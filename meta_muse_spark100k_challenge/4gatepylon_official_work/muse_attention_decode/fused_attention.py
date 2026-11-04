# okc: problem=muse_attention_decode
"""`fused_attention.py` computes muse_attention_decode, moving its steps into CUDA one at a time.

The step runs in two stages. A CUDA prologue kernel RMS-normalizes the queries and the block's new
keys, scales the queries by `qk_scale_factor`, applies RoPE on sliding layers, rounds the queries to
bf16 as `reference.py` does, and writes the new keys and values to the FP8 cache. The attention
stage, still PyTorch, then attends over each request's cache and applies the per-head RMSNorm and
the sigmoid gate. Later versions move it into the same CUDA kernel.

Interface:
    kernel_fn(data): `TASK.md`'s timed call; returns bfloat16 [B, L, Hq * 64].

Example:
    >>> out = kernel_fn(reference.generate_input(1, 2048, "tp8", "sliding", seed=1))

Must know:
    Performance:
        - The PyTorch attention gathers every position a request's page table can hold.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE: str = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 64;
constexpr int kWarpSize = 32;
constexpr int kPrologueWarps = 8;
constexpr float kRmsEps = 1e-5f;

struct Params {
  const __nv_bfloat16* qkv_device;
  __nv_fp8_e4m3* k_cache_device;
  __nv_fp8_e4m3* v_cache_device;
  const int* seq_lens_device;
  const int64_t* cache_loc_device;
  const float* k_scale_device;
  const float* v_scale_device;
  const float* inv_freq_device;
  __nv_bfloat16* q_device;
  int L, q_head_count, kv_head_count;
  float qk_scale_factor;
};

/**
 * @brief `warp_sum` returns the sum of `x` over the 32 lanes of a warp, in every lane.
 *
 * @param[in] x This lane's value.
 * @return The warp-wide sum.
 *
 * @par Example
 * @code
 * float total = warp_sum(partial);
 * @endcode
 */
__device__ float warp_sum(float x) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  }
  return x;
}

/**
 * @brief `normalize_and_rotate` RMS-normalizes one 64-wide head vector held by a warp, scales it,
 * and applies interleaved RoPE.
 *
 * @param[in] x This lane's pair of elements (2 * lane, 2 * lane + 1).
 * @param[in] gain Factor applied after the normalization.
 * @param[in] position Token position for RoPE.
 * @param[in] inv_freq_device RoPE inverse frequencies [32], or nullptr for no RoPE.
 * @param[in] lane This thread's lane.
 * @return This lane's normalized, rotated pair.
 *
 * @par Example
 * @code
 * float2 q = normalize_and_rotate(x, p.qk_scale_factor, seq + j, p.inv_freq_device, lane);
 * @endcode
 */
__device__ float2 normalize_and_rotate(float2 x, float gain, int position,
                                       const float* inv_freq_device, int lane) {
  const float inv_rms = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / kHeadDim + kRmsEps);
  x = make_float2(x.x * inv_rms * gain, x.y * inv_rms * gain);
  if (inv_freq_device == nullptr) {
    return x;
  }
  const float angle = static_cast<float>(position) * inv_freq_device[lane];
  const float cos_angle = cosf(angle);
  const float sin_angle = sinf(angle);
  return make_float2(x.x * cos_angle - x.y * sin_angle, x.x * sin_angle + x.y * cos_angle);
}

/**
 * @brief `prologue_kernel` prepares one new token: its normalized, rotated, bf16-rounded queries,
 * and its keys and values written to the FP8 cache.
 *
 * @param[in] p Kernel parameters; the grid has one block per token (B x L).
 *
 * @par Example
 * @code
 * prologue_kernel<<<B * L, kPrologueWarps * kWarpSize, 0, stream>>>(p);
 * @endcode
 */
__global__ void prologue_kernel(const Params p) {
  const int token = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int position = p.seq_lens_device[token / p.L] + token % p.L;
  const __nv_bfloat162* token_in = reinterpret_cast<const __nv_bfloat162*>(
      p.qkv_device + token * (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim);
  for (int head = warp; head < p.q_head_count + p.kv_head_count; head += kPrologueWarps) {
    const float2 x = __bfloat1622float2(token_in[head * kHeadDim / 2 + lane]);
    if (head < p.q_head_count) {
      const float2 q =
          normalize_and_rotate(x, p.qk_scale_factor, position, p.inv_freq_device, lane);
      __nv_bfloat162* q_out = reinterpret_cast<__nv_bfloat162*>(p.q_device);
      q_out[(token * p.q_head_count + head) * kHeadDim / 2 + lane] = __float22bfloat162_rn(q);
      continue;
    }
    const int kv_head = head - p.q_head_count;
    const float2 k = normalize_and_rotate(x, 1.0f, position, p.inv_freq_device, lane);
    const float2 v = __bfloat1622float2(token_in[(head + p.kv_head_count) * kHeadDim / 2 + lane]);
    const int64_t offset =
        (p.cache_loc_device[token] * p.kv_head_count + kv_head) * kHeadDim + 2 * lane;
    p.k_cache_device[offset] = __nv_fp8_e4m3(k.x / *p.k_scale_device);
    p.k_cache_device[offset + 1] = __nv_fp8_e4m3(k.y / *p.k_scale_device);
    p.v_cache_device[offset] = __nv_fp8_e4m3(v.x / *p.v_scale_device);
    p.v_cache_device[offset + 1] = __nv_fp8_e4m3(v.y / *p.v_scale_device);
  }
}

}  // namespace

/**
 * @brief `prologue` checks the tensors and launches `prologue_kernel` on the current stream.
 *
 * @param[in] qkv Fused projection output, bf16 [B, L, (2 Hq + 2 Hkv) * 64].
 * @param[in,out] k_cache FP8 key cache [pages, page_size, Hkv, 64]; the new keys are written.
 * @param[in,out] v_cache FP8 value cache, same shape; the new values are written.
 * @param[in] seq_lens int32 [B].
 * @param[in] cache_loc int64 [B * L], the new tokens' flat cache rows.
 * @param[in] k_scale float32 [], key = e4m3 * k_scale.
 * @param[in] v_scale float32 [], value = e4m3 * v_scale.
 * @param[in] inv_freq float32 [32] RoPE inverse frequencies, or empty for no RoPE.
 * @param[out] q bf16 [B, L, Hq, 64], the prepared queries.
 * @param[in] qk_scale_factor Query gain after the RMSNorm.
 * @throws c10::Error A tensor has the wrong dtype, shape or layout.
 *
 * @par Example
 * @code
 * prologue(qkv, k_cache, v_cache, seq_lens, cache_loc, k_scale, v_scale, inv_freq, q, 5.473);
 * @endcode
 */
void prologue(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache,
              torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor q,
              double qk_scale_factor) {
  TORCH_CHECK(qkv.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  q.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(k_cache.size(3) == kHeadDim && q.size(3) == kHeadDim, "head_dim must be ", kHeadDim);
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kFloat8_e4m3fn &&
                  seq_lens.scalar_type() == at::kInt && cache_loc.scalar_type() == at::kLong,
              "unexpected dtype");
  Params p{};
  p.qkv_device = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.k_cache_device = reinterpret_cast<__nv_fp8_e4m3*>(k_cache.data_ptr());
  p.v_cache_device = reinterpret_cast<__nv_fp8_e4m3*>(v_cache.data_ptr());
  p.seq_lens_device = seq_lens.data_ptr<int>();
  p.cache_loc_device = cache_loc.data_ptr<int64_t>();
  p.k_scale_device = k_scale.data_ptr<float>();
  p.v_scale_device = v_scale.data_ptr<float>();
  p.inv_freq_device = inv_freq.numel() == 0 ? nullptr : inv_freq.data_ptr<float>();
  p.q_device = reinterpret_cast<__nv_bfloat16*>(q.data_ptr());
  p.L = static_cast<int>(qkv.size(1));
  p.q_head_count = static_cast<int>(q.size(2));
  p.kv_head_count = static_cast<int>(k_cache.size(2));
  p.qk_scale_factor = static_cast<float>(qk_scale_factor);
  prologue_kernel<<<qkv.size(0) * qkv.size(1), kPrologueWarps * kWarpSize, 0,
                    at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
CPP_SOURCE: str = (
    "void prologue(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, "
    "torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_scale, "
    "torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor q, double qk_scale_factor);"
)
HEAD_DIM: int = 64
ROPE_PAIR_STRIDE: int = 2
RMS_EPS: float = 1e-5
_extension_cache: dict[str, object] = {}
_inv_freq_cache: dict[tuple[float, torch.device], torch.Tensor] = {}


def _get_extension() -> object:
    """`_get_extension` compiles the CUDA source once per process and returns the module.

    Returns:
        The compiled extension.

    Example:
        >>> _get_extension().prologue(...)
    """
    if "module" not in _extension_cache:
        _extension_cache["module"] = load_inline(
            name="muse_fused_attention_prologue",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["prologue"],
            extra_cuda_cflags=["-O3"],
        )
    return _extension_cache["module"]


def _get_inv_freq(theta: float, device: torch.device) -> torch.Tensor:
    """`_get_inv_freq` returns RoPE's inverse frequencies the way `reference._rope` computes them.

    Args:
        theta: RoPE base; 0 means no RoPE.
        device: Device for the tensor.

    Returns:
        float32 [32] inverse frequencies, or an empty tensor when `theta` is 0.

    Example:
        >>> inv_freq = _get_inv_freq(500_000.0, torch.device("cuda"))
    """
    key: tuple[float, torch.device] = (theta, device)
    if key not in _inv_freq_cache:
        exponents: torch.Tensor = torch.arange(0, HEAD_DIM, ROPE_PAIR_STRIDE, device=device).float()
        _inv_freq_cache[key] = (
            1.0 / (theta ** (exponents / HEAD_DIM)) if theta else torch.empty(0, device=device)
        )
    return _inv_freq_cache[key]


def _attend_and_gate(data: dict, q: torch.Tensor) -> torch.Tensor:
    """`_attend_and_gate` attends the prepared queries over the cache and gates the result.

    Args:
        data: The inputs; the caches already hold the block's new keys and values.
        q: bf16 [B, L, Hq, 64] prepared queries.

    Returns:
        bfloat16 [B, L, Hq * 64]: rmsnorm(attention) * sigmoid(gate).

    Example:
        >>> out = _attend_and_gate(data, q)
    """
    config: dict = data["config"]
    B, L, Hq = config["batch"], config["q_len"], config["num_q_heads"]
    Hkv, page_size, window = config["num_kv_heads"], config["page_size"], config["window"]
    k_cache: torch.Tensor = data["k_cache"].view(-1, Hkv, HEAD_DIM)
    v_cache: torch.Tensor = data["v_cache"].view(-1, Hkv, HEAD_DIM)
    positions: torch.Tensor = data["positions"].view(B, L)
    t: torch.Tensor = torch.arange(data["page_table"].shape[1] * page_size, device=q.device)
    out: torch.Tensor = torch.empty((B, L, Hq, HEAD_DIM), device=q.device)
    for b in range(B):
        rows: torch.Tensor = data["page_table"][b].long().clamp(min=0)[t // page_size] * page_size
        rows = rows + t % page_size
        K: torch.Tensor = k_cache[rows].float() * data["k_scale"]
        V: torch.Tensor = v_cache[rows].float() * data["v_scale"]
        is_visible: torch.Tensor = t[None, :] <= positions[b][:, None]
        if window:
            is_visible &= t[None, :] > positions[b][:, None] - window
        S: torch.Tensor = torch.einsum("lkgd,skd->lkgs", q[b].view(L, Hkv, -1, HEAD_DIM).float(), K)
        S = (S * config["sm_scale"]).masked_fill(~is_visible[:, None, None, :], float("-inf"))
        out[b] = torch.einsum("lkgs,skd->lkgd", S.softmax(-1), V).reshape(L, Hq, HEAD_DIM)
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    gate: torch.Tensor = data["qkv"][..., (Hq + 2 * Hkv) * HEAD_DIM :].float().view(B, L, Hq, -1)
    return (out * torch.sigmoid(gate)).reshape(B, L, Hq * HEAD_DIM).to(torch.bfloat16)


def kernel_fn(data: dict) -> torch.Tensor:
    """`kernel_fn` runs one decode or verify step of gated attention (see `TASK.md`).

    Args:
        data: The inputs `reference.generate_input` builds.

    Returns:
        bfloat16 [B, L, Hq * 64]; the block's new keys and values are written to the caches.

    Example:
        >>> out = kernel_fn(data)
    """
    config: dict = data["config"]
    q: torch.Tensor = torch.empty(
        (config["batch"], config["q_len"], config["num_q_heads"], HEAD_DIM),
        dtype=torch.bfloat16,
        device=data["qkv"].device,
    )
    _get_extension().prologue(
        data["qkv"],
        data["k_cache"],
        data["v_cache"],
        data["seq_lens"],
        data["cache_loc"],
        data["k_scale"],
        data["v_scale"],
        _get_inv_freq(config["rope_theta"], data["qkv"].device),
        q,
        config["qk_scale_factor"],
    )
    return _attend_and_gate(data, q)
