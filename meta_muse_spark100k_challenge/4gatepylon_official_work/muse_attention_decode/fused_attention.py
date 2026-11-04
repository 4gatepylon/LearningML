# okc: problem=muse_attention_decode
"""`fused_attention.py` computes muse_attention_decode, moving its steps into CUDA one at a time.

The step runs as two CUDA kernels. The prologue kernel RMS-normalizes the queries and the
block's new keys, scales the queries by `qk_scale_factor`, applies RoPE on sliding layers, rounds
the queries to bf16 as `reference.py` does, and writes the new keys and values to the FP8 cache.
The attention kernel runs one thread block per request and key/value head. It streams the
request's keys and values (the new ones included) through shared memory, 32 at a time, folds each
tile into an online softmax on CUDA cores, then applies the per-head RMSNorm and the sigmoid gate.

Interface:
    kernel_fn(data): `TASK.md`'s timed call; returns bfloat16 [B, L, Hq * 64].

Example:
    >>> out = kernel_fn(reference.generate_input(1, 2048, "tp8", "sliding", seed=1))

Must know:
    Performance:
        - With one block per request and KV head, small batches use only a few of the GPU's SMs.
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
constexpr int kMaxRows = 64;
constexpr int kTileKeys = 32;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kRowsPerPass = kThreads / kHeadDim;
constexpr int kAccumulatorCount = kMaxRows / kRowsPerPass;
constexpr int kBytesPerLoad = 16;
constexpr int kLoadsPerRow = kHeadDim / kBytesPerLoad;
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
  const int* page_table_device;
  __nv_bfloat16* out_device;
  int page_size, page_table_stride, window;
  float sm_scale;
};

struct SharedTile {
  float q[kMaxRows][kHeadDim];
  float k[kTileKeys][kHeadDim + 1];
  float v[kTileKeys][kHeadDim];
  float p[kMaxRows][kTileKeys];
  int key_position[kTileKeys];
  float row_max[kMaxRows];
  float row_sum[kMaxRows];
  float row_rescale[kMaxRows];
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
 * @brief `warp_max` returns the maximum of `x` over the 32 lanes of a warp, in every lane.
 *
 * @param[in] x This lane's value.
 * @return The warp-wide maximum.
 *
 * @par Example
 * @code
 * float tile_max = warp_max(score);
 * @endcode
 */
__device__ float warp_max(float x) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset));
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

/**
 * @brief `load_cached_tile` copies up to 32 keys and values of one request and KV head from the
 * paged FP8 cache into shared memory, and marks missing keys with position -1.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] first_position_incl First key position of the tile.
 * @param[in] end_position_excl Keys at or after this position are missing.
 *
 * @par Example
 * @code
 * load_cached_tile(p, tile, b, kv_head, start, seq_len + p.L);
 * @endcode
 */
__device__ void load_cached_tile(const Params& p, SharedTile& tile, int b, int kv_head,
                                 int first_position_incl, int end_position_excl) {
  const int key = threadIdx.x / kLoadsPerRow;
  const int chunk = threadIdx.x % kLoadsPerRow;
  const int position = first_position_incl + key;
  const bool is_present = position < end_position_excl;
  uint4 k_bytes = make_uint4(0, 0, 0, 0);
  uint4 v_bytes = make_uint4(0, 0, 0, 0);
  if (is_present) {
    const int page = p.page_table_device[b * p.page_table_stride + position / p.page_size];
    const int64_t row = static_cast<int64_t>(page) * p.page_size + position % p.page_size;
    const int64_t offset = (row * p.kv_head_count + kv_head) * kHeadDim + chunk * kBytesPerLoad;
    k_bytes = *reinterpret_cast<const uint4*>(p.k_cache_device + offset);
    v_bytes = *reinterpret_cast<const uint4*>(p.v_cache_device + offset);
  }
  const __nv_fp8_e4m3* k_values = reinterpret_cast<const __nv_fp8_e4m3*>(&k_bytes);
  const __nv_fp8_e4m3* v_values = reinterpret_cast<const __nv_fp8_e4m3*>(&v_bytes);
  for (int i = 0; i < kBytesPerLoad; ++i) {
    tile.k[key][chunk * kBytesPerLoad + i] = static_cast<float>(k_values[i]);
    tile.v[key][chunk * kBytesPerLoad + i] = static_cast<float>(v_values[i]);
  }
  if (chunk == 0) {
    tile.key_position[key] = is_present ? position : -1;
  }
}

/**
 * @brief `attend_tile` folds the keys in shared memory into every query row's online softmax and
 * output accumulator.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile; its scores and per-row softmax state are updated.
 * @param[in,out] accumulators This thread's unnormalized outputs, for rows
 * `threadIdx.x / 64 + 2 i` and dimension `threadIdx.x % 64`.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @param[in] seq_len Cached tokens of this request: the first new token's position.
 *
 * @par Example
 * @code
 * attend_tile(p, tile, accumulators, row_count, seq_len);
 * @endcode
 */
__device__ void attend_tile(const Params& p, SharedTile& tile, float* accumulators, int row_count,
                            int seq_len) {
  const int group_size = p.q_head_count / p.kv_head_count;
  const float logit_scale = p.sm_scale * *p.k_scale_device;
  for (int index = threadIdx.x; index < row_count * kTileKeys; index += kThreads) {
    const int row = index / kTileKeys;
    const int key = index % kTileKeys;
    float dot = 0.0f;
    for (int d = 0; d < kHeadDim; ++d) {
      dot += tile.q[row][d] * tile.k[key][d];
    }
    const int key_position = tile.key_position[key];
    const int query_position = seq_len + row / group_size;
    const bool is_visible = key_position >= 0 && key_position <= query_position &&
                            (p.window == 0 || key_position > query_position - p.window);
    tile.p[row][key] = is_visible ? dot * logit_scale : -INFINITY;
  }
  __syncthreads();
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int row = warp; row < row_count; row += kWarps) {
    const float score = tile.p[row][lane];
    const float old_max = tile.row_max[row];
    const float new_max = fmaxf(old_max, warp_max(score));
    const float safe_max = new_max == -INFINITY ? 0.0f : new_max;
    const float probability = expf(score - safe_max);
    const float tile_sum = warp_sum(probability);
    tile.p[row][lane] = probability;
    if (lane == 0) {
      tile.row_rescale[row] = expf(old_max - safe_max);
      tile.row_sum[row] = tile.row_sum[row] * tile.row_rescale[row] + tile_sum;
      tile.row_max[row] = new_max;
    }
  }
  __syncthreads();
  const int d = threadIdx.x % kHeadDim;
#pragma unroll
  for (int i = 0; i < kAccumulatorCount; ++i) {
    const int row = threadIdx.x / kHeadDim + kRowsPerPass * i;
    if (row < row_count) {
      float pv = 0.0f;
      for (int key = 0; key < kTileKeys; ++key) {
        pv += tile.p[row][key] * tile.v[key][d];
      }
      accumulators[i] = accumulators[i] * tile.row_rescale[row] + pv;
    }
  }
  __syncthreads();
}

/**
 * @brief `write_gated_output` divides each query row's accumulated output by its softmax sum,
 * applies the per-head RMSNorm and the sigmoid gate, and writes the result as bf16.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile; its query rows are reused for the outputs.
 * @param[in] accumulators This thread's unnormalized outputs (see `attend_tile`).
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] row_count Query rows (L x query heads per KV head).
 *
 * @par Example
 * @code
 * write_gated_output(p, tile, accumulators, b, kv_head, row_count);
 * @endcode
 */
__device__ void write_gated_output(const Params& p, SharedTile& tile, const float* accumulators,
                                   int b, int kv_head, int row_count) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int group_size = p.q_head_count / p.kv_head_count;
  const int qkv_row_size = (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim;
  const float v_scale = *p.v_scale_device;
#pragma unroll
  for (int i = 0; i < kAccumulatorCount; ++i) {
    const int row = threadIdx.x / kHeadDim + kRowsPerPass * i;
    if (row < row_count) {
      tile.q[row][threadIdx.x % kHeadDim] = accumulators[i] / tile.row_sum[row] * v_scale;
    }
  }
  __syncthreads();
  for (int row = warp; row < row_count; row += kWarps) {
    const int token = b * p.L + row / group_size;
    const int head = kv_head * group_size + row % group_size;
    const float2 o = make_float2(tile.q[row][2 * lane], tile.q[row][2 * lane + 1]);
    const float inv_rms = rsqrtf(warp_sum(o.x * o.x + o.y * o.y) / kHeadDim + kRmsEps);
    const __nv_bfloat162* gate = reinterpret_cast<const __nv_bfloat162*>(
        p.qkv_device + token * qkv_row_size +
        (p.q_head_count + 2 * p.kv_head_count + head) * kHeadDim);
    const float2 g = __bfloat1622float2(gate[lane]);
    const float2 out =
        make_float2(o.x * inv_rms / (1.0f + expf(-g.x)), o.y * inv_rms / (1.0f + expf(-g.y)));
    __nv_bfloat162* out_row = reinterpret_cast<__nv_bfloat162*>(
        p.out_device + (token * p.q_head_count + head) * kHeadDim);
    out_row[lane] = __float22bfloat162_rn(out);
  }
}

/**
 * @brief `attention_kernel` attends one request's prepared queries of one KV head group over its
 * cache, including the block's new keys, and writes the gated output.
 *
 * @param[in] p Kernel parameters; the grid is (KV heads, requests).
 *
 * @par Example
 * @code
 * attention_kernel<<<dim3(kv_head_count, B), kThreads, 0, stream>>>(p);
 * @endcode
 */
__global__ void __launch_bounds__(kThreads) attention_kernel(const Params p) {
  __shared__ SharedTile tile;
  const int kv_head = blockIdx.x;
  const int b = blockIdx.y;
  const int group_size = p.q_head_count / p.kv_head_count;
  const int row_count = p.L * group_size;
  const int seq_len = p.seq_lens_device[b];
  for (int index = threadIdx.x; index < row_count * kHeadDim; index += kThreads) {
    const int row = index / kHeadDim;
    const int head = kv_head * group_size + row % group_size;
    const int64_t q_offset = ((b * p.L + row / group_size) * p.q_head_count + head) * kHeadDim;
    tile.q[row][index % kHeadDim] = __bfloat162float(p.q_device[q_offset + index % kHeadDim]);
  }
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    tile.row_max[row] = -INFINITY;
    tile.row_sum[row] = 0.0f;
  }
  float accumulators[kAccumulatorCount] = {};
  const int first_position_incl = p.window == 0 ? 0 : max(0, seq_len - p.window + 1);
  for (int start = first_position_incl; start < seq_len + p.L; start += kTileKeys) {
    load_cached_tile(p, tile, b, kv_head, start, seq_len + p.L);
    __syncthreads();
    attend_tile(p, tile, accumulators, row_count, seq_len);
  }
  write_gated_output(p, tile, accumulators, b, kv_head, row_count);
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

/**
 * @brief `attend` checks the tensors and launches `attention_kernel` on the current stream.
 *
 * @param[in] qkv Fused projection output, bf16 [B, L, (2 Hq + 2 Hkv) * 64], for the gate.
 * @param[in] q bf16 [B, L, Hq, 64], the queries `prologue` prepared.
 * @param[in] k_cache FP8 key cache [pages, page_size, Hkv, 64], with the block's new keys.
 * @param[in] v_cache FP8 value cache, same shape, with the block's new values.
 * @param[in] page_table int32 [B, max_pages].
 * @param[in] seq_lens int32 [B].
 * @param[in] k_scale float32 [], key = e4m3 * k_scale.
 * @param[in] v_scale float32 [], value = e4m3 * v_scale.
 * @param[out] out bf16 [B, L, Hq * 64].
 * @param[in] window Sliding window in tokens, or 0 for full attention.
 * @param[in] sm_scale Softmax scale.
 * @throws c10::Error A tensor has the wrong dtype, shape or layout.
 *
 * @par Example
 * @code
 * attend(qkv, q, k_cache, v_cache, page_table, seq_lens, k_scale, v_scale, out, 2048, 0.125);
 * @endcode
 */
void attend(torch::Tensor qkv, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
            torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor k_scale,
            torch::Tensor v_scale, torch::Tensor out, int64_t window, double sm_scale) {
  TORCH_CHECK(page_table.is_contiguous() && out.is_contiguous() && q.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(q.size(1) * q.size(2) / k_cache.size(2) <= kMaxRows,
              "L * Hq / Hkv must be <= ", kMaxRows);
  TORCH_CHECK(page_table.scalar_type() == at::kInt && out.scalar_type() == at::kBFloat16,
              "unexpected dtype");
  Params p{};
  p.qkv_device = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.k_cache_device = reinterpret_cast<__nv_fp8_e4m3*>(k_cache.data_ptr());
  p.v_cache_device = reinterpret_cast<__nv_fp8_e4m3*>(v_cache.data_ptr());
  p.seq_lens_device = seq_lens.data_ptr<int>();
  p.k_scale_device = k_scale.data_ptr<float>();
  p.v_scale_device = v_scale.data_ptr<float>();
  p.q_device = reinterpret_cast<__nv_bfloat16*>(q.data_ptr());
  p.page_table_device = page_table.data_ptr<int>();
  p.out_device = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.L = static_cast<int>(q.size(1));
  p.q_head_count = static_cast<int>(q.size(2));
  p.kv_head_count = static_cast<int>(k_cache.size(2));
  p.page_size = static_cast<int>(k_cache.size(1));
  p.page_table_stride = static_cast<int>(page_table.size(1));
  p.window = static_cast<int>(window);
  p.sm_scale = static_cast<float>(sm_scale);
  attention_kernel<<<dim3(k_cache.size(2), q.size(0)), kThreads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
CPP_SOURCE: str = (
    "void prologue(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, "
    "torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_scale, "
    "torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor q, double qk_scale_factor);"
    "void attend(torch::Tensor qkv, torch::Tensor q, torch::Tensor k_cache, "
    "torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor seq_lens, "
    "torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor out, int64_t window, "
    "double sm_scale);"
)
HEAD_DIM: int = 64
ROPE_PAIR_STRIDE: int = 2
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
            name="muse_fused_attention_two_kernels",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["prologue", "attend"],
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
    out: torch.Tensor = torch.empty_like(q).view(q.shape[0], q.shape[1], -1)
    _get_extension().attend(
        data["qkv"],
        q,
        data["k_cache"],
        data["v_cache"],
        data["page_table"],
        data["seq_lens"],
        data["k_scale"],
        data["v_scale"],
        out,
        config["window"],
        config["sm_scale"],
    )
    return out
