# okc: problem=muse_attention_decode
"""`fused_attention.py` computes muse_attention_decode in one readable CUDA kernel.

Each request's keys are split into chunks, and one thread block handles one chunk of one request
and one key/value head, for every query head of that group and every token of the block. It:

1. RMS-normalizes the queries, scales them by `qk_scale_factor`, applies RoPE on sliding layers and
   rounds them to bf16, as `reference.py` does.
2. Streams its chunk of cached keys and values through shared memory, 32 keys at a time, and folds
   each tile into an online softmax on CUDA cores.
3. In the first chunk's block only: normalizes the block's new keys, applies RoPE, writes the new
   keys and values to the FP8 cache, and attends to them as one last, causal tile.
4. Publishes its softmax state. The last block of the request and KV head to finish merges every
   chunk's state, divides by the softmax sum, and applies the per-head RMSNorm and sigmoid gate.

Interface:
    kernel_fn(data): `TASK.md`'s timed call; returns bfloat16 [B, L, Hq * 64].

Example:
    >>> out = kernel_fn(reference.generate_input(1, 2048, "tp8", "sliding", seed=1))

Must know:
    Gotchas:
        - The arrival counters live across calls; the merging block resets its counter to 0.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE: str = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 64;
constexpr int kWarpSize = 32;
constexpr int kMaxRows = 64;
constexpr int kTileKeys = 32;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kRowsPerPass = kThreads / kHeadDim;
constexpr int kAccumulatorCount = kMaxRows / kRowsPerPass;
constexpr int kBytesPerLoad = 16;
constexpr int kLoadsPerRow = kHeadDim / kBytesPerLoad;
constexpr int kStepKeys = 64;
constexpr int kMmaRows = 16;
constexpr int kMmaCols = 8;
constexpr int kDimTiles = kHeadDim / kMmaCols;
constexpr int kRowStride = kHeadDim + 8;
constexpr int kLoadsPerThread = kStepKeys * kLoadsPerRow / kThreads;
constexpr int kMaxSplits = 64;
constexpr int kMmaDepth = 16;
constexpr int kChunkKeys = 16;
constexpr int kChunksPerStep = kStepKeys / kChunkKeys;
constexpr int kDepthSteps = kHeadDim / kMmaDepth;
constexpr float kRmsEps = 1e-5f;
constexpr float kLog2e = 1.4426950408889634f;

struct Params {
  const __nv_bfloat16* qkv_device;
  __nv_fp8_e4m3* k_cache_device;
  __nv_fp8_e4m3* v_cache_device;
  const int* seq_lens_device;
  const int64_t* cache_loc_device;
  const float* k_scale_device;
  const float* v_scale_device;
  const float* inv_freq_device;
  int L, q_head_count, kv_head_count;
  float qk_scale_factor;
  const int* page_table_device;
  __nv_bfloat16* out_device;
  int page_size, page_table_stride, window;
  float sm_scale;
  float* partial_out_device;
  float* partial_max_device;
  float* partial_sum_device;
  int* arrival_count_device;
  int split_count, keys_per_split;
};

struct ScalarTile {
  float q[kMaxRows][kHeadDim];
  float k[kTileKeys][kHeadDim + 1];
  float v[kTileKeys][kHeadDim];
  float p[kMaxRows][kTileKeys];
  int key_position[kTileKeys];
  float row_max[kMaxRows];
  float row_sum[kMaxRows];
  float row_rescale[kMaxRows];
  bool is_last_split;
};

struct SharedTile {
  __half k[kStepKeys][kRowStride];
  __half v[kStepKeys][kRowStride];
  union {
    __half q[kMaxRows][kRowStride];
    float out[kMaxRows][kHeadDim];
  };
  int key_position[kStepKeys];
  union {
    float warp_out[kWarps][kMmaRows][kHeadDim];
    float split_weight[kMaxSplits][kMaxRows];
  };
  float warp_max[kWarps][kMmaRows];
  float warp_sum[kWarps][kMmaRows];
  float row_max[kMaxRows];
  float row_sum[kMaxRows];
  bool is_last_split;
};

struct KeyAddresses {
  int page[kLoadsPerThread];
  int position[kLoadsPerThread];
};

struct RawKeys {
  uint4 k[kLoadsPerThread];
  uint4 v[kLoadsPerThread];
  int position[kLoadsPerThread];
};

struct WarpState {
  float out[kDimTiles][4];
  float max_log2[2];
  float sum[2];
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
__device__ void load_cached_tile(const Params& p, ScalarTile& tile, int b, int kv_head,
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
__device__ void attend_tile(const Params& p, ScalarTile& tile, float* accumulators, int row_count,
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
__device__ void write_gated_output(const Params& p, ScalarTile& tile, const float* accumulators,
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
 * @brief `load_new_tile` normalizes and rotates the block's L new keys, writes them and their
 * values to the FP8 cache, and puts the stored (FP8-rounded) values in shared memory.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] seq_len Cached tokens of this request: the first new token's position.
 *
 * @par Example
 * @code
 * load_new_tile(p, tile, b, kv_head, seq_len);
 * @endcode
 */
__device__ void load_new_tile(const Params& p, ScalarTile& tile, int b, int kv_head, int seq_len) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int qkv_row_size = (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim;
  for (int key = warp; key < kTileKeys; key += kWarps) {
    float2 k_stored = make_float2(0.0f, 0.0f);
    float2 v_stored = make_float2(0.0f, 0.0f);
    if (key < p.L) {
      const __nv_bfloat162* token =
          reinterpret_cast<const __nv_bfloat162*>(p.qkv_device + (b * p.L + key) * qkv_row_size);
      const float2 k_in =
          __bfloat1622float2(token[(p.q_head_count + kv_head) * kHeadDim / 2 + lane]);
      const float2 v_in = __bfloat1622float2(
          token[(p.q_head_count + p.kv_head_count + kv_head) * kHeadDim / 2 + lane]);
      const float2 k = normalize_and_rotate(k_in, 1.0f, seq_len + key, p.inv_freq_device, lane);
      const __nv_fp8_e4m3 k_fp8[2] = {__nv_fp8_e4m3(k.x / *p.k_scale_device),
                                      __nv_fp8_e4m3(k.y / *p.k_scale_device)};
      const __nv_fp8_e4m3 v_fp8[2] = {__nv_fp8_e4m3(v_in.x / *p.v_scale_device),
                                      __nv_fp8_e4m3(v_in.y / *p.v_scale_device)};
      const int64_t offset =
          (p.cache_loc_device[b * p.L + key] * p.kv_head_count + kv_head) * kHeadDim + 2 * lane;
      p.k_cache_device[offset] = k_fp8[0];
      p.k_cache_device[offset + 1] = k_fp8[1];
      p.v_cache_device[offset] = v_fp8[0];
      p.v_cache_device[offset + 1] = v_fp8[1];
      k_stored = make_float2(static_cast<float>(k_fp8[0]), static_cast<float>(k_fp8[1]));
      v_stored = make_float2(static_cast<float>(v_fp8[0]), static_cast<float>(v_fp8[1]));
    }
    tile.k[key][2 * lane] = k_stored.x;
    tile.k[key][2 * lane + 1] = k_stored.y;
    tile.v[key][2 * lane] = v_stored.x;
    tile.v[key][2 * lane + 1] = v_stored.y;
    if (lane == 0) {
      tile.key_position[key] = key < p.L ? seq_len + key : -1;
    }
  }
}

/**
 * @brief `mma_16x8x16` accumulates D += A B on tensor cores for one warp: A is 16 x 16 and B is
 * 16 x 8, both fp16, in the `mma.sync.m16n8k16` fragment layouts; D is fp32.
 *
 * @param[in,out] d This thread's 4 accumulator elements.
 * @param[in] a This thread's A fragment (4 registers of 2 halves).
 * @param[in] b0 First register of this thread's B fragment.
 * @param[in] b1 Second register of this thread's B fragment.
 *
 * @par Example
 * @code
 * mma_16x8x16(scores[n], q_fragments[s], b[0], b[1]);
 * @endcode
 */
__device__ void mma_16x8x16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/**
 * @brief `load_matrices` loads four 8 x 8 fp16 matrices from shared memory into one register each
 * per thread (`ldmatrix.x4`), optionally transposed.
 *
 * @tparam kIsTransposed Whether each matrix is transposed on the way.
 * @param[out] r This thread's 4 registers.
 * @param[in] row_address This thread's row: lanes 8 i to 8 i + 7 give the 8 rows of matrix i.
 *
 * @par Example
 * @code
 * load_matrices<false>(q_fragments[s], &tile.q[row][col]);
 * @endcode
 */
template <bool kIsTransposed>
__device__ void load_matrices(uint32_t (&r)[4], const __half* row_address) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(row_address));
  if (kIsTransposed) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
  }
}

/**
 * @brief `pack_halves` rounds two floats to fp16 and packs them in one register, `low` first.
 *
 * @param[in] low Element with the lower index.
 * @param[in] high Element with the higher index.
 * @return The packed pair.
 *
 * @par Example
 * @code
 * uint32_t pair = pack_halves(p0, p1);
 * @endcode
 */
__device__ uint32_t pack_halves(float low, float high) {
  const __half2 pair = __floats2half2_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

/**
 * @brief `store_fp8_as_halves` converts 16 FP8 E4M3 values to fp16 (exactly) and stores them.
 *
 * @param[in] bytes 16 FP8 values.
 * @param[out] destination 16 fp16 values in shared memory, 16-byte aligned.
 *
 * @par Example
 * @code
 * store_fp8_as_halves(raw.k[i], &tile.k[key][chunk * kBytesPerLoad]);
 * @endcode
 */
__device__ void store_fp8_as_halves(const uint4& bytes, __half* destination) {
  const __nv_fp8x2_storage_t* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&bytes);
  uint32_t halves[kBytesPerLoad / 2];
  for (int i = 0; i < kBytesPerLoad / 2; ++i) {
    const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(pairs[i], __NV_E4M3);
    halves[i] = static_cast<uint32_t>(pair.x) | (static_cast<uint32_t>(pair.y) << 16);
  }
  reinterpret_cast<uint4*>(destination)[0] = make_uint4(halves[0], halves[1], halves[2], halves[3]);
  reinterpret_cast<uint4*>(destination)[1] = make_uint4(halves[4], halves[5], halves[6], halves[7]);
}

/**
 * @brief `locate_keys` starts reading the page table for up to 64 keys of one request and KV head;
 * missing keys get position -1.
 *
 * @param[in] p Kernel parameters.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] first_position_incl First key position of the step.
 * @param[in] end_position_excl Keys at or after this position are missing.
 * @return The pages of this thread's pieces (thread t's piece i is 16 bytes of key (128 i + t) /
 * 4), still in flight: nothing here uses them, so the thread does not wait for the reads.
 *
 * @par Example
 * @code
 * KeyAddresses addresses = locate_keys(p, b, kv_head, start, end);
 * @endcode
 */
__device__ KeyAddresses locate_keys(const Params& p, int b, int kv_head, int first_position_incl,
                                    int end_position_excl) {
  KeyAddresses addresses;
  for (int i = 0; i < kLoadsPerThread; ++i) {
    const int piece = i * kThreads + threadIdx.x;
    const int position = first_position_incl + piece / kLoadsPerRow;
    addresses.position[i] = position < end_position_excl ? position : -1;
    addresses.page[i] = 0;
    if (position < end_position_excl) {
      addresses.page[i] = p.page_table_device[b * p.page_table_stride + position / p.page_size];
    }
  }
  return addresses;
}

/**
 * @brief `load_raw_keys` starts loading this thread's pieces of a step's FP8 keys and values into
 * registers; missing keys get zero bytes.
 *
 * @param[in] p Kernel parameters.
 * @param[in] kv_head Key/value head index.
 * @param[in] addresses The pieces' pages, from `locate_keys` one step earlier.
 * @return The pieces, not yet converted.
 *
 * @par Example
 * @code
 * RawKeys next = load_raw_keys(p, kv_head, next_addresses);
 * @endcode
 */
__device__ RawKeys load_raw_keys(const Params& p, int kv_head, const KeyAddresses& addresses) {
  RawKeys raw;
  for (int i = 0; i < kLoadsPerThread; ++i) {
    const int piece = i * kThreads + threadIdx.x;
    const int position = addresses.position[i];
    raw.position[i] = position;
    raw.k[i] = make_uint4(0, 0, 0, 0);
    raw.v[i] = make_uint4(0, 0, 0, 0);
    if (position >= 0) {
      const int64_t row =
          static_cast<int64_t>(addresses.page[i]) * p.page_size + position % p.page_size;
      const int64_t offset =
          (row * p.kv_head_count + kv_head) * kHeadDim + (piece % kLoadsPerRow) * kBytesPerLoad;
      raw.k[i] = *reinterpret_cast<const uint4*>(p.k_cache_device + offset);
      raw.v[i] = *reinterpret_cast<const uint4*>(p.v_cache_device + offset);
    }
  }
  return raw;
}

/**
 * @brief `store_raw_keys` converts a step's raw keys and values to fp16 in shared memory.
 *
 * @param[in] raw This thread's pieces from `load_raw_keys`.
 * @param[out] tile Shared memory tile: its keys, values and key positions.
 *
 * @par Example
 * @code
 * store_raw_keys(current, tile);
 * @endcode
 */
__device__ void store_raw_keys(const RawKeys& raw, SharedTile& tile) {
  for (int i = 0; i < kLoadsPerThread; ++i) {
    const int piece = i * kThreads + threadIdx.x;
    const int key = piece / kLoadsPerRow;
    const int dim = (piece % kLoadsPerRow) * kBytesPerLoad;
    store_fp8_as_halves(raw.k[i], &tile.k[key][dim]);
    store_fp8_as_halves(raw.v[i], &tile.v[key][dim]);
    if (piece % kLoadsPerRow == 0) {
      tile.key_position[key] = raw.position[i];
    }
  }
}

/**
 * @brief `store_new_keys` normalizes and rotates the block's L new keys, writes them and their
 * values to the FP8 cache, and puts the stored values in shared memory as a step of keys.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile: keys 0 to L - 1 are the new ones, the rest are missing.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] seq_len Cached tokens of this request: the first new token's position.
 *
 * @par Example
 * @code
 * store_new_keys(p, tile, b, kv_head, seq_len);
 * @endcode
 */
__device__ void store_new_keys(const Params& p, SharedTile& tile, int b, int kv_head, int seq_len) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int qkv_row_size = (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim;
  for (int key = warp; key < kStepKeys; key += kWarps) {
    __half2 k_stored = __floats2half2_rn(0.0f, 0.0f);
    __half2 v_stored = k_stored;
    if (key < p.L) {
      const __nv_bfloat162* token =
          reinterpret_cast<const __nv_bfloat162*>(p.qkv_device + (b * p.L + key) * qkv_row_size);
      const float2 k_in =
          __bfloat1622float2(token[(p.q_head_count + kv_head) * kHeadDim / 2 + lane]);
      const float2 v_in = __bfloat1622float2(
          token[(p.q_head_count + p.kv_head_count + kv_head) * kHeadDim / 2 + lane]);
      const float2 k = normalize_and_rotate(k_in, 1.0f, seq_len + key, p.inv_freq_device, lane);
      const __nv_fp8_e4m3 k_fp8[2] = {__nv_fp8_e4m3(k.x / *p.k_scale_device),
                                      __nv_fp8_e4m3(k.y / *p.k_scale_device)};
      const __nv_fp8_e4m3 v_fp8[2] = {__nv_fp8_e4m3(v_in.x / *p.v_scale_device),
                                      __nv_fp8_e4m3(v_in.y / *p.v_scale_device)};
      const int64_t offset =
          (p.cache_loc_device[b * p.L + key] * p.kv_head_count + kv_head) * kHeadDim + 2 * lane;
      p.k_cache_device[offset] = k_fp8[0];
      p.k_cache_device[offset + 1] = k_fp8[1];
      p.v_cache_device[offset] = v_fp8[0];
      p.v_cache_device[offset + 1] = v_fp8[1];
      k_stored = __floats2half2_rn(static_cast<float>(k_fp8[0]), static_cast<float>(k_fp8[1]));
      v_stored = __floats2half2_rn(static_cast<float>(v_fp8[0]), static_cast<float>(v_fp8[1]));
    }
    reinterpret_cast<__half2*>(&tile.k[key][0])[lane] = k_stored;
    reinterpret_cast<__half2*>(&tile.v[key][0])[lane] = v_stored;
    if (lane == 0) {
      tile.key_position[key] = key < p.L ? seq_len + key : -1;
    }
  }
}

/**
 * @brief `merge_splits` publishes this block's softmax state and, in the last block of its
 * request and KV head to arrive, merges every split's state.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile; the merging block gets the merged row sums.
 * @param[in,out] accumulators This thread's unnormalized outputs (see `attend_tile`); the merging
 * block gets the merged outputs.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] split This block's split.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @return Whether this block merged and must write the output.
 *
 * @par Example
 * @code
 * if (merge_splits(p, tile, accumulators, b, kv_head, split, row_count)) { ... }
 * @endcode
 */
__device__ bool merge_splits(const Params& p, ScalarTile& tile, float* accumulators, int b,
                             int kv_head, int split, int row_count) {
  const int group = b * p.kv_head_count + kv_head;
  const int64_t first_slot = static_cast<int64_t>(group) * p.split_count * kMaxRows;
  const int64_t own_slot = first_slot + static_cast<int64_t>(split) * kMaxRows;
  const int d = threadIdx.x % kHeadDim;
#pragma unroll
  for (int i = 0; i < kAccumulatorCount; ++i) {
    const int row = threadIdx.x / kHeadDim + kRowsPerPass * i;
    if (row < row_count) {
      p.partial_out_device[(own_slot + row) * kHeadDim + d] = accumulators[i];
    }
  }
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    p.partial_max_device[own_slot + row] = tile.row_max[row];
    p.partial_sum_device[own_slot + row] = tile.row_sum[row];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    tile.is_last_split = atomicAdd(&p.arrival_count_device[group], 1) == p.split_count - 1;
  }
  __syncthreads();
  if (!tile.is_last_split) {
    return false;
  }
  __threadfence();
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    float merged_max = -INFINITY;
    for (int s = 0; s < p.split_count; ++s) {
      merged_max =
          fmaxf(merged_max, __ldcg(&p.partial_max_device[first_slot + s * kMaxRows + row]));
    }
    float merged_sum = 0.0f;
    for (int s = 0; s < p.split_count; ++s) {
      const int64_t slot = first_slot + s * kMaxRows + row;
      merged_sum += __ldcg(&p.partial_sum_device[slot]) *
                    expf(__ldcg(&p.partial_max_device[slot]) - merged_max);
    }
    tile.row_max[row] = merged_max;
    tile.row_sum[row] = merged_sum;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kAccumulatorCount; ++i) {
    const int row = threadIdx.x / kHeadDim + kRowsPerPass * i;
    if (row < row_count) {
      float merged_out = 0.0f;
      for (int s = 0; s < p.split_count; ++s) {
        const int64_t slot = first_slot + s * kMaxRows + row;
        merged_out += __ldcg(&p.partial_out_device[slot * kHeadDim + d]) *
                      expf(__ldcg(&p.partial_max_device[slot]) - tile.row_max[row]);
      }
      accumulators[i] = merged_out;
    }
  }
  if (threadIdx.x == 0) {
    p.arrival_count_device[group] = 0;
  }
  return true;
}

/**
 * @brief `attention_kernel` attends one split of one request's keys for one KV head group; the
 * first split also writes the block's new keys and values, and the last to finish writes the
 * output.
 *
 * @param[in] p Kernel parameters; the grid is (splits, KV heads, requests).
 *
 * @par Example
 * @code
 * attention_kernel<<<dim3(split_count, kv_head_count, B), kThreads, 0, stream>>>(p);
 * @endcode
 */
__global__ void __launch_bounds__(kThreads) attention_kernel(const Params p) {
  __shared__ ScalarTile tile;
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int b = blockIdx.z;
  const int group_size = p.q_head_count / p.kv_head_count;
  const int row_count = p.L * group_size;
  const int seq_len = p.seq_lens_device[b];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int qkv_row_size = (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim;
  for (int row = warp; row < row_count; row += kWarps) {
    const int head = kv_head * group_size + row % group_size;
    const __nv_bfloat162* q_in = reinterpret_cast<const __nv_bfloat162*>(
        p.qkv_device + (b * p.L + row / group_size) * qkv_row_size + head * kHeadDim);
    const float2 q = normalize_and_rotate(__bfloat1622float2(q_in[lane]), p.qk_scale_factor,
                                          seq_len + row / group_size, p.inv_freq_device, lane);
    tile.q[row][2 * lane] = __bfloat162float(__float2bfloat16(q.x));
    tile.q[row][2 * lane + 1] = __bfloat162float(__float2bfloat16(q.y));
    if (lane == 0) {
      tile.row_max[row] = -INFINITY;
      tile.row_sum[row] = 0.0f;
    }
  }
  float accumulators[kAccumulatorCount] = {};
  const int window_start_incl = p.window == 0 ? 0 : max(0, seq_len - p.window + 1);
  const int first_position_incl = window_start_incl + split * p.keys_per_split;
  const int end_position_excl = min(first_position_incl + p.keys_per_split, seq_len);
  for (int start = first_position_incl; start < end_position_excl; start += kTileKeys) {
    load_cached_tile(p, tile, b, kv_head, start, end_position_excl);
    __syncthreads();
    attend_tile(p, tile, accumulators, row_count, seq_len);
  }
  if (split == 0) {
    load_new_tile(p, tile, b, kv_head, seq_len);
    __syncthreads();
    attend_tile(p, tile, accumulators, row_count, seq_len);
  }
  if (merge_splits(p, tile, accumulators, b, kv_head, split, row_count)) {
    write_gated_output(p, tile, accumulators, b, kv_head, row_count);
  }
}

/**
 * @brief `attend_chunk` folds 16 keys from shared memory into one warp's online softmax over its
 * 16 query rows: S = Q K^T and O += P V on tensor cores, the masks and the rescaling in between.
 *
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the step's keys, values and key positions.
 * @param[in] q_fragments This warp's query rows as A fragments, one per 16 head dimensions.
 * @param[in] first_key First of the 16 keys, within the step.
 * @param[in] first_row First of the warp's 16 query rows.
 * @param[in] seq_len Cached tokens of this request: the first new token's position.
 * @param[in] logit_scale_log2 Factor from Q K^T to base-2 logits.
 * @param[in,out] state This thread's part of the warp's softmax state and outputs.
 *
 * @par Example
 * @code
 * attend_chunk(p, tile, q_fragments, chunk * kChunkKeys, first_row, seq_len, scale, state);
 * @endcode
 */
__device__ void attend_chunk(const Params& p, const SharedTile& tile,
                             const uint32_t (&q_fragments)[kDepthSteps][4], int first_key,
                             int first_row, int seq_len, float logit_scale_log2, WarpState& state) {
  const int lane = threadIdx.x % kWarpSize;
  const int group_size = p.q_head_count / p.kv_head_count;
  float scores[2][4] = {};
  for (int n = 0; n < 2; ++n) {
    for (int pair = 0; pair < kDepthSteps / 2; ++pair) {
      uint32_t b[4];
      load_matrices<false>(
          b, &tile.k[first_key + n * kMmaCols + lane % 8][32 * pair + 8 * (lane / 8)]);
      mma_16x8x16(scores[n], q_fragments[2 * pair], b[0], b[1]);
      mma_16x8x16(scores[n], q_fragments[2 * pair + 1], b[2], b[3]);
    }
  }
  for (int half = 0; half < 2; ++half) {
    const int query_position = seq_len + (first_row + lane / 4 + 8 * half) / group_size;
    float chunk_max = -INFINITY;
    for (int n = 0; n < 2; ++n) {
      for (int e = 2 * half; e < 2 * half + 2; ++e) {
        const int key_position =
            tile.key_position[first_key + n * kMmaCols + 2 * (lane % 4) + e % 2];
        const bool is_visible = key_position >= 0 && key_position <= query_position &&
                                (p.window == 0 || key_position > query_position - p.window);
        scores[n][e] = is_visible ? scores[n][e] * logit_scale_log2 : -INFINITY;
        chunk_max = fmaxf(chunk_max, scores[n][e]);
      }
    }
    chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffffu, chunk_max, 1));
    chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffffu, chunk_max, 2));
    const float new_max = fmaxf(state.max_log2[half], chunk_max);
    const float safe_max = new_max == -INFINITY ? 0.0f : new_max;
    const float rescale = exp2f(state.max_log2[half] - safe_max);
    state.max_log2[half] = new_max;
    state.sum[half] *= rescale;
    for (int d = 0; d < kDimTiles; ++d) {
      state.out[d][2 * half] *= rescale;
      state.out[d][2 * half + 1] *= rescale;
    }
    for (int n = 0; n < 2; ++n) {
      for (int e = 2 * half; e < 2 * half + 2; ++e) {
        scores[n][e] = exp2f(scores[n][e] - safe_max);
        state.sum[half] += scores[n][e];
      }
    }
  }
  const uint32_t p_fragment[4] = {
      pack_halves(scores[0][0], scores[0][1]), pack_halves(scores[0][2], scores[0][3]),
      pack_halves(scores[1][0], scores[1][1]), pack_halves(scores[1][2], scores[1][3])};
  for (int pair = 0; pair < kDimTiles / 2; ++pair) {
    uint32_t b[4];
    load_matrices<true>(
        b, &tile.v[first_key + 8 * ((lane / 8) % 2) + lane % 8][16 * pair + 8 * (lane / 16)]);
    mma_16x8x16(state.out[2 * pair], p_fragment, b[0], b[1]);
    mma_16x8x16(state.out[2 * pair + 1], p_fragment, b[2], b[3]);
  }
}

/**
 * @brief `combine_warps` merges the softmax states of the warps that share query rows into the
 * block's state: unnormalized outputs in `out`, row maxima (natural-log units) and row sums.
 *
 * @param[in,out] tile Shared memory tile: reads the warps' states, writes `out`, `row_max` and
 * `row_sum`.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @param[in] m_tile_count Groups of 16 query rows.
 * @param[in] key_group_count Warps per group of 16 query rows, each over its own keys.
 *
 * @par Example
 * @code
 * combine_warps(tile, row_count, m_tile_count, key_group_count);
 * @endcode
 */
__device__ void combine_warps(SharedTile& tile, int row_count, int m_tile_count,
                              int key_group_count) {
  for (int element = threadIdx.x; element < row_count * kHeadDim; element += kThreads) {
    const int row = element / kHeadDim;
    const int d = element % kHeadDim;
    const int m_tile = row / kMmaRows;
    const int row_in_tile = row % kMmaRows;
    float merged_max = -INFINITY;
    for (int group = 0; group < key_group_count; ++group) {
      merged_max = fmaxf(merged_max, tile.warp_max[group * m_tile_count + m_tile][row_in_tile]);
    }
    const float safe_max = merged_max == -INFINITY ? 0.0f : merged_max;
    float merged_sum = 0.0f;
    float merged_out = 0.0f;
    for (int group = 0; group < key_group_count; ++group) {
      const int warp = group * m_tile_count + m_tile;
      const float weight = exp2f(tile.warp_max[warp][row_in_tile] - safe_max);
      merged_sum += weight * tile.warp_sum[warp][row_in_tile];
      merged_out += weight * tile.warp_out[warp][row_in_tile][d];
    }
    tile.out[row][d] = merged_out;
    if (d == 0) {
      tile.row_max[row] = merged_max / kLog2e;
      tile.row_sum[row] = merged_sum;
    }
  }
}

/**
 * @brief `merge_splits` publishes this block's softmax state and, in the last block of its
 * request and KV head to arrive, merges every split's state into `out` and `row_sum`.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile with this block's state; the merging block gets the
 * merged state and uses `split_weight` for each split's weight per row.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] split This block's split.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @return Whether this block merged and must write the output.
 *
 * @par Example
 * @code
 * if (merge_splits(p, tile, b, kv_head, split, row_count)) { ... }
 * @endcode
 */
__device__ bool merge_splits(const Params& p, SharedTile& tile, int b, int kv_head, int split,
                             int row_count) {
  const int group = b * p.kv_head_count + kv_head;
  const int64_t first_slot = static_cast<int64_t>(group) * p.split_count * kMaxRows;
  const int64_t own_slot = first_slot + static_cast<int64_t>(split) * kMaxRows;
  for (int element = threadIdx.x; element < row_count * kHeadDim; element += kThreads) {
    p.partial_out_device[own_slot * kHeadDim + element] =
        tile.out[element / kHeadDim][element % kHeadDim];
  }
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    p.partial_max_device[own_slot + row] = tile.row_max[row];
    p.partial_sum_device[own_slot + row] = tile.row_sum[row];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    tile.is_last_split = atomicAdd(&p.arrival_count_device[group], 1) == p.split_count - 1;
  }
  __syncthreads();
  if (!tile.is_last_split) {
    return false;
  }
  __threadfence();
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    float merged_max = -INFINITY;
    for (int s = 0; s < p.split_count; ++s) {
      tile.split_weight[s][row] = __ldcg(&p.partial_max_device[first_slot + s * kMaxRows + row]);
      merged_max = fmaxf(merged_max, tile.split_weight[s][row]);
    }
    float merged_sum = 0.0f;
    for (int s = 0; s < p.split_count; ++s) {
      tile.split_weight[s][row] = expf(tile.split_weight[s][row] - merged_max);
      merged_sum += tile.split_weight[s][row] *
                    __ldcg(&p.partial_sum_device[first_slot + s * kMaxRows + row]);
    }
    tile.row_sum[row] = merged_sum;
  }
  __syncthreads();
  for (int element = threadIdx.x; element < row_count * kHeadDim; element += kThreads) {
    float merged_out = 0.0f;
    for (int s = 0; s < p.split_count; ++s) {
      merged_out += tile.split_weight[s][element / kHeadDim] *
                    __ldcg(&p.partial_out_device[(first_slot + s * kMaxRows) * kHeadDim + element]);
    }
    tile.out[element / kHeadDim][element % kHeadDim] = merged_out;
  }
  if (threadIdx.x == 0) {
    p.arrival_count_device[group] = 0;
  }
  __syncthreads();
  return true;
}

}  // namespace

/**
 * @brief `run` checks the tensors and launches `attention_kernel` on the current stream.
 *
 * @param[in] qkv Fused projection output, bf16 [B, L, (2 Hq + 2 Hkv) * 64].
 * @param[in,out] k_cache FP8 key cache [pages, page_size, Hkv, 64]; the new keys are written.
 * @param[in,out] v_cache FP8 value cache, same shape; the new values are written.
 * @param[in] page_table int32 [B, max_pages].
 * @param[in] seq_lens int32 [B].
 * @param[in] cache_loc int64 [B * L], the new tokens' flat cache rows.
 * @param[in] k_scale float32 [], key = e4m3 * k_scale.
 * @param[in] v_scale float32 [], value = e4m3 * v_scale.
 * @param[in] inv_freq float32 [32] RoPE inverse frequencies, or empty for no RoPE.
 * @param[out] out bf16 [B, L, Hq * 64].
 * @param[in] window Sliding window in tokens, or 0 for full attention.
 * @param[in] qk_scale_factor Query gain after the RMSNorm.
 * @param[in] sm_scale Softmax scale.
 * @param[out] partial_out float32 [B, Hkv, splits, 64, 64] workspace: each split's outputs.
 * @param[out] partial_max float32 [B, Hkv, splits, 64] workspace: each split's row maxima.
 * @param[out] partial_sum float32 [B, Hkv, splits, 64] workspace: each split's row sums.
 * @param[in,out] arrival_count int32 [B, Hkv], zero before and after each call.
 * @param[in] keys_per_split Cached keys per split, a multiple of 32.
 * @throws c10::Error A tensor has the wrong dtype, shape or layout.
 *
 * @par Example
 * @code
 * run(qkv, k_cache, v_cache, page_table, seq_lens, cache_loc, k_scale, v_scale, inv_freq, out,
 *     2048, 5.473, 0.125, partial_out, partial_max, partial_sum, arrival_count, 64);
 * @endcode
 */
void run(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
         torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_scale,
         torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor out, int64_t window,
         double qk_scale_factor, double sm_scale, torch::Tensor partial_out,
         torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor arrival_count,
         int64_t keys_per_split) {
  const int64_t q_head_count = out.size(2) / kHeadDim;
  TORCH_CHECK(qkv.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  page_table.is_contiguous() && out.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(k_cache.size(3) == kHeadDim, "head_dim must be ", kHeadDim);
  TORCH_CHECK(qkv.size(1) * q_head_count / k_cache.size(2) <= kMaxRows,
              "L * Hq / Hkv must be <= ", kMaxRows);
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kFloat8_e4m3fn &&
                  page_table.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt &&
                  cache_loc.scalar_type() == at::kLong && out.scalar_type() == at::kBFloat16,
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
  p.page_table_device = page_table.data_ptr<int>();
  p.out_device = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.L = static_cast<int>(qkv.size(1));
  p.q_head_count = static_cast<int>(q_head_count);
  p.kv_head_count = static_cast<int>(k_cache.size(2));
  p.page_size = static_cast<int>(k_cache.size(1));
  p.page_table_stride = static_cast<int>(page_table.size(1));
  p.window = static_cast<int>(window);
  p.qk_scale_factor = static_cast<float>(qk_scale_factor);
  p.sm_scale = static_cast<float>(sm_scale);
  p.partial_out_device = partial_out.data_ptr<float>();
  p.partial_max_device = partial_max.data_ptr<float>();
  p.partial_sum_device = partial_sum.data_ptr<float>();
  p.arrival_count_device = arrival_count.data_ptr<int>();
  p.split_count = static_cast<int>(partial_max.size(2));
  p.keys_per_split = static_cast<int>(keys_per_split);
  attention_kernel<<<dim3(p.split_count, k_cache.size(2), qkv.size(0)), kThreads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
CPP_SOURCE: str = (
    "void run(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, "
    "torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor cache_loc, "
    "torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor out, "
    "int64_t window, double qk_scale_factor, double sm_scale, torch::Tensor partial_out, "
    "torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor arrival_count, "
    "int64_t keys_per_split);"
)
HEAD_DIM: int = 64
ROPE_PAIR_STRIDE: int = 2
MAX_ROWS: int = 64
TILE_KEYS: int = 32
TARGET_BLOCKS_PER_SM: int = 4
_extension_cache: dict[str, object] = {}
_inv_freq_cache: dict[tuple[float, torch.device], torch.Tensor] = {}
_workspace_cache: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, ...]] = {}


def _get_extension() -> object:
    """`_get_extension` compiles the CUDA source once per process and returns the module.

    Returns:
        The compiled extension.

    Example:
        >>> _get_extension().run(...)
    """
    if "module" not in _extension_cache:
        _extension_cache["module"] = load_inline(
            name="muse_fused_attention_mma_pieces",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["run"],
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


def _plan_splits(data: dict) -> tuple[int, int]:
    """`_plan_splits` picks how many blocks share each request's keys, from shapes only.

    Args:
        data: The inputs; only `config` and the page table's shape are read.

    Returns:
        The split count and the cached keys per split (a multiple of 32), so that every key a
        request can hold falls in a split and the grid has about `TARGET_BLOCKS_PER_SM` per SM.

    Example:
        >>> split_count, keys_per_split = _plan_splits(data)
    """
    config: dict = data["config"]
    span_keys: int = config["window"] or data["page_table"].shape[1] * config["page_size"]
    tile_count: int = -(-span_keys // TILE_KEYS)
    sm_count: int = torch.cuda.get_device_properties(data["qkv"].device).multi_processor_count
    group_count: int = config["batch"] * config["num_kv_heads"]
    wanted_split_count: int = max(1, -(-TARGET_BLOCKS_PER_SM * sm_count // group_count))
    keys_per_split: int = -(-tile_count // min(tile_count, wanted_split_count)) * TILE_KEYS
    return -(-span_keys // keys_per_split), keys_per_split


def _get_workspace(data: dict, split_count: int) -> tuple[torch.Tensor, ...]:
    """`_get_workspace` returns the per-split state buffers and zeroed arrival counters.

    Args:
        data: The inputs; only `config` and the device are read.
        split_count: Splits per request and KV head.

    Returns:
        partial_out [B, Hkv, splits, 64, 64], partial_max and partial_sum [B, Hkv, splits, 64]
        (float32), and arrival_count [B, Hkv] (int32, zero), allocated once per shape.

    Example:
        >>> partial_out, partial_max, partial_sum, arrival_count = _get_workspace(data, 8)
    """
    config: dict = data["config"]
    device: torch.device = data["qkv"].device
    shape: tuple[int, int, int] = (config["batch"], config["num_kv_heads"], split_count)
    key: tuple[int, int, int, torch.device] = (*shape, device)
    if key not in _workspace_cache:
        _workspace_cache[key] = (
            torch.empty((*shape, MAX_ROWS, HEAD_DIM), device=device),
            torch.empty((*shape, MAX_ROWS), device=device),
            torch.empty((*shape, MAX_ROWS), device=device),
            torch.zeros(shape[:2], dtype=torch.int32, device=device),
        )
    return _workspace_cache[key]


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
    split_count, keys_per_split = _plan_splits(data)
    out: torch.Tensor = torch.empty(
        (config["batch"], config["q_len"], config["num_q_heads"] * HEAD_DIM),
        dtype=torch.bfloat16,
        device=data["qkv"].device,
    )
    _get_extension().run(
        data["qkv"],
        data["k_cache"],
        data["v_cache"],
        data["page_table"],
        data["seq_lens"],
        data["cache_loc"],
        data["k_scale"],
        data["v_scale"],
        _get_inv_freq(config["rope_theta"], data["qkv"].device),
        out,
        config["window"],
        config["qk_scale_factor"],
        config["sm_scale"],
        *_get_workspace(data, split_count),
        keys_per_split,
    )
    return out
