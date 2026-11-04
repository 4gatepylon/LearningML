# okc: problem=muse_attention_decode
"""`fused_attention.py` computes muse_attention_decode in one readable CUDA kernel.

Each request's keys are split into chunks, and one thread block of 4 warps handles one chunk of
one request and one key/value head, for every query head of that group and every token of the
block. It:

1. Copies its query and gate rows to shared memory in one `cp.async` batch, RMS-normalizes the
   queries, scales them by `qk_scale_factor`, applies RoPE from a per-token cos/sin table and
   rounds them to bf16, as `reference.py` does.
2. Streams its chunk of cached FP8 keys and values 64 at a time through a 4-stage `cp.async`
   ring in shared memory, converting each step to fp16. Each warp attends its 16 query rows over
   its share of the step's keys: S = Q K^T and O += P V run on tensor cores (`mma.sync`), with one
   online-softmax update per step. Only steps at the edge of the window or chunk apply the mask.
3. In the first chunk's block only: normalizes the block's new keys, applies RoPE, writes the new
   keys and values to the FP8 cache, and attends to them as one last, causal step.
4. Combines its warps' softmax states and publishes the block's state. A two-level tree merges
   the chunks: the last block of each group of up to 16 chunks merges that group, then the last
   group merger merges the groups, divides by the softmax sum, and applies the per-head RMSNorm
   and the sigmoid gate.

Interface:
    kernel_fn(data): `TASK.md`'s timed call; returns bfloat16 [B, L, Hq * 64].

Example:
    >>> out = kernel_fn(reference.generate_input(1, 2048, "tp8", "sliding", seed=1))

Must know:
    Gotchas:
        - The arrival counters live across calls; each merging block resets its counter to 0.
    Performance:
        - The page-table reads for a step are issued one step before its keys and values.
        - The split count fills exactly one wave of resident blocks, from the occupancy calculator.
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
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kStepKeys = 64;
constexpr int kMmaRows = 16;
constexpr int kMmaCols = 8;
constexpr int kMmaDepth = 16;
constexpr int kChunkKeys = 16;
constexpr int kDimTiles = kHeadDim / kMmaCols;
constexpr int kDepthSteps = kHeadDim / kMmaDepth;
constexpr int kRowStride = kHeadDim + 8;
constexpr int kBytesPerLoad = 16;
constexpr int kLoadsPerRow = kHeadDim / kBytesPerLoad;
constexpr int kLoadsPerThread = kStepKeys * kLoadsPerRow / kThreads;
constexpr int kMaxSplits = 256;
constexpr int kStages = 4;
constexpr int kMaxTokens = 8;
constexpr int kMergeFanIn = 16;
constexpr int kMaxMergeGroups = kMaxSplits / kMergeFanIn;
constexpr int kCountersPerGroup = kMaxMergeGroups + 1;
constexpr int kFloatsPerVector = 4;
constexpr int kVectorsPerRow = kHeadDim / kFloatsPerVector;
constexpr int kPiecesPerRow = kHeadDim * 2 / kBytesPerLoad;
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

struct StepTile {
  __half k[kStepKeys][kRowStride];
  __half v[kStepKeys][kRowStride];
};

struct StartTile {
  __nv_bfloat16 q_in[kMaxRows][kHeadDim];
  __half q[kMaxRows][kRowStride];
};

struct MergeTile {
  float split_weight[kMergeFanIn][kMaxRows];
  __nv_bfloat16 gate[kMaxRows][kHeadDim];
};

struct SharedTile {
  union {
    StepTile step;
    float warp_out[kWarps][kMmaRows][kHeadDim];
    MergeTile merge;
  };
  union {
    StartTile start;
    float out[kMaxRows][kHeadDim];
  };
  float2 rotation[kMaxTokens][kHeadDim / 2];
  uint8_t raw_k[kStages][kStepKeys][kHeadDim];
  uint8_t raw_v[kStages][kStepKeys][kHeadDim];
  int raw_position[kStages][kStepKeys];
  int key_position[kStepKeys];
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
 * @brief `normalize_and_rotate` RMS-normalizes one 64-wide head vector held by a warp, scales it,
 * and applies interleaved RoPE.
 *
 * @param[in] x This lane's pair of elements (2 * lane, 2 * lane + 1).
 * @param[in] gain Factor applied after the normalization.
 * @param[in] rotation This pair's (cos, sin) at the token's position; (1, 0) for no RoPE.
 * @return This lane's normalized, rotated pair.
 *
 * @par Example
 * @code
 * float2 q = normalize_and_rotate(x, p.qk_scale_factor, tile.rotation[token][lane]);
 * @endcode
 */
__device__ float2 normalize_and_rotate(float2 x, float gain, float2 rotation) {
  const float inv_rms = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / kHeadDim + kRmsEps);
  x = make_float2(x.x * inv_rms * gain, x.y * inv_rms * gain);
  return make_float2(x.x * rotation.x - x.y * rotation.y, x.x * rotation.y + x.y * rotation.x);
}

/**
 * @brief `write_gated_output` divides each query row's merged output by its softmax sum, applies
 * the per-head RMSNorm and the sigmoid gate, and writes the result as bf16.
 *
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the merged, unnormalized outputs, row sums and gates.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] row_count Query rows (L x query heads per KV head).
 *
 * @par Example
 * @code
 * write_gated_output(p, tile, b, kv_head, row_count);
 * @endcode
 */
__device__ void write_gated_output(const Params& p, const SharedTile& tile, int b, int kv_head,
                                   int row_count) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int group_size = p.q_head_count / p.kv_head_count;
  for (int row = warp; row < row_count; row += kWarps) {
    const int token = b * p.L + row / group_size;
    const int head = kv_head * group_size + row % group_size;
    const float scale = *p.v_scale_device / tile.row_sum[row];
    const float2 o =
        make_float2(tile.out[row][2 * lane] * scale, tile.out[row][2 * lane + 1] * scale);
    const float inv_rms = rsqrtf(warp_sum(o.x * o.x + o.y * o.y) / kHeadDim + kRmsEps);
    const float2 g =
        __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&tile.merge.gate[row][0])[lane]);
    const float2 gated =
        make_float2(o.x * inv_rms / (1.0f + expf(-g.x)), o.y * inv_rms / (1.0f + expf(-g.y)));
    __nv_bfloat162* out_row = reinterpret_cast<__nv_bfloat162*>(
        p.out_device + (token * p.q_head_count + head) * kHeadDim);
    out_row[lane] = __float22bfloat162_rn(gated);
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
 * load_matrices<false>(q_fragments[s], &tile.start.q[row][col]);
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
 * store_fp8_as_halves(raw_bytes, &tile.step.k[key][dim]);
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
 * @brief `copy_async_16` starts copying 16 bytes from global to shared memory (`cp.async`), or
 * zero-fills the destination.
 *
 * @param[out] destination 16-byte aligned shared memory.
 * @param[in] source 16-byte aligned global memory, read only if `is_present`.
 * @param[in] is_present Whether to copy; otherwise the destination becomes zeros.
 *
 * @par Example
 * @code
 * copy_async_16(&tile.raw_k[stage][key][dim], p.k_cache_device + offset, is_present);
 * @endcode
 */
__device__ void copy_async_16(void* destination, const void* source, bool is_present) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  const int source_bytes = is_present ? kBytesPerLoad : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(source),
               "r"(source_bytes));
}

/**
 * @brief `wait_for_copies` waits until at most `kPendingCount` of this thread's committed
 * `cp.async` groups are still in flight.
 *
 * @tparam kPendingCount Groups that may stay in flight.
 *
 * @par Example
 * @code
 * wait_for_copies<kStages - 1>();
 * @endcode
 */
template <int kPendingCount>
__device__ void wait_for_copies() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPendingCount));
}

/**
 * @brief `start_rows_copy` starts copying one bf16 row per query row of this block's KV head
 * group from the fused projection output into shared memory, as one `cp.async` group.
 *
 * @param[in] p Kernel parameters.
 * @param[out] rows Shared memory destination, one 64-wide row per query row.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @param[in] first_head_column Column of query head 0 in a token's row: 0 for the queries,
 * (Hq + 2 Hkv) * 64 for the gate.
 *
 * @par Example
 * @code
 * start_rows_copy(p, tile.start.q_in, b, kv_head, row_count, 0);
 * @endcode
 */
__device__ void start_rows_copy(const Params& p, __nv_bfloat16 (*rows)[kHeadDim], int b,
                                int kv_head, int row_count, int first_head_column) {
  const int group_size = p.q_head_count / p.kv_head_count;
  const int qkv_row_size = (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim;
  for (int piece = threadIdx.x; piece < row_count * kPiecesPerRow; piece += kThreads) {
    const int row = piece / kPiecesPerRow;
    const int column = (piece % kPiecesPerRow) * (kBytesPerLoad / 2);
    const int head = kv_head * group_size + row % group_size;
    const __nv_bfloat16* source = p.qkv_device + (b * p.L + row / group_size) * qkv_row_size +
                                  first_head_column + head * kHeadDim + column;
    copy_async_16(&rows[row][column], source, true);
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

/**
 * @brief `fill_rotation_table` computes RoPE's (cos, sin) for each new token's position and each
 * pair of head dimensions, or (1, 0) when the layer has no RoPE.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile: its rotation table.
 * @param[in] seq_len Cached tokens of this request: the first new token's position.
 *
 * @par Example
 * @code
 * fill_rotation_table(p, tile, seq_len);
 * @endcode
 */
__device__ void fill_rotation_table(const Params& p, SharedTile& tile, int seq_len) {
  for (int index = threadIdx.x; index < p.L * kHeadDim / 2; index += kThreads) {
    const int token = index / (kHeadDim / 2);
    const int pair = index % (kHeadDim / 2);
    float2 rotation = make_float2(1.0f, 0.0f);
    if (p.inv_freq_device != nullptr) {
      const float angle = static_cast<float>(seq_len + token) * p.inv_freq_device[pair];
      rotation = make_float2(cosf(angle), sinf(angle));
    }
    tile.rotation[token][pair] = rotation;
  }
}

/**
 * @brief `start_step_copy` starts copying this thread's pieces of one step's FP8 keys and values
 * into a ring stage, as one `cp.async` group; missing keys become zeros.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile: its ring stage and the stage's key positions.
 * @param[in] kv_head Key/value head index.
 * @param[in] addresses The pieces' pages, from `locate_keys` one step earlier.
 * @param[in] stage Ring stage to fill.
 *
 * @par Example
 * @code
 * start_step_copy(p, tile, kv_head, addresses, (step + kStages - 1) % kStages);
 * @endcode
 */
__device__ void start_step_copy(const Params& p, SharedTile& tile, int kv_head,
                                const KeyAddresses& addresses, int stage) {
  for (int i = 0; i < kLoadsPerThread; ++i) {
    const int piece = i * kThreads + threadIdx.x;
    const int key = piece / kLoadsPerRow;
    const int dim = (piece % kLoadsPerRow) * kBytesPerLoad;
    const int position = addresses.position[i];
    const int64_t row = static_cast<int64_t>(addresses.page[i]) * p.page_size +
                        (position >= 0 ? position % p.page_size : 0);
    const int64_t offset = (row * p.kv_head_count + kv_head) * kHeadDim + dim;
    copy_async_16(&tile.raw_k[stage][key][dim], p.k_cache_device + offset, position >= 0);
    copy_async_16(&tile.raw_v[stage][key][dim], p.v_cache_device + offset, position >= 0);
    if (piece % kLoadsPerRow == 0) {
      tile.raw_position[stage][key] = addresses.position[i];
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

/**
 * @brief `convert_step` converts this thread's pieces of a ring stage to the fp16 keys and values
 * that the tensor cores read.
 *
 * @param[in,out] tile Shared memory tile: reads the stage, writes keys, values and key positions.
 * @param[in] stage Ring stage whose copies this thread has waited for.
 *
 * @par Example
 * @code
 * convert_step(tile, step % kStages);
 * @endcode
 */
__device__ void convert_step(SharedTile& tile, int stage) {
  for (int i = 0; i < kLoadsPerThread; ++i) {
    const int piece = i * kThreads + threadIdx.x;
    const int key = piece / kLoadsPerRow;
    const int dim = (piece % kLoadsPerRow) * kBytesPerLoad;
    store_fp8_as_halves(*reinterpret_cast<const uint4*>(&tile.raw_k[stage][key][dim]),
                        &tile.step.k[key][dim]);
    store_fp8_as_halves(*reinterpret_cast<const uint4*>(&tile.raw_v[stage][key][dim]),
                        &tile.step.v[key][dim]);
    if (piece % kLoadsPerRow == 0) {
      tile.key_position[key] = tile.raw_position[stage][key];
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
      const float2 k = normalize_and_rotate(k_in, 1.0f, tile.rotation[key][lane]);
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
    reinterpret_cast<__half2*>(&tile.step.k[key][0])[lane] = k_stored;
    reinterpret_cast<__half2*>(&tile.step.v[key][0])[lane] = v_stored;
    if (lane == 0) {
      tile.key_position[key] = key < p.L ? seq_len + key : -1;
    }
  }
}

/**
 * @brief `attend_keys` folds this warp's keys of a step (`kChunkCount` chunks of 16) into its
 * online softmax over its 16 query rows: S = Q K^T on tensor cores, one max and one rescale for
 * all of them, then O += P V on tensor cores.
 *
 * @tparam kChunkCount Chunks of 16 keys this warp attends in the step.
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the step's keys, values and key positions.
 * @param[in] q_fragments This warp's query rows as A fragments, one per 16 head dimensions.
 * @param[in] first_key First of the warp's keys, within the step.
 * @param[in] first_row First of the warp's 16 query rows.
 * @param[in] seq_len Cached tokens of this request: the first new token's position.
 * @param[in] logit_scale_log2 Factor from Q K^T to base-2 logits.
 * @param[in] is_masked Whether some key may be missing or invisible to some row; steps inside
 * the window and the split skip the mask.
 * @param[in,out] state This thread's part of the warp's softmax state and outputs.
 *
 * @par Example
 * @code
 * attend_keys<4>(p, tile, q_fragments, 0, first_row, seq_len, scale, is_masked, state);
 * @endcode
 */
template <int kChunkCount>
__device__ void attend_keys(const Params& p, const SharedTile& tile,
                            const uint32_t (&q_fragments)[kDepthSteps][4], int first_key,
                            int first_row, int seq_len, float logit_scale_log2, bool is_masked,
                            WarpState& state) {
  const int lane = threadIdx.x % kWarpSize;
  const int group_size = p.q_head_count / p.kv_head_count;
  float scores[kChunkCount][2][4] = {};
  for (int chunk = 0; chunk < kChunkCount; ++chunk) {
    for (int pair = 0; pair < kDepthSteps / 2; ++pair) {
      for (int n = 0; n < 2; ++n) {
        const int key = first_key + chunk * kChunkKeys + n * kMmaCols + lane % 8;
        uint32_t b[4];
        load_matrices<false>(b, &tile.step.k[key][32 * pair + 8 * (lane / 8)]);
        mma_16x8x16(scores[chunk][n], q_fragments[2 * pair], b[0], b[1]);
        mma_16x8x16(scores[chunk][n], q_fragments[2 * pair + 1], b[2], b[3]);
      }
    }
  }
  for (int half = 0; half < 2; ++half) {
    const int query_position = seq_len + (first_row + lane / 4 + 8 * half) / group_size;
    float step_max = -INFINITY;
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      for (int n = 0; n < 2; ++n) {
        for (int e = 2 * half; e < 2 * half + 2; ++e) {
          float score = scores[chunk][n][e] * logit_scale_log2;
          if (is_masked) {
            const int key_position = tile.key_position[first_key + chunk * kChunkKeys +
                                                       n * kMmaCols + 2 * (lane % 4) + e % 2];
            const bool is_visible = key_position >= 0 && key_position <= query_position &&
                                    (p.window == 0 || key_position > query_position - p.window);
            score = is_visible ? score : -INFINITY;
          }
          scores[chunk][n][e] = score;
          step_max = fmaxf(step_max, score);
        }
      }
    }
    step_max = fmaxf(step_max, __shfl_xor_sync(0xffffffffu, step_max, 1));
    step_max = fmaxf(step_max, __shfl_xor_sync(0xffffffffu, step_max, 2));
    const float new_max = fmaxf(state.max_log2[half], step_max);
    const float safe_max = new_max == -INFINITY ? 0.0f : new_max;
    const float rescale = exp2f(state.max_log2[half] - safe_max);
    state.max_log2[half] = new_max;
    state.sum[half] *= rescale;
    for (int d = 0; d < kDimTiles; ++d) {
      state.out[d][2 * half] *= rescale;
      state.out[d][2 * half + 1] *= rescale;
    }
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      for (int n = 0; n < 2; ++n) {
        for (int e = 2 * half; e < 2 * half + 2; ++e) {
          scores[chunk][n][e] = exp2f(scores[chunk][n][e] - safe_max);
          state.sum[half] += scores[chunk][n][e];
        }
      }
    }
  }
  for (int chunk = 0; chunk < kChunkCount; ++chunk) {
    const uint32_t p_fragment[4] = {pack_halves(scores[chunk][0][0], scores[chunk][0][1]),
                                    pack_halves(scores[chunk][0][2], scores[chunk][0][3]),
                                    pack_halves(scores[chunk][1][0], scores[chunk][1][1]),
                                    pack_halves(scores[chunk][1][2], scores[chunk][1][3])};
    const int key = first_key + chunk * kChunkKeys + 8 * ((lane / 8) % 2) + lane % 8;
    for (int pair = 0; pair < kDimTiles / 2; ++pair) {
      uint32_t b[4];
      load_matrices<true>(b, &tile.step.v[key][16 * pair + 8 * (lane / 16)]);
      mma_16x8x16(state.out[2 * pair], p_fragment, b[0], b[1]);
      mma_16x8x16(state.out[2 * pair + 1], p_fragment, b[2], b[3]);
    }
  }
}

/**
 * @brief `attend_step` hands this warp's share of a step's 64 keys to `attend_keys`.
 *
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the step's keys, values and key positions.
 * @param[in] q_fragments This warp's query rows as A fragments.
 * @param[in] key_group This warp's share: keys `key_group * 64 / key_group_count` onwards.
 * @param[in] key_group_count Warps sharing the same query rows: 4, 2 or 1.
 * @param[in] first_row First of the warp's 16 query rows.
 * @param[in] seq_len Cached tokens of this request.
 * @param[in] logit_scale_log2 Factor from Q K^T to base-2 logits.
 * @param[in] is_masked Whether the step needs the visibility mask.
 * @param[in,out] state This thread's part of the warp's softmax state and outputs.
 *
 * @par Example
 * @code
 * attend_step(p, tile, q_fragments, key_group, 4, first_row, seq_len, scale, false, state);
 * @endcode
 */
__device__ void attend_step(const Params& p, const SharedTile& tile,
                            const uint32_t (&q_fragments)[kDepthSteps][4], int key_group,
                            int key_group_count, int first_row, int seq_len, float logit_scale_log2,
                            bool is_masked, WarpState& state) {
  const int first_key = key_group * kStepKeys / key_group_count;
  if (key_group_count == 4) {
    attend_keys<1>(p, tile, q_fragments, first_key, first_row, seq_len, logit_scale_log2, is_masked,
                   state);
  } else if (key_group_count == 2) {
    attend_keys<2>(p, tile, q_fragments, first_key, first_row, seq_len, logit_scale_log2, is_masked,
                   state);
  } else {
    attend_keys<4>(p, tile, q_fragments, first_key, first_row, seq_len, logit_scale_log2, is_masked,
                   state);
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
 * @brief `publish_state` writes the block's softmax state (unnormalized outputs, row maxima in
 * natural-log units, row sums) to one workspace slot.
 *
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the state.
 * @param[in] slot Workspace slot: every slot holds 64 rows.
 * @param[in] row_count Query rows (L x query heads per KV head).
 *
 * @par Example
 * @code
 * publish_state(p, tile, first_slot + split, row_count);
 * @endcode
 */
__device__ void publish_state(const Params& p, const SharedTile& tile, int64_t slot,
                              int row_count) {
  float4* slot_out = reinterpret_cast<float4*>(p.partial_out_device + slot * kMaxRows * kHeadDim);
  for (int vector = threadIdx.x; vector < row_count * kVectorsPerRow; vector += kThreads) {
    slot_out[vector] = reinterpret_cast<const float4*>(&tile.out[0][0])[vector];
  }
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    p.partial_max_device[slot * kMaxRows + row] = tile.row_max[row];
    p.partial_sum_device[slot * kMaxRows + row] = tile.row_sum[row];
  }
}

/**
 * @brief `arrive` counts this block in at a workspace counter and tells whether it was the last
 * of `expected_count` blocks; the last block may then read what the others published.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile: its `is_last_split` flag.
 * @param[in] counter Index into the arrival counters.
 * @param[in] expected_count Blocks that arrive at this counter.
 * @return Whether this block arrived last.
 *
 * @par Example
 * @code
 * if (!arrive(p, tile, counter, 8)) { return false; }
 * @endcode
 */
__device__ bool arrive(const Params& p, SharedTile& tile, int counter, int expected_count) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    tile.is_last_split = atomicAdd(&p.arrival_count_device[counter], 1) == expected_count - 1;
  }
  __syncthreads();
  if (tile.is_last_split) {
    __threadfence();
  }
  return tile.is_last_split;
}

/**
 * @brief `merge_slots` merges up to 16 published softmax states into the block's state in `tile`.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile: gets the merged outputs, row maxima and row sums; uses
 * `split_weight` for each slot's weight per row.
 * @param[in] first_slot First workspace slot to merge.
 * @param[in] slot_stride Distance between merged slots.
 * @param[in] slot_count Slots to merge, at most `kMergeFanIn`.
 * @param[in] row_count Query rows (L x query heads per KV head).
 *
 * @par Example
 * @code
 * merge_slots(p, tile, first_slot, 1, 8, row_count);
 * @endcode
 */
__device__ void merge_slots(const Params& p, SharedTile& tile, int64_t first_slot, int slot_stride,
                            int slot_count, int row_count) {
  for (int index = threadIdx.x; index < slot_count * row_count; index += kThreads) {
    const int64_t slot = first_slot + (index / row_count) * slot_stride;
    tile.merge.split_weight[index / row_count][index % row_count] =
        __ldcg(&p.partial_max_device[slot * kMaxRows + index % row_count]);
  }
  __syncthreads();
  for (int row = threadIdx.x; row < row_count; row += kThreads) {
    float merged_max = -INFINITY;
    for (int s = 0; s < slot_count; ++s) {
      merged_max = fmaxf(merged_max, tile.merge.split_weight[s][row]);
    }
    const float safe_max = merged_max == -INFINITY ? 0.0f : merged_max;
    float merged_sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kMergeFanIn; ++s) {
      if (s < slot_count) {
        tile.merge.split_weight[s][row] = expf(tile.merge.split_weight[s][row] - safe_max);
        merged_sum +=
            tile.merge.split_weight[s][row] *
            __ldcg(&p.partial_sum_device[(first_slot + s * slot_stride) * kMaxRows + row]);
      }
    }
    tile.row_max[row] = merged_max;
    tile.row_sum[row] = merged_sum;
  }
  __syncthreads();
  for (int vector = threadIdx.x; vector < row_count * kVectorsPerRow; vector += kThreads) {
    const int row = vector / kVectorsPerRow;
    float4 merged = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int s = 0; s < kMergeFanIn; ++s) {
      if (s < slot_count) {
        const float4* slot_out = reinterpret_cast<const float4*>(
            p.partial_out_device + (first_slot + s * slot_stride) * kMaxRows * kHeadDim);
        const float4 x = __ldcg(&slot_out[vector]);
        const float weight = tile.merge.split_weight[s][row];
        merged = make_float4(merged.x + weight * x.x, merged.y + weight * x.y,
                             merged.z + weight * x.z, merged.w + weight * x.w);
      }
    }
    reinterpret_cast<float4*>(&tile.out[0][0])[vector] = merged;
  }
  __syncthreads();
}

/**
 * @brief `merge_splits` merges every split's softmax state with a two-level tree: the last block
 * of each group of 16 splits merges its group, and the last group merger merges the groups.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile with this block's state; the block that finishes the
 * tree gets the merged state.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] split This block's split.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @return Whether this block finished the tree and must write the output.
 *
 * @par Example
 * @code
 * const bool is_merging_block = merge_splits(p, tile, b, kv_head, split, row_count);
 * @endcode
 */
__device__ bool merge_splits(const Params& p, SharedTile& tile, int b, int kv_head, int split,
                             int row_count) {
  const int group = b * p.kv_head_count + kv_head;
  const int64_t first_slot = static_cast<int64_t>(group) * p.split_count;
  const int merge_group = split / kMergeFanIn;
  const int merge_group_count = (p.split_count + kMergeFanIn - 1) / kMergeFanIn;
  const int group_slot_count = min(kMergeFanIn, p.split_count - merge_group * kMergeFanIn);
  publish_state(p, tile, first_slot + split, row_count);
  if (!arrive(p, tile, group * kCountersPerGroup + merge_group, group_slot_count)) {
    return false;
  }
  merge_slots(p, tile, first_slot + merge_group * kMergeFanIn, 1, group_slot_count, row_count);
  if (threadIdx.x == 0) {
    p.arrival_count_device[group * kCountersPerGroup + merge_group] = 0;
  }
  if (merge_group_count == 1) {
    return true;
  }
  publish_state(p, tile, first_slot + merge_group * kMergeFanIn, row_count);
  if (!arrive(p, tile, group * kCountersPerGroup + kMaxMergeGroups, merge_group_count)) {
    return false;
  }
  merge_slots(p, tile, first_slot, kMergeFanIn, merge_group_count, row_count);
  if (threadIdx.x == 0) {
    p.arrival_count_device[group * kCountersPerGroup + kMaxMergeGroups] = 0;
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
 * attention_kernel<<<dim3(split_count, kv_head_count, B), kThreads, sizeof(SharedTile),
 * stream>>>(p);
 * @endcode
 */
__global__ void __launch_bounds__(kThreads) attention_kernel(const Params p) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  SharedTile& tile = *reinterpret_cast<SharedTile*>(shared_bytes);
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int b = blockIdx.z;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int group_size = p.q_head_count / p.kv_head_count;
  const int row_count = p.L * group_size;
  const int m_tile_count = (row_count + kMmaRows - 1) / kMmaRows;
  const int key_group_count = kWarps / m_tile_count;
  const int m_tile = warp % m_tile_count;
  const int key_group = warp / m_tile_count;
  const bool is_active = key_group < key_group_count;
  const int seq_len = p.seq_lens_device[b];
  const int window_start_incl = p.window == 0 ? 0 : max(0, seq_len - p.window + 1);
  const int first_position_incl = window_start_incl + split * p.keys_per_split;
  const int end_position_excl = min(first_position_incl + p.keys_per_split, seq_len);
  start_rows_copy(p, tile.start.q_in, b, kv_head, row_count, 0);
  KeyAddresses addresses = locate_keys(p, b, kv_head, first_position_incl, end_position_excl);
  for (int stage = 0; stage < kStages - 1; ++stage) {
    const KeyAddresses next_addresses = locate_keys(
        p, b, kv_head, first_position_incl + (stage + 1) * kStepKeys, end_position_excl);
    start_step_copy(p, tile, kv_head, addresses, stage);
    addresses = next_addresses;
  }
  fill_rotation_table(p, tile, seq_len);
  wait_for_copies<kStages - 1>();
  __syncthreads();
  for (int row = warp; row < m_tile_count * kMmaRows; row += kWarps) {
    float2 q = make_float2(0.0f, 0.0f);
    if (row < row_count) {
      const float2 q_in = __bfloat1622float2(
          reinterpret_cast<const __nv_bfloat162*>(&tile.start.q_in[row][0])[lane]);
      q = normalize_and_rotate(q_in, p.qk_scale_factor, tile.rotation[row / group_size][lane]);
    }
    const __nv_bfloat162 q_bf16 = __float22bfloat162_rn(q);
    reinterpret_cast<__half2*>(&tile.start.q[row][0])[lane] =
        __float22half2_rn(__bfloat1622float2(q_bf16));
  }
  __syncthreads();
  uint32_t q_fragments[kDepthSteps][4];
  for (int depth = 0; depth < kDepthSteps; ++depth) {
    load_matrices<false>(q_fragments[depth],
                         &tile.start.q[m_tile * kMmaRows + 8 * ((lane / 8) % 2) + lane % 8]
                                      [kMmaDepth * depth + 8 * (lane / 16)]);
  }
  WarpState state = {};
  state.max_log2[0] = state.max_log2[1] = -INFINITY;
  const float logit_scale_log2 = p.sm_scale * *p.k_scale_device * kLog2e;
  const int first_row = m_tile * kMmaRows;
  for (int step = 0; first_position_incl + step * kStepKeys < end_position_excl; ++step) {
    const KeyAddresses next_addresses = locate_keys(
        p, b, kv_head, first_position_incl + (step + kStages) * kStepKeys, end_position_excl);
    start_step_copy(p, tile, kv_head, addresses, (step + kStages - 1) % kStages);
    addresses = next_addresses;
    wait_for_copies<kStages - 1>();
    __syncthreads();
    convert_step(tile, step % kStages);
    __syncthreads();
    const int start = first_position_incl + step * kStepKeys;
    const bool is_masked = start + kStepKeys > end_position_excl ||
                           (p.window != 0 && start < window_start_incl + p.L - 1);
    if (is_active) {
      attend_step(p, tile, q_fragments, key_group, key_group_count, first_row, seq_len,
                  logit_scale_log2, is_masked, state);
    }
  }
  wait_for_copies<0>();
  __syncthreads();
  if (split == 0) {
    store_new_keys(p, tile, b, kv_head, seq_len);
    __syncthreads();
    if (is_active) {
      attend_step(p, tile, q_fragments, key_group, key_group_count, first_row, seq_len,
                  logit_scale_log2, true, state);
    }
  }
  __syncthreads();
  for (int half = 0; half < 2 && is_active; ++half) {
    state.sum[half] += __shfl_xor_sync(0xffffffffu, state.sum[half], 1);
    state.sum[half] += __shfl_xor_sync(0xffffffffu, state.sum[half], 2);
    const int row = lane / 4 + 8 * half;
    if (lane % 4 == 0) {
      tile.warp_max[warp][row] = state.max_log2[half];
      tile.warp_sum[warp][row] = state.sum[half];
    }
    for (int d = 0; d < kDimTiles; ++d) {
      tile.warp_out[warp][row][d * kMmaCols + 2 * (lane % 4)] = state.out[d][2 * half];
      tile.warp_out[warp][row][d * kMmaCols + 2 * (lane % 4) + 1] = state.out[d][2 * half + 1];
    }
  }
  __syncthreads();
  combine_warps(tile, row_count, m_tile_count, key_group_count);
  __syncthreads();
  start_rows_copy(p, tile.merge.gate, b, kv_head, row_count,
                  (p.q_head_count + 2 * p.kv_head_count) * kHeadDim);
  const bool is_merging_block = merge_splits(p, tile, b, kv_head, split, row_count);
  wait_for_copies<0>();
  __syncthreads();
  if (is_merging_block) {
    write_gated_output(p, tile, b, kv_head, row_count);
  }
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
 * @param[in,out] arrival_count int32 [B, Hkv, 17], zero before and after each call.
 * @param[in] keys_per_split Cached keys per split, a multiple of 64.
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
  TORCH_CHECK(qkv.size(1) <= kMaxTokens, "L must be <= ", kMaxTokens);
  TORCH_CHECK(partial_max.size(2) <= kMaxSplits, "at most ", kMaxSplits, " splits");
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
  C10_CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      sizeof(SharedTile)));
  attention_kernel<<<dim3(p.split_count, k_cache.size(2), qkv.size(0)), kThreads,
                     sizeof(SharedTile), at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
/**
 * @brief `resident_blocks_per_sm` returns how many blocks of `attention_kernel` fit on one SM.
 *
 * @return The occupancy calculator's count, for the kernel's threads and shared memory.
 *
 * @par Example
 * @code
 * int64_t slots = resident_blocks_per_sm() * sm_count;
 * @endcode
 */
int64_t resident_blocks_per_sm() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      sizeof(SharedTile)));
  int block_count = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_count, attention_kernel,
                                                               kThreads, sizeof(SharedTile)));
  return block_count;
}
"""
CPP_SOURCE: str = (
    "void run(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, "
    "torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor cache_loc, "
    "torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor out, "
    "int64_t window, double qk_scale_factor, double sm_scale, torch::Tensor partial_out, "
    "torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor arrival_count, "
    "int64_t keys_per_split); int64_t resident_blocks_per_sm();"
)
HEAD_DIM: int = 64
ROPE_PAIR_STRIDE: int = 2
MAX_ROWS: int = 64
TILE_KEYS: int = 64
MAX_SPLITS: int = 256
COUNTERS_PER_GROUP: int = 17
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
            name="muse_fused_attention_step_softmax",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["run", "resident_blocks_per_sm"],
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
        The split count and the cached keys per split (a multiple of 64), so that every key a
        request can hold falls in a split and the grid fits in one wave of resident blocks.

    Example:
        >>> split_count, keys_per_split = _plan_splits(data)
    """
    config: dict = data["config"]
    span_keys: int = config["window"] or data["page_table"].shape[1] * config["page_size"]
    tile_count: int = -(-span_keys // TILE_KEYS)
    sm_count: int = torch.cuda.get_device_properties(data["qkv"].device).multi_processor_count
    slot_count: int = sm_count * _get_extension().resident_blocks_per_sm()
    group_count: int = config["batch"] * config["num_kv_heads"]
    wanted_split_count: int = max(1, slot_count // group_count)
    wanted_split_count = min(wanted_split_count, MAX_SPLITS)
    keys_per_split: int = -(-tile_count // min(tile_count, wanted_split_count)) * TILE_KEYS
    return -(-span_keys // keys_per_split), keys_per_split


def _get_workspace(data: dict, split_count: int) -> tuple[torch.Tensor, ...]:
    """`_get_workspace` returns the per-split state buffers and zeroed arrival counters.

    Args:
        data: The inputs; only `config` and the device are read.
        split_count: Splits per request and KV head.

    Returns:
        partial_out [B, Hkv, splits, 64, 64], partial_max and partial_sum [B, Hkv, splits, 64]
        (float32), and arrival_count [B, Hkv, 17] (int32, zero), allocated once per shape.

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
            torch.zeros((*shape[:2], COUNTERS_PER_GROUP), dtype=torch.int32, device=device),
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
