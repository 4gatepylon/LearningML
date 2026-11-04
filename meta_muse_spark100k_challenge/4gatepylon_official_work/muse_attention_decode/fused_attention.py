# okc: problem=muse_attention_decode
"""`fused_attention.py` computes muse_attention_decode in one readable CUDA kernel.

Each request's keys are divided into splits, and one thread block handles one split of one request
and one key/value head, for every query head of that group and every token of the block. A block
has 4 warps, or 8 for verify on sliding layers or with few requests. It:

1. Copies its query rows to shared memory with `cp.async`, RMS-normalizes them, applies RoPE from
   a per-token cos/sin table and rounds them to FP8 E4M3, as FA3 does (`qk_scale_factor` joins
   the softmax scale).
2. Streams its split of cached FP8 keys and values 64 at a time through a 3-stage `cp.async`
   ring in shared memory, whose space the merge reuses afterwards. Each warp attends its 16
   query rows over its share of the step's keys: S = Q K^T runs on FP8 tensor cores straight
   from the ring, the values are converted to fp16 for O += P V, and the online softmax updates
   once per step. Only steps at the edge of the window or split apply the mask. In verify, each
   warpgroup of 4 warps runs both products on `wgmma` over its keys (all 64, or 32 of them with
   8 warps), so the keys and the fp16 values sit in `wgmma`'s 64- and 128-byte swizzles.
3. In the first split's block only: normalizes the block's new keys (staged in shared memory at
   the start), applies RoPE, writes the new keys and values to the FP8 cache, and attends to them
   as one last, causal step.
4. Combines its warps' softmax states, starts copying its gate rows, and publishes the block's
   state. In decode, a two-level tree merges the splits: the last block of each group of up to 16
   splits merges that group, then the last group merger merges the groups, divides by the
   softmax sum, and applies the per-head RMSNorm and the sigmoid gate. In verify, the last blocks
   to arrive (up to 16) wait for the rest, then each merges and writes its own slice of the rows.

Interface:
    kernel_fn(data): `TASK.md`'s timed call; returns bfloat16 [B, L, Hq * 64].

Example:
    >>> out = kernel_fn(reference.generate_input(1, 2048, "tp8", "sliding", seed=1))

Must know:
    Limitations:
        - The verify builds use `wgmma`, so the kernel builds for and runs on Hopper (sm_90a).
        - A block needs at most 16 query rows (decode) or 49 to 64 (verify) per KV head; both head
          layouts in `TASK.md` give 8 or 64.
    Gotchas:
        - The arrival counters live across calls; each merging block resets its counter to 0.
        - Verify launches cooperatively, so all of its blocks are resident while mergers wait.
    Performance:
        - The page-table reads for a step are issued one step before its keys and values.
        - The copies fetch whole 128-byte L2 lines, which hold a token's keys (or values) for both
          KV heads, so the other head's block hits in L2.
        - The split count fills one wave of resident blocks: the occupancy calculator's count,
          capped at each build's intended blocks per SM.
        - Each build compiles one path: decode (4 warps, 1 chunk per warp per step, 3 blocks per
          SM), full verify (4 warps, 4 chunks, 4 blocks per SM) or verify on sliding layers or with
          at most 8 request x KV-head groups (8 warps, 2 chunks, 2 blocks per SM).
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
constexpr int kMinThreads = 128;
constexpr int kMaxWarps = 8;
constexpr int kStepKeys = 64;
constexpr int kMmaRows = 16;
constexpr int kMmaCols = 8;
constexpr int kChunkKeys = 16;
constexpr int kDimTiles = kHeadDim / kMmaCols;
constexpr int kRawStride = kHeadDim + 16;
constexpr int kFp8MmaDepth = 32;
constexpr int kQkSteps = kHeadDim / kFp8MmaDepth;
constexpr int kBytesPerLoad = 16;
constexpr int kSwizzleRows = 8;
constexpr int kSwizzleAtomBytes = kSwizzleRows * kHeadDim;
constexpr uint64_t kSwizzle64BytesMode = 2;
constexpr uint64_t kSwizzle128BytesMode = 1;
constexpr int kHalvesPerLoad = kBytesPerLoad / 2;
constexpr int kValueAtomBytes = kSwizzleRows * kHeadDim * 2;
constexpr int kHalfMmaDepth = 16;
constexpr int kLoadsPerRow = kHeadDim / kBytesPerLoad;
constexpr int kPiecesPerStep = kStepKeys * kLoadsPerRow;
constexpr int kLoadsPerThread = kPiecesPerStep / kMinThreads;
constexpr int kMaxSplits = 256;
constexpr int kStages = 3;
constexpr int kDecodeBlocksPerSm = 3;
constexpr int kVerifyBlocksPerSm = 4;
constexpr int kWideVerifyBlocksPerSm = 2;
constexpr int kMaxWideFullGroups = 8;
constexpr int kMaxTokens = 8;
constexpr int kMergeFanIn = 16;
constexpr int kMaxMergeGroups = kMaxSplits / kMergeFanIn;
constexpr int kCountersPerGroup = kMaxMergeGroups + 1;
constexpr int kMaxMergers = 16;
constexpr int kMaxMergeWeights = kMaxSplits * kMaxRows / kMaxMergers;
constexpr int kMaxThreads = kMaxWarps * kWarpSize;
constexpr int kFloatsPerVector = 4;
constexpr int kVectorsPerRow = kHeadDim / kFloatsPerVector;
constexpr int kPiecesPerRow = kHeadDim * 2 / kBytesPerLoad;
constexpr float kRmsEps = 1e-5f;
constexpr int kLanesPerOutputRow = 8;
constexpr int kDimsPerOutputLane = kHeadDim / kLanesPerOutputRow;
constexpr float kLog2e = 1.4426950408889634f;

/**
 * @brief `Params` carries the kernel's inputs, outputs, shapes and workspace, built once by `run`.
 *
 * @par Example
 * @code
 * attention_kernel<4, 1, kDecodeBlocksPerSm><<<grid, kMinThreads, sizeof(SharedTile)>>>(p);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Pointers ending in `_device` point to device memory; `run` fills them from its tensors.
 */
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
  int page_size_log2, page_table_stride, window;
  float sm_scale;
  float* partial_out_device;
  float* partial_max_device;
  float* partial_sum_device;
  int* arrival_count_device;
  int split_count, keys_per_split;
};

/**
 * @brief `StartTile` holds the block's bf16 query rows as copied and their FP8 E4M3 version, padded
 * rows that `ldmatrix` turns into the query fragments.
 *
 * @par Example
 * @code
 * normalize_query_row(tile, tile.start.q[row], row, row_count, group_size, first_dim);
 * @endcode
 */
struct StartTile {
  __nv_bfloat16 q_in[kMaxRows][kHeadDim];
  uint8_t q[kMaxRows][kRawStride];
};

/**
 * @brief `RingTile` is the `cp.async` ring: per stage, a step's 64 FP8 keys (one 64-byte row each,
 * in `wgmma`'s 64-byte swizzle), their FP8 values and their positions (-1 for a missing key).
 *
 * @par Example
 * @code
 * copy_async_16(&tile.ring.raw_k[stage][key][swizzled_dim(key, dim)], source, is_present);
 * @endcode
 */
struct RingTile {
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN WGMMA WILL READ WRONGLY SWIZZLED KEYS
  alignas(kSwizzleAtomBytes) uint8_t raw_k[kStages][kStepKeys][kHeadDim];
  uint8_t raw_v[kStages][kStepKeys][kHeadDim];
  int raw_position[kStages][kStepKeys];
};

/**
 * @brief `MergeTile` holds what the merge and the output need after the loop: each merged slot's
 * row maximum (then its weight) and softmax sum per row (`split_*` for the decode tree, `row_*`
 * for the verify mergers' rows), the verify mergers' partial sums, and the block's gate rows.
 *
 * @par Example
 * @code
 * merge_slots(p, tile, first_slot, 1, slot_count, row_count);
 * @endcode
 */
struct MergeTile {
  float split_weight[kMergeFanIn][kMaxRows];
  float split_sum[kMergeFanIn][kMaxRows];
  float row_weight[kMaxMergeWeights];
  float row_split_sum[kMaxMergeWeights];
  float4 partial_sums[kMaxThreads];
  __nv_bfloat16 gate[kMaxRows][kHeadDim];
};

/**
 * @brief `SharedTile` lays out the block's dynamic shared memory. Its two unions share space
 * between phases: the query rows, then the step's fp16 values, then the merged outputs; and the
 * ring, then the merge tile and the warps' outputs past 64 rows.
 *
 * @par Example
 * @code
 * SharedTile& tile = *reinterpret_cast<SharedTile*>(shared_bytes);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Members that `cp.async` fills must stay 16-byte aligned: add new scalars at the end.
 */
struct SharedTile {
  union {
    StartTile start;
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN WGMMA WILL READ WRONGLY SWIZZLED VALUES
    alignas(kValueAtomBytes) __half v[kStepKeys][kHeadDim];
    float out[kMaxRows][kHeadDim];
  };
  union {
    RingTile ring;
    MergeTile merge;
    float spill_out[kMaxRows][kHeadDim];
  };
  float2 rotation[kMaxTokens][kHeadDim / 2];
  __nv_bfloat16 new_k_in[kMaxTokens][kHeadDim];
  __nv_bfloat16 new_v_in[kMaxTokens][kHeadDim];
  int key_position[kStepKeys];
  float warp_max[kMaxWarps][kMmaRows];
  float warp_sum[kMaxWarps][kMmaRows];
  float row_max[kMaxRows];
  float row_sum[kMaxRows];
  bool is_last_split;
  int merge_rank;
  int64_t new_cache_loc[kMaxTokens];
  float k_scale;
  float v_scale;
};

/**
 * @brief `KeyAddresses` holds this thread's page and position for each piece of one step's
 * copies, loaded a step before the copies start.
 *
 * @par Example
 * @code
 * const KeyAddresses addresses = locate_keys<128>(p, b, kv_head, first_position, end_position);
 * @endcode
 */
struct KeyAddresses {
  int page[kLoadsPerThread];
  int position[kLoadsPerThread];
};

/**
 * @brief `WarpState` is one thread's part of its warp's online softmax: unnormalized outputs (in
 * the `mma.sync` C layout of 8 dimension tiles), and the running max (base-2 logits) and sum of
 * its two rows.
 *
 * @par Example
 * @code
 * WarpState state = {};
 * @endcode
 */
struct WarpState {
  float out[kDimTiles][4];
  float max_log2[2];
  float sum[2];
};

/**
 * @brief `thread_count` returns the block's thread count, which depends on the kernel build.
 *
 * @return `blockDim.x`.
 *
 * @par Example
 * @code
 * for (int row = threadIdx.x; row < row_count; row += thread_count()) { ... }
 * @endcode
 */
__device__ int thread_count() { return static_cast<int>(blockDim.x); }

/**
 * @brief `warp_count` returns the block's warp count, which depends on the kernel build.
 *
 * @return `blockDim.x / 32`.
 *
 * @par Example
 * @code
 * for (int row = warp; row < row_count; row += warp_count()) { ... }
 * @endcode
 */
__device__ int warp_count() { return thread_count() / kWarpSize; }

/**
 * @brief `warp_out_row` returns where warp `warp` keeps row `row_in_tile` of its unnormalized
 * outputs: rows 16 w to 16 w + 15 of `out`, continued in `spill_out` past 64 rows.
 *
 * @param[in] tile Shared memory tile.
 * @param[in] warp Warp index.
 * @param[in] row_in_tile Row within the warp's 16 query rows.
 * @return The row's 64 floats.
 *
 * @par Example
 * @code
 * warp_out_row(tile, warp, row)[d] = state.out[d / 8][...];
 * @endcode
 */
__device__ float* warp_out_row(SharedTile& tile, int warp, int row_in_tile) {
  const int row = warp * kMmaRows + row_in_tile;
  return row < kMaxRows ? tile.out[row] : tile.spill_out[row - kMaxRows];
}

/**
 * @brief `lane_group_sum` returns the sum of `x` over each aligned group of `kLaneCount` lanes, in
 * every lane of the group.
 *
 * @tparam kLaneCount Lanes per group, a power of 2 up to 32.
 * @param[in] x This lane's value.
 * @return The group's sum.
 *
 * @par Example
 * @code
 * const float row_total = lane_group_sum<8>(partial);
 * @endcode
 */
template <int kLaneCount>
__device__ float lane_group_sum(float x) {
  for (int offset = kLaneCount / 2; offset > 0; offset /= 2) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  }
  return x;
}

/**
 * @brief `normalize_and_rotate` RMS-normalizes one 64-wide head vector held by a warp and applies
 * interleaved RoPE.
 *
 * @param[in] x This lane's pair of elements (2 * lane, 2 * lane + 1).
 * @param[in] rotation This pair's (cos, sin) at the token's position; (1, 0) for no RoPE.
 * @return This lane's normalized, rotated pair.
 *
 * @par Example
 * @code
 * const float2 k = normalize_and_rotate(k_in, tile.rotation[key][lane]);
 * @endcode
 */
__device__ float2 normalize_and_rotate(float2 x, float2 rotation) {
  const float inv_rms =
      rsqrtf(lane_group_sum<kWarpSize>(x.x * x.x + x.y * x.y) / kHeadDim + kRmsEps);
  x = make_float2(x.x * inv_rms, x.y * inv_rms);
  return make_float2(x.x * rotation.x - x.y * rotation.y, x.x * rotation.y + x.y * rotation.x);
}

/**
 * @brief `write_gated_output` divides each query row's merged output by its softmax sum, applies
 * the per-head RMSNorm and the sigmoid gate, and writes the result as bf16. Each row takes 8 lanes
 * of 8 dimensions, which load and store 16 bytes each.
 *
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the merged, unnormalized outputs, row sums and gates.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @param[in] first_row First row this block writes.
 * @param[in] row_step Distance between the rows this block writes.
 *
 * @par Example
 * @code
 * write_gated_output(p, tile, b, kv_head, row_count, merge_rank, merger_count);
 * @endcode
 */
__device__ void write_gated_output(const Params& p, const SharedTile& tile, int b, int kv_head,
                                   int row_count, int first_row, int row_step) {
  const int group_size = p.q_head_count / p.kv_head_count;
  const int rows_per_warp = kWarpSize / kLanesPerOutputRow;
  const int lane = threadIdx.x % kWarpSize;
  const int first_dim = lane % kLanesPerOutputRow * kDimsPerOutputLane;
  const int my_row_count = (row_count - first_row + row_step - 1) / row_step;
  for (int warp_row = threadIdx.x / kWarpSize * rows_per_warp; warp_row < my_row_count;
       warp_row += warp_count() * rows_per_warp) {
    const int row =
        first_row + min(warp_row + lane / kLanesPerOutputRow, my_row_count - 1) * row_step;
    const float scale = tile.v_scale / tile.row_sum[row];
    float o[kDimsPerOutputLane];
    float square_sum = 0.0f;
    for (int i = 0; i < kDimsPerOutputLane; ++i) {
      o[i] = tile.out[row][first_dim + i] * scale;
      square_sum += o[i] * o[i];
    }
    const float inv_rms =
        rsqrtf(lane_group_sum<kLanesPerOutputRow>(square_sum) / kHeadDim + kRmsEps);
    const uint4 gate_bits = *reinterpret_cast<const uint4*>(&tile.merge.gate[row][first_dim]);
    const __nv_bfloat162* gates = reinterpret_cast<const __nv_bfloat162*>(&gate_bits);
    uint4 out_bits;
    __nv_bfloat162* outs = reinterpret_cast<__nv_bfloat162*>(&out_bits);
    for (int i = 0; i < kDimsPerOutputLane / 2; ++i) {
      const float2 g = __bfloat1622float2(gates[i]);
      outs[i] = __float22bfloat162_rn(make_float2(o[2 * i] * inv_rms / (1.0f + expf(-g.x)),
                                                  o[2 * i + 1] * inv_rms / (1.0f + expf(-g.y))));
    }
    if (warp_row + lane / kLanesPerOutputRow < my_row_count) {
      const int token = b * p.L + row / group_size;
      const int head = kv_head * group_size + row % group_size;
      *reinterpret_cast<uint4*>(p.out_device + (token * p.q_head_count + head) * kHeadDim +
                                first_dim) = out_bits;
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
 * mma_16x8x16(state.out[2 * pair], p_fragment, b[0], b[1]);
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
 * @brief `mma_16x8x32_fp8` accumulates D += A B on tensor cores for one warp: A is 16 x 32 and B is
 * 32 x 8, both FP8 E4M3, in the `mma.sync.m16n8k32` fragment layouts; D is fp32.
 *
 * @param[in,out] d This thread's 4 accumulator elements.
 * @param[in] a This thread's A fragment (4 registers of 4 FP8 values).
 * @param[in] b0 First register of this thread's B fragment.
 * @param[in] b1 Second register of this thread's B fragment.
 *
 * @par Example
 * @code
 * mma_16x8x32_fp8(scores[n], q_fragments[s], b[0], b[1]);
 * @endcode
 */
__device__ void mma_16x8x32_fp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/**
 * @brief `swizzled_dim` returns where byte `dim` of key `key` sits in the key's 64-byte row, in
 * `wgmma`'s 64-byte swizzle: the row's 16-byte pieces are permuted by `piece ^ (key / 2 % 4)`.
 *
 * @param[in] key Key within the step.
 * @param[in] dim Byte (head dimension) within the key.
 * @return The byte within the key's row.
 *
 * @par Example
 * @code
 * copy_async_16(&tile.ring.raw_k[stage][key][swizzled_dim(key, dim)], source, is_present);
 * @endcode
 *
 * @par Must know
 * - Performance:
 *   - Each key keeps its own 64-byte row, so `cp.async` and `ldmatrix` stay at full speed.
 */
__device__ int swizzled_dim(int key, int dim) {
  const int piece = (dim / kBytesPerLoad) ^ (key / 2 % kLoadsPerRow);
  return piece * kBytesPerLoad + dim % kBytesPerLoad;
}

/**
 * @brief `make_key_descriptor` builds the `wgmma` shared-memory descriptor of 64 FP8 keys x 32 head
 * dimensions in the 64-byte swizzle of `swizzled_dim`.
 *
 * @param[in] first_byte The 32 dimensions' first byte in key 0's row.
 * @return The 64-bit descriptor.
 *
 * @par Example
 * @code
 * const uint64_t descriptor = make_key_descriptor(&keys[0][kFp8MmaDepth * depth]);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The step's keys must start at a multiple of `kSwizzleAtomBytes`, where the swizzle repeats.
 */
__device__ uint64_t make_key_descriptor(const uint8_t* first_byte) {
  const uint64_t address = static_cast<uint32_t>(__cvta_generic_to_shared(first_byte));
  const uint64_t unused_leading_bytes = 16;
  return ((address & 0x3FFFF) >> 4) | ((unused_leading_bytes >> 4) << 16) |
         (static_cast<uint64_t>(kSwizzleAtomBytes >> 4) << 32) | (kSwizzle64BytesMode << 62);
}

/**
 * @brief `wgmma_64x64x32_fp8` starts D (+)= A B for one warpgroup: A is 64 x 32 FP8 query rows in
 * registers (each warp its 16 rows, in the `mma.sync` A layout), B is 32 x 64 FP8 keys in shared
 * memory, D is 64 x 64 fp32 (each thread 32 values in the `mma.sync` C layout of 8 n-tiles).
 *
 * @param[in,out] d This thread's 32 accumulators.
 * @param[in] a This thread's A fragment.
 * @param[in] key_descriptor `make_key_descriptor` of the 32 dimensions.
 * @param[in] should_accumulate Whether to add to D (otherwise D = A B).
 *
 * @par Example
 * @code
 * wgmma_64x64x32_fp8(scores, q_fragments[0], descriptor, false);
 * @endcode
 */
__device__ void wgmma_64x64x32_fp8(float (&d)[32], const uint32_t (&a)[4], uint64_t key_descriptor,
                                   bool should_accumulate) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n.reg .pred accumulate;\nsetp.ne.b32 accumulate, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
      "%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, "
      "%27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, "
      "accumulate, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(key_descriptor),
        "r"(static_cast<int>(should_accumulate)));
#else
  __trap();
#endif
}

/**
 * @brief `wgmma_64x32x32_fp8` is `wgmma_64x64x32_fp8` for 32 keys: D is 64 x 32 fp32, 16 values
 * per thread in the `mma.sync` C layout of 4 n-tiles.
 *
 * @param[in,out] d This thread's 16 accumulators.
 * @param[in] a This thread's A fragment.
 * @param[in] key_descriptor `make_key_descriptor` of the 32 keys' 32 dimensions.
 * @param[in] should_accumulate Whether to add to D (otherwise D = A B).
 *
 * @par Example
 * @code
 * wgmma_64x32x32_fp8(scores, q_fragments[0], descriptor, false);
 * @endcode
 */
__device__ void wgmma_64x32x32_fp8(float (&d)[16], const uint32_t (&a)[4], uint64_t key_descriptor,
                                   bool should_accumulate) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n.reg .pred accumulate;\nsetp.ne.b32 accumulate, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
      "%9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, "
      "accumulate, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(key_descriptor),
        "r"(static_cast<int>(should_accumulate)));
#else
  __trap();
#endif
}

/**
 * @brief `score_keys_wgmma` computes S = Q K^T for the warpgroup's 64 query rows and `kKeyCount`
 * keys on `wgmma`, and waits for the result.
 *
 * @tparam kKeyCount Keys: 64 (the whole step) or 32.
 * @param[out] scores This thread's scores, in the `mma.sync` C layout of n-tiles of 8 keys.
 * @param[in] q_fragments This thread's FP8 query fragments, one per 32 head dimensions.
 * @param[in] keys The first key's row of the step's FP8 keys, swizzled by `swizzled_dim`.
 *
 * @par Example
 * @code
 * score_keys_wgmma<64>(accumulators, q_fragments, keys);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - All 128 threads of the warpgroup must call `score_keys_wgmma` together.
 */
template <int kKeyCount>
__device__ void score_keys_wgmma(float (&scores)[kKeyCount / 2],
                                 const uint32_t (&q_fragments)[kQkSteps][4],
                                 const uint8_t (*keys)[kHeadDim]) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  for (int depth = 0; depth < kQkSteps; ++depth) {
    const uint64_t descriptor = make_key_descriptor(&keys[0][kFp8MmaDepth * depth]);
    if constexpr (kKeyCount == kStepKeys) {
      wgmma_64x64x32_fp8(scores, q_fragments[depth], descriptor, depth > 0);
    } else {
      static_assert(kKeyCount == kStepKeys / 2, "wgmma scores 64 or 32 keys");
      wgmma_64x32x32_fp8(scores, q_fragments[depth], descriptor, depth > 0);
    }
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#else
  __trap();
#endif
}

/**
 * @brief `swizzled_value_dim` returns where value `dim` of key `key` sits in the key's 128-byte row
 * of fp16 values, in `wgmma`'s 128-byte swizzle: the row's 16-byte pieces are permuted by
 * `piece ^ (key % 8)`.
 *
 * @param[in] key Key within the step.
 * @param[in] dim Head dimension.
 * @return The half within the key's row.
 *
 * @par Example
 * @code
 * load_matrices<true>(b, &tile.v[key][swizzled_value_dim(key, dim)]);
 * @endcode
 */
__device__ int swizzled_value_dim(int key, int dim) {
  return ((dim / kHalvesPerLoad) ^ (key % kSwizzleRows)) * kHalvesPerLoad + dim % kHalvesPerLoad;
}

/**
 * @brief `make_value_descriptor` builds the `wgmma` shared-memory descriptor of 16 keys x 64 fp16
 * values, one 128-byte row per key (MN-major), in the 128-byte swizzle of `swizzled_value_dim`.
 *
 * @param[in] first_key_row The first of the 16 keys' rows.
 * @return The 64-bit descriptor.
 *
 * @par Example
 * @code
 * const uint64_t descriptor = make_value_descriptor(tile.v[kHalfMmaDepth * chunk]);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - `tile.v` must start at a multiple of `kValueAtomBytes`, where the swizzle repeats.
 */
__device__ uint64_t make_value_descriptor(const __half* first_key_row) {
  const uint64_t address = static_cast<uint32_t>(__cvta_generic_to_shared(first_key_row));
  const uint64_t unused_leading_bytes = 16;
  return ((address & 0x3FFFF) >> 4) | ((unused_leading_bytes >> 4) << 16) |
         (static_cast<uint64_t>(kValueAtomBytes >> 4) << 32) | (kSwizzle128BytesMode << 62);
}

/**
 * @brief `wgmma_64x64x16_f16` starts D += A B for one warpgroup: A is 64 x 16 fp16 probabilities in
 * registers (each warp its 16 rows, in the `mma.sync` A layout), B is 16 keys x 64 fp16 values in
 * shared memory, D is 64 x 64 fp32 (each thread 32 values in the `mma.sync` C layout).
 *
 * @param[in,out] d This thread's 32 accumulators.
 * @param[in] a This thread's A fragment.
 * @param[in] value_descriptor `make_value_descriptor` of the 16 keys.
 *
 * @par Example
 * @code
 * wgmma_64x64x16_f16(out, p_fragment, make_value_descriptor(tile.v[0]));
 * @endcode
 */
__device__ void wgmma_64x64x16_f16(float (&d)[32], const uint32_t (&a)[4],
                                   uint64_t value_descriptor) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
      "%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, "
      "%27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(value_descriptor));
#else
  __trap();
#endif
}

/**
 * @brief `load_matrices` loads four 8 x 8 matrices of 16-bit elements (fp16 values, or pairs of
 * FP8 bytes) from shared memory into one register each per thread (`ldmatrix.x4`), optionally
 * transposed.
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
__device__ void load_matrices(uint32_t (&r)[4], const void* row_address) {
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
 * @brief `fast_exp2` returns 2^x from the special-function unit (`ex2.approx.ftz`), with
 * denormal results flushed to 0.
 *
 * @param[in] x The exponent.
 * @return 2^x, or 0 for x = -inf.
 *
 * @par Example
 * @code
 * const float probability = fast_exp2(fmaf(score, logit_scale_log2, -max_log2));
 * @endcode
 */
__device__ float fast_exp2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(x));
  return result;
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
 * @param[out] low The first 8 fp16 values, in shared memory, 16-byte aligned.
 * @param[out] high The last 8 fp16 values, in shared memory, 16-byte aligned.
 *
 * @par Example
 * @code
 * store_fp8_as_halves(raw_bytes, &row[swizzled_value_dim(key, dim)],
 *                     &row[swizzled_value_dim(key, dim + 8)]);
 * @endcode
 */
__device__ void store_fp8_as_halves(const uint4& bytes, __half* low, __half* high) {
  const __nv_fp8x2_storage_t* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&bytes);
  uint32_t halves[kHalvesPerLoad];
  for (int i = 0; i < kHalvesPerLoad; ++i) {
    const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(pairs[i], __NV_E4M3);
    halves[i] = static_cast<uint32_t>(pair.x) | (static_cast<uint32_t>(pair.y) << 16);
  }
  *reinterpret_cast<uint4*>(low) = make_uint4(halves[0], halves[1], halves[2], halves[3]);
  *reinterpret_cast<uint4*>(high) = make_uint4(halves[4], halves[5], halves[6], halves[7]);
}

/**
 * @brief `locate_keys` starts reading the page table for up to 64 keys of one request and KV head;
 * missing keys get position -1.
 *
 * @tparam kThreadCount Threads per block, so the piece loop has fixed bounds.
 * @param[in] p Kernel parameters.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 * @param[in] first_position_incl First key position of the step.
 * @param[in] end_position_excl Keys at or after this position are missing.
 * @return The pages and positions of this thread's pieces (thread t's piece i is 16 bytes of key
 * (kThreadCount i + t) / 4); the pages are still in flight: nothing here uses them, so the thread
 * does not wait for the reads.
 *
 * @par Example
 * @code
 * KeyAddresses addresses = locate_keys<128>(p, b, kv_head, start, end);
 * @endcode
 */
template <int kThreadCount>
__device__ KeyAddresses locate_keys(const Params& p, int b, int kv_head, int first_position_incl,
                                    int end_position_excl) {
  KeyAddresses addresses;
  static_assert(kPiecesPerStep % kThreadCount == 0, "each thread copies whole pieces");
  for (int i = 0; i < kPiecesPerStep / kThreadCount; ++i) {
    const int piece = i * kThreadCount + threadIdx.x;
    const int position = first_position_incl + piece / kLoadsPerRow;
    addresses.position[i] = position < end_position_excl ? position : -1;
    addresses.page[i] = 0;
    if (position < end_position_excl) {
      addresses.page[i] =
          p.page_table_device[b * p.page_table_stride + (position >> p.page_size_log2)];
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
 * copy_async_16(&tile.ring.raw_k[stage][key][dim], p.k_cache_device + offset, is_present);
 * @endcode
 *
 * @par Must know
 * - Performance:
 *   - The `L2::128B` hint fetches whole 128-byte lines: the other KV head's block reads the rest.
 */
__device__ void copy_async_16(void* destination, const void* source, bool is_present) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  const int source_bytes = is_present ? kBytesPerLoad : 0;
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(address),
               "l"(source), "r"(source_bytes));
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
  for (int piece = threadIdx.x; piece < row_count * kPiecesPerRow; piece += thread_count()) {
    const int row = piece / kPiecesPerRow;
    const int column = (piece % kPiecesPerRow) * kHalvesPerLoad;
    const int head = kv_head * group_size + row % group_size;
    const __nv_bfloat16* source = p.qkv_device + (b * p.L + row / group_size) * qkv_row_size +
                                  first_head_column + head * kHeadDim + column;
    copy_async_16(&rows[row][column], source, true);
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

/**
 * @brief `start_new_token_copy` starts copying the block's new key and value rows for one KV head
 * from the fused projection output into shared memory; the next `cp.async` commit includes them.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile: its `new_k_in` and `new_v_in` rows.
 * @param[in] b Request index.
 * @param[in] kv_head Key/value head index.
 *
 * @par Example
 * @code
 * start_new_token_copy(p, tile, b, kv_head);
 * @endcode
 */
__device__ void start_new_token_copy(const Params& p, SharedTile& tile, int b, int kv_head) {
  const int qkv_row_size = (2 * p.q_head_count + 2 * p.kv_head_count) * kHeadDim;
  for (int piece = threadIdx.x; piece < p.L * kPiecesPerRow; piece += thread_count()) {
    const int token = piece / kPiecesPerRow;
    const int column = (piece % kPiecesPerRow) * kHalvesPerLoad;
    const __nv_bfloat16* row = p.qkv_device + (b * p.L + token) * qkv_row_size + column;
    copy_async_16(&tile.new_k_in[token][column], row + (p.q_head_count + kv_head) * kHeadDim, true);
    copy_async_16(&tile.new_v_in[token][column],
                  row + (p.q_head_count + p.kv_head_count + kv_head) * kHeadDim, true);
  }
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
  for (int index = threadIdx.x; index < p.L * kHeadDim / 2; index += thread_count()) {
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
 * @tparam kThreadCount Threads per block, so the piece loop has fixed bounds.
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile: its ring stage and the stage's key positions.
 * @param[in] kv_head Key/value head index.
 * @param[in] addresses The pieces' pages, from `locate_keys` one step earlier.
 * @param[in] stage Ring stage to fill.
 *
 * @par Example
 * @code
 * start_step_copy<128>(p, tile, kv_head, addresses, (step + kStages - 1) % kStages);
 * @endcode
 */
template <int kThreadCount>
__device__ void start_step_copy(const Params& p, SharedTile& tile, int kv_head,
                                const KeyAddresses& addresses, int stage) {
  static_assert(kPiecesPerStep % kThreadCount == 0, "each thread copies whole pieces");
  for (int i = 0; i < kPiecesPerStep / kThreadCount; ++i) {
    const int piece = i * kThreadCount + threadIdx.x;
    const int key = piece / kLoadsPerRow;
    const int dim = (piece % kLoadsPerRow) * kBytesPerLoad;
    const int position = addresses.position[i];
    const int64_t row = (static_cast<int64_t>(addresses.page[i]) << p.page_size_log2) +
                        (position >= 0 ? position & ((1 << p.page_size_log2) - 1) : 0);
    const int64_t offset = (row * p.kv_head_count + kv_head) * kHeadDim + dim;
    copy_async_16(&tile.ring.raw_k[stage][key][swizzled_dim(key, dim)], p.k_cache_device + offset,
                  position >= 0);
    copy_async_16(&tile.ring.raw_v[stage][key][dim], p.v_cache_device + offset, position >= 0);
    if (piece % kLoadsPerRow == 0) {
      tile.ring.raw_position[stage][key] = addresses.position[i];
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

/**
 * @brief `convert_step` converts this thread's pieces of a ring stage's values to the fp16 that
 * the P V tensor cores read; the keys stay FP8.
 *
 * @tparam kThreadCount Threads per block, so the piece loop has fixed bounds.
 * @param[in,out] tile Shared memory tile: reads the stage, writes values and key positions.
 * @param[in] stage Ring stage whose copies this thread has waited for.
 *
 * @par Example
 * @code
 * convert_step<128>(tile, step % kStages);
 * @endcode
 */
template <int kThreadCount>
__device__ void convert_step(SharedTile& tile, int stage) {
  static_assert(kPiecesPerStep % kThreadCount == 0, "each thread copies whole pieces");
  for (int i = 0; i < kPiecesPerStep / kThreadCount; ++i) {
    const int piece = i * kThreadCount + threadIdx.x;
    const int key = piece / kLoadsPerRow;
    const int dim = (piece % kLoadsPerRow) * kBytesPerLoad;
    store_fp8_as_halves(*reinterpret_cast<const uint4*>(&tile.ring.raw_v[stage][key][dim]),
                        &tile.v[key][swizzled_value_dim(key, dim)],
                        &tile.v[key][swizzled_value_dim(key, dim + kHalvesPerLoad)]);
    if (piece % kLoadsPerRow == 0) {
      tile.key_position[key] = tile.ring.raw_position[stage][key];
    }
  }
}

/**
 * @brief `normalize_query_row` RMS-normalizes this lane's 8 dimensions of one 64-wide query row
 * (8 lanes per row), applies interleaved RoPE, and stores them as FP8.
 *
 * @param[in] tile Shared memory tile with the bf16 query rows and the rotation table.
 * @param[out] q_row The row's FP8 bytes.
 * @param[in] row Query row; rows past `row_count` become zeros.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @param[in] group_size Query heads per KV head.
 * @param[in] first_dim This lane's first dimension, a multiple of 8.
 *
 * @par Example
 * @code
 * normalize_query_row(tile, tile.start.q[row], row, row_count, group_size, lane % 8 * 8);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - All 32 lanes must call `normalize_query_row` together (it sums over groups of 8 lanes).
 */
__device__ void normalize_query_row(const SharedTile& tile, uint8_t* q_row, int row, int row_count,
                                    int group_size, int first_dim) {
  float q[kDimsPerOutputLane] = {};
  float square_sum = 0.0f;
  if (row < row_count) {
    const uint4 bits = *reinterpret_cast<const uint4*>(&tile.start.q_in[row][first_dim]);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&bits);
    for (int i = 0; i < kDimsPerOutputLane / 2; ++i) {
      const float2 pair = __bfloat1622float2(pairs[i]);
      q[2 * i] = pair.x;
      q[2 * i + 1] = pair.y;
      square_sum += pair.x * pair.x + pair.y * pair.y;
    }
  }
  const float inv_rms = rsqrtf(lane_group_sum<kLanesPerOutputRow>(square_sum) / kHeadDim + kRmsEps);
  uint8_t bytes[kDimsPerOutputLane];
  for (int i = 0; i < kDimsPerOutputLane / 2; ++i) {
    const float2 rotation = row < row_count ? tile.rotation[row / group_size][first_dim / 2 + i]
                                            : make_float2(1.0f, 0.0f);
    const float x = q[2 * i] * inv_rms;
    const float y = q[2 * i + 1] * inv_rms;
    bytes[2 * i] = __nv_fp8_e4m3(x * rotation.x - y * rotation.y).__x;
    bytes[2 * i + 1] = __nv_fp8_e4m3(x * rotation.y + y * rotation.x).__x;
  }
  *reinterpret_cast<uint2*>(&q_row[first_dim]) = *reinterpret_cast<const uint2*>(bytes);
}

/**
 * @brief `store_new_keys` normalizes and rotates the block's L new keys (staged by
 * `start_new_token_copy`), writes them and their
 * values to the FP8 cache, and puts the stored values in shared memory as a step of keys.
 *
 * @param[in] p Kernel parameters.
 * @param[out] tile Shared memory tile: ring stage 0 holds the new FP8 keys in rows 0 to L - 1, the
 * fp16 values tile their values, and the other keys are missing (zero values; stale key bytes,
 * whose scores the mask replaces).
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
  for (int vector = threadIdx.x; vector < kStepKeys * kHeadDim / kHalvesPerLoad;
       vector += thread_count()) {
    if (vector / (kHeadDim / kHalvesPerLoad) >= p.L) {
      reinterpret_cast<uint4*>(&tile.v[0][0])[vector] = make_uint4(0, 0, 0, 0);
    }
  }
  for (int key = threadIdx.x; key < kStepKeys; key += thread_count()) {
    tile.key_position[key] = key < p.L ? seq_len + key : -1;
  }
  for (int key = warp; key < p.L; key += warp_count()) {
    const float2 k_in =
        __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&tile.new_k_in[key][0])[lane]);
    const float2 v_in =
        __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&tile.new_v_in[key][0])[lane]);
    const float2 k = normalize_and_rotate(k_in, tile.rotation[key][lane]);
    const __nv_fp8_e4m3 k_fp8[2] = {__nv_fp8_e4m3(k.x / tile.k_scale),
                                    __nv_fp8_e4m3(k.y / tile.k_scale)};
    const __nv_fp8_e4m3 v_fp8[2] = {__nv_fp8_e4m3(v_in.x / tile.v_scale),
                                    __nv_fp8_e4m3(v_in.y / tile.v_scale)};
    const int64_t offset =
        (tile.new_cache_loc[key] * p.kv_head_count + kv_head) * kHeadDim + 2 * lane;
    p.k_cache_device[offset] = k_fp8[0];
    p.k_cache_device[offset + 1] = k_fp8[1];
    p.v_cache_device[offset] = v_fp8[0];
    p.v_cache_device[offset + 1] = v_fp8[1];
    tile.ring.raw_k[0][key][swizzled_dim(key, 2 * lane)] = k_fp8[0].__x;
    tile.ring.raw_k[0][key][swizzled_dim(key, 2 * lane + 1)] = k_fp8[1].__x;
    *reinterpret_cast<__half2*>(&tile.v[key][swizzled_value_dim(key, 2 * lane)]) =
        __floats2half2_rn(static_cast<float>(v_fp8[0]), static_cast<float>(v_fp8[1]));
  }
}

/**
 * @brief `accumulate_values_wgmma` computes O += P V for the warpgroup's 64 query rows and
 * `kChunkCount` chunks of 16 keys on `wgmma`, and waits for the result.
 *
 * @tparam kChunkCount Chunks of 16 keys: 4 (the whole step) or 2.
 *
 * @param[in,out] out This thread's 32 output accumulators, in the `mma.sync` C layout.
 * @param[in] probabilities This thread's 8 probabilities per chunk, in the `mma.sync` C layout of
 * `kChunkCount` chunks of 16 keys.
 * @param[in] values The first key's row of the step's fp16 values, in the 128-byte swizzle of
 * `swizzled_value_dim`.
 *
 * @par Example
 * @code
 * accumulate_values_wgmma<4>(out, scores, tile.v);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - All 128 threads of the warpgroup must call `accumulate_values_wgmma` together.
 * - Performance:
 *   - All of P is packed before `wgmma.fence`, or ptxas serializes the `wgmma`s (warning C7520).
 */
template <int kChunkCount>
__device__ void accumulate_values_wgmma(float (&out)[32],
                                        const float (&probabilities)[kChunkCount][2][4],
                                        const __half (*values)[kHeadDim]) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  uint32_t p_fragments[kChunkCount][4];
  for (int chunk = 0; chunk < kChunkCount; ++chunk) {
    const float (&p)[2][4] = probabilities[chunk];
    p_fragments[chunk][0] = pack_halves(p[0][0], p[0][1]);
    p_fragments[chunk][1] = pack_halves(p[0][2], p[0][3]);
    p_fragments[chunk][2] = pack_halves(p[1][0], p[1][1]);
    p_fragments[chunk][3] = pack_halves(p[1][2], p[1][3]);
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  for (int chunk = 0; chunk < kChunkCount; ++chunk) {
    wgmma_64x64x16_f16(out, p_fragments[chunk],
                       make_value_descriptor(values[kHalfMmaDepth * chunk]));
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#else
  __trap();
#endif
}

/**
 * @brief `attend_keys` folds this warp's keys of a step (`kChunkCount` chunks of 16) into its
 * online softmax over its 16 query rows: S = Q K^T on tensor cores, one max and one rescale for
 * all of them, then O += P V on tensor cores. S uses FP8 keys and queries, P V uses fp16.
 *
 * @tparam kChunkCount Chunks of 16 keys this warp attends in the step.
 * @tparam kUseWgmma Whether each warpgroup runs S and P V on `wgmma` over its keys (verify).
 * @param[in] p Kernel parameters.
 * @param[in] tile Shared memory tile with the step's fp16 values and key positions.
 * @param[in] keys The step's FP8 keys, one 64-byte row per key, swizzled by `swizzled_dim`.
 * @param[in] q_fragments This warp's FP8 query rows as A fragments, one per 32 head dimensions.
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
 * attend_keys<4, true>(p, tile, keys, q_fragments, 0, first_row, seq_len, scale, is_masked,
 *                      state);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - `logit_scale_log2` must be positive: the step's max is taken before scaling.
 */
template <int kChunkCount, bool kUseWgmma>
__device__ void attend_keys(const Params& p, const SharedTile& tile,
                            const uint8_t (*keys)[kHeadDim],
                            const uint32_t (&q_fragments)[kQkSteps][4], int first_key,
                            int first_row, int seq_len, float logit_scale_log2, bool is_masked,
                            WarpState& state) {
  const int lane = threadIdx.x % kWarpSize;
  const int group_size = p.q_head_count / p.kv_head_count;
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN THE 4-WARP BUILD WILL LOSE 2% (D40)
  const int warpgroup_first_key = kChunkCount * kChunkKeys == kStepKeys ? 0 : first_key;
  float scores[kChunkCount][2][4] = {};
  if constexpr (kUseWgmma) {
    constexpr int kKeyCount = kChunkCount * kChunkKeys;
    float (&accumulators)[kKeyCount / 2] =
        *reinterpret_cast<float (*)[kKeyCount / 2]>(&scores[0][0][0]);
    score_keys_wgmma<kKeyCount>(accumulators, q_fragments, &keys[warpgroup_first_key]);
  } else {
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      for (int n = 0; n < 2; ++n) {
        const int key = first_key + chunk * kChunkKeys + n * kMmaCols + lane % 8;
        uint32_t b[4];
        load_matrices<false>(b, &keys[key][swizzled_dim(key, kBytesPerLoad * (lane / 8))]);
        mma_16x8x32_fp8(scores[chunk][n], q_fragments[0], b[0], b[1]);
        mma_16x8x32_fp8(scores[chunk][n], q_fragments[1], b[2], b[3]);
      }
    }
  }
  if (is_masked) {
    for (int half = 0; half < 2; ++half) {
      const int query_position = seq_len + (first_row + lane / 4 + 8 * half) / group_size;
      for (int chunk = 0; chunk < kChunkCount; ++chunk) {
        for (int n = 0; n < 2; ++n) {
          for (int e = 2 * half; e < 2 * half + 2; ++e) {
            const int key_position = tile.key_position[first_key + chunk * kChunkKeys +
                                                       n * kMmaCols + 2 * (lane % 4) + e % 2];
            const bool is_visible = key_position >= 0 && key_position <= query_position &&
                                    (p.window == 0 || key_position > query_position - p.window);
            scores[chunk][n][e] = is_visible ? scores[chunk][n][e] : -INFINITY;
          }
        }
      }
    }
  }
  for (int half = 0; half < 2; ++half) {
    float step_max = -INFINITY;
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      for (int n = 0; n < 2; ++n) {
        step_max =
            fmaxf(step_max, fmaxf(scores[chunk][n][2 * half], scores[chunk][n][2 * half + 1]));
      }
    }
    step_max = fmaxf(step_max, __shfl_xor_sync(0xffffffffu, step_max, 1));
    step_max = fmaxf(step_max, __shfl_xor_sync(0xffffffffu, step_max, 2));
    const float new_max = fmaxf(state.max_log2[half], step_max * logit_scale_log2);
    const float safe_max = new_max == -INFINITY ? 0.0f : new_max;
    const float rescale = fast_exp2(state.max_log2[half] - safe_max);
    state.max_log2[half] = new_max;
    state.sum[half] *= rescale;
    for (int d = 0; d < kDimTiles; ++d) {
      state.out[d][2 * half] *= rescale;
      state.out[d][2 * half + 1] *= rescale;
    }
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      for (int n = 0; n < 2; ++n) {
        for (int e = 2 * half; e < 2 * half + 2; ++e) {
          scores[chunk][n][e] = fast_exp2(fmaf(scores[chunk][n][e], logit_scale_log2, -safe_max));
          state.sum[half] += scores[chunk][n][e];
        }
      }
    }
  }
  if constexpr (kUseWgmma) {
    accumulate_values_wgmma<kChunkCount>(*reinterpret_cast<float (*)[32]>(&state.out[0][0]), scores,
                                         &tile.v[warpgroup_first_key]);
    return;
  }
  for (int chunk = 0; chunk < kChunkCount; ++chunk) {
    const uint32_t p_fragment[4] = {pack_halves(scores[chunk][0][0], scores[chunk][0][1]),
                                    pack_halves(scores[chunk][0][2], scores[chunk][0][3]),
                                    pack_halves(scores[chunk][1][0], scores[chunk][1][1]),
                                    pack_halves(scores[chunk][1][2], scores[chunk][1][3])};
    const int key = first_key + chunk * kChunkKeys + 8 * ((lane / 8) % 2) + lane % 8;
    for (int pair = 0; pair < kDimTiles / 2; ++pair) {
      uint32_t b[4];
      load_matrices<true>(b, &tile.v[key][swizzled_value_dim(key, 16 * pair + 8 * (lane / 16))]);
      mma_16x8x16(state.out[2 * pair], p_fragment, b[0], b[1]);
      mma_16x8x16(state.out[2 * pair + 1], p_fragment, b[2], b[3]);
    }
  }
}

/**
 * @brief `combine_warps` merges the softmax states of the warps that share query rows into the
 * block's state, in place: warp w's outputs sit in rows 16 w to 16 w + 15 of `out`, and the merged
 * output of query row r lands in row r. It also writes the row maxima (natural-log units) and sums.
 * One thread per row computes the row's merge weights once, then every thread merges `float4`s.
 *
 * @param[in,out] tile Shared memory tile: reads the warps' states, writes `out`, `row_max` and
 * `row_sum`, and overwrites `warp_max` with each warp's merge weights. Each output element reads
 * only its own rows, so merging in place is safe.
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
  for (int row = threadIdx.x; row < row_count; row += thread_count()) {
    const int m_tile = row / kMmaRows;
    const int row_in_tile = row % kMmaRows;
    float merged_max = -INFINITY;
    for (int group = 0; group < key_group_count; ++group) {
      merged_max = fmaxf(merged_max, tile.warp_max[group * m_tile_count + m_tile][row_in_tile]);
    }
    const float safe_max = merged_max == -INFINITY ? 0.0f : merged_max;
    float merged_sum = 0.0f;
    for (int group = 0; group < key_group_count; ++group) {
      const int warp = group * m_tile_count + m_tile;
      const float weight = exp2f(tile.warp_max[warp][row_in_tile] - safe_max);
      merged_sum += weight * tile.warp_sum[warp][row_in_tile];
      tile.warp_max[warp][row_in_tile] = weight;
    }
    tile.row_max[row] = merged_max / kLog2e;
    tile.row_sum[row] = merged_sum;
  }
  __syncthreads();
  for (int vector = threadIdx.x; vector < row_count * kVectorsPerRow; vector += thread_count()) {
    const int row = vector / kVectorsPerRow;
    const int m_tile = row / kMmaRows;
    const int row_in_tile = row % kMmaRows;
    float4 merged = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int group = 0; group < key_group_count; ++group) {
      const int warp = group * m_tile_count + m_tile;
      const float weight = tile.warp_max[warp][row_in_tile];
      const float4 x = reinterpret_cast<const float4*>(
          warp_out_row(tile, warp, row_in_tile))[vector % kVectorsPerRow];
      merged = make_float4(merged.x + weight * x.x, merged.y + weight * x.y,
                           merged.z + weight * x.z, merged.w + weight * x.w);
    }
    reinterpret_cast<float4*>(&tile.out[0][0])[vector] = merged;
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
  for (int vector = threadIdx.x; vector < row_count * kVectorsPerRow; vector += thread_count()) {
    slot_out[vector] = reinterpret_cast<const float4*>(&tile.out[0][0])[vector];
  }
  for (int row = threadIdx.x; row < row_count; row += thread_count()) {
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
 * @brief `merge_slots` merges up to 16 published softmax states into the block's state in `tile`,
 * with one warp per row and one lane per slot for the weights.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile: gets the merged outputs, row maxima and row sums; uses
 * `split_weight` for each slot's row maxima and then its weights, and `split_sum` for its softmax
 * sums, which the same pass loads with the maxima.
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
  for (int index = threadIdx.x; index < slot_count * row_count; index += thread_count()) {
    const int64_t slot = first_slot + (index / row_count) * slot_stride;
    tile.merge.split_weight[index / row_count][index % row_count] =
        __ldcg(&p.partial_max_device[slot * kMaxRows + index % row_count]);
    tile.merge.split_sum[index / row_count][index % row_count] =
        __ldcg(&p.partial_sum_device[slot * kMaxRows + index % row_count]);
  }
  __syncthreads();
  const int lane = threadIdx.x % kWarpSize;
  for (int row = threadIdx.x / kWarpSize; row < row_count; row += warp_count()) {
    const bool is_slot = lane < slot_count;
    const float slot_max = is_slot ? tile.merge.split_weight[lane][row] : -INFINITY;
    float merged_max = slot_max;
    for (int offset = kMergeFanIn / 2; offset > 0; offset /= 2) {
      merged_max = fmaxf(merged_max, __shfl_xor_sync(0xffffffffu, merged_max, offset));
    }
    const float safe_max = merged_max == -INFINITY ? 0.0f : merged_max;
    const float weight = is_slot ? expf(slot_max - safe_max) : 0.0f;
    if (is_slot) {
      tile.merge.split_weight[lane][row] = weight;
    }
    const float merged_sum =
        lane_group_sum<kWarpSize>(is_slot ? weight * tile.merge.split_sum[lane][row] : 0.0f);
    if (lane == 0) {
      tile.row_max[row] = merged_max;
      tile.row_sum[row] = merged_sum;
    }
  }
  __syncthreads();
  for (int vector = threadIdx.x; vector < row_count * kVectorsPerRow; vector += thread_count()) {
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
 * @brief `merger_count` returns how many blocks merge each request and KV head's splits: one
 * per slice of rows, at most `kMaxMergers`.
 *
 * @param[in] p Kernel parameters.
 * @param[in] row_count Query rows (L x query heads per KV head).
 * @return The merger count.
 *
 * @par Example
 * @code
 * const int mergers = merger_count(p, row_count);
 * @endcode
 */
__device__ int merger_count(const Params& p, int row_count) {
  return min(kMaxMergers, min(p.split_count, row_count));
}

/**
 * @brief `take_merge_rank` counts this block in once its state is published, and returns its rank
 * among the last `merger_count` blocks to arrive, which merge; earlier blocks get a negative rank.
 *
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile: its `merge_rank`.
 * @param[in] group Request x KV-head group.
 * @param[in] mergers `merger_count`.
 * @return The merge rank, or a negative number.
 *
 * @par Example
 * @code
 * const int rank = take_merge_rank(p, tile, group, mergers);
 * @endcode
 */
__device__ int take_merge_rank(const Params& p, SharedTile& tile, int group, int mergers) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    tile.merge_rank = atomicAdd(&p.arrival_count_device[kCountersPerGroup * group], 1) -
                      (p.split_count - mergers);
  }
  __syncthreads();
  return tile.merge_rank;
}

/**
 * @brief `wait_for_all_splits` returns once every split of the group has published its state.
 *
 * @param[in] p Kernel parameters.
 * @param[in] group Request x KV-head group.
 *
 * @par Example
 * @code
 * wait_for_all_splits(p, group);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The kernel launches cooperatively, so every split is resident and the wait ends.
 */
__device__ void wait_for_all_splits(const Params& p, int group) {
  if (threadIdx.x == 0) {
    const int* counter = &p.arrival_count_device[kCountersPerGroup * group];
    int arrived = 0;
    do {
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(arrived) : "l"(counter));
    } while (arrived < p.split_count);
  }
  __syncthreads();
  __threadfence();
}

/**
 * @brief `merge_rows` merges every split's state for this block's rows (`rank`, `rank + mergers`,
 * ...) into `tile`: row sums, and unnormalized outputs relative to the merged maximum.
 *
 * @tparam kSplitUnroll Splits each thread's output loop keeps in flight: 4 in the 8-warp build,
 * whose batch-1 merges run over 100 or more splits; 1 in the 4-warp build, whose registers are
 * full.
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile: gets `out` and `row_sum` for this block's rows.
 * @param[in] first_slot The group's first workspace slot.
 * @param[in] rank This block's merge rank.
 * @param[in] mergers `merger_count`.
 * @param[in] row_count Query rows (L x query heads per KV head).
 *
 * @par Example
 * @code
 * merge_rows<4>(p, tile, first_slot, rank, mergers, row_count);
 * @endcode
 */
template <int kSplitUnroll>
__device__ void merge_rows(const Params& p, SharedTile& tile, int64_t first_slot, int rank,
                           int mergers, int row_count) {
  const int my_row_count = (row_count - rank + mergers - 1) / mergers;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int index = threadIdx.x; index < p.split_count * my_row_count; index += thread_count()) {
    const int row = rank + index % my_row_count * mergers;
    const int64_t slot = first_slot + index / my_row_count;
    tile.merge.row_weight[index] = __ldcg(&p.partial_max_device[slot * kMaxRows + row]);
    tile.merge.row_split_sum[index] = __ldcg(&p.partial_sum_device[slot * kMaxRows + row]);
  }
  __syncthreads();
  for (int local_row = warp; local_row < my_row_count; local_row += warp_count()) {
    float merged_max = -INFINITY;
    for (int s = lane; s < p.split_count; s += kWarpSize) {
      merged_max = fmaxf(merged_max, tile.merge.row_weight[s * my_row_count + local_row]);
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      merged_max = fmaxf(merged_max, __shfl_xor_sync(0xffffffffu, merged_max, offset));
    }
    const float safe_max = merged_max == -INFINITY ? 0.0f : merged_max;
    float merged_sum = 0.0f;
    for (int s = lane; s < p.split_count; s += kWarpSize) {
      const int index = s * my_row_count + local_row;
      const float weight = expf(tile.merge.row_weight[index] - safe_max);
      tile.merge.row_weight[index] = weight;
      merged_sum += weight * tile.merge.row_split_sum[index];
    }
    merged_sum = lane_group_sum<kWarpSize>(merged_sum);
    if (lane == 0) {
      tile.row_sum[rank + local_row * mergers] = merged_sum;
    }
  }
  __syncthreads();
  const int vector_count = my_row_count * kVectorsPerRow;
  const int part_count = max(1, thread_count() / vector_count);
  for (int task = threadIdx.x; task < vector_count * part_count; task += thread_count()) {
    const int vector = task % vector_count;
    const int local_row = vector / kVectorsPerRow;
    const int row = rank + local_row * mergers;
    float4 merged = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll kSplitUnroll
    for (int s = task / vector_count; s < p.split_count; s += part_count) {
      const float4 x = __ldcg(&reinterpret_cast<const float4*>(
          p.partial_out_device +
          (first_slot + s) * kMaxRows * kHeadDim)[row * kVectorsPerRow + vector % kVectorsPerRow]);
      const float weight = tile.merge.row_weight[s * my_row_count + local_row];
      merged = make_float4(merged.x + weight * x.x, merged.y + weight * x.y,
                           merged.z + weight * x.z, merged.w + weight * x.w);
    }
    if (part_count == 1) {
      reinterpret_cast<float4*>(tile.out[row])[vector % kVectorsPerRow] = merged;
    } else {
      tile.merge.partial_sums[task] = merged;
    }
  }
  __syncthreads();
  for (int vector = threadIdx.x; vector < vector_count && part_count > 1;
       vector += thread_count()) {
    float4 merged = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int part = 0; part < part_count; ++part) {
      const float4 x = tile.merge.partial_sums[part * vector_count + vector];
      merged = make_float4(merged.x + x.x, merged.y + x.y, merged.z + x.z, merged.w + x.w);
    }
    const int row = rank + vector / kVectorsPerRow * mergers;
    reinterpret_cast<float4*>(tile.out[row])[vector % kVectorsPerRow] = merged;
  }
  __syncthreads();
}

/**
 * @brief `finish_merge` counts this merger out; the last one resets the group's counters to 0 for
 * the next call.
 *
 * @param[in] p Kernel parameters.
 * @param[in] group Request x KV-head group.
 * @param[in] mergers `merger_count`.
 *
 * @par Example
 * @code
 * finish_merge(p, group, mergers);
 * @endcode
 */
__device__ void finish_merge(const Params& p, int group, int mergers) {
  if (threadIdx.x == 0) {
    int* counters = &p.arrival_count_device[kCountersPerGroup * group];
    if (atomicAdd(&counters[1], 1) == mergers - 1) {
      counters[0] = 0;
      counters[1] = 0;
    }
  }
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
 * @brief `BlockWork` describes what one block of `attention_kernel` works on: its split, KV head
 * and request, its warp's place among the query rows and keys, and its split's key positions.
 *
 * @par Example
 * @code
 * const BlockWork work = describe_block<4>(p);
 * @endcode
 */
struct BlockWork {
  int split, kv_head, b, warp, lane;
  int group_size, row_count, m_tile_count, key_group_count, m_tile, key_group;
  bool is_active;
  int seq_len, window_start_incl, first_position_incl, end_position_excl;
};

/**
 * @brief `describe_block` returns this block's and this warp's `BlockWork`; it starts the read of
 * the request's length.
 *
 * @tparam kWarpCount Warps per block.
 * @param[in] p Kernel parameters; the grid is (splits, KV heads, requests).
 * @return The block's work.
 *
 * @par Example
 * @code
 * const BlockWork work = describe_block<kWarpCount>(p);
 * @endcode
 */
template <int kWarpCount>
__device__ BlockWork describe_block(const Params& p) {
  BlockWork work;
  work.split = blockIdx.x;
  work.kv_head = blockIdx.y;
  work.b = blockIdx.z;
  work.warp = threadIdx.x / kWarpSize;
  work.lane = threadIdx.x % kWarpSize;
  work.group_size = p.q_head_count / p.kv_head_count;
  work.row_count = p.L * work.group_size;
  work.m_tile_count = (work.row_count + kMmaRows - 1) / kMmaRows;
  work.key_group_count = kWarpCount / work.m_tile_count;
  work.m_tile = work.warp % work.m_tile_count;
  work.key_group = work.warp / work.m_tile_count;
  work.is_active = work.key_group < work.key_group_count;
  work.seq_len = p.seq_lens_device[work.b];
  work.window_start_incl = p.window == 0 ? 0 : max(0, work.seq_len - p.window + 1);
  work.first_position_incl = work.window_start_incl + work.split * p.keys_per_split;
  work.end_position_excl = min(work.first_position_incl + p.keys_per_split, work.seq_len);
  return work;
}

/**
 * @brief `start_block` starts copying the block's query rows, its new tokens (first split only)
 * and its first `kStages - 1` steps of keys and values, reads the scales and the new tokens'
 * cache rows, fills the RoPE table, and waits for the query rows.
 *
 * @tparam kThreadCount Threads per block.
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile.
 * @param[in] work The block's work.
 * @return The page-table entries of the next step to copy, still in flight.
 *
 * @par Example
 * @code
 * KeyAddresses addresses = start_block<128>(p, tile, work);
 * @endcode
 */
template <int kThreadCount>
__device__ KeyAddresses start_block(const Params& p, SharedTile& tile, const BlockWork& work) {
  if (work.split == 0) {
    start_new_token_copy(p, tile, work.b, work.kv_head);
  }
  start_rows_copy(p, tile.start.q_in, work.b, work.kv_head, work.row_count, 0);
  KeyAddresses addresses = locate_keys<kThreadCount>(
      p, work.b, work.kv_head, work.first_position_incl, work.end_position_excl);
  for (int stage = 0; stage < kStages - 1; ++stage) {
    const KeyAddresses next_addresses = locate_keys<kThreadCount>(
        p, work.b, work.kv_head, work.first_position_incl + (stage + 1) * kStepKeys,
        work.end_position_excl);
    start_step_copy<kThreadCount>(p, tile, work.kv_head, addresses, stage);
    addresses = next_addresses;
  }
  if (threadIdx.x < p.L) {
    tile.new_cache_loc[threadIdx.x] = p.cache_loc_device[work.b * p.L + threadIdx.x];
  }
  if (threadIdx.x == kWarpSize) {
    tile.k_scale = *p.k_scale_device;
    tile.v_scale = *p.v_scale_device;
  }
  fill_rotation_table(p, tile, work.seq_len);
  wait_for_copies<kStages - 1>();
  __syncthreads();
  return addresses;
}

/**
 * @brief `load_query_fragments` normalizes, rotates and rounds the block's query rows to FP8, then
 * loads this warp's 16 rows as `mma` A fragments.
 *
 * @param[in,out] tile Shared memory tile with the copied query rows.
 * @param[in] work The block's work.
 * @param[out] q_fragments This thread's query fragments, one per 32 head dimensions.
 *
 * @par Example
 * @code
 * load_query_fragments(tile, work, q_fragments);
 * @endcode
 */
__device__ void load_query_fragments(SharedTile& tile, const BlockWork& work,
                                     uint32_t (&q_fragments)[kQkSteps][4]) {
  constexpr int kQueryRowsPerWarp = kWarpSize / kLanesPerOutputRow;
  for (int warp_row = work.warp * kQueryRowsPerWarp; warp_row < work.m_tile_count * kMmaRows;
       warp_row += warp_count() * kQueryRowsPerWarp) {
    const int row = warp_row + work.lane / kLanesPerOutputRow;
    normalize_query_row(tile, tile.start.q[row], row, work.row_count, work.group_size,
                        work.lane % kLanesPerOutputRow * kDimsPerOutputLane);
  }
  __syncthreads();
  for (int depth = 0; depth < kQkSteps; ++depth) {
    load_matrices<false>(
        q_fragments[depth],
        &tile.start.q[work.m_tile * kMmaRows + 8 * ((work.lane / 8) % 2) + work.lane % 8]
                     [kFp8MmaDepth * depth + kBytesPerLoad * (work.lane / 16)]);
  }
}

/**
 * @brief `attend_split` streams the split's keys and values through the ring, 64 per step, and
 * folds each step into this warp's softmax state.
 *
 * @tparam kWarpCount Warps per block.
 * @tparam kChunksPerWarp Chunks of 16 keys each warp attends per step.
 * @tparam kIsVerify Whether this is a verify build (`wgmma`).
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile.
 * @param[in] work The block's work.
 * @param[in,out] addresses The next step's page-table entries, from `start_block`.
 * @param[in] q_fragments This thread's query fragments.
 * @param[in] logit_scale_log2 Factor from Q K^T to base-2 logits.
 * @param[in,out] state This thread's softmax state.
 *
 * @par Example
 * @code
 * attend_split<4, 4, true>(p, tile, work, addresses, q_fragments, logit_scale_log2, state);
 * @endcode
 */
template <int kWarpCount, int kChunksPerWarp, bool kIsVerify>
__device__ void attend_split(const Params& p, SharedTile& tile, const BlockWork& work,
                             KeyAddresses& addresses, const uint32_t (&q_fragments)[kQkSteps][4],
                             float logit_scale_log2, WarpState& state) {
  constexpr int kThreadCount = kWarpCount * kWarpSize;
  for (int step = 0; work.first_position_incl + step * kStepKeys < work.end_position_excl; ++step) {
    wait_for_copies<kStages - 2>();
    if constexpr (kIsVerify) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    convert_step<kThreadCount>(tile, step % kStages);
    const KeyAddresses next_addresses = locate_keys<kThreadCount>(
        p, work.b, work.kv_head, work.first_position_incl + (step + kStages) * kStepKeys,
        work.end_position_excl);
    start_step_copy<kThreadCount>(p, tile, work.kv_head, addresses, (step + kStages - 1) % kStages);
    addresses = next_addresses;
    if constexpr (kIsVerify) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    const int start_incl = work.first_position_incl + step * kStepKeys;
    const bool is_masked = start_incl + kStepKeys > work.end_position_excl ||
                           (p.window != 0 && start_incl < work.window_start_incl + p.L - 1);
    attend_keys<kChunksPerWarp, kIsVerify>(p, tile, tile.ring.raw_k[step % kStages], q_fragments,
                                           work.key_group * kChunksPerWarp * kChunkKeys,
                                           work.m_tile * kMmaRows, work.seq_len, logit_scale_log2,
                                           is_masked, state);
  }
  wait_for_copies<0>();
  __syncthreads();
}

/**
 * @brief `attend_new_tokens` writes the block's new keys and values to the cache and attends to
 * them as one last, causal step (first split only).
 *
 * @tparam kChunksPerWarp Chunks of 16 keys each warp attends per step.
 * @tparam kIsVerify Whether this is a verify build (`wgmma`).
 * @param[in] p Kernel parameters.
 * @param[in,out] tile Shared memory tile.
 * @param[in] work The block's work.
 * @param[in] q_fragments This thread's query fragments.
 * @param[in] logit_scale_log2 Factor from Q K^T to base-2 logits.
 * @param[in,out] state This thread's softmax state.
 *
 * @par Example
 * @code
 * attend_new_tokens<4, true>(p, tile, work, q_fragments, logit_scale_log2, state);
 * @endcode
 */
template <int kChunksPerWarp, bool kIsVerify>
__device__ void attend_new_tokens(const Params& p, SharedTile& tile, const BlockWork& work,
                                  const uint32_t (&q_fragments)[kQkSteps][4],
                                  float logit_scale_log2, WarpState& state) {
  store_new_keys(p, tile, work.b, work.kv_head, work.seq_len);
  if constexpr (kIsVerify) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  attend_keys<kChunksPerWarp, kIsVerify>(
      p, tile, tile.ring.raw_k[0], q_fragments, work.key_group * kChunksPerWarp * kChunkKeys,
      work.m_tile * kMmaRows, work.seq_len, logit_scale_log2, true, state);
}

/**
 * @brief `attention_kernel` attends one split of one request's keys for one KV head group; the
 * first split also writes the block's new keys and values, and the merging blocks write the
 * output.
 *
 * @tparam kWarpCount Warps per block: 4, or 8 for verify on sliding layers or with at most 8
 * request x KV-head groups.
 * @tparam kChunksPerWarp Chunks of 16 keys each warp attends per step, so that the warps cover the
 * step's 64 keys: 1 for decode (4 warps share 16 query rows), 2 for 8-warp verify, 4 for 4-warp
 * verify (each warp owns 16 rows).
 * @tparam kMinBlocksPerSm Blocks that must fit on one SM: the compiler caps registers to match.
 * @param[in] p Kernel parameters; the grid is (splits, KV heads, requests).
 *
 * @par Example
 * @code
 * attention_kernel<4, 1, 3><<<dim3(split_count, kv_head_count, B), kMinThreads,
 * sizeof(SharedTile), stream>>>(p);
 * @endcode
 */
template <int kWarpCount, int kChunksPerWarp, int kMinBlocksPerSm>
__global__ void __launch_bounds__(kWarpCount* kWarpSize, kMinBlocksPerSm)
    attention_kernel(const Params p) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  SharedTile& tile = *reinterpret_cast<SharedTile*>(shared_bytes);
  const BlockWork work = describe_block<kWarpCount>(p);
  const int split = work.split;
  const int kv_head = work.kv_head;
  const int b = work.b;
  const int warp = work.warp;
  const int lane = work.lane;
  const int row_count = work.row_count;
  const int m_tile_count = work.m_tile_count;
  const int key_group_count = work.key_group_count;
  const bool is_active = work.is_active;
  KeyAddresses addresses = start_block<kWarpCount * kWarpSize>(p, tile, work);
  uint32_t q_fragments[kQkSteps][4];
  load_query_fragments(tile, work, q_fragments);
  WarpState state = {};
  state.max_log2[0] = state.max_log2[1] = -INFINITY;
  const float logit_scale_log2 = p.sm_scale * p.qk_scale_factor * tile.k_scale * kLog2e;
  constexpr bool kIsVerify = kChunksPerWarp > 1;
  attend_split<kWarpCount, kChunksPerWarp, kIsVerify>(p, tile, work, addresses, q_fragments,
                                                      logit_scale_log2, state);
  if (split == 0) {
    attend_new_tokens<kChunksPerWarp, kIsVerify>(p, tile, work, q_fragments, logit_scale_log2,
                                                 state);
  }
  __syncthreads();
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN THE 4-WARP BUILD WILL LOSE 3% (D52)
  for (int half = 0; half < 2 && is_active; ++half) {
    state.sum[half] += __shfl_xor_sync(0xffffffffu, state.sum[half], 1);
    state.sum[half] += __shfl_xor_sync(0xffffffffu, state.sum[half], 2);
    const int row = lane / 4 + 8 * half;
    if (lane % 4 == 0) {
      tile.warp_max[warp][row] = state.max_log2[half];
      tile.warp_sum[warp][row] = state.sum[half];
    }
    for (int d = 0; d < kDimTiles; ++d) {
      warp_out_row(tile, warp, row)[d * kMmaCols + 2 * (lane % 4)] = state.out[d][2 * half];
      warp_out_row(tile, warp, row)[d * kMmaCols + 2 * (lane % 4) + 1] = state.out[d][2 * half + 1];
    }
  }
  __syncthreads();
  if (key_group_count > 1) {
    combine_warps(tile, row_count, m_tile_count, key_group_count);
  } else {
    for (int row = threadIdx.x; row < row_count; row += thread_count()) {
      tile.row_max[row] = tile.warp_max[row / kMmaRows][row % kMmaRows] / kLog2e;
      tile.row_sum[row] = tile.warp_sum[row / kMmaRows][row % kMmaRows];
    }
  }
  __syncthreads();
  start_rows_copy(p, tile.merge.gate, b, kv_head, row_count,
                  (p.q_head_count + 2 * p.kv_head_count) * kHeadDim);
  if constexpr (kIsVerify) {
    const int group = b * p.kv_head_count + kv_head;
    const int64_t first_slot = static_cast<int64_t>(group) * p.split_count;
    const int mergers = merger_count(p, row_count);
    publish_state(p, tile, first_slot + split, row_count);
    const int merge_rank = take_merge_rank(p, tile, group, mergers);
    if (merge_rank < 0) {
      wait_for_copies<0>();
      return;
    }
    wait_for_all_splits(p, group);
    merge_rows<kWarpCount == kMaxWarps ? 4 : 1>(p, tile, first_slot, merge_rank, mergers,
                                                row_count);
    wait_for_copies<0>();
    __syncthreads();
    write_gated_output(p, tile, b, kv_head, row_count, merge_rank, mergers);
    finish_merge(p, group, mergers);
  } else {
    const bool is_merging_block = merge_splits(p, tile, b, kv_head, split, row_count);
    wait_for_copies<0>();
    __syncthreads();
    if (is_merging_block) {
      write_gated_output(p, tile, b, kv_head, row_count, 0, 1);
    }
  }
}

using attention_kernel_t = void (*)(Params);

/**
 * @brief `KernelBuild` names one compiled kernel build: the kernel, its threads per block, and the
 * blocks per SM its launch bounds and the split plan target, and whether it launches
 * cooperatively (its mergers wait for the other blocks).
 *
 * @par Example
 * @code
 * const KernelBuild build = select_kernel(row_count, window, group_count);
 * @endcode
 */
struct KernelBuild {
  attention_kernel_t kernel;
  int thread_count;
  int blocks_per_sm;
  bool is_cooperative;
};

/**
 * @brief `select_kernel` picks the kernel build for a block's query rows and layer: warps per
 * block, the chunks each warp attends per step, and how many blocks share an SM.
 *
 * @param[in] row_count Query rows per block (L x query heads per KV head).
 * @param[in] window Sliding window in tokens, or 0 for full attention.
 * @param[in] group_count Requests x KV heads: full verify with few groups also gets 8 warps.
 * @return The kernel to launch, its threads per block, its blocks per SM, and whether it launches
 * cooperatively.
 * @throws c10::Error When a block has 17 to 48 query rows (2 or 3 row tiles).
 *
 * @par Example
 * @code
 * const KernelBuild build = select_kernel(64, 2048, 32);
 * @endcode
 */
KernelBuild select_kernel(int64_t row_count, int64_t window, int64_t group_count) {
  const int64_t m_tile_count = (row_count + kMmaRows - 1) / kMmaRows;
  if (m_tile_count == 1) {
    return {attention_kernel<4, 1, kDecodeBlocksPerSm>, kMinThreads, kDecodeBlocksPerSm, false};
  }
  TORCH_CHECK(m_tile_count == kMaxRows / kMmaRows,
              "a block needs at most 16 or 49 to 64 query rows (L x query heads per KV head), got ",
              row_count);
  if (window != 0 || group_count <= kMaxWideFullGroups) {
    return {attention_kernel<8, 2, kWideVerifyBlocksPerSm>, 2 * kMinThreads, kWideVerifyBlocksPerSm,
            true};
  }
  return {attention_kernel<4, 4, kVerifyBlocksPerSm>, kMinThreads, kVerifyBlocksPerSm, true};
}

}  // namespace

/**
 * @brief `run` checks the tensors and launches `attention_kernel` on the current stream.
 *
 * @param[in] qkv Fused projection output, bf16 [B, L, (2 Hq + 2 Hkv) * 64].
 * @param[in,out] k_cache FP8 key cache [pages, page_size, Hkv, 64], page_size a power of 2; the
 * new keys are written.
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
 * @throws c10::Error A tensor has the wrong dtype, shape or layout, or the page size is not a power
 * of 2.
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
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes() && v_cache.scalar_type() == k_cache.scalar_type(),
              "v_cache must match k_cache in shape and dtype");
  TORCH_CHECK(k_scale.numel() == 1 && v_scale.numel() == 1, "k_scale and v_scale must be scalars");
  TORCH_CHECK(keys_per_split > 0 && keys_per_split % kStepKeys == 0,
              "keys_per_split must be a positive multiple of ", kStepKeys, ", got ",
              keys_per_split);
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
  const int page_size = static_cast<int>(k_cache.size(1));
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
              "page size must be a power of 2, got ", page_size);
  p.page_size_log2 = __builtin_ctz(page_size);
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
  const KernelBuild build = select_kernel(p.L * (p.q_head_count / p.kv_head_count), window,
                                          qkv.size(0) * p.kv_head_count);
  C10_CUDA_CHECK(cudaFuncSetAttribute(build.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      sizeof(SharedTile)));
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(p.split_count, k_cache.size(2), qkv.size(0));
  config.blockDim = dim3(build.thread_count);
  config.dynamicSmemBytes = sizeof(SharedTile);
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute cooperative_attribute = {};
  cooperative_attribute.id = cudaLaunchAttributeCooperative;
  cooperative_attribute.val.cooperative = 1;
  if (build.is_cooperative) {
    config.attrs = &cooperative_attribute;
    config.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, build.kernel, p));
}
/**
 * @brief `resident_blocks_per_sm` returns how many blocks of `attention_kernel` fit on one SM.
 *
 * @param[in] row_count Query rows per block (L x query heads per KV head), which picks the build.
 * @param[in] window Sliding window in tokens, or 0 for full attention, which also picks it.
 * @param[in] group_count Requests x KV heads, which also picks it.
 * @return The occupancy calculator's count, for the kernel's threads and shared memory, capped at
 * the build's intended blocks per SM.
 *
 * @par Example
 * @code
 * int64_t slots = resident_blocks_per_sm(8, 0, 64) * sm_count;
 * @endcode
 */
int64_t resident_blocks_per_sm(int64_t row_count, int64_t window, int64_t group_count) {
  const KernelBuild build = select_kernel(row_count, window, group_count);
  C10_CUDA_CHECK(cudaFuncSetAttribute(build.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      sizeof(SharedTile)));
  int block_count = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_count, build.kernel, build.thread_count, sizeof(SharedTile)));
  return std::min(block_count, build.blocks_per_sm);
}
"""
CPP_SOURCE: str = (
    "void run(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, "
    "torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor cache_loc, "
    "torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor inv_freq, torch::Tensor out, "
    "int64_t window, double qk_scale_factor, double sm_scale, torch::Tensor partial_out, "
    "torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor arrival_count, "
    "int64_t keys_per_split); "
    "int64_t resident_blocks_per_sm(int64_t row_count, int64_t window, int64_t group_count);"
)
_HEAD_DIM: int = 64
_ROPE_PAIR_STRIDE: int = 2
_MAX_ROWS: int = 64
_TILE_KEYS: int = 64
_MAX_SPLITS: int = 256
_MERGE_FAN_IN: int = 16
_COUNTERS_PER_GROUP: int = 17
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
            name="muse_fused_attention_attend_split",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["run", "resident_blocks_per_sm"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
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
        exponents: torch.Tensor = torch.arange(
            0, _HEAD_DIM, _ROPE_PAIR_STRIDE, device=device
        ).float()
        _inv_freq_cache[key] = (
            1.0 / (theta ** (exponents / _HEAD_DIM)) if theta else torch.empty(0, device=device)
        )
    return _inv_freq_cache[key]


def _plan_splits(data: dict) -> tuple[int, int]:
    """`_plan_splits` picks how many blocks share each request's keys, from shapes only.

    Args:
        data: The inputs; only `config` and the page table's shape are read.

    Returns:
        The split count and the cached keys per split (a multiple of 64), so that every key a
        request can hold falls in a split and the grid fits in one wave of resident blocks. In
        decode, short spans (at most 32 steps, such as the 2,048-token window) use at most 16
        splits, so the merge tree needs one level.

    Example:
        >>> split_count, keys_per_split = _plan_splits(data)
    """
    config: dict = data["config"]
    span_keys: int = config["window"] or data["page_table"].shape[1] * config["page_size"]
    tile_count: int = -(-span_keys // _TILE_KEYS)
    sm_count: int = torch.cuda.get_device_properties(data["qkv"].device).multi_processor_count
    row_count: int = config["q_len"] * config["num_q_heads"] // config["num_kv_heads"]
    group_count: int = config["batch"] * config["num_kv_heads"]
    resident_count: int = _get_extension().resident_blocks_per_sm(
        row_count, config["window"], group_count
    )
    slot_count: int = sm_count * resident_count
    wanted_split_count: int = max(1, slot_count // group_count)
    wanted_split_count = min(wanted_split_count, _MAX_SPLITS)
    if tile_count <= 2 * _MERGE_FAN_IN and config["q_len"] == 1:
        wanted_split_count = min(wanted_split_count, _MERGE_FAN_IN)
    keys_per_split: int = -(-tile_count // min(tile_count, wanted_split_count)) * _TILE_KEYS
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
            torch.empty((*shape, _MAX_ROWS, _HEAD_DIM), device=device),
            torch.empty((*shape, _MAX_ROWS), device=device),
            torch.empty((*shape, _MAX_ROWS), device=device),
            torch.zeros((*shape[:2], _COUNTERS_PER_GROUP), dtype=torch.int32, device=device),
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
        (config["batch"], config["q_len"], config["num_q_heads"] * _HEAD_DIM),
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
