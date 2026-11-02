/**
 * @file
 * @brief `quad_from_tmastore.cu` defines `matmul_quad_from_tmastore`: `tmastore` with clusters
 * of 2 x 2 blocks that share A's tiles along rows and B's along columns.
 *
 * @par Interface
 * - `matmul_quad_from_tmastore`: Launches `matmul_quad_from_tmastore_kernel`.
 * - `QUAD_FROM_TMASTORE_STAGE_COUNT`: Option: K steps in shared memory at once, at least 2.
 * - `QUAD_FROM_TMASTORE_GROUP_M`: Option: row tiles that consecutive cluster tiles walk down
 *   before moving to the next column tile, a multiple of 2.
 * - `QUAD_FROM_TMASTORE_STAGING_COLUMNS`: Option: columns of C a consumer stages at once,
 *   128 or 256; 128 stages a tile in two halves and frees shared memory for another stage.
 *
 * @par Example
 * @code
 * matmul_quad_from_tmastore(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - It needs `NVCC_ARCH=sm_90a` (`.env`); built for another architecture, the kernel traps.
 */
#include <cuda.h>

#include <algorithm>
#include <cstdint>

#include "common/matmul.h"
#include "kernels/common/hopper.cuh"

#ifndef QUAD_FROM_TMASTORE_STAGE_COUNT
#define QUAD_FROM_TMASTORE_STAGE_COUNT 4
#endif
#ifndef QUAD_FROM_TMASTORE_GROUP_M
#define QUAD_FROM_TMASTORE_GROUP_M 16
#endif
#ifndef QUAD_FROM_TMASTORE_STAGING_COLUMNS
#define QUAD_FROM_TMASTORE_STAGING_COLUMNS 128
#endif

namespace {

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 256;
constexpr int BLOCK_TILE_K = hopper::BF16_PER_SWIZZLE_ROW;
constexpr int STAGE_COUNT = QUAD_FROM_TMASTORE_STAGE_COUNT;
constexpr int GROUP_M = QUAD_FROM_TMASTORE_GROUP_M;
constexpr int STAGING_COLUMNS = QUAD_FROM_TMASTORE_STAGING_COLUMNS;
constexpr int CLUSTER_ROWS = 2;
constexpr int CLUSTER_COLS = 2;
constexpr int CLUSTER_SIZE = CLUSTER_ROWS * CLUSTER_COLS;
constexpr int A_ROWS_PER_BLOCK = BLOCK_TILE_M / CLUSTER_COLS;
constexpr int CONSUMER_COUNT = BLOCK_TILE_M / hopper::WGMMA_M;
constexpr int THREAD_COUNT = (1 + CONSUMER_COUNT) * hopper::THREADS_PER_WARPGROUP;
constexpr int A_TILE_BYTES = BLOCK_TILE_M * hopper::SWIZZLE_BYTES;
constexpr int B_BOX_BYTES = BLOCK_TILE_K * hopper::SWIZZLE_BYTES;
constexpr int B_BOX_COUNT = BLOCK_TILE_N / hopper::BF16_PER_SWIZZLE_ROW;
constexpr int STAGE_BYTES = A_TILE_BYTES + (B_BOX_COUNT * B_BOX_BYTES);
constexpr int C_STAGING_BYTES =
    hopper::WGMMA_M * STAGING_COLUMNS * static_cast<int>(sizeof(bf16_t));
constexpr int SHARED_BYTES =
    (STAGE_COUNT * STAGE_BYTES) + (CONSUMER_COUNT * C_STAGING_BYTES) + hopper::SWIZZLE_ATOM_BYTES;
constexpr int MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90 = 227 * 1024;
constexpr int REGISTERS_PER_SM_ACCORDING_TO_CUDA_GUIDE_FOR_SM90 = 64 * 1024;
constexpr int PRODUCER_REGISTER_COUNT = 40;
constexpr int CONSUMER_REGISTER_COUNT = 232;

static_assert(STAGE_COUNT >= 2, "a pipeline needs a stage to load while another computes");
static_assert(BLOCK_TILE_N % STAGING_COLUMNS == 0 &&
                  STAGING_COLUMNS % hopper::BF16_PER_SWIZZLE_ROW == 0,
              "a tile is whole staging rounds, and a round is whole 64-column boxes");
static_assert(GROUP_M >= CLUSTER_ROWS && GROUP_M % CLUSTER_ROWS == 0,
              "a group is whole clusters of row tiles");
static_assert(B_BOX_COUNT % CLUSTER_ROWS == 0, "a column's blocks split B's boxes evenly");
static_assert(BLOCK_TILE_M % (CLUSTER_COLS * hopper::WGMMA_M) == 0,
              "a row's blocks split A's tile into whole 64-row boxes");
static_assert(SHARED_BYTES <= MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the stages must fit in one block's shared memory");
static_assert(hopper::THREADS_PER_WARPGROUP *
                      (PRODUCER_REGISTER_COUNT + (CONSUMER_COUNT * CONSUMER_REGISTER_COUNT)) <=
                  REGISTERS_PER_SM_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the warpgroups' registers must fit in one SM");

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)

constexpr int ACCUMULATOR_COUNT = 2 * hopper::WGMMA_ACCUMULATOR_COUNT;
constexpr int B_BOXES_PER_BLOCK = B_BOX_COUNT / CLUSTER_ROWS;
constexpr int THREADS_PER_WARP = 32;

using C_warpgroup_tile_t = float[ACCUMULATOR_COUNT];

/**
 * @brief `Problem` holds the sizes that every role of the kernel needs.
 *
 * @par Example
 * @code
 * const Problem problem = {M, N, step_count, tile_row_count, tile_col_count, cluster_tile_count};
 * @endcode
 */
struct Problem {
  int M;
  int N;
  int step_count;
  int tile_row_count;
  int tile_col_count;
  int cluster_tile_count;
};

/**
 * @brief `get_grouped_tile` maps a cluster tile, 2 x 2 tiles of C, to the block's tile in grouped
 * order.
 *
 * Cluster tiles are numbered from 0 over pairs of row tiles and pairs of column tiles. Each group
 * of `GROUP_M` row tiles is walked one column pair at a time, down the group's row pairs first.
 * The block of rank r sits at cluster row r % 2 and cluster column r / 2 of its cluster tile.
 *
 * @param[in] problem The problem's tile counts.
 * @param[in] cluster_tile The cluster tile's number.
 * @param[in] cluster_rank The block's rank in its cluster.
 * @param[out] block_i Row of C where the block's tile starts; it may be past C, which loads zeros.
 * @param[out] block_j Column of C where the block's tile starts; likewise.
 *
 * @par Example
 * @code
 * get_grouped_tile(problem, cluster_tile, cluster_rank, block_i, block_j);
 * @endcode
 */
__device__ __forceinline__ void get_grouped_tile(const Problem& problem, int cluster_tile,
                                                 int cluster_rank, int& block_i, int& block_j) {
  constexpr int row_pairs_per_group = GROUP_M / CLUSTER_ROWS;
  const int row_pair_count = (problem.tile_row_count + CLUSTER_ROWS - 1) / CLUSTER_ROWS;
  const int col_pair_count = (problem.tile_col_count + CLUSTER_COLS - 1) / CLUSTER_COLS;
  const int cluster_tiles_per_group = row_pairs_per_group * col_pair_count;
  const int group_first_row_pair = (cluster_tile / cluster_tiles_per_group) * row_pairs_per_group;
  const int group_row_pair_count = min(row_pair_count - group_first_row_pair, row_pairs_per_group);
  const int cluster_tile_in_group = cluster_tile % cluster_tiles_per_group;
  const int row_pair = group_first_row_pair + (cluster_tile_in_group % group_row_pair_count);
  const int col_pair = cluster_tile_in_group / group_row_pair_count;
  block_i = ((row_pair * CLUSTER_ROWS) + (cluster_rank % CLUSTER_ROWS)) * BLOCK_TILE_M;
  block_j = ((col_pair * CLUSTER_COLS) + (cluster_rank / CLUSTER_ROWS)) * BLOCK_TILE_N;
}

/**
 * @brief `produce_tiles` copies every K step of every tile this cluster owns into the stages:
 * half of A's tile, multicast to the blocks in this block's cluster row, and half of B's boxes,
 * multicast to the blocks in its cluster column.
 *
 * @param[in] A_tensor_map TMA descriptor of A, with 64 x 128 boxes (K x M).
 * @param[in] B_tensor_map TMA descriptor of B, with 64 x 64 boxes (N x K).
 * @param[out] stages The stages, 1024-byte aligned: each A's tile, then B's boxes.
 * @param[in] full_barriers One mbarrier per stage that completes when its copies land.
 * @param[in] empty_barriers One mbarrier per stage that completes when both blocks' consumers
 *   release it.
 * @param[in] problem The problem's sizes.
 * @param[in] cluster_rank The block's rank in its cluster.
 *
 * @par Example
 * @code
 * produce_tiles(A_tensor_map, B_tensor_map, stages, full_barriers, empty_barriers, problem,
 *               cluster_rank);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - One thread of the producer warpgroup calls it. The step counter runs across tiles, so the
 *     next tile's copies start as soon as stages free up, while the consumers store C.
 */
__device__ __forceinline__ void produce_tiles(const CUtensorMap& A_tensor_map,
                                              const CUtensorMap& B_tensor_map, uint8_t* stages,
                                              uint64_t* full_barriers, uint64_t* empty_barriers,
                                              const Problem& problem, int cluster_rank) {
  const int cluster = static_cast<int>(blockIdx.x) / CLUSTER_SIZE;
  const int cluster_count = static_cast<int>(gridDim.x) / CLUSTER_SIZE;
  const int cluster_row = cluster_rank % CLUSTER_ROWS;
  const int cluster_col = cluster_rank / CLUSTER_ROWS;
  uint16_t row_mask = 0;
  uint16_t col_mask = 0;
  for (int col = 0; col < CLUSTER_COLS; ++col) {
    row_mask |=
        static_cast<uint16_t>(1U << static_cast<unsigned>((col * CLUSTER_ROWS) + cluster_row));
  }
  for (int row = 0; row < CLUSTER_ROWS; ++row) {
    col_mask |=
        static_cast<uint16_t>(1U << static_cast<unsigned>((cluster_col * CLUSTER_ROWS) + row));
  }
  int global_step = 0;
  for (int cluster_tile = cluster; cluster_tile < problem.cluster_tile_count;
       cluster_tile += cluster_count) {
    int block_i = 0;
    int block_j = 0;
    get_grouped_tile(problem, cluster_tile, cluster_rank, block_i, block_j);
    for (int step = 0; step < problem.step_count; ++step, ++global_step) {
      const int stage = global_step % STAGE_COUNT;
      if (global_step >= STAGE_COUNT) {
        // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN TMA OVERWRITES TILES WGMMA READS
        hopper::wait_barrier(hopper::get_shared_address(&empty_barriers[stage]),
                             ((global_step / STAGE_COUNT) - 1) % 2);
      }
      const uint32_t full_barrier_address = hopper::get_shared_address(&full_barriers[stage]);
      const uint32_t stage_address =
          hopper::get_shared_address(stages + (static_cast<ptrdiff_t>(stage) * STAGE_BYTES));
      const int k_tile_start = step * BLOCK_TILE_K;
      hopper::expect_barrier_bytes(full_barrier_address, STAGE_BYTES);
      hopper::load_tile_multicast_async(
          A_tensor_map, stage_address + (cluster_col * A_ROWS_PER_BLOCK * hopper::SWIZZLE_BYTES),
          full_barrier_address, k_tile_start, block_i + (cluster_col * A_ROWS_PER_BLOCK), row_mask);
      for (int box = cluster_row * B_BOXES_PER_BLOCK; box < (cluster_row + 1) * B_BOXES_PER_BLOCK;
           ++box) {
        hopper::load_tile_multicast_async(
            B_tensor_map, stage_address + A_TILE_BYTES + (box * B_BOX_BYTES), full_barrier_address,
            block_j + (box * hopper::BF16_PER_SWIZZLE_ROW), k_tile_start, col_mask);
      }
    }
  }
}

/**
 * @brief `release_stage` arrives on a stage's empty mbarrier in every block whose producer writes
 * into this block: this block, the others in its cluster row (A) and in its cluster column (B).
 *
 * @param[in] empty_barriers One mbarrier per stage.
 * @param[in] global_step The step whose stage to release, counted across tiles.
 *
 * @par Example
 * @code
 * release_stage(empty_barriers, global_step - 1);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The diagonal block never writes here, so it isn't told; empty mbarriers count accordingly.
 */
__device__ __forceinline__ void release_stage(uint64_t* empty_barriers, int global_step) {
  const uint32_t empty_barrier_address =
      hopper::get_shared_address(&empty_barriers[global_step % STAGE_COUNT]);
  const int cluster_rank = hopper::get_cluster_rank();
  const int cluster_row = cluster_rank % CLUSTER_ROWS;
  const int cluster_col = cluster_rank / CLUSTER_ROWS;
  for (int col = 0; col < CLUSTER_COLS; ++col) {
    hopper::arrive_cluster_barrier(empty_barrier_address, (col * CLUSTER_ROWS) + cluster_row);
  }
  for (int row = 0; row < CLUSTER_ROWS; ++row) {
    if (row != cluster_row) {
      hopper::arrive_cluster_barrier(empty_barrier_address, (cluster_col * CLUSTER_ROWS) + row);
    }
  }
}

/**
 * @brief `consume_tile` multiplies one tile's K steps into a consumer warpgroup's 64 x 256
 * accumulators, releasing each stage in both blocks once its wgmmas finish.
 *
 * @param[in] stages The stages, 1024-byte aligned: each A's tile, then B's boxes.
 * @param[in] full_barriers One mbarrier per stage that completes when its copies land.
 * @param[in] empty_barriers One mbarrier per stage that completes when both blocks' consumers
 *   release it.
 * @param[in] consumer The warpgroup's index among the consumers, so its 64 rows.
 * @param[in] step_count K steps of 64.
 * @param[in,out] global_step The step counter across tiles; it advances by `step_count`.
 * @param[out] C_fp32 This thread's accumulators, zeroed first.
 *
 * @par Example
 * @code
 * consume_tile(stages, full_barriers, empty_barriers, consumer, step_count, global_step, C_fp32);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - It releases the tile's last stage too, so the producer can refill it during the store.
 */
__device__ __forceinline__ void consume_tile(const uint8_t* stages, uint64_t* full_barriers,
                                             uint64_t* empty_barriers, int consumer, int step_count,
                                             int& global_step, C_warpgroup_tile_t& C_fp32) {
  const bool is_releasing_thread = threadIdx.x % hopper::THREADS_PER_WARPGROUP == 0;
#pragma unroll
  for (int r = 0; r < ACCUMULATOR_COUNT; ++r) {
    C_fp32[r] = 0.0F;
  }
  for (int step = 0; step < step_count; ++step, ++global_step) {
    const int stage = global_step % STAGE_COUNT;
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN WGMMA READS TILES STILL IN FLIGHT
    hopper::wait_barrier(hopper::get_shared_address(&full_barriers[stage]),
                         (global_step / STAGE_COUNT) % 2);
    const uint32_t stage_address =
        hopper::get_shared_address(stages + (static_cast<ptrdiff_t>(stage) * STAGE_BYTES));
    const uint32_t A_address = stage_address + (consumer * hopper::WGMMA_M * hopper::SWIZZLE_BYTES);
    const uint32_t B_address = stage_address + A_TILE_BYTES;
    hopper::fence_wgmma();
#pragma unroll
    for (int k = 0; k < BLOCK_TILE_K; k += hopper::WGMMA_K) {
      const uint64_t A_descriptor = hopper::make_descriptor(
          A_address + (k * static_cast<int>(sizeof(bf16_t))), 16, hopper::SWIZZLE_ATOM_BYTES);
      hopper::wgmma_64x256x16(C_fp32, A_descriptor,
                              hopper::make_descriptor(B_address + (k * hopper::SWIZZLE_BYTES),
                                                      B_BOX_BYTES, hopper::SWIZZLE_ATOM_BYTES));
    }
    hopper::commit_wgmma_group();
    hopper::wait_wgmma_groups<1>();
    if (step > 0 && is_releasing_thread) {
      release_stage(empty_barriers, global_step - 1);
    }
  }
  hopper::wait_wgmma_groups<0>();
  if (is_releasing_thread) {
    release_stage(empty_barriers, global_step - 1);
  }
}

/**
 * @brief `store_tile_through_shared` writes a consumer's 64 x 256 accumulators to its staging
 * buffer as bf16 and starts TMA stores of it to C, `STAGING_COLUMNS` columns per round.
 *
 * The buffer holds boxes of 64 x 64, each laid out as TMA's 128-byte swizzle expects: row r's
 * 16-byte chunk q sits at chunk q ^ (r % 8). Each thread writes its 2-column pairs there (wgmma's
 * register layout, as in `wide`), which also spreads a warp's writes over all 32 banks. Then the
 * warpgroup fences and syncs, and its first thread starts the round's TMA stores and commits them.
 *
 * @param[in] C_fp32 This thread's 128 accumulators of the tile.
 * @param[out] staging The consumer's staging buffer, 64 x `STAGING_COLUMNS` bf16, 1024-byte
 *   aligned.
 * @param[in] C_tensor_map TMA descriptor of C, with 64 x 64 boxes.
 * @param[in] consumer The warpgroup's index among the consumers, so its named barrier.
 * @param[in] i_start Row of C where the tile starts.
 * @param[in] j_start Column of C where the tile starts.
 *
 * @par Example
 * @code
 * store_tile_through_shared(C_fp32, staging, C_tensor_map, consumer, i_start, j_start);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Each round first waits until the previous round's stores have read the buffer.
 */
__device__ __forceinline__ void store_tile_through_shared(const float (&C_fp32)[ACCUMULATOR_COUNT],
                                                          uint8_t* staging,
                                                          const CUtensorMap& C_tensor_map,
                                                          int consumer, int i_start, int j_start) {
  constexpr int rows_per_warp = 16;
  constexpr int lanes_per_row = 4;
  constexpr int columns_per_group = 8;
  constexpr int registers_per_group = 4;
  constexpr int half_rows_per_warp = 8;
  constexpr int bytes_per_chunk = 16;
  constexpr int rows_per_swizzle = 8;
  constexpr int box_bytes = hopper::WGMMA_M * hopper::SWIZZLE_BYTES;
  constexpr int round_count = BLOCK_TILE_N / STAGING_COLUMNS;
  constexpr int registers_per_round = ACCUMULATOR_COUNT / round_count;
  constexpr int boxes_per_round = STAGING_COLUMNS / hopper::BF16_PER_SWIZZLE_ROW;
  const bool is_lead_thread = threadIdx.x % hopper::THREADS_PER_WARPGROUP == 0;
  const int lane = static_cast<int>(threadIdx.x) % THREADS_PER_WARP;
  const int warp =
      (static_cast<int>(threadIdx.x) % hopper::THREADS_PER_WARPGROUP) / THREADS_PER_WARP;
  const int barrier_id = 1 + consumer;
#pragma unroll
  for (int round = 0; round < round_count; ++round) {
    if (is_lead_thread) {
      hopper::wait_bulk_stores_read<0>();
    }
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN THREADS OVERWRITE C BEFORE TMA READS IT
    hopper::sync_warpgroup(barrier_id);
#pragma unroll
    for (int r = round * registers_per_round; r < (round + 1) * registers_per_round; r += 2) {
      const int row =
          (warp * rows_per_warp) + (lane / lanes_per_row) + (((r / 2) % 2) * half_rows_per_warp);
      const int col = ((r / registers_per_group) * columns_per_group) +
                      (2 * (lane % lanes_per_row)) - (round * STAGING_COLUMNS);
      const int box = col / hopper::BF16_PER_SWIZZLE_ROW;
      const int row_byte = (col % hopper::BF16_PER_SWIZZLE_ROW) * static_cast<int>(sizeof(bf16_t));
      const int chunk = static_cast<int>(static_cast<unsigned>(row_byte / bytes_per_chunk) ^
                                         static_cast<unsigned>(row % rows_per_swizzle));
      const int offset = (box * box_bytes) + (row * hopper::SWIZZLE_BYTES) +
                         (chunk * bytes_per_chunk) + (row_byte % bytes_per_chunk);
      *reinterpret_cast<__nv_bfloat162*>(staging + offset) =
          __floats2bfloat162_rn(C_fp32[r], C_fp32[r + 1]);
    }
    hopper::fence_async_shared();
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN TMA STORES A HALF-WRITTEN TILE
    hopper::sync_warpgroup(barrier_id);
    if (is_lead_thread) {
      const uint32_t staging_address = hopper::get_shared_address(staging);
      for (int box = 0; box < boxes_per_round; ++box) {
        hopper::store_tile_async(
            C_tensor_map, staging_address + (box * box_bytes),
            j_start + (round * STAGING_COLUMNS) + (box * hopper::BF16_PER_SWIZZLE_ROW), i_start);
      }
      hopper::commit_bulk_stores();
    }
  }
}

/**
 * @brief `matmul_quad_from_tmastore_kernel` is a persistent kernel: each cluster of 2 x 2 blocks
 * loops over 2 x 2 groups of 128 x 256 tiles of C, sharing A's tiles along rows and B's along
 * columns, and stores C through shared memory with TMA.
 *
 * The grid has only as many clusters as can be resident at once. Cluster c takes cluster tiles c,
 * c + clusters, ..., each mapped to a pair of tiles in grouped order. Within a cluster everything
 * is as in `wide`: multicast B, a producer warpgroup and 2 consumer warpgroups, and per-stage
 * mbarriers. But the step counter, and so the stages and mbarrier phases, runs across tiles: the
 * producer loads the next tile while the consumers store the last one, and nothing is set up
 * again per tile. New: each consumer stages its tile in shared memory and stores it with
 * asynchronous TMA stores, which finish while it multiplies the next tile.
 * New here: each block copies half of its A tile and multicasts it to its cluster row, and half
 * of its B boxes, multicast to its cluster column; consumers release stages in all 4 blocks.
 *
 * @param[in] A_tensor_map TMA descriptor of A (M x K, row-major).
 * @param[in] B_tensor_map TMA descriptor of B (K x N, row-major).
 * @param[in] C_tensor_map TMA descriptor of C (M x N, row-major), with 64 x 64 boxes.
 * @param[in] problem The problem's sizes and tile counts.
 *
 * @par Example
 * @code
 * matmul_quad_from_tmastore_kernel<<<CLUSTER_SIZE * cluster_count, THREAD_COUNT,
 * SHARED_BYTES>>>( A_tensor_map, B_tensor_map, C_tensor_map, problem);
 * @endcode
 */
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(THREAD_COUNT, 1)
    matmul_quad_from_tmastore_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                                     const __grid_constant__ CUtensorMap B_tensor_map,
                                     const __grid_constant__ CUtensorMap C_tensor_map,
                                     const Problem problem) {
  extern __shared__ uint8_t shared_bytes[];
  __shared__ alignas(8) uint64_t full_barriers[STAGE_COUNT];
  __shared__ alignas(8) uint64_t empty_barriers[STAGE_COUNT];
  const uint32_t misaligned_bytes =
      hopper::get_shared_address(shared_bytes) % hopper::SWIZZLE_ATOM_BYTES;
  uint8_t* stages =
      shared_bytes + ((hopper::SWIZZLE_ATOM_BYTES - misaligned_bytes) % hopper::SWIZZLE_ATOM_BYTES);
  uint8_t* C_staging = stages + (static_cast<ptrdiff_t>(STAGE_COUNT) * STAGE_BYTES);
  const int warpgroup = static_cast<int>(threadIdx.x) / hopper::THREADS_PER_WARPGROUP;
  const int cluster_rank = hopper::get_cluster_rank();
  if (threadIdx.x == 0) {
    for (int stage = 0; stage < STAGE_COUNT; ++stage) {
      hopper::init_barrier(hopper::get_shared_address(&full_barriers[stage]), 1);
      hopper::init_barrier(hopper::get_shared_address(&empty_barriers[stage]),
                           CONSUMER_COUNT * (CLUSTER_ROWS + CLUSTER_COLS - 1));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN PEERS USE UNINITIALIZED BARRIERS
  hopper::sync_cluster();
  if (warpgroup == 0) {
    hopper::set_max_registers<PRODUCER_REGISTER_COUNT, false>();
    if (threadIdx.x == 0) {
      produce_tiles(A_tensor_map, B_tensor_map, stages, full_barriers, empty_barriers, problem,
                    cluster_rank);
    }
  } else {
    hopper::set_max_registers<CONSUMER_REGISTER_COUNT, true>();
    const int consumer = warpgroup - 1;
    const int cluster = static_cast<int>(blockIdx.x) / CLUSTER_SIZE;
    const int cluster_count = static_cast<int>(gridDim.x) / CLUSTER_SIZE;
    int global_step = 0;
    C_warpgroup_tile_t C_fp32;
    for (int cluster_tile = cluster; cluster_tile < problem.cluster_tile_count;
         cluster_tile += cluster_count) {
      int block_i = 0;
      int block_j = 0;
      get_grouped_tile(problem, cluster_tile, cluster_rank, block_i, block_j);
      consume_tile(stages, full_barriers, empty_barriers, consumer, problem.step_count, global_step,
                   C_fp32);
      store_tile_through_shared(
          C_fp32, C_staging + (static_cast<ptrdiff_t>(consumer) * C_STAGING_BYTES), C_tensor_map,
          consumer, block_i + (consumer * hopper::WGMMA_M), block_j);
    }
    if (threadIdx.x % hopper::THREADS_PER_WARPGROUP == 0) {
      hopper::wait_bulk_stores<0>();
    }
  }
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A PEER WRITES TO AN EXITED BLOCK
  hopper::sync_cluster();
}

#else

/**
 * @brief `Problem` holds the sizes that every role of the kernel needs.
 *
 * @par Example
 * @code
 * const Problem problem = {M, N, step_count, tile_row_count, tile_col_count, cluster_tile_count};
 * @endcode
 */
struct Problem {
  int M;
  int N;
  int step_count;
  int tile_row_count;
  int tile_col_count;
  int cluster_tile_count;
};

/**
 * @brief `matmul_quad_from_tmastore_kernel` traps: this build lacks sm_90a's wgmma and TMA.
 *
 * @param[in] A_tensor_map Unused.
 * @param[in] B_tensor_map Unused.
 * @param[in] C_tensor_map Unused.
 * @param[in] problem Unused.
 *
 * @par Example
 * @code
 * matmul_quad_from_tmastore_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_tensor_map, problem);
 * @endcode
 */
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
    matmul_quad_from_tmastore_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                                     const __grid_constant__ CUtensorMap B_tensor_map,
                                     const __grid_constant__ CUtensorMap C_tensor_map,
                                     const Problem problem) {
  __trap();
}

#endif

/**
 * @brief `get_resident_cluster_count` returns how many clusters of the kernel can run at once.
 *
 * @return The count from `cudaOccupancyMaxActiveClusters`, at least 1.
 *
 * @par Example
 * @code
 * const int cluster_count = get_resident_cluster_count();
 * @endcode
 */
int get_resident_cluster_count() {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(CLUSTER_SIZE);
  config.blockDim = dim3(THREAD_COUNT);
  config.dynamicSmemBytes = SHARED_BYTES;
  int cluster_count = 0;
  CHECK(cudaOccupancyMaxActiveClusters(&cluster_count, matmul_quad_from_tmastore_kernel, &config));
  return cluster_count > 0 ? cluster_count : 1;
}

}  // namespace

/**
 * @brief `matmul_quad_from_tmastore` builds A's, B's and C's TMA descriptors and launches as
 * many clusters of 2 blocks as can be resident, which loop over C's 128 x 256 tiles.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory, 16-byte aligned.
 * @param[in] B_device Right factor, K x N, row-major, in device memory, 16-byte aligned.
 * @param[out] C_device Product, M x N, row-major, in device memory, 16-byte aligned.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_quad_from_tmastore(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8: TMA needs every row to start on a 16-byte boundary.
 * - Gotchas:
 *   - The first call raises the kernel's shared-memory limit and asks how many clusters fit.
 */
void matmul_quad_from_tmastore(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                               int M, int N, int K) {
  static const int resident_cluster_count_by_first_call = [] {
    CHECK(cudaFuncSetAttribute(matmul_quad_from_tmastore_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
    return get_resident_cluster_count();
  }();
  const CUtensorMap A_tensor_map = hopper::make_tensor_map(A_device, M, K, A_ROWS_PER_BLOCK);
  const CUtensorMap B_tensor_map = hopper::make_tensor_map(B_device, K, N, BLOCK_TILE_K);
  const CUtensorMap C_tensor_map = hopper::make_tensor_map(C_device, M, N, hopper::WGMMA_M);
  Problem problem = {};
  problem.M = M;
  problem.N = N;
  problem.step_count = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
  problem.tile_row_count = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
  problem.tile_col_count = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
  problem.cluster_tile_count = ((problem.tile_row_count + CLUSTER_ROWS - 1) / CLUSTER_ROWS) *
                               ((problem.tile_col_count + CLUSTER_COLS - 1) / CLUSTER_COLS);
  const int cluster_count =
      std::min(resident_cluster_count_by_first_call, problem.cluster_tile_count);
  matmul_quad_from_tmastore_kernel<<<CLUSTER_SIZE * cluster_count, THREAD_COUNT, SHARED_BYTES>>>(
      A_tensor_map, B_tensor_map, C_tensor_map, problem);
}
