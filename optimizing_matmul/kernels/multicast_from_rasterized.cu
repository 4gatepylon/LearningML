/**
 * @file
 * @brief `multicast_from_rasterized.cu` defines `matmul_multicast_from_rasterized`: `rasterized`
 * with pairs of blocks in a cluster sharing B's tiles through TMA multicast.
 *
 * @par Interface
 * - `matmul_multicast_from_rasterized`: Launches `matmul_multicast_from_rasterized_kernel`.
 * - `MULTICAST_FROM_RASTERIZED_STAGE_COUNT`: Option: K steps in shared memory at once, at least 2.
 * - `MULTICAST_FROM_RASTERIZED_GROUP_M`: Option: row tiles that consecutive clusters walk down
 *   before moving to the next column tile, a multiple of 2.
 *
 * @par Example
 * @code
 * matmul_multicast_from_rasterized(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - It needs `NVCC_ARCH=sm_90a` (`.env`); built for another architecture, the kernel traps.
 */
#include <cuda.h>

#include <cstdint>

#include "common/matmul.h"
#include "kernels/common/hopper.cuh"

#ifndef MULTICAST_FROM_RASTERIZED_STAGE_COUNT
#define MULTICAST_FROM_RASTERIZED_STAGE_COUNT 3
#endif
#ifndef MULTICAST_FROM_RASTERIZED_GROUP_M
#define MULTICAST_FROM_RASTERIZED_GROUP_M 24
#endif

namespace {

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 256;
constexpr int BLOCK_TILE_K = hopper::BF16_PER_SWIZZLE_ROW;
constexpr int STAGE_COUNT = MULTICAST_FROM_RASTERIZED_STAGE_COUNT;
constexpr int GROUP_M = MULTICAST_FROM_RASTERIZED_GROUP_M;
constexpr int CLUSTER_SIZE = 2;
constexpr int CONSUMER_COUNT = BLOCK_TILE_M / hopper::WGMMA_M;
constexpr int THREAD_COUNT = (1 + CONSUMER_COUNT) * hopper::THREADS_PER_WARPGROUP;
constexpr int A_TILE_BYTES = BLOCK_TILE_M * hopper::SWIZZLE_BYTES;
constexpr int B_BOX_BYTES = BLOCK_TILE_K * hopper::SWIZZLE_BYTES;
constexpr int B_BOX_COUNT = BLOCK_TILE_N / hopper::BF16_PER_SWIZZLE_ROW;
constexpr int STAGE_BYTES = A_TILE_BYTES + (B_BOX_COUNT * B_BOX_BYTES);
constexpr int SHARED_BYTES = (STAGE_COUNT * STAGE_BYTES) + hopper::SWIZZLE_ATOM_BYTES;
constexpr int MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90 = 227 * 1024;
constexpr int REGISTERS_PER_SM_ACCORDING_TO_CUDA_GUIDE_FOR_SM90 = 64 * 1024;
constexpr int PRODUCER_REGISTER_COUNT = 40;
constexpr int CONSUMER_REGISTER_COUNT = 232;

static_assert(STAGE_COUNT >= 2, "a pipeline needs a stage to load while another computes");
static_assert(GROUP_M >= CLUSTER_SIZE && GROUP_M % CLUSTER_SIZE == 0,
              "a group is whole clusters of row tiles");
static_assert(B_BOX_COUNT % CLUSTER_SIZE == 0, "the blocks split B's boxes evenly");
static_assert(SHARED_BYTES <= MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the stages must fit in one block's shared memory");
static_assert(hopper::THREADS_PER_WARPGROUP *
                      (PRODUCER_REGISTER_COUNT + (CONSUMER_COUNT * CONSUMER_REGISTER_COUNT)) <=
                  REGISTERS_PER_SM_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the warpgroups' registers must fit in one SM");

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)

constexpr int WGMMAS_PER_ROW = BLOCK_TILE_N / hopper::WGMMA_N;
constexpr int B_BOXES_PER_BLOCK = B_BOX_COUNT / CLUSTER_SIZE;
constexpr uint16_t ALL_BLOCKS_CTA_MASK = (1U << static_cast<unsigned>(CLUSTER_SIZE)) - 1U;
constexpr int THREADS_PER_WARP = 32;

using C_warpgroup_tile_t = float[WGMMAS_PER_ROW][hopper::WGMMA_ACCUMULATOR_COUNT];

/**
 * @brief `get_grouped_tile` maps a cluster's block to its tile of C in grouped order.
 *
 * Clusters are numbered row-major over pairs of row tiles and column tiles. Each group of
 * `GROUP_M` row tiles is walked column tile by column tile, down the group's pairs first. The
 * block of rank r in a cluster takes row tile 2 * pair + r of its column tile.
 *
 * @param[in] tile_row_count Row tiles of C.
 * @param[in] tile_col_count Column tiles of C.
 * @param[in] cluster_rank The block's rank in its cluster.
 * @param[out] tile_row The block's row tile; it may be past C's last row tile, which loads zeros.
 * @param[out] tile_col The block's column tile.
 *
 * @par Example
 * @code
 * get_grouped_tile(tile_row_count, tile_col_count, cluster_rank, tile_row, tile_col);
 * @endcode
 */
__device__ __forceinline__ void get_grouped_tile(int tile_row_count, int tile_col_count,
                                                 int cluster_rank, int& tile_row, int& tile_col) {
  constexpr int pairs_per_group = GROUP_M / CLUSTER_SIZE;
  const int pair_row_count = (tile_row_count + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
  const int cluster = (static_cast<int>(blockIdx.x) / CLUSTER_SIZE) +
                      (static_cast<int>(blockIdx.y) * tile_col_count);
  const int clusters_per_group = pairs_per_group * tile_col_count;
  const int group_first_pair = (cluster / clusters_per_group) * pairs_per_group;
  const int group_pair_count = min(pair_row_count - group_first_pair, pairs_per_group);
  const int cluster_in_group = cluster % clusters_per_group;
  const int pair_row = group_first_pair + (cluster_in_group % group_pair_count);
  tile_row = (pair_row * CLUSTER_SIZE) + cluster_rank;
  tile_col = cluster_in_group / group_pair_count;
}

/**
 * @brief `produce_stages` copies every K step's tiles into the stages: this block's A tile, and
 * its share of B's boxes multicast to both blocks of the cluster.
 *
 * @param[in] A_tensor_map TMA descriptor of A, with 64 x 128 boxes (K x M).
 * @param[in] B_tensor_map TMA descriptor of B, with 64 x 64 boxes (N x K).
 * @param[out] stages The stages, 1024-byte aligned: each A's tile, then B's boxes.
 * @param[in] full_barriers One mbarrier per stage that completes when its copies land.
 * @param[in] empty_barriers One mbarrier per stage that completes when both blocks' consumers
 *   release it.
 * @param[in] cluster_rank The block's rank in its cluster.
 * @param[in] block_i Row of C where the block's tile starts.
 * @param[in] block_j Column of C where the cluster's tiles start.
 * @param[in] step_count K steps of 64.
 *
 * @par Example
 * @code
 * produce_stages(A_tensor_map, B_tensor_map, stages, full_barriers, empty_barriers, cluster_rank,
 *                block_i, block_j, step_count);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - One thread of the producer warpgroup calls it. A stage's B boxes come from both blocks, so
 *     its full mbarrier still expects the whole stage's bytes.
 */
__device__ __forceinline__ void produce_stages(const CUtensorMap& A_tensor_map,
                                               const CUtensorMap& B_tensor_map, uint8_t* stages,
                                               uint64_t* full_barriers, uint64_t* empty_barriers,
                                               int cluster_rank, int block_i, int block_j,
                                               int step_count) {
  for (int step = 0; step < step_count; ++step) {
    const int stage = step % STAGE_COUNT;
    if (step >= STAGE_COUNT) {
      // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN TMA OVERWRITES TILES WGMMA STILL
      // READS
      hopper::wait_barrier(hopper::get_shared_address(&empty_barriers[stage]),
                           ((step / STAGE_COUNT) - 1) % 2);
    }
    const uint32_t full_barrier_address = hopper::get_shared_address(&full_barriers[stage]);
    const uint32_t stage_address =
        hopper::get_shared_address(stages + (static_cast<ptrdiff_t>(stage) * STAGE_BYTES));
    const int k_tile_start = step * BLOCK_TILE_K;
    hopper::expect_barrier_bytes(full_barrier_address, STAGE_BYTES);
    hopper::load_tile_async(A_tensor_map, stage_address, full_barrier_address, k_tile_start,
                            block_i);
    for (int box = cluster_rank * B_BOXES_PER_BLOCK; box < (cluster_rank + 1) * B_BOXES_PER_BLOCK;
         ++box) {
      hopper::load_tile_multicast_async(
          B_tensor_map, stage_address + A_TILE_BYTES + (box * B_BOX_BYTES), full_barrier_address,
          block_j + (box * hopper::BF16_PER_SWIZZLE_ROW), k_tile_start, ALL_BLOCKS_CTA_MASK);
    }
  }
}

/**
 * @brief `consume_stages` multiplies every K step's tiles into a consumer warpgroup's 64 x 256
 * accumulators, releasing each stage in both blocks of the cluster once its wgmmas finish.
 *
 * @param[in] stages The stages, 1024-byte aligned: each A's tile, then B's boxes.
 * @param[in] full_barriers One mbarrier per stage that completes when its copies land.
 * @param[in] empty_barriers One mbarrier per stage that completes when both blocks' consumers
 *   release it.
 * @param[in] consumer The warpgroup's index among the consumers, so its 64 rows.
 * @param[in] step_count K steps of 64.
 * @param[in,out] C_fp32 This thread's accumulators.
 *
 * @par Example
 * @code
 * consume_stages(stages, full_barriers, empty_barriers, consumer, step_count, C_fp32);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Both blocks' producers multicast into this block's stage, so it is released in both.
 */
__device__ __forceinline__ void consume_stages(const uint8_t* stages, uint64_t* full_barriers,
                                               uint64_t* empty_barriers, int consumer,
                                               int step_count, C_warpgroup_tile_t& C_fp32) {
  const bool is_releasing_thread = threadIdx.x % hopper::THREADS_PER_WARPGROUP == 0;
  for (int step = 0; step < step_count; ++step) {
    const int stage = step % STAGE_COUNT;
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN WGMMA READS TILES STILL IN FLIGHT
    hopper::wait_barrier(hopper::get_shared_address(&full_barriers[stage]),
                         (step / STAGE_COUNT) % 2);
    const uint32_t stage_address =
        hopper::get_shared_address(stages + (static_cast<ptrdiff_t>(stage) * STAGE_BYTES));
    const uint32_t A_address = stage_address + (consumer * hopper::WGMMA_M * hopper::SWIZZLE_BYTES);
    const uint32_t B_address = stage_address + A_TILE_BYTES;
    hopper::fence_wgmma();
#pragma unroll
    for (int k = 0; k < BLOCK_TILE_K; k += hopper::WGMMA_K) {
      const uint64_t A_descriptor = hopper::make_descriptor(
          A_address + (k * static_cast<int>(sizeof(bf16_t))), 16, hopper::SWIZZLE_ATOM_BYTES);
#pragma unroll
      for (int n = 0; n < WGMMAS_PER_ROW; ++n) {
        const uint32_t B_slice_address =
            B_address + (n * (hopper::WGMMA_N / hopper::BF16_PER_SWIZZLE_ROW) * B_BOX_BYTES) +
            (k * hopper::SWIZZLE_BYTES);
        hopper::wgmma_64x128x16(
            C_fp32[n], A_descriptor,
            hopper::make_descriptor(B_slice_address, B_BOX_BYTES, hopper::SWIZZLE_ATOM_BYTES));
      }
    }
    hopper::commit_wgmma_group();
    hopper::wait_wgmma_groups<1>();
    if (step > 0 && is_releasing_thread) {
      const uint32_t empty_barrier_address =
          hopper::get_shared_address(&empty_barriers[(step - 1) % STAGE_COUNT]);
      for (int cta_rank = 0; cta_rank < CLUSTER_SIZE; ++cta_rank) {
        hopper::arrive_cluster_barrier(empty_barrier_address, cta_rank);
      }
    }
  }
  hopper::wait_wgmma_groups<0>();
}

/**
 * @brief `store_accumulators` writes a warpgroup's 64 x 128 accumulator tile to C as bf16.
 *
 * Thread t of the warpgroup holds rows 16 * (t / 32) + (t % 32) / 4 and 8 more, and for each 8
 * columns the 2 at 2 * (t % 4); register r is row +8 when r % 4 >= 2, column +1 when r is odd.
 *
 * @param[in] C_fp32 This thread's 64 accumulators of the tile.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of C.
 * @param[in] N Columns of C, a multiple of 8.
 * @param[in] i_start Row of C where the tile starts.
 * @param[in] j_start Column of C where the tile starts.
 *
 * @par Example
 * @code
 * store_accumulators(C_fp32[n], C_device, M, N, i_start, j_start);
 * @endcode
 */
__device__ __forceinline__ void store_accumulators(
    const float (&C_fp32)[hopper::WGMMA_ACCUMULATOR_COUNT], bf16_t* C_device, int M, int N,
    int64_t i_start, int64_t j_start) {
  constexpr int rows_per_warp = 16;
  constexpr int lanes_per_row = 4;
  constexpr int columns_per_group = 8;
  constexpr int registers_per_group = 4;
  constexpr int half_rows_per_warp = 8;
  const int lane = static_cast<int>(threadIdx.x) % THREADS_PER_WARP;
  const int warp =
      (static_cast<int>(threadIdx.x) % hopper::THREADS_PER_WARPGROUP) / THREADS_PER_WARP;
#pragma unroll
  for (int r = 0; r < hopper::WGMMA_ACCUMULATOR_COUNT; r += 2) {
    const int tile_row =
        (warp * rows_per_warp) + (lane / lanes_per_row) + (((r / 2) % 2) * half_rows_per_warp);
    const int tile_col =
        ((r / registers_per_group) * columns_per_group) + (2 * (lane % lanes_per_row));
    const int64_t i = i_start + tile_row;
    const int64_t j = j_start + tile_col;
    if (i < M && j < N) {
      *reinterpret_cast<__nv_bfloat162*>(&C_device[(i * N) + j]) =
          __floats2bfloat162_rn(C_fp32[r], C_fp32[r + 1]);
    }
  }
}

/**
 * @brief `matmul_multicast_from_rasterized_kernel` computes a 128 x 256 tile of C; the 2 blocks
 * of a cluster take vertically adjacent tiles and share B's.
 *
 * As in `rasterized`: warpgroup 0 copies with TMA, and warpgroups 1 and 2 multiply with wgmma,
 * synchronized by per-stage full and empty mbarriers, in grouped tile order. New: each block's
 * producer copies its own A tile and half of B's boxes, multicast into both blocks, so L2 serves
 * B once per pair of blocks. A stage is free only when both blocks' consumers released it, so
 * consumers arrive on both blocks' empty mbarriers. Cluster syncs after setup and before exit keep
 * a block from receiving copies or arrivals before its barriers exist or after it exits.
 *
 * @param[in] A_tensor_map TMA descriptor of A (M x K, row-major).
 * @param[in] B_tensor_map TMA descriptor of B (K x N, row-major).
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_multicast_from_rasterized_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The grid's x size is twice the column tiles: x / 2 is the column tile, x % 2 the rank.
 */
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(THREAD_COUNT, 1)
    matmul_multicast_from_rasterized_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                                            const __grid_constant__ CUtensorMap B_tensor_map,
                                            bf16_t* C_device, int M, int N, int K) {
  extern __shared__ uint8_t shared_bytes[];
  __shared__ alignas(8) uint64_t full_barriers[STAGE_COUNT];
  __shared__ alignas(8) uint64_t empty_barriers[STAGE_COUNT];
  const uint32_t misaligned_bytes =
      hopper::get_shared_address(shared_bytes) % hopper::SWIZZLE_ATOM_BYTES;
  uint8_t* stages =
      shared_bytes + ((hopper::SWIZZLE_ATOM_BYTES - misaligned_bytes) % hopper::SWIZZLE_ATOM_BYTES);
  const int warpgroup = static_cast<int>(threadIdx.x) / hopper::THREADS_PER_WARPGROUP;
  const int cluster_rank = hopper::get_cluster_rank();
  int tile_row = 0;
  int tile_col = 0;
  get_grouped_tile((M + BLOCK_TILE_M - 1) / BLOCK_TILE_M,
                   static_cast<int>(gridDim.x) / CLUSTER_SIZE, cluster_rank, tile_row, tile_col);
  const int block_i = tile_row * BLOCK_TILE_M;
  const int block_j = tile_col * BLOCK_TILE_N;
  const int step_count = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
  if (threadIdx.x == 0) {
    for (int stage = 0; stage < STAGE_COUNT; ++stage) {
      hopper::init_barrier(hopper::get_shared_address(&full_barriers[stage]), 1);
      hopper::init_barrier(hopper::get_shared_address(&empty_barriers[stage]),
                           CONSUMER_COUNT * CLUSTER_SIZE);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN PEERS USE UNINITIALIZED BARRIERS
  hopper::sync_cluster();
  if (warpgroup == 0) {
    hopper::set_max_registers<PRODUCER_REGISTER_COUNT, false>();
    if (threadIdx.x == 0) {
      produce_stages(A_tensor_map, B_tensor_map, stages, full_barriers, empty_barriers,
                     cluster_rank, block_i, block_j, step_count);
    }
  } else {
    hopper::set_max_registers<CONSUMER_REGISTER_COUNT, true>();
    const int consumer = warpgroup - 1;
    C_warpgroup_tile_t C_fp32 = {};
    consume_stages(stages, full_barriers, empty_barriers, consumer, step_count, C_fp32);
#pragma unroll
    for (int n = 0; n < WGMMAS_PER_ROW; ++n) {
      store_accumulators(C_fp32[n], C_device, M, N, block_i + (consumer * hopper::WGMMA_M),
                         block_j + (n * hopper::WGMMA_N));
    }
  }
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A PEER WRITES TO AN EXITED BLOCK
  hopper::sync_cluster();
}

#else

/**
 * @brief `matmul_multicast_from_rasterized_kernel` traps: this build lacks sm_90a's wgmma and TMA.
 *
 * @param[in] A_tensor_map Unused.
 * @param[in] B_tensor_map Unused.
 * @param[out] C_device Unused.
 * @param[in] M Unused.
 * @param[in] N Unused.
 * @param[in] K Unused.
 *
 * @par Example
 * @code
 * matmul_multicast_from_rasterized_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_device, M, N, K);
 * @endcode
 */
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
    matmul_multicast_from_rasterized_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                                            const __grid_constant__ CUtensorMap B_tensor_map,
                                            bf16_t* C_device, int M, int N, int K) {
  __trap();
}

#endif

}  // namespace

/**
 * @brief `matmul_multicast_from_rasterized` builds A's and B's TMA descriptors and launches
 * clusters of 2 blocks of 384 threads, one block per 128 x 256 tile of C.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory, 16-byte aligned.
 * @param[in] B_device Right factor, K x N, row-major, in device memory, 16-byte aligned.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_multicast_from_rasterized(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8: TMA needs every row to start on a 16-byte boundary.
 * - Gotchas:
 *   - The first call raises the kernel's dynamic shared-memory limit, which defaults to 48 KB.
 */
void matmul_multicast_from_rasterized(const bf16_t* A_device, const bf16_t* B_device,
                                      bf16_t* C_device, int M, int N, int K) {
  static const bool is_shared_limit_raised_by_first_call = [] {
    CHECK(cudaFuncSetAttribute(matmul_multicast_from_rasterized_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
    return true;
  }();
  (void)is_shared_limit_raised_by_first_call;
  const CUtensorMap A_tensor_map = hopper::make_tensor_map(A_device, M, K, BLOCK_TILE_M);
  const CUtensorMap B_tensor_map = hopper::make_tensor_map(B_device, K, N, BLOCK_TILE_K);
  const int tile_row_count = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
  const int tile_col_count = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
  const dim3 grid(CLUSTER_SIZE * tile_col_count,
                  (tile_row_count + CLUSTER_SIZE - 1) / CLUSTER_SIZE);
  matmul_multicast_from_rasterized_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(
      A_tensor_map, B_tensor_map, C_device, M, N, K);
}
