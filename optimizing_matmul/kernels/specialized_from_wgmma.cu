/**
 * @file
 * @brief `specialized_from_wgmma.cu` defines `matmul_specialized_from_wgmma`: `wgmma` with warp
 * specialization, a producer warpgroup that only copies and consumer warpgroups that only multiply.
 *
 * @par Interface
 * - `matmul_specialized_from_wgmma`: Launches `matmul_specialized_from_wgmma_kernel`.
 * - `SPECIALIZED_FROM_WGMMA_BLOCK_TILE_N`: Option: the block tile's width, 128 or 256.
 * - `SPECIALIZED_FROM_WGMMA_STAGE_COUNT`: Option: the K steps in shared memory at once, at least 2.
 *
 * @par Example
 * @code
 * matmul_specialized_from_wgmma(A_device, B_device, C_device, M, N, K);
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

#ifndef SPECIALIZED_FROM_WGMMA_BLOCK_TILE_N
#define SPECIALIZED_FROM_WGMMA_BLOCK_TILE_N 256
#endif
#ifndef SPECIALIZED_FROM_WGMMA_STAGE_COUNT
#define SPECIALIZED_FROM_WGMMA_STAGE_COUNT 3
#endif

namespace {

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = SPECIALIZED_FROM_WGMMA_BLOCK_TILE_N;
constexpr int BLOCK_TILE_K = hopper::BF16_PER_SWIZZLE_ROW;
constexpr int STAGE_COUNT = SPECIALIZED_FROM_WGMMA_STAGE_COUNT;
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

static_assert(BLOCK_TILE_N % hopper::WGMMA_N == 0, "the block tile is whole wgmma tiles");
static_assert(STAGE_COUNT >= 2, "a pipeline needs a stage to load while another computes");
static_assert(SHARED_BYTES <= MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the stages must fit in one block's shared memory");
static_assert(hopper::THREADS_PER_WARPGROUP *
                      (PRODUCER_REGISTER_COUNT + (CONSUMER_COUNT * CONSUMER_REGISTER_COUNT)) <=
                  REGISTERS_PER_SM_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the warpgroups' registers must fit in one SM");

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)

constexpr int WGMMAS_PER_ROW = BLOCK_TILE_N / hopper::WGMMA_N;
constexpr int THREADS_PER_WARP = 32;

using C_warpgroup_tile_t = float[WGMMAS_PER_ROW][hopper::WGMMA_ACCUMULATOR_COUNT];

/**
 * @brief `produce_stages` copies every K step's tiles into the stages, waiting for each stage to
 * be released before refilling it.
 *
 * @param[in] A_tensor_map TMA descriptor of A, with 64 x 128 boxes (K x M).
 * @param[in] B_tensor_map TMA descriptor of B, with 64 x 64 boxes (N x K).
 * @param[out] stages The stages, 1024-byte aligned: each A's tile, then B's boxes.
 * @param[in] full_barriers One mbarrier per stage that completes when its copies land.
 * @param[in] empty_barriers One mbarrier per stage that completes when the consumers release it.
 * @param[in] block_i Row of C where the block's tile starts.
 * @param[in] block_j Column of C where the block's tile starts.
 * @param[in] step_count K steps of 64.
 *
 * @par Example
 * @code
 * produce_stages(A_tensor_map, B_tensor_map, stages, full_barriers, empty_barriers, block_i,
 *                block_j, step_count);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - One thread of the producer warpgroup calls it.
 */
__device__ __forceinline__ void produce_stages(const CUtensorMap& A_tensor_map,
                                               const CUtensorMap& B_tensor_map, uint8_t* stages,
                                               uint64_t* full_barriers, uint64_t* empty_barriers,
                                               int block_i, int block_j, int step_count) {
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
    for (int box = 0; box < B_BOX_COUNT; ++box) {
      hopper::load_tile_async(B_tensor_map, stage_address + A_TILE_BYTES + (box * B_BOX_BYTES),
                              full_barrier_address, block_j + (box * hopper::BF16_PER_SWIZZLE_ROW),
                              k_tile_start);
    }
  }
}

/**
 * @brief `consume_stages` multiplies every K step's tiles into a consumer warpgroup's 64 x
 * `BLOCK_TILE_N` accumulators, releasing each stage once its wgmmas finish.
 *
 * @param[in] stages The stages, 1024-byte aligned: each A's tile, then B's boxes.
 * @param[in] full_barriers One mbarrier per stage that completes when its copies land.
 * @param[in] empty_barriers One mbarrier per stage that completes when the consumers release it.
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
 * - Performance:
 *   - It keeps one step's wgmmas in flight while issuing the next, and releases the step before.
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
      hopper::arrive_barrier(hopper::get_shared_address(&empty_barriers[(step - 1) % STAGE_COUNT]));
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
 * @brief `matmul_specialized_from_wgmma_kernel` computes a 128 x `BLOCK_TILE_N` tile of C with a
 * producer warpgroup and 2 consumer warpgroups.
 *
 * Warpgroup 0 gives up registers and one of its threads copies every K step's tiles with TMA,
 * waiting on a stage's empty mbarrier before refilling it. Warpgroups 1 and 2 take the registers,
 * wait on each stage's full mbarrier, multiply their 64 rows with wgmma, and release the stage.
 * No block-wide sync runs after setup, so copies and multiplies overlap.
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
 * matmul_specialized_from_wgmma_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_device, M, N, K);
 * @endcode
 */
__global__ void __launch_bounds__(THREAD_COUNT, 1)
    matmul_specialized_from_wgmma_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
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
  const int block_i = static_cast<int>(blockIdx.y) * BLOCK_TILE_M;
  const int block_j = static_cast<int>(blockIdx.x) * BLOCK_TILE_N;
  const int step_count = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
  if (threadIdx.x == 0) {
    for (int stage = 0; stage < STAGE_COUNT; ++stage) {
      hopper::init_barrier(hopper::get_shared_address(&full_barriers[stage]), 1);
      hopper::init_barrier(hopper::get_shared_address(&empty_barriers[stage]), CONSUMER_COUNT);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN THREADS USE UNINITIALIZED BARRIERS
  __syncthreads();
  if (warpgroup == 0) {
    hopper::set_max_registers<PRODUCER_REGISTER_COUNT, false>();
    if (threadIdx.x == 0) {
      produce_stages(A_tensor_map, B_tensor_map, stages, full_barriers, empty_barriers, block_i,
                     block_j, step_count);
    }
    return;
  }
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

#else

/**
 * @brief `matmul_specialized_from_wgmma_kernel` traps: this build lacks sm_90a's wgmma and TMA.
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
 * matmul_specialized_from_wgmma_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_device, M, N, K);
 * @endcode
 */
__global__ void matmul_specialized_from_wgmma_kernel(
    const __grid_constant__ CUtensorMap A_tensor_map,
    const __grid_constant__ CUtensorMap B_tensor_map, bf16_t* C_device, int M, int N, int K) {
  __trap();
}

#endif

}  // namespace

/**
 * @brief `matmul_specialized_from_wgmma` builds A's and B's TMA descriptors and launches a
 * 384-thread block per 128 x `BLOCK_TILE_N` tile of C.
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
 * matmul_specialized_from_wgmma(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8: TMA needs every row to start on a 16-byte boundary.
 * - Gotchas:
 *   - The first call raises the kernel's dynamic shared-memory limit, which defaults to 48 KB.
 */
void matmul_specialized_from_wgmma(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                                   int M, int N, int K) {
  static const bool is_shared_limit_raised_by_first_call = [] {
    CHECK(cudaFuncSetAttribute(matmul_specialized_from_wgmma_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
    return true;
  }();
  (void)is_shared_limit_raised_by_first_call;
  const CUtensorMap A_tensor_map = hopper::make_tensor_map(A_device, M, K, BLOCK_TILE_M);
  const CUtensorMap B_tensor_map = hopper::make_tensor_map(B_device, K, N, BLOCK_TILE_K);
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_specialized_from_wgmma_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(
      A_tensor_map, B_tensor_map, C_device, M, N, K);
}
