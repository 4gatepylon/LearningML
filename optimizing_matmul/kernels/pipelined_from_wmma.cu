/**
 * @file
 * @brief `pipelined_from_wmma.cu` defines `matmul_pipelined_from_wmma`: `wmma` with a multi-stage
 * `cp.async` pipeline, so the tiles for later steps along K load while the tensor cores work.
 *
 * @par Interface
 * - `matmul_pipelined_from_wmma`: Launches `matmul_pipelined_from_wmma_kernel`.
 * - `PIPELINED_FROM_WMMA_BLOCK_TILE_K`: Option: the width of a step along K, a multiple of 16.
 * - `PIPELINED_FROM_WMMA_STAGE_COUNT`: Option: the steps in shared memory at once, at least 2.
 *
 * @par Example
 * @code
 * matmul_pipelined_from_wmma(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdint>

#include "common/matmul.h"

#ifndef PIPELINED_FROM_WMMA_BLOCK_TILE_K
#define PIPELINED_FROM_WMMA_BLOCK_TILE_K 128
#endif
#ifndef PIPELINED_FROM_WMMA_STAGE_COUNT
#define PIPELINED_FROM_WMMA_STAGE_COUNT 2
#endif

namespace {

namespace wmma = nvcuda::wmma;

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 128;
constexpr int BLOCK_TILE_K = PIPELINED_FROM_WMMA_BLOCK_TILE_K;
constexpr int STAGE_COUNT = PIPELINED_FROM_WMMA_STAGE_COUNT;
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_N = 32;
constexpr int MMA_SIZE = 16;
constexpr int MMA_TILES_PER_WARP_M = WARP_TILE_M / MMA_SIZE;
constexpr int MMA_TILES_PER_WARP_N = WARP_TILE_N / MMA_SIZE;
constexpr int WARPS_PER_BLOCK_N = BLOCK_TILE_N / WARP_TILE_N;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARP_COUNT = (BLOCK_TILE_M / WARP_TILE_M) * WARPS_PER_BLOCK_N;
constexpr int THREAD_COUNT = WARP_COUNT * THREADS_PER_WARP;
constexpr int BF16_PER_VECTOR = 8;
constexpr int BYTES_PER_VECTOR = 16;
constexpr int PADDING_BF16 = 8;
constexpr int A_TILE_STRIDE = BLOCK_TILE_K + PADDING_BF16;
constexpr int B_TILE_STRIDE = BLOCK_TILE_N + PADDING_BF16;
constexpr int A_TILE_BF16_COUNT = BLOCK_TILE_M * A_TILE_STRIDE;
constexpr int STAGE_BF16_COUNT = A_TILE_BF16_COUNT + (BLOCK_TILE_K * B_TILE_STRIDE);
constexpr int SHARED_BYTES = STAGE_COUNT * STAGE_BF16_COUNT * static_cast<int>(sizeof(bf16_t));
constexpr int SCRATCH_FLOAT_COUNT = MMA_SIZE * MMA_SIZE;
constexpr int MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90 = 227 * 1024;

static_assert(BLOCK_TILE_K % MMA_SIZE == 0, "a step along K is a whole number of MMA steps");
static_assert(STAGE_COUNT >= 2, "a pipeline needs a stage to load while another computes");
static_assert(SHARED_BYTES <= MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the stages must fit in one block's shared memory");
static_assert(SHARED_BYTES >= WARP_COUNT * SCRATCH_FLOAT_COUNT * static_cast<int>(sizeof(float)),
              "the stages' memory is reused as the warps' scratch tiles");

using A_fragment_t =
    wmma::fragment<wmma::matrix_a, MMA_SIZE, MMA_SIZE, MMA_SIZE, bf16_t, wmma::row_major>;
using B_fragment_t =
    wmma::fragment<wmma::matrix_b, MMA_SIZE, MMA_SIZE, MMA_SIZE, bf16_t, wmma::row_major>;
using C_fragment_t = wmma::fragment<wmma::accumulator, MMA_SIZE, MMA_SIZE, MMA_SIZE, float>;
using C_warp_tile_t = C_fragment_t[MMA_TILES_PER_WARP_M][MMA_TILES_PER_WARP_N];

/**
 * @brief `issue_stage_loads` starts the asynchronous copies of one step's tiles into a stage.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] stage The stage: A's tile (stride `A_TILE_STRIDE`), then B's (`B_TILE_STRIDE`).
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 * @param[in] block_i Row of C where the block's tile starts.
 * @param[in] block_j Column of C where the block's tile starts.
 * @param[in] k_tile_start Column of A and row of B where the step starts.
 *
 * @par Example
 * @code
 * issue_stage_loads(A_device, B_device, stage, M, N, K, block_i, block_j, k_tile_start);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The copies land only after `__pipeline_wait_prior`; a vector outside A or B lands as zeros.
 */
__device__ __forceinline__ void issue_stage_loads(const bf16_t* A_device, const bf16_t* B_device,
                                                  bf16_t* stage, int M, int N, int K,
                                                  int64_t block_i, int64_t block_j,
                                                  int64_t k_tile_start) {
  bf16_t* A_tile = stage;
  bf16_t* B_tile = stage + A_TILE_BF16_COUNT;
  constexpr int A_vectors_per_row = BLOCK_TILE_K / BF16_PER_VECTOR;
  for (int v = static_cast<int>(threadIdx.x); v < BLOCK_TILE_M * A_vectors_per_row;
       v += THREAD_COUNT) {
    const int row = v / A_vectors_per_row;
    const int col = (v % A_vectors_per_row) * BF16_PER_VECTOR;
    const int64_t A_i = block_i + row;
    const int64_t A_k = k_tile_start + col;
    const bool is_inside = A_i < M && A_k < K;
    __pipeline_memcpy_async(&A_tile[(row * A_TILE_STRIDE) + col],
                            is_inside ? &A_device[(A_i * K) + A_k] : A_device, BYTES_PER_VECTOR,
                            is_inside ? 0 : BYTES_PER_VECTOR);
  }
  constexpr int B_vectors_per_row = BLOCK_TILE_N / BF16_PER_VECTOR;
  for (int v = static_cast<int>(threadIdx.x); v < BLOCK_TILE_K * B_vectors_per_row;
       v += THREAD_COUNT) {
    const int row = v / B_vectors_per_row;
    const int col = (v % B_vectors_per_row) * BF16_PER_VECTOR;
    const int64_t B_k = k_tile_start + row;
    const int64_t B_j = block_j + col;
    const bool is_inside = B_k < K && B_j < N;
    __pipeline_memcpy_async(&B_tile[(row * B_TILE_STRIDE) + col],
                            is_inside ? &B_device[(B_k * N) + B_j] : B_device, BYTES_PER_VECTOR,
                            is_inside ? 0 : BYTES_PER_VECTOR);
  }
}

/**
 * @brief `add_stage_products` adds one stage's products to a warp's 64 x 32 tile of C.
 *
 * @param[in] stage The stage: A's tile, then B's.
 * @param[in] warp_row_start Row of the block's tile where the warp's tile starts.
 * @param[in] warp_col_start Column of the block's tile where the warp's tile starts.
 * @param[in,out] C_fragments The warp's 4 x 2 accumulator fragments, 16 x 16 each.
 *
 * @par Example
 * @code
 * add_stage_products(stage, warp_row_start, warp_col_start, C_fragments);
 * @endcode
 */
__device__ __forceinline__ void add_stage_products(const bf16_t* stage, int warp_row_start,
                                                   int warp_col_start, C_warp_tile_t& C_fragments) {
  const bf16_t* A_tile = stage;
  const bf16_t* B_tile = stage + A_TILE_BF16_COUNT;
  A_fragment_t A_fragments[MMA_TILES_PER_WARP_M];
  B_fragment_t B_fragments[MMA_TILES_PER_WARP_N];
  for (int k = 0; k < BLOCK_TILE_K; k += MMA_SIZE) {
    for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
      const int row = warp_row_start + (m * MMA_SIZE);
      wmma::load_matrix_sync(A_fragments[m], &A_tile[(row * A_TILE_STRIDE) + k], A_TILE_STRIDE);
    }
    for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
      const int col = warp_col_start + (n * MMA_SIZE);
      wmma::load_matrix_sync(B_fragments[n], &B_tile[(k * B_TILE_STRIDE) + col], B_TILE_STRIDE);
    }
    for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
      for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
        wmma::mma_sync(C_fragments[m][n], A_fragments[m], B_fragments[n], C_fragments[m][n]);
      }
    }
  }
}

/**
 * @brief `store_warp_tile` writes a warp's 64 x 32 tile of C as bf16, skipping elements outside C.
 *
 * @param[in] C_fragments The warp's accumulator fragments.
 * @param[out] scratch The warp's 16 x 16 fp32 scratch tile, in shared memory.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of C.
 * @param[in] N Columns of C.
 * @param[in] i_start Row of C where the warp's tile starts.
 * @param[in] j_start Column of C where the warp's tile starts.
 *
 * @par Example
 * @code
 * store_warp_tile(C_fragments, scratch, C_device, M, N, i_start, j_start);
 * @endcode
 */
__device__ __forceinline__ void store_warp_tile(const C_warp_tile_t& C_fragments, float* scratch,
                                                bf16_t* C_device, int M, int N, int64_t i_start,
                                                int64_t j_start) {
  const int lane = static_cast<int>(threadIdx.x) % THREADS_PER_WARP;
  for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
    for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
      wmma::store_matrix_sync(scratch, C_fragments[m][n], MMA_SIZE, wmma::mem_row_major);
      // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN LANES READ UNWRITTEN SCRATCH
      __syncwarp();
      for (int e = lane; e < SCRATCH_FLOAT_COUNT; e += THREADS_PER_WARP) {
        const int64_t i = i_start + static_cast<int64_t>((m * MMA_SIZE) + (e / MMA_SIZE));
        const int64_t j = j_start + static_cast<int64_t>((n * MMA_SIZE) + (e % MMA_SIZE));
        if (i < M && j < N) {
          C_device[(i * N) + j] = __float2bfloat16(scratch[e]);
        }
      }
      // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN LANES OVERWRITE UNREAD SCRATCH
      __syncwarp();
    }
  }
}

/**
 * @brief `matmul_pipelined_from_wmma_kernel` computes a 128 x 128 tile of C on tensor cores while
 * the next steps' tiles load.
 *
 * Shared memory holds `STAGE_COUNT` stages, each one step's A and B tiles. The block first starts
 * the copies for steps 0 to `STAGE_COUNT` - 2. Then, for each step t, it waits for step t's copies,
 * syncs, starts the copies for step t + `STAGE_COUNT` - 1 into the stage that step t - 1 used, and
 * multiplies step t's tiles as in `wmma`. One sync per step is enough: after it, every warp is
 * done with step t - 1's stage.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_pipelined_from_wmma_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_device, B_device,
 *                                                                        C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Every step commits a copy group, even an empty one, so the wait counts stay right.
 */
__global__ void __launch_bounds__(THREAD_COUNT)
    matmul_pipelined_from_wmma_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                      bf16_t* C_device, int M, int N, int K) {
  extern __shared__ __align__(128) bf16_t stages[];
  const int warp = static_cast<int>(threadIdx.x) / THREADS_PER_WARP;
  const int warp_row_start = (warp / WARPS_PER_BLOCK_N) * WARP_TILE_M;
  const int warp_col_start = (warp % WARPS_PER_BLOCK_N) * WARP_TILE_N;
  const int64_t block_i = static_cast<int64_t>(blockIdx.y) * BLOCK_TILE_M;
  const int64_t block_j = static_cast<int64_t>(blockIdx.x) * BLOCK_TILE_N;
  const int64_t step_count = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
  C_warp_tile_t C_fragments;
  for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
    for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
      wmma::fill_fragment(C_fragments[m][n], 0.0F);
    }
  }
  for (int64_t step = 0; step < STAGE_COUNT - 1; ++step) {
    if (step < step_count) {
      issue_stage_loads(A_device, B_device, &stages[(step % STAGE_COUNT) * STAGE_BF16_COUNT], M, N,
                        K, block_i, block_j, step * BLOCK_TILE_K);
    }
    __pipeline_commit();
  }
  for (int64_t step = 0; step < step_count; ++step) {
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN STEPS READ TILES STILL IN FLIGHT
    __pipeline_wait_prior(STAGE_COUNT - 2);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
    const int64_t next_step = step + STAGE_COUNT - 1;
    if (next_step < step_count) {
      issue_stage_loads(A_device, B_device, &stages[(next_step % STAGE_COUNT) * STAGE_BF16_COUNT],
                        M, N, K, block_i, block_j, next_step * BLOCK_TILE_K);
    }
    __pipeline_commit();
    add_stage_products(&stages[(step % STAGE_COUNT) * STAGE_BF16_COUNT], warp_row_start,
                       warp_col_start, C_fragments);
  }
  __pipeline_wait_prior(0);
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN SCRATCH OVERWRITES TILES IN USE
  __syncthreads();
  float* scratch =
      reinterpret_cast<float*>(stages) + static_cast<ptrdiff_t>(warp * SCRATCH_FLOAT_COUNT);
  store_warp_tile(C_fragments, scratch, C_device, M, N, block_i + warp_row_start,
                  block_j + warp_col_start);
}

}  // namespace

/**
 * @brief `matmul_pipelined_from_wmma` launches a 256-thread block per 128 x 128 tile of C.
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
 * matmul_pipelined_from_wmma(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8, so that every row starts on a 16-byte boundary.
 * - Gotchas:
 *   - The first call raises the kernel's dynamic shared-memory limit, which defaults to 48 KB.
 */
void matmul_pipelined_from_wmma(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                                int M, int N, int K) {
  static const bool is_shared_limit_raised_by_first_call = [] {
    CHECK(cudaFuncSetAttribute(matmul_pipelined_from_wmma_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
    return true;
  }();
  (void)is_shared_limit_raised_by_first_call;
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_pipelined_from_wmma_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_device, B_device,
                                                                          C_device, M, N, K);
}
