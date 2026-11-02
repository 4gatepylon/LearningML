/**
 * @file
 * @brief `blocktile2d_from_blocktile1d.cu` defines `matmul_blocktile2d_from_blocktile1d`:
 * `blocktile1d` with each thread computing an 8 x 8 block of C.
 *
 * @par Interface
 * - `matmul_blocktile2d_from_blocktile1d`: Launches `matmul_blocktile2d_from_blocktile1d_kernel`.
 *
 * @par Example
 * @code
 * matmul_blocktile2d_from_blocktile1d(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <cstdint>

#include "common/matmul.h"

namespace {

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 128;
constexpr int BLOCK_TILE_K = 8;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int THREAD_COUNT = (BLOCK_TILE_M / THREAD_TILE_M) * (BLOCK_TILE_N / THREAD_TILE_N);
constexpr int A_TILE_ROWS_PER_LOAD = THREAD_COUNT / BLOCK_TILE_K;
constexpr int B_TILE_ROWS_PER_LOAD = THREAD_COUNT / BLOCK_TILE_N;

static_assert(BLOCK_TILE_M % A_TILE_ROWS_PER_LOAD == 0, "the threads load A's tile in full rows");
static_assert(BLOCK_TILE_K % B_TILE_ROWS_PER_LOAD == 0, "the threads load B's tile in full rows");

using A_tile_t = bf16_t[BLOCK_TILE_M][BLOCK_TILE_K];
using B_tile_t = bf16_t[BLOCK_TILE_K][BLOCK_TILE_N];
using C_thread_tile_t = float[THREAD_TILE_M][THREAD_TILE_N];

/**
 * @brief `load_tiles` loads A's and B's tiles for one step along K, zero-filling past the edges.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] A_tile A's 128 x 8 tile, in shared memory.
 * @param[out] B_tile B's 8 x 128 tile, in shared memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 * @param[in] block_i Row of C where the block's tile starts.
 * @param[in] block_j Column of C where the block's tile starts.
 * @param[in] k_tile_start Column of A and row of B where the step starts.
 *
 * @par Example
 * @code
 * load_tiles(A_device, B_device, A_tile, B_tile, M, N, K, block_i, block_j, k_tile_start);
 * @endcode
 */
__device__ __forceinline__ void load_tiles(const bf16_t* A_device, const bf16_t* B_device,
                                           A_tile_t& A_tile, B_tile_t& B_tile, int M, int N, int K,
                                           int64_t block_i, int64_t block_j, int64_t k_tile_start) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const bf16_t zero_bf16 = __float2bfloat16(0.0F);
  const int A_tile_col = thread_index % BLOCK_TILE_K;
  const int64_t A_k = k_tile_start + A_tile_col;
  for (int row = thread_index / BLOCK_TILE_K; row < BLOCK_TILE_M; row += A_TILE_ROWS_PER_LOAD) {
    const int64_t A_i = block_i + row;
    A_tile[row][A_tile_col] = (A_i < M && A_k < K) ? A_device[(A_i * K) + A_k] : zero_bf16;
  }
  const int B_tile_col = thread_index % BLOCK_TILE_N;
  const int64_t B_j = block_j + B_tile_col;
  for (int row = thread_index / BLOCK_TILE_N; row < BLOCK_TILE_K; row += B_TILE_ROWS_PER_LOAD) {
    const int64_t B_k = k_tile_start + row;
    B_tile[row][B_tile_col] = (B_k < K && B_j < N) ? B_device[(B_k * N) + B_j] : zero_bf16;
  }
}

/**
 * @brief `add_tile_products` adds one step's products to a thread's 8 x 8 block of C.
 *
 * @param[in] A_tile A's 128 x 8 tile, in shared memory.
 * @param[in] B_tile B's 8 x 128 tile, in shared memory.
 * @param[in] C_tile_row_start Row of the block's tile where the thread's block starts.
 * @param[in] C_tile_col_start Column of the block's tile where the thread's block starts.
 * @param[in,out] C_fp32 The thread's 8 x 8 accumulators.
 *
 * @par Example
 * @code
 * add_tile_products(A_tile, B_tile, C_tile_row_start, C_tile_col_start, C_fp32);
 * @endcode
 *
 * @par Must know
 * - Performance:
 *   - Per k, it reads 8 + 8 tile elements into registers and does their 64 multiply-adds.
 */
__device__ __forceinline__ void add_tile_products(const A_tile_t& A_tile, const B_tile_t& B_tile,
                                                  int C_tile_row_start, int C_tile_col_start,
                                                  C_thread_tile_t& C_fp32) {
  float A_fp32[THREAD_TILE_M];
  float B_fp32[THREAD_TILE_N];
  for (int k = 0; k < BLOCK_TILE_K; ++k) {
    for (int r = 0; r < THREAD_TILE_M; ++r) {
      A_fp32[r] = __bfloat162float(A_tile[C_tile_row_start + r][k]);
    }
    for (int c = 0; c < THREAD_TILE_N; ++c) {
      B_fp32[c] = __bfloat162float(B_tile[k][C_tile_col_start + c]);
    }
    for (int r = 0; r < THREAD_TILE_M; ++r) {
      for (int c = 0; c < THREAD_TILE_N; ++c) {
        C_fp32[r][c] += A_fp32[r] * B_fp32[c];
      }
    }
  }
}

/**
 * @brief `store_thread_tile` writes a thread's 8 x 8 block of C, skipping elements outside C.
 *
 * @param[in] C_fp32 The thread's 8 x 8 accumulators.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of C.
 * @param[in] N Columns of C.
 * @param[in] i_start Row of C where the thread's block starts.
 * @param[in] j_start Column of C where the thread's block starts.
 *
 * @par Example
 * @code
 * store_thread_tile(C_fp32, C_device, M, N, i_start, j_start);
 * @endcode
 */
__device__ __forceinline__ void store_thread_tile(const C_thread_tile_t& C_fp32, bf16_t* C_device,
                                                  int M, int N, int64_t i_start, int64_t j_start) {
  for (int r = 0; r < THREAD_TILE_M; ++r) {
    const int64_t i = i_start + r;
    for (int c = 0; c < THREAD_TILE_N; ++c) {
      const int64_t j = j_start + c;
      if (i < M && j < N) {
        C_device[(i * N) + j] = __float2bfloat16(C_fp32[r][c]);
      }
    }
  }
}

/**
 * @brief `matmul_blocktile2d_from_blocktile1d_kernel` computes an 8 x 8 block of C per thread.
 *
 * Each block computes a 128 x 128 tile of C, and each of its 256 threads an 8 x 8 block of it.
 * For each step of 8 along K, the threads load A's 128 x 8 tile and B's 8 x 128 tile into shared
 * memory, 4 elements of each per thread, and the block syncs. Then, for each k in the step, every
 * thread copies 8 elements of A's tile column and 8 of B's tile row into registers and adds their
 * outer product to its 64 accumulators.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B (the length of each dot product).
 *
 * @par Example
 * @code
 * matmul_blocktile2d_from_blocktile1d_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device,
 *                                                                    C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Every thread loads and syncs, even outside C, so no thread may return before the last sync.
 *   - Loads past the edges of A and B fill the tiles with zeros, so any M, N and K works.
 * - Performance:
 *   - A thread does 64 multiply-adds per 16 shared-memory reads, where `blocktile1d` did 8 per 9.
 */
__global__ void __launch_bounds__(THREAD_COUNT)
    matmul_blocktile2d_from_blocktile1d_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                               bf16_t* C_device, int M, int N, int K) {
  __shared__ A_tile_t A_tile;
  __shared__ B_tile_t B_tile;
  const int thread_index = static_cast<int>(threadIdx.x);
  const int C_tile_row_start = (thread_index / (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_M;
  const int C_tile_col_start = (thread_index % (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_N;
  const int64_t block_i = static_cast<int64_t>(blockIdx.y) * BLOCK_TILE_M;
  const int64_t block_j = static_cast<int64_t>(blockIdx.x) * BLOCK_TILE_N;
  float C_fp32[THREAD_TILE_M][THREAD_TILE_N] = {};
  for (int64_t k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
    load_tiles(A_device, B_device, A_tile, B_tile, M, N, K, block_i, block_j, k_tile_start);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
    add_tile_products(A_tile, B_tile, C_tile_row_start, C_tile_col_start, C_fp32);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
  }
  store_thread_tile(C_fp32, C_device, M, N, block_i + C_tile_row_start, block_j + C_tile_col_start);
}

}  // namespace

/**
 * @brief `matmul_blocktile2d_from_blocktile1d` launches a 256-thread block per 128 x 128 tile.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B (the length of each dot product).
 *
 * @par Example
 * @code
 * matmul_blocktile2d_from_blocktile1d(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The grid's y dimension caps M at 65535 * 128.
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
void matmul_blocktile2d_from_blocktile1d(const bf16_t* A_device, const bf16_t* B_device,
                                         bf16_t* C_device, int M, int N, int K) {
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_blocktile2d_from_blocktile1d_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device, C_device,
                                                                     M, N, K);
}
