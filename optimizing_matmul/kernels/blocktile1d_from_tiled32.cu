/**
 * @file
 * @brief `blocktile1d_from_tiled32.cu` defines `matmul_blocktile1d_from_tiled32`: `tiled32` with
 * each thread computing a column of 8 elements of C.
 *
 * @par Interface
 * - `matmul_blocktile1d_from_tiled32`: Launches `matmul_blocktile1d_from_tiled32_kernel`.
 *
 * @par Example
 * @code
 * matmul_blocktile1d_from_tiled32(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <cstdint>

#include "common/matmul.h"

namespace {

constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = 8;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_COUNT = BLOCK_TILE_M * BLOCK_TILE_N / THREAD_TILE_M;

static_assert(BLOCK_TILE_M * BLOCK_TILE_K == THREAD_COUNT, "each thread loads one element of A");
static_assert(BLOCK_TILE_K * BLOCK_TILE_N == THREAD_COUNT, "each thread loads one element of B");

/**
 * @brief `matmul_blocktile1d_from_tiled32_kernel` computes 8 elements of C per thread from tiles.
 *
 * Each block computes a 64 x 64 tile of C, and each of its 512 threads a column of 8 elements of
 * it. For each step of 8 along K, every thread loads one element of A's 64 x 8 tile and one of
 * B's 8 x 64 tile into shared memory, and the block syncs. Then every thread reads each element of
 * B's tile column once and multiplies it by 8 elements of A's tile, accumulating in 8 registers.
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
 * matmul_blocktile1d_from_tiled32_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device, C_device,
 *                                                                M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Every thread loads and syncs, even outside C, so no thread may return before the last sync.
 *   - Loads past the edges of A and B fill the tiles with zeros, so any M, N and K works.
 * - Performance:
 *   - A thread does 8 multiply-adds per element of B's tile it reads, where `tiled32` did 1.
 */
__global__ void __launch_bounds__(THREAD_COUNT)
    matmul_blocktile1d_from_tiled32_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                           bf16_t* C_device, int M, int N, int K) {
  __shared__ bf16_t A_tile[BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ bf16_t B_tile[BLOCK_TILE_K][BLOCK_TILE_N];
  const int thread_index = static_cast<int>(threadIdx.x);
  const int A_tile_row = thread_index / BLOCK_TILE_K;
  const int A_tile_col = thread_index % BLOCK_TILE_K;
  const int B_tile_row = thread_index / BLOCK_TILE_N;
  const int B_tile_col = thread_index % BLOCK_TILE_N;
  const int C_tile_row_start = (thread_index / BLOCK_TILE_N) * THREAD_TILE_M;
  const int C_tile_col = thread_index % BLOCK_TILE_N;
  const int64_t block_i = static_cast<int64_t>(blockIdx.y) * BLOCK_TILE_M;
  const int64_t block_j = static_cast<int64_t>(blockIdx.x) * BLOCK_TILE_N;
  const int64_t A_i = block_i + A_tile_row;
  const int64_t B_j = block_j + B_tile_col;
  const bf16_t zero_bf16 = __float2bfloat16(0.0F);
  float C_fp32[THREAD_TILE_M] = {};
  for (int64_t k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
    const int64_t A_k = k_tile_start + A_tile_col;
    const int64_t B_k = k_tile_start + B_tile_row;
    A_tile[A_tile_row][A_tile_col] = (A_i < M && A_k < K) ? A_device[(A_i * K) + A_k] : zero_bf16;
    B_tile[B_tile_row][B_tile_col] = (B_k < K && B_j < N) ? B_device[(B_k * N) + B_j] : zero_bf16;
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
    for (int k = 0; k < BLOCK_TILE_K; ++k) {
      const float B_kj = __bfloat162float(B_tile[k][C_tile_col]);
      for (int r = 0; r < THREAD_TILE_M; ++r) {
        C_fp32[r] += __bfloat162float(A_tile[C_tile_row_start + r][k]) * B_kj;
      }
    }
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
  }
  const int64_t j = block_j + C_tile_col;
  for (int r = 0; r < THREAD_TILE_M; ++r) {
    const int64_t i = block_i + C_tile_row_start + r;
    if (i < M && j < N) {
      C_device[(i * N) + j] = __float2bfloat16(C_fp32[r]);
    }
  }
}

}  // namespace

/**
 * @brief `matmul_blocktile1d_from_tiled32` launches one 512-thread block per 64 x 64 tile of C.
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
 * matmul_blocktile1d_from_tiled32(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The grid's y dimension caps M at 65535 * 64.
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
void matmul_blocktile1d_from_tiled32(const bf16_t* A_device, const bf16_t* B_device,
                                     bf16_t* C_device, int M, int N, int K) {
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_blocktile1d_from_tiled32_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device, C_device, M, N,
                                                                 K);
}
