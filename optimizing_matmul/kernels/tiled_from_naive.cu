/**
 * @file
 * @brief `tiled_from_naive.cu` defines `matmul_tiled_from_naive`: `naive` with shared-memory tiles.
 *
 * @par Interface
 * - `matmul_tiled_from_naive<TILE_SIZE>`: A `matmul_fn_t` for `TILE_SIZE` 4, 8, 16 or 32.
 *
 * @par Example
 * @code
 * matmul_tiled_from_naive<32>(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <cstdint>

#include "common/matmul.h"

namespace {

/**
 * @brief `matmul_tiled_from_naive_kernel` computes one element of C per thread, from tiles.
 *
 * Each block computes a `TILE_SIZE` x `TILE_SIZE` tile of C. For each step along K, every thread
 * loads one element of A's tile and one of B's tile, the block syncs, and every thread adds the
 * tiles' dot product for its element to a register. The thread writes C once, at the end.
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
 * matmul_tiled_from_naive_kernel<32><<<grid, block>>>(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - A block has `TILE_SIZE`^2 threads, so `TILE_SIZE` is at most 32 (1024 threads).
 * - Gotchas:
 *   - Every thread loads and syncs, even outside C, so no thread may return before the last sync.
 *   - Loads past the edges of A and B fill the tiles with zeros, so any M, N and K works.
 * - Performance:
 *   - Each element of A is read from global memory N / `TILE_SIZE` times, and of B M / `TILE_SIZE`.
 */
template <int TILE_SIZE>
__global__ void matmul_tiled_from_naive_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                               bf16_t* C_device, int M, int N, int K) {
  __shared__ bf16_t A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ bf16_t B_tile[TILE_SIZE][TILE_SIZE];
  const int tile_row = static_cast<int>(threadIdx.y);
  const int tile_col = static_cast<int>(threadIdx.x);
  const int64_t i = (static_cast<int64_t>(blockIdx.y) * TILE_SIZE) + tile_row;
  const int64_t j = (static_cast<int64_t>(blockIdx.x) * TILE_SIZE) + tile_col;
  const bf16_t zero_bf16 = __float2bfloat16(0.0F);
  float C_ij_fp32 = 0.0F;
  for (int64_t k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_SIZE) {
    const int64_t A_k = k_tile_start + tile_col;
    const int64_t B_k = k_tile_start + tile_row;
    A_tile[tile_row][tile_col] = (i < M && A_k < K) ? A_device[(i * K) + A_k] : zero_bf16;
    B_tile[tile_row][tile_col] = (B_k < K && j < N) ? B_device[(B_k * N) + j] : zero_bf16;
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
    for (int k = 0; k < TILE_SIZE; ++k) {
      C_ij_fp32 += __bfloat162float(A_tile[tile_row][k]) * __bfloat162float(B_tile[k][tile_col]);
    }
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
  }
  if (i < M && j < N) {
    C_device[(i * N) + j] = __float2bfloat16(C_ij_fp32);
  }
}

}  // namespace

/**
 * @brief `matmul_tiled_from_naive` launches the kernel with one thread per element of C.
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
 * matmul_tiled_from_naive<32>(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The grid's y dimension caps M at 65535 * `TILE_SIZE`.
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
template <int TILE_SIZE>
void matmul_tiled_from_naive(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                             int M, int N, int K) {
  const dim3 block(TILE_SIZE, TILE_SIZE);
  const dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_tiled_from_naive_kernel<TILE_SIZE><<<grid, block>>>(A_device, B_device, C_device, M, N, K);
}

template void matmul_tiled_from_naive<4>(const bf16_t*, const bf16_t*, bf16_t*, int, int, int);
template void matmul_tiled_from_naive<8>(const bf16_t*, const bf16_t*, bf16_t*, int, int, int);
template void matmul_tiled_from_naive<16>(const bf16_t*, const bf16_t*, bf16_t*, int, int, int);
template void matmul_tiled_from_naive<32>(const bf16_t*, const bf16_t*, bf16_t*, int, int, int);
