/**
 * @file
 * @brief `naive.cu` defines `matmul_naive`, a kernel that gives each thread one element of C.
 *
 * @par Interface
 * - `matmul_naive_kernel`: Computes one element of C per thread.
 * - `matmul_naive`: Launches `matmul_naive_kernel`; it has the `matmul_fn_t` signature.
 *
 * @par Example
 * @code
 * matmul_naive(A_device, B_device, C_device, N);
 * @endcode
 */
#include <cstdint>

#include "common/matmul.h"

constexpr int BLOCK_COLS = 32;
constexpr int BLOCK_ROWS = 8;

/**
 * @brief `matmul_naive_kernel` computes C[i][j] = sum over k of A[i][k] * B[k][j], one per thread.
 *
 * @param[in] A_device Left factor, N x N, row-major, in device memory.
 * @param[in] B_device Right factor, N x N, row-major, in device memory.
 * @param[out] C_device Product, N x N, row-major, in device memory.
 * @param[in] N Matrix size.
 *
 * @par Example
 * @code
 * matmul_naive_kernel<<<grid, block>>>(A_device, B_device, C_device, N);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The grid's y dimension caps N at 65535 * `BLOCK_ROWS` = 524280.
 * - Performance:
 *   - `threadIdx.x` walks the columns, so each warp reads B and writes C coalesced.
 */
__global__ void matmul_naive_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                    bf16_t* C_device, int N) {
  const int64_t i = (static_cast<int64_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
  const int64_t j = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= N || j >= N) {
    return;
  }
  float C_ij_fp32 = 0.0F;
  for (int64_t k = 0; k < N; ++k) {
    C_ij_fp32 += __bfloat162float(A_device[(i * N) + k]) * __bfloat162float(B_device[(k * N) + j]);
  }
  C_device[(i * N) + j] = __float2bfloat16(C_ij_fp32);
}

/**
 * @brief `matmul_naive` launches `matmul_naive_kernel` with one thread per element of C.
 *
 * @param[in] A_device Left factor, N x N, row-major, in device memory.
 * @param[in] B_device Right factor, N x N, row-major, in device memory.
 * @param[out] C_device Product, N x N, row-major, in device memory.
 * @param[in] N Matrix size.
 *
 * @par Example
 * @code
 * matmul_naive(A_device, B_device, C_device, N);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
void matmul_naive(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device, int N) {
  const dim3 block(BLOCK_COLS, BLOCK_ROWS);
  const dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS, (N + BLOCK_ROWS - 1) / BLOCK_ROWS);
  matmul_naive_kernel<<<grid, block>>>(A_device, B_device, C_device, N);
}
