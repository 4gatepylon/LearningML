/**
 * @file
 * @brief `naive.cu` defines `matmul_naive`, a kernel that gives each thread one element of C.
 *
 * @par Interface
 * - `matmul_naive`: Launches `matmul_naive_kernel`; it has the `matmul_fn_t` signature.
 *
 * @par Example
 * @code
 * matmul_naive(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <cstdint>

#include "common/matmul.h"

namespace {

constexpr int BLOCK_COL_COUNT = 32;
constexpr int BLOCK_ROW_COUNT = 8;

/**
 * @brief `matmul_naive_kernel` computes C[i][j] = sum over k of A[i][k] * B[k][j], one per thread.
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
 * matmul_naive_kernel<<<grid, block>>>(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The grid's y dimension caps M at 65535 * `BLOCK_ROW_COUNT` = 524280.
 * - Performance:
 *   - `threadIdx.x` walks the columns, so each warp reads B and writes C coalesced.
 */
__global__ void matmul_naive_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                    bf16_t* C_device, int M, int N, int K) {
  const int64_t i = (static_cast<int64_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
  const int64_t j = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= M || j >= N) {
    return;
  }
  float C_ij_fp32 = 0.0F;
  for (int64_t k = 0; k < K; ++k) {
    C_ij_fp32 += __bfloat162float(A_device[(i * K) + k]) * __bfloat162float(B_device[(k * N) + j]);
  }
  C_device[(i * N) + j] = __float2bfloat16(C_ij_fp32);
}

}  // namespace

/**
 * @brief `matmul_naive` launches `matmul_naive_kernel` with one thread per element of C.
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
 * matmul_naive(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
void matmul_naive(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device, int M, int N,
                  int K) {
  const dim3 block(BLOCK_COL_COUNT, BLOCK_ROW_COUNT);
  const dim3 grid((N + BLOCK_COL_COUNT - 1) / BLOCK_COL_COUNT,
                  (M + BLOCK_ROW_COUNT - 1) / BLOCK_ROW_COUNT);
  matmul_naive_kernel<<<grid, block>>>(A_device, B_device, C_device, M, N, K);
}
