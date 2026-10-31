/**
 * @file
 * @brief `cublas.cu` defines `matmul_cublas`, the cuBLAS reference kernel.
 *
 * @par Interface
 * - `matmul_cublas`: C = A * B with cuBLAS.
 *
 * @par Example
 * @code
 * matmul_cublas(A_device, B_device, C_device, N);
 * @endcode
 */
#include <cublas_v2.h>

#include "common/matmul.h"

/**
 * @brief `matmul_cublas` computes C = A * B for N x N row-major bf16 matrices with cuBLAS.
 *
 * @param[in] A_device Left factor, N x N, in device memory.
 * @param[in] B_device Right factor, N x N, in device memory.
 * @param[out] C_device Product, N x N, in device memory.
 * @param[in] N Matrix size.
 *
 * @par Example
 * @code
 * matmul_cublas(A_device, B_device, C_device, N);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - cuBLAS is column-major, so passing B before A makes it compute row-major A * B.
 * - Performance:
 *   - The first call creates the cuBLAS handle, so it is much slower than later calls.
 */
void matmul_cublas(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device, int N) {
  static cublasHandle_t cublas_handle_created_by_first_call = [] {
    cublasHandle_t new_cublas_handle = nullptr;
    CHECK(cublasCreate(&new_cublas_handle));
    return new_cublas_handle;
  }();
  const float alpha = 1.0F;
  const float beta = 0.0F;
  CHECK(cublasGemmEx(cublas_handle_created_by_first_call, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                     B_device, CUDA_R_16BF, N, A_device, CUDA_R_16BF, N, &beta, C_device,
                     CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
