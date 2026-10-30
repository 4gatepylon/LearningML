// Reference kernel: cuBLAS bf16 GEMM with fp32 accumulation.
#include <cublas_v2.h>

#include "common/matmul.h"

void matmul_cublas(const bf16* A_d, const bf16* B_d, bf16* C_d, int n) {
  static cublasHandle_t handle = [] {
    cublasHandle_t h = nullptr;
    CHECK(cublasCreate(&h));
    return h;
  }();
  const float alpha = 1.0F;
  const float beta = 0.0F;
  // cuBLAS is column-major, so it reads our row-major matrices as their transposes. Computing
  // C^T = B^T * A^T (B passed first) therefore writes C = A * B in row-major order.
  CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B_d, CUDA_R_16BF, n, A_d,
                     CUDA_R_16BF, n, &beta, C_d, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT));
}
