// Shared by the kernels, tests and benchmark: every kernel computes C = A * B for n x n row-major
// bf16 matrices in device memory, accumulating in fp32.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using bf16 = __nv_bfloat16;
using MatmulFn = void (*)(const bf16* A_d, const bf16* B_d, bf16* C_d, int n);

// Aborts with the call's text if a CUDA or cuBLAS call fails (both return 0 on success).
#define CHECK(call) check_status(static_cast<int>(call), #call, __FILE__, __LINE__)
inline void check_status(int err, const char* call, const char* file, int line) {
  if (err != 0) {
    std::fprintf(stderr, "%s:%d: %s failed with error %d\n", file, line, call, err);
    std::abort();
  }
}

void matmul_cublas(const bf16* A_d, const bf16* B_d, bf16* C_d, int n);

// Every kernel the tests and benchmark run. A new kernel is declared above and added here.
struct Kernel {
  const char* name;
  MatmulFn fn;
};
inline constexpr Kernel kKernels[] = {{"cublas", matmul_cublas}};

// Copies a host matrix into a new device buffer, rounding each value to bf16.
inline bf16* to_device(const std::vector<float>& x_h) {
  const std::vector<bf16> x_bf16_h(x_h.begin(), x_h.end());
  bf16* x_d = nullptr;
  CHECK(cudaMalloc(&x_d, x_bf16_h.size() * sizeof(bf16)));
  CHECK(cudaMemcpy(x_d, x_bf16_h.data(), x_bf16_h.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  return x_d;
}
