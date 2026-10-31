/**
 * @file
 * @brief `matmul.h` holds the kernel interface, the list of kernels and the shared helpers.
 *
 * @par Interface
 * - `bf16_t`: The element type of every matrix.
 * - `matmul_fn_t`: The signature that every kernel has.
 * - `Kernel`: A kernel's name and function.
 * - `KERNELS`: Every kernel that the tests and the benchmark run.
 * - `matmul_cublas`: The cuBLAS reference kernel, defined in `kernels/cublas.cu`.
 * - `check_status`: Aborts on a failed CUDA or cuBLAS call.
 * - `CHECK`: Calls `check_status` with the call's text, file and line.
 * - `copy_to_device`: Copies a host array into a new device buffer.
 *
 * @par Example
 * @code
 * for (const Kernel& kernel : KERNELS) kernel.matmul_fn(A_device, B_device, C_device, N);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - Every kernel computes C = A * B for N x N row-major bf16 matrices, accumulating in fp32.
 * - Gotchas:
 *   - A new kernel needs a declaration here and an entry in `KERNELS`, or nothing runs it.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) check_status(static_cast<int>(call), #call, __FILE__, __LINE__)

using bf16_t = __nv_bfloat16;
using matmul_fn_t = void (*)(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                             int N);

/**
 * @brief `check_status` prints the failed call and aborts if `status_code` is not a success.
 *
 * @param[in] status_code Status that a CUDA or cuBLAS call returned.
 * @param[in] call_text Source text of the call.
 * @param[in] file_name File that holds the call.
 * @param[in] line_number Line that holds the call.
 *
 * @par Example
 * @code
 * CHECK(cudaDeviceSynchronize());
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Code calls it through `CHECK`, which fills in the call's text, file and line.
 */
inline void check_status(int status_code, const char* call_text, const char* file_name,
                         int line_number) {
  constexpr int success_status_code = 0;
  if (status_code != success_status_code) {
    std::fprintf(stderr, "%s:%d: %s failed with status %d\n", file_name, line_number, call_text,
                 status_code);
    std::abort();
  }
}

void matmul_cublas(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device, int N);

/**
 * @brief `Kernel` pairs a kernel's name with its function.
 *
 * @par Example
 * @code
 * Kernel kernel = {"cublas", matmul_cublas};
 * @endcode
 */
struct Kernel {
  const char* name;
  matmul_fn_t matmul_fn;
};

inline constexpr Kernel KERNELS[] = {{"cublas", matmul_cublas}};

/**
 * @brief `copy_to_device` copies a host array into a new device buffer, rounding to bf16.
 *
 * @param[in] X_host Values to copy.
 * @return A new device buffer that holds `X_host` rounded to bf16.
 *
 * @par Example
 * @code
 * bf16_t* A_device = copy_to_device(A_host);
 * @endcode
 *
 * @par Must know
 * - Side effects:
 *   - The caller owns the returned buffer and must `cudaFree` it.
 */
inline bf16_t* copy_to_device(const std::vector<float>& X_host) {
  const std::vector<bf16_t> X_bf16_host(X_host.begin(), X_host.end());
  const size_t X_bytes = X_bf16_host.size() * sizeof(bf16_t);
  bf16_t* X_device = nullptr;
  CHECK(cudaMalloc(&X_device, X_bytes));
  CHECK(cudaMemcpy(X_device, X_bf16_host.data(), X_bytes, cudaMemcpyHostToDevice));
  return X_device;
}
