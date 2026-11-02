/**
 * @file
 * @brief `matmul.h` holds the kernel interface, the list of kernels and the shared helpers.
 *
 * @par Interface
 * - `bf16_t`: The element type of every matrix.
 * - `matmul_fn_t`: The signature that every kernel has.
 * - `Kernel`: A kernel's name, function and size multiple.
 * - `KERNELS`: Every kernel that the tests and the benchmark run.
 * - `matmul_cublas`: The cuBLAS reference kernel, defined in `kernels/cublas.cu`.
 * - `matmul_naive`: The one-thread-per-element kernel, defined in `kernels/naive.cu`.
 * - `matmul_tiled_from_naive`: Shared-memory tiles, defined in `kernels/tiled_from_naive.cu`.
 * - `matmul_blocktile1d_from_tiled32`: 8 outputs per thread, in its own file under `kernels/`.
 * - `matmul_blocktile2d_from_blocktile1d`: 8 x 8 outputs per thread, in its own file too.
 * - `matmul_vectorized_from_blocktile2d`: 16-byte loads and stores, in its own file too.
 * - `matmul_wmma_from_vectorized`: Tensor cores through WMMA, in its own file too.
 * - `check_status`: Aborts on a failed CUDA or cuBLAS call.
 * - `CHECK`: Calls `check_status` with the call's text, file and line.
 * - `copy_to_device`: Copies a host array into a new device buffer.
 *
 * @par Example
 * @code
 * for (const Kernel& kernel : KERNELS) kernel.matmul_fn(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - Every kernel computes C = A * B (A: M x K, B: K x N, row-major bf16), accumulating in fp32.
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
                             int M, int N, int K);

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

void matmul_cublas(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device, int M, int N,
                   int K);
void matmul_naive(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device, int M, int N,
                  int K);
template <int TILE_SIZE>
void matmul_tiled_from_naive(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                             int M, int N, int K);
void matmul_blocktile1d_from_tiled32(const bf16_t* A_device, const bf16_t* B_device,
                                     bf16_t* C_device, int M, int N, int K);
void matmul_blocktile2d_from_blocktile1d(const bf16_t* A_device, const bf16_t* B_device,
                                         bf16_t* C_device, int M, int N, int K);
void matmul_vectorized_from_blocktile2d(const bf16_t* A_device, const bf16_t* B_device,
                                        bf16_t* C_device, int M, int N, int K);
void matmul_wmma_from_vectorized(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                                 int M, int N, int K);

/**
 * @brief `Kernel` holds a kernel's name, its function and the sizes it supports.
 *
 * @par Example
 * @code
 * Kernel kernel = {"cublas", matmul_cublas, 1};
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The kernel supports every M, N and K that is a multiple of `size_multiple`, a power of two.
 * - Gotchas:
 *   - The tests round every size up to `size_multiple`, so a kernel with 1 runs every size.
 */
struct Kernel {
  const char* name;
  matmul_fn_t matmul_fn;
  int size_multiple;
};

inline constexpr Kernel KERNELS[] = {
    {"cublas", matmul_cublas, 1},
    {"naive", matmul_naive, 1},
    {"tiled4_from_naive", matmul_tiled_from_naive<4>, 1},
    {"tiled8_from_naive", matmul_tiled_from_naive<8>, 1},
    {"tiled16_from_naive", matmul_tiled_from_naive<16>, 1},
    {"tiled32_from_naive", matmul_tiled_from_naive<32>, 1},
    {"blocktile1d_from_tiled32", matmul_blocktile1d_from_tiled32, 1},
    {"blocktile2d_from_blocktile1d", matmul_blocktile2d_from_blocktile1d, 1},
    {"vectorized_from_blocktile2d", matmul_vectorized_from_blocktile2d, 8},
    {"wmma_from_vectorized", matmul_wmma_from_vectorized, 8},
};

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
