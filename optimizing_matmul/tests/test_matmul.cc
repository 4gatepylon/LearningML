/**
 * @file
 * @brief `test_matmul.cc` checks every kernel in `KERNELS` against known products.
 *
 * @par Interface
 * - `matmul.known_2x2`: Checks a hand-computed 2x2 product exactly.
 *
 * @par Example
 * @code
 * make test
 * @endcode
 */
#include <gtest/gtest.h>

#include <vector>

#include "common/matmul.h"

namespace {

constexpr float UNWRITTEN_SENTINEL = -1.0F;

/**
 * @brief `run_kernel` runs one kernel on host matrices and returns C = A * B as floats.
 *
 * @param[in] matmul_fn Kernel to run.
 * @param[in] A_host Left factor, M x K, row-major.
 * @param[in] B_host Right factor, K x N, row-major.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 * @return C = A * B, M x N, row-major.
 *
 * @par Example
 * @code
 * std::vector<float> C_host = run_kernel(matmul_cublas, A_host, B_host, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - C starts as `UNWRITTEN_SENTINEL`, so entries that a kernel never writes come back wrong.
 */
std::vector<float> run_kernel(matmul_fn_t matmul_fn, const std::vector<float>& A_host,
                              const std::vector<float>& B_host, int M, int N, int K) {
  const size_t C_element_count = static_cast<size_t>(M) * N;
  bf16_t* A_device = copy_to_device(A_host);
  bf16_t* B_device = copy_to_device(B_host);
  bf16_t* C_device = copy_to_device(std::vector<float>(C_element_count, UNWRITTEN_SENTINEL));
  matmul_fn(A_device, B_device, C_device, M, N, K);
  CHECK(cudaGetLastError());
  std::vector<bf16_t> C_host(C_element_count);
  CHECK(
      cudaMemcpy(C_host.data(), C_device, C_host.size() * sizeof(bf16_t), cudaMemcpyDeviceToHost));
  for (bf16_t* X_device : {A_device, B_device, C_device}) {
    CHECK(cudaFree(X_device));
  }
  return {C_host.begin(), C_host.end()};
}

}  // namespace

/**
 * @brief `matmul.known_2x2` checks every kernel on a hand-computed 2x2 product, exactly.
 *
 * @par Must know
 * - Gotchas:
 *   - B * A, A^T * B, A * B^T and A^T * B^T all differ from A * B, so swaps and transposes fail.
 *   - Every value is a small integer, so bf16 holds it exactly.
 */
TEST(matmul, known_2x2) {
  constexpr int N = 2;
  const std::vector<float> A_host = {1, 2, 3, 4};
  const std::vector<float> B_host = {5, 6, 7, 8};
  const std::vector<float> C_expected_host = {19, 22, 43, 50};
  for (const Kernel& kernel : KERNELS) {
    SCOPED_TRACE(kernel.name);
    EXPECT_EQ(run_kernel(kernel.matmul_fn, A_host, B_host, N, N, N), C_expected_host);
  }
}
