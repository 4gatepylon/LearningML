/**
 * @file
 * @brief `test_matmul.cc` checks every kernel in `KERNELS` against known products and a reference.
 *
 * @par Interface
 * - `matmul.known_2x2`: Checks a hand-computed 2x2 product exactly.
 * - `matmul.reference_matches_known_2x2`: Checks the CPU reference on the same 2x2 product.
 * - `matmul.matches_reference_random_normal`: Checks every kernel, cuBLAS too, against the
 * reference.
 * - `matmul.matches_cublas_large_random_normal`: Checks every kernel against cuBLAS at a large N.
 *
 * @par Example
 * @code
 * make test
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The reference is too slow for large N, so large matrices are checked against cuBLAS.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common/matmul.h"

namespace {

constexpr float UNWRITTEN_SENTINEL = -1.0F;
constexpr double BF16_UNIT_ROUNDOFF = 0x1p-8;
constexpr double FP32_UNIT_ROUNDOFF = 0x1p-24;
constexpr int REFERENCE_N = 96;
constexpr int LARGE_N = 4096;
constexpr unsigned A_SEED = 1;
constexpr unsigned B_SEED = 2;

/**
 * @brief `run_kernel` runs one kernel on host matrices and returns C = A * B as floats.
 *
 * @param[in] matmul_fn Kernel to run.
 * @param[in] A_host Left factor, N x N, row-major.
 * @param[in] B_host Right factor, N x N, row-major.
 * @param[in] N Matrix size.
 * @return C = A * B, N x N, row-major.
 *
 * @par Example
 * @code
 * std::vector<float> C_host = run_kernel(matmul_cublas, A_host, B_host, N);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - C starts as `UNWRITTEN_SENTINEL`, so entries that a kernel never writes come back wrong.
 */
std::vector<float> run_kernel(matmul_fn_t matmul_fn, const std::vector<float>& A_host,
                              const std::vector<float>& B_host, int N) {
  bf16_t* A_device = copy_to_device(A_host);
  bf16_t* B_device = copy_to_device(B_host);
  bf16_t* C_device = copy_to_device(std::vector<float>(A_host.size(), UNWRITTEN_SENTINEL));
  matmul_fn(A_device, B_device, C_device, N);
  CHECK(cudaGetLastError());
  std::vector<bf16_t> C_host(A_host.size());
  CHECK(
      cudaMemcpy(C_host.data(), C_device, C_host.size() * sizeof(bf16_t), cudaMemcpyDeviceToHost));
  for (bf16_t* X_device : {A_device, B_device, C_device}) {
    CHECK(cudaFree(X_device));
  }
  return {C_host.begin(), C_host.end()};
}

/**
 * @brief `make_random_normal_matrix` returns an N x N matrix of N(0, 1) samples rounded to bf16.
 *
 * @param[in] N Matrix size.
 * @param[in] seed Seed for the random number generator.
 * @return The matrix, row-major, with every value exactly representable in bf16.
 *
 * @par Example
 * @code
 * std::vector<float> A_host = make_random_normal_matrix(N, A_SEED);
 * @endcode
 */
std::vector<float> make_random_normal_matrix(int N, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> standard_normal;
  std::vector<float> X_host(static_cast<size_t>(N) * N);
  for (float& x : X_host) {
    x = static_cast<float>(static_cast<bf16_t>(standard_normal(rng)));
  }
  return X_host;
}

/**
 * @brief `Reference` holds the exact product C = A * B and, per element, the sum of |A||B|.
 *
 * @par Example
 * @code
 * Reference reference = compute_reference(A_host, B_host, N);
 * @endcode
 */
struct Reference {
  std::vector<double> C;
  std::vector<double> abs_product_sums;
};

/**
 * @brief `compute_reference` multiplies A and B on the CPU in double precision.
 *
 * @param[in] A_host Left factor, N x N, row-major, with values exactly representable in bf16.
 * @param[in] B_host Right factor, N x N, row-major, with values exactly representable in bf16.
 * @param[in] N Matrix size.
 * @return C = A * B and, for each element, the sum over k of |A[i][k] * B[k][j]|.
 *
 * @par Example
 * @code
 * Reference reference = compute_reference(A_host, B_host, N);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - It takes O(N^3) time on one CPU thread, so it suits N up to a few hundred.
 */
Reference compute_reference(const std::vector<float>& A_host, const std::vector<float>& B_host,
                            int N) {
  Reference reference{std::vector<double>(A_host.size()), std::vector<double>(A_host.size())};
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        const double product = static_cast<double>(A_host[(i * N) + k]) * B_host[(k * N) + j];
        reference.C[(i * N) + j] += product;
        reference.abs_product_sums[(i * N) + j] += std::abs(product);
      }
    }
  }
  return reference;
}

/**
 * @brief `compute_tolerance` bounds how far a correct kernel's output can be from the exact C.
 *
 * @param[in] C_exact The exact value of one element of C.
 * @param[in] abs_product_sum The sum over k of |A[i][k] * B[k][j]| for that element.
 * @param[in] K Length of the dot product.
 * @return One rounding of the fp32 sum to bf16 (the inputs are already bf16) plus fp32
 *         accumulation over K terms.
 *
 * @par Example
 * @code
 * double tolerance = compute_tolerance(reference.C[e], reference.abs_product_sums[e], N);
 * @endcode
 */
double compute_tolerance(double C_exact, double abs_product_sum, int K) {
  return (BF16_UNIT_ROUNDOFF * std::abs(C_exact)) +
         ((1.0 + BF16_UNIT_ROUNDOFF) * K * FP32_UNIT_ROUNDOFF * abs_product_sum);
}

/**
 * @brief `expect_close_to_reference` checks every kernel's C = A * B against the CPU reference.
 *
 * @param[in] A_host Left factor, N x N, row-major, with values exactly representable in bf16.
 * @param[in] B_host Right factor, N x N, row-major, with values exactly representable in bf16.
 * @param[in] N Matrix size.
 *
 * @par Example
 * @code
 * expect_close_to_reference(A_host, B_host, N);
 * @endcode
 */
void expect_close_to_reference(const std::vector<float>& A_host, const std::vector<float>& B_host,
                               int N) {
  const Reference reference = compute_reference(A_host, B_host, N);
  for (const Kernel& kernel : KERNELS) {
    SCOPED_TRACE(kernel.name);
    const std::vector<float> C_host = run_kernel(kernel.matmul_fn, A_host, B_host, N);
    for (size_t e = 0; e < C_host.size(); ++e) {
      const double tolerance = compute_tolerance(reference.C[e], reference.abs_product_sums[e], N);
      ASSERT_LE(std::abs(C_host[e] - reference.C[e]), tolerance)
          << "i = " << e / N << ", j = " << e % N << ", C = " << C_host[e]
          << ", reference = " << reference.C[e];
    }
  }
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
    EXPECT_EQ(run_kernel(kernel.matmul_fn, A_host, B_host, N), C_expected_host);
  }
}

/**
 * @brief `matmul.reference_matches_known_2x2` checks the CPU reference on the hand-computed
 * product.
 */
TEST(matmul, reference_matches_known_2x2) {
  constexpr int N = 2;
  const Reference reference = compute_reference({1, 2, 3, 4}, {5, 6, 7, 8}, N);
  EXPECT_EQ(reference.C, (std::vector<double>{19, 22, 43, 50}));
}

/**
 * @brief `matmul.matches_reference_random_normal` checks every kernel against the CPU reference.
 *
 * @par Must know
 * - Gotchas:
 *   - cuBLAS is in `KERNELS`, so this also checks cuBLAS against the reference.
 */
TEST(matmul, matches_reference_random_normal) {
  expect_close_to_reference(make_random_normal_matrix(REFERENCE_N, A_SEED),
                            make_random_normal_matrix(REFERENCE_N, B_SEED), REFERENCE_N);
}

/**
 * @brief `matmul.matches_cublas_large_random_normal` checks every kernel against cuBLAS at
 * `LARGE_N`.
 *
 * @par Must know
 * - Gotchas:
 *   - The tolerance bounds the sum of |A||B| by the row norm of A times the column norm of B.
 */
TEST(matmul, matches_cublas_large_random_normal) {
  const std::vector<float> A_host = make_random_normal_matrix(LARGE_N, A_SEED);
  const std::vector<float> B_host = make_random_normal_matrix(LARGE_N, B_SEED);
  std::vector<double> A_row_squared_norms(LARGE_N);
  std::vector<double> B_col_squared_norms(LARGE_N);
  for (int i = 0; i < LARGE_N; ++i) {
    for (int k = 0; k < LARGE_N; ++k) {
      A_row_squared_norms[i] += A_host[(i * LARGE_N) + k] * A_host[(i * LARGE_N) + k];
      B_col_squared_norms[i] += B_host[(k * LARGE_N) + i] * B_host[(k * LARGE_N) + i];
    }
  }
  const std::vector<float> C_cublas_host = run_kernel(matmul_cublas, A_host, B_host, LARGE_N);
  for (const Kernel& kernel : KERNELS) {
    SCOPED_TRACE(kernel.name);
    const std::vector<float> C_host = run_kernel(kernel.matmul_fn, A_host, B_host, LARGE_N);
    for (size_t e = 0; e < C_host.size(); ++e) {
      const double abs_product_sum_bound =
          std::sqrt(A_row_squared_norms[e / LARGE_N] * B_col_squared_norms[e % LARGE_N]);
      constexpr double compared_output_count = 2.0;
      const double tolerance =
          compared_output_count *
          compute_tolerance(std::max(std::abs(C_host[e]), std::abs(C_cublas_host[e])),
                            abs_product_sum_bound, LARGE_N);
      const bool is_finite = std::isfinite(C_host[e]) && std::isfinite(C_cublas_host[e]);
      ASSERT_TRUE(is_finite && std::abs(C_host[e] - C_cublas_host[e]) <= tolerance)
          << "i = " << e / LARGE_N << ", j = " << e % LARGE_N << ", C = " << C_host[e]
          << ", cuBLAS = " << C_cublas_host[e];
    }
  }
}
