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
 * - `matmul.sign_*`: One test per part of the sign partitioning (see Coverage).
 * - `matmul.magnitude_*`: Tests for the tiny and huge parts of the magnitude partitioning.
 * - `matmul.shape_*`: Tests for the non-square parts of the shape partitioning.
 *
 * @par Coverage
 * Every kernel is checked against the CPU reference on one exemplar per part. A is M x K and B
 * is K x N.
 * | Partitioning | Part | Tests |
 * | --- | --- | --- |
 * | Sign of every value in A and B | All zero | `matmul.sign_all_zero` |
 * | | Non-negative, some positive | `matmul.sign_non_negative` |
 * | | Non-positive, some negative | `matmul.sign_non_positive` |
 * | | Both signs | `matmul.sign_mixed`, `matmul.matches_reference_random_normal` |
 * | Largest abs value m in A and B | Tiny: 0 < m < 2^-63 | `matmul.magnitude_tiny_*` |
 * | | Normal: m = 0, or 2^-63 <= m <= 2^64 | `matmul.matches_reference_random_normal` |
 * | | Huge: m > 2^64 | `matmul.magnitude_huge_*` |
 * | Shape | Both square: M = K = N | `matmul.matches_reference_random_normal` |
 * | | Only A square: M = K != N | `matmul.shape_only_a_square` |
 * | | Only B square: K = N != M | `matmul.shape_only_b_square` |
 * | | Neither square, K > M and K > N | `matmul.shape_k_longest` |
 * | | Neither square, K < M and K < N | `matmul.shape_k_shortest` |
 * | | Neither square, K between M and N | `matmul.shape_k_between` |
 *
 * The magnitude limits are the square roots of bf16's smallest normal (2^-126) and largest value
 * (about 2^128): below and above them, products of two inputs leave bf16's normal range.
 *
 * @par Must know
 * - Gotchas:
 *   - The reference is too slow for large N, so large matrices are checked against cuBLAS.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "common/matmul.h"

namespace {

constexpr float UNWRITTEN_SENTINEL = -1.0F;
constexpr double BF16_UNIT_ROUNDOFF = 0x1p-8;
constexpr double FP32_UNIT_ROUNDOFF = 0x1p-24;
constexpr double BF16_HALF_MIN_SUBNORMAL = 0x1p-134;
constexpr float BF16_MIN_NORMAL = 0x1p-126F;
constexpr float BF16_MAX = 0x1.fep127F;
constexpr float TINY_MAGNITUDE_LIMIT = 0x1p-63F;
constexpr float HUGE_MAGNITUDE_LIMIT = 0x1p64F;
constexpr int REFERENCE_N = 96;
constexpr int LARGE_N = 4096;
constexpr unsigned A_SEED = 1;
constexpr unsigned B_SEED = 2;

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

/**
 * @brief `make_random_matrix` returns a matrix of values from `draw_value`, rounded to bf16.
 *
 * @param[in] row_count Number of rows.
 * @param[in] col_count Number of columns.
 * @param[in] seed Seed for the random number generator.
 * @param[in] draw_value Draws one value from the random number generator.
 * @return The matrix, row-major, with every value exactly representable in bf16.
 *
 * @par Example
 * @code
 * std::vector<float> A_host = make_random_matrix(M, K, A_SEED, draw_standard_normal);
 * @endcode
 */
std::vector<float> make_random_matrix(int row_count, int col_count, unsigned seed,
                                      const std::function<float(std::mt19937&)>& draw_value) {
  std::mt19937 rng(seed);
  std::vector<float> X_host(static_cast<size_t>(row_count) * col_count);
  for (float& x : X_host) {
    x = static_cast<float>(static_cast<bf16_t>(draw_value(rng)));
  }
  return X_host;
}

/**
 * @brief `draw_standard_normal` draws one N(0, 1) sample.
 *
 * @param[in,out] rng Random number generator.
 * @return The sample.
 *
 * @par Example
 * @code
 * std::vector<float> A_host = make_random_matrix(M, K, A_SEED, draw_standard_normal);
 * @endcode
 */
float draw_standard_normal(std::mt19937& rng) { return std::normal_distribution<float>()(rng); }

/**
 * @brief `Reference` holds the exact product C = A * B and, per element, the sum of |A||B|.
 *
 * @par Example
 * @code
 * Reference reference = compute_reference(A_host, B_host, M, N, K);
 * @endcode
 */
struct Reference {
  std::vector<double> C;
  std::vector<double> abs_product_sums;
};

/**
 * @brief `compute_reference` multiplies A and B on the CPU in double precision.
 *
 * @param[in] A_host Left factor, M x K, row-major, with values exactly representable in bf16.
 * @param[in] B_host Right factor, K x N, row-major, with values exactly representable in bf16.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 * @return C = A * B and, for each element, the sum over k of |A[i][k] * B[k][j]|.
 *
 * @par Example
 * @code
 * Reference reference = compute_reference(A_host, B_host, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - It takes O(N^3) time on one CPU thread, so it suits N up to a few hundred.
 */
Reference compute_reference(const std::vector<float>& A_host, const std::vector<float>& B_host,
                            int M, int N, int K) {
  const size_t C_element_count = static_cast<size_t>(M) * N;
  Reference reference{std::vector<double>(C_element_count), std::vector<double>(C_element_count)};
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        const double product = static_cast<double>(A_host[(i * K) + k]) * B_host[(k * N) + j];
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
 * @return Rounding to bf16 (twice, to cover the input side) plus fp32 accumulation over K terms,
 *         plus half of bf16's smallest subnormal for outputs that round to subnormals.
 *
 * @par Example
 * @code
 * double tolerance = compute_tolerance(reference.C[e], reference.abs_product_sums[e], K);
 * @endcode
 */
double compute_tolerance(double C_exact, double abs_product_sum, int K) {
  constexpr double rounding_count = 2.0;
  return (rounding_count * BF16_UNIT_ROUNDOFF * std::abs(C_exact)) +
         (K * FP32_UNIT_ROUNDOFF * abs_product_sum) + BF16_HALF_MIN_SUBNORMAL;
}

/**
 * @brief `expect_close_to_reference` checks every kernel's C = A * B against the CPU reference.
 *
 * @param[in] A_host Left factor, M x K, row-major, with values exactly representable in bf16.
 * @param[in] B_host Right factor, K x N, row-major, with values exactly representable in bf16.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 *
 * @par Example
 * @code
 * expect_close_to_reference(A_host, B_host, M, N, K);
 * @endcode
 */
void expect_close_to_reference(const std::vector<float>& A_host, const std::vector<float>& B_host,
                               int M, int N, int K) {
  const Reference reference = compute_reference(A_host, B_host, M, N, K);
  for (const Kernel& kernel : KERNELS) {
    SCOPED_TRACE(kernel.name);
    const std::vector<float> C_host = run_kernel(kernel.matmul_fn, A_host, B_host, M, N, K);
    for (size_t e = 0; e < C_host.size(); ++e) {
      const double tolerance = compute_tolerance(reference.C[e], reference.abs_product_sums[e], K);
      ASSERT_LE(std::abs(C_host[e] - reference.C[e]), tolerance)
          << "i = " << e / N << ", j = " << e % N << ", C = " << C_host[e]
          << ", reference = " << reference.C[e];
    }
  }
}

/**
 * @brief `find_max_abs_value` returns the largest absolute value in A and B.
 *
 * @param[in] A_host Left factor.
 * @param[in] B_host Right factor.
 * @return The largest absolute value, which picks the part of the magnitude partitioning.
 *
 * @par Example
 * @code
 * ASSERT_LT(find_max_abs_value(A_host, B_host), TINY_MAGNITUDE_LIMIT);
 * @endcode
 */
float find_max_abs_value(const std::vector<float>& A_host, const std::vector<float>& B_host) {
  float max_abs_value = 0.0F;
  for (const std::vector<float>* X_host : {&A_host, &B_host}) {
    for (const float x : *X_host) {
      max_abs_value = std::max(max_abs_value, std::abs(x));
    }
  }
  return max_abs_value;
}

/**
 * @brief `make_identity_matrix` returns the N x N identity matrix.
 *
 * @param[in] N Matrix size.
 * @return The identity, row-major.
 *
 * @par Example
 * @code
 * std::vector<float> B_host = make_identity_matrix(N);
 * @endcode
 */
std::vector<float> make_identity_matrix(int N) {
  std::vector<float> I_host(static_cast<size_t>(N) * N, 0.0F);
  for (int i = 0; i < N; ++i) {
    I_host[(i * N) + i] = 1.0F;
  }
  return I_host;
}

/**
 * @brief `expect_random_normal_close_to_reference` checks every kernel on random normal A and B.
 *
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 *
 * @par Example
 * @code
 * expect_random_normal_close_to_reference(64, 96, 64);
 * @endcode
 */
void expect_random_normal_close_to_reference(int M, int N, int K) {
  expect_close_to_reference(make_random_matrix(M, K, A_SEED, draw_standard_normal),
                            make_random_matrix(K, N, B_SEED, draw_standard_normal), M, N, K);
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

/**
 * @brief `matmul.reference_matches_known_2x2` checks the CPU reference on the hand-computed
 * product.
 */
TEST(matmul, reference_matches_known_2x2) {
  constexpr int N = 2;
  const Reference reference = compute_reference({1, 2, 3, 4}, {5, 6, 7, 8}, N, N, N);
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
  expect_close_to_reference(
      make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_standard_normal),
      make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_standard_normal), REFERENCE_N,
      REFERENCE_N, REFERENCE_N);
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
  const std::vector<float> A_host =
      make_random_matrix(LARGE_N, LARGE_N, A_SEED, draw_standard_normal);
  const std::vector<float> B_host =
      make_random_matrix(LARGE_N, LARGE_N, B_SEED, draw_standard_normal);
  std::vector<double> A_row_norms(LARGE_N);
  std::vector<double> B_col_norms(LARGE_N);
  for (int i = 0; i < LARGE_N; ++i) {
    for (int k = 0; k < LARGE_N; ++k) {
      A_row_norms[i] += A_host[(i * LARGE_N) + k] * A_host[(i * LARGE_N) + k];
      B_col_norms[i] += B_host[(k * LARGE_N) + i] * B_host[(k * LARGE_N) + i];
    }
  }
  const std::vector<float> C_cublas_host =
      run_kernel(matmul_cublas, A_host, B_host, LARGE_N, LARGE_N, LARGE_N);
  for (const Kernel& kernel : KERNELS) {
    SCOPED_TRACE(kernel.name);
    const std::vector<float> C_host =
        run_kernel(kernel.matmul_fn, A_host, B_host, LARGE_N, LARGE_N, LARGE_N);
    for (size_t e = 0; e < C_host.size(); ++e) {
      const double abs_product_sum_bound =
          std::sqrt(A_row_norms[e / LARGE_N]) * std::sqrt(B_col_norms[e % LARGE_N]);
      constexpr double compared_output_count = 2.0;
      const double tolerance =
          compared_output_count *
          compute_tolerance(std::max(std::abs(C_host[e]), std::abs(C_cublas_host[e])),
                            abs_product_sum_bound, LARGE_N);
      ASSERT_LE(std::abs(C_host[e] - C_cublas_host[e]), tolerance)
          << "i = " << e / LARGE_N << ", j = " << e % LARGE_N << ", C = " << C_host[e]
          << ", cuBLAS = " << C_cublas_host[e];
    }
  }
}

/**
 * @brief `matmul.sign_all_zero` covers the all-zero part: A and B are all zeros, so C must be too.
 */
TEST(matmul, sign_all_zero) {
  const std::vector<float> zero_matrix(static_cast<size_t>(REFERENCE_N) * REFERENCE_N, 0.0F);
  expect_close_to_reference(zero_matrix, zero_matrix, REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.sign_non_negative` covers the non-negative part with random 0s and 1s.
 */
TEST(matmul, sign_non_negative) {
  const auto draw_zero_or_one = [](std::mt19937& rng) {
    return static_cast<float>(std::uniform_int_distribution<int>(0, 1)(rng));
  };
  expect_close_to_reference(make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_zero_or_one),
                            make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_zero_or_one),
                            REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.sign_non_positive` covers the non-positive part with minus absolute normals.
 */
TEST(matmul, sign_non_positive) {
  const auto draw_non_positive = [](std::mt19937& rng) {
    return -std::abs(draw_standard_normal(rng));
  };
  expect_close_to_reference(make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_non_positive),
                            make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_non_positive),
                            REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.sign_mixed` covers the mixed-sign part with random -1s, 0s and 1s.
 */
TEST(matmul, sign_mixed) {
  const auto draw_minus_one_zero_or_one = [](std::mt19937& rng) {
    return static_cast<float>(std::uniform_int_distribution<int>(-1, 1)(rng));
  };
  expect_close_to_reference(
      make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_minus_one_zero_or_one),
      make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_minus_one_zero_or_one), REFERENCE_N,
      REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.magnitude_tiny_near_limit` covers the tiny part with +-{1, 2, 3} * 2^-65.
 *
 * @par Must know
 * - Gotchas:
 *   - Products of 2^-130 to 9 * 2^-130 are fp32 subnormals, so this checks subnormal accumulation.
 */
TEST(matmul, magnitude_tiny_near_limit) {
  constexpr float tiny_scale = 0x1p-65F;
  const auto draw_tiny = [&](std::mt19937& rng) {
    return tiny_scale * static_cast<float>(std::uniform_int_distribution<int>(-3, 3)(rng));
  };
  const std::vector<float> A_host = make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_tiny);
  const std::vector<float> B_host = make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_tiny);
  ASSERT_LT(find_max_abs_value(A_host, B_host), TINY_MAGNITUDE_LIMIT);
  expect_close_to_reference(A_host, B_host, REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.magnitude_tiny_at_bf16_min` covers the tiny part with +-bf16's smallest normal.
 *
 * @par Must know
 * - Gotchas:
 *   - Every product underflows, so every kernel must return zeros, not garbage or NaN.
 */
TEST(matmul, magnitude_tiny_at_bf16_min) {
  const auto draw_signed_min_normal = [](std::mt19937& rng) {
    return std::bernoulli_distribution()(rng) ? BF16_MIN_NORMAL : -BF16_MIN_NORMAL;
  };
  const std::vector<float> A_host =
      make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_signed_min_normal);
  const std::vector<float> B_host =
      make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_signed_min_normal);
  ASSERT_LT(find_max_abs_value(A_host, B_host), TINY_MAGNITUDE_LIMIT);
  expect_close_to_reference(A_host, B_host, REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.magnitude_huge_few_large_values` covers the huge part with a few +-2^100 values.
 *
 * @par Must know
 * - Gotchas:
 *   - A is normal except for `outlier_count` entries of +-2^100, so C stays finite.
 */
TEST(matmul, magnitude_huge_few_large_values) {
  constexpr float outlier_value = 0x1p100F;
  constexpr int outlier_count = 8;
  constexpr unsigned outlier_seed = 3;
  std::vector<float> A_host =
      make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_standard_normal);
  std::mt19937 outlier_rng(outlier_seed);  // NOLINT(bugprone-random-generator-seed)
  std::uniform_int_distribution<size_t> outlier_index_distribution(0, A_host.size() - 1);
  for (int outlier = 0; outlier < outlier_count; ++outlier) {
    const bool is_negative = std::bernoulli_distribution()(outlier_rng);
    A_host[outlier_index_distribution(outlier_rng)] = is_negative ? -outlier_value : outlier_value;
  }
  const std::vector<float> B_host =
      make_random_matrix(REFERENCE_N, REFERENCE_N, B_SEED, draw_standard_normal);
  ASSERT_GT(find_max_abs_value(A_host, B_host), HUGE_MAGNITUDE_LIMIT);
  expect_close_to_reference(A_host, B_host, REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.magnitude_huge_at_bf16_max` covers the huge part with +-bf16's largest value.
 *
 * @par Must know
 * - Gotchas:
 *   - B is the identity, so C = A must reproduce bf16's largest value exactly, without overflow.
 */
TEST(matmul, magnitude_huge_at_bf16_max) {
  const auto draw_signed_max = [](std::mt19937& rng) {
    return std::bernoulli_distribution()(rng) ? BF16_MAX : -BF16_MAX;
  };
  const std::vector<float> A_host =
      make_random_matrix(REFERENCE_N, REFERENCE_N, A_SEED, draw_signed_max);
  const std::vector<float> B_host = make_identity_matrix(REFERENCE_N);
  ASSERT_GT(find_max_abs_value(A_host, B_host), HUGE_MAGNITUDE_LIMIT);
  expect_close_to_reference(A_host, B_host, REFERENCE_N, REFERENCE_N, REFERENCE_N);
}

/**
 * @brief `matmul.shape_only_a_square` covers M = K != N.
 */
TEST(matmul, shape_only_a_square) {
  constexpr int M = 64;
  constexpr int N = 96;
  constexpr int K = 64;
  static_assert(M == K && K != N);
  expect_random_normal_close_to_reference(M, N, K);
}

/**
 * @brief `matmul.shape_only_b_square` covers K = N != M.
 */
TEST(matmul, shape_only_b_square) {
  constexpr int M = 96;
  constexpr int N = 64;
  constexpr int K = 64;
  static_assert(K == N && N != M);
  expect_random_normal_close_to_reference(M, N, K);
}

/**
 * @brief `matmul.shape_k_longest` covers K > M and K > N, with neither matrix square.
 */
TEST(matmul, shape_k_longest) {
  constexpr int M = 48;
  constexpr int N = 64;
  constexpr int K = 128;
  static_assert(K > M && K > N);
  expect_random_normal_close_to_reference(M, N, K);
}

/**
 * @brief `matmul.shape_k_shortest` covers K < M and K < N, with neither matrix square.
 */
TEST(matmul, shape_k_shortest) {
  constexpr int M = 96;
  constexpr int N = 80;
  constexpr int K = 32;
  static_assert(K < M && K < N);
  expect_random_normal_close_to_reference(M, N, K);
}

/**
 * @brief `matmul.shape_k_between` covers K between M and N, with neither matrix square.
 */
TEST(matmul, shape_k_between) {
  constexpr int M = 40;
  constexpr int N = 112;
  constexpr int K = 72;
  static_assert(M < K && K < N);
  expect_random_normal_close_to_reference(M, N, K);
}
