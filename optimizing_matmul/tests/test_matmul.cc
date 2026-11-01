/**
 * @file
 * @brief `test_matmul.cc` checks every kernel in `KERNELS` exactly against a CPU reference.
 *
 * @par Interface
 * - `matmul/KernelTest.known_2x2/<kernel>`: Checks a hand-computed 2x2 product.
 * - `matmul/KernelTest.random_square/<kernel>`: Checks random 0/1 matrices, 200 x 200.
 *
 * @par Example
 * @code
 * make test
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - Inputs are 0 or 1 outside `known_2x2`, so the tests check data movement, not rounding.
 * - Gotchas:
 *   - Every output is an integer of at most 255, so bf16 holds it and the check is exact.
 *   - The cap is 255, not 256: bf16(257) is 256, so 256 would hide one extra product.
 *   - Every row of A has at most 255 ones, so every output is at most 255, whatever M, N and K.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/matmul.h"

namespace {

constexpr float UNWRITTEN_SENTINEL = -1.0F;
constexpr int MAX_OUTPUT_VALUE = 255;
constexpr double B_ONE_PROBABILITY = 0.5;
constexpr uint32_t A_SEED = 1;
constexpr uint32_t B_SEED = 2;
constexpr size_t REPORTED_MISMATCH_COUNT = 5;
constexpr size_t REPORTED_DIFFERENCE_COUNT = 8;

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
 * @brief `make_random_A_with_at_most_255_ones_per_row` makes a random 0/1 M x K matrix.
 *
 * @param[in] M Rows.
 * @param[in] K Columns.
 * @param[in] seed Seed for the choice of columns.
 * @return A, M x K, row-major: each row has min(ceil(K / 2), 255) ones, in random columns.
 *
 * @par Must know
 * - Gotchas:
 *   - C[i][j] is at most the count of ones in row i of A, so every output is at most 255.
 */
std::vector<float> make_random_A_with_at_most_255_ones_per_row(int M, int K, uint32_t seed) {
  const int ones_per_row = std::min((K + 1) / 2, MAX_OUTPUT_VALUE);
  std::mt19937 rng(seed);
  std::vector<int> columns(K);
  std::iota(columns.begin(), columns.end(), 0);
  std::vector<float> A_host(static_cast<size_t>(M) * K, 0.0F);
  for (int64_t i = 0; i < M; ++i) {
    std::shuffle(columns.begin(), columns.end(), rng);
    std::for_each_n(columns.begin(), ones_per_row, [&](int k) { A_host[(i * K) + k] = 1.0F; });
  }
  return A_host;
}

/**
 * @brief `make_random_B_with_half_ones_on_average` makes a random 0/1 K x N matrix.
 *
 * @param[in] K Rows.
 * @param[in] N Columns.
 * @param[in] seed Seed for the entries.
 * @return B, K x N, row-major: each entry is 1 with probability 1/2.
 */
std::vector<float> make_random_B_with_half_ones_on_average(int K, int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::bernoulli_distribution draw_is_one(B_ONE_PROBABILITY);
  std::vector<float> B_host(static_cast<size_t>(K) * N);
  std::generate(B_host.begin(), B_host.end(), [&] { return draw_is_one(rng) ? 1.0F : 0.0F; });
  return B_host;
}

/**
 * @brief `compute_reference_matmul_on_cpu` computes C = A * B on the CPU in double precision.
 *
 * @param[in] A_host Left factor, M x K, row-major.
 * @param[in] B_host Right factor, K x N, row-major.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 * @return C = A * B, M x N, row-major.
 *
 * @par Must know
 * - Gotchas:
 *   - On 0/1 inputs every partial sum is a small integer, which double holds exactly.
 * - Performance:
 *   - It skips the zeros of A, so it costs M * N times the ones per row of A, not M * N * K.
 *   - It loops i, k, j so it reads rows of B; a dot product per output is 10x slower at 8192.
 */
std::vector<double> compute_reference_matmul_on_cpu(const std::vector<float>& A_host,
                                                    const std::vector<float>& B_host, int M, int N,
                                                    int K) {
  std::vector<double> C_host(static_cast<size_t>(M) * N, 0.0);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      const double A_ik = A_host[(i * K) + k];
      if (A_ik == 0.0) {
        continue;
      }
      for (int64_t j = 0; j < N; ++j) {
        C_host[(i * N) + j] += A_ik * B_host[(k * N) + j];
      }
    }
  }
  return C_host;
}

/**
 * @brief `expect_kernel_output_matches_reference` checks a kernel's output against the reference.
 *
 * @param[in] matmul_fn Kernel to run.
 * @param[in] A_host Left factor, M x K, row-major, 0/1.
 * @param[in] B_host Right factor, K x N, row-major, 0/1.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C.
 * @param[in] K Columns of A and rows of B.
 *
 * @par Must know
 * - Gotchas:
 *   - It first asserts that no reference output passes 255, so a test can't break the cap.
 *   - A failure prints the mismatch count, the first mismatches and the differences got - expected.
 */
void expect_kernel_output_matches_reference(matmul_fn_t matmul_fn, const std::vector<float>& A_host,
                                            const std::vector<float>& B_host, int M, int N, int K) {
  const std::vector<double> C_reference_host =
      compute_reference_matmul_on_cpu(A_host, B_host, M, N, K);
  ASSERT_LE(*std::max_element(C_reference_host.begin(), C_reference_host.end()), MAX_OUTPUT_VALUE);
  const std::vector<float> C_host = run_kernel(matmul_fn, A_host, B_host, M, N, K);
  size_t mismatch_count = 0;
  std::set<double> differences;
  std::ostringstream report;
  for (size_t e = 0; e < C_host.size(); ++e) {
    const double C_expected = C_reference_host[e];
    if (C_host[e] == C_expected) {
      continue;
    }
    if (mismatch_count < REPORTED_MISMATCH_COUNT) {
      report << " C[" << e / N << "][" << e % N << "] = " << C_host[e] << ", not " << C_expected
             << ";";
    }
    ++mismatch_count;
    differences.insert(C_host[e] - C_expected);
  }
  report << " differences:";
  std::for_each_n(differences.begin(), std::min(differences.size(), REPORTED_DIFFERENCE_COUNT),
                  [&](double difference) { report << " " << difference; });
  EXPECT_EQ(mismatch_count, 0U) << mismatch_count << " of " << C_host.size()
                                << " outputs differ:" << report.str();
}

}  // namespace

/**
 * @brief `PrintTo` prints a kernel as its name in gtest's messages.
 *
 * @param[in] kernel Kernel to print.
 * @param[out] stream Stream to print to.
 *
 * @par Must know
 * - Gotchas:
 *   - It sits outside the anonymous namespace because gtest finds it by argument-dependent lookup.
 */
void PrintTo(const Kernel& kernel, std::ostream* stream) { *stream << kernel.name; }

/**
 * @brief `KernelTest` runs each test once per kernel in `KERNELS`, named after the kernel.
 */
class KernelTest : public testing::TestWithParam<Kernel> {};

/**
 * @brief `KernelTest.known_2x2` checks a hand-computed 2x2 product.
 *
 * @par Must know
 * - Gotchas:
 *   - B * A, A^T * B, A * B^T and A^T * B^T all differ from A * B, so swaps and transposes fail.
 *   - Its inputs run from 1 to 8, the one exception to the 0/1 rule; its largest output is 50.
 */
TEST_P(KernelTest, known_2x2) {
  constexpr int N = 2;
  const std::vector<float> A_host = {1, 2, 3, 4};
  const std::vector<float> B_host = {5, 6, 7, 8};
  const std::vector<float> C_expected_host = {19, 22, 43, 50};
  EXPECT_EQ(run_kernel(GetParam().matmul_fn, A_host, B_host, N, N, N), C_expected_host);
}

/**
 * @brief `KernelTest.random_square` checks random 0/1 matrices, 200 x 200, against the reference.
 *
 * @par Must know
 * - Gotchas:
 *   - 200 spans several blocks and is a multiple of 8 but of no larger power of two.
 */
TEST_P(KernelTest, random_square) {
  constexpr int N = 200;
  expect_kernel_output_matches_reference(
      GetParam().matmul_fn, make_random_A_with_at_most_255_ones_per_row(N, N, A_SEED),
      make_random_B_with_half_ones_on_average(N, N, B_SEED), N, N, N);
}

INSTANTIATE_TEST_SUITE_P(matmul, KernelTest, testing::ValuesIn(KERNELS),
                         [](const testing::TestParamInfo<Kernel>& info) {
                           return std::string(info.param.name);
                         });
