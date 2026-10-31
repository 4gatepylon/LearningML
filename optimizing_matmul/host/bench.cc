/**
 * @file
 * @brief `bench.cc` times one call of each kernel and prints one CSV line per kernel.
 *
 * @par Interface
 * - Arguments: `N warmup_iterations`.
 * - Output: one `name,milliseconds` line per kernel on stdout.
 *
 * @par Example
 * @code
 * .build/bench 8192 1
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The timed call includes cuBLAS setup when `warmup_iterations` is 0.
 *   - `scripts/bench.py` runs this binary with its defaults and prints a table.
 */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "common/matmul.h"

constexpr int EXPECTED_ARG_COUNT = 3;
constexpr int PROGRAM_NAME_ARG_INDEX = 0;
constexpr int N_ARG_INDEX = 1;
constexpr int WARMUP_ITERATIONS_ARG_INDEX = 2;
constexpr unsigned RNG_SEED = 0;

/**
 * @brief `upload_random_matrix` fills `temp_X_host` with N(0, 1) samples and copies them over.
 *
 * @param[in,out] rng Random number generator that supplies the samples.
 * @param[out] temp_X_host Scratch buffer; its size sets the matrix size.
 * @return A new device buffer that holds the samples rounded to bf16.
 *
 * @par Example
 * @code
 * bf16_t* A_device = upload_random_matrix(rng, temp_X_host);
 * @endcode
 *
 * @par Must know
 * - Side effects:
 *   - The caller owns the returned buffer and must `cudaFree` it.
 */
static bf16_t* upload_random_matrix(std::mt19937& rng, std::vector<float>& temp_X_host) {
  std::normal_distribution<float> standard_normal;
  for (float& x : temp_X_host) {
    x = standard_normal(rng);
  }
  return copy_to_device(temp_X_host);
}

/**
 * @brief `main` times one call of each kernel after `warmup_iterations` untimed calls.
 *
 * @param[in] argc Argument count, `EXPECTED_ARG_COUNT`.
 * @param[in] argv Program name, `N` and `warmup_iterations`.
 * @return `EXIT_SUCCESS`, or `EXIT_FAILURE` after a usage error.
 *
 * @par Example
 * @code
 * .build/bench 8192 1
 * @endcode
 */
int main(int argc, char** argv) {
  if (argc != EXPECTED_ARG_COUNT) {
    std::fprintf(stderr, "usage: %s N warmup_iterations\n", argv[PROGRAM_NAME_ARG_INDEX]);
    return EXIT_FAILURE;
  }
  const int N = std::stoi(argv[N_ARG_INDEX]);
  const int warmup_iterations = std::stoi(argv[WARMUP_ITERATIONS_ARG_INDEX]);
  std::mt19937 rng(RNG_SEED);  // NOLINT(bugprone-random-generator-seed)
  std::vector<float> temp_X_host(static_cast<size_t>(N) * N);
  bf16_t* A_device = upload_random_matrix(rng, temp_X_host);
  bf16_t* B_device = upload_random_matrix(rng, temp_X_host);
  bf16_t* C_device = nullptr;
  CHECK(cudaMalloc(&C_device, temp_X_host.size() * sizeof(bf16_t)));

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CHECK(cudaEventCreate(&start_event));
  CHECK(cudaEventCreate(&stop_event));
  for (const Kernel& kernel : KERNELS) {
    for (int i = 0; i < warmup_iterations; ++i) {
      kernel.matmul_fn(A_device, B_device, C_device, N);
    }
    CHECK(cudaEventRecord(start_event));
    kernel.matmul_fn(A_device, B_device, C_device, N);
    CHECK(cudaEventRecord(stop_event));
    CHECK(cudaEventSynchronize(stop_event));
    CHECK(cudaGetLastError());
    float elapsed_ms_according_to_cuda_events = 0.0F;
    CHECK(cudaEventElapsedTime(&elapsed_ms_according_to_cuda_events, start_event, stop_event));
    std::printf("%s,%.4f\n", kernel.name, elapsed_ms_according_to_cuda_events);
  }
  return EXIT_SUCCESS;
}
