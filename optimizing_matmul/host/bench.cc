/**
 * @file
 * @brief `bench.cc` times `trial_count` calls of each kernel and prints one CSV line per call.
 *
 * @par Interface
 * - Arguments: `N warmup_iteration_count trial_count`.
 * - Output: one `name,elapsed_ms_according_to_cuda_events` line per timed call on stdout.
 *
 * @par Example
 * @code
 * .build/bench 8192 1 20
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The first timed cuBLAS call includes cuBLAS setup when `warmup_iteration_count` is 0.
 *   - Inputs are normal with sigma `INPUT_SIGMA` and never all zeros, which cuBLAS runs faster.
 *   - Each call is timed alone with CUDA events, so launch overhead inflates small-N times.
 *   - This timing method is provisional and may need to change, e.g. to timing batches of calls.
 *   - Times are burst times; sustained load hits the 310 W cap and slows cuBLAS to ~3.0 ms.
 *   - Adding or removing kernels changes the gaps between calls, and so every kernel's times.
 * - Performance:
 *   - Trials run trial-major (each kernel once per trial), so kernels don't run back to back.
 */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/matmul.h"

namespace {

constexpr int EXPECTED_ARG_COUNT = 4;
constexpr int PROGRAM_NAME_ARG_INDEX = 0;
constexpr int N_ARG_INDEX = 1;
constexpr int WARMUP_ITERATION_COUNT_ARG_INDEX = 2;
constexpr int TRIAL_COUNT_ARG_INDEX = 3;
constexpr int MIN_N = 1;
constexpr int MIN_WARMUP_ITERATION_COUNT = 0;
constexpr int MIN_TRIAL_COUNT = 1;
constexpr unsigned RNG_SEED = 0;
constexpr float INPUT_MEAN = 0.0F;
constexpr float INPUT_SIGMA = 1.0F;

/**
 * @brief `parse_int_at_least` parses all of `text` as a base-10 int that is at least `min_value`.
 *
 * @param[in] text Text to parse.
 * @param[in] min_value Smallest value to accept.
 * @return The parsed value.
 * @throws std::invalid_argument `text` is not a whole integer, or its value is below `min_value`.
 * @throws std::out_of_range `text` does not fit in an `int`.
 *
 * @par Example
 * @code
 * int N = parse_int_at_least("8192", MIN_N);
 * @endcode
 */
int parse_int_at_least(const std::string& text, int min_value) {
  size_t parsed_char_count = 0;
  const int value = std::stoi(text, &parsed_char_count);
  if (parsed_char_count != text.size() || value < min_value) {
    throw std::invalid_argument(text);
  }
  return value;
}

/**
 * @brief `upload_random_matrix` fills `temp_X_host` with normal samples and copies them over.
 *
 * @param[in,out] rng Random number generator that supplies the samples.
 * @param[in,out] temp_X_host Scratch buffer; its size sets the matrix size.
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
bf16_t* upload_random_matrix(std::mt19937& rng, std::vector<float>& temp_X_host) {
  std::normal_distribution<float> input_distribution(INPUT_MEAN, INPUT_SIGMA);
  for (float& x : temp_X_host) {
    x = input_distribution(rng);
  }
  return copy_to_device(temp_X_host);
}

}  // namespace

/**
 * @brief `main` warms up every kernel, then times `trial_count` interleaved calls of each.
 *
 * @param[in] argc Argument count, `EXPECTED_ARG_COUNT`.
 * @param[in] argv Program name, `N` (at least `MIN_N`), `warmup_iteration_count` (at least
 *                 `MIN_WARMUP_ITERATION_COUNT`) and `trial_count` (at least `MIN_TRIAL_COUNT`).
 * @return `EXIT_SUCCESS`, or `EXIT_FAILURE` after a usage error such as a negative argument.
 *
 * @par Example
 * @code
 * .build/bench 8192 1 20
 * @endcode
 */
int main(int argc, char** argv) {
  int N = 0;
  int warmup_iteration_count = 0;
  int trial_count = 0;
  try {
    if (argc != EXPECTED_ARG_COUNT) {
      throw std::invalid_argument("wrong argument count");
    }
    N = parse_int_at_least(argv[N_ARG_INDEX], MIN_N);
    warmup_iteration_count =
        parse_int_at_least(argv[WARMUP_ITERATION_COUNT_ARG_INDEX], MIN_WARMUP_ITERATION_COUNT);
    trial_count = parse_int_at_least(argv[TRIAL_COUNT_ARG_INDEX], MIN_TRIAL_COUNT);
  } catch (const std::logic_error&) {
    std::fprintf(stderr,
                 "usage: %s N warmup_iteration_count trial_count (N >= %d, "
                 "warmup_iteration_count >= %d, trial_count >= %d)\n",
                 argv[PROGRAM_NAME_ARG_INDEX], MIN_N, MIN_WARMUP_ITERATION_COUNT, MIN_TRIAL_COUNT);
    return EXIT_FAILURE;
  }
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
    for (int i = 0; i < warmup_iteration_count; ++i) {
      kernel.matmul_fn(A_device, B_device, C_device, N);
    }
  }
  for (int trial = 0; trial < trial_count; ++trial) {
    for (const Kernel& kernel : KERNELS) {
      CHECK(cudaEventRecord(start_event));
      kernel.matmul_fn(A_device, B_device, C_device, N);
      CHECK(cudaEventRecord(stop_event));
      CHECK(cudaEventSynchronize(stop_event));
      CHECK(cudaGetLastError());
      float elapsed_ms_according_to_cuda_events = 0.0F;
      CHECK(cudaEventElapsedTime(&elapsed_ms_according_to_cuda_events, start_event, stop_event));
      std::printf("%s,%.4f\n", kernel.name, elapsed_ms_according_to_cuda_events);
    }
  }
  return EXIT_SUCCESS;
}
