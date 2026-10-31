/**
 * @file
 * @brief `bench.cc` times `trial_count` batches of `batch_size` calls per kernel, one line each.
 *
 * @par Interface
 * - Arguments: `N warmup_iteration_count trial_count batch_size [kernel_name ...]`, where no
 *   kernel names selects every kernel; or `--list-kernels` alone.
 * - Output: one `name,ms_per_call_according_to_cuda_events` line per batch on stdout; or, with
 *   `--list-kernels`, one kernel name per line.
 *
 * @par Example
 * @code
 * .build/bench 8192 1 3 100 cublas
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The first timed cuBLAS call includes cuBLAS setup when `warmup_iteration_count` is 0.
 *   - Inputs are normal with sigma `INPUT_SIGMA` and never all zeros, which cuBLAS runs faster.
 *   - A batch's stubs overlap GPU work, so only its first stub and launch are in its time.
 *   - Calls in a batch run back to back, so long batches reach the 310 W cap's throttled clocks.
 *   - Adding or removing kernels changes the gaps between batches, and so the kernels' times.
 * - Performance:
 *   - Trials run trial-major (each kernel's batch once per trial), so batches interleave.
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/matmul.h"

namespace {

constexpr int MIN_ARG_COUNT = 5;
constexpr int LIST_KERNELS_ARG_COUNT = 2;
constexpr int FLAG_ARG_INDEX = 1;
constexpr std::string_view LIST_KERNELS_FLAG = "--list-kernels";
constexpr int PROGRAM_NAME_ARG_INDEX = 0;
constexpr int N_ARG_INDEX = 1;
constexpr int WARMUP_ITERATION_COUNT_ARG_INDEX = 2;
constexpr int TRIAL_COUNT_ARG_INDEX = 3;
constexpr int BATCH_SIZE_ARG_INDEX = 4;
constexpr int FIRST_KERNEL_NAME_ARG_INDEX = 5;
constexpr int MIN_N = 1;
constexpr int MIN_WARMUP_ITERATION_COUNT = 0;
constexpr int MIN_TRIAL_COUNT = 1;
constexpr int MIN_BATCH_SIZE = 1;
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

/**
 * @brief `select_kernels` returns the kernels in `KERNELS` named in `kernel_names`, or all of them.
 *
 * @param[in] kernel_names Names of the kernels to select; empty selects every kernel.
 * @return The selected kernels, in `KERNELS` order.
 * @throws std::invalid_argument A name in `kernel_names` is not in `KERNELS`.
 *
 * @par Example
 * @code
 * std::vector<Kernel> selected_kernels = select_kernels({"cublas"});
 * @endcode
 */
std::vector<Kernel> select_kernels(const std::vector<std::string>& kernel_names) {
  for (const std::string& kernel_name : kernel_names) {
    const bool is_known =
        std::any_of(std::begin(KERNELS), std::end(KERNELS),
                    [&](const Kernel& kernel) { return kernel_name == kernel.name; });
    if (!is_known) {
      throw std::invalid_argument(kernel_name);
    }
  }
  std::vector<Kernel> selected_kernels;
  for (const Kernel& kernel : KERNELS) {
    const bool is_selected =
        kernel_names.empty() ||
        std::find(kernel_names.begin(), kernel_names.end(), kernel.name) != kernel_names.end();
    if (is_selected) {
      selected_kernels.push_back(kernel);
    }
  }
  return selected_kernels;
}

}  // namespace

/**
 * @brief `main` warms up the selected kernels, then times `trial_count` interleaved batches of
 * each.
 *
 * @param[in] argc Argument count, at least `MIN_ARG_COUNT`, or `LIST_KERNELS_ARG_COUNT`.
 * @param[in] argv Program name, `N` (at least `MIN_N`), `warmup_iteration_count` (at least
 *                 `MIN_WARMUP_ITERATION_COUNT`), `trial_count` (at least `MIN_TRIAL_COUNT`),
 *                 `batch_size` (at least `MIN_BATCH_SIZE`) and optional kernel names; or
 *                 program name and `--list-kernels`.
 * @return `EXIT_SUCCESS`, or `EXIT_FAILURE` after a usage error such as a negative argument.
 *
 * @par Example
 * @code
 * .build/bench 8192 1 3 100 cublas naive
 * @endcode
 */
int main(int argc, char** argv) {
  int N = 0;
  int warmup_iteration_count = 0;
  int trial_count = 0;
  if (argc == LIST_KERNELS_ARG_COUNT && argv[FLAG_ARG_INDEX] == LIST_KERNELS_FLAG) {
    for (const Kernel& kernel : KERNELS) {
      std::printf("%s\n", kernel.name);
    }
    return EXIT_SUCCESS;
  }
  int batch_size = 0;
  std::vector<Kernel> selected_kernels;
  try {
    if (argc < MIN_ARG_COUNT) {
      throw std::invalid_argument("too few arguments");
    }
    N = parse_int_at_least(argv[N_ARG_INDEX], MIN_N);
    warmup_iteration_count =
        parse_int_at_least(argv[WARMUP_ITERATION_COUNT_ARG_INDEX], MIN_WARMUP_ITERATION_COUNT);
    trial_count = parse_int_at_least(argv[TRIAL_COUNT_ARG_INDEX], MIN_TRIAL_COUNT);
    batch_size = parse_int_at_least(argv[BATCH_SIZE_ARG_INDEX], MIN_BATCH_SIZE);
    selected_kernels = select_kernels({argv + FIRST_KERNEL_NAME_ARG_INDEX, argv + argc});
  } catch (const std::logic_error&) {
    std::fprintf(stderr,
                 "usage: %s N warmup_iteration_count trial_count batch_size [kernel_name ...] "
                 "(N >= %d, warmup_iteration_count >= %d, trial_count >= %d, batch_size >= %d), "
                 "or %s --list-kernels\n",
                 argv[PROGRAM_NAME_ARG_INDEX], MIN_N, MIN_WARMUP_ITERATION_COUNT, MIN_TRIAL_COUNT,
                 MIN_BATCH_SIZE, argv[PROGRAM_NAME_ARG_INDEX]);
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
  for (const Kernel& kernel : selected_kernels) {
    for (int i = 0; i < warmup_iteration_count; ++i) {
      kernel.matmul_fn(A_device, B_device, C_device, N, N, N);
    }
  }
  for (int trial = 0; trial < trial_count; ++trial) {
    for (const Kernel& kernel : selected_kernels) {
      CHECK(cudaEventRecord(start_event));
      for (int i = 0; i < batch_size; ++i) {
        kernel.matmul_fn(A_device, B_device, C_device, N, N, N);
      }
      CHECK(cudaEventRecord(stop_event));
      CHECK(cudaEventSynchronize(stop_event));
      CHECK(cudaGetLastError());
      float batch_ms_according_to_cuda_events = 0.0F;
      CHECK(cudaEventElapsedTime(&batch_ms_according_to_cuda_events, start_event, stop_event));
      const float ms_per_call_according_to_cuda_events =
          batch_ms_according_to_cuda_events / static_cast<float>(batch_size);
      std::printf("%s,%.6f\n", kernel.name, ms_per_call_according_to_cuda_events);
    }
  }
  return EXIT_SUCCESS;
}
