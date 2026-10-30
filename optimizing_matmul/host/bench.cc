// Times one call of each kernel on random n x n matrices after `warmup` untimed calls, printing
// one CSV line per kernel (name,milliseconds). Run it through scripts/bench.py.
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common/matmul.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s n warmup_iterations\n", argv[0]);
    return 1;
  }
  const int n = std::stoi(argv[1]);
  const int warmup = std::stoi(argv[2]);
  std::mt19937 rng(0);  // NOLINT(bugprone-random-generator-seed): same inputs on every run
  std::normal_distribution<float> normal;
  std::vector<float> x_h(static_cast<size_t>(n) * n);
  const auto random_matrix = [&] {
    for (float& x : x_h) {
      x = normal(rng);
    }
    return to_device(x_h);
  };
  bf16* A_d = random_matrix();
  bf16* B_d = random_matrix();
  bf16* C_d = nullptr;
  CHECK(cudaMalloc(&C_d, x_h.size() * sizeof(bf16)));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  for (const Kernel& kernel : kKernels) {
    // Warmup keeps one-time setup (cuBLAS handle, module loading) out of the timed call.
    for (int i = 0; i < warmup; ++i) {
      kernel.fn(A_d, B_d, C_d, n);
    }
    CHECK(cudaEventRecord(start));
    kernel.fn(A_d, B_d, C_d, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());
    float ms = 0.0F;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::printf("%s,%.4f\n", kernel.name, ms);
  }
}
