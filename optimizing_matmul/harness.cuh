// Correctness tests and timing harness for N x N bf16 matmuls (fp32 accumulate, bf16 output).
//
// Matrices are row-major device arrays and C = A * B. An implementation is a function
//
//     void my_matmul(const bf16 *A, const bf16 *B, bf16 *C, int n);
//
// that launches its work on the default stream. Its .cu file includes this header and ends with
//
//     MATMUL_TESTS(MyMatmul, my_matmul)
//     int main(int argc, char **argv) { return run_matmul("my_matmul", my_matmul, argc, argv); }
//
// run_matmul runs the GoogleTest tests (the implementation must match cuBLAS on every entry at
// several sizes), then, if they pass, benchmarks it. Flags: --n N (benchmark size, default 8192),
// --tries T (timed calls, default 100), --warmup-ms MS (default 2000), --no-bench, plus the usual
// --gtest_* flags.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;
using MatmulFn = void (*)(const bf16* A, const bf16* B, bf16* C, int n);

inline void check_cuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s\n", file, line, cudaGetErrorString(err));
    exit(1);
  }
}

inline void check_cublas(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s:%d: cuBLAS error %d\n", file, line, status);
    exit(1);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)
#define CHECK_CUBLAS(call) check_cublas((call), __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS reference
////////////////////////////////////////////////////////////////////////////////////////////////////

// cuBLAS is column-major, and a row-major matrix read as column-major is its transpose. So the
// column-major product C^T = B^T * A^T, i.e. passing B before A, gives row-major C = A * B.
inline void matmul_cublas(const bf16* A, const bf16* B, bf16* C, int n) {
  static cublasHandle_t handle = [] {
    cublasHandle_t h = nullptr;
    CHECK_CUBLAS(cublasCreate(&h));
    return h;
  }();
  const float alpha = 1.0F;
  const float beta = 0.0F;
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, CUDA_R_16BF, n, A,
                            CUDA_R_16BF, n, &beta, C, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// Fills x with pseudo-random values in [-1, 1) (splitmix64 hash of index and seed). Random data
// matters for timing: all-zero inputs draw less power, so the GPU clocks higher and matmuls look
// about 1.6x faster than they are on real data.
__global__ void fill_uniform(bf16* x, size_t count, uint64_t seed) {
  for (size_t i = (size_t(blockIdx.x) * blockDim.x) + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    uint64_t z = i + (seed * 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z ^= z >> 31U;
    x[i] = __float2bfloat16((float(z >> 40U) / float(1U << 23U)) - 1.0F);
  }
}

// Allocates and randomly fills A and B, and allocates C, all n x n.
inline void alloc_inputs(int n, bf16** A, bf16** B, bf16** C) {
  size_t count = size_t(n) * n;
  CHECK_CUDA(cudaMalloc(A, count * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(B, count * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(C, count * sizeof(bf16)));
  fill_uniform<<<1024, 256>>>(*A, count, 2 * uint64_t(n));
  fill_uniform<<<1024, 256>>>(*B, count, (2 * uint64_t(n)) + 1);
  CHECK_CUDA(cudaGetLastError());
}

// Two correct implementations differ by bf16 rounding of the output (relative 2^-8 each) plus
// small fp32 accumulation-order differences: allow 2^-7 relative error plus a little absolute
// slack.
inline bool close(double got, double want) {
  return std::fabs(got - want) <= (std::fabs(want) * 0x1p-7) + 1e-2;
}

inline std::vector<bf16> to_host(const bf16* x_d, int n) {
  std::vector<bf16> x_h(size_t(n) * n);
  CHECK_CUDA(cudaMemcpy(x_h.data(), x_d, x_h.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
  return x_h;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Correctness tests
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Mismatches {
  size_t count = 0;
  size_t first = 0;  // index of the first mismatch, if count > 0
};

inline Mismatches find_mismatches(const std::vector<bf16>& got, const std::vector<bf16>& want) {
  Mismatches m;
  for (size_t idx = 0; idx < got.size(); ++idx) {
    if (!close(__bfloat162float(got[idx]), __bfloat162float(want[idx])) && m.count++ == 0) {
      m.first = idx;
    }
  }
  return m;
}

// Expects impl to match cuBLAS on every entry of an n x n product of random matrices.
inline void expect_matches_cublas(MatmulFn impl, int n) {
  bf16* A = nullptr;
  bf16* B = nullptr;
  bf16* C = nullptr;
  bf16* C_ref = nullptr;
  alloc_inputs(n, &A, &B, &C);
  CHECK_CUDA(cudaMalloc(&C_ref, size_t(n) * n * sizeof(bf16)));
  // 0xFF bytes are bf16 NaNs, so entries the implementation never writes fail the check.
  CHECK_CUDA(cudaMemset(C, 0xFF, size_t(n) * n * sizeof(bf16)));
  impl(A, B, C, n);
  CHECK_CUDA(cudaGetLastError());
  matmul_cublas(A, B, C_ref, n);
  std::vector<bf16> C_h = to_host(C, n);
  std::vector<bf16> C_ref_h = to_host(C_ref, n);

  Mismatches m = find_mismatches(C_h, C_ref_h);
  EXPECT_EQ(m.count, 0U) << "n=" << n << ": " << m.count << " of " << C_h.size()
                         << " entries differ from cuBLAS; first is C[" << m.first / n << "]["
                         << m.first % n << "] = " << __bfloat162float(C_h[m.first])
                         << ", cuBLAS has " << __bfloat162float(C_ref_h[m.first]);

  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C));
  CHECK_CUDA(cudaFree(C_ref));
}

// Defines the correctness tests for one implementation: it must match cuBLAS at a small size, a
// medium one and the benchmark size.
#define MATMUL_TESTS(Suite, impl)                                        \
  TEST(Suite, MatchesCublasN128) { expect_matches_cublas(impl, 128); }   \
  TEST(Suite, MatchesCublasN1024) { expect_matches_cublas(impl, 1024); } \
  TEST(Suite, MatchesCublasN8192) { expect_matches_cublas(impl, 8192); }

// Runs cuBLAS on random n x n inputs and returns how many of `samples` random entries differ from
// an fp64 CPU dot product.
inline int count_cublas_fp64_mismatches(int n, int samples) {
  bf16* A = nullptr;
  bf16* B = nullptr;
  bf16* C = nullptr;
  alloc_inputs(n, &A, &B, &C);
  matmul_cublas(A, B, C, n);
  std::vector<bf16> A_h = to_host(A, n);
  std::vector<bf16> B_h = to_host(B, n);
  std::vector<bf16> C_h = to_host(C, n);
  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C));

  std::mt19937 rng(n);
  std::uniform_int_distribution<int> index(0, n - 1);
  int bad = 0;
  for (int s = 0; s < samples; ++s) {
    int i = index(rng);
    int j = index(rng);
    double want = 0;
    for (int k = 0; k < n; ++k) {
      want += double(__bfloat162float(A_h[(size_t(i) * n) + k])) *
              double(__bfloat162float(B_h[(size_t(k) * n) + j]));
    }
    if (!close(__bfloat162float(C_h[(size_t(i) * n) + j]), want)) {
      ++bad;
    }
  }
  return bad;
}

// The reference itself: cuBLAS must match an fp64 CPU dot product on a random sample of entries.
TEST(CublasReference, MatchesFp64OnSampledEntries) {
  for (int n : {128, 1024, 8192}) {
    EXPECT_EQ(count_cublas_fp64_mismatches(n, 1024), 0)
        << "n=" << n << ": cuBLAS disagrees with fp64 on sampled entries";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark
////////////////////////////////////////////////////////////////////////////////////////////////////

// Calls impl back to back until warmup_ms has passed, so clocks and power settle at their
// sustained level before timing.
inline void warm_up(MatmulFn impl, const bf16* A, const bf16* B, bf16* C, int n, float warmup_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t now = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&now));
  CHECK_CUDA(cudaEventRecord(start));
  for (float elapsed = 0; elapsed < warmup_ms;) {
    impl(A, B, C, n);
    CHECK_CUDA(cudaEventRecord(now));
    CHECK_CUDA(cudaEventSynchronize(now));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, now));
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(now));
}

// Times `tries` back-to-back calls, with no host sync in between; events[t] and events[t + 1]
// bracket call t. Returns the per-call times in ms, sorted.
inline std::vector<float> time_calls(MatmulFn impl, const bf16* A, const bf16* B, bf16* C, int n,
                                     int tries) {
  std::vector<cudaEvent_t> events(tries + 1);
  for (auto& e : events) {
    CHECK_CUDA(cudaEventCreate(&e));
  }
  CHECK_CUDA(cudaEventRecord(events[0]));
  for (int t = 0; t < tries; ++t) {
    impl(A, B, C, n);
    CHECK_CUDA(cudaEventRecord(events[t + 1]));
  }
  CHECK_CUDA(cudaEventSynchronize(events[tries]));
  CHECK_CUDA(cudaGetLastError());
  std::vector<float> ms(tries);
  for (int t = 0; t < tries; ++t) {
    CHECK_CUDA(cudaEventElapsedTime(&ms[t], events[t], events[t + 1]));
  }
  for (auto& e : events) {
    CHECK_CUDA(cudaEventDestroy(e));
  }
  std::sort(ms.begin(), ms.end());
  return ms;
}

// Benchmarks impl on random n x n inputs: warm up, then time `tries` calls and report the minimum
// (best case) and median (typical sustained) times.
inline void benchmark(const char* name, MatmulFn impl, int n, int tries, float warmup_ms) {
  bf16* A = nullptr;
  bf16* B = nullptr;
  bf16* C = nullptr;
  alloc_inputs(n, &A, &B, &C);
  warm_up(impl, A, B, C, n, warmup_ms);
  std::vector<float> ms = time_calls(impl, A, B, C, n, tries);
  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C));

  double flops = 2.0 * n * n * n;
  auto tflops = [&](float t) { return flops / (t * 1e-3) / 1e12; };
  printf("bench n=%d %s: min %.3f ms (%.1f TFLOP/s), median %.3f ms (%.1f TFLOP/s), %d tries\n", n,
         name, ms[0], tflops(ms[0]), ms[tries / 2], tflops(ms[tries / 2]), tries);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Entry point
////////////////////////////////////////////////////////////////////////////////////////////////////

// Runs the GoogleTest tests, then (if they all pass and --no-bench isn't given) benchmarks impl.
inline int run_matmul(const char* name, MatmulFn impl, int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);  // removes the --gtest_* flags it recognizes
  int n = 8192;
  int tries = 100;
  float warmup_ms = 2000;
  bool bench = true;
  for (int a = 1; a < argc; ++a) {
    std::string flag = argv[a];
    bool has_value = a + 1 < argc;
    if (flag == "--no-bench") {
      bench = false;
    } else if (has_value && flag == "--n") {
      n = std::stoi(argv[++a]);
    } else if (has_value && flag == "--tries") {
      tries = std::stoi(argv[++a]);
    } else if (has_value && flag == "--warmup-ms") {
      warmup_ms = std::stof(argv[++a]);
    } else {
      fprintf(stderr, "unknown flag %s\n", flag.c_str());
      return 2;
    }
  }

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  const char* visible = getenv("CUDA_VISIBLE_DEVICES");
  printf("%s on %s (CUDA_VISIBLE_DEVICES=%s)\n", name, prop.name,
         visible != nullptr ? visible : "unset");

  if (RUN_ALL_TESTS() != 0) {
    printf("Correctness tests failed; not benchmarking\n");
    return 1;
  }
  if (bench) {
    benchmark(name, impl, n, tries, warmup_ms);
  }
  return 0;
}
