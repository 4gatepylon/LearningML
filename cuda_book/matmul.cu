#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU implementations and CPU launch code/helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: kernel. C (m x n) = A (m x k) * B (k x n), all row-major.
__global__ void matmul_kernel_on_gpu(const float* a, const float* b, float* c, int m, int n,
                                     int k) {}

// C = A * B on the GPU, with A m x k, B k x n and C m x n, all row-major.
void matmul_gpu(const float* a, const float* b, float* c, int m, int n, int k) {
  // TODO: allocate, copy, launch, copy back, free.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU reference implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

// C = A * B on the CPU, with A m x k, B k x n and C m x n, all row-major.
void matmul_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
  // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (main comes from gtest_main)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Runs both implementations on the same inputs and expects the outputs to match (stops at the first
// mismatch). The tolerance scales with k since summation order differs between CPU and GPU.
static void ExpectCpuGpuMatch(const std::vector<float>& a, const std::vector<float>& b, int m,
                              int n, int k) {
  ASSERT_EQ(a.size(), static_cast<size_t>(m) * static_cast<size_t>(k));
  ASSERT_EQ(b.size(), static_cast<size_t>(k) * static_cast<size_t>(n));
  std::vector<float> c_cpu(static_cast<size_t>(m) * n, 0.0F);
  std::vector<float> c_gpu(static_cast<size_t>(m) * n, 0.0F);

  matmul_cpu(a.data(), b.data(), c_cpu.data(), m, n, k);
  matmul_gpu(a.data(), b.data(), c_gpu.data(), m, n, k);

  const float tol = 1e-5F * static_cast<float>(k > 1 ? k : 1);
  for (size_t i = 0; i < c_cpu.size(); ++i) {
    ASSERT_NEAR(c_cpu[i], c_gpu[i], tol) << "first mismatch at index " << i;
  }
}

// Fills v with a deterministic pattern of small values in [-1, 1).
static std::vector<float> Pattern(int size, int seed) {
  std::vector<float> v(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    v[static_cast<size_t>(i)] = (static_cast<float>(((i * 31) + (seed * 17)) % 64) / 32.0F) - 1.0F;
  }
  return v;
}

// Happy path: small square matrices.
TEST(Matmul, HappyPath) {
  const int m = 8;
  const int n = 8;
  const int k = 8;
  ExpectCpuGpuMatch(Pattern(m * k, 1), Pattern(k * n, 2), m, n, k);
}

// Non-square sizes that aren't multiples of a typical tile or block size.
TEST(Matmul, NonMultipleSizes) {
  const int m = 37;
  const int n = 53;
  const int k = 29;
  ExpectCpuGpuMatch(Pattern(m * k, 3), Pattern(k * n, 4), m, n, k);
}

// Degenerate shapes: matrix-vector (n = 1) and a single inner element (k = 1).
TEST(Matmul, VectorAndOuterProduct) {
  ExpectCpuGpuMatch(Pattern(16 * 10, 5), Pattern(10 * 1, 6), 16, 1, 10);
  ExpectCpuGpuMatch(Pattern(9 * 1, 7), Pattern(1 * 12, 8), 9, 12, 1);
}
