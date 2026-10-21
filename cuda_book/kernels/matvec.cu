#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU implementations and CPU launch code/helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: kernel. a = B * c, with B an N x N row-major matrix and a, c vectors of length N.
// One thread computes one element of a.
__global__ void matvec_kernel_on_gpu(float* a, const float* b, const float* c, int n) {}

// a = B * c on the GPU, with B N x N row-major and a, c of length N.
void matvec_gpu(float* a, const float* b, const float* c, int n) {
  // TODO: allocate, copy, launch, copy back, free.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU reference implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

// a = B * c on the CPU, with B N x N row-major and a, c of length N.
void matvec_cpu(float* a, const float* b, const float* c, int n) {
  // TODO: a_i = sum_j B[i][j] * c[j].
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (main comes from gtest_main)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Fails if `a` is exactly zero everywhere. Outputs start zeroed and every test input is positive,
// so a correct product is never all zero; an all-zero output means nothing was ever written.
static ::testing::AssertionResult NotAllZero(const std::vector<float>& a, const char* who) {
  for (float x : a) {
    if (x != 0.0F) {
      return ::testing::AssertionSuccess();
    }
  }
  return ::testing::AssertionFailure()
         << who << " left all " << a.size()
         << " output values exactly zero. The test inputs are all positive, so a correct product "
            "is never all zero: "
         << who << " is probably unimplemented or never writes to its output.";
}

// Runs both implementations on the same inputs and expects the outputs to match (stops at the first
// mismatch). Inputs must be positive so the expected output is never all zero. The tolerance scales
// with n since summation order may differ between CPU and GPU.
static void ExpectCpuGpuMatch(const std::vector<float>& b, const std::vector<float>& c, int n) {
  ASSERT_EQ(b.size(), static_cast<size_t>(n) * static_cast<size_t>(n));
  ASSERT_EQ(c.size(), static_cast<size_t>(n));
  std::vector<float> a_cpu(static_cast<size_t>(n), 0.0F);
  std::vector<float> a_gpu(static_cast<size_t>(n), 0.0F);

  matvec_cpu(a_cpu.data(), b.data(), c.data(), n);
  matvec_gpu(a_gpu.data(), b.data(), c.data(), n);

  ASSERT_TRUE(NotAllZero(a_cpu, "matvec_cpu"));
  ASSERT_TRUE(NotAllZero(a_gpu, "matvec_gpu"));
  const float tol = 1e-5F * static_cast<float>(n > 1 ? n : 1);
  for (size_t i = 0; i < a_cpu.size(); ++i) {
    ASSERT_NEAR(a_cpu[i], a_gpu[i], tol) << "first mismatch at index " << i;
  }
}

// Fills v with a deterministic pattern of strictly positive values in (0, 2].
static std::vector<float> Pattern(int size, int seed) {
  std::vector<float> v(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    v[static_cast<size_t>(i)] = static_cast<float>((((i * 31) + (seed * 17)) % 64) + 1) / 32.0F;
  }
  return v;
}

// Happy path: small square matrix.
TEST(Matvec, HappyPath) {
  const int n = 8;
  ExpectCpuGpuMatch(Pattern(n * n, 1), Pattern(n, 2), n);
}

// Sizes that aren't multiples of a typical block size.
TEST(Matvec, NonMultipleSize) {
  const int n = 37;
  ExpectCpuGpuMatch(Pattern(n * n, 3), Pattern(n, 4), n);
}

// Degenerate 1x1 case.
TEST(Matvec, SingleElement) { ExpectCpuGpuMatch(Pattern(1, 5), Pattern(1, 6), 1); }

// Spans many thread blocks.
TEST(Matvec, Large) {
  const int n = 1000;
  ExpectCpuGpuMatch(Pattern(n * n, 7), Pattern(n, 8), n);
}

// CPU only: runs matvec_cpu and expects exactly the hand-computed `expected` product.
static void ExpectCpuEquals(const std::vector<float>& b, const std::vector<float>& c, int n,
                            const std::vector<float>& expected) {
  ASSERT_EQ(expected.size(), static_cast<size_t>(n));
  std::vector<float> a(expected.size(), 0.0F);
  matvec_cpu(a.data(), b.data(), c.data(), n);
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_NEAR(a[i], expected[i], 1e-5F) << "first mismatch at index " << i;
  }
}

// CPU only: [[1, 2], [3, 4]] * [5, 6] = [17, 39].
TEST(MatvecCpu, TwoByTwo) { ExpectCpuEquals({1, 2, 3, 4}, {5, 6}, 2, {17, 39}); }

// CPU only: the identity matrix leaves the vector unchanged.
TEST(MatvecCpu, IdentityLeavesVectorUnchanged) {
  const std::vector<float> c = {1, 2, 3};
  ExpectCpuEquals({1, 0, 0, 0, 1, 0, 0, 0, 1}, c, 3, c);
}
