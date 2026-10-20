#include <gtest/gtest.h>

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU implementations and CPU launch code/helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// Element-wise c[i] = a[i] + b[i] on the GPU.
// TODO: not implemented yet; leaves c untouched.
// NOLINTNEXTLINE(readability-non-const-parameter): remove once c is actually written.
void vector_add_gpu(const float* a, const float* b, float* c, int n) {
  (void)a;
  (void)b;
  (void)c;
  (void)n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU reference implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

// Element-wise c[i] = a[i] + b[i] on the CPU.
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (main comes from gtest_main)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Runs both implementations on a, b and expects their outputs to match (stops at the first
// mismatch).
static void ExpectCpuGpuMatch(const std::vector<float>& a, const std::vector<float>& b) {
  const int n = static_cast<int>(a.size());
  std::vector<float> c_cpu(n, 0.0F);
  std::vector<float> c_gpu(n, 0.0F);

  vector_add_cpu(a.data(), b.data(), c_cpu.data(), n);
  vector_add_gpu(a.data(), b.data(), c_gpu.data(), n);

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(c_cpu[i], c_gpu[i], 1e-5F) << "first mismatch at index " << i;
  }
}

// Happy path: small ramp.
TEST(VectorAdd, HappyPath) {
  const int n = 8;
  std::vector<float> a(n);
  std::vector<float> b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
  ExpectCpuGpuMatch(a, b);
}

// Larger size that isn't a multiple of a typical block size, with negatives.
TEST(VectorAdd, NonMultipleSize) {
  const int n = 1000;
  std::vector<float> a(n);
  std::vector<float> b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) - 500.0F;
    b[i] = 0.5F * static_cast<float>(i);
  }
  ExpectCpuGpuMatch(a, b);
}
