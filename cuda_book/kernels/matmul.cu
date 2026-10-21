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
  // TODO(hadriano) support tile-based coarsening in both axes (so each individual does a single tile loop-nest and the tile
  // size can default be 1 for fully parallel; however the k tile dim is 1 always because idk how to do synchronization)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU reference implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

// C = A * B on the CPU, with A m x k, B k x n and C m x n, all row-major.
void matmul_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
  // TODO(hadriano) implement some way of parameterizing the loop orderings and be able to
  // cost-model.
  // TODO(hadriano) learn to use sanitize.

  // Left matrix is A: [m x k]
  // Right matrix is B: [k x n]
  // Output matrix is C: [m x n]
  const int m_tile_size = 16;
  const int n_tile_size = 16;
  const int k_tile_size = 16;
  // Outer nest
  for (int m_tile_offset_2d = 0; m_tile_offset_2d < m; m_tile_offset_2d += m_tile_size) {
    for (int n_tile_offset_2d = 0; n_tile_offset_2d < n; n_tile_offset_2d += n_tile_size) {
      for (int k_tile_offset_2d = 0; k_tile_offset_2d < k; k_tile_offset_2d += k_tile_size) {
        // Inner nest
        for (int m_tile_idx_2d = 0;
             m_tile_idx_2d < m_tile_size && m_tile_offset_2d + m_tile_idx_2d < m; m_tile_idx_2d++) {
          for (int n_tile_idx_2d = 0;
               n_tile_idx_2d < n_tile_size && n_tile_offset_2d + n_tile_idx_2d < n;
               n_tile_idx_2d++) {
            // Output write index
            for (int k_tile_idx_2d = 0;
                 k_tile_idx_2d < k_tile_size && k_tile_offset_2d + k_tile_idx_2d < k;
                 k_tile_idx_2d++) {
              // Get locations in the array
              const int m_idx_2d = m_tile_offset_2d + m_tile_idx_2d;
              const int n_idx_2d = n_tile_offset_2d + n_tile_idx_2d;
              const int k_idx_2d = k_tile_offset_2d + k_tile_idx_2d;
              // We want to do this:
              // c[m_global_idx][n_global_idx] +=
              //     a[m_global_idx][k_global_idx] * b[k_global_idx][n_global_idx];
              const int a_idx_1d =
                  m_idx_2d * k +
                  k_idx_2d;  // a is [m x k] where k is the width, so we need to increase by k
              const int b_idx_1d =
                  k_idx_2d * n +
                  n_idx_2d;  // b is [k x n] where n is the width, so we need to increase by n
              const int c_idx_1d =
                  m_idx_2d * n +
                  n_idx_2d;  // c is [m x n] where n is the width, so we need to increase by n
              c[c_idx_1d] += a[a_idx_1d] * b[b_idx_1d];
            }
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (main comes from gtest_main)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Fails if `c` is exactly zero everywhere. Outputs start zeroed and every test input is positive,
// so a correct product is never all zero; an all-zero output means nothing was ever written.
static ::testing::AssertionResult NotAllZero(const std::vector<float>& c, const char* who) {
  for (float x : c) {
    if (x != 0.0F) {
      return ::testing::AssertionSuccess();
    }
  }
  return ::testing::AssertionFailure()
         << who << " left all " << c.size()
         << " output values exactly zero. The test inputs are all positive, so a correct product "
            "is never all zero: "
         << who << " is probably unimplemented or never writes to its output.";
}

// Runs both implementations on the same inputs and expects the outputs to match (stops at the first
// mismatch). Inputs must be positive so the expected output is never all zero. The tolerance scales
// with k since summation order differs between CPU and GPU.
static void ExpectCpuGpuMatch(const std::vector<float>& a, const std::vector<float>& b, int m,
                              int n, int k) {
  ASSERT_EQ(a.size(), static_cast<size_t>(m) * static_cast<size_t>(k));
  ASSERT_EQ(b.size(), static_cast<size_t>(k) * static_cast<size_t>(n));
  std::vector<float> c_cpu(static_cast<size_t>(m) * n, 0.0F);
  std::vector<float> c_gpu(static_cast<size_t>(m) * n, 0.0F);

  matmul_cpu(a.data(), b.data(), c_cpu.data(), m, n, k);
  matmul_gpu(a.data(), b.data(), c_gpu.data(), m, n, k);

  ASSERT_TRUE(NotAllZero(c_cpu, "matmul_cpu"));
  ASSERT_TRUE(NotAllZero(c_gpu, "matmul_gpu"));
  const float tol = 1e-5F * static_cast<float>(k > 1 ? k : 1);
  for (size_t i = 0; i < c_cpu.size(); ++i) {
    ASSERT_NEAR(c_cpu[i], c_gpu[i], tol) << "first mismatch at index " << i;
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

// CPU only: runs matmul_cpu and expects exactly the hand-computed `expected` product.
static void ExpectCpuEquals(const std::vector<float>& a, const std::vector<float>& b, int m, int n,
                            int k, const std::vector<float>& expected) {
  ASSERT_EQ(expected.size(), static_cast<size_t>(m) * static_cast<size_t>(n));
  std::vector<float> c(expected.size(), 0.0F);
  matmul_cpu(a.data(), b.data(), c.data(), m, n, k);
  for (size_t i = 0; i < c.size(); ++i) {
    ASSERT_NEAR(c[i], expected[i], 1e-5F) << "first mismatch at index " << i;
  }
}

// CPU only: [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]].
TEST(MatmulCpu, TwoByTwo) {
  ExpectCpuEquals({1, 2, 3, 4}, {5, 6, 7, 8}, 2, 2, 2, {19, 22, 43, 50});
}

// CPU only: a 2x3 matrix times the 3x3 identity is unchanged.
TEST(MatmulCpu, IdentityLeavesMatrixUnchanged) {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  ExpectCpuEquals(a, identity, 2, 3, 3, a);
}

// CPU only: a 1x3 row times a 3x1 column is their dot product: 1*4 + 2*5 + 3*6 = 32.
TEST(MatmulCpu, RowTimesColumnIsDotProduct) {
  ExpectCpuEquals({1, 2, 3}, {4, 5, 6}, 1, 1, 3, {32});
}
