#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU implementations and CPU launch code/helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void vector_add_kernel_on_gpu(const float* a, const float* b, float* c, int n) {
  int i = static_cast<int>((blockDim.x * blockIdx.x) + threadIdx.x);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Element-wise c[i] = a[i] + b[i] on the GPU.
void vector_add_gpu(const float* a, const float* b, float* c, int n) {
  // Allocate on CUDA memory
  // (we can store as a then b then c)
  float* ptr_to_abc_on_cuda;
  const size_t n_bytes = static_cast<size_t>(n) * sizeof(float);
  const std::ptrdiff_t n_elems = n;
  cudaError_t malloc_err = cudaMalloc((void**)&ptr_to_abc_on_cuda, 3 * n_bytes);
  if (malloc_err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(malloc_err));
  }
  cudaMemcpy(ptr_to_abc_on_cuda + (0 * n_elems), a, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_to_abc_on_cuda + (1 * n_elems), b, n_bytes, cudaMemcpyHostToDevice);

  // Do computation
  vector_add_kernel_on_gpu<<<((n + 255) / 256), 256>>>(ptr_to_abc_on_cuda + (0 * n_elems),
                                                       ptr_to_abc_on_cuda + (1 * n_elems),
                                                       ptr_to_abc_on_cuda + (2 * n_elems), n);

  // Free on Cuda
  cudaMemcpy(c, ptr_to_abc_on_cuda + (2 * n_elems), n_bytes, cudaMemcpyDeviceToHost);
  cudaFree(ptr_to_abc_on_cuda);
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
