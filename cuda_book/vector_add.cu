#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU implementations and CPU launch code/helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void vector_add_kernel_on_gpu(const float* a, const float* b, float* c, int n, int k) {
  // Threads index i maps onto index j as such:
  // Thread 0: 0, 1, 2, 3, ..., k-1
  // Thread 1: k, ..., 2*k-1
  // ...
  // Thread i: i*k, ..., (i+1)*k-1
  // ...
  int i_thread = static_cast<int>((blockDim.x * blockIdx.x) + threadIdx.x);
  int i_arrays = i_thread * k;
  for (int j_arrays = i_arrays; j_arrays < i_arrays + k && j_arrays < n; j_arrays++) {
    c[j_arrays] = a[j_arrays] + b[j_arrays];
  }
}

// Element-wise c[i] = a[i] + b[i] on the GPU.
void vector_add_gpu(const float* a, const float* b, float* c, int n) {
  // Allocate on CUDA memory
  // (we can store as a then b then c)
  float* ptr_to_abc_on_cuda;
  const size_t n_bytes = static_cast<size_t>(n) * sizeof(float);
  const std::ptrdiff_t n_elems = n;
  const size_t block_size_gpu = 256;
  const size_t thread_size_gpu = 32;

  const size_t n_elems_per_block_gpu = block_size_gpu * thread_size_gpu;
  const size_t n_blocks_gpu = (n + n_elems_per_block_gpu - 1) / n_elems_per_block_gpu;
  assert(n_blocks_gpu >= 1);

  cudaError_t malloc_err = cudaMalloc((void**)&ptr_to_abc_on_cuda, 3 * n_bytes);
  if (malloc_err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(malloc_err));
  }
  cudaMemcpy(ptr_to_abc_on_cuda + (0 * n_elems), a, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_to_abc_on_cuda + (1 * n_elems), b, n_bytes, cudaMemcpyHostToDevice);

  // Do computation
  vector_add_kernel_on_gpu<<<n_blocks_gpu, block_size_gpu>>>(
      ptr_to_abc_on_cuda + (0 * n_elems), ptr_to_abc_on_cuda + (1 * n_elems),
      ptr_to_abc_on_cuda + (2 * n_elems), n, thread_size_gpu);

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
