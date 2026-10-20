#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU implementations and CPU launch code/helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: kernel. Each output pixel is the mean of the (2*radius+1)^2 window around it. Pick an edge
// rule (e.g. average only the in-bounds pixels) and use the same one in box_blur_cpu.
__global__ void box_blur_kernel_on_gpu(const float* in, float* out, int width, int height,
                                       int radius) {}

// Box blur of a row-major width x height single-channel image on the GPU.
void box_blur_gpu(const float* in, float* out, int width, int height, int radius) {
  // TODO: allocate, copy, launch, copy back, free.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU reference implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

// Box blur of a row-major width x height single-channel image on the CPU.
void box_blur_cpu(const float* in, float* out, int width, int height, int radius) {
  // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (main comes from gtest_main)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Runs both implementations on the same image and expects the outputs to match (stops at the first
// mismatch).
static void ExpectCpuGpuMatch(const std::vector<float>& in, int width, int height, int radius) {
  ASSERT_EQ(in.size(), static_cast<size_t>(width) * static_cast<size_t>(height));
  std::vector<float> out_cpu(in.size(), 0.0F);
  std::vector<float> out_gpu(in.size(), 0.0F);

  box_blur_cpu(in.data(), out_cpu.data(), width, height, radius);
  box_blur_gpu(in.data(), out_gpu.data(), width, height, radius);

  for (size_t i = 0; i < in.size(); ++i) {
    ASSERT_NEAR(out_cpu[i], out_gpu[i], 1e-4F) << "first mismatch at index " << i;
  }
}

// Happy path: small square ramp image, radius 1.
TEST(BoxBlur, HappyPath) {
  const int width = 8;
  const int height = 8;
  std::vector<float> in(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i);
  }
  ExpectCpuGpuMatch(in, width, height, 1);
}

// Non-square size that isn't a multiple of a typical block size, larger radius.
TEST(BoxBlur, NonMultipleSize) {
  const int width = 67;
  const int height = 33;
  std::vector<float> in(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>((i * 37) % 101) - 50.0F;
  }
  ExpectCpuGpuMatch(in, width, height, 3);
}

// Radius 0 is the identity.
TEST(BoxBlur, RadiusZero) {
  const int width = 5;
  const int height = 4;
  std::vector<float> in(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i) * 0.25F;
  }
  ExpectCpuGpuMatch(in, width, height, 0);
}
