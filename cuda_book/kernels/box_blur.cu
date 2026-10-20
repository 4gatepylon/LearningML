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

// Fails if `out` is exactly zero everywhere. Outputs start zeroed and every test input is positive,
// so a correct result is never all zero; an all-zero output means the function never wrote to it.
static ::testing::AssertionResult NotAllZero(const std::vector<float>& out, const char* who) {
  for (float x : out) {
    if (x != 0.0F) {
      return ::testing::AssertionSuccess();
    }
  }
  return ::testing::AssertionFailure()
         << who << " left all " << out.size()
         << " output values exactly zero. The test inputs are all positive, so a correct blur is "
            "never all zero: "
         << who << " is probably unimplemented or never writes to its output.";
}

// Runs both implementations on the same image and expects the outputs to match (stops at the first
// mismatch). Inputs must be positive so the expected output is never all zero.
static void ExpectCpuGpuMatch(const std::vector<float>& in, int width, int height, int radius) {
  ASSERT_EQ(in.size(), static_cast<size_t>(width) * static_cast<size_t>(height));
  std::vector<float> out_cpu(in.size(), 0.0F);
  std::vector<float> out_gpu(in.size(), 0.0F);

  box_blur_cpu(in.data(), out_cpu.data(), width, height, radius);
  box_blur_gpu(in.data(), out_gpu.data(), width, height, radius);

  ASSERT_TRUE(NotAllZero(out_cpu, "box_blur_cpu"));
  ASSERT_TRUE(NotAllZero(out_gpu, "box_blur_gpu"));
  for (size_t i = 0; i < in.size(); ++i) {
    ASSERT_NEAR(out_cpu[i], out_gpu[i], 1e-4F) << "first mismatch at index " << i;
  }
}

// Happy path: small square ramp image (starting at 1), radius 1.
TEST(BoxBlur, HappyPath) {
  const int width = 8;
  const int height = 8;
  std::vector<float> in(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i + 1);
  }
  ExpectCpuGpuMatch(in, width, height, 1);
}

// Non-square size that isn't a multiple of a typical block size, larger radius.
TEST(BoxBlur, NonMultipleSize) {
  const int width = 67;
  const int height = 33;
  std::vector<float> in(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>((i * 37) % 101) + 1.0F;
  }
  ExpectCpuGpuMatch(in, width, height, 3);
}

// Radius 0 is the identity.
TEST(BoxBlur, RadiusZero) {
  const int width = 5;
  const int height = 4;
  std::vector<float> in(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i + 1) * 0.25F;
  }
  ExpectCpuGpuMatch(in, width, height, 0);
}

// CPU only: runs box_blur_cpu and expects exactly the hand-computed `expected` image.
static void ExpectCpuEquals(const std::vector<float>& in, int width, int height, int radius,
                            const std::vector<float>& expected) {
  ASSERT_EQ(in.size(), expected.size());
  std::vector<float> out(in.size(), 0.0F);
  box_blur_cpu(in.data(), out.data(), width, height, radius);
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_NEAR(out[i], expected[i], 1e-5F) << "first mismatch at index " << i;
  }
}

// CPU only: blurring a constant image leaves it unchanged, whatever the radius. This also checks
// that edge pixels are averaged over only their in-bounds neighbors.
TEST(BoxBlurCpu, ConstantImageStaysConstant) {
  const int width = 6;
  const int height = 4;
  const std::vector<float> in(static_cast<size_t>(width) * height, 3.0F);
  ExpectCpuEquals(in, width, height, 1, in);
  ExpectCpuEquals(in, width, height, 2, in);
}

// CPU only: a single 9 in the middle of a zero 5x5 image with radius 1 spreads to a 3x3 patch of 1s
// (9 / 9 each). The result doesn't depend on the edge rule since the patch is away from the edges.
TEST(BoxBlurCpu, SinglePixelSpreadsToThreeByThree) {
  const int width = 5;
  const int height = 5;
  std::vector<float> in(static_cast<size_t>(width) * height, 0.0F);
  in[(2 * width) + 2] = 9.0F;

  std::vector<float> expected(in.size(), 0.0F);
  for (int i = 1; i <= 3; ++i) {
    for (int j = 1; j <= 3; ++j) {
      expected[(i * width) + j] = 1.0F;
    }
  }
  ExpectCpuEquals(in, width, height, 1, expected);
}
