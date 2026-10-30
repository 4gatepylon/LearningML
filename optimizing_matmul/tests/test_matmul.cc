#include <gtest/gtest.h>

#include <vector>

#include "common/matmul.h"

// Runs one kernel on host matrices and returns C = A * B as floats. C starts out as -1 everywhere,
// so entries the kernel never writes show up as wrong.
static std::vector<float> run(MatmulFn fn, const std::vector<float>& A_h,
                              const std::vector<float>& B_h, int n) {
  bf16* A_d = to_device(A_h);
  bf16* B_d = to_device(B_h);
  bf16* C_d = to_device(std::vector<float>(A_h.size(), -1.0F));
  fn(A_d, B_d, C_d, n);
  CHECK(cudaGetLastError());
  std::vector<bf16> C_h(A_h.size());
  CHECK(cudaMemcpy(C_h.data(), C_d, C_h.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
  for (bf16* x_d : {A_d, B_d, C_d}) {
    CHECK(cudaFree(x_d));
  }
  return {C_h.begin(), C_h.end()};
}

// A hand-computed 2x2 product. B * A, A^T * B, A * B^T and A^T * B^T all differ from A * B, so a
// kernel that swaps the operands or reads a matrix transposed fails. All values are exact in bf16.
TEST(Matmul, Known2x2) {
  const std::vector<float> A_h = {1, 2, 3, 4};
  const std::vector<float> B_h = {5, 6, 7, 8};
  const std::vector<float> C_expected_h = {19, 22, 43, 50};
  for (const Kernel& kernel : kKernels) {
    SCOPED_TRACE(kernel.name);
    EXPECT_EQ(run(kernel.fn, A_h, B_h, 2), C_expected_h);
  }
}
