#include <cmath>
#include <cstdio>
#include <vector>

// Element-wise c[i] = a[i] + b[i] on the CPU.
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

// Element-wise c[i] = a[i] + b[i] on the GPU.
// TODO: not implemented yet; leaves c untouched.
void vector_add_gpu(const float* a, const float* b, float* c, int n) {
  (void)a;
  (void)b;
  (void)c;
  (void)n;
}

// Runs both implementations on a, b and returns true if their outputs match.
static bool run_test(const char* name, const std::vector<float>& a,
                     const std::vector<float>& b) {
  const int n = static_cast<int>(a.size());
  std::vector<float> c_cpu(n, 0.0f), c_gpu(n, 0.0f);

  vector_add_cpu(a.data(), b.data(), c_cpu.data(), n);
  vector_add_gpu(a.data(), b.data(), c_gpu.data(), n);

  bool match = true;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(c_cpu[i] - c_gpu[i]) > 1e-5f) {
      match = false;
      break;
    }
  }
  std::printf("[%s] n=%d: %s\n", name, n, match ? "MATCH" : "MISMATCH");
  return match;
}

int main() {
  bool all_match = true;

  // Happy path: small ramp.
  {
    const int n = 8;
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(2 * i);
    }
    all_match &= run_test("happy path", a, b);
  }

  // Larger size that isn't a multiple of a typical block size, with negatives.
  {
    const int n = 1000;
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) {
      a[i] = static_cast<float>(i) - 500.0f;
      b[i] = 0.5f * static_cast<float>(i);
    }
    all_match &= run_test("non-multiple size", a, b);
  }

  std::printf("%s\n", all_match ? "All tests: CPU and GPU answers matched"
                                : "Some tests: CPU and GPU answers did NOT match");
  return all_match ? 0 : 1;
}
