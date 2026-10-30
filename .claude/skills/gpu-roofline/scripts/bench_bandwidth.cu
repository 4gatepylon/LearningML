// Measures achievable memory bandwidth: DRAM read, write, copy (kernel and cudaMemcpy), and L2
// read (a buffer small enough to stay resident in L2). Reports the best of several timed runs.
// Output lines starting with "RESULT" are parsed by measure_peaks.py.
//
// Usage: bench_bandwidth [dram_buffer_MiB=1024]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                                 \
  do {                                                                              \
    cudaError_t err_ = (call);                                                      \
    if (err_ != cudaSuccess) {                                                      \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

constexpr int kThreads = 256;
constexpr int kWarmup = 3;
constexpr int kIters = 20;

__global__ void read_kernel(const float4* src, size_t n, int reps, float* sink) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  float acc = 0.f;
  // Each rep shifts the start index so the compiler can't reuse loads across reps.
  for (int r = 0; r < reps; ++r) {
    size_t offset = (size_t(r) * 32) % n;
    for (size_t i = tid; i < n; i += stride) {
      size_t j = i + offset;
      float4 v = __ldcg(&src[j < n ? j : j - n]);
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (acc == -1.2345f) *sink = acc;  // never true for our data; keeps the loads alive
}

__global__ void write_kernel(float4* dst, size_t n, float val) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    dst[i] = make_float4(val, val, val, val);
}

__global__ void copy_kernel(const float4* src, float4* dst, size_t n) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    dst[i] = src[i];
}

// Times `launch` and returns the best (minimum) duration in seconds.
template <typename F>
static double best_seconds(F launch) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < kWarmup; ++i) launch();
  CHECK(cudaGetLastError());
  std::vector<float> ms(kIters);
  for (int i = 0; i < kIters; ++i) {
    CHECK(cudaEventRecord(start));
    launch();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms[i], start, stop));
  }
  CHECK(cudaGetLastError());
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return *std::min_element(ms.begin(), ms.end()) / 1e3;
}

static void report(const char* key, const char* label, double bytes, double seconds) {
  double gbs = bytes / seconds / 1e9;
  printf("  %-34s %8.1f GB/s\n", label, gbs);
  printf("RESULT %s %.1f\n", key, gbs);
}

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1024;
  size_t bytes = mib << 20;
  size_t n = bytes / sizeof(float4);

  int dev = 0, sms = 0, blocks_per_sm = 0, l2_bytes = 0;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, copy_kernel, kThreads, 0));
  // Enough blocks to fill every SM several times over; the grid-stride loops cover the rest.
  int grid = sms * blocks_per_sm * 4;

  float4 *a = nullptr, *b = nullptr;
  float* sink = nullptr;
  CHECK(cudaMalloc(&a, bytes));
  CHECK(cudaMalloc(&b, bytes));
  CHECK(cudaMalloc(&sink, sizeof(float)));
  CHECK(cudaMemset(a, 0, bytes));

  printf("DRAM buffers: %zu MiB each, grid %d x %d threads\n", mib, grid, kThreads);
  report("dram_read_gbs", "DRAM read (kernel)", bytes,
         best_seconds([&] { read_kernel<<<grid, kThreads>>>(a, n, 1, sink); }));
  report("dram_write_gbs", "DRAM write (kernel)", bytes,
         best_seconds([&] { write_kernel<<<grid, kThreads>>>(b, n, 1.f); }));
  report("dram_copy_gbs", "DRAM copy (kernel, read+write)", 2.0 * bytes,
         best_seconds([&] { copy_kernel<<<grid, kThreads>>>(a, b, n); }));
  report("dram_memcpy_gbs", "DRAM copy (cudaMemcpy D2D)", 2.0 * bytes,
         best_seconds([&] { CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice)); }));

  // L2: a buffer a quarter of the L2 size, re-read many times within one launch.
  size_t l2_buf = size_t(l2_bytes) / 4;
  size_t l2_n = l2_buf / sizeof(float4);
  int reps = 200;
  report("l2_read_gbs", "L2 read (buffer = L2/4, re-read)", double(l2_buf) * reps,
         best_seconds([&] { read_kernel<<<grid, kThreads>>>(a, l2_n, reps, sink); }));

  CHECK(cudaFree(a));
  CHECK(cudaFree(b));
  CHECK(cudaFree(sink));
  return 0;
}
