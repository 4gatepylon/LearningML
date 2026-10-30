// Prints the current GPU's properties and its theoretical peaks (FP32/FP64 FMA throughput, DRAM
// bandwidth) computed from the SM count, clocks and bus width the driver reports.
// Output lines starting with "RESULT" are parsed by measure_peaks.py.
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                                 \
  do {                                                                              \
    cudaError_t err_ = (call);                                                      \
    if (err_ != cudaSuccess) {                                                      \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

// FP32 and FP64 FMA lanes per SM by compute capability (major*10 + minor). -1 means unknown.
struct LanesPerSm {
  int cc;
  int fp32;
  int fp64;
};
static const LanesPerSm kLanes[] = {
    {60, 64, 32},   {61, 128, 4},   {70, 64, 32},  {72, 64, 32},  {75, 64, 2},
    {80, 64, 32},   {86, 128, 2},   {87, 128, 2},  {89, 128, 2},  {90, 128, 64},
    {100, 128, -1}, {103, 128, -1}, {120, 128, 2}, {121, 128, 2},
};

int main() {
  int dev = 0;
  CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, dev));
  int sm_clock_khz = 0, mem_clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev));
  CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));

  int cc = p.major * 10 + p.minor;
  int fp32_lanes = -1, fp64_lanes = -1;
  for (const auto& l : kLanes) {
    if (l.cc == cc) {
      fp32_lanes = l.fp32;
      fp64_lanes = l.fp64;
    }
  }

  printf("Device %d: %s (compute capability %d.%d)\n", dev, p.name, p.major, p.minor);
  printf("  SMs                  %d\n", p.multiProcessorCount);
  printf("  Max SM clock         %.0f MHz\n", sm_clock_khz / 1e3);
  printf("  Memory clock         %.0f MHz, bus %d bits\n", mem_clock_khz / 1e3, p.memoryBusWidth);
  printf("  Global memory        %.1f GiB\n", p.totalGlobalMem / double(1 << 30));
  printf("  L2 cache             %.1f MiB\n", p.l2CacheSize / double(1 << 20));
  printf("  Shared mem per SM    %zu KiB (per block opt-in max %zu KiB)\n",
         p.sharedMemPerMultiprocessor >> 10, p.sharedMemPerBlockOptin >> 10);
  printf("  Registers per SM     %d\n", p.regsPerMultiprocessor);
  printf("  Max threads per SM   %d\n", p.maxThreadsPerMultiProcessor);
  printf("  FP32/FP64 lanes/SM   %d / %d (-1 = unknown for this arch)\n", fp32_lanes, fp64_lanes);

  // An FMA counts as 2 FLOPs. DRAM is double data rate, hence the factor of 2 on the clock.
  double clk_hz = sm_clock_khz * 1e3;
  double dram_bw = 2.0 * mem_clock_khz * 1e3 * p.memoryBusWidth / 8.0;
  printf(
      "\nTheoretical peaks (at max SM clock; sustained clocks may be lower under power limits):\n");
  printf("  DRAM bandwidth       %.1f GB/s\n", dram_bw / 1e9);
  printf("RESULT name %s\n", p.name);
  printf("RESULT cc %d.%d\n", p.major, p.minor);
  printf("RESULT sms %d\n", p.multiProcessorCount);
  printf("RESULT sm_clock_mhz %.0f\n", sm_clock_khz / 1e3);
  printf("RESULT l2_bytes %d\n", p.l2CacheSize);
  printf("RESULT theoretical_dram_gbs %.1f\n", dram_bw / 1e9);
  if (fp32_lanes > 0) {
    double fp32 = p.multiProcessorCount * fp32_lanes * 2.0 * clk_hz;
    printf("  FP32 (CUDA cores)    %.1f TFLOP/s\n", fp32 / 1e12);
    printf("RESULT theoretical_fp32_gflops %.1f\n", fp32 / 1e9);
  }
  if (fp64_lanes > 0) {
    double fp64 = p.multiProcessorCount * fp64_lanes * 2.0 * clk_hz;
    printf("  FP64 (CUDA cores)    %.1f TFLOP/s\n", fp64 / 1e12);
    printf("RESULT theoretical_fp64_gflops %.1f\n", fp64 / 1e9);
  }
  printf(
      "  Tensor-core peaks are not derivable from device properties; see the datasheet or the\n"
      "  measured cuBLAS numbers from bench_flops.\n");
  return 0;
}
