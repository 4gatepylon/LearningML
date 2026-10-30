// Measures achievable compute throughput: FP32, FP64 and FP16 (half2) FMA on CUDA cores, and
// TF32 / BF16 / FP16 tensor-core GEMMs through cuBLAS. Reports the best of several timed runs.
// Output lines starting with "RESULT" are parsed by measure_peaks.py.
//
// Usage: bench_flops [gemm_size=8192]   (build with -lcublas)
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

#define CHECK_CUBLAS(call)                                                  \
  do {                                                                      \
    cublasStatus_t st_ = (call);                                            \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "%s:%d: cuBLAS error %d\n", __FILE__, __LINE__, st_); \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

constexpr int kThreads = 256;
constexpr int kWarmup = 3;
constexpr int kIters = 10;
constexpr int kChains = 8;   // independent FMA chains per thread, to hide FMA latency
constexpr int kLoop = 4096;  // FMAs per chain

// Each thread runs kChains independent chains of kLoop dependent FMAs (x = x * m + a).
// m and a come from kernel arguments so the compiler can't fold the loop away.
template <typename T>
__global__ void fma_kernel(T m, T a, T* sink) {
  T x[kChains];
#pragma unroll
  for (int c = 0; c < kChains; ++c) x[c] = T(threadIdx.x + c);
  for (int i = 0; i < kLoop; ++i) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) x[c] = x[c] * m + a;
  }
  T sum = T(0);
#pragma unroll
  for (int c = 0; c < kChains; ++c) sum += x[c];
  if (sum == T(-1.2345)) *sink = sum;  // never true for our data; keeps the FMAs alive
}

__global__ void hfma2_kernel(__half2 m, __half2 a, __half2* sink) {
  __half2 x[kChains];
#pragma unroll
  for (int c = 0; c < kChains; ++c) x[c] = __float2half2_rn(float(threadIdx.x % 7 + c));
  for (int i = 0; i < kLoop; ++i) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) x[c] = __hfma2(x[c], m, a);
  }
  __half2 sum = x[0];
#pragma unroll
  for (int c = 1; c < kChains; ++c) sum = __hadd2(sum, x[c]);
  if (__low2float(sum) == -1.2345f) *sink = sum;
}

// Fills x with pseudo-random values in [-1, 1). Random data matters for GEMMs: all-zero inputs
// draw less power, so clocks stay higher and the measured peak comes out unrealistically high.
template <typename T>
__global__ void fill_random(T* x, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned h = unsigned(i) * 2654435761u;
    h ^= h >> 16;
    x[i] = T(float(h & 0xffff) / 32768.f - 1.f);
  }
}

static void fill(void* x, size_t n, cudaDataType_t type) {
  if (type == CUDA_R_32F) fill_random<<<1024, 256>>>((float*)x, n);
  if (type == CUDA_R_16F) fill_random<<<1024, 256>>>((__half*)x, n);
  if (type == CUDA_R_16BF) fill_random<<<1024, 256>>>((__nv_bfloat16*)x, n);
  CHECK(cudaGetLastError());
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

static void report(const char* key, const char* label, double flops, double seconds) {
  double gflops = flops / seconds / 1e9;
  printf("  %-36s %10.1f GFLOP/s\n", label, gflops);
  printf("RESULT %s %.1f\n", key, gflops);
}

// One n x n x n GEMM, C = A * B, with the given storage and compute types.
static void bench_gemm(cublasHandle_t h, int n, cudaDataType_t ab_type, size_t ab_elem,
                       cudaDataType_t c_type, size_t c_elem, cublasComputeType_t compute,
                       const char* key, const char* label) {
  void *A, *B, *C;
  CHECK(cudaMalloc(&A, size_t(n) * n * ab_elem));
  CHECK(cudaMalloc(&B, size_t(n) * n * ab_elem));
  CHECK(cudaMalloc(&C, size_t(n) * n * c_elem));
  fill(A, size_t(n) * n, ab_type);
  fill(B, size_t(n) * n, ab_type);
  float alpha = 1.f, beta = 0.f;
  double s = best_seconds([&] {
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, ab_type, n, B,
                              ab_type, n, &beta, C, c_type, n, compute, CUBLAS_GEMM_DEFAULT));
  });
  report(key, label, 2.0 * n * n * n, s);
  CHECK(cudaFree(A));
  CHECK(cudaFree(B));
  CHECK(cudaFree(C));
}

int main(int argc, char** argv) {
  int gemm_n = argc > 1 ? atoi(argv[1]) : 8192;

  int dev = 0, sms = 0, max_threads = 0;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  // Full occupancy, many times over, so every SM stays busy for the whole launch.
  int grid = sms * (max_threads / kThreads) * 8;
  double threads = double(grid) * kThreads;
  double fma_flops = threads * kChains * kLoop * 2.0;

  void* sink = nullptr;
  CHECK(cudaMalloc(&sink, 16));

  printf("CUDA cores: grid %d x %d threads, %d chains x %d FMAs per thread\n", grid, kThreads,
         kChains, kLoop);
  report("fp32_gflops", "FP32 FMA", fma_flops, best_seconds([&] {
           fma_kernel<float><<<grid, kThreads>>>(0.999f, 0.001f, (float*)sink);
         }));
  report("fp64_gflops", "FP64 FMA", fma_flops, best_seconds([&] {
           fma_kernel<double><<<grid, kThreads>>>(0.999, 0.001, (double*)sink);
         }));
  // Each __hfma2 does two FMAs, so twice the FLOPs.
  report("fp16_gflops", "FP16 FMA (half2)", 2.0 * fma_flops, best_seconds([&] {
           hfma2_kernel<<<grid, kThreads>>>(__float2half2_rn(0.999f), __float2half2_rn(0.001f),
                                            (__half2*)sink);
         }));

  cublasHandle_t h;
  CHECK_CUBLAS(cublasCreate(&h));
  printf("Tensor cores (cuBLAS GEMM, %d x %d x %d):\n", gemm_n, gemm_n, gemm_n);
  bench_gemm(h, gemm_n, CUDA_R_32F, 4, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F_FAST_TF32,
             "tf32_tc_gflops", "TF32 GEMM (FP32 in/out)");
  bench_gemm(h, gemm_n, CUDA_R_16BF, 2, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, "bf16_tc_gflops",
             "BF16 GEMM (FP32 accumulate)");
  bench_gemm(h, gemm_n, CUDA_R_16F, 2, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, "fp16_tc_gflops",
             "FP16 GEMM (FP32 accumulate)");
  // Also measure plain FP32 SGEMM (no tensor cores) as a realistic FP32 ceiling for GEMM-like code.
  bench_gemm(h, gemm_n, CUDA_R_32F, 4, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, "fp32_sgemm_gflops",
             "FP32 SGEMM (no tensor cores)");
  CHECK_CUBLAS(cublasDestroy(h));
  CHECK(cudaFree(sink));
  return 0;
}
