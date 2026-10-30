# Optimizing matmul

A three-hour-ish exercise in optimizing a matmul kernel with AI help. The user calls the algorithmic
shots; the AI runs the experiments. Everything runs on a single GPU.

## The problem

Multiply two `N x N` bf16 matrices (fp32 accumulate, bf16 output) with the naive `O(N^3)` algorithm (no Strassen,
no [AlphaTensor](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/)-style
tricks): `N^3` multiplications and `N^3` additions, so `FLOPs = 2 * N^3`. The target is
`N = 8192 = 2**13`, i.e. `2^40 = 1.0995e12` FLOPs (about 1.1 TFLOP) per matmul.

## The machine

GPU 2 of 4, selected with `CUDA_VISIBLE_DEVICES=2` (set by `.env`).

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 PCIe, compute capability 9.0 (`sm_90`), 114 SMs, 1755 MHz max SM clock |
| Memory | 80 GB HBM2e, 5120-bit bus, 1593 MHz |
| OS / CPU | Ubuntu 22.04.5, kernel 5.15.0-176, AMD EPYC 7543, 1 TiB RAM |
| Driver / CUDA | 545.23.08 / toolkit 12.8 (`nvcc` 12.8.93, with `ncu` and `nsys`) |

`nvcc` is not on `PATH` by default. Copy `.env.example` to `.env` (git-ignored, since the CUDA
version is per host) and `source .env`.

## Testing and benchmarking

Each implementation is a `<name>.cu` file that defines
`void <name>(const bf16 *A, const bf16 *B, bf16 *C, int n)` (row-major device arrays, `C = A * B`,
launched on the default stream) and ends with:

```cpp
MATMUL_TESTS(<Suite>, <name>)
int main(int argc, char **argv) { return run_matmul("<name>", <name>, argc, argv); }
```

`harness.cuh` supplies the rest, and `cublas.cu` is the example. `make run-<name>` builds the file
into `.build/`, runs its GoogleTest tests, and benchmarks it only if they all pass:

- **Correctness**: the output must match cuBLAS on every entry at `N` = 128, 1024 and 8192, within
  `2^-7` relative error plus `0.01` absolute (bf16 output rounding). `C` starts as NaNs, so unwritten
  entries fail. A separate test checks cuBLAS itself against fp64 CPU dot products on sampled entries.
- **Timing**: random inputs in `[-1, 1)`, then a 2 s warmup so clocks and power settle at their
  sustained level, then 100 back-to-back calls, each timed with its own CUDA events. It reports the
  min and median.

```
make run-cublas                                  # test, then benchmark at N = 8192
make run-cublas MATMUL_ARGS='--n 4096 --tries 50 --warmup-ms 1000'
make test-cublas MATMUL_ARGS=--gtest_filter=Cublas.*   # tests only, optionally filtered
make fmt && make tidy                            # clang-format / clang-tidy (cuda_book's configs)
```

## What the hardware can do

```
tensor cores: 114 SMs * 4096 dense bf16 FLOP/clk/SM * 1.755 GHz = 819.5 TFLOPS (max clock)
                                                    * 1.620 GHz = 756.4 TFLOPS (datasheet boost)
fp32 CUDA cores: 114 * 128 lanes * 2 FLOP * 1.755 GHz           =  51.2 TFLOPS
memory bandwidth: 5120 bit / 8 * 1593 MHz * 2 (DDR)             =   2.04 TB/s
ridge point: 756.4e12 / 2.039e12                                =    371 FLOP/byte
matmul intensity: 2N^3 / (3 * N^2 * 2 B) = N / 3 = 8192 / 3     =   2731 FLOP/byte
```

The matmul is compute-bound (intensity ~7x the ridge point), so the target is the dense bf16 tensor
core ceiling of about 756 TFLOPS (clocks drop under load). A kernel without tensor cores (`mma` /
WGMMA) is capped near 51 TFLOPS.

```
time at N = 8192: 1.0995e12 / 756.4e12 = 1.45 ms (datasheet boost), / 819.5e12 = 1.34 ms (max clock)
```

**cuBLAS reference** (`make run-cublas`): min 2.72-2.82 ms (390-404 TFLOPS), median 2.72-2.92 ms
(376-404 TFLOPS) across runs, about 53% of the datasheet ceiling. Under sustained load the card sits
at its 310 W power cap, which is why it falls well short of the ceiling. This is the bar to approach.

Times land on discrete steps (2.725, 2.772, 2.820, 2.923 ms) as the clock settles at a different
level from run to run, so compare implementations by running them back to back, more than once.

## Notes

> NOTE: This is the PCIe H100, not SXM5: 114 SMs (SXM5 has 132) and a lower power cap, so published
> SXM5 numbers overstate it.

> NOTE: The headline 1979 TFLOPS bf16 figure is SXM5 *with 2:4 structured sparsity*. This kernel is
> dense; its ceiling here is about 756 TFLOPS.

> NOTE: Benchmark on random data after a warmup. All-zero inputs draw less power and run at about
> 650 TFLOPS (1.69 ms), and without a warmup the fastest call is a 2.04 ms burst before the power
> cap kicks in. Both overstate sustained performance by 1.3-1.6x.

> NOTE: `nvidia-smi` reports CUDA 12.3 (the driver's maximum) but the toolkit is 12.8. This works via
> minor-version compatibility; check it first if a launch fails with a driver/runtime mismatch.

> NOTE: `ncu` may fail with `ERR_NVGPUCTRPERM` if the driver restricts performance counters to root.
