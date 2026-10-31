# Optimizing matmul

A three-hour-ish exercise in optimizing a matmul kernel with AI help. The user calls the algorithmic
shots; the AI runs the experiments. Everything runs on a single GPU.

## The problem

Multiply two `N x N` bf16 matrices (fp32 accumulate) with the naive `O(N^3)` algorithm (no Strassen,
no [AlphaTensor](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/)-style
tricks): `N^3` multiplications and `N^3` additions, so `FLOPs = 2 * N^3`. The target is
`N = 8192 = 2**13`, i.e. `2^40 = 1.0995e12` FLOPs (about 1.1 TFLOP) per matmul.

## The machine

GPU 2 of 4, selected with `CUDA_VISIBLE_DEVICES=2` (set by `.env`).

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 PCIe, compute capability 9.0 (`sm_90`), 114 SMs, 1755 MHz max SM clock |
| Power limit | 310 W (the default) |
| Memory | 80 GB HBM2e, 5120-bit bus, 1593 MHz |
| OS / CPU | Ubuntu 22.04.5, kernel 5.15.0-176, AMD EPYC 7543, 1 TiB RAM |
| Driver / CUDA | 545.23.08 / toolkit 12.8 (`nvcc` 12.8.93, with `ncu` and `nsys`) |

`nvcc` is not on `PATH` by default. Copy `.env.example` to `.env` (git-ignored, since the CUDA
version is per host) and `source .env`.

## Running it

```bash
cp .env.example .env && source .env                # nvcc on PATH, GPU 2
make test                                          # build and run the GoogleTest tests
make bench-kernels                                 # time each kernel at N = 8192
make bench-kernels BENCH_ARGS="-N 4096"            # pass flags (`uv run scripts/bench.py -h`)
make scan-batch-sizes SCAN_ARGS="--kernel cublas"  # time batch sizes; fit the fixed cost per batch
make fmt && make tidy                              # format and lint
```

Every kernel in `KERNELS` (`common/matmul.h`) is tested and benchmarked.

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
time at N = 8192: 1.0995e12 / 756.4e12 = 1.45 ms (realistic), / 819.5e12 = 1.34 ms (max clock)
```

`make bench-kernels` times 3 trials per kernel, interleaved across kernels. Each trial is a batch of
100 back-to-back calls timed with one pair of CUDA events, and its time is divided by 100. Inputs
are normal random values (mean 0, sigma 1), never all zeros, which cuBLAS multiplies faster.

**cuBLAS reference** (`cublasGemmEx`, bf16, fp32 accumulate, `N = 8192`, `make bench-kernels`):
median 2.62 ms per call, 420 TFLOPS, about 55% of the 756 TFLOPS ceiling. This is the bar to
approach. A batch runs long enough for the 310 W power cap to lower SM clocks. A lone call on idle
clocks takes about 2.3 ms.

## Notes

> NOTE: This is the PCIe H100, not SXM5: 114 SMs (SXM5 has 132) and a lower power cap, so published
> SXM5 numbers overstate it.

> NOTE: The headline 1979 TFLOPS bf16 figure is SXM5 *with 2:4 structured sparsity*. This kernel is
> dense; its ceiling here is about 756 TFLOPS.

> NOTE: `nvidia-smi` reports CUDA 12.3 (the driver's maximum) but the toolkit is 12.8. This works via
> minor-version compatibility; check it first if a launch fails with a driver/runtime mismatch.

> NOTE: `ncu` may fail with `ERR_NVGPUCTRPERM` if the driver restricts performance counters to root.
