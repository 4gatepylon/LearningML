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
| Memory | 80 GB HBM2e, 5120-bit bus, 1593 MHz |
| OS / CPU | Ubuntu 22.04.5, kernel 5.15.0-176, AMD EPYC 7543, 1 TiB RAM |
| Driver / CUDA | 545.23.08 / toolkit 12.8 (`nvcc` 12.8.93, with `ncu` and `nsys`) |

`nvcc` is not on `PATH` by default. Copy `.env.example` to `.env` (git-ignored, since the CUDA
version is per host) and `source .env`.

## Code layout

`common/` holds shared headers, `kernels/` one matmul implementation per `.cu` file, and `tests/`
tests of the shared code. Every `.cu` file is a standalone GoogleTest executable, built into
`.build/` (git-ignored) by the `Makefile`:

```
make test-<name>   # build and run kernels/<name>.cu or tests/<name>.cu
make test          # every kernel and test
make fmt           # clang-format, configured like cuda_book (Google style, 100 columns)
make tidy          # clang-tidy, configured like cuda_book
```

`make fmt` and `make tidy` use the tools from `cuda_book`'s uv environment (run `uv sync` there).

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

Benchmarks loop many iterations rather than timing one call.

**cuBLAS reference** (`cublasGemmEx`, bf16, fp32 accumulate, `N = 8192`, 100 iterations): 1.68-1.72 ms,
640-653 TFLOPS, about 85% of the ceiling. This is the bar to approach.

## Notes

> NOTE: This is the PCIe H100, not SXM5: 114 SMs (SXM5 has 132) and a lower power cap, so published
> SXM5 numbers overstate it.

> NOTE: The headline 1979 TFLOPS bf16 figure is SXM5 *with 2:4 structured sparsity*. This kernel is
> dense; its ceiling here is about 756 TFLOPS.

> NOTE: `nvidia-smi` reports CUDA 12.3 (the driver's maximum) but the toolkit is 12.8. This works via
> minor-version compatibility; check it first if a launch fails with a driver/runtime mismatch.

> NOTE: `ncu` may fail with `ERR_NVGPUCTRPERM` if the driver restricts performance counters to root.
