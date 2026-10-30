# Optimizing matmul

A three-hour-ish exercise in optimizing a matmul kernel with AI help. The user calls the algorithmic
shots; the AI runs the experiments. Everything runs on a single GPU.

## The problem

Multiply two `N x N` bf16 matrices (fp32 accumulate, bf16 output) with the naive `O(N^3)` algorithm
(no Strassen, no
[AlphaTensor](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/)-style
tricks): `N^3` multiplications and `N^3` additions, so `FLOPs = 2 * N^3`. The target is
`N = 8192 = 2**13`, i.e. `2^40 = 1.0995e12` FLOPs (about 1.1 TFLOP) per matmul.

## The machine

GPU 2 of 4, selected with `CUDA_VISIBLE_DEVICES=2` (set by `.env`).

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 PCIe, compute capability 9.0 (`sm_90`), 114 SMs, 1755 MHz max SM clock |
| Memory | 80 GB HBM2e, 5120-bit bus, 1593 MHz |
| Power limit | 310 W (default and maximum) |
| OS / CPU | Ubuntu 22.04.5, kernel 5.15.0-176, AMD EPYC 7543, 1 TiB RAM |
| Driver / CUDA | 545.23.08 / toolkit 12.8 (`nvcc` 12.8.93, with `ncu` and `nsys`) |

`nvcc` is not on `PATH` by default. Copy `.env.example` to `.env` (git-ignored, since the CUDA
version is per host) and `source .env`.

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
core ceiling of about 756 TFLOPS (clocks drop under load, since the card is capped at 310 W). A
kernel without tensor cores (`mma` / WGMMA) is capped near 51 TFLOPS.

```
time at N = 8192: 1.0995e12 / 756.4e12 = 1.45 ms (datasheet boost), / 819.5e12 = 1.34 ms (max clock)
```

Benchmarks use random inputs and warm up for about 2 s so clocks and power settle. They then time
many back-to-back calls individually and report the min and the median.

**cuBLAS reference** (`cublasGemmEx`, bf16 in, fp32 accumulate, bf16 out, `N = 8192`, 100 calls):
min 2.72-2.82 ms, median 2.72-2.92 ms across runs, i.e. 376-404 TFLOPS, about 50-53% of the
ceiling. This is the bar to approach. Under sustained load the card sits at its power cap, which is
why even cuBLAS falls well short of the ceiling. Times land on discrete steps (2.725, 2.772, 2.820,
2.923 ms) as the clock settles at a different level from run to run, so compare implementations by
running them back to back, more than once.

## Notes

> NOTE: This is the PCIe H100, not SXM5: 114 SMs (SXM5 has 132) and a lower power cap, so published
> SXM5 numbers overstate it.

> NOTE: The headline 1979 TFLOPS bf16 figure is SXM5 *with 2:4 structured sparsity*. This kernel is
> dense; its ceiling here is about 756 TFLOPS.

> NOTE: Benchmark on random data after a warmup. All-zero inputs draw less power, so cuBLAS runs
> at about 650 TFLOPS (1.69 ms) on them. Without a warmup, the fastest call is a 2.04 ms burst
> before the power cap kicks in. Both overstate sustained performance by 1.3-1.6x.

> NOTE: `nvidia-smi` reports CUDA 12.3 (the driver's maximum) but the toolkit is 12.8. This works via
> minor-version compatibility; check it first if a launch fails with a driver/runtime mismatch.

> NOTE: `ncu` may fail with `ERR_NVGPUCTRPERM` if the driver restricts performance counters to root.

> NOTE: The "datasheet boost" clock (1.620 GHz) is not measured. It's implied by NVIDIA's datasheet
> peak of about 756 TFLOPS dense bf16: 756.4e12 / (114 SMs * 4096 FLOP/clk/SM) = 1.620 GHz. The
> driver reports a higher max SM clock (1.755 GHz, hence 819.5 TFLOPS). Under sustained load, the
> 310 W power cap pulls the clock below both, so cuBLAS gets about 400 TFLOPS.
