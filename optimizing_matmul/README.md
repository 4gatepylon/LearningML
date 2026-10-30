# Optimizing matmul

A three-hour-ish exercise in optimizing a matmul kernel with AI help. The user calls all the
algorithmic shots and the AI runs the experiments. Everything runs on a single GPU (see
[The GPU](#the-gpu)).

## The problem

The kernel multiplies two `N x N` matrices, with `N` a power of two, using the naive algorithm
(no Strassen, no [AlphaTensor](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/)-style
tricks). The algorithm is `O(N^3)`: `N^3` multiplications and `N^3` additions, so

```
FLOPs = 2 * N^3
```

Inputs are bf16 (accumulation in fp32). The target size is `N = 8192 = 2**13`:

```
2 * 8192^3 = 2^40 = 1.0995e12 FLOPs   (about 1.1 TFLOP per matmul)
```

## The GPU

The GPU is device 2 of the 4 on the host, selected with `CUDA_VISIBLE_DEVICES=2` (set by `.env`;
the GPU then appears as device 0). Values come from `nvidia-smi`, `cudaGetDeviceProperties` and
`nvcc`.

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 PCIe, Hopper, compute capability 9.0 (`sm_90`) |
| SMs | 114 |
| Max SM clock | 1755 MHz |
| Memory | 80 GB HBM2e, 5120-bit bus, 1593 MHz memory clock |
| L2 cache | 50 MiB |
| Shared memory per SM | 228 KiB |
| Registers per SM | 65536 x 32-bit |

## The system

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-176-generic |
| CPU | AMD EPYC 7543, 1 TiB RAM |
| NVIDIA driver | 545.23.08 |
| CUDA toolkit | 12.8 (`nvcc` 12.8.93) |

Profilers `ncu` and `nsys` ship with the toolkit.

### Setup

`nvcc` is not on `PATH` by default. Copy `.env.example` to `.env` (git-ignored, because the CUDA
version is per host) and source it:

```
cp .env.example .env
source .env
nvcc --version
```

## What the hardware can do

**Tensor cores.** An SM has 4 tensor cores that together perform 4096 dense bf16 FLOPs per clock
(2048 FMAs per clock). With the values above:

```
114 SMs * 4096 FLOP/clk/SM * 1.755 GHz = 819.5 TFLOPS   (at the max SM clock)
114 SMs * 4096 FLOP/clk/SM * 1.620 GHz = 756.4 TFLOPS   (at the 1.62 GHz boost clock implied by the datasheet)
```

Under load the clock drops below the max, so about 756 TFLOPS is the realistic dense bf16 ceiling.

**fp32 without tensor cores.** 114 SMs * 128 FP32 lanes * 2 FLOP (FMA) * 1.755 GHz = 51.2 TFLOPS,
about 16x below the tensor cores. A kernel that does not use tensor cores (`mma` / WGMMA) cannot get
close to the bf16 ceiling.

**Memory bandwidth.** `5120 bit / 8 * 1593 MHz * 2 (DDR) = 2.04 TB/s`.

**Compute- or memory-bound.** The ridge point is `756.4e12 / 2.039e12 = 371 FLOP/byte`. A matmul
that reads `A` and `B` and writes `C` once each has arithmetic intensity

```
2N^3 FLOP / (3 * N^2 * 2 bytes) = N / 3 FLOP/byte  ->  8192 / 3 = 2731 FLOP/byte
```

That is about 7x above the ridge point, so a good kernel is compute-bound. The three bf16 matrices
total 3 * 8192^2 * 2 B = 403 MB.

**Time budget at N = 8192.**

```
1.0995e12 FLOP / 756.4e12 FLOP/s = 1.45 ms   (realistic ceiling)
1.0995e12 FLOP / 819.5e12 FLOP/s = 1.34 ms   (max-clock ceiling)
```

Benchmarks loop many iterations rather than timing one call.

**Measured reference (cuBLAS).** `cublasGemmEx`, bf16 inputs, fp32 accumulate, `N = 8192`, 100
iterations, gave 1.68-1.72 ms, i.e. 640-653 TFLOPS, about 85% of the 756 TFLOPS ceiling. This is the
bar to approach with a hand-written kernel.

## Notes

> NOTE: This is the PCIe H100, not the SXM5 part. It has 114 SMs (SXM5 has 132) and a lower power
> cap, so published SXM5 numbers overstate it.

> NOTE: The headline 1979 TFLOPS bf16 figure for the H100 is the SXM5 number *with 2:4 structured
> sparsity*. This kernel is dense, and dense bf16 on this card tops out around 756 TFLOPS (see
> above), so 1979 TFLOPS is not a valid target.

> NOTE: `nvidia-smi` reports CUDA 12.3 (the driver's maximum) while the toolkit is 12.8. This works
> through CUDA minor-version compatibility, but if a launch fails with a driver/runtime mismatch,
> check this first.

> NOTE: `ncu` may need access to the GPU performance counters, which the driver can restrict to
> root (`ERR_NVGPUCTRPERM`).
