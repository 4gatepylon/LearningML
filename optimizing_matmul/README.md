# Optimizing matmul

A three-hour-ish exercise in optimizing a matmul kernel with AI help. The user calls all the
algorithmic shots and the AI runs the experiments. Everything runs on a single H100 (GPU 2 of the
four on this host; see [The GPU](#the-gpu)).

## The problem

The kernel multiplies two `N x N` matrices, with `N` a power of two, using the naive algorithm
(no Strassen, no [AlphaTensor](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/)-style
tricks). The algorithm is `O(N^3)`: `N^3` multiplications and `N^3` additions, so

```
FLOPs = 2 * N^3
```

Inputs are bf16 (accumulation in fp32). The target size is `N = 8192 = 2**13`:

```
2 * 8192^3 = 2 * 2^39 = 2^40 = 1.0995e12 FLOPs   (about 1.1 TFLOP per matmul)
```

At `N = 1000` a matmul is 2 GFLOP; growing each axis by about 8x multiplies the work by 8^3 = 512x.

## The GPU

The numbers below are measured on this host (`nvidia-smi`, `cudaGetDeviceProperties`, `nvcc`) unless
marked otherwise.

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 **PCIe** (not SXM), 4 on the host, GPU 2 is used (PCI bus `00000000:C1:00.0`) |
| Architecture | Hopper, compute capability 9.0 (`sm_90`) |
| SMs | 114 (SXM5 has 132) |
| Max SM clock | 1755 MHz |
| Memory | 80 GB HBM2e (81559 MiB), 5120-bit bus, 1593 MHz memory clock |
| L2 cache | 50 MiB |
| Shared memory per SM | 228 KiB (227 KiB opt-in per block) |
| Registers per SM | 65536 x 32-bit |
| Max threads per SM | 2048 |
| Power limit | 310 W (min 200 W) |
| Topology | GPUs 2 and 3 share NUMA node 1 (CPU cores 32-63, 96-127); all GPU pairs connect over `SYS` (no NVLink) |

Select it with `CUDA_VISIBLE_DEVICES=2` (set by `.env`; the GPU then appears as device 0).

### System

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-176-generic |
| CPU | AMD EPYC 7543 (32 cores / 64 threads per socket, two sockets), 1 TiB RAM |
| NVIDIA driver | 545.23.08 (supports up to CUDA 12.3 per `nvidia-smi`) |
| CUDA toolkit | 12.8 (`nvcc` 12.8.93 at `/usr/local/cuda-12.8`); 12.x and 13.4 are also installed |

The driver advertises CUDA 12.3 while the toolkit is 12.8. This works because of CUDA minor-version
compatibility within the 12.x family, but PTX and features newer than the driver supports may not
load. If a kernel launch fails with a driver/runtime mismatch, check this first.

### Setup

`nvcc` is not on `PATH` by default. Copy `.env.example` to `.env` (git-ignored, because the CUDA
version is per host) and source it:

```
cp .env.example .env
source .env
nvcc --version
```

## What the hardware can do

The H100 numbers found online are usually for the SXM5 part, and several quote the figure *with 2:4
structured sparsity*. The 1979 TFLOPS bf16 figure is exactly that: SXM5 with sparsity. It is not
what this GPU delivers for a dense matmul. Derivation for the PCIe card on this host:

**Tensor core throughput.** A Hopper SM has 4 tensor cores, and together they perform 4096 dense
bf16 FLOPs per clock (2048 FMAs per clock, 512 per tensor core). Check against the published SXM5
dense figure: `132 SMs * 4096 FLOP/clk * 1.83 GHz = 989.4 TFLOPS`, which matches the 989 TFLOPS in
the datasheet.

**This card (dense bf16):**

```
114 SMs * 4096 FLOP/clk/SM * 1.755 GHz = 819.5 TFLOPS   (at max SM clock)
114 SMs * 4096 FLOP/clk/SM * 1.620 GHz = 756.4 TFLOPS   (the 756 TFLOPS PCIe datasheet value, implying a 1.62 GHz boost clock)
```

The measured SM count and max clock reproduce the datasheet number, so the datasheet matches the
hardware. With 2:4 sparsity the figures double (about 1513 TFLOPS), and 1979 TFLOPS is not reachable
on a PCIe H100 at all. Real sustained clocks fall below the max under load because of the 310 W cap,
so about 756 TFLOPS is the realistic ceiling for dense bf16.

**Non-tensor-core fp32 (for comparison):** 114 SMs * 128 FP32 lanes * 2 FLOP (FMA) * 1.755 GHz =
51.2 TFLOPS, about 16x below the tensor cores. A kernel that does not use tensor cores (`mma` /
WGMMA) cannot get close to the bf16 ceiling.

**Memory bandwidth:** `5120 bit / 8 * 1593 MHz * 2 (DDR) = 2.04 TB/s`.

**Compute- or memory-bound?** The ridge point is `756.4e12 / 2.039e12 = 371 FLOP/byte`. The
arithmetic intensity of a matmul that reads `A` and `B` and writes `C` once each is

```
2N^3 FLOP / (3 * N^2 * 2 bytes) = N / 3 FLOP/byte  ->  8192 / 3 = 2731 FLOP/byte
```

That is about 7x above the ridge point, so a good kernel is compute-bound; the three bf16 matrices
total only 3 * 8192^2 * 2 B = 403 MB and fit in the 80 GB of HBM trivially.

**Time budget at N = 8192:**

```
1.0995e12 FLOP / 756.4e12 FLOP/s = 1.45 ms   (realistic ceiling)
1.0995e12 FLOP / 819.5e12 FLOP/s = 1.34 ms   (max-clock ceiling)
```

So one matmul takes roughly 1.5 ms, not 1 second; reaching about a second would need around 1000x
more work (`N` of about 80000). Benchmarks loop many iterations rather than timing one call.

**Measured reference (cuBLAS).** `cublasGemmEx`, bf16 inputs, fp32 accumulate, `N = 8192`, 100
iterations on GPU 2, gave 1.68-1.72 ms, i.e. **640-653 TFLOPS**, which is about 85-86% of the 756
TFLOPS ceiling. This is the bar to beat or approach with a hand-written kernel.
