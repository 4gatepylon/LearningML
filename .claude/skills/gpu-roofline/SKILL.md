---
name: gpu-roofline
description: Measure the current GPU's peak compute (FP32/FP64/FP16, TF32/BF16/FP16 tensor cores) and memory bandwidth (DRAM, L2), and place a CUDA kernel on a roofline. Use when the user asks about GPU specs, peak FLOPs or bandwidth, arithmetic intensity, whether a kernel is memory- or compute-bound, how close a kernel is to peak, or wants a roofline plot.
---

# GPU roofline

Tools live in `scripts/` next to this file. They need `nvcc` on `PATH`. In this repo, source a
kernel folder's env file first, e.g. `source optimizing_matmul/.env` (copied from the committed
`.env.example`, which also sets `CUDA_VISIBLE_DEVICES`).

**Pick the GPU with `CUDA_VISIBLE_DEVICES`** on every command. Check `nvidia-smi` for an idle
device first, since other jobs on the same GPU distort the numbers. On the `align-4` host the user
wants device 2 (`CUDA_VISIBLE_DEVICES=2`).

## 1. Get the peaks (once per GPU)

```
CUDA_VISIBLE_DEVICES=2 python3 scripts/measure_peaks.py      # ~1 min; --only bandwidth|flops|specs
```

This builds the benchmarks into `.build/` (with `-arch=native`) and prints and caches results in
`.cache/peaks-<host>-<gpu>.json`. Reuse the cache if it exists and is for the right GPU; just
`cat` it. The benchmarks are:

- `gpu_specs.cu`: device properties plus theoretical FP32/FP64 and DRAM peaks from SM count,
  clocks and bus width.
- `bench_bandwidth.cu`: DRAM read, write and copy (kernel and `cudaMemcpy`), plus L2 read on a
  buffer of L2/4. Arg: DRAM buffer MiB (default 1024).
- `bench_flops.cu`: FP32, FP64 and FP16 (half2) FMA loops on CUDA cores, and cuBLAS 8192³ GEMMs
  for TF32, BF16, FP16 (tensor cores) and FP32 SGEMM. Arg: GEMM size.

Measured values are the realistic roofs; theoretical ones are upper bounds at max clock.

## 2. Measure the kernel

The roofline needs three numbers for the kernel:

- **FLOPs**: count analytically from the algorithm (an FMA is 2 FLOPs). Count useful work, not
  instructions; e.g. an M×N×K matmul is `2*M*N*K`.
- **DRAM bytes**: the minimum bytes the kernel must move to and from global memory (each input
  read once, each output written once). Mention to the user that the real traffic can be higher
  when caching is poor; that gap is often the point of the exercise.
- **Time**: time the kernel alone with `cudaEvent`s, after a warmup launch, taking the best or
  median of several runs. Exclude allocation, `cudaMemcpy` H2D/D2H and the CPU reference.

This repo's `cuda_book/` is a learning project: the user writes the kernels. Adding timing
boilerplate is fine; don't rewrite or optimize the kernel itself unless asked.

## 3. Place it on the roofline

```
python3 scripts/roofline.py --kernel NAME:FLOPS:BYTES:MS [--kernel ...] \
    [--compute fp32|fp64|fp16|tf32_tc|bf16_tc|fp16_tc|fp32_sgemm] [--memory dram|l2] \
    [--theoretical] [--plot out.png]
```

It prints arithmetic intensity, achieved GFLOP/s and GB/s, the roof at that intensity, % of roof
and whether the kernel sits left (memory-bound) or right (compute-bound) of the ridge point. Pick
`--compute` to match the kernel's math (CUDA-core FP32 is `fp32`; tensor-core kernels use `*_tc`).
Put the plot in the scratchpad unless the user wants it saved, and show it to them.

## Nsight Compute

`ncu --set roofline` measures FLOPs and bytes from hardware counters and draws a hierarchical
roofline. Before relying on it, check that it runs with a quick
`ncu --launch-count 1 --metrics dram__bytes_read.sum <binary>`. By default `ncu` locks clocks to
base (`--clock-control none` disables that), so its timings are slower than normal runs.

## Reference: NVIDIA H100 PCIe on align-4, device 2 (2026-10-30)

114 SMs, 1755 MHz, 50 MiB L2, power limit 310 W. Theoretical: 51.2 TFLOP/s FP32,
25.6 TFLOP/s FP64, 2039 GB/s DRAM. Measured: DRAM read 1936, write 1983, copy 1786 GB/s; L2
read about 5100 GB/s; FP32 49.7, FP64 25.5, FP16 (half2) 101 TFLOP/s; cuBLAS TF32 274, BF16 506,
FP16 474, SGEMM 38 TFLOP/s. FP32 ridge ≈ 25 FLOP/byte. Use these to sanity-check a fresh run.

The tensor-core GEMM numbers depend heavily on the data and on run length, because the card hits
its 310 W cap. For a bf16 8192³ cuBLAS GEMM averaged over 100 back-to-back calls: about 650 TFLOP/s
with all-zero inputs, but only about 400–420 TFLOP/s with random inputs. `bench_flops` (random
data, best of 10) reports about 506. Benchmark kernels on random data and compare against cuBLAS
run the same way; zero-filled inputs overstate throughput by about 1.6×.
