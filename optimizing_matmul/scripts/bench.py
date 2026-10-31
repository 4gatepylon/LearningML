"""`bench.py` times kernels over trials of batched calls and prints a table of time statistics.

`bench.py` is more than a shim around `.build/bench`. The C++ binary does the timing, since the
kernels and CUDA events live in C++. `bench.py` owns the command line (flags, defaults and
validation) and turns the binary's raw times into the statistics table, which is shorter to write
in Python.

Interface:
    -N: Matrix size (default 8192, at least 1).
    --warmup-iteration-count: Untimed calls per kernel before the trials (default 1, at least 0).
    --trial-count: Timed batches per kernel, interleaved across kernels (default 3, at least 1).
    --batch-size: Back-to-back calls per timed batch (default 100, at least 1).
    --kernel: A kernel to time; repeat it for several (default: every kernel).
    --list-kernels: Print the kernel names and exit.

Example:
    $ make bench BENCH_ARGS="-N 4096 --batch-size 1000 --kernel cublas"

Must know:
    Gotchas:
        - `bench.py` reads the roofline peaks from `.env`, so `source .env` must run first.
        - TFLOPS and %roof come from the median time.
        - Times are per call: each batch's time divided by `--batch-size`.
        - Calls in a batch run back to back, so long batches run at the 310 W cap's clocks.
"""

import os
import statistics

import click
from bench_binary import check_kernel_names, list_kernel_names, run_bench

_DEFAULT_N: int = 8192
_DEFAULT_WARMUP_ITERATION_COUNT: int = 1
_DEFAULT_TRIAL_COUNT: int = 3
_DEFAULT_BATCH_SIZE: int = 100
_MIN_N: int = 1
_MIN_WARMUP_ITERATION_COUNT: int = 0
_MIN_TRIAL_COUNT: int = 1
_MIN_BATCH_SIZE: int = 1
_FLOP_PER_MULTIPLY_ADD: int = 2
_DRAM_CROSSING_MATRIX_COUNT: int = 3
_BYTES_PER_BF16: int = 2
_MS_PER_S: float = 1e3
_FLOP_PER_TFLOP: float = 1e12
_ROW_FORMAT: str = "{:<8}{:>6}{:>8}{:>7}{:>10}{:>11}{:>10}{:>10}{:>9}{:>10}{:>7}"


@click.command(
    help="Times kernels over trials of batched calls and prints time statistics.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-N",
    "N",
    type=click.IntRange(min=_MIN_N),
    default=_DEFAULT_N,
    show_default=True,
    help="Matrix size (N x N).",
)
@click.option(
    "--warmup-iteration-count",
    type=click.IntRange(min=_MIN_WARMUP_ITERATION_COUNT),
    default=_DEFAULT_WARMUP_ITERATION_COUNT,
    show_default=True,
    help="Untimed calls per kernel before the trials.",
)
@click.option(
    "--trial-count",
    type=click.IntRange(min=_MIN_TRIAL_COUNT),
    default=_DEFAULT_TRIAL_COUNT,
    show_default=True,
    help="Timed batches per kernel, interleaved across kernels.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=_MIN_BATCH_SIZE),
    default=_DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Back-to-back calls per timed batch.",
)
@click.option(
    "--kernel",
    "kernel_names",
    multiple=True,
    help="A kernel to time; repeat it for several. Default: every kernel.",
)
@click.option("--list-kernels", "is_listing_kernels", is_flag=True, help="Print the kernel names.")
def main(
    N: int,
    warmup_iteration_count: int,
    trial_count: int,
    batch_size: int,
    kernel_names: tuple[str, ...],
    is_listing_kernels: bool,
) -> None:
    """`main` runs the benchmark binary and prints one row of time statistics per kernel.

    Args:
        N: Matrix size.
        warmup_iteration_count: Untimed calls per kernel before the trials.
        trial_count: Timed batches per kernel.
        batch_size: Back-to-back calls per timed batch.
        kernel_names: Kernels to time; empty times every kernel.
        is_listing_kernels: Whether to print the kernel names and exit instead.

    Example:
        $ uv run scripts/bench.py -N 4096 --batch-size 1000

    Must know:
        Limitations:
            - The roofline assumes that A, B and C each cross DRAM exactly once.
    """
    if is_listing_kernels:
        click.echo("\n".join(list_kernel_names()))
        return
    check_kernel_names(kernel_names)
    peak_dense_bf16_tflops_according_to_nvidia_spec: float = float(
        os.environ["PEAK_DENSE_BF16_TFLOPS_ACCORDING_TO_NVIDIA_SPEC"]
    )
    mem_bandwidth_tbytes_per_s_according_to_nvidia_spec: float = float(
        os.environ["MEM_BANDWIDTH_TBYTES_PER_S_ACCORDING_TO_NVIDIA_SPEC"]
    )
    matmul_flop: int = _FLOP_PER_MULTIPLY_ADD * N**3
    matmul_traffic_bytes: int = _DRAM_CROSSING_MATRIX_COUNT * N**2 * _BYTES_PER_BF16
    intensity_flop_per_byte: float = matmul_flop / matmul_traffic_bytes
    roofline_tflops_according_to_nvidia_spec: float = min(
        peak_dense_bf16_tflops_according_to_nvidia_spec,
        intensity_flop_per_byte * mem_bandwidth_tbytes_per_s_according_to_nvidia_spec,
    )

    kernel_name_to_ms_per_call_samples_according_to_cuda_events: dict[str, list[float]] = run_bench(
        N, warmup_iteration_count, trial_count, batch_size, kernel_names
    )

    print(
        _ROW_FORMAT.format(
            "kernel",
            "N",
            "trials",
            "batch",
            "min ms",
            "median ms",
            "mean ms",
            "max ms",
            "TFLOPS",
            "roofline",
            "%roof",
        )
    )
    for kernel_name in kernel_name_to_ms_per_call_samples_according_to_cuda_events:
        ms_per_call_samples_according_to_cuda_events: list[float] = (
            kernel_name_to_ms_per_call_samples_according_to_cuda_events[kernel_name]
        )
        median_ms_per_call_according_to_cuda_events: float = statistics.median(
            ms_per_call_samples_according_to_cuda_events
        )
        achieved_tflops_according_to_cuda_events: float = (
            matmul_flop
            / (median_ms_per_call_according_to_cuda_events / _MS_PER_S)
            / _FLOP_PER_TFLOP
        )
        roofline_frac: float = (
            achieved_tflops_according_to_cuda_events / roofline_tflops_according_to_nvidia_spec
        )
        print(
            _ROW_FORMAT.format(
                kernel_name,
                N,
                len(ms_per_call_samples_according_to_cuda_events),
                batch_size,
                f"{min(ms_per_call_samples_according_to_cuda_events):.3f}",
                f"{median_ms_per_call_according_to_cuda_events:.3f}",
                f"{statistics.mean(ms_per_call_samples_according_to_cuda_events):.3f}",
                f"{max(ms_per_call_samples_according_to_cuda_events):.3f}",
                f"{achieved_tflops_according_to_cuda_events:.1f}",
                f"{roofline_tflops_according_to_nvidia_spec:.1f}",
                f"{roofline_frac:.0%}",
            )
        )


if __name__ == "__main__":
    main()
