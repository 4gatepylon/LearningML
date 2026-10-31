"""`bench.py` times every kernel over several trials and prints a table of time statistics.

`bench.py` is more than a shim around `.build/bench`. The C++ binary does the timing, since the
kernels and CUDA events live in C++. `bench.py` owns the command line (flags, defaults and
validation) and turns the binary's raw times into the statistics table, which is shorter to write
in Python.

Interface:
    -N: Matrix size (default 8192, at least 1).
    --warmup-iteration-count: Untimed calls per kernel before the trials (default 1, at least 0).
    --trial-count: Timed calls per kernel, interleaved across kernels (default 20, at least 1).

Example:
    $ make bench BENCH_ARGS="-N 4096 --trial-count 50"

Must know:
    Gotchas:
        - `bench.py` reads the roofline peaks from `.env`, so `source .env` must run first.
        - TFLOPS and %roof come from the median time.
        - Each call is timed alone, so small-N times include launch overhead; this may change.
        - The times are burst times; under sustained load cuBLAS slows from ~2.3 to ~3.0 ms.
"""

import os
import statistics
import subprocess
from pathlib import Path

import click

_BENCH_BINARY_PATH: Path = Path(__file__).resolve().parent.parent / ".build" / "bench"
_DEFAULT_N: int = 8192
_DEFAULT_WARMUP_ITERATION_COUNT: int = 1
_DEFAULT_TRIAL_COUNT: int = 20
_MIN_N: int = 1
_MIN_WARMUP_ITERATION_COUNT: int = 0
_MIN_TRIAL_COUNT: int = 1
_FLOP_PER_MULTIPLY_ADD: int = 2
_DRAM_CROSSING_MATRIX_COUNT: int = 3
_BYTES_PER_BF16: int = 2
_MS_PER_S: float = 1e3
_FLOP_PER_TFLOP: float = 1e12
_ROW_FORMAT: str = "{:<8}{:>6}{:>8}{:>10}{:>11}{:>10}{:>10}{:>9}{:>10}{:>7}"


@click.command(
    help="Times every kernel over several trials and prints a table of time statistics.",
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
    help="Timed calls per kernel, interleaved across kernels.",
)
def main(N: int, warmup_iteration_count: int, trial_count: int) -> None:
    """`main` runs the benchmark binary and prints one row of time statistics per kernel.

    Args:
        N: Matrix size.
        warmup_iteration_count: Untimed calls per kernel before the trials.
        trial_count: Timed calls per kernel.

    Example:
        $ uv run scripts/bench.py -N 4096 --trial-count 50

    Must know:
        Limitations:
            - The roofline assumes that A, B and C each cross DRAM exactly once.
    """
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

    bench_command: list[str] = [
        str(_BENCH_BINARY_PATH),
        str(N),
        str(warmup_iteration_count),
        str(trial_count),
    ]
    bench_stdout: str = subprocess.run(
        bench_command, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    kernel_name_to_call_times_ms_according_to_cuda_events: dict[str, list[float]] = {}
    for csv_line in bench_stdout.splitlines():
        kernel_name, elapsed_ms_text = csv_line.split(",")
        kernel_name_to_call_times_ms_according_to_cuda_events.setdefault(kernel_name, []).append(
            float(elapsed_ms_text)
        )

    print(
        _ROW_FORMAT.format(
            "kernel",
            "N",
            "trials",
            "min ms",
            "median ms",
            "mean ms",
            "max ms",
            "TFLOPS",
            "roofline",
            "%roof",
        )
    )
    for kernel_name in kernel_name_to_call_times_ms_according_to_cuda_events:
        call_times_ms_according_to_cuda_events: list[float] = (
            kernel_name_to_call_times_ms_according_to_cuda_events[kernel_name]
        )
        median_call_time_ms_according_to_cuda_events: float = statistics.median(
            call_times_ms_according_to_cuda_events
        )
        achieved_tflops_according_to_cuda_events: float = (
            matmul_flop
            / (median_call_time_ms_according_to_cuda_events / _MS_PER_S)
            / _FLOP_PER_TFLOP
        )
        roofline_frac: float = (
            achieved_tflops_according_to_cuda_events / roofline_tflops_according_to_nvidia_spec
        )
        print(
            _ROW_FORMAT.format(
                kernel_name,
                N,
                len(call_times_ms_according_to_cuda_events),
                f"{min(call_times_ms_according_to_cuda_events):.3f}",
                f"{median_call_time_ms_according_to_cuda_events:.3f}",
                f"{statistics.mean(call_times_ms_according_to_cuda_events):.3f}",
                f"{max(call_times_ms_according_to_cuda_events):.3f}",
                f"{achieved_tflops_according_to_cuda_events:.1f}",
                f"{roofline_tflops_according_to_nvidia_spec:.1f}",
                f"{roofline_frac:.0%}",
            )
        )


if __name__ == "__main__":
    main()
