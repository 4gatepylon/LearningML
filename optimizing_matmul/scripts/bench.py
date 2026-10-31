"""`bench.py` times every kernel and prints a table of time, TFLOPS and roofline share.

`bench.py` is more than a shim around `.build/bench`. The C++ binary does the timing, since the
kernels and CUDA events live in C++. `bench.py` owns the command line (flags, defaults and
validation) and turns the binary's raw times into the table, which is shorter to write in Python.

Interface:
    -N: Matrix size (default 8192, at least 1).
    --warmup-iteration-count: Untimed calls per kernel before the timed one (default 1, at least 0).
    stdout: One table row per kernel.

Example:
    $ make bench BENCH_ARGS="-N 4096 --warmup-iteration-count 0"

Must know:
    Gotchas:
        - `bench.py` reads the roofline peaks from `.env`, so `source .env` must run first.
"""

import os
import subprocess
from pathlib import Path

import click

BENCH_BINARY_PATH: Path = Path(__file__).resolve().parent.parent / ".build" / "bench"
DEFAULT_N: int = 8192
DEFAULT_WARMUP_ITERATION_COUNT: int = 1
MIN_N: int = 1
MIN_WARMUP_ITERATION_COUNT: int = 0
FLOP_PER_MULTIPLY_ADD: int = 2
DRAM_CROSSING_MATRIX_COUNT: int = 3
BYTES_PER_BF16: int = 2
MS_PER_S: float = 1e3
FLOP_PER_TFLOP: float = 1e12
ROW_FORMAT: str = "{:<12}{:>6}{:>10}{:>9}{:>10}{:>7}"


@click.command(
    help="Times every kernel and prints a table of time, TFLOPS and roofline share.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-N",
    "N",
    type=click.IntRange(min=MIN_N),
    default=DEFAULT_N,
    show_default=True,
    help="Matrix size (N x N).",
)
@click.option(
    "--warmup-iteration-count",
    type=click.IntRange(min=MIN_WARMUP_ITERATION_COUNT),
    default=DEFAULT_WARMUP_ITERATION_COUNT,
    show_default=True,
    help="Untimed calls per kernel before the timed one.",
)
def main(N: int, warmup_iteration_count: int) -> None:
    """`main` runs the benchmark binary and prints one table row per kernel.

    Args:
        N: Matrix size.
        warmup_iteration_count: Untimed calls per kernel before the timed one.

    Example:
        $ uv run scripts/bench.py -N 4096

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
    matmul_flop: int = FLOP_PER_MULTIPLY_ADD * N**3
    matmul_traffic_bytes: int = DRAM_CROSSING_MATRIX_COUNT * N**2 * BYTES_PER_BF16
    intensity_flop_per_byte: float = matmul_flop / matmul_traffic_bytes
    roofline_tflops_according_to_nvidia_spec: float = min(
        peak_dense_bf16_tflops_according_to_nvidia_spec,
        intensity_flop_per_byte * mem_bandwidth_tbytes_per_s_according_to_nvidia_spec,
    )

    bench_command: list[str] = [str(BENCH_BINARY_PATH), str(N), str(warmup_iteration_count)]
    bench_stdout: str = subprocess.run(
        bench_command, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    print(ROW_FORMAT.format("kernel", "N", "ms", "TFLOPS", "roofline", "%roof"))
    for csv_line in bench_stdout.splitlines():
        kernel_name, elapsed_ms_text = csv_line.split(",")
        elapsed_ms_according_to_cuda_events: float = float(elapsed_ms_text)
        achieved_tflops_according_to_cuda_events: float = (
            matmul_flop / (elapsed_ms_according_to_cuda_events / MS_PER_S) / FLOP_PER_TFLOP
        )
        roofline_frac: float = (
            achieved_tflops_according_to_cuda_events / roofline_tflops_according_to_nvidia_spec
        )
        print(
            ROW_FORMAT.format(
                kernel_name,
                N,
                f"{elapsed_ms_according_to_cuda_events:.3f}",
                f"{achieved_tflops_according_to_cuda_events:.1f}",
                f"{roofline_tflops_according_to_nvidia_spec:.1f}",
                f"{roofline_frac:.0%}",
            )
        )


if __name__ == "__main__":
    main()
