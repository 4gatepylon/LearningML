"""`bench.py` times every kernel and prints a table of time, TFLOPS and roofline share.

Interface:
    -N: Matrix size (default 8192).
    --warmup-iterations: Untimed calls per kernel before the timed one (default 1).

Example:
    $ make bench BENCH_ARGS="-N 4096 --warmup-iterations 0"

Must know:
    Gotchas:
        - `bench.py` reads the roofline peaks from `.env`, so `source .env` must run first.
"""

import argparse
import os
import subprocess
from pathlib import Path

BENCH_BINARY_PATH: Path = Path(__file__).resolve().parent.parent / ".build" / "bench"
DEFAULT_N: int = 8192
DEFAULT_WARMUP_ITERATIONS: int = 1
FLOP_PER_MULTIPLY_ADD: int = 2
MATRICES_CROSSING_DRAM_COUNT: int = 3
BF16_BYTES: int = 2
MS_PER_S: float = 1e3
FLOP_PER_TFLOP: float = 1e12
ROW_FORMAT: str = "{:<12}{:>6}{:>10}{:>9}{:>10}{:>7}"


def main() -> None:
    """`main` runs the benchmark binary and prints one table row per kernel.

    Example:
        >>> main()

    Must know:
        Limitations:
            - The roofline assumes that A, B and C each cross DRAM exactly once.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-N", type=int, default=DEFAULT_N)
    parser.add_argument("--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    args: argparse.Namespace = parser.parse_args()
    N: int = args.N

    peak_dense_bf16_tflops_according_to_nvidia_spec: float = float(
        os.environ["PEAK_DENSE_BF16_TFLOPS_ACCORDING_TO_NVIDIA_SPEC"]
    )
    mem_bandwidth_tbytes_per_s_according_to_nvidia_spec: float = float(
        os.environ["MEM_BANDWIDTH_TBYTES_PER_S_ACCORDING_TO_NVIDIA_SPEC"]
    )
    matmul_flop: int = FLOP_PER_MULTIPLY_ADD * N**3
    matmul_traffic_bytes: int = MATRICES_CROSSING_DRAM_COUNT * N**2 * BF16_BYTES
    intensity_flop_per_byte: float = matmul_flop / matmul_traffic_bytes
    roofline_tflops_according_to_nvidia_spec: float = min(
        peak_dense_bf16_tflops_according_to_nvidia_spec,
        intensity_flop_per_byte * mem_bandwidth_tbytes_per_s_according_to_nvidia_spec,
    )

    bench_command: list[str] = [str(BENCH_BINARY_PATH), str(N), str(args.warmup_iterations)]
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
