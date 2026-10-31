"""`bench_binary.py` runs `.build/bench` and parses its output for the benchmark scripts.

Interface:
    list_kernel_names: Returns the names of the kernels that `.build/bench` can run.
    check_kernel_names: Raises a click error that lists the valid names if any name is unknown.
    run_bench: Runs `.build/bench` and returns each kernel's milliseconds per call, per trial.

Example:
    >>> samples: dict[str, list[float]] = run_bench(8192, 1, 3, 100, ("cublas",))

Must know:
    Gotchas:
        - `.build/bench` must be built first; `make bench` builds it.
"""

import subprocess
from pathlib import Path

import click

_BENCH_BINARY_PATH: Path = Path(__file__).resolve().parent.parent / ".build" / "bench"
_LIST_KERNELS_FLAG: str = "--list-kernels"


def list_kernel_names() -> list[str]:
    """`list_kernel_names` returns the names of the kernels that `.build/bench` can run.

    Returns:
        The kernel names, in the order of `KERNELS` in `common/matmul.h`.

    Example:
        >>> kernel_names: list[str] = list_kernel_names()
    """
    bench_command: list[str] = [str(_BENCH_BINARY_PATH), _LIST_KERNELS_FLAG]
    return subprocess.run(
        bench_command, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.split()


def check_kernel_names(kernel_names: tuple[str, ...]) -> None:
    """`check_kernel_names` rejects kernel names that `.build/bench` doesn't know.

    Args:
        kernel_names: Names passed with `--kernel`.

    Raises:
        click.BadParameter: A name is not a kernel; the message lists the valid names.

    Example:
        >>> check_kernel_names(("cublas",))
    """
    valid_kernel_names: list[str] = list_kernel_names()
    unknown_kernel_names: list[str] = [
        name for name in kernel_names if name not in valid_kernel_names
    ]
    if unknown_kernel_names:
        unknown_text: str = ", ".join(unknown_kernel_names)
        valid_text: str = ", ".join(valid_kernel_names)
        raise click.BadParameter(
            f"unknown {unknown_text}; choose from {valid_text}", param_hint="--kernel"
        )


def run_bench(
    N: int,
    warmup_iteration_count: int,
    trial_count: int,
    batch_size: int,
    kernel_names: tuple[str, ...],
) -> dict[str, list[float]]:
    """`run_bench` runs `.build/bench` and returns each kernel's milliseconds per call, per trial.

    Args:
        N: Matrix size.
        warmup_iteration_count: Untimed calls per kernel before the trials.
        trial_count: Timed batches per kernel.
        batch_size: Back-to-back calls per timed batch.
        kernel_names: Kernels to run; an empty tuple runs every kernel.

    Returns:
        A map from kernel name to its milliseconds per call according to CUDA events, one per trial.

    Raises:
        subprocess.CalledProcessError: `.build/bench` failed; its error message went to stderr.

    Example:
        >>> samples: dict[str, list[float]] = run_bench(8192, 1, 3, 100, ("cublas",))
    """
    bench_command: list[str] = [
        str(_BENCH_BINARY_PATH),
        str(N),
        str(warmup_iteration_count),
        str(trial_count),
        str(batch_size),
        *kernel_names,
    ]
    bench_stdout: str = subprocess.run(
        bench_command, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    kernel_name_to_ms_per_call_samples_according_to_cuda_events: dict[str, list[float]] = {}
    for csv_line in bench_stdout.splitlines():
        kernel_name, ms_per_call_text = csv_line.split(",")
        kernel_name_to_ms_per_call_samples_according_to_cuda_events.setdefault(
            kernel_name, []
        ).append(float(ms_per_call_text))
    return kernel_name_to_ms_per_call_samples_according_to_cuda_events
