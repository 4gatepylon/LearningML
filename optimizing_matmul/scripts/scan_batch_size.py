"""`scan_batch_size.py` times kernels at a range of batch sizes and fits the fixed cost per batch.

`scan_batch_size.py` shows how the batch size changes the time per call. A batch of K calls takes
about `fixed_ms_per_batch + K * ms_per_call`, where the fixed part is the batch's first stub and
launch. The script times each batch size with `.build/bench`, fits that line by least squares to
every trial's batch time, and reports the fit's R^2 and the smallest batch size that keeps the
fixed part under 1% of the time per call.

Interface:
    -N: Matrix size (default 8192, at least 1).
    --warmup-iteration-count: Untimed calls per kernel before each size's trials (default 1).
    --trial-count: Timed batches per kernel and batch size (default 3, at least 1).
    --min-batch-size: Smallest batch size (default 1, at least 1).
    --max-batch-size: Largest batch size (default 256, at least 1).
    --sample-count: Number of batch sizes for linear spacing (default 8, at least 2).
    --spacing: linear, or exponential: min, min + 1, min + 3, min + 7, ... (default).
    --kernel: A kernel to time; repeat it for several (default: every kernel).
    --list-kernels: Print the kernel names and exit.

Example:
    $ make scan SCAN_ARGS="--kernel cublas --max-batch-size 512"

Must know:
    Limitations:
        - Exponential spacing ignores `--sample-count`.
        - The fit needs at least two batch sizes.
    Gotchas:
        - The GPU stays warm between batch sizes, so each size's clocks depend on earlier ones.
        - Long batches reach the 310 W cap's clocks, so ms per call can rise with the batch size.
        - When ms per call rises with batch size, the fit's fixed cost goes negative; it shows n/a.
"""

import math
import statistics

import click
from bench_binary import check_kernel_names, list_kernel_names, run_bench

_DEFAULT_N: int = 8192
_DEFAULT_WARMUP_ITERATION_COUNT: int = 1
_DEFAULT_TRIAL_COUNT: int = 3
_DEFAULT_MIN_BATCH_SIZE: int = 1
_DEFAULT_MAX_BATCH_SIZE: int = 256
_DEFAULT_SAMPLE_COUNT: int = 8
_MIN_N: int = 1
_MIN_WARMUP_ITERATION_COUNT: int = 0
_MIN_TRIAL_COUNT: int = 1
_MIN_BATCH_SIZE: int = 1
_MIN_SAMPLE_COUNT: int = 2
_LINEAR_SPACING: str = "linear"
_EXPONENTIAL_SPACING: str = "exponential"
_MAX_FIXED_COST_FRAC: float = 0.01
_TIMING_ROW_FORMAT: str = "{:<8}{:>7}{:>8}{:>11}{:>11}{:>11}{:>11}"
_FIT_ROW_FORMAT: str = "{:<8}{:>16}{:>12}{:>9}{:>26}"


def _compute_batch_sizes(
    min_batch_size: int, max_batch_size: int, sample_count: int, spacing: str
) -> list[int]:
    """`_compute_batch_sizes` returns the batch sizes to time, smallest first.

    Args:
        min_batch_size: Smallest batch size.
        max_batch_size: Largest batch size.
        sample_count: Number of batch sizes for linear spacing.
        spacing: `_LINEAR_SPACING` or `_EXPONENTIAL_SPACING`.

    Returns:
        Distinct batch sizes from `min_batch_size` to at most `max_batch_size`.

    Example:
        >>> _compute_batch_sizes(5, 20, 8, _EXPONENTIAL_SPACING)
        [5, 6, 8, 12, 20]
    """
    if spacing == _LINEAR_SPACING:
        step: float = (max_batch_size - min_batch_size) / (sample_count - 1)
        return sorted({round(min_batch_size + i * step) for i in range(sample_count)})
    batch_sizes: list[int] = []
    batch_size: int = min_batch_size
    increment: int = 1
    while batch_size <= max_batch_size:
        batch_sizes.append(batch_size)
        batch_size += increment
        increment *= 2
    return batch_sizes


def _fit_batch_time(
    batch_sizes: list[int], batch_ms_values: list[float]
) -> tuple[float, float, float]:
    """`_fit_batch_time` fits `batch_ms = fixed_ms_per_batch + batch_size * ms_per_call`.

    Args:
        batch_sizes: The batch size of each timed batch.
        batch_ms_values: Each timed batch's milliseconds, in the same order.

    Returns:
        `fixed_ms_per_batch`, `ms_per_call` and the fit's R^2.

    Raises:
        statistics.StatisticsError: The batch sizes are all the same.

    Example:
        >>> _fit_batch_time([1, 2, 4], [3.0, 5.0, 9.0])
        (1.0, 2.0, 1.0)
    """
    ms_per_call, fixed_ms_per_batch = statistics.linear_regression(batch_sizes, batch_ms_values)
    r_squared: float = statistics.correlation(batch_sizes, batch_ms_values) ** 2
    return fixed_ms_per_batch, ms_per_call, r_squared


@click.command(
    help="Times kernels at a range of batch sizes and fits the fixed cost per batch.",
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
    help="Untimed calls per kernel before each batch size's trials.",
)
@click.option(
    "--trial-count",
    type=click.IntRange(min=_MIN_TRIAL_COUNT),
    default=_DEFAULT_TRIAL_COUNT,
    show_default=True,
    help="Timed batches per kernel and batch size.",
)
@click.option(
    "--min-batch-size",
    type=click.IntRange(min=_MIN_BATCH_SIZE),
    default=_DEFAULT_MIN_BATCH_SIZE,
    show_default=True,
    help="Smallest batch size.",
)
@click.option(
    "--max-batch-size",
    type=click.IntRange(min=_MIN_BATCH_SIZE),
    default=_DEFAULT_MAX_BATCH_SIZE,
    show_default=True,
    help="Largest batch size.",
)
@click.option(
    "--sample-count",
    type=click.IntRange(min=_MIN_SAMPLE_COUNT),
    default=_DEFAULT_SAMPLE_COUNT,
    show_default=True,
    help="Number of batch sizes for linear spacing.",
)
@click.option(
    "--spacing",
    type=click.Choice([_LINEAR_SPACING, _EXPONENTIAL_SPACING]),
    default=_EXPONENTIAL_SPACING,
    show_default=True,
    help="Exponential: min, min + 1, min + 3, min + 7, ...",
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
    min_batch_size: int,
    max_batch_size: int,
    sample_count: int,
    spacing: str,
    kernel_names: tuple[str, ...],
    is_listing_kernels: bool,
) -> None:
    """`main` times each batch size, then prints a timing table and one line fit per kernel.

    Args:
        N: Matrix size.
        warmup_iteration_count: Untimed calls per kernel before each batch size's trials.
        trial_count: Timed batches per kernel and batch size.
        min_batch_size: Smallest batch size.
        max_batch_size: Largest batch size.
        sample_count: Number of batch sizes for linear spacing.
        spacing: `_LINEAR_SPACING` or `_EXPONENTIAL_SPACING`.
        kernel_names: Kernels to time; empty times every kernel.
        is_listing_kernels: Whether to print the kernel names and exit instead.

    Example:
        $ uv run scripts/scan_batch_size.py --kernel cublas --spacing linear --sample-count 5
    """
    if is_listing_kernels:
        click.echo("\n".join(list_kernel_names()))
        return
    check_kernel_names(kernel_names)
    if min_batch_size > max_batch_size:
        raise click.BadParameter("must be at most --max-batch-size", param_hint="--min-batch-size")

    kernel_name_to_batch_size_to_ms_per_call_samples_according_to_cuda_events: dict[
        str, dict[int, list[float]]
    ] = {}
    for batch_size in _compute_batch_sizes(min_batch_size, max_batch_size, sample_count, spacing):
        kernel_name_to_samples: dict[str, list[float]] = run_bench(
            N, warmup_iteration_count, trial_count, batch_size, kernel_names
        )
        for kernel_name, ms_per_call_samples in kernel_name_to_samples.items():
            kernel_name_to_batch_size_to_ms_per_call_samples_according_to_cuda_events.setdefault(
                kernel_name, {}
            )[batch_size] = ms_per_call_samples

    print(f"N = {N}, {trial_count} trials per batch size; times are milliseconds per call.")
    print(
        _TIMING_ROW_FORMAT.format(
            "kernel", "batch", "trials", "min ms", "median ms", "mean ms", "max ms"
        )
    )
    for (
        kernel_name,
        batch_size_to_samples,
    ) in kernel_name_to_batch_size_to_ms_per_call_samples_according_to_cuda_events.items():
        for batch_size, ms_per_call_samples in batch_size_to_samples.items():
            print(
                _TIMING_ROW_FORMAT.format(
                    kernel_name,
                    batch_size,
                    len(ms_per_call_samples),
                    f"{min(ms_per_call_samples):.4f}",
                    f"{statistics.median(ms_per_call_samples):.4f}",
                    f"{statistics.mean(ms_per_call_samples):.4f}",
                    f"{max(ms_per_call_samples):.4f}",
                )
            )

    print()
    print("Fit: batch_ms = fixed_ms_per_batch + batch_size * ms_per_call, over every trial.")
    print(
        _FIT_ROW_FORMAT.format(
            "kernel", "fixed ms/batch", "ms/call", "R^2", "batch for <1% fixed cost"
        )
    )
    for (
        kernel_name,
        batch_size_to_samples,
    ) in kernel_name_to_batch_size_to_ms_per_call_samples_according_to_cuda_events.items():
        trial_batch_sizes: list[int] = [
            batch_size for batch_size, samples in batch_size_to_samples.items() for _ in samples
        ]
        trial_batch_ms_values: list[float] = [
            batch_size * ms_per_call
            for batch_size, samples in batch_size_to_samples.items()
            for ms_per_call in samples
        ]
        if len(batch_size_to_samples) < _MIN_SAMPLE_COUNT:
            print(f"{kernel_name}: the fit needs at least {_MIN_SAMPLE_COUNT} batch sizes")
            continue
        fixed_ms_per_batch, ms_per_call, r_squared = _fit_batch_time(
            trial_batch_sizes, trial_batch_ms_values
        )
        is_fixed_cost_negative: bool = fixed_ms_per_batch < 0
        smallest_batch_size_under_fixed_cost_target: str = (
            "n/a (negative fixed cost)"
            if is_fixed_cost_negative
            else str(math.ceil(fixed_ms_per_batch / (_MAX_FIXED_COST_FRAC * ms_per_call)))
        )
        print(
            _FIT_ROW_FORMAT.format(
                kernel_name,
                f"{fixed_ms_per_batch:.4f}",
                f"{ms_per_call:.4f}",
                f"{r_squared:.5f}",
                smallest_batch_size_under_fixed_cost_target,
            )
        )


if __name__ == "__main__":
    main()
