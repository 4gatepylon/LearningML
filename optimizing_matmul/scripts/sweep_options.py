"""`sweep_options.py` times one kernel built with each value of one compile-time option.

A kernel's tunable sizes are compile-time options: macros with defaults that `-D` overrides (see
`CLAUDE.md`). For each value, `sweep_options.py` builds the benchmark binary into its own folder,
`.build/sweep_options/<option>-<value>/`, with `-D<option>=<value>`, and runs the kernel's tests
with that build, so a value that breaks the kernel stops the sweep. It then times the kernel with
each build, interleaving the trials across values, and prints one row per value with the fastest
marked. The results also go to a new run folder.

Interface:
    --kernel: The kernel to time (required).
    --option: The option to sweep: a macro that a kernel file guards with `#ifndef` (required).
    --value: A value of the option; repeat it for each value (required, distinct).
    -N: Matrix size (default 8192, at least 1).
    --warmup-iteration-count: Untimed calls before each value's trials (default 1, at least 0).
    --trial-count: Timed batches per value, interleaved across values (default 3, at least 1).
    --batch-size: Back-to-back calls per timed batch (default 100, at least 1).
    --dry-run: Validate, print the builds and the run folder, and stop.
    --json: Print the results as one JSON object instead of the table.

Example:
    $ make sweep-options SWEEP_ARGS="--kernel pipelined_from_wmma \\
          --option PIPELINED_FROM_WMMA_STAGE_COUNT --value 2 --value 3 --value 4"

Must know:
    Gotchas:
        - `--kernel` is checked against `.build/bench`, which `make sweep-options` builds first.
        - A value that breaks a `static_assert` or a test stops the sweep before any timing.
    Side effects:
        - Each run writes `.artifacts/sweep_options/<run_id>/` with `command.txt` and `results.json`.
"""

import datetime
import json
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import click
from bench_binary import check_kernel_names, run_bench

_PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
_KERNEL_DIR: Path = _PROJECT_DIR / "kernels"
_SWEEP_BUILD_DIR: Path = _PROJECT_DIR / ".build" / "sweep_options"
_RUN_ROOT_DIR: Path = _PROJECT_DIR / ".artifacts" / "sweep_options"
_RUN_ID_FORMAT: str = "%Y%m%d-%H%M%S"
_OPTION_PATTERN: re.Pattern[str] = re.compile(r"[A-Z][A-Z0-9_]*")
_DEFAULT_N: int = 8192
_DEFAULT_WARMUP_ITERATION_COUNT: int = 1
_DEFAULT_TRIAL_COUNT: int = 3
_DEFAULT_BATCH_SIZE: int = 100
_MIN_N: int = 1
_MIN_WARMUP_ITERATION_COUNT: int = 0
_MIN_TRIAL_COUNT: int = 1
_MIN_BATCH_SIZE: int = 1
_SINGLE_TRIAL_COUNT: int = 1
_FLOP_PER_MULTIPLY_ADD: int = 2
_MS_PER_S: float = 1e3
_FLOP_PER_TFLOP: float = 1e12
_ROW_FORMAT: str = "{:<8}{:>12}{:>12}{:>12}{:>10}{:>7}"


def _validate_option(option: str) -> None:
    """`_validate_option` rejects an option that no kernel file guards with `#ifndef`.

    Args:
        option: The macro passed with `--option`.

    Raises:
        click.BadParameter: The name is not UPPER_SNAKE_CASE, or no kernel file defines it.

    Example:
        >>> _validate_option("PIPELINED_FROM_WMMA_STAGE_COUNT")
    """
    if not _OPTION_PATTERN.fullmatch(option):
        raise click.BadParameter(
            f"{option} is not an UPPER_SNAKE_CASE macro", param_hint="--option"
        )
    guard: str = f"#ifndef {option}\n"
    if not any(guard in kernel_path.read_text() for kernel_path in _KERNEL_DIR.glob("*.cu")):
        raise click.BadParameter(
            f"no file in kernels/ has `#ifndef {option}`; options are macros that kernels guard",
            param_hint="--option",
        )


def _validate_values(values: tuple[int, ...]) -> None:
    """`_validate_values` rejects repeated values.

    Args:
        values: The values passed with `--value`.

    Raises:
        click.BadParameter: A value appears more than once.

    Example:
        >>> _validate_values((16, 32, 64))
    """
    if len(set(values)) != len(values):
        raise click.BadParameter(
            f"values must be distinct, got {list(values)}", param_hint="--value"
        )


def _build_and_test(option: str, value: int, kernel_name: str) -> Path:
    """`_build_and_test` builds the binaries with `-D<option>=<value>` and runs the kernel's tests.

    Args:
        option: The macro to define.
        value: Its value.
        kernel_name: The kernel whose tests to run.

    Returns:
        The path of the benchmark binary.

    Raises:
        subprocess.CalledProcessError: The build or a test failed; their output went to stderr.

    Example:
        >>> bench_binary_path: Path = _build_and_test("PIPELINED_FROM_WMMA_STAGE_COUNT", 3, "cublas")
    """
    build_dir: Path = _SWEEP_BUILD_DIR / f"{option}-{value}"
    bench_binary_path: Path = build_dir / "bench"
    test_binary_path: Path = build_dir / "test"
    make_command: list[str] = [
        "make",
        "--no-print-directory",
        f"BUILD_DIR={build_dir.relative_to(_PROJECT_DIR)}",
        f"NVCC_DEFINES=-D{option}={value}",
        str(bench_binary_path.relative_to(_PROJECT_DIR)),
        str(test_binary_path.relative_to(_PROJECT_DIR)),
    ]
    subprocess.run(make_command, cwd=_PROJECT_DIR, check=True, stdout=sys.stderr)
    click.echo(f"testing {kernel_name} with {option}={value}", err=True)
    test_command: list[str] = [
        str(test_binary_path),
        f"--gtest_filter=*/{kernel_name}:*_{kernel_name}",
        "--gtest_brief=1",
    ]
    subprocess.run(test_command, check=True, stdout=sys.stderr)
    return bench_binary_path


def _describe_git_commit() -> str:
    """`_describe_git_commit` returns `git describe --always --dirty` for the project.

    Returns:
        The commit's description, with `-dirty` if there are uncommitted changes.

    Example:
        >>> commit: str = _describe_git_commit()
    """
    describe_command: list[str] = ["git", "describe", "--always", "--dirty"]
    return subprocess.run(
        describe_command, cwd=_PROJECT_DIR, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()


@click.command(
    help="Times one kernel built with each value of one compile-time option.",
    epilog='Example: make sweep-options SWEEP_ARGS="--kernel pipelined_from_wmma '
    '--option PIPELINED_FROM_WMMA_STAGE_COUNT --value 2 --value 3 --value 4"',
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--kernel", "kernel_name", required=True, help="The kernel to time.")
@click.option("--option", required=True, help="The macro to sweep, guarded by #ifndef in kernels/.")
@click.option(
    "--value", "values", type=int, multiple=True, required=True, help="Repeat for each value."
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
    help="Untimed calls before each value's trials.",
)
@click.option(
    "--trial-count",
    type=click.IntRange(min=_MIN_TRIAL_COUNT),
    default=_DEFAULT_TRIAL_COUNT,
    show_default=True,
    help="Timed batches per value, interleaved across values.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=_MIN_BATCH_SIZE),
    default=_DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Back-to-back calls per timed batch.",
)
@click.option("--dry-run", "is_dry_run", is_flag=True, help="Print the plan and stop.")
@click.option("--json", "is_json", is_flag=True, help="Print the results as one JSON object.")
def main(
    kernel_name: str,
    option: str,
    values: tuple[int, ...],
    N: int,
    warmup_iteration_count: int,
    trial_count: int,
    batch_size: int,
    is_dry_run: bool,
    is_json: bool,
) -> None:
    """`main` builds one binary per value, times the kernel with each, and reports the fastest.

    Args:
        kernel_name: The kernel to time.
        option: The macro to sweep.
        values: Its values.
        N: Matrix size.
        warmup_iteration_count: Untimed calls before each value's trials.
        trial_count: Timed batches per value.
        batch_size: Back-to-back calls per timed batch.
        is_dry_run: Whether to stop after printing the plan.
        is_json: Whether to print JSON instead of the table.

    Example:
        >>> main()
    """
    _validate_option(option)
    _validate_values(values)
    check_kernel_names((kernel_name,))
    run_id: str = datetime.datetime.now(datetime.timezone.utc).strftime(_RUN_ID_FORMAT)
    run_dir: Path = _RUN_ROOT_DIR / run_id
    if is_dry_run:
        for value in values:
            click.echo(f"would build and test {_SWEEP_BUILD_DIR / f'{option}-{value}'}")
        click.echo(f"would time {kernel_name} {trial_count} times per value into {run_dir}")
        return

    run_dir.mkdir(parents=True)
    command_text: str = f"{shlex.join(sys.argv)}\n{_describe_git_commit()}\n"
    (run_dir / "command.txt").write_text(command_text)
    value_to_bench_binary_path: dict[int, Path] = {
        value: _build_and_test(option, value, kernel_name) for value in values
    }
    value_to_ms_per_call_samples_according_to_cuda_events: dict[int, list[float]] = {
        value: [] for value in values
    }
    for _ in range(trial_count):
        for value in values:
            value_to_ms_per_call_samples_according_to_cuda_events[value].extend(
                run_bench(
                    N,
                    warmup_iteration_count,
                    _SINGLE_TRIAL_COUNT,
                    batch_size,
                    (kernel_name,),
                    value_to_bench_binary_path[value],
                )[kernel_name]
            )

    matmul_flop: int = _FLOP_PER_MULTIPLY_ADD * N**3
    rows: list[dict[str, float]] = []
    for value, samples in value_to_ms_per_call_samples_according_to_cuda_events.items():
        median_ms_per_call: float = statistics.median(samples)
        rows.append(
            {
                "value": value,
                "median_ms_per_call": median_ms_per_call,
                "min_ms_per_call": min(samples),
                "max_ms_per_call": max(samples),
                "tflops": matmul_flop / (median_ms_per_call / _MS_PER_S) / _FLOP_PER_TFLOP,
            }
        )
    best_row: dict[str, float] = min(rows, key=lambda row: row["median_ms_per_call"])
    results: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "kernel": kernel_name,
        "option": option,
        "N": N,
        "trial_count": trial_count,
        "batch_size": batch_size,
        "best_value": best_row["value"],
        "rows": rows,
        "ms_per_call_samples_according_to_cuda_events": {
            str(value): samples
            for value, samples in value_to_ms_per_call_samples_according_to_cuda_events.items()
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    if is_json:
        click.echo(json.dumps(results))
        return
    click.echo(f"{kernel_name}, N = {N}, {trial_count} trials of {batch_size} calls per value")
    click.echo(_ROW_FORMAT.format("value", "median ms", "min ms", "max ms", "TFLOPS", ""))
    for row in rows:
        best_mark: str = "best" if row is best_row else ""
        click.echo(
            _ROW_FORMAT.format(
                int(row["value"]),
                f"{row['median_ms_per_call']:.3f}",
                f"{row['min_ms_per_call']:.3f}",
                f"{row['max_ms_per_call']:.3f}",
                f"{row['tflops']:.1f}",
                best_mark,
            )
        )
    click.echo(f"run {run_id}: {run_dir}")


if __name__ == "__main__":
    main()
