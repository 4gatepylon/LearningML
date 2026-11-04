"""`bench_attention_decode.py` checks and times muse_attention_decode kernels on the local GPU.

First, the script runs every test of `TASK.md` with each seed, eagerly and as a CUDA graph replayed
on the next seed's inputs, and checks each output with `reference.check_implementation`. Then, for
each scored shape, the script builds the judge's inputs with `reference.generate_input`, checks
every implementation with `reference.check_implementation`, and times CUDA graph replays the way
`TASK.md` describes: each replay starts with a cold L2, the implementations take turns in an order
that flips every repetition, and the median replay time of an empty graph is subtracted. It writes
the medians as a measurements file for `roofline_attention_decode.py`, and each implementation's
speedup over the first: the median, over repetitions, of the first's time over its own.

Interface:
    --implementation: `NAME=PATH` of a file with `kernel_fn(data)`; repeatable, at least one.
    --gpu: Key of the local GPU in `roofline_attention_decode.GPUS`, written to the measurements.
    --workload: Shape key to run, e.g. `d16f`; repeatable (default: all 12 scored shapes).
    --repetition-count: Timed replays per implementation and shape (default 20).
    --seed: Input seed of the scored shapes (default 1).
    --test-seed-count: Seeds per `TASK.md` test; 0 skips the tests (default 1).
    --label: Suffix for the run folder's name (default: none).
    --json: Prints the summary as one JSON object instead of a table.

Example:
    $ CUDA_VISIBLE_DEVICES=2 python scripts/bench_attention_decode.py --gpu h100-pcie \\
        --implementation starter=muse_attention_decode/submission.py

Must know:
    Limitations:
        - Local times rank memory-bound kernels differently from the judge's H200.
    Gotchas:
        - The script needs a Python with torch (CUDA) and click; `uv run` would download torch.
    Side effects:
        - Each run writes `command.txt`, `gpu_state.txt` (nvidia-smi before and after),
          `tests.json`, `measurements.json` and `summary.json` (checks, times, paired speedups).
"""

import importlib.util
import json
import re
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import click
import torch

SCRIPTS_DIR: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import roofline_attention_decode as roofline  # noqa: E402

PROBLEM_DIR: Path = SCRIPTS_DIR.parent / "muse_attention_decode"
ARTIFACTS_DIR: Path = roofline.CHALLENGE_DIR / ".artifacts" / "bench_attention_decode"
L2_FLUSH_BYTES: int = 1 << 30
BYTES_PER_FP32: int = 4
US_PER_MS: float = 1e3
TESTS_PATTERN: re.Pattern = re.compile(r"must pass these \d+ tests.*?```\n(.*?)```", re.S)
GPU_STATE_QUERY: list[str] = [
    "nvidia-smi",
    "--query-gpu=index,name,clocks.sm,clocks.mem,utilization.gpu,memory.used",
    "--format=csv",
]


def _load_module(path: Path, name: str) -> ModuleType:
    """`_load_module` imports a Python file under a given module name.

    Args:
        path: The file to import.
        name: The module name to give it.

    Returns:
        The imported module.

    Example:
        >>> reference = _load_module(PROBLEM_DIR / "reference.py", "reference")
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_kwargs(workload: roofline.Workload) -> dict:
    """`_generate_kwargs` gives `reference.generate_input`'s arguments for one scored shape.

    Args:
        workload: The shape, as `TASK.md`'s Shapes table defines it.

    Returns:
        Keyword arguments, without the seed.

    Example:
        >>> data = reference.generate_input(seed=1, **_generate_kwargs(workload))
    """
    return {
        "batch": workload.batch_size,
        "ctx": roofline.CONTEXT_TOKEN_COUNT,
        "heads": "tp8",
        "layer": "sliding" if workload.is_sliding else "full",
        "q_len": workload.query_token_count,
        "spread": "fixed" if workload.batch_size == 1 else "agentic",
        "dist": "soft" if workload.is_sliding else "coherent",
        "page": 1,
    }


def _check_output(reference: ModuleType, data: dict, out: torch.Tensor, description: str) -> str:
    """`_check_output` checks one output with `reference.check_implementation`.

    Args:
        reference: The problem's `reference` module.
        data: The inputs the output was computed from, after the call wrote the cache.
        out: The implementation's output.
        description: Names the implementation and case in the error.

    Returns:
        The check's message.

    Raises:
        click.ClickException: The output fails the check.

    Example:
        >>> message = _check_output(reference, data, module.kernel_fn(data), "fused on d1s")
    """
    is_correct, message = reference.check_implementation(data, reference.collect(data, out))
    if not is_correct:
        raise click.ClickException(f"{description} fails: {message}")
    return message


def _run_tests(reference: ModuleType, modules: dict[str, ModuleType], seed_count: int) -> dict:
    """`_run_tests` checks every implementation on every `TASK.md` test, eagerly and as a graph.

    Args:
        reference: The problem's `reference` module.
        modules: Name -> implementation module.
        seed_count: Seeds per test; the graph captured on seed s is replayed on seed s + 1.

    Returns:
        Name -> list of check messages, eager then replayed, per test and seed.

    Raises:
        click.ClickException: An output fails its check.

    Example:
        >>> messages = _run_tests(reference, modules, 2)
    """
    messages: dict[str, list[str]] = {name: [] for name in modules}
    for line in TESTS_PATTERN.search((PROBLEM_DIR / "TASK.md").read_text()).group(1).splitlines():
        kwargs: dict = {}
        for field in line.split(","):
            key, value = field.split()
            kwargs[key] = int(value) if value.isdigit() else float(value) if "." in value else value
        for seed in range(seed_count):
            for name, module in modules.items():
                data: dict = reference.generate_input(seed=seed, **kwargs)
                messages[name].append(_check_output(reference, data, module.kernel_fn(data), name))
                graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out: torch.Tensor = module.kernel_fn(data)
                for key, value in reference.generate_input(seed=seed + 1, **kwargs).items():
                    if isinstance(value, torch.Tensor):
                        data[key].copy_(value)
                graph.replay()
                messages[name].append(_check_output(reference, data, out, f"{name} replayed"))
    return messages


def _time_replays(runs: dict[str, Callable[[], object]], repetition_count: int) -> dict:
    """`_time_replays` captures each call as a CUDA graph and times interleaved, cold-L2 replays.

    Args:
        runs: Name -> call; each is run once, then captured. The order flips every repetition.
        repetition_count: Replays per graph.

    Returns:
        Name -> list of replay times in microseconds, in repetition order.

    Example:
        >>> times_us = _time_replays({"starter": lambda: module.kernel_fn(data)}, 20)
    """
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, run in runs.items():
        run()
        torch.cuda.synchronize()
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            run()
    flush: torch.Tensor = torch.zeros(L2_FLUSH_BYTES // BYTES_PER_FP32, device="cuda")
    sink: torch.Tensor = torch.empty((), device="cuda")
    times_us: dict[str, list[float]] = {name: [] for name in graphs}
    for repetition in range(repetition_count):
        names: list[str] = list(graphs)[:: 1 if repetition % 2 == 0 else -1]
        for name in names:
            sink.copy_(flush.sum())
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            torch.cuda.synchronize()
            times_us[name].append(start.elapsed_time(end) * US_PER_MS)
    return times_us


@click.command(
    epilog="Example: python scripts/bench_attention_decode.py --gpu h100-pcie "
    "--implementation starter=muse_attention_decode/submission.py"
)
@click.option(
    "--implementation",
    "implementation_specs",
    multiple=True,
    required=True,
    help="NAME=PATH of a file with kernel_fn(data); repeatable.",
)
@click.option(
    "--gpu",
    "gpu_key",
    required=True,
    type=click.Choice(list(roofline.GPUS)),
    help="Key of the local GPU in roofline_attention_decode.GPUS.",
)
@click.option(
    "--workload",
    "workload_keys",
    multiple=True,
    type=click.Choice([w.key for w in roofline.WORKLOADS]),
    help="Shape key; repeatable (default: all 12).",
)
@click.option("--repetition-count", default=20, show_default=True, type=click.IntRange(min=1))
@click.option("--seed", default=1, show_default=True, type=int)
@click.option("--test-seed-count", default=1, show_default=True, type=click.IntRange(min=0))
@click.option("--label", default=None, help="Suffix for the run folder's name (default: none).")
@click.option("--json", "should_print_json", is_flag=True, help="Print one JSON object.")
def main(
    implementation_specs: tuple[str, ...],
    gpu_key: str,
    workload_keys: tuple[str, ...],
    repetition_count: int,
    seed: int,
    test_seed_count: int,
    label: str | None,
    should_print_json: bool,
) -> None:
    """`main` checks and times every implementation on every chosen shape."""
    paths: dict[str, Path] = {}
    for spec in implementation_specs:
        name, separator, path = spec.partition("=")
        if not separator or name in paths or not Path(path).is_file():
            raise click.BadParameter(f"--implementation {spec!r} must be a new NAME=existing PATH.")
        paths[name] = Path(path)
    if label is not None and not roofline.LABEL_PATTERN.match(label):
        raise click.BadParameter(f"--label {label!r} must match {roofline.LABEL_PATTERN.pattern}.")
    sys.path.insert(0, str(PROBLEM_DIR))
    reference: ModuleType = _load_module(PROBLEM_DIR / "reference.py", "reference")
    modules: dict[str, ModuleType] = {
        n: _load_module(p, f"impl_{i}") for i, (n, p) in enumerate(paths.items())
    }
    run_id: str = f"{datetime.now(UTC):%Y%m%d-%H%M%S}" + (f"-{label}" if label else "")
    run_dir: Path = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True)
    git_commit: str = subprocess.check_output(
        ["git", "describe", "--always", "--dirty"], cwd=SCRIPTS_DIR, text=True
    ).strip()
    (run_dir / "command.txt").write_text(f"{shlex.join(sys.argv)}\ngit: {git_commit}\n")
    gpu_state: str = subprocess.check_output(GPU_STATE_QUERY, text=True)
    tests: dict = _run_tests(reference, modules, test_seed_count)
    (run_dir / "tests.json").write_text(json.dumps(tests, indent=2))
    elapsed_us: dict[str, dict[str, float]] = {name: {} for name in modules}
    first_name: str = next(iter(modules))
    summary: dict[str, dict] = {}
    for workload in roofline.WORKLOADS:
        if workload_keys and workload.key not in workload_keys:
            continue
        data: dict = reference.generate_input(seed=seed, **_generate_kwargs(workload))
        summary[workload.key] = {}
        for name, module in modules.items():
            out: torch.Tensor = module.kernel_fn(data)
            message: str = _check_output(reference, data, out, f"{name} on {workload.key}")
            summary[workload.key][name] = {"check": message}
        runs: dict[str, Callable[[], object]] = {"empty graph": lambda: None}
        for name, module in modules.items():
            runs[name] = lambda module=module, data=data: module.kernel_fn(data)
        times_us: dict[str, list[float]] = _time_replays(runs, repetition_count)
        empty_us: float = statistics.median(times_us.pop("empty graph"))
        for name, replay_us in times_us.items():
            elapsed_us[name][workload.key] = statistics.median(replay_us) - empty_us
            summary[workload.key][name]["median_us"] = elapsed_us[name][workload.key]
            summary[workload.key][name]["empty_graph_us"] = empty_us
            summary[workload.key][name][f"speedup_over_{first_name}"] = statistics.median(
                first_us / us for first_us, us in zip(times_us[first_name], replay_us, strict=True)
            )
        torch.cuda.empty_cache()
    gpu_state += subprocess.check_output(GPU_STATE_QUERY, text=True)
    (run_dir / "gpu_state.txt").write_text(gpu_state)
    measurements: dict = {
        "gpu": gpu_key,
        "source": f"bench_attention_decode run {run_id}",
        "elapsed_us": elapsed_us,
    }
    (run_dir / "measurements.json").write_text(json.dumps(measurements, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if should_print_json:
        click.echo(json.dumps({"run_id": run_id, "summary": summary}))
        return
    for key, row in summary.items():
        cells: list[str] = [
            f"{n} {v['median_us']:9.1f} us (x{v[f'speedup_over_{first_name}']:.2f})"
            for n, v in row.items()
        ]
        click.echo(f"{key:5} " + "  ".join(cells))
    click.echo(f"run {run_id}: {run_dir / 'measurements.json'}")


if __name__ == "__main__":
    main()
