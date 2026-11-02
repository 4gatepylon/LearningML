"""`local_judge.py` checks and times an okc kernel on a local GPU, the way the okc judge does.

`local_judge.py` checks the kernel on every test and scored shape in `TASK.md`, then times each
scored shape as a CUDA graph replay after an L2 flush, per shape and per load (weighted).

Interface:
    KERNEL: The submission file to judge.
    --compare: Another submission file to time on the same shapes (e.g. the parent kernel).
    --seeds: Seeds per test and scored shape for the correctness check (default 2).
    --reps: Timed replays per shape (default 30).
    --only: Only scored shapes whose load contains this text (e.g. "decode 32").
    --no-check, --no-time: Skip the correctness check or the timing.

Example:
    $ python scripts/local_judge.py muse_attention_decode/kernels/002-per-head-dots.py --compare \
        muse_attention_decode/kernels/001-triton-splitkv.py

Must know:
    Limitations:
        - `local_judge.py` handles problems without weights (muse_attention_decode) only.
    Gotchas:
        - A local H100 has 2.0 TB/s, not the judge H200's 4.8 TB/s: compare kernels, not times.
"""

import importlib.util
import re
import statistics
import sys
from pathlib import Path
from types import ModuleType

import click
import torch

L2_FLUSH_BYTES: int = 512 * 2**20
US_PER_MS: float = 1e3
WARMUP_CALLS: int = 2
TEST_SEED_BASE: int = 1000
FILE_TYPE: click.Path = click.Path(exists=True, dir_okay=False, path_type=Path)


def _load_module(path: Path, name: str) -> ModuleType:
    """`_load_module` imports a Python file as a module named `name` and returns it.

    Args:
        path: The file to import.
        name: The module's name.

    Example:
        >>> reference = _load_module(problem_dir / "reference.py", "reference")
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_case(text: str) -> dict:
    """`_parse_case` returns "batch 1, ctx 10, layer sliding" as generate_input's keyword arguments.

    Args:
        text: Comma-separated `name value` pairs, as TASK.md writes them.

    Example:
        >>> _parse_case("batch 1, ctx 10, heads tp8, layer sliding")["ctx"]
        10
    """
    case: dict = {}
    for pair in text.split(","):
        key, value = pair.strip().split(None, 1)
        is_int: bool = re.fullmatch(r"-?\d+", value) is not None
        is_float: bool = re.fullmatch(r"-?\d*\.\d+", value) is not None
        case[key] = int(value) if is_int else float(value) if is_float else value
    return case


def _read_cases(task_path: Path) -> tuple[list[dict], list[dict]]:
    """`_read_cases` returns the tests and the scored shapes (with load and weight) from TASK.md.

    Args:
        task_path: The problem's TASK.md.

    Returns:
        The tests, then the scored shapes.

    Example:
        >>> tests, shapes = _read_cases(problem_dir / "TASK.md")
    """
    text: str = task_path.read_text()
    test_block: str = re.search(r"must pass these \d+ tests.*?```\n(.*?)```", text, re.S).group(1)
    tests: list[dict] = [_parse_case(line) for line in test_block.strip().splitlines()]
    shapes: list[dict] = []
    for load, shape, weight in re.findall(
        r"^\| ((?:decode|verify)[^|]*?) \| ([^|]+?) \| (\d+) \|", text, re.M
    ):
        shapes.append({"load": load, "weight": int(weight), "case": _parse_case(shape)})
    return tests, shapes


def _check_kernel(
    reference: ModuleType, kernel: ModuleType, cases: list[dict], seed_count: int
) -> int:
    """`_check_kernel` runs the kernel on each case and seed and prints each failure.

    Args:
        reference: The problem's reference module.
        kernel: The submission module.
        cases: generate_input keyword arguments, one dict per case.
        seed_count: Seeds per case.

    Returns:
        The number of failing (case, seed) pairs.

    Example:
        >>> failure_count = _check_kernel(reference, kernel, tests, 2)
    """
    failure_count: int = 0
    for case_index, case in enumerate(cases):
        for seed_index in range(seed_count):
            seed: int = TEST_SEED_BASE * (case_index + 1) + seed_index
            data: dict = reference.generate_input(**case, seed=seed)
            raw: torch.Tensor = reference.collect(data, kernel.kernel_fn(data))
            is_ok, message = reference.check_implementation(data, raw)
            if not is_ok:
                failure_count += 1
                click.echo(f"FAIL {case} seed {seed}: {message}")
    return failure_count


def _capture_graph(kernel: ModuleType, data: dict) -> torch.cuda.CUDAGraph:
    """`_capture_graph` warms the kernel up on a side stream and captures one call as a CUDA graph.

    Args:
        kernel: The submission module.
        data: The inputs the graph will read.

    Returns:
        The captured graph.

    Example:
        >>> graph = _capture_graph(kernel, data)
    """
    stream: torch.cuda.Stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_CALLS):
            kernel.kernel_fn(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel.kernel_fn(data)
    return graph


def _time_graph_us(graph: torch.cuda.CUDAGraph, flush: torch.Tensor, rep_count: int) -> float:
    """`_time_graph_us` returns the median replay time of `graph` after an L2 flush, in µs.

    Args:
        graph: The graph to replay.
        flush: A buffer larger than L2, summed before each replay.
        rep_count: Timed replays.

    Returns:
        The median replay time in µs.

    Example:
        >>> elapsed_us = _time_graph_us(graph, flush, 30)
    """
    elapsed_us_per_rep: list[float] = []
    for _ in range(rep_count):
        flush.sum()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        elapsed_us_per_rep.append(start.elapsed_time(stop) * US_PER_MS)
    return statistics.median(elapsed_us_per_rep)


@click.command(epilog="Example: python scripts/local_judge.py KERNEL_PATH --compare PARENT_PATH")
@click.argument("kernel_path", type=FILE_TYPE)
@click.option("--compare", "compare_path", type=FILE_TYPE, help="Another kernel to time too.")
@click.option("--seeds", "seed_count", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--reps", "rep_count", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--only", default=None, help="Only scored shapes whose load contains this text.")
@click.option("--no-check", is_flag=True, help="Skip the correctness check.")
@click.option("--no-time", is_flag=True, help="Skip the timing.")
def main(
    kernel_path: Path,
    compare_path: Path | None,
    seed_count: int,
    rep_count: int,
    only: str | None,
    no_check: bool,
    no_time: bool,
) -> None:
    """`main` checks KERNEL_PATH on the problem's tests and scored shapes, then times it."""
    problem_dir: Path = kernel_path.resolve().parent
    while not (problem_dir / "reference.py").exists():
        if problem_dir == problem_dir.parent:
            raise click.BadParameter("no reference.py above the kernel", param_hint="KERNEL_PATH")
        problem_dir = problem_dir.parent
    sys.path.insert(0, str(problem_dir))
    reference: ModuleType = _load_module(problem_dir / "reference.py", "reference")
    tests, shapes = _read_cases(problem_dir / "TASK.md")
    shapes = [shape for shape in shapes if only is None or only in shape["load"]]
    kernels: dict[str, ModuleType] = {"kernel": _load_module(kernel_path, "kernel")}
    if compare_path is not None:
        kernels["compare"] = _load_module(compare_path, "compare")
    if not no_check:
        cases: list[dict] = tests + [shape["case"] for shape in shapes]
        failure_count: int = _check_kernel(reference, kernels["kernel"], cases, seed_count)
        click.echo(f"correctness: {'PASS' if failure_count == 0 else f'{failure_count} FAIL'}")
    if no_time:
        return
    flush: torch.Tensor = torch.ones(L2_FLUSH_BYTES // 4, device="cuda")
    load_us_per_kernel: dict[str, dict[str, float]] = {name: {} for name in kernels}
    for shape in shapes:
        data: dict = reference.generate_input(**shape["case"], seed=TEST_SEED_BASE)
        line: str = f"{shape['load']:<10} {shape['case']['layer']:<8}"
        for name, kernel in kernels.items():
            elapsed_us: float = _time_graph_us(_capture_graph(kernel, data), flush, rep_count)
            loads: dict[str, float] = load_us_per_kernel[name]
            loads[shape["load"]] = loads.get(shape["load"], 0.0) + shape["weight"] * elapsed_us
            line += f"  {name} {elapsed_us:8.1f} us"
        click.echo(line)
        del data
        torch.cuda.empty_cache()
    for load, kernel_us in load_us_per_kernel["kernel"].items():
        line = f"load {load:<10} kernel {kernel_us:9.1f} us"
        if compare_path is not None:
            compare_us: float = load_us_per_kernel["compare"][load]
            line += f"  compare {compare_us:9.1f} us  ratio {compare_us / kernel_us:.3f}x"
        click.echo(line)


if __name__ == "__main__":
    main()
