# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1"]
# ///
"""`roofline_attention_decode.py` computes the roofline of every scored muse_attention_decode shape.

For each scored shape in `muse_attention_decode/TASK.md`, the script counts the matmul FLOP and
the bytes a kernel must move from the model's sizes. From the GPU's spec peaks it then gives the
time each would take alone, which one bounds the shape, and the roofline: the TFLOP/s the peaks
allow at that shape's arithmetic intensity. Every model and GPU input is a module constant, and
`SOURCES` says where each comes from. Measured times, read from JSON files that name their GPU and
source, become achieved TFLOP/s (the matmul FLOP over the time) and a percent of the roofline.

Interface:
    --measurements-file: A measurements JSON file; repeat for several (default: every file in
        `muse_attention_decode/measurements/`). Each holds `gpu` (a key of `GPUS`), `source`, and
        `elapsed_us`: implementation -> shape key (e.g. `d16f`) -> microseconds.
    --label: Suffix for the run folder's name (default: none).
    --json: Prints the results as one JSON object instead of a table.

Example:
    $ uv run --script scripts/roofline_attention_decode.py

Must know:
    Limitations:
        - The FLOP count covers the two attention matmuls only, not norms, RoPE, softmax or gate.
        - Batched shapes use the mean context; the judge draws each length within +-7.5 % of it.
    Side effects:
        - Each run writes `command.txt` and `rows.json` to its run folder under `.artifacts/`.
"""

import json
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import click

CHALLENGE_DIR: Path = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR: Path = CHALLENGE_DIR / ".artifacts" / "roofline_attention_decode"
MEASUREMENTS_DIR: Path = (
    CHALLENGE_DIR / "4gatepylon_official_work/muse_attention_decode/measurements"
)
LABEL_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9_.-]+$")

Q_HEAD_COUNT: int = 16
KV_HEAD_COUNT: int = 2
HEAD_DIM: int = 64
SLIDING_WINDOW_TOKEN_COUNT: int = 2048
CONTEXT_TOKEN_COUNT: int = 80_000
QKV_HEAD_COUNT_PER_TOKEN: int = Q_HEAD_COUNT + KV_HEAD_COUNT + KV_HEAD_COUNT + Q_HEAD_COUNT
CACHE_TENSOR_COUNT: int = 2
MATMUL_COUNT: int = 2
FLOP_PER_MAC: int = 2
BYTES_PER_FP8: int = 1
BYTES_PER_BF16: int = 2
BYTES_PER_PAGE_TABLE_ENTRY: int = 4
SPARSE_TFLOPS_PER_DENSE_TFLOPS: int = 2
US_PER_S: float = 1e6
FLOP_PER_TFLOP: float = 1e12
BYTES_PER_TBYTE: float = 1e12
FLOP_PER_GFLOP: float = 1e9
BYTES_PER_MBYTE: float = 1e6
PCT_PER_FRAC: float = 100.0


class Workload(NamedTuple):
    """`Workload` is one scored shape of `TASK.md`.

    Attributes:
        key: Short name, e.g. `d16f`: d(ecode) or v(erify), batch, s(liding) or f(ull).
        batch_size: Requests per call (B).
        query_token_count: New tokens per request (L): 1 for decode, 8 for verify.
        is_sliding: Whether the layer attends a 2,048-token window (else the whole context).

    Example:
        >>> Workload("d16f", 16, 1, False)
    """

    key: str
    batch_size: int
    query_token_count: int
    is_sliding: bool


class Gpu(NamedTuple):
    """`Gpu` holds the peaks that bound a GPU's roofline.

    Attributes:
        name: Display name.
        peak_dense_bf16_tflops_according_to_nvidia_spec: Dense BF16 tensor-core peak.
        hbm_tbytes_per_s_according_to_nvidia_spec: Peak memory bandwidth.

    Example:
        >>> Gpu("H200 SXM", 989.5, 4.8)
    """

    name: str
    peak_dense_bf16_tflops_according_to_nvidia_spec: float
    hbm_tbytes_per_s_according_to_nvidia_spec: float


WORKLOADS: tuple[Workload, ...] = tuple(
    Workload(f"{'dv'[length > 1]}{batch}{'sf'[not is_sliding]}", batch, length, is_sliding)
    for length in (1, 8)
    for batch in (1, 16, 32)
    for is_sliding in (True, False)
)
GPUS: dict[str, Gpu] = {
    "h200-sxm": Gpu("H200 SXM", 1979 / SPARSE_TFLOPS_PER_DENSE_TFLOPS, 4.8),
}
SOURCES: dict[str, str] = {
    "Q_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM, SLIDING_WINDOW_TOKEN_COUNT": (
        "4gatepylon_official_work/muse_attention_decode/reference.py: HEADS['tp8'], HEAD_DIM, "
        "SLIDING_WINDOW"
    ),
    "WORKLOADS, CONTEXT_TOKEN_COUNT": "4gatepylon_official_work/muse_attention_decode/TASK.md",
    "BYTES_PER_*": "reference.py: FP8 E4M3 K/V cache, int32 page_table, bf16 qkv and output",
    "h200-sxm": "https://www.nvidia.com/en-us/data-center/h200/: BF16 1,979 TFLOPS sparse; 4.8TB/s",
}


def _count_flop_and_bytes(workload: Workload) -> tuple[float, float]:
    """`_count_flop_and_bytes` counts the matmul FLOP and the bytes one call must move.

    Args:
        workload: The shape to count.

    Returns:
        The FLOP of Q K^T and P V, and the bytes of the cached K/V rows read, their page-table
        entries, the qkv input, the new FP8 K/V rows written and the bf16 output.

    Example:
        >>> flop, byte_count = _count_flop_and_bytes(WORKLOADS[0])
    """
    window_token_count: int = SLIDING_WINDOW_TOKEN_COUNT if workload.is_sliding else sys.maxsize
    cached_key_count: int = min(CONTEXT_TOKEN_COUNT, window_token_count - 1)
    attended_key_count: int = sum(
        min(window_token_count, CONTEXT_TOKEN_COUNT + j + 1)
        for j in range(workload.query_token_count)
    )
    flop: float = workload.batch_size * attended_key_count * Q_HEAD_COUNT * HEAD_DIM
    flop *= MATMUL_COUNT * FLOP_PER_MAC
    token_count: int = workload.batch_size * workload.query_token_count
    cache_row_bytes: int = KV_HEAD_COUNT * HEAD_DIM * BYTES_PER_FP8
    key_bytes: int = CACHE_TENSOR_COUNT * cache_row_bytes + BYTES_PER_PAGE_TABLE_ENTRY
    token_bytes: int = (
        QKV_HEAD_COUNT_PER_TOKEN * HEAD_DIM * BYTES_PER_BF16
        + CACHE_TENSOR_COUNT * cache_row_bytes
        + Q_HEAD_COUNT * HEAD_DIM * BYTES_PER_BF16
    )
    return flop, workload.batch_size * cached_key_count * key_bytes + token_count * token_bytes


def _load_measurements(paths: tuple[Path, ...]) -> dict[str, dict[str, dict[str, float]]]:
    """`_load_measurements` merges measurement files into GPU -> implementation -> shape -> us.

    Args:
        paths: Measurement JSON files (see the Interface above).

    Returns:
        The merged times, keyed by GPU key (every key of `GPUS`, measured or not), then
        implementation, then workload key.

    Raises:
        click.BadParameter: A file names an unknown GPU, repeats an implementation of that GPU,
            or lacks a time for some workload.

    Example:
        >>> elapsed_us_per_gpu = _load_measurements(tuple(MEASUREMENTS_DIR.glob("*.json")))
    """
    elapsed_us_per_gpu: dict[str, dict[str, dict[str, float]]] = {gpu_key: {} for gpu_key in GPUS}
    for path in paths:
        measurements: dict = json.loads(path.read_text())
        if measurements["gpu"] not in GPUS:
            raise click.BadParameter(f"{path}: gpu must be one of {', '.join(GPUS)}.")
        per_name: dict[str, dict[str, float]] = elapsed_us_per_gpu[measurements["gpu"]]
        for name, elapsed_us in measurements["elapsed_us"].items():
            missing_keys: list[str] = [w.key for w in WORKLOADS if w.key not in elapsed_us]
            if name in per_name or missing_keys:
                raise click.BadParameter(f"{path}: {name!r} is repeated or lacks {missing_keys}.")
            per_name[name] = elapsed_us
    return elapsed_us_per_gpu


def _compute_rows(gpu_key: str, elapsed_us_per_name: dict[str, dict[str, float]]) -> list[dict]:
    """`_compute_rows` gives every workload's counts, bound, roofline and measured points on a GPU.

    Args:
        gpu_key: Key into `GPUS`.
        elapsed_us_per_name: Implementation -> workload key -> microseconds measured on that GPU.

    Returns:
        One dict per workload with its FLOP, bytes, intensity, the time each would take at the
        spec peak, the bound, the roofline TFLOP/s, and each implementation's time, achieved
        TFLOP/s and percent of the roofline.

    Example:
        >>> rows = _compute_rows("h200-sxm", {"SGLang": {"d1s": 20.91, ...}})
    """
    gpu: Gpu = GPUS[gpu_key]
    rows: list[dict] = []
    for workload in WORKLOADS:
        flop, byte_count = _count_flop_and_bytes(workload)
        flop_per_byte: float = flop / byte_count
        peak_tflops: float = gpu.peak_dense_bf16_tflops_according_to_nvidia_spec
        hbm_tbytes_per_s: float = gpu.hbm_tbytes_per_s_according_to_nvidia_spec
        compute_us: float = flop / (peak_tflops * FLOP_PER_TFLOP) * US_PER_S
        memory_us: float = byte_count / (hbm_tbytes_per_s * BYTES_PER_TBYTE) * US_PER_S
        roof_tflops: float = min(peak_tflops, flop_per_byte * hbm_tbytes_per_s)
        elapsed_us: dict[str, float] = {
            name: us[workload.key] for name, us in elapsed_us_per_name.items()
        }
        achieved_tflops: dict[str, float] = {
            name: flop / (us / US_PER_S) / FLOP_PER_TFLOP for name, us in elapsed_us.items()
        }
        rows.append(
            {
                "gpu": gpu_key,
                "workload": workload.key,
                "gflop": flop / FLOP_PER_GFLOP,
                "mbytes": byte_count / BYTES_PER_MBYTE,
                "flop_per_byte": flop_per_byte,
                "compute_us_at_peak": compute_us,
                "memory_us_at_peak": memory_us,
                "bound": "memory" if memory_us > compute_us else "compute",
                "roof_tflops": roof_tflops,
                "elapsed_us": elapsed_us,
                "achieved_tflops": achieved_tflops,
                "roof_pct": {n: PCT_PER_FRAC * t / roof_tflops for n, t in achieved_tflops.items()},
            }
        )
    return rows


def _format_table(rows: list[dict]) -> str:
    """`_format_table` renders rows as a fixed-width table, one line per GPU and shape.

    Args:
        rows: Rows from `_compute_rows`.

    Returns:
        The table, ending each line with every implementation's TFLOP/s and percent of the roof.

    Example:
        >>> click.echo(_format_table(rows))
    """
    lines: list[str] = [
        f"{'gpu':9} {'shape':5} {'GFLOP':>8} {'MB':>7} {'FLOP/B':>6} {'compute us':>10} "
        f"{'memory us':>9} {'bound':>7} {'roof TFLOP/s':>12}  measured TFLOP/s (% of roof)"
    ]
    for row in rows:
        lines.append(
            f"{row['gpu']:9} {row['workload']:5} {row['gflop']:8.3f} {row['mbytes']:7.1f} "
            f"{row['flop_per_byte']:6.1f} {row['compute_us_at_peak']:10.2f} "
            f"{row['memory_us_at_peak']:9.2f} {row['bound']:>7} {row['roof_tflops']:12.1f}  "
            + "  ".join(
                f"{name} {tflops:.2f} ({row['roof_pct'][name]:.1f} %)"
                for name, tflops in row["achieved_tflops"].items()
            )
        )
    return "\n".join(lines)


@click.command(epilog="Example: uv run --script scripts/roofline_attention_decode.py")
@click.option(
    "--measurements-file",
    "measurements_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Measurements JSON, repeatable (default: muse_attention_decode/measurements/*.json).",
)
@click.option("--label", default=None, help="Suffix for the run folder's name (default: none).")
@click.option("--json", "should_print_json", is_flag=True, help="Print one JSON object.")
def main(measurements_paths: tuple[Path, ...], label: str | None, should_print_json: bool) -> None:
    """`main` writes the roofline table of every GPU in `GPUS` to a new run folder."""
    if label is not None and not LABEL_PATTERN.match(label):
        raise click.BadParameter(f"--label {label!r} must match {LABEL_PATTERN.pattern}.")
    paths: tuple[Path, ...] = measurements_paths or tuple(sorted(MEASUREMENTS_DIR.glob("*.json")))
    elapsed_us_per_gpu: dict[str, dict[str, dict[str, float]]] = _load_measurements(paths)
    run_id: str = f"{datetime.now(UTC):%Y%m%d-%H%M%S}" + (f"-{label}" if label else "")
    run_dir: Path = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True)
    git_describe: list[str] = ["git", "describe", "--always", "--dirty"]
    git_commit: str = subprocess.check_output(git_describe, cwd=CHALLENGE_DIR, text=True).strip()
    (run_dir / "command.txt").write_text(f"{shlex.join(sys.argv)}\ngit: {git_commit}\n")
    rows: list[dict] = [
        row
        for gpu_key, times in elapsed_us_per_gpu.items()
        for row in _compute_rows(gpu_key, times)
    ]
    output: dict = {"run_id": run_id, "measurements_files": [str(p) for p in paths], "rows": rows}
    output["sources"] = SOURCES
    (run_dir / "rows.json").write_text(json.dumps(output, indent=2))
    if should_print_json:
        click.echo(json.dumps(output))
        return
    click.echo(_format_table(rows))
    click.echo(f"run {run_id}: {run_dir / 'rows.json'}")


if __name__ == "__main__":
    main()
