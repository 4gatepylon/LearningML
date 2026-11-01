# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1"]
# ///
"""`okc_runs.py` sends kernels to the okc judge and keeps each run's files in its own folder.

Each run lives in `meta_muse_spark100k_challenge/.artifacts/okc_runs/<run_id>/`, with the file
sent (`submission.py`), `command.txt`, `run.json` (problem, mode, sha256, job id),
`submit.json` (okc's reply) and, once judged, okc's full `result.json`, `profile.txt` (GPU time
per kernel for each shape) and a small `summary.json`.

Interface:
    create FILE: Sends FILE to `okc check` (or `okc submit` with `--mode submit`) without waiting.
        --mode: `check` (default, off the board) or `submit` (scored, on the board).
        --label: Run label (default: FILE's stem).
        --min-left: Refuses when this many or fewer runs of that mode are left today (default 2).
        --yes-to-spend-scored-run, --interactive: Confirm `--mode submit`.
        --dry-run, --json.
    get RUN_ID: Waits for the run's job, saves `result.json` and `summary.json`, prints the summary.
        --no-wait, --json.
    list: Prints one row per run (run id, problem, mode, job, sha256, status, speedup, minutes).
        --limit: Newest runs to show (default 20).
        --json.

Example:
    $ uv run --script scripts/okc_runs.py create muse_bf16_proj/submission.py --label tiled
    $ uv run --script scripts/okc_runs.py get 20261101-003443-muse_bf16_proj-tiled
    $ uv run --script scripts/okc_runs.py list

Must know:
    Gotchas:
        - okc dedupes by content: resending a judged file returns `duplicate` and spends nothing.
        - `okc` must be on PATH and signed in (`okc login`).
    Side effects:
        - `okc status` rewrites profiles beside the file a job was first sent from.
"""

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click

ARTIFACTS_DIR: Path = Path(__file__).resolve().parents[2] / ".artifacts" / "okc_runs"
LIMIT_KEY_PER_MODE: dict[str, str] = {"check": "checks", "submit": "scored"}
FINAL_STATUSES: tuple[str, ...] = ("done", "failed", "rejected", "duplicate", "timeout")
PROBLEM_LINE_PATTERN: re.Pattern = re.compile(r"^# okc: problem=(\w+)$")
LABEL_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
US_PER_NS: float = 1e-3
S_PER_MIN: float = 60.0
SHA256_PREFIX_LENGTH: int = 12


def _run_okc(args: list[str], cwd: Path | None = None) -> dict:
    """`_run_okc` runs one `okc ... --json` command and returns its parsed JSON.

    Args:
        args: The arguments after `okc`, including `--json`.
        cwd: Folder to run in; okc writes profiles beside the file it sends.

    Returns:
        The JSON object okc printed.

    Raises:
        click.ClickException: `okc` is missing, or printed no JSON.

    Example:
        >>> problems = _run_okc(["problems", "--json"])["problems"]
    """
    okc_path: str | None = shutil.which("okc")
    if okc_path is None:
        raise click.ClickException("okc is not on PATH; install it and run `okc login`.")
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [okc_path, *args], cwd=cwd, capture_output=True, text=True
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise click.ClickException(
            f"okc {' '.join(args)} exited {result.returncode}:\n{result.stderr}{result.stdout}"
        ) from error


def _summarize_result(run: dict, job: dict) -> dict:
    """`_summarize_result` keeps the few KB of a judged job that a reader needs.

    Args:
        run: The run's `run.json`.
        job: okc's status JSON for the job that holds the result.

    Returns:
        Status, speedup, timings, test counts and one row per timed shape.

    Example:
        >>> summary = _summarize_result(run, job)
    """
    result: dict = job.get("result") or {}
    records: list[dict] = result.get("records") or []
    tests: list[dict] = [record for record in records if record["event"] == "test"]
    failures: list[dict] = [test for test in tests if test["status"] != "pass"]
    shapes: list[dict] = [
        {
            "load": record["case"].get("load"),
            "case": {k: v for k, v in record["case"].items() if k not in ("seed", "load")},
            "baseline_us": record["baseline_ns"] * US_PER_NS,
            "yours_us": record["median_ns"] * US_PER_NS,
            "floor_us": record["speed_of_light_ns"] * US_PER_NS,
            "speedup": record["speedup"],
        }
        for record in records
        if record["event"] == "benchmark"
    ]
    started_at: float | None = job.get("started_at")
    finished_at: float | None = job.get("finished_at")
    return {
        **{key: run[key] for key in ("run_id", "problem", "mode", "job_id")},
        "result_job_id": job.get("id"),
        "status": job.get("status"),
        "passed": result.get("passed"),
        "speedup": result.get("speedup"),
        "speedup_ci": result.get("speedup_ci"),
        "score_us": result.get("score_us"),
        "baseline_us": result.get("baseline_us"),
        "error": result.get("error"),
        "queue_min": started_at and (started_at - job["created_at"]) / S_PER_MIN,
        "run_min": finished_at and started_at and (finished_at - started_at) / S_PER_MIN,
        "tests_passed": len(tests) - len(failures),
        "tests_total": len(tests),
        "first_failure": failures[0] if failures else None,
        "shapes": shapes,
    }


def _confirm_destructive_action(
    warning: str, yes_flag: str, is_confirmed: bool, is_interactive: bool
) -> None:
    """`_confirm_destructive_action` returns only when the user agreed to a destructive action.

    Args:
        warning: What exactly will happen, and what undoing it costs.
        yes_flag: The `--yes-to-<action>` flag that skips the question.
        is_confirmed: Whether that flag was passed.
        is_interactive: Whether `--interactive` was passed.

    Raises:
        click.UsageError: Neither flag was passed (exit code 2).
        click.Abort: The user answered no (exit code 1).

    Example:
        >>> _confirm_destructive_action(warning, "--yes-to-spend-scored-run", False, True)
    """
    message: str = f"{warning}\nTo skip this question, pass {yes_flag}."
    if is_confirmed:
        return
    if not is_interactive:
        raise click.UsageError(f"{message}\nTo be asked instead, pass --interactive.")
    click.confirm(f"{message}\nContinue?", abort=True)


@click.group()
def cli() -> None:
    """`cli` groups the `create`, `get` and `list` commands."""


@cli.command(epilog="Example: okc_runs.py create muse_bf16_proj/submission.py --label tiled")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--mode", type=click.Choice(list(LIMIT_KEY_PER_MODE)), default="check")
@click.option("--label", default=None, help="Run label (default: FILE's stem).")
@click.option(
    "--min-left",
    type=click.IntRange(min=0),
    default=2,
    show_default=True,
    help="Refuse at or below this many runs left today.",
)
@click.option("--yes-to-spend-scored-run", is_flag=True, help="Confirm --mode submit.")
@click.option("--interactive", is_flag=True, help="Ask before spending a scored run.")
@click.option("--dry-run", is_flag=True, help="Print the plan and send nothing.")
@click.option("--json", "should_print_json", is_flag=True, help="Print the result as JSON.")
def create(
    file: Path,
    mode: str,
    label: str | None,
    min_left: int,
    yes_to_spend_scored_run: bool,
    interactive: bool,
    dry_run: bool,
    should_print_json: bool,
) -> None:
    """`create` sends FILE to the judge in a new run folder and prints the run id."""
    label = label or file.stem
    if not LABEL_PATTERN.match(label):
        raise click.BadParameter(
            f"{label!r} must match {LABEL_PATTERN.pattern}.", param_hint="--label"
        )
    first_line: str = file.read_text().splitlines()[0] if file.read_text() else ""
    problem_match: re.Match | None = PROBLEM_LINE_PATTERN.match(first_line)
    if problem_match is None:
        raise click.BadParameter(f"{file}'s first line must be `# okc: problem=<problem>`.")
    problem: str = problem_match.group(1)
    problems: dict[str, dict] = {p["name"]: p for p in _run_okc(["problems", "--json"])["problems"]}
    if problem not in problems:
        raise click.BadParameter(f"{problem!r} is not one of {', '.join(problems)}.")
    left_count: int = problems[problem]["limits"][LIMIT_KEY_PER_MODE[mode]]["left"]
    if left_count <= min_left:
        raise click.UsageError(f"Only {left_count} {mode} runs of {problem} are left today.")
    sha256: str = hashlib.sha256(file.read_bytes()).hexdigest()
    previous_run_ids: list[str] = sorted(
        path.parent.name
        for path in ARTIFACTS_DIR.glob("*/run.json")
        if json.loads(path.read_text())["sha256"] == sha256
    )
    run_id: str = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{problem}-{label}"
    plan: dict = {
        "run_id": run_id,
        "problem": problem,
        "mode": mode,
        "source": str(file.resolve()),
        "sha256": sha256,
        "left_today": left_count,
        "previous_run_ids": previous_run_ids,
    }
    if dry_run:
        click.echo(json.dumps(plan, indent=None if should_print_json else 2))
        return
    if mode == "submit":
        _confirm_destructive_action(
            f"This spends 1 of {left_count} scored runs of {problem} left today and puts "
            f"{file} on the public leaderboard.",
            "--yes-to-spend-scored-run",
            yes_to_spend_scored_run,
            interactive,
        )
    run_dir: Path = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True)
    shutil.copyfile(file, run_dir / "submission.py")
    git_commit: str = subprocess.run(
        ["git", "describe", "--always", "--dirty"], cwd=file.parent, capture_output=True, text=True
    ).stdout.strip()
    (run_dir / "command.txt").write_text(f"{shlex.join(sys.argv)}\ngit: {git_commit}\n")
    job: dict = _run_okc([mode, "submission.py", "--no-wait", "--json"], cwd=run_dir)
    (run_dir / "submit.json").write_text(json.dumps(job, indent=2))
    run: dict = {**plan, "label": label, "job_id": job["job_id"], "submitted_at": time.time()}
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))
    output: dict = {"run_id": run_id, "job_id": job["job_id"], "previous_run_ids": previous_run_ids}
    click.echo(
        json.dumps(output)
        if should_print_json
        else "\n".join(f"{key}: {value}" for key, value in output.items())
    )


@cli.command(epilog="Example: okc_runs.py get 20261101-003443-muse_bf16_proj-tiled")
@click.argument("run_id")
@click.option("--no-wait", is_flag=True, help="Report the job's state now instead of waiting.")
@click.option("--json", "should_print_json", is_flag=True, help="Print the summary as JSON.")
def get(run_id: str, no_wait: bool, should_print_json: bool) -> None:
    """`get` saves a run's result and summary, waiting for the job unless `--no-wait`."""
    run_dir: Path = ARTIFACTS_DIR / run_id
    if not (run_dir / "run.json").exists():
        raise click.BadParameter(f"{run_dir} has no run.json.", param_hint="RUN_ID")
    run: dict = json.loads((run_dir / "run.json").read_text())
    wait_args: list[str] = [] if no_wait else ["--wait"]
    job: dict = _run_okc(["status", run["job_id"], *wait_args, "--json"], cwd=run_dir)
    if job.get("status") == "duplicate":
        job = _run_okc(["status", job["duplicate_of"], *wait_args, "--json"], cwd=run_dir)
    if job.get("status") in FINAL_STATUSES:
        (run_dir / "result.json").write_text(json.dumps(job, indent=2))
        profile_reports: list[str] = [
            f"profile {record['index']}: {record['case']}\n{record['report']}"
            for record in (job.get("result") or {}).get("records") or []
            if record["event"] == "profile"
        ]
        (run_dir / "profile.txt").write_text("\n\n".join(profile_reports))
    summary: dict = _summarize_result(run, job)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if should_print_json:
        click.echo(json.dumps(summary))
        return
    rounded: dict = {
        key: round(value, 3) if isinstance(value, float) else value
        for key, value in summary.items()
    }
    click.echo(
        f"{run_id}: {summary['status']}, passed {summary['passed']}, "
        f"speedup {rounded['speedup']}, 95% {[round(x, 3) for x in summary['speedup_ci'] or []]}, "
        f"tests {summary['tests_passed']}/{summary['tests_total']}, "
        f"queue {rounded['queue_min']} min, run {rounded['run_min']} min"
    )
    for shape in summary["shapes"]:
        click.echo(
            f"  {shape['load']:<12} {shape['baseline_us']:9.1f} us -> "
            f"{shape['yours_us']:9.1f} us  {shape['speedup']:.3f}x  {shape['case']}"
        )
    if summary["first_failure"]:
        click.echo(f"first failure: {summary['first_failure']}")


@cli.command("list", epilog="Example: okc_runs.py list --limit 5")
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--json", "should_print_json", is_flag=True, help="Print the rows as JSON.")
def list_runs(limit: int, should_print_json: bool) -> None:
    """`list_runs` prints one row per run, newest last, from `run.json` and `summary.json`."""
    rows: list[dict] = []
    for run_path in sorted(ARTIFACTS_DIR.glob("*/run.json")):
        run: dict = json.loads(run_path.read_text())
        summary_path: Path = run_path.parent / "summary.json"
        summary: dict = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rows.append(
            {
                **{key: run[key] for key in ("run_id", "problem", "mode", "job_id")},
                "sha256": run["sha256"][:SHA256_PREFIX_LENGTH],
                **{key: summary.get(key) for key in ("status", "speedup", "queue_min", "run_min")},
            }
        )
    shown_rows: list[dict] = rows[-limit:]
    if should_print_json:
        click.echo(json.dumps({"runs": shown_rows, "total": len(rows)}))
        return
    if shown_rows:
        click.echo("  ".join(shown_rows[0]))
    for row in shown_rows:
        click.echo(
            "  ".join(
                f"{value:.3f}" if isinstance(value, float) else str(value) for value in row.values()
            )
        )
    if len(rows) > limit:
        click.echo(f"Showed the newest {limit} of {len(rows)} runs; pass --limit to see more.")


if __name__ == "__main__":
    cli()
