# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1"]
# ///
"""`okc_runs.py` sends kernels to the okc judge and keeps each run's files in its own folder.

Each run lives in `meta_muse_spark100k_challenge/.artifacts/okc_runs/<run_id>/`, with the file
sent (`submission.py`), `command.txt`, `run.json` (problem, mode, sha256, job id) and
`submit.json` (okc's reply).

Interface:
    create FILE: Sends FILE to `okc check` (or `okc submit` with `--mode submit`) without waiting.
        --mode: `check` (default, off the board) or `submit` (scored, on the board).
        --label: Run label (default: FILE's stem).
        --min-left: Refuses when this many or fewer runs of that mode are left today (default 2).
        --yes-to-spend-scored-run, --interactive: Confirm `--mode submit`.
        --dry-run, --json.

Example:
    $ uv run --script scripts/okc_runs.py create muse_bf16_proj/submission.py --label tiled
    $ okc status <job_id> --wait

Must know:
    Gotchas:
        - okc dedupes by content: resending a judged file returns `duplicate` and spends nothing.
        - `okc` must be on PATH and signed in (`okc login`).
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
PROBLEM_LINE_PATTERN: re.Pattern = re.compile(r"^# okc: problem=(\w+)$")
LABEL_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9_.-]+$")


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
    """`cli` groups the `create` command."""


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


if __name__ == "__main__":
    cli()
