# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1"]
# ///
"""`regenerate_starters.py` reruns `okc init` for every problem and compares its files with ours.

`regenerate_starters.py` runs `okc init` for each problem that `okc problems` lists, into a
temporary folder, and compares every file it writes with the same file in `<problem>/` here. It
writes files that are missing, leaves identical files alone, and overwrites files that differ only
when asked. Right after the starters are committed, a run changes nothing, which shows that the
committed files are what `okc init` writes.

Interface:
    --dry-run: Print the plan and change nothing.
    --yes-to-overwrite-changed-files: Overwrite the files that differ from `okc init`'s output.
    --interactive: Ask before overwriting the files that differ.
    --json: Print the result as one JSON object.

Example:
    $ uv run --script scripts/regenerate_starters.py --dry-run

Must know:
    Gotchas:
        - `okc` must be on PATH and signed in (`okc login`).
        - An edited `submission.py` differs from its starter by design, so it is listed as changed.
    Side effects:
        - `okc init` downloads each problem's starter from openkernelchallenge.dev.
"""

import filecmp
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import click

WORK_DIR: Path = Path(__file__).resolve().parent.parent
OVERWRITE_FLAG: str = "--yes-to-overwrite-changed-files"


def _find_okc() -> str:
    """`_find_okc` returns the path of the `okc` executable.

    Returns:
        The path of `okc` on PATH.

    Raises:
        click.ClickException: `okc` is not on PATH.

    Example:
        >>> okc_path = _find_okc()
    """
    okc_path: str | None = shutil.which("okc")
    if okc_path is None:
        raise click.ClickException(
            "okc is not on PATH. Install it (see https://openkernelchallenge.dev), run "
            "`okc login`, then rerun: uv run --script scripts/regenerate_starters.py --dry-run"
        )
    return okc_path


def _run_okc(okc_path: str, args: list[str]) -> str:
    """`_run_okc` runs one `okc` command and returns what it printed.

    Args:
        okc_path: The path of `okc`.
        args: The arguments after `okc`.

    Returns:
        The command's stdout.

    Raises:
        click.ClickException: The command exited with an error.

    Example:
        >>> problems_json = _run_okc(okc_path, ["problems", "--json"])
    """
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [okc_path, *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise click.ClickException(f"okc {' '.join(args)} failed:\n{result.stderr}{result.stdout}")
    return result.stdout


def _plan_changes(generated_dir: Path) -> dict[str, list[str]]:
    """`_plan_changes` sorts every file `okc init` wrote into same, missing and changed.

    Args:
        generated_dir: The folder that holds one `okc init` output folder per problem.

    Returns:
        Paths relative to `WORK_DIR`, under the keys "same", "missing" and "changed".

    Example:
        >>> plan = _plan_changes(Path("/tmp/starters"))
    """
    plan: dict[str, list[str]] = {"same": [], "missing": [], "changed": []}
    for generated_path in sorted(path for path in generated_dir.rglob("*") if path.is_file()):
        relative_path: Path = generated_path.relative_to(generated_dir)
        work_path: Path = WORK_DIR / relative_path
        if not work_path.exists():
            plan["missing"].append(str(relative_path))
        elif filecmp.cmp(generated_path, work_path, shallow=False):
            plan["same"].append(str(relative_path))
        else:
            plan["changed"].append(str(relative_path))
    return plan


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
        >>> _confirm_destructive_action(warning, OVERWRITE_FLAG, False, True)
    """
    message: str = f"{warning}\nTo skip this question, pass {yes_flag}."
    if is_confirmed:
        return
    if not is_interactive:
        raise click.UsageError(f"{message}\nTo be asked instead, pass --interactive.")
    click.confirm(f"{message}\nContinue?", abort=True)


@click.command(epilog="Example: uv run --script scripts/regenerate_starters.py --dry-run")
@click.option("--dry-run", is_flag=True, help="Print the plan and change nothing.")
@click.option(
    OVERWRITE_FLAG,
    "yes_to_overwrite_changed_files",
    is_flag=True,
    help="Overwrite the files that differ from okc init's output.",
)
@click.option("--interactive", is_flag=True, help="Ask before overwriting the files that differ.")
@click.option("--json", "should_print_json", is_flag=True, help="Print the result as JSON.")
def main(
    dry_run: bool, yes_to_overwrite_changed_files: bool, interactive: bool, should_print_json: bool
) -> None:
    """`main` regenerates every problem's starter and writes or reports the differences."""
    okc_path: str = _find_okc()
    problems_json: dict = json.loads(_run_okc(okc_path, ["problems", "--json"]))
    problem_names: list[str] = [problem["name"] for problem in problems_json["problems"]]
    with tempfile.TemporaryDirectory() as temp_dir:
        generated_dir: Path = Path(temp_dir)
        for problem_name in problem_names:
            click.echo(f"okc init {problem_name}", err=True)
            _run_okc(okc_path, ["init", problem_name, str(generated_dir / problem_name)])
        plan: dict[str, list[str]] = _plan_changes(generated_dir)
        written_paths: list[str] = []
        if not dry_run:
            if plan["changed"]:
                _confirm_destructive_action(
                    f"This overwrites {len(plan['changed'])} files with okc init's output, losing "
                    f"their edits: {', '.join(plan['changed'])}.",
                    OVERWRITE_FLAG,
                    yes_to_overwrite_changed_files,
                    interactive,
                )
            written_paths = plan["missing"] + plan["changed"]
            for relative_path in written_paths:
                (WORK_DIR / relative_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(generated_dir / relative_path, WORK_DIR / relative_path)
    if should_print_json:
        click.echo(json.dumps({"problems": problem_names, **plan, "written": written_paths}))
        return
    click.echo(f"{len(plan['same'])} files match okc init's output in {WORK_DIR}.")
    for key in ("missing", "changed"):
        for relative_path in plan[key]:
            click.echo(f"{key}: {relative_path}")
    click.echo(f"Wrote {len(written_paths)} files." if not dry_run else "Dry run: wrote nothing.")


if __name__ == "__main__":
    main()
