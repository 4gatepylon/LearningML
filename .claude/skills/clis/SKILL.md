---
name: clis
description: Design command-line interfaces in this repo - Python scripts (click) and C, C++ and CUDA mains (getopt_long) - so they are safe by default and use the same flags everywhere. Use whenever you write or change a script's or main's arguments, flags, prompts or output, or add a destructive, expensive or cached operation (including a Makefile target that wraps one).
---

# CLIs

Every CLI here is run by people and by agents. So a CLI is safe by default, says exactly what it
is about to do, never hangs waiting for input, and uses the same vocabulary as every other CLI in
the repo. The background is Trevin Chow's
[10 Principles for Agent-Native CLIs](https://x.com/trevin/status/2051316002730991795); where it
and this skill differ, this skill wins.

## Libraries

- **Python:** always [click](https://click.palletsprojects.com/), never `argparse` or `sys.argv`.
- **C, C++ and CUDA mains:** always `getopt_long` (`<getopt.h>`, part of glibc), with long
  options. No other parser.

## Defaults

- **Non-interactive:** a CLI never prompts unless it gets `--interactive`, so it can't hang
  under an agent. Where it would have asked, it fails instead (see Destructive actions).
- **Idempotent:** running a command twice with the same arguments gives the same result and
  destroys nothing. A new run writes a new run folder (`.artifacts/<script>/<run_id>/`, see
  "Uncommitted files" in the root `CLAUDE.md`) and never replaces an old one.
- **No cache:** a run computes everything fresh unless it gets `--reuse-cache` (see Caching).
- **Bounded output:** see Output.

## Validate first

- A CLI validates every argument before any side effect: before it writes a file, starts GPU
  work or downloads anything.
- Each check is a click type (`click.IntRange`, `click.Choice`, `click.Path(exists=True)`) or its
  own `_validate_<argument>` function, so checks are small, reusable and testable.
- Checks that involve several arguments (e.g. `--min-batch-size` ≤ `--max-batch-size`) go in one
  `_validate_args` that `main` calls first.
- An error names the argument, says what was wrong, lists the valid values or range (all of them
  when it is a set), and shows a working invocation.

## Destructive actions

A destructive action deletes or overwrites something that is costly to recreate: a dataset,
weights, a cache, a results folder.

- Each destructive action has its own kebab-case flag, `--yes-to-<action>`:
  `--yes-to-overwrite-dataset`, `--yes-to-delete-checkpoints`. There is no blanket `--yes` or
  `--force`.
- The warning says exactly what will happen (which paths, how many files, what recreating them
  costs) and names the `--yes-to-<action>` flag that skips the question.
- Without its flag and without `--interactive`, the CLI prints the warning and fails before doing
  anything.
- With `--interactive`, the CLI prints the warning and asks yes/no. Any answer but yes aborts it
  (`click.confirm(..., abort=True)`; in C, a non-zero exit).
- Every destructive or expensive CLI has `--dry-run`: it validates, prints what it would do, and
  changes nothing.
- The order in `main` is: validate, plan, stop here on `--dry-run`, confirm, act.

```python
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
        >>> _confirm_destructive_action(warning, "--yes-to-overwrite-dataset", False, True)
    """
    message: str = f"{warning}\nTo skip this question, pass {yes_flag}."
    if is_confirmed:
        return
    if not is_interactive:
        raise click.UsageError(f"{message}\nTo be asked instead, pass --interactive.")
    click.confirm(f"{message}\nContinue?", abort=True)


_confirm_destructive_action(
    f"This deletes {dataset_dir} ({file_count} files) and downloads it again (about 2 h).",
    "--yes-to-overwrite-dataset",
    yes_to_overwrite_dataset,
    interactive,
)
```

A Makefile target that wraps a destructive action takes the same switches as variables:
`make reset-artifacts YES_TO_RESET_ARTIFACTS=1` or `make reset-artifacts INTERACTIVE=1`.

## Output

- Results go to stdout. Progress, warnings and errors go to stderr. Colors only when stdout is a
  terminal.
- Every command that returns data takes `--json`, which prints the same results as a single JSON
  object on stdout instead of the human-readable table.
- Every command that creates something prints its identifier (run ID, cache handle or output
  path), also in its `--json` output.
- Output is bounded. A list prints at most `--limit` rows (with a default), and when it truncates
  it says so and how to see more.
- Output that is long by nature goes to a file in the run folder, and stdout gets a summary plus
  the file's path. `--output-file <path>` writes it elsewhere, and `--allow-long-output` prints all
  of it to stdout.
- Exit codes: 0 on success, 1 when the work fails or a prompt is declined, 2 on a usage error (bad
  arguments, or a missing `--yes-to-<action>`). These match click's defaults.

## Caching

- A cache is never used by default: every run computes fresh.
- A run whose result can be reused keeps it in its run folder and prints a cache handle: its run
  ID (`cache_handle` in `--json`).
- `--reuse-cache <handle>` uses that run's result instead of computing it. It fails if the
  handle doesn't exist or the run's arguments (its `command.txt`) differ from this call's.
- `--overwrite-cache <handle>` recomputes into that run's folder. That's destructive, so it also
  needs `--yes-to-overwrite-cache` or `--interactive`.

## Vocabulary

The same concept has the same name in every CLI:

| Concept | Name | Never |
| --- | --- | --- |
| Ask before destructive actions | `--interactive` | `-i`, `--prompt`, `--confirm` |
| Skip one confirmation | `--yes-to-<action>` | `--yes`, `-y`, `--force`, `--skip-confirmations` |
| Show the plan, change nothing | `--dry-run` | `-n`, `--plan`, `--simulate` |
| Structured output | `--json` | `--format json`, `--output json` |
| Reuse a cached result | `--reuse-cache <handle>` | `--cache`, `--use-cache` |
| Recompute a cached result | `--overwrite-cache <handle>` | `--refresh`, `--no-cache` |
| Cap the rows printed | `--limit <count>` | `--max`, `--top`, `--head` |
| Print long output in full | `--allow-long-output` | `--all`, `--full`, `--verbose` |
| Write output to a file | `--output-file <path>` | `-o`, `--out` |
| Subcommand verbs | `get`, `list`, `create`, `update`, `delete` | `info`, `ls`, `new`, `rm` |

- Flags are kebab-case long options. The only short options are `-h` and the math letters of
  the root `CLAUDE.md` (`-N`, `-K`).
- A new concept needs the user's agreement; add its row to this table in the same change.

## Help

- `--help` lists every flag with its default and ends with an example invocation (click's
  `epilog`). It matches the Interface section of the file's docstring.
