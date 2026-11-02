# Kernel versions

How we store and review our own kernels for each okc problem, so that every version we judged can
be found, rerun and compared, and every change reads as a small diff.

## Where kernels live

- Each meaningfully different kernel is one file, `<problem>/kernels/<NNN>-<slug>.py`:
  - `NNN` is the next free three-digit number within that problem (`001`, `002`, ...).
  - `<slug>` names what the kernel does differently from its parent, in kebab-case
    (`packed-decode`).
- Meaningfully different means the algorithm or launch structure changed, or the judged speed
  moved by more than the judge's 95% interval. A pure retune of constants that changes little
  stays in the same file before its PR merges, and becomes a new kernel after.
- A kernel file is a complete okc submission: its first line is `# okc: problem=<problem>`, and
  `okc check <file>` works on it as it is.
- Its docstring's first line says what it does, and its second paragraph starts with
  "Kernel NNN's parent is MMM." (or "Kernel NNN has no parent.").
- After its PR merges, a kernel file doesn't change. Fixes and improvements become a new kernel.
- `<problem>/submission.py` stays `okc init`'s starter (see `make regenerate-starters`).

## Adding a kernel

1. Copy the parent in its own commit: `cp kernels/MMM-*.py kernels/NNN-<slug>.py`, committed as
   "<problem>: copy kernel MMM as NNN-<slug>". A verbatim copy doesn't count toward the PR's
   reviewable lines.
2. Make the change in the next commit. Reviewers read that commit's diff, or
   `git diff --no-index kernels/MMM-*.py kernels/NNN-*.py`.
3. Check and time it locally against its parent:
   `python scripts/local_judge.py <problem>/kernels/NNN-<slug>.py --compare <problem>/kernels/MMM-*.py`.
4. Judge it with `uv run --script scripts/okc_runs.py create <problem>/kernels/NNN-<slug>.py
   --label NNN-<slug>`, then `get` the run.
5. Add its row to `<problem>/kernels/README.md`: parent, change, local load times against the
   parent, and the judge's speedup with its run id.
6. One kernel per PR, stacked on the PR of its parent.

Experiments that we measured and didn't keep get a line under "Tried and dropped" in the
problem's `kernels/README.md`, with what was tried and what it measured, but no file.
