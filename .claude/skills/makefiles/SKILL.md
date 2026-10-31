---
name: makefiles
description: Write or change a Makefile in this repo - which targets it must have and how to name them - for C, C++ and CUDA folders (kernel folders especially) and for Python projects. Use whenever you add a Makefile, add, rename or document a Makefile target, or tell the user which make command to run.
---

# Makefiles

A folder's Makefile is its entry point: `make help` says how to use it, and every other target
says what it does and what it does it to, so `make <target>` reads without opening the Makefile.

## Every Makefile

- `help` is the first target, so a plain `make` runs it. It lists every target with its arguments
  (e.g. `BENCH_ARGS="-N 4096"`) and a one-line description, and gives an example command.
- When a target is added, renamed or removed, `help` and the Makefile's `#` docstring change in the
  same commit.
- `clean` removes only build outputs that are git-ignored and can be rebuilt (e.g. `.build/`),
  never committed files, data or results.

## Required targets

C, C++ and CUDA folders, such as kernel folders (see the root `CLAUDE.md`):

- `make help`: lists the targets.
- `make test`: builds and runs all the tests.
- `make fmt`: formats the folder's files (clang-format and ruff).
- `make tidy`: lints the folder's files (clang-tidy and ruff).
- `make clean`: removes the build outputs (`.build/`).

Python projects:

- `make help`: lists the targets.
- `make test`: runs all the tests.
- `make fmt`: formats and lints the code with ruff (`ruff format` and `ruff check`), so a separate
  `tidy` is usually not needed.
- `make clean`: only when the project has build outputs to remove (`.build/`). It never removes
  `.data/` or `.artifacts/` (see "Uncommitted files" in the root `CLAUDE.md`).

ML training projects: the Python targets, plus:

- `make reset-artifacts`: deletes `.artifacts/` (weights, checkpoints, logs, results). Like any
  destructive CLI (`clis` skill), it fails by default with a warning that recreating
  `.artifacts/` may mean rerunning costly experiments. `YES_TO_RESET_ARTIFACTS=1` deletes it
  without asking; `INTERACTIVE=1` asks, and deletes only if the user types `yes`:

```make
RESET_ARTIFACTS_WARNING := This deletes .artifacts/ (weights, checkpoints, logs, results). \
  Recreating it may mean rerunning costly experiments. \
  To skip this question, pass YES_TO_RESET_ARTIFACTS=1.

reset-artifacts:
ifeq ($(YES_TO_RESET_ARTIFACTS),1)
	rm -rf .artifacts
else ifeq ($(INTERACTIVE),1)
	@printf '%s\nType yes to confirm: ' "$(RESET_ARTIFACTS_WARNING)"; read answer; [ "$$answer" = yes ]
	rm -rf .artifacts
else
	@printf '%s\nTo be asked instead, pass INTERACTIVE=1.\n' "$(RESET_ARTIFACTS_WARNING)" >&2; exit 2
endif
```

## Target names

- Every other target that a person runs (every `.PHONY` target) is `<verb>-<noun>`:
  `bench-kernels`, `scan-batch-sizes`, `plot-roofline`, `build-tests`.
- A bare verb such as `make bench`, `make run`, `make build` or `make plot` is not allowed: it
  doesn't say what it benchmarks, runs, builds or plots.
- The only bare verbs allowed are `help`, `test`, `fmt`, `tidy` and `clean`, with the meanings
  above. A new one needs the user's agreement; add it to this skill in the same change.

## Must know

- Gotchas:
  - File targets such as `$(BUILD_DIR)/bench` are paths, not commands, so they keep path names.
