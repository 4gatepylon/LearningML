---
name: make-target-names
description: Name Makefile targets (make commands) in this repo. Use whenever you add, rename or document a Makefile target, write a new Makefile, or tell the user which make command to run.
---

# Make target names

A make command says what it does and what it does it to, so `make <target>` reads without opening
the Makefile.

## Rule

- Every target that a person runs (every `.PHONY` target) is `<verb>-<noun>`: `bench-kernels`,
  `run-naive`, `plot-roofline`, `build-tests`.
- A bare verb such as `make bench`, `make run`, `make build` or `make plot` is not allowed: it
  doesn't say what it benchmarks, runs, builds or plots.

## Exceptions

These are so common that the bare verb is clear on its own:

- `make help` lists the targets.
- `make test` builds and runs all the tests.
- `make fmt` formats the folder's files (clang-format and ruff).
- `make tidy` lints the folder's files (clang-tidy and ruff).
- `make clean` removes `.build/`.

A new exception needs the user's agreement; add it to this list in the same change.

## Must know

- Gotchas:
  - File targets such as `$(BUILD_DIR)/bench` are paths, not commands, so this rule doesn't apply.
