These rules apply to the whole repository. Folders can add their own rules in their own `CLAUDE.md`
(e.g. `cuda_book/CLAUDE.md`).

## Stacked pull requests

- When a change splits into a sequence of dependent pieces, make one PR per logical piece and stack
  them. Each piece's branch is built on top of the previous piece's branch, and each PR's base is
  the branch below it, so every PR shows only its own piece. Each layer should build and pass its
  own tests.
- Before building a stack, tell the user the planned layers and how they will be stacked, and wait
  for their agreement.
- By default, stack with GitHub's native stacked PRs, using the `gh stack` CLI extension
  (`gh extension install github/gh-stack`). It needs Git 2.36 or later, and stacked PRs must be
  enabled for the repository.
  - `gh stack init <branch>` starts the stack with its first branch. Commit the first piece there.
  - `gh stack add <branch>` adds a branch on top of the stack. Commit the next piece there, and
    repeat for each piece.
  - `gh stack submit` pushes every branch and opens the linked PRs.
  - To change a lower piece (e.g. after review), switch to its branch with
    `gh stack checkout <branch>` or `gh stack down`, and commit there rather than in a higher layer.
    Then run `gh stack rebase --upstack` and `gh stack push`.
  - PRs merge bottom-up, with `gh stack merge` or from the merge box on GitHub. Afterwards,
    `gh stack sync --prune` rebases the remaining branches onto `main` and removes merged ones.
- If `gh stack` can't be used (the extension is missing, Git is older than 2.36, or it exits with
  code 9 because stacked PRs aren't enabled), tell the user and ask before falling back to stacking
  by hand: open the PRs with `gh pr create --base <branch below>`, then link them into a stack on
  GitHub.

## Kernel folders

A kernel folder (e.g. `optimizing_matmul/`) mixes three languages. CUDA (`.cu`) implements the
kernels. C++ (`.cc`) provides host-side stubs, such as launch wrappers and `main`s, and light
orchestration. Python (`.py`) is for scripts: building, orchestrating tests, benchmarking and
plotting.

Every kernel folder implements this tree. Entries marked "optional" may be left out; everything
else is required.

```
<folder>/
├── README.md                    # what it does and how to run it (present tense)
├── CLAUDE.md                    # optional: folder-specific rules
├── Makefile                     # entry points: make test, make fmt, make tidy, make clean
├── .gitignore                   # ignores .build/
├── .env.example                 # puts nvcc on PATH and sets CUDA_VISIBLE_DEVICES; copy to .env
├── .clang-format, .clang-tidy   # copies of cuda_book's (Google style, 100 columns)
├── pyproject.toml, ruff.toml    # uv project for the Python scripts and tools (ruff, clang-*)
├── common/                      # optional: shared code used by both kernels and host code
├── kernels/                     # .cu: the kernels, one kernel/task per file
│   └── common/                  # optional: shared CUDA code (.cuh)
├── host/                        # optional: .cc stubs and orchestration
│   └── common/                  # optional: shared host code (.h)
├── tests/                       # GoogleTest tests (.cu/.cc)
├── scripts/                     # .py: build/test orchestration, benchmarks, plots
└── .build/                      # build outputs (git-ignored)
```

- Before committing, run `make fmt` (clang-format and ruff) and `make tidy` (clang-tidy and ruff).
  Both must come back clean for the folder's own files.
- Never hardcode host-specific settings (CUDA version or path, GPU architecture). Take them from
  the environment (`.env`), and build with `-arch=native`.

> NOTE: `cuda_book/` is separate from this template and doesn't necessarily follow it. It keeps its
> own layout and rules (see `cuda_book/CLAUDE.md`).

> NOTE: if any of this is outdated notify the user.
