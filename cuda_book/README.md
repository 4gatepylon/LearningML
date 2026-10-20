I read through this book (edition 5): https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311

I did the exercises as I went through and wrote them down below in this folder.


## Prerequisites

Install [uv](https://docs.astral.sh/uv/), then run:

```
uv sync
```

This installs the dev tools (`ruff`, `clang-format`, `clang-tidy`) into `.venv`, and the Makefile
runs them through `uv run`. You also need a C++ compiler (`g++`).

Alternatively, install the tools yourself (e.g. `sudo apt install clang-format clang-tidy`, or
`pip install clang-format clang-tidy ruff`) and point the Makefile at them, e.g.
`make fmt CLANG_FORMAT=clang-format RUFF=ruff`. Other install options exist too.

## Running things

Each `.cc` file is a standalone executable with its own `main`. Build and run one with:

```
make run <name>    # the .cc extension is optional; a unique prefix (e.g. `vec`) also works
make run           # no name: build and run every .cc file
```

For example, `make run vector_add`. Run `make help` to see all targets.

Object files and binaries are stored in `./.artifacts/` (gitignored). `make clean` removes that folder.
