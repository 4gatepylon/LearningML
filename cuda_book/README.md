I read through this book (edition 5): https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311

I did the exercises as I went through and wrote them down below in this folder.


## Prerequisites

Install [uv](https://docs.astral.sh/uv/), then run:

```
uv sync
```

This installs the dev tools (`ruff`, `clang-format`, `clang-tidy`) into `.venv`, and the Makefile
runs them through `uv run`. You also need a C++ compiler (`g++`).

Tests use [GoogleTest](https://github.com/google/googletest), which uv can't install. On
Ubuntu/Debian:

```
sudo apt install libgtest-dev
```

On Ubuntu 22.04 this ships the headers plus the static `libgtest.a` and `libgtest_main.a`, so
`-lgtest -lgtest_main -pthread` links with no extra flags. Without sudo, you can use conda
(`conda install -c conda-forge gtest`), but then you must pass the include and library paths
(`-I$CONDA_PREFIX/include`, `-L$CONDA_PREFIX/lib`) yourself.

Alternatively, install the tools yourself (e.g. `sudo apt install clang-format clang-tidy`, or
`pip install clang-format clang-tidy ruff`) and point the Makefile at them, e.g.
`make fmt CLANG_FORMAT=clang-format RUFF=ruff`. Other install options exist too.

## Running things

Each `.cc` file is a standalone executable with its own `main`. Build and test one with:

```
make test <name>   # the .cc extension is optional; a unique prefix (e.g. `vec`) also works
make test           # no name: build and test every .cc file
```

For example, `make test vector_add`. Run `make help` to see all targets.

Object files and binaries are stored in `./.artifacts/` (gitignored). `make clean` removes that folder.
