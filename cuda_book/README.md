I read through this book (edition 5): https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311

I did the exercises as I went through and wrote them down below in this folder.


## Prerequisites

`make fmt` and `make tidy` need `clang-format`, `clang-tidy` and `ruff` (for the Python files). Install the clang tools with apt:

```
sudo apt install clang-format
sudo apt install clang-tidy
```

or with pip (no sudo needed):

```
pip install clang-format clang-tidy ruff
```

Other install options exist too (conda, LLVM release tarballs, etc.).

## Running things

Each `.cc` file is a standalone executable with its own `main`. Build and run one with:

```
make run <name>    # the .cc extension is optional; a unique prefix (e.g. `vec`) also works
make run           # no name: build and run every .cc file
```

For example, `make run vector_add`. Run `make help` to see all targets.

Object files and binaries are stored in `./.artifacts/` (gitignored). `make clean` removes that folder.
