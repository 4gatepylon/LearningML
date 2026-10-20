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

`nvcc` must be on your `PATH`. If `nvcc --version` says "command not found", the CUDA toolkit is installed but not on your `PATH` (this happens on some lab machines that Adriano uses). Run this from this folder in each new shell, or add it to your `~/.bashrc`:

```
source .env
```

`.env` is gitignored, so create it yourself if it's missing (adjust the CUDA version to one installed under `/usr/local`; it must be supported by your driver, see `nvidia-smi`):

```
export CUDA_HOME=/insert/your/path/here/cuda_version
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Running things

Each `.cc` file is a standalone executable with its own `main`. Build and run one with:

```
make run <name>    # the .cc extension is optional; a unique prefix (e.g. `vec`) also works
make run           # no name: build and run every .cc file
```

For example, `make run vector_add`. Run `make help` to see all targets.

Object files and binaries are stored in `./.artifacts/` (gitignored). `make clean` removes that folder.
