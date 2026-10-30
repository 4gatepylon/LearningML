# Entry points; see README.md. Run `source .env` first so nvcc is on PATH.
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -arch=native -Xcompiler=-Wall,-Wextra -I.
BUILD := .build
# Dev tools come from the uv environment (see pyproject.toml); override to use system installs.
CLANG_FORMAT ?= uv run clang-format
CLANG_TIDY ?= uv run clang-tidy
RUFF ?= uv run ruff

KERNELS := $(wildcard kernels/*.cu)
HEADERS := $(wildcard common/*.h)
HOST := $(wildcard host/*.cc tests/*.cc)

.PHONY: test bench fmt tidy clean

test: $(BUILD)/test
	$(BUILD)/test

# e.g. `make bench ARGS="--n 4096"`; the empty default keeps out any ARGS set in the shell.
ARGS :=
bench: $(BUILD)/bench
	uv run scripts/bench.py $(ARGS)

$(BUILD)/test: tests/test_matmul.cc $(KERNELS) $(HEADERS)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(filter-out %.h,$^) -o $@ -lcublas -lgtest -lgtest_main -lpthread

$(BUILD)/bench: host/bench.cc $(KERNELS) $(HEADERS)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(filter-out %.h,$^) -o $@ -lcublas

fmt:
	$(CLANG_FORMAT) -i $(KERNELS) $(HEADERS) $(HOST)
	$(RUFF) format .
	$(RUFF) check --select I --fix .

# clang-tidy needs g++'s libstdc++ and the CUDA headers; the toolkit root comes from nvcc on PATH.
CUDA_PATH ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc))))
TIDY_FLAGS := -std=c++17 -I. --gcc-install-dir=$(shell dirname $$(g++ -print-libgcc-file-name))

tidy:
	$(CLANG_TIDY) --header-filter=common/ $(HOST) -- $(TIDY_FLAGS) -isystem $(CUDA_PATH)/include
	$(CLANG_TIDY) $(KERNELS) -- -x cuda --cuda-path=$(CUDA_PATH) --cuda-host-only -nocudalib $(TIDY_FLAGS)
	$(RUFF) check .

clean:
	rm -rf $(BUILD)
