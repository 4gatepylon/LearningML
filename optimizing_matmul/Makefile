# `Makefile` builds and runs the tests and the benchmark, and formats and lints the code.
#
# Interface:
#     make help: list the targets.
#     make test: build and run the GoogleTest tests.
#     make bench-kernels [BENCH_ARGS="-N 4096"]: build the benchmark and print its table.
#     make scan-batch-sizes [SCAN_ARGS="--kernel cublas"]: time a range of batch sizes and fit the fixed cost.
#     make fmt, make tidy: format and lint with clang-format, clang-tidy and ruff.
#     make clean: remove .build/.
#
# Example:
#     source .env && make test
#
# Must know:
#     Gotchas:
#         - nvcc comes from PATH, so `source .env` must run first.
#         - The dev tools come from uv; set CLANG_FORMAT, CLANG_TIDY or RUFF to use others.
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -arch=native -Xcompiler=-Wall,-Wextra -I.
BUILD_DIR := .build
CLANG_FORMAT ?= uv run clang-format
CLANG_TIDY ?= uv run clang-tidy
RUFF ?= uv run ruff

KERNEL_SOURCES := $(wildcard kernels/*.cu)
HEADER_FILES := $(wildcard common/*.h)
HOST_SOURCES := $(wildcard host/*.cc tests/*.cc)

.PHONY: help test bench-kernels scan-batch-sizes fmt tidy clean

help:
	@echo "Targets:"
	@echo "  make help                                   List the targets"
	@echo "  make test                                   Build and run the GoogleTest tests"
	@echo "  make bench-kernels [BENCH_ARGS=\"-N 4096\"]   Time each kernel and print a table"
	@echo "  make scan-batch-sizes [SCAN_ARGS=\"...\"]     Time a range of batch sizes; fit the fixed cost"
	@echo "  make fmt                                    Format C++/CUDA (clang-format) and Python (ruff)"
	@echo "  make tidy                                   Lint C++/CUDA (clang-tidy) and Python (ruff)"
	@echo "  make clean                                  Remove $(BUILD_DIR)/"
	@echo ""
	@echo "Example: source .env && make bench-kernels BENCH_ARGS=\"--kernel cublas\""
	@echo "Flags for BENCH_ARGS and SCAN_ARGS: uv run scripts/bench.py -h, uv run scripts/scan_batch_size.py -h"

test: $(BUILD_DIR)/test
	$(BUILD_DIR)/test

bench-kernels: $(BUILD_DIR)/bench
	uv run scripts/bench.py $(BENCH_ARGS)

scan-batch-sizes: $(BUILD_DIR)/bench
	uv run scripts/scan_batch_size.py $(SCAN_ARGS)

$(BUILD_DIR)/test: tests/test_matmul.cc $(KERNEL_SOURCES) $(HEADER_FILES)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(filter-out %.h,$^) -o $@ -lcublas -lgtest -lgtest_main -lpthread

$(BUILD_DIR)/bench: host/bench.cc $(KERNEL_SOURCES) $(HEADER_FILES)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(filter-out %.h,$^) -o $@ -lcublas

fmt:
	$(CLANG_FORMAT) -i $(KERNEL_SOURCES) $(HEADER_FILES) $(HOST_SOURCES)
	$(RUFF) format .
	$(RUFF) check --select I --fix .

CUDA_PATH ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc))))
TIDY_FLAGS := -std=c++17 -I. --gcc-install-dir=$(shell dirname $$(g++ -print-libgcc-file-name))

tidy:
	$(CLANG_TIDY) --header-filter=common/ $(HOST_SOURCES) -- $(TIDY_FLAGS) \
	  -isystem $(CUDA_PATH)/include
	$(CLANG_TIDY) $(KERNEL_SOURCES) -- -x cuda --cuda-path=$(CUDA_PATH) --cuda-host-only \
	  -nocudalib $(TIDY_FLAGS)
	$(RUFF) check .

clean:
	rm -rf $(BUILD_DIR)
