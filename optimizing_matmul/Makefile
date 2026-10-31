# `Makefile` builds and runs the tests and the benchmark, and formats and lints the code.
#
# Interface:
#     make test: build and run the GoogleTest tests.
#     make bench-kernels [BENCH_ARGS="-N 4096"]: build the benchmark and print its table.
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

.PHONY: test bench-kernels fmt tidy clean

test: $(BUILD_DIR)/test
	$(BUILD_DIR)/test

bench-kernels: $(BUILD_DIR)/bench
	uv run scripts/bench.py $(BENCH_ARGS)

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
