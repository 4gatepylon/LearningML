# nvcc comes from PATH and the GPU from CUDA_VISIBLE_DEVICES (both set by `source .env`).
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=native
# gtest_main supplies main() only to binaries that don't define their own.
LIBS := -lcublas -lgtest_main -lgtest -lpthread
BUILD := .build
HEADERS := $(wildcard common/*.cuh)
SOURCES := $(wildcard kernels/*.cu tests/*.cu)
NAMES := $(basename $(notdir $(SOURCES)))
vpath %.cu kernels tests

# Lint/format tools come from cuda_book's uv environment (`uv sync` there), configured by the
# .clang-format/.clang-tidy files here (copies of cuda_book's). Override to use system installs.
CLANG_FORMAT ?= uv run --project ../cuda_book clang-format
CLANG_TIDY ?= uv run --project ../cuda_book clang-tidy
# clang-tidy can't always find libstdc++ headers on its own, so point it at g++'s install.
GCC_INSTALL := --gcc-install-dir=$(shell dirname $$(g++ -print-libgcc-file-name))
# .cu files are linted as host-only CUDA; the toolkit root is derived from nvcc on PATH.
CUDA_PATH ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc))))

.PHONY: help test fmt tidy clean
.PRECIOUS: $(BUILD)/%

help:
	@echo "make test-<name>   Build kernels/<name>.cu or tests/<name>.cu and run its tests"
	@echo "make test          Build and run every kernel and test"
	@echo "make fmt           Format .cu/.cuh files in place (clang-format)"
	@echo "make tidy          Lint .cu files and the headers they include (clang-tidy; needs nvcc)"
	@echo "make clean         Remove $(BUILD)/"
	@echo "Pass extra flags to a binary with MATMUL_ARGS='...' (e.g. --gtest_filter=...)."

$(BUILD)/%: %.cu $(HEADERS)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I. $< -o $@ $(LIBS)

test-%: $(BUILD)/%
	./$< $(MATMUL_ARGS)

test: $(addprefix test-,$(NAMES))

fmt:
	$(if $(SOURCES)$(HEADERS),$(CLANG_FORMAT) -i $(SOURCES) $(HEADERS))

tidy:
	$(if $(SOURCES),$(CLANG_TIDY) $(SOURCES) -- -x cuda --cuda-path=$(CUDA_PATH) --cuda-host-only -nocudalib -std=c++17 -I. $(GCC_INSTALL))

clean:
	rm -rf $(BUILD)
