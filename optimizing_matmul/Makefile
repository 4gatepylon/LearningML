# nvcc comes from PATH and the GPU from CUDA_VISIBLE_DEVICES (both set by `source .env`).
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=native
LIBS := -lcublas -lgtest -lpthread
BUILD := .build
SOURCES := $(wildcard *.cu *.cuh)
# Lint/format tools come from cuda_book's uv environment (`uv sync` there), configured by the
# .clang-format/.clang-tidy files here (copies of cuda_book's). Override to use system installs.
CLANG_FORMAT ?= uv run --project ../cuda_book clang-format
CLANG_TIDY ?= uv run --project ../cuda_book clang-tidy
# clang-tidy can't always find libstdc++ headers on its own, so point it at g++'s install.
GCC_INSTALL := --gcc-install-dir=$(shell dirname $$(g++ -print-libgcc-file-name))
# .cu files are linted as host-only CUDA; the toolkit root is derived from nvcc on PATH.
CUDA_PATH ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc))))

.PHONY: help fmt tidy clean
.PRECIOUS: $(BUILD)/%

help:
	@echo "make run-<name> [MATMUL_ARGS='--n 4096 --tries 50']   Build <name>.cu, test it, then benchmark it"
	@echo "make test-<name>                                     Build <name>.cu and run only the tests"
	@echo "make fmt                                             Format .cu/.cuh files in place (clang-format)"
	@echo "make tidy                                            Lint .cu files (clang-tidy; needs nvcc on PATH)"
	@echo "make clean                                           Remove $(BUILD)/"

$(BUILD)/%: %.cu harness.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

run-%: $(BUILD)/%
	./$< $(MATMUL_ARGS)

test-%: $(BUILD)/%
	./$< --no-bench $(MATMUL_ARGS)

fmt:
	$(CLANG_FORMAT) -i $(SOURCES)

tidy:
	$(CLANG_TIDY) $(wildcard *.cu) -- -x cuda --cuda-path=$(CUDA_PATH) --cuda-host-only -nocudalib -std=c++17 $(GCC_INSTALL)

clean:
	rm -rf $(BUILD)
