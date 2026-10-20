CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ARTIFACTS := .artifacts
# Dev tools come from the uv environment (see pyproject.toml); override to use system installs.
CLANG_FORMAT ?= uv run clang-format
CLANG_TIDY ?= uv run clang-tidy
RUFF ?= uv run ruff
SOURCE_DIRS := . kernels
SOURCES := $(foreach d,$(SOURCE_DIRS),$(wildcard $(d)/*.cc $(d)/*.cu $(d)/*.cuh $(d)/*.h))

# No implicit rules: `make test vector_add` must not try to build `vector_add` itself.
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

.PHONY: help test fmt tidy clean

help:
	@echo "Targets:"
	@echo "  make help             Show this message"
	@echo "  make test [name ...]  Build and test <name>.cc (.cc optional); with no name, test every .cc"
	@echo "  make fmt              Format C++ (clang-format) and Python (ruff) files in place"
	@echo "  make tidy             Lint .cc/.cu (clang-tidy; .cu needs nvcc on PATH) and Python (ruff) files"
	@echo "  make clean            Remove $(ARTIFACTS)/"
	@echo ""
	@echo "Example: make test vector_add"
	@echo "Binaries are written to $(ARTIFACTS)/ (see test.py)"

# `make test` just forwards to test.py; extra goals after `test` are its arguments.
test:
	@uv run ./test.py $(filter-out test,$(MAKECMDGOALS))

# Swallow the names given after `test` so make doesn't complain they have no rule.
ifneq ($(filter test,$(MAKECMDGOALS)),)
%:
	@:
endif

fmt:
	$(CLANG_FORMAT) -i $(SOURCES)
	$(RUFF) format .
	$(RUFF) check --select I --fix .

# clang-tidy can't always find libstdc++ headers on its own, so point it at g++'s install.
GCC_INSTALL := --gcc-install-dir=$(shell dirname $$($(CXX) -print-libgcc-file-name))
# .cu files are linted as host-only CUDA; the toolkit root is derived from nvcc on PATH.
CUDA_PATH ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc))))
CC_SOURCES := $(strip $(foreach d,$(SOURCE_DIRS),$(wildcard $(d)/*.cc)))
CU_SOURCES := $(strip $(foreach d,$(SOURCE_DIRS),$(wildcard $(d)/*.cu)))

tidy:
	$(if $(CC_SOURCES),$(CLANG_TIDY) $(CC_SOURCES) -- $(CXXFLAGS) $(GCC_INSTALL))
	$(if $(CU_SOURCES),$(CLANG_TIDY) $(CU_SOURCES) -- -x cuda --cuda-path=$(or $(CUDA_PATH),$(error nvcc not found on PATH; source .env or set CUDA_PATH)) --cuda-host-only -nocudalib $(CXXFLAGS) $(GCC_INSTALL))
	$(RUFF) check .

clean:
	rm -rf $(ARTIFACTS)
