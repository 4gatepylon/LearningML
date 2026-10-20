CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ARTIFACTS := .artifacts
# Dev tools come from the uv environment (see pyproject.toml); override to use system installs.
CLANG_FORMAT ?= uv run clang-format
CLANG_TIDY ?= uv run clang-tidy
RUFF ?= uv run ruff
SOURCES := $(wildcard *.cc *.cu *.cuh *.h)

# No implicit rules: `make test vector_add` must not try to build `vector_add` itself.
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

.PHONY: help test fmt tidy clean

help:
	@echo "Targets:"
	@echo "  make help             Show this message"
	@echo "  make test [name ...]  Build and test <name>.cc (.cc optional); with no name, test every .cc"
	@echo "  make fmt              Format C++ (clang-format) and Python (ruff) files in place"
	@echo "  make tidy             Lint C++ (clang-tidy) and Python (ruff) files"
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
tidy:
	$(CLANG_TIDY) $(wildcard *.cc) -- $(CXXFLAGS) --gcc-install-dir=$(shell dirname $$($(CXX) -print-libgcc-file-name))
	$(RUFF) check .

clean:
	rm -rf $(ARTIFACTS)
