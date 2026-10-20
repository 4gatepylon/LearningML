CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ARTIFACTS := .artifacts
# Dev tools come from the uv environment (see pyproject.toml); override to use system installs.
CLANG_FORMAT ?= uv run clang-format
CLANG_TIDY ?= uv run clang-tidy
RUFF ?= uv run ruff
SOURCES := $(wildcard *.cc *.cu *.cuh *.h)

# No implicit rules: `make run vector_add` must not try to build `vector_add` itself.
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

.PHONY: help run fmt tidy clean

help:
	@echo "Targets:"
	@echo "  make help             Show this message"
	@echo "  make run [name ...]   Build and run <name>.cc (.cc optional); with no name, run every .cc"
	@echo "  make fmt              Format C++ (clang-format) and Python (ruff) files in place"
	@echo "  make tidy             Lint C++ (clang-tidy) and Python (ruff) files"
	@echo "  make clean            Remove $(ARTIFACTS)/"
	@echo ""
	@echo "Example: make run vector_add"
	@echo "Binaries are written to $(ARTIFACTS)/ (see run.py)"

# `make run` just forwards to run.py; extra goals after `run` are its arguments.
run:
	@uv run ./run.py $(filter-out run,$(MAKECMDGOALS))

# Swallow the names given after `run` so make doesn't complain they have no rule.
ifneq ($(filter run,$(MAKECMDGOALS)),)
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
