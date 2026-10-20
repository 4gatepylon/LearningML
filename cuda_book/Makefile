CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ARTIFACTS := .artifacts
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
SOURCES := $(wildcard *.cc *.cu *.cuh *.h)

# No implicit rules: `make run vector_add` must not try to build `vector_add` itself.
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

.PHONY: help run fmt tidy clean

help:
	@echo "Targets:"
	@echo "  make help             Show this message"
	@echo "  make run [name ...]   Build and run <name>.cc (.cc optional); with no name, run every .cc"
	@echo "  make fmt              Format all .cc/.cu/.cuh/.h files in place with clang-format"
	@echo "  make tidy             Lint all .cc files with clang-tidy"
	@echo "  make clean            Remove $(ARTIFACTS)/"
	@echo ""
	@echo "Example: make run vector_add"
	@echo "Binaries are written to $(ARTIFACTS)/ (see run.py)"

# `make run` just forwards to run.py; extra goals after `run` are its arguments.
run:
	@./run.py $(filter-out run,$(MAKECMDGOALS))

# Swallow the names given after `run` so make doesn't complain they have no rule.
ifneq ($(filter run,$(MAKECMDGOALS)),)
%:
	@:
endif

fmt:
	$(CLANG_FORMAT) -i $(SOURCES)

# clang-tidy can't always find libstdc++ headers on its own, so point it at g++'s install.
tidy:
	$(CLANG_TIDY) $(wildcard *.cc) -- $(CXXFLAGS) --gcc-install-dir=$(shell dirname $$($(CXX) -print-libgcc-file-name))

clean:
	rm -rf $(ARTIFACTS)
