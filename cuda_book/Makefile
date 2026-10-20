CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ARTIFACTS := .artifacts
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
SOURCES := $(wildcard *.cc *.cu *.cuh *.h)

# Usage: make run FILE=vector_add   (the .cc extension is omitted)
FILE ?=

.PHONY: help run fmt tidy clean

help:
	@echo "Targets:"
	@echo "  make help             Show this message"
	@echo "  make run FILE=<name>  Build <name>.cc and run it (omit the .cc extension)"
	@echo "  make fmt              Format all .cc/.cu/.cuh/.h files in place with clang-format"
	@echo "  make tidy             Lint all .cc files with clang-tidy"
	@echo "  make clean            Remove $(ARTIFACTS)/"
	@echo ""
	@echo "Example: make run FILE=vector_add"
	@echo "Object files and binaries are written to $(ARTIFACTS)/"

$(ARTIFACTS):
	@mkdir -p $(ARTIFACTS)

$(ARTIFACTS)/%.o: %.cc | $(ARTIFACTS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ARTIFACTS)/%: $(ARTIFACTS)/%.o
	$(CXX) $(CXXFLAGS) $< -o $@

run:
	@test -n "$(FILE)" || { echo "error: FILE is required, e.g. make run FILE=vector_add"; exit 1; }
	@test -f "$(FILE).cc" || { echo "error: $(FILE).cc not found"; exit 1; }
	@$(MAKE) --no-print-directory $(ARTIFACTS)/$(FILE)
	./$(ARTIFACTS)/$(FILE)

fmt:
	$(CLANG_FORMAT) -i $(SOURCES)

# clang-tidy can't always find libstdc++ headers on its own, so point it at g++'s install.
tidy:
	$(CLANG_TIDY) $(wildcard *.cc) -- $(CXXFLAGS) --gcc-install-dir=$(shell dirname $$($(CXX) -print-libgcc-file-name))

clean:
	rm -rf $(ARTIFACTS)
