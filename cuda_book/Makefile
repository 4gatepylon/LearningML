CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ARTIFACTS := .artifacts

# Usage: make run FILE=vector_add_cpu   (the .cc extension is omitted)
FILE ?=

.PHONY: help run clean

help:
	@echo "Targets:"
	@echo "  make help             Show this message"
	@echo "  make run FILE=<name>  Build <name>.cc and run it (omit the .cc extension)"
	@echo "  make clean            Remove $(ARTIFACTS)/"
	@echo ""
	@echo "Example: make run FILE=vector_add_cpu"
	@echo "Object files and binaries are written to $(ARTIFACTS)/"

$(ARTIFACTS):
	@mkdir -p $(ARTIFACTS)

$(ARTIFACTS)/%.o: %.cc | $(ARTIFACTS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ARTIFACTS)/%: $(ARTIFACTS)/%.o
	$(CXX) $(CXXFLAGS) $< -o $@

run:
	@test -n "$(FILE)" || { echo "error: FILE is required, e.g. make run FILE=vector_add_cpu"; exit 1; }
	@test -f "$(FILE).cc" || { echo "error: $(FILE).cc not found"; exit 1; }
	@$(MAKE) --no-print-directory $(ARTIFACTS)/$(FILE)
	./$(ARTIFACTS)/$(FILE)

clean:
	rm -rf $(ARTIFACTS)
