# `Makefile` regenerates the okc starters and formats this folder's scripts.
#
# Interface:
#     help: Lists the targets.
#     regenerate-starters: Reruns okc init for every problem and writes or reports differences.
#         REGENERATE_STARTERS_ARGS: Extra flags, e.g. "--dry-run" or "--json".
#         YES_TO_OVERWRITE_CHANGED_FILES=1: Overwrites the files that differ from okc init's output.
#         INTERACTIVE=1: Asks before overwriting them.
#     fmt: Formats and lints scripts/ with ruff.
#
# Example:
#     $ make regenerate-starters REGENERATE_STARTERS_ARGS=--dry-run

REGENERATE_STARTERS_ARGS ?=
REGENERATE_STARTERS_FLAGS := $(REGENERATE_STARTERS_ARGS) \
  $(if $(filter 1,$(YES_TO_OVERWRITE_CHANGED_FILES)),--yes-to-overwrite-changed-files) \
  $(if $(filter 1,$(INTERACTIVE)),--interactive)

.PHONY: help regenerate-starters fmt

help:
	@echo "make regenerate-starters  Reruns okc init for every problem and compares with ours."
	@echo "    REGENERATE_STARTERS_ARGS=\"--dry-run\"  Extra flags (--dry-run, --json)."
	@echo "    YES_TO_OVERWRITE_CHANGED_FILES=1      Overwrites files that differ from okc init's."
	@echo "    INTERACTIVE=1                         Asks before overwriting them."
	@echo "make fmt                  Formats and lints scripts/ with ruff."
	@echo "Example: make regenerate-starters REGENERATE_STARTERS_ARGS=--dry-run"

regenerate-starters:
	uv run --script scripts/regenerate_starters.py $(REGENERATE_STARTERS_FLAGS)

fmt:
	uvx ruff format --line-length 100 scripts
	uvx ruff check --line-length 100 --select E,F,I,B,UP --fix scripts
