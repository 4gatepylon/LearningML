.DEFAULT_GOAL := help
.PHONY: help test fmt lint doc check

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-10s %s\n", $$1, $$2}'

test: ## Run all tests (unit and doc tests)
	cargo test

fmt: ## Format the code with rustfmt
	cargo fmt

lint: ## Check formatting and run clippy with warnings as errors
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

doc: ## Build the docs, failing on warnings such as broken links
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

check: lint doc test ## Run lint, doc and test (use before committing)
