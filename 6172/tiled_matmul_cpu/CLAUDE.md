# tiled_matmul_cpu

This folder is only for matmul optimization on the CPU.

This folder is entirely Rust. All code, tests, benchmarks, and tooling here are written in Rust (use cargo). Do not add C, C++, or other languages.

## Goals

The user's goals for this folder are in `README.md`. Read it before starting work, and keep optimization work tied to them (measurement and cost-modeling, not only code).

## Commands

Prefer the `make` targets over raw cargo commands so flags stay consistent. Run `make help` to list them.

- `make check`: lint, doc and test. Run before committing.
- `make fmt`: format with rustfmt.
- `make lint`: `cargo fmt --check` and clippy with warnings as errors.
- `make doc`: build docs, failing on warnings such as broken links.
- `make test`: unit tests and doctests.

## Docs convention

- `///` on every public item: a one-sentence summary, then `# Panics` and `# Examples` sections where they apply. `#![warn(missing_docs)]` is on.
- `//!` documents the crate or module.
- `//` comments explain why, not what.
- Examples are doctests and run under `make test`, so keep them correct.

Template:

````rust
/// Computes `C = A * B`, overwriting `c` (it is not accumulated into).
///
/// `a` is `m x k`, `b` is `k x n` and `c` is `m x n`, all row-major.
///
/// # Panics
///
/// Panics if a slice length does not match its dimensions.
///
/// # Examples
///
/// ```
/// use tiled_matmul_cpu::matmul_naive;
///
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [5.0, 6.0, 7.0, 8.0];
/// let mut c = [0.0; 4];
/// matmul_naive(&a, &b, &mut c, 2, 2, 2);
/// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn matmul_naive(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    // Explain why here, not what.
}
````
