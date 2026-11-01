These rules apply to `optimizing_matmul/`, on top of the root `CLAUDE.md`.

## Kernels

- **One new kernel per PR:** a PR that tries an optimization adds a new kernel and leaves the
  kernel it started from unchanged (bug fixes aside), so the benchmark keeps comparing every step.
- **Names:** a kernel built from another is named `<new>_from_<old>`. `<new>` is a short name that
  no other kernel here uses, and `<old>` is the short name of the kernel it starts from. Following
  the `_from_` links traces every kernel back to a root: `naive`, then `tiled32_from_naive`, then
  `<next>_from_tiled32`. Roots, built from no other kernel (`cublas`, `naive`), have plain names.
- **Variants:** when a PR sweeps a parameter, each value is its own kernel, with the value in its
  short name (`tiled4_from_naive`, `tiled8_from_naive`, ...), so a later kernel can name the
  variant it starts from.
- **Where the name appears:** in `KERNELS` (`"tiled32_from_naive"`), in the function name
  (`matmul_<new>_from_<old>`; variants share a template, `matmul_tiled_from_naive<32>`) and in the
  file name (`kernels/<new>_from_<old>.cu`; variants share one file, `kernels/tiled_from_naive.cu`).
- **Experimental PRs:** an optimization that Claude tries on its own goes in a PR whose title
  starts with `[experimental claude optimization]`.
