These rules apply to `optimizing_matmul/`, on top of the root `CLAUDE.md`.

## Kernels

- **One new kernel per PR:** a PR that tries an optimization adds a new kernel and leaves the
  kernel it started from unchanged (bug fixes aside), so the benchmark keeps comparing every step.
  A new kernel changes how the work is done, not only the values of its options.
- **Names:** a kernel built from another is named `<new>_from_<old>`. `<new>` is a short name that
  no other kernel here uses, and `<old>` is the short name of the kernel it starts from. Following
  the `_from_` links traces every kernel back to a root: `naive`, then `blocktile1d_from_tiled32`,
  then `<next>_from_blocktile1d`. Roots, built from no other kernel (`cublas`, `naive`), have plain
  names.
- **Compile-time options:** a kernel's tunable sizes (tile sizes, threads per block, pipeline
  depth) are compile-time options of that one kernel, never separate kernels. Each is a macro with
  a default, named after the kernel (`WMMA_FROM_VECTORIZED_BLOCK_TILE_K`), that `-D` overrides.
  The PR that adds the kernel sweeps its options with `make sweep-options`, sets the defaults
  to the best values and reports the sweep. (`tiled4_from_naive` to `tiled32_from_naive` predate
  this rule.)
- **Where the name appears:** in `KERNELS` (`"blocktile1d_from_tiled32"`), in the function name
  (`matmul_<new>_from_<old>`) and in the file name (`kernels/<new>_from_<old>.cu`).
- **Experimental PRs:** an optimization that Claude tries on its own goes in a PR whose title
  starts with `[experimental claude optimization]`.
