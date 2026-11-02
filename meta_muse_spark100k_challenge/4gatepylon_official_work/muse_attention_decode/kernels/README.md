# muse_attention_decode kernels

Our kernels for `muse_attention_decode`, stored as `../../KERNELS.md` describes. The leaderboard's
top entry is 3.339x (1,128.0 µs per decode step) as of 2026-11-02; SGLang is 3,766.3 µs.

| Kernel | Parent | Change | Judge speedup (95%) | Judge run |
| --- | --- | --- | --- | --- |
| [001](001-triton-splitkv.py) | none | Three Triton kernels: new KV; split-KV attend with both heads' queries block-diagonal (one 128-byte-row load and one dot per tile); one combine program per request looping over the splits | 0.339x (0.333-0.345) | `20261102-190031-muse_attention_decode-001-triton-splitkv` |
| [002](002-per-head-dots.py) | 001 | Separate [64, 64] Q K^T and P V dots per KV head; one combine program per query row, reading splits in blocks | 0.897x (0.883-0.910) | `20261102-190033-muse_attention_decode-002-per-head-dots` |
| [003](003-packed-decode.py) | 002 | Decode blocks use 001's block-diagonal loop (one 128-byte-row load and one dot per tile) with 002's partials; tile size and splits per layer type (sliding 64 and 4 per SM, full 128 and 8) | 1.062x (1.046-1.079) | `20261102-201241-muse_attention_decode-003-packed-decode` |
| [004](004-tuned-per-shape.py) | 003 | Tile size, warps, stages and splits per (layer, verify or decode, batched or not), from an H200 sweep | 1.256x (1.237-1.275) | `20261102-202500-muse_attention_decode-004-tuned-per-shape` |
| [005](005-fa3-verify.py) | 004 | Verify blocks of 4+ requests: FA3 (sgl_kernel) between one fused new-KV-and-FP8-query kernel and one norm-and-gate kernel | 1.388x (1.368-1.414) | `20261102-203127-muse_attention_decode-005-fa3-verify` |

Local times are from `scripts/local_judge.py` on an H100 PCIe (2.0 TB/s), which hides some of
the latency problems the judge's H200 (4.8 TB/s) shows, so a kernel's judge result decides.

## Tried and dropped

- Batched 3D `tl.dot` over the two heads (`[2, rows, 64]`) instead of 002's per-head 2D dots: Triton's
  code was much slower locally (best verify-32 full 1,637 µs, against 686 µs for 001).
- One fused kernel per step (new tokens in an extra split, the last split to finish combining via an
  atomic counter): slower locally on sliding layers (batch-1 decode 20.7-41 µs, against 17.3 µs for
  002), since the last split's serial combine and the new-token work sit on the critical path.
- New tokens attended in the combine kernel, plus prefetching the next key tile into registers: slower
  locally than 002 on five of six loads (0.72-1.01x).
