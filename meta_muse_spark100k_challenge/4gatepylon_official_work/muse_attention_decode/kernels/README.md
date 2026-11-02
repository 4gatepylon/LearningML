# muse_attention_decode kernels

Our kernels for `muse_attention_decode`, stored as `../../KERNELS.md` describes. The leaderboard's
top entry is 3.339x (1,128.0 µs per decode step) as of 2026-11-02; SGLang is 3,766.3 µs.

| Kernel | Parent | Change | Judge speedup (95%) | Judge run |
| --- | --- | --- | --- | --- |
| [001](001-triton-splitkv.py) | none | Three Triton kernels: new KV; split-KV attend with both heads' queries block-diagonal (one 128-byte-row load and one dot per tile); one combine program per request looping over the splits | 0.339x (0.333-0.345) | `20261102-190031-muse_attention_decode-001-triton-splitkv` |

Local times are from `scripts/local_judge.py` on an H100 PCIe (2.0 TB/s), which hides some of
the latency problems the judge's H200 (4.8 TB/s) shows, so a kernel's judge result decides.

## Tried and dropped
