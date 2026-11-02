# muse_attention_decode kernels

Our kernels for `muse_attention_decode`, stored as `../../KERNELS.md` describes. The leaderboard's
top entry is 3.339x (1,128.0 µs per decode step) as of 2026-11-02; SGLang is 3,766.3 µs.

| Kernel | Parent | Change | Judge speedup (95%) | Judge run |
| --- | --- | --- | --- | --- |

Local times are from `scripts/local_judge.py` on an H100 PCIe (2.0 TB/s), which hides some of
the latency problems the judge's H200 (4.8 TB/s) shows, so a kernel's judge result decides.

## Tried and dropped
