# 4gatepylon_official_work

Our entries for the Open Kernel Challenge (https://openkernelchallenge.dev), one folder per `okc`
problem:

| Folder | Problem |
| --- | --- |
| `muse_bf16_proj/` | Dense BF16 projections and the FP32 router |
| `muse_attention_decode/` | Gated attention decode over a paged FP8 cache |
| `muse_moe_w4a16/` | Routed experts with NVFP4 weights |
| `muse_decode_layer/` | The whole decoder layer, decode shapes |
| `muse_prefill_layer/` | The whole decoder layer, prefill shapes |

Each folder starts as the output of `okc init <problem> <folder>` (okc 0.7.2): `submission.py` (the
kernel we edit), `baseline.py` (SGLang's kernel, the one timed against), `reference.py` and
`checks.py` (what the judge checks), `nvfp4.py` where the problem uses NVFP4 weights, `TASK.md` and
`AGENTS.md`. Read `AGENTS.md`, then `TASK.md`, before working on a problem.

From inside a problem's folder:

```bash
okc check submission.py --json    # tests and timing, kept off the leaderboard
okc submit submission.py --json   # the same run, scored on the leaderboard
```

Both write the run's profile and log to `profiles/` beside `submission.py`, which is git-ignored.
