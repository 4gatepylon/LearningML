---
name: okc
description: Test, check, submit and track Open Kernel Challenge kernels with the okc CLI without wasting the daily per-problem allowance, and keep every run's outputs in a well-named run folder. Use whenever you run any okc command (init, check, submit, status, jobs, leaderboard, withdraw) or scripts/okc_runs.py, read an okc result, or work in meta_muse_spark100k_challenge/.
---

# okc

The Open Kernel Challenge (https://openkernelchallenge.dev) times kernels against SGLang's on one
H200 through the `okc` CLI. Every timed run comes out of a small daily allowance per problem, and
the scored ones go on a public leaderboard under our GitHub name (`4gatepylon`). This skill is how
to use that allowance on purpose and keep each run's outputs where they can be found again.

Our work lives in `meta_muse_spark100k_challenge/4gatepylon_official_work/<problem>/`, and every
run's files in `meta_muse_spark100k_challenge/.artifacts/okc_runs/<run_id>/` (git-ignored). Read
`meta_muse_spark100k_challenge/CLAUDE.md` first, then the problem's `AGENTS.md` and `TASK.md`.

## Setup

- `okc` lives in `~/.okc/bin`. A non-interactive shell may not read `.bashrc`, so put it on PATH
  yourself: `export PATH=$HOME/.okc/bin:$PATH`.
- `okc whoami` must print `4gatepylon`. If it doesn't, ask the user to run `okc login` in a
  terminal: it accepts the challenge rules, which an agent can't do for anyone.
- `okc_runs.py` needs `uv`; run it as `uv run --script scripts/okc_runs.py` from
  `4gatepylon_official_work/`.

## The allowance

| Kind | Command | Per problem per day | Where it shows |
| --- | --- | --- | --- |
| Check | `okc check` | 20 | Nowhere public; off the board |
| Scored run | `okc submit` | 10 | The public leaderboard |

- The allowance resets at 00:00 UTC. `okc problems --json` shows `limits.checks.left` and
  `limits.scored.left` for each problem; read it before sending anything.
- A run is spent when the judge accepts the job (`queued`), before it runs, so assume a kernel
  that fails its tests or times out still costs one (`TASK.md` says a timeout is charged). Make
  sure the file imports and passes a local test first.
- Identical content is free: resending a file okc has already judged returns `status: "duplicate"`
  with `duplicate_of: <job>` and spends nothing. A single changed byte is new content.
- Keep a reserve: `okc_runs.py create` refuses when 2 or fewer runs of that kind are left
  (`--min-left`), so the last runs of the day stay for a deliberate choice.
- Ask the user before spending checks, and never `okc submit` unless the user asked for that
  submission. Never submit someone else's code (such as `lee_101_openkernels_example/`) under our
  name: the organizers review the top entries and remove ones that break the spirit of the rules.

## Workflows

### Check a kernel (the normal loop)

```bash
cd meta_muse_spark100k_challenge/4gatepylon_official_work
export PATH=$HOME/.okc/bin:$PATH
uv run --script scripts/okc_runs.py create muse_bf16_proj/submission.py --label tiled --dry-run
uv run --script scripts/okc_runs.py create muse_bf16_proj/submission.py --label tiled
uv run --script scripts/okc_runs.py get <run_id>        # waits; run it in the background
uv run --script scripts/okc_runs.py list
```

1. `create --dry-run` prints the problem, the file's sha256, how many runs are left today and any
   earlier runs of the same sha256 (`previous_run_ids`), and sends nothing. If an earlier run has
   the same sha256, read its `summary.json` instead of sending it again.
2. `create` copies the file into a new run folder, writes `command.txt` and `run.json`, sends it
   with `okc check --no-wait`, and prints the run id and job id in under a second.
3. `get` waits for the job (minutes; see Timings), follows `duplicate_of` to the job that holds the
   result, and writes `result.json`, `profile.txt` and `summary.json` into the run folder. It is
   safe to run again: `get --no-wait` reports where the job is without waiting.
4. `list` shows every run with its sha256 prefix, status, speedup and queue and run minutes.

Name runs with `--label` after what changed (`tiled`, `splitk4`), so `list` reads as a history.

### Submit a kernel (only when the user asks)

```bash
uv run --script scripts/okc_runs.py create muse_bf16_proj/submission.py --label tiled \
  --mode submit --yes-to-spend-scored-run
```

Submit only a file whose check passed, with exactly the bytes that were checked (same sha256), so
the scored run holds no surprises. `okc withdraw <job>` takes a run off the board for good.

### Raw okc, when the script doesn't cover it

- `okc init <problem> <dir>` writes a problem's starter; it never overwrites a file that exists.
  `make regenerate-starters` checks our starters against it.
- `okc status <job>` prints the per-shape table in about 0.2 s; `okc jobs --json` lists recent jobs
  and their states; `okc leaderboard <problem> --json` shows `you.rank` and `you.to_first`.
- `okc check`/`okc submit` without `--no-wait` block until the result and write `profiles/` beside
  the file sent, so only ever run them from a run folder, never from a source folder.

## Reading results

- Read `summary.json` (a few KB): status, passed, speedup and its 95% interval, `score_us`, queue and
  run minutes, test counts, the first failing test, and per shape the SGLang time, ours, the floor
  (`floor_us`, the speed of light) and the speedup.
- Read `profile.txt` for where the GPU time goes: each shape's kernels, their GPU time, share and
  call count.
- Don't print `result.json` (40 to 210 KB; almost all of it is `result.records`, each benchmark
  holding 60 raw timings). Pull single fields out with `python3 -c` or `jq` instead, e.g.
  `jq '.result.records[] | select(.event == "test" and .status != "pass")' result.json`.
- A gain inside the 95% interval (at least plus or minus 1.5%) may be noise.

## Timings

Measured on 2026-11-01 (UTC), with okc 0.7.2 and about 380 jobs waiting on 112 judge GPUs, for
lee101's best files (`okc check`):

| Problem | File | Queue | Run | Send to result | Speedup |
| --- | --- | --- | --- | --- | --- |
| `muse_bf16_proj` | `submission_v11.py` | 3.2 min | 1.5 min | 4.7 min | 1.140x |
| `muse_attention_decode` | `submission_v38.py` | 4.5 min | 5.6 min | 10.1 min | 1.602x |
| `muse_moe_w4a16` | `submission_v17.py` | 5.9 min | 2.2 min | 8.2 min | 1.415x |
| `muse_decode_layer` | `submission_v7.py` | 9.0 min | 5.1 min | 14.3 min | 1.347x |
| `muse_prefill_layer` | `submission_v6.py` | 11.8 min | 6.7 min | 18.5 min | 1.026x |

All five were sent within 4 s of each other, so the later queue times also include waiting behind
our own jobs. Each speedup matched lee101's own judge numbers within 1%.

- The queue dominates and grows near the deadline (`curl -s
  https://openkernelchallenge.dev/api/v1/queue` shows `backlog` and `running`). Send several
  problems at once rather than one after another; each problem has its own allowance.
- A run's time grows with what `prepare()` compiles. A run has 10 minutes in all, compiling
  included; past that it is charged as a timeout.
- A duplicate comes back in about 11 s; `okc status`, `okc jobs` and `okc problems` in under 0.3 s.

## Gotchas

- A `duplicate` reply exits with code 1 and has `result: null`; the result is on `duplicate_of`.
- `okc check --no-wait --json` prints only `{"job_id", "problem"}`; the job's state comes from
  `okc status <job> --json`.
- `okc status` rewrites the profile and log beside the file a job was first sent from, wherever
  that is (even another checkout), and writes nothing where you run it.
- The first line of a submission, `# okc: problem=<problem>`, picks the problem; keep it.
