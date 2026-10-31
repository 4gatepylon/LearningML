---
name: github-issue
description: Open, format, label or split GitHub issues in this repo using its standard template and labels. Use whenever you create or edit an issue (gh issue create / edit), the user asks to file, track or write up a bug, task, experiment or question, or work should be split into several issues.
---

# GitHub issue

Every issue uses the same title format, body template and labels, so issues are easy to scan and
parse. As with the PR template (`pr-description` skill), this is a living document: if an issue
doesn't fit, stop and ask the user rather than improvising, and change this skill in its own PR.

## Scope: one issue or several

- **One issue = one outcome** that a single PR (or a single PR stack) closes, e.g. "add a tiled
  SGEMM kernel", "loss diverges with BF16 at lr 3e-4", "compare FlashAttention v2 vs v3 on H100".
- **Split** when parts can be finished, merged or dropped independently, or when they belong to
  different areas (below). A bug found while doing a task is its own issue.
- **Big efforts** (e.g. "multi-GPU training for model X") get a tracking issue whose **Done when**
  is a checklist of the sub-issues (`- [ ] #31`). The tracking issue holds no work of its own.

## Title

`<folder or area>: <what>`, same as PR titles. Short and specific; state the problem or goal,
not the fix you guess at. E.g. `optimizing_matmul: tiled kernel slower than naive at N=256`.

## Body template

```markdown
## Goal

<1-3 sentences: what should be true when this is done, and why.>

## Context

<What is known: links, files, measurements, logs. "None." if nothing.>

## Done when

- [ ] <checkable criterion>

## Notes

<Open questions, constraints, related issues/PRs. "None." if nothing.>
```

Every section is always present. What goes in **Context** depends on the type:

- **bug**: steps to reproduce (exact command), expected vs. actual, error output in a code block,
  and the setup that matters (GPU, driver/CUDA version, `CUDA_VISIBLE_DEVICES`, commit).
- **experiment**: the question and the options being compared. **Done when** includes "decisions
  recorded per the `experiment-decisions` skill".
- **enhancement / documentation / question**: what exists now and what's missing.

## Labels

Exactly one **type** label and at least one **area** label.

| Type | Use for |
| --- | --- |
| `bug` | something is wrong: crash, wrong result, failing test, broken build |
| `enhancement` | new code or a feature: a kernel, model, training loop, tool |
| `experiment` | a benchmark, evaluation or ablation that answers a question |
| `documentation` | READMEs, CLAUDE.md, skills |
| `question` | needs an answer or a decision before work can start |

| Area | Covers |
| --- | --- |
| `area:kernels` | CUDA kernels and their tests and benchmarks (kernel folders, `cuda_book/`) |
| `area:models` | model code, training, data, inference and evaluation |
| `area:compilers` | ML compilers and code generation |
| `area:distributed` | multi-GPU and multi-node code (NCCL, data/tensor/pipeline parallelism) |
| `area:tooling` | build, scripts, environment, repo rules |

## Commands

```bash
gh issue create --title "<title>" --body-file body.md --label enhancement --label area:kernels
gh issue edit <n> --add-label <label>      # works, unlike `gh pr edit` on align-4
gh issue close <n> --reason "not planned"  # or "completed"; close, don't delete
```

- Write the body to a file in the scratchpad and pass `--body-file`, which avoids shell-quoting
  the Markdown.
- If any `--label` doesn't exist, `gh issue create` fails and creates nothing. Check with
  `gh label list` first.
- Only GitHub's default labels exist out of the box. `experiment` and the `area:*` labels have to
  be created once (repo admin or maintain rights needed). Don't create labels yourself; ask the
  user, who can run or approve:

```bash
gh label create experiment       --color 5319e7 --description "Benchmark, evaluation or ablation"
gh label create area:kernels     --color 1d76db --description "CUDA kernels, their tests and benchmarks"
gh label create area:models      --color 1d76db --description "Model code, training, data, inference, eval"
gh label create area:compilers   --color 1d76db --description "ML compilers and code generation"
gh label create area:distributed --color 1d76db --description "Multi-GPU and multi-node code"
gh label create area:tooling     --color 1d76db --description "Build, scripts, environment, repo rules"
```
