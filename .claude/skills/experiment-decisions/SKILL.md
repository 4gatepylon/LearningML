---
name: experiment-decisions
description: Record the decisions behind an experiment, evaluation or benchmark in a standard template, in the PR description or in a per-experiment decisions.md. Use whenever you build or change anything that compares options or measures something - a performance benchmark, a model evaluation, an A/B test, or an ablation of a kernel, algorithm, model architecture, training scheme, training data, hyperparameters and so on - and you open or update a PR for it.
---

# Experiment decisions

An experiment is only as trustworthy as the choices behind it. When the user hasn't fixed every
detail, you fill the gaps: which baselines, which sizes, how many repeats, which metric, what to
hold fixed. Those choices must be visible to the reviewer, so every such PR lists them.

## When this applies

Any PR that adds or changes an experiment, evaluation or benchmark: anything that compares two or
more options or measures how well something works. Examples: timing kernel implementations,
comparing algorithms, ablating model architecture, a training scheme or training data, sweeping
hyperparameters, evaluating a model on a task.

## What counts as a decision

A decision is any choice that could change the results or how they read, and that the user didn't
state exactly. Typical ones:

- **Scope**: which options are compared, which are left out, and the question being answered.
- **Baselines**: what each option is compared against.
- **Workload / data**: problem sizes, shapes, dtypes, datasets, splits, preprocessing.
- **Procedure**: warmup, repeats, seeds, stopping criteria, hyperparameters, what is held fixed.
- **Metrics**: what is measured, how it is aggregated (mean, median, min), and the units.
- **Environment**: hardware, clocks, library versions, anything pinned or assumed.
- **Analysis**: how results are summarized, plotted or judged (thresholds, significance).

Don't list choices the user made explicitly; put those in the **Given** line instead. Don't list
pure implementation details that can't affect the results (file names, variable names).

## Where the decisions go

- **One folder per experiment** (the PR adds or changes experiment folders, e.g.
  `experiments/<name>/`): write the decisions to `decisions.md` in each experiment's folder. The PR
  description gets only a `## Decisions` section that links each file, one line per experiment.
- **Otherwise**: put the decisions in the PR description under `## Decisions`.

Either way, use the template below unchanged in structure. If a decision changes after review,
update it in place (the file or the PR description) rather than appending a new list.

## Template

In `decisions.md`, the top heading is `# Decisions: <experiment name>`; in a PR description, it is
`## Decisions` and every heading below it moves down one level (as in the example).

```markdown
# Decisions: <experiment name>

**Question:** <one sentence: what the experiment compares or measures, and why>
**Given:** <what the user specified; "none" if nothing>

## <Category, e.g. Scope, Baselines, Workload / data, Procedure, Metrics, Environment, Analysis>

### D1. <short name of the decision>

- **Choice:** <what was chosen>
- **Alternatives:** <what else was considered; "none considered" if so>
- **Rationale:** <one or two sentences on why>
- **Revisit if:** <optional: what result or constraint would change this choice>

### D2. ...
```

Rules for filling it in:

- Number decisions `D1`, `D2`, ... across the whole list (not per category), so review comments
  can refer to them. Don't renumber after review; mark dropped ones `(removed)`.
- Only include categories that have decisions, in the order listed above.
- Keep each rationale short. If a choice was arbitrary, say so ("arbitrary; any value in 5-20
  would do") rather than inventing a reason.
- Flag decisions you are unsure about with **(needs input)** after the name, and mention them to
  the user.

### Example

```markdown
## Decisions

**Question:** Is the tiled SGEMM kernel faster than the naive one, and how far is it from cuBLAS?
**Given:** compare naive, tiled and cuBLAS; FP32.

### Workload / data

#### D1. Square sizes 256-8192, powers of two

- **Choice:** M = N = K in {256, 512, ..., 8192}.
- **Alternatives:** non-square shapes; sizes that aren't multiples of the tile.
- **Rationale:** covers launch-bound to compute-bound; odd shapes are a separate question.
- **Revisit if:** the tiled kernel is meant for real model shapes.

### Procedure

#### D2. 10 warmup runs, median of 50 timed runs

- **Choice:** time with CUDA events; report the median.
- **Alternatives:** mean; min.
- **Rationale:** the median ignores one-off stalls from other jobs on the GPU.
```
