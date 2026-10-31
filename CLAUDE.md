These rules apply to the whole repository. Folders can add their own rules in their own `CLAUDE.md`
(e.g. `cuda_book/CLAUDE.md`).

## Pull requests

- Keep each PR small enough to review: at most 200 non-blank lines of hand-written code (source,
  scripts, build files and configs, comments included), counting added and changed lines. Prose
  such as READMEs doesn't count, but keep it minimal. Split larger changes into stacked PRs (below).
- Lockfiles and verbatim copies of existing files don't count.
- Generated code (output of a script or compiler) doesn't count either, but commit it only if the
  user agrees. A PR may go well over 200 lines this way: the limit is on what reviewers read.
- State the reviewable line count in the PR description.
- Write every PR description with the `pr-description` skill (`.claude/skills/pr-description/`).
  Every PR uses its template unchanged.

## Stacked pull requests

- When a change splits into a sequence of dependent pieces, make one PR per logical piece and stack
  them. Each piece's branch is built on top of the previous piece's branch, and each PR's base is
  the branch below it, so every PR shows only its own piece. Each layer should build and pass its
  own tests.
- Before building a stack, tell the user the planned layers and how they will be stacked, and wait
  for their agreement.
- By default, stack with GitHub's native stacked PRs, using the `gh stack` CLI extension
  (`gh extension install github/gh-stack`). It needs Git 2.36 or later, and stacked PRs must be
  enabled for the repository.
  - `gh stack init <branch>` starts the stack with its first branch. Commit the first piece there.
  - `gh stack add <branch>` adds a branch on top of the stack. Commit the next piece there, and
    repeat for each piece.
  - `gh stack submit` pushes every branch and opens the linked PRs.
  - To change a lower piece (e.g. after review), switch to its branch with
    `gh stack checkout <branch>` or `gh stack down`, and commit there rather than in a higher layer.
    Then run `gh stack rebase --upstack` and `gh stack push`.
  - PRs merge bottom-up, with `gh stack merge` or from the merge box on GitHub. Afterwards,
    `gh stack sync --prune` rebases the remaining branches onto `main` and removes merged ones.
- If `gh stack` can't be used (the extension is missing, Git is older than 2.36, or it exits with
  code 9 because stacked PRs aren't enabled), tell the user and ask before falling back to stacking
  by hand: open the PRs with `gh pr create --base <branch below>`, then link them into a stack on
  GitHub.

## Issues

Open, format, label and split GitHub issues with the `github-issue` skill
(`.claude/skills/github-issue/`). Every issue uses its title format, template and labels.

## Experiments, evaluations and benchmarks

For experiments, evaluations and benchmarks (anything that compares options or ablates
something), let the user make the important decisions: ask rather than choose. Whatever you do
decide, list in the PR with a short rationale. Follow the `experiment-decisions` skill
(`.claude/skills/experiment-decisions/`) for when to ask, the template, and where the list goes
(PR description or a per-experiment `DECISIONS.md`).

## Kernel folders

A kernel folder (e.g. `optimizing_matmul/`) mixes three languages. CUDA (`.cu`) implements the
kernels. C++ (`.cc`) provides host-side stubs, such as launch wrappers and `main`s, and light
orchestration. Python (`.py`) is for scripts: building, orchestrating tests, benchmarking and
plotting.

Every kernel folder implements this tree. Entries marked "optional" may be left out; everything
else is required.

```
<folder>/
├── README.md                    # what it does and how to run it (present tense)
├── CLAUDE.md                    # optional: folder-specific rules
├── Makefile                     # entry points: make test, make fmt, make tidy, make clean
├── .gitignore                   # ignores .build/
├── .env.example                 # puts nvcc on PATH and sets CUDA_VISIBLE_DEVICES; copy to .env
├── .clang-format, .clang-tidy   # copies of cuda_book's (Google style, 100 columns)
├── pyproject.toml, ruff.toml    # uv project for the Python scripts and tools (ruff, clang-*)
├── common/                      # optional: shared code used by both kernels and host code
├── kernels/                     # .cu: the kernels, one kernel/task per file
│   └── common/                  # optional: shared CUDA code (.cuh)
├── host/                        # optional: .cc stubs and orchestration
│   └── common/                  # optional: shared host code (.h)
├── tests/                       # GoogleTest tests (.cu/.cc)
├── scripts/                     # .py: build/test orchestration, benchmarks, plots
└── .build/                      # build outputs (git-ignored)
```

- Before committing, run `make fmt` (clang-format and ruff) and `make tidy` (clang-tidy and ruff).
  Both must come back clean for the folder's own files.
- Never hardcode host-specific settings (CUDA version or path, GPU architecture). Take them from
  the environment (`.env`), and build with `-arch=native`.

> NOTE: `cuda_book/` is separate from this template and doesn't necessarily follow it. It keeps its
> own layout and rules (see `cuda_book/CLAUDE.md`).

## Code practices

These rules are for new code in every language; don't rewrite existing code only to follow them.
Inside `cuda_book/`, `cuda_book/CLAUDE.md` wins where the two differ (e.g. `_h`/`_d` instead of
`_host`/`_device`); the rest applies there too.

### Names and declarations

A name should say what a value is without making the reader look at the code around it. Match a
name's length to its scope: `i`, `j` and `x` are fine for a few lines, but a name that lives across
a function or a file, or that a decision depends on, spells out its meaning.

- **Intent:** say what the value is for, not what type it is: `active_user_count`, not `count`.
- **Number:** collections are plural, scalars singular: `for kernel in kernels`.
- **Units:** every quantity that has a unit carries a suffix from the table below: `timeout_ms`,
  `tile_bytes`.
- **Ratios:** a value that is one quantity per another, such as a conversion factor, is named
  `<numerator>_per_<denominator>`: `bytes_per_bf16` and `meters_per_mile`, not `bf16_bytes`,
  `bytes_bf16`, `mile_meters` or `meters_in_mile`.
- **Source:** a measured value, or one taken from a spec or other document, says where it comes
  from with `_according_to_<source>`, and `<source>` makes clear whether it is a spec or a
  measurement: `elapsed_ms_according_to_cuda_events`,
  `PEAK_DENSE_BF16_TFLOPS_ACCORDING_TO_NVIDIA_SPEC`, not `_according_to_nvidia`.
- **Maker:** when it matters what made a thing, say so with `_created_by_<maker>`:
  `plan_created_by_autotuner`.
- **Range bounds:** an index or pointer that starts or ends a range (other than a loop variable)
  carries `_incl` or `_excl`: `end_excl`, `last_row_incl`.
- **Booleans:** read as a yes/no question, starting with a verb such as `is_`, `has_`, `can_` or
  `should_`: `is_too_big`, not `too_big`.
- **Temporaries:** a throwaway value starts with `temp_`: `temp_sum`.
- **Functions:** `<verb>_<noun>`, the action and what it acts on, optionally followed by context:
  `copy_to_device`, `move_box_outside`. Interchangeable implementations of one operation may be
  `<operation>_<variant>`: `matmul_cublas`.
- **Dangerous functions:** a function that is dangerous to call ends in
  `_DANGEROUS_DANGEROUS_DANGEROUS_BECAUSE_<REASON>`:
  `free_buffer_DANGEROUS_DANGEROUS_DANGEROUS_BECAUSE_KERNELS_MAY_STILL_READ_IT`.
- **Types:** classes and structs are nouns for the object, not the action: `Dataloader`, not
  `Dataload`. Type aliases end in `_t`: `bf16_t`, `matmul_fn_t`.
- **Math letters:** where code follows a formula or a standard algorithm (kernels, linear algebra),
  use the standard letters. Arrays, matrices and tensors are capitals (`A`, `B`, `C`, `D`); vectors
  are lowercase (`x`, `b`); sizes and lengths are capitals (`N`, `M`, `K`); loop indices are `i`,
  `j`, `k`, or the lowercase of the size they run over (`m` over `M`, `n` over `N`, `k` over `K`);
  spatial coordinates and quaternion parts are `x`, `y`, `z`, `w`. There these letters override the
  length, intent and unit rules, so a matrix size stays `N` across a whole file. Elsewhere, name
  arrays and sizes by what they hold: `kernels`, `sample_count`.
- **Location:** an array, tensor or buffer, or a pointer to one, says where it lives with `_host` or
  `_device` (`A_host`, `A_device`), with `_device0`, `_device1`, ... when there are several devices
  and it matters which. Plain scalars, and code that never touches a device, don't take it.
- **Order:** when a name takes several suffixes, they go `<what>_<unit>`, then `_incl`/`_excl`,
  then `_according_to_<source>` or `_created_by_<maker>`, then the location:
  `elapsed_ms_according_to_cuda_events`, `end_offset_bytes_excl_device`.
- **Case:** snake_case for variables and free functions, in every language. Classes are the
  exception, and their rules take precedence: class and struct names are UpperCamelCase
  (`Dataloader`), methods are lowerCamelCase (`moveBoxOutside`), and static constructors are
  UpperCamelCase (`Dataloader::FromFiles`). Globals, global constants, macros and environment
  variables are UPPER_SNAKE_CASE (`KERNELS`, `DEFAULT_N`). The math letters above keep their case.
  A Python script's work goes in a `main()` function, so its working variables are locals and only
  constants sit at module level.
- **No magic numbers:** every literal with a meaning gets a named constant: `constexpr` in C++, a
  module-level constant in Python, `#define` only where a constant can't work. Bare `0` and `1` as
  start values, identities or increments are fine.
- **Declared types:** write the type when declaring a variable, in Python too
  (`timeout_ms: float = ...`). Use C++ `auto` only where the type can't be written (lambdas).

| Quantity | Suffixes | Meaning |
| --- | --- | --- |
| Time | `_ns`, `_us`, `_ms`, `_s` | |
| Clock | `_cycles` | clock cycles, e.g. from `clock64()` |
| Frequency | `_mhz`, `_ghz` | 10^6, 10^9 Hz |
| Data | `_bits`, `_bytes` | |
| Data, decimal | `_kbytes`, `_mbytes`, `_gbytes`, `_tbytes` | 10^3, 10^6, 10^9, 10^12 bytes |
| Data, decimal bits | `_kbits`, `_mbits`, `_gbits` | 10^3, 10^6, 10^9 bits |
| Data, binary | `_kibytes`, `_mibytes`, `_gibytes`, `_tibytes` | 2^10, 2^20, 2^30, 2^40 bytes |
| Bandwidth | `_gbytes_per_s`, `_tbytes_per_s`, `_gbits_per_s` | not `_gbps`: bits or bytes? |
| Work | `_flop`, `_gflop`, `_tflop` | 1, 10^9, 10^12 floating-point operations; never `_flops` |
| Throughput | `_gflops`, `_tflops` | 10^9, 10^12 FLOP per second: the `s` means per second |
| Arithmetic intensity | `_flop_per_byte` | |
| Power | `_w` | watts |
| Count | `_count` | a number of items: `element_count` |
| Fraction | `_frac`, `_pct` | 0 to 1, 0 to 100 |

For a unit the table lacks, build the suffix the same way, from lowercase unit names joined with
`_per_` (`_hz`, `_flop_per_cycle`), and add its row here in the same change.

### Comments

Code has no comments except docstrings and DANGEROUS comments. Tool directives such as
`// NOLINT(...)`, `# noqa`, `#pragma` and shebangs aren't comments for this rule.

- **Docstrings:** every file, class and function has one docstring, in the template for its
  language below. It gives the purpose, the interface, an example and, optionally, what a reader
  must know.
- **DANGEROUS comments:** a line that someone MUST know something about before changing it gets a
  one-line comment right above it:
  `// DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN <CONSEQUENCE> WILL HAPPEN` (`#` in
  Python). In a dangerous section, every line gets one. A DANGEROUS comment longer than one line
  needs the user's approval. Nothing else gets a comment.
- **Phrasing:** every sentence names its subject and uses the active voice: "`copy_to_device`
  copies a host array to the device.", not "Copies a host array to the device.". Write directly,
  clearly and briefly. Separate points are bullets; a line of reasoning is a paragraph.
- **Sections:** a docstring's sections come in the template's order. A section with nothing to say
  is left out, except Example, which every docstring has except a test's.
- **Interface:** a function's interface is its arguments, return value and errors. A file's is its
  command-line arguments (scripts and `main`s) or its public names (libraries). A Python class's is
  its constructor arguments and attributes; a C++ class documents its interface on its public
  methods and static constructors.
- **Must know:** an optional list grouped under these categories, in this order: Limitations,
  Gotchas, Side effects, Performance. Each item is one bullet of one sentence that fits on one
  line. A new category needs the user's agreement.
- **Files without docstrings:** Makefiles, shell scripts, TOML and `.env` files start with a `#`
  block shaped like the Python file docstring.

Python docstrings use the Google style:

```python
"""`bench.py` times every kernel and prints a table of time, TFLOPS and roofline share.

Interface:
    -N: Matrix size (default 8192).
    --warmup-iterations: Untimed calls per kernel before the timed one (default 1).

Example:
    $ uv run scripts/bench.py -N 4096

Must know:
    Gotchas:
        - `bench.py` reads the roofline peaks from `.env`, so `source .env` must run first.
"""


class Dataloader:
    """`Dataloader` loads batches of samples from files on disk.

    Args:
        paths: Files that the `Dataloader` reads, in order.

    Attributes:
        batch_count: Number of batches loaded so far.

    Example:
        >>> dataloader = Dataloader.FromFiles(paths)

    Must know:
        Limitations:
            - A `Dataloader` is not thread-safe.
    """


def copy_to_device(X_host: np.ndarray) -> DeviceArray:
    """`copy_to_device` copies a host array into a new device buffer.

    Args:
        X_host: The array to copy.

    Returns:
        A new device buffer that holds a copy of `X_host`.

    Raises:
        MemoryError: The device has no room for the copy.

    Example:
        >>> X_device = copy_to_device(X_host)

    Must know:
        Side effects:
            - The caller owns the new buffer and must free it.
    """
```

C++ and CUDA docstrings use Doxygen's Javadoc style:

```cpp
/**
 * @file
 * @brief `bench.cc` times one call of each kernel and prints one CSV line per kernel.
 *
 * @par Interface
 * - Arguments: `N warmup_iterations`.
 * - Output: one `name,milliseconds` line per kernel on stdout.
 *
 * @par Example
 * @code
 * .build/bench 8192 1
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - The timed call includes cuBLAS setup when `warmup_iterations` is 0.
 */

/**
 * @brief `Dataloader` loads batches of samples from files on disk.
 *
 * @par Example
 * @code
 * Dataloader dataloader = Dataloader::FromFiles(paths);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - A `Dataloader` is not thread-safe.
 */

/**
 * @brief `matmul_cublas` computes C = A * B for N x N row-major bf16 matrices with cuBLAS.
 *
 * @param[in] A_device Left factor, N x N, in device memory.
 * @param[in] B_device Right factor, N x N, in device memory.
 * @param[out] C_device Product, N x N, in device memory.
 * @param[in] N Matrix size.
 *
 * @par Example
 * @code
 * matmul_cublas(A_device, B_device, C_device, N);
 * @endcode
 *
 * @par Must know
 * - Performance:
 *   - The first call creates the cuBLAS handle, so it is much slower than later calls.
 */
```

A C++ function that returns a value adds `@return <meaning>.` after its `@param` lines, and one that
throws adds `@throws <type> <when>.`.

### Why

These practices aim for fewer bugs, code that reads quickly, and short code. They get there by
keeping code standard, informative and correct. When something you write isn't covered by a rule
here, ask the user whether to set one, so that conventions don't drift apart.

## Machine-specific workarounds

This repo is used from several machines, so don't write a workaround for one machine (e.g. a
broken `gh` command) as if it held everywhere. In `CLAUDE.md` files and skills, write it as a
condition, a one-line note on where it was seen, and the fix. Keep it to a line or two:

> If `gh pr edit` fails with a Projects (classic) error (seen once with gh 2.55 on align-4,
> 2026-10-30), set the body with `gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -f body=...`.

## Documenting the system

Documentation of the system (GPU model, compute capability, clocks, power limit, driver and toolkit
versions, OS, and so on) exists so that results can be reproduced. Record the settings that
determine or explain a result. Don't record observations that are likely temporary or that don't
matter for reproducing results, such as a tool failing to run, a permission or install quirk, or
which GPUs are busy right now. If something like that gets in your way, tell the user instead of
writing it into the docs.

> NOTE: if any of this is outdated notify the user.
