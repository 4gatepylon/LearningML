- This is an educational project for learning CUDA, so don't just solve the problems for the user. Boilerplate (Makefile, scaffolding, printing/timing helpers) is fine to write, but leave the core exercise logic to the user unless they explicitly ask for it.
- Every C++ file (including ones with CUDA code) lives in `kernels/` and is a standalone executable tested with `make test <name>` (the `.cc` extension is optional; `make test` alone tests every file). Rules for these files:
  - One file per task/function. File names must NOT say cpu/gpu (e.g. `kernels/vector_add.cc`, not `kernels/vector_add_cpu.cc`).
  - Each file contains both a CPU reference implementation and a GPU implementation, named `<name>_cpu` and `<name>_gpu`. Usually the user implements both; if the user asks you to implement something, it's fine to implement only the CPU one.
  - Tests use GoogleTest (`TEST(...)`); do not write a `main`, since `gtest_main` supplies it.
  - Every test runs both implementations on the same inputs and compares their outputs with `ASSERT_*`/`EXPECT_*` (use `*_NEAR` for floats).
  - Write one or more tests, including at least one happy-path test.
- Read `README.md` in this folder before starting work.
- New exercise files go in `kernels/`. `megakernels/` holds megakernels, each in its own subfolder (currently `naive_pytorch_v0`) with its own `README.md`; it is not part of the test setup (see `megakernels/README.md`).
- Keep plans out of `README.md`s (they describe what exists and how to use it). Multi-step work gets an ephemeral `PLAN.md` in the folder it concerns, so agents and the user can share the plan: check items off and append to the work log as you go, and delete `PLAN.md` once every item is done. Template:

  ```markdown
  # Plan: <title>

  ## Description
  <What we're building and why, in a few sentences.>

  ## Checklist
  - [ ] <step>
  - [ ] <step>

  ## Work log
  - YYYY-MM-DD: <what was done, decisions made, open issues>
  ```
- Use a `.cu` file for anything with CUDA code (built with `nvcc`) and a `.cc` file for plain C++ (built with `g++`). `test.py` picks the compiler from the extension.
- Never hardcode flags, paths or versions that are specific to a host's build setup (CUDA version or location, GPU architecture, driver-specific workarounds) in `test.py`, the Makefile or the source. The user may use different CUDA versions on different hosts; take them from the environment (e.g. `nvcc` from `PATH` via `.env`). If hardcoding seems necessary, ask the user first.
- Lay out each file in three sections, in this order, each introduced by a banner (a line of 100 `/` characters, a title comment, another line of 100 `/` characters). `kernels/vector_add.cc` follows this template:

  ```cpp
  #include <gtest/gtest.h>
  #include <...>

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // GPU implementations and CPU launch code/helpers
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  void <name>_gpu(...) { ... }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // CPU reference implementations
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  void <name>_cpu(...) { ... }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Tests (main comes from gtest_main)
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // Runs both implementations on the same inputs and expects the outputs to match.
  static void ExpectCpuGpuMatch(...) { ... }

  // Happy path.
  TEST(<Name>, HappyPath) { ... ExpectCpuGpuMatch(...); }

  // Further tests: edge cases, odd sizes, etc.
  TEST(<Name>, <Case>) { ... }
  ```
- Variable naming convention (for new code; existing kernels are older and don't follow it yet, so don't rewrite them just for this):
  - Matrices and higher-order tensors have capitalized names (`A`, `B`); vectors and scalars have lowercase names (`a`, `b`).
  - Suffix `_h` for data on the host/CPU and `_d` for data on the device/GPU (e.g. `A_h`, `A_d`, `x_h`, `x_d`).
  - `kernels/matvec.cu` follows this convention; use it as the example.

> NOTE: if any of this is outdated notify the user.