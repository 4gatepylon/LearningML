- This is an educational project for learning CUDA, so don't just solve the problems for the user. Boilerplate (Makefile, scaffolding, printing/timing helpers) is fine to write, but leave the core exercise logic to the user unless they explicitly ask for it.
- Every C++ file (including ones with CUDA code) is a standalone executable run with `make run <name>` (the `.cc` extension is optional; `make run` alone runs every file). Rules for these files:
  - One file per task/function. File names must NOT say cpu/gpu (e.g. `vector_add.cc`, not `vector_add_cpu.cc`).
  - Each file contains both a CPU reference implementation and a GPU implementation, named `<name>_cpu` and `<name>_gpu`. Usually the user implements both; if the user asks you to implement something, it's fine to implement only the CPU one.
  - `main` always runs both implementations on the same inputs and compares their outputs.
  - `main` runs one or more tests, including at least one happy-path test, and prints whether the CPU and GPU answers matched.
- Read `README.md` in this folder before starting work.
- Lay out each file in three sections, in this order, each introduced by a banner (a line of 100 `/` characters, a title comment, another line of 100 `/` characters). `vector_add.cc` follows this template:

  ```cpp
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
  // Tests and main
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // Runs both implementations on the same inputs; returns true if outputs match.
  static bool run_test(const char* name, ...) { ... }

  int main() {
    // one or more run_test calls (at least one happy path), then print the overall result
    // and return nonzero if any test mismatched
  }
  ```
