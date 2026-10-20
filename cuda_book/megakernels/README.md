# Megakernels

Megakernels will be added here in the future. They are not yet part of this project, and nothing in
this folder is built or tested by `make test`, `make tidy` or `make fmt`.

Each megakernel will live in its own subfolder of `megakernels/`. Unlike the single-file programs in
`../kernels/`, a megakernel may need extra toolchain elements, such as code generation in Python or
additional compilation steps, so each subfolder will document how to build and run it.
