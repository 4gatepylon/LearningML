# Megakernels

Megakernels live here. They are not part of the `make test`, `make tidy` or `make fmt` setup, and
nothing in this folder is built or tested by them.

Each megakernel lives in its own subfolder of `megakernels/`. Unlike the single-file programs in
`../kernels/`, a megakernel may need extra toolchain elements, such as code generation in Python or
additional compilation steps, so each subfolder documents how to build and run it.

Current megakernels:

- [`naive_pytorch_v0/`](naive_pytorch_v0/README.md): compiles static PyTorch graphs (via
  `torch.export`) into a single megakernel that does the forward pass.

> NOTE: if the list of megakernels above doesn't match the subfolders here, then this README is outdated and must be updated. All agents working on `megakernels/` must update this README (and let the user know) upon completion of their task if they see this note.