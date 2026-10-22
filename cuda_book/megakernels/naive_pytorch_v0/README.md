# naive_pytorch_v0

Compiles static PyTorch graphs, captured via the `torch.export` mechanism, into a single megakernel
that performs the whole forward pass.
