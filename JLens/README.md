# JLens model playground

This directory is deliberately only model boilerplate: it contains no JLens
implementation. It runs one small language model, demonstrates a standard PyTorch
forward hook, and can optionally use one streamed text sample from The Pile.

The default checkpoint is
[`LiquidAI/LFM2.5-1.2B-JP-202606`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606),
an official Liquid AI checkpoint from June 2026. It has 1.17B parameters and
supports English and Japanese. The checkpoint is ungated and about 2.3 GB in
BF16 form.

## Run it

From this directory:

```sh
uv sync
uv run python inference.py
```

On Apple Silicon, `--device auto` selects PyTorch's `mps` backend. To require
it instead of allowing a CPU fallback:

```sh
uv run python inference.py --device mps
```

The first run downloads and caches the model weights. The script loads them as
FP16 on MPS, hooks layer 5 (one of the model's full-attention blocks) for one
forward pass, prints the activation tensor's shape and a few values, removes the
hook, and generates a short answer. Try another layer with `--layer 10`.

## Use a small amount of The Pile

```sh
uv run python inference.py --pile
```

This uses the uncopyrighted Pile mirror
[`monology/pile-uncopyrighted`](https://huggingface.co/datasets/monology/pile-uncopyrighted).
It passes `streaming=True`, reads only the first example, and truncates that
example to 1,000 characters. It does **not** download the complete dataset. You
can lower the text bound further with `--pile-max-characters 500`.

Run `uv run python inference.py --help` to see all options.
