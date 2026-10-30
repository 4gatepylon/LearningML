"""Run LFM inference and inspect one layer with a PyTorch hook."""

import argparse
from collections.abc import Sequence
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
PILE_DATASET_ID = "monology/pile-uncopyrighted"


def choose_device(requested: str) -> torch.device:
    """Prefer Apple's Metal backend, while keeping a CPU escape hatch."""
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return torch.device("mps")

    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def prompt_from_pile(max_characters: int) -> tuple[str, str]:
    """Read one bounded example without downloading the full Pile dataset."""
    pile = load_dataset(PILE_DATASET_ID, split="train", streaming=True)
    example = next(iter(pile))
    passage = example["text"].strip()[:max_characters]
    source = example.get("meta", {}).get("pile_set_name", "unknown subset")
    prompt = (
        "Briefly summarize the following passage in one sentence.\n\n"
        f"{passage}"
    )
    return prompt, source


def extract_tensor(output: Any) -> torch.Tensor:
    """Get the hidden-state tensor from a hooked layer's output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Sequence) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Expected a tensor or tensor-first sequence, got {type(output)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID")
    parser.add_argument(
        "--prompt",
        default="Explain in two sentences why the sky appears blue.",
        help="Text to send to the model",
    )
    parser.add_argument(
        "--pile",
        action="store_true",
        help="Use one streamed example from The Pile instead of --prompt",
    )
    parser.add_argument(
        "--pile-max-characters",
        type=int,
        default=1_000,
        help="Maximum characters to keep from the streamed Pile example",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Model layer to hook (layer 5 is a full-attention block)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="auto prefers MPS when it is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = torch.float16 if device.type == "mps" else torch.float32

    prompt = args.prompt
    if args.pile:
        prompt, pile_source = prompt_from_pile(args.pile_max_characters)
        print(f"Streamed one bounded example from The Pile ({pile_source}).")

    print(f"Loading {args.model} on {device} with {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    layers = model.model.layers
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"--layer must be between 0 and {len(layers) - 1}")

    activations: dict[str, torch.Tensor] = {}

    def capture_activation(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        # Copy to CPU so this small example does not keep extra MPS memory alive.
        activations["hooked_layer"] = extract_tensor(output).detach().cpu()

    hook = layers[args.layer].register_forward_hook(capture_activation)
    try:
        # A plain forward pass gives one activation vector per prompt token.
        with torch.inference_mode():
            model(**model_inputs, use_cache=False)
    finally:
        hook.remove()

    activation = activations["hooked_layer"]
    print(f"Layer {args.layer} activation shape: {tuple(activation.shape)}")
    print(f"First token, first 8 values: {activation[0, 0, :8].float().tolist()}")

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    prompt_length = model_inputs["input_ids"].shape[-1]
    answer = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    print(f"\nModel response:\n{answer.strip()}")


if __name__ == "__main__":
    main()
