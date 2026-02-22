from __future__ import annotations

"""Interactive text generation from a trained model.

Usage: python scripts/generate.py --checkpoint checkpoints/step_10000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import torch.nn.functional as F
from config import Config
from src.model import build_model
from src.tokenizer import load_tokenizer, encode, decode
from safetensors.torch import load_model


def top_k_top_p_filter(
    logits: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits."""
    if top_k > 0:
        top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        threshold = top_k_vals[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        # Scatter back to original ordering
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    config: Config,
    max_new_tokens: int = 200,
) -> str:
    """Autoregressive generation with top-k/top-p sampling."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([encode(tokenizer, prompt)], device=device)

    neuron_states = None
    for _ in range(max_new_tokens):
        # Truncate to max_seq_len
        x = input_ids[:, -config.max_seq_len :]
        logits, _, neuron_states = model(x, neuron_states=neuron_states)

        next_logits = logits[:, -1, :] / config.temperature
        next_logits = top_k_top_p_filter(next_logits, config.top_k, config.top_p)

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        # Stop on EOS token
        if next_id.item() == 3:
            break

    return decode(tokenizer, input_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (overrides config)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    config = Config.load(ckpt_dir / "config.json")

    if args.temperature is not None:
        config.temperature = args.temperature

    model = build_model(config)
    load_model(model, str(ckpt_dir / "model.safetensors"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tokenizer = load_tokenizer(config.tokenizer_dir)

    model_type = "Baseline" if config.use_baseline else f"Squishy ({config.neuron_variant})"
    print(f"Model: {model_type} | Params: {model.count_parameters():,}")
    print(f"Temperature: {config.temperature} | Top-k: {config.top_k} | Top-p: {config.top_p}")
    print("Interactive generation (Ctrl+C to exit)\n")

    try:
        while True:
            prompt = input("Prompt: ")
            if not prompt.strip():
                continue
            output = generate(model, tokenizer, prompt, config, args.max_tokens)
            print(f"\n{output}\n")
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
