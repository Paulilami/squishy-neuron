from __future__ import annotations

"""Quick end-to-end test: synthetic data -> train both models -> compare.

Generates a small dataset with learnable sequential patterns,
trains both squishy and baseline models, and reports insights.

Usage: python3 scripts/quick_test.py
"""

import sys
import random
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import Config
from src.tokenizer import train_tokenizer, load_tokenizer, encode, decode
from src.model import build_model, SquishyTransformer
from src.trainer import Trainer, TextDataset
from src.analysis import extract_neuron_params, print_neuron_summary


# ── 1. Generate synthetic dataset with learnable patterns ──

def generate_synthetic_stories(n_stories: int = 2000, seed: int = 42) -> list[str]:
    """Generate simple stories with repeating patterns and temporal structure.

    The patterns are designed so that stateful neurons *should* have an
    advantage: predicting words requires remembering context from earlier
    in the sequence (e.g., character names, locations, weather).
    """
    rng = random.Random(seed)

    characters = ["Alice", "Bob", "Cat", "Dog", "Fox", "Bear", "Owl", "Fish"]
    places = ["forest", "garden", "river", "mountain", "house", "cave", "field", "lake"]
    actions = ["walked", "ran", "jumped", "sat", "played", "looked", "found", "saw"]
    objects = ["flower", "stone", "stick", "leaf", "berry", "shell", "feather", "acorn"]
    feelings = ["happy", "sad", "excited", "tired", "curious", "brave", "calm", "proud"]
    weather = ["sunny", "rainy", "cloudy", "windy", "snowy", "warm", "cold", "foggy"]
    colors = ["red", "blue", "green", "yellow", "white", "brown", "golden", "silver"]

    templates = [
        # Pattern 1: Character consistency (must remember who is in the story)
        "Once upon a time, {char} lived near the {place}. One {weather} day, {char} {action} to the {place}. {char} found a {color} {obj} and felt {feeling}. {char} was very {feeling}.",
        # Pattern 2: Cause and effect chains
        "It was a {weather} morning. {char} {action} through the {place}. Because it was {weather}, {char} felt {feeling}. {char} picked up a {color} {obj} from the {place}.",
        # Pattern 3: Two characters interacting (must track both)
        "{char1} and {char2} went to the {place}. {char1} {action} while {char2} looked for a {obj}. Then {char2} found a {color} {obj}. {char1} and {char2} were both {feeling}.",
        # Pattern 4: Repetitive structure with variation
        "The {color} {obj} sat in the {place}. {char} saw the {color} {obj}. {char} {action} toward it. The {color} {obj} was very pretty. {char} felt {feeling}.",
        # Pattern 5: Temporal sequences
        "First, {char} went to the {place}. Next, {char} {action} around. Then {char} found a {obj}. After that, {char} felt {feeling}. Finally, {char} went home.",
        # Pattern 6: Counting/listing patterns
        "In the {place}, there were many things. There was a {color} {obj}. There was a {color2} {obj2}. {char} liked them all. {char} was {feeling}.",
    ]

    stories = []
    for _ in range(n_stories):
        template = rng.choice(templates)
        story = template.format(
            char=rng.choice(characters),
            char1=rng.choice(characters),
            char2=rng.choice(characters),
            place=rng.choice(places),
            action=rng.choice(actions),
            obj=rng.choice(objects),
            obj2=rng.choice(objects),
            feeling=rng.choice(feelings),
            weather=rng.choice(weather),
            color=rng.choice(colors),
            color2=rng.choice(colors),
        )
        stories.append(story)

    return stories


def main() -> None:
    print("=" * 70)
    print("SQUISHY NEURON - Quick End-to-End Test")
    print("=" * 70)

    # ── Config: small and fast ──
    config = Config(
        d_model=128,
        n_heads=4,
        n_layers=3,
        vocab_size=2048,
        max_seq_len=128,
        seq_length=128,
        dropout=0.1,
        batch_size=16,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=50,
        max_steps=500,
        grad_accum_steps=1,
        mixed_precision="no",  # CPU-safe
        log_interval=50,
        eval_interval=100,
        save_interval=500,
        tokenizer_vocab_size=2048,
        seed=42,
        wandb_project="squishy-test",
    )

    # ── Generate synthetic data ──
    print("\n[1/6] Generating synthetic dataset...")
    stories = generate_synthetic_stories(n_stories=3000)
    val_stories = generate_synthetic_stories(n_stories=300, seed=123)
    print(f"  Train: {len(stories)} stories")
    print(f"  Val:   {len(val_stories)} stories")
    print(f"  Sample: {stories[0][:100]}...")

    # ── Train tokenizer ──
    print("\n[2/6] Training BPE tokenizer...")
    config.tokenizer_dir = Path("data/tokenizer")
    tokenizer = train_tokenizer(stories, config, save_path=config.tokenizer_dir)
    actual_vocab = tokenizer.get_vocab_size()
    config.vocab_size = actual_vocab
    print(f"  Vocab size: {actual_vocab}")

    # Sample encoding
    sample = stories[0]
    sample_ids = encode(tokenizer, sample)
    decoded = decode(tokenizer, sample_ids)
    print(f"  Sample encode length: {len(sample_ids)} tokens")
    print(f"  Round-trip: {decoded[:80]}...")

    # ── Tokenize datasets ──
    print("\n[3/6] Tokenizing datasets...")
    train_ids = []
    for s in stories:
        train_ids.extend(encode(tokenizer, s))
    val_ids = []
    for s in val_stories:
        val_ids.extend(encode(tokenizer, s))

    train_tensor = torch.tensor(train_ids, dtype=torch.int32)
    val_tensor = torch.tensor(val_ids, dtype=torch.int32)
    print(f"  Train: {len(train_tensor):,} tokens")
    print(f"  Val:   {len(val_tensor):,} tokens")

    train_dataset = TextDataset(train_tensor, config.seq_length)
    val_dataset = TextDataset(val_tensor, config.seq_length)
    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences:   {len(val_dataset):,}")

    # ── Train Squishy model ──
    print("\n[4/6] Training SQUISHY model...")
    print("-" * 50)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    squishy_config = Config(**{
        k: v for k, v in config.__dict__.items()
        if k != 'head_dim'
    })
    squishy_config.use_baseline = False
    squishy_config.wandb_run_name = "test-squishy"
    squishy_config.checkpoint_dir = Path("checkpoints/test_squishy")

    squishy_model = build_model(squishy_config)
    s_params = squishy_model.count_parameters()
    print(f"  Parameters: {s_params:,}")

    squishy_trainer = Trainer(squishy_model, train_dataset, val_dataset, squishy_config)
    t0 = time.time()
    squishy_trainer.train()
    squishy_time = time.time() - t0
    squishy_val_loss = squishy_trainer.evaluate()
    print(f"  Final val loss: {squishy_val_loss:.4f}")
    print(f"  Final val ppl:  {math.exp(min(squishy_val_loss, 20)):.2f}")
    print(f"  Training time:  {squishy_time:.1f}s")

    # ── Train Baseline model ──
    print("\n[5/6] Training BASELINE model...")
    print("-" * 50)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    baseline_config = Config(**{
        k: v for k, v in config.__dict__.items()
        if k != 'head_dim'
    })
    baseline_config.use_baseline = True
    baseline_config.wandb_run_name = "test-baseline"
    baseline_config.checkpoint_dir = Path("checkpoints/test_baseline")

    baseline_model = build_model(baseline_config)
    b_params = baseline_model.count_parameters()
    print(f"  Parameters: {b_params:,}")

    baseline_trainer = Trainer(baseline_model, train_dataset, val_dataset, baseline_config)
    t0 = time.time()
    baseline_trainer.train()
    baseline_time = time.time() - t0
    baseline_val_loss = baseline_trainer.evaluate()
    print(f"  Final val loss: {baseline_val_loss:.4f}")
    print(f"  Final val ppl:  {math.exp(min(baseline_val_loss, 20)):.2f}")
    print(f"  Training time:  {baseline_time:.1f}s")

    # ── Analysis & Comparison ──
    print("\n[6/6] Analysis & Comparison")
    print("=" * 70)

    # Parameter comparison
    print("\n--- Parameter Counts ---")
    print(f"  Squishy:  {s_params:>10,}")
    print(f"  Baseline: {b_params:>10,}")
    print(f"  Ratio:    {s_params/b_params:.3f}x")

    # Loss comparison
    print("\n--- Final Validation Metrics ---")
    s_ppl = math.exp(min(squishy_val_loss, 20))
    b_ppl = math.exp(min(baseline_val_loss, 20))
    print(f"  {'':15s} {'Loss':>10s} {'Perplexity':>12s} {'Time (s)':>10s}")
    print(f"  {'Squishy':15s} {squishy_val_loss:>10.4f} {s_ppl:>12.2f} {squishy_time:>10.1f}")
    print(f"  {'Baseline':15s} {baseline_val_loss:>10.4f} {b_ppl:>12.2f} {baseline_time:>10.1f}")

    diff = baseline_val_loss - squishy_val_loss
    if abs(diff) < 0.01:
        print(f"\n  Result: Roughly tied (diff={diff:.4f})")
    elif diff > 0:
        print(f"\n  Result: SQUISHY wins by {diff:.4f} loss ({diff/baseline_val_loss*100:.1f}% relative)")
    else:
        print(f"\n  Result: BASELINE wins by {-diff:.4f} loss ({-diff/squishy_val_loss*100:.1f}% relative)")

    # Efficiency comparison (loss per parameter)
    print("\n--- Efficiency (loss per 1K params) ---")
    print(f"  Squishy:  {squishy_val_loss / (s_params/1000):.6f}")
    print(f"  Baseline: {baseline_val_loss / (b_params/1000):.6f}")

    # Speed comparison
    print("\n--- Training Speed ---")
    s_tok_per_sec = (config.batch_size * config.seq_length * config.max_steps) / squishy_time
    b_tok_per_sec = (config.batch_size * config.seq_length * config.max_steps) / baseline_time
    print(f"  Squishy:  {s_tok_per_sec:,.0f} tok/s")
    print(f"  Baseline: {b_tok_per_sec:,.0f} tok/s")
    print(f"  Speed ratio: {b_tok_per_sec/s_tok_per_sec:.2f}x (baseline/squishy)")

    # Neuron analysis (squishy only)
    if isinstance(squishy_model, SquishyTransformer):
        print("\n--- Squishy Neuron Analysis ---")
        print_neuron_summary(squishy_model)

        params = extract_neuron_params(squishy_model)
        for i, (decay, sw) in enumerate(zip(params["decay"], params["state_weight"])):
            d = decay.numpy()
            s = sw.numpy()
            print(f"  Layer {i} decay  : min={d.min():.4f} max={d.max():.4f} "
                  f"range={d.max()-d.min():.4f} "
                  f"long_memory(>0.9)={int((d > 0.9).sum())} "
                  f"short_memory(<0.5)={int((d < 0.5).sum())}")
            print(f"  Layer {i} st_wt  : min={s.min():.4f} max={s.max():.4f} "
                  f"range={s.max()-s.min():.4f} "
                  f"state_dominant(>0.7)={int((s > 0.7).sum())} "
                  f"raw_dominant(<0.3)={int((s < 0.3).sum())}")

    # Sample generation from both models
    print("\n--- Sample Generation ---")
    tokenizer = load_tokenizer(config.tokenizer_dir)
    prompts = ["Once upon a time,", "The red", "Alice and Bob"]

    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")

        for name, model in [("Squishy", squishy_model), ("Baseline", baseline_model)]:
            model.eval()
            device = next(model.parameters()).device
            input_ids = torch.tensor([encode(tokenizer, prompt)], device=device)

            with torch.no_grad():
                neuron_states = None
                for _ in range(40):
                    x = input_ids[:, -config.max_seq_len:]
                    logits, _, neuron_states = model(x, neuron_states=neuron_states)
                    next_logits = logits[:, -1, :] / 0.7
                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_id], dim=1)
                    if next_id.item() == 3:  # EOS
                        break

            output = decode(tokenizer, input_ids[0].tolist())
            # Clean up output
            output = output.replace("[BOS]", "").replace("[EOS]", "").strip()
            print(f"    {name:10s}: {output[:120]}")

    print("\n" + "=" * 70)
    print("Quick test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
