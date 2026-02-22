from __future__ import annotations

"""Training entry point.

Usage:
  python scripts/train.py                           # squishy model
  python scripts/train.py --baseline                # standard transformer
  python scripts/train.py --variant gated           # gated variant
  python scripts/train.py --resume checkpoints/step_5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
from config import Config
from src.model import build_model
from src.trainer import Trainer, TextDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Squishy Neuron model")
    parser.add_argument(
        "--baseline", action="store_true",
        help="Train standard transformer instead of squishy",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Neuron variant (overrides config.neuron_variant)",
    )
    args = parser.parse_args()

    config = Config()

    if args.baseline:
        config.use_baseline = True
        config.wandb_run_name = "baseline"
    else:
        if args.variant:
            config.neuron_variant = args.variant
        config.wandb_run_name = f"squishy-{config.neuron_variant}"

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load pre-tokenized data
    train_path = config.data_dir / "processed" / "train.npy"
    val_path = config.data_dir / "processed" / "validation.npy"
    assert train_path.exists(), f"Training data not found at {train_path}. Run scripts/prepare_data.py first."

    train_ids = torch.from_numpy(np.load(str(train_path)))
    val_ids = torch.from_numpy(np.load(str(val_path)))

    train_dataset = TextDataset(train_ids, config.seq_length)
    val_dataset = TextDataset(val_ids, config.seq_length)

    print(f"Train: {len(train_ids):,} tokens, {len(train_dataset):,} sequences")
    print(f"Val:   {len(val_ids):,} tokens, {len(val_dataset):,} sequences")

    # Build model
    model = build_model(config)
    model_type = "Baseline" if config.use_baseline else f"Squishy ({config.neuron_variant})"
    print(f"Model: {model_type}")
    print(f"Parameters: {model.count_parameters():,}")

    # Train
    trainer = Trainer(model, train_dataset, val_dataset, config)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    trainer.train()


if __name__ == "__main__":
    main()
