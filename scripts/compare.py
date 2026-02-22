from __future__ import annotations

"""A/B comparison: train squishy and baseline with identical data/config.

Trains both models sequentially with the same seed and data,
then prints a summary comparison.

Usage: python scripts/compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from copy import deepcopy
from config import Config
from src.model import build_model
from src.trainer import Trainer, TextDataset


def main() -> None:
    config = Config()

    # Load data
    train_ids = torch.from_numpy(
        np.load(str(config.data_dir / "processed" / "train.npy"))
    )
    val_ids = torch.from_numpy(
        np.load(str(config.data_dir / "processed" / "validation.npy"))
    )
    train_dataset = TextDataset(train_ids, config.seq_length)
    val_dataset = TextDataset(val_ids, config.seq_length)

    # ── Train squishy model ──
    print("=" * 60)
    print("Training SQUISHY model")
    print("=" * 60)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    squishy_config = deepcopy(config)
    squishy_config.use_baseline = False
    squishy_config.wandb_run_name = "squishy-compare"
    squishy_config.checkpoint_dir = Path(config.checkpoint_dir) / "squishy"

    squishy_model = build_model(squishy_config)
    print(f"Squishy params: {squishy_model.count_parameters():,}")
    squishy_trainer = Trainer(
        squishy_model, train_dataset, val_dataset, squishy_config
    )
    squishy_trainer.train()
    squishy_final_loss = squishy_trainer.evaluate()

    # ── Train baseline model ──
    print("\n" + "=" * 60)
    print("Training BASELINE model")
    print("=" * 60)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    baseline_config = deepcopy(config)
    baseline_config.use_baseline = True
    baseline_config.wandb_run_name = "baseline-compare"
    baseline_config.checkpoint_dir = Path(config.checkpoint_dir) / "baseline"

    baseline_model = build_model(baseline_config)
    print(f"Baseline params: {baseline_model.count_parameters():,}")
    baseline_trainer = Trainer(
        baseline_model, train_dataset, val_dataset, baseline_config
    )
    baseline_trainer.train()
    baseline_final_loss = baseline_trainer.evaluate()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Squishy  - params: {squishy_model.count_parameters():>10,} | "
          f"val loss: {squishy_final_loss:.4f} | "
          f"val ppl: {np.exp(min(squishy_final_loss, 20)):.2f}")
    print(f"Baseline - params: {baseline_model.count_parameters():>10,} | "
          f"val loss: {baseline_final_loss:.4f} | "
          f"val ppl: {np.exp(min(baseline_final_loss, 20)):.2f}")

    diff = baseline_final_loss - squishy_final_loss
    if diff > 0:
        print(f"\nSquishy wins by {diff:.4f} loss ({diff/baseline_final_loss*100:.1f}% relative)")
    elif diff < 0:
        print(f"\nBaseline wins by {-diff:.4f} loss ({-diff/squishy_final_loss*100:.1f}% relative)")
    else:
        print("\nExact tie (unlikely!)")


if __name__ == "__main__":
    main()
