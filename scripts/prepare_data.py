from __future__ import annotations

"""Download dataset, train tokenizer, pre-tokenize and save to disk.

Usage: python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_dataset
from config import Config
from src.tokenizer import train_tokenizer, load_tokenizer, encode


def main() -> None:
    config = Config()

    # 1. Download dataset
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    # 2. Train tokenizer on train split
    print(f"Training tokenizer with vocab_size={config.tokenizer_vocab_size}")
    train_texts = (ex["text"] for ex in dataset["train"])
    tokenizer = train_tokenizer(train_texts, config, save_path=config.tokenizer_dir)
    print(f"Tokenizer saved to {config.tokenizer_dir}")

    # Reload to ensure consistency
    tokenizer = load_tokenizer(config.tokenizer_dir)

    # 3. Tokenize train and validation splits
    for split_name in ["train", "validation"]:
        print(f"Tokenizing {split_name} split...")
        all_ids: list[int] = []
        for ex in dataset[split_name]:
            ids = encode(tokenizer, ex["text"])
            all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.int32)
        out_path = config.data_dir / "processed" / f"{split_name}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), arr)
        print(f"  {split_name}: {len(arr):,} tokens -> {out_path}")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
