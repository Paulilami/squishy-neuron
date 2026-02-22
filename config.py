from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    vocab_size: int = 8192
    max_seq_len: int = 256
    dropout: float = 0.1

    neuron_variant: str = "selective"
    decay_init: float = 0.95
    state_weight_init: float = 0.5
    selective_decay: bool = True
    use_baseline: bool = False

    use_rope: bool = True
    rope_theta: float = 10000.0
    use_gradient_checkpointing: bool = False

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    grad_accum_steps: int = 1
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    seed: int = 42

    dataset: str = "roneneldan/TinyStories"
    tokenizer_vocab_size: int = 8192
    seq_length: int = 256
    data_dir: str = "data/processed"
    tokenizer_dir: str = "data/tokenizer"

    wandb_project: str = "squishy-neuron"
    wandb_run_name: str = ""
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> Config:
        with open(path) as f:
            data = json.load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)