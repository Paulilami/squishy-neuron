from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_model, load_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class TextDataset(Dataset):

    def __init__(self, token_ids: torch.Tensor, seq_length: int) -> None:
        self.data = token_ids
        self.seq_length = seq_length

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_length
        chunk = self.data[start : start + self.seq_length + 1]
        return chunk[:-1].long(), chunk[1:].long()


def get_lr(step: int, config: Config) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    decay_ratio = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate * 0.9)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_dataset: TextDataset,
        val_dataset: TextDataset,
        config: Config,
    ) -> None:
        self.model = model
        self.config = config
        self.device = self._get_device()
        self.model.to(self.device)

        decay_params = [
            p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
        ]
        no_decay_params = [
            p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )

        self.use_amp = config.mixed_precision != "no"
        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[config.mixed_precision]

        use_scaler = config.mixed_precision == "fp16" and torch.cuda.is_available()
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except (TypeError, AttributeError):
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=use_scaler)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.train_iter: Iterator = iter(self.train_loader)
        self.global_step = 0

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return batch[0].to(self.device), batch[1].to(self.device)

    def train(self) -> None:
        config = self.config

        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or None,
                config={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in config.__dict__.items()
                },
            )
            use_wandb = True
        except Exception:
            print("wandb not available, logging to stdout only")
            use_wandb = False

        self.model.train()
        t0 = time.time()

        while self.global_step < config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(config.grad_accum_steps):
                x, y = self._get_batch()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    _, loss, _ = self.model(x, targets=y)
                    loss = loss / config.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.max_grad_norm
            )

            lr = get_lr(self.global_step, config)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1

            if self.global_step % config.log_interval == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    config.batch_size
                    * config.seq_length
                    * config.grad_accum_steps
                    * config.log_interval
                ) / max(dt, 1e-6)
                log_data = {
                    "train/loss": accum_loss,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": self.global_step,
                }
                if use_wandb:
                    import wandb
                    wandb.log(log_data)
                print(
                    f"step {self.global_step:>6d} | "
                    f"loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tokens_per_sec:.0f}"
                )
                t0 = time.time()

            if self.global_step % config.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                log_data = {
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "train/step": self.global_step,
                }
                if use_wandb:
                    import wandb
                    wandb.log(log_data)
                print(
                    f"         val loss {val_loss:.4f} | "
                    f"val ppl {val_ppl:.2f}"
                )
                self.model.train()

            if self.global_step % config.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        if use_wandb:
            import wandb
            wandb.finish()

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for i, (x, y) in enumerate(self.val_loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                _, loss, _ = self.model(x, targets=y)
            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)

    def save_checkpoint(self) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir) / f"step_{self.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_model(self.model, str(ckpt_dir / "model.safetensors"))
        self.config.save(ckpt_dir / "config.json")

        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.global_step,
            },
            ckpt_dir / "training_state.pt",
        )
        print(f"Saved checkpoint to {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir: Path) -> None:
        ckpt_dir = Path(ckpt_dir)
        load_model(self.model, str(ckpt_dir / "model.safetensors"))

        state = torch.load(
            ckpt_dir / "training_state.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["step"]
        print(f"Resumed from {ckpt_dir} at step {self.global_step}")