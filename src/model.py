from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from src.neuron import build_neuron_bank
from src.attention import CausalSelfAttention


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class StandardFFN(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        d_ff = config.d_model * 4
        self.w1 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SquishyBlock(nn.Module):

    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.neuron_bank = build_neuron_bank(config)
        self.layer_idx = layer_idx

    def forward(
        self, x: torch.Tensor, neuron_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        neuron_out, new_state = self.neuron_bank(self.ln2(x), neuron_state)
        x = x + neuron_out
        return x, new_state


class BaselineBlock(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = StandardFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class SquishyTransformer(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._use_checkpoint = config.use_gradient_checkpointing

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_emb = None

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            SquishyBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        self.apply(_init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        neuron_states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        B, T = input_ids.shape

        x = self.token_emb(input_ids)

        if self.pos_emb is not None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(positions)

        x = self.drop(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            layer_state = neuron_states[i] if neuron_states else None
            if self._use_checkpoint and self.training:
                x, ns = checkpoint(block, x, layer_state, use_reentrant=False)
            else:
                x, ns = block(x, layer_state)
            new_states.append(ns)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss, new_states

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineTransformer(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._use_checkpoint = config.use_gradient_checkpointing

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_emb = None

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            BaselineBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        self.apply(_init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        neuron_states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        B, T = input_ids.shape

        x = self.token_emb(input_ids)

        if self.pos_emb is not None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(positions)

        x = self.drop(x)

        for block in self.blocks:
            if self._use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss, []

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: Config) -> SquishyTransformer | BaselineTransformer:
    if config.use_baseline:
        return BaselineTransformer(config)
    return SquishyTransformer(config)