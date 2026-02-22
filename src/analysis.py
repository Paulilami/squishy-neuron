from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from src.model import SquishyTransformer


def extract_neuron_params(model: SquishyTransformer) -> dict:
    decays = []
    state_weights = []
    has_selective = []

    for block in model.blocks:
        bank = block.neuron_bank

        if hasattr(bank, "decay_rates"):
            decays.append(bank.decay_rates.detach().cpu())
            has_selective.append(True)
        elif hasattr(bank, "decay"):
            decays.append(bank.decay.detach().cpu())
            has_selective.append(False)

        if hasattr(bank, "state_weight"):
            state_weights.append(bank.state_weight.detach().cpu())

    return {
        "decay": decays,
        "state_weight": state_weights,
        "has_selective": has_selective,
    }


def plot_decay_distribution(
    model: SquishyTransformer,
    save_path: Optional[Path] = None,
) -> None:
    params = extract_neuron_params(model)
    n_layers = len(params["decay"])
    if n_layers == 0:
        print("No stateful neuron banks found in model.")
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), squeeze=False)
    for i, decay in enumerate(params["decay"]):
        ax = axes[0, i]
        vals = decay.numpy()
        ax.hist(vals, bins=50, range=(0, 1), color="steelblue", alpha=0.8)
        label = "Selective" if params["has_selective"][i] else "Fixed"
        ax.set_title(f"Layer {i} Decay ({label})")
        ax.set_xlabel("Decay Rate")
        ax.set_ylabel("Count")
        ax.axvline(x=vals.mean(), color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_state_weight_distribution(
    model: SquishyTransformer,
    save_path: Optional[Path] = None,
) -> None:
    params = extract_neuron_params(model)
    n_layers = len(params["state_weight"])
    if n_layers == 0:
        print("No stateful neuron banks found in model.")
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), squeeze=False)
    for i, sw in enumerate(params["state_weight"]):
        ax = axes[0, i]
        ax.hist(sw.numpy(), bins=50, range=(0, 1), color="coral", alpha=0.8)
        ax.set_title(f"Layer {i} State Weight Distribution")
        ax.set_xlabel("State Weight")
        ax.set_ylabel("Count")
        ax.axvline(x=sw.mean().item(), color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def trace_neuron_states(
    model: SquishyTransformer,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    neuron_indices: list[int] | None = None,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    block = model.blocks[layer_idx]
    bank = block.neuron_bank

    has_state = hasattr(bank, "A_log") or hasattr(bank, "_decay_logit")
    if not has_state:
        raise ValueError(f"Layer {layer_idx} does not have a stateful neuron bank.")

    if neuron_indices is None:
        neuron_indices = list(range(min(8, bank.d_neuron)))

    B, T = input_ids.shape

    x = model.token_emb(input_ids)
    if model.pos_emb is not None:
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = x + model.pos_emb(positions)
    x = model.drop(x)

    for i, blk in enumerate(model.blocks):
        if i < layer_idx:
            x, _ = blk(x, None)
        elif i == layer_idx:
            x_normed = blk.ln1(x)
            x = x + blk.attn(x_normed)
            x_normed = blk.ln2(x)

            raw = bank.W_main(x_normed)
            gate = torch.sigmoid(bank.W_gate(x_normed))

            if hasattr(bank, "A_log"):
                delta = F.softplus(bank.W_delta(x_normed))
                A = -torch.exp(bank.A_log)
                decay_t = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
            elif hasattr(bank, "_decay_logit"):
                if hasattr(bank, "_use_selective") and bank._use_selective:
                    delta = F.softplus(bank.W_delta(x_normed))
                    A = -torch.exp(bank.A_log)
                    decay_t = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
                else:
                    decay_t = bank.decay.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            else:
                raise ValueError("Cannot determine decay mechanism")

            state = torch.zeros(B, bank.d_neuron, device=device, dtype=x.dtype)
            traced = []
            for t in range(T):
                raw_t = raw[:, t, :]
                gate_t = gate[:, t, :]
                d_t = decay_t[:, t, :] if decay_t.dim() == 3 else decay_t
                update_t = (1.0 - d_t) * gate_t * torch.tanh(raw_t)
                state = d_t * state + update_t
                traced.append(state[0, neuron_indices].cpu())

            return torch.stack(traced, dim=0)

    raise RuntimeError("Should not reach here")


def plot_state_trajectories(
    states: torch.Tensor,
    neuron_indices: list[int],
    save_path: Optional[Path] = None,
    title: str = "Neuron State Trajectories",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    T = states.shape[0]
    for i, idx in enumerate(neuron_indices):
        ax.plot(range(T), states[:, i].numpy(), label=f"Neuron {idx}", alpha=0.8)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("State Value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_metrics(
    squishy_losses: list[float],
    baseline_losses: list[float],
    eval_interval: int = 1,
    save_path: Optional[Path] = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps_s = [i * eval_interval for i in range(len(squishy_losses))]
    steps_b = [i * eval_interval for i in range(len(baseline_losses))]

    ax1.plot(steps_s, squishy_losses, label="Squishy", color="steelblue")
    ax1.plot(steps_b, baseline_losses, label="Baseline", color="coral")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    squishy_ppl = [np.exp(min(l, 20)) for l in squishy_losses]
    baseline_ppl = [np.exp(min(l, 20)) for l in baseline_losses]
    ax2.plot(steps_s, squishy_ppl, label="Squishy", color="steelblue")
    ax2.plot(steps_b, baseline_ppl, label="Baseline", color="coral")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("Perplexity Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_neuron_summary(model: SquishyTransformer) -> None:
    params = extract_neuron_params(model)
    if not params["decay"]:
        print("No stateful neuron banks found.")
        return

    print(f"\n{'Layer':>6} | {'Type':>10} | {'Decay Mean':>11} | {'Decay Std':>10} | "
          f"{'SW Mean':>8} | {'SW Std':>7}")
    print("-" * 72)
    for i, decay in enumerate(params["decay"]):
        sw = params["state_weight"][i] if i < len(params["state_weight"]) else None
        variant = "Selective" if params["has_selective"][i] else "Fixed"
        sw_mean = f"{sw.mean().item():>8.4f}" if sw is not None else "     N/A"
        sw_std = f"{sw.std().item():>7.4f}" if sw is not None else "    N/A"
        print(
            f"{i:>6d} | {variant:>10} | {decay.mean().item():>11.4f} | "
            f"{decay.std().item():>10.4f} | {sw_mean} | {sw_std}"
        )
    print()


def plot_selectivity_heatmap(
    model: SquishyTransformer,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    n_neurons: int = 32,
    save_path: Optional[Path] = None,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    block = model.blocks[layer_idx]
    bank = block.neuron_bank

    if not hasattr(bank, "W_delta"):
        print("No selective decay in this layer.")
        return

    B, T = input_ids.shape
    x = model.token_emb(input_ids)
    if model.pos_emb is not None:
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = x + model.pos_emb(positions)
    x = model.drop(x)

    with torch.no_grad():
        for i, blk in enumerate(model.blocks):
            if i < layer_idx:
                x, _ = blk(x, None)
            elif i == layer_idx:
                x_normed = blk.ln1(x)
                x = x + blk.attn(x_normed)
                x_normed = blk.ln2(x)
                delta = F.softplus(bank.W_delta(x_normed))
                A = -torch.exp(bank.A_log)
                decay_vals = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
                decay_np = decay_vals[0, :, :n_neurons].cpu().numpy()
                break

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(decay_np.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Neuron Index")
    ax.set_title(f"Layer {layer_idx} Input-Dependent Decay Rates")
    plt.colorbar(im, ax=ax, label="Decay Rate")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()