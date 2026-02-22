from __future__ import annotations

import math
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class NeuronBankBase(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_neuron = config.d_model * 4

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def _inv_sigmoid(x: float) -> float:
    x = max(min(x, 0.999), 0.001)
    return math.log(x / (1.0 - x))


def _s4_init_log_decay(d: int) -> torch.Tensor:
    log_min = math.log(0.001)
    log_max = math.log(0.1)
    return torch.linspace(log_min, log_max, d)


def _sequential_scan(
    decay: torch.Tensor,
    update: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = decay.shape
    outputs = []
    for t in range(T):
        state = decay[:, t] * state + update[:, t]
        outputs.append(state)
    return torch.stack(outputs, dim=1), state


def _parallel_scan_doubling(
    decay: torch.Tensor,
    update: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = decay.shape

    if T == 0:
        return torch.empty(B, 0, D, device=decay.device), state
    if T == 1:
        out = decay[:, 0] * state + update[:, 0]
        return out.unsqueeze(1), out

    cur_a = decay.clone()
    cur_b = update.clone()

    d = 1
    while d < T:
        prev_a = torch.ones_like(cur_a)
        prev_b = torch.zeros_like(cur_b)
        prev_a[:, d:] = cur_a[:, :-d]
        prev_b[:, d:] = cur_b[:, :-d]
        cur_b = cur_a * prev_b + cur_b
        cur_a = cur_a * prev_a
        d *= 2

    result = cur_a * state.unsqueeze(1) + cur_b
    final_state = result[:, -1]
    return result, final_state


def scan(
    decay: torch.Tensor,
    update: torch.Tensor,
    state: torch.Tensor,
    use_parallel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_parallel and decay.shape[1] >= 64:
        return _parallel_scan_doubling(decay, update, state)
    return _sequential_scan(decay, update, state)


class SelectiveNeuronBank(NeuronBankBase):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.W_main = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_gate = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_mod = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_delta = nn.Linear(self.d_model, self.d_neuron, bias=True)
        self.W_out = nn.Linear(self.d_neuron, self.d_model, bias=False)

        self.A_log = nn.Parameter(_s4_init_log_decay(self.d_neuron))

        self._sw_logit = nn.Parameter(
            torch.full((self.d_neuron,), _inv_sigmoid(config.state_weight_init))
        )

        nn.init.constant_(self.W_delta.bias, -3.0)

        self.dropout = nn.Dropout(config.dropout)
        self._use_parallel = True

    @property
    def decay_rates(self) -> torch.Tensor:
        return torch.exp(-torch.exp(self.A_log))

    @property
    def state_weight(self) -> torch.Tensor:
        return torch.sigmoid(self._sw_logit)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        raw = self.W_main(x)
        gate = torch.sigmoid(self.W_gate(x))
        mod = torch.tanh(self.W_mod(x))
        delta = F.softplus(self.W_delta(x))

        A = -torch.exp(self.A_log)
        decay = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))

        if state is None:
            state = torch.zeros(B, self.d_neuron, device=x.device, dtype=x.dtype)

        update = (1.0 - decay) * gate * torch.tanh(raw)

        all_states, final_state = scan(decay, update, state, self._use_parallel)

        sw = self.state_weight
        mixed = (1.0 - sw) * F.gelu(raw) + sw * all_states
        output = mixed * (1.0 + mod)

        output = self.dropout(output)
        output = self.W_out(output)

        return output, final_state.detach()


class StatefulNeuronBank(NeuronBankBase):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.W_main = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_gate = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_mod = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_out = nn.Linear(self.d_neuron, self.d_model, bias=False)

        if config.selective_decay:
            self.W_delta = nn.Linear(self.d_model, self.d_neuron, bias=True)
            self.A_log = nn.Parameter(_s4_init_log_decay(self.d_neuron))
            nn.init.constant_(self.W_delta.bias, -3.0)
            self._use_selective = True
        else:
            self._decay_logit = nn.Parameter(
                torch.full((self.d_neuron,), _inv_sigmoid(config.decay_init))
            )
            self._use_selective = False

        self._sw_logit = nn.Parameter(
            torch.full((self.d_neuron,), _inv_sigmoid(config.state_weight_init))
        )

        self.dropout = nn.Dropout(config.dropout)

    @property
    def decay(self) -> torch.Tensor:
        if self._use_selective:
            return torch.exp(-torch.exp(self.A_log))
        return torch.sigmoid(self._decay_logit)

    @property
    def state_weight(self) -> torch.Tensor:
        return torch.sigmoid(self._sw_logit)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        raw = self.W_main(x)
        gate = torch.sigmoid(self.W_gate(x))
        mod = torch.tanh(self.W_mod(x))

        if state is None:
            state = torch.zeros(B, self.d_neuron, device=x.device, dtype=x.dtype)

        if self._use_selective:
            delta = F.softplus(self.W_delta(x))
            A = -torch.exp(self.A_log)
            decay_t = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
            update = (1.0 - decay_t) * gate * torch.tanh(raw)
            all_states, final_state = _sequential_scan(decay_t, update, state)
        else:
            d = self.decay
            update = gate * torch.tanh(raw)
            outputs = []
            for t in range(T):
                state = d * state + (1.0 - d) * update[:, t]
                outputs.append(state)
            all_states = torch.stack(outputs, dim=1)
            final_state = state

        sw = self.state_weight
        mixed = (1.0 - sw) * F.gelu(raw) + sw * all_states
        output = mixed * (1.0 + mod)

        output = self.dropout(output)
        output = self.W_out(output)

        return output, final_state.detach()


class GatedNeuronBank(NeuronBankBase):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.W_main = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_gate = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_mod = nn.Linear(self.d_model, self.d_neuron, bias=False)
        self.W_out = nn.Linear(self.d_neuron, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        raw = F.gelu(self.W_main(x))
        gate = torch.sigmoid(self.W_gate(x))
        mod = torch.tanh(self.W_mod(x))

        output = raw * gate * (1.0 + mod)
        output = self.dropout(output)
        output = self.W_out(output)

        dummy_state = torch.zeros(B, self.d_neuron, device=x.device, dtype=x.dtype)
        return output, dummy_state


NEURON_REGISTRY: Dict[str, Type[NeuronBankBase]] = {
    "stateful": StatefulNeuronBank,
    "selective": SelectiveNeuronBank,
    "gated": GatedNeuronBank,
}


def build_neuron_bank(config: Config) -> NeuronBankBase:
    variant = config.neuron_variant
    if variant not in NEURON_REGISTRY:
        raise ValueError(
            f"Unknown neuron variant '{variant}'. "
            f"Available: {list(NEURON_REGISTRY.keys())}"
        )
    return NEURON_REGISTRY[variant](config)