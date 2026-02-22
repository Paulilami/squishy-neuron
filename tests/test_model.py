import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from config import Config
from src.model import (
    SquishyTransformer,
    BaselineTransformer,
    build_model,
    RMSNorm,
)


@pytest.fixture
def config():
    return Config(
        d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32,
        dropout=0.0, neuron_variant="selective", use_rope=True,
        selective_decay=True, use_gradient_checkpointing=False,
    )


@pytest.fixture
def config_no_rope():
    return Config(
        d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32,
        dropout=0.0, neuron_variant="selective", use_rope=False,
        selective_decay=True, use_gradient_checkpointing=False,
    )


class TestRMSNorm:

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == (2, 8, 64)

    def test_gradients(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None


class TestSquishyTransformer:

    def test_forward_logits_shape(self, config):
        model = SquishyTransformer(config)
        ids = torch.randint(0, 100, (2, 16))
        logits, loss, states = model(ids)
        assert logits.shape == (2, 16, 100)
        assert loss is None
        assert len(states) == 2

    def test_forward_with_targets(self, config):
        model = SquishyTransformer(config)
        ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss, states = model(ids, targets=targets)
        assert loss is not None
        assert loss.ndim == 0

    def test_forward_no_rope(self, config_no_rope):
        model = SquishyTransformer(config_no_rope)
        assert model.pos_emb is not None
        ids = torch.randint(0, 100, (2, 16))
        logits, loss, states = model(ids)
        assert logits.shape == (2, 16, 100)

    def test_backward(self, config):
        model = SquishyTransformer(config)
        ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        _, loss, _ = model(ids, targets=targets)
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_training_step_reduces_loss(self, config):
        model = SquishyTransformer(config)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ids = torch.randint(0, 100, (4, 16))
        targets = torch.randint(0, 100, (4, 16))
        _, loss0, _ = model(ids, targets=targets)
        for _ in range(20):
            opt.zero_grad()
            _, loss, _ = model(ids, targets=targets)
            loss.backward()
            opt.step()
        _, loss_final, _ = model(ids, targets=targets)
        assert loss_final.item() < loss0.item()

    def test_neuron_states_carry(self, config):
        model = SquishyTransformer(config)
        model.eval()
        ids = torch.randint(0, 100, (2, 16))
        _, _, states1 = model(ids)
        _, _, states2 = model(ids, neuron_states=states1)
        for s1, s2 in zip(states1, states2):
            assert not torch.allclose(s1, s2, atol=1e-6)

    def test_count_parameters(self, config):
        model = SquishyTransformer(config)
        n = model.count_parameters()
        assert n > 0
        assert n < 10_000_000

    def test_gradient_checkpointing(self, config):
        config.use_gradient_checkpointing = True
        model = SquishyTransformer(config)
        model.train()
        ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        _, loss, _ = model(ids, targets=targets)
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


class TestBaselineTransformer:

    def test_forward_shape(self, config):
        config.use_baseline = True
        model = BaselineTransformer(config)
        ids = torch.randint(0, 100, (2, 16))
        logits, loss, states = model(ids)
        assert logits.shape == (2, 16, 100)
        assert states == []

    def test_forward_with_targets(self, config):
        config.use_baseline = True
        model = BaselineTransformer(config)
        ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        _, loss, _ = model(ids, targets=targets)
        assert loss is not None

    def test_same_api_as_squishy(self, config):
        squishy = SquishyTransformer(config)
        config.use_baseline = True
        baseline = BaselineTransformer(config)
        ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        s_out = squishy(ids, targets=targets)
        b_out = baseline(ids, targets=targets)
        assert len(s_out) == len(b_out) == 3

    def test_gradient_checkpointing(self, config):
        config.use_gradient_checkpointing = True
        config.use_baseline = True
        model = BaselineTransformer(config)
        model.train()
        ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        _, loss, _ = model(ids, targets=targets)
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


class TestBuildModel:

    def test_builds_squishy_by_default(self, config):
        config.use_baseline = False
        model = build_model(config)
        assert isinstance(model, SquishyTransformer)

    def test_builds_baseline_when_flagged(self, config):
        config.use_baseline = True
        model = build_model(config)
        assert isinstance(model, BaselineTransformer)

    def test_param_counts_same_order_of_magnitude(self, config):
        config.use_baseline = False
        squishy = build_model(config)
        config.use_baseline = True
        baseline = build_model(config)
        ratio = squishy.count_parameters() / baseline.count_parameters()
        assert 0.5 < ratio < 5.0