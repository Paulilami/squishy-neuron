import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from config import Config
from src.neuron import (
    StatefulNeuronBank,
    SelectiveNeuronBank,
    GatedNeuronBank,
    NEURON_REGISTRY,
    build_neuron_bank,
    _sequential_scan,
    _parallel_scan_doubling,
)


@pytest.fixture
def config():
    return Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0)


@pytest.fixture
def selective_config():
    return Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, neuron_variant="selective", selective_decay=True)


class TestSelectiveNeuronBank:

    def test_output_shape(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        x = torch.randn(2, 8, 64)
        out, state = bank(x)
        assert out.shape == (2, 8, 64)
        assert state.shape == (2, 256)

    def test_state_changes_output(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        bank.eval()
        x = torch.randn(2, 8, 64)
        out1, s1 = bank(x, None)
        out2, s2 = bank(x, s1)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_gradients_flow(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, state = bank(x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in bank.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_state_is_detached(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        x = torch.randn(2, 8, 64)
        _, state = bank(x)
        assert not state.requires_grad

    def test_decay_rates_bounded(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        rates = bank.decay_rates
        assert (rates > 0).all()
        assert (rates < 1).all()

    def test_state_weight_bounded(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        sw = bank.state_weight
        assert (sw > 0).all()
        assert (sw < 1).all()

    def test_batch_independence(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        bank.eval()
        x = torch.randn(3, 8, 64)
        out_full, _ = bank(x)
        out_single, _ = bank(x[1:2])
        assert torch.allclose(out_full[1], out_single[0], atol=1e-5)

    def test_single_timestep(self, selective_config):
        bank = SelectiveNeuronBank(selective_config)
        x = torch.randn(2, 1, 64)
        out, state = bank(x)
        assert out.shape == (2, 1, 64)
        assert state.shape == (2, 256)


class TestStatefulNeuronBank:

    def test_output_shape(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        x = torch.randn(2, 8, 64)
        out, state = bank(x)
        assert out.shape == (2, 8, 64)
        assert state.shape == (2, 256)

    def test_state_changes_output(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        bank.eval()
        x = torch.randn(2, 8, 64)
        out1, s1 = bank(x, None)
        out2, s2 = bank(x, s1)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_selective_mode(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=True)
        bank = StatefulNeuronBank(cfg)
        assert bank._use_selective
        x = torch.randn(2, 8, 64)
        out, state = bank(x)
        assert out.shape == (2, 8, 64)

    def test_decay_in_range(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        d = bank.decay
        assert (d > 0).all()
        assert (d < 1).all()

    def test_state_weight_in_range(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        sw = bank.state_weight
        assert (sw > 0).all()
        assert (sw < 1).all()

    def test_gradients_flow(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, state = bank(x)
        out.sum().backward()
        assert x.grad is not None

    def test_zero_state_init(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        x = torch.randn(2, 8, 64)
        out1, _ = bank(x, None)
        out2, _ = bank(x, torch.zeros(2, 256))
        assert torch.allclose(out1, out2)

    def test_state_is_detached(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        x = torch.randn(2, 8, 64)
        _, state = bank(x)
        assert not state.requires_grad

    def test_single_timestep(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        x = torch.randn(2, 1, 64)
        out, state = bank(x)
        assert out.shape == (2, 1, 64)

    def test_batch_independence(self, config):
        cfg = Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, selective_decay=False)
        bank = StatefulNeuronBank(cfg)
        bank.eval()
        x = torch.randn(3, 8, 64)
        out_full, _ = bank(x)
        out_single, _ = bank(x[1:2])
        assert torch.allclose(out_full[1], out_single[0], atol=1e-5)


class TestGatedNeuronBank:

    def test_output_shape(self, config):
        bank = GatedNeuronBank(config)
        x = torch.randn(2, 8, 64)
        out, state = bank(x)
        assert out.shape == (2, 8, 64)

    def test_no_temporal_dependence(self, config):
        bank = GatedNeuronBank(config)
        bank.eval()
        x = torch.randn(2, 8, 64)
        out1, _ = bank(x)
        out2, _ = bank(x, torch.randn(2, 256))
        assert torch.allclose(out1, out2)

    def test_gradients_flow(self, config):
        bank = GatedNeuronBank(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = bank(x)
        out.sum().backward()
        assert x.grad is not None


class TestScanFunctions:

    def test_sequential_scan_basic(self):
        B, T, D = 2, 4, 8
        decay = torch.ones(B, T, D) * 0.9
        update = torch.ones(B, T, D) * 0.1
        s0 = torch.zeros(B, D)
        states, final = _sequential_scan(decay, update, s0)
        assert states.shape == (B, T, D)
        assert final.shape == (B, D)
        assert torch.allclose(final, states[:, -1])

    def test_parallel_scan_matches_sequential(self):
        B, T, D = 2, 16, 8
        torch.manual_seed(42)
        decay = torch.sigmoid(torch.randn(B, T, D))
        update = torch.randn(B, T, D) * 0.1
        s0 = torch.randn(B, D) * 0.1

        seq_states, seq_final = _sequential_scan(decay, update, s0)
        par_states, par_final = _parallel_scan_doubling(decay, update, s0)

        assert torch.allclose(seq_states, par_states, atol=1e-4), \
            f"Max diff: {(seq_states - par_states).abs().max().item()}"
        assert torch.allclose(seq_final, par_final, atol=1e-4)

    def test_parallel_scan_single_step(self):
        B, D = 2, 8
        decay = torch.ones(B, 1, D) * 0.9
        update = torch.ones(B, 1, D) * 0.1
        s0 = torch.ones(B, D)
        states, final = _parallel_scan_doubling(decay, update, s0)
        expected = 0.9 * s0 + 0.1
        assert torch.allclose(states[:, 0], expected, atol=1e-5)

    def test_parallel_scan_non_power_of_two(self):
        B, T, D = 2, 13, 8
        torch.manual_seed(42)
        decay = torch.sigmoid(torch.randn(B, T, D))
        update = torch.randn(B, T, D) * 0.1
        s0 = torch.zeros(B, D)

        seq_states, _ = _sequential_scan(decay, update, s0)
        par_states, _ = _parallel_scan_doubling(decay, update, s0)
        assert torch.allclose(seq_states, par_states, atol=1e-4)


class TestRegistry:

    def test_build_stateful(self, config):
        config.neuron_variant = "stateful"
        bank = build_neuron_bank(config)
        assert isinstance(bank, StatefulNeuronBank)

    def test_build_selective(self, config):
        config.neuron_variant = "selective"
        bank = build_neuron_bank(config)
        assert isinstance(bank, SelectiveNeuronBank)

    def test_build_gated(self, config):
        config.neuron_variant = "gated"
        bank = build_neuron_bank(config)
        assert isinstance(bank, GatedNeuronBank)

    def test_unknown_variant_raises(self, config):
        config.neuron_variant = "nonexistent"
        with pytest.raises(ValueError):
            build_neuron_bank(config)

    def test_all_registry_entries_instantiate(self, config):
        for name in NEURON_REGISTRY:
            config.neuron_variant = name
            bank = build_neuron_bank(config)
            x = torch.randn(2, 4, config.d_model)
            out, state = bank(x)
            assert out.shape == (2, 4, config.d_model)