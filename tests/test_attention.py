import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from config import Config
from src.attention import CausalSelfAttention, RotaryEmbedding, apply_rotary_emb


@pytest.fixture
def config():
    return Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, use_rope=True)


@pytest.fixture
def config_no_rope():
    return Config(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32, dropout=0.0, use_rope=False)


class TestCausalSelfAttention:

    def test_output_shape(self, config):
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_output_shape_no_rope(self, config_no_rope):
        attn = CausalSelfAttention(config_no_rope)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_causal_masking(self, config):
        attn = CausalSelfAttention(config)
        attn.eval()
        x = torch.randn(1, 8, 64)
        out1 = attn(x)
        x_modified = x.clone()
        x_modified[0, 5:, :] = torch.randn(3, 64)
        out2 = attn(x_modified)
        assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-5)

    def test_gradients_flow(self, config):
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_seq_lengths(self, config):
        attn = CausalSelfAttention(config)
        for t in [1, 4, 16, 32]:
            x = torch.randn(2, t, 64)
            out = attn(x)
            assert out.shape == (2, t, 64)

    def test_single_token(self, config):
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 1, 64)
        out = attn(x)
        assert out.shape == (2, 1, 64)

    def test_batch_independence(self, config):
        attn = CausalSelfAttention(config)
        attn.eval()
        x = torch.randn(3, 8, 64)
        out_full = attn(x)
        out_single = attn(x[1:2])
        assert torch.allclose(out_full[1], out_single[0], atol=1e-5)

    def test_deterministic_in_eval(self, config):
        attn = CausalSelfAttention(config)
        attn.eval()
        x = torch.randn(2, 8, 64)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)


class TestRotaryEmbedding:

    def test_cos_sin_shapes(self):
        rope = RotaryEmbedding(16, max_seq_len=64)
        cos, sin = rope(32)
        assert cos.shape == (32, 16)
        assert sin.shape == (32, 16)

    def test_dynamic_extension(self):
        rope = RotaryEmbedding(16, max_seq_len=32)
        cos, sin = rope(64)
        assert cos.shape == (64, 16)

    def test_apply_rotary_preserves_shape(self):
        rope = RotaryEmbedding(16, max_seq_len=32)
        cos, sin = rope(8)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_equivariance(self):
        rope = RotaryEmbedding(16, max_seq_len=32)
        cos, sin = rope(8)
        q = torch.randn(1, 1, 8, 16)
        k = q.clone()
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        dots_orig = (q * k).sum(-1)
        dots_rot = (q_rot * k_rot).sum(-1)
        assert torch.allclose(dots_orig, dots_rot, atol=1e-5)