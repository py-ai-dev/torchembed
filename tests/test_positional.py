"""Tests for positional embedding modules."""

import pytest
import torch

from torchembed.positional import (
    ALiBiEmbedding,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalEmbedding,
)


class TestRotaryEmbedding:
    def test_output_shape(self, batch_size, num_heads, seq_len, dim):
        rope = RotaryEmbedding(dim=dim)
        q = torch.randn(batch_size, num_heads, seq_len, dim)
        k = torch.randn(batch_size, num_heads, seq_len, dim)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(dim=63)

    def test_rotation_preserves_norms(self, dim):
        """RoPE is a rotation — it should preserve vector norms."""
        rope = RotaryEmbedding(dim=dim)
        q = torch.randn(2, 4, 8, dim)
        k = torch.randn(2, 4, 8, dim)
        q_rot, k_rot = rope(q, k)
        torch.testing.assert_close(
            q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_different_positions_differ(self, dim):
        """Different sequence positions should produce different rotations."""
        rope = RotaryEmbedding(dim=dim)
        q = torch.ones(1, 1, 4, dim)
        k = torch.ones(1, 1, 4, dim)
        q_rot, _ = rope(q, k)
        # Consecutive positions should produce different results
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1])

    def test_extends_beyond_max_seq_len(self, dim):
        """Should handle sequences longer than max_seq_len gracefully."""
        rope = RotaryEmbedding(dim=dim, max_seq_len=8)
        q = torch.randn(1, 1, 32, dim)
        k = torch.randn(1, 1, 32, dim)
        q_rot, k_rot = rope(q, k)  # Should not raise
        assert q_rot.shape == q.shape

    def test_no_learned_parameters(self, dim):
        """RotaryEmbedding should have no trainable parameters."""
        rope = RotaryEmbedding(dim=dim)
        assert sum(p.numel() for p in rope.parameters()) == 0

    def test_gradient_flows_through(self, dim):
        rope = RotaryEmbedding(dim=dim)
        q = torch.randn(2, 4, 8, dim, requires_grad=True)
        k = torch.randn(2, 4, 8, dim, requires_grad=True)
        q_rot, k_rot = rope(q, k)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None

    def test_custom_base(self, dim):
        """LLaMA 3 uses base=500000 — should work without errors."""
        rope = RotaryEmbedding(dim=dim, base=500_000)
        q = torch.randn(2, 4, 8, dim)
        k = torch.randn(2, 4, 8, dim)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


class TestFusedRotaryEmbedding:
    """Tests for the triton-fused RoPE kernel (GPU only)."""

    @pytest.fixture
    def rope(self, dim):
        return RotaryEmbedding(dim=dim)

    @pytest.fixture
    def rope_cuda(self, dim):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return RotaryEmbedding(dim=dim, device="cuda")

    def test_matches_vanilla(self, rope_cuda, dim):
        """Fused and vanilla should produce identical results."""
        rope = rope_cuda
        q = torch.randn(2, 4, 8, dim, device="cuda")
        k = torch.randn(2, 4, 8, dim, device="cuda")
        q_ref, k_ref = rope(q, k)
        cos, sin = rope.cos_cache[:8], rope.sin_cache[:8]
        q_fused, k_fused = rope._fused_forward(q, k, cos, sin)
        torch.testing.assert_close(q_fused, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_fused, k_ref, atol=1e-5, rtol=1e-5)

    def test_use_fused_flag(self, dim):
        """Setting use_fused=True should dispatch to the fused kernel."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rope = RotaryEmbedding(dim=dim, use_fused=True).to("cuda")
        q = torch.randn(2, 4, 8, dim, device="cuda")
        k = torch.randn(2, 4, 8, dim, device="cuda")
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_various_shapes(self):
        """Fused kernel should handle various batch/head/seq/dim combos."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        for dim in [32, 64, 128]:
            rope = RotaryEmbedding(dim=dim, device="cuda")
            q = torch.randn(1, 4, 16, dim, device="cuda")
            k = torch.randn(1, 4, 16, dim, device="cuda")
            q_ref, k_ref = rope(q, k)
            cos, sin = rope.cos_cache[:16], rope.sin_cache[:16]
            q_fused, k_fused = rope._fused_forward(q, k, cos, sin)
            torch.testing.assert_close(q_fused, q_ref, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(k_fused, k_ref, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self, dim):
        """Gradients should flow through the fused kernel."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rope = RotaryEmbedding(dim=dim, device="cuda")
        q = torch.randn(2, 4, 8, dim, device="cuda", requires_grad=True)
        k = torch.randn(2, 4, 8, dim, device="cuda", requires_grad=True)
        q_rot, k_rot = rope._fused_forward(q, k, rope.cos_cache[:8], rope.sin_cache[:8])
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None

    def test_gradient_correctness(self, dim):
        """Fused kernel gradients must numerically match the vanilla path's.

        RoPE's forward is a per-position rotation matrix [[c, -s], [s, c]] applied
        to each (x0, x1) pair; its backward is that matrix's transpose, not the
        forward transform reapplied unchanged. A prior bug reused the forward
        kernel verbatim for backward, which silently produced wrong gradients
        for every dim with sin != 0. `torch.testing.assert_close` on the grads
        (not just shape/is-not-None) is what catches a regression here.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rope = RotaryEmbedding(dim=dim, device="cuda")
        cos, sin = rope.cos_cache[:8], rope.sin_cache[:8]

        torch.manual_seed(0)
        q_base = torch.randn(2, 4, 8, dim, device="cuda")
        k_base = torch.randn(2, 4, 8, dim, device="cuda")
        grad_q_up = torch.randn_like(q_base)
        grad_k_up = torch.randn_like(k_base)

        q_ref = q_base.clone().requires_grad_(True)
        k_ref = k_base.clone().requires_grad_(True)
        q_rot_ref, k_rot_ref = rope(q_ref, k_ref)
        q_rot_ref.backward(grad_q_up, retain_graph=True)
        k_rot_ref.backward(grad_k_up)

        q_fused = q_base.clone().requires_grad_(True)
        k_fused = k_base.clone().requires_grad_(True)
        q_rot_fused, k_rot_fused = rope._fused_forward(q_fused, k_fused, cos, sin)
        q_rot_fused.backward(grad_q_up, retain_graph=True)
        k_rot_fused.backward(grad_k_up)

        torch.testing.assert_close(q_fused.grad, q_ref.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_fused.grad, k_ref.grad, atol=1e-4, rtol=1e-4)

    def test_fused_forward_function(self, dim):
        """Direct call to fused_rope_forward should match."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from torchembed._triton import fused_rope_forward

        rope = RotaryEmbedding(dim=dim, device="cuda")
        q = torch.randn(2, 4, 8, dim, device="cuda")
        k = torch.randn(2, 4, 8, dim, device="cuda")
        q_ref, k_ref = rope(q, k)
        cos, sin = rope.cos_cache[:8], rope.sin_cache[:8]
        q_fused, k_fused = fused_rope_forward(q, k, cos, sin)
        torch.testing.assert_close(q_fused, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_fused, k_ref, atol=1e-5, rtol=1e-5)


class TestALiBiEmbedding:
    def test_output_shape(self, batch_size, num_heads, seq_len):
        alibi = ALiBiEmbedding(num_heads=num_heads)
        attn = torch.randn(batch_size, num_heads, seq_len, seq_len)
        out = alibi(attn)
        assert out.shape == attn.shape

    def test_bias_is_non_positive(self, num_heads, seq_len):
        """ALiBi bias should penalize (not reward) distance, so bias ≤ 0."""
        alibi = ALiBiEmbedding(num_heads=num_heads)
        attn = torch.zeros(1, num_heads, seq_len, seq_len)
        out = alibi(attn)
        assert (out <= 1e-6).all(), "ALiBi bias should be non-positive"

    def test_non_power_of_two_heads(self):
        """Should work for non-power-of-2 head counts (e.g. 6, 10)."""
        for n_heads in [3, 5, 6, 10, 12]:
            alibi = ALiBiEmbedding(num_heads=n_heads)
            attn = torch.randn(2, n_heads, 8, 8)
            out = alibi(attn)
            assert out.shape == attn.shape

    def test_no_parameters(self, num_heads):
        alibi = ALiBiEmbedding(num_heads=num_heads)
        assert sum(p.numel() for p in alibi.parameters()) == 0

    def test_diagonal_is_zero_penalty(self, num_heads):
        """Self-attention positions (distance=0) should have zero bias."""
        alibi = ALiBiEmbedding(num_heads=num_heads, max_seq_len=16)
        bias = alibi.bias  # (heads, seq, seq)
        diagonal = torch.diagonal(bias, dim1=-2, dim2=-1)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))

    def test_gradient_flows(self, batch_size, num_heads, seq_len):
        alibi = ALiBiEmbedding(num_heads=num_heads)
        attn = torch.randn(batch_size, num_heads, seq_len, seq_len, requires_grad=True)
        out = alibi(attn)
        out.sum().backward()
        assert attn.grad is not None


class TestSinusoidalEmbedding:
    def test_output_shape(self, batch_size, seq_len, dim):
        emb = SinusoidalEmbedding(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        out = emb(x)
        assert out.shape == x.shape

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            SinusoidalEmbedding(dim=65)

    def test_no_parameters_by_default(self, dim):
        emb = SinusoidalEmbedding(dim=dim)
        assert sum(p.numel() for p in emb.parameters()) == 0

    def test_learned_scale_adds_one_parameter(self, dim):
        emb = SinusoidalEmbedding(dim=dim, learned_scale=True)
        params = list(emb.parameters())
        assert len(params) == 1
        assert params[0].numel() == 1

    def test_learned_scale_forward(self, batch_size, seq_len, dim):
        """Forward pass with learned_scale=True should work and change output."""
        emb = SinusoidalEmbedding(dim=dim, learned_scale=True)
        x = torch.randn(batch_size, seq_len, dim)
        out = emb(x)
        assert out.shape == x.shape

    def test_deterministic(self, batch_size, seq_len, dim):
        """Same input should always produce same output."""
        emb = SinusoidalEmbedding(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        out1 = emb(x)
        out2 = emb(x)
        torch.testing.assert_close(out1, out2)

    def test_different_positions_differ(self, dim):
        """Different positions should produce different encodings."""
        emb = SinusoidalEmbedding(dim=dim)
        x = torch.zeros(1, 8, dim)
        out = emb(x)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_dropout_variant(self, batch_size, seq_len, dim):
        emb = SinusoidalEmbedding(dim=dim, dropout=0.1)
        emb.train()
        x = torch.randn(batch_size, seq_len, dim)
        out = emb(x)
        assert out.shape == x.shape


class TestLearnedPositionalEmbedding:
    def test_output_shape(self, batch_size, seq_len, dim):
        emb = LearnedPositionalEmbedding(max_seq_len=512, dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        out = emb(x)
        assert out.shape == x.shape

    def test_has_parameters(self, dim):
        emb = LearnedPositionalEmbedding(max_seq_len=64, dim=dim)
        assert sum(p.numel() for p in emb.parameters()) == 64 * dim

    def test_offset_parameter(self, dim):
        """Offset should shift the position indices."""
        emb = LearnedPositionalEmbedding(max_seq_len=128, dim=dim)
        x = torch.randn(1, 4, dim)
        out_0 = emb(x, offset=0)
        out_4 = emb(x, offset=4)
        assert not torch.allclose(out_0, out_4)

    def test_gradient_flows(self, batch_size, seq_len, dim):
        emb = LearnedPositionalEmbedding(max_seq_len=512, dim=dim)
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        out = emb(x)
        out.sum().backward()
        assert x.grad is not None
