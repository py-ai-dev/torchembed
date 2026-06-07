"""
Positional embedding strategies.

Includes:
- RotaryEmbedding (RoPE): used in LLaMA, Mistral, Falcon, etc.
- ALiBiEmbedding: Attention with Linear Biases (Press et al., 2022)
- SinusoidalEmbedding: original fixed positional encoding (Vaswani et al., 2017)
- LearnedPositionalEmbedding: standard learned position lookup table
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes position by rotating query and key vectors in 2D subspaces.
    Unlike additive embeddings, RoPE is applied directly to Q and K inside
    the attention layer, not to the input sequence.

    Used in: LLaMA, Mistral, Falcon, PaLM, GPT-NeoX, and most modern LLMs.

    Reference:
        Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        https://arxiv.org/abs/2104.09864

    Args:
        dim: Head dimension (must be even). Typically d_model // num_heads.
        max_seq_len: Maximum sequence length to precompute. Longer sequences
            will be computed on the fly.
        base: Base for the geometric progression of frequencies. Default 10000
            matches the original paper. LLaMA 3 uses 500000.
        use_fused: If True, uses a fused triton kernel for the forward pass
            (requires GPU and ``triton``). Default False.
        device: Device to create buffers on.

    Example::

        rope = RotaryEmbedding(dim=64)
        q = torch.randn(2, 8, 16, 64)  # (batch, heads, seq, dim)
        k = torch.randn(2, 8, 16, 64)
        q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10_000,
        use_fused: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.use_fused = use_fused

        # Precompute inverse frequencies: shape (dim/2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )  # noqa: E501
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        t = torch.arange(seq_len, device=device or self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate the last dimension by splitting and negating halves."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _vanilla_forward(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor]:  # noqa: E501
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def _fused_forward(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor]:  # noqa: E501
        from torchembed._triton import fused_rope_forward

        return fused_rope_forward(q, k, cos, sin)

    def forward(self, q: Tensor, k: Tensor, seq_dim: int = -2) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (..., seq_len, dim).
            k: Key tensor of shape (..., seq_len, dim).
            seq_dim: Dimension along which sequence length lives. Default -2.

        Returns:
            Tuple of (rotated_q, rotated_k) with the same shapes as inputs.
        """
        seq_len = q.shape[seq_dim]

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, q.device)
            self.max_seq_len = seq_len

        cos = self.cos_cache[:seq_len].to(device=q.device)
        sin = self.sin_cache[:seq_len].to(device=q.device)

        if self.use_fused and q.is_cuda and k.is_cuda:
            try:
                return self._fused_forward(q, k, cos, sin)
            except (ImportError, RuntimeError):
                pass

        return self._vanilla_forward(q, k, cos, sin)


class ALiBiEmbedding(nn.Module):
    """Attention with Linear Biases (ALiBi).

    Instead of adding positional information to token embeddings, ALiBi
    adds a fixed, non-learned bias to attention scores that penalizes
    distance between tokens linearly. This allows strong extrapolation
    to longer sequences than seen during training.

    Used in: BLOOM, MPT, and other long-context models.

    Reference:
        Press et al., "Train Short, Test Long: Attention with Linear Biases
        Enables Input Length Extrapolation" https://arxiv.org/abs/2108.12409

    Args:
        num_heads: Number of attention heads. Each head gets a different slope.
        max_seq_len: Maximum sequence length to precompute biases for.

    Example::

        alibi = ALiBiEmbedding(num_heads=8)
        # attn_scores: (batch, heads, seq, seq)
        attn_scores = torch.randn(2, 8, 16, 16)
        biased_scores = alibi(attn_scores)
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.num_heads = num_heads

        slopes = self._get_slopes(num_heads)  # (num_heads,)
        bias = self._build_bias(slopes, max_seq_len)  # (num_heads, seq, seq)
        self.register_buffer("bias", bias, persistent=False)

    @staticmethod
    def _get_slopes(num_heads: int) -> Tensor:
        """Compute ALiBi slopes following the original paper's geometric sequence."""
        # Nearest power of 2 >= num_heads
        n = 2 ** math.ceil(math.log2(num_heads))
        slopes = torch.pow(2, -torch.arange(1, n + 1) * (8 / n))
        if n > num_heads:
            # Interleave to handle non-power-of-2 head counts
            slopes = torch.cat([slopes[1::2], slopes[::2]])[:num_heads]
        return slopes

    @staticmethod
    def _build_bias(slopes: Tensor, max_seq_len: int) -> Tensor:
        positions = torch.arange(max_seq_len)
        # Relative distances: (seq, seq) lower-triangular distance matrix
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        dist = -dist.abs()
        # Scale by each head's slope: (num_heads, seq, seq)
        bias = slopes.unsqueeze(-1).unsqueeze(-1) * dist.unsqueeze(0)
        return bias

    def forward(self, attn_scores: Tensor) -> Tensor:
        """Add ALiBi positional bias to attention scores.

        Args:
            attn_scores: Attention logits of shape (batch, heads, seq_q, seq_k).

        Returns:
            Attention scores with ALiBi bias added, same shape as input.
        """
        seq_len = attn_scores.shape[-1]
        bias = self.bias[:, :seq_len, :seq_len]  # (heads, seq, seq)
        return attn_scores + bias.unsqueeze(0)  # broadcast over batch


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding from "Attention Is All You Need".

    Adds a non-learned, frequency-based positional signal to input embeddings.
    The encoding is deterministic and can generalize slightly beyond the
    training sequence length.

    Reference:
        Vaswani et al., "Attention Is All You Need" https://arxiv.org/abs/1706.03762

    Args:
        dim: Embedding dimension (must be even).
        max_seq_len: Maximum supported sequence length.
        dropout: Optional dropout rate applied after adding the embedding.
        learned_scale: If True, adds a single learned scalar to scale the
            sinusoidal signal (a light touch of trainability).

    Example::

        emb = SinusoidalEmbedding(dim=512)
        x = torch.randn(2, 16, 512)   # (batch, seq, dim)
        x = emb(x)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        learned_scale: bool = False,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.ones(1)) if learned_scale else None

        pe = self._build_pe(dim, max_seq_len)  # (1, max_seq_len, dim)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(dim: int, max_seq_len: int) -> Tensor:
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(1, max_seq_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Add sinusoidal positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Tensor of same shape with positional encoding added.
        """
        seq_len = x.shape[1]
        pe = self.pe[:, :seq_len, :]
        if self.scale is not None:
            pe = pe * self.scale
        return self.dropout(x + pe)


class LearnedPositionalEmbedding(nn.Module):
    """Standard learned positional embedding.

    A simple lookup table mapping each position index to a learnable vector.
    Used in BERT, GPT-2, and many other models. Simpler than sinusoidal but
    cannot extrapolate beyond the training sequence length.

    Args:
        max_seq_len: Maximum sequence length (vocabulary size for positions).
        dim: Embedding dimension.
        dropout: Optional dropout rate.
        padding_idx: If set, the embedding at this index is not updated.

    Example::

        emb = LearnedPositionalEmbedding(max_seq_len=512, dim=768)
        x = torch.randn(2, 16, 768)
        x = emb(x)
    """

    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Add learned positional embeddings to input.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            offset: Starting position index. Useful for KV-cache inference
                where you process one token at a time.

        Returns:
            Tensor of same shape with positional embeddings added.
        """
        seq_len = x.shape[1]
        positions = torch.arange(offset, offset + seq_len, device=x.device)
        return self.dropout(x + self.embedding(positions))
