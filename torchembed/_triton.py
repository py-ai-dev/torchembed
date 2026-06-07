"""Triton kernels for accelerated embedding operations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_b,
    stride_x_s,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    stride_out_b,
    stride_out_s,
    stride_out_d,
    HALF_D,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE forward: out = x * cos + rotate_half(x) * sin.

    rotate_half splits the last dimension in two halves and swaps them:
      rotate_half(x)[i] = -x[i + dim/2]  for i < dim/2
      rotate_half(x)[i] =  x[i - dim/2]  for i >= dim/2

    Since cos/sin are duplicated across the two halves, each program
    processes pairs (i, i + half_dim) together:
      out[i]        = x[i] * cos[i] - x[i+half] * sin[i]
      out[i+half]   = x[i+half] * cos[i] + x[i] * sin[i]
    """
    b = tl.program_id(0)
    s = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HALF_D

    # First half of x
    x0 = tl.load(
        x_ptr + b * stride_x_b + s * stride_x_s + offsets * stride_x_d,
        mask=mask,
    )
    # Second half of x
    x1 = tl.load(
        x_ptr + b * stride_x_b + s * stride_x_s + (HALF_D + offsets) * stride_x_d,
        mask=mask,
    )

    # cos and sin at first-half positions (same as second-half)
    c = tl.load(
        cos_ptr + s * stride_cos_s + offsets * stride_cos_d,
        mask=mask,
    )
    sn = tl.load(
        sin_ptr + s * stride_cos_s + offsets * stride_cos_d,
        mask=mask,
    )

    out0 = x0 * c - x1 * sn
    out1 = x1 * c + x0 * sn

    base = out_ptr + b * stride_out_b + s * stride_out_s
    tl.store(base + offsets * stride_out_d, out0, mask=mask)
    tl.store(base + (HALF_D + offsets) * stride_out_d, out1, mask=mask)


class _FusedRoPE(torch.autograd.Function):
    """Autograd wrapper for the fused RoPE kernel.

    RoPE is an orthogonal transformation (rotation), so the backward pass
    applies the same forward kernel to the gradient.
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _fused_rope_forward_core(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad_input = _fused_rope_forward_core(grad_output, cos, sin)
        return grad_input, None, None


def _fused_rope_forward_core(x, cos, sin):
    orig_shape = x.shape
    *leading, seq_len, dim = orig_shape
    batch_total = 1
    for d in leading:
        batch_total *= d

    x_2d = x.reshape(batch_total, seq_len, dim)
    out = torch.empty_like(x_2d)

    half_dim = dim // 2
    block_size = triton.next_power_of_2(half_dim)

    grid = (batch_total, seq_len)
    _fused_rope_kernel[grid](
        x_2d,
        cos,
        sin,
        out,
        x_2d.stride(0),
        x_2d.stride(1),
        x_2d.stride(2),
        cos.stride(0),
        cos.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        half_dim,
        BLOCK_SIZE=block_size,
    )
    return out.reshape(orig_shape)


def fused_rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K using a fused triton kernel.

    The backward pass is also fused (RoPE is an orthogonal transformation).

    Args:
        q: Query tensor of shape (..., seq_len, dim).
        k: Key tensor of shape (..., seq_len, dim).
        cos: Cosine cache of shape (seq_len, dim).
        sin: Sine cache of shape (seq_len, dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with the same shapes as inputs.
    """
    q_out = _FusedRoPE.apply(q, cos, sin)
    k_out = _FusedRoPE.apply(k, cos, sin)
    return q_out, k_out
