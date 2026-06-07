"""Triton kernels for accelerated embedding operations."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def fused_rope_forward(q, k, cos, sin):
    """Apply RoPE to Q and K using a fused triton kernel.

    Falls back to a clear ImportError if triton is not installed.

    Args:
        q: Query tensor of shape (..., seq_len, dim).
        k: Key tensor of shape (..., seq_len, dim).
        cos: Cosine cache of shape (seq_len, dim).
        sin: Sine cache of shape (seq_len, dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with the same shapes as inputs.
    """
    if triton is None:
        raise ImportError(
            "triton is required. Install it with: pip install triton"
        )
    return _fused_rope_forward_impl(q, k, cos, sin)


if triton is not None:

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
        b = tl.program_id(0)
        s = tl.program_id(1)

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < HALF_D

        x0 = tl.load(
            x_ptr + b * stride_x_b + s * stride_x_s + offsets * stride_x_d,
            mask=mask,
        )
        x1 = tl.load(
            x_ptr + b * stride_x_b + s * stride_x_s
            + (HALF_D + offsets) * stride_x_d,
            mask=mask,
        )

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

    def _fused_rope_forward_impl(q, k, cos, sin):
        q_out = _FusedRoPE.apply(q, cos, sin)
        k_out = _FusedRoPE.apply(k, cos, sin)
        return q_out, k_out
