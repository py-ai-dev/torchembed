# torchembed

<p align="center">
  <a href="https://github.com/liodon-ai/torchembed">
    <img src="assets/logo/torchembed.png" width="750" height="300" alt="torchembed Logo"/>
  </a>
</p>

**Modern embedding strategies for PyTorch — the ones missing from `torch.nn`.**

`torch.nn` gives you `nn.Embedding` (a lookup table). That's it. The moment you work with continuous inputs, modern transformer architectures, coordinates, time, or tabular data, you're on your own — copy-pasting RoPE implementations across projects. **torchembed** is a single, well-tested, pip-installable home for all of them.

> **torchembed is integrated into [DeepSpeed](https://github.com/microsoft/DeepSpeed) and ships as a dependency.** The fused RoPE kernel is used by DeepSpeed's transformer engine for accelerated attention.

## Features

- **Positional embeddings** — `RotaryEmbedding` (RoPE, LLaMA/Mistral-style), `ALiBiEmbedding` (long-context extrapolation), `SinusoidalEmbedding`, `LearnedPositionalEmbedding`.
- **Fourier features** — `RandomFourierFeatures` (coordinate/kernel encoding), `LearnedFourierFeatures`, `GaussianFourierProjection` (diffusion timestep embedding).
- **Categorical embeddings** — `EntityEmbedding` and `MultiCategoricalEmbedding` for tabular data, with auto-sized embedding dimensions.
- **Patch embeddings** — `PatchEmbedding` (ViT) and `TubeletEmbedding` (video transformers: VideoMAE, ViViT).
- **Temporal embeddings** — `CyclicEmbedding`, `TimestampEmbedding`, `FrequencyEmbedding` for hour/day/month and periodic time series.
- **Fused Triton kernels** — optional GPU-accelerated RoPE, ~4x faster than plain PyTorch and ~2x faster than `torch.compile`, with full autograd support and automatic CPU fallback.
- **Zero required dependencies beyond PyTorch** — no transformers, no numpy, nothing pulled in you didn't ask for.

## Install

```bash
pip install torchembed
```

For GPU-accelerated kernels:

```bash
pip install torchembed[triton]
```

Requires Python >= 3.9 and PyTorch >= 2.0.

## Triton Kernels

torchembed includes optional Triton-accelerated kernels for GPU. Install with `pip install torchembed[triton]`, then enable with `use_fused=True`:

```python
rope = RotaryEmbedding(dim=64, use_fused=True)
```

The fused RoPE kernel combines cos/sin lookup, rotate-half, and element-wise multiplication into a single Triton launch, reducing memory traffic. Supports any even dim (32, 64, 128, etc.) and full autograd support. Falls back to vanilla PyTorch automatically when Triton is unavailable or inputs are on CPU.

RoPE forward pass on NVIDIA GB10 (float16):

![RoPE benchmark — forward pass latency and Triton speedup across sequence lengths 256–32K](assets/benchmark_rope.svg)

![RoPE benchmark bar charts — grouped latency and speedup bars across sequence lengths 256–32K](assets/benchmark_bars.svg)

<details>
<summary>Full numbers</summary>

| Shape (B,H,S,D,rot) | PyTorch (ms) | torch.compile (ms) | Triton (ms) | Speedup |
|---|---|---|---|---|
| (1,32,2048,128,128) | 1.364 | 0.620 | 0.305 | **4.47x** |
| (1,32,4096,128,128) | 2.946 | 1.284 | 0.579 | **5.09x** |
| (1,32,8192,128,128) | 5.909 | 2.476 | 1.224 | **4.83x** |
| (2,32,2048,128,128) | 2.949 | 1.293 | 0.630 | **4.68x** |
| (1,32,2048,256,128) | 2.869 | 1.323 | 0.646 | **4.44x** |

</details>

The fused Triton kernel is **4–7× faster than pure PyTorch** and **~2× faster than `torch.compile`**. `torch.compile` reduces overhead but cannot eliminate intermediate tensor allocations from `chunk`/`cat` — the fused kernel reads and writes each element exactly once.

**Does `view_as_complex` help?** No — on float16 it's roughly the same speed as rotate-half or slightly slower. The float16→float32 cast required by `torch.view_as_complex` eats the savings from avoiding the `cat`. `torch.compile` makes it worse, not better (inductor has no complex op codegen). The Triton kernel beats both approaches by 3.5–9×.

## Python API

### Rotary Embedding (RoPE) — LLaMA / Mistral style

```python
import torch
from torchembed.positional import RotaryEmbedding

rope = RotaryEmbedding(dim=64)  # head_dim

# Inside your attention layer:
q = torch.randn(batch, heads, seq_len, 64)
k = torch.randn(batch, heads, seq_len, 64)
q, k = rope(q, k)  # apply rotation in-place
```

RoPE has no trainable parameters and preserves vector norms (it's a pure rotation).
The default base of 10,000 matches the original paper; use `base=500_000` for LLaMA 3.

For GPU-accelerated inference:

```python
rope = RotaryEmbedding(dim=128, use_fused=True).to("cuda")
q, k = rope(q.cuda(), k.cuda())
```

### ALiBi — long context with length extrapolation

```python
from torchembed.positional import ALiBiEmbedding

alibi = ALiBiEmbedding(num_heads=8)

# After computing raw attention scores:
attn_scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
attn_scores = alibi(attn_scores)   # adds learned distance penalty
attn_weights = attn_scores.softmax(-1)
```

### Gaussian Fourier Projection — diffusion model timestep embedding

```python
from torchembed.fourier import GaussianFourierProjection
import torch.nn as nn

class DiffusionTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fourier = GaussianFourierProjection(embed_dim=embed_dim, scale=16)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t):
        return self.mlp(self.fourier(t))

t_emb = DiffusionTimeEmbedding(embed_dim=256)
t = torch.rand(32)   # normalized timesteps
emb = t_emb(t)       # (32, 256) — condition your UNet on this
```

### ViT Patch Embedding

```python
from torchembed.patch import PatchEmbedding

patch_emb = PatchEmbedding(
    image_size=224,
    patch_size=16,
    embed_dim=768,
)

images = torch.randn(4, 3, 224, 224)
tokens = patch_emb(images)    # (4, 196, 768)
print(patch_emb.num_patches)  # 196
```

### Tubelet Embedding — Video Transformers

```python
from torchembed.patch import TubeletEmbedding

tubelet_emb = TubeletEmbedding(
    image_size=224,
    patch_size=16,
    tubelet_size=2,
    embed_dim=768,
)

video = torch.randn(2, 3, 16, 224, 224)   # (B, C, T, H, W)
tokens = tubelet_emb(video)                # (2, 1568, 768)
# 1568 = (16/2) * (224/16) * (224/16) = 8 * 14 * 14
```

### Tabular categorical features

```python
from torchembed.categorical import MultiCategoricalEmbedding

# A tabular dataset with 3 categorical columns:
# country (50 unique values), day of week (7), product category (120)
emb = MultiCategoricalEmbedding(cardinalities=[50, 7, 120])
print(emb.output_dim)   # sum of auto-sized embed dims

x = torch.stack([country_ids, dow_ids, category_ids], dim=1)   # (batch, 3)
features = emb(x)   # (batch, output_dim)
```

### Cyclic time features

```python
from torchembed.temporal import CyclicEmbedding
import torch

hour_enc  = CyclicEmbedding(period=24)
dow_enc   = CyclicEmbedding(period=7)
month_enc = CyclicEmbedding(period=12)

hour   = torch.tensor([0.0, 6.0, 12.0, 18.0])
dow    = torch.tensor([0.0, 1.0, 2.0, 3.0])
month  = torch.tensor([1.0, 4.0, 7.0, 10.0])

time_features = torch.cat([
    hour_enc(hour),    # (4, 2)
    dow_enc(dow),      # (4, 2)
    month_enc(month),  # (4, 2)
], dim=-1)             # (4, 6)
```

### Random Fourier Features for coordinate encoding

```python
from torchembed.fourier import RandomFourierFeatures

# Encode 2D spatial coordinates for a neural field / NeRF-style model
rff = RandomFourierFeatures(in_features=2, out_features=256, sigma=1.0)

coords = torch.rand(1024, 2)   # (x, y) pairs in [0, 1]
features = rff(coords)          # (1024, 256)
```

### Frequency Embedding — learnable periodic decomposition

```python
from torchembed.temporal import FrequencyEmbedding

# Discover periodic structure in time series automatically
freq_emb = FrequencyEmbedding(embed_dim=32)

t = torch.linspace(0, 100, 512).unsqueeze(0)   # (1, 512) time steps
out = freq_emb(t)                               # (1, 512, 33)
# 33 = 1 linear trend + 32 sinusoidal components
```

## Documentation

Full API reference: [liodon-ai.github.io/torchembed](https://liodon-ai.github.io/torchembed/torchembed.html). Hand-written guides for each module are in [`docs/`](docs/):

| Module | Guide |
|---|---|
| Positional (RoPE, ALiBi, Sinusoidal, Learned) | [docs/positional.md](docs/positional.md) |
| Fourier features | [docs/fourier.md](docs/fourier.md) |
| Categorical embeddings | [docs/categorical.md](docs/categorical.md) |
| Patch embeddings (ViT, video) | [docs/patch.md](docs/patch.md) |
| Temporal embeddings | [docs/temporal.md](docs/temporal.md) |

## Development

```bash
pip install torchembed[dev]
pytest
```

Building API docs:

```bash
make docs         # generates docs/api/
make docs-serve   # serves at http://localhost:8080
```

API docs are generated from Google-style docstrings using [pdoc](https://pdoc.dev/).

## Related projects

| Project | Focus |
|---|---|
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | Full training optimization suite including fused RoPE + RMSNorm + SwiGLU with automatic model patching |
| [torchnorm](https://github.com/liodon-ai/torchnorm) | Companion library: fused RMSNorm, FusedAddRMSNorm, LayerNorm, GroupNorm kernels |
| [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) | Pure PyTorch RoPE, no Triton |
| [flash-attn](https://github.com/Dao-AILab/flash-attention) | Flash Attention with built-in fused RoPE support |

**torchembed vs Liger Kernel for RoPE**: Liger patches model classes wholesale — the right choice when you want `use_liger_kernel=True` and one-line integration. torchembed provides individual `nn.Module` drop-ins (both `rotate_half` and `adjacent_pairs` conventions) for use in custom architectures, ViT patch embeddings, diffusion models, and audio models where full model patching isn't the goal.

## Contributing

Contributions welcome! If there's an embedding strategy you find yourself copy-pasting into projects, open a PR with a clear docstring (paper reference included), tests covering shape/gradients/key mathematical properties, and a README example.

## License

MIT
