"""A small, real transformer block expressed functionally, in fp64.

Two properties are deliberate:

1. **Functional, not `nn.Module`.** `apply_block(x, params, cfg)` takes its
   weights as an explicit dict. Streaming means the weights are not attributes
   of a persistent object -- they arrive, get used, and leave -- so the compute
   has to be callable against whichever copy is currently resident.

2. **Everything is smooth.** LayerNorm, softmax and the exact (erf) GELU are all
   C-infinity, so `gradcheck`'s central differences converge cleanly. The tanh
   GELU approximation and anything ReLU-like would introduce kinks that produce
   spurious finite-difference mismatches, which would be indistinguishable from
   a real streaming bug.

The block is a pre-norm attention + MLP residual pair -- the same shape as a
MiniCPM-V vision-encoder layer, just tiny. Dimensions are kept small because
gradcheck costs two forward passes per input element.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DTYPE = torch.float64

# Order matters: it fixes the flattened parameter layout used by gradcheck.
PARAM_NAMES: tuple[str, ...] = (
    "ln1_w",
    "ln1_b",
    "qkv_w",
    "qkv_b",
    "attn_out_w",
    "attn_out_b",
    "ln2_w",
    "ln2_b",
    "ff1_w",
    "ff1_b",
    "ff2_w",
    "ff2_b",
)


@dataclass(frozen=True)
class BlockConfig:
    d_model: int = 6
    n_heads: int = 2
    d_ff: int = 12
    ln_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def shapes(self) -> dict[str, tuple[int, ...]]:
        d, f = self.d_model, self.d_ff
        return {
            "ln1_w": (d,),
            "ln1_b": (d,),
            "qkv_w": (3 * d, d),
            "qkv_b": (3 * d,),
            "attn_out_w": (d, d),
            "attn_out_b": (d,),
            "ln2_w": (d,),
            "ln2_b": (d,),
            "ff1_w": (f, d),
            "ff1_b": (f,),
            "ff2_w": (d, f),
            "ff2_b": (d,),
        }


def init_block_params(
    cfg: BlockConfig, seed: int, requires_grad: bool = True
) -> dict[str, torch.Tensor]:
    """Deterministically initialise one block's host-side master weights."""
    gen = torch.Generator().manual_seed(seed)
    params: dict[str, torch.Tensor] = {}
    for name, shape in cfg.shapes().items():
        t = torch.randn(*shape, generator=gen, dtype=DTYPE)
        if name.endswith("_w") and name.startswith("ln"):
            # LayerNorm gain: near 1, but perturbed so its gradient is exercised.
            t = 1.0 + 0.05 * t
        elif name.endswith("_b"):
            t = 0.05 * t
        else:
            t = t / math.sqrt(shape[-1])
        params[name] = t.requires_grad_(requires_grad)
    return params


def init_layers(
    cfg: BlockConfig, n_layers: int, seed: int = 0, requires_grad: bool = True
) -> list[dict[str, torch.Tensor]]:
    return [
        init_block_params(cfg, seed=seed * 1000 + i, requires_grad=requires_grad)
        for i in range(n_layers)
    ]


def layer_bytes(layer: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in layer.values())


def _layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * w + b


def _lora_proj(xin: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Low-rank projection residual: `(xin @ A) @ B`.

    `A: (in, r)`, `B: (r, out)`; result is `(..., out)`, broadcast with the base
    linear's output on the SAME broadcast dims, so the residual threads straight
    into the block's graph (and thus into StreamedBlock's recompute).
    """
    return (xin @ A) @ B


def apply_block(
    x: torch.Tensor,
    p: dict[str, torch.Tensor],
    cfg: BlockConfig,
    lora: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Pre-norm attention + MLP block. `p` may be host or device weights.

    `lora` optionally injects a LoRA residual into each targeted projection:
    `{target: (A, B)}` with `A: (in, r)`, `B: (r, out)`. When `lora` is None the
    computation is bit-identical to the Phase 2 block, which is what keeps the
    resident/streamed gate apples-to-apples.
    """
    B, T, D = x.shape
    H, dh = cfg.n_heads, cfg.d_head

    h = _layer_norm(x, p["ln1_w"], p["ln1_b"], cfg.ln_eps)
    qkv = h @ p["qkv_w"].transpose(0, 1) + p["qkv_b"]
    if lora and lora.get("qkv") is not None:
        qkv = qkv + _lora_proj(h, *lora["qkv"])
    q, k, v = qkv.split(D, dim=-1)
    # (B, T, D) -> (B, H, T, dh)
    q = q.view(B, T, H, dh).transpose(1, 2)
    k = k.view(B, T, H, dh).transpose(1, 2)
    v = v.view(B, T, H, dh).transpose(1, 2)

    att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    att = att.softmax(dim=-1)
    o = (att @ v).transpose(1, 2).reshape(B, T, D)
    aout = o @ p["attn_out_w"].transpose(0, 1) + p["attn_out_b"]
    if lora and lora.get("attn_out") is not None:
        aout = aout + _lora_proj(o, *lora["attn_out"])
    x = x + aout

    h = _layer_norm(x, p["ln2_w"], p["ln2_b"], cfg.ln_eps)
    f1 = h @ p["ff1_w"].transpose(0, 1) + p["ff1_b"]
    if lora and lora.get("ff1") is not None:
        f1 = f1 + _lora_proj(h, *lora["ff1"])
    g = F.gelu(f1)  # exact erf GELU
    out = g @ p["ff2_w"].transpose(0, 1) + p["ff2_b"]
    if lora and lora.get("ff2") is not None:
        out = out + _lora_proj(g, *lora["ff2"])
    return x + out


def resident_stack_forward(
    x: torch.Tensor, layers: list[dict[str, torch.Tensor]], cfg: BlockConfig
) -> torch.Tensor:
    """The full-VRAM baseline: every layer resident, ordinary autograd."""
    for layer in layers:
        x = apply_block(x, layer, cfg)
    return x


def make_input(cfg: BlockConfig, batch: int = 2, seq: int = 3, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(
        batch, seq, cfg.d_model, generator=gen, dtype=DTYPE
    ).requires_grad_(True)
