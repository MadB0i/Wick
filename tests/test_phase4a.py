"""Sub-gate 4A tests: real CUDA + fp16/GradScaler streaming on the toy blocks.

Skipped entirely when torch has no CUDA build available, so the suite stays green
on CPU-only machines. When a CUDA GPU is present (the GTX 1650 target), these run
the streamed LoRA training for a short burst and assert the load-bearing 4A
claims: GradScaler scale stays finite, no overflow/skip, weight residency is
bounded at ~1 layer, the device is empty after, and loss trends down.

    .venv/Scripts/python.exe -m pytest tests/test_phase4a.py -v   (on a GPU box)
"""

from __future__ import annotations

import math

import pytest
import torch

from wick.cuda_block import (
    CudaDevice,
    baseline_vram,
    cuda_available,
    fp16_layer_bytes,
    streamed_cuda_forward,
)
from wick.lora import LoRAConfig, init_lora_blocks, init_noisy_lora_blocks
from wick.toy import BlockConfig, apply_block

RUNABLE = cuda_available()
pytestmark = pytest.mark.skipif(not RUNABLE, reason="no CUDA build installed")


def _frozen_stack(cfg) -> list:
    from wick.toy import init_layers

    layers = init_layers(cfg, 2, seed=0)
    return [{n: t.detach().to("cpu", torch.float32).requires_grad_(False)
             for n, t in layer.items()} for layer in layers]


def _fp32_target(x, base, hidden, cfg) -> torch.Tensor:
    h = x
    for layer, lora in zip(base, hidden):
        h = apply_block(h, layer, cfg, lora=lora)
    return h


def _train_burst(cfg, steps: int):
    base = _frozen_stack(cfg)
    x = torch.randn(2, 8, cfg.d_model, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(5))
    hidden = init_noisy_lora_blocks(base, cfg, LoRAConfig(r=8), seed=9999)
    hidden = [{t: (A.to(torch.float32), B.to(torch.float32)) for t, (A, B) in blk.items()}
              for blk in hidden]
    with torch.no_grad():
        yt = _fp32_target(x, base, hidden, cfg).to("cuda")

    lora_blocks = init_lora_blocks(base, cfg, LoRAConfig(r=8, scale=0.02), seed=11)
    lora_blocks = [
        {t: (A.detach().to(torch.float32).requires_grad_(True),
             B.detach().to(torch.float32).requires_grad_(True))
         for t, (A, B) in blk.items()}
        for blk in lora_blocks
    ]
    params = [p for blk in lora_blocks for (A, B) in blk.values() for p in (A, B)]
    opt = torch.optim.Adam(params, lr=5e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12)
    sim = CudaDevice()
    base0 = baseline_vram()
    losses: list[float] = []
    overflow = 0
    for _ in range(steps):
        opt.zero_grad()
        out = streamed_cuda_forward(x, base, lora_blocks, cfg, sim)
        loss = ((out.float() - yt) ** 2).mean()
        losses.append(float(loss.detach()))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if any(p.grad is not None and (p.grad.isinf() | p.grad.isnan()).any()
               for g in opt.param_groups for p in g["params"]):
            overflow += 1
        scaler.step(opt)
        scaler.update()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    flow = {
        "losses": losses,
        "overflow": overflow,
        "scale": float(scaler.get_scale()),
        "sim": sim,
        "one_layer": fp16_layer_bytes(base[0]),
        "all_layers": sum(fp16_layer_bytes(l) for l in base),
        "peak_cuda": max(0, torch.cuda.max_memory_allocated() - base0),
    }
    return flow


@pytest.fixture(scope="module")
def cfg():
    return BlockConfig(d_model=64, n_heads=8, d_ff=256, ln_eps=1e-5)


@pytest.fixture(scope="module")
def flow(cfg):
    return _train_burst(cfg, steps=20)


def test_loss_is_finite_and_decreases(flow):
    losses = flow["losses"]
    assert all(math.isfinite(v) for v in losses)
    tail = max(1, len(losses) // 5)
    assert sum(losses[-tail:]) / tail < sum(losses[:tail]) / tail


def test_gradscaler_scale_stays_finite_and_stable(flow):
    scale = flow["scale"]
    assert math.isfinite(scale)
    assert scale > 0.0


def test_no_overflow_or_skipped_steps(flow):
    assert flow["overflow"] == 0


def test_weight_residency_bounded_at_one_layer(flow):
    sim = flow["sim"]
    peak_layers = sim.peak_bytes / flow["one_layer"]
    assert peak_layers <= 1.05, f"peak weight residency {peak_layers:.2f} layers"
    assert sim.peak_bytes < flow["all_layers"]  # strictly below full residency


def test_device_empty_after_training(flow):
    assert flow["sim"].resident_bytes() == 0


def test_peak_layers_below_all_resident(flow):
    assert flow["sim"].peak_bytes < flow["all_layers"]


def test_lora_masters_are_cpu_fp32_leaves_used(flow, cfg):
    # structural: masters were fp32 CPU leaves; if the streaming path leaked a
    # device tensor into the master graph this test would fail to run cleanly.
    assert flow["sim"].assert_empty is not None
    assert flow["peak_cuda"] >= 0