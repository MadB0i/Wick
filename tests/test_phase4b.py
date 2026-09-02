"""Sub-gate 4B tests: real SigLIP streaming on GTX 1650.

Skipped if no CUDA or if the local SigLIP weights are missing.
"""

from __future__ import annotations

import pytest

import torch

CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not CUDA, reason="CUDA required for sub-gate 4B")


def _local_siglip_exists() -> bool:
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "models" / "siglip-so400m-patch14-384" / "model.safetensors"
    return p.exists() and p.stat().st_size == 3_511_950_624


WEIGHTS = pytest.mark.skipif(not _local_siglip_exists(), reason="SigLIP weights not downloaded")


# ── Model loading ──────────────────────────────────────────────────────────


@WEIGHTS
def test_siglip_layer_count():
    from wick.model_loader import load_siglip_vision
    m = load_siglip_vision()
    assert len(m.encoder.layers) == 27


@WEIGHTS
def test_siglip_first_layer_params():
    from wick.model_loader import load_siglip_vision
    m = load_siglip_vision()
    n = sum(p.numel() for p in m.encoder.layers[0].parameters())
    assert n == 15_239_504


@WEIGHTS
def test_siglip_total_params_in_range():
    from wick.model_loader import load_siglip_vision
    m = load_siglip_vision()
    total = sum(p.numel() for p in m.parameters())
    assert 400_000_000 < total < 500_000_000


# ── StreamedModule ──────────────────────────────────────────────────────────


@WEIGHTS
def test_streamed_module_forward():
    from wick.model_loader import load_siglip_vision
    from wick.phase4b import StreamedModule
    m = load_siglip_vision()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    layer = m.encoder.layers[0]
    x = torch.randn(1, 729, 1152, dtype=torch.float16, device="cuda")
    y = StreamedModule.apply(x, layer)
    assert y.shape == (1, 729, 1152)
    assert y.device.type == "cuda"
    assert y.dtype == torch.float16


@WEIGHTS
def test_streamed_module_evicts():
    """After StreamedModule forward, no extra CUDA weight memory is held."""
    import gc
    from wick.model_loader import load_siglip_vision
    from wick.phase4b import StreamedModule
    m = load_siglip_vision()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    layer = m.encoder.layers[0]
    x = torch.randn(1, 729, 1152, dtype=torch.float16, device="cuda")
    before = torch.cuda.memory_allocated()
    y = StreamedModule.apply(x, layer)
    del y
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    # After eviction, allocated should be back to baseline (CUDA context only)
    assert after - before < 50 * 1024**2, f"residual {after - before} bytes > 50 MiB"


# ── Training loop ───────────────────────────────────────────────────────────


@WEIGHTS
def test_training_loss_decreases():
    from wick.phase4b import run_phase4b_gate
    stats = run_phase4b_gate(steps=3, batch_size=1, grad_accum=2, lr=1e-3, verbose=False)
    assert stats.step_loss[-1] < stats.step_loss[0]


@WEIGHTS
def test_peak_vram_under_ceiling():
    from wick.phase4b import run_phase4b_gate
    stats = run_phase4b_gate(steps=2, batch_size=1, grad_accum=2, lr=1e-3, verbose=False)
    assert stats.peak_vram_bytes < 2339 * 1024**2


@WEIGHTS
def test_grad_scaler_finite():
    from wick.phase4b import run_phase4b_gate
    stats = run_phase4b_gate(steps=2, batch_size=1, grad_accum=2, lr=1e-3, verbose=False)
    assert stats.grad_scaler_final_scale != float("inf")
    assert stats.grad_scaler_final_scale == stats.grad_scaler_final_scale  # not NaN


@WEIGHTS
def test_zero_overflows():
    from wick.phase4b import run_phase4b_gate
    stats = run_phase4b_gate(steps=2, batch_size=1, grad_accum=2, lr=1e-3, verbose=False)
    assert stats.overflow_count == 0
    assert stats.skipped_count == 0


@WEIGHTS
def test_peak_residency_one_layer():
    from wick.phase4b import run_phase4b_gate
    stats = run_phase4b_gate(steps=2, batch_size=1, grad_accum=2, lr=1e-3, verbose=False)
    assert stats.peak_layer_residency <= 1.01
