"""Phase 3 gate, as assertions: LoRA training via backward-pass streaming.

Structured like the Phase 2 suite. The load-bearing claim is `run_phase3`: after
training LoRA adapters on the same frozen base, same data, same optimizer, and
same step count, the *streamed* loss trajectory must track the *full-resident*
trajectory within the gate tolerance. Because the base recompute is bit-exact and
the LoRA math is shared line-for-line, we additionally assert the trajectories
coincide exactly at fp64.

    .venv/Scripts/python.exe -m pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from wick.lora import (
    LORA_TARGETS,
    LoRAConfig,
    LoRAStreamedBlock,
    TrainerConfig,
    _generic_train,
    frozen_base,
    init_lora_block,
    init_lora_blocks,
    init_noisy_lora_blocks,
    lora_flat,
    lora_unflatten,
    make_task,
    resident_lora_forward,
    run_phase3,
    streamed_lora_forward,
    train_resident,
    train_streamed,
)
from wick.simdevice import SimDevice
from wick.toy import PARAM_NAMES, BlockConfig, apply_block, init_layers
from wick.streaming import BlockSpec, streamed_stack_forward

CFG = BlockConfig()
TC = TrainerConfig(steps=120, seed=0)


@pytest.fixture(scope="module")
def task():
    return make_task(CFG, TC)


# -- structural sanity ------------------------------------------------------


def test_lora_shapes():
    blk = init_lora_block(CFG, LoRAConfig(r=3), seed=0)
    d, f = CFG.d_model, CFG.d_ff
    assert blk["qkv"][0].shape == (d, 3) and blk["qkv"][1].shape == (3, 3 * d)
    assert blk["attn_out"][0].shape == (d, 3) and blk["attn_out"][1].shape == (3, d)
    assert blk["ff1"][0].shape == (d, 3) and blk["ff1"][1].shape == (3, f)
    assert blk["ff2"][0].shape == (f, 3) and blk["ff2"][1].shape == (3, d)


def test_zero_b_init_is_a_noop(task):
    frozen, *_ = task
    layers = init_layers(CFG, 3, 0)
    frozen_single = frozen_base(layers)
    x = task[2]
    fresh = init_lora_block(CFG, TC.lora, seed=999)
    ro = resident_lora_forward(x, [frozen_single[0]], [fresh], CFG)
    base = apply_block(x, frozen_single[0], CFG)
    assert torch.equal(base, ro)


def test_lora_flat_roundtrips(task):
    _, lora_blocks, _, _ = task
    for blk in lora_blocks:
        assert lora_unflatten(lora_flat(blk)) == blk


def test_frozen_base_has_no_grads(task):
    frozen, _, _, _ = task
    assert all(not t.requires_grad for layer in frozen for t in layer.values())


def test_targets_are_learnable():
    noisy = init_noisy_lora_blocks(init_layers(CFG, 3, 0), CFG, LoRAConfig(r=CFG.d_model))
    for blk in noisy:
        for (_A, B) in blk.values():
            assert B.abs().sum() > 0


# -- forward equivalence ------------------------------------------------------


def test_streamed_forward_matches_resident(task):
    frozen, lora_blocks, x, _ = task
    sr = streamed_lora_forward(x, frozen, lora_blocks, CFG, SimDevice(), evict=True)
    rr = resident_lora_forward(x, frozen, lora_blocks, CFG)
    assert torch.equal(sr, rr)


def test_streamed_matches_phase2_base(task):
    """With B=0 init the adapter residual is a no-op, so the streamed LoRA stack must
    equal the Phase 2 streamed (no-LoRA) base stack."""
    frozen, lora_blocks, x, _ = task
    src = streamed_lora_forward(x, frozen, lora_blocks, CFG, SimDevice(), evict=True)
    base = streamed_stack_forward(x, frozen, CFG, SimDevice(), evict=True)
    assert torch.equal(src, base)


# -- gradient equivalence ------------------------------------------------------


def test_streamed_grads_match_resident_on_step_0(task):
    frozen, lora_blocks, x, yt = task
    rv = resident_lora_forward(x, frozen, lora_blocks, CFG)
    rp = [p for b in lora_blocks for (A, B) in b.values() for p in (A, B)]
    rg = torch.autograd.grad(F.mse_loss(rv, yt), rp)

    sv = streamed_lora_forward(x, frozen, lora_blocks, CFG, SimDevice(), evict=True)
    sg = torch.autograd.grad(F.mse_loss(sv, yt), rp)

    assert len(rg) == len(sg) == 24
    assert all(torch.equal(a, b) for a, b in zip(rg, sg))


def test_streamed_gradcheck_passes(task):
    frozen, lora_blocks, x, _ = task
    flat = lora_flat(lora_blocks[0])

    def fn(x, *f):
        # one layer, so layers and lora_blocks have matching length
        blk = lora_unflatten(f)
        return streamed_lora_forward(x, frozen[:1], [blk], CFG, SimDevice(), evict=True)

    x0 = x.detach().requires_grad_(True)
    f0 = [t.detach().requires_grad_(True) for t in flat]
    torch.autograd.gradcheck(fn, (x0, *f0), eps=1e-6, atol=1e-7, rtol=1e-5)


# -- training trajectory ------------------------------------------------------


def test_run_phase3_passes_small():
    v = run_phase3(tc=TrainerConfig(steps=120, seed=0))
    assert v.passed
    assert v.gap < TC.tol
    assert len(v.resident) == len(v.streamed) == 120


def test_run_phase3_trajectories_coincide_exactly():
    v = run_phase3(tc=TrainerConfig(steps=120, seed=0))
    assert v.resident == v.streamed


def test_run_phase3_trains_the_signal_down():
    v = run_phase3(tc=TrainerConfig(steps=120, seed=0))
    assert v.resident[0] > v.resident[-1]
    assert v.streamed[0] > v.streamed[-1]


def test_run_phase3_deterministic():
    a = run_phase3(tc=TrainerConfig(steps=60, seed=7))
    b = run_phase3(tc=TrainerConfig(steps=60, seed=7))
    assert a.resident == b.resident
    assert a.streamed == b.streamed


# -- memory / lifecycle ------------------------------------------------------


def test_streamed_training_leaves_device_empty():
    frozen, lora_blocks, x, yt = make_task(CFG, TC)
    sim = SimDevice()
    losses = train_streamed(frozen, lora_blocks, x.detach(), yt, CFG, TC)
    assert len(losses) == TC.steps
    sim.assert_empty("streamed training")


def test_modes_do_not_share_params(task):
    frozen, lora_blocks, x, yt = task
    r_clone = copy.deepcopy(lora_blocks)
    s_clone = copy.deepcopy(lora_blocks)
    train_resident(frozen, r_clone, x.detach(), yt, CFG, TC)
    train_streamed(frozen, s_clone, x.detach(), yt, CFG, TC)
    assert all(p.grad is None for b in lora_blocks for (A, B) in b.values() for p in (A, B))


@pytest.mark.parametrize("freeze_x", [False, True])
def test_resident_vs_streamed_with_x_requires_grad(freeze_x):
    frozen, lora_blocks, x, yt = make_task(CFG, TC)
    x_in = x.detach().requires_grad_(freeze_x)
    r = _generic_train(
        resident_lora_forward, frozen, copy.deepcopy(lora_blocks), x_in, yt, CFG, TC
    )
    sim = SimDevice()

    def fwd(x, layers, loras, cfg):
        h = x
        for idx in range(len(layers)):
            spec = BlockSpec(cfg=cfg, sim=sim, idx=idx, all_layers=layers, evict=True)
            h = LoRAStreamedBlock.apply(
                spec, h, *[layers[idx][n] for n in PARAM_NAMES], *lora_flat(loras[idx])
            )
        return h

    s = _generic_train(fwd, frozen, copy.deepcopy(lora_blocks), x_in, yt, CFG, TC)
    assert s == r


def test_lora_targets_are_expected_set():
    assert set(LORA_TARGETS) == {"qkv", "attn_out", "ff1", "ff2"}