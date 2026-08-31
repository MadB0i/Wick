"""Phase 2 gate, as assertions.

    .venv/Scripts/python.exe -m pytest tests/ -v

`scripts/run_gate.py` prints the same measurements as a human-readable report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wick.gate import (
    GateConfig,
    check_faults,
    check_hook_variant,
    compare,
    fresh_sim,
    grads_for,
    make_problem,
    measure_residency,
    resident_callable,
    run_gate,
    run_gradcheck,
    streamed_callable,
)
from wick.simdevice import ResidencyBudgetExceeded, SimDevice
from wick.streaming import FAULTS, streamed_stack_forward
from wick.toy import (
    PARAM_NAMES,
    BlockConfig,
    init_layers,
    make_input,
    resident_stack_forward,
)

GC = GateConfig()


@pytest.fixture(scope="module")
def problem():
    return make_problem(GC)


@pytest.fixture(scope="module")
def faults():
    """Runs the whole fault sweep once -- each sweep is 5 gradchecks."""
    return check_faults(GC)


@pytest.fixture(scope="module")
def hooks():
    return check_hook_variant(GC)


@pytest.fixture(scope="module")
def residency(problem):
    return measure_residency(problem, GC, evict=True)


# -- the ladder ------------------------------------------------------------


def test_dtype_is_float64(problem):
    """gradcheck's finite differences are meaningless below fp64."""
    assert problem.x.dtype is torch.float64
    assert all(t.dtype is torch.float64 for t in problem.flat)


def test_baseline_gradcheck_passes(problem):
    """Rung 1: if this fails the toy block is not smooth enough to gradcheck."""
    res = run_gradcheck(resident_callable(problem.cfg, problem.n_layers), problem, GC)
    assert res.passed, f"harness bug, not a streaming bug: {res.error}"


def test_recompute_without_eviction_gradcheck_passes(problem):
    """Rung 2: isolates the activation-checkpointing recompute."""
    fn = streamed_callable(problem.cfg, problem.n_layers, fresh_sim(), evict=False)
    res = run_gradcheck(fn, problem, GC)
    assert res.passed, f"recompute math is wrong: {res.error}"


def test_streamed_gradcheck_passes(problem):
    """Rung 3: the real mechanism, under a ~1-layer device budget."""
    budget = int(problem.one_layer_bytes() * 1.25)
    fn = streamed_callable(
        problem.cfg, problem.n_layers, fresh_sim(budget_bytes=budget), evict=True
    )
    res = run_gradcheck(fn, problem, GC)
    assert res.passed, f"eviction breaks gradients: {res.error}"


def test_streamed_gradients_match_baseline(problem):
    """The Phase 2 acceptance criterion."""
    base = grads_for(resident_callable(problem.cfg, problem.n_layers), problem)
    fn = streamed_callable(problem.cfg, problem.n_layers, fresh_sim(), evict=True)
    cmp_ = compare(base, grads_for(fn, problem), problem.flat_labels())

    assert cmp_.n_nan == 0
    assert not math.isnan(cmp_.rel_diff_norm)
    assert cmp_.rel_diff_norm < GC.tol, (
        f"rel_diff_norm={cmp_.rel_diff_norm:.3e} exceeds {GC.tol:.0e}; "
        f"worst tensor {cmp_.worst_param} at {cmp_.worst_param_rel:.3e}"
    )
    assert cmp_.rel_norm_gap < GC.tol


def test_streamed_gradients_are_bitwise_identical(problem):
    """Stronger than the gate: deterministic recompute should reproduce exactly.

    Not the acceptance criterion -- kept separate so that a future change which
    trades exactness for speed (fused kernels, different reduction order) fails
    here and is noticed, without failing the gate itself.
    """
    base = grads_for(resident_callable(problem.cfg, problem.n_layers), problem)
    fn = streamed_callable(problem.cfg, problem.n_layers, fresh_sim(), evict=True)
    cmp_ = compare(base, grads_for(fn, problem), problem.flat_labels())
    assert cmp_.max_abs_diff == 0.0


# -- memory claims ---------------------------------------------------------


def test_peak_residency_is_one_layer(residency):
    r = residency
    assert r.peak_layers == pytest.approx(1.0, abs=0.01), (
        f"peak was {r.peak_layers:.2f} layers; streaming is not bounding memory"
    )
    assert r.peak_bytes < r.all_layers_bytes


def test_nothing_resident_at_end(residency):
    assert residency.resident_at_end == []


def test_graph_retains_no_device_allocation(residency):
    """The claim streaming lives or dies on, measured while the graph is alive."""
    assert residency.retained_by_graph == [], (
        f"autograd still points at evicted device memory: {residency.retained_by_graph}"
    )


def test_graph_walk_is_not_vacuous(residency):
    """Guard against the retention check passing because it found nothing."""
    assert residency.n_graph_saved > 0


def test_budget_actually_trips(problem):
    """The residency budget must be capable of failing."""
    budget = int(problem.one_layer_bytes() * 1.25)
    fn = streamed_callable(
        problem.cfg, problem.n_layers, fresh_sim(budget_bytes=budget), evict=False
    )
    with pytest.raises(ResidencyBudgetExceeded):
        grads_for(fn, problem)


def test_saved_tensors_are_host_only(problem):
    """Structural: the strict check inside backward is on and does something."""
    sim = fresh_sim()
    fn = streamed_callable(problem.cfg, problem.n_layers, sim, evict=True)
    grads_for(fn, problem)  # would raise AssertionError if a device copy leaked


# -- fault injection -------------------------------------------------------


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_injected_fault_is_caught(fault, faults):
    """A gate that cannot fail is not evidence."""
    res = faults[fault]
    assert res["caught"], (
        f"injected bug '{fault}' ({res['suspect']}) slipped through the gate: "
        f"{res['description']}"
    )


# -- hook-based control ----------------------------------------------------


def test_hook_streaming_retains_device_memory(hooks):
    """Documents why module hooks cannot carry the backward pass.

    Hook timing here is correct -- load before, evict after, in both directions.
    It still retains the weight matrices, because autograd saved views that
    alias the device storage, and no amount of hook reordering changes that.
    """
    nopoison = hooks["nopoison"]
    assert nopoison["ran"]
    # Gradients are correct...
    assert nopoison["n_nan_params"] == 0
    assert nopoison["rel_diff_norm"] < GC.tol
    # ...but the memory was never released, so streaming bought nothing.
    assert nopoison["retained_by_graph"], (
        "expected hook-based eviction to retain device allocations; if this "
        "now passes, re-examine whether the custom Function is still needed"
    )


def test_hook_streaming_corrupts_when_storage_is_reused(hooks):
    """If the allocator reuses the evicted storage, hook streaming goes silent-wrong.

    Poisoning stands in for a real allocator handing that block to someone else,
    which is exactly what happens on a 4 GB card under pressure.
    """
    poisoned = hooks["poison"]
    assert poisoned["ran"]
    assert poisoned["n_nan_params"] > 0
    assert math.isnan(poisoned["rel_diff_norm"])


# -- partial requires_grad (the LoRA case) ---------------------------------


def _mixed_problem(freeze_idx: int = 1, x_requires_grad: bool = False):
    cfg = BlockConfig()
    layers = init_layers(cfg, 3, seed=0)
    for t in layers[freeze_idx].values():
        t.requires_grad_(False)
    x = make_input(cfg, seed=7).requires_grad_(x_requires_grad)
    return cfg, layers, x


def _grad_snapshot(layers):
    return {
        f"L{i}.{n}": (None if t.grad is None else t.grad.clone())
        for i, layer in enumerate(layers)
        for n, t in layer.items()
    }


def _zero_grads(layers):
    for layer in layers:
        for t in layer.values():
            t.grad = None


@pytest.mark.parametrize("x_requires_grad", [False, True])
def test_frozen_layer_matches_baseline(x_requires_grad):
    """Phase 3 keeps the LLM frozen and trains adapters, so mixed
    requires_grad is the normal case, not an edge case."""
    cfg, layers, x = _mixed_problem(freeze_idx=1, x_requires_grad=x_requires_grad)

    (resident_stack_forward(x, layers, cfg) ** 2).sum().backward()
    base = _grad_snapshot(layers)
    _zero_grads(layers)

    sim = SimDevice()
    (streamed_stack_forward(x, layers, cfg, sim) ** 2).sum().backward()
    got = _grad_snapshot(layers)

    for name in base:
        if name.startswith("L1."):
            assert base[name] is None and got[name] is None, f"{name} should be frozen"
        else:
            assert got[name] is not None, f"{name} lost its gradient"
            assert torch.equal(base[name], got[name]), f"{name} differs from baseline"

    sim.assert_empty("frozen-layer backward")


# -- end to end ------------------------------------------------------------


def test_gate_passes():
    verdict = run_gate(GC)
    assert verdict.passed, f"gate failed at {verdict.failed_at}: {verdict.diagnosis}"
