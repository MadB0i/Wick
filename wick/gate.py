"""The Phase 2 gate: does streaming produce the same gradients as full residency?

Everything here runs on CPU in float64. fp64 is not a stylistic choice --
`gradcheck` compares an analytical Jacobian against central finite differences,
and in fp32 the finite-difference error alone (roundoff ~1e-8/eps) sits at or
above the 1e-5 threshold the gate is trying to measure. The gate would be
reading its own numerical noise. The fp16/GradScaler design is a Phase 3
concern and deliberately absent.

The gate is a ladder, run in order, so a failure localises itself:

    1. resident baseline   ordinary autograd. Fails => the toy block is not
                           smooth enough to gradcheck; harness bug, not Wick.
    2. recompute only      custom Function, no eviction. Fails => the
                           activation-checkpointing recompute is wrong.
    3. recompute + evict   the real mechanism. Fails => eviction order /
                           residency.
    4. liveness            were the evicted copies actually freed? Fails =>
                           a device tensor leaked into the autograd graph.
    5. hook-based          expected to fail. Isolates hook timing, and shows
                           why the custom Function is necessary.

Then `check_faults()` injects known bugs and asserts each one is caught, because
a gate that cannot fail is not evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .simdevice import SimDevice
from .streaming import (
    FAULT_NONE,
    FAULTS,
    hook_stack_grads,
    streamed_stack_forward,
)
from .toy import (
    PARAM_NAMES,
    BlockConfig,
    init_layers,
    layer_bytes,
    make_input,
    resident_stack_forward,
)


@dataclass
class GateConfig:
    n_layers: int = 3
    batch: int = 2
    seq: int = 3
    block: BlockConfig = field(default_factory=BlockConfig)
    seed: int = 0
    #: the Phase 2 acceptance threshold
    tol: float = 1e-5
    gradcheck_eps: float = 1e-6
    gradcheck_atol: float = 1e-7
    gradcheck_rtol: float = 1e-5


@dataclass
class Problem:
    cfg: BlockConfig
    layers: list[dict[str, torch.Tensor]]
    x: torch.Tensor

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def flat(self) -> list[torch.Tensor]:
        return [self.layers[i][n] for i in range(self.n_layers) for n in PARAM_NAMES]

    def flat_labels(self) -> list[str]:
        return [f"L{i}.{n}" for i in range(self.n_layers) for n in PARAM_NAMES]

    def one_layer_bytes(self) -> int:
        return layer_bytes(self.layers[0])


def make_problem(gc_: GateConfig) -> Problem:
    layers = init_layers(gc_.block, gc_.n_layers, seed=gc_.seed)
    x = make_input(gc_.block, batch=gc_.batch, seq=gc_.seq, seed=gc_.seed + 7)
    return Problem(cfg=gc_.block, layers=layers, x=x)


def unflatten(flat: tuple[torch.Tensor, ...] | list[torch.Tensor], n_layers: int):
    k = len(PARAM_NAMES)
    return [dict(zip(PARAM_NAMES, flat[i * k : (i + 1) * k])) for i in range(n_layers)]


# -- callables under test --------------------------------------------------


def resident_callable(cfg: BlockConfig, n_layers: int):
    def fn(x, *flat):
        return resident_stack_forward(x, unflatten(flat, n_layers), cfg)

    return fn


def streamed_callable(
    cfg: BlockConfig,
    n_layers: int,
    sim: SimDevice,
    evict: bool = True,
    fault: str = FAULT_NONE,
):
    def fn(x, *flat):
        # The layers must be rebuilt from the *arguments*, so that gradcheck's
        # perturbed tensors are the ones that get streamed.
        return streamed_stack_forward(
            x, unflatten(flat, n_layers), cfg, sim, evict=evict, fault=fault
        )

    return fn


def fresh_sim(
    budget_bytes: int | None = None,
    poison: bool = True,
    liveness: bool = False,
    log: bool = False,
) -> SimDevice:
    return SimDevice(
        budget_bytes=budget_bytes,
        poison_on_evict=poison,
        track_liveness=liveness,
        log_events=log,
    )


# -- measurements ----------------------------------------------------------


@dataclass
class GradcheckResult:
    passed: bool
    error: str | None = None
    exc_type: str | None = None


def run_gradcheck(fn, problem: Problem, gc_: GateConfig) -> GradcheckResult:
    x = problem.x.detach().requires_grad_(True)
    flat = [p.detach().requires_grad_(True) for p in problem.flat]
    try:
        torch.autograd.gradcheck(
            fn,
            (x, *flat),
            eps=gc_.gradcheck_eps,
            atol=gc_.gradcheck_atol,
            rtol=gc_.gradcheck_rtol,
            check_undefined_grad=True,
            check_batched_grad=False,
        )
    except Exception as exc:  # GradcheckError, RuntimeError, AssertionError, ...
        first = str(exc).strip().splitlines()
        return GradcheckResult(
            False, error=first[0] if first else repr(exc), exc_type=type(exc).__name__
        )
    return GradcheckResult(True)


def grads_for(fn, problem: Problem) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Gradients of a scalar loss, via autograd.grad so no `.grad` is mutated."""
    x = problem.x.detach().requires_grad_(True)
    flat = [p.detach().requires_grad_(True) for p in problem.flat]
    y = fn(x, *flat)
    loss = (y**2).sum()
    grads = torch.autograd.grad(loss, [x, *flat])
    return grads[0], list(grads[1:])


@dataclass
class Comparison:
    """How far a streamed gradient set is from the fully-resident one."""

    #: ||g_test - g_base|| / ||g_base||, over all params concatenated. The gate
    #: criterion -- strictly harder than comparing norms, which can cancel.
    rel_diff_norm: float
    #: | ||g_test|| - ||g_base|| | / ||g_base||
    rel_norm_gap: float
    max_abs_diff: float
    worst_param: str
    worst_param_rel: float
    n_nan: int
    base_norm: float
    test_norm: float


def compare(
    base: tuple[torch.Tensor, list[torch.Tensor]],
    test: tuple[torch.Tensor, list[torch.Tensor]],
    labels: list[str],
) -> Comparison:
    bx, bp = base
    tx, tp = test
    all_labels = ["x", *labels]
    b_all = [bx, *bp]
    t_all = [tx, *tp]

    sq_diff = 0.0
    sq_base = 0.0
    sq_test = 0.0
    max_abs = 0.0
    worst, worst_rel = "-", 0.0
    n_nan = 0

    for label, b, t in zip(all_labels, b_all, t_all):
        if t is None:
            t = torch.zeros_like(b)
        n_nan += int(torch.isnan(t).sum().item())
        d = (t - b).abs()
        sq_diff += float((d**2).sum())
        sq_base += float((b**2).sum())
        sq_test += float(torch.nan_to_num(t, nan=0.0).pow(2).sum())
        m = float(d.max()) if d.numel() else 0.0
        if m > max_abs or math.isnan(m):
            max_abs = m
        bn = float(b.norm())
        rel = float((t - b).norm()) / bn if bn > 0 else float((t - b).norm())
        if rel > worst_rel or math.isnan(rel):
            worst, worst_rel = label, rel

    base_norm = math.sqrt(sq_base)
    test_norm = math.sqrt(sq_test)
    return Comparison(
        rel_diff_norm=math.sqrt(sq_diff) / base_norm if base_norm else math.sqrt(sq_diff),
        rel_norm_gap=abs(test_norm - base_norm) / base_norm if base_norm else 0.0,
        max_abs_diff=max_abs,
        worst_param=worst,
        worst_param_rel=worst_rel,
        n_nan=n_nan,
        base_norm=base_norm,
        test_norm=test_norm,
    )


@dataclass
class Residency:
    peak_bytes: int
    peak_after_forward: int
    all_layers_bytes: int
    one_layer_bytes: int
    n_loads: int
    n_evicts: int
    bytes_moved: int
    #: measured while the autograd graph is still alive -- the only moment a
    #: leaked saved tensor is detectable
    leaked_after_forward: list[str]
    leaked_after_backward: list[str]
    #: evicted device *allocations* the graph still points at, by data_ptr
    retained_by_graph: list[str]
    #: total tensors found in the graph walk; 0 would make the above vacuous
    n_graph_saved: int
    resident_at_end: list[str]

    @property
    def peak_layers(self) -> float:
        return self.peak_bytes / self.one_layer_bytes if self.one_layer_bytes else 0.0

    @property
    def saving(self) -> float:
        return 1.0 - self.peak_bytes / self.all_layers_bytes if self.all_layers_bytes else 0.0


def measure_residency(problem: Problem, gc_: GateConfig, evict: bool = True) -> Residency:
    """One forward+backward, with a liveness snapshot taken between them.

    The snapshot timing is the whole point. Checking liveness *after* backward
    proves nothing: by then the graph has been freed, so every device tensor is
    dead whether or not it was ever leaked. Peak device memory is decided while
    the graph is alive, so that is when the question has to be asked.
    """
    sim = fresh_sim(liveness=True, log=True)
    fn = streamed_callable(problem.cfg, problem.n_layers, sim, evict=evict)

    x = problem.x.detach().requires_grad_(True)
    flat = [p.detach().requires_grad_(True) for p in problem.flat]
    y = fn(x, *flat)
    loss = (y**2).sum()

    leaked_fwd = sim.leaked_names()  # graph alive
    retained, n_graph_saved = sim.retained_by_graph(loss)
    peak_fwd = sim.peak_bytes

    torch.autograd.grad(loss, [x, *flat])

    return Residency(
        peak_bytes=sim.peak_bytes,
        peak_after_forward=peak_fwd,
        all_layers_bytes=problem.one_layer_bytes() * problem.n_layers,
        one_layer_bytes=problem.one_layer_bytes(),
        n_loads=sim.n_loads,
        n_evicts=sim.n_evicts,
        bytes_moved=sim.bytes_moved,
        leaked_after_forward=leaked_fwd,
        leaked_after_backward=sim.leaked_names(),
        retained_by_graph=retained,
        n_graph_saved=n_graph_saved,
        resident_at_end=sim.resident_names(),
    )


# -- the ladder ------------------------------------------------------------


@dataclass
class Rung:
    name: str
    suspect: str
    gradcheck: GradcheckResult
    comparison: Comparison | None
    note: str = ""

    @property
    def ok(self) -> bool:
        if not self.gradcheck.passed:
            return False
        return self.comparison is None or (
            self.comparison.n_nan == 0
            and not math.isnan(self.comparison.rel_diff_norm)
        )


def run_ladder(gc_: GateConfig | None = None) -> tuple[list[Rung], Residency, Problem, GateConfig]:
    gc_ = gc_ or GateConfig()
    problem = make_problem(gc_)
    labels = problem.flat_labels()

    base_fn = resident_callable(problem.cfg, problem.n_layers)
    base_grads = grads_for(base_fn, problem)

    rungs: list[Rung] = []

    # 1. baseline -- gradcheck the toy block itself.
    rungs.append(
        Rung(
            "resident baseline",
            "harness / block smoothness",
            run_gradcheck(base_fn, problem, gc_),
            None,
            "ordinary autograd, every layer resident",
        )
    )

    # 2. recompute only, no eviction -> isolates the checkpointing math.
    sim = fresh_sim()
    fn = streamed_callable(problem.cfg, problem.n_layers, sim, evict=False)
    rungs.append(
        Rung(
            "recompute, no eviction",
            "activation checkpointing",
            run_gradcheck(fn, problem, gc_),
            compare(base_grads, grads_for(fn, problem), labels),
            "custom Function recomputes in backward; weights never evicted",
        )
    )

    # 3. the real mechanism, with a residency budget of ~1 layer so that any
    #    accidental retention trips ResidencyBudgetExceeded rather than passing
    #    quietly.
    budget = int(problem.one_layer_bytes() * 1.25)
    sim = fresh_sim(budget_bytes=budget)
    fn = streamed_callable(problem.cfg, problem.n_layers, sim, evict=True)
    rungs.append(
        Rung(
            "recompute + eviction",
            "eviction order / residency",
            run_gradcheck(fn, problem, gc_),
            compare(base_grads, grads_for(fn, problem), labels),
            f"device budget capped at {budget}B (~1.25 layers)",
        )
    )

    residency = measure_residency(problem, gc_, evict=True)
    return rungs, residency, problem, gc_


def check_faults(gc_: GateConfig | None = None) -> dict[str, dict]:
    """Inject each known bug; every one must be caught by the gate."""
    gc_ = gc_ or GateConfig()
    problem = make_problem(gc_)
    labels = problem.flat_labels()
    base_grads = grads_for(resident_callable(problem.cfg, problem.n_layers), problem)

    out: dict[str, dict] = {}
    for fault, (suspect, description) in FAULTS.items():
        sim = fresh_sim()
        fn = streamed_callable(problem.cfg, problem.n_layers, sim, evict=True, fault=fault)
        res = run_gradcheck(fn, problem, gc_)

        detail = res.error
        how = f"gradcheck raised {res.exc_type}" if not res.passed else None
        cmp_ = None
        if res.passed:
            # gradcheck let it through -- does the baseline comparison catch it?
            try:
                sim.reset()
                cmp_ = compare(base_grads, grads_for(fn, problem), labels)
                if cmp_.n_nan or cmp_.rel_diff_norm > gc_.tol or math.isnan(cmp_.rel_diff_norm):
                    how = f"baseline comparison: rel_diff_norm={cmp_.rel_diff_norm:.3e}"
                    detail = f"{cmp_.n_nan} NaN grads, worst={cmp_.worst_param}"
            except Exception as exc:
                how = f"baseline comparison raised {type(exc).__name__}"
                detail = str(exc).strip().splitlines()[0]

        out[fault] = {
            "suspect": suspect,
            "description": description,
            "caught": how is not None,
            "how": how or "NOT CAUGHT",
            "detail": detail,
        }
    return out


def check_hook_variant(gc_: GateConfig | None = None) -> dict:
    """The expected-fail control: hooks with correct timing, and why it breaks."""
    gc_ = gc_ or GateConfig()
    problem = make_problem(gc_)
    base_x, base_p = grads_for(
        resident_callable(problem.cfg, problem.n_layers), problem
    )
    labels = problem.flat_labels()

    report: dict = {}
    for poison in (True, False):
        sim = fresh_sim(poison=poison, liveness=True)
        snapshot: dict = {}

        def snap(s: SimDevice, root: torch.Tensor, out: dict = snapshot) -> None:
            retained, n_saved = s.retained_by_graph(root)
            out.update(
                leaked=s.leaked_names(),
                peak=s.peak_bytes,
                retained=retained,
                n_graph_saved=n_saved,
            )

        try:
            _, grads = hook_stack_grads(
                problem.x.detach().requires_grad_(True),
                problem.layers,
                problem.cfg,
                sim,
                after_forward=snap,
            )
            flat = [grads[i][n] for i in range(problem.n_layers) for n in PARAM_NAMES]
            nan_params = [
                lbl
                for lbl, g in zip(labels, flat)
                if g is not None and bool(torch.isnan(g).any())
            ]
            cmp_ = compare((base_x, base_p), (base_x, flat), labels)
            report["poison" if poison else "nopoison"] = {
                "ran": True,
                "n_nan_params": len(nan_params),
                "nan_params": nan_params[:6],
                "rel_diff_norm": cmp_.rel_diff_norm,
                "leaked_after_forward": snapshot.get("leaked", []),
                "retained_by_graph": snapshot.get("retained", []),
                "n_graph_saved": snapshot.get("n_graph_saved", 0),
                "n_params_total": len(flat),
                "peak_bytes": sim.peak_bytes,
                "peak_after_forward": snapshot.get("peak", 0),
            }
        except Exception as exc:
            report["poison" if poison else "nopoison"] = {
                "ran": False,
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            }
    return report


@dataclass
class GateVerdict:
    passed: bool
    rungs: list[Rung]
    residency: Residency
    faults: dict[str, dict]
    hooks: dict
    problem: Problem
    cfg: GateConfig
    failed_at: str | None = None
    diagnosis: str | None = None


def run_gate(gc_: GateConfig | None = None) -> GateVerdict:
    gc_ = gc_ or GateConfig()
    rungs, residency, problem, gc_ = run_ladder(gc_)

    failed_at = None
    diagnosis = None
    for rung in rungs:
        if not rung.ok:
            failed_at = rung.name
            diagnosis = (
                f"discrepancy introduced at '{rung.name}' -- suspect: {rung.suspect}. "
                f"{rung.gradcheck.error or ''}".strip()
            )
            break

    final = rungs[-1]
    tol_ok = (
        final.comparison is not None
        and final.comparison.rel_diff_norm < gc_.tol
        and final.comparison.n_nan == 0
    )
    if failed_at is None and not tol_ok:
        failed_at = "tolerance"
        c = final.comparison
        diagnosis = (
            f"gradients differ by rel_diff_norm={c.rel_diff_norm:.3e} "
            f"(threshold {gc_.tol:.0e}); worst tensor {c.worst_param} "
            f"at {c.worst_param_rel:.3e}"
        )

    if failed_at is None and (residency.leaked_after_forward or residency.retained_by_graph):
        failed_at = "liveness"
        leaked = residency.leaked_after_forward or residency.retained_by_graph
        how = (
            "a live tensor object"
            if residency.leaked_after_forward
            else "the autograd graph (storage retained through a view)"
        )
        diagnosis = (
            f"eviction did not free {len(leaked)} device tensor(s) "
            f"({', '.join(leaked[:4])}...): held by {how}, so streaming saves "
            "no memory"
        )
    if failed_at is None and residency.n_graph_saved == 0:
        failed_at = "liveness (vacuous)"
        diagnosis = (
            "the graph walk found 0 saved tensors, so the retention check could "
            "not have failed; it proves nothing as written"
        )

    faults = check_faults(gc_)
    hooks = check_hook_variant(gc_)
    uncaught = [f for f, r in faults.items() if not r["caught"]]
    if failed_at is None and uncaught:
        failed_at = "fault detection"
        diagnosis = f"the gate failed to catch injected bug(s): {', '.join(uncaught)}"

    return GateVerdict(
        passed=failed_at is None,
        rungs=rungs,
        residency=residency,
        faults=faults,
        hooks=hooks,
        problem=problem,
        cfg=gc_,
        failed_at=failed_at,
        diagnosis=diagnosis,
    )
