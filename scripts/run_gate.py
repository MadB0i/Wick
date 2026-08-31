"""Phase 2 gate report. Exit code 0 = gate passed, 1 = gate failed.

    .venv/Scripts/python.exe scripts/run_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from wick.gate import GateConfig, run_gate


def kb(n: int) -> str:
    return f"{n / 1024:.1f} KiB"


def main() -> int:
    cfg = GateConfig()
    verdict = run_gate(cfg)
    p = verdict.problem

    n_params = sum(t.numel() for t in p.flat)
    print("=" * 78)
    print("WICK -- PHASE 2 GATE: backward-pass streaming gradient equivalence")
    print("=" * 78)
    print(f"torch            {torch.__version__}  (device: CPU, simulated stream)")
    print(f"dtype            {p.x.dtype}   <- required for gradcheck finite differences")
    print(
        f"toy stack        {p.n_layers} pre-norm attn+MLP blocks, "
        f"d_model={p.cfg.d_model}, heads={p.cfg.n_heads}, d_ff={p.cfg.d_ff}"
    )
    print(f"input            {tuple(p.x.shape)}")
    print(f"parameters       {n_params} across {p.n_layers} layers ({kb(p.one_layer_bytes())}/layer)")
    print(
        f"gradcheck        eps={cfg.gradcheck_eps:.0e} atol={cfg.gradcheck_atol:.0e} "
        f"rtol={cfg.gradcheck_rtol:.0e}"
    )
    print(f"gate threshold   rel_diff_norm < {cfg.tol:.0e} vs full-resident backward")

    print()
    print("-" * 78)
    print("LADDER  (each rung isolates one mechanism)")
    print("-" * 78)
    for i, rung in enumerate(verdict.rungs, 1):
        mark = "PASS" if rung.ok else "FAIL"
        print(f"\n{i}. {rung.name}   [{mark}]")
        print(f"   isolates    {rung.suspect}")
        print(f"   setup       {rung.note}")
        gcr = rung.gradcheck
        print(f"   gradcheck   {'passed' if gcr.passed else f'FAILED ({gcr.exc_type})'}")
        if gcr.error:
            print(f"               {gcr.error[:150]}")
        c = rung.comparison
        if c is not None:
            print(f"   vs baseline rel_diff_norm = {c.rel_diff_norm:.3e}   (threshold {cfg.tol:.0e})")
            print(f"               rel_norm_gap  = {c.rel_norm_gap:.3e}")
            print(f"               max |diff|    = {c.max_abs_diff:.3e}")
            print(f"               worst tensor  = {c.worst_param} @ {c.worst_param_rel:.3e}")
            print(f"               NaN grads     = {c.n_nan}")
            print(f"               ||g_base||    = {c.base_norm:.6f}")
            print(f"               ||g_stream||  = {c.test_norm:.6f}")

    r = verdict.residency
    print()
    print("-" * 78)
    print("RESIDENCY  (did streaming actually bound device memory?)")
    print("-" * 78)
    print(f"peak resident         {kb(r.peak_bytes)}  = {r.peak_layers:.2f} layers")
    print(f"all layers resident   {kb(r.all_layers_bytes)}  (what the baseline needs)")
    print(f"saving                {r.saving * 100:.1f}%")
    print(f"loads / evicts        {r.n_loads} / {r.n_evicts}")
    print(f"bytes over the bus    {kb(r.bytes_moved)}")
    print(f"resident at end       {r.resident_at_end or 'none'}")
    print()
    print("  liveness -- was the evicted device memory actually released?")
    print(
        f"    retained by autograd graph (by data_ptr, graph alive)   "
        f"{r.retained_by_graph or 'none'}"
        + ("   <- eviction freed nothing!" if r.retained_by_graph else "")
    )
    print(f"    tensors found in graph walk                            {r.n_graph_saved}"
          + ("  <- walk found nothing; check above is vacuous" if not r.n_graph_saved else ""))
    print(
        f"    device tensor objects still alive after forward         "
        f"{r.leaked_after_forward or 'none'}"
    )
    print(
        f"    same, after backward (graph freed -- cannot fail)       "
        f"{r.leaked_after_backward or 'none'}"
    )

    print()
    print("-" * 78)
    print("FAULT INJECTION  (a gate that cannot fail is not evidence)")
    print("-" * 78)
    for fault, res in verdict.faults.items():
        mark = "caught" if res["caught"] else "MISSED"
        print(f"\n  {fault:22s} [{mark}]  suspect: {res['suspect']}")
        print(f"    bug       {res['description']}")
        print(f"    detected  {res['how']}")
        if res["detail"]:
            print(f"              {res['detail'][:130]}")

    print()
    print("-" * 78)
    print("HOOK-BASED STREAMING  (expected-fail control -- why not just use hooks)")
    print("-" * 78)
    for key, label in (("poison", "eviction poisons freed storage"), ("nopoison", "eviction is a no-op on storage")):
        h = verdict.hooks.get(key, {})
        print(f"\n  {label}:")
        if not h.get("ran"):
            print(f"    crashed: {h.get('error')}")
            continue
        print(f"    params with NaN grad  {h['n_nan_params']} / {h['n_params_total']}  {h['nan_params']}")
        print(f"    rel_diff_norm         {h['rel_diff_norm']:.3e}")
        print(
            f"    retained by graph     {len(h['retained_by_graph'])} allocations"
            f"  (of {h['n_graph_saved']} tensors in graph)"
        )
        if h["retained_by_graph"]:
            print(f"                          first: {h['retained_by_graph'][:3]}")

    print()
    print("=" * 78)
    if verdict.passed:
        final = verdict.rungs[-1].comparison
        print(f"GATE PASSED -- rel_diff_norm {final.rel_diff_norm:.3e} < {cfg.tol:.0e}")
        print("Backward-pass streaming is gradient-exact. Phase 2 may proceed.")
    else:
        print(f"GATE FAILED at: {verdict.failed_at}")
        print(f"Diagnosis: {verdict.diagnosis}")
    print("=" * 78)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
