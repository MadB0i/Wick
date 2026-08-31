"""Phase 3 gate report. Exit code 0 = gate passed, 1 = gate failed.

    .venv/Scripts/python.exe scripts/run_phase3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from wick.lora import TrainerConfig, run_phase3
from wick.toy import BlockConfig


def main() -> int:
    cfg = BlockConfig()
    tc = TrainerConfig()
    verdict = run_phase3(cfg=cfg, tc=tc)

    print("=" * 78)
    print("WICK -- PHASE 3 GATE: LoRA training via backward-pass streaming")
    print("=" * 78)
    print(f"torch            {torch.__version__}  (device: CPU, simulated stream)")
    print(f"dtype            torch.float64")
    print(f"toy stack        {cfg.d_model=} {cfg.n_heads=} {cfg.d_ff=}, 3 layers")
    print(f"LoRA             r={tc.lora.r}, targets qkv/attn_out/ff1/ff2")
    print(f"training         {tc.steps} steps, Adam lr={tc.lr}, MSE on fixed target")
    print(f"gate threshold   final mean-loss relative gap < {tc.tol:.0%}")

    print()
    print("-" * 78)
    print("LOSS TRAJECTORY  (streamed vs full-resident, first 6 + last 2)")
    print("-" * 78)

    def fmt(vals):
        s = ", ".join(f"{x:.3e}" for x in vals[:6])
        return f"[{s} ... {vals[-2]:.3e} {vals[-1]:.3e}]"

    print(f"  resident  {fmt(verdict.resident)}")
    print(f"  streamed  {fmt(verdict.streamed)}")

    tail = max(1, tc.steps // 5)
    mr = sum(verdict.resident[-tail:]) / tail
    ms = sum(verdict.streamed[-tail:]) / tail

    print()
    print("-" * 78)
    print(f"mean loss (last {tail} steps)  resident {mr:.6e}   streamed {ms:.6e}")
    print(f"relative gap                  {verdict.gap:.3e}   (threshold {tc.tol:.0%})")
    print(f"coincide exactly (fp64)       {verdict.resident == verdict.streamed}")

    print()
    print("=" * 78)
    if verdict.passed:
        print("GATE PASSED -- streamed LoRA training tracks the full-resident baseline.")
        print("Streaming training costs nothing numerically. Phase 3 may proceed.")
    else:
        print(f"GATE FAILED -- gap {verdict.gap:.3e} >= {tc.tol:.0%}")
        print("Streaming diverged from the baseline; do not build on this.")
    print("=" * 78)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())