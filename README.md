# 🔥 Wick

**VLM fine-tuning on low-VRAM consumer GPUs via CPU↔GPU layer streaming — through the *backward* pass.**

Wick lets you fine-tune Vision-Language Models on hardware that was never meant
for it (think 4 GB GTX 1650s) by streaming the vision encoder's layers in and
out of VRAM, one layer at a time — **not just for inference, but for training itself.**

> Inference-side streaming was already solved (AirLLM, oLLM). Wick solves the harder
> half nobody's cracked yet: **computing gradients through streamed layers.**

**55 tests passing | GTX 1650 4GB | fp16 stable**

---

## 🚀 Quick Demo

```text
========================================================================
WICK DEVICE PROFILER
========================================================================
Your GPU: NVIDIA GeForce GTX 1650
  VRAM      4.00 GB (4.00 GB)
  compute   7.5 (no bf16, fp16)
  system RAM 15 GB (host masters / optimizer offload)

  Peak-VRAM model (Wick streaming): 1 vision layer fp16 + resident LLM
  + LoRA fp32 + LoRA optimizer + ~1-layer activation + 10% headroom

  ✅ MiniCPM-V 1.3B     best(fp16)   2.60 GB of 4.00 GB VRAM
          fp16 needs   2.60 GB, int4 needs   0.82 GB
  ✅ MiniCPM-V 2.6B     best(int4)   1.69 GB of 4.00 GB VRAM
          fp16 needs   5.56 GB, int4 needs   1.69 GB
  ✅ Qwen2-VL 2B        best(int4)   1.35 GB of 4.00 GB VRAM
          fp16 needs   4.32 GB, int4 needs   1.35 GB
  ❌ LLaVA-1.5 7B       best(int4)   4.49 GB of 4.00 GB VRAM
          fp16 needs  14.89 GB, int4 needs   4.49 GB
  ✅ Phi-3.5 Vision     best(int4)   2.69 GB of 4.00 GB VRAM
          fp16 needs   8.93 GB, int4 needs   2.69 GB

  ✅ Can fine-tune:
     - MiniCPM-V 1.3B
     - MiniCPM-V 2.6B
     - Qwen2-VL 2B
     - Phi-3.5 Vision
  ❌ Cannot fit:
     - LLaVA-1.5 7B
========================================================================
```

Run it yourself with `python -m wick.profiler`.

---

## ✨ Why Wick?

| | AirLLM / oLLM (inference-only) | **Wick (training)** |
|---|---|
| Forward pass streaming | ✅ one layer at a time | ✅ one layer at a time |
| **Backward pass streaming** | ❌ not supported | ✅ **bit-exact gradients |
| Peak VRAM residency | ~1 layer | ~1 layer |
| Fine-tuning | ❌ | ✅ LoRA / QLoRA |

---

## 🧪 What's Proven

- ✅ **Phase 2 — backward-pass streaming is gradient-exact.** `rel_diff_norm = 0.0`
  vs a fully-resident baseline (bitwise), fp64, CPU sim.
- ✅ **Peak residency = 1 layer.** No device storage leaks into the autograd graph
  (40 saved tensors walked via `data_ptr`, non-vacuous check).
- ✅ **AirLLM-style hooks rejected with evidence.** Hooks retain device allocations
  and produce **NaN gradients** under storage pressure. A custom
  `torch.autograd.Function` is required — and proven.
- ✅ **Phase 3 — streamed LoRA training** (frozen base + resident adapters) yields
  **bit-identical loss trajectories** vs full-resident (fp64, CPU).
- ✅ **Sub-gate 4A — real GTX 1650, fp16 + GradScaler.** Stable scale (final 4e3,
  init 2^12), zero overflow across 100 streamed steps / 4800 real PCIe
  load-evict cycles; peak weight residency 1.00 fp16 layer; loss 7.3e-4 → 2.8e-6.
- ✅ **The gate can fail** — injected fault types are all caught.

## 🚧 What's Not Proven Yet

- ⏳ No real VLM modules — toy blocks (real-GPU fp16) only, not MiniCPM-V's
  SigLIP/ViT encoder.
- ⏳ Sub-gate 4B (MiniCPM-V vision-encoder streaming) not started.
- ⏳ The real trainer (streamed encoder → resident LLM, GradScaler end-to-end,
  1000-step loss-curve gate vs full-VRAM baseline) not built.
- ⏳ Realistic-scale VRAM: at toy size, fixed CUDA/autograd/optimizer overhead
  dominates the 97.6 KiB layer, so only the weight-residency bound is honest
  until layers reach MB scale.

---

## 🎯 Target

**MiniCPM-V 1.3B** — stream only the vision encoder (~400 M params), keep the LLM
+ LoRA adapters resident in VRAM.

**GPU floor:** GTX 1650 · 4 GB VRAM · TU117 die — no bf16, no tensor cores.
Mixed precision is **fp16 + GradScaler only**; fp32 master weights for
streamed layers live on CPU always.

---

## 🛠️ Stack

Python + PyTorch (`torch.autograd.Function`) · pure local, no cloud calls.
Real-GPU runs need a CUDA build — for the GTX 1650 (sm_75) that is
`pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124`.

---

## 📅 Roadmap

- **Phase 2 ✅** — backward-pass streaming proof (gate passes)
- **Phase 3 ✅** — LoRA training via streamed backward (CPU sim, exact)
- **Phase 4A ✅** — real GTX 1650: fp16 + GradScaler stability, 1-layer residency
- **Phase 4B ⏳** — real MiniCPM-V vision-encoder streaming
- **Phase 5 ⏳** — trainer wiring (1000-step gate on GPU) + benchmarks + demo

---

## 📊 Results — Sub-gate 4A (real GTX 1650)

Measured on a real NVIDIA GeForce GTX 1650 (4 GB, sm_75, fp16 only), 100 streamed
training steps with Adam + GradScaler.

| Metric | Value |
|---|---|
| GradScaler final scale | `4e3` (init `2^12` = 4096) |
| Overflow / skipped steps | `0` |
| Peak weight residency | `1.00` fp16 layer (97.6 KiB; all-resident would need 195.2 KiB) |
| Loss trajectory | `7.35e-4` → `2.77e-6` over 100 steps |
| Real PCIe load / evict cycles | `4800` / `4800` |
| Device bytes resident after run | `0.00` MiB |

```text
==============================================================================
SUB-GATE 4A PASSED --
  final GradScaler scale 4e+03 (finite, stable, 0 overflow steps)
  peak weight residency 97.6 KiB = 1.00 fp16 layers (bounded at ~1; all-resident would need 195.2 KiB)
  loss 7.349e-04 -> 2.775e-06 over 100 steps
  4800 real PCIe loads, 4800 evicts, device empty at end
==============================================================================
```

Run it yourself with `python scripts/run_phase4a.py`.

---

## 🚀 Get Started

```bash
pip install -e .
# Phase 2 gate report (exit 0 = pass)
python scripts/run_gate.py
# Phase 3 gate report (LoRA via streamed backward, CPU sim)
python scripts/run_phase3.py
# Sub-gate 4A report (real GPU: fp16 + GradScaler stability) -- needs CUDA build
python scripts/run_phase4a.py
# Device profiler: which VLMs fit your hardware (no model download)
python -m wick.profiler
# Full assertion suite
python -m pytest tests/ -v
```

---

## 📄 License

Apache-2.0 · built for the low-VRAM community.