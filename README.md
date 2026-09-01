# 🔥 Wick

**VLM fine-tuning on low-VRAM consumer GPUs via CPU↔GPU layer streaming — through the *backward* pass.**

Wick lets you fine-tune Vision-Language Models on hardware that was never meant
for it (think 4 GB GTX 1650s) by streaming the vision encoder's layers in and
out of VRAM, one layer at a time — **not just for inference, but for training itself.**

> Inference-side streaming was already solved (AirLLM, oLLM). Wick solves the harder
> half nobody's cracked yet: **computing gradients through streamed layers.**

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

## 🚀 Get Started

```bash
pip install -e .
# Phase 2 gate report (exit 0 = pass)
python scripts/run_gate.py
# Phase 3 gate report (LoRA via streamed backward, CPU sim)
python scripts/run_phase3.py
# Sub-gate 4A report (real GPU: fp16 + GradScaler stability) -- needs CUDA build
python scripts/run_phase4a.py
# Full assertion suite
python -m pytest tests/ -v
```

---

## 📄 License

Apache-2.0 · built for the low-VRAM community.