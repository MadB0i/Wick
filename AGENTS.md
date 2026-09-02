# Wick
VLM fine-tuning on low-VRAM consumer GPUs via CPU<->GPU layer streaming
through BOTH forward and backward pass.

## What's proven
- Phase 2 gate passes: backward-pass layer streaming yields bit-exact gradients
  (rel_diff_norm = 0.0) vs full-resident baseline, at fp64, CPU simulation.
- Peak residency = 1 layer. No device storage leaks into autograd graph (40
  saved tensors walked via data_ptr, non-vacuous check).
- Hook-based approach (AirLLM-style) is rejected — retains 12 device allocations
  in graph, produces NaN gradients. Custom autograd.Function is required.
- 5 injected fault types all caught by the gate (gate can fail = gate is real).
- Phase 3 gate passes: streamed LoRA training (frozen base + resident adapters)
  yields bit-identical loss trajectories vs full-resident baseline (fp64, CPU).
- Sub-gate 4A passes on the real GTX 1650: fp16 + GradScaler held a stable scale
  (final 4e3, init 2^12) across 100 streamed training steps and 4800 real PCIe
  load/evict cycles with zero overflow/skips. Peak weight residency bounded at
  1.00 fp16 layers; device empty after; loss 7.3e-4 -> 2.8e-6.
- Sub-gate 4B (100-step smoke gate) passes on the real GTX 1650: REAL SigLIP
  SO-400M encoder (27 layers, 15,239,504 params each, 29.07 MB fp16) streamed
  layer-by-layer through forward AND backward. Peak VRAM 84.8 MiB (well under
  the 2339 MiB ceiling); peak residency 1.00 layer; GradScaler scale finite and
  stable (65536, 0 overflow/skipped); loss 1.38e0 -> 9.68e-6 over 100 steps;
  43200 real PCIe load/evict cycles; device empty after.

## What is NOT proven yet
- Sub-gate 4B 1000-step full gate: the 100-step smoke gate passes; the 1000-step
  loss-curve-within-5%-of-full-VRAM-baseline run on the real GPU is still pending.
- The LLM backbone is not yet resident/quantized (int4 via bitsandbytes) or wired
  to the streamed vision encoder. Only the frozen SigLIP tower is streamed; the
  trainable head used in 4B is a stand-in, not the real LLM + LoRA.
- Peak-VRAM-at-realistic-scale: the 84.8 MiB peak reflects frozen-SigLIP + small
  head only. With int4 LLM resident (~1 GB) + LoRA, the honest ceiling is the
  ~2339 MiB budget from the fit check — not yet measured on the real GPU.
- Real dataset / task loss not yet tested (synthetic random target used).

## Target model
MiniCPM-V 1.3B. Stream only the vision encoder (~400M params), keep LLM +
LoRA adapters resident in VRAM.

### Sub-gate 4B — real architecture recon (from HF configs, no weights)
MiniCPM-V 1.3B (`openbmb/MiniCPM-V`), model_type=minicpmv:
- Vision tower: `vit_so400m_patch14_siglip_384.webli` = SigLIP SO-400M
  (`google/siglip-so400m-patch14-384`). hidden=1152, intermediate=4304,
  layers=27, heads=16, patch=14, image=384.   Per-layer params = 15,239,504
  (~29.07 MB fp16 / 58.1 MB fp32). Measured full tower = 428,225,600 params
  (411M across the 27 encoder layers + embeddings + post_layernorm);
  816.8 MB fp16 / 1633.5 MB fp32 on CPU masters.
- LLM backbone: hidden=2304, intermediate=5760, layers=40, heads=36, kv=36.
  Per-layer = 47,780,352 params. Full LLM ~2.19B params
  (4185 MB fp16 / 1046 MB int4).
- LoRA on LLM ONLY (vision frozen), r=8 over q/k/v/proj/fc1/fc2 (6 proj):
  ~8.85M params = 33.8 MB fp32 + 67.5 MB Adam m+v.

### 4B fit check (GTX 1650 = 4096 MiB = 4.0 GiB)
Wick streams 1 SigLIP layer at a time, so resident = 1 layer only:
- 1 SigLIP layer fp16: 29.0 MB
- LLM int4: 1046 MB
- LoRA fp32 + Adam m+v: 33.8 + 67.5 MB
- activations (small batch): ~350 MB (est.)
- CUDA/autograd/allocator overhead: ~600 MB (est.)
- subtotal 2126 MB, +10% headroom = **~2339 MiB (2.28 GiB) peak**,
  margin ~1757 MiB free (43%). **FITS.**
- LLM in fp16 instead of int4 would be ~5792 MiB -> **does NOT fit**.
  int4 LLM + fp16 streamed SigLIP + frozen vision is REQUIRED.

## GPU constraints
GTX 1650, 4GB VRAM, TU117 die — NO bf16, NO tensor cores.
Mixed precision = fp16 + GradScaler only. fp32 master weights for streamed
layers live on CPU always.

## Phase gate rule
Any new phase must have a validation gate BEFORE full implementation.
Phase 2 gate = gradcheck on 2-3 toy layers. Phase 3 gate = loss curve within
5% of full-VRAM baseline after 1000 steps on real GPU.

### Sub-gate 4B gate criteria (real SigLIP layers, GTX 1650)
Based on real layer sizes from recon (1 SigLIP layer = 29.04 MB fp16):
1. Peak weight residency = 1.00 SigLIP layer(s) fp16 (~29 MB), bounded across
   ALL steps, device empty after run (allocator trace == 0 resident weight bytes).
2.   Peak VRAM must stay under **~2339 MiB** (2.28 GiB) — the real Wick-strategy
   budget for resident LLM int4 + LoRA + 1 streamed layer. See fit check above.
   1 SigLIP layer = 29.07 MB fp16, total encoder = 816.8 MB fp16.
3. Loss curve within 5% of the full-VRAM baseline after **1000** steps on the
   real GPU (gap < 5% relative).
4. GradScaler scale stays finite and stable, **0 overflow / skipped steps**,
   across all 1000 streamed steps (fp16 + GradScaler only; no bf16 on TU117).

### Stack
Python + PyTorch only. No cloud API calls. Everything local.