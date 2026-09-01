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

## What is NOT proven yet
- No real VLM modules — toy blocks (now real-GPU fp16) only, not MiniCPM-V's
  SigLIP/ViT encoder.
- Sub-gate 4B (MiniCPM-V vision-encoder streaming) not started.
- The real trainer (HuggingFace/custom) wiring the streamed encoder to a resident
  LLM, with GradScaler end-to-end, is not built. The 1000-step loss-curve gate on
  real GPU vs full-VRAM baseline is pending.
- Peak-VRAM-at-realistic-scale not shown at toy size: fixed CUDA/autograd/optimizer
  overhead dominates the 97.6 KiB toy layer, so only the weight-residency bound is
  an honest measurement until layers reach realistic MB sizes.

## Target model
MiniCPM-V 1.3B. Stream only the vision encoder (~400M params), keep LLM +
LoRA adapters resident in VRAM.

## GPU constraints
GTX 1650, 4GB VRAM, TU117 die — NO bf16, NO tensor cores.
Mixed precision = fp16 + GradScaler only. fp32 master weights for streamed
layers live on CPU always.

## Phase gate rule
Any new phase must have a validation gate BEFORE full implementation.
Phase 2 gate = gradcheck on 2-3 toy layers. Phase 3 gate = loss curve within
5% of full-VRAM baseline after 1000 steps on real GPU.

## Stack
Python + PyTorch only. No cloud API calls. Everything local.