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

## What is NOT proven yet
- No real GPU involved — everything is CPU simulation. Actual VRAM behavior unproven.
- No real VLM modules — toy blocks only, not MiniCPM-V's SigLIP/ViT encoder.
- Phase 3 done only in CPU sim: the real trainer (HuggingFace/custom) on real
  GPU, with fp16/amp, is not started. 1000-step loss-curve gate on real GPU pending.
- fp16/amp regime not tested — fp64 determinism is a lab condition.

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