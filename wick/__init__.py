"""Wick -- VLM fine-tuning via CPU<->GPU layer streaming through forward AND backward.

This package currently contains the Phase 2 gate harness (CPU-only, fp64 proof
that streaming a layer's parameters in and out of a device produces gradients
identical to a fully-resident backward pass) and the Phase 3 gate harness (LoRA
adapters trained on top of streamed, frozen base weights -- the MiniCPM-V
deployment the project targets).
"""

__all__ = ["simdevice", "toy", "streaming", "lora", "gate"]
