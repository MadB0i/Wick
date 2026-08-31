"""Wick -- VLM fine-tuning via CPU<->GPU layer streaming through forward AND backward.

This package currently contains only the Phase 2 gate harness: a CPU-only, fp64
proof that streaming a layer's parameters in and out of a device produces
gradients identical to a fully-resident backward pass.
"""

__all__ = ["simdevice", "toy", "streaming", "gate"]
