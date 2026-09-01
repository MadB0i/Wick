"""Sub-gate 4A: the Phase 2/3 streaming mechanism on REAL CUDA, in fp16.

Phase 2/3 ran on CPU with `SimDevice` bookkeeping. 4A replaces the *simulation*
with reality, one mechanism at a time:

* "load"   = `master.to('cuda')`            (a real PCIe host->device transfer)
* "evict"  = `del device_copy` + empty_cache (a real VRAM release)
* "device" = CUDA fp16 tensor                (GTX 1650: no bf16, no tensor cores)
* "master" = fp32 tensor on CPU, always.     (AGENTS.md: fp32 master weights for
                                             streamed layers live on CPU always)

fp16 + GradScaler is the only mixed-precision path on this die. The reason to
keep the master fp32 on CPU is exactly what GradScaler exists to protect: when a
layer is evicted mid-backward and reloaded, every intermediate is fp16, so the
backward pass must run on GradScaler's scaled loss or the fp16 gradients
under/overflow. This module's key deliverable is *proving the scale stays stable
across hundreds of streamed load/evict cycles* -- not just once.

Design notes:

* The autograd.Function's **differentiable inputs are the fp32 CPU masters**. The
  forward saves only those; the backward reloads fp16 device copies, recomputes
  under grad, and hands fp32 gradients back. autograd accumulates onto the fp32
  CPU masters, so `GradScaler.unscale_/step` sees an ordinary parameter graph.
* Base encoder weights stream and are frozen (`requires_grad=False`). LoRA
  adapters stream too (device fp16 copies), but their fp32 CPU masters are the
  trainable leaves.
* VRAM is measured with `torch.cuda.max_memory_allocated()`, the incremental peak
  above the CUDA-context baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora import LORA_TARGETS, lora_unflatten
from .toy import PARAM_NAMES, BlockConfig, apply_block


def cuda_available() -> bool:
    return torch.cuda.is_available()


class StreamedBackwardError(RuntimeError):
    """The backward pass broke in a way GradScaler cannot be expected to fix."""


class CudaDevice:
    """The real-device analog of `SimDevice`: genuine PCIe transfers.

    load  = fp32-CPU master -> fp16-CUDA copy (a real host->device transfer)
    evict = drop the reference and empty_cache (a real VRAM release)
    """

    def __init__(self) -> None:
        self._resident: dict[str, torch.Tensor] = {}
        self.n_loads = 0
        self.n_evicts = 0
        self.bytes_moved = 0
        self.peak_bytes = 0

    def load(self, name: str, host: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        dev = host.to("cuda", torch.float16)
        if requires_grad:
            dev = dev.requires_grad_(True)
        self._resident[name] = dev
        self.n_loads += 1
        self.bytes_moved += dev.numel() * 2
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes())
        return dev

    def get(self, name: str) -> torch.Tensor:
        return self._resident[name]

    def evict(self, name: str) -> None:
        self._resident.pop(name)
        self.n_evicts += 1

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._resident.values())

    def resident_names(self) -> list[str]:
        return sorted(self._resident)

    def assert_empty(self, where: str = "") -> None:
        if self._resident:
            raise AssertionError(
                f"CUDA device not empty{(' after ' + where) if where else ''}: "
                f"{sorted(self._resident)}"
            )


@dataclass
class CudaStreamSpec:
    """Non-tensor context threaded through the autograd Function."""

    cfg: BlockConfig
    sim: CudaDevice
    idx: int
    all_layers: list
    evict_lora: bool = True  # evict the (small) LoRA device copies after compute
    fault: str = "none"  # fault injection, for the expected-fail control

    @property
    def prefix(self) -> str:
        return f"L{self.idx}"


class CudaStreamedLoRABlock(torch.autograd.Function):
    """One transformer block: streamed fp32-CPU base weights + resident fp32-CPU
    LoRA masters, computed as fp16 on CUDA. Load/evict are real transfers."""

    @staticmethod
    def forward(ctx, spec, x: torch.Tensor, *host_params: torch.Tensor):
        n_base = len(PARAM_NAMES)
        host_base = host_params[:n_base]
        host_lora = host_params[n_base:]

        sim = spec.sim
        # ---- LOAD: fp32 CPU master -> fp16 CUDA device copy ----------------
        for name, host in zip(PARAM_NAMES, host_base):
            sim.load(f"{spec.prefix}.{name}", host)
        dev_base = {n: sim.get(f"{spec.prefix}.{n}") for n in PARAM_NAMES}
        dev_lora = {
            t: (A.to("cuda", torch.float16), B.to("cuda", torch.float16))
            for t, (A, B) in zip(LORA_TARGETS, zip(host_lora[0::2], host_lora[1::2]))
        }

        # ---- COMPUTE in fp16, no gradient graph kept ---------------------------
        with torch.no_grad():
            xf = x.detach().to("cuda", torch.float16)
            apply_lora = {t: (dev_lora[t][0], dev_lora[t][1]) for t in LORA_TARGETS}
            y = apply_block(xf, dev_base, spec.cfg, lora=apply_lora)

        ctx.spec = spec
        # Only host fp32 masters cross into backward. Nothing device-resident.
        ctx.save_for_backward(x.detach(), *host_base, *host_lora)

        # ---- EVICT: real releases --------------------------------------------
        for name in PARAM_NAMES:
            sim.evict(f"{spec.prefix}.{name}")
        if spec.evict_lora:
            del dev_lora, dev_base, xf
            torch.cuda.empty_cache()

        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        spec: CudaStreamSpec = ctx.spec
        sim = spec.sim
        saved = ctx.saved_tensors
        n_base = len(PARAM_NAMES)
        x = saved[0]
        host_base = saved[1 : 1 + n_base]
        host_lora = saved[1 + n_base :]

        want_x = ctx.needs_input_grad[1]
        want_lora = tuple(ctx.needs_input_grad[2 + n_base :])

        # ---- RELOAD fp16 device copies ----------------------------------------
        dev_base = {
            n: sim.load(f"{spec.prefix}.{n}", spec.all_layers[spec.idx][n], requires_grad=False)
            for n in PARAM_NAMES
        }
        dev_lora = {
            t: (A.to("cuda", torch.float16).requires_grad_(True),
                B.to("cuda", torch.float16).requires_grad_(True))
            for t, (A, B) in zip(LORA_TARGETS, zip(host_lora[0::2], host_lora[1::2]))
        }

        # ---- RECOMPUTE with grad enabled --------------------------------------
        with torch.enable_grad():
            xd = x.detach().to("cuda", torch.float16)
            if want_x:
                xd.requires_grad_(True)
            apply_lora = {t: (dev_lora[t][0], dev_lora[t][1]) for t in LORA_TARGETS}
            y = apply_block(xd, dev_base, spec.cfg, lora=apply_lora)

        targets: list[torch.Tensor] = []
        if want_x:
            targets.append(xd)
        for t in LORA_TARGETS:
            targets.append(dev_lora[t][0])
            targets.append(dev_lora[t][1])

        grads = torch.autograd.grad(y, targets, grad_y, allow_unused=True)
        it = iter(grads)
        grad_x = (next(it).float().to("cuda" if x.is_cuda else "cpu")) if want_x else None
        grad_lora: list[torch.Tensor | None] = []
        for wA, wB in zip(want_lora[0::2], want_lora[1::2]):
            ga = next(it).float().to("cpu") if wA else None
            gb = next(it).float().to("cpu") if wB else None
            grad_lora.extend([ga, gb])

        # ---- EVICT ------------------------------------------------------------
        for name in PARAM_NAMES:
            sim.evict(f"{spec.prefix}.{name}")
        del dev_lora, dev_base, xd
        torch.cuda.empty_cache()

        grad_base = [None for _ in PARAM_NAMES]  # frozen in 4A
        return (None, grad_x, *grad_base, *grad_lora)


def streamed_cuda_forward(
    x, layers, lora_blocks, cfg, sim, evict_lora: bool = True, fault: str = "none"
) -> torch.Tensor:
    """Run the stack on real CUDA, each layer streamed in and out via PCIe."""
    h = x.to("cuda") if not x.is_cuda else x
    for idx in range(len(layers)):
        spec = CudaStreamSpec(
            cfg=cfg, sim=sim, idx=idx, all_layers=layers, evict_lora=evict_lora, fault=fault
        )
        base_flat = tuple(layers[idx][n] for n in PARAM_NAMES)
        lora_flat = tuple(v for t in LORA_TARGETS for v in lora_blocks[idx][t])
        h = CudaStreamedLoRABlock.apply(spec, h, *base_flat, *lora_flat)
    return h


# -- VRAM instrumentation --------------------------------------------------


def baseline_vram() -> int:
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated()


def peak_incremental_vram(baseline: int) -> int:
    return max(0, torch.cuda.max_memory_allocated() - baseline)


def fp16_layer_bytes(layer) -> int:
    return sum(t.numel() * 2 for t in layer.values())