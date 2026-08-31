"""Streaming a layer's parameters through both the forward and backward pass.

The mechanism
-------------
A streamed layer is a `torch.autograd.Function` whose *host* master weights are
its differentiable inputs:

    forward:   load host->device, compute under no_grad, save only host tensors,
               evict device copies
    backward:  reload host->device, recompute the block with grad enabled,
               take grads w.r.t. the device copies, evict, and hand those grads
               back so autograd accumulates them into the host masters

Why not module hooks
--------------------
Forward-only streaming (AirLLM-style) can be done with `forward_pre_hook` /
`forward_hook` swapping `Parameter.data`. That does not extend to backward, and
the reason is not fixable by better hook timing: during forward, autograd saves
whatever tensors the ops needed for their own backward -- and those saves alias
the *device* copies. Evicting afterwards drops SimDevice's reference but not
autograd's, so the memory is never released, and if the storage is reused or
poisoned the gradients silently corrupt. `hook_stack_grads()` below builds that
version specifically so the gate can demonstrate the failure rather than assert
it.

The custom Function avoids the problem by construction: its forward runs under
`no_grad`, so no intermediate is saved at all, and the only tensors handed to
`save_for_backward` are host-resident. `_assert_saved_are_host` checks that
invariant on every backward instead of trusting it.

Fault injection
---------------
A gate that cannot fail proves nothing. `FAULTS` enumerates realistic bugs --
each one maps to one of the three suspects named in the Phase 2 plan
(activation checkpointing, hook timing, eviction order) -- and the harness
asserts that every one of them is *caught*.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

from .simdevice import SimDevice
from .toy import PARAM_NAMES, BlockConfig, apply_block

# -- fault injection ------------------------------------------------------

FAULT_NONE = "none"
FAULT_NO_RELOAD = "no_reload"
FAULT_WRONG_LAYER_RELOAD = "wrong_layer_reload"
FAULT_DROP_PARAM_GRADS = "drop_param_grads"
FAULT_NO_ENABLE_GRAD = "no_enable_grad"
FAULT_EVICT_BEFORE_COMPUTE = "evict_before_compute"

#: fault -> (which Phase 2 suspect it models, what the bug is in plain terms)
FAULTS: dict[str, tuple[str, str]] = {
    FAULT_NO_RELOAD: (
        "eviction order",
        "backward reuses the device reference captured in forward instead of "
        "reloading from the host master",
    ),
    FAULT_WRONG_LAYER_RELOAD: (
        "eviction order",
        "backward reloads the neighbouring layer's weights (off-by-one in the "
        "streaming bookkeeping)",
    ),
    FAULT_DROP_PARAM_GRADS: (
        "activation checkpointing",
        "backward computes grad w.r.t. the input but forgets to return "
        "parameter grads for the evicted layer",
    ),
    FAULT_NO_ENABLE_GRAD: (
        "activation checkpointing",
        "the recompute in backward runs without torch.enable_grad(), so no "
        "graph is built to differentiate",
    ),
    FAULT_EVICT_BEFORE_COMPUTE: (
        "hook timing",
        "forward evicts the weights before the block is computed",
    ),
}


@dataclass
class BlockSpec:
    """Non-tensor context threaded through the autograd Function."""

    cfg: BlockConfig
    sim: SimDevice
    idx: int
    all_layers: list[dict[str, torch.Tensor]] = field(repr=False)
    evict: bool = True
    fault: str = FAULT_NONE
    strict_saved_check: bool = True

    @property
    def prefix(self) -> str:
        return f"L{self.idx}"

    def host_layer(self) -> dict[str, torch.Tensor]:
        return self.all_layers[self.idx]


def _assert_saved_are_host(
    saved_params: tuple[torch.Tensor, ...], host_layer: dict[str, torch.Tensor]
) -> None:
    """Fail if anything other than the host master got saved for backward.

    Storage identity, not object identity: `save_for_backward` may hand back a
    different Python wrapper, but a host master and a device clone never share a
    `data_ptr`.
    """
    for name, saved in zip(PARAM_NAMES, saved_params):
        host = host_layer[name]
        if saved.data_ptr() != host.data_ptr():
            raise AssertionError(
                f"{name}: save_for_backward captured a tensor that is not the host "
                f"master (saved {saved.data_ptr():#x} vs host {host.data_ptr():#x}). "
                "A device-resident copy leaked into the autograd graph, so eviction "
                "cannot free it."
            )


class StreamedBlock(torch.autograd.Function):
    """One transformer block whose weights are streamed in both directions."""

    @staticmethod
    def forward(ctx, spec: BlockSpec, x: torch.Tensor, *host_params: torch.Tensor):
        sim = spec.sim
        sim.phase = "forward"

        for name, host in zip(PARAM_NAMES, host_params):
            sim.load(f"{spec.prefix}.{name}", host)

        if spec.fault == FAULT_EVICT_BEFORE_COMPUTE:
            for name in PARAM_NAMES:
                sim.evict(f"{spec.prefix}.{name}")

        # Read through sim.get() so that a residency violation raises rather
        # than silently reading a stale local reference.
        dev_p = {n: sim.get(f"{spec.prefix}.{n}") for n in PARAM_NAMES}
        with torch.no_grad():
            y = apply_block(x, dev_p, spec.cfg)

        ctx.spec = spec
        # Only host tensors cross into backward. Nothing device-resident.
        ctx.save_for_backward(x, *host_params)

        if spec.fault == FAULT_NO_RELOAD:
            ctx.leaked = dev_p  # deliberately smuggle device refs across
        del dev_p

        if spec.evict:
            for name in PARAM_NAMES:
                sim.evict(f"{spec.prefix}.{name}")

        sim.phase = "-"
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        spec: BlockSpec = ctx.spec
        sim = spec.sim
        saved = ctx.saved_tensors
        x, host_params = saved[0], saved[1:]

        if spec.strict_saved_check:
            _assert_saved_are_host(host_params, spec.host_layer())

        sim.phase = "backward"
        want_x = ctx.needs_input_grad[1]
        want_p = tuple(ctx.needs_input_grad[2:])

        # -- RELOAD ------------------------------------------------------
        source = spec.host_layer()
        if spec.fault == FAULT_WRONG_LAYER_RELOAD:
            source = spec.all_layers[(spec.idx + 1) % len(spec.all_layers)]

        if spec.fault == FAULT_NO_RELOAD:
            dev_p = {n: t.requires_grad_(True) for n, t in ctx.leaked.items()}
        else:
            dev_p = {
                n: sim.load(f"{spec.prefix}.{n}", source[n], requires_grad=w)
                for n, w in zip(PARAM_NAMES, want_p)
            }

        # -- RECOMPUTE ---------------------------------------------------
        # Grad mode is off inside backward unless create_graph was set, so
        # enable_grad here is mandatory, not defensive.
        grad_ctx = (
            contextlib.nullcontext()
            if spec.fault == FAULT_NO_ENABLE_GRAD
            else torch.enable_grad()
        )
        with grad_ctx:
            xd = x.detach()
            if want_x:
                xd.requires_grad_(True)
            y = apply_block(xd, dev_p, spec.cfg)

            targets: list[torch.Tensor] = ([xd] if want_x else []) + [
                dev_p[n] for n, w in zip(PARAM_NAMES, want_p) if w
            ]
            raw = (
                torch.autograd.grad(y, targets, grad_y, allow_unused=True)
                if targets
                else ()
            )

        # -- EVICT -------------------------------------------------------
        if spec.evict and spec.fault != FAULT_NO_RELOAD:
            for name in PARAM_NAMES:
                sim.evict(f"{spec.prefix}.{name}")

        # -- unpack back into the input signature ------------------------
        it = iter(raw)
        grad_x = next(it) if want_x else None
        grad_p: list[torch.Tensor | None] = []
        for name, w in zip(PARAM_NAMES, want_p):
            if not w:
                grad_p.append(None)
                continue
            g = next(it)
            if g is None:  # unused param: report an explicit zero
                g = torch.zeros_like(spec.host_layer()[name])
            grad_p.append(g)

        if spec.fault == FAULT_DROP_PARAM_GRADS:
            grad_p = [
                None if g is None else torch.zeros_like(g) for g in grad_p
            ]

        sim.phase = "-"
        return (None, grad_x, *grad_p)


def streamed_stack_forward(
    x: torch.Tensor,
    layers: list[dict[str, torch.Tensor]],
    cfg: BlockConfig,
    sim: SimDevice,
    evict: bool = True,
    fault: str = FAULT_NONE,
    strict_saved_check: bool = True,
) -> torch.Tensor:
    """Run the stack with every layer streamed in and out."""
    for idx in range(len(layers)):
        spec = BlockSpec(
            cfg=cfg,
            sim=sim,
            idx=idx,
            all_layers=layers,
            evict=evict,
            fault=fault,
            strict_saved_check=strict_saved_check,
        )
        x = StreamedBlock.apply(spec, x, *[layers[idx][n] for n in PARAM_NAMES])
    return x


# -- hook-based variant: an expected-fail control --------------------------


class HookedBlock(torch.nn.Module):
    """Forward-only streaming, done the way AirLLM does it, extended to backward.

    Included so the gate can *demonstrate* why this approach cannot carry the
    backward pass. The hook timing is correct -- load before, evict after, in
    both directions -- and it still breaks, because autograd's saved tensors
    alias the device copies and nothing about hook ordering changes that.
    """

    def __init__(
        self, cfg: BlockConfig, host_layer: dict[str, torch.Tensor], sim: SimDevice, idx: int
    ) -> None:
        super().__init__()
        self.cfg, self.sim, self.prefix = cfg, sim, f"L{idx}"
        for name in PARAM_NAMES:
            self.register_parameter(
                name, torch.nn.Parameter(host_layer[name].detach().clone())
            )
        self._host_data: dict[str, torch.Tensor] = {}
        self.register_forward_pre_hook(type(self)._pre_forward)
        self.register_forward_hook(type(self)._post_forward)
        self.register_full_backward_pre_hook(type(self)._pre_backward)
        self.register_full_backward_hook(type(self)._post_backward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_block(x, {n: getattr(self, n) for n in PARAM_NAMES}, self.cfg)

    def _load(self) -> None:
        for name in PARAM_NAMES:
            p = getattr(self, name)
            self._host_data[name] = p.data
            p.data = self.sim.load(f"{self.prefix}.{name}", p.data)

    def _evict(self) -> None:
        for name in PARAM_NAMES:
            self.sim.evict(f"{self.prefix}.{name}")
            getattr(self, name).data = self._host_data.pop(name)

    @staticmethod
    def _pre_forward(mod: "HookedBlock", args):
        mod.sim.phase = "forward"
        mod._load()

    @staticmethod
    def _post_forward(mod: "HookedBlock", args, output):
        mod._evict()
        mod.sim.phase = "-"
        return output

    @staticmethod
    def _pre_backward(mod: "HookedBlock", grad_output):
        mod.sim.phase = "backward"
        mod._load()

    @staticmethod
    def _post_backward(mod: "HookedBlock", grad_input, grad_output):
        mod._evict()
        mod.sim.phase = "-"


def hook_stack_grads(
    x: torch.Tensor,
    layers: list[dict[str, torch.Tensor]],
    cfg: BlockConfig,
    sim: SimDevice,
    after_forward=None,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Forward+backward the hook-streamed stack; return (loss, per-layer grads).

    `after_forward(sim, loss)` is called with the graph still alive, so a caller
    can ask whether the evicted device copies were actually released.
    """
    blocks = [HookedBlock(cfg, layer, sim, idx) for idx, layer in enumerate(layers)]
    h = x
    for blk in blocks:
        h = blk(h)
    loss = (h**2).sum()
    if after_forward is not None:
        after_forward(sim, loss)
    loss.backward()
    grads = [
        {n: getattr(blk, n).grad for n in PARAM_NAMES} for blk in blocks
    ]
    return loss.detach(), grads
