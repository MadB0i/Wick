"""Phase 3: LoRA fine-tuning with the backward-pass streaming mechanism.

The Phase 2 proof established that a *frozen* streamed layer reproduces the
full-resident backward pass bit-exactly (`test_frozen_layer_matches_baseline`).
Phase 3 is the step that makes that useful: real LoRA adapters, kept resident,
trained on top of streamed (frozen) base weights -- exactly the MiniCPM-V
deployment the project targets ("keep LLM + LoRA adapters resident in VRAM,
stream the vision encoder").

Two forwards must exist and must share every line of math except the weight
handling:

* `resident_lora_forward` -- base weights resident, LoRA resident, ordinary
  autograd. The baseline.
* `streamed_lora_forward` -- base weights streamed through `SimDevice`
  (loaded->compute->evicted around each custom Function), LoRA adapters kept
  resident. The mechanism under test.

Because the base recompute is bit-exact and the LoRA math is identical, the two
training trajectories must coincide (fp64, deterministic). The Phase 3 gate
asserts that formally: it runs both trainers for the same steps and requires the
final loss curves to agree within a tolerance, proving streaming training costs
nothing numerically.

LoRA is injected per projection (qkv / attn_out / ff1 / ff2) as
`(x @ A) @ B`, A: (in, r), B: (r, out). Adapters are initialised to zero B so the
pre-training outputs are exactly the base block's -- a clean starting point.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

from .simdevice import SimDevice
from .toy import PARAM_NAMES, BlockConfig, apply_block

#: the per-layer projections we attach a LoRA adapter to.
LORA_TARGETS: tuple[str, ...] = ("qkv", "attn_out", "ff1", "ff2")


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 4
    scale: float = 0.02  # std of the Gaussian used to init A

    #: how the raw (x@A)@B residual is scaled before adding. alpha/rLora is
    #: the conventional HF choice; with plain rLora<=r it is just 1.0.
    alpha: float | None = None

    @property
    def rank_scale(self) -> float:
        if self.alpha is None:
            return 1.0
        return self.alpha / self.r


def shape_for(target: str, cfg: BlockConfig) -> tuple[int, int]:
    """(in, out) of the linear a `target` projection multiplies."""
    d, f = cfg.d_model, cfg.d_ff
    return {
        "qkv": (d, 3 * d),
        "attn_out": (d, d),
        "ff1": (d, f),
        "ff2": (f, d),
    }[target]


def init_lora_block(
    cfg: BlockConfig, lc: LoRAConfig, seed: int, requires_grad: bool = True
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """One layer's adapters: `{target: (A, B)}`. B is zero => a no-op residual."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for target in LORA_TARGETS:
        d_in, d_out = shape_for(target, cfg)
        A = (torch.randn(d_in, lc.r, generator=gen, dtype=torch.float64) * lc.scale).requires_grad_(
            requires_grad
        )
        B = torch.zeros(lc.r, d_out, dtype=torch.float64).requires_grad_(requires_grad)
        out[target] = (A, B)
    return out


def init_lora_blocks(
    layers, cfg: BlockConfig, lc: LoRAConfig, seed: int = 1000, requires_grad: bool = True
) -> list[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    return [
        init_lora_block(cfg, lc, seed=seed + i, requires_grad=requires_grad)
        for i in range(len(layers))
    ]


def init_noisy_lora_block(
    cfg: BlockConfig, lc: LoRAConfig, seed: int, requires_grad: bool = True
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Like `init_lora_block` but with a *non-zero* B, so the residual actually
    moves the output. Used only to fabricate a private target the adapters learn
    to reproduce -- never as the training init (which must stay a no-op)."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for target in LORA_TARGETS:
        d_in, d_out = shape_for(target, cfg)
        A = (torch.randn(d_in, lc.r, generator=gen, dtype=torch.float64) * lc.scale).requires_grad_(
            requires_grad
        )
        B = (torch.randn(lc.r, d_out, generator=gen, dtype=torch.float64) * lc.scale).requires_grad_(
            requires_grad
        )
        out[target] = (A, B)
    return out


def init_noisy_lora_blocks(layers, cfg: BlockConfig, lc: LoRAConfig, seed: int = 9999):
    return [
        init_noisy_lora_block(cfg, lc, seed=seed + i) for i in range(len(layers))
    ]


def lora_flat(blk: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    """Deterministic flat ordering of one layer's adapters for the Function."""
    return tuple(v for t in LORA_TARGETS for v in blk[t])


def lora_unflatten(
    flat: tuple[torch.Tensor, ...],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    it = iter(flat)
    return {t: (next(it), next(it)) for t in LORA_TARGETS}


def frozen_base(layers) -> list[dict[str, torch.Tensor]]:
    """Detached copies of the base weights, so their grads are never computed."""
    return [
        {n: t.detach().requires_grad_(False) for n, t in layer.items()} for layer in layers
    ]


def resident_lora_forward(
    x: torch.Tensor,
    layers: list[dict[str, torch.Tensor]],
    lora_blocks: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    cfg: BlockConfig,
) -> torch.Tensor:
    """Baseline: base weights + LoRA all resident, ordinary autograd."""
    h = x
    for layer, lora in zip(layers, lora_blocks):
        h = apply_block(h, layer, cfg, lora=lora)
    return h


# -- the streamed, LoRA-aware autograd Function -----------------------------


class LoRAStreamedBlock(torch.autograd.Function):
    """One block with streamed (frozen) base weights + resident LoRA adapters.

    The base weights follow the exact Phase 2 lifecycle (load->compute under
    no_grad->evict for forward; reload->recompute with grad->evict for backward).
    The LoRA adapters are never evicted: as the project targets, small adapter
    params stay resident while the big base weights stream.
    """

    @staticmethod
    def forward(ctx, spec, x: torch.Tensor, *host_params: torch.Tensor):
        n_base = len(PARAM_NAMES)
        base_flat = host_params[:n_base]
        lora_flat = host_params[n_base:]

        sim: SimDevice = spec.sim
        sim.phase = "forward"
        for name, host in zip(PARAM_NAMES, base_flat):
            sim.load(f"{spec.prefix}.{name}", host)

        dev_p = {n: sim.get(f"{spec.prefix}.{n}") for n in PARAM_NAMES}
        lora = lora_unflatten(tuple(lora_flat))
        with torch.no_grad():
            y = apply_block(x, dev_p, spec.cfg, lora=lora)
        del dev_p

        ctx.spec = spec
        ctx.save_for_backward(x, *base_flat, *lora_flat)

        if spec.evict:
            for name in PARAM_NAMES:
                sim.evict(f"{spec.prefix}.{name}")
        sim.phase = "-"
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        spec = ctx.spec
        sim = spec.sim
        saved = ctx.saved_tensors
        n_base = len(PARAM_NAMES)
        x = saved[0]
        base_flat = saved[1 : 1 + n_base]
        lora_flat = saved[1 + n_base :]

        n_in = len(spec.all_layers[spec.idx])  # == n_base
        want_x = ctx.needs_input_grad[1]
        want_base = tuple(ctx.needs_input_grad[2 : 2 + n_in])
        want_lora = tuple(ctx.needs_input_grad[2 + n_in :])

        sim.phase = "backward"
        source = spec.all_layers[spec.idx]
        dev_p = {
            n: sim.load(f"{spec.prefix}.{n}", source[n], requires_grad=w)
            for n, w in zip(PARAM_NAMES, want_base)
        }
        lora = lora_unflatten(lora_flat)

        # Recompute the block with grad enabled, threading the resident LoRA
        # adapters (saved leaves) straight into the graph.
        with torch.enable_grad():
            xd = x.detach()
            if want_x:
                xd.requires_grad_(True)
            y = apply_block(xd, dev_p, spec.cfg, lora=lora)

        # Collect grad targets. Frozen base weights stay out (their grads are
        # None); frozen LoRA adapters are likewise skipped via needs_input_grad.
        targets: list[torch.Tensor] = []
        if want_x:
            targets.append(xd)
        for name, w in zip(PARAM_NAMES, want_base):
            if w:
                targets.append(dev_p[name])
        want_idx = iter(want_lora)
        for t in LORA_TARGETS:
            A, B = lora[t]
            if next(want_idx):
                targets.append(A)
            if next(want_idx):
                targets.append(B)

        raw = torch.autograd.grad(y, targets, grad_y, allow_unused=True) if targets else ()

        if spec.evict:
            for name in PARAM_NAMES:
                sim.evict(f"{spec.prefix}.{name}")

        it = iter(raw)
        grad_x = next(it) if want_x else None
        grad_base = [None if not w else next(it) for w in want_base]
        grad_lora: list[torch.Tensor | None] = []
        for wA, wB in zip(want_lora[0::2], want_lora[1::2]):
            grad_lora.append(None if not wA else next(it))
            grad_lora.append(None if not wB else next(it))

        sim.phase = "-"
        return (None, grad_x, *grad_base, *grad_lora)


def streamed_lora_forward(
    x: torch.Tensor,
    layers: list[dict[str, torch.Tensor]],
    lora_blocks: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    cfg: BlockConfig,
    sim: SimDevice,
    evict: bool = True,
) -> torch.Tensor:
    """Mechanism: streamed base weights + resident LoRA adapters."""
    from .streaming import BlockSpec

    h = x
    for idx in range(len(layers)):
        spec = BlockSpec(
            cfg=cfg,
            sim=sim,
            idx=idx,
            all_layers=layers,
            evict=evict,
        )
        args = (h, *[layers[idx][n] for n in PARAM_NAMES], *lora_flat(lora_blocks[idx]))
        h = LoRAStreamedBlock.apply(spec, *args)
    return h


# -- the Phase 3 gate ------------------------------------------------------


@dataclass
class TrainerConfig:
    steps: int = 400
    lr: float = 5e-3
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    batch: int = 2
    seq: int = 3
    seed: int = 0
    tol: float = 0.05  # Phase 3 gate: final mean-loss relative gap


def make_task(cfg: BlockConfig, tc: TrainerConfig):
    """Shared data + frozen base + fresh adapters for a train/compare run.

    The target is the constant output of the resident stack with a *hidden* set
    of adapters (`target_blocks`, its own seed), so there is a genuinely
    learnable signal and every trainer optimises against the same fixed target.
    """
    layers = _base_layers_raw(cfg, tc.seed)
    frozen = frozen_base(layers)
    x = torch.randn(
        tc.batch, tc.seq, cfg.d_model, dtype=torch.float64,
        generator=torch.Generator().manual_seed(tc.seed + 5),
    )
    lora_blocks = init_lora_blocks(layers, cfg, tc.lora, seed=tc.seed + 11)

    hidden = init_noisy_lora_blocks(layers, cfg, LoRAConfig(r=cfg.d_model), seed=9999)
    with torch.no_grad():
        yt = resident_lora_forward(x, frozen, hidden, cfg).detach()
    return frozen, lora_blocks, x, yt


def _base_layers_raw(cfg: BlockConfig, seed: int):
    from .toy import init_layers

    return init_layers(cfg, 3, seed=seed)


def _adapter_params(lora_blocks):
    return [p for blk in lora_blocks for (A, B) in blk.values() for p in (A, B)]


def _generic_train(forward_fn, frozen, lora_blocks, x, yt, cfg, tc) -> list[float]:
    """One Adam loop body; only `forward_fn` differs between modes."""
    params = _adapter_params(lora_blocks)
    opt = torch.optim.Adam(params, lr=tc.lr)
    losses: list[float] = []
    for _ in range(tc.steps):
        opt.zero_grad()
        out = forward_fn(x, frozen, lora_blocks, cfg)
        loss = torch.nn.functional.mse_loss(out, yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_resident(frozen, lora_blocks, x, yt, cfg: BlockConfig, tc: TrainerConfig) -> list[float]:
    return _generic_train(resident_lora_forward, frozen, lora_blocks, x, yt, cfg, tc)


def train_streamed(frozen, lora_blocks, x, yt, cfg: BlockConfig, tc: TrainerConfig) -> list[float]:
    sim = SimDevice()

    def fwd(x, layers, loras, cfg):
        h = x
        from .streaming import BlockSpec

        for idx in range(len(layers)):
            spec = BlockSpec(cfg=cfg, sim=sim, idx=idx, all_layers=layers, evict=True)
            h = LoRAStreamedBlock.apply(
                spec, h, *[layers[idx][n] for n in PARAM_NAMES], *lora_flat(loras[idx])
            )
        return h

    losses = _generic_train(fwd, frozen, lora_blocks, x, yt, cfg, tc)
    sim.assert_empty("streamed training")
    return losses


@dataclass
class Phase3Verdict:
    passed: bool
    resident: list[float]
    streamed: list[float]
    gap: float
    tol: float
    n_steps: int


def _clone_blocks(lora_blocks):
    """Deep, value-identical copy so two modes start from the same point and
    their optimisers never touch each other's tensors."""
    import copy

    return copy.deepcopy(lora_blocks)


def run_phase3(cfg: BlockConfig | None = None, tc: TrainerConfig | None = None):
    cfg = cfg or BlockConfig()
    tc = tc or TrainerConfig()
    frozen, lora_blocks, x, yt = make_task(cfg, tc)
    x_detached = x.detach()  # never optimised on either side
    resident = train_resident(
        frozen, _clone_blocks(lora_blocks), x_detached, yt, cfg, tc
    )
    streamed = train_streamed(
        frozen, _clone_blocks(lora_blocks), x_detached, yt, cfg, tc
    )
    tail = max(1, tc.steps // 5)
    mean_r = sum(resident[-tail:]) / tail
    mean_s = sum(streamed[-tail:]) / tail
    gap = abs(mean_s - mean_r) / mean_r if mean_r else 0.0
    return Phase3Verdict(
        passed=gap < tc.tol,
        resident=resident,
        streamed=streamed,
        gap=gap,
        tol=tc.tol,
        n_steps=tc.steps,
    )