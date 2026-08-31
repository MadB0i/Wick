"""A fake accelerator, used to test streaming correctness without a GPU.

Both "host" and "device" tensors physically live in CPU RAM. What makes the
simulation meaningful is that a device copy is a *distinct tensor object*,
created by `SimDevice.load()` and destroyed by `SimDevice.evict()`. That is
enough to test the only thing the Phase 2 gate cares about: whether a layer's
parameters are present exactly when they are needed, and genuinely gone
otherwise.

Two features do the real work of catching bugs:

* `get()` raises `NotResidentError` if a layer is read while evicted. This
  catches ordering bugs -- computing before a load, or after an evict.

* `evict()` overwrites the departing tensor with NaN ("poisoning"). A correct
  implementation never touches a device tensor after eviction, so poisoning is
  invisible to it. An implementation that leaked a device tensor into the
  autograd graph -- the classic forward-only-streaming trap -- will silently
  read the poisoned storage during backward and produce NaN gradients, which
  gradcheck reports immediately.

Poisoning writes through `.data` so it never bumps a version counter. The goal
is a clean NaN signal, not an autograd in-place error.

On top of that, `SimDevice` keeps a weakref to every tensor it has ever handed
out. Its own byte accounting only tracks what it was *told* to forget, which is
not the same question as whether the memory was actually released: if autograd
saved a device tensor for backward, that tensor stays alive no matter how
diligently we call `evict()`. `leaked_names()` answers the question that
actually matters for a 4 GB card -- did the eviction free anything?
"""

from __future__ import annotations

import gc
import math
import weakref
from dataclasses import dataclass, field

import torch


class NotResidentError(RuntimeError):
    """A layer's parameters were read while evicted from the simulated device."""


class ResidencyBudgetExceeded(RuntimeError):
    """More bytes were resident at once than the simulated device allows."""


@dataclass
class Event:
    """One movement across the simulated PCIe bus."""

    op: str  # "load" | "evict"
    name: str
    phase: str  # "forward" | "backward" | "-"
    nbytes: int
    resident_after: int

    def __str__(self) -> str:  # pragma: no cover - display only
        arrow = "->" if self.op == "load" else "<-"
        return f"[{self.phase:8s}] {arrow} {self.op:5s} {self.name:20s} resident={self.resident_after}B"


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def graph_saved_tensors(root: torch.Tensor) -> list[torch.Tensor]:
    """Every tensor the autograd graph rooted at `root` is holding for backward.

    This is the forensic tool the gate needs. A weakref to a device tensor
    *object* is not a memory measurement: `p.transpose(0, 1)` is a different
    object that shares `p`'s storage, so the object can die while the
    allocation it pointed at stays alive inside the graph. Walking the graph and
    reading each node's saved tensors asks the question directly -- what is
    autograd still holding, and where does it point?
    """
    seen: set[int] = set()
    found: list[torch.Tensor] = []
    stack = [root.grad_fn] if root.grad_fn is not None else []
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))

        names = [a for a in dir(node) if a.startswith("_saved_")]
        # Custom autograd.Function nodes expose their saves here instead.
        if hasattr(node, "saved_tensors"):
            names.append("saved_tensors")
        for attr in names:
            try:
                val = getattr(node, attr)
            except (RuntimeError, AttributeError):
                continue  # SavedVariable already freed, or not readable
            if isinstance(val, torch.Tensor):
                found.append(val)
            elif isinstance(val, (tuple, list)):
                found.extend(v for v in val if isinstance(v, torch.Tensor))

        for nxt in getattr(node, "next_functions", ()):
            stack.append(nxt[0] if isinstance(nxt, tuple) else nxt)
    return found


@dataclass
class SimDevice:
    """Tracks which named tensors are resident on the simulated device."""

    budget_bytes: int | None = None
    poison_on_evict: bool = True
    log_events: bool = False
    track_liveness: bool = True

    phase: str = "-"
    events: list[Event] = field(default_factory=list)
    _resident: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _wrefs: dict[str, list] = field(default_factory=dict, repr=False)
    _handed_out: list[tuple[int, str]] = field(default_factory=list, repr=False)
    peak_bytes: int = 0
    n_loads: int = 0
    n_evicts: int = 0
    bytes_moved: int = 0

    # -- residency ---------------------------------------------------------

    def load(self, name: str, host: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """Copy `host` onto the simulated device and register it under `name`.

        Returns a leaf tensor. When `requires_grad` is set the returned copy is
        a differentiable leaf, so a caller can use it directly as an
        `autograd.grad` target instead of making an untracked second copy --
        which would understate device memory use.
        """
        dev = host.detach().clone()
        if requires_grad:
            dev.requires_grad_(True)
        self._resident[name] = dev
        if self.track_liveness:
            self._wrefs.setdefault(name, []).append(weakref.ref(dev))
            self._handed_out.append((dev.data_ptr(), name))

        nbytes = tensor_bytes(dev)
        self.n_loads += 1
        self.bytes_moved += nbytes
        resident = self.resident_bytes()
        self.peak_bytes = max(self.peak_bytes, resident)
        if self.budget_bytes is not None and resident > self.budget_bytes:
            raise ResidencyBudgetExceeded(
                f"loading {name} put {resident}B resident, over the "
                f"{self.budget_bytes}B budget; resident={sorted(self._resident)}"
            )
        self._record("load", name, nbytes, resident)
        return dev

    def get(self, name: str) -> torch.Tensor:
        """Read a resident device tensor, or fail loudly if it is not there."""
        try:
            return self._resident[name]
        except KeyError:
            raise NotResidentError(
                f"{name!r} was read during {self.phase} but is not resident "
                f"(resident={sorted(self._resident)})"
            ) from None

    def evict(self, name: str) -> None:
        """Free a device tensor, poisoning it so stale references are visible."""
        dev = self._resident.pop(name, None)
        if dev is None:
            raise NotResidentError(f"cannot evict {name!r}: not resident")
        if self.poison_on_evict:
            # Write through .data: corrupt the storage without touching the
            # autograd version counter.
            dev.data.fill_(math.nan)
        self.n_evicts += 1
        self._record("evict", name, tensor_bytes(dev), self.resident_bytes())

    def evict_all(self) -> None:
        for name in list(self._resident):
            self.evict(name)

    # -- accounting --------------------------------------------------------

    def resident_bytes(self) -> int:
        return sum(tensor_bytes(t) for t in self._resident.values())

    def resident_names(self) -> list[str]:
        return sorted(self._resident)

    def leaked_names(self) -> list[str]:
        """Names whose device copy was evicted but is *still alive* in memory.

        This is the load-bearing measurement for Wick. `evict()` only drops
        SimDevice's own reference; if anything else still holds the tensor --
        most importantly autograd, having saved it for backward -- the memory is
        not returned, and streaming buys nothing. A correct implementation
        reports an empty list here.
        """
        if not self.track_liveness:
            raise RuntimeError("liveness tracking is disabled")
        gc.collect()
        leaked = []
        for name, refs in self._wrefs.items():
            if name in self._resident:
                continue  # legitimately held right now
            if any(ref() is not None for ref in refs):
                leaked.append(name)
        return sorted(leaked)

    def retained_by_graph(self, root: torch.Tensor) -> tuple[list[str], int]:
        """Which evicted device allocations is the autograd graph still holding?

        Returns `(names, n_graph_saved)`. The second value is the total number of
        tensors found in the graph: if it is 0 the walk found nothing and an
        empty `names` proves nothing, so callers should report it rather than
        read a vacuous pass as a real one.

        Compares by `data_ptr`, so it sees storage retained through a view whose
        own tensor object has already died -- exactly the case that makes
        hook-based eviction unsafe in backward.
        """
        saved = graph_saved_tensors(root)
        graph_ptrs = {t.data_ptr() for t in saved}
        names = {
            name
            for ptr, name in self._handed_out
            if ptr in graph_ptrs and name not in self._resident
        }
        return sorted(names), len(saved)

    def assert_empty(self, where: str = "") -> None:
        if self._resident:
            raise AssertionError(
                f"device not empty{' after ' + where if where else ''}: "
                f"{sorted(self._resident)}"
            )

    def reset(self) -> None:
        self._resident.clear()
        self._wrefs.clear()
        self._handed_out.clear()
        self.events.clear()
        self.peak_bytes = 0
        self.n_loads = self.n_evicts = self.bytes_moved = 0
        self.phase = "-"

    def _record(self, op: str, name: str, nbytes: int, resident_after: int) -> None:
        if self.log_events:
            self.events.append(Event(op, name, self.phase, nbytes, resident_after))
