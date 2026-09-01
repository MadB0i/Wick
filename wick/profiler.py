"""Wick device profiler: what can I fine-tune on my hardware?

Runs BEFORE any training and tells you, from pure lookup-table math + a real
VRAM read, which VLM the Wick streaming approach can fine-tune on your machine.
No model downloads, no weights, no CUDA required to *answer* — but a CUDA GPU is
required to actually train.

Peak-VRAM model (the whole point of Wick):

    peak = (  one vision-encoder layer, fp16 )     <- streamed, only 1 resident
          + ( LLM backbone resident )               <- fp16 OR int4 (fp32 master
                                                       lives on CPU, never in VRAM)
          + ( LoRA adapters, fp32 )                 <- resident, trainable
          + ( optimizer states for LoRA only, fp32 ) <- Adam keeps 2 fp32 states
          + ( activation buffer, ~1 layer )
          + ( safety headroom, 10% )

Everything not in that sum is either streamed (vision encoder) or host RAM
(fp32 masters / optimizer we do NOT store on device for streamed layers).

Usage
-----
    python -m wick.profiler                 # standalone CLI
    from wick.profiler import recommend     # from code
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared dependency
    psutil = None

#: fp16 bytes/param, fp32 bytes/param, packed int4 bytes/param (incl. overhead)
FP16 = 2
FP32 = 4
INT4 = 0.55


@dataclass(frozen=True)
class ModelSpec:
    """Hardcoded (no download) footprint facts for a common VLM.

    `llm_params` truthfully drives the resident LLM term. `vision_layer_fp16`
    is ONE encoder layer's fp16 size (what Wick keeps resident while streaming).
    `lora_params` / `activation_bytes` are conservative hardcodes.
    """

    name: str
    llm_params: int
    vision_layer_fp16: int
    lora_params: int
    activation_bytes: int

    @property
    def llm_fp16(self) -> int:
        return int(self.llm_params * FP16)

    @property
    def llm_int4(self) -> int:
        return int(self.llm_params * INT4)

    @property
    def lora_fp32(self) -> int:
        return int(self.lora_params * FP32)

    @property
    def lora_optimizer_fp32(self) -> int:
        return int(self.lora_params * FP32 * 2)  # Adam state m + v, both fp32


#: the lookup table. All sizes are hardcoded — nothing is downloaded or scanned.
MODEL_TABLE: list[ModelSpec] = [
    ModelSpec("MiniCPM-V 1.3B", 1_200_000_000, 33_000_000, 3_600_000, 64_000_000),
    ModelSpec("MiniCPM-V 2.6B", 2_600_000_000, 33_000_000, 7_800_000, 96_000_000),
    ModelSpec("Qwen2-VL 2B", 2_000_000_000, 21_000_000, 6_000_000, 128_000_000),
    ModelSpec("LLaVA-1.5 7B", 7_000_000_000, 25_000_000, 21_000_000, 256_000_000),
    ModelSpec("Phi-3.5 Vision", 4_200_000_000, 33_000_000, 12_600_000, 128_000_000),
]

HEADROOM = 1.10


def peak_for(model: ModelSpec, llm_dtype: str) -> int:
    """Peak streamed-training VRAM (bytes) for one model + one resident LLM dtype.

    `llm_dtype` is ``"fp16"`` or ``"int4"``. The vision encoder is always
    streamed (one fp16 layer resident); the LLM is whichever is resident.
    """
    llm = model.llm_fp16 if llm_dtype == "fp16" else model.llm_int4
    base = (
        model.vision_layer_fp16  # streamed, 1 layer
        + llm
        + model.lora_fp32
        + model.lora_optimizer_fp32
        + model.activation_bytes
    )
    return int(base * HEADROOM)


@dataclass
class ModelFit:
    name: str
    best_dtype: str  # "fp16" | "int4" | None
    peak: int  # bytes at the best (lowest) dtype
    peak_fp16: int
    peak_int4: int
    status: str  # "can" | "borderline" | "cannot"


@dataclass(frozen=True)
class Hardware:
    kind: str  # "cuda" | "cpu-only"
    name: str
    vram_bytes: int
    capability: tuple[int, int] | None
    bf16: bool
    fp16: bool
    ram_bytes: int

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024**3)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / (1024**3)


def _system_ram() -> int:
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().total)
        except Exception:  # pragma: no cover - defensive
            return 0
    return 0


def detect_hardware() -> Hardware:
    """Auto-detect the GPU (and CPU-only fallback). Never downloads a model."""
    if not torch.cuda.is_available():
        return Hardware("cpu-only", "no CUDA device", 0, None, False, False, _system_ram())
    props = torch.cuda.get_device_properties(0)
    cap = (int(props.major), int(props.minor))
    total = int(props.total_memory)
    return Hardware(
        kind="cuda",
        name=str(props.name),
        vram_bytes=total,
        capability=cap,
        bf16=cap >= (8, 0),
        fp16=cap >= (7, 0),
        ram_bytes=_system_ram(),
    )


@dataclass
class Recommendation:
    hardware: Hardware
    fits: "list[ModelFit]" = field(default_factory=list)

    @property
    def can_fit(self) -> "list[ModelFit]":
        return [f for f in self.fits if f.status == "can"]

    @property
    def borderline(self) -> "list[ModelFit]":
        return [f for f in self.fits if f.status == "borderline"]

    @property
    def cannot_fit(self) -> "list[ModelFit]":
        return [f for f in self.fits if f.status == "cannot"]


def recommend(hardware: Hardware | None = None) -> Recommendation:
    """Classify every model in the table against the given hardware.

    Defaults to auto-detection (`detect_hardware`). A ``Hardware`` object can be
    injected for tests / headless planning.
    """
    hw = hardware or detect_hardware()
    if hw.kind == "cpu-only":
        return Recommendation(hardware=hw)  # empty fits; CLI prints a graceful note

    budget = hw.vram_bytes
    fits: list[ModelFit] = []
    for m in MODEL_TABLE:
        pf = peak_for(m, "fp16")
        pi = peak_for(m, "int4")

        # Prefer fp16 (no int4 quality loss) whenever it actually fits; fall
        # back to int4 only if fp16 exceeds the card.
        if pf <= budget:
            best, best_dtype = pf, "fp16"
        elif pi <= budget:
            best, best_dtype = pi, "int4"
        elif pf <= pi:
            best, best_dtype = pf, "fp16"
        else:
            best, best_dtype = pi, "int4"

        if best > budget:
            status = "cannot"
        elif best > 0.9 * budget:
            status = "borderline"
        else:
            status = "can"
        fits.append(ModelFit(m.name, best_dtype, best, pf, pi, status))
    return Recommendation(hardware=hw, fits=fits)


def _gb(b: int) -> str:
    return f"{b / (1024**3):.2f} GB"


def _fmt_line(fit: ModelFit, budget: int, emoji: str) -> str:
    note = ""
    if fit.best_dtype == "int4":
        note = " (with int4 LLM)"
    elif fit.peak_fp16 > budget and fit.peak_int4 <= budget:
        note += " [int4 makes it fit]"
    head = f"{emoji} {fit.name} ← in {_gb(fit.peak)}{' int4' if fit.best_dtype == 'int4' else ''}"
    return head


def main() -> int:
    hw = detect_hardware()
    if hw.kind == "cpu-only":
        # Still print the usable hardware picture so a CPU-only user knows why.
        print("=" * 72)
        print("WICK DEVICE PROFILER")
        print("=" * 72)
        print("No CUDA GPU detected on this machine.")
        print("Wick fine-tuning is VRAM-streaming and therefore needs a CUDA GPU.")
        if hw.ram_gb:
            print(f"System RAM available: {hw.ram_gb:.0f} GB (enough to hold model masters).")
        print("Install a CUDA-enabled torch (e.g. torch==2.6.0+cu124), then re-run.")
        print("=" * 72)
        return 0

    r = recommend(hw)
    budget = hw.vram_bytes
    print("=" * 72)
    print("WICK DEVICE PROFILER")
    print("=" * 72)
    print(f"Your GPU: {hw.name}")
    print(f"  VRAM      {hw.vram_gb:.2f} GB ({_gb(budget)})")
    print(f"  compute   {hw.capability[0]}.{hw.capability[1]} "
          f"({('bf16' if hw.bf16 else 'no bf16')}, "
          f"{'fp16' if hw.fp16 else 'no fp16'})")
    if hw.ram_gb:
        print(f"  system RAM {hw.ram_gb:.0f} GB (host masters / optimizer offload)")
    print()
    print("  Peak-VRAM model (Wick streaming): 1 vision layer fp16 + resident LLM")
    print("  + LoRA fp32 + LoRA optimizer + ~1-layer activation + 10% headroom")
    print()
    for fit in r.fits:
        tag = "✅" if fit.status == "can" else ("⚠️" if fit.status == "borderline" else "❌")
        print(
            f"  {tag} {fit.name:<18} best({fit.best_dtype:>4}) "
            f"{_gb(fit.peak):>9} of {_gb(budget):>7} VRAM"
        )
        print(
            f"          fp16 needs {_gb(fit.peak_fp16):>9}, "
            f"int4 needs {_gb(fit.peak_int4):>9}"
        )
    print()
    print("  ✅ Can fine-tune:")
    for f in r.can_fit:
        print(f"     - {f.name}")
    if r.borderline:
        print("  ⚠️  Borderline:")
        for f in r.borderline:
            print(f"     - {f.name}")
    if r.cannot_fit:
        print("  ❌ Cannot fit:")
        for f in r.cannot_fit:
            print(f"     - {f.name}")
    print("=" * 72)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())