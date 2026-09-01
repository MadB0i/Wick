"""Device profiler tests: deterministic, mocked torch.cuda + psutil.

No GPU needed and nothing is downloaded. We patch `torch.cuda` availability and
`get_device_properties` and (optionally) psutil so the three required scenarios --
4 GB, 8 GB, CPU-only -- can be asserted exactly on any machine.
"""

from __future__ import annotations

import types

import pytest

import wick.profiler as prof
from wick.profiler import MODEL_TABLE, detect_hardware, peak_for, recommend


class _Props:
    def __init__(self, name, total_memory, major, minor):
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor


def _fake_cuda(monkeypatch, available, props=None):
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: available
    cuda.get_device_properties = lambda i: props
    monkeypatch.setattr(prof.torch, "cuda", cuda)


def _fake_ram(monkeypatch, gb: float):
    class VM:
        def __init__(s):
            s.total = int(gb * (1024**3))
    vmm = types.SimpleNamespace()
    vmm.virtual_memory = lambda: VM()
    monkeypatch.setattr(prof, "psutil", vmm)


def _gpu(monkeypatch, name, vram, cap):
    _fake_ram(monkeypatch, 16.0)
    _fake_cuda(monkeypatch, True, _Props(name, vram, cap[0], cap[1]))


def test_detect_4gb_gpu(monkeypatch):
    _gpu(monkeypatch, "GTX 1650", 4294967296, (7, 5))
    hw = detect_hardware()
    assert hw.kind == "cuda"
    assert hw.name == "GTX 1650"
    assert hw.capability == (7, 5)
    assert hw.fp16 is True and hw.bf16 is False  # 7.x: fp16 yes, bf16 no
    assert hw.vram_gb == pytest.approx(4.0)


def test_bf16_needs_cap_8(monkeypatch):
    _gpu(monkeypatch, "A", 4294967296, (8, 0))
    assert detect_hardware().bf16 is True
    _gpu(monkeypatch, "B", 4294967296, (6, 1))
    assert detect_hardware().fp16 is False


def test_peak_formula_includes_all_terms():
    m = MODEL_TABLE[0]
    p = peak_for(m, "fp16")
    expect = int((m.vision_layer_fp16 + m.llm_fp16 + m.lora_fp32
                  + m.lora_optimizer_fp32 + m.activation_bytes) * prof.HEADROOM)
    assert p == expect


def test_int4_smaller_than_fp16_for_llm():
    m = MODEL_TABLE[0]
    assert peak_for(m, "int4") < peak_for(m, "fp16")


def test_4gb_recommendations(monkeypatch):
    _gpu(monkeypatch, "GTX 1650", 4294967296, (7, 5))
    r = recommend()
    names = {f.name: f for f in r.fits}
    assert names["MiniCPM-V 1.3B"].status == "can"
    assert names["MiniCPM-V 1.3B"].best_dtype == "fp16"
    assert names["MiniCPM-V 2.6B"].status == "can"
    assert names["MiniCPM-V 2.6B"].best_dtype == "int4"
    assert names["Qwen2-VL 2B"].status == "can"
    assert names["Qwen2-VL 2B"].best_dtype == "int4"
    assert names["LLaVA-1.5 7B"].status == "cannot"
    assert names["Phi-3.5 Vision"].status == "can"
    assert names["Phi-3.5 Vision"].best_dtype == "int4"
    assert {f.name for f in r.can_fit} == {"MiniCPM-V 1.3B", "MiniCPM-V 2.6B",
                                           "Qwen2-VL 2B", "Phi-3.5 Vision"}
    assert {f.name for f in r.cannot_fit} == {"LLaVA-1.5 7B"}


def test_8gb_recommends_more(monkeypatch):
    _gpu(monkeypatch, "RTX 2080", 8589934592, (7, 5))
    r = recommend()
    names = {f.name: f for f in r.fits}
    # 8 GB lets LLaVA-1.5 7B fit via int4 (fp16 still won't on 8 GB).
    assert names["LLaVA-1.5 7B"].status == "can"
    assert names["LLaVA-1.5 7B"].best_dtype == "int4"
    # and the 2.6B now fits at fp16.
    assert names["MiniCPM-V 2.6B"].status == "can"
    assert names["MiniCPM-V 2.6B"].best_dtype == "fp16"
    assert not r.cannot_fit


def test_cpu_only_graceful(monkeypatch):
    _fake_ram(monkeypatch, 8.0)
    _fake_cuda(monkeypatch, False)
    hw = detect_hardware()
    assert hw.kind == "cpu-only"
    r = recommend()
    assert r.hardware.kind == "cpu-only"
    assert r.fits == []
    assert r.can_fit == [] and r.cannot_fit == []


def test_detect_hardware_injectable():
    hw = prof.Hardware("cuda", "fake", 4294967296, (7, 5), False, True,
                     16_000_000_000)
    r = recommend(hardware=hw)
    assert {f.name for f in r.can_fit} == {"MiniCPM-V 1.3B", "MiniCPM-V 2.6B",
                                         "Qwen2-VL 2B", "Phi-3.5 Vision"}