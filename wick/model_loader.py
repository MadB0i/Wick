"""Local-path model loader for SigLIP vision tower.

Loads from D:\\Projects\\Wick\\models\\siglip-so400m-patch14-384\\ which
must contain model.safetensors + config files placed manually by the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import SiglipVisionModel


LOCAL_DIR = Path(__file__).resolve().parent.parent / "models" / "siglip-so400m-patch14-384"
WEIGHTS_FILE = LOCAL_DIR / "model.safetensors"
EXPECTED_WEIGHTS_BYTES = 3_511_950_624
EXPECTED_LAYERS = 27
EXPECTED_PER_LAYER_PARAMS = 15_239_504  # includes biases (hand-calc 15,227,136 omitted them)
EXPECTED_TOTAL_PARAMS = 428_225_600  # 27 layers + embeddings + post_layernorm
PER_LAYER_FP16_MB = 29.04


def verify_local_folder() -> None:
    if not LOCAL_DIR.exists():
        print(
            f"ERROR: Local model folder not found: {LOCAL_DIR}\n"
            f"Create it and place model.safetensors inside."
        )
        sys.exit(1)
    if not WEIGHTS_FILE.exists():
        print(
            f"ERROR: model.safetensors not found in {LOCAL_DIR}\n"
            f"Expected file: {WEIGHTS_FILE}\n"
            f"Expected size: {EXPECTED_WEIGHTS_BYTES:,} bytes (3.27 GiB)\n"
            f"Download from: https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors\n"
            f"Then place it in {LOCAL_DIR}"
        )
        sys.exit(1)
    actual = WEIGHTS_FILE.stat().st_size
    if actual != EXPECTED_WEIGHTS_BYTES:
        print(
            f"ERROR: model.safetensors size mismatch\n"
            f"  expected: {EXPECTED_WEIGHTS_BYTES:,} bytes\n"
            f"  actual:   {actual:,} bytes\n"
            f"  diff:     {abs(EXPECTED_WEIGHTS_BYTES - actual):,} bytes\n"
            f"File may be corrupted or incomplete. Re-download."
        )
        sys.exit(1)


def load_siglip_vision() -> SiglipVisionModel:
    """Load SigLIP vision tower from the local folder, verify, and return.

    Exits with a clear message if verification fails (layer count or param
    count mismatch vs architecture recon numbers in AGENTS.md).
    """
    verify_local_folder()

    print(f"Loading SiglipVisionModel from {LOCAL_DIR} ...")
    model = SiglipVisionModel.from_pretrained(
        str(LOCAL_DIR),
        local_files_only=True,
    )
    model = model.float()
    enc = model.encoder

    # --- Verification ---
    n_layers = len(enc.layers)
    if n_layers != EXPECTED_LAYERS:
        print(
            f"FAIL: layer count mismatch\n"
            f"  expected: {EXPECTED_LAYERS}\n"
            f"  actual:   {n_layers}"
        )
        sys.exit(1)
    print(f"  layer count: {n_layers} (matches recon)")

    first_layer = enc.layers[0]
    first_params = sum(p.numel() for p in first_layer.parameters())
    if first_params != EXPECTED_PER_LAYER_PARAMS:
        print(
            f"FAIL: first-layer param count mismatch\n"
            f"  expected: {EXPECTED_PER_LAYER_PARAMS:,}\n"
            f"  actual:   {first_params:,}"
        )
        sys.exit(1)
    print(f"  first-layer params: {first_params:,} (matches recon ~15.23M)")

    total_params = sum(p.numel() for p in model.parameters())
    # total includes 27 encoder layers + embeddings + post_layernorm;
    # check it's in the ~400-450M range (not a hard constant).
    if total_params < 350_000_000 or total_params > 500_000_000:
        print(
            f"FAIL: total param count out of range\n"
            f"  expected: ~400-450M\n"
            f"  actual:   {total_params:,}"
        )
        sys.exit(1)
    print(f"  total vision params: {total_params:,} (~{total_params/1e6:.0f}M, within range)")
    print(f"  fp16 per-layer: {PER_LAYER_FP16_MB:.2f} MB")
    print(f"  fp16 total:     {total_params*2/1024**2:.1f} MB")

    return model
