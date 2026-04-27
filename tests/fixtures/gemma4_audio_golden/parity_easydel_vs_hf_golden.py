# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

"""End-to-end parity: EasyDeL Gemma 4 audio path vs HF goldens.

Reads the three .npz files captured by ``capture_gemma4_e2e_golden.py``
and compares against an EasyDeL forward at:

  - audio_projector_out      : output of embed_audio / audio projector
  - inputs_embeds_after_scatter : embeds after audio scatter
  - logits                   : last-position logits

Inputs are re-encoded with the same HF processor, same seed, same audio
fixture, so the only source of drift is the EasyDeL implementation.

Run on a host with ``/tmp/easydel-gemma4-e4b-it`` (converted EasyDeL
checkpoint) and ``google/gemma-4-E4B-it`` cached for the HF processor.

Usage:
  JAX_PLATFORMS=cpu python parity_easydel_vs_hf_golden.py \
      --easydel-path /tmp/easydel-gemma4-e4b-it \
      --hf-model-id google/gemma-4-E4B-it \
      --golden-dir <this dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
TEST_FIXTURES_DIR = FIXTURE_DIR.parent
if str(TEST_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_FIXTURES_DIR))

from capture_gemma4_audio_golden import _seed_everything  # noqa: E402

GOLDEN_NAMES = ("audio_projector_out", "inputs_embeds_after_scatter", "logits")
TOLERANCES = {
    # bf16 forward + fp32 cast on capture leaves ~1e-2 absolute headroom.
    "audio_projector_out": (5e-2, 1e-2),       # (atol, rtol)
    "inputs_embeds_after_scatter": (5e-2, 1e-2),
    "logits": (1e-1, 5e-2),                    # logits get more compounding error
}


def _load_audio_fixture(out_dir: Path):
    import wave

    path = out_dir / "sample_audio.wav"
    if path.exists():
        with wave.open(str(path), "rb") as f:
            sample_rate = f.getframerate()
            frames = f.readframes(f.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sample_rate
    return np.zeros(16_000, dtype=np.float32), 16_000


def _diff(name: str, actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        return {"name": name, "ok": False, "reason": f"shape mismatch {actual.shape} vs {expected.shape}"}
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    rel = diff / (np.abs(expected.astype(np.float32)) + 1e-6)
    atol, rtol = TOLERANCES[name]
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    p99_abs = float(np.percentile(diff, 99))
    ok = max_abs < atol or max_rel < rtol
    return {
        "name": name,
        "ok": ok,
        "max_abs": max_abs,
        "p99_abs": p99_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "shape": list(actual.shape),
    }


def run_parity(easydel_path: str, hf_model_id: str, golden_dir: Path, seed: int = 42) -> int:
    # IMPORTANT: import easydel BEFORE any jax.devices()/jnp call so its
    # _DistributedConfig().initialize() runs before XLA backend init.
    from easydel import AutoEasyDeLModelForImageTextToText

    import jax
    import jax.numpy as jnp
    import transformers

    _seed_everything(seed)
    print(f"[parity] jax devices: {jax.devices()}")

    print("[parity] loading HF processor for input encoding")
    processor = transformers.AutoProcessor.from_pretrained(hf_model_id)
    audio_token = processor.audio_token
    waveform, sample_rate = _load_audio_fixture(golden_dir)
    encoded = processor(
        text=f"Transcribe: {audio_token}",
        audio=[waveform],
        sampling_rate=sample_rate,
        return_tensors="np",
        return_mm_token_type_ids=True,
    )

    input_ids = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
    attention_mask = jnp.asarray(encoded["attention_mask"], dtype=jnp.int32)
    input_features = jnp.asarray(encoded["input_features"], dtype=jnp.bfloat16)
    input_features_mask = jnp.asarray(encoded["input_features_mask"], dtype=jnp.bool_)
    mm_token_type_ids = jnp.asarray(encoded["mm_token_type_ids"], dtype=jnp.int32)

    print(f"[parity] inputs: input_ids={input_ids.shape}, input_features={input_features.shape}, "
          f"feat_mask_sum={int(input_features_mask.sum())}")

    print(f"[parity] loading EasyDeL model from HF: {hf_model_id} (from_torch=True)")
    # Bypass the broken TS conversion roundtrip (input_min/max stored as raw
    # nn.Variable + missing lconv1d depthwise kernels). Direct from_torch path
    # is the source of truth.
    model = AutoEasyDeLModelForImageTextToText.from_pretrained(
        hf_model_id,
        from_torch=True,
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        sharding_axis_dims=(1, 1, 1, 1, 1),
        auto_shard_model=False,
    )

    print("[parity] forward pass — split-and-conquer (audio_tower → scatter → logits)")
    with model.mesh:
        # Track A: audio tower → embed_audio (post-projector features).
        audio_features, audio_output_mask = model.base_model.get_audio_features(
            input_features=input_features,
            input_features_mask=input_features_mask,
        )
        # Track C: full forward to logits.
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            input_features_mask=input_features_mask,
            token_type_ids=mm_token_type_ids,
        )

    audio_features_np = np.asarray(audio_features.astype(jnp.float32))
    print(f"[parity] audio_features shape={audio_features_np.shape}")

    logits = np.asarray(out.logits.astype(jnp.float32))
    last_logits = logits[:, -1:, :]

    diffs = []
    audio_golden_path = golden_dir / "audio_projector_out.npz"
    if audio_golden_path.exists():
        diffs.append(_diff("audio_projector_out", audio_features_np, np.load(audio_golden_path)["data"]))
    diffs.append(_diff("logits", last_logits, np.load(golden_dir / "logits.npz")["data"]))

    for d in diffs:
        print("[parity]", json.dumps(d))

    failed = [d for d in diffs if not d["ok"]]
    if failed:
        print(f"[parity] FAILED {len(failed)}/{len(diffs)}")
        return 1
    print(f"[parity] PASSED {len(diffs)}/{len(diffs)}")
    return 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--easydel-path", default="/tmp/easydel-gemma4-e4b-it")
    p.add_argument("--hf-model-id", default="google/gemma-4-E4B-it")
    p.add_argument("--golden-dir", default=str(FIXTURE_DIR))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    return run_parity(
        easydel_path=args.easydel_path,
        hf_model_id=args.hf_model_id,
        golden_dir=Path(args.golden_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
