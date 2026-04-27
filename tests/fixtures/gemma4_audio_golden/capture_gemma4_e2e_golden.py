# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Capture end-to-end HuggingFace Gemma 4 audio golden tensors.

This extends the audio-tower golden fixture set with tensors that require a
full ``Gemma4ForConditionalGeneration`` forward pass:

    audio_projector_out.npz
    inputs_embeds_after_scatter.npz
    logits.npz

The script intentionally lives beside the generated fixture files so capture
runs do not modify any existing tests or CI wiring.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


FIXTURE_DIR = Path(__file__).resolve().parent
TEST_FIXTURES_DIR = FIXTURE_DIR.parent
if str(TEST_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_FIXTURES_DIR))

from capture_gemma4_audio_golden import _flatten_to_arrays, _seed_everything  # noqa: E402


TARGETS = {
    "audio_projector_out": FIXTURE_DIR / "audio_projector_out.npz",
    "inputs_embeds_after_scatter": FIXTURE_DIR / "inputs_embeds_after_scatter.npz",
    "logits": FIXTURE_DIR / "logits.npz",
}


def _save_npz(path: Path, payload) -> None:
    arrays = _flatten_to_arrays(payload)
    if not arrays:
        raise RuntimeError(f"{path.name}: payload produced no numpy arrays")
    np.savez(path, **arrays)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        return signed.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        with wave.open(str(path), "rb") as fh:
            sample_rate = fh.getframerate()
            channels = fh.getnchannels()
            sample_width = fh.getsampwidth()
            frames = fh.readframes(fh.getnframes())
        waveform = _decode_pcm(frames, sample_width)
        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"{path} must decode to mono or stereo audio, got shape {waveform.shape}")
    return np.clip(waveform, -1.0, 1.0).astype(np.float32), int(sample_rate)


def _load_audio_fixture(path: Path) -> tuple[np.ndarray, int]:
    if path.exists():
        return _load_wav(path)
    return np.zeros(16_000, dtype=np.float32), 16_000


def _find_submodule(root, candidate_names: tuple[str, ...]):
    modules = dict(root.named_modules())
    for name in candidate_names:
        if name in modules:
            return name, modules[name]
    suffixes = tuple(f".{name}" for name in candidate_names)
    for name, module in modules.items():
        if name.endswith(suffixes):
            return name, module
    return None, None


def _hook_language_model_inputs(captured: dict[str, object]):
    def hook(_module, args, kwargs, _output):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and len(args) >= 4:
            inputs_embeds = args[3]
        if inputs_embeds is None:
            raise RuntimeError("language_model hook could not locate inputs_embeds")
        captured["inputs_embeds_after_scatter"] = inputs_embeds.detach()

    return hook


def capture(model_id: str, out_dir: Path, seed: int, revision: str | None = None) -> None:
    import torch
    import transformers

    _seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_audio = out_dir / "sample_audio.wav"
    waveform, sample_rate = _load_audio_fixture(sample_audio)

    processor = transformers.AutoProcessor.from_pretrained(model_id, revision=revision)
    audio_token = getattr(processor, "audio_token", None)
    if not audio_token:
        raise RuntimeError("Gemma4 processor does not expose audio_token")

    encoded = processor(
        text=f"Transcribe: {audio_token}",
        audio=[waveform],
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )

    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    captured: dict[str, object] = {}
    handles = []

    projector_name, projector = _find_submodule(
        model,
        ("model.audio_projector", "audio_projector", "model.embed_audio", "embed_audio"),
    )
    if projector is None:
        available = [name for name, _ in model.named_modules() if "audio" in name or "embed" in name]
        raise RuntimeError(f"Could not locate audio projector. Candidates: {available[:80]}")

    language_model_name, language_model = _find_submodule(
        model,
        ("model.language_model", "language_model"),
    )
    if language_model is None:
        raise RuntimeError("Could not locate language_model for inputs_embeds scatter capture")

    def projector_hook(_module, _inputs, output):
        captured["audio_projector_out"] = output

    try:
        handles.append(projector.register_forward_hook(projector_hook))
        handles.append(language_model.register_forward_hook(_hook_language_model_inputs(captured), with_kwargs=True))

        model_inputs = {
            key: value.to("cpu") if hasattr(value, "to") else value
            for key, value in encoded.items()
            if key
            in {
                "input_ids",
                "attention_mask",
                "input_features",
                "input_features_mask",
                "mm_token_type_ids",
            }
        }
        if "input_features" in model_inputs:
            model_inputs["input_features"] = model_inputs["input_features"].to(torch.bfloat16)

        with torch.inference_mode():
            output = model(
                **model_inputs,
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            )
            captured["logits"] = output.logits.detach()
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(TARGETS) - set(captured))
    if missing:
        raise RuntimeError(
            f"Missing captures {missing}; hooked projector={projector_name}, language_model={language_model_name}"
        )

    for name, path in TARGETS.items():
        _save_npz(path, captured[name])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default="google/gemma-4-E4B")
    parser.add_argument("--out-dir", default=str(FIXTURE_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    capture(
        model_id=args.model_id,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        revision=args.revision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
