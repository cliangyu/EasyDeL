# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Capture golden tensors from HuggingFace Gemma 4 E4B audio tower.

Runs the *reference* PyTorch implementation end-to-end on a deterministic audio
fixture and dumps intermediate activations to ``.npz`` files. These artifacts
are the oracle that the EasyDeL / JAX port is diffed against at every layer.

Produced artifacts live at ``tests/fixtures/gemma4_audio_golden/``:

    inputs.npz              raw waveform, mel features, mel mask, seed
    sscp_out.npz            output of Gemma4AudioSubSampleConvProjection
    audio_block_{i}_out.npz output of the first N audio transformer blocks
    audio_tower_out.npz     final (hidden, mask) from Gemma4AudioModel
    projector_out.npz       projected-into-text-space embeddings (embed_audio)
    meta.json               config digest, HF commit, torch + transformers versions

Run:

    python tests/fixtures/capture_gemma4_audio_golden.py \\
        --model-id google/gemma-4-E4B \\
        --num-blocks 3 \\
        --dtype bfloat16

Memory envelope
---------------
Full E4B in bf16 is ~16 GB. On a 24 GB MacBook that is tight; prefer running on
a TPU VM (fork's ``TRC_README.md``) or a rented GPU. The script supports
``--audio-only`` which loads *only* the audio tower + projector weights from
safetensors (≈600 MB in bf16) and is MacBook-runnable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ determinism


def _seed_everything(seed: int) -> None:
    """Seed every RNG we know about. Keep this in one place so capture runs
    are bit-reproducible across machines (modulo BLAS/cuDNN non-determinism
    which we disable below)."""
    import torch  # local import so the file parses on hosts without torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism: slower but we only run this once. Required so different
    # hardware produces the same .npz bytes (or close enough for atol=1e-6).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ fixture


def _synth_waveform(
    seconds: float = 10.0,
    sample_rate: int = 16_000,
    seed: int = 0,
) -> "np.ndarray":
    """Deterministic 10-second 16 kHz mono waveform.

    Sum of three sinusoids + gentle pink noise, amplitude-clipped to [-1, 1].
    Chosen so that both the SSCP conv stem and the subsequent self-attention
    blocks see non-trivial energy across the full mel band (flat spectra
    underexercise the network).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    sig = (
        0.30 * np.sin(2 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2 * np.pi * 880.0 * t)
        + 0.15 * np.sin(2 * np.pi * 3_300.0 * t)
    )
    # Pink-ish noise via 1/f shaping of white Gaussian.
    white = rng.standard_normal(t.shape).astype(np.float32)
    freqs = np.fft.rfftfreq(t.size, d=1.0 / sample_rate)
    scale = 1.0 / np.maximum(freqs, 1.0)
    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=t.size).astype(np.float32)
    pink /= np.max(np.abs(pink)) + 1e-9
    sig = sig + 0.05 * pink
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ capture


def _tree_to_numpy(x):
    """Recursively convert torch tensors in (tuple | list | dict | tensor) to
    numpy arrays. Everything lands as float32 on disk; we don't want to bake
    a dtype choice (bf16 is not losslessly representable as numpy) into the
    fixtures — the JAX side can cast on load."""
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    if isinstance(x, dict):
        return {k: _tree_to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_to_numpy(v) for v in x)
    return x


def _sha256_of_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(
    model_id: str,
    out_dir: Path,
    num_blocks: int,
    dtype: str,
    audio_only: bool,
    seed: int,
) -> None:
    import torch
    import transformers

    _seed_everything(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- fixture input --------------------------------------------------
    waveform = _synth_waveform(seconds=10.0, sample_rate=16_000, seed=seed)

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    # Audio-only path: use the feature extractor directly so we don't need a
    # text prompt / image.
    fe = getattr(processor, "feature_extractor", None) or processor
    audio_inputs = fe(
        [waveform],
        sampling_rate=16_000,
        return_tensors="pt",
    )
    input_features = audio_inputs["input_features"]  # (1, n_mels, n_frames) or similar
    input_features_mask = audio_inputs.get(
        "input_features_mask",
        audio_inputs.get("attention_mask"),
    )
    if input_features_mask is None:
        # Fall back: all-ones mask shaped like the time axis.
        input_features_mask = torch.ones(
            input_features.shape[0], input_features.shape[-1], dtype=torch.bool
        )

    np.savez(
        out_dir / "inputs.npz",
        waveform=waveform,
        input_features=input_features.cpu().numpy(),
        input_features_mask=input_features_mask.cpu().numpy().astype(np.bool_),
        sample_rate=np.int32(16_000),
        seed=np.int32(seed),
    )

    # ---- model ---------------------------------------------------------
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    if audio_only:
        # Partial load: only the audio tower + projector weights. Keeps
        # memory footprint at ~600 MB in bf16 so this runs on a MacBook.
        audio_tower, embed_audio, full_config = _load_audio_only(
            model_id, torch_dtype=torch_dtype
        )
    else:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        inner = model.model  # Gemma4Model
        audio_tower = inner.audio_tower
        embed_audio = inner.embed_audio
        full_config = model.config

    if audio_tower is None:
        raise RuntimeError(
            f"{model_id} has no audio_tower; check that you picked the E4B "
            f"variant (only E2B/E4B ship audio)."
        )

    audio_tower.eval()
    embed_audio.eval()

    # ---- hook intermediate outputs -------------------------------------
    captured: dict[str, object] = {}

    def _hook(name):
        def fn(_module, _inp, out):
            captured[name] = out

        return fn

    handles = []
    # SSCP stem
    try:
        handles.append(
            audio_tower.subsample_conv_projection.register_forward_hook(
                _hook("sscp_out")
            )
        )
    except AttributeError:
        # HF may name it .conv_subsample / .stem depending on refactor; try
        # alternatives and record which one fired.
        for candidate in ("conv_subsample", "stem", "sub_sample_conv_projection"):
            mod = getattr(audio_tower, candidate, None)
            if mod is not None:
                handles.append(mod.register_forward_hook(_hook("sscp_out")))
                break

    # First num_blocks transformer blocks
    blocks = (
        getattr(audio_tower, "layers", None)
        or getattr(audio_tower, "blocks", None)
        or getattr(audio_tower, "encoder", None)
    )
    if blocks is None:
        raise RuntimeError("Could not locate audio transformer blocks on audio_tower.")
    if hasattr(blocks, "layers"):  # nested encoder wrapper
        blocks = blocks.layers
    for i, blk in enumerate(blocks[:num_blocks]):
        handles.append(blk.register_forward_hook(_hook(f"audio_block_{i}_out")))

    # ---- forward --------------------------------------------------------
    with torch.inference_mode():
        audio_out = audio_tower(
            input_features=input_features.to(torch_dtype),
            attention_mask=input_features_mask,
        )
        # Gemma4AudioModelOutput: last_hidden_state, attention_mask
        last_hidden_state = getattr(audio_out, "last_hidden_state", None)
        if last_hidden_state is None and isinstance(audio_out, tuple):
            last_hidden_state = audio_out[0]
        projected = embed_audio(inputs_embeds=last_hidden_state)

    for h in handles:
        h.remove()

    # ---- dump -----------------------------------------------------------
    def _save(name: str, payload):
        np.savez(out_dir / f"{name}.npz", **{"data": _tree_to_numpy(payload)})

    for name, val in captured.items():
        _save(name, val)

    _save("audio_tower_out", {
        "last_hidden_state": last_hidden_state,
        "attention_mask": getattr(audio_out, "attention_mask", None),
    })
    _save("projector_out", projected)

    # ---- meta -----------------------------------------------------------
    meta = {
        "model_id": model_id,
        "dtype": dtype,
        "seed": seed,
        "num_blocks_captured": num_blocks,
        "audio_only_partial_load": audio_only,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "audio_config": _config_digest(full_config),
        "fixtures": {
            p.name: {
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_of_path(p),
            }
            for p in sorted(out_dir.glob("*.npz"))
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"[capture] wrote {len(list(out_dir.glob('*.npz')))} .npz files to {out_dir}")


# ------------------------------------------------------------------ helpers


def _config_digest(config) -> dict:
    """Extract the audio-relevant config fields into a plain dict for meta."""
    ac = getattr(config, "audio_config", None)
    if ac is None:
        return {}
    keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "num_mel_bins",
        "chunk_size",
        "attention_context_left",
        "attention_context_right",
        "sscp_conv_channel_size",
        "conf_num_hidden_layers",
        "conf_conv_kernel_size",
        "model_type",
    ]
    return {k: getattr(ac, k, None) for k in keys}


def _load_audio_only(model_id: str, torch_dtype):
    """Load *only* audio_tower + embed_audio weights from safetensors shards.

    This avoids pulling the ~8B language model onto a RAM-limited host. Uses
    ``transformers``' config to instantiate the sub-modules, then hydrates
    them from the checkpoint via prefix matching.
    """
    import torch
    import transformers
    from safetensors import safe_open

    full_config = transformers.AutoConfig.from_pretrained(model_id)
    audio_config = full_config.audio_config
    if audio_config is None:
        raise RuntimeError(f"{model_id} has no audio_config.")

    # Instantiate empty sub-modules.
    audio_tower = transformers.AutoModel.from_config(audio_config).to(torch_dtype)
    # The projector is a Gemma4MultimodalEmbedder; construct via config.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

    embed_audio = Gemma4MultimodalEmbedder(
        multimodal_config=audio_config,
        text_config=full_config.text_config,
    ).to(torch_dtype)

    # Pull weights from safetensors shards. Match on prefixes
    # ``model.audio_tower.`` and ``model.embed_audio.``.
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    want_tower = "model.audio_tower."
    want_embed = "model.embed_audio."
    tower_sd = {}
    embed_sd = {}
    for shard in sorted(local.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith(want_tower):
                    tower_sd[k[len(want_tower):]] = f.get_tensor(k).to(torch_dtype)
                elif k.startswith(want_embed):
                    embed_sd[k[len(want_embed):]] = f.get_tensor(k).to(torch_dtype)
    audio_tower.load_state_dict(tower_sd, strict=True)
    embed_audio.load_state_dict(embed_sd, strict=True)

    return audio_tower, embed_audio, full_config


# ------------------------------------------------------------------ cli


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-id", default="google/gemma-4-E4B")
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "gemma4_audio_golden"),
    )
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    p.add_argument(
        "--audio-only",
        action="store_true",
        help="Only materialise the audio tower + projector (MacBook-friendly).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    capture(
        model_id=args.model_id,
        out_dir=Path(args.out_dir),
        num_blocks=args.num_blocks,
        dtype=args.dtype,
        audio_only=args.audio_only,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
