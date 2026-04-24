# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Sanity-check the Gemma 4 E4B audio-tower golden fixtures.

This is a *shape and finiteness* smoke test. It does not exercise any JAX
code — its job is to fail fast when the capture pipeline regresses (wrong
shapes, NaNs, missing files, HF version bump that silently changed the
numerics). The real parity tests land alongside the JAX port.

The fixtures are regenerated on demand by
``tests/fixtures/capture_gemma4_audio_golden.py``. They are not committed
to git (see the sibling README + ``.gitignore``). If the fixtures are
missing locally, this test is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "gemma4_audio_golden"


# Expected shapes for a deterministic 10 s / 16 kHz waveform through
# google/gemma-4-E4B. Any drift here means the HF reference changed or the
# capture script regressed — either way, the JAX port's oracle is invalid
# until this is re-greened.
EXPECTED = {
    "inputs.npz": {
        "waveform": (160_000,),
        "input_features": (1, 999, 128),
        "input_features_mask": (1, 999),
    },
    "sscp_out.npz": {
        "0": (1, 250, 1024),  # subsampled hidden
        "1": (1, 250),  # subsampled mask
    },
    "audio_block_0_out.npz": {"data": (1, 250, 1024)},
    "audio_block_1_out.npz": {"data": (1, 250, 1024)},
    "audio_block_2_out.npz": {"data": (1, 250, 1024)},
    "audio_tower_out.npz": {
        "last_hidden_state": (1, 250, 1536),
        "attention_mask": (1, 250),
    },
    "projector_out.npz": {"data": (1, 250, 2560)},
}


def _require_fixtures() -> None:
    if not FIXTURE_DIR.exists() or not any(FIXTURE_DIR.glob("*.npz")):
        pytest.skip(
            "Gemma 4 audio golden fixtures missing; run tests/fixtures/capture_gemma4_audio_golden.py to produce them.",
            allow_module_level=False,
        )


@pytest.mark.parametrize(("filename", "fields"), sorted(EXPECTED.items()))
def test_fixture_shapes(filename: str, fields: dict[str, tuple[int, ...]]) -> None:
    _require_fixtures()
    path = FIXTURE_DIR / filename
    assert path.exists(), f"missing fixture: {path}"
    with np.load(path) as z:
        for key, shape in fields.items():
            assert key in z.files, f"{filename} missing key {key!r}; has {z.files}"
            arr = z[key]
            assert arr.shape == shape, f"{filename}:{key} shape {arr.shape} != expected {shape}"
            assert not np.isnan(arr).any(), f"{filename}:{key} contains NaN"
            assert np.isfinite(arr).all(), f"{filename}:{key} contains inf"


def test_meta_json_records_versions_and_hashes() -> None:
    _require_fixtures()
    meta_path = FIXTURE_DIR / "meta.json"
    assert meta_path.exists(), "meta.json missing — capture script did not finish"
    meta = json.loads(meta_path.read_text())

    for required in ("model_id", "transformers_version", "torch_version", "seed", "audio_config", "fixtures"):
        assert required in meta, f"meta.json missing {required!r}"

    # Every captured .npz must have a sha256 recorded; any silent
    # truncation will trip this on a later rerun.
    for name in EXPECTED:
        assert name in meta["fixtures"], f"meta.json missing sha for {name}"
        assert len(meta["fixtures"][name]["sha256"]) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
