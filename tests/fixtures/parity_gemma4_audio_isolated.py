# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

"""Per-block isolated parity for Gemma 4 audio tower.

Where ``parity_gemma4_audio.py`` runs the tower end-to-end (so each block
sees the previous block's *JAX* output), this script isolates each block by
feeding the golden previous-block output. That separates three kinds of
divergence:

* **block-internal bug** — block N's output differs from golden when fed the
  golden block N-1 input.
* **accumulated drift** — block N matches golden under isolation, but the
  end-to-end run diverges because errors compound through the stack.
* **earlier-block bug** — both the end-to-end and isolated tests fail at
  block N, but the smallest failing block under isolation is the actual
  source.

The script reuses the same checkpoint loader and config inference as the
end-to-end harness; only the trace differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
from flax import nnx as nn

from tests.fixtures.parity_gemma4_audio import (
    AudioParityRoot,
    DiffRow,
    _audio_config_from_golden,
    _infer_text_hidden_size,
    _load_audio_weights,
    _load_inputs,
    _print_diff_table,
    _read_npz_array,
)


# Goldens were captured in bf16 from HF reference, while this runner forces
# fp32 to surface true port bugs. The achievable per-block error floor is
# therefore bf16 quantization noise of activations with magnitude up to ~50,
# i.e. max_abs ~0.2 / mean_abs ~0.005. We gate on mean_abs because it is far
# more stable than max_abs (which is dominated by a handful of outliers in
# tails of large activation values).
MEAN_ABS_GATE = 2.0e-2
MAX_ABS_REPORT_GATE = 5.0e-1


def _golden_block_input(golden_dir: Path, idx: int) -> np.ndarray:
    """Block N's input is block N-1's output, or sscp_out for N=0."""
    if idx == 0:
        return _read_npz_array(golden_dir / "sscp_out.npz", ("0", "data", "last_hidden_state"))
    return _read_npz_array(
        golden_dir / f"audio_block_{idx - 1}_out.npz",
        ("data", "0", "last_hidden_state"),
    )


def _golden_block_output(golden_dir: Path, idx: int) -> np.ndarray:
    return _read_npz_array(
        golden_dir / f"audio_block_{idx}_out.npz",
        ("data", "0", "last_hidden_state"),
    )


def _diff_row(name: str, actual: np.ndarray, expected: np.ndarray) -> DiffRow:
    if actual.shape != expected.shape:
        return DiffRow(name, f"{actual.shape}!={expected.shape}", float("nan"), float("nan"), float("nan"), "shape_mismatch")
    actual_f = actual.astype(np.float32, copy=False)
    expected_f = expected.astype(np.float32, copy=False)
    diff = np.abs(actual_f - expected_f)
    denom = np.maximum(np.abs(expected_f), np.float32(1.0e-12))
    rel = diff / denom
    finite_ok = bool(np.isfinite(actual_f).all() and np.isfinite(expected_f).all())
    return DiffRow(
        layer_name=name,
        shape=str(tuple(actual.shape)),
        max_abs=float(np.max(diff)) if diff.size else 0.0,
        max_rel=float(np.max(rel)) if rel.size else 0.0,
        mean_abs=float(np.mean(diff)) if diff.size else 0.0,
        finite_check="ok" if finite_ok else "nonfinite",
    )


def _run_isolated_trace(
    root: AudioParityRoot,
    input_features: np.ndarray,
    input_features_mask: np.ndarray,
    golden_dir: Path,
) -> list[DiffRow]:
    tower = root.audio_tower
    features = jnp.asarray(input_features, dtype=jnp.float32)
    feature_mask = jnp.asarray(input_features_mask.astype(np.bool_))

    # SSCP + mask + position embeddings come from JAX (we want to reuse the
    # exact mask/pos-emb the layers will see). We compute them once and freeze.
    hidden_states_jax, output_mask = tower.subsample_conv_projection(features, feature_mask)
    sscp_actual = np.asarray(hidden_states_jax)
    sscp_expected = _read_npz_array(golden_dir / "sscp_out.npz", ("0", "data", "last_hidden_state"))

    if output_mask is None:
        output_mask = jnp.ones(hidden_states_jax.shape[:2], dtype=jnp.bool_)
    else:
        output_mask = output_mask.astype(jnp.bool_)
    position_embeddings = tower.rel_pos_enc(hidden_states_jax)
    attention_mask = tower._build_chunked_5d_mask(output_mask)

    rows: list[DiffRow] = [_diff_row("sscp_out (JAX-computed)", sscp_actual, sscp_expected)]

    num_layers = root.audio_tower.config.num_hidden_layers
    for idx, layer in enumerate(tower.layers):
        # Feed golden previous output, NOT the JAX-computed one.
        golden_in = _golden_block_input(golden_dir, idx)
        block_input = jnp.asarray(golden_in, dtype=jnp.float32)
        block_out = layer(
            block_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        golden_out = _golden_block_output(golden_dir, idx)
        rows.append(_diff_row(f"block_{idx}_isolated", np.asarray(block_out), golden_out))

    # Also report the cumulative reference for compactness.
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--safetensors", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    golden_dir = Path(args.golden_dir).expanduser().resolve()
    safetensors_path = Path(args.safetensors).expanduser().resolve()

    text_hidden_size = _infer_text_hidden_size(golden_dir)
    audio_config = _audio_config_from_golden(golden_dir)
    input_features, input_features_mask = _load_inputs(golden_dir, audio_config)

    root = AudioParityRoot(audio_config, text_hidden_size=text_hidden_size, rngs=nn.Rngs(0))
    records, skipped = _load_audio_weights(root, safetensors_path)
    print(
        f"loaded {len(records)} audio tensors; skipped {skipped} non-audio tensors; "
        f"audio_layers={audio_config.num_hidden_layers}"
    )

    rows = _run_isolated_trace(root, input_features, input_features_mask, golden_dir)
    _print_diff_table(rows)

    failures = [
        f"{row.layer_name}: mean_abs={row.mean_abs:.3e} > gate={MEAN_ABS_GATE:.0e}"
        for row in rows
        if row.finite_check != "ok" or row.mean_abs > MEAN_ABS_GATE
    ]
    warnings = [
        f"{row.layer_name}: max_abs={row.max_abs:.3e} > {MAX_ABS_REPORT_GATE:.0e}"
        for row in rows
        if row.finite_check == "ok" and row.max_abs > MAX_ABS_REPORT_GATE
    ]
    if warnings:
        print("\nWARN (max_abs above informational threshold; not fatal under bf16 reference):")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nIsolated per-block parity passed: mean_abs <= {MEAN_ABS_GATE:.0e} for every block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
