# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

"""Parity test for Bug -1.B: per-row audio scatter must not leak across
rows when batches contain different numbers of valid post-SSCP frames.

The previous implementation flattened across batch and did a single
``cumsum(special_mask.reshape(-1))``. That works only when every row
has the same number of valid features and the same number of
placeholders. Real batches in this pipeline are right-padded with mel
zero-frames per HF's audio processor, so different rows have different
valid counts (``audio_output_mask.sum(-1)``). Under that condition the
flattened cumsum places the k-th valid feature of row 0 into a
placeholder slot of row 1.

The fix uses ``jax.vmap`` over the batch axis with a stable argsort
compaction inside, plus a ``checkify.check`` runtime assertion that the
per-row placeholder count equals the per-row valid count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from easydel.modules.gemma4.modeling_gemma4 import Gemma4Model


def _reference_scatter(
    inputs_embeds: np.ndarray,
    input_ids: np.ndarray,
    features: np.ndarray,
    valid_mask: np.ndarray,
    token_id: int,
) -> np.ndarray:
    """Numpy reference: per-row, gather k-th valid feature into k-th
    placeholder slot."""
    batch, seq, dim = inputs_embeds.shape
    out = inputs_embeds.copy()
    for b in range(batch):
        valid_features = features[b][valid_mask[b].astype(bool)]
        placeholder_positions = np.flatnonzero(input_ids[b] == token_id)
        assert valid_features.shape[0] == placeholder_positions.shape[0], (
            f"row {b} mismatch: valid={valid_features.shape[0]} "
            f"placeholders={placeholder_positions.shape[0]}"
        )
        for k, pos in enumerate(placeholder_positions):
            out[b, pos] = valid_features[k]
    return out


def _make_inputs(
    batch: int,
    seq: int,
    feat_t: int,
    dim: int,
    valid_per_row: list[int],
    placeholders_per_row: list[int],
    token_id: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    inputs_embeds = rng.normal(size=(batch, seq, dim)).astype(np.float32)
    features = rng.normal(size=(batch, feat_t, dim)).astype(np.float32)
    valid_mask = np.zeros((batch, feat_t), dtype=bool)
    input_ids = np.zeros((batch, seq), dtype=np.int32)

    other_token = token_id + 1
    for b in range(batch):
        valid_mask[b, : valid_per_row[b]] = True
        # zero-pad invalid frames as the production pipeline does
        features[b, valid_per_row[b]:] = 0.0
        positions = rng.choice(seq, size=placeholders_per_row[b], replace=False)
        input_ids[b] = other_token
        input_ids[b, positions] = token_id

    return inputs_embeds, input_ids, features, valid_mask


def test_unequal_valid_lengths_no_cross_row_leak():
    """The original bug: row 0 has 3 valid features and 3 placeholders,
    row 1 has 5 valid and 5 placeholders. Old flattened scatter would
    pull row-0 features into row-1 placeholders. The fix must keep each
    row independent."""
    batch, seq, feat_t, dim = 2, 12, 8, 4
    token_id = 99
    inputs_embeds, input_ids, features, valid_mask = _make_inputs(
        batch=batch, seq=seq, feat_t=feat_t, dim=dim,
        valid_per_row=[3, 5], placeholders_per_row=[3, 5],
        token_id=token_id, seed=42,
    )

    expected = _reference_scatter(inputs_embeds, input_ids, features, valid_mask, token_id)
    actual = np.asarray(
        Gemma4Model._scatter_audio_features_at_token(
            inputs_embeds=jnp.asarray(inputs_embeds),
            input_ids=jnp.asarray(input_ids),
            features=jnp.asarray(features),
            valid_mask=jnp.asarray(valid_mask),
            token_id=token_id,
        )
    )

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_equal_lengths_matches_reference():
    """Sanity: when valid counts match across rows, output matches reference."""
    batch, seq, feat_t, dim = 3, 10, 6, 4
    token_id = 7
    inputs_embeds, input_ids, features, valid_mask = _make_inputs(
        batch=batch, seq=seq, feat_t=feat_t, dim=dim,
        valid_per_row=[4, 4, 4], placeholders_per_row=[4, 4, 4],
        token_id=token_id, seed=1,
    )

    expected = _reference_scatter(inputs_embeds, input_ids, features, valid_mask, token_id)
    actual = np.asarray(
        Gemma4Model._scatter_audio_features_at_token(
            inputs_embeds=jnp.asarray(inputs_embeds),
            input_ids=jnp.asarray(input_ids),
            features=jnp.asarray(features),
            valid_mask=jnp.asarray(valid_mask),
            token_id=token_id,
        )
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_runtime_assertion_fires_on_count_mismatch():
    """When a row has more placeholders than valid features, the
    embedded ``checkify.check`` must raise (under a checkify wrapper)
    rather than silently producing zeros / pad-leakage. Production
    callers wrap the model __call__ with ``checkify.checkify`` to opt
    into runtime enforcement; this test exercises that path directly."""
    from jax.experimental import checkify

    batch, seq, feat_t, dim = 2, 8, 6, 4
    token_id = 13
    # Row 1: 5 placeholders but only 3 valid features → must throw.
    inputs_embeds, input_ids, features, valid_mask = _make_inputs(
        batch=batch, seq=seq, feat_t=feat_t, dim=dim,
        valid_per_row=[3, 3], placeholders_per_row=[3, 5],
        token_id=token_id, seed=2,
    )

    checked = checkify.checkify(Gemma4Model._scatter_audio_features_at_token)
    err, _ = checked(
        inputs_embeds=jnp.asarray(inputs_embeds),
        input_ids=jnp.asarray(input_ids),
        features=jnp.asarray(features),
        valid_mask=jnp.asarray(valid_mask),
        token_id=token_id,
    )
    with pytest.raises(Exception) as info:
        err.throw()
    msg = str(info.value).lower()
    assert "placeholder" in msg or "valid" in msg


def test_jit_compatibility_with_checkify_wrapper():
    """Production path is jit'd with a top-level checkify.checkify
    wrapper. Verify that path works (and that bare jit without
    functionalisation correctly *refuses* to stage the embedded
    checkify.check, instead of silently dropping it)."""
    from jax.experimental import checkify

    batch, seq, feat_t, dim = 2, 10, 6, 4
    token_id = 5
    inputs_embeds, input_ids, features, valid_mask = _make_inputs(
        batch=batch, seq=seq, feat_t=feat_t, dim=dim,
        valid_per_row=[4, 2], placeholders_per_row=[4, 2],
        token_id=token_id, seed=3,
    )

    def scatter(ie, ii, ff, vm):
        return Gemma4Model._scatter_audio_features_at_token(
            inputs_embeds=ie, input_ids=ii, features=ff, valid_mask=vm,
            token_id=token_id,
        )

    jitted = jax.jit(checkify.checkify(scatter))
    err, actual = jitted(
        jnp.asarray(inputs_embeds),
        jnp.asarray(input_ids),
        jnp.asarray(features),
        jnp.asarray(valid_mask),
    )
    err.throw()  # equal counts → no error

    expected = _reference_scatter(inputs_embeds, input_ids, features, valid_mask, token_id)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6, rtol=1e-6)

    # Bare jit (no checkify wrap) must fail loudly so the assertion is never
    # silently dropped in production.
    with pytest.raises(ValueError, match="checkify"):
        jax.jit(scatter)(
            jnp.asarray(inputs_embeds),
            jnp.asarray(input_ids),
            jnp.asarray(features),
            jnp.asarray(valid_mask),
        )
