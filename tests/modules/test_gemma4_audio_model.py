# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structural tests for ``Gemma4AudioModel``.

The model is the top-level audio tower assembly. Three contracts to pin
without an HF oracle (golden-tensor diffs cover bit-level parity):

1. End-to-end shape: ``(B, T_mel, F_mel)`` -> ``(B, T_mel/4, output_proj_dims)``.
2. Output mask is downsampled 4x and is bool.
3. The ``_build_chunked_5d_mask`` helper produces the documented shape and
   correctly enforces the sliding window — keys far outside the window
   are False even when both q and k are within the padding mask.
4. Layers list has exactly ``num_hidden_layers`` Gemma4AudioLayer instances.
5. ``output_proj`` has bias=True (the only audio-tower linear that does).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig
from easydel.modules.gemma4.modeling_gemma4_audio import (
    Gemma4AudioLayer,
    Gemma4AudioModel,
)


def _make_model(
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    chunk: int = 4,
    past: int = 8,
    future: int = 0,
    output_proj_dims: int = 96,
    subsampling_conv_channels: tuple[int, int] = (32, 8),
):
    cfg = Gemma4AudioConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        attention_chunk_size=chunk,
        attention_context_left=past + 1,
        attention_context_right=future,
        output_proj_dims=output_proj_dims,
        subsampling_conv_channels=list(subsampling_conv_channels),
    )
    rngs = nn.Rngs(0)
    return cfg, Gemma4AudioModel(
        cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


# -- Output shape contracts --------------------------------------------------


def test_output_shape_quarter_time_proj_dims() -> None:
    """(B, T_mel, F_mel) -> (B, T_mel/4, output_proj_dims)."""
    cfg, model = _make_model(hidden_size=64, output_proj_dims=96)
    # F_mel must equal subsampling_conv_channels[0] (HF's stale formula).
    B, T_mel = 2, 16
    F_mel = cfg.subsampling_conv_channels[0]
    x = jax.random.normal(jax.random.key(1), (B, T_mel, F_mel), dtype=jnp.float32)
    out, mask = model(x)
    assert out.shape == (B, T_mel // 4, cfg.output_proj_dims), f"got {out.shape}"
    assert mask is not None
    assert mask.shape == (B, T_mel // 4)
    assert mask.dtype == jnp.bool_


def test_forward_is_finite() -> None:
    """At random init, full forward must not produce NaN/Inf."""
    cfg, model = _make_model(hidden_size=32, num_layers=2)
    F_mel = cfg.subsampling_conv_channels[0]
    x = jax.random.normal(jax.random.key(2), (1, 16, F_mel), dtype=jnp.float32)
    out, _ = model(x)
    assert np.isfinite(np.asarray(out)).all()


def test_no_input_mask_passes_through() -> None:
    """Without an input mask, the model still runs and returns a True-everywhere mask."""
    cfg, model = _make_model()
    F_mel = cfg.subsampling_conv_channels[0]
    x = jax.random.normal(jax.random.key(3), (1, 16, F_mel), dtype=jnp.float32)
    out, mask = model(x)
    assert out.shape[1] == 16 // 4
    assert mask is not None
    # All True since no input mask was provided.
    assert np.asarray(mask).all()


# -- Wiring contracts --------------------------------------------------------


def test_num_layers_matches_config() -> None:
    """Layers list must contain exactly ``num_hidden_layers`` instances."""
    _, model = _make_model(num_layers=3)
    assert len(model.layers) == 3
    for layer in model.layers:
        assert isinstance(layer, Gemma4AudioLayer)


def test_output_proj_has_bias() -> None:
    """output_proj is the only audio-tower linear with bias=True (HF parity)."""
    _, model = _make_model()
    assert model.output_proj.use_bias is True
    params = nn.state(model.output_proj, nn.Param).flat_state()
    names = {"/".join(str(s) for s in path) for path, _ in params}
    assert any("kernel" in n for n in names)
    assert any("bias" in n for n in names)


def test_subsample_and_rel_pos_registered() -> None:
    """Both stem sub-modules must be attributes."""
    _, model = _make_model()
    assert hasattr(model, "subsample_conv_projection")
    assert hasattr(model, "rel_pos_enc")


# -- Mask construction -------------------------------------------------------


def test_chunked_mask_shape() -> None:
    """_build_chunked_5d_mask produces (B, 1, NB, chunk, context_size)."""
    _cfg, model = _make_model(chunk=4, past=8, future=0)
    B, T = 2, 12  # NB = 3
    output_mask = jnp.ones((B, T), dtype=jnp.bool_)
    m = model._build_chunked_5d_mask(output_mask)
    assert m.shape == (B, 1, 3, 4, 12), f"got {m.shape}"  # context=4+8+0
    assert m.dtype == jnp.bool_


def test_chunked_mask_padding_zeroed() -> None:
    """Positions where the *key* is padded must be False in the chunked mask.

    Construct: T=8, chunk=4, NB=2. Mark only the first 4 timesteps as valid
    (output_mask=[1,1,1,1,0,0,0,0]). Block 0's queries may attend the first
    4 keys; block 1's queries should see *no* valid keys (all 4 of its
    chunked window's real-key entries are padded, and the past-window
    overlap with block 0 is 4 positions which are real).
    """
    _cfg, model = _make_model(chunk=4, past=8, future=0)
    # B=1, T=8 -> 2 blocks of chunk=4.
    output_mask = jnp.array([[True, True, True, True, False, False, False, False]])
    m = np.asarray(model._build_chunked_5d_mask(output_mask))

    # context_size = 12 = 4 (chunk) + 8 (past). Block 0's keys index into:
    #   raw padded seq (length 8) padded left by 8 zeros => len 16.
    #   block 0 (start=0) gathers indices 0..11 of the 16-length array.
    #   Indices 0..7 are left-pad zeros (False); 8..11 are real (and valid
    #   per the input mask: True). So block 0's row should look like
    #   [F]*8 + [T]*4 along context axis.
    # And q_idx must also be valid: for block 0 (chunk_q 0..3), q_idx=0..3
    # are all True (valid).
    block_0 = m[0, 0, 0]  # (chunk=4, context=12)
    expected_keys = np.array([False] * 8 + [True] * 4)
    for q in range(4):
        # The mask AND-s the sliding window with the padding mask. Sliding
        # window allows q-k in [0, past-1] = [0, 7], so for q=0 only k=0
        # (relative dist=0) is in window; rel-pos within block-0's gathered
        # slice maps offset o to k_index = o (since past pad is left-padded
        # and offsets start at 0). So actually testing this requires
        # accounting for the window. We just assert: any True position in
        # block_0 must be where the input mask is True.
        true_offsets = np.where(block_0[q])[0]
        assert all(expected_keys[o] for o in true_offsets), (
            f"block 0 q={q}: True at offsets {true_offsets}, but mask says only {expected_keys}"
        )

    # Block 1's queries (q_idx 4..7) are all padded. Padding is bidirectional
    # (q & k both must be valid), so all entries of block_1 must be False.
    block_1 = m[0, 0, 1]
    assert not block_1.any(), f"block 1 should be all False (padded queries), got\n{block_1}"


def test_chunked_mask_window_excludes_far_keys() -> None:
    """Sliding window must exclude keys outside [-(future-1), past-1] in q-k."""
    _, model = _make_model(chunk=4, past=2, future=0)  # past=2: only q-k in {0,1}
    # context = 4 + 2 + 0 = 6. _build_chunked_5d_mask doesn't go through
    # _rel_shift, so the < 12 lower bound on context_size doesn't apply here.
    B, T = 1, 4  # one block
    output_mask = jnp.ones((B, T), dtype=jnp.bool_)
    m = np.asarray(model._build_chunked_5d_mask(output_mask))
    # Block 0 gathers offsets 0..5 from a left-padded array of length 4+2=6.
    # That maps to absolute key indices [-2, -1, 0, 1, 2, 3].
    # For q=0: window allows k in [0-(past-1), 0] = [-1, 0] (q-k in [0, 1]).
    #   So valid (post-window) keys are abs k = 0 (offset 2) and k = -1
    #   (offset 1, but that's outside the array -> padded False).
    #   So q=0 row should have True only at offset 2.
    block_0 = m[0, 0, 0]
    # q=0 -> True only at offset where abs_k = 0 (offset 2).
    np.testing.assert_array_equal(block_0[0], np.array([False, False, True, False, False, False]))
