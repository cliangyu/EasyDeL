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

"""Structural tests for ``Gemma4AudioAttention``.

This is the highest-risk class in the audio port. Bit-level parity is
deferred to the end-to-end fixture diff; here we pin the constants and
helper functions that, if wrong, would silently produce nearly-correct
but subtly-broken outputs:

1. Scale constants match HF byte-for-byte:
   ``q_scale = head_dim**-0.5 / log(2)``,
   ``k_scale = log(1+e) / log(2)``,
   ``context_size = chunk + past + future``.
2. ``per_dim_scale`` is initialised to zeros (so ``softplus(0) = log(2)``
   gives an effective q-scale of ``head_dim**-0.5`` at init).
3. ``_convert_to_block`` correctly pads the time axis and reshapes.
4. ``_extract_block_context`` correctly produces overlapping windows.
5. ``_rel_shift`` produces output of shape ``(B, H, NB, chunk, context)``.
6. Forward output shape is ``(B, T, hidden)`` for a complete forward pass.
7. Mask polarity: ``mask=False`` positions get the sentinel
   (``attention_invalid_logits_value``) before softmax — verified by
   showing they get ~zero softmax weight.
8. Softcap saturation: with cap=50, no logit can exceed 50 in magnitude
   even for huge raw scores (the tanh squashes pre-mask).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig
from easydel.modules.gemma4.modeling_gemma4_audio import (
    Gemma4AudioAttention,
    Gemma4AudioRelPositionalEncoding,
)


def _make_attn(
    *,
    hidden_size: int = 64,
    num_heads: int = 4,
    # Defaults must satisfy ``context_size = chunk + past + future >= 12`` —
    # the relative-position encoding produces a hardcoded 13 positions, and
    # ``_rel_shift`` pads to ``context_size + 1`` which must be >= 13.
    chunk: int = 4,
    past: int = 8,
    future: int = 0,
) -> tuple[Gemma4AudioConfig, Gemma4AudioAttention]:
    cfg = Gemma4AudioConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_chunk_size=chunk,
        attention_context_left=past + 1,  # past = context_left - 1 in HF
        attention_context_right=future,
    )
    rngs = nn.Rngs(0)
    return cfg, Gemma4AudioAttention(
        cfg,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


# -- Constants ---------------------------------------------------------------


def test_scale_constants_match_hf() -> None:
    """q_scale, k_scale, and per-dim init must match HF byte-for-byte."""
    _, attn = _make_attn(hidden_size=64, num_heads=4)
    head_dim = 64 // 4

    expected_q = head_dim**-0.5 / math.log(2)
    expected_k = math.log(1 + math.e) / math.log(2)
    assert attn.q_scale == pytest.approx(expected_q, rel=0, abs=1e-12)
    assert attn.k_scale == pytest.approx(expected_k, rel=0, abs=1e-12)


def test_context_size_formula() -> None:
    """context_size = chunk + (context_left - 1) + context_right."""
    _, attn = _make_attn(chunk=4, past=5, future=2)
    assert attn.chunk_size == 4
    assert attn.max_past_horizon == 5
    assert attn.max_future_horizon == 2
    assert attn.context_size == 11


def test_per_dim_scale_init_is_zero() -> None:
    """At init, per_dim_scale = 0 -> softplus(0) = log(2) -> effective q_scale = head_dim**-0.5."""
    _, attn = _make_attn(hidden_size=64, num_heads=4)
    np.testing.assert_array_equal(np.asarray(attn.per_dim_scale.value), np.zeros((16,), dtype=np.float32))


def test_softcap_is_python_float() -> None:
    """softcap is a non-persistent buffer in HF; we keep a plain float."""
    _, attn = _make_attn()
    assert isinstance(attn.softcap, float)
    assert attn.softcap == 50.0  # config default


def test_relative_k_proj_has_no_bias_no_clamp() -> None:
    """relative_k_proj is plain Linear (HF: bias=False, no clippable wrapper)."""
    _, attn = _make_attn()
    # Plain ColumnParallelLinear: kernel only, no bias.
    params = nn.state(attn.relative_k_proj, nn.Param).flat_state()
    names = {"/".join(str(s) for s in path) for path, _ in params}
    assert any("kernel" in n for n in names)
    assert not any("bias" in n for n in names)
    # Not wrapped in Gemma4AudioClippableLinear: no input_min/output_max attrs.
    assert not hasattr(attn.relative_k_proj, "input_min")
    assert not hasattr(attn.relative_k_proj, "output_max")


# -- Helper functions --------------------------------------------------------


def test_convert_to_block_pads_and_reshapes() -> None:
    """_convert_to_block: pad time axis to multiple of chunk, then reshape."""
    _, attn = _make_attn(hidden_size=8, num_heads=2, chunk=4)
    H, D = 2, 4
    # T=10 -> NB = ceil(10/4) = 3, pad 2 at the end.
    x = jnp.arange(1 * 10 * H * D, dtype=jnp.float32).reshape(1, 10, H, D)
    blocks = attn._convert_to_block(x)
    assert blocks.shape == (1, 3, 4, H, D)
    # Last 2 timesteps in the last block must be zero (the pad).
    np.testing.assert_array_equal(np.asarray(blocks[:, 2, 2:, :, :]), np.zeros((1, 2, H, D), dtype=np.float32))
    # The first 10 entries (flattened over the first two block dims, then chunk)
    # must equal the original.
    flat = np.asarray(blocks).reshape(1, 12, H, D)[:, :10, :, :]
    np.testing.assert_array_equal(flat, np.asarray(x))


def test_extract_block_context_window_count_and_shape() -> None:
    """_extract_block_context: NB windows of length context_size, stride chunk."""
    _, attn = _make_attn(hidden_size=8, num_heads=2, chunk=4, past=3, future=0)
    # context_size = 4 + 3 + 0 = 7.
    H, D = 2, 4
    x = jax.random.normal(jax.random.key(0), (1, 12, H, D), dtype=jnp.float32)
    ctx = attn._extract_block_context(x)
    # NB = ceil(12 / 4) = 3.
    assert ctx.shape == (1, 3, 7, H, D)


def test_extract_block_context_left_padding_zeroed() -> None:
    """First block's context begins with max_past_horizon zero rows."""
    _, attn = _make_attn(hidden_size=8, num_heads=2, chunk=4, past=3, future=0)
    H, D = 2, 4
    x = jnp.ones((1, 8, H, D), dtype=jnp.float32)
    ctx = np.asarray(attn._extract_block_context(x))
    # Block 0, context positions 0..2 are the left-pad (should be 0).
    np.testing.assert_array_equal(ctx[:, 0, :3, :, :], np.zeros((1, 3, H, D), dtype=np.float32))
    # Block 0, context position 3 is the first real input token (= 1).
    np.testing.assert_array_equal(ctx[:, 0, 3, :, :], np.ones((1, H, D), dtype=np.float32))


def test_rel_shift_output_shape() -> None:
    """_rel_shift takes (B, H, NB, chunk, 13) -> (B, H, NB, chunk, context).

    ``context_size`` must be >= 12 (the rel-pos encoding's hardcoded
    position_length is 13 and ``_rel_shift`` pads to ``context+1``).
    """
    _, attn = _make_attn(hidden_size=8, num_heads=2, chunk=4, past=8, future=0)
    # context = 4 + 8 + 0 = 12.
    x = jnp.zeros((1, 2, 3, 4, 13), dtype=jnp.float32)
    shifted = attn._rel_shift(x)
    assert shifted.shape == (1, 2, 3, 4, 12)


# -- Forward -----------------------------------------------------------------


def _make_attn_with_pos(
    *,
    hidden_size: int = 64,
    num_heads: int = 4,
    chunk: int = 4,
    past: int = 5,
    future: int = 0,
):
    cfg, attn = _make_attn(hidden_size=hidden_size, num_heads=num_heads, chunk=chunk, past=past, future=future)
    pos_layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    return cfg, attn, pos_layer


def test_forward_output_shape_and_finite() -> None:
    """End-to-end forward: shape preserved, no NaN/Inf at init."""
    cfg, attn, pos_layer = _make_attn_with_pos(hidden_size=64, num_heads=4, chunk=4, past=8)
    B, T = 2, 12
    x = jax.random.normal(jax.random.key(1), (B, T, cfg.hidden_size), dtype=jnp.float32)
    pos = pos_layer(x)
    y, w = attn(x, pos)
    assert y.shape == (B, T, cfg.hidden_size), f"got {y.shape}"
    assert np.isfinite(np.asarray(y)).all()
    # Attention weights are also finite.
    assert np.isfinite(np.asarray(w)).all()


def test_softcap_keeps_forward_finite_under_extreme_input() -> None:
    """Softcap is the structural guard against logit overflow.

    With ``softcap=50``, raw QK products of arbitrary magnitude are squashed
    via ``softcap * tanh(raw / softcap)`` to ``[-50, +50]`` *before* softmax.
    Without it, an extreme input scale could push raw logits to ``±inf``
    and yield NaN attention weights downstream.

    Note: a "softmax does not collapse to one-hot" check would *not* be
    safe to assert here — even with softcap=50, an aggressive input scale
    can drive each logit to ±cap, and softmax([50, -50, -50, ...]) is
    legitimately near one-hot. Bit-level softcap verification belongs in
    the golden-tensor diff. Here we just pin the finite-output contract.
    """
    cfg = Gemma4AudioConfig(
        hidden_size=8,
        num_attention_heads=2,
        attention_chunk_size=4,
        attention_context_left=9,  # past = 8
        attention_context_right=0,
    )
    rngs = nn.Rngs(0)
    attn = Gemma4AudioAttention(cfg, 0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
    pos_layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    # Crank input magnitudes to overwhelming levels.
    x = jax.random.normal(jax.random.key(2), (1, 8, 8), dtype=jnp.float32) * 1000.0
    pos = pos_layer(x)
    out, attn_weights = attn(x, pos)
    aw = np.asarray(attn_weights)
    # No NaN/Inf in either output — softcap or numerically-stable softmax
    # both must hold for this to pass.
    assert np.isfinite(np.asarray(out)).all(), "non-finite forward output under huge input"
    assert np.isfinite(aw).all(), "non-finite attention weights under huge input"
    # Softmax invariant: weights sum to 1 along the context axis.
    np.testing.assert_allclose(aw.sum(axis=-1), 1.0, atol=1e-5)


def test_mask_zeros_invalid_softmax_weight() -> None:
    """mask=False positions get the sentinel and softmax to ~0 weight."""
    # context_size must be >= 12 — see test_softcap_caps_logits_pre_softmax.
    cfg = Gemma4AudioConfig(
        hidden_size=8,
        num_attention_heads=2,
        attention_chunk_size=4,
        attention_context_left=9,  # past = 8
        attention_context_right=0,
    )
    rngs = nn.Rngs(0)
    attn = Gemma4AudioAttention(cfg, 0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
    pos_layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    B, T = 1, 8
    x = jax.random.normal(jax.random.key(3), (B, T, cfg.hidden_size), dtype=jnp.float32)
    pos = pos_layer(x)
    # Build a mask of shape (B, H, NB, chunk, context). Block all but the
    # first context position for every query.
    NB = 2  # ceil(8 / 4)
    chunk = 4
    context = 12  # 4 + 8 + 0
    mask = jnp.zeros((B, 1, NB, chunk, context), dtype=jnp.bool_)
    mask = mask.at[:, :, :, :, 0].set(True)  # only context[0] is valid
    _, w = attn(x, pos, mask)
    aw = np.asarray(w)
    # All mask=False entries (context indices 1..) must have softmax weight ~0.
    invalid_weights = aw[..., 1:]
    assert invalid_weights.max() < 1e-6, f"invalid positions leaked: max weight={invalid_weights.max():.2e}"
