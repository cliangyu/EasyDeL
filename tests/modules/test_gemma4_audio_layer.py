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

"""Structural tests for ``Gemma4AudioLayer``.

The layer is an assembly of already-tested sub-modules (FFN x2, Attention,
LightConv1d, three RMSNorms). Bit-level parity against HF lives in the
end-to-end fixture diff; here we pin the *wiring* — what would silently
produce wrong outputs even though every sub-module is correct:

1. Output shape preserved: ``(B, T, hidden)`` in -> same out.
2. All four sub-modules exist as attributes (``feed_forward1``,
   ``feed_forward2``, ``self_attn``, ``lconv1d``).
3. The two FFNs are *distinct* instances (independent weights, not aliased).
4. The three RMSNorms are distinct instances (each layer needs three
   separate parameter sets).
5. ``gradient_clipping`` matches the FFN's own pre-computed value (so the
   in-layer clamp magnitude lines up with what the FFNs use internally).
6. Forward output is finite at random init.
7. Attention mask is plumbed through: passing a mask that zeros all
   positions changes the output vs. no-mask, proving the layer forwards
   it to ``self_attn``.
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
    Gemma4AudioAttention,
    Gemma4AudioFeedForward,
    Gemma4AudioLayer,
    Gemma4AudioLightConv1d,
    Gemma4AudioRelPositionalEncoding,
)


def _make_layer(
    *,
    hidden_size: int = 64,
    num_heads: int = 4,
    # context_size = chunk + past + future must be >= 12 (rel-pos encoding
    # has a hardcoded 13 positions; ``_rel_shift`` pads to ``context+1``).
    chunk: int = 4,
    past: int = 8,
    future: int = 0,
):
    cfg = Gemma4AudioConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_chunk_size=chunk,
        attention_context_left=past + 1,
        attention_context_right=future,
    )
    rngs = nn.Rngs(0)
    layer = Gemma4AudioLayer(
        cfg,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )
    pos_layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    return cfg, layer, pos_layer


def test_output_shape_preserved() -> None:
    """(B, T, hidden) in -> (B, T, hidden) out."""
    cfg, layer, pos_layer = _make_layer(hidden_size=64, num_heads=4, chunk=4, past=8)
    B, T = 2, 12
    x = jax.random.normal(jax.random.key(1), (B, T, cfg.hidden_size), dtype=jnp.float32)
    pos = pos_layer(x)
    y = layer(x, pos)
    assert y.shape == (B, T, cfg.hidden_size), f"got {y.shape}"


def test_forward_is_finite() -> None:
    """Random init must not produce NaN/Inf through the full Macaron block."""
    cfg, layer, pos_layer = _make_layer(hidden_size=32, num_heads=4, chunk=4, past=8)
    x = jax.random.normal(jax.random.key(2), (1, 8, cfg.hidden_size), dtype=jnp.float32)
    pos = pos_layer(x)
    y = np.asarray(layer(x, pos))
    assert np.isfinite(y).all()


# -- Wiring contracts --------------------------------------------------------


def test_submodules_registered() -> None:
    """All four sub-modules + three RMSNorms must be attributes."""
    _, layer, _ = _make_layer()
    assert isinstance(layer.feed_forward1, Gemma4AudioFeedForward)
    assert isinstance(layer.feed_forward2, Gemma4AudioFeedForward)
    assert isinstance(layer.self_attn, Gemma4AudioAttention)
    assert isinstance(layer.lconv1d, Gemma4AudioLightConv1d)
    assert hasattr(layer, "norm_pre_attn")
    assert hasattr(layer, "norm_post_attn")
    assert hasattr(layer, "norm_out")


def test_two_ffns_are_distinct_instances() -> None:
    """feed_forward1 and feed_forward2 must have independent parameters."""
    _, layer, _ = _make_layer()
    assert layer.feed_forward1 is not layer.feed_forward2
    # Their first kernel must be a different array object.
    k1 = layer.feed_forward1.ffw_layer_1.linear.kernel.value
    k2 = layer.feed_forward2.ffw_layer_1.linear.kernel.value
    # Different RNG draws => different values (overwhelmingly likely).
    assert not np.array_equal(np.asarray(k1), np.asarray(k2))


def test_three_rms_norms_distinct() -> None:
    """norm_pre_attn, norm_post_attn, norm_out must be three separate modules."""
    _, layer, _ = _make_layer()
    norms = [layer.norm_pre_attn, layer.norm_post_attn, layer.norm_out]
    for i in range(3):
        for j in range(i + 1, 3):
            assert norms[i] is not norms[j], f"norm[{i}] aliased to norm[{j}]"


def test_gradient_clipping_matches_ffn() -> None:
    """Layer's clamp bound must equal the FFN's own clamp bound (HF parity)."""
    _, layer, _ = _make_layer()
    assert layer.gradient_clipping == layer.feed_forward1.gradient_clipping


# -- Attention mask plumbing -------------------------------------------------


def test_mask_is_forwarded_to_attention() -> None:
    """Passing a restrictive mask changes output vs. no-mask, proving plumbing."""
    cfg, layer, pos_layer = _make_layer(hidden_size=32, num_heads=4, chunk=4, past=8)
    B, T = 1, 8
    x = jax.random.normal(jax.random.key(3), (B, T, cfg.hidden_size), dtype=jnp.float32)
    pos = pos_layer(x)

    y_unmasked = np.asarray(layer(x, pos))

    # Build a mask that lets only the first context position through everywhere.
    NB = 2  # ceil(8 / 4)
    chunk = 4
    context = 4 + 8 + 0  # = 12
    mask = jnp.zeros((B, 1, NB, chunk, context), dtype=jnp.bool_)
    mask = mask.at[:, :, :, :, 0].set(True)
    y_masked = np.asarray(layer(x, pos, attention_mask=mask))

    # Outputs must differ — if the mask weren't forwarded, they'd be identical.
    assert not np.allclose(y_unmasked, y_masked, atol=1e-6), (
        "mask had no effect on the layer output — attention_mask is not being forwarded"
    )
