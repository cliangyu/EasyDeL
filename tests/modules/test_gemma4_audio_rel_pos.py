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

"""Parity test for ``Gemma4AudioRelPositionalEncoding``.

We don't need the HF oracle installed to cross-check this layer: the forward
pass is ~10 lines of pure arithmetic with no learnable parameters, so a
reference implementation in ``numpy`` is as trustworthy as the HF PyTorch
one. We compute both, diff them, and require float32 bit-for-bit equality
(sin/cos are deterministic in IEEE-754).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig
from easydel.modules.gemma4.modeling_gemma4_audio import Gemma4AudioRelPositionalEncoding


def _numpy_reference(hidden_size: int) -> np.ndarray:
    """Port of HF's forward pass using only numpy — deterministic ground truth."""
    min_timescale = 1.0
    max_timescale = 10_000.0
    num_timescales = hidden_size // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment)
    inv_timescales = inv_timescales[None, None, :]

    position_ids = np.arange(12, -1, -1, dtype=np.float32)[:, None]
    scaled_time = position_ids * inv_timescales  # (1, 13, num_timescales)
    pos_embed = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=-1)
    return pos_embed  # shape (1, 13, hidden_size)


@pytest.mark.parametrize("hidden_size", [1024, 512, 768])
def test_rel_pos_matches_numpy_reference(hidden_size: int) -> None:
    """JAX layer must match the numpy reference at float32 bit precision."""
    cfg = Gemma4AudioConfig(hidden_size=hidden_size)
    layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    dummy = jnp.zeros((1, 1, hidden_size), dtype=jnp.float32)

    ed_out = np.asarray(layer(dummy))
    ref_out = _numpy_reference(hidden_size)

    assert ed_out.shape == (1, 13, hidden_size)
    assert ed_out.shape == ref_out.shape
    # sin/cos are IEEE-deterministic — this should be exact.
    np.testing.assert_allclose(ed_out, ref_out, atol=0.0, rtol=0.0)


def test_rel_pos_output_dtype_follows_input() -> None:
    """Output dtype must be cast to match ``hidden_states`` (HF parity)."""
    cfg = Gemma4AudioConfig()
    layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
        dummy = jnp.zeros((1, 1, cfg.hidden_size), dtype=dtype)
        out = layer(dummy)
        assert out.dtype == dtype


def test_rel_pos_is_deterministic() -> None:
    """No RNG should leak into the forward pass — two calls must be bit-identical."""
    cfg = Gemma4AudioConfig()
    layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    dummy = jnp.zeros((1, 1, cfg.hidden_size), dtype=jnp.float32)

    a = np.asarray(layer(dummy))
    b = np.asarray(layer(dummy))
    np.testing.assert_array_equal(a, b)


def test_rel_pos_has_no_learnable_params() -> None:
    """HF registers ``inv_timescales`` as a non-persistent buffer (not learned).

    Our port stores it as a plain jax.Array attribute rather than an
    ``nnx.Param``, so ``nn.state(layer, nn.Param)`` should be empty.
    """
    cfg = Gemma4AudioConfig()
    layer = Gemma4AudioRelPositionalEncoding(cfg, dtype=jnp.float32)
    params = nn.state(layer, nn.Param)
    assert len(params.flat_state()) == 0
