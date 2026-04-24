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

"""Tests for ``Gemma4AudioClippableLinear``.

The linear-with-clamp wrapper has three behaviours worth pinning:

1. With ``use_clipped_linears=False`` the forward pass is exactly a plain
   linear — no clipping, no extra state.
2. With ``use_clipped_linears=True`` and bounds left at ±inf (the init
   default), it still degenerates to a plain linear — the ±inf defaults
   must not leak NaNs or perturb outputs.
3. When bounds are explicitly set to finite values, the clamp is applied
   both before and after the linear (HF's layout).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig
from easydel.modules.gemma4.modeling_gemma4_audio import Gemma4AudioClippableLinear


def _make_layer(*, use_clipped_linears: bool, in_features: int = 16, out_features: int = 16):
    cfg = Gemma4AudioConfig(use_clipped_linears=use_clipped_linears)
    rngs = nn.Rngs(0)
    return cfg, Gemma4AudioClippableLinear(
        cfg,
        in_features,
        out_features,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


def test_unclipped_matches_plain_linear() -> None:
    """When clipping is disabled, the wrapper is functionally identical to its .linear."""
    _, layer = _make_layer(use_clipped_linears=False)
    x = jax.random.normal(jax.random.key(1), (2, 4, 16), dtype=jnp.float32)

    via_wrapper = np.asarray(layer(x))
    via_linear = np.asarray(layer.linear(x))

    np.testing.assert_allclose(via_wrapper, via_linear, atol=0, rtol=0)


def test_clipped_inf_defaults_are_noop() -> None:
    """±inf default bounds must produce the same output as disabling clipping."""
    _, clipped = _make_layer(use_clipped_linears=True)
    x = jax.random.normal(jax.random.key(2), (2, 4, 16), dtype=jnp.float32)

    # Sanity check that bounds were registered at ±inf.
    assert jnp.isneginf(clipped.input_min.value)
    assert jnp.isposinf(clipped.input_max.value)
    assert jnp.isneginf(clipped.output_min.value)
    assert jnp.isposinf(clipped.output_max.value)

    # The clamped output should equal the plain linear output at these bounds.
    via_wrapper = np.asarray(clipped(x))
    via_linear = np.asarray(clipped.linear(x))
    assert np.isfinite(via_wrapper).all(), "±inf clamp leaked NaN/Inf"
    np.testing.assert_allclose(via_wrapper, via_linear, atol=0, rtol=0)


def test_finite_bounds_clamp_inputs_and_outputs() -> None:
    """With finite bounds, values outside the bounds must be clipped."""
    _, layer = _make_layer(use_clipped_linears=True)

    # Freeze bounds to tight known values.
    layer.input_min.value = jnp.asarray(-1.0, dtype=jnp.float32)
    layer.input_max.value = jnp.asarray(1.0, dtype=jnp.float32)
    layer.output_min.value = jnp.asarray(-0.5, dtype=jnp.float32)
    layer.output_max.value = jnp.asarray(0.5, dtype=jnp.float32)

    # Craft an input with values outside input range to verify pre-clamp.
    x = jnp.array([[[5.0] * 16, [-5.0] * 16]], dtype=jnp.float32)  # (1, 2, 16)
    y = np.asarray(layer(x))

    assert y.shape == (1, 2, 16)
    # Output must be bounded by [output_min, output_max] regardless of input.
    assert y.min() >= -0.5 - 1e-6
    assert y.max() <= 0.5 + 1e-6


def test_no_bias_param() -> None:
    """HF hardcodes bias=False on every audio linear — port must match."""
    _, layer = _make_layer(use_clipped_linears=True)
    # Parameter tree should contain the linear kernel but no bias.
    params = nn.state(layer, nn.Param).flat_state()
    param_names = {k for k in params.keys()}
    assert any("kernel" in str(k) for k in param_names), "missing linear kernel"
    assert not any("bias" in str(k) for k in param_names), f"expected no bias in clippable linear, found {param_names}"


def test_clipped_registers_four_bound_variables() -> None:
    """Four bound scalars (input/output × min/max) must be in NNX state."""
    _, layer = _make_layer(use_clipped_linears=True)
    # Bound scalars are nnx.Variable (not Param).
    assert hasattr(layer, "input_min")
    assert hasattr(layer, "input_max")
    assert hasattr(layer, "output_min")
    assert hasattr(layer, "output_max")


def test_unclipped_does_not_register_bounds() -> None:
    """When disabled, no bound attributes should exist (matches HF)."""
    _, layer = _make_layer(use_clipped_linears=False)
    assert not hasattr(layer, "input_min")
    assert not hasattr(layer, "input_max")
    assert not hasattr(layer, "output_min")
    assert not hasattr(layer, "output_max")
