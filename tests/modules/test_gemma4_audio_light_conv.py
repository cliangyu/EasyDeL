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

"""Tests for ``Gemma4AudioCausalConv1d`` and ``Gemma4AudioLightConv1d``.

Causality is the critical contract: the value at time ``t`` must depend only
on inputs at times ``<= t``. We pin that explicitly because the JAX port
uses manual left-padding instead of inheriting torch's ``F.pad`` call.

For LightConv1d we verify structural wiring (shape, residual, clamp
propagation, distinct RMSNorms) rather than bit-level parity against HF —
that lives in the end-to-end fixture diff.
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
    Gemma4AudioCausalConv1d,
    Gemma4AudioClippableLinear,
    Gemma4AudioLightConv1d,
)

# -- Causal conv -------------------------------------------------------------


def _make_causal_conv(
    *,
    kernel_size: int = 5,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    channels: int = 8,
    use_bias: bool = False,
) -> Gemma4AudioCausalConv1d:
    rngs = nn.Rngs(0)
    return Gemma4AudioCausalConv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


def test_left_pad_matches_hf_formula() -> None:
    """left_pad = (kernel-1)*dilation + 1 - stride. Pin against HF."""
    for k, s, d, expected in [
        (5, 1, 1, 4),  # default audio config
        (3, 1, 1, 2),
        (5, 2, 1, 3),  # non-trivial stride
        (5, 1, 2, 8),  # dilation > 1
        (1, 1, 1, 0),  # degenerate
    ]:
        layer = _make_causal_conv(kernel_size=k, stride=s, dilation=d)
        assert layer.left_pad == expected, f"k={k} s={s} d={d}: got {layer.left_pad}, want {expected}"


def test_stride1_dilation1_preserves_length() -> None:
    """With left-pad and VALID conv, output length equals input length."""
    layer = _make_causal_conv(kernel_size=5, channels=8)
    x = jax.random.normal(jax.random.key(1), (2, 13, 8), dtype=jnp.float32)
    y = np.asarray(layer(x))
    assert y.shape == (2, 13, 8)


def test_causal_property_strict() -> None:
    """Output at time t must not depend on inputs at times > t.

    Method: run on an input, then perturb a *future* timestep only and
    verify every output up to that timestep is bit-identical. If the conv
    leaked future information, the outputs would diverge.
    """
    layer = _make_causal_conv(kernel_size=5, channels=4)
    x = jax.random.normal(jax.random.key(2), (1, 16, 4), dtype=jnp.float32)
    y1 = np.asarray(layer(x))

    # Perturb only timestep 10.
    perturbation = jax.random.normal(jax.random.key(3), (1, 1, 4), dtype=jnp.float32) * 10.0
    x_perturbed = x.at[:, 10:11, :].add(perturbation)
    y2 = np.asarray(layer(x_perturbed))

    # Every timestep < 10 must be identical (no leak from future).
    np.testing.assert_array_equal(y1[:, :10, :], y2[:, :10, :])
    # Timestep 10 itself is allowed to change (input at t=10 affects output at t=10).
    # Beyond t=10 we also expect changes because kernel reaches backwards.


def test_depthwise_groups_param_shape() -> None:
    """Groups == channels should allocate one kernel slot per channel.

    Checking kernel shape loosely — flax stores kernels as
    ``(kernel_size, in/groups, out)`` so depthwise has ``in/groups = 1``.
    """
    layer = _make_causal_conv(kernel_size=5, channels=16, groups=16)
    kernel_shape = layer.conv.kernel.value.shape
    assert kernel_shape == (5, 1, 16), f"expected (5, 1, 16), got {kernel_shape}"


# -- Light conv --------------------------------------------------------------


def _make_light_conv(*, hidden_size: int = 64) -> tuple[Gemma4AudioConfig, Gemma4AudioLightConv1d]:
    cfg = Gemma4AudioConfig(hidden_size=hidden_size)
    rngs = nn.Rngs(0)
    return cfg, Gemma4AudioLightConv1d(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)


def test_light_conv_preserves_shape() -> None:
    """Output shape == input shape (full residual, no downsampling)."""
    _, block = _make_light_conv(hidden_size=64)
    x = jax.random.normal(jax.random.key(4), (2, 8, 64), dtype=jnp.float32)
    y = np.asarray(block(x))
    assert y.shape == (2, 8, 64)


def test_light_conv_is_finite() -> None:
    """Fresh (+/- inf clamp) weights must not leak NaN/Inf through GLU + conv + norm."""
    _, block = _make_light_conv(hidden_size=32)
    x = jax.random.normal(jax.random.key(5), (1, 4, 32), dtype=jnp.float32)
    y = np.asarray(block(x))
    assert np.isfinite(y).all()


def test_light_conv_uses_clippable_linears() -> None:
    """linear_start and linear_end must be Gemma4AudioClippableLinear."""
    _, block = _make_light_conv()
    assert isinstance(block.linear_start, Gemma4AudioClippableLinear)
    assert isinstance(block.linear_end, Gemma4AudioClippableLinear)


def test_light_conv_has_depthwise_conv() -> None:
    """depthwise_conv1d must be a Gemma4AudioCausalConv1d with groups=hidden_size."""
    _, block = _make_light_conv(hidden_size=64)
    assert isinstance(block.depthwise_conv1d, Gemma4AudioCausalConv1d)
    # Groups == hidden_size implies kernel shape (K, 1, hidden_size).
    kernel_shape = block.depthwise_conv1d.conv.kernel.value.shape
    assert kernel_shape == (5, 1, 64), f"expected (5, 1, 64), got {kernel_shape}"


def test_light_conv_distinct_rms_norms() -> None:
    """pre_layer_norm and conv_norm must be distinct modules (independent params)."""
    _, block = _make_light_conv()
    assert block.pre_layer_norm is not block.conv_norm


def test_light_conv_residual_wired() -> None:
    """At small init the output stays close to the input (residual path is live)."""
    _, block = _make_light_conv(hidden_size=64)
    x = jax.random.normal(jax.random.key(6), (1, 4, 64), dtype=jnp.float32)
    y = np.asarray(block(x))
    x_np = np.asarray(x)
    rel = np.linalg.norm(y - x_np) / np.linalg.norm(x_np)
    assert rel < 1.0, f"residual path may be broken (rel diff = {rel:.3f})"
