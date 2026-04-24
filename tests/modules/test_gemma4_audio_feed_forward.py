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

"""Structural tests for ``Gemma4AudioFeedForward``.

Golden-tensor parity against HF lives in the end-to-end fixture test; here we
only validate pieces we can verify *without* the HF oracle:

1. Output shape matches input shape.
2. Macaron residual: forward output is within the residual band (i.e. the
   FFN contribution is bounded; verifies residual add is wired).
3. ``post_layer_scale`` equals ``config.residual_weight`` (0.5).
4. ``gradient_clipping`` is floored by ``finfo(param_dtype).max``: bf16
   leaves the 1e10 default intact; fp16 clamps it to 65504.
5. With fresh (small-init) weights and clamp bounds at ±inf, the output is
   finite and non-NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig
from easydel.modules.gemma4.modeling_gemma4_audio import Gemma4AudioFeedForward


def _make_ffn(*, param_dtype=jnp.float32, hidden_size: int = 64):
    from flax import nnx as nn  # local import keeps top-level import cheap

    cfg = Gemma4AudioConfig(hidden_size=hidden_size)
    rngs = nn.Rngs(0)
    return cfg, Gemma4AudioFeedForward(
        cfg,
        dtype=jnp.float32,
        param_dtype=param_dtype,
        rngs=rngs,
    )


def test_output_shape_preserved() -> None:
    """FFN preserves (batch, seq, hidden) shape."""
    _, ffn = _make_ffn(hidden_size=64)
    x = jax.random.normal(jax.random.key(1), (2, 8, 64), dtype=jnp.float32)
    y = np.asarray(ffn(x))
    assert y.shape == (2, 8, 64)


def test_post_layer_scale_matches_residual_weight() -> None:
    """The Macaron half-step must read from config.residual_weight (0.5)."""
    cfg, ffn = _make_ffn()
    assert ffn.post_layer_scale == cfg.residual_weight == 0.5


def test_gradient_clipping_floored_by_param_dtype_max() -> None:
    """bf16 keeps 1e10; fp16 clamps to 65504 (finfo max)."""
    # bf16: finfo.max ~= 3.39e38 > 1e10, so no change.
    _, ffn_bf16 = _make_ffn(param_dtype=jnp.bfloat16)
    assert ffn_bf16.gradient_clipping == 1e10

    # fp16: finfo.max = 65504 < 1e10, so clamped down.
    _, ffn_f16 = _make_ffn(param_dtype=jnp.float16)
    assert ffn_f16.gradient_clipping == pytest.approx(65504.0, rel=0, abs=0)


def test_forward_is_finite_with_default_bounds() -> None:
    """±inf clamp bounds must not leak NaN/Inf through the FFN."""
    _, ffn = _make_ffn(hidden_size=32)
    x = jax.random.normal(jax.random.key(2), (1, 4, 32), dtype=jnp.float32)
    y = np.asarray(ffn(x))
    assert np.isfinite(y).all()


def test_residual_pathway_dominates_at_small_init() -> None:
    """With initializer_range=0.02 the FFN contribution is tiny; output ≈ input.

    This checks that the residual is actually added (if it weren't, the
    output at near-zero weights would be dominated by the scaled RMSNorm of
    the near-zero ffw_layer_2 output, which has very different magnitude
    than the input).
    """
    _, ffn = _make_ffn(hidden_size=64)
    x = jax.random.normal(jax.random.key(3), (1, 4, 64), dtype=jnp.float32)
    y = np.asarray(ffn(x))
    x_np = np.asarray(x)

    # The residual must pass through — output should be close to input (not
    # identical, because the small FFN contribution is non-zero).
    diff = np.linalg.norm(y - x_np) / np.linalg.norm(x_np)
    # Diff dominated by post_layer_scale * norm(ffw_layer_2(silu(ffw_layer_1(norm(x)))))
    # With initializer_range=0.02 and post_layer_scale=0.5, this is small.
    assert diff < 1.0, f"FFN contribution swamps residual (diff={diff:.3f})"


def test_two_ffw_layers_are_clippable_linears() -> None:
    """Both ffw layers must be Gemma4AudioClippableLinear (enables HF weight mapping)."""
    from easydel.modules.gemma4.modeling_gemma4_audio import Gemma4AudioClippableLinear

    _, ffn = _make_ffn()
    assert isinstance(ffn.ffw_layer_1, Gemma4AudioClippableLinear)
    assert isinstance(ffn.ffw_layer_2, Gemma4AudioClippableLinear)


def test_pre_and_post_layer_norms_registered() -> None:
    """Both RMSNorms must be registered (HF has them as distinct modules)."""
    _, ffn = _make_ffn()
    assert hasattr(ffn, "pre_layer_norm")
    assert hasattr(ffn, "post_layer_norm")
    # They must be distinct instances (not aliased).
    assert ffn.pre_layer_norm is not ffn.post_layer_norm
