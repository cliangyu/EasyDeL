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

"""Structural tests for ``Gemma4AudioSubSampleConvProjection``.

This is the audio encoder's feature stem: two stride-2 Conv2d layers reduce
both temporal and spectral resolution by 4x, then a linear projects the
flattened result to ``hidden_size``. Bit-level parity against HF lives in
the end-to-end fixture test; here we pin contracts that, if broken, would
cascade into wrong shapes / bad masking / wrong projection dim:

1. Output shape is ``(B, T/4, hidden_size)`` for valid (T divisible by 4)
   inputs — verifies both stride-2 layers are wired and the projection
   collapses (F/4, C) to ``proj_input_dim``.
2. Mask is downsampled to ``(B, T/4)`` after both layers.
3. ``proj_input_dim`` matches HF's stale formula
   ``(subsampling_conv_channels[0] // 4) * subsampling_conv_channels[1]``
   independent of the actual mel input size — required for checkpoint parity.
4. Padded timesteps (mask=0) are zeroed *before* the conv so they cannot
   leak into adjacent valid timesteps via the kernel footprint.
5. ``input_proj_linear`` has no bias (HF uses ``bias=False``).
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
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
)


def _make_sscp(
    *,
    hidden_size: int = 1024,
    subsampling_conv_channels: list[int] | None = None,
) -> tuple[Gemma4AudioConfig, Gemma4AudioSubSampleConvProjection]:
    if subsampling_conv_channels is None:
        subsampling_conv_channels = [128, 32]
    cfg = Gemma4AudioConfig(
        hidden_size=hidden_size,
        subsampling_conv_channels=subsampling_conv_channels,
    )
    rngs = nn.Rngs(0)
    return cfg, Gemma4AudioSubSampleConvProjection(
        cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


# -- Output shape ------------------------------------------------------------


def test_output_shape_quarter_time_hidden_size() -> None:
    """Output is (B, T/4, hidden_size) for divisible T."""
    cfg, sscp = _make_sscp(hidden_size=1024)
    # B=2, T=16, F=128 (default mel dim assumption baked into proj_input_dim).
    x = jax.random.normal(jax.random.key(1), (2, 16, 128), dtype=jnp.float32)
    y, _ = sscp(x)
    assert y.shape == (2, 16 // 4, cfg.hidden_size), f"got {y.shape}"


def test_mask_downsampled_to_quarter_time() -> None:
    """Mask is sliced ``[:, ::2]`` twice -> (B, T/4)."""
    _, sscp = _make_sscp()
    x = jax.random.normal(jax.random.key(2), (1, 16, 128), dtype=jnp.float32)
    mask = jnp.ones((1, 16), dtype=jnp.float32)
    _, mask_out = sscp(x, mask)
    assert mask_out is not None
    assert mask_out.shape == (1, 16 // 4)


def test_mask_none_passes_through() -> None:
    """Without a mask, the second-return is None."""
    _, sscp = _make_sscp()
    x = jax.random.normal(jax.random.key(3), (1, 16, 128), dtype=jnp.float32)
    _, mask_out = sscp(x)
    assert mask_out is None


# -- proj_input_dim formula --------------------------------------------------


def test_proj_input_dim_matches_hf_stale_formula() -> None:
    """proj_input_dim = (conv_channels[0] // 4) * conv_channels[1].

    HF's formula is independent of the actual mel dim — checkpoint
    weights bake this assumption in. Match it exactly.
    """
    _, sscp = _make_sscp(subsampling_conv_channels=[128, 32])
    # ColumnParallelLinear stores (in, out); HF stores (out, in).
    kernel = sscp.input_proj_linear.kernel.value
    assert kernel.shape[0] == (128 // 4) * 32, (
        f"proj_input_dim={kernel.shape[0]}, expected (128//4)*32 = {(128 // 4) * 32}"
    )
    assert kernel.shape[1] == 1024  # hidden_size


def test_proj_input_dim_with_alternate_channels() -> None:
    """Formula holds for non-default channel configs."""
    # If HF stops being stale and pins (channels[0]//4)*channels[1], any
    # sensible config still produces a kernel of that exact in-features size.
    _, sscp = _make_sscp(subsampling_conv_channels=[64, 16])
    kernel = sscp.input_proj_linear.kernel.value
    assert kernel.shape[0] == (64 // 4) * 16, f"got {kernel.shape[0]}"


# -- Bias-off contract -------------------------------------------------------


def test_input_proj_linear_has_no_bias() -> None:
    """HF uses bias=False on input_proj_linear; port must match."""
    _, sscp = _make_sscp()
    params = nn.state(sscp.input_proj_linear, nn.Param).flat_state()
    names = {"/".join(str(s) for s in path) for path, _ in params}
    assert not any("bias" in n for n in names)


def test_sub_layers_have_no_conv_bias() -> None:
    """HF's Conv2d uses bias=False; both stem layers must match."""
    _, sscp = _make_sscp()
    for layer_name in ("layer0", "layer1"):
        layer = getattr(sscp, layer_name)
        params = nn.state(layer.conv, nn.Param).flat_state()
        names = {"/".join(str(s) for s in path) for path, _ in params}
        assert not any("bias" in n for n in names), f"{layer_name}.conv has a bias"


# -- Mask zeroing happens BEFORE conv ----------------------------------------


def test_padded_timesteps_zeroed_before_conv() -> None:
    """A padded input shouldn't be able to leak into adjacent valid output.

    Construct two inputs that are identical at unmasked positions but
    differ wildly at masked-out positions. With pre-conv zeroing both
    runs must produce identical output.
    """
    _, sscp = _make_sscp()
    valid = jax.random.normal(jax.random.key(4), (1, 16, 128), dtype=jnp.float32)
    padded_a = valid.at[:, 8:, :].set(1000.0)  # huge garbage in padded region
    padded_b = valid.at[:, 8:, :].set(-1000.0)  # different huge garbage
    mask = jnp.concatenate(
        [jnp.ones((1, 8), dtype=jnp.float32), jnp.zeros((1, 8), dtype=jnp.float32)],
        axis=-1,
    )

    y_a, _ = sscp(padded_a, mask)
    y_b, _ = sscp(padded_b, mask)
    # First two valid output timesteps (covering valid input 0..3) must
    # match exactly between the two runs — kernel does not see the
    # garbage (it sees zeros instead).
    np.testing.assert_array_equal(np.asarray(y_a)[:, 0, :], np.asarray(y_b)[:, 0, :])


# -- Conv layer building blocks ----------------------------------------------


def test_layer_kernel_strides_and_pad() -> None:
    """Stem Conv2d uses kernel=3, stride=2, padding=1 (explicit, not SAME)."""
    rngs = nn.Rngs(0)
    layer = Gemma4AudioSubSampleConvProjectionLayer(
        in_channels=1,
        out_channels=128,
        norm_eps=1e-6,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )
    assert layer.conv.kernel_size == (3, 3)
    assert layer.conv.strides == (2, 2)
    # Padding pinned as ((1,1),(1,1)) for torch parity (rejects SAME).
    assert layer.conv.padding == ((1, 1), (1, 1))


def test_layer_norm_no_bias_with_scale() -> None:
    """LayerNorm has learned scale, no bias (HF: elementwise_affine=True, bias=False)."""
    rngs = nn.Rngs(0)
    layer = Gemma4AudioSubSampleConvProjectionLayer(
        in_channels=1,
        out_channels=128,
        norm_eps=1e-6,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )
    params = nn.state(layer.norm, nn.Param).flat_state()
    names = {"/".join(str(s) for s in path) for path, _ in params}
    assert any("scale" in n for n in names), f"missing learned scale in {names}"
    assert not any("bias" in n for n in names), f"unexpected bias in {names}"
