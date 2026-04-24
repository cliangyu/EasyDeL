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

"""Gemma 4 USM-style audio encoder (JAX / Flax NNX).

This module is the JAX port of HuggingFace's ``Gemma4AudioModel`` family of
classes, kept in a dedicated file because the audio tower is a large,
self-contained subsystem (~10 classes) and ``modeling_gemma4.py`` is already
over 3k lines.

Port strategy: match HF arithmetic exactly, diff against golden reference
tensors captured from the PyTorch model (``tests/fixtures/gemma4_audio_golden``).
See ``docs/superpowers/plans/2026-04-24-gemma4-audio-port-risks.md`` for the
enumerated risk landscape this port must avoid.

Classes land in the order specified by the port plan (lowest risk first):

1. :class:`Gemma4AudioRelPositionalEncoding` — sinusoidal Shaw-style relative
   position embedding (pure math, no learnable parameters). **Landed.**
2. :class:`Gemma4AudioClippableLinear` — linear with optional per-layer
   input/output clamp buffers. **Landed.**
3. :class:`Gemma4AudioFeedForward` — Macaron FFN with pre+post RMSNorm and
   residual half-step (``residual_weight=0.5``). **Landed.**
4. :class:`Gemma4AudioCausalConv1d` — left-padded depthwise-capable 1-D
   convolution used inside the light-conv module. **Landed.**
5. :class:`Gemma4AudioLightConv1d` — GLU + depthwise causal conv + residual
   (Macaron companion to the FFN). **Landed.**
6. :class:`Gemma4AudioSubSampleConvProjectionLayer` and
   :class:`Gemma4AudioSubSampleConvProjection` — feature-extractor stem
   that downsamples mel features 4x and projects to ``hidden_size``.
   **Landed.**
7. :class:`Gemma4AudioAttention` — chunked local attention with Shaw-style
   relative position bias, fp32 islands, softplus per-head scale, softcap
   before mask. **Landed.**
8. :class:`Gemma4AudioLayer` — Macaron conformer block: FFN → norm-clamp →
   self-attention → norm-clamp + residual → light-conv → FFN → norm-clamp.
   **Landed.**
9. :class:`Gemma4AudioModel` — full audio tower: SSCP stem → relative-pos
   encoding + chunked-attention mask → ``num_hidden_layers`` conformer
   layers → output projection (1024→1536). **Landed.**
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from eformer.common_types import Replicated
from flax import nnx as nn
from jaxtyping import Array, Float

from easydel.infra.utils import ACT2FN, ArrayParam
from easydel.layers import ColumnParallelLinear
from easydel.layers.norms import LayerNorm

from .gemma4_configuration import Gemma4AudioConfig
from .modeling_gemma4 import Gemma4RMSNorm


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal Shaw-style relative positional encoding.

    Direct JAX port of ``transformers.models.gemma4.modeling_gemma4
    .Gemma4AudioRelPositionalEncoding``. Produces a tensor of shape
    ``[1, 13, hidden_size]`` with concatenated ``[sin(sτ), cos(sτ)]`` layout
    (not interleaved) at fixed integer positions ``[12, 11, …, 1, 0]``.

    Subtle points that must match HF exactly for parity:

    * **Hardcoded position range.** Upstream uses
      ``torch.arange(12, -1, -1)`` — a literal 12, **not** derived from
      ``attention_chunk_size``. This is a stale hardcode in HF (the
      ``context_size`` computed in ``__init__`` is never referenced at
      forward time). We mirror the literal so parity holds, even for
      non-default chunk sizes.
    * **Leading batch dim.** HF registers ``inv_timescales`` with two
      unsqueezes so the runtime product broadcasts to shape
      ``[1, 13, num_timescales]``, and the final ``torch.cat(..., dim=-1)``
      concatenates along the *last* axis, giving
      ``[1, 13, hidden_size]``. Downstream attention relies on this rank-3
      layout.
    * **Concatenated, not interleaved** ``sin`` / ``cos``: the first half of
      the hidden dimension is all sines, the second half all cosines —
      matching Shaw et al. and the HF implementation.
    * **num_timescales = hidden_size // 2.** Odd hidden sizes silently
      round down — we match HF's behaviour rather than raising.
    * **Division-by-zero guard.** ``max(num_timescales - 1, 1)`` protects the
      edge case where ``hidden_size == 2`` (single timescale).
    * ``inv_timescales`` is a non-persistent buffer in HF (not saved to the
      checkpoint). We therefore materialise it as a plain array rather than
      a learnable parameter, avoiding accidental param registration.
    * The function is ``@torch.no_grad`` in HF; JAX handles that implicitly
      because no learnable parameters are involved.
    """

    def __init__(self, config: Gemma4AudioConfig, dtype: jnp.dtype = jnp.bfloat16):
        self.hidden_size = config.hidden_size
        self.dtype = dtype

        min_timescale = 1.0
        max_timescale = 10_000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        # Shape: (1, 1, num_timescales) — matches HF's double-unsqueeze so the
        # runtime broadcast produces (1, 13, num_timescales).
        self.inv_timescales = (
            min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
        )[None, None, :]

    def __call__(self, hidden_states: Float[Array, "batch seq hidden"]) -> Float[Array, "1 13 hidden"]:
        """Return the positional bias tensor that attention blocks mix into logits.

        The output is deterministic in ``config`` (does not depend on
        ``hidden_states`` except for output dtype). ``hidden_states`` is
        accepted for API parity with HF, which reads ``device`` and ``dtype``
        off the input tensor.
        """
        # position_ids: shape (13, 1) — literal 12..0 per HF.
        position_ids = jnp.arange(12, -1, -1, dtype=jnp.float32)[:, None]
        # scaled_time broadcasts (13, 1) * (1, 1, num_timescales) -> (1, 13, num_timescales).
        scaled_time = position_ids * self.inv_timescales
        # Final shape (1, 13, hidden_size) with concatenated sin/cos halves.
        pos_embed = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
        return pos_embed.astype(hidden_states.dtype)


class Gemma4AudioClippableLinear(nn.Module):
    """Linear with optional per-layer input/output activation clamping.

    Direct port of HF's ``Gemma4ClippableLinear``. When
    ``config.use_clipped_linears=True`` (the audio default; contrast with the
    vision tower where it defaults ``False``), the forward pass sandwiches
    the linear between two element-wise clamps:

    .. code-block:: text

        x -> clamp(x, input_min, input_max)
          -> linear(x)                                 # bias=False
          -> clamp(y, output_min, output_max)

    The four clamp bounds are **scalar buffers** in HF — initialised to
    ``±inf`` so that an untrained / freshly-loaded model produces a no-op
    clamp, and overwritten with **trained bounds from the checkpoint** when
    loading the released E4B weights. This matches the USM training
    pipeline's practice of tracking activation percentiles as part of the
    parameter set, analogous to how BatchNorm stores running stats.

    Implementation notes:

    * We use ``ColumnParallelLinear`` matching the style of
      :class:`easydel.modules.gemma4.modeling_gemma4.Gemma4VisionClippableLinear`,
      so the HF kernel name ``*.linear.weight`` maps to our ``*.linear.kernel``
      via the standard EasyDeL weight-conversion path.
    * The four clamp scalars are stored as ``nnx.Variable`` non-Param state,
      not ``ArrayParam``. This keeps them out of the optimizer while still
      being tracked by NNX so HF checkpoint conversion can overwrite them.
      They map to HF buffer names ``input_min`` / ``input_max`` /
      ``output_min`` / ``output_max`` — identical attribute names here.
    * When ``use_clipped_linears=False`` the scalars are not registered at
      all and the forward pass degenerates to a plain linear, matching HF.
    * Biases are always off (HF hardcodes ``bias=False`` for every linear
      in the audio tower).
    """

    def __init__(
        self,
        config: Gemma4AudioConfig,
        in_features: int,
        out_features: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.use_clipped_linears = config.use_clipped_linears

        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.linear = ColumnParallelLinear(
            in_features,
            out_features,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        if self.use_clipped_linears:
            # Scalar buffers, ±inf by default — no-op clamp until a trained
            # checkpoint overwrites them. Kept in param_dtype to match HF.
            neg_inf = jnp.asarray(-jnp.inf, dtype=param_dtype)
            pos_inf = jnp.asarray(jnp.inf, dtype=param_dtype)
            self.input_min = nn.Variable(neg_inf)
            self.input_max = nn.Variable(pos_inf)
            self.output_min = nn.Variable(neg_inf)
            self.output_max = nn.Variable(pos_inf)

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Replicate the (tiny, scalar) clamp buffers across all devices."""
        if not self.use_clipped_linears:
            return {}
        return {
            "input_min": Replicated,
            "input_max": Replicated,
            "output_min": Replicated,
            "output_max": Replicated,
        }

    def __call__(self, hidden_states: Array) -> Array:
        if self.use_clipped_linears:
            hidden_states = jnp.clip(hidden_states, self.input_min.value, self.input_max.value)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = jnp.clip(hidden_states, self.output_min.value, self.output_max.value)
        return hidden_states


class Gemma4AudioFeedForward(nn.Module):
    """Macaron-style feed-forward block for the audio conformer.

    Direct port of HF's ``Gemma4AudioFeedForward``. The structure matches the
    Macaron FFN from the Conformer paper (Gulati et al., 2020): the block
    contributes a *half-step* residual (``post_layer_scale = residual_weight
    = 0.5``) rather than a full residual, because each conformer layer sandwiches
    a Macaron FFN on *either side* of the attention block — two halves
    summing to one full residual pass.

    .. code-block:: text

        residual = x
        x = clamp(x, -G, +G)           # pre-FFN gradient clip
        x = pre_layer_norm(x)
        x = ffw_layer_1(x)             # ClippableLinear, hidden -> 4*hidden
        x = act_fn(x)                  # silu
        x = ffw_layer_2(x)             # ClippableLinear, 4*hidden -> hidden
        x = clamp(x, -G, +G)           # post-FFN gradient clip
        x = post_layer_norm(x)
        x = x * post_layer_scale       # 0.5 (Macaron half-step)
        x = x + residual

    Subtleties that must match HF for parity:

    * **Gradient-clip magnitude** (``G``) is ``min(config.gradient_clipping,
      finfo(kernel_dtype).max)``. The config default is ``1e10`` — well below
      bf16's max (~3.39e38), so bf16 weights leave it untouched. fp16 weights
      would clamp it down to 65504. HF computes this at every forward pass
      from the live kernel dtype; we compute once at init time because
      ``param_dtype`` doesn't change.
    * **Clamps are on the input activations, not the weights.** The name
      ``gradient_clipping`` is a historical artefact from USM training;
      at inference it acts as an activation range guard.
    * **Post-layer-norm is applied to the already-scaled output**, i.e. the
      RMSNorm happens *before* the residual add but *after* the output
      clamp. Getting the order wrong changes the activation distribution.
    * **Scale-by-0.5 is in-place in HF** (``hidden_states *= ...``); we use an
      explicit multiply for JAX functional purity.
    * **Biases always off** (inherited from ``Gemma4AudioClippableLinear``).
    """

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.config = config

        self.ffw_layer_1 = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size * 4,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffw_layer_2 = Gemma4AudioClippableLinear(
            config,
            config.hidden_size * 4,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # HF passes a positional hidden_size but our Gemma4RMSNorm reads it
        # from the config. Pass the audio config through — it exposes both
        # ``hidden_size`` and ``rms_norm_eps`` like the text/vision configs.
        self.pre_layer_norm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.post_layer_norm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.act_fn = ACT2FN[config.hidden_act]

        # Match HF: min(config.gradient_clipping, finfo(kernel_dtype).max).
        # Precompute — param_dtype is fixed after __init__.
        self.gradient_clipping = float(min(config.gradient_clipping, float(jnp.finfo(param_dtype).max)))
        self.post_layer_scale = config.residual_weight

    def __call__(self, hidden_states: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        residual = hidden_states

        hidden_states = jnp.clip(hidden_states, -self.gradient_clipping, self.gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = jnp.clip(hidden_states, -self.gradient_clipping, self.gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states = hidden_states * self.post_layer_scale
        hidden_states = hidden_states + residual

        return hidden_states


class Gemma4AudioCausalConv1d(nn.Module):
    """Left-padded 1-D convolution — the causal variant used in light-conv blocks.

    Direct port of HF's ``Gemma4AudioCausalConv1d``, which subclasses
    ``nn.Conv1d`` and overrides ``forward`` to left-pad the input sequence
    before invoking the parent convolution. The HF version derives
    ``left_pad`` from the dilated kernel size minus the stride so the
    computation works for non-default strides/dilations too:

    .. code-block:: python

        left_pad = (kernel_size - 1) * dilation + 1 - stride

    For the default ``conv_kernel_size=5`` (stride=1, dilation=1) this is
    simply ``kernel - 1 = 4``.

    Layout notes for the JAX port
    -----------------------------
    * HF's convolution operates on ``(N, C, L)`` and wraps the input in
      ``F.pad(x, (left_pad, 0))`` (last-dim pad in PyTorch = L-axis).
    * Our JAX implementation uses ``(B, L, C)`` throughout — the natural
      Flax layout — so we bypass the HF ``transpose(1,2)`` ping-pong in
      :class:`Gemma4AudioLightConv1d`.
    * ``flax.nnx.Conv`` accepts either VALID padding with manual left pad
      (our choice) or ``padding=((left, right),)`` tuples. We pad manually
      with ``jnp.pad`` so the left-pad is stateless and transparent when
      debugging.
    * HF stores the conv kernel as ``(out_channels, in_channels/groups,
      kernel_size)``; ``nnx.Conv`` stores ``(kernel_size,
      in_features/groups, out_features)``. The weight converter will
      transpose at load time — the shape and semantics are equivalent.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        # Matches HF's ``left_pad`` cached property.
        self.left_pad = (kernel_size - 1) * dilation + 1 - stride

        self.conv = nn.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(stride,),
            kernel_dilation=(dilation,),
            feature_group_count=groups,
            padding="VALID",  # Manual left-pad handles causality.
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "batch length channels"]) -> Array:
        # Left-pad the length axis; leave batch and channel axes untouched.
        x = jnp.pad(x, ((0, 0), (self.left_pad, 0), (0, 0)))
        return self.conv(x)


class Gemma4AudioLightConv1d(nn.Module):
    """GLU-gated depthwise causal convolution — conformer light-conv block.

    Direct port of HF's ``Gemma4AudioLightConv1d``. Adds a residual
    around::

        x -> pre_layer_norm(x)
          -> linear_start(x)                  # hidden -> 2*hidden
          -> GLU(x, dim=-1)                   # -> hidden (gated halves)
          -> depthwise_causal_conv1d(x)       # along length axis
          -> clamp(x, -G, +G)
          -> conv_norm(x)
          -> act_fn(x)                        # silu
          -> linear_end(x)                    # hidden -> hidden
          -> x + residual

    Conformer-specific subtleties
    -----------------------------
    * **Gated Linear Unit (GLU) collapses ``2*hidden`` back to ``hidden``**
      by splitting the last dim in half and multiplying by ``sigmoid(second
      half)``. Equivalent to ``a * sigmoid(b)`` — but matching
      PyTorch's ``F.glu(..., dim=-1)`` convention. Implemented inline
      (``a * sigmoid(b)``) rather than via ``jax.nn.glu`` to avoid
      coupling to any particular JAX-version alias.
    * **Depthwise conv** (``groups = hidden_size``) — a per-channel 1-D
      filter over the length axis. HF uses kernel size 5, default for
      Gemma 4 audio.
    * **Single, full residual** — unlike the Macaron FFN there is no 0.5
      weight here. The light-conv block is the full residual.
    * **Clamp + conv_norm is applied *after* the conv and *before* the
      activation**, not symmetrically around the FFN. Order matters for
      activation-bound accuracy.
    * HF transposes ``(B,T,C) -> (B,C,T)`` before the conv and back after,
      because its conv expects channels-first. Our JAX conv already runs
      on channels-last ``(B,L,C)`` — we skip the transpose entirely.
    """

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.config = config

        self.linear_start = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size * 2,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.linear_end = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Depthwise: groups == hidden_size so each channel has its own filter.
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pre_layer_norm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.conv_norm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.act_fn = ACT2FN[config.hidden_act]

        # Same finfo-clamp rationale as Gemma4AudioFeedForward.
        self.gradient_clipping = float(min(config.gradient_clipping, float(jnp.finfo(param_dtype).max)))

    def __call__(self, hidden_states: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        # GLU over last axis: (..., 2H) -> (..., H). Split then multiply
        # first half by sigmoid(second half); matches torch F.glu.
        gate, value = jnp.split(hidden_states, 2, axis=-1)
        hidden_states = gate * jax.nn.sigmoid(value)

        # Depthwise causal conv runs on (B, L, C) directly — no transpose needed.
        hidden_states = self.depthwise_conv1d(hidden_states)

        hidden_states = jnp.clip(hidden_states, -self.gradient_clipping, self.gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    """Stride-2 Conv2d + LayerNorm + ReLU stem block.

    Direct port of HF's ``Gemma4AudioSubSampleConvProjectionLayer``. Two of
    these stack to give 4x temporal/spectral downsampling before the
    conformer body. Mask-aware: padded timesteps in the input are zeroed
    *before* the conv so they cannot leak into adjacent valid timesteps
    via the kernel footprint, then the mask itself is downsampled by
    stride-2 slicing.

    Layout note: HF runs everything in NCHW and shuffles to channels-last
    just for the LayerNorm. We use NHWC throughout (the natural Flax
    layout), so the LayerNorm slot is a no-op transpose-wise.

    Padding parity
    --------------
    HF uses ``padding=1`` (PyTorch symmetric pad). Flax's ``padding="SAME"``
    can pad asymmetrically (e.g. left=0/right=1) when the spatial size
    forces it, which would diverge from PyTorch on odd-length inputs.
    We pin explicit ``((1, 1), (1, 1))`` padding to guarantee bit-level
    parity with HF for any input size.

    LayerNorm parity
    ----------------
    HF: ``nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True,
    bias=False)`` — learned scale, no bias. We pass
    ``use_scale=True, use_bias=False`` to match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.conv = nn.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # explicit symmetric pad to match torch
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # HF: LayerNorm with elementwise_affine=True, bias=False.
        self.norm = LayerNorm(
            num_features=out_channels,
            epsilon=norm_eps,
            use_scale=True,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # HF uses nn.ReLU; jax.nn.relu is the elementwise equivalent.
        self.act = jax.nn.relu

    def __call__(
        self,
        hidden_states: Float[Array, "batch height width channels"],
        mask: Float[Array, "batch height"] | None = None,
    ) -> tuple[Array, Array | None]:
        if mask is not None:
            # Broadcast over W and C so padded H positions get zeroed pre-conv.
            hidden_states = hidden_states * mask[:, :, None, None]

        hidden_states = self.conv(hidden_states)
        # Channels-last NHWC: LayerNorm acts on the last axis directly,
        # no transpose required.
        hidden_states = self.act(self.norm(hidden_states))

        if mask is not None:
            # Match HF stride-2 slicing along the temporal (H) axis.
            mask = mask[:, ::2]

        return hidden_states, mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    """Mel-features -> conformer-input projection stem.

    Direct port of HF's ``Gemma4AudioSubSampleConvProjection``. Two
    stride-2 Conv2d layers give 4x downsampling on each spatial axis,
    then a linear projects the flattened ``(F/4) * subsampling_conv_channels[1]``
    vector to ``hidden_size``.

    Forward (NHWC):

    .. code-block:: text

        (B, T, F)         input mel features
        -> (B, T, F, 1)   add channel dim
        -> layer0(.)      stride-2 Conv2d, in=1, out=conv_channels[0]
        -> (B, T/2, F/2, conv_channels[0])
        -> layer1(.)      stride-2 Conv2d, in=conv_channels[0], out=conv_channels[1]
        -> (B, T/4, F/4, conv_channels[1])
        -> reshape        (B, T/4, F/4 * conv_channels[1])
        -> linear         (B, T/4, hidden_size)

    Subtleties
    ----------
    * **proj_input_dim formula is HF-stale** — it computes
      ``(subsampling_conv_channels[0] // 4) * subsampling_conv_channels[1]``,
      which equals ``F/4 * conv_channels[1]`` *only* when the input mel
      dimension equals ``conv_channels[0]`` (= 128 by default). The
      formula is wrong for any other mel size, but mirroring HF preserves
      checkpoint compatibility — the released E4B weights expect this
      exact ``proj_input_dim``.
    * **No clippable wrapper on ``input_proj_linear``** — HF uses a plain
      ``nn.Linear``, so trained activation bounds do not apply here. We
      use ``ColumnParallelLinear`` directly, matching the kernel-name
      convention (``input_proj_linear.linear.kernel`` ↔ HF's
      ``input_proj_linear.weight``).
    * **Mask is downsampled by stride-2 slicing twice** giving (B, T/4)
      after both layers. Downstream attention reads this final mask.
    * **Channel dim added on the last axis** (NHWC). HF adds it on dim=1
      (NCHW); our equivalent is ``hidden_states[..., None]``.
    """

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.config = config

        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Mirror HF's stale formula exactly; required for checkpoint parity.
        proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.input_proj_linear = ColumnParallelLinear(
            proj_input_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(
        self,
        input_features: Float[Array, "batch time freq"],
        input_features_mask: Float[Array, "batch time"] | None = None,
    ) -> tuple[Float[Array, "batch t4 hidden"], Array | None]:
        # NHWC: add channel as the last axis -> (B, T, F, 1).
        hidden_states = input_features[..., None]
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)

        batch_size, t4, f4, c4 = hidden_states.shape
        # Flatten (F/4, C) -> (F/4 * C); already channels-last so no permute.
        hidden_states = hidden_states.reshape(batch_size, t4, f4 * c4)
        return self.input_proj_linear(hidden_states), mask


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with Shaw-style relative position bias.

    Direct port of HF's ``Gemma4AudioAttention``. This is the most
    arithmetic-sensitive class in the audio tower; bit-level parity with
    HF requires getting all of the following right simultaneously:

    Per-block / chunked structure
    -----------------------------
    Each token attends to a *local context window* of length
    ``chunk_size + max_past_horizon + max_future_horizon`` rather than
    the full sequence. Tokens are grouped into non-overlapping
    ``chunk_size``-length blocks (queries) but each block looks at an
    *overlapping* context window of ``context_size`` keys/values
    (with stride ``chunk_size``). This is implemented via:

    * :meth:`_convert_to_block` — pads sequence to next multiple of
      chunk and reshapes ``(B, T, H, D)`` → ``(B, NB, chunk, H, D)``.
    * :meth:`_extract_block_context` — pads with ``max_past_horizon`` on
      the left and ``max_future_horizon + chunk - 1`` on the right, then
      gathers overlapping windows ``(B, NB, context, H, D)``.

    For audio config defaults: chunk=12, past=12, future=0,
    so context = 12 + 12 + 0 = 24.

    Scaling subtleties (must match HF byte-for-byte)
    ------------------------------------------------
    * ``q_scale = (head_dim**-0.5) / log(2)`` — natural log, *not*
      log2; this is paired with ``F.softplus(per_dim_scale)`` which
      equals ``log(2)`` at init, so the effective initial scale is
      ``head_dim**-0.5``.
    * ``k_scale = log(1 + e) / log(2) ≈ 1.895`` — a constant scalar
      key boost; combined with q_scale at init the effective
      ``q·k`` factor is ``log(1+e) / (sqrt(d) * log(2))``, ~1.895x
      the standard ``1/sqrt(d)``. Don't normalise this away.
    * ``per_dim_scale`` is a learnable head-dim vector initialised to
      zeros (so ``softplus(0) = log(2)`` at start). Trained
      checkpoints will populate it.

    fp32 islands
    ------------
    HF casts q/k/v to ``float32`` immediately after projection and runs
    the full attention computation in fp32. The softmax explicitly uses
    ``dtype=torch.float32`` and the result is cast to ``value_states.dtype``
    (which is fp32 — so the final cast is a no-op). Only the input to
    ``self.post`` is cast back to the kernel's ``param_dtype``. We mirror
    this exactly because softmax in bf16 over wide context windows can
    underflow.

    Softcap-then-mask ordering
    --------------------------
    HF applies the softcap (``softcap * tanh(logits / softcap)``) *before*
    the mask, then masks invalid positions to ``-1e9``. Reversing the
    order would clip ``-1e9`` to ``-50`` and destroy the mask. We pin
    this ordering exactly.

    Mask polarity
    -------------
    HF: ``masked_fill(attention_mask.logical_not(), -1e9)`` — i.e.
    ``mask=True`` means *valid*, the negation flips to invalid which
    gets the sentinel. Our port mirrors this: pass a boolean mask
    where ``True`` = attend, ``False`` = sentinel.

    Relative position bias and ``_rel_shift``
    -----------------------------------------
    The position embeddings (shape ``(1, 13, hidden)`` from
    :class:`Gemma4AudioRelPositionalEncoding`) are projected via
    ``relative_k_proj`` (a *plain* linear, no clamp) and combined with
    queries to produce ``matrix_bd`` of shape ``(B, H, NB, chunk, 13)``.
    :meth:`_rel_shift` then transforms this to ``(B, H, NB, chunk,
    context)`` via the Shaw/Transformer-XL right-pad-then-slice trick
    (Appendix B of arxiv 1901.02860). Implementation must match HF
    *exactly*; we copy the pad/reshape/slice pattern verbatim.

    Layer-name convention
    ---------------------
    * ``q_proj``, ``k_proj``, ``v_proj``, ``post``: clippable linears
      (``self.q_proj.linear.kernel`` ↔ HF's ``q_proj.linear.weight``).
    * ``relative_k_proj``: plain ColumnParallelLinear (HF: plain Linear).
    * ``per_dim_scale``: learnable parameter, scalar buffer-style.
    * ``softcap``: stored as a Python float (HF: non-persistent buffer);
      we don't need it in NNX state.
    """

    def __init__(
        self,
        config: Gemma4AudioConfig,
        layer_idx: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.param_dtype = param_dtype

        # Match HF *exactly* — note natural-log denominators.
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.q_proj = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post = Gemma4AudioClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # HF: nn.Linear(..., bias=False). NOT a clippable linear.
        self.relative_k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        # HF: nn.Parameter(torch.zeros(head_dim)). Learned per-dim scale.
        self.per_dim_scale = ArrayParam.bound(
            shape=(self.head_dim,),
            dtype=param_dtype,
            init_method="zeros",
            key=None,
        )

        # HF stores softcap as a non-persistent buffer; we just keep the
        # Python float — no learnable, no checkpoint slot, no NNX state.
        self.softcap = float(self.attention_logits_soft_cap)

    # -- chunking helpers ----------------------------------------------------

    def _convert_to_block(self, hidden_states: Array) -> Array:
        """Reshape (B, T, H, D) -> (B, NB, chunk, H, D), zero-padding to a
        multiple of chunk_size on the time axis.

        Matches HF's ``F.pad(x, (0, 0, 0, 0, 0, pad))``: pad only the time
        axis with ``pad`` zeros on the right.
        """
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        # Pad spec: ((B), (T_left=0, T_right=pad), (H), (D)).
        hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad), (0, 0), (0, 0)))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim)

    def _extract_block_context(self, hidden_states: Array) -> Array:
        """Build overlapping ``context_size`` windows for each block.

        HF uses ``F.pad`` followed by ``tensor.unfold(1, context, chunk)``;
        we replicate that with explicit gather indices. Pad amounts:
        ``max_past_horizon`` on the left, ``max_future_horizon + chunk - 1``
        on the right of the time axis.

        Output shape: ``(B, NB, context_size, H, D)``.
        """
        seq_len = hidden_states.shape[1]
        # HF pad spec (last-to-first): (D=0,0), (H=0,0), (T=past, future + chunk - 1).
        left = self.max_past_horizon
        right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = jnp.pad(hidden_states, ((0, 0), (left, right), (0, 0), (0, 0)))
        # Number of stride-chunk windows of length context_size that fit.
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        # Gather indices: window b spans [b*chunk, b*chunk + context).
        block_starts = jnp.arange(num_blocks) * self.chunk_size  # (NB,)
        offsets = jnp.arange(self.context_size)  # (context,)
        idx = block_starts[:, None] + offsets[None, :]  # (NB, context)
        # Fancy index over the time axis.
        return hidden_states[:, idx, :, :]  # (B, NB, context, H, D)

    def _rel_shift(self, x: Array) -> Array:
        """Shaw / Transformer-XL relative-position shift for blocked attention.

        Reshapes ``(B, H, NB, chunk, position_length)`` to
        ``(B, H, NB, chunk, context_size)`` via right-pad + reshape +
        slice + reshape. See Appendix B of
        https://huggingface.co/papers/1901.02860. Bit-for-bit copy of HF's
        five-line implementation.
        """
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        # Pad last axis with zeros: (context+1 - position_length) zeros on the right.
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, context_size + 1 - position_length)))
        x = x.reshape(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.reshape(batch_size, num_heads, num_blocks, block_size, context_size)

    # -- forward -------------------------------------------------------------

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: Float[Array, "1 13 hidden"],
        attention_mask: Array | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Array]:
        batch_size, seq_length, _ = hidden_states.shape

        # Project Q/K/V then immediately upcast to fp32 (HF parity).
        query_states = self.q_proj(hidden_states).astype(jnp.float32)
        key_states = self.k_proj(hidden_states).astype(jnp.float32)
        value_states = self.v_proj(hidden_states).astype(jnp.float32)

        # Reshape to (B, T, H, D).
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        query_states = query_states.reshape(hidden_shape)
        key_states = key_states.reshape(hidden_shape)
        value_states = value_states.reshape(hidden_shape)

        # Per-dim softplus-scaled query, scalar-scaled key. Cast scale to fp32.
        per_dim = jax.nn.softplus(self.per_dim_scale.value.astype(jnp.float32))
        query_states = query_states * jnp.float32(self.q_scale) * per_dim
        key_states = key_states * jnp.float32(self.k_scale)

        # Block / context decomposition.
        query_states = self._convert_to_block(query_states)  # (B, NB, chunk, H, D)
        key_states = self._extract_block_context(key_states)  # (B, NB, context, H, D)
        value_states = self._extract_block_context(value_states)  # (B, NB, context, H, D)
        num_blocks = query_states.shape[1]

        # Relative-position keys: (1, 13, H*D) -> (13, H, D), in fp32.
        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.reshape(-1, self.num_heads, self.head_dim).astype(jnp.float32)

        # Permute queries to (B, H, NB, chunk, D) for batched matmul.
        queries = jnp.transpose(query_states, (0, 3, 1, 2, 4))

        # matrix_ac = queries @ keys^T per block, per head.
        # keys.permute(0,3,1,4,2) -> (B, H, NB, D, context).
        keys_t = jnp.transpose(key_states, (0, 3, 1, 4, 2))
        matrix_ac = jnp.matmul(queries, keys_t)  # (B, H, NB, chunk, context)

        # matrix_bd: queries vs. relative-position keys, broadcast over heads.
        # queries_flat: (B, H, NB*chunk, D); rel_k.permute(1,2,0): (H, D, 13).
        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        rel_k_t = jnp.transpose(relative_key_states, (1, 2, 0))  # (H, D, 13)
        matrix_bd = jnp.matmul(queries_flat, rel_k_t)  # (B, H, NB*chunk, 13)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)  # (B, H, NB, chunk, context)

        # Softcap BEFORE mask.
        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / jnp.float32(self.softcap)
        attn_weights = jnp.tanh(attn_weights)
        attn_weights = attn_weights * jnp.float32(self.softcap)

        # Mask: True = attend, False = sentinel (matches HF logical_not flip).
        if attention_mask is not None:
            attn_weights = jnp.where(
                attention_mask,
                attn_weights,
                jnp.float32(self.config.attention_invalid_logits_value),
            )

        # Softmax in fp32.
        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(value_states.dtype)

        # values.permute(0,3,1,2,4) -> (B, H, NB, context, D).
        values_t = jnp.transpose(value_states, (0, 3, 1, 2, 4))
        attn_output = jnp.matmul(attn_weights, values_t)  # (B, H, NB, chunk, D)

        # (B, H, NB, chunk, D) -> (B, NB, chunk, H, D) -> (B, NB*chunk, H*D).
        attn_output = jnp.transpose(attn_output, (0, 2, 3, 1, 4))
        attn_output = attn_output.reshape(batch_size, num_blocks * self.chunk_size, self.num_heads * self.head_dim)

        # Strip the chunk-pad rows.
        attn_output = attn_output[:, :seq_length]

        # Cast back to the post-projection kernel dtype (HF parity), then
        # apply the clippable output projection.
        attn_output = attn_output.astype(self.post.linear.kernel.value.dtype)
        attn_output = self.post(attn_output)

        return attn_output, attn_weights


class Gemma4AudioLayer(nn.Module):
    """One conformer block of the audio tower.

    Direct port of HF's ``Gemma4AudioLayer``. Each layer is a Macaron
    conformer: two half-step feed-forward blocks sandwich a self-attention
    block plus a depthwise GLU-conv block. Three RMSNorms — one before each
    of the (attention, post-attention residual, output) stages — and a
    *gradient_clipping* clamp before each norm act as activation guards.

    Shape and dtype contract::

        hidden_states : (B, T, hidden) float
        position_embeddings : (1, 13, hidden) float (from RelPositionalEncoding)
        attention_mask : (B, 1, NB, chunk, context) bool, optional

    The block flow exactly mirrors HF (line numbers in
    ``transformers/models/gemma4/modeling_gemma4.py``)::

        x = feed_forward1(x)              # Macaron half (full block, *0.5 inside)
        residual = x                      # snapshot AFTER the first FFN
        x = clamp(x, ±G); x = norm_pre_attn(x)
        x, _ = self_attn(x, position_embeddings, attention_mask)
        x = clamp(x, ±G); x = norm_post_attn(x)
        x = x + residual
        x = lconv1d(x)                    # GLU + depthwise causal conv + residual
        x = feed_forward2(x)              # Macaron half (full block, *0.5 inside)
        x = clamp(x, ±G); x = norm_out(x)

    Subtleties that must match HF for parity:

    * The residual snapshot is taken **after** the first FFN, *not* on the
      raw input. Both Macaron halves contribute their internal half-step
      residual to their inputs; the *attention* sub-block's residual is
      added at the explicit ``+= residual`` line on the post-attn-norm output.
    * The light-conv block carries its own residual internally (see
      :class:`Gemma4AudioLightConv1d`); the layer adds none on top.
    * ``gradient_clipping`` floors at ``finfo(weight_dtype).max`` per HF —
      the FFN already pre-computes this in init; we mirror it here so the
      clamp magnitude exactly matches what the FFN sub-blocks use.
    * Attention returns ``(out, weights)``; we discard the weights to keep
      the layer signature identical to HF's (returns just ``hidden_states``).
    """

    def __init__(
        self,
        config: Gemma4AudioConfig,
        layer_idx: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.config = config

        self.feed_forward1 = Gemma4AudioFeedForward(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.feed_forward2 = Gemma4AudioFeedForward(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.self_attn = Gemma4AudioAttention(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lconv1d = Gemma4AudioLightConv1d(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.norm_pre_attn = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.norm_post_attn = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.norm_out = Gemma4RMSNorm(config, param_dtype=param_dtype)

        # Pre-compute clamp bound the same way the FFN does, so attention
        # input/output and FFN clamp magnitudes match byte-for-byte.
        self.gradient_clipping = float(min(config.gradient_clipping, float(jnp.finfo(param_dtype).max)))

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: Float[Array, "1 13 hidden"],
        attention_mask: Array | None = None,
    ) -> Float[Array, "batch seq hidden"]:
        # Macaron half #1.
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        # Pre-attn clamp + norm.
        hidden_states = jnp.clip(hidden_states, -self.gradient_clipping, self.gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)

        # Self-attention. Drop the returned weights — HF only stores them via hooks.
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        # Post-attn clamp + norm + residual add.
        hidden_states = jnp.clip(hidden_states, -self.gradient_clipping, self.gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = hidden_states + residual

        # Light-conv block (carries its own residual internally).
        hidden_states = self.lconv1d(hidden_states)

        # Macaron half #2.
        hidden_states = self.feed_forward2(hidden_states)

        # Output clamp + norm.
        hidden_states = jnp.clip(hidden_states, -self.gradient_clipping, self.gradient_clipping)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states


class Gemma4AudioModel(nn.Module):
    """USM-style conformer audio encoder — full tower.

    Direct JAX port of HuggingFace's ``Gemma4AudioModel``. Forward pipeline::

        input_features (B, T_mel, F_mel)  # log-mel spectrogram
        + input_features_mask (B, T_mel)  # True = valid frame
                |
                v
        SubSampleConvProjection         # 4x downsample in time, project to hidden
                |
        (hidden_states (B, T, hidden), output_mask (B, T))
                |
                v
        rel_pos_enc(hidden_states) -> position_embeddings (1, 13, hidden)
                |
                v
        attention_mask = build_chunked_5d_mask(output_mask)
                |
                v
        for L layers: Gemma4AudioLayer(hidden_states, position_embeddings, mask)
                |
                v
        output_proj(hidden_states)      # hidden -> output_proj_dims (1024 -> 1536)
                |
                v
        return last_hidden_state, output_mask

    The mask construction is the *only* novel logic in this class — every
    other piece is already-tested. We re-implement HF's
    ``create_bidirectional_mask + sliding_window_mask_function +
    _convert_4d_mask_to_blocked_5d`` pipeline directly in JAX, matching the
    upstream algorithm step-for-step:

    1. Build a 4D bidirectional mask ``(B, 1, T, T)`` of valid (q, k) pairs
       under both the padding mask AND the sliding window:
       ``valid(b, q, k) = output_mask[b, q] & output_mask[b, k] &
                           in_window(q - k)``
       where ``in_window(d) = (0 <= d < past) | (-future < d < 0)``.
       (Note STRICT inequality on both bounds — matches HF's
       ``sliding_window_mask_function`` byte-for-byte.)
    2. Pad the time axis up to a multiple of ``chunk_size`` (with False).
    3. Reshape to ``(B, 1, NB, chunk, padded_T)`` of (block, query, key) layout.
    4. Pad the key axis on the left by ``past`` and on the right by ``future``
       so block ``b``'s keys ``[b·chunk - past, b·chunk + chunk + future)``
       index into the padded array contiguously.
    5. Gather the slice ``[b·chunk, b·chunk + context_size)`` for each block
       to produce the final ``(B, 1, NB, chunk, context_size)`` mask.

    The output ``output_mask`` (1D, downsampled by SSCP) is also returned
    for downstream consumers — ``Gemma4MultimodalEmbedder`` uses it to
    locate valid audio tokens when projecting into text-embedding space.
    """

    config: Gemma4AudioConfig

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config, dtype=dtype)

        self.layers = nn.List(
            [
                Gemma4AudioLayer(
                    config,
                    layer_idx=i,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # output_proj has bias=True in HF (the only audio-tower linear that does).
        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.output_proj = ColumnParallelLinear(
            config.hidden_size,
            config.output_proj_dims,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    # -- mask construction ---------------------------------------------------

    def _build_chunked_5d_mask(self, output_mask: Array) -> Array:
        """Bidirectional padding mask + sliding window, in 5D blocked layout.

        Args:
            output_mask: ``(B, T)`` bool — True = valid (post-SSCP downsampling).

        Returns:
            mask of shape ``(B, 1, NB, chunk, context_size)`` bool —
            True = (q, k) is a valid attention pair.
        """
        chunk = self.config.attention_chunk_size
        past = self.config.attention_context_left - 1
        future = self.config.attention_context_right
        context_size = chunk + past + future

        batch_size, seq_len = output_mask.shape

        # ---- step 1: 4D bidirectional mask with sliding window -------------
        q_pos = jnp.arange(seq_len)[:, None]  # (T, 1)
        k_pos = jnp.arange(seq_len)[None, :]  # (1, T)
        dist = q_pos - k_pos  # (T, T)
        # HF's sliding_window_mask_function: STRICT inequality on both sides.
        left_window = (dist >= 0) & (dist < past)
        right_window = (dist < 0) & (-dist < future)
        window = left_window | right_window  # (T, T)

        # Bidirectional padding: both query and key positions must be valid.
        pad_mask = output_mask[:, :, None] & output_mask[:, None, :]  # (B, T, T)
        mask_4d = (pad_mask & window[None, :, :])[:, None, :, :]  # (B, 1, T, T)

        # ---- step 2: pad to chunk multiple, reshape to 5D blocked ---------
        num_blocks = (seq_len + chunk - 1) // chunk
        padded_seq_len = num_blocks * chunk
        pad_amount = padded_seq_len - seq_len
        if pad_amount > 0:
            mask_4d = jnp.pad(
                mask_4d,
                ((0, 0), (0, 0), (0, pad_amount), (0, pad_amount)),
                constant_values=False,
            )
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk, padded_seq_len)

        # ---- step 3: pad key axis by (past, future) -----------------------
        mask_5d = jnp.pad(
            mask_5d,
            ((0, 0), (0, 0), (0, 0), (0, 0), (past, future)),
            constant_values=False,
        )
        # mask_5d now: (B, 1, NB, chunk, padded_seq_len + past + future)

        # ---- step 4: gather block-relative key indices ---------------------
        block_starts = jnp.arange(num_blocks) * chunk  # (NB,)
        offsets = jnp.arange(context_size)  # (context_size,)
        kv_indices = block_starts[:, None] + offsets[None, :]  # (NB, context_size)
        # Broadcast to (B, 1, NB, chunk, context_size).
        kv_indices = jnp.broadcast_to(
            kv_indices[None, None, :, None, :],
            (batch_size, 1, num_blocks, chunk, context_size),
        )
        return jnp.take_along_axis(mask_5d, kv_indices, axis=-1)

    # -- forward -------------------------------------------------------------

    def __call__(
        self,
        input_features: Float[Array, "batch t_mel f_mel"],
        input_features_mask: Array | None = None,
    ) -> tuple[Float[Array, "batch t hidden_out"], Array | None]:
        """Encode audio mel features to text-embedding-space pre-projection vectors.

        Args:
            input_features: ``(B, T_mel, F_mel)`` log-mel spectrogram.
            input_features_mask: ``(B, T_mel)`` float (1.0=valid) or bool, optional.

        Returns:
            ``(last_hidden_state, output_mask)`` where ``last_hidden_state``
            has shape ``(B, T_mel/4, output_proj_dims)`` and ``output_mask``
            has shape ``(B, T_mel/4)`` (or None if no input mask).
        """
        # SSCP downsamples 4x in time and projects to hidden_size.
        hidden_states, output_mask = self.subsample_conv_projection(input_features, input_features_mask)

        # Relative position embeddings: (1, 13, hidden_size). Independent of T.
        position_embeddings = self.rel_pos_enc(hidden_states)

        # Build chunked 5D attention mask (only when we have a 1D padding mask).
        # If output_mask is None, every position is valid -> we still need the
        # window structure to forbid attending outside the receptive field.
        if output_mask is None:
            output_mask = jnp.ones(hidden_states.shape[:2], dtype=jnp.bool_)
        else:
            # SSCP returns the mask as the same dtype as the input mask
            # (typically float). Convert to bool — True = valid.
            output_mask = output_mask.astype(jnp.bool_)

        attention_mask = self._build_chunked_5d_mask(output_mask)

        # Stack of conformer layers.
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        # Final projection: hidden_size -> output_proj_dims (1024 -> 1536).
        hidden_states = self.output_proj(hidden_states)

        return hidden_states, output_mask
