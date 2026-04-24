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
2. ``Gemma4AudioClippableLinear`` — TBD
3. ``Gemma4AudioFeedForward`` — TBD
4. ``Gemma4AudioLightConv1d`` — TBD
5. ``Gemma4AudioSubSampleConvProjection`` — TBD
6. ``Gemma4AudioAttention`` — TBD
7. ``Gemma4AudioLayer`` — TBD
8. ``Gemma4AudioModel`` — TBD
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from flax import nnx as nn
from jaxtyping import Array, Float

from .gemma4_configuration import Gemma4AudioConfig


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
