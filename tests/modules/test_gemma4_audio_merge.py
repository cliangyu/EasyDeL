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

"""Audio-soft-token merge contract for ``Gemma4Model.compute_embedding``.

Pins the cumsum-gather scatter that places audio features into
``input_ids == audio_token_id`` slots. Two failure modes this test
shields against:

1. **Silent no-op**: the audio path forgets to call ``embed_audio`` /
   ``audio_tower`` and placeholder tokens remain at their (untrained)
   text embedding values. The test asserts placeholder slots strictly
   *change* relative to a no-audio call.
2. **Slot misalignment**: the gather uses the wrong cumulative count
   so placeholder slot ``k`` gets the ``j``-th feature for ``j != k``.
   The test seeds each post-SSCP audio frame with a unique scalar
   pattern and checks the merge preserves it 1:1 in placeholder order.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
)
from easydel.modules.gemma4.modeling_gemma4 import Gemma4Model


def _make_tiny_model() -> Gemma4Model:
    text = Gemma4TextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=512,
        # Disable per-layer-inputs to keep the test focused on the merge.
        vocab_size_per_layer_input=512,
        hidden_size_per_layer_input=0,
    )
    audio = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        attention_chunk_size=4,
        attention_context_left=9,
        attention_context_right=0,
        output_proj_dims=48,
        subsampling_conv_channels=[16, 8],
    )
    # Override audio_token_id so it fits inside the tiny vocab — production
    # default sits well above the test vocab_size and would NaN the lookup.
    cfg = Gemma4Config(text_config=text, vision_config=None, audio_config=audio, audio_token_id=42)
    return Gemma4Model(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))


def test_audio_placeholders_are_replaced() -> None:
    """Embedding at audio-placeholder positions must change once audio is fed."""
    m = _make_tiny_model()
    audio_id = m.config.audio_token_id

    # Sequence: [<bos>, <bos>, <audio>, <audio>, <audio>, <audio>, <bos>, <bos>]
    bos = 1
    n_audio_tokens = 4  # T_mel=16 -> T_audio=4
    seq = [bos, bos] + [audio_id] * n_audio_tokens + [bos, bos]
    input_ids = jnp.asarray([seq], dtype=jnp.int32)

    # T_mel must satisfy SSCP requirements: divisible by 4 (subsample x4) and
    # F_mel = subsampling_conv_channels[0] (HF stale formula).
    T_mel = n_audio_tokens * 4
    F_mel = m.config.audio_config.subsampling_conv_channels[0]
    input_features = jax.random.normal(jax.random.key(7), (1, T_mel, F_mel), dtype=jnp.float32)

    embeds_no_audio = np.asarray(m.compute_embedding(input_ids))
    embeds_with_audio = np.asarray(m.compute_embedding(input_ids, input_features=input_features))

    # Placeholder slots must differ; non-placeholder slots must be identical.
    audio_slots = np.array([False, False, True, True, True, True, False, False])
    placeholder_diff = np.linalg.norm(embeds_with_audio[0, audio_slots] - embeds_no_audio[0, audio_slots], axis=-1)
    assert (placeholder_diff > 1e-3).all(), "audio merge had no effect on placeholder slots"

    np.testing.assert_array_equal(
        embeds_with_audio[0, ~audio_slots],
        embeds_no_audio[0, ~audio_slots],
    )


def test_audio_merge_preserves_per_slot_ordering() -> None:
    """Slot k must receive the k-th audio feature (no shuffle, no off-by-one).

    Uses ``compute_embedding`` directly with ``inputs_embeds=None`` to get
    the merge step in isolation. Probes ordering via the helper
    ``_scatter_features_at_token`` so we don't have to reason about the
    audio tower's nonlinearities — the helper *is* the full scatter kernel
    used in the audio path.
    """
    m = _make_tiny_model()
    audio_id = m.config.audio_token_id
    text_hidden = m.config.text_config.hidden_size

    # Construct a synthetic feature stream where feature k is constant value k.
    n_tokens = 5
    features = jnp.broadcast_to(
        jnp.arange(n_tokens, dtype=jnp.float32)[:, None],
        (n_tokens, text_hidden),
    )[None, :, :]  # (1, n_tokens, D)

    # Sequence has 5 placeholders interspersed with non-placeholder tokens.
    seq = [10, audio_id, 11, audio_id, audio_id, 12, audio_id, audio_id, 13]
    input_ids = jnp.asarray([seq], dtype=jnp.int32)

    # Start from zero embeddings so we cleanly see the scatter.
    inputs_embeds = jnp.zeros((1, len(seq), text_hidden), dtype=jnp.float32)
    out = np.asarray(
        m._scatter_features_at_token(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            features=features,
            token_id=audio_id,
        )
    )

    # Placeholder positions in order: [1, 3, 4, 6, 7] -> features 0..4.
    expected_per_pos = {1: 0.0, 3: 1.0, 4: 2.0, 6: 3.0, 7: 4.0}
    for pos, val in expected_per_pos.items():
        np.testing.assert_array_equal(
            out[0, pos],
            np.full((text_hidden,), val, dtype=np.float32),
            err_msg=f"placeholder #{pos} expected scalar {val}",
        )

    # Non-placeholder positions stay zero.
    for pos in (0, 2, 5, 8):
        np.testing.assert_array_equal(
            out[0, pos],
            np.zeros((text_hidden,), dtype=np.float32),
        )


def test_get_audio_features_shape() -> None:
    """``get_audio_features`` returns ``(B, T_mel/4, text_hidden)`` + bool mask."""
    m = _make_tiny_model()
    F_mel = m.config.audio_config.subsampling_conv_channels[0]
    B, T_mel = 1, 16
    x = jax.random.normal(jax.random.key(11), (B, T_mel, F_mel), dtype=jnp.float32)
    feats, mask = m.get_audio_features(x)
    assert feats.shape == (B, T_mel // 4, m.config.text_config.hidden_size)
    assert mask.shape == (B, T_mel // 4)
    assert mask.dtype == jnp.bool_


def test_get_audio_features_raises_without_config() -> None:
    """``get_audio_features`` must raise when audio_config is missing."""
    text = Gemma4TextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size_per_layer_input=0,
    )
    cfg = Gemma4Config(text_config=text, vision_config=None, audio_config=None)
    m = Gemma4Model(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match="without an audio config"):
        m.get_audio_features(jnp.zeros((1, 16, 16), dtype=jnp.float32))


def test_per_layer_inputs_masks_audio_token() -> None:
    """``_compute_per_layer_inputs`` must replace audio_token_id with pad_token_id.

    Otherwise the per-layer-input lookup tries to embed an OOV token id
    (audio_token_id sits well above ``vocab_size_per_layer_input``).
    """
    text = Gemma4TextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size_per_layer_input=8,
        pad_token_id=0,
    )
    audio = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        attention_chunk_size=4,
        attention_context_left=9,
        attention_context_right=0,
        output_proj_dims=48,
        subsampling_conv_channels=[16, 8],
    )
    cfg = Gemma4Config(text_config=text, vision_config=None, audio_config=audio)
    m = Gemma4Model(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))

    audio_id = m.config.audio_token_id
    input_ids = jnp.asarray([[1, audio_id, audio_id, 5]], dtype=jnp.int32)
    # Should not raise — audio_token_id positions get pad_token_id (0).
    out = m._compute_per_layer_inputs(input_ids)
    assert out is not None
    assert np.isfinite(np.asarray(out)).all()
