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

"""Video soft-token merge contract for Gemma4."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp
from flax import nnx as nn
from jax.sharding import Mesh

from easydel.modules.gemma4.gemma4_configuration import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from easydel.modules.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


def _make_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:1]), ("data",))


def _make_tiny_model() -> Gemma4ForConditionalGeneration:
    text = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        head_dim=8,
        global_head_dim=8,
        sliding_window=16,
        hidden_size_per_layer_input=0,
        tie_word_embeddings=True,
        attn_mechanism="vanilla",
    )
    vision = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=2,
        pooling_kernel_size=1,
        position_embedding_size=8,
    )
    cfg = Gemma4Config(
        text_config=text,
        vision_config=vision,
        image_token_id=121,
        video_token_id=120,
        audio_token_id=119,
        tie_word_embeddings=True,
    )
    return Gemma4ForConditionalGeneration(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))


def _patch_dim(model: Gemma4ForConditionalGeneration) -> int:
    patch_size = model.config.vision_config.patch_size
    return 3 * patch_size * patch_size


def _position_ids() -> jax.Array:
    return jnp.asarray([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=jnp.int32)


def _text_embeddings(model: Gemma4ForConditionalGeneration, input_ids: jax.Array) -> jax.Array:
    return model.base_model.language_model.embed_tokens(input_ids.astype("i4")) * (
        model.config.text_config.hidden_size**0.5
    )


def test_video_features_scatter_basic() -> None:
    with _make_mesh():
        model = _make_tiny_model()
        video_id = model.config.video_token_id
        patch_dim = _patch_dim(model)

        input_ids = jnp.asarray([[1, *([video_id] * 8), 2]], dtype=jnp.int32)
        pixel_values_videos = jnp.linspace(0.0, 1.0, 1 * 2 * 4 * patch_dim, dtype=jnp.float32).reshape(
            1, 2, 4, patch_dim
        )
        video_position_ids = jnp.broadcast_to(_position_ids()[:, None, :, :], (1, 2, 4, 2))

        outputs = model(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_position_ids=video_position_ids,
            apply_lm_head=False,
        )
        assert outputs.last_hidden_state.shape[:2] == input_ids.shape

        text_embeds = np.asarray(_text_embeddings(model, input_ids))
        merged = np.asarray(
            model.compute_embedding(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_position_ids=video_position_ids,
            )
        )
        video_features = np.asarray(model.get_video_features(pixel_values_videos, video_position_ids))

    video_slots = np.asarray(input_ids == video_id)[0]
    np.testing.assert_allclose(
        merged[0, video_slots],
        video_features.reshape(-1, video_features.shape[-1]),
        atol=1e-5,
    )
    assert not np.allclose(merged[0, video_slots], text_embeds[0, video_slots])
    np.testing.assert_allclose(merged[0, ~video_slots], text_embeds[0, ~video_slots], atol=1e-6)


def test_video_and_image_coexist() -> None:
    with _make_mesh():
        model = _make_tiny_model()
        image_id = model.config.image_token_id
        video_id = model.config.video_token_id
        patch_dim = _patch_dim(model)

        input_ids = jnp.asarray([[3, *([image_id] * 4), 4, *([video_id] * 8), 5]], dtype=jnp.int32)
        image_position_ids = _position_ids()
        video_position_ids = jnp.broadcast_to(image_position_ids[:, None, :, :], (1, 2, 4, 2))
        pixel_values = jnp.linspace(0.0, 0.5, 1 * 4 * patch_dim, dtype=jnp.float32).reshape(1, 4, patch_dim)
        pixel_values_videos = jnp.linspace(0.5, 1.0, 1 * 2 * 4 * patch_dim, dtype=jnp.float32).reshape(
            1, 2, 4, patch_dim
        )

        text_embeds = np.asarray(_text_embeddings(model, input_ids))
        merged = np.asarray(
            model.compute_embedding(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                pixel_values_videos=pixel_values_videos,
                video_position_ids=video_position_ids,
            )
        )
        image_features = np.asarray(model.get_image_features(pixel_values, image_position_ids))
        video_features = np.asarray(model.get_video_features(pixel_values_videos, video_position_ids))

    image_slots = np.asarray(input_ids == image_id)[0]
    video_slots = np.asarray(input_ids == video_id)[0]
    text_slots = ~(image_slots | video_slots)

    np.testing.assert_allclose(
        merged[0, image_slots],
        image_features.reshape(-1, image_features.shape[-1]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        merged[0, video_slots],
        video_features.reshape(-1, video_features.shape[-1]),
        atol=1e-5,
    )
    assert not np.allclose(merged[0, image_slots], text_embeds[0, image_slots])
    assert not np.allclose(merged[0, video_slots], text_embeds[0, video_slots])
    np.testing.assert_allclose(merged[0, text_slots], text_embeds[0, text_slots], atol=1e-6)


def test_no_video_path_unchanged() -> None:
    with _make_mesh():
        model = _make_tiny_model()
        input_ids = jnp.asarray([[1, 5, 9, 13, 17]], dtype=jnp.int32)
        outputs = model(input_ids=input_ids, pixel_values_videos=None, apply_lm_head=True)

    logits = np.asarray(outputs.logits, dtype=np.float32)
    digest = hashlib.sha256(logits.tobytes()).hexdigest()
    assert digest == "68ab7cb11bbff9b7312208ae14806ae03a9d881feb5a9812200683c9da331859"
