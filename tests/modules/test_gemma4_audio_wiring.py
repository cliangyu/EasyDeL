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

"""Wiring contract for ``Gemma4AudioModel`` inside ``Gemma4Model``.

The HF reference assembles the audio tower at ``model.audio_tower`` and
the audio→text projector at ``model.embed_audio``. To make the standard
EasyDeL HF→JAX converter discover audio params automatically (via
``traversals.iter_module_search``), both must be registered as direct
attributes on ``Gemma4Model`` whenever ``config.audio_config is not None``.

Without this wiring the converter silently drops every audio param
(no module to write into), which would only surface as garbage outputs
at inference time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
)
from easydel.modules.gemma4.modeling_gemma4 import (
    Gemma4Model,
    Gemma4MultimodalEmbedder,
)
from easydel.modules.gemma4.modeling_gemma4_audio import Gemma4AudioModel


def _make_tiny_model(audio: bool = True) -> Gemma4Model:
    text = Gemma4TextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=512,
        vocab_size_per_layer_input=512,
    )
    audio_cfg = (
        Gemma4AudioConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            attention_chunk_size=4,
            attention_context_left=9,
            attention_context_right=0,
            output_proj_dims=48,
            subsampling_conv_channels=[16, 8],
        )
        if audio
        else None
    )
    cfg = Gemma4Config(text_config=text, vision_config=None, audio_config=audio_cfg)
    return Gemma4Model(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))


def test_audio_tower_registered_when_audio_config_present() -> None:
    """``audio_tower`` and ``embed_audio`` must both exist on the model graph."""
    m = _make_tiny_model(audio=True)
    assert isinstance(m.audio_tower, Gemma4AudioModel)
    assert isinstance(m.embed_audio, Gemma4MultimodalEmbedder)


def test_audio_tower_none_when_audio_config_missing() -> None:
    """Without an audio config the audio sub-modules must be ``None``.

    HF parity: ``self.audio_tower = AutoModel.from_config(...) if audio_config
    is not None else None``. Critical for vision-only / text-only deployments
    so they don't pay parameter cost for an unused tower.
    """
    m = _make_tiny_model(audio=False)
    assert m.audio_tower is None
    assert m.embed_audio is None


def test_audio_params_visible_in_state_tree() -> None:
    """Audio params must show up under ``audio_tower/...`` in the param tree.

    This is the contract the HF→EasyDeL converter relies on: it looks up
    PT keys like ``audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight``
    and writes them into the corresponding JAX param path. Missing wiring
    here = silent param drop at conversion time.
    """
    m = _make_tiny_model(audio=True)
    paths = ["/".join(str(s) for s in path) for path, _ in nn.state(m, nn.Param).flat_state()]
    audio_paths = [p for p in paths if p.startswith("audio_tower/")]
    # Spot-check a few representative parameter sites.
    assert any("audio_tower/output_proj/kernel" == p for p in audio_paths)
    assert any("audio_tower/output_proj/bias" == p for p in audio_paths)
    # FFN layer kernels (Macaron block).
    assert any("layers/0/feed_forward1/ffw_layer_1/linear/kernel" in p for p in audio_paths)
    # SSCP convolutional stem.
    assert any("subsample_conv_projection" in p for p in audio_paths)


def test_embed_audio_uses_output_proj_dims() -> None:
    """``embed_audio.embedding_projection`` must accept the audio tower's
    output dim and produce the text hidden dim — otherwise the audio
    features won't fit when scattered into the embedding stream.
    """
    m = _make_tiny_model(audio=True)
    kernel = m.embed_audio.embedding_projection.kernel.value
    # nnx.Linear stores kernel as (in, out).
    assert kernel.shape == (
        m.config.audio_config.output_proj_dims,
        m.config.text_config.hidden_size,
    ), f"got {kernel.shape}"


def test_require_audio_tower_passthrough() -> None:
    """``_require_audio_tower`` returns silently when configured."""
    m = _make_tiny_model(audio=True)
    m._require_audio_tower()  # must not raise


def test_require_audio_tower_raises_without_config() -> None:
    """When no audio config is given, ``_require_audio_tower`` must raise."""
    m = _make_tiny_model(audio=False)
    with pytest.raises(ValueError, match="without an audio config"):
        m._require_audio_tower()
