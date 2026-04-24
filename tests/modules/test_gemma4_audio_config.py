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

"""Hermetic defaults-pinning test for ``Gemma4AudioConfig``.

The audio config is the lowest-risk port artefact in the Gemma 4 E4B audio
tower port (see ``docs/superpowers/plans/2026-04-24-gemma4-audio-port-risks.md``).
This test locks every default to the value that HuggingFace's
``transformers.models.gemma4.configuration_gemma4.Gemma4AudioConfig`` ships
with, so accidental drift during future refactors is caught in CI.

The test is fully hermetic: it imports only the EasyDeL config class (no JAX,
no torch, no transformers) and asserts on plain Python attributes.
"""

from __future__ import annotations

import pytest

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig

# Defaults taken verbatim from HF ``Gemma4AudioConfig`` (transformers main as of
# 2026-04-24). If HF changes a default, update this dict and bump the expected
# value in the corresponding assertion below.
HF_AUDIO_DEFAULTS: dict[str, object] = {
    "hidden_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "hidden_act": "silu",
    "subsampling_conv_channels": [128, 32],
    "conv_kernel_size": 5,
    "residual_weight": 0.5,
    "attention_chunk_size": 12,
    "attention_context_left": 13,
    "attention_context_right": 0,
    "attention_logit_cap": 50.0,
    "attention_invalid_logits_value": -1.0e9,
    "use_clipped_linears": True,
    "rms_norm_eps": 1e-6,
    "gradient_clipping": 1e10,
    "output_proj_dims": 1536,
    "initializer_range": 0.02,
}


@pytest.mark.parametrize("field,expected", list(HF_AUDIO_DEFAULTS.items()))
def test_default_matches_hf(field: str, expected: object) -> None:
    """Every default must match the HF oracle exactly."""
    cfg = Gemma4AudioConfig()
    assert getattr(cfg, field) == expected, f"{field}: expected {expected!r}, got {getattr(cfg, field)!r}"


def test_subsampling_channels_list_not_tuple() -> None:
    """HF converts tuple → list in ``__post_init__`` for JSON round-trips.

    Our JAX port does the same so that checkpoints serialised by HF and loaded
    by EasyDeL (and vice-versa) round-trip without type drift.
    """
    cfg = Gemma4AudioConfig()
    assert isinstance(cfg.subsampling_conv_channels, list)


def test_overrides_applied() -> None:
    """Keyword overrides must replace defaults, not append to them."""
    cfg = Gemma4AudioConfig(hidden_size=2048, num_hidden_layers=6, attention_chunk_size=24)
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 6
    assert cfg.attention_chunk_size == 24
    # untouched fields still match HF
    assert cfg.output_proj_dims == 1536
    assert cfg.residual_weight == 0.5


def test_model_type_is_gemma4_audio() -> None:
    """HF sets ``model_type = "gemma4_audio"``; factory registration depends on it."""
    cfg = Gemma4AudioConfig()
    assert cfg.model_type == "gemma4_audio"
