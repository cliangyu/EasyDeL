# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

"""Parity test for Bug -1.C: VLM tied LM head must match text-only path.

The VLM ``Gemma4ForConditionalGeneration.apply_lm_head`` previously called
``super().apply_lm_head`` which routes through the generic
``ColumnParallelLinear`` projection. The text-only ``Gemma4ForCausalLM``
instead reads the tied embedding via ``embed_tokens.attend(h)``. Under
TPU tensor-parallel layouts the two paths are not bit-identical, so the
VLM produces logits that drift from the text-only model even though
their tied weights are identical.

This test exercises both ``apply_lm_head`` and ``make_lm_head_fn`` under
``tie_word_embeddings=True``: it builds a tiny VLM, copies the language
model's embedding weights into a tiny CausalLM with a matching text
config, and verifies that:

* VLM and text-only ``apply_lm_head`` produce identical logits.
* VLM ``make_lm_head_fn`` returns a closure that produces the same
  result as the closure from the text-only model.
* The capping branch is exercised when ``final_logit_softcapping`` is set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx as nn

import easydel as ed
from easydel.modules.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
)


def _make_text_config(vocab_size: int = 256, hidden_size: int = 32, softcap: float | None = None):
    return ed.Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        head_dim=8,
        global_head_dim=8,
        sliding_window=32,
        hidden_size_per_layer_input=0,
        tie_word_embeddings=True,
        final_logit_softcapping=softcap,
    )


def _make_vlm_config(text_cfg: ed.Gemma4TextConfig):
    vision_cfg = ed.Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=4,
        pooling_kernel_size=1,
        position_embedding_size=8,
    )
    return ed.Gemma4Config(
        text_config=text_cfg,
        vision_config=vision_cfg,
        tie_word_embeddings=True,
    )


def _copy_embedding(src_module: nn.Module, dst_module: nn.Module) -> None:
    """Make dst's input embedding share weights with src's, so apply_lm_head
    on both classes is operating on the same underlying matrix."""
    src_embed = src_module.get_embedding().embedding[...]
    dst_module.get_embedding().embedding[...] = src_embed


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_vlm_apply_lm_head_matches_text_only(softcap):
    rng = nn.Rngs(0)
    text_cfg = _make_text_config(softcap=softcap)
    vlm_cfg = _make_vlm_config(text_cfg)

    text_model = Gemma4ForCausalLM(
        config=text_cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rng,
    )
    vlm_model = Gemma4ForConditionalGeneration(
        config=vlm_cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nn.Rngs(0),
    )
    _copy_embedding(text_model, vlm_model)

    key = jax.random.PRNGKey(7)
    hidden = jax.random.normal(key, (1, 4, text_cfg.hidden_size), dtype=jnp.float32)

    text_logits = text_model.apply_lm_head(hidden)
    vlm_logits = vlm_model.apply_lm_head(hidden)

    np.testing.assert_allclose(np.asarray(vlm_logits), np.asarray(text_logits), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_vlm_make_lm_head_fn_matches_text_only(softcap):
    rng = nn.Rngs(0)
    text_cfg = _make_text_config(softcap=softcap)
    vlm_cfg = _make_vlm_config(text_cfg)

    text_model = Gemma4ForCausalLM(
        config=text_cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rng,
    )
    vlm_model = Gemma4ForConditionalGeneration(
        config=vlm_cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nn.Rngs(0),
    )
    _copy_embedding(text_model, vlm_model)

    text_fn = text_model.make_lm_head_fn()
    vlm_fn = vlm_model.make_lm_head_fn()

    key = jax.random.PRNGKey(11)
    hidden = jax.random.normal(key, (1, 4, text_cfg.hidden_size), dtype=jnp.float32)

    np.testing.assert_allclose(np.asarray(vlm_fn(hidden)), np.asarray(text_fn(hidden)), atol=1e-6, rtol=1e-6)
