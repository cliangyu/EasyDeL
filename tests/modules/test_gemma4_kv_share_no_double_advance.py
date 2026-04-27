# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

"""Parity test for Bug -1.E+F: shared decoder layers must NOT advance the
donor's cache index.

Before this fix, the shared layer's ``concatenate`` call hit the write path
in ``_handle_cache_concat``, which calls ``cache_view.concatenate_to_cache``
and advances ``cache_view.indexs += q_len``. The donor (non-shared) layer
also writes via the same path. Aliased to the donor view, the shared layer
double-advances the index by another ``q_len`` → cache appears to contain
``2*q_len`` tokens, future reads pull garbage from beyond the actual
contents, and decode-time KV pages get corrupted.

The fix passes ``write_cache=False`` from the shared-layer call site, which
hits a new read-only branch in ``_flexible.py:concatenate`` that reads from
``cache_view.key/value`` and applies ``mask_info.apply_kv_lengths`` without
re-writing or re-advancing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx as nn
from jax.sharding import Mesh

from easydel.modules.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig


def _make_mesh():
    return Mesh(np.array(jax.devices()[:1]), ("data",))


def _config(**overrides):
    defaults = dict(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        max_position_embeddings=64,
        sliding_window=32,
        layer_types=["full_attention", "full_attention"],
        attn_mechanism="vanilla",
        num_kv_shared_layers=1,
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def test_kv_share_prefill_no_double_advance():
    """Prefill of q_len tokens must leave donor.indexs == q_len, not 2*q_len."""
    config = _config()
    q_len = 4
    input_ids = jnp.array([[2, 17, 23, 29]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)

    with _make_mesh():
        model = Gemma4ForCausalLM(
            config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0)
        )
        model_kwargs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            pad_token_id=config.pad_token_id,
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

    donor_view = outputs.past_key_values.views[0]
    indexs = np.asarray(donor_view.indexs)
    assert indexs.tolist() == [q_len], (
        f"donor.indexs={indexs.tolist()}, expected [{q_len}] — shared layer "
        "double-advanced the donor's cache index."
    )


def test_kv_share_decode_step_advances_by_one_only():
    """One decode step must advance donor.indexs by exactly 1."""
    config = _config()
    input_ids = jnp.array([[2, 17, 23, 29]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)

    with _make_mesh():
        model = Gemma4ForCausalLM(
            config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0)
        )
        model_kwargs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            pad_token_id=config.pad_token_id,
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
        prefill_indexs = np.asarray(outputs.past_key_values.views[0].indexs).tolist()

        # One decode step.
        current_kwargs = model.update_inputs_for_generation(outputs, model_kwargs)
        next_token = jnp.array([[42]], dtype=jnp.int32)
        call_kwargs = model._prepare_mask_info_for_generation_step(next_token, current_kwargs)
        outputs2 = model(next_token, **call_kwargs)

    decode_indexs = np.asarray(outputs2.past_key_values.views[0].indexs).tolist()
    assert decode_indexs == [prefill_indexs[0] + 1], (
        f"donor.indexs went {prefill_indexs} → {decode_indexs} after one decode step; "
        "expected single advance, not double."
    )


def test_kv_share_without_share_advances_normally():
    """Sanity: when num_kv_shared_layers=0, both layers write so donor.indexs
    still reflects exactly q_len for layer 0 (each layer has its own view)."""
    config = _config(num_kv_shared_layers=0)
    q_len = 4
    input_ids = jnp.array([[2, 17, 23, 29]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)

    with _make_mesh():
        model = Gemma4ForCausalLM(
            config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0)
        )
        model_kwargs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            pad_token_id=config.pad_token_id,
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

    for layer_idx in range(config.num_hidden_layers):
        view = outputs.past_key_values.views[layer_idx]
        indexs = np.asarray(view.indexs).tolist()
        assert indexs == [q_len], (
            f"layer {layer_idx} indexs={indexs}, expected [{q_len}] (no sharing)"
        )
