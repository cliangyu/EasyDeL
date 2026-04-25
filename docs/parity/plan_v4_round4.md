# Gemma4 Audio Port — Plan v4 (Round 4 dispatch input)

Source: Round 3 critique by Codex (agent `a003c36abc4c4fd24`) + Claude verification.
Round 3 confirmed -1.A landed (`cc056b25`), broke -1.B design, expanded -1.C scope, surfaced new Bug E.
All paths relative to repo root unless absolute. All claims cite `file:line`.

## What's new since v3

1. **Bug E added**: shared-layer cache double-advance at `modeling_gemma4.py:1503-1519`. Confirmed via inspection of `concatenate` at `_flexible.py:1090-1165` — both ragged and standard paths call `concatenate_to_cache` which advances `indexs` (cache.py:596-733). Donor's `indexs` is advanced once by donor's call, then again by shared layer's call when `cache_view` is borrowed from donor (`modeling_gemma4.py:2587-2590`). Cache pointer corrupts after every donor+shared pair. **Blocks all Phase C decode-with-cache.** Tracked as task #36.

2. **-1.C scope expanded**: Codex caught that `make_lm_head_fn` at `modeling_gemma4.py:3443-3455` has the same tie-check bug as `apply_lm_head` at `:3436`. Both call `super()` which reads top-level `config.tie_word_embeddings` (`base_module.py:2819-2826, 2862-2870`); VLM has it on `config.text_config`. Text-only overrides BOTH at `:2776-2789` and `:2791-2815` — VLM must too.

3. **-1.B design rewrite**: Codex rejected the masking-only approach. New design uses argsort-based stable compaction. Detailed below.

4. **Phase 0.0 thresholds**: gate from CPU-FP32 floors only; final B1/B2/B3 thresholds bound to TPU-BF16 calibration after first successful Tier 1 run.

## Phase -1: Port-level fixes (all gates required before TPU work)

### -1.A KV-share side channel — LANDED in `cc056b25`

(see plan_v3_round3.md for full diff). Unchanged.

### -1.B Scatter batch-leak fix — REWRITTEN

**Bug** at `modeling_gemma4.py:3149-3163`: global cumsum across flattened batch (`special_mask.reshape(-1)` then `cumsum`) means B1's first placeholder gathers from B0's row range. Confirmed walkthrough: B=2, B0=100 valid frames, B1=50, L=200 → B1's first placeholder at flat position 230 → cumsum=5 → gathers `features_flat[4]` = B0's 5th feature.

**Fix design (v4)** — per-row compaction via argsort-stable, then per-row gather:

```python
def _scatter_one_row(input_ids_b, inputs_embeds_b, features_b, valid_mask_b, token_id, T):
    """Per-row scatter for one batch element.
    Args:
      input_ids_b:    (L,)   int32
      inputs_embeds_b: (L, D) float
      features_b:     (T, D) float — audio tower output for this row
      valid_mask_b:   (T,)   bool   — audio output valid frames
      token_id:       int    — placeholder id (audio_token_id)
      T:              static int — features axis length
    Returns:
      (L, D) float — embeddings with valid features scattered into placeholder slots.
    """
    # 1. Compact valid features to the front using stable argsort.
    #    -valid_int sorts True before False; stable=True preserves order within group.
    valid_int = valid_mask_b.astype(jnp.int32)
    sort_idx = jnp.argsort(-valid_int, stable=True)            # (T,)
    packed_features = features_b[sort_idx]                      # (T, D)

    # 2. Per-row placeholder positions (local cumsum, NOT global).
    placeholder_mask = (input_ids_b == token_id)                # (L,)
    row_cumsum = jnp.cumsum(placeholder_mask.astype(jnp.int32)) - 1   # (L,)

    # 3. Bounds-safe gather.
    #    If n_placeholders > T (shouldn't happen if HF preflight is correct),
    #    gather indices clamp to T-1 — caller must Python-assert before jit.
    gather_idx = jnp.where(placeholder_mask, jnp.minimum(row_cumsum, T - 1), 0)
    gathered = packed_features[gather_idx]                      # (L, D)

    # 4. Scatter only at placeholder positions.
    return jnp.where(placeholder_mask[:, None], gathered, inputs_embeds_b)


# Caller (compute_embedding):
B, T, D = features.shape
inputs_embeds_new = jax.vmap(
    _scatter_one_row, in_axes=(0, 0, 0, 0, None, None)
)(input_ids, inputs_embeds, features, audio_output_mask, audio_token_id, T)
```

**Pre-flight assertion** (Python, before jit): for each row,
```python
n_placeholders_per_row = (input_ids == audio_token_id).sum(axis=-1)  # (B,)
n_valid_per_row = audio_output_mask.sum(axis=-1)                     # (B,)
assert (n_placeholders_per_row == n_valid_per_row).all(), (
    f"Per-row placeholder count must equal per-row valid feature count. "
    f"Got placeholders={n_placeholders_per_row}, valid={n_valid_per_row}"
)
```

**Why this is correct** (vs. v3 design):
- v3 used `packed_features = features * valid_mask[:, None]` which only zeros invalid frames — does NOT compact. Codex correctly flagged this.
- v4 uses `argsort(-valid_int, stable=True)`: produces a permutation that puts valid frames first (in original order), invalid frames after. `features[sort_idx]` is true compaction.
- `jnp.argsort(stable=True)` is documented JAX API — confidence: 95% from training data, verify against current jax.numpy docs.
- Bounds clamp `jnp.minimum(row_cumsum, T - 1)` prevents OOB indexing if pre-flight is somehow bypassed; combined with Python assert, both belt and suspenders.
- vmap-over-batch composes with downstream `compute_embedding` because the result has the same `(B, L, D)` shape and dtype as `inputs_embeds` (just with placeholder slots replaced).

**Open questions for Codex**:
- Is `jnp.argsort(stable=True)` actually stable on TPU? (HF training data says yes for CPU; TPU implementation uses a different sort kernel.)
- Is per-row Python assert OK as the only enforcement, given JAX-traced calls would skip it? Or do we need an in-graph `checkify`?

### -1.C VLM tied LM head — EXPANDED to two methods

**Site 1**: `Gemma4ForConditionalGeneration.apply_lm_head` at `:3436`. Replace `super().apply_lm_head` call with explicit tie-attend matching text-only's pattern.

**Site 2**: `Gemma4ForConditionalGeneration.make_lm_head_fn` at `:3443-3455`. Same fix.

**Reference**: text-only `Gemma4ForCausalLM.apply_lm_head` at `:2776-2789` and `make_lm_head_fn` at `:2791-2815`.

**Diff design**:

```python
# Replace lines 3422-3441 (apply_lm_head):
def apply_lm_head(self, hidden_states):
    if getattr(self.config.text_config, "tie_word_embeddings", False):
        lm_logits = self.get_embedding().attend(hidden_states)
    else:
        lm_logits = super().apply_lm_head(hidden_states)
    cap = getattr(self.config.text_config, "final_logit_softcapping", None)
    if cap is not None:
        cap = jnp.array(cap, dtype=lm_logits.dtype)
        lm_logits = cap * jax.nn.tanh(lm_logits / cap)
    return lm_logits

# Replace lines 3443-3455 (make_lm_head_fn):
def make_lm_head_fn(self):
    cap_value = getattr(self.config.text_config, "final_logit_softcapping", None)
    if getattr(self.config.text_config, "tie_word_embeddings", False):
        _attend = self.get_embedding().attend
        def _project(hidden_states):
            lm_logits = _attend(hidden_states)
            if cap_value is not None:
                cap = jnp.array(cap_value, dtype=lm_logits.dtype)
                lm_logits = cap * jax.nn.tanh(lm_logits / cap)
            return lm_logits
    else:
        base_fn = super().make_lm_head_fn()
        if cap_value is None:
            return base_fn
        def _project(hidden_states):
            logits = base_fn(hidden_states)
            cap = jnp.array(cap_value, dtype=logits.dtype)
            return cap * jax.nn.tanh(logits / cap)
    return _project
```

**Verification plan**: CPU parity test comparing `Gemma4ForCausalLM.apply_lm_head(h)` to `Gemma4ForConditionalGeneration.apply_lm_head(h)` for matching configs. Same for `make_lm_head_fn`-returned closures.

### -1.D Bug D `_causal_baked` impurity — DEFERRED

Static bool, works under jit via tracing. Codex Round 3 partial-flagged a decode-path concern in `mixins/generation.py:3063-3069` where the attr is re-propagated after `apply_kv_lengths`. Watch for it during Phase B/C; clean up post-smoke as task #35.

### -1.E Bug E shared-layer cache double-advance — NEW

**Bug** (`modeling_gemma4.py:1503-1519`):

```python
# In Gemma4Attention.__call__ KV-sharing branch:
if cache_view is not None:
    (key_states, value_states, mask_info, init_attention_bias,
     cache_view, cache_metadata) = self.concatenate(...)
```

`self.concatenate` at `_flexible.py:1090-1165` calls `concatenate_to_cache` (line 954 ragged, line 1165 standard) which writes K/V and advances `cache_view.indexs`. The donor layer already did this for the same K/V at `modeling_gemma4.py:1361-1369`. Result: shared layer borrows donor's `cache_view`, calls `concatenate` again, advances `indexs` a 2nd time → cache pointer drifts forward by 1 per donor+shared pair per token.

**Verification of root cause**: line 2587-2590 confirms shared layers borrow donor's cache_view:
```python
cache_view = past_key_values.views[idx]
if cache_view is None and attn.is_kv_shared_layer:
    cache_view = donor_cache_views.get(attn.kv_shared_layer_index)
```
And line 2617 confirms shared layers don't write back to `past_key_values`:
```python
if not attn.is_kv_shared_layer:
    past_key_values[idx] = layer_outputs.cache_view
```
So the donor's `past_key_values` slot gets overwritten with the shared layer's (double-advanced) cache_view at the next iteration of the donor's family — except: only the donor's `past_key_values[donor_idx]` is written; the shared layer's mutation lives in `donor_cache_views[donor_idx]` but never flows back. **The exact corruption mechanism needs more analysis** — Codex was confident this is broken; I want a second pass.

**Fix design (v4)** — Option A: `write_cache: bool` flag on `concatenate`.

Add an optional kwarg `write_cache: bool = True` to `AttentionModule.concatenate` at `_flexible.py:1090`. When `False`, the function still does mask/bias setup and reads K/V history from cache, but skips `concatenate_to_cache` (so no `indexs` advance, no K/V write).

```python
# _flexible.py change (~line 1165):
if is_ragged_cache:
    if write_cache:
        cache_view = cache_view.concatenate_to_cache(...)
    # else: cache_view unchanged, K/V already in cache from donor
    ...
```

**Gemma4Attention shared-path call** (`modeling_gemma4.py:1503-1519`):

```python
if cache_view is not None:
    (key_states, value_states, mask_info, init_attention_bias,
     cache_view, cache_metadata) = self.concatenate(
        query=query_states, key=key_states, value=value_states,
        cache_view=cache_view, cache_metadata=cache_metadata,
        mask_info=mask_info, sliding_window=sliding_window_for_kernel,
        write_cache=False,   # NEW: shared layer reuses donor's writes
    )
```

**Cross-model impact**: adding optional `write_cache` kwarg with default `True` is backward-compatible across all `concatenate` callers. Default is True so all non-shared layers behave as before.

**Open questions for Codex**:
- Is "skip `concatenate_to_cache` but still read mask/bias" actually doable cleanly, or does mask/bias setup depend on intermediate state that `concatenate_to_cache` also updates?
- Does the standard-cache (non-ragged) path at `_flexible.py` also need the flag? (Yes — verify.)
- Alternative design: factor mask/bias setup into `prepare_attention_inputs(cache_view, write=False)` and have shared layers call that directly. Cleaner separation but wider refactor.

**Verification plan**: integration test under jit with `num_kv_shared_layers=1`, decode 4 tokens, assert `cache_view.indexs == 4` after generation (not 8). Add to Phase 0.x.

## Phases 0–D unchanged (see plan_v3_round3.md)

Plus new test in Phase 0.x: KV-share cache integration test under decode mode (Bug E verifier).

## Sequencing decision

- **-1.A**: LANDED.
- **-1.C** (LM head): independent of -1.B and -1.E. Can land next; ~30 LoC mirror of text-only methods.
- **-1.B** (scatter): independent of -1.E.
- **-1.E** (cache): touches base infra (`_flexible.py` `concatenate` signature). Largest blast radius.

**Recommended order**: -1.C → -1.B → -1.E (LoC ascending, blast radius ascending). Each commits separately, Codex reviews each diff.

**Alternative**: parallelize -1.B and -1.C (independent), serialize -1.E last.

## Round 4 attack questions for Codex

1. **Bug E mechanism**: Walk through one decode step with `num_kv_shared_layers=1` (donor=layer N, shared=layer N+1). At iteration N (donor): cache_view writes K/V to slot k, indexs advances to k+1, donor_cache_views[N] = updated_view. At iteration N+1 (shared): cache_view = donor_cache_views[N] (indexs=k+1), concatenate writes K/V to slot k+1 (NOT k+2), indexs advances to k+2. Then `past_key_values[N] = updated_view` does NOT happen for shared layer (line 2617 guard). At next decode step: donor reads from `past_key_values[N]` whose indexs is still k+1 (not k+2 because shared's updated view didn't flow back). Donor writes at slot k+1 → CORRUPTS the K/V the shared layer wrote previously. Confirm or refute this exact failure pattern.

2. **-1.B argsort-stable**: Is `jnp.argsort(stable=True)` actually stable on TPU? Cite JAX source or doc URL.

3. **-1.B per-row Python assert**: When the model is called from `jax.jit`, is the Python assert evaluated at trace time (once on concrete shapes/values) or runtime? If trace-time only, the assert is bypassed for actual mismatched data at runtime — do we need `jax.experimental.checkify`?

4. **-1.C tied attend semantics**: Is `Embed.attend(hidden_states)` numerically equivalent to `super().apply_lm_head(hidden_states)` when `tie_word_embeddings=True`? Specifically: does the base `apply_lm_head` use `embedding.value.T` differently from `attend()`'s internal `jnp.dot(query, embedding.T)`? If they differ, the parity test in Phase 0.8 will catch it but I want to know in advance.

5. **-1.E concatenate refactor**: Is `write_cache=False` actually correct? Specifically: when the donor advances indexs but the shared layer also needs to "see" the post-write state for mask alignment — does `cache_view.read()` (or whatever `concatenate` does for read-only) give the same K/V history view that the donor's `concatenate_to_cache` produced?

6. **What v4 still misses**: Highest-leverage finding I haven't accounted for. Specifically check: (a) the `attentions.cache_view` write-back at `:1541-1542` (does this also need a guard for shared layers?), (b) the donor's `past_key_values[idx] = layer_outputs.cache_view` at `:2617` — does this happen with the donor's cache_view BEFORE the shared layer borrowed it, or AFTER (if Python references)?

7. **Sequencing**: -1.C → -1.B → -1.E. Are there hidden cross-dependencies? E.g., does the Phase 0 vmap-scatter test depend on -1.E being landed?

8. **Phase 0.0 floor strategy**: After Round 3 you flagged CPU-FP32 floors don't translate to TPU-BF16 gates. v4 says "gate from CPU floors only; bind final TPU thresholds to BF16 calibration after first Tier 1 run". Is this the right approach, or is there a better calibration path I'm missing (e.g., synthetic BF16 simulation on CPU)?

Cite file:line for every code claim. Hard cap: 1500 words.
