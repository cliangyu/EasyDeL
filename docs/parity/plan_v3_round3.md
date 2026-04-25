# Gemma4 Audio Port — Plan v3 (Round 3 dispatch input)

Source: independent verification by Claude after Round 2 + adversarial fact-finding while Codex usage was rate-limited.
Round 2 surfaced 3 bugs; Round 3 fact-finding surfaced a 4th (Bug D) and concretized fixes.
All paths relative to repo root unless absolute. All claims cite `file:line`.

## What's new since v2

1. **Phase -1.A is now landed code (commit `cc056b25`)** — diff in §-1.A below. Codex review should now attack the actual diff, not just the design.
2. **Bug D added** (Phase -1.D, lower priority): `object.__setattr__(mask_info, "_causal_baked", True)` at `modeling_gemma4.py:2542-2543`. Static bool, works under jit via tracing; code smell, not a runtime bug. Tracked as task #35.
3. **VLM tied embedding confirmed safe** for the proposed -1.C fix:
   - `Gemma4ForConditionalGeneration.config.text_config.tie_word_embeddings` accessed at `:3326`
   - `Gemma4ForConditionalGeneration.get_embedding()` returns text embedding only (`:3281-3283 → Gemma4TextModel.get_embedding() :2640-2642 → embed_tokens`)
4. **AttentionLayerOutput / DecoderLayerOutput field-add is wide but kwarg-safe**: 13 + 24+ call sites across all model families verified via grep — every site uses kwargs, so optional fields with `None` default are backward-compatible.
5. **Audio/vision towers are clean** — no `object.__setattr__` impurities outside the language model.
6. **Bug B concrete walkthrough** confirms the leak: B=2, B0=100 valid frames, B1=50, L=200, B1's first placeholder at flat-index 230 → cumsum=5 → gathers `features_flat[4]` = B0's 5th audio feature.

## Phase -1: Port-level fixes (NO TPU work proceeds without these)

### -1.A KV-share side channel — LANDED in `cc056b25`

**Diff** (2 files, +20 −8):

```python
# easydel/infra/modeling_outputs.py
@auto_pytree
class AttentionLayerOutput(ModelOutput):
    attention_output: Array
    attention_weight: Array | None = None
    cache_view: TransformerCacheView | None = None
+   captured_kv: tuple[Array, Array] | None = None

@auto_pytree
class DecoderLayerOutput(ModelOutput):
    hidden_states: Array
    ...
    cache_view: TransformerCacheView | None = None
+   captured_kv: tuple[Array, Array] | None = None
```

```python
# easydel/modules/gemma4/modeling_gemma4.py
# Site 1 (Gemma4Attention._forward_with_kv_capture, ~line 1344):
- object.__setattr__(self, "_captured_kv", (key_states, value_states))
+ captured_kv: tuple[Array, Array] = (key_states, value_states)

# Site 2 (return at line 1404):
return AttentionLayerOutput(
    attention_output=attn_output,
    attention_weight=attentions.attention_weight if output_attentions else None,
    cache_view=cache_view,
+   captured_kv=captured_kv,
)

# Site 3 (Gemma4DecoderLayer return at line 2187):
return DecoderLayerOutput(
    hidden_states=hidden_states,
    attention_weight=attn_outputs.attention_weight,
    cache_view=attn_outputs.cache_view,
+   captured_kv=attn_outputs.captured_kv,
)

# Site 4 (Gemma4TextModel layer loop at lines 2604-2608):
- captured = getattr(attn, "_captured_kv", None)
- if captured is not None and not attn.is_kv_shared_layer:
-     shared_kv[idx] = captured
-     object.__setattr__(attn, "_captured_kv", None)
+ if layer_outputs.captured_kv is not None and not attn.is_kv_shared_layer:
+     shared_kv[idx] = layer_outputs.captured_kv
      donor_cache_views[idx] = layer_outputs.cache_view
```

**Why this is correct:**
- Captured K/V flow: attention local var → `AttentionLayerOutput.captured_kv` → `DecoderLayerOutput.captured_kv` → model loop reads `layer_outputs.captured_kv`
- Loop is unrolled (`for idx, block in enumerate(self.layers)`) — at trace time, each iteration's tracer is stored in Python `dict`, next iteration reads it. JAX threads dependencies through.
- KV-sharing path (`__call__` when `shared_key_value is not None`, line 1462+) returns at `:1551` without populating `captured_kv` — donor doesn't double-capture.

**Cross-model impact**: 13 AttentionLayerOutput + 24+ DecoderLayerOutput callers. All use kwargs (verified via grep). No functional change for non-Gemma4 models — field defaults to None.

### -1.B Scatter batch-leak fix — DESIGN PROPOSAL

**Bug at** `modeling_gemma4.py:3145-3158` — confirmed:
- `cumsum(special_mask.reshape(-1))` is global across flattened batch
- `features_flat = features.reshape(-1, D)` flattens all batches
- B1's first placeholder gets `cumsum_global - 1` which indexes into B0's row range

**Proposed fix**: per-row compaction via `jax.vmap` over batch axis:

```python
def _scatter_one_row(input_ids_b, inputs_embeds_b, features_b, valid_mask_b, token_id):
    # input_ids_b: (L,), inputs_embeds_b: (L, D), features_b: (T, D), valid_mask_b: (T,)
    placeholder_mask = (input_ids_b == token_id)               # (L,)
    n_placeholders = placeholder_mask.sum()                    # scalar tracer
    # Pre-pack valid features to the front (preserve order, mask out invalid)
    valid_idx = jnp.cumsum(valid_mask_b.astype(jnp.int32)) - 1  # (T,)
    packed_features = features_b * valid_mask_b[:, None]        # (T, D), invalid → 0
    # Per-row gather using local cumsum
    row_cumsum = jnp.cumsum(placeholder_mask.astype(jnp.int32)) - 1  # (L,)
    gather_idx = jnp.where(placeholder_mask, row_cumsum, 0)
    gathered = packed_features[gather_idx]                     # (L, D)
    # Scatter: replace embeddings only at placeholder positions
    return jnp.where(placeholder_mask[:, None], gathered, inputs_embeds_b)

scattered = jax.vmap(_scatter_one_row, in_axes=(0, 0, 0, 0, None))(
    input_ids, inputs_embeds, features, valid_mask, token_id
)
```

**Per-row count assertion**: cannot be a Python `assert` (would force trace evaluation). Use `jax.experimental.checkify` or fail-soft + return diagnostic. Default: bake the count check into a Python-level pre-flight before jit on real input shapes.

### -1.C VLM tied LM head — DESIGN PROPOSAL

**Bug at** `modeling_gemma4.py:3432`: `Gemma4ForConditionalGeneration.apply_lm_head` calls `super().apply_lm_head(hidden_states)`, which goes through the generic untied path. Text-only `Gemma4ForCausalLM.apply_lm_head` at `:2774-2779` explicitly avoids this:

```python
# :2774-2779 (text-only)
if getattr(self.config, "tie_word_embeddings", False):
    return self.get_embedding().attend(hidden_states)
return super().apply_lm_head(hidden_states)
```

**Proposed fix at `:3432`**:
```python
def apply_lm_head(self, hidden_states):
    if getattr(self.config.text_config, "tie_word_embeddings", False):
        return self.get_embedding().attend(hidden_states)
    return super().apply_lm_head(hidden_states)
```

Verified safe:
- `self.config.text_config.tie_word_embeddings` already accessed at `:3326` → field exists
- `self.get_embedding()` returns `self.language_model.get_embedding()` → `embed_tokens` (text only) at `:3281-3283 → :2640-2642`. `.attend()` on the text embedding is the right operation.

### -1.D Bug D `_causal_baked` impurity — LOW PRIORITY

`object.__setattr__(mask_info_full, "_causal_baked", True)` at `:2542-2543`. The flag is a static bool, not an Array. Under jit at trace time the bool is read by `getattr(mask_info, "_causal_baked", False)` at `:1347, :1486` and the branch is selected and baked into the trace. Functionally correct but bypasses pytree validation.

**Fix when convenient**: add `causal_baked: bool = False` as a real field on the mask_info type (`MaskInfo`), then `mask_info.replace(causal_baked=True)`. Defer until Phase 0.x or post-TPU-smoke.

## Phase 0–D unchanged from v2

(See Plan v2 in main session log. Phase 0 calibration, Phase A HF baseline, Phase B TPU parity, Phase C smoke decode, Phase D converter. Quality gates table unchanged.)

---

## Round 3 attack questions for Codex

1. **§-1.A landed diff**: Walk the captured_kv lifecycle through one full layer iteration under `jax.jit`. Does the unrolled-loop pattern with Python-dict storage work? Specifically: at iteration `idx`, `shared_kv[idx] = layer_outputs.captured_kv` stores tracers; at iteration `idx+1`, `shared_kv.get(attn.kv_shared_layer_index)` retrieves them. Does this compose correctly when the loop is jit-traced as a single graph? Are there any second-order issues with `auto_pytree` flattening of `tuple[Array, Array]` inside `AttentionLayerOutput`?

2. **§-1.B vmap design**: Does `jax.vmap` over the batch axis compose with the rest of `compute_embedding` (`:3038-3116`)? Specifically: `_scatter_one_row` returns `(L, D)`, vmapped to `(B, L, D)` matching `inputs_embeds`. But the cumsum count `n_placeholders` is dynamic; does `jnp.where(placeholder_mask[:, None], gathered, inputs_embeds_b)` handle the case where `n_placeholders > T_max_features` (edge: more placeholders than features) without leaking?

3. **§-1.C tied attend**: Verify `self.get_embedding().attend(hidden_states)` is correct for the VLM. `attend` is presumably `embed_tokens.attend` — what's its semantics under tensor parallelism? Does it differ from `super().apply_lm_head` only in the TP-safe path, or are there other differences (bias, normalization)?

4. **Bug D severity**: Confirm or refute "_causal_baked is a static bool, baked into the jit trace, functionally correct". Specifically: if `mask_info` is `@auto_pytree`, does the un-flatten step preserve `_causal_baked` as a non-field attribute? If `mask_info` is reconstructed during checkpoint-recovery or during a 2nd jit invocation with different shapes, is the flag lost?

5. **Hidden-dep audit on `compute_embedding` and `_compute_per_layer_inputs`**: Empirical fact-finder reported these are clean (only `self.config.*` reads). Independently verify and surface anything that mutates self-state during forward.

6. **Validator scope creep**: Round 2 raised "extending the validator to full-model is more work than using the converter." Now that -1.A is landed, does the converter strategy in `docs/parity/e4b_converter_strategy.md` need any updates for the new `captured_kv` field (none expected since it's a runtime-only field, not in checkpoint)? Confirm.

7. **Phase 0.0 empirical floor on Mac**: Realistic? `jax-on-CPU-FP32` floors will be tighter than `jax-on-TPU-BF16`. Does it make sense to gate B1/B2/B3 thresholds on CPU-FP32 floors, or do we need a TPU-BF16 calibration run first (chicken-and-egg with the validator)?

8. **What's still missing?** Specifically: any other shared-state pattern across layers that I haven't found (RoPE freq cache mutation? Mask cache?). Independently inspect `Gemma4Attention.concatenate` and `attention_performer.forward` for hidden state mutation.

Cite file:line for every code claim. Hard cap: 1300 words.
