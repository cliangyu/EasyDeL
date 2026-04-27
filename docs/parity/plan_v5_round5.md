# Gemma4 Audio Port — Plan v5 (Round 5 dispatch input)

Source: Codex Round 4 critique (agent `a663d0157275d8e0a`) + Claude verification.
Round 4 partially refuted v4's Bug E mechanism, surfaced **new Bug -1.F**, and revealed v4's `write_cache=False` was incomplete.
All paths relative to repo root unless absolute. All claims cite `file:line`.

## What's new since v4

1. **Bug E mechanism corrected**: v4 said "donor's slot k+1 overwrites what shared wrote." Codex Round 4 found shared writes are *silently dropped* (the shared layer's mutation lives in a local variable, never flows back to `past_key_values`). Same blocker, cleaner mechanism. Refute summary at `plan_v4_round4.md` Q1.
2. **New Bug -1.F discovered**: `write_cache=False` alone in v4's design is insufficient. `_handle_cache_concat` at `_flexible.py:1191-1208` is the **only path** that materializes full K/V history AND updates mask lengths from `cache_view.indexs`. Skipping it leaves shared attention with current-token K/V only. Need a read-only prepare path that reads cache_view + reconstructs mask without writing.
3. **-1.B clarified**: `jnp.argsort(stable=True)` is documented as `is_stable=true` in OpenXLA Sort spec; CPU/TPU XLA delegates accordingly. Confidence raised to 95%. Verify on TPU once Phase 0.0 calibration runs.
4. **-1.B assert**: Python `assert` over JAX-traced values fires at *trace time only* — at runtime, mismatched data bypasses the check silently. **Fix**: wrap the placeholder/feature count check in `jax.experimental.checkify.check` for in-graph runtime assertion.
5. **-1.C numeric**: `Embed.attend(query)` calls `jnp.dot(query, embedding.T)` without a precision argument at `easydel/layers/embeddings/_embeddings.py:226-227`; base tied path at `_linear.py:292-306` passes `precision=self.precision`. **Mathematically equivalent, not bitwise**. Phase 0.8 parity test will surface any drift.

## Phase -1: port-level fixes (all gates required before TPU work)

### -1.A KV-share side channel — LANDED in `cc056b25`

(see plan_v3_round3.md). Unchanged.

### -1.B Scatter batch-leak fix — argsort + checkify

**Bug** at `modeling_gemma4.py:3149-3163`: global cumsum across flattened batch leaks features across rows. Confirmed.

**Fix** (per-row compaction via stable argsort, per-row gather, in-graph checkify):

```python
from jax.experimental import checkify

def _scatter_one_row(input_ids_b, inputs_embeds_b, features_b, valid_mask_b, token_id, T):
    # 1. Compact valid features to the front (stable argsort).
    valid_int = valid_mask_b.astype(jnp.int32)
    sort_idx = jnp.argsort(-valid_int, stable=True)             # (T,)
    packed_features = features_b[sort_idx]                       # (T, D)

    # 2. Per-row placeholder positions.
    placeholder_mask = (input_ids_b == token_id)                 # (L,)
    n_placeholders = placeholder_mask.sum()
    n_valid = valid_int.sum()

    # 3. In-graph runtime assertion (replaces broken Python assert).
    checkify.check(
        n_placeholders == n_valid,
        "Per-row placeholder count must equal per-row valid feature count "
        "(got placeholders={p}, valid={v})",
        p=n_placeholders, v=n_valid,
    )

    # 4. Bounds-safe gather.
    row_cumsum = jnp.cumsum(placeholder_mask.astype(jnp.int32)) - 1   # (L,)
    gather_idx = jnp.where(placeholder_mask, jnp.minimum(row_cumsum, T - 1), 0)
    gathered = packed_features[gather_idx]                       # (L, D)

    # 5. Scatter only at placeholder positions.
    return jnp.where(placeholder_mask[:, None], gathered, inputs_embeds_b)


# Caller (compute_embedding) — wrap in checkify if not already in graph:
B, T, D = features.shape
checked_fn = checkify.checkify(
    jax.vmap(_scatter_one_row, in_axes=(0, 0, 0, 0, None, None))
)
err, inputs_embeds_new = checked_fn(
    input_ids, inputs_embeds, features, audio_output_mask, audio_token_id, T
)
err.throw()  # raises if any row's count check failed
```

**Why this is correct**:
- v4 used Python `assert` — Codex confirmed that fires at trace time only, not at runtime.
- `checkify.check` injects a runtime check that doesn't break jit and throws a Python error after execution if the check fails.
- Argsort-stable compaction unchanged from v4.

**Open question for Codex**: Does `checkify` impose meaningful TPU overhead in the hot path, or is it negligible (single boolean reduce per row)?

### -1.C VLM tied LM head — TWO methods (both `apply_lm_head` and `make_lm_head_fn`)

(Unchanged from v4 — see `plan_v4_round4.md` lines 90-135. Codex Round 4 confirmed equivalence "mathematically equivalent, not bitwise" — Phase 0.8 parity test will detect any drift.)

### -1.D `_causal_baked` impurity — DEFERRED (low priority)

Round 4 watch-out: `mixins/generation.py:3063-3069` re-propagates the static-bool flag after `apply_kv_lengths`. Inspect during Phase B/C; clean up post-smoke.

### -1.E + -1.F COMBINED: Shared-layer read-only attention path

**Bug E (corrected mechanism)**: shared layer borrows donor's `cache_view`, calls `concatenate` again at `:1503-1519`, calls `concatenate_to_cache` (writes to slot, advances `indexs`), but the resulting view is bound only to a local var. Shared layers fail the `not attn.is_kv_shared_layer` guard at `:2617-2618`, so the shared layer's writes are *silently dropped*. Donor's view in `past_key_values[N]` is untouched (Python-reference semantics: `past_key_values[idx] = layer_outputs.cache_view` was set at the *donor's* iteration with the donor's then-current view). Net effect: shared layer's attention is computed using a single-token window of K/V (just the current token), not the full history.

**Bug -1.F (new, from Codex Round 4)**: `_handle_cache_concat` at `_flexible.py:1191-1208` is the only path that:
1. Reads `cache_view.key`/`cache_view.value` to assemble full K/V history.
2. Updates `mask_info.kv_lengths` from `cache_view.indexs`.

If we skip the call entirely (v4's `write_cache=False` design), shared attention sees only the current-token projection, not history. Wrong but undetected by current tests.

**Fix** — read-only prepare path. Add `write_cache: bool = True` kwarg. When `False`:
- Skip `concatenate_to_cache` (no `indexs` advance, no K/V write).
- Read full history from `cache_view.key`/`cache_view.value`.
- Apply mask lengths from `cache_view.indexs` to `mask_info`.

Concrete diff (proposed by Codex, validated by Claude):

```python
# easydel/layers/attention/_flexible.py — concatenate signature + handler:
def concatenate(
    self,
    *,
    query, key, value,
    cache_view, cache_metadata, mask_info,
    sliding_window=None,
    write_cache: bool = True,   # NEW
) -> tuple[...]:
    ...
    if write_cache:
        key, value, mask_info, cache_view, _md = self._handle_cache_concat(
            query=query, key=key, value=value, mode=mode,
            mask_info=mask_info, cache_view=cache_view, cache_metadata=cache_metadata,
        )
    else:
        # Read-only: full history without writing.
        # cache_view.key/value already hold history through indexs-1.
        key   = cache_view.key.astype(key.dtype)
        value = cache_view.value.astype(value.dtype)
        if mask_info is not None and hasattr(cache_view, "indexs"):
            mask_info = mask_info.apply_kv_lengths(
                kv_lengths=cache_view.indexs,
                q_len=query.shape[1],
                end_index=cache_view.indexs,
            )
    return key, value, mask_info, init_attention_bias, cache_view, cache_metadata
```

```python
# easydel/modules/gemma4/modeling_gemma4.py:1503-1519 (shared path):
if cache_view is not None:
    (key_states, value_states, mask_info, init_attention_bias,
     cache_view, cache_metadata) = self.concatenate(
        query=query_states, key=key_states, value=value_states,
        cache_view=cache_view, cache_metadata=cache_metadata,
        mask_info=mask_info, sliding_window=sliding_window_for_kernel,
        write_cache=False,   # shared layer reuses donor's writes
    )
```

**Cross-model impact**: optional kwarg defaulting to `True` is backward-compatible. All existing callers behave unchanged.

**Verification plan**: integration test under jit with `num_kv_shared_layers=1`, decode 4 tokens, assert `cache_view.indexs == 4` after generation (NOT 8) AND attention output matches HF reference. Add to Phase 0.x.

**Open questions for Codex Round 5**:
- Does `cache_view.key` / `cache_view.value` actually hold the full padded buffer through `indexs - 1`? Or only the slice up to the most recent write? Verify by reading `easydel/caching/transformer/cache.py:704-733`.
- Does `mask_info.apply_kv_lengths` mutate or return-new? If mutate, we'd need to `replace`. The `_handle_cache_concat` path calls it via the same kwargs, so behavior should match.
- Does the `ragged_page` cache backend (`ragged_page/cache.py:930`) require the same read-only treatment? Confirm by checking `_handle_cache_concat`'s mode dispatch at `_flexible.py:1135-1190`.

### -1.G (NEW from Codex Round 4 cross-deps): None-cache guard for shared layers

Codex flagged: shared layers are allocated as `None` at `easydel/infra/mixins/generation.py:2463-2468` and `:2664-2668`. The `write_cache=False` path must early-out when `cache_view is None`.

Already covered by the `if cache_view is not None:` guard at `modeling_gemma4.py:1502`. **No code change needed**, but add a unit test: shared layer at first decode step with no cache → behaves like donor's first call.

## Phases 0–D unchanged from v3

### New Phase 0 sub-tests (added by Round 4)

- **0.6** KV-share decode integration: `num_kv_shared_layers=1`, decode 4 tokens, assert `indexs == 4`. Validates -1.E + -1.F.
- **0.7** Audio scatter batch>1 unequal-length: B=2, T=64, valid={50, 30}; assert per-row scatter correct + checkify trips on count mismatch.
- **0.8** VLM tied LM-head parity: compare `Gemma4ForCausalLM.apply_lm_head(h)` vs `Gemma4ForConditionalGeneration.apply_lm_head(h)` on identical config + tied weights.

## Phase 0.0 floor strategy (Codex Round 4 confirmed)

CPU-FP32 floors gate logic only. Final B1/B2/B3 thresholds bind to **a calibration matrix** (multiple shapes/seeds) of TPU-BF16 runs after first successful Tier 1 — NOT a single run. v4 single-run binding was naive.

## Sequencing decision (Round 5)

- **-1.A**: LANDED.
- **-1.C** (LM head, ~30 LoC): independent. Lowest blast radius, ship first.
- **-1.B** (scatter, +checkify): independent. Ship second.
- **-1.E + -1.F** (read-only cache prepare): touches `_flexible.py` base infra. Highest blast radius. Ship last.

**Recommended order**: -1.C → -1.B → -1.E+F. Each commits separately; Codex reviews each diff before next starts.

## Round 5 attack questions for Codex

1. **-1.F read-only path correctness**: Does `cache_view.key.astype(key.dtype)` give the full padded history (size `cache_max_length`) with valid entries through `indexs - 1` and zeros after? Or does it require explicit slicing/masking? Cite `transformer/cache.py:704-733` and `ragged_page/cache.py:920-940`.

2. **-1.F mask_info.apply_kv_lengths**: Read-only branch calls it with `kv_lengths=cache_view.indexs, q_len=query.shape[1], end_index=cache_view.indexs`. The write-path in `_handle_cache_concat` calls it with `kv_lengths=cache_view.indexs + q_len` (after the write). Which is correct for the shared layer — should it see the *post-write* `indexs` of the donor (i.e., the donor's current step has been incorporated), or the *pre-write* `indexs` (the previous step's end)?

3. **-1.B checkify overhead**: Is `checkify.check` a no-op when wrapped in a hot decode loop, or does it add per-step overhead on TPU? Cite JAX docs or experiment.

4. **-1.E+F decode integration test design**: Does our test need to (a) hit both transformer and ragged_page cache backends, or (b) is one sufficient since Gemma4's prefill+decode uses only one path? Check `modeling_gemma4.py` for `cache_metadata.cache_mode` selection.

5. **Bug E + Bug -1.F joint correctness**: After applying both -1.E (`write_cache=False`) and -1.F (read-only branch), does the donor's K/V written at iteration N flow correctly to the shared layer at iteration N+1 within the same decode step? Specifically: the shared layer at N+1 reads `cache_view.key` from the donor's borrowed view — is the donor's just-written entry visible there?

6. **-1.G shared layer first-token init**: Round 4 flagged shared cache allocated as None. At decode step 0, `cache_view` is None for the shared layer; the `if cache_view is not None` guard short-circuits the read-only path. But the shared layer still needs to compute attention over the donor's just-projected K/V. Trace: at iteration N (donor), `cache_view` is allocated and concatenate writes; at iteration N+1 (shared), the `if cache_view is not None` guard at `:1502` evaluates against `donor_cache_views.get(...)` which has the donor's view. Correct?

7. **Compaction + scatter on TPU**: vmap of argsort over batch dim + scatter — does this generate efficient TPU code, or does it inhibit fusion? If inhibited, alternative design: process all rows at once with a 2D argsort + 2D gather. Worth exploring before implementation?

8. **Anything I'm still missing**: Independently re-inspect:
   - `Gemma4Attention.__call__` post-attention output processing (`:1521-1546`) — any state leaks?
   - The donor-cache-view `donor_cache_views[idx]` write at `:2610-2613` — does this happen only when `idx == kv_shared_layer_index`? Check the sharing topology (which layer is donor for which shared layer).
   - Cross-device behavior: under multi-host, does `cache_view` actually represent a sharded array? If so, does the read-only path's `astype` trigger an undesirable resharding?

Cite file:line for every code claim. Hard cap: 1500 words.

---

## Plan-v5 confidence summary

| Bug | Status | Risk if wrong |
|---|---|---|
| -1.A KV-share side channel | LANDED in `cc056b25` | None |
| -1.B scatter batch-leak | DESIGN FINAL (argsort+checkify) | Med — wrong audio routing |
| -1.C VLM tied LM head | DESIGN FINAL (mirror text-only) | Low — slight numeric drift |
| -1.D `_causal_baked` | DEFERRED | Low — code smell |
| -1.E+F shared-layer cache | DESIGN PROPOSED (read-only path) | **HIGH — silently breaks decode** |
| -1.G None-cache guard | NO CODE CHANGE (existing guard suffices) | Low |

**Phase -1 status**: 1 landed, 4 with concrete designs, 1 deferred. Round 5 must validate -1.E+F before any code lands.
