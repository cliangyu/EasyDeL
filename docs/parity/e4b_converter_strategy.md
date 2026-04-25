# Gemma 4 E4B HF→EasyDeL Converter — Strategy Memo

Source: Codex audit task `afc9cc77ef780e626`, 2026-04-25. All paths relative to repo root unless absolute. Citations point to specific file:line.

## 1. Reuse vs. new code

EasyDeL has a generic HF→EasyDeL sequential converter:
- Entrypoint script: `easydel/scripts/convert_hf_to_easydel.py:68-90, 391-404` — selects `gemma4` as `image_text_to_text`, calls `EasyDeLBaseModule.huggingface_to_easydel_sequential`.
- Implementation: `bridge.py:2269-2299` — streams shards and writes TensorStore without full param materialization.
- Generic mapper: `parameters_transformation.py:241-264` — applies embedding, LayerNorm, rank-based `.weight → .kernel`.

**Strategy: reuse loader + output path, add Gemma4-specific resolver.** The validator's resolver (`parity_gemma4_audio.py:226-239, 262-288`) already resolves against actual NNX state paths — extend that pattern instead of building a parallel registry.

## 2. Mapping coverage

| Family | Path examples | Status | Action |
|---|---|---|---|
| Audio tower | `model.audio_tower.*`, `model.embed_audio.*` | Covered by validator | Reuse |
| Language embed | `language_model.embed_tokens.weight`, `embed_tokens_per_layer.weight` | `parameters_transformation.py:241-242` | Add per-layer mapping |
| Decoder attn/MLP | `q/k/v/o_proj.weight`, `gate/up/down_proj.weight` | `parameters_transformation.py:247-264` | Generic transpose-to-kernel |
| RMSNorm | `*_layernorm.weight` | Maps to `.kernel` (NOT `.scale`) | Confirm |
| Per-layer gates | `layer_scalar`, etc. (`modeling_gemma4.py:2044-2064`) | Identity | Add identity rule |
| RoPE | n/a | Computed from config (`modeling_gemma4.py:2351-2447`) | Verify no checkpoint key |
| Vision patch | `input_proj`, `position_embedding_table`, ClippableLinear q/k/v/o | `modeling_gemma4.py:333-547, 624-657, 690-701, 919-931` | Add nested-`.linear` rule |
| `embed_vision` | `model.embed_vision.*` | `modeling_gemma4.py:2830-2884` | Mirror `embed_audio` rules |
| `lm_head` | `lm_head.weight` | `modeling_gemma4.py:3322-3327` | Tied or retained — verify |

## 3. Sharded loading

HF shard discovery via `model.safetensors.index.json`:
- EasyDeL already loads index, reads `weight_map`, maps each key to shard (`bridge.py:2544-2602`).
- Streaming pattern: group keys per filename → resolve/download shard → `safe_open(..., device="cpu")` → convert each tensor → write immediately → release shard (`bridge.py:2700-2725, 2755-2858, 2887`).

No host RAM blowup; works for E4B's multi-shard layout.

## 4. Output format

EasyDeL TensorStore/Zarr checkpoint (`bridge.py:2641-2678`):
- Arrays written under `model/<key path>` with TensorStore index + checkpoint metadata.
- Matches the maintained large-checkpoint path (`bridge.py:2269-2299`).

## 5. Entrypoint signature

```python
def convert_gemma4_e4b(
    hf_repo_or_path: str,
    output_dir: str,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    sharding_axis_dims: tuple[int, ...] = (1, -1, 1, 1, 1),
) -> Path:
```

Default `sharding_axis_dims` mirrors `convert_hf_to_easydel.py:258-264`. Pass through to existing converter.

## 6. Validation hook

Current validator loads raw safetensors via `--safetensors` (`parity_gemma4_audio.py:141-144, 362-378`). For post-conversion validation:

1. Reuse `AudioParityRoot`.
2. Populate state from TensorStore instead of `_load_audio_weights`.
3. Run the same touched-state / mapping gate and audio trace/diff (`parity_gemma4_audio.py:117-134, 395-408, 424-491, 501-551`).

Tier 3 VLM/logit parity is not in this validator; current top-level forwards exist for full model comparison (`modeling_gemma4.py:3196-3267, 3453-3529`).

## 7. Top 3 risks

1. **Audio resolver gaps.** Square linear kernels, depthwise Conv1d, and clamp `nn.Variable` leaves can silently map wrong unless the validator's state-path and untouched checks remain hard failures (`parity_gemma4_audio.py:618-623`).
2. **Vision `.linear` nesting.** `Gemma4VisionClippableLinear` preserves nested `.linear` layout — stripping or double-stripping `linear` can misplace q/k/v/o and MLP weights (`modeling_gemma4.py:379-397, 494-537, 624-657`).
3. **Language v_proj aliasing.** Global attention may alias `v_proj = k_proj` (`modeling_gemma4.py:1191-1195`). If converter blindly maps both, we either duplicate weight or miss `v_proj` in coverage gate. **Must verify before running converter.**
