# End-to-End Audio Goldens — Design Memo

Source: Codex audit task `aabfef7a3f8c8d497`, 2026-04-25. All HF citations
reference the version pinned in `.venv/lib/python3.13/site-packages/transformers/models/gemma4/modeling_gemma4.py`.

## Tier 2: Post-Projection
- Input: same Tier 1 deterministic mel fixture from `capture_gemma4_audio_golden.py:178-204` — 10 s waveform → `input_features` + `input_features_mask` from the HF feature extractor. Isolates `embed_audio` because HF runs audio tower then projector in `get_audio_features` (`modeling_gemma4.py:2328-2329`).
- Output shape/dtype: `pooler_output` from `Gemma4MultimodalEmbedder.embedding_projection` (`modeling_gemma4.py:1941-1952`). For the existing E4B 10 s fixture: projected audio `(1,250,2560)` plus encoder mask `(1,250)` (`test_gemma4_audio_golden_shapes.py:50-54`). Store as float32 `.npz` matching the existing serializer (`capture_gemma4_audio_golden.py:117-129`).
- HF capture call:
```python
audio_out = model.model.get_audio_features(input_features, input_features_mask, return_dict=True)
projected = audio_out.pooler_output
projected_mask = audio_out.attention_mask
```
(`modeling_gemma4.py:2310-2329`)

## Tier 3: Full-Stack
- Input: one deterministic 4 s audio chunk through the HF feature extractor. Text `Transcribe:` plus a contiguous block of `config.audio_token_id` placeholders whose count equals `audio_out.attention_mask.sum()` (NOT one token). HF strips padded audio features (`modeling_gemma4.py:2240-2245`) and validates placeholder count matches feature count before scatter (`modeling_gemma4.py:2247-2257`).
- Output: greedy 1-step prefill logits — `outputs.logits` with `logits_to_keep=1`, shape `(1,1,vocab_size)`. Greedy first-token logits over N-step decode because HF forward returns logits directly (`modeling_gemma4.py:2444-2448, 2475-2478`); N-step adds cache/generation drift orthogonal to scatter parity.
- Tolerance: `atol=1e-3, rtol=1e-3`. Tier 1 uses 1e-4 (`parity_gemma4_audio.py:76-77`); Tier 3 adds language attention masks, RoPE, LM layers, and logit softcapping (`modeling_gemma4.py:2265-2297, 1116-1122, 2448-2451`), each contributing FP rounding.
- HF capture call:
```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                input_features=input_features, input_features_mask=input_features_mask,
                logits_to_keep=1, use_cache=False, return_dict=True)
logits = outputs.logits
```
(`modeling_gemma4.py:2396-2414, 2425-2448`)

## HF Forward Path (cited)
Real entry: `Gemma4ForConditionalGeneration.forward(input_ids, input_features, input_features_mask, ...)` — not `audio_features` or `audio_token_id_mask`.

- `modeling_gemma4.py:2396-2414` — `Gemma4ForConditionalGeneration.forward` signature.
- `modeling_gemma4.py:2425-2442` — wrapper forwards to `self.model`.
- `modeling_gemma4.py:2123-2127, 2182-2190` — `Gemma4Model` derives `audio_mask` from `input_ids == config.audio_token_id`, replaces multimodal ids with pad ids, embeds text.
- `modeling_gemma4.py:2237-2257` — audio branch calls `get_audio_features`, strips invalid encoder positions, validates placeholder count, `masked_scatter`s into `inputs_embeds`.
- `modeling_gemma4.py:2328-2329` — `get_audio_features` runs `audio_tower(...)` then `embed_audio(...)`.
- `modeling_gemma4.py:2288-2297, 2444-2451, 2475-2478` — merged embeddings → language model → `lm_head` → optional softcap → logits.

## Validator Integration
- Choice: A — extend `parity_gemma4_audio.py` with `--tier {1,2,3}`. The file already owns golden loading, audio weight mapping, projector validation, tolerance reporting, `projector_out` comparison (`parity_gemma4_audio.py:424-490`). Add lazy full-model loading only for Tier 3 so Tier 1/2 stays fast without language weights. Separate file would duplicate infrastructure with no isolation benefit.

## Top 3 Risks
1. **Placeholder/mask mismatch.** HF strips audio features by encoder mask before scatter and checks exact element count (`modeling_gemma4.py:2240-2257`); EasyDeL's scatter assumes aligned placeholder counts (`modeling_gemma4.py:3058-3064, 3119-3159`). If the dummy input produces a different valid-frame count across frameworks the golden is captured under mismatched conditions.
2. **BF16/FP32 accumulation drift.** HF RoPE computes float32 then casts back (`modeling_gemma4.py:1116-1122`); logits are not upcast unless loss is computed (`modeling_gemma4.py:2444-2456`). JAX may accumulate attention in BF16 by default on TPU, shifting Tier 3 first-token logits by more than 1e-3.
3. **RoPE base / scaling conventions.** Text layers apply per-layer RoPE after scatter (`modeling_gemma4.py:1595-1610`); causal mask is built from scattered position indices, not original token positions. Any difference in RoPE theta, scaling factors, or position-id construction between HF and the JAX port dominates Tier 3 error and cannot be absorbed by `atol=1e-3`.
