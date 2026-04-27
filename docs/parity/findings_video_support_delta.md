# Gemma 4 video-support delta (Codex audit, 2026-04-27)

Source: codex-rescue agent `a5f12c0326671a9da`. All file:line claims independently verified by Codex against HF transformers (model_type=gemma4) and EasyDeL `feat/gemma4-audio` HEAD.

## TL;DR

Video is NOT just stacked images. HF Gemma 4 has a dedicated video path
(`pixel_values_videos` + `video_position_ids`, separate `Gemma4VideoProcessor`,
distinct `<|video|>` token, frame-aware insertion of `<boi><|video|>*n<eoi>`).
EasyDeL config has `video_token_id` but no plumbing.

Smallest delta to enable video: ~150-200 LoC in `easydel/modules/gemma4/modeling_gemma4.py`. Single file, no infra changes.

## What HF does

- `Gemma4VideoProcessor` patchifies each `(F, C, H, W)` video into `(F, num_patches, patch_dim)` and stacks across the batch as `pixel_values_videos` + `video_position_ids` (`video_processing_gemma4.py:65-78,225-232`).
- `Gemma4ImageProcessor` is image-only — emits `pixel_values`, `image_position_ids`, `num_soft_tokens_per_image` (`image_processing_gemma4.py:93,208-215`). Different field names.
- `Gemma4Processor` adds `<|video|>` token; insertion replaces each `<|video|>` with `<boi><|video|>*n<eoi>` per frame, with timestamps (`processing_gemma4.py:83-86,170-192`).
- Model side: `pixel_values_videos` flow flattens video frames before the vision tower, then scatter into `video_token_id` slots (`modeling_gemma4.py:2155-2167,2217-2234,2335-2354`).

## What EasyDeL has today

- Config: `video_token_id` defined (`gemma4_configuration.py:662-688`).
- Forward + embedding + generation: only `pixel_values` and `image_position_ids` exposed (`modeling_gemma4.py:3048-3051,3267-3278,3546-3554`).
- Vision input normalization rejects rank-5 (`modeling_gemma4.py:133-136,276-295`).
- Scatter logic: `_scatter_features_at_token` exists and is general — same path could scatter video features at `video_token_id` (`modeling_gemma4.py:3130-3164`).

## Concrete delta (single file: `easydel/modules/gemma4/modeling_gemma4.py`)

1. Add `pixel_values_videos`, `video_position_ids` kwargs to:
   - `Gemma4Model.__call__`
   - `Gemma4ForConditionalGeneration.__call__`
   - `compute_embedding` (or wherever `pixel_values` is currently consumed) — branch into a `get_video_features` path mirroring HF (`HF modeling_gemma4.py:2155-2167,2335-2354`).

2. New `get_video_features` method:
   - Flatten `(num_videos, F, max_patches, patch_dim)` across axes 0-1.
   - Run existing vision tower.
   - Project via `embed_vision`.
   - Use existing `_scatter_features_at_token(... token_id=config.video_token_id)`.

3. `prepare_inputs_for_generation` cleanup: pop `pixel_values_videos` + `video_position_ids` after first decode step (currently only pops image/audio fields at lines 3625-3645).

4. Set `_supports_video = True` (currently `False` at `modeling_gemma4.py:3378`).

## Risks (silent / wrong-logits)

- **`mm_token_type_ids` vs `token_type_ids`**: HF processor returns `mm_token_type_ids` (`processing_gemma4.py:239-240`); EasyDeL expects `token_type_ids` (`modeling_gemma4.py:2532-2544,3553-3600`). Rename or add an adapter — otherwise bidirectional vision/video masking is silently absent.
- **Video placeholder masking**: Currently `<|video|>` tokens are only masked out of per-layer inputs (`modeling_gemma4.py:3237-3241`), NOT feature-scattered (no scatter at video_token_id today). After fix, scatter must run BEFORE the per-layer-input masking.
- **Position-id padding**: HF supplies padded `video_position_ids` (`video_processing_gemma4.py:226-232`); EasyDeL's flat-patch path uses position ids for attention (`modeling_gemma4.py:291-316,967-980`). If padded positions aren't honored, attention will leak across pad slots.

## Reuse hooks

- The audio scatter fix (`-1.B`, `_scatter_audio_features_at_token`) is the right pattern — vmap + per-row stable argsort + checkify. The video scatter should reuse the generic `_scatter_features_at_token` already in place; only the `token_id` arg changes.
- Gemma 4's vision tower is unchanged for video; it processes flattened frames as if they were images.

## Estimate

~2-3 days focused implementation + parity test against HF goldens. Not 5-10 days as previously feared.
