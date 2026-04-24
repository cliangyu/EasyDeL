# Gemma 4 E4B audio golden fixtures

These `.npz` files are the reference activations for the HuggingFace
`Gemma4AudioModel` (USM-style conformer) on a deterministic 10 s / 16 kHz
synthetic waveform. The EasyDeL JAX port of the audio tower is diffed
against these tensors at every layer to catch numerical drift.

## Producing the fixtures

From the fork root:

```bash
# Option A — full model load (needs ~16 GB RAM in bf16; use a TPU VM):
python tests/fixtures/capture_gemma4_audio_golden.py \
    --model-id google/gemma-4-E4B --num-blocks 3 --dtype bfloat16

# Option B — audio-only partial load (~600 MB, fits on a 24 GB MacBook):
python tests/fixtures/capture_gemma4_audio_golden.py \
    --audio-only --model-id google/gemma-4-E4B --num-blocks 3 --dtype bfloat16
```

The script seeds every RNG, disables non-deterministic kernels, and writes
a `meta.json` with the `transformers` / `torch` versions and per-file
sha256s so we can detect silent drift when the reference library updates.

## Tolerance contract

The JAX port is considered parity-passing when, for each captured tensor:

| artifact                 | atol    | rtol | rationale                    |
|--------------------------|---------|------|------------------------------|
| `sscp_out`               | `1e-5`  | `0`  | dense conv, linear op        |
| `audio_block_{i}_out`    | `1e-4`  | `0`  | softmax / norm               |
| `audio_tower_out`        | `1e-3`  | `0`  | accumulated over ~32 layers  |
| `projector_out`          | `1e-3`  | `0`  | linear of the above          |

Tolerances mirror the EasyDeL convention used in
`tests/modules/test_utils/comparators.py`.

## Not committed to git

Only the **script** and this README live in git. The `.npz` artifacts are
regenerated on demand (they are deterministic) and are too large
(>100 MB total) to belong in the source tree — see `.gitignore`.
