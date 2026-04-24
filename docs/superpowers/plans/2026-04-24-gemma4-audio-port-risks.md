# Gemma 4 E4B Audio Tower Port — Risk Landscape & Port Order

> Captured 2026-04-24. Before patching anything, triangulate against HF (the
> oracle, now frozen as `.npz` fixtures on `feat/gemma4-audio-golden-fixtures`).
> Don't hurry to a fix — enumerate first, discuss, then plan.

## Oracle

- **HF class:** `transformers.models.gemma4.modeling_gemma4.Gemma4AudioModel`
- **Frozen reference tensors:** PR #1, `tests/fixtures/gemma4_audio_golden/*.npz`
  (SSCP stem, first N conformer blocks, full tower, projector). Shapes pinned
  by `tests/fixtures/test_gemma4_audio_golden_shapes.py`.

## Shapes locked in

| Tensor | Shape |
|---|---|
| `input_features` | `(1, 999, 128)` mel-spec |
| `sscp_out` | `(1, 250, 1024)` after 4× subsampling |
| `audio_tower_out.last_hidden_state` | `(1, 250, 1536)` |
| `projector_out` | `(1, 250, 2560)` into LM space |

## EasyDeL current state

- `easydel/modules/gemma4/` has text + vision fully ported (Flax/NNX).
- **Audio model is stub only.** `Gemma4Config` accepts `audio_config` dict
  but does not instantiate an audio tower.
- `Gemma4MultimodalEmbedder` exists — ready to accept a 1536-dim projector input.
- Reusable: `Gemma4RMSNorm` (`with_scale` flag), `Gemma4ClippableLinear` (vision).

## Risk landscape (enumerate, don't patch)

1. **Relative attention mask polarity.** HF uses `logical_not()` to flip
   boolean mask before `masked_fill`; JAX attention conventions differ. Silent
   reversal is plausible.
2. **Block reshape / padding boundaries.** Unfold + chunk + trim; off-by-one
   on seq_len after attention is easy.
3. **Hardcoded relative position range** (`arange(12, -1, -1)`, 13 positions).
   Long audio may fail.
4. **Manual `torch.clamp` at ±1e10** in attention input, FFN input, output.
   JAX must replicate — no autograd magic.
5. **fp32 upcast path.** QKV → fp32 → matmul → softcap → tanh → downcast.
   Precision regain must be matched.
6. **Softcap before mask.** `tanh(x/softcap)*softcap` is applied *before*
   logit replacement with -1e9. Order matters.
7. **Per-head softplus scale.** `q_scale * F.softplus(per_dim_scale)` where
   `per_dim_scale` init is zeros. `softplus(0) ≈ 1.31`, not 1.
8. **Conv mask downsampling.** `mask[:, ::2]` after each SSCP conv; asymmetric
   with hidden-state reshapes.
9. **Depthwise causal conv padding formula.** `left_pad = (k-1)*d - s + 1`.
   Kernel 5, dilation 1, stride 1 ⇒ left_pad = 3.
10. **Embedder shape routing.** Audio tower outputs 1536, LM hidden is 2560.
    Embedder: 1536 → RMSNorm(scale=False) → 2560.
11. **LayerNorm in SSCP vs RMSNorm in conformer.** Inconsistent norms.
12. **Attention validity mask threading.** Must survive every reshape and
    match post-trim seq_len.

## Port order (lowest risk → highest)

1. `Gemma4AudioConfig` dataclass (hermetic test first)
2. `Gemma4AudioRelPositionalEncoding` — pure math, no params
3. `Gemma4AudioFeedForward` — two ClippableLinears + activation + residual
4. `Gemma4AudioLightConv1d` — depthwise conv + GLU + causal padding
5. `Gemma4AudioSubSampleConvProjection` — 2D convs + LN + ReLU + mask
6. `Gemma4AudioAttention` — **highest risk** (rel_shift, softcap, chunk)
7. `Gemma4AudioLayer` — Macaron FFN + attn + light conv + FFN residual order
8. `Gemma4AudioModel` — full encoder stack + output projection
9. Multimodal embedder wiring in `Gemma4ForConditionalGeneration`
10. Weight-conversion utility (HF safetensors → EasyDeL params)
11. Parity test (JAX output vs `.npz` oracle, tol 1e-3 end-to-end)

## Tolerance contract (mirrors `tests/fixtures/README.md`)

- SSCP stem: `atol=1e-5`
- Per-conformer-block output: `atol=1e-4`
- End-to-end audio tower: `atol=1e-3`
- Projector output: `atol=1e-3`

Parity looser than 1e-3 on end-to-end is a bug until proven otherwise.

## Open questions (defer until Step 2 design)

- Do we need chunked attention for training, or is full-seq attention
  acceptable for our audio lengths (≤30s ⇒ seq_len 250)?
- Does the TRC v6e-8 memory budget allow materialising full rel-attn matrices,
  or must we port `_rel_shift`?
- Should the port target bfloat16 throughout with fp32 stability islands
  (matches HF), or fp32 throughout for first parity pass?
