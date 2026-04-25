# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate EasyDeL Gemma 4 audio parity against captured HF goldens.

This script is intentionally a checkpoint loader plus forward-diff harness,
not a generic converter. Audio parity is the fragile boundary for the
audio->video->RL chain, so the implementation optimises for inspectability and
early failure instead of hiding drift behind fuzzy key walking.

Plan critique baked into the implementation:

* Constructing the full ``Gemma4Model`` would allocate the language stack just
  to reach ``audio_tower`` and ``embed_audio``. A tiny NNX wrapper keeps those
  two attributes at the same state paths (``audio_tower/...`` and
  ``embed_audio/...``), avoids language weights, and still exercises the direct
  audio tower/projector calls this validator cares about.
* Resolving every PyTorch key by dynamically walking object attributes is
  risky: lists, NNX containers, and naming aliases make failures look like
  missing attrs rather than state mismatches. The loader first walks
  ``nn.state(root, nn.Variable)`` once and treats that flat state path map as
  the source of truth. Every HF key must resolve to exactly one path in that
  map, and every path in that map must be touched.
* Current Flax NNX in this repo rejects ``nn.Param | nn.Variable`` as a filter,
  while ``nn.Variable`` includes ``Param`` and non-Param array variables. The
  clippable linear bounds are therefore loaded through the same state map as
  params. Array updates use ``leaf[...] = value``; writing ``.value`` still
  works in older NNX but is deprecated here and emits warnings.
* Naming quirks are explicit: HF ``weight`` maps to JAX ``kernel`` for
  ``Gemma4RMSNorm`` and linear/conv kernels, to JAX ``scale`` for the SSCP
  ``LayerNorm``, and remains shape-identical for scalar clamp buffers,
  ``per_dim_scale``, and biases. Tensor layout transforms are keyed by the
  matched target path and rank, not by shape alone, so square linear kernels
  still transpose.
* Golden and checkpoint locations are CLI arguments. TPU VM paths are the
  common case, but the script has no baked-in VM assumption and accepts either
  a safetensors file or a directory of shards.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.gemma4.gemma4_configuration import Gemma4AudioConfig
from easydel.modules.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
from easydel.modules.gemma4.modeling_gemma4_audio import Gemma4AudioModel


ATOL = 1.0e-4
RTOL = 1.0e-4
AUDIO_PREFIXES = ("model.audio_tower.", "model.embed_audio.")
CLAMP_BUFFER_NAMES = {"input_min", "input_max", "output_min", "output_max"}


@dataclass(frozen=True)
class StateLeaf:
    path: str
    leaf: nn.Variable
    shape: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class CandidatePath:
    path: str
    resolver_rule: str


@dataclass(frozen=True)
class MappingRecord:
    pt_key: str
    jax_path: str
    resolver_rule: str
    transform: str
    pt_shape: tuple[int, ...]
    jax_shape: tuple[int, ...]
    leaf_kind: str


@dataclass(frozen=True)
class DiffRow:
    layer_name: str
    shape: str
    max_abs: float
    max_rel: float
    mean_abs: float
    finite_check: str


class AudioParityRoot(nn.Module):
    """Keep checkpoint paths stable without paying for the text model."""

    def __init__(self, audio_config: Gemma4AudioConfig, text_hidden_size: int, *, rngs: nn.Rngs):
        self.audio_tower = Gemma4AudioModel(
            config=audio_config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.embed_audio = Gemma4MultimodalEmbedder(
            multimodal_hidden_size=audio_config.output_proj_dims,
            text_hidden_size=text_hidden_size,
            rms_norm_eps=audio_config.rms_norm_eps,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden-dir", required=True, help="Directory containing Gemma 4 audio golden .npz files.")
    parser.add_argument(
        "--safetensors",
        required=True,
        help="Path to model.safetensors or a directory containing safetensors shards.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print the full HF safetensors -> NNX state map.")
    return parser.parse_args()


def _read_npz_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing golden file: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in preferred_keys:
            if key in data.files:
                return np.asarray(data[key]).copy()
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]]).copy()
        raise KeyError(f"{path} has keys {data.files}, none of preferred keys {preferred_keys}")


def _infer_text_hidden_size(golden_dir: Path) -> int:
    projected = _read_npz_array(golden_dir / "projector_out.npz", ("data", "0", "projected"))
    if projected.ndim < 1:
        raise ValueError("projector_out.npz must contain at least one dimension")
    return int(projected.shape[-1])


def _audio_config_from_golden(golden_dir: Path) -> Gemma4AudioConfig:
    kwargs: dict[str, object] = {}
    meta_path = golden_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta_audio = meta.get("audio_config") or {}
        signature = inspect.signature(Gemma4AudioConfig.__init__)
        allowed = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        kwargs.update({key: value for key, value in meta_audio.items() if key in allowed and value is not None})

    if "hidden_size" not in kwargs:
        sscp_hidden = _read_npz_array(golden_dir / "sscp_out.npz", ("0", "data", "last_hidden_state"))
        kwargs["hidden_size"] = int(sscp_hidden.shape[-1])
    if "output_proj_dims" not in kwargs:
        tower_hidden = _read_npz_array(golden_dir / "audio_tower_out.npz", ("last_hidden_state", "0", "data"))
        kwargs["output_proj_dims"] = int(tower_hidden.shape[-1])

    return Gemma4AudioConfig(**kwargs)


def _validate_inputs(input_features: np.ndarray, input_features_mask: np.ndarray, audio_config: Gemma4AudioConfig) -> None:
    if input_features.ndim != 3:
        raise ValueError(f"input_features must be rank 3 (B, T, F), got shape {input_features.shape}")
    if input_features_mask.shape != input_features.shape[:2]:
        raise ValueError(
            "input_features_mask must match input_features batch/time axes, "
            f"got mask {input_features_mask.shape} for features {input_features.shape}"
        )
    expected_mels = audio_config.subsampling_conv_channels[0]
    if input_features.shape[-1] != expected_mels:
        raise ValueError(
            "input_features must be time-major (B, T, F). "
            f"Expected F={expected_mels} from audio_config.subsampling_conv_channels[0], "
            f"got shape {input_features.shape}."
        )


def _load_inputs(golden_dir: Path, audio_config: Gemma4AudioConfig) -> tuple[np.ndarray, np.ndarray]:
    with np.load(golden_dir / "inputs.npz", allow_pickle=False) as data:
        if "input_features" not in data.files:
            raise KeyError("inputs.npz is missing input_features")
        mask_key = "input_features_mask" if "input_features_mask" in data.files else "attention_mask"
        if mask_key not in data.files:
            raise KeyError("inputs.npz is missing input_features_mask/attention_mask")
        input_features = np.asarray(data["input_features"], dtype=np.float32).copy()
        input_features_mask = np.asarray(data[mask_key]).astype(np.bool_, copy=True)
    _validate_inputs(input_features, input_features_mask, audio_config)
    return input_features, input_features_mask


def _state_path(path: tuple[object, ...]) -> str:
    return "/".join(str(part) for part in path)


def _build_state_map(root: AudioParityRoot) -> dict[str, StateLeaf]:
    state_map: dict[str, StateLeaf] = {}
    for path, leaf in nn.state(root, nn.Variable).flat_state():
        path_str = _state_path(path)
        value = leaf[...]
        state_map[path_str] = StateLeaf(
            path=path_str,
            leaf=leaf,
            shape=tuple(int(dim) for dim in value.shape),
            kind=type(leaf).__name__,
        )
    if not state_map:
        raise RuntimeError("NNX state map is empty; audio root did not register any Param/Variable leaves")
    return state_map


def _safetensor_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if shards:
            return shards
    raise FileNotFoundError(f"no safetensors file(s) found at {path}")


def _is_audio_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in AUDIO_PREFIXES)


def _strip_model_prefix(key: str) -> str:
    if not _is_audio_key(key):
        raise ValueError(f"not an audio safetensors key: {key}")
    return key[len("model.") :] if key.startswith("model.") else key


def _candidate_paths_for_pt_key(pt_key: str) -> list[CandidatePath]:
    stripped = _strip_model_prefix(pt_key)
    segments = stripped.split(".")
    candidates: list[CandidatePath] = []

    def add(parts: list[str], rule: str) -> None:
        path = "/".join(parts)
        if not any(existing.path == path for existing in candidates):
            candidates.append(CandidatePath(path=path, resolver_rule=rule))

    add(segments, "identity")

    if segments[-1] == "weight":
        # HF uses one leaf name for several semantics; the actual NNX state
        # path tells us whether this is a norm scale or a kernel.
        add([*segments[:-1], "kernel"], "weight->kernel")
        add([*segments[:-1], "conv", "kernel"], "weight->conv/kernel")
        add([*segments[:-1], "scale"], "weight->scale")
        if len(segments) >= 2 and segments[-2] == "linear":
            # Some upstream wrappers expose a plain Linear directly while
            # clippable linears keep the nested ".linear". Keep both candidates
            # and require the NNX state map to disambiguate.
            add([*segments[:-2], "kernel"], "drop-linear.weight->kernel")
    elif segments[-1] == "bias":
        add([*segments[:-1], "conv", "bias"], "bias->conv/bias")

    return candidates


def _resolve_pt_key(pt_key: str, state_map: dict[str, StateLeaf]) -> CandidatePath:
    candidates = _candidate_paths_for_pt_key(pt_key)
    matches = [candidate for candidate in candidates if candidate.path in state_map]
    if len(matches) == 1:
        return matches[0]
    candidate_text = ", ".join(candidate.path for candidate in candidates)
    if not matches:
        raise KeyError(f"{pt_key} did not match any NNX state path; candidates: {candidate_text}")
    match_text = ", ".join(match.path for match in matches)
    raise KeyError(f"{pt_key} matched multiple NNX state paths ({match_text}); candidates: {candidate_text}")


def _convert_tensor(pt_key: str, jax_path: str, array: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    source_leaf = pt_key.rsplit(".", 1)[-1]
    target_leaf = jax_path.rsplit("/", 1)[-1]
    value = np.asarray(array).astype(np.float32, copy=False)

    if source_leaf == "weight" and target_leaf == "kernel":
        if value.ndim == 2:
            converted = value.T
            transform = "Linear (out,in)->(in,out)"
        elif value.ndim == 4:
            converted = np.transpose(value, (2, 3, 1, 0))
            transform = "Conv2d (out,in,kH,kW)->(kH,kW,in,out)"
        elif value.ndim == 3:
            converted = np.transpose(value, (2, 1, 0))
            transform = "Conv1d (out,in/groups,K)->(K,in/groups,out)"
        elif value.ndim == 1:
            converted = value
            transform = "RMSNorm weight->kernel"
        else:
            raise ValueError(f"unsupported weight rank {value.ndim} for {pt_key} -> {jax_path}")
    elif source_leaf == "weight" and target_leaf == "scale":
        converted = value
        transform = "LayerNorm weight->scale"
    else:
        converted = value
        if source_leaf in CLAMP_BUFFER_NAMES:
            transform = "ClippableLinear scalar buffer"
        else:
            transform = "identity"

    if tuple(converted.shape) != target_shape:
        raise ValueError(
            f"shape mismatch for {pt_key} -> {jax_path}: PT {tuple(value.shape)} "
            f"converted by {transform} to {tuple(converted.shape)}, expected {target_shape}"
        )
    return converted, transform


def _assign_leaf(leaf: nn.Variable, value: np.ndarray) -> None:
    # Bracket assignment is the current NNX array-variable mutation API and
    # handles both Param and plain Variable leaves, including scalar clamp
    # buffers. The fallback keeps the script usable with older NNX builds.
    try:
        leaf[...] = jnp.asarray(value, dtype=jnp.float32)
    except (AttributeError, TypeError):
        leaf.value = jnp.asarray(value, dtype=jnp.float32)


def _load_audio_weights(
    root: AudioParityRoot,
    safetensors_path: Path,
) -> tuple[list[MappingRecord], int]:
    state_map = _build_state_map(root)
    touched_paths: set[str] = set()
    seen_pt_keys: set[str] = set()
    records: list[MappingRecord] = []
    errors: list[str] = []
    skipped_non_audio = 0

    for shard in _safetensor_paths(safetensors_path):
        with safe_open(shard, framework="numpy") as handle:
            for pt_key in handle.keys():
                if not _is_audio_key(pt_key):
                    skipped_non_audio += 1
                    continue
                if pt_key in seen_pt_keys:
                    errors.append(f"duplicate audio key across shards: {pt_key}")
                    continue
                seen_pt_keys.add(pt_key)
                try:
                    candidate = _resolve_pt_key(pt_key, state_map)
                    state_leaf = state_map[candidate.path]
                    if candidate.path in touched_paths:
                        raise KeyError(f"{pt_key} maps to already-touched NNX path {candidate.path}")
                    pt_array = handle.get_tensor(pt_key)
                    converted, transform = _convert_tensor(pt_key, candidate.path, pt_array, state_leaf.shape)
                    _assign_leaf(state_leaf.leaf, converted)
                    touched_paths.add(candidate.path)
                    records.append(
                        MappingRecord(
                            pt_key=pt_key,
                            jax_path=candidate.path,
                            resolver_rule=candidate.resolver_rule,
                            transform=transform,
                            pt_shape=tuple(int(dim) for dim in pt_array.shape),
                            jax_shape=state_leaf.shape,
                            leaf_kind=state_leaf.kind,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - batch all mapping failures for one report.
                    errors.append(f"{pt_key}: {exc}")

    if not seen_pt_keys:
        errors.append(f"no audio keys found; expected prefixes {AUDIO_PREFIXES}")

    untouched = sorted(path for path in state_map if path not in touched_paths)
    if untouched:
        errors.append(
            "untouched NNX Param/Variable leaves:\n"
            + "\n".join(f"  {path} shape={state_map[path].shape} kind={state_map[path].kind}" for path in untouched)
        )

    if errors:
        raise RuntimeError("audio checkpoint load failed:\n" + "\n".join(f"- {error}" for error in errors))

    return records, skipped_non_audio


def _print_mapping_table(records: list[MappingRecord]) -> None:
    print("\nHF safetensors -> NNX state mapping")
    print(
        "pt_key | jax_path | resolver_rule | transform | pt_shape | jax_shape | leaf_kind"
    )
    print("-" * 160)
    for record in records:
        print(
            f"{record.pt_key} | {record.jax_path} | {record.resolver_rule} | "
            f"{record.transform} | {record.pt_shape} | {record.jax_shape} | {record.leaf_kind}"
        )


def _run_audio_trace(
    root: AudioParityRoot,
    input_features: np.ndarray,
    input_features_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    tower = root.audio_tower
    features = jnp.asarray(input_features, dtype=jnp.float32)
    feature_mask = jnp.asarray(input_features_mask.astype(np.bool_))

    # The manual trace mirrors Gemma4AudioModel.__call__ because NNX modules do
    # not offer PyTorch-style forward hooks. The public direct call below still
    # provides the tower output used by downstream code.
    hidden_states, output_mask = tower.subsample_conv_projection(features, feature_mask)
    position_embeddings = tower.rel_pos_enc(hidden_states)
    if output_mask is None:
        output_mask = jnp.ones(hidden_states.shape[:2], dtype=jnp.bool_)
    else:
        output_mask = output_mask.astype(jnp.bool_)
    attention_mask = tower._build_chunked_5d_mask(output_mask)

    actual: dict[str, np.ndarray] = {
        "sscp_out.hidden": np.asarray(hidden_states),
        "sscp_out.mask": np.asarray(output_mask),
    }

    for idx, layer in enumerate(tower.layers):
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        actual[f"audio_block_{idx}_out"] = np.asarray(hidden_states)

    last_hidden_state, final_mask = root.audio_tower(
        input_features=features,
        input_features_mask=feature_mask,
    )
    projected = root.embed_audio(inputs_embeds=last_hidden_state)
    actual["audio_tower_out.last_hidden_state"] = np.asarray(last_hidden_state)
    actual["audio_tower_out.attention_mask"] = np.asarray(final_mask)
    actual["projector_out"] = np.asarray(projected)
    return actual


def _expected_pairs(golden_dir: Path, num_layers: int) -> list[tuple[str, np.ndarray]]:
    pairs = [
        ("sscp_out.hidden", _read_npz_array(golden_dir / "sscp_out.npz", ("0", "data", "last_hidden_state"))),
        ("sscp_out.mask", _read_npz_array(golden_dir / "sscp_out.npz", ("1", "attention_mask", "mask"))),
    ]
    for idx in range(num_layers):
        pairs.append(
            (
                f"audio_block_{idx}_out",
                _read_npz_array(golden_dir / f"audio_block_{idx}_out.npz", ("data", "0", "last_hidden_state")),
            )
        )
    pairs.extend(
        [
            (
                "audio_tower_out.last_hidden_state",
                _read_npz_array(golden_dir / "audio_tower_out.npz", ("last_hidden_state", "0", "data")),
            ),
            (
                "audio_tower_out.attention_mask",
                _read_npz_array(golden_dir / "audio_tower_out.npz", ("attention_mask", "1", "mask")),
            ),
            ("projector_out", _read_npz_array(golden_dir / "projector_out.npz", ("data", "0", "projected"))),
        ]
    )
    return pairs


def _compact_assertion(exc: AssertionError) -> str:
    lines = str(exc).splitlines()
    return "\n".join(lines[:8])


def _compare_outputs(actual: dict[str, np.ndarray], expected_pairs: list[tuple[str, np.ndarray]]) -> tuple[list[DiffRow], list[str]]:
    rows: list[DiffRow] = []
    failures: list[str] = []

    for name, expected in expected_pairs:
        if name not in actual:
            failures.append(f"{name}: validator did not produce this capture")
            rows.append(DiffRow(name, "<missing>", float("nan"), float("nan"), float("nan"), "missing"))
            continue

        actual_value = np.asarray(actual[name])
        expected_value = np.asarray(expected)
        if actual_value.shape != expected_value.shape:
            failures.append(f"{name}: shape mismatch actual={actual_value.shape} expected={expected_value.shape}")
            rows.append(
                DiffRow(
                    name,
                    f"{actual_value.shape}!={expected_value.shape}",
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    "shape_mismatch",
                )
            )
            continue

        actual_float = actual_value.astype(np.float32, copy=False)
        expected_float = expected_value.astype(np.float32, copy=False)
        diff = np.abs(actual_float - expected_float)
        finite_ok = bool(np.isfinite(actual_float).all() and np.isfinite(expected_float).all())
        denom = np.maximum(np.abs(expected_float), np.float32(1.0e-12))
        rel = diff / denom

        row = DiffRow(
            layer_name=name,
            shape=str(tuple(actual_value.shape)),
            max_abs=float(np.max(diff)) if diff.size else 0.0,
            max_rel=float(np.max(rel)) if rel.size else 0.0,
            mean_abs=float(np.mean(diff)) if diff.size else 0.0,
            finite_check="ok" if finite_ok else "nonfinite",
        )
        rows.append(row)

        try:
            if not finite_ok:
                raise AssertionError("actual or expected contains non-finite values")
            np.testing.assert_allclose(actual_float, expected_float, atol=ATOL, rtol=RTOL)
        except AssertionError as exc:
            failures.append(f"{name}: {_compact_assertion(exc)}")

    return rows, failures


def _print_diff_table(rows: list[DiffRow]) -> None:
    print("\nlayer_name | shape | max_abs | max_rel | mean_abs | finite_check")
    print("-" * 112)
    for row in rows:
        print(
            f"{row.layer_name:<38} | {row.shape:<22} | "
            f"{row.max_abs:>10.4e} | {row.max_rel:>10.4e} | {row.mean_abs:>10.4e} | {row.finite_check}"
        )


def main() -> int:
    args = _parse_args()
    golden_dir = Path(args.golden_dir).expanduser().resolve()
    safetensors_path = Path(args.safetensors).expanduser().resolve()

    try:
        text_hidden_size = _infer_text_hidden_size(golden_dir)
        audio_config = _audio_config_from_golden(golden_dir)
        input_features, input_features_mask = _load_inputs(golden_dir, audio_config)

        root = AudioParityRoot(audio_config, text_hidden_size=text_hidden_size, rngs=nn.Rngs(0))
        records, skipped_non_audio = _load_audio_weights(root, safetensors_path)
        print(
            f"loaded {len(records)} audio tensors; skipped {skipped_non_audio} non-audio tensors; "
            f"text_hidden_size={text_hidden_size}; audio_layers={audio_config.num_hidden_layers}"
        )
        if args.verbose:
            _print_mapping_table(records)

        actual = _run_audio_trace(root, input_features, input_features_mask)
        expected = _expected_pairs(golden_dir, audio_config.num_hidden_layers)
        rows, failures = _compare_outputs(actual, expected)
        _print_diff_table(rows)

        if failures:
            print("\nFAILURES")
            print("-" * 80)
            for failure in failures:
                print(f"- {failure}")
            return 1

        print(f"\nGemma 4 audio parity passed at atol={ATOL:g}, rtol={RTOL:g}.")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should exit non-zero with a clear error.
        if args.verbose:
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# Gap analysis
#
# Quickest path from "audio parity passes" to TPU end-to-end inference on a
# TikTok mp4: keep this validator as the checkpoint gate, then wire the same
# audio tower/projector load into the multimodal inference script that already
# handles video frames. Use the HF processor or an equivalent feature extractor
# to turn the mp4 audio track into the exact (B, T, 128) mel tensor shape used
# here, scatter projected audio tokens through Gemma4Model.compute_embedding,
# and run one deterministic TPU decode before adding batching or RL plumbing.
#
# The likely first failures are: (1) a naming mismatch around norm weights
# (Gemma4RMSNorm.kernel versus LayerNorm.scale), (2) a silent transpose mistake
# on square linear kernels or the depthwise Conv1d kernel, and (3) clamp scalar
# buffers not being loaded because they are nn.Variable, not nn.Param. Those
# failures should remain loud; do not relax untouched-state or unmapped-key
# checks to get a forward pass.
#
# Do not prematurely optimise by sharding, quantising, jitting the whole trace,
# caching projected audio, or replacing the HF feature extractor. The
# load-bearing work for the audio->video->RL chain is correctness of
# checkpoint mapping, feature layout, token scattering, and a reproducible
# single-example TPU inference path. Throughput work is useful only after
# those invariants hold under this parity gate.
