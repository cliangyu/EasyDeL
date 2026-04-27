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

"""Property + boundary tests for the GRPO clipped policy surrogate.

The production formula at ``_fn.py:556`` and ``:710`` is

    per_token_loss = -jnp.minimum(coef_1 * A, jnp.clip(coef_1, 1-eps, 1+eps_high) * A)

A previous version conditioned on the sign of A and used ``jnp.maximum`` for
A < 0 (commit b4a2008c reverted that). For A < 0 with a small ratio, the
buggy form returned the *optimistic* surrogate, which let the policy update
without the low clip ever binding — an off-policy runaway hazard.

These tests pin (a) the canonical pessimism property and (b) the specific
A < 0, r < 1-eps boundary case the bug missed. Pure-math, no model, runs
in milliseconds on CPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="jax not installed in this environment")

import jax
import jax.numpy as jnp


def _clipped_surrogate(coef_1: jax.Array, advantages: jax.Array, eps: float, eps_high: float) -> jax.Array:
    """Reference implementation of the production clipped surrogate."""
    coef_2 = jnp.clip(coef_1, 1.0 - eps, 1.0 + eps_high)
    loss1 = coef_1 * advantages
    loss2 = coef_2 * advantages
    return -jnp.minimum(loss1, loss2)


def _buggy_sign_conditional_surrogate(
    coef_1: jax.Array, advantages: jax.Array, eps: float, eps_high: float
) -> jax.Array:
    """The pre-fix version — reproduced here only to assert the new form differs."""
    coef_2 = jnp.clip(coef_1, 1.0 - eps, 1.0 + eps_high)
    loss1 = coef_1 * advantages
    loss2 = coef_2 * advantages
    return -jnp.where(advantages >= 0, jnp.minimum(loss1, loss2), jnp.maximum(loss1, loss2))


def test_clipped_surrogate_is_pessimistic_vs_unclipped() -> None:
    """For all (r, A): clipped_loss >= unclipped_loss = -r*A.

    This is the defining property of the PPO clipped surrogate: the clip
    can only make the surrogate smaller (the loss larger), never the
    other way. Random sweep with both signs of A.
    """
    key = jax.random.PRNGKey(0)
    k_r, k_a = jax.random.split(key)
    coef = jax.random.uniform(k_r, (4096,), minval=0.3, maxval=2.5, dtype=jnp.float32)
    adv = jax.random.normal(k_a, (4096,), dtype=jnp.float32) * 1.5

    eps = 0.2
    clipped = _clipped_surrogate(coef, adv, eps, eps)
    unclipped = -coef * adv

    # Allow tiny float slack but no monotonic violation.
    assert bool(jnp.all(clipped >= unclipped - 1e-6))


def test_no_clip_in_band_matches_unclipped() -> None:
    """For r in (1-eps, 1+eps_high), the clip is a no-op; surrogate == -r*A."""
    eps = 0.2
    coef = jnp.array([1.0 - eps + 0.01, 1.0, 1.0 + eps - 0.01], dtype=jnp.float32)
    adv = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float32)

    clipped = _clipped_surrogate(coef, adv, eps, eps)
    unclipped = -coef * adv

    assert jnp.allclose(clipped, unclipped, atol=1e-6)


def test_high_clip_binds_for_positive_advantage() -> None:
    """A > 0, r > 1+eps: clip caps surrogate at -(1+eps)*A < -r*A."""
    eps = 0.2
    coef = jnp.array([1.5], dtype=jnp.float32)  # r = 1.5 > 1+eps = 1.2
    adv = jnp.array([1.0], dtype=jnp.float32)

    surrogate = _clipped_surrogate(coef, adv, eps, eps)
    expected = jnp.array([-(1.0 + eps) * 1.0], dtype=jnp.float32)  # -1.2

    assert jnp.allclose(surrogate, expected, atol=1e-6)
    assert float(surrogate[0]) > float(-coef[0] * adv[0])  # pessimistic vs unclipped (-1.5)


def test_low_clip_binds_for_negative_advantage_regression() -> None:
    """A < 0, r < 1-eps: low clip MUST bind. The pre-fix code missed this case.

    Numbers: r=0.5, A=-1, eps=0.2 → 1-eps=0.8.
        loss1 = 0.5 * -1 = -0.5
        loss2 = 0.8 * -1 = -0.8
        new (correct) = -min(-0.5, -0.8) = +0.8
        old (buggy)   = -max(-0.5, -0.8) = +0.5
    The new form is +0.3 more pessimistic — exactly the regime the bug exempted.
    """
    eps = 0.2
    coef = jnp.array([0.5], dtype=jnp.float32)
    adv = jnp.array([-1.0], dtype=jnp.float32)

    new_form = _clipped_surrogate(coef, adv, eps, eps)
    old_form = _buggy_sign_conditional_surrogate(coef, adv, eps, eps)

    assert jnp.allclose(new_form, jnp.array([0.8]), atol=1e-6), f"new={new_form}"
    assert jnp.allclose(old_form, jnp.array([0.5]), atol=1e-6), f"old={old_form}"
    assert float(new_form[0]) > float(old_form[0])


def test_buggy_form_only_disagrees_for_negative_advantage_outside_band() -> None:
    """Sanity-check that the old/new forms agree everywhere except A<0 with low-clip-binding.

    For A>=0, both forms collapse to -min(loss1, loss2) by the where-branch.
    For A<0 with r in band [1-eps, 1+eps], coef_2 == coef_1 so loss1 == loss2,
    and min == max trivially. So disagreement only happens for A<0 AND r<1-eps
    AND r>1+eps_high (the low/high clip regions).
    """
    eps = 0.2
    # Uniform sweep, including both clip regions and both signs of A.
    coef_pos = jnp.linspace(0.4, 1.6, 40, dtype=jnp.float32)
    adv_pos = jnp.full_like(coef_pos, 1.0)
    adv_neg_in_band = jnp.full_like(coef_pos, -1.0)

    new_pos = _clipped_surrogate(coef_pos, adv_pos, eps, eps)
    old_pos = _buggy_sign_conditional_surrogate(coef_pos, adv_pos, eps, eps)
    assert jnp.allclose(new_pos, old_pos, atol=1e-6), "must agree for all A>0"

    # For A<0 and r in band, agreement is trivial (loss1 == loss2). Use a
    # ratio strictly inside [1-eps, 1+eps].
    coef_in_band = jnp.linspace(1.0 - eps + 0.01, 1.0 + eps - 0.01, 20, dtype=jnp.float32)
    adv_in_band_neg = jnp.full_like(coef_in_band, -1.0)
    new_in = _clipped_surrogate(coef_in_band, adv_in_band_neg, eps, eps)
    old_in = _buggy_sign_conditional_surrogate(coef_in_band, adv_in_band_neg, eps, eps)
    assert jnp.allclose(new_in, old_in, atol=1e-6), "must agree for A<0 with r in band"

    # And confirm strict disagreement at the negative-advantage low-clip boundary.
    coef_low = jnp.array([0.4, 0.5, 0.6], dtype=jnp.float32)  # all < 1-eps
    adv_neg = jnp.full_like(coef_low, -1.0)
    new_low = _clipped_surrogate(coef_low, adv_neg, eps, eps)
    old_low = _buggy_sign_conditional_surrogate(coef_low, adv_neg, eps, eps)
    assert bool(jnp.all(new_low > old_low + 1e-6)), f"new={new_low} old={old_low}"


def test_production_source_does_not_contain_buggy_sign_conditional() -> None:
    """Source-level pin: prevent a future refactor from re-introducing the
    sign-conditional surrogate at ``_fn.py``. The buggy form looked like::

        -jnp.where(advantages >= 0,
                   jnp.minimum(loss1, loss2),
                   jnp.maximum(loss1, loss2))

    The defining red flag is ``advantages`` (or ``chunk_advantages``)
    compared with ``>= 0`` — note ``> 0`` is fine and used legitimately
    at the high-clip metric (line 586).
    """
    from pathlib import Path

    fn_py = (
        Path(__file__).resolve().parent.parent.parent
        / "easydel"
        / "trainers"
        / "group_relative_policy_optimization"
        / "_fn.py"
    )
    src = fn_py.read_text()
    assert "advantages >= 0" not in src, (
        "Found 'advantages >= 0' in grpo_step._fn.py — possible re-introduction of the sign-conditional "
        "clip surrogate. See commit b4a2008c; use the unconditional `-jnp.minimum(loss1, loss2)` form."
    )
