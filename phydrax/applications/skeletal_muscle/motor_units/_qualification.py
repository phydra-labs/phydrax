#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent qualification evidence for the 1993 motor-unit fidelity."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._fuglevand_winter_patla_1993 import PreparedFuglevandWinterPatla1993


class FuglevandWinterPatla1993QualificationEvidence(
    StrictModule, NonTrainableState
):
    """Replay, renewal-distribution, twitch, and force-statistics evidence."""

    replay_exact: Array
    normal_score_mean: Array
    normal_score_standard_deviation: Array
    normal_score_minimum: Array
    normal_score_maximum: Array
    twitch_peak_relative_error: Array
    force_mean_arbitrary: Array
    force_standard_deviation_arbitrary: Array
    force_coefficient_of_variation: Array
    event_count: Array
    finite: Array
    distribution_within_tolerance: Array
    twitch_within_tolerance: Array
    force_variable: Array
    valid: Array
    topology_gradient_supported: bool = eqx.field(static=True)
    claim_scope: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


class FuglevandWinterPatla1993QualificationPlan(
    StrictModule, NonTrainableState
):
    """Finite-sample source qualification policy, not a universal force law."""

    normal_mean_tolerance: float = eqx.field(static=True)
    normal_standard_deviation_tolerance: float = eqx.field(static=True)
    twitch_relative_tolerance: float = eqx.field(static=True)
    minimum_force_standard_deviation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        normal_mean_tolerance: float = 0.08,
        normal_standard_deviation_tolerance: float = 0.08,
        twitch_relative_tolerance: float = 2.0e-6,
        minimum_force_standard_deviation: float = 1.0e-8,
    ):
        values = (
            float(normal_mean_tolerance),
            float(normal_standard_deviation_tolerance),
            float(twitch_relative_tolerance),
            float(minimum_force_standard_deviation),
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Qualification tolerances must be positive and finite.")
        (
            self.normal_mean_tolerance,
            self.normal_standard_deviation_tolerance,
            self.twitch_relative_tolerance,
            self.minimum_force_standard_deviation,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fuglevand-winter-patla-1993-qualification",
                "normal_mean_tolerance": values[0].hex(),
                "normal_standard_deviation_tolerance": values[1].hex(),
                "twitch_relative_tolerance": values[2].hex(),
                "minimum_force_standard_deviation": values[3].hex(),
            }
        )

    def evaluate(
        self,
        prepared: PreparedFuglevandWinterPatla1993,
        normal_scores: ArrayLike,
        event_mask: ArrayLike,
        force_samples_arbitrary: ArrayLike,
        replay_force_samples_arbitrary: ArrayLike,
        /,
    ) -> FuglevandWinterPatla1993QualificationEvidence:
        """Evaluate a realized trial and an independently replayed copy."""

        if not isinstance(prepared, PreparedFuglevandWinterPatla1993):
            raise TypeError("prepared must be PreparedFuglevandWinterPatla1993.")
        scores = jnp.asarray(normal_scores)
        mask = jnp.asarray(event_mask, dtype=bool)
        force = jnp.asarray(force_samples_arbitrary)
        replay = jnp.asarray(replay_force_samples_arbitrary)
        if scores.shape != mask.shape or scores.ndim < 2:
            raise ValueError("normal_scores and event_mask must share rank >= 2.")
        if force.ndim != 1 or force.shape[0] < 2 or replay.shape != force.shape:
            raise ValueError(
                "force and replay samples must be equal-shaped vectors with at least two entries."
            )
        if not jnp.issubdtype(scores.dtype, jnp.inexact):
            scores = scores.astype(float)
        force = force.astype(scores.dtype)
        replay = replay.astype(scores.dtype)
        count = jnp.sum(mask).astype(jnp.int32)
        denominator = jnp.maximum(count, jnp.asarray(1, dtype=jnp.int32)).astype(
            scores.dtype
        )
        selected = jnp.where(mask, scores, 0.0)
        score_mean = jnp.sum(selected) / denominator
        centered = jnp.where(mask, scores - score_mean, 0.0)
        score_std = jnp.sqrt(
            jnp.sum(centered**2)
            / jnp.maximum(denominator - 1.0, jnp.asarray(1.0, dtype=scores.dtype))
        )
        score_min = jnp.min(jnp.where(mask, scores, jnp.inf), initial=jnp.inf)
        score_max = jnp.max(jnp.where(mask, scores, -jnp.inf), initial=-jnp.inf)

        contraction = prepared.contraction_time_ms
        peak = prepared.peak_twitch_force_arbitrary
        twitch_at_peak = peak * (contraction / contraction) * jnp.exp(
            1.0 - contraction / contraction
        )
        twitch_error = jnp.max(
            jnp.abs(twitch_at_peak - peak) / jnp.maximum(jnp.abs(peak), 1.0)
        )
        force_mean = jnp.mean(force)
        force_std = jnp.std(force, ddof=1)
        force_cv = force_std / jnp.maximum(
            jnp.abs(force_mean), jnp.finfo(force.dtype).tiny
        )
        replay_exact = jnp.array_equal(force, replay)
        finite = (
            (count > 1)
            & jnp.all(jnp.isfinite(selected))
            & jnp.all(jnp.isfinite(force))
            & jnp.isfinite(force_cv)
        )
        distribution = (
            (jnp.abs(score_mean) <= self.normal_mean_tolerance)
            & (
                jnp.abs(score_std - 1.0)
                <= self.normal_standard_deviation_tolerance
            )
            & (score_min >= -3.9)
            & (score_max <= 3.9)
        )
        twitch_within = twitch_error <= self.twitch_relative_tolerance
        force_variable = force_std >= self.minimum_force_standard_deviation
        valid = finite & replay_exact & distribution & twitch_within & force_variable
        return FuglevandWinterPatla1993QualificationEvidence(
            replay_exact,
            score_mean,
            score_std,
            score_min,
            score_max,
            twitch_error,
            force_mean,
            force_std,
            force_cv,
            count,
            finite,
            distribution,
            twitch_within,
            force_variable,
            valid,
            False,
            (
                "finite realized renewal/force trial; no gradients through event "
                "topology and no population-wide force-variability claim"
            ),
            canonical_fingerprint(
                {
                    "kind": "fuglevand-winter-patla-1993-qualified-trial",
                    "plan": self.plan_id,
                    "model": prepared.plan.model_id,
                }
            ),
        )


__all__ = [
    "FuglevandWinterPatla1993QualificationEvidence",
    "FuglevandWinterPatla1993QualificationPlan",
]
