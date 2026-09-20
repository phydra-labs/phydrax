#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class BinnedLikelihoodPlan(StrictModule, NonTrainableState):
    """Limited linear-nuisance Poisson IR; external calculators remain authoritative."""

    nominal_expectation: Array
    nuisance_effects: Array
    constraint_standard_deviations: Array
    nuisance_names: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nominal_expectation: ArrayLike,
        nuisance_effects: ArrayLike,
        constraint_standard_deviations: ArrayLike,
        /,
        *,
        nuisance_names: Sequence[str],
        channel_names: Sequence[str],
    ):
        nominal = np.asarray(nominal_expectation, dtype=np.float64)
        effects = np.asarray(nuisance_effects, dtype=np.float64)
        constraints = np.asarray(constraint_standard_deviations, dtype=np.float64)
        nuisances = tuple(str(value).strip() for value in nuisance_names)
        channels = tuple(str(value).strip() for value in channel_names)
        if (
            nominal.ndim != 1
            or nominal.size < 1
            or np.any(~np.isfinite(nominal))
            or np.any(nominal < 0.0)
        ):
            raise ValueError(
                "nominal_expectation must be a finite nonnegative bin vector."
            )
        if effects.shape != (len(nuisances), nominal.size) or constraints.shape != (
            len(nuisances),
        ):
            raise ValueError("Nuisance arrays must align with names and bins.")
        if (
            len(channels) != nominal.size
            or any(not value for value in nuisances + channels)
            or len(set(nuisances)) != len(nuisances)
        ):
            raise ValueError("Likelihood names must be non-empty with unique nuisances.")
        if (
            np.any(~np.isfinite(effects))
            or np.any(~np.isfinite(constraints))
            or np.any(constraints <= 0.0)
        ):
            raise ValueError(
                "Nuisance effects and constraints must be finite with positive scales."
            )
        self.nominal_expectation = jnp.asarray(nominal)
        self.nuisance_effects = jnp.asarray(effects)
        self.constraint_standard_deviations = jnp.asarray(constraints)
        self.nuisance_names = nuisances
        self.channel_names = channels
        self.plan_id = canonical_fingerprint(
            {
                "kind": "limited-binned-poisson-likelihood",
                "nominal": array_tree_fingerprint(nominal),
                "effects": array_tree_fingerprint(effects),
                "constraints": array_tree_fingerprint(constraints),
                "nuisances": list(nuisances),
                "channels": list(channels),
            }
        )


class BinnedLikelihoodEvaluation(StrictModule, NonTrainableState):
    expected: Array
    poisson_log_likelihood: Array
    constraint_log_likelihood: Array
    total_log_likelihood: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def evaluate_binned_likelihood(
    plan: BinnedLikelihoodPlan,
    observations: ArrayLike,
    nuisance_parameters: ArrayLike,
    /,
) -> BinnedLikelihoodEvaluation:
    if not isinstance(plan, BinnedLikelihoodPlan):
        raise TypeError("plan must be BinnedLikelihoodPlan.")
    observations_ = jnp.asarray(observations, dtype=plan.nominal_expectation.dtype)
    nuisance = jnp.asarray(nuisance_parameters, dtype=plan.nominal_expectation.dtype)
    if observations_.shape != plan.nominal_expectation.shape or nuisance.shape != (
        len(plan.nuisance_names),
    ):
        raise ValueError("Likelihood observations or nuisance support is invalid.")
    expected = plan.nominal_expectation + nuisance @ plan.nuisance_effects
    possible = jnp.all(expected > 0.0) & jnp.all(observations_ >= 0.0)
    safe = jnp.maximum(expected, jnp.finfo(expected.dtype).tiny)
    poisson = jnp.sum(
        observations_ * jnp.log(safe) - safe - jsp.special.gammaln(observations_ + 1.0)
    )
    standardized = nuisance / plan.constraint_standard_deviations
    constraint = -0.5 * jnp.sum(
        standardized * standardized
        + jnp.log(2.0 * jnp.pi * plan.constraint_standard_deviations**2)
    )
    total = poisson + constraint
    finite = jnp.all(jnp.isfinite(expected)) & jnp.isfinite(total)
    return BinnedLikelihoodEvaluation(
        expected,
        poisson,
        constraint,
        total,
        finite,
        finite & possible,
        plan.plan_id,
    )


__all__ = [
    "BinnedLikelihoodEvaluation",
    "BinnedLikelihoodPlan",
    "evaluate_binned_likelihood",
]
