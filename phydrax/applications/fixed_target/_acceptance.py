#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DecayVolumePlan(StrictModule, NonTrainableState):
    longitudinal_start: float = eqx.field(static=True)
    longitudinal_end: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    length_unit_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        longitudinal_start: float,
        longitudinal_end: float,
        maximum_radius: float,
        /,
        *,
        length_unit_id: str,
    ):
        start = float(longitudinal_start)
        end = float(longitudinal_end)
        radius = float(maximum_radius)
        unit = str(length_unit_id).strip()
        if (
            not all(math.isfinite(value) for value in (start, end, radius))
            or not 0.0 <= start < end
            or radius <= 0.0
            or not unit
        ):
            raise ValueError("Decay-volume geometry and unit are invalid.")
        self.longitudinal_start = start
        self.longitudinal_end = end
        self.maximum_radius = radius
        self.length_unit_id = unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-target-decay-volume",
                "geometry": [start, end, radius],
                "unit": unit,
            }
        )


class LongLivedAcceptanceResult(StrictModule, NonTrainableState):
    decay_probability: Array
    geometric_acceptance: Array
    total_acceptance: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def long_lived_particle_acceptance(
    plan: DecayVolumePlan,
    momentum: ArrayLike,
    rest_energy: ArrayLike,
    proper_decay_length: ArrayLike,
    /,
) -> LongLivedAcceptanceResult:
    """Evaluate exponential decay and cylindrical straight-line geometric acceptance."""
    if not isinstance(plan, DecayVolumePlan):
        raise TypeError("plan must be DecayVolumePlan.")
    momentum_ = jnp.asarray(momentum)
    mass = jnp.asarray(rest_energy, dtype=momentum_.dtype)
    decay_length = jnp.asarray(proper_decay_length, dtype=momentum_.dtype)
    if (
        momentum_.shape[-1:] != (3,)
        or mass.shape != momentum_.shape[:-1]
        or decay_length.shape != mass.shape
    ):
        raise ValueError(
            "Momentum, mass, and proper decay length supports are incompatible."
        )
    momentum_magnitude = jnp.linalg.norm(momentum_, axis=-1)
    beta_gamma = momentum_magnitude / jnp.maximum(mass, jnp.finfo(momentum_.dtype).tiny)
    lab_decay_length = beta_gamma * decay_length
    probability = jnp.exp(
        -plan.longitudinal_start
        / jnp.maximum(lab_decay_length, jnp.finfo(momentum_.dtype).tiny)
    ) - jnp.exp(
        -plan.longitudinal_end
        / jnp.maximum(lab_decay_length, jnp.finfo(momentum_.dtype).tiny)
    )
    transverse_momentum = jnp.linalg.norm(momentum_[..., :2], axis=-1)
    longitudinal_momentum = momentum_[..., 2]
    radius_at_end = (
        plan.longitudinal_end
        * transverse_momentum
        / jnp.maximum(jnp.abs(longitudinal_momentum), jnp.finfo(momentum_.dtype).tiny)
    )
    geometric = (longitudinal_momentum > 0.0) & (radius_at_end <= plan.maximum_radius)
    total = jnp.where(geometric, probability, 0.0)
    finite = jnp.isfinite(total)
    valid = (
        finite
        & (mass > 0.0)
        & (decay_length > 0.0)
        & (probability >= 0.0)
        & (probability <= 1.0)
    )
    return LongLivedAcceptanceResult(
        probability, geometric, total, finite, valid, plan.plan_id
    )


__all__ = [
    "DecayVolumePlan",
    "LongLivedAcceptanceResult",
    "long_lived_particle_acceptance",
]
