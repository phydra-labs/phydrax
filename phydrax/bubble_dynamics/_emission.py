#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Far-field monopole emission of a bubble cloud from its dense output.

A compact pulsating bubble radiates the linear acoustic monopole field
`p(x, t) = ρ V̈(t − r/c)/(4π r)` with `r = |x − x_b|`; the cloud field is the
superposition over bubbles. `V̈` is the exact time derivative of the volume
flow rate `V̇ = 4πR²Ṙ` along the solver's dense interpolant (one forward-mode
derivative per retarded time), so no finite difference of saved samples enters.
The formula is valid in the far field (`r ≫ R`) of acoustically compact
bubbles; the distance ratio is reported, and retarded times outside the solved
interval are uncovered (NaN), never extrapolated.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field
from .._validation import positive_finite_float, positive_integer


class FarFieldEmissionPlan(StrictModule):
    """Observer positions and emission times of one far-field evaluation.

    `observers` are absolute positions (m) in the cloud frame and `times` the
    observer times (s). `minimum_distance_ratio` is the far-field support
    threshold on `r/R_max`; `working_set` bounds the number of dense-output
    evaluations held at once.
    """

    observers: Array = fixed_field()
    times: Array = fixed_field()
    minimum_distance_ratio: float = eqx.field(static=True)
    working_set: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        observers: ArrayLike,
        times: ArrayLike,
        /,
        *,
        minimum_distance_ratio: float = 10.0,
        working_set: int = 65536,
    ) -> None:
        points = np.asarray(observers, dtype=np.float64)
        instants = np.asarray(times, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("observers must have shape (M, 3) with M >= 1.")
        if not np.all(np.isfinite(points)):
            raise ValueError("observers must be finite.")
        if (
            instants.ndim != 1
            or instants.shape[0] == 0
            or not np.all(np.isfinite(instants))
        ):
            raise ValueError("times must be a finite non-empty rank-1 array.")
        ratio = positive_finite_float(minimum_distance_ratio, "minimum_distance_ratio")
        budget = positive_integer(working_set, "working_set")
        self.observers = jnp.asarray(points, dtype=jnp.float64)
        self.times = jnp.asarray(instants, dtype=jnp.float64)
        self.minimum_distance_ratio = ratio
        self.working_set = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubble-far-field-emission-plan",
                "observers": points,
                "times": instants,
                "minimum_distance_ratio": ratio,
                "working_set": budget,
            }
        )


class FarFieldEmissionEvidence(StrictModule):
    """Retarded-time coverage and far-field support of one evaluation.

    `covered[t, m]` is true when every bubble's retarded time lies inside the
    solved interval. `minimum_distance_ratio` is `min r/R_max` over observers
    and bubbles with the maximum radius reached on the saved trajectory.
    """

    covered: Array
    covered_fraction: Array
    minimum_distance_ratio: Array
    within_far_field: Array
    finite: Array
    evaluation_count: int = eqx.field(static=True)


class FarFieldEmissionResult(StrictModule):
    """Acoustic pressure `p(t, m)` (Pa) at every observer and time."""

    times: Array
    observers: Array
    pressure: Array
    evidence: FarFieldEmissionEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether every sample is covered, finite and in the far field."""
        evidence = self.evidence
        return jnp.all(evidence.covered) & evidence.finite & evidence.within_far_field


def far_field_emission(
    plan: FarFieldEmissionPlan,
    volume_rate: Callable[[Array], Array],
    positions: Array,
    maximum_radius: Array,
    density: Array,
    sound_speed: Array,
    start_time: Array,
    end_time: Array,
    /,
) -> FarFieldEmissionResult:
    """Superpose `ρ V̈_b(t − r/c)/(4πr)` over bubbles from a dense volume-rate map.

    `volume_rate(t)` returns `V̇` of every bubble at the scalar time `t` from the
    dense output; its forward-mode time derivative is `V̈`.
    """
    observers = plan.observers
    count = positions.shape[0]
    distance = jnp.sqrt(
        jnp.sum((observers[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    )
    delay = distance / sound_speed
    retarded = plan.times[:, None, None] - delay[None, :, :]
    inside = (retarded >= start_time) & (retarded <= end_time)
    covered = jnp.all(inside, axis=-1)
    query = jnp.clip(retarded, start_time, end_time)
    index = jnp.arange(count)

    def volume_acceleration(time: Array, bubble: Array) -> Array:
        _, tangent = jax.jvp(volume_rate, (time,), (jnp.ones_like(time),))
        return tangent[bubble]

    flat_query = query.reshape((-1, count))

    def sample(times: Array) -> Array:
        return jax.vmap(volume_acceleration)(times, index)

    batch = max(1, plan.working_set // max(count * count, 1))
    accelerations = jax.lax.map(sample, flat_query, batch_size=batch).reshape(query.shape)
    contribution = density * accelerations / (4.0 * jnp.pi * distance[None, :, :])
    pressure = jnp.where(covered, jnp.sum(contribution, axis=-1), jnp.nan)
    ratio = jnp.min(distance / maximum_radius[None, :])
    finite = jnp.all(jnp.where(covered, jnp.isfinite(pressure), True))
    evidence = FarFieldEmissionEvidence(
        covered,
        jnp.mean(covered.astype(jnp.float64)),
        ratio,
        ratio >= plan.minimum_distance_ratio,
        finite,
        evaluation_count=plan.times.shape[0] * observers.shape[0] * count,
    )
    return FarFieldEmissionResult(
        plan.times, observers, pressure, evidence, plan_id=plan.plan_id
    )


__all__ = [
    "FarFieldEmissionEvidence",
    "FarFieldEmissionPlan",
    "FarFieldEmissionResult",
    "far_field_emission",
]
