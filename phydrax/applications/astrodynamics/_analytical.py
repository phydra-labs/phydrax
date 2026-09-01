#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._elements import classical_to_cartesian, ClassicalOrbitalElements
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


class J2SecularResult(StrictModule):
    elements: ClassicalOrbitalElements
    state: CartesianOrbitState
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class J2SecularPlan(StrictModule, NonTrainableState):
    mu: Array
    reference_radius: Array
    j2: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, mu: ArrayLike, reference_radius: ArrayLike, j2: ArrayLike, /):
        self.mu = jnp.asarray(mu).reshape(())
        self.reference_radius = jnp.asarray(reference_radius).reshape(())
        self.j2 = jnp.asarray(j2).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "j2-secular-propagator",
                "mu": float(self.mu),
                "radius": float(self.reference_radius),
                "j2": float(self.j2),
            }
        )

    def propagate(
        self, initial: ClassicalOrbitalElements, delta_time: ArrayLike, /
    ) -> J2SecularResult:
        p, eccentricity, inclination, raan, argument, anomaly = initial.values
        semi_major = p / (1.0 - eccentricity**2)
        mean_motion = jnp.sqrt(self.mu / semi_major**3)
        factor = 1.5 * self.j2 * mean_motion * (self.reference_radius / p) ** 2
        cosine = jnp.cos(inclination)
        dt = jnp.asarray(delta_time)
        next_values = jnp.asarray(
            (
                p,
                eccentricity,
                inclination,
                raan - factor * cosine * dt,
                argument + 0.5 * factor * (5.0 * cosine**2 - 1.0) * dt,
                anomaly
                + (
                    mean_motion
                    + 0.5
                    * factor
                    * jnp.sqrt(1.0 - eccentricity**2)
                    * (3.0 * cosine**2 - 1.0)
                )
                * dt,
            )
        )
        elements = ClassicalOrbitalElements(next_values, initial.context)
        state, state_valid, _ = classical_to_cartesian(elements, self.mu)
        valid = state_valid & (eccentricity < 1.0) & (p > 0.0)
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.UNSUPPORTED_REGIME),
        ).astype(jnp.int32)
        return J2SecularResult(elements, state, valid, status, self.plan_id)


__all__ = ["J2SecularPlan", "J2SecularResult"]
