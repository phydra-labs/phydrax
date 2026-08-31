#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


class ChargedParticlePlan(StrictModule, NonTrainableState):
    """Extensive macrocharge attached to one stable particle support."""

    charges: Array
    species_id: str = eqx.field(static=True)
    require_constant_specific_charge: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        charges: ArrayLike,
        species_id: str,
        /,
        *,
        require_constant_specific_charge: bool = True,
        tolerance: float = 1.0e-12,
    ):
        values = np.asarray(charges)
        identifier = str(species_id)
        tolerance_ = float(tolerance)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("charges must be a nonempty rank-one array.")
        if not np.issubdtype(values.dtype, np.inexact):
            values = values.astype(float)
        if not identifier:
            raise ValueError("species_id must be nonempty.")
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.charges = jnp.asarray(values)
        self.species_id = identifier
        self.require_constant_specific_charge = bool(require_constant_specific_charge)
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charged-particle-plan",
                "charges": array_tree_fingerprint(values),
                "species": identifier,
                "constant_specific_charge": bool(require_constant_specific_charge),
                "tolerance": tolerance_,
            }
        )

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedChargedParticles:
        return PreparedChargedParticles(self, particles)


class PreparedChargedParticles(StrictModule, NonTrainableState):
    """Prepared macrocharge and charge-to-mass data over material particles."""

    plan: ChargedParticlePlan
    particles: ParticleDiscretization
    charges: Array
    specific_charge: Array
    reference_specific_charge: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ChargedParticlePlan, particles: ParticleDiscretization, /):
        if not isinstance(plan, ChargedParticlePlan):
            raise TypeError("plan must be ChargedParticlePlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if plan.charges.shape != (particles.capacity,):
            raise ValueError("charges must have the particle-capacity shape.")
        dtype = jnp.result_type(plan.charges, particles.safe_masses)
        charges = jnp.asarray(plan.charges, dtype=dtype)
        active = particles.active_mask
        valid = jnp.all(
            jnp.where(active, jnp.isfinite(charges) & (charges != 0.0), charges == 0.0)
        )
        if not bool(valid):
            raise ValueError(
                "Active macrocharges must be finite and nonzero; inactive charges must be zero."
            )
        specific = jnp.where(
            active,
            charges / particles.safe_masses.astype(dtype),
            jnp.zeros((), dtype=dtype),
        )
        first_active = int(np.flatnonzero(np.asarray(active))[0])
        reference = specific[first_active]
        if plan.require_constant_specific_charge:
            scale = jnp.maximum(jnp.abs(reference), 1.0)
            defect = jnp.max(jnp.where(active, jnp.abs(specific - reference), 0.0))
            if not bool(defect <= plan.tolerance * scale):
                raise ValueError(
                    "Active particles in one charged species must share specific charge."
                )
        self.plan = plan
        self.particles = particles
        self.charges = charges
        self.specific_charge = specific
        self.reference_specific_charge = reference
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-charged-particles",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
            }
        )

    @property
    def capacity(self) -> int:
        return self.particles.capacity

    @property
    def spatial_dimension(self) -> int:
        return self.particles.ambient_dimension


__all__ = ["ChargedParticlePlan", "PreparedChargedParticles"]
