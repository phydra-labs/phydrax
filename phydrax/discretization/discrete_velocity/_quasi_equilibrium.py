#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._kinetic_entropy import solve_kinetic_entropy_root
from ._compressible_contracts import (
    CompressibleKineticConservationEvidence,
    CompressibleKineticPopulationState,
    CompressibleKineticStepResult,
)
from ._positive_kinetic import PositiveCompressibleKineticPlan


QuasiEquilibriumSlowFamily = Literal["heat-flux", "stress"]


class QuasiEquilibriumEvidence(StrictModule):
    requested_prandtl: Array
    beta_one: Array
    beta_two: Array
    slow_moment_defect: Array
    conserved_moment_defect: Array
    minimum_quasi_population: Array
    slow_family: QuasiEquilibriumSlowFamily = eqx.field(static=True)
    successful: Array


class FullRangeQuasiEquilibriumPlan(StrictModule, NonTrainableState):
    """Two-rate quasi-equilibrium closure on a positive kinetic model."""

    model: PositiveCompressibleKineticPlan
    particle_lift: Array
    internal_lift: Array
    prandtl_number: float = eqx.field(static=True)
    slow_family: QuasiEquilibriumSlowFamily = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: PositiveCompressibleKineticPlan,
        /,
        *,
        prandtl_number: float,
    ):
        if not isinstance(model, PositiveCompressibleKineticPlan):
            raise TypeError("model must be a PositiveCompressibleKineticPlan.")
        prandtl = float(prandtl_number)
        if not np.isfinite(prandtl) or prandtl <= 0.0:
            raise ValueError("prandtl_number must be finite and positive.")
        velocities = np.asarray(model.rule.velocities, dtype=np.float64)
        speed_squared = np.sum(velocities * velocities, axis=1)
        conserved = np.vstack(
            (
                np.ones(model.rule.population_count),
                velocities.T,
                speed_squared,
            )
        )
        if prandtl <= 1.0:
            slow_family: QuasiEquilibriumSlowFamily = "heat-flux"
            slow = (velocities * speed_squared[:, None]).T
        else:
            slow_family = "stress"
            cx, cy, cz = velocities.T
            slow = np.vstack(
                (
                    cx * cx - cz * cz,
                    cy * cy - cz * cz,
                    cx * cy,
                    cx * cz,
                    cy * cz,
                )
            )
        constraints = np.vstack((conserved, slow))
        gram = constraints @ constraints.T
        if np.linalg.matrix_rank(gram) != gram.shape[0]:
            raise ValueError(
                "Velocity rule cannot represent the quasi-equilibrium moments."
            )
        inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
        lift = constraints.T @ inverse
        slow_lift = lift[:, conserved.shape[0] :]
        internal_constraints = np.vstack(
            (np.ones(model.rule.population_count), velocities.T)
        )
        internal_gram = internal_constraints @ internal_constraints.T
        internal_inverse = np.linalg.solve(internal_gram, np.eye(internal_gram.shape[0]))
        internal_lift = (internal_constraints.T @ internal_inverse)[:, 1:]
        self.model = model
        self.particle_lift = jnp.asarray(slow_lift)
        self.internal_lift = jnp.asarray(internal_lift)
        self.prandtl_number = prandtl
        self.slow_family = slow_family
        self.plan_id = canonical_fingerprint(
            {
                "kind": "full-range-quasi-equilibrium",
                "model": model.model_id,
                "prandtl_number": prandtl,
                "slow_family": slow_family,
                "particle_lift": slow_lift.tolist(),
                "internal_lift": internal_lift.tolist(),
            }
        )

    def _particle_slow_moments(self, populations: Array, /) -> Array:
        velocities = self.model.rule.velocities.astype(populations.dtype)
        speed_squared = jnp.sum(velocities * velocities, axis=-1)
        if self.slow_family == "heat-flux":
            features = velocities * speed_squared[:, None]
        else:
            cx, cy, cz = jnp.moveaxis(velocities, -1, 0)
            features = jnp.stack(
                (
                    cx * cx - cz * cz,
                    cy * cy - cz * cz,
                    cx * cy,
                    cx * cz,
                    cy * cz,
                ),
                axis=-1,
            )
        return populations @ features

    def collide(
        self,
        state: CompressibleKineticPopulationState,
        relaxation_rate: ArrayLike,
        /,
    ) -> tuple[CompressibleKineticStepResult, QuasiEquilibriumEvidence]:
        old = self.model.moments(state)
        rate = jnp.broadcast_to(
            jnp.asarray(relaxation_rate, dtype=old.density.dtype), old.density.shape
        )
        rate = eqx.error_if(
            rate,
            jnp.any(~jnp.isfinite(rate) | (rate <= 0.0) | (rate >= 2.0)),
            "Quasi-equilibrium relaxation_rate must lie in (0, 2).",
        )
        equilibrium, dual, _ = self.model.equilibrium(
            old.density,
            old.velocity,
            old.temperature,
            initial_dual=state.equilibrium_dual,
        )
        particle = state.population("particle")
        particle_equilibrium = equilibrium[0]
        current_slow = self._particle_slow_moments(particle)
        equilibrium_slow = self._particle_slow_moments(particle_equilibrium)
        particle_quasi = particle_equilibrium + (
            current_slow - equilibrium_slow
        ) @ jnp.swapaxes(self.particle_lift.astype(particle.dtype), -1, -2)
        quasi: list[Array] = [particle_quasi]
        if len(self.model.layout.fields) == 2:
            internal = state.population("internal-energy")
            internal_equilibrium = equilibrium[1]
            velocities = self.model.rule.velocities.astype(internal.dtype)
            current_flux = internal @ velocities
            equilibrium_flux = internal_equilibrium @ velocities
            internal_quasi = internal_equilibrium + (
                current_flux - equilibrium_flux
            ) @ jnp.swapaxes(self.internal_lift.astype(internal.dtype), -1, -2)
            quasi.append(internal_quasi)
        beta_one = 0.5 * rate
        prandtl = jnp.asarray(self.prandtl_number, dtype=rate.dtype)
        beta_two = prandtl * beta_one / (1.0 - beta_one + prandtl * beta_one)
        if self.model.collision_kind == "entropic":
            root = solve_kinetic_entropy_root(
                self.model.entropy_root,
                particle,
                particle_equilibrium - particle,
                base_measure=self.model.rule.base_probabilities,
                initial=state.stabilizer,
            )
            alpha = root.evidence.alpha
            root_success = root.evidence.successful
        else:
            alpha = jnp.full(rate.shape, 2.0, dtype=rate.dtype)
            root_success = jnp.ones(rate.shape, dtype=jnp.bool_)
        candidates = tuple(
            value
            + (alpha * beta_one)[..., None] * (target - value)
            + (2.0 * (beta_one - beta_two))[..., None] * (slow - target)
            for value, target, slow in zip(
                state.populations, equilibrium, tuple(quasi), strict=True
            )
        )
        candidate = CompressibleKineticPopulationState(
            candidates,
            dual,
            alpha,
            state.frame_velocity,
            state.frame_temperature_scale,
            self.model.layout,
        )
        new = self.model.moments(candidate)
        mass_defect = new.density - old.density
        momentum_defect = jnp.max(jnp.abs(new.momentum - old.momentum), axis=-1)
        energy_defect = new.total_energy - old.total_energy
        minimum_quasi = jnp.min(
            jnp.stack(tuple(jnp.min(value, axis=-1) for value in quasi), axis=0),
            axis=0,
        )
        minimum_candidate = jnp.min(
            jnp.stack(tuple(jnp.min(value, axis=-1) for value in candidates), axis=0),
            axis=0,
        )
        scale = jnp.maximum(jnp.abs(old.total_energy), 1.0)
        numerical_tolerance = jnp.maximum(
            512.0 * jnp.finfo(rate.dtype).eps,
            8.0 * self.model.family.solve_plan.residual_tolerance,
        )
        density_scale = jnp.maximum(old.density, 1.0)
        successful = (
            root_success
            & old.admissible
            & new.admissible
            & (minimum_quasi > 0.0)
            & (minimum_candidate > 0.0)
            & (jnp.abs(mass_defect) <= numerical_tolerance * density_scale)
            & (jnp.abs(energy_defect) <= numerical_tolerance * scale)
        )
        accepted = CompressibleKineticPopulationState(
            tuple(
                jnp.where(successful[..., None], new_value, old_value)
                for new_value, old_value in zip(
                    candidates, state.populations, strict=True
                )
            ),
            jnp.where(successful[..., None], dual, state.equilibrium_dual),
            jnp.where(successful, alpha, state.stabilizer),
            state.frame_velocity,
            state.frame_temperature_scale,
            self.model.layout,
        )
        conservation = CompressibleKineticConservationEvidence(
            mass_defect=mass_defect,
            momentum_defect=momentum_defect,
            energy_defect=energy_defect,
            entropy_change=jnp.zeros_like(mass_defect),
            minimum_population=minimum_candidate,
            finite=new.finite,
            successful=successful,
        )
        result = CompressibleKineticStepResult(
            candidate=candidate,
            accepted=accepted,
            macroscopic=new,
            conservation=conservation,
            status=jnp.where(successful, 0, 1).astype(jnp.int32),
            successful=successful,
            model_id=self.model.model_id,
        )
        evidence = QuasiEquilibriumEvidence(
            requested_prandtl=jnp.broadcast_to(prandtl, rate.shape),
            beta_one=beta_one,
            beta_two=beta_two,
            slow_moment_defect=jnp.max(
                jnp.abs(self._particle_slow_moments(particle_quasi) - current_slow),
                axis=-1,
            ),
            conserved_moment_defect=jnp.maximum(
                jnp.abs(mass_defect), jnp.maximum(momentum_defect, jnp.abs(energy_defect))
            ),
            minimum_quasi_population=minimum_quasi,
            slow_family=self.slow_family,
            successful=successful,
        )
        return result, evidence


__all__ = [
    "FullRangeQuasiEquilibriumPlan",
    "QuasiEquilibriumEvidence",
    "QuasiEquilibriumSlowFamily",
]
