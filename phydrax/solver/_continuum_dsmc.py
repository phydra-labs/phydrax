#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class MaxwellianReservoirResult(StrictModule):
    velocities: Array
    momentum_defect: Array
    kinetic_energy_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MaxwellianReservoirPlan(StrictModule, NonTrainableState):
    """Quiet-start Maxwellian samples corrected to exact mean and kinetic energy."""

    particle_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_count: int,
        dimension: int,
        /,
        *,
        boltzmann_constant: float = 1.380649e-23,
    ):
        count = int(particle_count)
        dimension_ = int(dimension)
        boltzmann = float(boltzmann_constant)
        if (
            count <= dimension_
            or dimension_ not in (1, 2, 3)
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError("Maxwellian reservoir capacity or dimension is invalid.")
        self.particle_count = count
        self.dimension = dimension_
        self.boltzmann_constant = boltzmann
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwellian-reservoir",
                "particle_count": count,
                "dimension": dimension_,
                "boltzmann_constant": boltzmann,
            }
        )

    def sample(
        self,
        key: PRNGKeyArray,
        mean_velocity: ArrayLike,
        temperature: ArrayLike,
        molecular_mass: ArrayLike,
        /,
    ) -> MaxwellianReservoirResult:
        mean = jnp.asarray(mean_velocity)
        temperature_ = jnp.asarray(temperature, dtype=mean.dtype)
        mass = jnp.asarray(molecular_mass, dtype=mean.dtype)
        if (
            mean.shape != (self.dimension,)
            or temperature_.shape != ()
            or mass.shape != ()
        ):
            raise ValueError("Reservoir mean, temperature, and mass shapes are invalid.")
        draws = jax.random.normal(
            key, (self.particle_count, self.dimension), dtype=mean.dtype
        )
        centered = draws - jnp.mean(draws, axis=0)
        raw_energy = jnp.sum(centered * centered)
        target_energy = (
            self.particle_count
            * self.dimension
            * self.boltzmann_constant
            * temperature_
            / mass
        )
        scaled = centered * jnp.sqrt(
            target_energy / jnp.maximum(raw_energy, jnp.finfo(mean.dtype).tiny)
        )
        velocity = mean + scaled
        momentum_defect = jnp.mean(velocity, axis=0) - mean
        kinetic_defect = (
            0.5 * mass * jnp.sum((velocity - mean) ** 2)
            - 0.5
            * self.particle_count
            * self.dimension
            * self.boltzmann_constant
            * temperature_
        )
        finite = jnp.all(jnp.isfinite(velocity)) & jnp.isfinite(kinetic_defect)
        tolerance = (
            512.0 * jnp.finfo(mean.dtype).eps * jnp.maximum(jnp.abs(target_energy), 1.0)
        )
        successful = (
            finite
            & (temperature_ > 0.0)
            & (mass > 0.0)
            & jnp.all(jnp.abs(momentum_defect) <= tolerance)
            & (jnp.abs(kinetic_defect) <= tolerance)
        )
        return MaxwellianReservoirResult(
            velocity, momentum_defect, kinetic_defect, finite, successful, self.plan_id
        )


class ContinuumKineticExchangeLedger(StrictModule):
    common_flux: Array
    continuum_exchange: Array
    kinetic_exchange: Array
    conservation_defect: Array
    statistical_weight: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class FixedContinuumDSMCInterfacePlan(StrictModule, NonTrainableState):
    """One uncertainty-weighted common interface flux with equal opposite updates."""

    component_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, component_count: int, /):
        count = int(component_count)
        if count <= 0:
            raise ValueError("Hybrid interface component count must be positive.")
        self.component_count = count
        self.plan_id = canonical_fingerprint(
            {"kind": "fixed-continuum-dsmc-interface", "component_count": count}
        )

    def exchange(
        self,
        continuum_flux: ArrayLike,
        kinetic_flux: ArrayLike,
        kinetic_variance: ArrayLike,
        step_size: ArrayLike,
        face_measure: ArrayLike,
        /,
    ) -> ContinuumKineticExchangeLedger:
        continuum = jnp.asarray(continuum_flux)
        kinetic = jnp.asarray(kinetic_flux, dtype=continuum.dtype)
        variance = jnp.asarray(kinetic_variance, dtype=continuum.dtype)
        step = jnp.asarray(step_size, dtype=continuum.dtype)
        measure = jnp.asarray(face_measure, dtype=continuum.dtype)
        if (
            continuum.shape[-1] != self.component_count
            or kinetic.shape != continuum.shape
            or variance.shape != continuum.shape
            or measure.shape != continuum.shape[:-1]
        ):
            raise ValueError(
                "Hybrid interface flux, variance, or measure shapes are invalid."
            )
        precision = 1.0 / jnp.maximum(variance, jnp.finfo(continuum.dtype).tiny)
        kinetic_weight = precision / (precision + 1.0)
        common = (1.0 - kinetic_weight) * continuum + kinetic_weight * kinetic
        extensive = step * measure[..., None] * common
        continuum_exchange = -extensive
        kinetic_exchange = extensive
        defect = continuum_exchange + kinetic_exchange
        finite = jnp.all(jnp.isfinite(common)) & jnp.all(jnp.isfinite(defect))
        successful = finite & jnp.all(variance >= 0.0) & jnp.all(defect == 0.0)
        return ContinuumKineticExchangeLedger(
            common,
            continuum_exchange,
            kinetic_exchange,
            defect,
            kinetic_weight,
            finite,
            successful,
            self.plan_id,
        )


class HybridRegionState(StrictModule):
    kinetic_mask: Array
    dwell_steps: Array
    epoch: Array
    policy_id: str = eqx.field(static=True)


class DynamicHybridOwnershipResult(StrictModule):
    candidate: HybridRegionState
    accepted: HybridRegionState
    entered: Array
    left: Array
    required_particles: Array
    capacity_available: Array
    finite: Array
    successful: Array
    policy_id: str = eqx.field(static=True)


class DynamicHybridOwnershipPlan(StrictModule, NonTrainableState):
    enter_threshold: float = eqx.field(static=True)
    leave_threshold: float = eqx.field(static=True)
    minimum_dwell_steps: int = eqx.field(static=True)
    buffer_layers: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        enter_threshold: float,
        leave_threshold: float,
        minimum_dwell_steps: int = 4,
        buffer_layers: int = 1,
    ):
        enter = float(enter_threshold)
        leave = float(leave_threshold)
        dwell = int(minimum_dwell_steps)
        buffers = int(buffer_layers)
        if (
            not np.isfinite(enter)
            or not np.isfinite(leave)
            or not 0.0 <= leave < enter
            or dwell < 0
            or buffers < 0
        ):
            raise ValueError("Dynamic hybrid thresholds, dwell, or buffer are invalid.")
        self.enter_threshold = enter
        self.leave_threshold = leave
        self.minimum_dwell_steps = dwell
        self.buffer_layers = buffers
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dynamic-hybrid-ownership",
                "enter_threshold": enter,
                "leave_threshold": leave,
                "minimum_dwell_steps": dwell,
                "buffer_layers": buffers,
            }
        )

    def initialize(self, kinetic_mask: ArrayLike, /) -> HybridRegionState:
        mask = jnp.asarray(kinetic_mask, dtype=bool)
        return HybridRegionState(
            mask,
            jnp.zeros(mask.shape, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            self.policy_id,
        )

    def update(
        self,
        state: HybridRegionState,
        breakdown_evidence: ArrayLike,
        adjacency: ArrayLike,
        particles_per_new_cell: int,
        particle_capacity_available: ArrayLike,
        /,
    ) -> DynamicHybridOwnershipResult:
        evidence = jnp.asarray(breakdown_evidence)
        adjacency_ = jnp.asarray(adjacency, dtype=bool)
        available = jnp.asarray(particle_capacity_available, dtype=jnp.int32)
        if (
            state.policy_id != self.policy_id
            or evidence.shape != state.kinetic_mask.shape
            or adjacency_.shape != (evidence.size, evidence.size)
            or int(particles_per_new_cell) <= 0
        ):
            raise ValueError("Dynamic hybrid state, evidence, or adjacency is invalid.")
        flattened = state.kinetic_mask.reshape((-1,))
        dwell = state.dwell_steps.reshape((-1,))
        requested = jnp.where(
            evidence.reshape((-1,)) >= self.enter_threshold,
            True,
            jnp.where(
                (evidence.reshape((-1,)) <= self.leave_threshold)
                & (dwell >= self.minimum_dwell_steps),
                False,
                flattened,
            ),
        )
        buffered = requested
        for _ in range(self.buffer_layers):
            buffered = buffered | jnp.any(adjacency_ & buffered[None, :], axis=1)
        entered = buffered & ~flattened
        left = flattened & ~buffered
        required = jnp.sum(entered) * int(particles_per_new_cell)
        capacity_ok = available >= required
        candidate_mask = buffered.reshape(state.kinetic_mask.shape)
        candidate_dwell = jnp.where(
            candidate_mask == state.kinetic_mask, state.dwell_steps + 1, 0
        )
        candidate = HybridRegionState(
            candidate_mask, candidate_dwell, state.epoch + 1, self.policy_id
        )
        accepted = HybridRegionState(
            jnp.where(capacity_ok, candidate.kinetic_mask, state.kinetic_mask),
            jnp.where(capacity_ok, candidate.dwell_steps, state.dwell_steps),
            jnp.where(capacity_ok, candidate.epoch, state.epoch),
            self.policy_id,
        )
        finite = jnp.all(jnp.isfinite(evidence))
        successful = finite & capacity_ok
        return DynamicHybridOwnershipResult(
            candidate,
            accepted,
            entered.reshape(state.kinetic_mask.shape),
            left.reshape(state.kinetic_mask.shape),
            required,
            available,
            finite,
            successful,
            self.policy_id,
        )


__all__ = [
    "ContinuumKineticExchangeLedger",
    "DynamicHybridOwnershipPlan",
    "DynamicHybridOwnershipResult",
    "FixedContinuumDSMCInterfacePlan",
    "HybridRegionState",
    "MaxwellianReservoirPlan",
    "MaxwellianReservoirResult",
]
