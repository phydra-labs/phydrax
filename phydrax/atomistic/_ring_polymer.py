#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization import AbstractPreparedParticleNeighborhood
from ._potential_program import PreparedAtomisticPotentialProgram
from ._thermal import stable_particle_normals


class RingPolymerPlan(StrictModule, NonTrainableState):
    bead_count: int = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    centroid_friction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bead_count: int,
        temperature: float,
        step_size: float,
        /,
        *,
        centroid_friction: float = 1.0,
    ):
        beads = int(bead_count)
        thermal = float(temperature)
        step = float(step_size)
        friction = float(centroid_friction)
        if (
            beads <= 0
            or not math.isfinite(thermal)
            or thermal <= 0.0
            or not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(friction)
            or friction <= 0.0
        ):
            raise ValueError(
                "Ring-polymer beads, temperature, step, and friction are invalid."
            )
        self.bead_count = beads
        self.temperature = thermal
        self.step_size = step
        self.centroid_friction = friction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ring-polymer-plan",
                "bead_count": beads,
                "temperature": thermal,
                "step_size": step,
                "centroid_friction": friction,
            }
        )


class RingPolymerState(StrictModule):
    positions: Array
    momenta: Array
    step_index: Array
    random_key: Array
    physical_potential: Array
    spring_energy: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class RingPolymerEstimators(StrictModule):
    centroid: Array
    radius_of_gyration: Array
    mean_potential_energy: Array
    spring_energy: Array
    primitive_total_energy: Array


class RingPolymerStepEvaluation(StrictModule):
    state: RingPolymerState
    estimators: RingPolymerEstimators
    successful: Array


class PreparedRingPolymerDynamics(StrictModule):
    plan: RingPolymerPlan
    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    spring_frequency: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RingPolymerPlan,
        potential: PreparedAtomisticPotentialProgram,
        neighborhood: AbstractPreparedParticleNeighborhood,
        /,
    ):
        if not isinstance(plan, RingPolymerPlan):
            raise TypeError("plan must be RingPolymerPlan.")
        if not isinstance(potential, PreparedAtomisticPotentialProgram):
            raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        if (
            neighborhood.particle_discretization_id
            != potential.system.particles.prepared_id
        ):
            raise ValueError("Ring-polymer neighborhood belongs to another system.")
        units = potential.system.plan.units
        frequency = (
            plan.bead_count
            * units.boltzmann_constant
            * plan.temperature
            / units.reduced_planck_constant
        )
        self.plan = plan
        self.potential = potential
        self.neighborhood = neighborhood
        self.spring_frequency = frequency
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-ring-polymer-dynamics",
                "plan": plan.plan_id,
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "spring_frequency": frequency,
            }
        )

    def _physical(self, positions: Array, /) -> tuple[Array, Array, Array]:
        energies = []
        forces = []
        successes = []
        for bead in range(self.plan.bead_count):
            neighborhood = self.neighborhood.build(positions[bead])
            evaluation = self.potential.evaluate(
                positions[bead],
                neighborhood,
                species=self.potential.system.plan.atom_type_ids,
            )
            energies.append(evaluation.energy)
            forces.append(evaluation.forces)
            successes.append(evaluation.successful & neighborhood.successful)
        return (
            jnp.stack(tuple(energies)),
            jnp.stack(tuple(forces)),
            jnp.all(jnp.stack(tuple(successes))),
        )

    def _spring(self, positions: Array, /) -> tuple[Array, Array]:
        previous = jnp.roll(positions, 1, axis=0)
        following = jnp.roll(positions, -1, axis=0)
        mass = self.potential.system.plan.masses[None, :, None]
        factor = self.potential.system.plan.units.kinetic_to_energy
        omega2 = self.spring_frequency**2
        displacement = positions - previous
        energy = 0.5 * factor * omega2 * jnp.sum(mass * displacement * displacement)
        force = -factor * omega2 * mass * (2.0 * positions - previous - following)
        return energy, force

    def initialize_state(
        self,
        positions: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        key: Key[Array, ""],
    ) -> RingPolymerState:
        base = jnp.asarray(positions, dtype=self.potential.system.plan.coordinate_dtype)
        expected = (self.potential.system.capacity, 3)
        if base.shape == expected:
            position = jnp.broadcast_to(base, (self.plan.bead_count,) + expected)
        elif base.shape == (self.plan.bead_count,) + expected:
            position = base
        else:
            raise ValueError("Ring-polymer positions have invalid shape.")
        if (velocity is None) == (momentum is None):
            raise ValueError("Supply exactly one of velocity or momentum.")
        masses = self.potential.system.plan.masses
        raw = (
            momentum
            if momentum is not None
            else masses[..., None] * jnp.asarray(velocity)
        )
        raw_momentum = jnp.asarray(raw, dtype=position.dtype)
        momenta = (
            jnp.broadcast_to(raw_momentum, position.shape)
            if raw_momentum.shape == expected
            else raw_momentum
        )
        if momenta.shape != position.shape:
            raise ValueError("Ring-polymer momentum or velocity has invalid shape.")
        physical, _, physical_success = self._physical(position)
        spring, _ = self._spring(position)
        successful = (
            physical_success & jnp.isfinite(spring) & jnp.all(jnp.isfinite(momenta))
        )
        return RingPolymerState(
            position,
            momenta,
            jnp.zeros((), dtype=jnp.int32),
            jr.key_data(key).astype(jnp.uint32),
            physical,
            spring,
            successful,
            self.prepared_id,
        )

    def apply_pile(self, state: RingPolymerState, /) -> tuple[Array, Array]:
        bead_count = self.plan.bead_count
        dtype = state.momenta.dtype
        modes = jnp.fft.fft(state.momenta, axis=0) / jnp.sqrt(bead_count)
        mode_indices = jnp.arange(bead_count, dtype=dtype)
        mode_frequency = (
            2.0 * self.spring_frequency * jnp.sin(jnp.pi * mode_indices / bead_count)
        )
        friction = jnp.where(
            mode_indices == 0,
            self.plan.centroid_friction,
            2.0 * jnp.abs(mode_frequency),
        )
        decay = jnp.exp(-friction * self.plan.step_size)
        mass = self.potential.system.plan.masses
        thermal_variance = (
            mass
            * self.potential.system.plan.units.boltzmann_constant
            * self.plan.temperature
            * bead_count
            / self.potential.system.plan.units.kinetic_to_energy
        )
        normals = []
        for mode in range(bead_count):
            normals.append(
                stable_particle_normals(
                    state.random_key,
                    self.potential.system.plan.particle_ids,
                    state.step_index,
                    operator_id=1000 + mode,
                    realization_id=0,
                    dtype=dtype,
                )
            )
        bead_noise = jnp.stack(tuple(normals))
        noise = (jnp.fft.fft(bead_noise, axis=0) / jnp.sqrt(bead_count)).astype(
            modes.dtype
        )
        mode_momenta = (
            decay[:, None, None] * modes
            + jnp.sqrt(thermal_variance)[None, :, None]
            * jnp.sqrt(1.0 - decay * decay)[:, None, None]
            * noise
        )
        momenta = jnp.real(jnp.fft.ifft(mode_momenta, axis=0) * jnp.sqrt(bead_count))
        heat = self._kinetic(momenta) - self._kinetic(state.momenta)
        return momenta, heat

    def _kinetic(self, momenta: Array, /) -> Array:
        factor = self.potential.system.plan.units.kinetic_to_energy
        inverse_mass = self.potential.system.inverse_masses[None, :, None]
        return 0.5 * factor * jnp.sum(momenta * momenta * inverse_mass)

    def estimators(self, state: RingPolymerState, /) -> RingPolymerEstimators:
        centroid = jnp.mean(state.positions, axis=0)
        radius = jnp.sqrt(
            jnp.mean(jnp.sum((state.positions - centroid[None, :, :]) ** 2, axis=-1))
        )
        mean_potential = jnp.mean(state.physical_potential)
        primitive = (
            1.5
            * self.potential.system.particles.active_count
            * self.plan.bead_count
            * self.potential.system.plan.units.boltzmann_constant
            * self.plan.temperature
            - state.spring_energy
            + mean_potential
        )
        return RingPolymerEstimators(
            centroid, radius, mean_potential, state.spring_energy, primitive
        )

    def step(self, state: RingPolymerState, /) -> RingPolymerStepEvaluation:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Ring-polymer state belongs to another runtime.")
        dt = jnp.asarray(self.plan.step_size, dtype=state.positions.dtype)
        factor = self.potential.system.plan.units.force_to_momentum_rate
        _, physical_force, success_0 = self._physical(state.positions)
        _, spring_force = self._spring(state.positions)
        half = state.momenta + 0.5 * dt * factor * (physical_force + spring_force)
        position = (
            state.positions
            + dt * half * self.potential.system.inverse_masses[None, :, None]
        )
        physical_next, physical_force_next, success_1 = self._physical(position)
        spring_next, spring_force_next = self._spring(position)
        momentum = half + 0.5 * dt * factor * (physical_force_next + spring_force_next)
        staged = RingPolymerState(
            position,
            momentum,
            state.step_index + 1,
            state.random_key,
            physical_next,
            spring_next,
            success_0 & success_1,
            self.prepared_id,
        )
        thermostatted, _ = self.apply_pile(staged)
        successor = eqx.tree_at(lambda value: value.momenta, staged, thermostatted)
        successful = staged.successful & jnp.all(jnp.isfinite(thermostatted))
        successor = eqx.tree_at(lambda value: value.successful, successor, successful)
        accepted = tree_where(successful, successor, state)
        return RingPolymerStepEvaluation(accepted, self.estimators(accepted), successful)


__all__ = [
    "PreparedRingPolymerDynamics",
    "RingPolymerEstimators",
    "RingPolymerPlan",
    "RingPolymerState",
    "RingPolymerStepEvaluation",
]
