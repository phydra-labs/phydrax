#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..typing import checked, PRNGKey
from ._cell_dynamics import bind_atomistic_bundle_geometry
from ._potential_program import PreparedAtomisticPotentialProgram
from ._thermal import stable_particle_normals


class RingPolymerPlan(StrictModule, NonTrainableState):
    """Ring-polymer plan: ``bead_count`` is layout; the scalars are dynamic.

    ``plan_identity`` is the exact digest of the host scalar values; ``plan_id``
    decodes it on the host.
    """

    bead_count: int = eqx.field(static=True)
    temperature: Array
    step_size: Array
    centroid_friction: Array
    plan_identity: IdentityWords

    def __init__(
        self,
        bead_count: int,
        temperature: float,
        step_size: float,
        /,
        *,
        centroid_friction: float = 1.0,
    ) -> None:
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
        self.temperature = jnp.asarray(thermal, dtype=jnp.float64)
        self.step_size = jnp.asarray(step, dtype=jnp.float64)
        self.centroid_friction = jnp.asarray(friction, dtype=jnp.float64)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "ring-polymer-plan",
                "bead_count": beads,
                "temperature": thermal,
                "step_size": step,
                "centroid_friction": friction,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decode of ``plan_identity``."""
        return self.plan_identity.fingerprint()


class RingPolymerState(StrictModule):
    positions: Array
    momenta: Array
    step_index: Array
    random_key: Array
    physical_potential: Array
    spring_energy: Array
    successful: Array
    prepared_identity: IdentityWords

    @property
    def prepared_id(self) -> str:
        """Host decode of ``prepared_identity``."""
        return self.prepared_identity.fingerprint()


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
    """Ring-polymer dynamics whose bead forces use the Hamiltonian's bundle.

    Each bead configuration is an independent geometry: its neighborhood state
    is formed from the prepared bundle of ``potential`` at that bead (no state
    is shared across beads or steps), exactly as the bead relations were
    formed per evaluation before.
    """

    plan: RingPolymerPlan
    potential: PreparedAtomisticPotentialProgram
    spring_frequency: Array
    prepared_identity: IdentityWords

    @checked
    def __init__(
        self,
        plan: RingPolymerPlan,
        potential: PreparedAtomisticPotentialProgram,
        /,
    ) -> None:
        if potential.neighborhood_bundle.plan.case_count != 1:
            raise ValueError("Ring-polymer beads bind a single-case neighborhood bundle.")
        units = potential.system.plan.units
        frequency = (
            plan.bead_count
            * units.boltzmann_constant
            * float(plan.temperature)
            / units.reduced_planck_constant
        )
        self.plan = plan
        self.potential = potential
        self.spring_frequency = jnp.asarray(frequency, dtype=jnp.float64)
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-ring-polymer-dynamics",
                "plan": plan.plan_id,
                "potential": potential.prepared_id,
                "spring_frequency": frequency,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decode of ``prepared_identity``."""
        return self.prepared_identity.fingerprint()

    def _physical(self, positions: Array, /) -> tuple[Array, Array, Array]:
        system = self.potential.system
        bundle = self.potential.neighborhood_bundle
        cell = system.cell

        def evaluate_bead(bead_positions: Array) -> tuple[Array, Array, Array]:
            species = system.plan.atom_type_ids
            if cell is None:
                geometry = bind_atomistic_bundle_geometry(bundle, system, bead_positions)
                neighborhoods = bundle.initialize(geometry)
                evaluation = self.potential.evaluate(
                    bead_positions, neighborhoods, species=species
                )
            else:
                wrapped, images = cell.wrap(bead_positions)
                vectors = cell.vectors.astype(bead_positions.dtype)
                geometry = bind_atomistic_bundle_geometry(
                    bundle, system, wrapped, cell_vectors=vectors, image_counts=images
                )
                neighborhoods = bundle.initialize(geometry)
                evaluation = self.potential.evaluate(
                    wrapped,
                    neighborhoods,
                    species=species,
                    unwrapped_positions=bead_positions,
                    cell=cell,
                    cell_vectors=vectors,
                    image_counts=images,
                )
            return (
                evaluation.energy,
                evaluation.forces,
                evaluation.successful & jnp.all(neighborhoods.successful),
            )

        energies, forces, successes = eqx.filter_vmap(evaluate_bead)(positions)
        return energies, forces, jnp.all(successes)

    def _spring(self, positions: Array, /) -> tuple[Array, Array]:
        active = self.potential.system.active_mask[None, :, None]
        safe_positions = jnp.where(active, positions, 0.0)
        previous = jnp.roll(safe_positions, 1, axis=0)
        following = jnp.roll(safe_positions, -1, axis=0)
        mass = self.potential.system.plan.masses[None, :, None]
        factor = self.potential.system.plan.units.kinetic_to_energy
        omega2 = self.spring_frequency.astype(positions.dtype) ** 2
        displacement = safe_positions - previous
        energy = 0.5 * factor * omega2 * jnp.sum(mass * displacement * displacement)
        force = -factor * omega2 * mass * (2.0 * safe_positions - previous - following)
        return energy, jnp.where(active, force, 0.0)

    def initialize_state(
        self,
        positions: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        key: PRNGKey,
    ) -> RingPolymerState:
        base = jnp.asarray(positions, dtype=self.potential.system.plan.coordinate_dtype)
        expected = (self.potential.system.capacity, 3)
        if base.shape == expected:
            position = jnp.broadcast_to(base, (self.plan.bead_count,) + expected)
        elif base.shape == (self.plan.bead_count,) + expected:
            position = base
        else:
            raise ValueError("Ring-polymer positions have invalid shape.")
        position = jnp.where(
            self.potential.system.active_mask[None, :, None], position, 0.0
        )
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
        momenta = jnp.where(
            self.potential.system.mobile_mask[None, :, None], momenta, 0.0
        )
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
            self.prepared_identity,
        )

    def apply_pile(self, state: RingPolymerState, /) -> tuple[Array, Array]:
        incoming_success = state.successful & state.prepared_identity.equals(
            self.prepared_identity
        )
        bead_count = self.plan.bead_count
        dtype = state.momenta.dtype
        modes = jnp.fft.fft(state.momenta, axis=0) / jnp.sqrt(bead_count)
        mode_indices = jnp.arange(bead_count, dtype=dtype)
        mode_frequency = (
            2.0
            * self.spring_frequency.astype(dtype)
            * jnp.sin(jnp.pi * mode_indices / bead_count)
        )
        friction = jnp.where(
            mode_indices == 0,
            self.plan.centroid_friction.astype(dtype),
            2.0 * jnp.abs(mode_frequency),
        )
        decay = jnp.exp(-friction * self.plan.step_size.astype(dtype))
        mass = self.potential.system.plan.masses
        thermal_variance = (
            mass
            * self.potential.system.plan.units.boltzmann_constant
            * self.plan.temperature.astype(dtype)
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
        momenta = jnp.where(
            self.potential.system.mobile_mask[None, :, None], momenta, 0.0
        )
        momenta = jnp.where(incoming_success, momenta, state.momenta)
        heat = jnp.where(
            incoming_success,
            self._kinetic(momenta) - self._kinetic(state.momenta),
            0.0,
        )
        return momenta, heat

    def _kinetic(self, momenta: Array, /) -> Array:
        factor = self.potential.system.plan.units.kinetic_to_energy
        inverse_mass = self.potential.system.inverse_masses[None, :, None]
        return 0.5 * factor * jnp.sum(momenta * momenta * inverse_mass)

    def estimators(self, state: RingPolymerState, /) -> RingPolymerEstimators:
        active = self.potential.system.active_mask
        safe_positions = jnp.where(active[None, :, None], state.positions, 0.0)
        centroid = jnp.mean(safe_positions, axis=0)
        squared_radius = jnp.mean(
            jnp.sum((safe_positions - centroid[None, :, :]) ** 2, axis=-1),
            axis=0,
        )
        radius = jnp.sqrt(
            jnp.sum(jnp.where(active, squared_radius, 0.0))
            / jnp.maximum(jnp.sum(active), 1)
        )
        mean_potential = jnp.mean(state.physical_potential)
        primitive = (
            1.5
            * jnp.sum(self.potential.system.active_mask, dtype=jnp.int32)
            * self.plan.bead_count
            * self.potential.system.plan.units.boltzmann_constant
            * self.plan.temperature.astype(state.physical_potential.dtype)
            - state.spring_energy
            + mean_potential
        )
        return RingPolymerEstimators(
            centroid, radius, mean_potential, state.spring_energy, primitive
        )

    def step(self, state: RingPolymerState, /) -> RingPolymerStepEvaluation:
        owned = state.prepared_identity.equals(self.prepared_identity)
        dt = self.plan.step_size.astype(state.positions.dtype)
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
            state.successful & owned & success_0 & success_1,
            self.prepared_identity,
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
