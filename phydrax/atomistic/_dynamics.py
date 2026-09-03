#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization import (
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
    ParticleVerletState,
    PreparedVerletParticleNeighborhood,
)
from ._constraints import PreparedDistanceConstraints
from ._potential_program import (
    AtomisticPotentialEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._system import PreparedAtomisticSystem
from ._thermal import (
    apply_baoab_ornstein_uhlenbeck,
    BAOABLangevinPlan,
)


class AtomisticDynamicsStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    REJECTED = 2
    NONFINITE = 3


class AtomisticStepRejectionReason(IntFlag):
    NONE = 0
    CELL_CAPACITY = 1 << 0
    PAIR_CAPACITY = 1 << 1
    DOMAIN = 1 << 2
    RELATION = 1 << 3
    IMAGE_AMBIGUITY = 1 << 4
    POTENTIAL = 1 << 5
    CONSTRAINT = 1 << 6
    NONFINITE = 1 << 7
    STALE_FORCE = 1 << 8
    UNIT_SYSTEM = 1 << 9
    THERMOSTAT = 1 << 10


class AtomisticKinematics(StrictModule):
    positions: Array
    momenta: Array
    image_counts: Array


class AtomisticForceState(StrictModule):
    forces: Array
    potential_energy: Array
    term_energies: Array
    atom_energy: Array
    virial: Array
    neighborhood_epoch: Array
    position_epoch: Array
    successful: Array
    program_id: str = eqx.field(static=True)


class AtomisticEnergyLedgerState(StrictModule):
    initial_kinetic_energy: Array
    initial_potential_energy: Array
    kinetic_energy: Array
    potential_energy: Array
    total_energy: Array
    thermostat_heat: Array
    barostat_work: Array
    external_work: Array
    constraint_work: Array
    cumulative_balance_residual: Array
    last_relative_energy_change: Array
    accepted_steps: Array


class AtomisticDynamicsState(StrictModule):
    time: Array
    step_index: Array
    kinematics: AtomisticKinematics
    species: Array
    cell_vectors: Array
    neighborhood: ParticleNeighborhoodState
    neighborhood_cache: ParticleVerletState | None
    force: AtomisticForceState
    constraint_lagrange: Array
    constraint_position_residual: Array
    constraint_velocity_residual: Array
    thermostat_state: Array
    barostat_state: Array
    random_key: Array
    energy: AtomisticEnergyLedgerState
    last_status: Array
    last_rejection_reasons: Array
    prepared_dynamics_id: str = eqx.field(static=True)


class AtomisticDynamicsDiagnostics(StrictModule):
    kinetic_energy: Array
    potential_energy: Array
    total_energy: Array
    temperature: Array
    pressure: Array
    total_linear_momentum: Array
    total_angular_momentum: Array
    net_internal_force: Array
    net_internal_torque: Array
    minimum_pair_distance: Array
    cutoff_margin: Array
    image_uniqueness_margin: Array
    neighborhood_rebuilt: Array
    neighborhood_rebuild_count: Array
    neighborhood_certificate_margin: Array
    constraint_residual: Array
    energy: AtomisticEnergyLedgerState
    successful: Array
    rejection_reasons: Array


class AtomisticStepEvaluation(StrictModule):
    candidate_state: AtomisticDynamicsState
    accepted_state: AtomisticDynamicsState
    potential: AtomisticPotentialEvaluation
    diagnostics: AtomisticDynamicsDiagnostics
    successful: Array
    residual: Array
    work: Array
    rejection_reasons: Array


class VelocityVerletPlan(StrictModule, NonTrainableState):
    step_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, step_size: float, /):
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        self.step_size = step
        self.plan_id = canonical_fingerprint(
            {"kind": "atomistic-velocity-verlet", "step_size": step}
        )


AtomisticIntegratorPlan: TypeAlias = VelocityVerletPlan | BAOABLangevinPlan


class AtomisticDynamicsPlan(StrictModule, NonTrainableState):
    system: PreparedAtomisticSystem
    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    integrator: AtomisticIntegratorPlan
    constraints: PreparedDistanceConstraints | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        potential: PreparedAtomisticPotentialProgram,
        neighborhood: AbstractPreparedParticleNeighborhood,
        integrator: AtomisticIntegratorPlan,
        /,
        *,
        constraints: PreparedDistanceConstraints | None = None,
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be a PreparedAtomisticSystem.")
        if not isinstance(potential, PreparedAtomisticPotentialProgram):
            raise TypeError("potential must be a PreparedAtomisticPotentialProgram.")
        if potential.system.prepared_id != system.prepared_id:
            raise ValueError("Potential program belongs to another atomistic system.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        if neighborhood.particle_discretization_id != system.particles.prepared_id:
            raise ValueError("Neighborhood belongs to another particle support.")
        if not isinstance(integrator, (VelocityVerletPlan, BAOABLangevinPlan)):
            raise TypeError("integrator must be VelocityVerletPlan or BAOABLangevinPlan.")
        if constraints is not None:
            if not isinstance(constraints, PreparedDistanceConstraints):
                raise TypeError(
                    "constraints must be PreparedDistanceConstraints or None."
                )
            if constraints.system.prepared_id != system.prepared_id:
                raise ValueError("Constraints belong to another atomistic system.")
        cutoff = potential.plan.requirements.cutoff
        if cutoff is not None and system.cell is not None:
            skin = (
                neighborhood.plan.skin
                if isinstance(neighborhood, PreparedVerletParticleNeighborhood)
                else 0.0
            )
            system.cell.require_unique_image(cutoff + skin)
        self.system = system
        self.potential = potential
        self.neighborhood = neighborhood
        self.integrator = integrator
        self.constraints = constraints
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-dynamics-plan",
                "system": system.prepared_id,
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "integrator": integrator.plan_id,
                "constraints": None if constraints is None else constraints.prepared_id,
            }
        )

    def prepare(self, /) -> "PreparedAtomisticDynamics":
        return PreparedAtomisticDynamics(self)


class PreparedAtomisticDynamics(StrictModule):
    plan: AtomisticDynamicsPlan
    system: PreparedAtomisticSystem
    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    integrator: AtomisticIntegratorPlan
    constraints: PreparedDistanceConstraints | None
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: AtomisticDynamicsPlan, /):
        if not isinstance(plan, AtomisticDynamicsPlan):
            raise TypeError("plan must be an AtomisticDynamicsPlan.")
        self.plan = plan
        self.system = plan.system
        self.potential = plan.potential
        self.neighborhood = plan.neighborhood
        self.integrator = plan.integrator
        self.constraints = plan.constraints
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-atomistic-dynamics", "plan": plan.plan_id}
        )

    def _unwrapped(
        self,
        kinematics: AtomisticKinematics,
        cell_vectors: Array | None = None,
        /,
    ) -> Array:
        cell = self.system.cell
        if cell is None:
            return kinematics.positions
        vectors = (
            cell.vectors.astype(kinematics.positions.dtype)
            if cell_vectors is None or cell_vectors.size == 0
            else jnp.asarray(cell_vectors, dtype=kinematics.positions.dtype)
        )
        return kinematics.positions + contract(
            "ni,ij->nj",
            kinematics.image_counts.astype(kinematics.positions.dtype),
            vectors,
        )

    def velocity(self, state: AtomisticDynamicsState, /) -> Array:
        return state.kinematics.momenta * self.system.inverse_masses[:, None]

    def interaction_sites(self, state: AtomisticDynamicsState, /):
        if state.prepared_dynamics_id != self.prepared_id:
            raise ValueError("State belongs to another atomistic dynamics runtime.")
        if self.system.cell is None or state.cell_vectors.size == 0:
            return self.system.coordinate_map.realize(state.kinematics.positions)
        fractional = self.system.cell.fractional_with_vectors(
            state.kinematics.positions, state.cell_vectors
        )
        return self.system.coordinate_map.realize(
            state.kinematics.positions,
            cell=self.system.cell,
            fractional_positions=fractional,
            cell_vectors=state.cell_vectors,
        )

    def kinetic_energy(self, momenta: Array, /) -> Array:
        kinetic_factor = self.system.plan.units.kinetic_to_energy
        per_atom = (
            0.5
            * kinetic_factor
            * jnp.sum(momenta * momenta * self.system.inverse_masses[:, None], axis=-1)
        )
        return jnp.sum(jnp.where(self.system.mobile_mask, per_atom, 0.0))

    def _build_neighborhood(
        self,
        positions: Array,
        previous: ParticleVerletState | None,
        cell_vectors: Array | None = None,
        /,
    ) -> tuple[ParticleNeighborhoodState, ParticleVerletState | None]:
        if isinstance(self.neighborhood, PreparedVerletParticleNeighborhood):
            cache = (
                self.neighborhood.initialize(positions, cell_vectors=cell_vectors)
                if previous is None
                else self.neighborhood.update(
                    positions, previous, cell_vectors=cell_vectors
                )
            )
            return cache.neighborhood, cache
        return self.neighborhood.build(positions), None

    def _force_state(
        self,
        evaluation: AtomisticPotentialEvaluation,
        neighborhood_cache: ParticleVerletState | None,
        position_epoch: Array,
        /,
    ) -> AtomisticForceState:
        neighborhood_epoch = (
            jnp.zeros((), dtype=jnp.int32)
            if neighborhood_cache is None
            else neighborhood_cache.epoch
        )
        return AtomisticForceState(
            forces=evaluation.forces,
            potential_energy=evaluation.energy,
            term_energies=evaluation.term_energies,
            atom_energy=evaluation.atom_energy,
            virial=evaluation.virial,
            neighborhood_epoch=neighborhood_epoch,
            position_epoch=jnp.asarray(position_epoch, dtype=jnp.int32),
            successful=evaluation.successful,
            program_id=evaluation.program_id,
        )

    def initialize_state(
        self,
        positions: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        species: ArrayLike | None = None,
        key: Key[Array, ""],
    ) -> AtomisticDynamicsState:
        if (velocity is None) == (momentum is None):
            raise ValueError("Supply exactly one of velocity or momentum.")
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        expected = (self.system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        active_finite = jnp.all(
            jnp.isfinite(jnp.where(self.system.active_mask[:, None], position, 0.0))
        )
        position = eqx.error_if(
            position,
            ~active_finite,
            "Active atomistic positions must be finite.",
        )
        position = jnp.where(self.system.active_mask[:, None], position, 0.0)
        if self.system.cell is None:
            wrapped = position
            image_counts = jnp.zeros(expected, dtype=jnp.int32)
            cell_vectors = jnp.zeros((0, 0), dtype=position.dtype)
        else:
            wrapped, image_counts = self.system.cell.wrap(position)
            cell_vectors = self.system.cell.vectors.astype(position.dtype)
        masses = self.system.plan.masses.astype(position.dtype)
        momenta = (
            jnp.asarray(momentum, dtype=position.dtype)
            if momentum is not None
            else masses[:, None] * jnp.asarray(velocity, dtype=position.dtype)
        )
        if momenta.shape != expected:
            raise ValueError(f"momentum or velocity must have shape {expected}.")
        mobile = self.system.mobile_mask[:, None]
        momenta = jnp.where(mobile, momenta, 0.0)
        constraint_lagrange = jnp.zeros(
            (self.system.topology.constraint_count,), dtype=position.dtype
        )
        constraint_position_residual = jnp.zeros((), dtype=position.dtype)
        constraint_velocity_residual = jnp.zeros((), dtype=position.dtype)
        constraint_successful = jnp.asarray(True)
        if self.constraints is not None:
            projection = self.constraints.project_positions(position, position, momenta)
            momenta = projection.momenta
            constraint_lagrange = projection.multipliers
            constraint_position_residual = projection.position_residual
            constraint_velocity_residual = projection.velocity_residual
            constraint_successful = projection.successful
            if self.system.cell is None:
                wrapped = projection.positions
                image_counts = jnp.zeros(expected, dtype=jnp.int32)
            else:
                wrapped, image_counts = self.system.cell.wrap(projection.positions)
        kinematics = AtomisticKinematics(wrapped, momenta, image_counts)
        neighborhood, cache = self._build_neighborhood(wrapped, None, cell_vectors)
        species_ = (
            self.system.plan.atom_type_ids
            if species is None
            else jnp.asarray(species, dtype=jnp.int32)
        )
        potential_kwargs: dict[str, Any] = {
            "unwrapped_positions": self._unwrapped(kinematics, cell_vectors),
            "species": species_,
            "cell": self.system.cell,
        }
        if self.system.cell is not None and cell_vectors is not None:
            dynamic_vectors = jnp.asarray(cell_vectors, dtype=wrapped.dtype)
            potential_kwargs["fractional_positions"] = (
                self.system.cell.fractional_with_vectors(wrapped, dynamic_vectors)
            )
            potential_kwargs["cell_vectors"] = dynamic_vectors
        evaluation = self.potential.evaluate(
            wrapped,
            neighborhood,
            **potential_kwargs,
        )
        force = self._force_state(evaluation, cache, jnp.zeros((), dtype=jnp.int32))
        kinetic = self.kinetic_energy(momenta)
        total = kinetic + evaluation.energy
        zero = jnp.zeros((), dtype=position.dtype)
        ledger = AtomisticEnergyLedgerState(
            initial_kinetic_energy=kinetic,
            initial_potential_energy=evaluation.energy,
            kinetic_energy=kinetic,
            potential_energy=evaluation.energy,
            total_energy=total,
            thermostat_heat=zero,
            barostat_work=zero,
            external_work=zero,
            constraint_work=zero,
            cumulative_balance_residual=zero,
            last_relative_energy_change=zero,
            accepted_steps=jnp.zeros((), dtype=jnp.int32),
        )
        successful = (
            neighborhood.successful
            & evaluation.successful
            & constraint_successful
            & tree_allfinite((kinematics, ledger))
        )
        checked = eqx.error_if(
            wrapped,
            ~successful,
            "Initial atomistic dynamics state is not admissible.",
        )
        return AtomisticDynamicsState(
            time=jnp.asarray(time, dtype=position.dtype).reshape(()),
            step_index=jnp.zeros((), dtype=jnp.int32),
            kinematics=AtomisticKinematics(checked, momenta, image_counts),
            species=species_,
            cell_vectors=cell_vectors,
            neighborhood=neighborhood,
            neighborhood_cache=cache,
            force=force,
            constraint_lagrange=constraint_lagrange,
            constraint_position_residual=constraint_position_residual,
            constraint_velocity_residual=constraint_velocity_residual,
            thermostat_state=jnp.zeros(
                (2 if isinstance(self.integrator, BAOABLangevinPlan) else 0,),
                dtype=position.dtype,
            ),
            barostat_state=jnp.zeros((0,), dtype=position.dtype),
            random_key=jr.key_data(key).astype(jnp.uint32),
            energy=ledger,
            last_status=jnp.asarray(
                int(AtomisticDynamicsStatus.SUCCESS), dtype=jnp.int32
            ),
            last_rejection_reasons=jnp.zeros((), dtype=jnp.int32),
            prepared_dynamics_id=self.prepared_id,
        )

    def _rejection_reasons(
        self,
        state: AtomisticDynamicsState,
        neighborhood: ParticleNeighborhoodState,
        potential: AtomisticPotentialEvaluation,
        candidate_finite: Array,
        constraint_successful: Array,
        thermostat_successful: Array,
        /,
    ) -> Array:
        reasons = jnp.zeros((), dtype=jnp.int32)
        reasons = reasons | jnp.where(
            neighborhood.cell_overflow,
            int(AtomisticStepRejectionReason.CELL_CAPACITY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            neighborhood.pair_overflow,
            int(AtomisticStepRejectionReason.PAIR_CAPACITY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            neighborhood.domain_violation,
            int(AtomisticStepRejectionReason.DOMAIN),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~potential.successful,
            int(AtomisticStepRejectionReason.POTENTIAL),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~constraint_successful,
            int(AtomisticStepRejectionReason.CONSTRAINT),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~thermostat_successful,
            int(AtomisticStepRejectionReason.THERMOSTAT),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~candidate_finite,
            int(AtomisticStepRejectionReason.NONFINITE),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            state.force.position_epoch != state.step_index,
            int(AtomisticStepRejectionReason.STALE_FORCE),
            0,
        ).astype(jnp.int32)
        return reasons

    def step_detailed(self, state: AtomisticDynamicsState, /) -> AtomisticStepEvaluation:
        if not isinstance(state, AtomisticDynamicsState):
            raise TypeError("state must be an AtomisticDynamicsState.")
        if state.prepared_dynamics_id != self.prepared_id:
            raise ValueError(
                "State belongs to another prepared atomistic dynamics runtime."
            )
        dt = jnp.asarray(
            self.integrator.step_size, dtype=state.kinematics.positions.dtype
        )
        half = 0.5 * dt
        force_scale = self.system.plan.units.force_to_momentum_rate
        mobile = self.system.mobile_mask[:, None]
        inverse_mass = self.system.inverse_masses[:, None]
        previous_unwrapped = self._unwrapped(state.kinematics, state.cell_vectors)
        half_momentum = state.kinematics.momenta + half * force_scale * state.force.forces
        half_momentum = jnp.where(mobile, half_momentum, 0.0)
        thermostat_heat = jnp.zeros((), dtype=dt.dtype)
        thermostat_state = state.thermostat_state
        thermostat_successful = jnp.asarray(True)
        if isinstance(self.integrator, BAOABLangevinPlan):
            middle_position = previous_unwrapped + half * half_momentum * inverse_mass
            thermostat = apply_baoab_ornstein_uhlenbeck(
                self.integrator,
                state.random_key,
                self.system.plan.particle_ids,
                half_momentum,
                self.system.plan.masses,
                self.system.mobile_mask,
                state.step_index,
                boltzmann_constant=self.system.plan.units.boltzmann_constant,
                kinetic_to_energy=self.system.plan.units.kinetic_to_energy,
            )
            moving_momentum = thermostat.momenta
            proposed_unwrapped = middle_position + half * moving_momentum * inverse_mass
            thermostat_heat = thermostat.heat
            thermostat_state = jnp.stack((thermostat.heat, thermostat.decay))
            thermostat_successful = thermostat.successful
        else:
            moving_momentum = half_momentum
            proposed_unwrapped = previous_unwrapped + dt * moving_momentum * inverse_mass
        proposed_unwrapped = jnp.where(mobile, proposed_unwrapped, previous_unwrapped)
        constraint_lagrange = state.constraint_lagrange
        constraint_position_residual = jnp.zeros((), dtype=dt.dtype)
        constraint_velocity_residual = jnp.zeros((), dtype=dt.dtype)
        constraint_successful = jnp.asarray(True)
        constraint_work = jnp.zeros((), dtype=dt.dtype)
        if self.constraints is not None:
            kinetic_before_constraint = self.kinetic_energy(moving_momentum)
            projection = self.constraints.project_positions(
                previous_unwrapped, proposed_unwrapped, moving_momentum
            )
            proposed_unwrapped = projection.positions
            moving_momentum = projection.momenta
            constraint_lagrange = projection.multipliers
            constraint_position_residual = projection.position_residual
            constraint_velocity_residual = projection.velocity_residual
            constraint_successful = projection.successful
            constraint_work = (
                self.kinetic_energy(moving_momentum) - kinetic_before_constraint
            )
        if self.system.cell is None:
            next_position = proposed_unwrapped
            next_images = state.kinematics.image_counts
        else:
            next_position, next_images = self.system.cell.wrap_with_vectors(
                proposed_unwrapped, state.cell_vectors
            )
        neighborhood, cache = self._build_neighborhood(
            next_position, state.neighborhood_cache, state.cell_vectors
        )
        potential_kwargs: dict[str, Any] = {
            "unwrapped_positions": proposed_unwrapped,
            "species": state.species,
            "cell": self.system.cell,
        }
        if (
            self.system.cell is not None
            and not self.potential.plan.requirements.directed_graph
        ):
            potential_kwargs["fractional_positions"] = (
                self.system.cell.fractional_with_vectors(
                    next_position, state.cell_vectors
                )
            )
            potential_kwargs["cell_vectors"] = state.cell_vectors
        evaluation = self.potential.evaluate(
            next_position,
            neighborhood,
            **potential_kwargs,
        )
        next_momentum_raw = moving_momentum + half * force_scale * evaluation.forces
        next_momentum_raw = jnp.where(mobile, next_momentum_raw, 0.0)
        next_momentum = next_momentum_raw
        if self.constraints is not None:
            projected_momentum, velocity_residual = self.constraints.project_momenta(
                proposed_unwrapped, next_momentum_raw
            )
            constraint_work = constraint_work + (
                self.kinetic_energy(projected_momentum)
                - self.kinetic_energy(next_momentum_raw)
            )
            next_momentum = projected_momentum
            constraint_velocity_residual = jnp.maximum(
                constraint_velocity_residual, velocity_residual
            )
            constraint_successful = constraint_successful & (
                velocity_residual <= self.constraints.plan.tolerance
            )
        next_kinematics = AtomisticKinematics(next_position, next_momentum, next_images)
        kinetic = self.kinetic_energy(next_momentum)
        total = kinetic + evaluation.energy
        previous_total = state.energy.total_energy
        delta = total - previous_total
        balance = delta - thermostat_heat - constraint_work
        scale = jnp.maximum(jnp.maximum(jnp.abs(total), jnp.abs(previous_total)), 1.0e-30)
        candidate_ledger = AtomisticEnergyLedgerState(
            initial_kinetic_energy=state.energy.initial_kinetic_energy,
            initial_potential_energy=state.energy.initial_potential_energy,
            kinetic_energy=kinetic,
            potential_energy=evaluation.energy,
            total_energy=total,
            thermostat_heat=state.energy.thermostat_heat + thermostat_heat,
            barostat_work=state.energy.barostat_work,
            external_work=state.energy.external_work,
            constraint_work=state.energy.constraint_work + constraint_work,
            cumulative_balance_residual=(
                state.energy.cumulative_balance_residual + balance
            ),
            last_relative_energy_change=jnp.abs(balance) / scale,
            accepted_steps=state.energy.accepted_steps + 1,
        )
        next_epoch = state.step_index + 1
        force = self._force_state(evaluation, cache, next_epoch)
        candidate_finite = tree_allfinite((next_kinematics, candidate_ledger, force))
        reasons = self._rejection_reasons(
            state,
            neighborhood,
            evaluation,
            candidate_finite,
            constraint_successful,
            thermostat_successful,
        )
        successful = reasons == 0
        status = jnp.where(
            successful,
            int(AtomisticDynamicsStatus.SUCCESS),
            int(AtomisticDynamicsStatus.REJECTED),
        ).astype(jnp.int32)
        candidate = AtomisticDynamicsState(
            time=state.time + dt,
            step_index=next_epoch,
            kinematics=next_kinematics,
            species=state.species,
            cell_vectors=state.cell_vectors,
            neighborhood=neighborhood,
            neighborhood_cache=cache,
            force=force,
            constraint_lagrange=constraint_lagrange,
            constraint_position_residual=constraint_position_residual,
            constraint_velocity_residual=constraint_velocity_residual,
            thermostat_state=thermostat_state,
            barostat_state=state.barostat_state,
            random_key=state.random_key,
            energy=candidate_ledger,
            last_status=status,
            last_rejection_reasons=reasons,
            prepared_dynamics_id=self.prepared_id,
        )
        accepted = tree_where(successful, candidate, state)
        diagnostics = self.diagnostics(candidate, successful, reasons)
        pair_count = neighborhood.candidate_pair_count
        work = pair_count + jnp.asarray(len(self.potential.terms), dtype=jnp.int32)
        return AtomisticStepEvaluation(
            candidate_state=candidate,
            accepted_state=accepted,
            potential=evaluation,
            diagnostics=diagnostics,
            successful=successful,
            residual=candidate_ledger.last_relative_energy_change,
            work=work,
            rejection_reasons=reasons,
        )

    def diagnostics(
        self,
        state: AtomisticDynamicsState,
        successful: Array | None = None,
        rejection_reasons: Array | None = None,
        /,
    ) -> AtomisticDynamicsDiagnostics:
        velocity = self.velocity(state)
        mass = self.system.plan.masses
        mobile = self.system.mobile_mask
        momentum = jnp.sum(
            jnp.where(mobile[:, None], state.kinematics.momenta, 0.0), axis=0
        )
        unwrapped = self._unwrapped(state.kinematics, state.cell_vectors)
        mass_sum = jnp.sum(jnp.where(mobile, mass, 0.0))
        center = (
            jnp.sum(jnp.where(mobile[:, None], mass[:, None] * unwrapped, 0.0), axis=0)
            / mass_sum
        )
        angular = jnp.sum(jnp.cross(unwrapped - center, state.kinematics.momenta), axis=0)
        force = jnp.where(self.system.active_mask[:, None], state.force.forces, 0.0)
        net_force = jnp.sum(force, axis=0)
        net_torque = jnp.sum(jnp.cross(unwrapped - center, force), axis=0)
        valid_pairs = state.neighborhood.pair_relation.valid
        diagnostic_kwargs: dict[str, Any] = {
            "unwrapped_positions": unwrapped,
            "species": state.species,
            "cell": self.system.cell,
        }
        if (
            self.system.cell is not None
            and not self.potential.plan.requirements.directed_graph
        ):
            diagnostic_kwargs["fractional_positions"] = (
                self.system.cell.fractional_with_vectors(
                    state.kinematics.positions, state.cell_vectors
                )
            )
            diagnostic_kwargs["cell_vectors"] = state.cell_vectors
        distances = self.potential.context(
            state.kinematics.positions,
            state.neighborhood,
            **diagnostic_kwargs,
        ).pair_distance
        minimum_distance = jnp.min(jnp.where(valid_pairs, distances, jnp.inf))
        cutoff = self.potential.plan.requirements.cutoff
        cutoff_margin = (
            jnp.asarray(jnp.inf, dtype=distances.dtype)
            if cutoff is None
            else jnp.min(jnp.where(valid_pairs, jnp.abs(distances - cutoff), jnp.inf))
        )
        image_margin = (
            jnp.asarray(jnp.inf, dtype=distances.dtype)
            if self.system.cell is None or cutoff is None
            else jnp.asarray(
                self.system.cell.unique_image_radius - cutoff, dtype=distances.dtype
            )
        )
        rebuilt = (
            jnp.asarray(True)
            if state.neighborhood_cache is None
            else state.neighborhood_cache.rebuilt
        )
        rebuild_count = (
            jnp.zeros((), dtype=jnp.int32)
            if state.neighborhood_cache is None
            else state.neighborhood_cache.rebuild_count
        )
        certificate = (
            jnp.asarray(jnp.inf, dtype=distances.dtype)
            if state.neighborhood_cache is None
            else state.neighborhood_cache.certificate_margin
        )
        temperature = (
            2.0
            * state.energy.kinetic_energy
            / (self.system.degrees_of_freedom * self.system.plan.units.boltzmann_constant)
        )
        kinetic_virial = self.system.plan.units.kinetic_to_energy * contract(
            "ni,nj,n->ij", velocity, velocity, mass
        )
        current_volume = (
            jnp.asarray(jnp.nan, dtype=distances.dtype)
            if self.system.cell is None
            else jnp.abs(
                jnp.sum(
                    state.cell_vectors[0]
                    * jnp.cross(state.cell_vectors[1], state.cell_vectors[2])
                )
            )
        )
        pressure = (
            jnp.asarray(jnp.nan, dtype=distances.dtype)
            if self.system.cell is None
            else jnp.trace(kinetic_virial + state.force.virial) / (3.0 * current_volume)
        )
        success = (
            state.last_status == int(AtomisticDynamicsStatus.SUCCESS)
            if successful is None
            else jnp.asarray(successful, dtype=bool)
        )
        reasons = (
            state.last_rejection_reasons
            if rejection_reasons is None
            else jnp.asarray(rejection_reasons, dtype=jnp.int32)
        )
        return AtomisticDynamicsDiagnostics(
            kinetic_energy=state.energy.kinetic_energy,
            potential_energy=state.energy.potential_energy,
            total_energy=state.energy.total_energy,
            temperature=temperature,
            pressure=pressure,
            total_linear_momentum=momentum,
            total_angular_momentum=angular,
            net_internal_force=net_force,
            net_internal_torque=net_torque,
            minimum_pair_distance=minimum_distance,
            cutoff_margin=cutoff_margin,
            image_uniqueness_margin=image_margin,
            neighborhood_rebuilt=rebuilt,
            neighborhood_rebuild_count=rebuild_count,
            neighborhood_certificate_margin=certificate,
            constraint_residual=jnp.maximum(
                state.constraint_position_residual,
                state.constraint_velocity_residual,
            ),
            energy=state.energy,
            successful=success,
            rejection_reasons=reasons,
        )


__all__ = [
    "AtomisticDynamicsDiagnostics",
    "AtomisticDynamicsPlan",
    "AtomisticDynamicsState",
    "AtomisticDynamicsStatus",
    "AtomisticEnergyLedgerState",
    "AtomisticForceState",
    "AtomisticKinematics",
    "AtomisticStepEvaluation",
    "AtomisticStepRejectionReason",
    "PreparedAtomisticDynamics",
    "VelocityVerletPlan",
]
