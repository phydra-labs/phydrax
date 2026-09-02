#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..fem import PreparedFiniteElementCellMap
from ..particle import ParticlePopulationState
from ..splatting import (
    ParticleGridSplatEpoch,
    prepare_particle_grid_splat_transition,
    PreparedMeshParticleGridSplat,
)
from ._types import FLIPParticleState


class ALEFLIPPlan(StrictModule, NonTrainableState):
    splat: PreparedMeshParticleGridSplat
    cell_map: PreparedFiniteElementCellMap
    gcl_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        splat: PreparedMeshParticleGridSplat,
        cell_map: PreparedFiniteElementCellMap,
        /,
        *,
        gcl_tolerance: float = 1.0e-9,
    ):
        if not isinstance(splat, PreparedMeshParticleGridSplat):
            raise TypeError("splat must be PreparedMeshParticleGridSplat.")
        if not isinstance(cell_map, PreparedFiniteElementCellMap):
            raise TypeError("cell_map must be PreparedFiniteElementCellMap.")
        if splat.target.mesh.topology_id != cell_map.topology_id:
            raise ValueError("ALE FLIP splat and FE cell map topologies differ.")
        if gcl_tolerance <= 0.0:
            raise ValueError("gcl_tolerance must be positive.")
        self.splat = splat
        self.cell_map = cell_map
        self.gcl_tolerance = float(gcl_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ale-flip-plan",
                "splat": splat.prepared_id,
                "cell_map": cell_map.cell_map_id,
                "gcl_tolerance": self.gcl_tolerance,
            }
        )


class PreparedALEFLIP(StrictModule, NonTrainableState):
    plan: ALEFLIPPlan
    epoch_number: Array
    epoch_id: str = eqx.field(static=True)


class ALEFLIPState(StrictModule):
    epoch: ParticleGridSplatEpoch
    particles: FLIPParticleState
    coordinates: Array
    pressure: Array
    history: Array


class ALEFLIPStepResult(StrictModule):
    candidate_state: ALEFLIPState
    accepted_state: ALEFLIPState
    relative_particle_velocity: Array
    deposited_mass: Array
    deposited_momentum: Array
    mass_defect: Array
    momentum_defect: Array
    gcl_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def prepare_ale_flip(
    plan: ALEFLIPPlan, epoch: ParticleGridSplatEpoch, /
) -> PreparedALEFLIP:
    if not isinstance(plan, ALEFLIPPlan):
        raise TypeError("plan must be ALEFLIPPlan.")
    if not isinstance(epoch, ParticleGridSplatEpoch):
        raise TypeError("epoch must be ParticleGridSplatEpoch.")
    if epoch.prepared.prepared_id != plan.splat.prepared_id:
        raise ValueError("ALE FLIP epoch does not match its prepared splat.")
    return PreparedALEFLIP(
        plan,
        epoch.epoch_number,
        epoch_id=canonical_fingerprint(
            {"kind": "prepared-ale-flip", "plan": plan.plan_id, "epoch": epoch.epoch_id}
        ),
    )


def advance_ale_flip(
    prepared: PreparedALEFLIP,
    state: ALEFLIPState,
    target_velocity: ArrayLike,
    target_grid_velocity: ArrayLike,
    new_coordinates: ArrayLike,
    step_size: ArrayLike,
    /,
) -> ALEFLIPStepResult:
    """Advance physical particles with u and report fixed-topology relative u-w/GCL."""

    if not isinstance(prepared, PreparedALEFLIP):
        raise TypeError("prepared must be PreparedALEFLIP.")
    if not isinstance(state, ALEFLIPState):
        raise TypeError("state must be ALEFLIPState.")
    if state.epoch.prepared.prepared_id != prepared.plan.splat.prepared_id:
        raise ValueError("ALE FLIP state belongs to a different splat epoch.")
    velocity = jnp.asarray(target_velocity)
    grid_velocity = jnp.asarray(target_grid_velocity, dtype=velocity.dtype)
    coordinates = jnp.asarray(new_coordinates, dtype=state.coordinates.dtype)
    dt = jnp.asarray(step_size, dtype=state.particles.position.dtype).reshape(())
    target_shape = (
        prepared.plan.splat.target.entity_count,
        prepared.plan.splat.target.mesh.ambient_dimension,
    )
    coordinate_shape = (
        prepared.plan.cell_map.coordinate_count,
        prepared.plan.cell_map.ambient_dimension,
    )
    if velocity.shape != grid_velocity.shape or velocity.shape != target_shape:
        raise ValueError("ALE target/grid velocities have incompatible target shape.")
    if (
        state.coordinates.shape != coordinate_shape
        or coordinates.shape != coordinate_shape
    ):
        raise ValueError("ALE coordinates do not match the prepared FE cell map.")
    if (
        state.particles.position.shape != state.epoch.positions.shape
        or state.epoch.population.active.shape != (state.particles.position.shape[0],)
    ):
        raise ValueError("ALE particle and splat epoch capacities differ.")
    particle_velocity = prepared.plan.splat.gather(
        state.particles.position, state.epoch.population.active, velocity
    )
    particle_grid_velocity = prepared.plan.splat.gather(
        state.particles.position, state.epoch.population.active, grid_velocity
    )
    relative = particle_velocity.values - particle_grid_velocity.values
    candidate_position = state.particles.position + dt * particle_velocity.values
    mass = state.epoch.population.mass
    momentum = mass[:, None] * particle_velocity.values
    deposited_mass = prepared.plan.splat.deposit(
        candidate_position, state.epoch.population.active, mass
    )
    deposited_momentum = prepared.plan.splat.deposit(
        candidate_position, state.epoch.population.active, momentum
    )
    target_mass = jnp.sum(deposited_mass.content)
    represented_mass = deposited_mass.represented_content
    mass_defect = target_mass - represented_mass
    target_momentum = jnp.sum(deposited_momentum.content, axis=0)
    represented_momentum = deposited_momentum.represented_content
    momentum_defect = target_momentum - represented_momentum
    coordinate_displacement = coordinates - state.coordinates
    grid_displacement = dt * grid_velocity
    gcl_shape_compatible = grid_displacement.shape == coordinate_displacement.shape
    if gcl_shape_compatible:
        gcl_defect = jnp.max(jnp.abs(coordinate_displacement - grid_displacement))
    else:
        gcl_defect = jnp.asarray(jnp.inf, dtype=dt.dtype)
    particle_candidate = eqx.tree_at(
        lambda value: (value.position, value.velocity),
        state.particles,
        (candidate_position, particle_velocity.values),
    )
    epoch_candidate = ParticleGridSplatEpoch(
        state.epoch.prepared,
        state.epoch.population,
        candidate_position,
        jnp.concatenate(
            (deposited_mass.content[..., None], deposited_momentum.content),
            axis=-1,
        ),
        epoch_number=state.epoch.epoch_number,
    )
    candidate = ALEFLIPState(
        epoch_candidate, particle_candidate, coordinates, state.pressure, state.history
    )
    finite = (
        jnp.all(jnp.isfinite(candidate_position))
        & jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(relative))
        & jnp.all(jnp.isfinite(deposited_mass.content))
        & jnp.all(jnp.isfinite(deposited_momentum.content))
        & jnp.all(jnp.isfinite(momentum_defect))
        & jnp.isfinite(dt)
    )
    mass_scale = jnp.maximum(jnp.abs(represented_mass), 1.0)
    momentum_scale = jnp.maximum(jnp.max(jnp.abs(represented_momentum)), 1.0)
    epoch_consistent = (state.epoch.epoch_number == prepared.epoch_number) & jnp.all(
        jnp.where(
            state.epoch.population.active[:, None],
            state.epoch.positions == state.particles.position,
            True,
        )
    )
    successful = (
        epoch_consistent
        & (dt > 0.0)
        & particle_velocity.successful
        & particle_grid_velocity.successful
        & deposited_mass.successful
        & deposited_momentum.successful
        & finite
        & (jnp.abs(mass_defect) <= prepared.plan.gcl_tolerance * mass_scale)
        & jnp.all(
            jnp.abs(momentum_defect) <= prepared.plan.gcl_tolerance * momentum_scale
        )
        & (gcl_defect <= prepared.plan.gcl_tolerance)
    )
    accepted = jax_tree_select(successful, candidate, state)
    return ALEFLIPStepResult(
        candidate,
        accepted,
        relative,
        deposited_mass.content,
        deposited_momentum.content,
        mass_defect,
        momentum_defect,
        gcl_defect,
        finite,
        successful,
        prepared.plan.plan_id,
    )


def jax_tree_select(predicate: Array, candidate, old):
    import jax

    return jax.tree.map(
        lambda new, prior: jnp.where(predicate, new, prior), candidate, old
    )


def transition_ale_flip_epoch(
    state: ALEFLIPState,
    target_prepared: PreparedMeshParticleGridSplat,
    target_population: ParticlePopulationState,
    target_positions: ArrayLike,
    target_coordinates: ArrayLike,
    /,
    *,
    target_transfer: ArrayLike | None = None,
    target_pressure: ArrayLike | None = None,
    target_history: ArrayLike | None = None,
) -> ALEFLIPState:
    """Atomically accept one canonical splat/remesh epoch transition.

    Changed target topology also requires caller-prepared pressure and history
    transfers.  The coupling never invents a remap for those state channels.
    """

    if not isinstance(state, ALEFLIPState):
        raise TypeError("state must be ALEFLIPState.")
    transition = prepare_particle_grid_splat_transition(
        state.epoch,
        target_prepared,
        target_population,
        target_positions,
        target_transfer=target_transfer,
    )
    changed_topology = (
        state.epoch.prepared.target.target_id != target_prepared.target.target_id
    )
    transferred_values = tuple(
        value for value in (target_pressure, target_history) if value is not None
    )
    state_transfer_complete = (
        not changed_topology
        or (target_pressure is not None and target_history is not None)
    ) and all(
        bool(np.all(np.isfinite(np.asarray(value)))) for value in transferred_values
    )
    coordinates = jnp.asarray(target_coordinates)
    coordinate_compatible = (
        coordinates.ndim == 2
        and coordinates.shape[1] == target_prepared.target.mesh.ambient_dimension
        and bool(np.all(np.isfinite(np.asarray(coordinates))))
        and bool(np.all(np.isfinite(np.asarray(target_positions))))
    )
    if bool(transition.successful) and state_transfer_complete and coordinate_compatible:
        source_keys = np.stack(
            (
                np.asarray(state.epoch.prepared.stable_source_ids),
                np.asarray(state.epoch.population.incarnation),
            ),
            axis=-1,
        )
        target_keys = np.stack(
            (
                np.asarray(target_prepared.stable_source_ids),
                np.asarray(target_population.incarnation),
            ),
            axis=-1,
        )
        source_lookup = {tuple(key): index for index, key in enumerate(source_keys)}
        source_indices = np.asarray(
            [source_lookup.get(tuple(key), -1) for key in target_keys],
            dtype=np.int32,
        )
        safe_indices = jnp.asarray(np.maximum(source_indices, 0))
        velocity = state.particles.velocity[safe_indices]
        velocity = jnp.where(
            target_population.active[:, None],
            velocity,
            0.0,
        )
        velocity_finite = bool(np.all(np.isfinite(np.asarray(velocity))))
        if not velocity_finite:
            return state
        particles = FLIPParticleState(
            transition.accepted_epoch.positions,
            velocity,
        )
        pressure = (
            state.pressure if target_pressure is None else jnp.asarray(target_pressure)
        )
        history = state.history if target_history is None else jnp.asarray(target_history)
        return ALEFLIPState(
            transition.accepted_epoch,
            particles,
            coordinates,
            pressure,
            history,
        )
    return state


__all__ = [
    "ALEFLIPPlan",
    "ALEFLIPState",
    "ALEFLIPStepResult",
    "PreparedALEFLIP",
    "advance_ale_flip",
    "prepare_ale_flip",
    "transition_ale_flip_epoch",
]
