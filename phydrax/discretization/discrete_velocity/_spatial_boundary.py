#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._quadrature import CertifiedDiscreteVelocityQuadrature
from ._smooth_compressible import SmoothCompressibleKineticState


class SmoothCompressibleD2VBoundaryStatus(IntEnum):
    """Stable outcome codes for one coupled boundary transaction."""

    SUCCESS = 0
    INVALID_INPUT_STATE = 1
    INVALID_TARGET = 2
    INCOMPATIBLE_RESERVOIR_CORNER = 3
    BACKFLOW = 4
    DEGENERATE_DIFFUSE_FLUX = 5
    NONFINITE_CANDIDATE = 6
    NONPOSITIVE_CANDIDATE = 7


class SmoothCompressibleD2VLinkOwner(IntEnum):
    """Exclusive owner of one destination-cell D2V17 population pair."""

    LOCAL = 0
    PERIODIC = 1
    SPECULAR_ADIABATIC_WALL = 2
    EQUILIBRIUM_RESERVOIR = 3
    OUTWARD_EXTRAPOLATION = 4
    MAXWELL_THERMAL_WALL = 5


class SmoothCompressibleD2VBoundaryFace(IntEnum):
    """Stable row ordering for axis-face reservoir populations."""

    X_LOWER = 0
    X_UPPER = 1
    Y_LOWER = 2
    Y_UPPER = 3


class SmoothCompressibleD2VBoundaryCorner(IntEnum):
    """Stable row ordering for explicit reservoir-corner populations."""

    X_LOWER_Y_LOWER = 0
    X_LOWER_Y_UPPER = 1
    X_UPPER_Y_LOWER = 2
    X_UPPER_Y_UPPER = 3


class CompiledD2V17BoundaryTopology(StrictModule, NonTrainableState):
    """Frozen, exclusive pull source and boundary ownership for every D2V17 link."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    owner: Array
    source_x: Array
    source_y: Array
    source_direction: Array
    physical_axis_mask: Array
    physical_side_sign: Array
    primary_face_index: Array
    secondary_face_index: Array
    corner_index: Array
    spatial_shape: tuple[int, int] = eqx.field(static=True)
    cell_spacing: tuple[float, float] = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    cell_volume: float = eqx.field(static=True)
    pull_offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)
    maximum_reach: tuple[int, int] = eqx.field(static=True)
    periodic_axes: tuple[bool, bool] = eqx.field(static=True)
    physical_owner: SmoothCompressibleD2VLinkOwner = eqx.field(static=True)
    population_shape: tuple[int, int, int] = eqx.field(static=True)
    owner_counts: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        periodic_axes: tuple[bool, bool],
        physical_owner: SmoothCompressibleD2VLinkOwner,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if (
            quadrature.name != "D2V17"
            or quadrature.dimension != 2
            or quadrature.population_count != 17
            or quadrature.transport_kind != "integer_lattice"
        ):
            raise ValueError("Compiled boundary routing requires certified D2V17.")
        if not isinstance(physical_owner, SmoothCompressibleD2VLinkOwner):
            raise TypeError("physical_owner must be SmoothCompressibleD2VLinkOwner.")
        if physical_owner is SmoothCompressibleD2VLinkOwner.LOCAL:
            raise ValueError("physical_owner cannot be LOCAL.")
        shape = tuple(spatial_shape)
        spacing = tuple(float(value) for value in cell_spacing)
        step = float(time_step)
        periodic = tuple(bool(value) for value in periodic_axes)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("spatial_shape must contain two positive extents.")
        if len(spacing) != 2 or any(
            not isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("cell_spacing must contain two finite positive values.")
        if not isfinite(step) or step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if len(periodic) != 2:
            raise ValueError("periodic_axes must contain exactly two booleans.")

        velocities = np.asarray(quadrature.velocities, dtype=np.float64)
        scaled = velocities * step / np.asarray(spacing)[None, :]
        rounded = np.rint(scaled)
        tolerance = (
            64.0 * np.finfo(scaled.dtype).eps * max(float(np.max(np.abs(scaled))), 1.0)
        )
        if float(np.max(np.abs(scaled - rounded))) > tolerance:
            raise ValueError(
                "D2V17 boundary routing requires velocity*time_step/cell_spacing to be integer."
            )
        offsets = tuple(tuple(row) for row in rounded)
        reach = tuple(max(abs(row[axis]) for row in offsets) for axis in range(2))
        if any(shape[axis] <= 2 * reach[axis] for axis in range(2)):
            raise ValueError(
                "D2V17 domains must exceed twice the maximum pull reach on each axis."
            )

        velocity_tuples = tuple(
            tuple(int(round(value)) for value in row) for row in velocities
        )
        velocity_to_direction = {
            velocity: direction for direction, velocity in enumerate(velocity_tuples)
        }
        if len(velocity_to_direction) != quadrature.population_count:
            raise ValueError("D2V17 velocities must be unique integer links.")

        population_shape = shape + (quadrature.population_count,)
        owner = np.empty(population_shape, dtype=np.int8)
        source_x = np.empty(population_shape, dtype=np.int32)
        source_y = np.empty(population_shape, dtype=np.int32)
        source_direction = np.empty(population_shape, dtype=np.int32)
        physical_axis_mask = np.zeros(population_shape + (2,), dtype=np.bool_)
        physical_side_sign = np.zeros(population_shape + (2,), dtype=np.int8)
        primary_face_index = np.full(population_shape, -1, dtype=np.int8)
        secondary_face_index = np.full(population_shape, -1, dtype=np.int8)
        corner_index = np.full(population_shape, -1, dtype=np.int8)

        for cell in np.ndindex(shape):
            for direction, offset in enumerate(offsets):
                index = cell + (direction,)
                raw_source = [cell[axis] - offset[axis] for axis in range(2)]
                routed_source = list(raw_source)
                crossed_any = False
                physical_crossings: list[tuple[int, int]] = []
                reflected_velocity = list(velocity_tuples[direction])
                for axis in range(2):
                    source = raw_source[axis]
                    if source < 0:
                        crossed_any = True
                        side_sign = -1
                    elif source >= shape[axis]:
                        crossed_any = True
                        side_sign = 1
                    else:
                        continue
                    if periodic[axis]:
                        routed_source[axis] = source % shape[axis]
                        continue
                    physical_crossings.append((axis, side_sign))
                    physical_axis_mask[index + (axis,)] = True
                    physical_side_sign[index + (axis,)] = side_sign
                    if physical_owner in (
                        SmoothCompressibleD2VLinkOwner.SPECULAR_ADIABATIC_WALL,
                        SmoothCompressibleD2VLinkOwner.MAXWELL_THERMAL_WALL,
                    ):
                        routed_source[axis] = (
                            -source - 1 if side_sign < 0 else 2 * shape[axis] - source - 1
                        )
                        reflected_velocity[axis] = -reflected_velocity[axis]
                    else:
                        routed_source[axis] = 0 if side_sign < 0 else shape[axis] - 1

                if physical_crossings:
                    owner[index] = int(physical_owner)
                    faces = tuple(
                        2 * axis + (1 if side_sign > 0 else 0)
                        for axis, side_sign in physical_crossings
                    )
                    primary_face_index[index] = faces[0]
                    if len(faces) == 2:
                        secondary_face_index[index] = faces[1]
                        x_sign = physical_side_sign[index + (0,)]
                        y_sign = physical_side_sign[index + (1,)]
                        corner_index[index] = (2 if x_sign > 0 else 0) + (
                            1 if y_sign > 0 else 0
                        )
                elif crossed_any:
                    owner[index] = int(SmoothCompressibleD2VLinkOwner.PERIODIC)
                else:
                    owner[index] = int(SmoothCompressibleD2VLinkOwner.LOCAL)

                source_x[index] = routed_source[0]
                source_y[index] = routed_source[1]
                source_direction[index] = velocity_to_direction[tuple(reflected_velocity)]

        counts = tuple(
            int(np.count_nonzero(owner == int(candidate)))
            for candidate in SmoothCompressibleD2VLinkOwner
        )
        if sum(counts) != int(np.prod(population_shape)):
            raise RuntimeError(
                "Every D2V17 destination link must have exactly one owner."
            )
        topology_id = canonical_fingerprint(
            {
                "kind": "compiled-d2v17-coupled-boundary-topology",
                "quadrature": quadrature.quadrature_id,
                "spatial_shape": list(shape),
                "cell_spacing": list(spacing),
                "time_step": step,
                "pull_offsets": [list(row) for row in offsets],
                "periodic_axes": list(periodic),
                "physical_owner": physical_owner.name,
                "owner": array_tree_fingerprint(owner),
                "source_x": array_tree_fingerprint(source_x),
                "source_y": array_tree_fingerprint(source_y),
                "source_direction": array_tree_fingerprint(source_direction),
                "physical_axis_mask": array_tree_fingerprint(physical_axis_mask),
                "physical_side_sign": array_tree_fingerprint(physical_side_sign),
            }
        )
        self.quadrature = quadrature
        self.owner = jnp.asarray(owner, dtype=jnp.int8)
        self.source_x = jnp.asarray(source_x, dtype=jnp.int32)
        self.source_y = jnp.asarray(source_y, dtype=jnp.int32)
        self.source_direction = jnp.asarray(source_direction, dtype=jnp.int32)
        self.physical_axis_mask = jnp.asarray(physical_axis_mask, dtype=jnp.bool_)
        self.physical_side_sign = jnp.asarray(physical_side_sign, dtype=jnp.int8)
        self.primary_face_index = jnp.asarray(primary_face_index, dtype=jnp.int8)
        self.secondary_face_index = jnp.asarray(secondary_face_index, dtype=jnp.int8)
        self.corner_index = jnp.asarray(corner_index, dtype=jnp.int8)
        self.spatial_shape = shape
        self.cell_spacing = spacing
        self.time_step = step
        self.cell_volume = prod(spacing)
        self.pull_offsets = offsets
        self.maximum_reach = reach
        self.periodic_axes = periodic
        self.physical_owner = physical_owner
        self.population_shape = population_shape
        self.owner_counts = counts
        self.topology_id = topology_id

    @property
    def boundary_mask(self) -> Array:
        return self.owner >= int(SmoothCompressibleD2VLinkOwner.SPECULAR_ADIABATIC_WALL)

    def validate_state(self, state: SmoothCompressibleKineticState, /) -> None:
        if not isinstance(state, SmoothCompressibleKineticState):
            raise TypeError("state must be SmoothCompressibleKineticState.")
        if (
            state.particle_populations.shape != self.population_shape
            or state.total_energy_populations.shape != self.population_shape
        ):
            raise ValueError(
                "Boundary state populations must have shape spatial_shape + (17,)."
            )
        if state.particle_populations.dtype != state.total_energy_populations.dtype:
            raise TypeError("Coupled boundary populations must use one dtype.")

    def gather(self, populations: ArrayLike, /) -> Array:
        values = jnp.asarray(populations)
        if values.shape != self.population_shape:
            raise ValueError("Populations do not match the compiled D2V17 topology.")
        return values[self.source_x, self.source_y, self.source_direction]


class SmoothCompressibleD2VReservoirParameters(StrictModule):
    """Prepared incoming f/g distributions in face and optional corner order."""

    face_particle_populations: Array
    face_total_energy_populations: Array
    corner_particle_populations: Array
    corner_total_energy_populations: Array
    corner_data_present: Array

    def __init__(
        self,
        face_particle_populations: ArrayLike,
        face_total_energy_populations: ArrayLike,
        /,
        *,
        corner_particle_populations: ArrayLike | None = None,
        corner_total_energy_populations: ArrayLike | None = None,
        corner_data_present: ArrayLike | None = None,
    ):
        particles = jnp.asarray(face_particle_populations)
        energy = jnp.asarray(face_total_energy_populations)
        if particles.shape == (17,):
            particles = jnp.broadcast_to(particles, (4, 17))
        if energy.shape == (17,):
            energy = jnp.broadcast_to(energy, (4, 17))
        if particles.shape != (4, 17) or energy.shape != (4, 17):
            raise ValueError(
                "Reservoir face populations must have shape (17,) or (4, 17)."
            )
        if not jnp.issubdtype(particles.dtype, jnp.inexact) or not jnp.issubdtype(
            energy.dtype, jnp.inexact
        ):
            raise TypeError("Reservoir populations must use inexact dtypes.")
        if particles.dtype != energy.dtype:
            raise TypeError("Coupled reservoir face populations must use one dtype.")
        if (corner_particle_populations is None) != (
            corner_total_energy_populations is None
        ):
            raise ValueError("Explicit reservoir corners require both f and g values.")
        if corner_particle_populations is None:
            if corner_data_present is not None:
                raise ValueError(
                    "corner_data_present requires explicit corner f/g values."
                )
            corner_particles = jnp.zeros((4, 17), dtype=particles.dtype)
            corner_energy = jnp.zeros((4, 17), dtype=energy.dtype)
            present = jnp.zeros((4,), dtype=jnp.bool_)
        else:
            corner_particles = jnp.asarray(corner_particle_populations)
            corner_energy = jnp.asarray(corner_total_energy_populations)
            if corner_particles.shape == (17,):
                corner_particles = jnp.broadcast_to(corner_particles, (4, 17))
            if corner_energy.shape == (17,):
                corner_energy = jnp.broadcast_to(corner_energy, (4, 17))
            if corner_particles.shape != (4, 17) or corner_energy.shape != (4, 17):
                raise ValueError(
                    "Reservoir corner populations must have shape (17,) or (4, 17)."
                )
            if (
                corner_particles.dtype != particles.dtype
                or corner_energy.dtype != energy.dtype
            ):
                raise TypeError(
                    "Reservoir face and corner populations must use one dtype."
                )
            present = jnp.asarray(
                jnp.ones((4,), dtype=jnp.bool_)
                if corner_data_present is None
                else corner_data_present,
                dtype=jnp.bool_,
            )
            if present.shape != (4,):
                raise ValueError("corner_data_present must have shape (4,).")
        self.face_particle_populations = particles
        self.face_total_energy_populations = energy
        self.corner_particle_populations = corner_particles
        self.corner_total_energy_populations = corner_energy
        self.corner_data_present = present


class SmoothCompressibleD2VBoundaryHistory(StrictModule):
    """Optional per-link audit detail for a retained boundary transaction."""

    source_particle_populations: Array
    source_total_energy_populations: Array
    boundary_written: Array
    backflow: Array


class SmoothCompressibleD2VBoundaryResult(StrictModule):
    """Candidate and atomically accepted coupled D2V17 boundary transaction."""

    candidate_state: SmoothCompressibleKineticState
    accepted_state: SmoothCompressibleKineticState
    mass_momentum_energy_exchange: Array
    wall_momentum_impulse: Array
    heat_exchange: Array
    wall_work: Array
    successful: Array
    status: Array
    rollback_applied: Array
    history: SmoothCompressibleD2VBoundaryHistory | None


class AbstractSmoothCompressibleD2VBoundaryPlan(StrictModule, NonTrainableState):
    """Typed contract for one compiled, coupled D2V17 boundary router."""

    topology: eqx.AbstractVar[CompiledD2V17BoundaryTopology]
    retain_history: eqx.AbstractVar[bool]
    plan_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def route(
        self,
        post_collision_state: SmoothCompressibleKineticState,
        parameters: Any = None,
        /,
    ) -> SmoothCompressibleD2VBoundaryResult:
        raise NotImplementedError


def _validate_retain_history(value: bool, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError("retain_history must be bool.")
    return value


def _select_state(
    predicate: Array,
    candidate: SmoothCompressibleKineticState,
    original: SmoothCompressibleKineticState,
    /,
) -> SmoothCompressibleKineticState:
    return SmoothCompressibleKineticState(
        jnp.where(
            predicate,
            candidate.particle_populations,
            original.particle_populations,
        ),
        jnp.where(
            predicate,
            candidate.total_energy_populations,
            original.total_energy_populations,
        ),
    )


def _state_is_valid(state: SmoothCompressibleKineticState, /) -> Array:
    return (
        jnp.all(jnp.isfinite(state.particle_populations))
        & jnp.all(jnp.isfinite(state.total_energy_populations))
        & jnp.all(state.particle_populations >= 0.0)
        & jnp.all(state.total_energy_populations >= 0.0)
    )


def _content(
    topology: CompiledD2V17BoundaryTopology,
    state: SmoothCompressibleKineticState,
    /,
) -> Array:
    dtype = state.particle_populations.dtype
    volume = jnp.asarray(topology.cell_volume, dtype=dtype)
    mass = jnp.sum(state.particle_populations) * volume
    momentum = (
        jnp.sum(
            ein.contract(
                "xyq,qd->xyd",
                state.particle_populations,
                topology.quadrature.velocities.astype(dtype),
            ),
            axis=(0, 1),
        )
        * volume
    )
    energy = jnp.sum(state.total_energy_populations) * volume
    return jnp.concatenate((mass[None], momentum, energy[None]), axis=0)


def _result(
    topology: CompiledD2V17BoundaryTopology,
    original: SmoothCompressibleKineticState,
    candidate: SmoothCompressibleKineticState,
    successful: Array,
    status: Array,
    wall: bool,
    retain_history: bool,
    gathered_particles: Array,
    gathered_energy: Array,
    backflow: Array,
    /,
    *,
    mechanical_wall_work: Array | None = None,
) -> SmoothCompressibleD2VBoundaryResult:
    accepted = _select_state(successful, candidate, original)
    exchange = _content(topology, accepted) - _content(topology, original)
    zero = jnp.asarray(0.0, dtype=exchange.dtype)
    wall_impulse = -exchange[1:3] if wall else jnp.zeros((2,), dtype=exchange.dtype)
    if mechanical_wall_work is None:
        wall_work = zero
        heat_exchange = zero
    else:
        wall_work = jnp.where(successful, mechanical_wall_work, zero)
        heat_exchange = exchange[3] - wall_work
    history = (
        SmoothCompressibleD2VBoundaryHistory(
            source_particle_populations=gathered_particles,
            source_total_energy_populations=gathered_energy,
            boundary_written=topology.boundary_mask,
            backflow=backflow,
        )
        if retain_history
        else None
    )
    return SmoothCompressibleD2VBoundaryResult(
        candidate_state=candidate,
        accepted_state=accepted,
        mass_momentum_energy_exchange=exchange,
        wall_momentum_impulse=wall_impulse,
        heat_exchange=heat_exchange,
        wall_work=wall_work,
        successful=successful,
        status=status.astype(jnp.int32),
        rollback_applied=~successful,
        history=history,
    )


def _route_gathered(
    topology: CompiledD2V17BoundaryTopology,
    state: SmoothCompressibleKineticState,
    retain_history: bool,
    /,
    *,
    wall: bool,
    backflow: Array | None = None,
) -> SmoothCompressibleD2VBoundaryResult:
    topology.validate_state(state)
    gathered_particles = topology.gather(state.particle_populations)
    gathered_energy = topology.gather(state.total_energy_populations)
    candidate = SmoothCompressibleKineticState(gathered_particles, gathered_energy)
    input_valid = _state_is_valid(state)
    candidate_valid = _state_is_valid(candidate)
    backflow_mask = (
        jnp.zeros(topology.population_shape, dtype=jnp.bool_)
        if backflow is None
        else backflow
    )
    backflow_free = ~jnp.any(backflow_mask)
    successful = input_valid & candidate_valid & backflow_free
    status = jnp.asarray(
        int(SmoothCompressibleD2VBoundaryStatus.SUCCESS), dtype=jnp.int32
    )
    status = jnp.where(
        ~input_valid,
        int(SmoothCompressibleD2VBoundaryStatus.INVALID_INPUT_STATE),
        status,
    )
    status = jnp.where(
        input_valid & ~candidate_valid,
        int(SmoothCompressibleD2VBoundaryStatus.INVALID_TARGET),
        status,
    )
    status = jnp.where(
        input_valid & candidate_valid & ~backflow_free,
        int(SmoothCompressibleD2VBoundaryStatus.BACKFLOW),
        status,
    )
    return _result(
        topology,
        state,
        candidate,
        successful,
        status,
        wall,
        retain_history,
        gathered_particles,
        gathered_energy,
        backflow_mask,
    )


class PeriodicD2VBoundaryPlan(AbstractSmoothCompressibleD2VBoundaryPlan):
    """Exact periodic D2V17 pull routing for coupled f/g fields."""

    topology: CompiledD2V17BoundaryTopology
    retain_history: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        retain_history: bool = False,
    ):
        history = _validate_retain_history(retain_history)
        topology = CompiledD2V17BoundaryTopology(
            quadrature,
            spatial_shape,
            cell_spacing,
            time_step,
            periodic_axes=(True, True),
            physical_owner=SmoothCompressibleD2VLinkOwner.PERIODIC,
        )
        self.topology = topology
        self.retain_history = history
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-d2v17-coupled-boundary",
                "topology": topology.topology_id,
                "retain_history": history,
            }
        )

    def route(
        self,
        post_collision_state: SmoothCompressibleKineticState,
        parameters: Any = None,
        /,
    ) -> SmoothCompressibleD2VBoundaryResult:
        if parameters is not None:
            raise TypeError("Periodic D2V17 routing does not accept parameters.")
        return _route_gathered(
            self.topology,
            post_collision_state,
            self.retain_history,
            wall=False,
        )


class SpecularAdiabaticD2VBoundaryPlan(AbstractSmoothCompressibleD2VBoundaryPlan):
    """Stationary specular, adiabatic walls on every nonperiodic axis face."""

    topology: CompiledD2V17BoundaryTopology
    retain_history: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        periodic_axes: tuple[bool, bool] = (False, False),
        retain_history: bool = False,
    ):
        history = _validate_retain_history(retain_history)
        topology = CompiledD2V17BoundaryTopology(
            quadrature,
            spatial_shape,
            cell_spacing,
            time_step,
            periodic_axes=periodic_axes,
            physical_owner=SmoothCompressibleD2VLinkOwner.SPECULAR_ADIABATIC_WALL,
        )
        if all(topology.periodic_axes):
            raise ValueError("Use PeriodicD2VBoundaryPlan when both axes are periodic.")
        self.topology = topology
        self.retain_history = history
        self.plan_id = canonical_fingerprint(
            {
                "kind": "specular-adiabatic-d2v17-coupled-boundary",
                "topology": topology.topology_id,
                "retain_history": history,
            }
        )

    def route(
        self,
        post_collision_state: SmoothCompressibleKineticState,
        parameters: Any = None,
        /,
    ) -> SmoothCompressibleD2VBoundaryResult:
        if parameters is not None:
            raise TypeError("Specular D2V17 routing does not accept parameters.")
        return _route_gathered(
            self.topology,
            post_collision_state,
            self.retain_history,
            wall=True,
        )


class EquilibriumReservoirD2VBoundaryPlan(AbstractSmoothCompressibleD2VBoundaryPlan):
    """Incoming-link reservoir using prepared f/g values, without model evaluation."""

    topology: CompiledD2V17BoundaryTopology
    default_parameters: SmoothCompressibleD2VReservoirParameters | None
    retain_history: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        periodic_axes: tuple[bool, bool] = (False, False),
        incoming_particle_populations: ArrayLike | None = None,
        incoming_total_energy_populations: ArrayLike | None = None,
        corner_particle_populations: ArrayLike | None = None,
        corner_total_energy_populations: ArrayLike | None = None,
        corner_data_present: ArrayLike | None = None,
        retain_history: bool = False,
    ):
        history = _validate_retain_history(retain_history)
        topology = CompiledD2V17BoundaryTopology(
            quadrature,
            spatial_shape,
            cell_spacing,
            time_step,
            periodic_axes=periodic_axes,
            physical_owner=SmoothCompressibleD2VLinkOwner.EQUILIBRIUM_RESERVOIR,
        )
        if all(topology.periodic_axes):
            raise ValueError("A reservoir plan requires at least one physical axis.")
        if (incoming_particle_populations is None) != (
            incoming_total_energy_populations is None
        ):
            raise ValueError("A default reservoir requires both incoming f and g values.")
        if incoming_particle_populations is None:
            if (
                corner_particle_populations is not None
                or corner_total_energy_populations is not None
                or corner_data_present is not None
            ):
                raise ValueError("Default corner values require default face f/g values.")
            default_parameters = None
            parameters_id = None
        else:
            default_parameters = SmoothCompressibleD2VReservoirParameters(
                incoming_particle_populations,
                incoming_total_energy_populations,
                corner_particle_populations=corner_particle_populations,
                corner_total_energy_populations=corner_total_energy_populations,
                corner_data_present=corner_data_present,
            )
            parameters_id = canonical_fingerprint(
                {
                    "face_particles": array_tree_fingerprint(
                        default_parameters.face_particle_populations
                    ),
                    "face_energy": array_tree_fingerprint(
                        default_parameters.face_total_energy_populations
                    ),
                    "corner_particles": array_tree_fingerprint(
                        default_parameters.corner_particle_populations
                    ),
                    "corner_energy": array_tree_fingerprint(
                        default_parameters.corner_total_energy_populations
                    ),
                    "corner_present": array_tree_fingerprint(
                        default_parameters.corner_data_present
                    ),
                }
            )
        self.topology = topology
        self.default_parameters = default_parameters
        self.retain_history = history
        self.plan_id = canonical_fingerprint(
            {
                "kind": "equilibrium-reservoir-d2v17-coupled-boundary",
                "topology": topology.topology_id,
                "default_parameters": parameters_id,
                "retain_history": history,
            }
        )

    def route(
        self,
        post_collision_state: SmoothCompressibleKineticState,
        parameters: Any = None,
        /,
    ) -> SmoothCompressibleD2VBoundaryResult:
        self.topology.validate_state(post_collision_state)
        if parameters is not None and not isinstance(
            parameters, SmoothCompressibleD2VReservoirParameters
        ):
            raise TypeError("Reservoir parameters must be prepared D2V17 f/g values.")
        selected_parameters = (
            self.default_parameters if parameters is None else parameters
        )
        supplied = selected_parameters is not None
        if selected_parameters is None:
            selected_parameters = SmoothCompressibleD2VReservoirParameters(
                jnp.zeros((4, 17), dtype=post_collision_state.particle_populations.dtype),
                jnp.zeros(
                    (4, 17), dtype=post_collision_state.total_energy_populations.dtype
                ),
            )

        gathered_particles = self.topology.gather(
            post_collision_state.particle_populations
        )
        gathered_energy = self.topology.gather(
            post_collision_state.total_energy_populations
        )
        owner_mask = self.topology.owner == int(
            SmoothCompressibleD2VLinkOwner.EQUILIBRIUM_RESERVOIR
        )
        primary = jnp.where(
            self.topology.primary_face_index >= 0,
            self.topology.primary_face_index,
            0,
        )
        secondary = jnp.where(
            self.topology.secondary_face_index >= 0,
            self.topology.secondary_face_index,
            0,
        )
        corner = jnp.where(self.topology.corner_index >= 0, self.topology.corner_index, 0)
        direction = jnp.broadcast_to(
            jnp.arange(17, dtype=jnp.int32)[None, None, :],
            self.topology.population_shape,
        )
        face_particles = selected_parameters.face_particle_populations.astype(
            post_collision_state.particle_populations.dtype
        )
        face_energy = selected_parameters.face_total_energy_populations.astype(
            post_collision_state.total_energy_populations.dtype
        )
        corner_particles = selected_parameters.corner_particle_populations.astype(
            post_collision_state.particle_populations.dtype
        )
        corner_energy = selected_parameters.corner_total_energy_populations.astype(
            post_collision_state.total_energy_populations.dtype
        )
        primary_particles = face_particles[primary, direction]
        primary_energy = face_energy[primary, direction]
        secondary_particles = face_particles[secondary, direction]
        secondary_energy = face_energy[secondary, direction]
        explicit_particles = corner_particles[corner, direction]
        explicit_energy = corner_energy[corner, direction]
        corner_present = selected_parameters.corner_data_present[corner]
        corner_mask = owner_mask & (self.topology.secondary_face_index >= 0)
        use_corner = corner_mask & corner_present
        incoming_particles = jnp.where(use_corner, explicit_particles, primary_particles)
        incoming_energy = jnp.where(use_corner, explicit_energy, primary_energy)
        incompatible_corner = (
            corner_mask
            & ~corner_present
            & (
                (primary_particles != secondary_particles)
                | (primary_energy != secondary_energy)
            )
        )
        candidate = SmoothCompressibleKineticState(
            jnp.where(owner_mask, incoming_particles, gathered_particles),
            jnp.where(owner_mask, incoming_energy, gathered_energy),
        )

        input_valid = _state_is_valid(post_collision_state)
        face_valid = (
            jnp.all(jnp.isfinite(face_particles))
            & jnp.all(jnp.isfinite(face_energy))
            & jnp.all(face_particles >= 0.0)
            & jnp.all(face_energy >= 0.0)
        )
        present = selected_parameters.corner_data_present[:, None]
        corner_valid = (
            jnp.all(~present | jnp.isfinite(corner_particles))
            & jnp.all(~present | jnp.isfinite(corner_energy))
            & jnp.all(~present | (corner_particles >= 0.0))
            & jnp.all(~present | (corner_energy >= 0.0))
        )
        target_valid = jnp.asarray(supplied) & face_valid & corner_valid
        corners_compatible = ~jnp.any(incompatible_corner)
        candidate_valid = _state_is_valid(candidate)
        successful = input_valid & target_valid & corners_compatible & candidate_valid
        status = jnp.asarray(
            int(SmoothCompressibleD2VBoundaryStatus.SUCCESS), dtype=jnp.int32
        )
        status = jnp.where(
            ~input_valid,
            int(SmoothCompressibleD2VBoundaryStatus.INVALID_INPUT_STATE),
            status,
        )
        status = jnp.where(
            input_valid & (~target_valid | ~candidate_valid),
            int(SmoothCompressibleD2VBoundaryStatus.INVALID_TARGET),
            status,
        )
        status = jnp.where(
            input_valid & target_valid & candidate_valid & ~corners_compatible,
            int(SmoothCompressibleD2VBoundaryStatus.INCOMPATIBLE_RESERVOIR_CORNER),
            status,
        )
        return _result(
            self.topology,
            post_collision_state,
            candidate,
            successful,
            status,
            False,
            self.retain_history,
            gathered_particles,
            gathered_energy,
            jnp.zeros(self.topology.population_shape, dtype=jnp.bool_),
        )


class OutwardExtrapolationD2VBoundaryPlan(AbstractSmoothCompressibleD2VBoundaryPlan):
    """Zero-gradient outward routing with strict local backflow refusal."""

    topology: CompiledD2V17BoundaryTopology
    retain_history: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        periodic_axes: tuple[bool, bool] = (False, False),
        retain_history: bool = False,
    ):
        history = _validate_retain_history(retain_history)
        topology = CompiledD2V17BoundaryTopology(
            quadrature,
            spatial_shape,
            cell_spacing,
            time_step,
            periodic_axes=periodic_axes,
            physical_owner=SmoothCompressibleD2VLinkOwner.OUTWARD_EXTRAPOLATION,
        )
        if all(topology.periodic_axes):
            raise ValueError("An outward-flow plan requires at least one physical axis.")
        self.topology = topology
        self.retain_history = history
        self.plan_id = canonical_fingerprint(
            {
                "kind": "outward-extrapolation-d2v17-coupled-boundary",
                "topology": topology.topology_id,
                "retain_history": history,
            }
        )

    def route(
        self,
        post_collision_state: SmoothCompressibleKineticState,
        parameters: Any = None,
        /,
    ) -> SmoothCompressibleD2VBoundaryResult:
        if parameters is not None:
            raise TypeError(
                "Outward-extrapolation D2V17 routing does not accept parameters."
            )
        self.topology.validate_state(post_collision_state)
        momentum = ein.contract(
            "xyq,qd->xyd",
            post_collision_state.particle_populations,
            self.topology.quadrature.velocities.astype(
                post_collision_state.particle_populations.dtype
            ),
        )
        momentum_x = momentum[..., 0, None]
        momentum_y = momentum[..., 1, None]
        signed_normal_momentum = jnp.where(
            self.topology.physical_axis_mask[..., 0],
            self.topology.physical_side_sign[..., 0] * momentum_x,
            0.0,
        ) + jnp.where(
            self.topology.physical_axis_mask[..., 1],
            self.topology.physical_side_sign[..., 1] * momentum_y,
            0.0,
        )
        each_face_outward = jnp.all(
            (~self.topology.physical_axis_mask)
            | (self.topology.physical_side_sign * momentum[..., None, :] >= 0.0),
            axis=-1,
        )
        owner_mask = self.topology.owner == int(
            SmoothCompressibleD2VLinkOwner.OUTWARD_EXTRAPOLATION
        )
        backflow = owner_mask & (
            (~each_face_outward) | ~jnp.isfinite(signed_normal_momentum)
        )
        return _route_gathered(
            self.topology,
            post_collision_state,
            self.retain_history,
            wall=False,
            backflow=backflow,
        )


def _positive_face_populations(values: ArrayLike, name: str, /) -> np.ndarray:
    populations = np.asarray(values)
    if populations.shape == (17,):
        populations = np.broadcast_to(populations, (4, 17)).copy()
    if populations.shape != (4, 17):
        raise ValueError(f"{name} must have shape (17,) or (4, 17).")
    if not np.issubdtype(populations.dtype, np.inexact):
        raise TypeError(f"{name} must use an inexact dtype.")
    if np.any(~np.isfinite(populations)) or np.any(populations <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return populations


class MaxwellThermalD2VBoundaryPlan(AbstractSmoothCompressibleD2VBoundaryPlan):
    """Diffuse/specular thermal reflection from explicitly supplied populations.

    The supplied f distribution represents unit mass density. The g distribution
    supplies its associated energy populations; no exact wall-temperature claim is
    inferred. Static Cartesian faces may move tangentially, but not normally.
    """

    topology: CompiledD2V17BoundaryTopology
    diffuse_particle_populations: Array
    diffuse_total_energy_populations: Array
    wall_velocity: Array
    accommodation: tuple[float, float, float, float] = eqx.field(static=True)
    retain_history: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        diffuse_particle_populations: ArrayLike,
        diffuse_total_energy_populations: ArrayLike,
        accommodation: ArrayLike,
        wall_velocity: ArrayLike,
        periodic_axes: tuple[bool, bool] = (False, False),
        retain_history: bool = False,
    ):
        history = _validate_retain_history(retain_history)
        topology = CompiledD2V17BoundaryTopology(
            quadrature,
            spatial_shape,
            cell_spacing,
            time_step,
            periodic_axes=periodic_axes,
            physical_owner=SmoothCompressibleD2VLinkOwner.MAXWELL_THERMAL_WALL,
        )
        if all(topology.periodic_axes):
            raise ValueError("A Maxwell thermal plan requires a physical axis.")
        diffuse_particles = _positive_face_populations(
            diffuse_particle_populations,
            "diffuse_particle_populations",
        )
        diffuse_energy = _positive_face_populations(
            diffuse_total_energy_populations,
            "diffuse_total_energy_populations",
        )
        if diffuse_particles.dtype != diffuse_energy.dtype:
            raise TypeError("Coupled Maxwell diffuse f/g values must use one dtype.")
        density = np.sum(diffuse_particles, axis=-1)
        density_tolerance = (
            256.0
            * np.finfo(diffuse_particles.dtype).eps
            * np.maximum(np.abs(density), 1.0)
        )
        if np.any(np.abs(density - 1.0) > density_tolerance):
            raise ValueError(
                "Each Maxwell diffuse particle distribution must have unit density."
            )

        accommodation_values = np.asarray(accommodation, dtype=np.float64)
        if accommodation_values.ndim == 0:
            accommodation_values = np.full((4,), float(accommodation_values))
        if accommodation_values.shape != (4,) or np.any(
            ~np.isfinite(accommodation_values)
        ):
            raise ValueError(
                "accommodation must be one finite value or four face values."
            )
        if np.any((accommodation_values < 0.0) | (accommodation_values > 1.0)):
            raise ValueError("Maxwell accommodation must lie in [0, 1].")

        velocity = np.asarray(wall_velocity, dtype=diffuse_particles.dtype)
        if velocity.shape == (2,):
            velocity = np.broadcast_to(velocity, (4, 2)).copy()
        if velocity.shape != (4, 2) or np.any(~np.isfinite(velocity)):
            raise ValueError("wall_velocity must be one finite vector or four vectors.")
        for face in range(4):
            axis = face // 2
            if not topology.periodic_axes[axis] and velocity[face, axis] != 0.0:
                raise ValueError(
                    "Static Cartesian Maxwell faces require zero normal wall velocity."
                )

        self.topology = topology
        self.diffuse_particle_populations = jnp.asarray(diffuse_particles)
        self.diffuse_total_energy_populations = jnp.asarray(diffuse_energy)
        self.wall_velocity = jnp.asarray(velocity)
        self.accommodation = tuple(float(value) for value in accommodation_values)
        self.retain_history = history
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-thermal-d2v17-coupled-boundary",
                "topology": topology.topology_id,
                "diffuse_particles": array_tree_fingerprint(diffuse_particles),
                "diffuse_energy": array_tree_fingerprint(diffuse_energy),
                "accommodation": list(self.accommodation),
                "wall_velocity": array_tree_fingerprint(velocity),
                "corner_ownership": "first-crossed-axis-face",
                "retain_history": history,
            }
        )

    def route(
        self,
        post_collision_state: SmoothCompressibleKineticState,
        parameters: Any = None,
        /,
    ) -> SmoothCompressibleD2VBoundaryResult:
        if parameters is not None:
            raise TypeError("Maxwell thermal D2V17 routing does not accept parameters.")
        self.topology.validate_state(post_collision_state)
        gathered_particles = self.topology.gather(
            post_collision_state.particle_populations
        )
        gathered_energy = self.topology.gather(
            post_collision_state.total_energy_populations
        )
        owner_mask = self.topology.owner == int(
            SmoothCompressibleD2VLinkOwner.MAXWELL_THERMAL_WALL
        )
        face_index = jnp.where(
            self.topology.primary_face_index >= 0,
            self.topology.primary_face_index,
            0,
        )
        direction = jnp.broadcast_to(
            jnp.arange(17, dtype=jnp.int32)[None, None, :],
            self.topology.population_shape,
        )
        diffuse_particles = self.diffuse_particle_populations.astype(
            post_collision_state.particle_populations.dtype
        )
        diffuse_energy = self.diffuse_total_energy_populations.astype(
            post_collision_state.total_energy_populations.dtype
        )
        unit_incoming_particles = diffuse_particles[face_index, direction]
        unit_incoming_energy = diffuse_energy[face_index, direction]

        incoming_half_range_fluxes = []
        outgoing_half_range_fluxes = []
        active_faces = []
        for face in range(4):
            axis = face // 2
            active = not self.topology.periodic_axes[axis]
            active_faces.append(active)
            face_mask = owner_mask & (face_index == face)
            incoming_half_range_fluxes.append(
                jnp.sum(jnp.where(face_mask, unit_incoming_particles, 0.0))
            )
            outgoing_half_range_fluxes.append(
                jnp.sum(jnp.where(face_mask, gathered_particles, 0.0))
            )
        incoming_half_range_flux = jnp.stack(incoming_half_range_fluxes)
        outgoing_half_range_flux = jnp.stack(outgoing_half_range_fluxes)
        active_face = jnp.asarray(active_faces, dtype=jnp.bool_)
        safe_half_range_flux = jnp.where(
            incoming_half_range_flux > 0.0,
            incoming_half_range_flux,
            1.0,
        )
        diffuse_density = outgoing_half_range_flux / safe_half_range_flux
        density_at_link = diffuse_density[face_index]
        diffuse_candidate_particles = density_at_link * unit_incoming_particles
        diffuse_candidate_energy = density_at_link * unit_incoming_energy
        accommodation = jnp.asarray(
            self.accommodation,
            dtype=post_collision_state.particle_populations.dtype,
        )[face_index]
        candidate_particles = (
            1.0 - accommodation
        ) * gathered_particles + accommodation * diffuse_candidate_particles
        candidate_energy = (
            1.0 - accommodation
        ) * gathered_energy + accommodation * diffuse_candidate_energy
        candidate = SmoothCompressibleKineticState(
            jnp.where(owner_mask, candidate_particles, gathered_particles),
            jnp.where(owner_mask, candidate_energy, gathered_energy),
        )

        input_valid = _state_is_valid(post_collision_state)
        flux_valid = jnp.all(
            (~active_face)
            | (
                jnp.isfinite(incoming_half_range_flux)
                & jnp.isfinite(outgoing_half_range_flux)
                & (incoming_half_range_flux > 0.0)
                & (outgoing_half_range_flux >= 0.0)
            )
        )
        candidate_finite = jnp.all(
            jnp.isfinite(candidate.particle_populations)
        ) & jnp.all(jnp.isfinite(candidate.total_energy_populations))
        candidate_positive = jnp.all(candidate.particle_populations > 0.0) & jnp.all(
            candidate.total_energy_populations > 0.0
        )
        successful = input_valid & flux_valid & candidate_finite & candidate_positive
        status = jnp.asarray(
            int(SmoothCompressibleD2VBoundaryStatus.SUCCESS),
            dtype=jnp.int32,
        )
        status = jnp.where(
            ~input_valid,
            int(SmoothCompressibleD2VBoundaryStatus.INVALID_INPUT_STATE),
            status,
        )
        status = jnp.where(
            input_valid & ~flux_valid,
            int(SmoothCompressibleD2VBoundaryStatus.DEGENERATE_DIFFUSE_FLUX),
            status,
        )
        status = jnp.where(
            input_valid & flux_valid & ~candidate_finite,
            int(SmoothCompressibleD2VBoundaryStatus.NONFINITE_CANDIDATE),
            status,
        )
        status = jnp.where(
            input_valid & flux_valid & candidate_finite & ~candidate_positive,
            int(SmoothCompressibleD2VBoundaryStatus.NONPOSITIVE_CANDIDATE),
            status,
        )

        incoming_velocity = self.topology.quadrature.velocities.astype(
            candidate.particle_populations.dtype
        ).reshape((1, 1, 17, 2))
        outgoing_velocity = self.topology.quadrature.velocities.astype(
            candidate.particle_populations.dtype
        )[self.topology.source_direction]
        link_fluid_impulse = (
            candidate.particle_populations[..., None] * incoming_velocity
            - gathered_particles[..., None] * outgoing_velocity
        )
        face_fluid_impulses = []
        for face in range(4):
            face_mask = owner_mask & (face_index == face)
            face_fluid_impulses.append(
                jnp.sum(
                    jnp.where(face_mask[..., None], link_fluid_impulse, 0.0),
                    axis=(0, 1, 2),
                )
                * jnp.asarray(
                    self.topology.cell_volume,
                    dtype=candidate.particle_populations.dtype,
                )
            )
        face_fluid_impulse = jnp.stack(face_fluid_impulses)
        mechanical_wall_work = jnp.sum(
            face_fluid_impulse
            * self.wall_velocity.astype(candidate.particle_populations.dtype)
        )
        return _result(
            self.topology,
            post_collision_state,
            candidate,
            successful,
            status,
            True,
            self.retain_history,
            gathered_particles,
            gathered_energy,
            jnp.zeros(self.topology.population_shape, dtype=jnp.bool_),
            mechanical_wall_work=mechanical_wall_work,
        )


__all__ = [
    "AbstractSmoothCompressibleD2VBoundaryPlan",
    "CompiledD2V17BoundaryTopology",
    "EquilibriumReservoirD2VBoundaryPlan",
    "MaxwellThermalD2VBoundaryPlan",
    "OutwardExtrapolationD2VBoundaryPlan",
    "PeriodicD2VBoundaryPlan",
    "SmoothCompressibleD2VBoundaryCorner",
    "SmoothCompressibleD2VBoundaryFace",
    "SmoothCompressibleD2VBoundaryHistory",
    "SmoothCompressibleD2VBoundaryResult",
    "SmoothCompressibleD2VBoundaryStatus",
    "SmoothCompressibleD2VLinkOwner",
    "SmoothCompressibleD2VReservoirParameters",
    "SpecularAdiabaticD2VBoundaryPlan",
]
