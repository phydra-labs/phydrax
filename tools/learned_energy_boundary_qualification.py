#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    PreparedSmoothCompressibleD2V17SpatialDynamics,
    SmoothCompressibleD2V17SpatialPlan,
)
from phydrax.discretization.discrete_velocity._spatial_boundary import (
    EquilibriumReservoirD2VBoundaryPlan,
    MaxwellThermalD2VBoundaryPlan,
    OutwardExtrapolationD2VBoundaryPlan,
    PeriodicD2VBoundaryPlan,
    SmoothCompressibleD2VBoundaryCorner,
    SmoothCompressibleD2VBoundaryFace,
    SmoothCompressibleD2VBoundaryStatus,
    SmoothCompressibleD2VLinkOwner,
    SmoothCompressibleD2VReservoirParameters,
    SpecularAdiabaticD2VBoundaryPlan,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport


SPATIAL_SHAPE = (16, 16)
CELL_SPACING = (1.0 / 16.0, 1.0 / 16.0)
TIME_STEP = 1.0 / 16.0
ABSOLUTE_TOLERANCE = 1.0e-11
_OUTPUT = Path("benchmarks/learned_energy_boundaries.json")
_PROTECTED_ARTIFACT_NAMES = frozenset(
    {
        "learned_energy_equilibrium.json",
        "learned_energy_spatial.json",
        "learned_energy_stage_two.json",
    }
)


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value)
    if not identifier or identifier != identifier.strip():
        raise ValueError(
            f"{role} must be a non-empty identifier without outer whitespace."
        )
    return identifier


def _array_equal(left: Any, right: Any, /) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _allclose(
    left: Any,
    right: Any,
    /,
    *,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
) -> bool:
    return bool(
        np.allclose(
            np.asarray(left),
            np.asarray(right),
            rtol=0.0,
            atol=absolute_tolerance,
        )
    )


def _maximum_absolute_difference(left: Any, right: Any, /) -> float:
    difference = np.abs(np.asarray(left) - np.asarray(right))
    return float(np.max(difference))


def _direction(quadrature: Any, velocity: tuple[int, int], /) -> int:
    velocities = np.asarray(quadrature.velocities)
    matches = np.all(velocities == np.asarray(velocity), axis=1)
    indices = np.flatnonzero(matches)
    if indices.shape != (1,):
        raise RuntimeError(f"D2V17 must contain velocity {velocity!r} exactly once.")
    return int(indices[0])


def _unique_state(
    shape: tuple[int, int] = SPATIAL_SHAPE,
) -> SmoothCompressibleKineticState:
    count = int(np.prod(shape) * 17)
    particles = jnp.arange(1, count + 1, dtype=jnp.float64).reshape(shape + (17,))
    energy = 2.0 * particles + 1.0
    return SmoothCompressibleKineticState(particles, energy)


def _uniform_state(
    particles: Any,
    energy: Any,
    shape: tuple[int, int] = SPATIAL_SHAPE,
) -> SmoothCompressibleKineticState:
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(jnp.asarray(particles), shape + (17,)),
        jnp.broadcast_to(jnp.asarray(energy), shape + (17,)),
    )


def _content(topology: Any, state: SmoothCompressibleKineticState, /) -> np.ndarray:
    particles = np.asarray(state.particle_populations, dtype=np.float64)
    energy = np.asarray(state.total_energy_populations, dtype=np.float64)
    velocities = np.asarray(topology.quadrature.velocities, dtype=np.float64)
    volume = float(topology.cell_volume)
    mass = float(np.sum(particles) * volume)
    momentum = np.sum(np.einsum("xyq,qd->xyd", particles, velocities), axis=(0, 1))
    total_energy = float(np.sum(energy) * volume)
    return np.concatenate(
        (
            np.asarray((mass,), dtype=np.float64),
            np.asarray(momentum * volume, dtype=np.float64),
            np.asarray((total_energy,), dtype=np.float64),
        )
    )


def _prepare_runtime() -> tuple[
    SmoothCompressibleD2V17SpatialPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
]:
    quadrature = d2v17_quadrature(dtype=jnp.float64)
    material = IdealGasMaterial(1.4, 1.0)
    transport = ConstantTransport(0.03, 0.04)
    method = SmoothCompressibleD2VKineticMethod(quadrature, material, transport)
    energy_plan = PositiveEnergyEquilibriumPlan(quadrature)
    plan = SmoothCompressibleD2V17SpatialPlan(
        method,
        energy_plan,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
    )
    return plan, plan.prepare()


def _periodic_compatibility(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    /,
) -> tuple[dict[str, Any], PeriodicD2VBoundaryPlan]:
    plan = PeriodicD2VBoundaryPlan(
        runtime.method.quadrature,
        runtime.transport.spatial_shape,
        runtime.transport.cell_spacing,
        runtime.required_step_size,
        retain_history=True,
    )
    result = plan.route(state)
    expected_particles = runtime.transport.transport(state.particle_populations)
    expected_energy = runtime.transport.transport(state.total_energy_populations)
    history_matches = bool(result.history is not None)
    if result.history is not None:
        history_matches = bool(
            _array_equal(result.history.source_particle_populations, expected_particles)
            and _array_equal(
                result.history.source_total_energy_populations, expected_energy
            )
            and _array_equal(result.history.boundary_written, plan.topology.boundary_mask)
            and not bool(np.any(np.asarray(result.history.backflow)))
        )
    compatible = bool(
        bool(result.successful)
        and not bool(result.rollback_applied)
        and tuple(plan.topology.pull_offsets) == tuple(runtime.transport.pull_offsets)
        and plan.topology.quadrature.quadrature_id
        == runtime.transport.quadrature.quadrature_id
        and plan.topology.spatial_shape == runtime.transport.spatial_shape
        and plan.topology.cell_spacing == runtime.transport.cell_spacing
        and plan.topology.time_step == runtime.transport.time_step
        and _array_equal(result.candidate_state.particle_populations, expected_particles)
        and _array_equal(result.candidate_state.total_energy_populations, expected_energy)
        and _allclose(result.mass_momentum_energy_exchange, np.zeros((4,)))
        and _allclose(result.wall_momentum_impulse, np.zeros((2,)))
        and _array_equal(result.heat_exchange, jnp.asarray(0.0, dtype=jnp.float64))
        and _array_equal(result.wall_work, jnp.asarray(0.0, dtype=jnp.float64))
        and history_matches
    )
    return (
        {
            "passed": compatible,
            "boundary_plan_id": plan.plan_id,
            "boundary_topology_id": plan.topology.topology_id,
            "parent_periodic_transport_id": runtime.transport.plan_id,
            "particle_maximum_absolute_difference": _maximum_absolute_difference(
                result.candidate_state.particle_populations, expected_particles
            ),
            "energy_maximum_absolute_difference": _maximum_absolute_difference(
                result.candidate_state.total_energy_populations, expected_energy
            ),
            "exchange": [
                float(value) for value in np.asarray(result.mass_momentum_energy_exchange)
            ],
            "history_retained_and_exact": history_matches,
        },
        plan,
    )


def _long_link_routes(
    plan: SpecularAdiabaticD2VBoundaryPlan,
    state: SmoothCompressibleKineticState,
    /,
) -> tuple[dict[str, Any], Any]:
    result = plan.route(state)
    topology = plan.topology
    velocities = np.asarray(topology.quadrature.velocities, dtype=np.int64)
    particles = np.asarray(state.particle_populations)
    energy = np.asarray(state.total_energy_populations)
    candidate_particles = np.asarray(result.candidate_state.particle_populations)
    candidate_energy = np.asarray(result.candidate_state.total_energy_populations)
    source_x = np.asarray(topology.source_x)
    source_y = np.asarray(topology.source_y)
    source_direction = np.asarray(topology.source_direction)
    owner = np.asarray(topology.owner)
    physical_axis_mask = np.asarray(topology.physical_axis_mask)
    physical_side_sign = np.asarray(topology.physical_side_sign)
    primary_face = np.asarray(topology.primary_face_index)
    secondary_face = np.asarray(topology.secondary_face_index)
    corner_index = np.asarray(topology.corner_index)
    velocity_to_direction = {
        (int(velocity[0]), int(velocity[1])): index
        for index, velocity in enumerate(velocities)
    }
    owner_inventory_exact = bool(
        sum(topology.owner_counts) == int(np.prod(topology.population_shape))
        and set(np.unique(owner)).issubset(
            {int(candidate) for candidate in SmoothCompressibleD2VLinkOwner}
        )
    )

    face_counts = {face.name: 0 for face in SmoothCompressibleD2VBoundaryFace}
    corner_counts = {corner.name: 0 for corner in SmoothCompressibleD2VBoundaryCorner}
    direction_face_counts: dict[str, int] = {}
    direction_corner_counts: dict[str, int] = {}
    metadata_exact = True
    populations_exact = True
    checked_links = 0

    for direction, velocity_array in enumerate(velocities):
        velocity = (int(velocity_array[0]), int(velocity_array[1]))
        if max(abs(velocity[0]), abs(velocity[1])) != 2:
            continue
        velocity_name = f"({velocity[0]},{velocity[1]})"
        direction_face_counts[velocity_name] = 0
        direction_corner_counts[velocity_name] = 0
        for x in range(topology.spatial_shape[0]):
            for y in range(topology.spatial_shape[1]):
                raw_source = (x - velocity[0], y - velocity[1])
                crossings: list[tuple[int, int]] = []
                routed_source = [raw_source[0], raw_source[1]]
                reflected_velocity = [velocity[0], velocity[1]]
                for axis in range(2):
                    if raw_source[axis] < 0:
                        side_sign = -1
                    elif raw_source[axis] >= topology.spatial_shape[axis]:
                        side_sign = 1
                    else:
                        continue
                    crossings.append((axis, side_sign))
                    routed_source[axis] = (
                        -raw_source[axis] - 1
                        if side_sign < 0
                        else 2 * topology.spatial_shape[axis] - raw_source[axis] - 1
                    )
                    reflected_velocity[axis] = -reflected_velocity[axis]
                if not crossings:
                    continue

                checked_links += 1
                expected_faces = tuple(
                    2 * axis + (1 if side_sign > 0 else 0)
                    for axis, side_sign in crossings
                )
                expected_corner = -1
                if len(crossings) == 1:
                    face = SmoothCompressibleD2VBoundaryFace(expected_faces[0])
                    face_counts[face.name] += 1
                    direction_face_counts[velocity_name] += 1
                else:
                    x_sign = crossings[0][1]
                    y_sign = crossings[1][1]
                    expected_corner = (2 if x_sign > 0 else 0) + (1 if y_sign > 0 else 0)
                    corner = SmoothCompressibleD2VBoundaryCorner(expected_corner)
                    corner_counts[corner.name] += 1
                    direction_corner_counts[velocity_name] += 1

                index = (x, y, direction)
                expected_direction = velocity_to_direction[
                    (reflected_velocity[0], reflected_velocity[1])
                ]
                expected_axis_mask = tuple(
                    any(axis == crossing_axis for crossing_axis, _ in crossings)
                    for axis in range(2)
                )
                expected_side_sign = tuple(
                    next(
                        (
                            side_sign
                            for crossing_axis, side_sign in crossings
                            if crossing_axis == axis
                        ),
                        0,
                    )
                    for axis in range(2)
                )
                metadata_exact = bool(
                    metadata_exact
                    and int(owner[index])
                    == int(SmoothCompressibleD2VLinkOwner.SPECULAR_ADIABATIC_WALL)
                    and int(source_x[index]) == routed_source[0]
                    and int(source_y[index]) == routed_source[1]
                    and int(source_direction[index]) == expected_direction
                    and tuple(bool(value) for value in physical_axis_mask[index])
                    == expected_axis_mask
                    and tuple(int(value) for value in physical_side_sign[index])
                    == expected_side_sign
                    and int(primary_face[index]) == expected_faces[0]
                    and int(secondary_face[index])
                    == (expected_faces[1] if len(expected_faces) == 2 else -1)
                    and int(corner_index[index]) == expected_corner
                )
                expected_index = (
                    routed_source[0],
                    routed_source[1],
                    expected_direction,
                )
                populations_exact = bool(
                    populations_exact
                    and candidate_particles[index] == particles[expected_index]
                    and candidate_energy[index] == energy[expected_index]
                )

    expected_long_velocities = {
        "(2,0)",
        "(-2,0)",
        "(0,2)",
        "(0,-2)",
        "(2,2)",
        "(2,-2)",
        "(-2,2)",
        "(-2,-2)",
    }
    diagonal_velocities = {
        name
        for name in expected_long_velocities
        if name not in {"(2,0)", "(-2,0)", "(0,2)", "(0,-2)"}
    }
    coverage_complete = bool(
        set(direction_face_counts) == expected_long_velocities
        and all(count > 0 for count in direction_face_counts.values())
        and all(direction_corner_counts[name] > 0 for name in diagonal_velocities)
        and all(
            direction_corner_counts[name] == 0
            for name in expected_long_velocities - diagonal_velocities
        )
        and all(count > 0 for count in face_counts.values())
        and all(count > 0 for count in corner_counts.values())
    )
    passed = bool(
        bool(result.successful)
        and not bool(result.rollback_applied)
        and checked_links > 0
        and owner_inventory_exact
        and coverage_complete
        and metadata_exact
        and populations_exact
    )
    return (
        {
            "passed": passed,
            "checked_long_boundary_links": checked_links,
            "long_velocity_count": len(direction_face_counts),
            "direction_face_route_counts": direction_face_counts,
            "direction_corner_route_counts": direction_corner_counts,
            "face_route_counts": face_counts,
            "corner_route_counts": corner_counts,
            "coverage_complete": coverage_complete,
            "exclusive_owner_inventory_exact": owner_inventory_exact,
            "topology_metadata_exact": metadata_exact,
            "coupled_population_routes_exact": populations_exact,
        },
        result,
    )


def _specular_adiabatic(
    plan: SpecularAdiabaticD2VBoundaryPlan,
    state: SmoothCompressibleKineticState,
    result: Any,
    /,
) -> dict[str, Any]:
    actual_exchange = _content(plan.topology, result.accepted_state) - _content(
        plan.topology, state
    )
    reported_exchange = np.asarray(result.mass_momentum_energy_exchange)
    history_exact = bool(result.history is not None)
    if result.history is not None:
        history_exact = bool(
            _array_equal(
                result.history.source_particle_populations,
                plan.topology.gather(state.particle_populations),
            )
            and _array_equal(
                result.history.source_total_energy_populations,
                plan.topology.gather(state.total_energy_populations),
            )
            and _array_equal(result.history.boundary_written, plan.topology.boundary_mask)
            and not bool(np.any(np.asarray(result.history.backflow)))
        )
    passed = bool(
        bool(result.successful)
        and not bool(result.rollback_applied)
        and _array_equal(
            result.accepted_state.particle_populations,
            result.candidate_state.particle_populations,
        )
        and _array_equal(
            result.accepted_state.total_energy_populations,
            result.candidate_state.total_energy_populations,
        )
        and _allclose(reported_exchange, actual_exchange)
        and abs(float(reported_exchange[0])) <= ABSOLUTE_TOLERANCE
        and abs(float(reported_exchange[3])) <= ABSOLUTE_TOLERANCE
        and float(np.linalg.norm(reported_exchange[1:3])) > ABSOLUTE_TOLERANCE
        and _allclose(result.wall_momentum_impulse, -reported_exchange[1:3])
        and _array_equal(result.heat_exchange, jnp.asarray(0.0, dtype=jnp.float64))
        and _array_equal(result.wall_work, jnp.asarray(0.0, dtype=jnp.float64))
        and history_exact
    )
    return {
        "passed": passed,
        "plan_id": plan.plan_id,
        "topology_id": plan.topology.topology_id,
        "reported_mass_momentum_energy_exchange": [
            float(value) for value in reported_exchange
        ],
        "independent_mass_momentum_energy_exchange": [
            float(value) for value in actual_exchange
        ],
        "wall_momentum_impulse": [
            float(value) for value in np.asarray(result.wall_momentum_impulse)
        ],
        "heat_exchange": float(result.heat_exchange),
        "wall_work": float(result.wall_work),
        "history_exact": history_exact,
    }


def _maxwell_thermal(
    quadrature: Any,
    /,
) -> tuple[dict[str, Any], MaxwellThermalD2VBoundaryPlan]:
    diffuse_particles = quadrature.weights
    diffuse_energy = 3.0 * diffuse_particles
    state = _unique_state()

    zero_plan = MaxwellThermalD2VBoundaryPlan(
        quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        diffuse_particle_populations=diffuse_particles,
        diffuse_total_energy_populations=diffuse_energy,
        accommodation=0.0,
        wall_velocity=jnp.zeros((2,), dtype=jnp.float64),
        retain_history=True,
    )
    zero_specular_plan = SpecularAdiabaticD2VBoundaryPlan(
        quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        retain_history=True,
    )
    zero_result = zero_plan.route(state)
    zero_specular_result = zero_specular_plan.route(state)
    zero_matches_specular = bool(
        bool(zero_result.successful)
        and bool(zero_specular_result.successful)
        and int(zero_result.status) == int(zero_specular_result.status)
        and bool(zero_result.rollback_applied)
        == bool(zero_specular_result.rollback_applied)
        and _array_equal(
            zero_result.candidate_state.particle_populations,
            zero_specular_result.candidate_state.particle_populations,
        )
        and _array_equal(
            zero_result.candidate_state.total_energy_populations,
            zero_specular_result.candidate_state.total_energy_populations,
        )
        and _array_equal(
            zero_result.accepted_state.particle_populations,
            zero_specular_result.accepted_state.particle_populations,
        )
        and _array_equal(
            zero_result.accepted_state.total_energy_populations,
            zero_specular_result.accepted_state.total_energy_populations,
        )
        and _array_equal(
            zero_result.mass_momentum_energy_exchange,
            zero_specular_result.mass_momentum_energy_exchange,
        )
        and _array_equal(
            zero_result.wall_momentum_impulse,
            zero_specular_result.wall_momentum_impulse,
        )
        and _array_equal(zero_result.heat_exchange, zero_specular_result.heat_exchange)
        and _array_equal(zero_result.wall_work, zero_specular_result.wall_work)
    )

    wall_velocity = jnp.asarray(
        (
            (0.0, 0.2),
            (0.0, -0.1),
            (0.15, 0.0),
            (-0.05, 0.0),
        ),
        dtype=jnp.float64,
    )
    positive_plan = MaxwellThermalD2VBoundaryPlan(
        quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        diffuse_particle_populations=diffuse_particles,
        diffuse_total_energy_populations=diffuse_energy,
        accommodation=0.75,
        wall_velocity=wall_velocity,
        retain_history=True,
    )
    positive_result = positive_plan.route(state)
    topology = positive_plan.topology
    owner_mask = np.asarray(topology.owner) == int(
        SmoothCompressibleD2VLinkOwner.MAXWELL_THERMAL_WALL
    )
    primary_face = np.asarray(topology.primary_face_index)
    gathered_particles = np.asarray(topology.gather(state.particle_populations))
    accepted_particles = np.asarray(positive_result.accepted_state.particle_populations)
    face_flux_residuals: list[float] = []
    face_link_counts: list[int] = []
    for face in SmoothCompressibleD2VBoundaryFace:
        face_mask = owner_mask & (primary_face == int(face))
        face_link_counts.append(int(np.count_nonzero(face_mask)))
        face_flux_residuals.append(
            float(
                np.sum(accepted_particles[face_mask])
                - np.sum(gathered_particles[face_mask])
            )
        )
    mass_flux_tolerance = 5.0e-10
    half_range_mass_flux_closed = bool(
        all(count > 0 for count in face_link_counts)
        and all(abs(residual) <= mass_flux_tolerance for residual in face_flux_residuals)
    )
    actual_exchange = _content(topology, positive_result.accepted_state) - _content(
        topology, state
    )
    reported_exchange = np.asarray(positive_result.mass_momentum_energy_exchange)
    heat_exchange = float(positive_result.heat_exchange)
    wall_work = float(positive_result.wall_work)
    accepted_energy_exchange = float(actual_exchange[3])
    exchange_scale = max(float(np.max(np.abs(actual_exchange))), 1.0)
    exchange_tolerance = 512.0 * np.finfo(np.float64).eps * exchange_scale
    energy_scale = max(abs(accepted_energy_exchange), 1.0)
    energy_tolerance = 512.0 * np.finfo(np.float64).eps * energy_scale
    heat_work_residual = float(
        jnp.abs(
            positive_result.heat_exchange
            + positive_result.wall_work
            - positive_result.mass_momentum_energy_exchange[3]
        )
    )
    heat_and_work_separated = bool(
        abs(heat_exchange) > ABSOLUTE_TOLERANCE and abs(wall_work) > ABSOLUTE_TOLERANCE
    )
    heat_and_work_close_energy = bool(heat_work_residual <= energy_tolerance)
    positive_accommodation_passed = bool(
        bool(positive_result.successful)
        and int(positive_result.status)
        == int(SmoothCompressibleD2VBoundaryStatus.SUCCESS)
        and not bool(positive_result.rollback_applied)
        and _array_equal(
            positive_result.accepted_state.particle_populations,
            positive_result.candidate_state.particle_populations,
        )
        and _array_equal(
            positive_result.accepted_state.total_energy_populations,
            positive_result.candidate_state.total_energy_populations,
        )
        and np.all(accepted_particles > 0.0)
        and np.all(
            np.asarray(positive_result.accepted_state.total_energy_populations) > 0.0
        )
        and half_range_mass_flux_closed
        and _allclose(
            reported_exchange,
            actual_exchange,
            absolute_tolerance=exchange_tolerance,
        )
        and _allclose(
            positive_result.wall_momentum_impulse,
            -reported_exchange[1:3],
        )
        and heat_and_work_separated
        and heat_and_work_close_energy
        and positive_result.history is not None
    )

    tiny = np.asarray(1.0e-300, dtype=np.float64)
    degenerate_particles = np.full((4, 17), tiny, dtype=np.float64)
    rest_direction = _direction(quadrature, (0, 0))
    degenerate_particles[:, rest_direction] = 1.0
    degenerate_energy = np.broadcast_to(
        3.0 * np.asarray(diffuse_particles, dtype=np.float64),
        (4, 17),
    ).copy()
    degenerate_plan = MaxwellThermalD2VBoundaryPlan(
        quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        diffuse_particle_populations=degenerate_particles,
        diffuse_total_energy_populations=degenerate_energy,
        accommodation=1.0,
        wall_velocity=np.zeros((2,), dtype=np.float64),
        periodic_axes=(False, True),
        retain_history=True,
    )
    degenerate_state = SmoothCompressibleKineticState(
        jnp.ones(SPATIAL_SHAPE + (17,), dtype=jnp.float32),
        2.0 * jnp.ones(SPATIAL_SHAPE + (17,), dtype=jnp.float32),
    )
    degenerate_result = degenerate_plan.route(degenerate_state)
    degenerate_particle_rollback = _array_equal(
        degenerate_result.accepted_state.particle_populations,
        degenerate_state.particle_populations,
    )
    degenerate_energy_rollback = _array_equal(
        degenerate_result.accepted_state.total_energy_populations,
        degenerate_state.total_energy_populations,
    )
    degenerate_ledger_zero = bool(
        _array_equal(
            degenerate_result.mass_momentum_energy_exchange,
            np.zeros((4,), dtype=np.float32),
        )
        and _array_equal(
            degenerate_result.wall_momentum_impulse,
            np.zeros((2,), dtype=np.float32),
        )
        and _array_equal(
            degenerate_result.heat_exchange,
            np.asarray(0.0, dtype=np.float32),
        )
        and _array_equal(
            degenerate_result.wall_work,
            np.asarray(0.0, dtype=np.float32),
        )
    )
    degenerate_refused = bool(
        not bool(degenerate_result.successful)
        and int(degenerate_result.status)
        == int(SmoothCompressibleD2VBoundaryStatus.DEGENERATE_DIFFUSE_FLUX)
        and bool(degenerate_result.rollback_applied)
        and degenerate_particle_rollback
        and degenerate_energy_rollback
        and degenerate_ledger_zero
    )

    nonpositive_state = SmoothCompressibleKineticState(
        jnp.zeros(SPATIAL_SHAPE + (17,), dtype=jnp.float64),
        jnp.zeros(SPATIAL_SHAPE + (17,), dtype=jnp.float64),
    )
    nonpositive_result = positive_plan.route(nonpositive_state)
    nonpositive_particle_rollback = _array_equal(
        nonpositive_result.accepted_state.particle_populations,
        nonpositive_state.particle_populations,
    )
    nonpositive_energy_rollback = _array_equal(
        nonpositive_result.accepted_state.total_energy_populations,
        nonpositive_state.total_energy_populations,
    )
    nonpositive_ledger_zero = bool(
        _array_equal(
            nonpositive_result.mass_momentum_energy_exchange,
            np.zeros((4,), dtype=np.float64),
        )
        and _array_equal(
            nonpositive_result.wall_momentum_impulse,
            np.zeros((2,), dtype=np.float64),
        )
        and _array_equal(
            nonpositive_result.heat_exchange,
            np.asarray(0.0, dtype=np.float64),
        )
        and _array_equal(
            nonpositive_result.wall_work,
            np.asarray(0.0, dtype=np.float64),
        )
    )
    nonpositive_refused = bool(
        not bool(nonpositive_result.successful)
        and int(nonpositive_result.status)
        == int(SmoothCompressibleD2VBoundaryStatus.NONPOSITIVE_CANDIDATE)
        and bool(nonpositive_result.rollback_applied)
        and nonpositive_particle_rollback
        and nonpositive_energy_rollback
        and nonpositive_ledger_zero
    )

    zero_case = {
        "passed": zero_matches_specular,
        "plan_id": zero_plan.plan_id,
        "reference_specular_plan_id": zero_specular_plan.plan_id,
        "status": SmoothCompressibleD2VBoundaryStatus(int(zero_result.status)).name,
        "coupled_state_and_ledgers_exactly_equal": zero_matches_specular,
    }
    positive_case = {
        "passed": positive_accommodation_passed,
        "plan_id": positive_plan.plan_id,
        "status": SmoothCompressibleD2VBoundaryStatus(int(positive_result.status)).name,
        "accommodation": list(positive_plan.accommodation),
        "tangential_wall_velocity": [
            [float(value) for value in row] for row in np.asarray(wall_velocity)
        ],
        "face_link_counts": {
            face.name: face_link_counts[int(face)]
            for face in SmoothCompressibleD2VBoundaryFace
        },
        "face_half_range_mass_flux_residuals": {
            face.name: face_flux_residuals[int(face)]
            for face in SmoothCompressibleD2VBoundaryFace
        },
        "mass_flux_absolute_tolerance": mass_flux_tolerance,
        "half_range_mass_flux_closed": half_range_mass_flux_closed,
        "reported_mass_momentum_energy_exchange": [
            float(value) for value in reported_exchange
        ],
        "independent_mass_momentum_energy_exchange": [
            float(value) for value in actual_exchange
        ],
        "accepted_energy_exchange": accepted_energy_exchange,
        "heat_exchange": heat_exchange,
        "heat_plus_work_residual": heat_work_residual,
        "heat_plus_work_tolerance": energy_tolerance,
        "reported_exchange_tolerance": exchange_tolerance,
        "tangential_wall_work": wall_work,
        "heat_and_tangential_wall_work_both_exercised": heat_and_work_separated,
        "heat_plus_work_equals_accepted_energy_exchange": heat_and_work_close_energy,
    }
    refusal_case = {
        "passed": bool(degenerate_refused and nonpositive_refused),
        "degenerate_diffuse_target": {
            "refused": degenerate_refused,
            "plan_id": degenerate_plan.plan_id,
            "status": SmoothCompressibleD2VBoundaryStatus(
                int(degenerate_result.status)
            ).name,
            "rollback_applied": bool(degenerate_result.rollback_applied),
            "particle_state_fully_restored": degenerate_particle_rollback,
            "energy_state_fully_restored": degenerate_energy_rollback,
            "exchange_ledgers_exactly_zero": degenerate_ledger_zero,
            "supplied_positive_value_underflowing_in_route_dtype": float(tiny),
            "route_dtype": "float32",
        },
        "nonpositive_diffuse_candidate": {
            "refused": nonpositive_refused,
            "plan_id": positive_plan.plan_id,
            "status": SmoothCompressibleD2VBoundaryStatus(
                int(nonpositive_result.status)
            ).name,
            "rollback_applied": bool(nonpositive_result.rollback_applied),
            "particle_state_fully_restored": nonpositive_particle_rollback,
            "energy_state_fully_restored": nonpositive_energy_rollback,
            "exchange_ledgers_exactly_zero": nonpositive_ledger_zero,
        },
    }
    return (
        {
            "passed": bool(
                zero_matches_specular
                and positive_accommodation_passed
                and degenerate_refused
                and nonpositive_refused
            ),
            "topology_id": positive_plan.topology.topology_id,
            "degenerate_topology_id": degenerate_plan.topology.topology_id,
            "zero_accommodation_specular_limit": zero_case,
            "positive_accommodation": positive_case,
            "refused_targets": refusal_case,
        },
        positive_plan,
    )


def _matching_reservoir(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> tuple[dict[str, Any], EquilibriumReservoirD2VBoundaryPlan]:
    particles = runtime.safe_state.particle_populations
    energy = runtime.safe_state.total_energy_populations
    state = _uniform_state(particles, energy)
    plan = EquilibriumReservoirD2VBoundaryPlan(
        runtime.method.quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        incoming_particle_populations=particles,
        incoming_total_energy_populations=energy,
        retain_history=True,
    )
    result = plan.route(state)
    exact = bool(
        bool(result.successful)
        and int(result.status) == int(SmoothCompressibleD2VBoundaryStatus.SUCCESS)
        and not bool(result.rollback_applied)
        and _array_equal(
            result.candidate_state.particle_populations, state.particle_populations
        )
        and _array_equal(
            result.candidate_state.total_energy_populations,
            state.total_energy_populations,
        )
        and _array_equal(
            result.accepted_state.particle_populations, state.particle_populations
        )
        and _array_equal(
            result.accepted_state.total_energy_populations,
            state.total_energy_populations,
        )
        and _allclose(result.mass_momentum_energy_exchange, np.zeros((4,)))
        and result.history is not None
    )
    return (
        {
            "passed": exact,
            "plan_id": plan.plan_id,
            "topology_id": plan.topology.topology_id,
            "input_source": "prepared runtime analytic safe-equilibrium f/g populations",
            "incoming_values_supplied_directly": True,
            "state_exactly_stationary": exact,
            "exchange": [
                float(value) for value in np.asarray(result.mass_momentum_energy_exchange)
            ],
        },
        plan,
    )


def _reservoir_exchange_and_corners(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> tuple[dict[str, Any], dict[str, Any], EquilibriumReservoirD2VBoundaryPlan]:
    base_particles = runtime.safe_state.particle_populations
    base_energy = runtime.safe_state.total_energy_populations
    state = _uniform_state(base_particles, base_energy)
    directions = jnp.arange(1, 18, dtype=jnp.float64)
    face_scales = jnp.asarray((1.1, 1.4, 1.8, 2.3), dtype=jnp.float64)
    corner_scales = jnp.asarray((2.7, 3.1, 3.8, 4.6), dtype=jnp.float64)
    face_particles = jnp.stack(
        tuple(scale * base_particles + 1.0e-4 * directions for scale in face_scales)
    )
    face_energy = jnp.stack(
        tuple(scale * base_energy + 2.0e-4 * directions for scale in face_scales)
    )
    corner_particles = jnp.stack(
        tuple(scale * base_particles + 3.0e-4 * directions for scale in corner_scales)
    )
    corner_energy = jnp.stack(
        tuple(scale * base_energy + 5.0e-4 * directions for scale in corner_scales)
    )
    plan = EquilibriumReservoirD2VBoundaryPlan(
        runtime.method.quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        retain_history=True,
    )
    parameters = SmoothCompressibleD2VReservoirParameters(
        face_particles,
        face_energy,
        corner_particle_populations=corner_particles,
        corner_total_energy_populations=corner_energy,
    )
    result = plan.route(state, parameters)
    topology = plan.topology
    owner = np.asarray(topology.owner)
    primary = np.asarray(topology.primary_face_index)
    secondary = np.asarray(topology.secondary_face_index)
    corners = np.asarray(topology.corner_index)
    candidate_particles = np.asarray(result.candidate_state.particle_populations)
    candidate_energy = np.asarray(result.candidate_state.total_energy_populations)
    face_particles_host = np.asarray(face_particles)
    face_energy_host = np.asarray(face_energy)
    corner_particles_host = np.asarray(corner_particles)
    corner_energy_host = np.asarray(corner_energy)
    boundary_owner = int(SmoothCompressibleD2VLinkOwner.EQUILIBRIUM_RESERVOIR)
    route_values_exact = True
    face_route_count = 0
    corner_route_counts = {
        corner.name: 0 for corner in SmoothCompressibleD2VBoundaryCorner
    }
    for index in np.ndindex(topology.population_shape):
        if int(owner[index]) != boundary_owner:
            continue
        direction = index[2]
        if int(secondary[index]) >= 0:
            corner = SmoothCompressibleD2VBoundaryCorner(int(corners[index]))
            corner_route_counts[corner.name] += 1
            expected_particles = corner_particles_host[int(corners[index]), direction]
            expected_energy = corner_energy_host[int(corners[index]), direction]
        else:
            face_route_count += 1
            expected_particles = face_particles_host[int(primary[index]), direction]
            expected_energy = face_energy_host[int(primary[index]), direction]
        route_values_exact = bool(
            route_values_exact
            and candidate_particles[index] == expected_particles
            and candidate_energy[index] == expected_energy
        )

    actual_exchange = _content(topology, result.accepted_state) - _content(
        topology, state
    )
    reported_exchange = np.asarray(result.mass_momentum_energy_exchange)
    all_exchange_components_nonzero = bool(
        np.all(np.abs(reported_exchange) > ABSOLUTE_TOLERANCE)
    )
    exchange_passed = bool(
        bool(result.successful)
        and int(result.status) == int(SmoothCompressibleD2VBoundaryStatus.SUCCESS)
        and not bool(result.rollback_applied)
        and route_values_exact
        and face_route_count > 0
        and all(count > 0 for count in corner_route_counts.values())
        and _allclose(reported_exchange, actual_exchange)
        and all_exchange_components_nonzero
        and _allclose(result.wall_momentum_impulse, np.zeros((2,)))
        and _array_equal(result.heat_exchange, jnp.asarray(0.0, dtype=jnp.float64))
        and _array_equal(result.wall_work, jnp.asarray(0.0, dtype=jnp.float64))
    )

    incompatible = SmoothCompressibleD2VReservoirParameters(face_particles, face_energy)
    refused = plan.route(state, incompatible)
    incompatible_rollback = bool(
        not bool(refused.successful)
        and int(refused.status)
        == int(SmoothCompressibleD2VBoundaryStatus.INCOMPATIBLE_RESERVOIR_CORNER)
        and bool(refused.rollback_applied)
        and _array_equal(
            refused.accepted_state.particle_populations, state.particle_populations
        )
        and _array_equal(
            refused.accepted_state.total_energy_populations,
            state.total_energy_populations,
        )
        and _allclose(refused.mass_momentum_energy_exchange, np.zeros((4,)))
    )
    return (
        {
            "passed": exchange_passed,
            "plan_id": plan.plan_id,
            "topology_id": topology.topology_id,
            "face_route_count": face_route_count,
            "explicit_corner_route_counts": corner_route_counts,
            "coupled_route_values_exact": route_values_exact,
            "all_exchange_components_nonzero": all_exchange_components_nonzero,
            "reported_mass_momentum_energy_exchange": [
                float(value) for value in reported_exchange
            ],
            "independent_mass_momentum_energy_exchange": [
                float(value) for value in actual_exchange
            ],
            "wall_momentum_impulse": [
                float(value) for value in np.asarray(result.wall_momentum_impulse)
            ],
            "heat_exchange": float(result.heat_exchange),
            "wall_work": float(result.wall_work),
        },
        {
            "passed": incompatible_rollback,
            "status": SmoothCompressibleD2VBoundaryStatus(int(refused.status)).name,
            "rollback_applied": bool(refused.rollback_applied),
            "particle_state_fully_restored": _array_equal(
                refused.accepted_state.particle_populations,
                state.particle_populations,
            ),
            "energy_state_fully_restored": _array_equal(
                refused.accepted_state.total_energy_populations,
                state.total_energy_populations,
            ),
        },
        plan,
    )


def _invalid_reservoir_rollback(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> tuple[dict[str, Any], EquilibriumReservoirD2VBoundaryPlan]:
    particles = runtime.safe_state.particle_populations
    energy = runtime.safe_state.total_energy_populations
    state = _uniform_state(particles, energy)
    invalid_particles = jnp.broadcast_to(particles, (4, 17)).at[0, 1].set(-1.0)
    parameters = SmoothCompressibleD2VReservoirParameters(
        invalid_particles,
        jnp.broadcast_to(energy, (4, 17)),
    )
    plan = EquilibriumReservoirD2VBoundaryPlan(
        runtime.method.quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        periodic_axes=(False, True),
        retain_history=True,
    )
    result = plan.route(state, parameters)
    particle_restored = _array_equal(
        result.accepted_state.particle_populations, state.particle_populations
    )
    energy_restored = _array_equal(
        result.accepted_state.total_energy_populations,
        state.total_energy_populations,
    )
    candidate_rejected = bool(
        not _array_equal(
            result.candidate_state.particle_populations, state.particle_populations
        )
    )
    passed = bool(
        not bool(result.successful)
        and int(result.status) == int(SmoothCompressibleD2VBoundaryStatus.INVALID_TARGET)
        and bool(result.rollback_applied)
        and particle_restored
        and energy_restored
        and candidate_rejected
        and _allclose(result.mass_momentum_energy_exchange, np.zeros((4,)))
        and _allclose(result.wall_momentum_impulse, np.zeros((2,)))
        and _array_equal(result.heat_exchange, jnp.asarray(0.0, dtype=jnp.float64))
        and _array_equal(result.wall_work, jnp.asarray(0.0, dtype=jnp.float64))
    )
    return (
        {
            "passed": passed,
            "plan_id": plan.plan_id,
            "topology_id": plan.topology.topology_id,
            "status": SmoothCompressibleD2VBoundaryStatus(int(result.status)).name,
            "rollback_applied": bool(result.rollback_applied),
            "rejected_candidate_differs": candidate_rejected,
            "particle_state_fully_restored": particle_restored,
            "energy_state_fully_restored": energy_restored,
            "exchange": [
                float(value) for value in np.asarray(result.mass_momentum_energy_exchange)
            ],
        },
        plan,
    )


def _outward_extrapolation(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> tuple[dict[str, Any], OutwardExtrapolationD2VBoundaryPlan]:
    quadrature = runtime.method.quadrature
    positive_x = _direction(quadrature, (1, 0))
    negative_x = _direction(quadrature, (-1, 0))
    base_particles = jnp.ones(SPATIAL_SHAPE + (17,), dtype=jnp.float64)
    base_energy = 2.0 * jnp.ones(SPATIAL_SHAPE + (17,), dtype=jnp.float64)
    reach = 2
    outward_particles = base_particles.at[:reach, :, negative_x].add(1.0)
    outward_particles = outward_particles.at[-reach:, :, positive_x].add(1.0)
    outward_state = SmoothCompressibleKineticState(outward_particles, base_energy)
    plan = OutwardExtrapolationD2VBoundaryPlan(
        quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        periodic_axes=(False, True),
        retain_history=True,
    )
    accepted = plan.route(outward_state)
    accepted_actual_exchange = _content(
        plan.topology, accepted.accepted_state
    ) - _content(plan.topology, outward_state)
    outward_accepted = bool(
        bool(accepted.successful)
        and int(accepted.status) == int(SmoothCompressibleD2VBoundaryStatus.SUCCESS)
        and not bool(accepted.rollback_applied)
        and _allclose(accepted.mass_momentum_energy_exchange, accepted_actual_exchange)
        and accepted.history is not None
        and not bool(np.any(np.asarray(accepted.history.backflow)))
    )

    backflow_particles = base_particles.at[:reach, :, positive_x].add(1.0)
    backflow_particles = backflow_particles.at[-reach:, :, positive_x].add(1.0)
    backflow_state = SmoothCompressibleKineticState(backflow_particles, base_energy)
    refused = plan.route(backflow_state)
    particle_restored = _array_equal(
        refused.accepted_state.particle_populations,
        backflow_state.particle_populations,
    )
    energy_restored = _array_equal(
        refused.accepted_state.total_energy_populations,
        backflow_state.total_energy_populations,
    )
    backflow_detected = bool(
        refused.history is not None
        and np.any(np.asarray(refused.history.backflow))
        and np.all(
            ~np.asarray(refused.history.backflow)
            | (
                np.asarray(plan.topology.owner)
                == int(SmoothCompressibleD2VLinkOwner.OUTWARD_EXTRAPOLATION)
            )
        )
    )
    backflow_refused = bool(
        not bool(refused.successful)
        and int(refused.status) == int(SmoothCompressibleD2VBoundaryStatus.BACKFLOW)
        and bool(refused.rollback_applied)
        and particle_restored
        and energy_restored
        and backflow_detected
        and _allclose(refused.mass_momentum_energy_exchange, np.zeros((4,)))
        and _allclose(refused.wall_momentum_impulse, np.zeros((2,)))
        and _array_equal(refused.heat_exchange, jnp.asarray(0.0, dtype=jnp.float64))
        and _array_equal(refused.wall_work, jnp.asarray(0.0, dtype=jnp.float64))
    )
    return (
        {
            "passed": bool(outward_accepted and backflow_refused),
            "plan_id": plan.plan_id,
            "topology_id": plan.topology.topology_id,
            "periodic_axes": list(plan.topology.periodic_axes),
            "physical_axis": "x",
            "outward_case": {
                "accepted": outward_accepted,
                "status": SmoothCompressibleD2VBoundaryStatus(int(accepted.status)).name,
                "reported_mass_momentum_energy_exchange": [
                    float(value)
                    for value in np.asarray(accepted.mass_momentum_energy_exchange)
                ],
                "independent_mass_momentum_energy_exchange": [
                    float(value) for value in accepted_actual_exchange
                ],
            },
            "backflow_case": {
                "refused": backflow_refused,
                "status": SmoothCompressibleD2VBoundaryStatus(int(refused.status)).name,
                "backflow_links_detected": int(
                    np.count_nonzero(np.asarray(refused.history.backflow))
                    if refused.history is not None
                    else 0
                ),
                "particle_state_fully_restored": particle_restored,
                "energy_state_fully_restored": energy_restored,
            },
        },
        plan,
    )


def _qualification(
    parent_spatial_id: str,
    model_id: str,
    support_id: str,
    /,
) -> dict[str, Any]:
    parent = _identifier(parent_spatial_id, "parent spatial ID")
    model = _identifier(model_id, "model ID")
    support = _identifier(support_id, "support ID")
    spatial_plan, runtime = _prepare_runtime()
    state = _unique_state()

    periodic, periodic_plan = _periodic_compatibility(runtime, state)
    specular_plan = SpecularAdiabaticD2VBoundaryPlan(
        runtime.method.quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        retain_history=True,
    )
    long_links, specular_result = _long_link_routes(specular_plan, state)
    specular = _specular_adiabatic(specular_plan, state, specular_result)
    maxwell, maxwell_plan = _maxwell_thermal(runtime.method.quadrature)
    matching_reservoir, matching_reservoir_plan = _matching_reservoir(runtime)
    (
        reservoir_exchange,
        incompatible_corner,
        reservoir_exchange_plan,
    ) = _reservoir_exchange_and_corners(runtime)
    invalid_reservoir, invalid_reservoir_plan = _invalid_reservoir_rollback(runtime)
    outward, outward_plan = _outward_extrapolation(runtime)

    topology_ids = {
        "periodic": periodic_plan.topology.topology_id,
        "specular_adiabatic": specular_plan.topology.topology_id,
        "maxwell_thermal": maxwell_plan.topology.topology_id,
        "maxwell_thermal_degenerate": maxwell["degenerate_topology_id"],
        "matching_equilibrium_reservoir": matching_reservoir_plan.topology.topology_id,
        "exchange_equilibrium_reservoir": reservoir_exchange_plan.topology.topology_id,
        "invalid_target_reservoir": invalid_reservoir_plan.topology.topology_id,
        "outward_extrapolation": outward_plan.topology.topology_id,
    }
    topology_set_id = canonical_fingerprint(
        {
            "kind": "learned-energy-boundary-topology-set",
            "topologies": topology_ids,
        }
    )
    gates = {
        "periodic_matches_parent_spatial_transport": bool(periodic["passed"]),
        "all_d2v17_long_face_and_corner_routes": bool(long_links["passed"]),
        "stationary_specular_adiabatic_ledger": bool(specular["passed"]),
        "zero_accommodation_maxwell_exactly_matches_specular": bool(
            maxwell["zero_accommodation_specular_limit"]["passed"]
        ),
        "positive_accommodation_maxwell_mass_heat_and_work": bool(
            maxwell["positive_accommodation"]["passed"]
        ),
        "degenerate_and_nonpositive_maxwell_full_rollback": bool(
            maxwell["refused_targets"]["passed"]
        ),
        "matching_equilibrium_reservoir_stationarity": bool(matching_reservoir["passed"]),
        "explicit_reservoir_corners_and_actual_exchange": bool(
            reservoir_exchange["passed"]
        ),
        "incompatible_reservoir_corner_full_rollback": bool(
            incompatible_corner["passed"]
        ),
        "invalid_reservoir_target_full_rollback": bool(invalid_reservoir["passed"]),
        "restricted_outward_extrapolation_and_backflow_refusal": bool(outward["passed"]),
    }
    report: dict[str, Any] = {
        "tool": "learned_energy_boundary_qualification",
        "scope": {
            "claim": "deterministic coupled D2V17 boundary qualification only",
            "included": [
                "periodic coupled f/g pull compatibility with the constructed parent runtime",
                "all D2V17 reach-two face and corner routes",
                "stationary specular adiabatic reflection",
                "supplied-population Maxwell thermal reflection, including the exact zero-accommodation specular limit",
                "positive-accommodation half-range mass-flux closure and separate heat/tangential-wall-work ledgers",
                "matching prepared equilibrium reservoir populations",
                "explicit equilibrium-reservoir corner routing and incompatible-corner refusal",
                "restricted zero-gradient outward extrapolation and strict backflow refusal",
                "accepted mass/momentum/energy exchange, stationary-wall impulse, heat, and work ledgers",
                "atomic full f/g state rollback on refused transactions",
            ],
            "excluded": [
                "exact target-temperature enforcement by Maxwell supplied populations",
                "exact no-slip enforcement by Maxwell supplied populations",
                "general moving-wall boundaries beyond static Cartesian faces with declared tangential wall velocity",
                "boundary-time learned-model evaluation",
                "forcing",
                "shock qualification",
                "production-readiness claims",
            ],
        },
        "lineage": {
            "parent_spatial_id": parent,
            "runtime_id": runtime.prepared_id,
            "model_id": model,
            "support_id": support,
            "topology_id": topology_set_id,
        },
        "identities": {
            "quadrature": runtime.method.quadrature.quadrature_id,
            "material": runtime.method.material.material_id,
            "transport_closure": runtime.method.transport.closure_id,
            "kinetic_method": runtime.method.method_id,
            "energy_plan": runtime.energy_plan.plan_id,
            "spatial_plan": spatial_plan.plan_id,
            "runtime": runtime.prepared_id,
            "runtime_program_manifest": runtime.program_manifest.manifest_id,
            "parent_periodic_transport": runtime.transport.plan_id,
            "boundary_topology_set": topology_set_id,
            "boundary_topologies": topology_ids,
        },
        "configuration": {
            "spatial_shape": list(SPATIAL_SHAPE),
            "cell_spacing": list(CELL_SPACING),
            "time_step": TIME_STEP,
            "population_count": 17,
            "primary_dtype": "float64",
            "degenerate_diffuse_flux_route_dtype": "float32",
            "deterministic": True,
        },
        "periodic_compatibility": periodic,
        "long_link_routes": long_links,
        "specular_adiabatic": specular,
        "maxwell_thermal": maxwell,
        "matching_equilibrium_reservoir": matching_reservoir,
        "reservoir_exchange_and_corners": reservoir_exchange,
        "incompatible_reservoir_corner": incompatible_corner,
        "invalid_reservoir_target": invalid_reservoir,
        "outward_extrapolation": outward,
        "gates": gates,
        "passed": all(gates.values()),
    }
    if report["passed"]:
        report["qualification_id"] = canonical_fingerprint(report)
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify the implemented coupled D2V17 periodic, specular-adiabatic, "
            "Maxwell-thermal, equilibrium-reservoir, and restricted outward "
            "boundary plans."
        )
    )
    parser.add_argument(
        "--parent-spatial-id",
        required=True,
        help="exact identity of the parent spatial qualification/artifact",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="exact frozen or accepted model identity used by the parent runtime",
    )
    parser.add_argument(
        "--support-id",
        required=True,
        help="exact learned-energy support identity used by the parent runtime",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT,
        help="atomic JSON output path; written only when every gate passes",
    )
    return parser.parse_args()


def _validate_output_path(path: Path, /) -> Path:
    output = Path(path)
    if output.name in _PROTECTED_ARTIFACT_NAMES:
        raise ValueError(
            "Boundary qualification cannot overwrite a stage-one, spatial, or "
            "stage-two artifact."
        )
    if output.suffix != ".json":
        raise ValueError("Boundary qualification output must be a JSON artifact.")
    return output


def main() -> int:
    arguments = _parse_arguments()
    output = _validate_output_path(arguments.output)
    with jax.enable_x64(True):
        report = _qualification(
            arguments.parent_spatial_id,
            arguments.model_id,
            arguments.support_id,
        )
    if not report["passed"]:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
        return 1
    write_json_atomic(output, report)
    print(f"wrote {output} ({report['qualification_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
