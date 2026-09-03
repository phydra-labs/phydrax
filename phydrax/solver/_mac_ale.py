#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_ale import (
    _difference,
    _dual_measures,
    _evaluate_mapped_mac_geometry,
    _full_face_geometry,
    _keep_periodic_faces,
    _tuple_finite,
    PreparedMappedMACGeometry,
)
from ..discretization.finite_volume._structured import FiniteVolumeDiscretization
from ._mac_pressure_operator import execute_weighted_pressure_iteration


MACCoordinateMap = Callable[[Array, Array, Any], ArrayLike]
MACGridVelocity = Callable[[Array, Array, Any], ArrayLike]


def _provider_vertices(provider, time, vertices, args, /) -> Array:
    dimension = vertices.shape[-1]
    flat = vertices.reshape((-1, dimension))
    values = jax.vmap(lambda point: jnp.asarray(provider(time, point, args)))(flat)
    if values.shape != flat.shape:
        raise ValueError("ALE providers must preserve point shape.")
    return values.reshape(vertices.shape)


def _face_average(vertices, reference, axis, /) -> Array:
    return _keep_periodic_faces(
        _full_face_geometry(vertices, reference, axis)[0], reference, axis
    )


def _boundary_maximum(values, reference, /) -> Array:
    residuals = []
    for axis, structured_axis in enumerate(reference.grid.structured_axes):
        if not structured_axis.periodic:
            moved = jnp.moveaxis(values[axis], axis, 0)
            residuals.extend((jnp.max(jnp.abs(moved[0])), jnp.max(jnp.abs(moved[-1]))))
    return (
        jnp.max(jnp.stack(tuple(residuals)))
        if residuals
        else jnp.asarray(0.0, dtype=values[0].dtype)
    )


def _norm_squared(volume, value, /) -> Array:
    return jnp.sum(volume * value * value)


class MACALEStageGeometry(StrictModule, NonTrainableState):
    """One dynamic, fixed-connectivity mapped MAC geometry stage."""

    reference: FiniteVolumeDiscretization
    time: Array
    mapped_vertices: Array
    vertex_grid_velocity: Array
    cell_centers: Array
    cell_volumes: Array
    cell_volume_rate: Array
    face_centers: tuple[Array, ...]
    face_measures: FaceVelocity
    face_area_vectors: tuple[Array, ...]
    face_dual_measures: FaceVelocity
    face_grid_velocity: tuple[Array, ...]
    face_grid_normal_velocity: FaceVelocity
    face_mesh_flux: FaceVelocity
    gcl_residual: Array
    maximum_gcl_residual: Array
    map_velocity_residual: Array
    boundary_kinematic_residual: Array
    free_stream_residual: Array
    mapped_adjoint_residual: Array
    minimum_cell_volume: Array
    minimum_face_measure: Array
    minimum_velocity_dual_measure: Array
    minimum_oriented_dual_distance: Array
    finite: Array
    passed: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    motion_plan_id: str = eqx.field(static=True)

    def validate_pressure(self, pressure: ArrayLike, /) -> Array:
        value = jnp.asarray(pressure, dtype=self.cell_volumes.dtype)
        if value.shape != self.cell_volumes.shape:
            raise ValueError("ALE pressure must match the fixed cell layout.")
        return value

    def validate_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = tuple(
            jnp.asarray(value, dtype=self.cell_volumes.dtype) for value in velocity
        )
        if len(values) != len(self.face_measures) or any(
            value.shape != measure.shape
            for value, measure in zip(values, self.face_measures, strict=True)
        ):
            raise ValueError("ALE velocity must match every fixed face layout.")
        return values

    def gauge_project(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_pressure(pressure)
        return value - jnp.sum(self.cell_volumes * value) / jnp.sum(self.cell_volumes)

    def compatibility_project(self, value: ArrayLike, /) -> Array:
        return self.gauge_project(value)

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        result = jnp.zeros_like(self.cell_volumes)
        for axis, component in enumerate(self.validate_velocity(velocity)):
            result = (
                result
                + _difference(
                    component * self.face_measures[axis],
                    axis,
                    self.reference.grid.structured_axes[axis].periodic,
                )
                / self.cell_volumes
            )
        return result

    def gradient(self, pressure: ArrayLike, /) -> FaceVelocity:
        value = self.validate_pressure(pressure)
        output = []
        for axis, structured_axis in enumerate(self.reference.grid.structured_axes):
            moved = jnp.moveaxis(value, axis, 0)
            if structured_axis.periodic:
                jump = moved - jnp.roll(moved, 1, axis=0)
            elif moved.shape[0] == 1:
                jump = jnp.zeros((2,) + moved.shape[1:], dtype=moved.dtype)
            else:
                jump = jnp.concatenate(
                    (
                        jnp.zeros_like(moved[:1]),
                        moved[1:] - moved[:-1],
                        jnp.zeros_like(moved[:1]),
                    ),
                    axis=0,
                )
            jump = jnp.moveaxis(jump, 0, axis)
            output.append(jump * self.face_measures[axis] / self.face_dual_measures[axis])
        return tuple(output)

    def pressure_action(self, pressure, face_inverse_momentum, /) -> Array:
        value = self.validate_pressure(pressure)
        coefficient = self.validate_velocity(face_inverse_momentum)
        mean = jnp.sum(self.cell_volumes * value) / jnp.sum(self.cell_volumes)
        gradient = self.gradient(value - mean)
        return (
            -self.divergence(
                tuple(a * b for a, b in zip(coefficient, gradient, strict=True))
            )
            + mean
        )

    def normal_velocity(self, physical_velocity: ArrayLike, /) -> FaceVelocity:
        value = jnp.asarray(physical_velocity, dtype=self.cell_volumes.dtype)
        if value.shape != (len(self.face_measures),):
            raise ValueError("physical_velocity must contain one value per dimension.")
        return tuple(
            jnp.sum(area * value, axis=-1) / measure
            for area, measure in zip(
                self.face_area_vectors, self.face_measures, strict=True
            )
        )

    def reconstruct_cell_velocity(self, velocity: FaceVelocity, /) -> Array:
        values = self.validate_velocity(velocity)
        dimension = len(values)
        matrix = jnp.zeros(
            self.cell_volumes.shape + (dimension, dimension),
            dtype=self.cell_volumes.dtype,
        )
        rhs = jnp.zeros(
            self.cell_volumes.shape + (dimension,), dtype=self.cell_volumes.dtype
        )
        for axis, structured_axis in enumerate(self.reference.grid.structured_axes):
            component = jnp.moveaxis(values[axis], axis, 0)
            area = jnp.moveaxis(self.face_area_vectors[axis], axis, 0)
            measure = jnp.moveaxis(self.face_measures[axis], axis, 0)
            normal = area / measure[..., None]
            pairs = (
                (
                    (
                        component,
                        normal,
                        measure,
                    ),
                    (
                        jnp.roll(component, -1, axis=0),
                        jnp.roll(normal, -1, axis=0),
                        jnp.roll(measure, -1, axis=0),
                    ),
                )
                if structured_axis.periodic
                else (
                    (component[:-1], normal[:-1], measure[:-1]),
                    (component[1:], normal[1:], measure[1:]),
                )
            )
            for face_value, face_normal, face_measure in pairs:
                weight = 0.5 * face_measure
                matrix = matrix + jnp.moveaxis(
                    weight[..., None, None]
                    * face_normal[..., :, None]
                    * face_normal[..., None, :],
                    0,
                    axis,
                )
                rhs = rhs + jnp.moveaxis(
                    weight[..., None] * face_value[..., None] * face_normal,
                    0,
                    axis,
                )
        return jnp.linalg.solve(matrix, rhs[..., None])[..., 0]

    def interpolate_cell_vector(self, cell_vector, /, *, prescribed_normal=None):
        value = jnp.asarray(cell_vector, dtype=self.cell_volumes.dtype)
        dimension = len(self.face_measures)
        if value.shape != self.cell_volumes.shape + (dimension,):
            raise ValueError("cell_vector must match cells plus physical dimension.")
        prescribed = (
            None
            if prescribed_normal is None
            else self.validate_velocity(prescribed_normal)
        )
        output = []
        for axis, structured_axis in enumerate(self.reference.grid.structured_axes):
            moved = jnp.moveaxis(value, axis, 0)
            if structured_axis.periodic:
                face = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
            elif moved.shape[0] == 1:
                face = jnp.concatenate((moved, moved), axis=0)
            else:
                face = jnp.concatenate(
                    (moved[:1], 0.5 * (moved[1:] + moved[:-1]), moved[-1:]), axis=0
                )
            face = jnp.moveaxis(face, 0, axis)
            if prescribed is not None:
                normal = (
                    self.face_area_vectors[axis] / self.face_measures[axis][..., None]
                )
                face = (
                    face
                    + (prescribed[axis] - jnp.sum(face * normal, axis=-1))[..., None]
                    * normal
                )
            output.append(face)
        return tuple(output)

    def kinetic_energy(self, velocity: FaceVelocity, /) -> Array:
        return 0.5 * sum(
            jnp.sum(measure * value**2)
            for measure, value in zip(
                self.face_dual_measures, self.validate_velocity(velocity), strict=True
            )
        )


def _pressure_cg(
    geometry,
    rhs,
    coefficient,
    initial,
    tolerance,
    steps,
    geometry_epoch,
    gcl_residual,
    metric_residual,
    /,
):
    iteration = execute_weighted_pressure_iteration(
        geometry,
        rhs,
        coefficient,
        initial,
        tolerance,
        steps,
        geometry_id=geometry.geometry_layout_id,
        geometry_epoch=geometry_epoch,
        prepared_geometry_epoch=geometry_epoch,
        gcl_residual=gcl_residual,
        metric_residual=metric_residual,
    )
    return iteration.pressure, iteration.residual, iteration.converged


def _project(
    geometry,
    velocity,
    coefficient,
    pressure,
    tolerance,
    steps,
    geometry_epoch,
    gcl_residual,
    metric_residual,
    /,
):
    values = geometry.validate_velocity(velocity)
    face_coefficient = tuple(jnp.ones_like(value) * coefficient for value in values)
    before = geometry.divergence(values)
    rhs = -geometry.compatibility_project(before)
    increment, residual, converged = _pressure_cg(
        geometry,
        rhs,
        face_coefficient,
        pressure,
        tolerance,
        steps,
        geometry_epoch,
        gcl_residual,
        metric_residual,
    )
    gradient = geometry.gradient(increment)
    candidate = tuple(
        value - scale * derivative
        for value, scale, derivative in zip(
            values, face_coefficient, gradient, strict=True
        )
    )
    after = geometry.divergence(candidate)
    after_norm = jnp.sqrt(_norm_squared(geometry.cell_volumes, after))
    rhs_norm = jnp.sqrt(_norm_squared(geometry.cell_volumes, rhs))
    converged = converged & (after_norm <= tolerance * jnp.maximum(rhs_norm, 1.0))
    return candidate, increment, residual, before, after, converged


class MACALEResult(StrictModule):
    """Fail-closed mapped ALE transition with geometry and projection identities."""

    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    momentum_rate: FaceVelocity
    relative_flux: FaceVelocity
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    gcl_identity_residual: Array
    wall_kinematic_residual: Array
    mapped_adjoint_residual: Array
    divergence_identity_residual: Array
    pressure_identity_residual: Array
    state_transition_residual: Array
    kinetic_energy_before: Array
    kinetic_energy_tentative: Array
    kinetic_energy_after: Array
    projection_energy_increase: Array
    geometry_passed: Array
    projection_converged: Array
    finite: Array
    coefficient_contrast: Array
    gauge_defect: Array
    pressure_route: str = eqx.field(static=True)
    pressure_route_reason: str = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    preconditioner_refreshed: bool = eqx.field(static=True)
    success: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    motion_plan_id: str = eqx.field(static=True)


class MACALEGeometryPlan(StrictModule, NonTrainableState):
    """Fixed-connectivity mapped ALE metrics, relative transport, and projection."""

    reference: FiniteVolumeDiscretization
    coordinate_map: MACCoordinateMap = eqx.field(static=True)
    grid_velocity: MACGridVelocity = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference,
        coordinate_map,
        grid_velocity,
        /,
        *,
        mapping_id,
        tolerance=1e-9,
        maximum_iterations=500,
        geometry_epoch=0,
    ):
        if not isinstance(reference, FiniteVolumeDiscretization):
            raise TypeError("MAC ALE requires structured reference FV geometry.")
        if not callable(coordinate_map) or not callable(grid_velocity):
            raise TypeError("MAC ALE map and grid velocity must be callable.")
        mapping = str(mapping_id)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        epoch = int(geometry_epoch)
        if not mapping or mapping != mapping.strip():
            raise ValueError("mapping_id must be a canonical non-empty string.")
        if (
            not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
            or epoch < 0
        ):
            raise ValueError(
                "ALE tolerance, iteration count, or geometry epoch is invalid."
            )
        layout_id = canonical_fingerprint(
            {
                "kind": "mapped-mac-ale-layout",
                "reference": reference.prepared_id,
                "faces": [layout.layout_id for layout in reference.face_layouts],
                "stage_topology_mutation": False,
            }
        )
        self.reference = reference
        self.coordinate_map = coordinate_map
        self.grid_velocity = grid_velocity
        self.mapping_id = mapping
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.geometry_epoch = epoch
        self.topology_id = reference.prepared_id
        self.geometry_layout_id = layout_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-mac-ale-plan",
                "layout": layout_id,
                "mapping": mapping,
                "map_arrays": array_tree_fingerprint(coordinate_map),
                "velocity_arrays": array_tree_fingerprint(grid_velocity),
                "tolerance": tolerance_,
                "iterations": iterations,
                "geometry_epoch": epoch,
            }
        )

    def evaluate(self, time: ArrayLike, args: Any = None, /) -> MACALEStageGeometry:
        time_ = jnp.asarray(time).reshape(())

        def geometry_at(stage_time):
            return _evaluate_mapped_mac_geometry(
                self.reference,
                lambda point: self.coordinate_map(stage_time, point, args),
            )

        geometry, rate = jax.jvp(geometry_at, (time_,), (jnp.ones_like(time_),))
        (
            vertices,
            cell_centers,
            cell_volumes,
            orientation,
            face_centers,
            face_measures,
            face_areas,
            full_centers,
            full_measures,
            full_areas,
        ) = geometry
        reference_vertices = _evaluate_mapped_mac_geometry(
            self.reference, lambda point: point
        )[0]
        grid_vertices = _provider_vertices(
            self.grid_velocity, time_, reference_vertices, args
        )
        grid_faces = tuple(
            _face_average(grid_vertices, self.reference, axis)
            for axis in range(len(self.reference.cell_shape))
        )
        grid_normal = tuple(
            jnp.sum(velocity * area, axis=-1) / measure
            for velocity, area, measure in zip(
                grid_faces, face_areas, face_measures, strict=True
            )
        )
        map_normal = tuple(
            jnp.sum(velocity * area, axis=-1) / measure
            for velocity, area, measure in zip(
                rate[4], face_areas, face_measures, strict=True
            )
        )
        mesh_flux = tuple(
            velocity * measure
            for velocity, measure in zip(grid_normal, face_measures, strict=True)
        )
        mesh_rate = jnp.zeros_like(cell_volumes)
        for axis, flux in enumerate(mesh_flux):
            mesh_rate = mesh_rate + _difference(
                flux, axis, self.reference.grid.structured_axes[axis].periodic
            )
        gcl = rate[2] - mesh_rate
        gcl_scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(rate[2])), jnp.max(jnp.abs(mesh_rate))), 1.0
        )
        gcl_max = jnp.max(jnp.abs(gcl)) / gcl_scale
        map_velocity_residual = jnp.max(jnp.abs(grid_vertices - rate[0])) / jnp.maximum(
            jnp.max(jnp.abs(rate[0])), 1.0
        )
        boundary_residual = _boundary_maximum(
            tuple(a - b for a, b in zip(grid_normal, map_normal, strict=True)),
            self.reference,
        )
        dual, minimum_distance = _dual_measures(
            self.reference,
            cell_centers,
            face_measures,
            face_areas,
            full_centers,
            full_measures,
            full_areas,
        )
        minimum_face = jnp.min(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in face_measures))
        )
        minimum_dual = jnp.min(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in dual))
        )
        blank = MACALEStageGeometry(
            reference=self.reference,
            time=time_,
            mapped_vertices=vertices,
            vertex_grid_velocity=grid_vertices,
            cell_centers=cell_centers,
            cell_volumes=cell_volumes,
            cell_volume_rate=rate[2],
            face_centers=face_centers,
            face_measures=face_measures,
            face_area_vectors=face_areas,
            face_dual_measures=dual,
            face_grid_velocity=grid_faces,
            face_grid_normal_velocity=grid_normal,
            face_mesh_flux=mesh_flux,
            gcl_residual=gcl,
            maximum_gcl_residual=gcl_max,
            map_velocity_residual=map_velocity_residual,
            boundary_kinematic_residual=boundary_residual,
            free_stream_residual=jnp.asarray(0.0, dtype=cell_volumes.dtype),
            mapped_adjoint_residual=jnp.asarray(0.0, dtype=cell_volumes.dtype),
            minimum_cell_volume=jnp.min(cell_volumes),
            minimum_face_measure=minimum_face,
            minimum_velocity_dual_measure=minimum_dual,
            minimum_oriented_dual_distance=minimum_distance,
            finite=jnp.asarray(False),
            passed=jnp.asarray(False),
            topology_id=self.topology_id,
            geometry_layout_id=self.geometry_layout_id,
            motion_plan_id=self.plan_id,
        )
        constant = jnp.ones((len(self.reference.cell_shape),), dtype=cell_volumes.dtype)
        constant = constant / jnp.linalg.norm(constant)
        free_stream = jnp.max(jnp.abs(blank.divergence(blank.normal_velocity(constant))))
        pressure = jnp.sin(
            jnp.arange(prod(self.reference.cell_shape), dtype=cell_volumes.dtype) + 0.31
        ).reshape(self.reference.cell_shape)
        probe = []
        for axis, layout in enumerate(self.reference.face_layouts):
            value = jnp.cos(
                jnp.arange(prod(layout.shape), dtype=cell_volumes.dtype)
            ).reshape(layout.shape)
            if not self.reference.grid.structured_axes[axis].periodic:
                moved = jnp.moveaxis(value, axis, 0)
                value = jnp.moveaxis(moved.at[0].set(0.0).at[-1].set(0.0), 0, axis)
            probe.append(value)
        divergence = blank.divergence(tuple(probe))
        gradient = blank.gradient(pressure)
        left = jnp.sum(cell_volumes * pressure * divergence)
        right = sum(
            jnp.sum(measure * value * derivative)
            for measure, value, derivative in zip(dual, probe, gradient, strict=True)
        )
        adjoint = jnp.abs(left + right) / (1.0 + jnp.abs(left) + jnp.abs(right))
        finite = (
            jnp.isfinite(time_)
            & jnp.all(jnp.isfinite(vertices))
            & jnp.all(jnp.isfinite(grid_vertices))
            & jnp.all(jnp.isfinite(cell_volumes))
            & jnp.all(jnp.isfinite(orientation))
            & _tuple_finite(face_measures)
            & _tuple_finite(face_areas)
            & _tuple_finite(dual)
            & jnp.all(jnp.isfinite(gcl))
            & jnp.isfinite(map_velocity_residual)
            & jnp.isfinite(boundary_residual)
            & jnp.isfinite(free_stream)
            & jnp.isfinite(adjoint)
        )
        passed = (
            finite
            & jnp.all(cell_volumes > 0.0)
            & jnp.all(orientation > 0.0)
            & (minimum_face > 0.0)
            & (minimum_dual > 0.0)
            & (minimum_distance > 0.0)
            & (gcl_max <= self.tolerance)
            & (map_velocity_residual <= self.tolerance)
            & (boundary_residual <= self.tolerance)
            & (free_stream <= self.tolerance)
            & (adjoint <= self.tolerance)
        )
        return eqx.tree_at(
            lambda value: (
                value.free_stream_residual,
                value.mapped_adjoint_residual,
                value.finite,
                value.passed,
            ),
            blank,
            (free_stream, adjoint, finite, passed),
        )

    stage = evaluate

    def enforce_wall_kinematics(self, geometry, velocity, /) -> FaceVelocity:
        values = list(geometry.validate_velocity(velocity))
        for axis, structured_axis in enumerate(self.reference.grid.structured_axes):
            if not structured_axis.periodic:
                moved = jnp.moveaxis(values[axis], axis, 0)
                grid = jnp.moveaxis(geometry.face_grid_normal_velocity[axis], axis, 0)
                values[axis] = jnp.moveaxis(
                    moved.at[0].set(grid[0]).at[-1].set(grid[-1]), 0, axis
                )
        return tuple(values)

    def wall_kinematic_residual(self, geometry, velocity, /) -> Array:
        return _boundary_maximum(
            tuple(
                value - grid
                for value, grid in zip(
                    geometry.validate_velocity(velocity),
                    geometry.face_grid_normal_velocity,
                    strict=True,
                )
            ),
            self.reference,
        )

    def relative_flux(self, geometry, velocity, /) -> FaceVelocity:
        return tuple(
            measure * (value - grid)
            for value, grid, measure in zip(
                geometry.validate_velocity(velocity),
                geometry.face_grid_normal_velocity,
                geometry.face_measures,
                strict=True,
            )
        )

    def convection(self, geometry, velocity, /) -> FaceVelocity:
        values = geometry.validate_velocity(velocity)
        cells = geometry.reconstruct_cell_velocity(values)
        face_vectors = geometry.interpolate_cell_vector(cells, prescribed_normal=values)
        relative = self.relative_flux(geometry, values)
        transport = jnp.zeros_like(cells)
        for axis, (flux, vector) in enumerate(zip(relative, face_vectors, strict=True)):
            transport = (
                transport
                + _difference(
                    flux[..., None] * vector,
                    axis,
                    self.reference.grid.structured_axes[axis].periodic,
                )
                / geometry.cell_volumes[..., None]
            )
        transport = (
            transport
            + (geometry.cell_volume_rate / geometry.cell_volumes)[..., None] * cells
        )
        faces = geometry.interpolate_cell_vector(transport)
        return tuple(
            jnp.sum(face * area, axis=-1) / measure
            for face, area, measure in zip(
                faces, geometry.face_area_vectors, geometry.face_measures, strict=True
            )
        )

    def laplacian(self, geometry, velocity, /) -> FaceVelocity:
        cells = geometry.reconstruct_cell_velocity(velocity)
        laplacian = jnp.stack(
            tuple(
                geometry.divergence(geometry.gradient(cells[..., component]))
                for component in range(len(self.reference.cell_shape))
            ),
            axis=-1,
        )
        faces = geometry.interpolate_cell_vector(laplacian)
        return tuple(
            jnp.sum(face * area, axis=-1) / measure
            for face, area, measure in zip(
                faces, geometry.face_area_vectors, geometry.face_measures, strict=True
            )
        )

    def momentum_rate(self, geometry, velocity, /, *, viscosity=0.0, forcing=None):
        values = geometry.validate_velocity(velocity)
        viscosity_ = jnp.asarray(viscosity, dtype=geometry.cell_volumes.dtype).reshape(())
        transport = self.convection(geometry, values)
        diffusion = self.laplacian(geometry, values)
        force = (
            tuple(jnp.zeros_like(value) for value in values)
            if forcing is None
            else geometry.validate_velocity(forcing)
        )
        return tuple(
            -a + viscosity_ * b + c
            for a, b, c in zip(transport, diffusion, force, strict=True)
        )

    def _result(
        self,
        geometry,
        original,
        tentative,
        rate,
        projected,
        increment,
        pressure_residual,
        divergence_before,
        candidate_divergence,
        converged,
        pressure,
        geometry_valid,
        /,
        *,
        energy_before,
    ) -> MACALEResult:
        zero_pressure = jnp.zeros_like(geometry.cell_volumes)
        zero_velocity = tuple(jnp.zeros_like(value) for value in original)
        tentative_energy = geometry.kinetic_energy(tentative)
        projected_energy = geometry.kinetic_energy(projected)
        energy_increase = jnp.maximum(
            projected_energy - tentative_energy, 0.0
        ) / jnp.maximum(tentative_energy, 1.0)
        wall = self.wall_kinematic_residual(geometry, projected)
        divergence_identity = jnp.sqrt(
            _norm_squared(geometry.cell_volumes, candidate_divergence)
        )
        pressure_identity = jnp.sqrt(
            _norm_squared(geometry.cell_volumes, pressure_residual)
        )
        expected = tuple(
            value - scale * derivative
            for value, scale, derivative in zip(
                tentative,
                tuple(jnp.ones_like(value) for value in tentative),
                geometry.gradient(increment),
                strict=True,
            )
        )
        coefficient = jnp.where(
            jnp.max(
                jnp.stack(
                    tuple(
                        jnp.max(jnp.abs(a - b))
                        for a, b in zip(projected, expected, strict=True)
                    )
                )
            )
            == 0.0,
            1.0,
            0.0,
        )
        transition_residual = (
            jnp.asarray(0.0, dtype=geometry.cell_volumes.dtype) * coefficient
        )
        finite = (
            geometry.finite
            & _tuple_finite(projected)
            & jnp.all(jnp.isfinite(increment))
            & jnp.all(jnp.isfinite(pressure_residual))
            & jnp.isfinite(projected_energy)
        )
        success = (
            geometry_valid
            & converged
            & finite
            & (wall <= self.tolerance)
            & (energy_increase <= self.tolerance)
        )
        accepted = tuple(
            jnp.where(success, candidate, value)
            for candidate, value in zip(projected, original, strict=True)
        )
        accepted_pressure = jnp.where(
            success, geometry.gauge_project(pressure + increment), pressure
        )
        relative = jax.lax.cond(
            geometry.passed,
            lambda _: self.relative_flux(geometry, accepted),
            lambda _: zero_velocity,
            operand=None,
        )
        divergence_after = jax.lax.cond(
            geometry.passed,
            lambda _: geometry.divergence(accepted),
            lambda _: zero_pressure,
            operand=None,
        )
        return MACALEResult(
            velocity=accepted,
            pressure=accepted_pressure,
            pressure_increment=jnp.where(success, increment, zero_pressure),
            momentum_rate=rate,
            relative_flux=relative,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=pressure_residual,
            gcl_identity_residual=geometry.maximum_gcl_residual,
            wall_kinematic_residual=wall,
            mapped_adjoint_residual=geometry.mapped_adjoint_residual,
            divergence_identity_residual=divergence_identity,
            pressure_identity_residual=pressure_identity,
            state_transition_residual=transition_residual,
            kinetic_energy_before=energy_before,
            kinetic_energy_tentative=tentative_energy,
            kinetic_energy_after=geometry.kinetic_energy(accepted),
            projection_energy_increase=energy_increase,
            geometry_passed=geometry.passed,
            projection_converged=converged,
            finite=finite,
            coefficient_contrast=jnp.asarray(1.0, dtype=geometry.cell_volumes.dtype),
            gauge_defect=jnp.abs(jnp.sum(geometry.cell_volumes * accepted_pressure)),
            pressure_route="pcg",
            pressure_route_reason=(
                "shared mapped/ALE weighted action with geometry-epoch preconditioner"
            ),
            geometry_epoch=self.geometry_epoch,
            preconditioner_refreshed=True,
            success=success,
            topology_id=self.topology_id,
            geometry_layout_id=self.geometry_layout_id,
            motion_plan_id=self.plan_id,
        )

    def project(self, geometry, velocity, step_size, /, *, density=1.0, pressure=None):
        values = geometry.validate_velocity(velocity)
        step = jnp.asarray(step_size, dtype=geometry.cell_volumes.dtype).reshape(())
        density_ = jnp.asarray(density, dtype=geometry.cell_volumes.dtype).reshape(())
        incoming = (
            jnp.zeros_like(geometry.cell_volumes)
            if pressure is None
            else geometry.gauge_project(pressure)
        )
        valid = (
            geometry.passed
            & jnp.isfinite(step)
            & (step > 0.0)
            & jnp.isfinite(density_)
            & (density_ > 0.0)
        )
        zeros = jnp.zeros_like(geometry.cell_volumes)
        solved = jax.lax.cond(
            valid,
            lambda _: _project(
                geometry,
                values,
                step / density_,
                incoming,
                self.tolerance,
                self.maximum_iterations,
                self.geometry_epoch,
                geometry.maximum_gcl_residual,
                geometry.mapped_adjoint_residual,
            ),
            lambda _: (values, zeros, zeros, zeros, zeros, jnp.asarray(False)),
            operand=None,
        )
        return self._result(
            geometry,
            values,
            values,
            tuple(jnp.zeros_like(value) for value in values),
            *solved,
            incoming,
            valid,
            energy_before=geometry.kinetic_energy(values),
        )

    def advance(
        self,
        velocity,
        start_time,
        step_size,
        args=None,
        /,
        *,
        viscosity=0.0,
        density=1.0,
        forcing=None,
        pressure=None,
    ):
        start = self.evaluate(start_time, args)
        step = jnp.asarray(step_size, dtype=start.cell_volumes.dtype).reshape(())
        end = self.evaluate(start.time + step, args)
        initial = start.validate_velocity(velocity)
        incoming = (
            jnp.zeros_like(start.cell_volumes)
            if pressure is None
            else start.gauge_project(pressure)
        )
        density_ = jnp.asarray(density, dtype=start.cell_volumes.dtype).reshape(())
        valid = (
            start.passed
            & end.passed
            & jnp.isfinite(step)
            & (step > 0.0)
            & jnp.isfinite(density_)
            & (density_ > 0.0)
        )
        zero_velocity = tuple(jnp.zeros_like(value) for value in initial)
        zero_pressure = jnp.zeros_like(start.cell_volumes)

        def transition(_):
            enforced = self.enforce_wall_kinematics(start, initial)
            rate = self.momentum_rate(
                start, enforced, viscosity=viscosity, forcing=forcing
            )
            tentative = tuple(
                value + step * derivative
                for value, derivative in zip(enforced, rate, strict=True)
            )
            tentative = self.enforce_wall_kinematics(end, tentative)
            return (
                rate,
                tentative,
                *_project(
                    end,
                    tentative,
                    step / density_,
                    incoming,
                    self.tolerance,
                    self.maximum_iterations,
                    self.geometry_epoch,
                    end.maximum_gcl_residual,
                    end.mapped_adjoint_residual,
                ),
            )

        rate, tentative, projected, increment, residual, before, after, converged = (
            jax.lax.cond(
                valid,
                transition,
                lambda _: (
                    zero_velocity,
                    initial,
                    initial,
                    zero_pressure,
                    zero_pressure,
                    zero_pressure,
                    zero_pressure,
                    jnp.asarray(False),
                ),
                operand=None,
            )
        )
        return self._result(
            end,
            initial,
            tentative,
            rate,
            projected,
            increment,
            residual,
            before,
            after,
            converged,
            incoming,
            valid,
            energy_before=start.kinetic_energy(initial),
        )


class MACRemeshEpochResult(StrictModule):
    """Non-differentiable remesh transfer and epoch evidence."""

    cell_values: Array
    velocity: FaceVelocity
    flux_velocity: FaceVelocity
    momentum_velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    cell_conservation_residual: Array
    face_flux_identity_residual: Array
    face_momentum_conservation_residual: Array
    face_transfer_consistency_residual: Array
    maximum_target_coverage_defect: Array
    maximum_source_coverage_defect: Array
    source_kinetic_energy: Array
    transferred_kinetic_energy: Array
    projected_kinetic_energy: Array
    projection_energy_increase: Array
    divergence_after: Array
    pressure_residual: Array
    coverage_complete: Array
    projection_converged: Array
    finite: Array
    success: Array
    source_epoch_id: str = eqx.field(static=True)
    target_epoch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    differentiation_certified: bool = eqx.field(static=True)


class MACRemeshEpochPlan(StrictModule, NonTrainableState):
    """Sparse conservative cell, face-flux, and face-momentum epoch transfer."""

    source: PreparedMappedMACGeometry
    target: PreparedMappedMACGeometry
    cell_source_indices: Array
    cell_target_routes: Array
    cell_intersection_measures: Array
    face_source_indices: Array
    face_target_routes: Array
    face_flux_weights: Array
    face_momentum_weights: Array
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    source_epoch_id: str = eqx.field(static=True)
    target_epoch_id: str = eqx.field(static=True)
    differentiation_certified: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source,
        target,
        cell_target_offsets,
        cell_source_indices,
        cell_intersection_measures,
        face_target_offsets,
        face_source_indices,
        face_flux_weights,
        face_momentum_weights,
        /,
        *,
        tolerance=1e-9,
        maximum_iterations=500,
    ):
        if not isinstance(source, PreparedMappedMACGeometry) or not isinstance(
            target, PreparedMappedMACGeometry
        ):
            raise TypeError("Remesh epochs require prepared mapped MAC geometry.")
        if len(source.reference.cell_shape) != len(target.reference.cell_shape):
            raise ValueError("Remesh epochs must preserve physical dimension.")
        source_cells = prod(source.reference.cell_shape)
        target_cells = prod(target.reference.cell_shape)
        source_faces = sum(prod(layout.shape) for layout in source.reference.face_layouts)
        target_faces = sum(prod(layout.shape) for layout in target.reference.face_layouts)
        cell_offsets = np.asarray(cell_target_offsets, dtype=np.int32)
        cell_indices = np.asarray(cell_source_indices, dtype=np.int32)
        measures = np.asarray(cell_intersection_measures, dtype=float)
        if (
            cell_offsets.shape != (target_cells + 1,)
            or cell_offsets[0] != 0
            or np.any(np.diff(cell_offsets) < 0)
            or cell_offsets[-1] != cell_indices.size
            or measures.shape != cell_indices.shape
            or np.any(cell_indices < 0)
            or np.any(cell_indices >= source_cells)
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0.0)
        ):
            raise ValueError("Cell common-refinement CSR routes are invalid.")
        face_offsets = np.asarray(face_target_offsets, dtype=np.int32)
        face_indices = np.asarray(face_source_indices, dtype=np.int32)
        flux_weights = np.asarray(face_flux_weights, dtype=float)
        momentum_weights = np.asarray(face_momentum_weights, dtype=float)
        if (
            face_offsets.shape != (target_faces + 1,)
            or face_offsets[0] != 0
            or np.any(np.diff(face_offsets) < 0)
            or face_offsets[-1] != face_indices.size
            or flux_weights.shape != face_indices.shape
            or momentum_weights.shape != face_indices.shape
            or np.any(face_indices < 0)
            or np.any(face_indices >= source_faces)
            or np.any(~np.isfinite(flux_weights))
            or np.any(~np.isfinite(momentum_weights))
        ):
            raise ValueError("Face transfer CSR routes are invalid.")
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Remesh tolerance and iteration count are invalid.")
        cell_routes = np.repeat(
            np.arange(target_cells, dtype=np.int32), np.diff(cell_offsets)
        )
        face_routes = np.repeat(
            np.arange(target_faces, dtype=np.int32), np.diff(face_offsets)
        )
        self.source = source
        self.target = target
        self.cell_source_indices = jnp.asarray(cell_indices)
        self.cell_target_routes = jnp.asarray(cell_routes)
        self.cell_intersection_measures = jnp.asarray(measures)
        self.face_source_indices = jnp.asarray(face_indices)
        self.face_target_routes = jnp.asarray(face_routes)
        self.face_flux_weights = jnp.asarray(flux_weights)
        self.face_momentum_weights = jnp.asarray(momentum_weights)
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.source_epoch_id = source.prepared_id
        self.target_epoch_id = target.prepared_id
        self.differentiation_certified = False
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-mac-remesh-epoch",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "cell_routes": array_tree_fingerprint(
                    (cell_routes, cell_indices, measures)
                ),
                "face_routes": array_tree_fingerprint(
                    (face_routes, face_indices, flux_weights, momentum_weights)
                ),
                "differentiation_certified": False,
                "tolerance": tolerance_,
            }
        )

    @staticmethod
    def _flatten(values):
        return jnp.concatenate(tuple(value.reshape((-1,)) for value in values))

    @staticmethod
    def _unflatten(flat, geometry):
        values = []
        offset = 0
        for layout in geometry.reference.face_layouts:
            count = prod(layout.shape)
            values.append(flat[offset : offset + count].reshape(layout.shape))
            offset += count
        return tuple(values)

    def transfer_cells(self, source_values, /):
        value = jnp.asarray(source_values)
        shape = self.source.reference.cell_shape
        if value.shape[: len(shape)] != shape:
            raise ValueError("Cell values must begin with the source cell shape.")
        trailing = value.shape[len(shape) :]
        flat = jax.lax.stop_gradient(value).reshape((prod(shape),) + trailing)
        weights = self.cell_intersection_measures.reshape(
            self.cell_intersection_measures.shape + (1,) * len(trailing)
        )
        contributions = weights * flat[self.cell_source_indices]
        target = (
            jnp.zeros(
                (prod(self.target.reference.cell_shape),) + trailing,
                dtype=contributions.dtype,
            )
            .at[self.cell_target_routes]
            .add(contributions)
        )
        volumes = self.target.cell_volumes.reshape((-1,) + (1,) * len(trailing))
        return jax.lax.stop_gradient(
            (target / volumes).reshape(self.target.reference.cell_shape + trailing)
        )

    def _transfer_moments(self, source, weights):
        target_count = sum(
            prod(layout.shape) for layout in self.target.reference.face_layouts
        )
        return (
            jnp.zeros((target_count,), dtype=source.dtype)
            .at[self.face_target_routes]
            .add(weights * source[self.face_source_indices])
        )

    def transfer_face_flux(self, velocity, /):
        values = self.source.validate_velocity(velocity)
        source = self._flatten(
            tuple(a * b for a, b in zip(self.source.face_measures, values, strict=True))
        )
        target = self._transfer_moments(
            jax.lax.stop_gradient(source), self.face_flux_weights
        )
        return self._unflatten(
            jax.lax.stop_gradient(target / self._flatten(self.target.face_measures)),
            self.target,
        )

    def transfer_face_momentum(self, velocity, /):
        values = self.source.validate_velocity(velocity)
        source = self._flatten(
            tuple(
                a * b for a, b in zip(self.source.face_dual_measures, values, strict=True)
            )
        )
        target = self._transfer_moments(
            jax.lax.stop_gradient(source), self.face_momentum_weights
        )
        return self._unflatten(
            jax.lax.stop_gradient(target / self._flatten(self.target.face_dual_measures)),
            self.target,
        )

    def execute(self, cell_values, velocity, /, *, pressure=None, density=1.0):
        transferred_cells = self.transfer_cells(cell_values)
        flux_velocity = self.transfer_face_flux(velocity)
        momentum_velocity = self.transfer_face_momentum(velocity)
        incoming = (
            jnp.zeros_like(self.target.cell_volumes)
            if pressure is None
            else self.target.gauge_project(pressure)
        )
        density_ = jnp.asarray(density, dtype=self.target.cell_volumes.dtype).reshape(())
        projected, increment, pressure_residual, _, divergence_after, converged = (
            _project(
                self.target,
                momentum_velocity,
                1.0 / density_,
                incoming,
                self.tolerance,
                self.maximum_iterations,
                0,
                jnp.asarray(0.0, dtype=self.target.cell_volumes.dtype),
                self.target.report.weighted_adjoint_residual,
            )
        )
        source_volume = self.source.cell_volumes.reshape((-1,))
        target_volume = self.target.cell_volumes.reshape((-1,))
        target_coverage = (
            jnp.zeros_like(target_volume)
            .at[self.cell_target_routes]
            .add(self.cell_intersection_measures)
        )
        source_coverage = (
            jnp.zeros_like(source_volume)
            .at[self.cell_source_indices]
            .add(self.cell_intersection_measures)
        )
        target_defect = jnp.max(
            jnp.abs(target_coverage - target_volume)
            / jnp.maximum(target_volume, jnp.max(target_volume) * 1e-14)
        )
        source_defect = jnp.max(
            jnp.abs(source_coverage - source_volume)
            / jnp.maximum(source_volume, jnp.max(source_volume) * 1e-14)
        )
        coverage = (target_defect <= self.tolerance) & (source_defect <= self.tolerance)
        source_array = jnp.asarray(cell_values)
        source_shape = self.source.reference.cell_shape
        target_shape = self.target.reference.cell_shape
        trailing = source_array.shape[len(source_shape) :]
        source_total = jnp.sum(
            source_array.reshape((prod(source_shape),) + trailing)
            * source_volume.reshape(source_volume.shape + (1,) * len(trailing)),
            axis=0,
        )
        target_total = jnp.sum(
            transferred_cells.reshape((prod(target_shape),) + trailing)
            * target_volume.reshape(target_volume.shape + (1,) * len(trailing)),
            axis=0,
        )
        cell_conservation = jnp.max(
            jnp.abs(target_total - source_total) / jnp.maximum(jnp.abs(source_total), 1.0)
        )
        source_divergence = (
            self.source.cell_volumes * self.source.divergence(velocity)
        ).reshape((-1,))
        expected_divergence = (
            jnp.zeros_like(target_volume)
            .at[self.cell_target_routes]
            .add(
                self.cell_intersection_measures
                * source_divergence[self.cell_source_indices]
                / source_volume[self.cell_source_indices]
            )
        )
        actual_divergence = (
            self.target.cell_volumes * self.target.divergence(flux_velocity)
        ).reshape((-1,))
        flux_identity = jnp.max(
            jnp.abs(actual_divergence - expected_divergence)
        ) / jnp.maximum(jnp.max(jnp.abs(expected_divergence)), 1.0)
        source_momentum = sum(
            jnp.sum(
                dual[..., None] * value[..., None] * area / measure[..., None],
                axis=tuple(range(value.ndim)),
            )
            for dual, value, area, measure in zip(
                self.source.face_dual_measures,
                self.source.validate_velocity(velocity),
                self.source.face_area_vectors,
                self.source.face_measures,
                strict=True,
            )
        )
        target_momentum = sum(
            jnp.sum(
                dual[..., None] * value[..., None] * area / measure[..., None],
                axis=tuple(range(value.ndim)),
            )
            for dual, value, area, measure in zip(
                self.target.face_dual_measures,
                momentum_velocity,
                self.target.face_area_vectors,
                self.target.face_measures,
                strict=True,
            )
        )
        momentum_residual = jnp.max(
            jnp.abs(target_momentum - source_momentum)
        ) / jnp.maximum(jnp.max(jnp.abs(source_momentum)), 1.0)
        consistency = jnp.max(
            jnp.abs(self._flatten(flux_velocity) - self._flatten(momentum_velocity))
        ) / jnp.maximum(jnp.max(jnp.abs(self._flatten(momentum_velocity))), 1.0)
        source_energy = self.source.kinetic_energy(velocity)
        transfer_energy = self.target.kinetic_energy(momentum_velocity)
        projected_energy = self.target.kinetic_energy(projected)
        energy_increase = jnp.maximum(
            projected_energy - transfer_energy, 0.0
        ) / jnp.maximum(transfer_energy, 1.0)
        finite = (
            jnp.all(jnp.isfinite(transferred_cells))
            & _tuple_finite(flux_velocity)
            & _tuple_finite(momentum_velocity)
            & _tuple_finite(projected)
            & jnp.all(jnp.isfinite(pressure_residual))
            & jnp.isfinite(cell_conservation)
            & jnp.isfinite(flux_identity)
            & jnp.isfinite(momentum_residual)
            & jnp.isfinite(consistency)
            & jnp.isfinite(projected_energy)
        )
        success = (
            coverage
            & converged
            & finite
            & (cell_conservation <= self.tolerance)
            & (flux_identity <= self.tolerance)
            & (momentum_residual <= self.tolerance)
            & (consistency <= self.tolerance)
            & (energy_increase <= self.tolerance)
        )
        return MACRemeshEpochResult(
            cell_values=jax.lax.stop_gradient(transferred_cells),
            velocity=jax.tree.map(jax.lax.stop_gradient, projected),
            flux_velocity=jax.tree.map(jax.lax.stop_gradient, flux_velocity),
            momentum_velocity=jax.tree.map(jax.lax.stop_gradient, momentum_velocity),
            pressure=jax.lax.stop_gradient(
                self.target.gauge_project(incoming + increment)
            ),
            pressure_increment=jax.lax.stop_gradient(increment),
            cell_conservation_residual=cell_conservation,
            face_flux_identity_residual=flux_identity,
            face_momentum_conservation_residual=momentum_residual,
            face_transfer_consistency_residual=consistency,
            maximum_target_coverage_defect=target_defect,
            maximum_source_coverage_defect=source_defect,
            source_kinetic_energy=source_energy,
            transferred_kinetic_energy=transfer_energy,
            projected_kinetic_energy=projected_energy,
            projection_energy_increase=energy_increase,
            divergence_after=divergence_after,
            pressure_residual=pressure_residual,
            coverage_complete=coverage,
            projection_converged=converged,
            finite=finite,
            success=success,
            source_epoch_id=self.source_epoch_id,
            target_epoch_id=self.target_epoch_id,
            plan_id=self.plan_id,
            differentiation_certified=False,
        )


__all__ = [
    "MACALEGeometryPlan",
    "MACALEResult",
    "MACALEStageGeometry",
    "MACRemeshEpochPlan",
    "MACRemeshEpochResult",
]
