#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, BlockSpace, DiagonalPairing
from .._axis import broadcasted_grid
from ._incompressible import FaceVelocity
from ._structured import FiniteVolumeDiscretization


CoordinateMap = Callable[[Array], ArrayLike]


def _difference(integrated: Array, axis: int, periodic: bool, /) -> Array:
    if periodic:
        return jnp.roll(integrated, -1, axis=axis) - integrated
    lower = [slice(None)] * integrated.ndim
    upper = [slice(None)] * integrated.ndim
    lower[axis] = slice(0, integrated.shape[axis] - 1)
    upper[axis] = slice(1, integrated.shape[axis])
    return integrated[tuple(upper)] - integrated[tuple(lower)]


def _reference_vertices(reference: FiniteVolumeDiscretization, /) -> Array:
    axes = tuple(
        axis.bounds[0]
        + jnp.concatenate(
            (
                jnp.zeros((1,), dtype=axis.interval_widths.dtype),
                jnp.cumsum(axis.interval_widths),
            )
        )
        for axis in reference.grid.structured_axes
    )
    return broadcasted_grid(axes)


def _map_vertices(vertices: Array, coordinate_map: CoordinateMap, /) -> Array:
    dimension = vertices.shape[-1]
    flat = vertices.reshape((-1, dimension))
    mapped = jax.vmap(lambda point: jnp.asarray(coordinate_map(point)))(flat)
    if mapped.shape != flat.shape:
        raise ValueError(
            "coordinate_map must preserve the physical coordinate dimension."
        )
    return mapped.reshape(vertices.shape)


def _corner(
    vertices: Array, bits: tuple[int, ...], cell_shape: tuple[int, ...], /
) -> Array:
    index = tuple(slice(bit, bit + cell_shape[axis]) for axis, bit in enumerate(bits))
    return vertices[index]


def _cross2(left: Array, right: Array, /) -> Array:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _cell_geometry(
    vertices: Array, cell_shape: tuple[int, ...], /
) -> tuple[Array, Array, Array]:
    dimension = len(cell_shape)
    corners = {
        bits: _corner(vertices, bits, cell_shape)
        for bits in product((0, 1), repeat=dimension)
    }
    center = jnp.mean(jnp.stack(tuple(corners.values()), axis=0), axis=0)
    if dimension == 1:
        volume = corners[(1,)][..., 0] - corners[(0,)][..., 0]
        return center, volume, volume
    if dimension == 2:
        p00 = corners[(0, 0)]
        p10 = corners[(1, 0)]
        p11 = corners[(1, 1)]
        p01 = corners[(0, 1)]
        volume = 0.5 * (
            _cross2(p00, p10) + _cross2(p10, p11) + _cross2(p11, p01) + _cross2(p01, p00)
        )
        return center, volume, volume
    if dimension != 3:
        raise ValueError("Mapped MAC geometry supports one, two, or three dimensions.")

    def tetra(a: Array, b: Array, c: Array, d: Array) -> Array:
        return jnp.linalg.det(jnp.stack((b - a, c - a, d - a), axis=-1)) / 6.0

    p000 = corners[(0, 0, 0)]
    p100 = corners[(1, 0, 0)]
    p010 = corners[(0, 1, 0)]
    p110 = corners[(1, 1, 0)]
    p001 = corners[(0, 0, 1)]
    p101 = corners[(1, 0, 1)]
    p011 = corners[(0, 1, 1)]
    p111 = corners[(1, 1, 1)]
    parts = (
        tetra(p000, p100, p010, p001),
        tetra(p100, p110, p010, p111),
        tetra(p100, p010, p001, p111),
        tetra(p100, p001, p101, p111),
        tetra(p010, p001, p111, p011),
    )
    volume = sum(jnp.abs(part) for part in parts)
    orientation = jnp.linalg.det(
        jnp.stack(
            (
                0.25 * (p100 + p110 + p101 + p111 - p000 - p010 - p001 - p011),
                0.25 * (p010 + p110 + p011 + p111 - p000 - p100 - p001 - p101),
                0.25 * (p001 + p101 + p011 + p111 - p000 - p100 - p010 - p110),
            ),
            axis=-1,
        )
    )
    return center, volume, orientation


def _full_face_geometry(
    vertices: Array,
    reference: FiniteVolumeDiscretization,
    axis: int,
    /,
) -> tuple[Array, Array, Array]:
    dimension = len(reference.cell_shape)
    tangential = tuple(index for index in range(dimension) if index != axis)
    full_shape = list(reference.face_layouts[axis].shape)
    if reference.grid.structured_axes[axis].periodic:
        full_shape[axis] += 1

    def face_corner(bits: tuple[int, ...]) -> Array:
        bit_by_axis = {tangential[index]: bits[index] for index in range(len(tangential))}
        slices = []
        for current_axis in range(dimension):
            if current_axis == axis:
                slices.append(slice(0, full_shape[current_axis]))
            else:
                bit = bit_by_axis[current_axis]
                slices.append(slice(bit, bit + reference.cell_shape[current_axis]))
        return vertices[tuple(slices)]

    corners = tuple(face_corner(bits) for bits in product((0, 1), repeat=dimension - 1))
    center = jnp.mean(jnp.stack(corners, axis=0), axis=0)
    if dimension == 1:
        area_vector = jnp.ones(center.shape)
    elif dimension == 2:
        tangent = corners[1] - corners[0]
        area_vector = (
            jnp.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
            if axis == 0
            else jnp.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        )
    else:
        p00, p01, p10, p11 = corners
        first = jnp.cross(p10 - p00, p11 - p00)
        second = jnp.cross(p11 - p00, p01 - p00)
        area_vector = 0.5 * (first + second)
        if axis == 1:
            area_vector = -area_vector
    return center, jnp.linalg.norm(area_vector, axis=-1), area_vector


def _keep_periodic_faces(
    value: Array,
    reference: FiniteVolumeDiscretization,
    axis: int,
    /,
) -> Array:
    if not reference.grid.structured_axes[axis].periodic:
        return value
    keep = [slice(None)] * value.ndim
    keep[axis] = slice(0, reference.face_layouts[axis].shape[axis])
    return value[tuple(keep)]


def _evaluate_mapped_mac_geometry(
    reference: FiniteVolumeDiscretization,
    coordinate_map: CoordinateMap,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
]:
    """Evaluate fixed-connectivity mapped polytopes without masking invalid geometry."""
    if not isinstance(reference, FiniteVolumeDiscretization) or not callable(
        coordinate_map
    ):
        raise TypeError(
            "Mapped MAC geometry requires a reference grid and coordinate map."
        )
    vertices = _map_vertices(_reference_vertices(reference), coordinate_map)
    cell_centers, cell_volumes, cell_orientation = _cell_geometry(
        vertices, reference.cell_shape
    )
    full = tuple(
        _full_face_geometry(vertices, reference, axis)
        for axis in range(len(reference.cell_shape))
    )
    face_centers = tuple(
        _keep_periodic_faces(values[0], reference, axis)
        for axis, values in enumerate(full)
    )
    face_measures = tuple(
        _keep_periodic_faces(values[1], reference, axis)
        for axis, values in enumerate(full)
    )
    face_area_vectors = tuple(
        _keep_periodic_faces(values[2], reference, axis)
        for axis, values in enumerate(full)
    )
    return (
        vertices,
        cell_centers,
        cell_volumes,
        cell_orientation,
        face_centers,
        face_measures,
        face_area_vectors,
        tuple(values[0] for values in full),
        tuple(values[1] for values in full),
        tuple(values[2] for values in full),
    )


def _dual_measures(
    reference: FiniteVolumeDiscretization,
    cell_centers: Array,
    face_measures: FaceVelocity,
    face_area_vectors: tuple[Array, ...],
    full_face_centers: tuple[Array, ...],
    full_face_measures: tuple[Array, ...],
    full_face_area_vectors: tuple[Array, ...],
    /,
) -> tuple[FaceVelocity, Array]:
    dual_measures = []
    signed_distances = []
    for axis, structured_axis in enumerate(reference.grid.structured_axes):
        cells = jnp.moveaxis(cell_centers, axis, 0)
        centers = jnp.moveaxis(full_face_centers[axis], axis, 0)
        areas = jnp.moveaxis(full_face_area_vectors[axis], axis, 0)
        measures = jnp.moveaxis(full_face_measures[axis], axis, 0)
        normals = areas / measures[..., None]
        lower = jnp.sum((cells[0] - centers[0]) * normals[0], axis=-1)
        upper = jnp.sum((centers[-1] - cells[-1]) * normals[-1], axis=-1)
        if cells.shape[0] == 1:
            interior = jnp.empty((0,) + lower.shape, dtype=cell_centers.dtype)
        else:
            interior = jnp.sum((cells[1:] - cells[:-1]) * normals[1:-1], axis=-1)
        if structured_axis.periodic:
            distance = jnp.concatenate(((lower + upper)[None, ...], interior), axis=0)
        else:
            distance = jnp.concatenate(
                (lower[None, ...], interior, upper[None, ...]), axis=0
            )
        signed_distances.append(distance.reshape((-1,)))
        kept_distance = jnp.moveaxis(distance, 0, axis)
        dual_measures.append(face_measures[axis] * kept_distance)
    minimum_distance = jnp.min(jnp.concatenate(tuple(signed_distances)))
    return tuple(dual_measures), minimum_distance


def _tuple_finite(values: tuple[Array, ...], /) -> Array:
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values)))


class MappedMACReport(StrictModule, NonTrainableState):
    """Geometry, free-stream, weighted-adjoint, and pressure-action evidence."""

    minimum_cell_volume: Array
    minimum_face_measure: Array
    minimum_velocity_dual_measure: Array
    minimum_oriented_dual_distance: Array
    free_stream_residual: Array
    weighted_adjoint_residual: Array
    constant_pressure_residual: Array
    pressure_symmetry_residual: Array
    pressure_energy: Array
    tolerance: float = eqx.field(static=True)
    finite: Array
    passed: Array
    report_id: str = eqx.field(static=True)


class MappedMACGeometryPlan(StrictModule, NonTrainableState):
    """Fixed-reference stationary mapped MAC geometry preparation."""

    reference: FiniteVolumeDiscretization
    coordinate_map: CoordinateMap = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: FiniteVolumeDiscretization,
        coordinate_map: CoordinateMap,
        /,
        *,
        mapping_id: str,
        tolerance: float = 1e-9,
    ):
        if not isinstance(reference, FiniteVolumeDiscretization) or not callable(
            coordinate_map
        ):
            raise TypeError("Mapped MAC plans require reference FV geometry and a map.")
        if len(reference.cell_shape) not in (1, 2, 3):
            raise ValueError(
                "Mapped MAC geometry supports one, two, or three dimensions."
            )
        identifier = str(mapping_id)
        tolerance_ = float(tolerance)
        if not identifier or identifier != identifier.strip():
            raise ValueError("mapping_id must be a non-empty canonical stripped string.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Mapped MAC tolerance must be positive and finite.")
        self.reference = reference
        self.coordinate_map = coordinate_map
        self.mapping_id = identifier
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-mac-geometry-plan",
                "reference": reference.prepared_id,
                "mapping": identifier,
                "map_arrays": array_tree_fingerprint(coordinate_map),
                "tolerance": tolerance_,
                "metric_rule": "mapped-polytope-incidence-dual",
            }
        )

    def prepare(self, /) -> "PreparedMappedMACGeometry":
        return PreparedMappedMACGeometry(self)


class PreparedMappedMACGeometry(StrictModule, NonTrainableState):
    """Positive mapped cells and compatible normal-face MAC operators."""

    reference: FiniteVolumeDiscretization
    mapped_vertices: Array
    cell_centers: Array
    cell_volumes: Array
    face_centers: tuple[Array, ...]
    face_measures: FaceVelocity
    face_area_vectors: tuple[Array, ...]
    face_dual_measures: FaceVelocity
    pressure_space: ArraySpace
    velocity_space: BlockSpace
    mapping_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    report: MappedMACReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MappedMACGeometryPlan, /):
        if not isinstance(plan, MappedMACGeometryPlan):
            raise TypeError("plan must be MappedMACGeometryPlan.")
        (
            vertices,
            cell_centers,
            cell_volumes,
            cell_orientation,
            face_centers,
            face_measures,
            face_area_vectors,
            full_face_centers,
            full_face_measures,
            full_face_area_vectors,
        ) = _evaluate_mapped_mac_geometry(plan.reference, plan.coordinate_map)
        face_dual_measures, minimum_distance = _dual_measures(
            plan.reference,
            cell_centers,
            face_measures,
            face_area_vectors,
            full_face_centers,
            full_face_measures,
            full_face_area_vectors,
        )
        pressure_space = ArraySpace(
            plan.reference.cell_shape,
            dtype=cell_volumes.dtype,
            pairing=DiagonalPairing(cell_volumes),
        )
        velocity_space = BlockSpace(
            tuple(
                ArraySpace(
                    layout.shape,
                    dtype=cell_volumes.dtype,
                    pairing=DiagonalPairing(measure),
                )
                for layout, measure in zip(
                    plan.reference.face_layouts, face_dual_measures, strict=True
                )
            ),
            names=plan.reference.grid.axis_names,
        )
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "mapped-mac-geometry-layout",
                "reference": plan.reference.prepared_id,
                "cell_shape": plan.reference.cell_shape,
                "face_layouts": [
                    layout.layout_id for layout in plan.reference.face_layouts
                ],
            }
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mapped-mac-geometry",
                "plan": plan.plan_id,
                "geometry_layout": geometry_layout_id,
                "pressure_space": pressure_space.space_id,
                "velocity_space": velocity_space.space_id,
            }
        )
        self.reference = plan.reference
        self.mapped_vertices = vertices
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes
        self.face_centers = face_centers
        self.face_measures = face_measures
        self.face_area_vectors = face_area_vectors
        self.face_dual_measures = face_dual_measures
        self.pressure_space = pressure_space
        self.velocity_space = velocity_space
        self.mapping_id = plan.mapping_id
        self.geometry_layout_id = geometry_layout_id
        self.prepared_id = prepared_id

        dtype = cell_volumes.dtype
        cell_count = prod(plan.reference.cell_shape)
        pressure = jnp.sin(jnp.arange(cell_count, dtype=dtype) + 0.37).reshape(
            plan.reference.cell_shape
        )
        second_pressure = jnp.cos(
            jnp.arange(cell_count, dtype=dtype) * 0.71 + 0.19
        ).reshape(plan.reference.cell_shape)
        velocity_values = []
        for axis, layout in enumerate(plan.reference.face_layouts):
            component = jnp.cos(
                jnp.arange(prod(layout.shape), dtype=dtype) * (0.31 + 0.07 * axis)
            ).reshape(layout.shape)
            if not plan.reference.grid.structured_axes[axis].periodic:
                lower = [slice(None)] * component.ndim
                upper = [slice(None)] * component.ndim
                lower[axis] = 0
                upper[axis] = component.shape[axis] - 1
                component = component.at[tuple(lower)].set(0.0)
                component = component.at[tuple(upper)].set(0.0)
            velocity_values.append(component)
        velocity = tuple(velocity_values)
        divergence = self.divergence(velocity)
        gradient = self.gradient(pressure)
        left = jnp.sum(cell_volumes * pressure * divergence)
        right = sum(
            jnp.sum(measure * component * derivative)
            for measure, component, derivative in zip(
                face_dual_measures, velocity, gradient, strict=True
            )
        )
        adjoint_scale = 1.0 + jnp.abs(left) + jnp.abs(right)
        adjoint_residual = jnp.abs(left + right) / adjoint_scale
        physical_constant = jnp.ones((len(plan.reference.cell_shape),), dtype=dtype)
        physical_constant = physical_constant / jnp.linalg.norm(physical_constant)
        free_stream = self.divergence(self.normal_velocity(physical_constant))
        free_stream_residual = jnp.max(jnp.abs(free_stream))
        unit_face = tuple(jnp.ones_like(value) for value in face_measures)
        action_pressure = self.pressure_action(pressure, unit_face)
        action_second = self.pressure_action(second_pressure, unit_face)
        symmetry_left = jnp.sum(cell_volumes * pressure * action_second)
        symmetry_right = jnp.sum(cell_volumes * action_pressure * second_pressure)
        symmetry_residual = jnp.abs(symmetry_left - symmetry_right) / (
            1.0 + jnp.abs(symmetry_left) + jnp.abs(symmetry_right)
        )
        gauged_pressure = self.gauge_project(pressure)
        pressure_energy = jnp.sum(cell_volumes * gauged_pressure * action_pressure)
        constant = jnp.ones(plan.reference.cell_shape, dtype=dtype)
        constant_residual = jnp.max(
            jnp.abs(self.pressure_action(constant, unit_face) - constant)
        )
        minimum_volume = jnp.min(cell_volumes)
        minimum_face = jnp.min(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in face_measures))
        )
        minimum_dual = jnp.min(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in face_dual_measures))
        )
        finite = (
            jnp.all(jnp.isfinite(vertices))
            & jnp.all(jnp.isfinite(cell_centers))
            & jnp.all(jnp.isfinite(cell_volumes))
            & jnp.all(jnp.isfinite(cell_orientation))
            & _tuple_finite(face_centers)
            & _tuple_finite(face_measures)
            & _tuple_finite(face_area_vectors)
            & _tuple_finite(face_dual_measures)
            & jnp.isfinite(free_stream_residual)
            & jnp.isfinite(adjoint_residual)
            & jnp.isfinite(constant_residual)
            & jnp.isfinite(symmetry_residual)
            & jnp.isfinite(pressure_energy)
        )
        passed = (
            finite
            & jnp.all(cell_volumes > 0.0)
            & jnp.all(cell_orientation > 0.0)
            & (minimum_face > 0.0)
            & (minimum_dual > 0.0)
            & (minimum_distance > 0.0)
            & (free_stream_residual <= plan.tolerance)
            & (adjoint_residual <= plan.tolerance)
            & (constant_residual <= plan.tolerance)
            & (symmetry_residual <= plan.tolerance)
            & (pressure_energy >= -plan.tolerance)
        )
        self.report = MappedMACReport(
            minimum_cell_volume=minimum_volume,
            minimum_face_measure=minimum_face,
            minimum_velocity_dual_measure=minimum_dual,
            minimum_oriented_dual_distance=minimum_distance,
            free_stream_residual=free_stream_residual,
            weighted_adjoint_residual=adjoint_residual,
            constant_pressure_residual=constant_residual,
            pressure_symmetry_residual=symmetry_residual,
            pressure_energy=pressure_energy,
            tolerance=plan.tolerance,
            finite=finite,
            passed=passed,
            report_id=canonical_fingerprint(
                {
                    "kind": "mapped-mac-report",
                    "geometry": prepared_id,
                    "metric_identity": "oriented-face-closure",
                    "adjoint_identity": "D=-G-star",
                    "pressure_identity": "positive-gauged-minus-DG",
                    "tolerance": plan.tolerance,
                }
            ),
        )
        if not bool(passed):
            raise RuntimeError(
                "Mapped MAC geometry failed mandatory compatibility evidence: "
                f"free_stream={free_stream_residual}, adjoint={adjoint_residual}, "
                f"constant={constant_residual}, symmetry={symmetry_residual}, "
                f"pressure_energy={pressure_energy}, minimum_distance={minimum_distance}."
            )

    def validate_pressure(self, pressure: ArrayLike, /) -> Array:
        return self.pressure_space.validate(jnp.asarray(pressure))

    def validate_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = tuple(jnp.asarray(component) for component in velocity)
        if len(values) != len(self.reference.cell_shape):
            raise ValueError(
                "Mapped MAC velocity requires one normal component per axis."
            )
        return tuple(self.velocity_space.validate(values))

    def gauge_project(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_pressure(pressure)
        volume = self.cell_volumes.astype(value.dtype)
        return value - jnp.sum(volume * value) / jnp.sum(volume)

    def compatibility_project(self, right_hand_side: ArrayLike, /) -> Array:
        return self.gauge_project(right_hand_side)

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        values = self.validate_velocity(velocity)
        result = jnp.zeros(self.reference.cell_shape, dtype=self.pressure_space.dtype)
        for axis, component in enumerate(values):
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
                interior = moved[1:] - moved[:-1]
                jump = jnp.concatenate(
                    (jnp.zeros_like(moved[:1]), interior, jnp.zeros_like(moved[:1])),
                    axis=0,
                )
            jump = jnp.moveaxis(jump, 0, axis)
            output.append(jump * self.face_measures[axis] / self.face_dual_measures[axis])
        return tuple(output)

    def pressure_action(
        self,
        pressure: ArrayLike,
        face_inverse_momentum: FaceVelocity,
        /,
    ) -> Array:
        value = self.validate_pressure(pressure)
        coefficient = self.validate_velocity(face_inverse_momentum)
        volume = self.cell_volumes.astype(value.dtype)
        mean = jnp.sum(volume * value) / jnp.sum(volume)
        gauged = value - mean
        gradient = self.gradient(gauged)
        flux = tuple(
            inverse * derivative
            for inverse, derivative in zip(coefficient, gradient, strict=True)
        )
        return -self.divergence(flux) + mean

    def normal_velocity(self, physical_velocity: ArrayLike, /) -> FaceVelocity:
        value = jnp.asarray(physical_velocity, dtype=self.pressure_space.dtype)
        dimension = len(self.reference.cell_shape)
        if value.shape != (dimension,):
            raise ValueError("physical_velocity must contain one value per dimension.")
        return tuple(
            jnp.sum(area_vector * value, axis=-1) / measure
            for area_vector, measure in zip(
                self.face_area_vectors, self.face_measures, strict=True
            )
        )

    def reconstruct_cell_velocity(self, velocity: FaceVelocity, /) -> Array:
        values = self.validate_velocity(velocity)
        dimension = len(self.reference.cell_shape)
        matrix = jnp.zeros(
            self.reference.cell_shape + (dimension, dimension),
            dtype=self.pressure_space.dtype,
        )
        right_hand_side = jnp.zeros(
            self.reference.cell_shape + (dimension,), dtype=self.pressure_space.dtype
        )
        for axis, structured_axis in enumerate(self.reference.grid.structured_axes):
            component = jnp.moveaxis(values[axis], axis, 0)
            area = jnp.moveaxis(self.face_area_vectors[axis], axis, 0)
            measure = jnp.moveaxis(self.face_measures[axis], axis, 0)
            normal = area / measure[..., None]
            if structured_axis.periodic:
                lower_component = component
                upper_component = jnp.roll(component, -1, axis=0)
                lower_normal = normal
                upper_normal = jnp.roll(normal, -1, axis=0)
                lower_measure = measure
                upper_measure = jnp.roll(measure, -1, axis=0)
            else:
                lower_component = component[:-1]
                upper_component = component[1:]
                lower_normal = normal[:-1]
                upper_normal = normal[1:]
                lower_measure = measure[:-1]
                upper_measure = measure[1:]
            component_pair = (lower_component, upper_component)
            normal_pair = (lower_normal, upper_normal)
            measure_pair = (lower_measure, upper_measure)
            for face_component, face_normal, face_measure in zip(
                component_pair, normal_pair, measure_pair, strict=True
            ):
                weight = 0.5 * face_measure
                contribution_matrix = (
                    weight[..., None, None]
                    * face_normal[..., :, None]
                    * face_normal[..., None, :]
                )
                contribution_right = (
                    weight[..., None] * face_component[..., None] * face_normal
                )
                matrix = matrix + jnp.moveaxis(contribution_matrix, 0, axis)
                right_hand_side = right_hand_side + jnp.moveaxis(
                    contribution_right, 0, axis
                )
        return jnp.linalg.solve(matrix, right_hand_side[..., None])[..., 0]

    def interpolate_cell_vector(
        self,
        cell_vector: ArrayLike,
        /,
        *,
        prescribed_normal: FaceVelocity | None = None,
    ) -> tuple[Array, ...]:
        value = jnp.asarray(cell_vector, dtype=self.pressure_space.dtype)
        dimension = len(self.reference.cell_shape)
        if value.shape != self.reference.cell_shape + (dimension,):
            raise ValueError("cell_vector must have cell shape plus physical dimension.")
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
                    (moved[:1], 0.5 * (moved[1:] + moved[:-1]), moved[-1:]),
                    axis=0,
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
        values = self.validate_velocity(velocity)
        return 0.5 * sum(
            jnp.sum(measure * component**2)
            for measure, component in zip(self.face_dual_measures, values, strict=True)
        )


__all__ = [
    "MappedMACGeometryPlan",
    "MappedMACReport",
    "PreparedMappedMACGeometry",
]
