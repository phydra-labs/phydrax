#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class HydroelasticPressureFieldPlan(StrictModule, NonTrainableState):
    vertices: Array
    tetrahedra: Array
    pressure: Array
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        tetrahedra: ArrayLike,
        pressure: ArrayLike,
        /,
    ):
        vertices_ = np.asarray(vertices, dtype=float)
        tetrahedra_ = np.asarray(tetrahedra)
        pressure_ = np.asarray(pressure, dtype=float)
        if vertices_.ndim != 2 or vertices_.shape[1:] != (3,):
            raise ValueError("Hydroelastic pressure vertices require three dimensions.")
        if (
            tetrahedra_.ndim != 2
            or tetrahedra_.shape[1:] != (4,)
            or not np.issubdtype(tetrahedra_.dtype, np.integer)
        ):
            raise TypeError("Hydroelastic pressure topology requires tetrahedra.")
        if pressure_.shape != (vertices_.shape[0],):
            raise ValueError("Hydroelastic pressure requires one scalar per vertex.")
        if (
            np.any(tetrahedra_ < 0)
            or np.any(tetrahedra_ >= vertices_.shape[0])
            or np.any(~np.isfinite(vertices_))
            or np.any(~np.isfinite(pressure_))
            or np.any(pressure_ < 0.0)
        ):
            raise ValueError("Hydroelastic pressure field data is invalid.")
        self.vertices = jnp.asarray(vertices_)
        self.tetrahedra = jnp.asarray(tetrahedra_, dtype=jnp.int32)
        self.pressure = jnp.asarray(pressure_)
        self.field_id = canonical_fingerprint(
            {
                "kind": "hydroelastic-pressure-field-plan",
                "vertices": array_tree_fingerprint(vertices_),
                "tetrahedra": array_tree_fingerprint(tetrahedra_),
                "pressure": array_tree_fingerprint(pressure_),
            }
        )


class HydroelasticPressurePatch(StrictModule, NonTrainableState):
    quadrature_point: Array
    normal: Array
    pressure: Array
    quadrature_weight: Array
    source_tetrahedron: Array
    valid: Array
    patch_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return int(self.valid.size)


class HydroelasticPatchEvidence(StrictModule):
    intersected_tetrahedra: Array
    triangle_count: Array
    overflow_count: Array
    total_area: Array
    minimum_pressure: Array
    finite: Array
    complete: Array
    successful: Array
    plus_field_id: str = eqx.field(static=True)
    minus_field_id: str = eqx.field(static=True)


class HydroelasticPatchExtraction(StrictModule):
    patch: HydroelasticPressurePatch
    evidence: HydroelasticPatchEvidence


def _unique_points(points, pressures, tolerance):
    unique_points = []
    unique_pressures = []
    for point, pressure in zip(points, pressures, strict=True):
        if any(
            np.dot(point - other, point - other) <= tolerance * tolerance
            for other in unique_points
        ):
            continue
        unique_points.append(point)
        unique_pressures.append(pressure)
    return unique_points, unique_pressures


def _ordered_polygon(points):
    values = np.asarray(points)
    centroid = values.mean(axis=0)
    normal = np.cross(values[1] - values[0], values[2] - values[0])
    norm = np.sqrt(np.dot(normal, normal))
    if norm == 0.0:
        return values, np.asarray((0.0, 0.0, 1.0))
    normal = normal / norm
    first = values[0] - centroid
    first = first / np.sqrt(np.dot(first, first))
    second = np.cross(normal, first)
    angles = np.arctan2(
        (values - centroid) @ second,
        (values - centroid) @ first,
    )
    return values[np.argsort(angles, kind="stable")], normal


def extract_hydroelastic_pressure_patch(
    plus: HydroelasticPressureFieldPlan,
    minus: HydroelasticPressureFieldPlan,
    /,
    *,
    capacity: int,
    tolerance: float = 1.0e-12,
) -> HydroelasticPatchExtraction:
    """Extract the equal-pressure patch on one common tetrahedral partition."""
    if not isinstance(plus, HydroelasticPressureFieldPlan) or not isinstance(
        minus, HydroelasticPressureFieldPlan
    ):
        raise TypeError("plus/minus must be hydroelastic pressure fields.")
    if (
        plus.vertices.shape != minus.vertices.shape
        or plus.tetrahedra.shape != minus.tetrahedra.shape
    ):
        raise ValueError("Hydroelastic fields must share one tetrahedral partition.")
    if not bool(jnp.all(plus.tetrahedra == minus.tetrahedra)) or not bool(
        jnp.allclose(plus.vertices, minus.vertices)
    ):
        raise ValueError("Hydroelastic field topology/coordinates must agree exactly.")
    count = int(capacity)
    tolerance_ = float(tolerance)
    if count <= 0 or tolerance_ <= 0.0:
        raise ValueError("Hydroelastic patch capacity/tolerance is invalid.")
    vertices = np.asarray(plus.vertices)
    tetrahedra = np.asarray(plus.tetrahedra)
    plus_pressure = np.asarray(plus.pressure)
    minus_pressure = np.asarray(minus.pressure)
    difference = plus_pressure - minus_pressure
    records = []
    intersected = 0
    for tetrahedron_index, tetrahedron in enumerate(tetrahedra):
        local_difference = difference[tetrahedron]
        local_vertices = vertices[tetrahedron]
        local_plus = plus_pressure[tetrahedron]
        local_minus = minus_pressure[tetrahedron]
        points = []
        pressures = []
        for first, second in _TETRAHEDRON_EDGES:
            first_value = local_difference[first]
            second_value = local_difference[second]
            if first_value * second_value > 0.0:
                continue
            denominator = first_value - second_value
            if abs(denominator) <= tolerance_:
                parameter = 0.5
            else:
                parameter = first_value / denominator
            if not -tolerance_ <= parameter <= 1.0 + tolerance_:
                continue
            parameter = float(np.clip(parameter, 0.0, 1.0))
            point = (1.0 - parameter) * local_vertices[
                first
            ] + parameter * local_vertices[second]
            pressure = 0.5 * (
                (1.0 - parameter) * (local_plus[first] + local_minus[first])
                + parameter * (local_plus[second] + local_minus[second])
            )
            points.append(point)
            pressures.append(max(float(pressure), 0.0))
        points, pressures = _unique_points(points, pressures, tolerance_)
        if len(points) < 3:
            continue
        intersected += 1
        polygon, normal = _ordered_polygon(points)
        pressure_by_point = []
        for point in polygon:
            index = min(
                range(len(points)),
                key=lambda candidate: float(
                    np.dot(point - points[candidate], point - points[candidate])
                ),
            )
            pressure_by_point.append(pressures[index])
        for local_index in range(1, len(polygon) - 1):
            triangle = np.asarray(
                (polygon[0], polygon[local_index], polygon[local_index + 1])
            )
            cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            area = 0.5 * np.sqrt(np.dot(cross, cross))
            if area <= tolerance_:
                continue
            pressure = (
                pressure_by_point[0]
                + pressure_by_point[local_index]
                + pressure_by_point[local_index + 1]
            ) / 3.0
            records.append(
                (
                    triangle.mean(axis=0),
                    normal,
                    pressure,
                    area,
                    tetrahedron_index,
                )
            )
    actual = len(records)
    overflow = max(actual - count, 0)
    points = np.zeros((count, 3), dtype=float)
    normals = np.zeros((count, 3), dtype=float)
    normals[:, 2] = 1.0
    pressures = np.zeros((count,), dtype=float)
    weights = np.zeros((count,), dtype=float)
    sources = np.zeros((count,), dtype=np.int32)
    valid = np.zeros((count,), dtype=bool)
    for slot, record in enumerate(records[:count]):
        points[slot], normals[slot], pressures[slot], weights[slot], sources[slot] = (
            record
        )
    if overflow == 0:
        valid[:actual] = True
    patch_id = canonical_fingerprint(
        {
            "kind": "hydroelastic-pressure-patch",
            "plus": plus.field_id,
            "minus": minus.field_id,
            "capacity": count,
        }
    )
    patch = HydroelasticPressurePatch(
        jnp.asarray(points),
        jnp.asarray(normals),
        jnp.asarray(pressures),
        jnp.asarray(weights),
        jnp.asarray(sources),
        jnp.asarray(valid),
        patch_id,
    )
    finite = np.all(np.isfinite(points)) and np.all(np.isfinite(pressures))
    complete = finite and overflow == 0
    evidence = HydroelasticPatchEvidence(
        jnp.asarray(intersected, dtype=jnp.int32),
        jnp.asarray(actual, dtype=jnp.int32),
        jnp.asarray(overflow, dtype=jnp.int32),
        jnp.asarray(weights.sum()),
        jnp.asarray(pressures[valid].min() if np.any(valid) else np.inf),
        jnp.asarray(finite),
        jnp.asarray(complete),
        jnp.asarray(complete),
        plus.field_id,
        minus.field_id,
    )
    return HydroelasticPatchExtraction(patch, evidence)


__all__ = [
    "HydroelasticPatchEvidence",
    "HydroelasticPatchExtraction",
    "HydroelasticPressureFieldPlan",
    "HydroelasticPressurePatch",
    "extract_hydroelastic_pressure_patch",
]
