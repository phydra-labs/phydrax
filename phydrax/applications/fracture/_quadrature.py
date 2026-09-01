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
from ...discretization import CellMesh
from ._geometry import (
    _cross_2d,
    _point_in_triangle,
    SharpCrackTopology,
)


def _clip_polygon(
    vertices: np.ndarray,
    signed_distances: np.ndarray,
    *,
    positive: bool,
    tolerance: float,
) -> np.ndarray:
    polygon = [
        (np.asarray(vertex), float(distance))
        for vertex, distance in zip(vertices, signed_distances, strict=True)
    ]
    result: list[tuple[np.ndarray, float]] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_inside = current[1] >= -tolerance if positive else current[1] <= tolerance
        previous_inside = (
            previous[1] >= -tolerance if positive else previous[1] <= tolerance
        )
        if current_inside != previous_inside:
            denominator = previous[1] - current[1]
            if abs(denominator) <= tolerance:
                raise ValueError(
                    "A crack-side clipping intersection is numerically indeterminate."
                )
            fraction = previous[1] / denominator
            result.append((previous[0] + fraction * (current[0] - previous[0]), 0.0))
        if current_inside:
            result.append(current)
    if not result:
        return np.empty((0, 2), dtype=vertices.dtype)
    unique: list[np.ndarray] = []
    for point, _ in result:
        if not unique or np.linalg.norm(point - unique[-1]) > tolerance:
            unique.append(point)
    if len(unique) > 1 and np.linalg.norm(unique[0] - unique[-1]) <= tolerance:
        unique.pop()
    return np.asarray(unique)


def _triangle_rule(
    triangle: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    determinant = abs(first[0] * second[1] - first[1] * second[0])
    if determinant <= 0.0:
        return np.empty((0, 2), dtype=triangle.dtype), np.empty(
            (0,), dtype=triangle.dtype
        )
    if order == 1:
        return np.mean(triangle, axis=0, keepdims=True), np.asarray((0.5 * determinant,))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    points: list[np.ndarray] = []
    physical_weights: list[float] = []
    for radial, radial_weight in zip(unit_nodes, unit_weights, strict=True):
        for transverse, transverse_weight in zip(unit_nodes, unit_weights, strict=True):
            edge_point = (1.0 - transverse) * triangle[1] + transverse * triangle[2]
            points.append((1.0 - radial) * triangle[0] + radial * edge_point)
            physical_weights.append(
                float(radial_weight * transverse_weight * radial * determinant)
            )
    return np.asarray(points), np.asarray(physical_weights, dtype=triangle.dtype)


def _polygon_rule(
    polygon: np.ndarray,
    order: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if polygon.shape[0] < 3:
        return np.empty((0, 2), dtype=polygon.dtype), np.empty((0,), dtype=polygon.dtype)
    all_points: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for index in range(1, polygon.shape[0] - 1):
        triangle = np.stack((polygon[0], polygon[index], polygon[index + 1]))
        points, weights = _triangle_rule(triangle, order)
        if weights.size and float(np.sum(weights)) > tolerance:
            all_points.append(points)
            all_weights.append(weights)
    if not all_points:
        return np.empty((0, 2), dtype=polygon.dtype), np.empty((0,), dtype=polygon.dtype)
    return np.concatenate(all_points, axis=0), np.concatenate(all_weights, axis=0)


def _reference_points(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    jacobian = np.stack((triangle[1] - triangle[0], triangle[2] - triangle[0]), axis=1)
    return np.linalg.solve(jacobian, (points - triangle[0]).T).T


class CrackVolumeQuadrature(StrictModule, NonTrainableState):
    """Physical and parent-reference volume points for one crack side."""

    points: Array
    reference_points: Array
    weights: Array
    cell_ids: Array
    segment_ids: Array
    side: int = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        reference_points: ArrayLike,
        weights: ArrayLike,
        cell_ids: ArrayLike,
        segment_ids: ArrayLike,
        /,
        *,
        side: int,
    ):
        points_ = np.asarray(points)
        reference = np.asarray(reference_points)
        weights_ = np.asarray(weights)
        cells = np.asarray(cell_ids, dtype=np.int64)
        segments = np.asarray(segment_ids, dtype=np.int64)
        side_ = int(side)
        count = weights_.size
        if (
            side_ not in (-1, 1)
            or points_.shape != (count, 2)
            or reference.shape != (count, 2)
            or weights_.shape != (count,)
            or cells.shape != (count,)
            or segments.shape != (count,)
            or np.any(~np.isfinite(points_))
            or np.any(~np.isfinite(reference))
            or np.any(~np.isfinite(weights_))
            or np.any(weights_ <= 0.0)
        ):
            raise ValueError("Crack-side volume quadrature arrays are inconsistent.")
        self.points = jnp.asarray(points_)
        self.reference_points = jnp.asarray(reference)
        self.weights = jnp.asarray(weights_)
        self.cell_ids = jnp.asarray(cells)
        self.segment_ids = jnp.asarray(segments)
        self.side = side_


class CrackFaceQuadrature(StrictModule, NonTrainableState):
    """Common physical crack-face points duplicated with opposite orientations."""

    points: Array
    weights: Array
    parameters: Array
    normals: Array
    segment_ids: Array
    side: Array

    def __init__(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        parameters: ArrayLike,
        normals: ArrayLike,
        segment_ids: ArrayLike,
        side: ArrayLike,
        /,
    ):
        points_ = np.asarray(points)
        weights_ = np.asarray(weights)
        parameters_ = np.asarray(parameters)
        normals_ = np.asarray(normals)
        segments = np.asarray(segment_ids, dtype=np.int64)
        side_ = np.asarray(side, dtype=np.int8)
        count = weights_.size
        if (
            points_.shape != (count, 2)
            or weights_.shape != (count,)
            or parameters_.shape != (count,)
            or normals_.shape != (count, 2)
            or segments.shape != (count,)
            or side_.shape != (count,)
            or np.any(~np.isfinite(points_))
            or np.any(~np.isfinite(weights_))
            or np.any(weights_ <= 0.0)
            or np.any((parameters_ <= 0.0) | (parameters_ >= 1.0))
            or np.any(np.abs(np.linalg.norm(normals_, axis=1) - 1.0) > 1.0e-10)
            or np.any(~np.isin(side_, (-1, 1)))
        ):
            raise ValueError("Crack-face quadrature arrays are inconsistent.")
        self.points = jnp.asarray(points_)
        self.weights = jnp.asarray(weights_)
        self.parameters = jnp.asarray(parameters_)
        self.normals = jnp.asarray(normals_)
        self.segment_ids = jnp.asarray(segments)
        self.side = jnp.asarray(side_)


class CrackTipQuadrature(StrictModule, NonTrainableState):
    """Duffy points in tip cells; no point evaluates the singular tip itself."""

    points: Array
    reference_points: Array
    weights: Array
    radii: Array
    angles: Array
    side: Array
    cell_ids: Array
    tip_ids: Array

    def __init__(
        self,
        points: ArrayLike,
        reference_points: ArrayLike,
        weights: ArrayLike,
        radii: ArrayLike,
        angles: ArrayLike,
        side: ArrayLike,
        cell_ids: ArrayLike,
        tip_ids: ArrayLike,
        /,
    ):
        points_ = np.asarray(points)
        reference = np.asarray(reference_points)
        weights_ = np.asarray(weights)
        radii_ = np.asarray(radii)
        angles_ = np.asarray(angles)
        side_ = np.asarray(side, dtype=np.int8)
        cells = np.asarray(cell_ids, dtype=np.int64)
        tips = np.asarray(tip_ids, dtype=np.int64)
        count = weights_.size
        if (
            points_.shape != (count, 2)
            or reference.shape != (count, 2)
            or weights_.shape != (count,)
            or radii_.shape != (count,)
            or angles_.shape != (count,)
            or side_.shape != (count,)
            or cells.shape != (count,)
            or tips.shape != (count,)
            or np.any(~np.isfinite(points_))
            or np.any(~np.isfinite(reference))
            or np.any(~np.isfinite(weights_))
            or np.any(weights_ <= 0.0)
            or np.any(radii_ <= 0.0)
            or np.any(~np.isin(side_, (-1, 1)))
        ):
            raise ValueError("Crack-tip Duffy quadrature arrays are inconsistent.")
        self.points = jnp.asarray(points_)
        self.reference_points = jnp.asarray(reference)
        self.weights = jnp.asarray(weights_)
        self.radii = jnp.asarray(radii_)
        self.angles = jnp.asarray(angles_)
        self.side = jnp.asarray(side_)
        self.cell_ids = jnp.asarray(cells)
        self.tip_ids = jnp.asarray(tips)


class CrackQuadratureEvidence(StrictModule, NonTrainableState):
    """Conservation and singular-point evidence for a sharp realization."""

    cut_cell_area: Array
    integrated_area: Array
    relative_area_defect: Array
    plus_face_measure: Array
    minus_face_measure: Array
    face_measure_defect: Array
    minimum_tip_radius: Array
    order: int = eqx.field(static=True)


class SharpCrackQuadrature(StrictModule, NonTrainableState):
    """Immutable plus/minus, face, and tip quadrature for one topology snapshot."""

    plus: CrackVolumeQuadrature
    minus: CrackVolumeQuadrature
    faces: CrackFaceQuadrature
    tips: CrackTipQuadrature
    evidence: CrackQuadratureEvidence
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        plus: CrackVolumeQuadrature,
        minus: CrackVolumeQuadrature,
        faces: CrackFaceQuadrature,
        tips: CrackTipQuadrature,
        evidence: CrackQuadratureEvidence,
        /,
        *,
        topology_id: str,
        geometry_id: str,
    ):
        if not isinstance(plus, CrackVolumeQuadrature) or plus.side != 1:
            raise TypeError("plus must be a positive CrackVolumeQuadrature.")
        if not isinstance(minus, CrackVolumeQuadrature) or minus.side != -1:
            raise TypeError("minus must be a negative CrackVolumeQuadrature.")
        if not isinstance(faces, CrackFaceQuadrature) or not isinstance(
            tips, CrackTipQuadrature
        ):
            raise TypeError(
                "Sharp quadrature requires crack-face and crack-tip realizations."
            )
        if not isinstance(evidence, CrackQuadratureEvidence):
            raise TypeError("evidence must be CrackQuadratureEvidence.")
        topology_identifier = str(topology_id)
        geometry_identifier = str(geometry_id)
        if not topology_identifier or not geometry_identifier:
            raise ValueError("Sharp quadrature provenance IDs must be nonempty.")
        self.plus = plus
        self.minus = minus
        self.faces = faces
        self.tips = tips
        self.evidence = evidence
        self.topology_id = topology_identifier
        self.geometry_id = geometry_identifier
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "sharp-crack-quadrature",
                "topology": topology_identifier,
                "geometry": geometry_identifier,
                "order": evidence.order,
                "plus_count": int(plus.weights.size),
                "minus_count": int(minus.weights.size),
                "face_count": int(faces.weights.size),
                "tip_count": int(tips.weights.size),
                "area": float(evidence.integrated_area),
            }
        )


def _empty_volume(side: int, dtype: np.dtype) -> CrackVolumeQuadrature:
    return CrackVolumeQuadrature(
        np.empty((0, 2), dtype=dtype),
        np.empty((0, 2), dtype=dtype),
        np.empty((0,), dtype=dtype),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        side=side,
    )


def build_sharp_crack_quadrature(
    mesh: CellMesh,
    topology: SharpCrackTopology,
    /,
    *,
    order: int = 2,
    tolerance: float = 1.0e-12,
) -> SharpCrackQuadrature:
    """Split non-tip cells by side and replace every tip cell by a Duffy fan."""

    if not isinstance(mesh, CellMesh) or not isinstance(topology, SharpCrackTopology):
        raise TypeError("Sharp quadrature requires CellMesh and SharpCrackTopology.")
    order_ = int(order)
    tolerance_ = float(tolerance)
    if order_ < 1 or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Sharp quadrature order and tolerance are invalid.")
    if mesh.mesh_id != topology.mesh_id:
        raise ValueError("Sharp topology and mesh IDs do not match.")
    if topology.geometry_id != topology.geometry.geometry_id:
        raise ValueError("Sharp topology carries stale geometry provenance.")
    if (
        mesh.ambient_dimension != 2
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind != "triangle"
    ):
        raise ValueError(
            "Sharp quadrature currently requires one two-dimensional T3 block."
        )

    geometry = topology.geometry
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    global_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    local_by_id = {int(identifier): index for index, identifier in enumerate(global_ids)}
    segment_index_by_id = {
        int(identifier): index
        for index, identifier in enumerate(np.asarray(geometry.segment_ids))
    }
    tip_cells = set(np.asarray(topology.tip_cell_ids).tolist())

    side_points: dict[int, list[np.ndarray]] = {1: [], -1: []}
    side_reference: dict[int, list[np.ndarray]] = {1: [], -1: []}
    side_weights: dict[int, list[np.ndarray]] = {1: [], -1: []}
    side_cells: dict[int, list[np.ndarray]] = {1: [], -1: []}
    side_segments: dict[int, list[np.ndarray]] = {1: [], -1: []}
    tip_points: list[np.ndarray] = []
    tip_reference: list[np.ndarray] = []
    tip_weights: list[np.ndarray] = []
    tip_radii: list[np.ndarray] = []
    tip_angles: list[np.ndarray] = []
    tip_sides: list[np.ndarray] = []
    tip_cell_values: list[np.ndarray] = []
    tip_id_values: list[np.ndarray] = []
    cut_area = 0.0

    for cell_id, segment_id in zip(
        np.asarray(topology.cut_cell_ids),
        np.asarray(topology.cell_segment_ids),
        strict=True,
    ):
        local_cell = local_by_id[int(cell_id)]
        triangle = coordinates[cells[local_cell]]
        area = 0.5 * abs(_cross_2d(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if area <= tolerance_:
            raise ValueError("Sharp quadrature cannot integrate a degenerate cut cell.")
        cut_area += area
        segment_index = segment_index_by_id[int(segment_id)]
        segment = np.asarray(geometry.segments)[segment_index]
        start = np.asarray(geometry.vertices)[segment[0]]
        normal = np.asarray(geometry.segment_normals())[segment_index]
        signed = (triangle - start) @ normal

        if int(cell_id) not in tip_cells:
            for side in (1, -1):
                polygon = _clip_polygon(
                    triangle,
                    signed,
                    positive=side == 1,
                    tolerance=tolerance_,
                )
                points, weights = _polygon_rule(polygon, order_, tolerance_)
                if weights.size == 0 or float(np.sum(weights)) <= tolerance_:
                    raise ValueError("A split crack cell produced a zero-measure side.")
                side_points[side].append(points)
                side_reference[side].append(_reference_points(points, triangle))
                side_weights[side].append(weights)
                side_cells[side].append(
                    np.full(weights.shape, int(cell_id), dtype=np.int64)
                )
                side_segments[side].append(
                    np.full(weights.shape, int(segment_id), dtype=np.int64)
                )
            continue

        live_tips: list[int] = []
        for tip_id, tip_vertex in zip(
            np.asarray(geometry.tip_ids),
            np.asarray(geometry.tip_vertex_ids),
            strict=True,
        ):
            if _point_in_triangle(
                np.asarray(geometry.vertices)[int(tip_vertex)], triangle, tolerance_
            ):
                live_tips.append(int(tip_id))
        if len(live_tips) != 1:
            raise ValueError("Every tip cell must contain exactly one live crack tip.")
        tip_id = live_tips[0]
        tip_origin, _, _ = geometry.tip_frame(tip_id)
        tip_origin_ = np.asarray(tip_origin)
        for index in range(3):
            fan_triangle = np.stack(
                (tip_origin_, triangle[index], triangle[(index + 1) % 3])
            )
            points, weights = _triangle_rule(fan_triangle, order_)
            if weights.size == 0 or float(np.sum(weights)) <= tolerance_:
                continue
            local = np.asarray(geometry.tip_local_coordinates(points, tip_id))
            sides = np.where(np.asarray(geometry.signed_distance(points)) >= 0.0, 1, -1)
            tip_points.append(points)
            tip_reference.append(_reference_points(points, triangle))
            tip_weights.append(weights)
            tip_radii.append(local[:, 0])
            tip_angles.append(local[:, 1])
            tip_sides.append(sides.astype(np.int8))
            tip_cell_values.append(np.full(weights.shape, int(cell_id), dtype=np.int64))
            tip_id_values.append(np.full(weights.shape, tip_id, dtype=np.int64))

    dtype = coordinates.dtype

    def volume(side: int) -> CrackVolumeQuadrature:
        if not side_points[side]:
            return _empty_volume(side, dtype)
        return CrackVolumeQuadrature(
            np.concatenate(side_points[side], axis=0),
            np.concatenate(side_reference[side], axis=0),
            np.concatenate(side_weights[side], axis=0),
            np.concatenate(side_cells[side], axis=0),
            np.concatenate(side_segments[side], axis=0),
            side=side,
        )

    plus = volume(1)
    minus = volume(-1)
    if tip_points:
        tips = CrackTipQuadrature(
            np.concatenate(tip_points, axis=0),
            np.concatenate(tip_reference, axis=0),
            np.concatenate(tip_weights, axis=0),
            np.concatenate(tip_radii, axis=0),
            np.concatenate(tip_angles, axis=0),
            np.concatenate(tip_sides, axis=0),
            np.concatenate(tip_cell_values, axis=0),
            np.concatenate(tip_id_values, axis=0),
        )
    else:
        tips = CrackTipQuadrature(
            np.empty((0, 2), dtype=dtype),
            np.empty((0, 2), dtype=dtype),
            np.empty((0,), dtype=dtype),
            np.empty((0,), dtype=dtype),
            np.empty((0,), dtype=dtype),
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    face_nodes, face_weights = np.polynomial.legendre.leggauss(order_)
    face_parameters = 0.5 * (face_nodes + 1.0)
    face_unit_weights = 0.5 * face_weights
    physical_points: list[np.ndarray] = []
    physical_weights: list[np.ndarray] = []
    physical_parameters: list[np.ndarray] = []
    physical_normals: list[np.ndarray] = []
    physical_segments: list[np.ndarray] = []
    physical_sides: list[np.ndarray] = []
    normals = np.asarray(geometry.segment_normals())
    for segment_index, (segment, segment_id) in enumerate(
        zip(np.asarray(geometry.segments), np.asarray(geometry.segment_ids), strict=True)
    ):
        start = np.asarray(geometry.vertices)[segment[0]]
        end = np.asarray(geometry.vertices)[segment[1]]
        length = float(np.linalg.norm(end - start))
        points = start + face_parameters[:, None] * (end - start)
        weights = face_unit_weights * length
        for side in (1, -1):
            physical_points.append(points)
            physical_weights.append(weights)
            physical_parameters.append(face_parameters)
            physical_normals.append(
                np.broadcast_to(side * normals[segment_index], points.shape)
            )
            physical_segments.append(
                np.full(weights.shape, int(segment_id), dtype=np.int64)
            )
            physical_sides.append(np.full(weights.shape, side, dtype=np.int8))
    faces = CrackFaceQuadrature(
        np.concatenate(physical_points, axis=0),
        np.concatenate(physical_weights, axis=0),
        np.concatenate(physical_parameters, axis=0),
        np.concatenate(physical_normals, axis=0),
        np.concatenate(physical_segments, axis=0),
        np.concatenate(physical_sides, axis=0),
    )

    integrated_area = float(
        jnp.sum(plus.weights) + jnp.sum(minus.weights) + jnp.sum(tips.weights)
    )
    plus_face_measure = float(
        np.sum(np.asarray(faces.weights)[np.asarray(faces.side) == 1])
    )
    minus_face_measure = float(
        np.sum(np.asarray(faces.weights)[np.asarray(faces.side) == -1])
    )
    evidence = CrackQuadratureEvidence(
        cut_cell_area=jnp.asarray(cut_area),
        integrated_area=jnp.asarray(integrated_area),
        relative_area_defect=jnp.asarray(
            abs(integrated_area - cut_area) / max(cut_area, np.finfo(float).eps)
        ),
        plus_face_measure=jnp.asarray(plus_face_measure),
        minus_face_measure=jnp.asarray(minus_face_measure),
        face_measure_defect=jnp.asarray(abs(plus_face_measure - minus_face_measure)),
        minimum_tip_radius=jnp.asarray(
            float(np.min(np.asarray(tips.radii))) if tips.radii.size else np.inf
        ),
        order=order_,
    )
    return SharpCrackQuadrature(
        plus,
        minus,
        faces,
        tips,
        evidence,
        topology_id=topology.topology_id,
        geometry_id=geometry.geometry_id,
    )


__all__ = [
    "CrackFaceQuadrature",
    "CrackQuadratureEvidence",
    "CrackTipQuadrature",
    "CrackVolumeQuadrature",
    "SharpCrackQuadrature",
    "build_sharp_crack_quadrature",
]
