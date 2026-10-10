#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable sampled cell quality and finite-volume face quality.

Owning coordinate maps supply signed measures and sampled Jacobian metric
quality; explicit corner-coordinate evaluations retain straight-corner/star
quality. Neither sampling route proves validity: the Bernstein certificate in
``phydrax.discretization._cell_geometry_validity`` is the validity owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import charge_native_geometry_queries
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometryElement,
    CellVertexGeometryElement,
    LayerColumnCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    polyhedral_star_tables,
    PolyhedralStarTables,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._motion_validity import reference_corner_frames
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._generic import _degree_aware_reference_rule
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..typing import checked
from ._metric import interpolate_mesh_metric, MeshMetricField
from ._scope import MeshingEntityKind


class CellQualityEvaluation(StrictModule):
    """Per-cell sampled quality in mesh block order.

    Angles are radians. Without ``geometry_layout_id``, every shape statistic
    uses straight corner frames (or vertex-star simplices). With an owning
    scalar coordinate map, measures integrate its signed Jacobian; shape,
    radius/sliver, condition, and metric statistics sample its Jacobian images
    of the canonical reference shape. Boundary angles use actual source-map
    corner tangents or edge dihedrals; warpage samples mapped face normals.
    Vertex-defined polygon/polyhedron maps retain their
    owning star convention. Rational/embedded measures are sampled quadrature,
    not exact integration or a whole-map validity claim. ``metric_quality`` is
    NaN without a metric; radius/sliver values are NaN for nonsimplicial kinds.
    """

    cell_global_ids: Array
    measures: Array
    mean_ratios: Array
    aspect_ratios: Array
    radius_ratios: Array
    scaled_jacobian: Array
    minimum_angle: Array
    maximum_angle: Array
    condition_number: Array
    sliver_measures: Array
    warpage: Array
    metric_quality: Array
    sampled_valid: Array
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str | None = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    metric: MeshMetricField | None


def _nan_extreme(values: np.ndarray, reducer: Any, /) -> float:
    finite = values[~np.isnan(values)]
    return float(reducer(finite)) if finite.size else float("nan")


class CellQualityReport(StrictModule, NonTrainableState):
    """Summary of source-map or explicitly straight-corner sampled quality.

    The evaluation's geometry layout distinguishes its actual source-map
    measurements from corner-only measurements. This is quality evidence only:
    mapped-cell validity is owned by the Bernstein validity certificate and
    global embedding by geometry mesh certificates.
    ``sampled_invalid_cell_global_ids`` lists failed sampled Jacobian frames.
    Empty sampled arrays are admitted only with an actual zero-resident
    distributed mesh. Their extrema are the neutral reductions over an empty
    local set, not measurements or a global quality certificate.
    """

    evaluation: CellQualityEvaluation
    topology_id: str = eqx.field(static=True)
    minimum_measure: float = eqx.field(static=True)
    maximum_measure: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    minimum_angle: float = eqx.field(static=True)
    maximum_angle: float = eqx.field(static=True)
    maximum_warpage: float = eqx.field(static=True)
    minimum_metric_quality: float = eqx.field(static=True)
    sampled_invalid_count: int = eqx.field(static=True)
    sampled_invalid_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    worst_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, evaluation: CellQualityEvaluation, /, *, mesh: CellMesh | None = None
    ) -> None:
        measures = np.asarray(evaluation.measures, dtype=np.float64)
        ratios = np.asarray(evaluation.mean_ratios, dtype=np.float64)
        aspects = np.asarray(evaluation.aspect_ratios, dtype=np.float64)
        scaled = np.asarray(evaluation.scaled_jacobian, dtype=np.float64)
        condition = np.asarray(evaluation.condition_number, dtype=np.float64)
        minimum_angle = np.asarray(evaluation.minimum_angle, dtype=np.float64)
        maximum_angle = np.asarray(evaluation.maximum_angle, dtype=np.float64)
        warpage = np.asarray(evaluation.warpage, dtype=np.float64)
        metric = np.asarray(evaluation.metric_quality, dtype=np.float64)
        valid = np.asarray(evaluation.sampled_valid, dtype=np.bool_)
        identifiers = np.asarray(evaluation.cell_global_ids, dtype=np.int64)
        if identifiers.size == 0:
            if (
                mesh is None
                or mesh.storage is None
                or (
                    mesh.storage.global_entity_counts[mesh.topological_dimension] <= 0
                    or mesh.coordinates.shape[0] != 0
                    or any(block.global_ids.shape[0] != 0 for block in mesh.blocks)
                    or evaluation.topology_id != mesh.topology_id
                    or evaluation.block_names
                    != tuple(block.name for block in mesh.blocks)
                    or evaluation.block_offsets
                    != tuple(0 for _ in range(len(mesh.blocks) + 1))
                    or any(
                        values.shape != (0,)
                        for values in (
                            measures,
                            ratios,
                            aspects,
                            scaled,
                            condition,
                            minimum_angle,
                            maximum_angle,
                            warpage,
                            metric,
                            valid,
                        )
                    )
                )
            ):
                raise ValueError(
                    "Empty sampled quality requires an exact zero-resident globally nonempty mesh."
                )
        order = np.argsort(np.where(valid, ratios, -np.inf), kind="stable")
        self.evaluation = evaluation
        self.topology_id = evaluation.topology_id
        self.minimum_measure = float(np.min(measures, initial=np.inf))
        self.maximum_measure = float(np.max(measures, initial=-np.inf))
        self.minimum_mean_ratio = float(np.min(ratios, initial=np.inf))
        self.maximum_aspect_ratio = float(np.max(aspects, initial=-np.inf))
        self.minimum_scaled_jacobian = float(np.min(scaled, initial=np.inf))
        self.maximum_condition_number = float(np.max(condition, initial=-np.inf))
        self.minimum_angle = _nan_extreme(minimum_angle, np.min)
        self.maximum_angle = _nan_extreme(maximum_angle, np.max)
        self.maximum_warpage = float(np.max(warpage, initial=-np.inf))
        self.minimum_metric_quality = _nan_extreme(metric, np.min)
        self.sampled_invalid_count = int(np.count_nonzero(~valid))
        self.sampled_invalid_cell_global_ids = tuple(
            int(value) for value in identifiers[~valid]
        )
        self.worst_cell_global_ids = tuple(
            int(identifiers[index]) for index in order[: min(10, order.size)]
        )
        self.report_id = canonical_fingerprint(
            {
                "kind": "cell-quality-report",
                "evaluation": evaluation.evaluation_id,
                "fields": [
                    array_tree_fingerprint(value)
                    for value in (
                        measures,
                        ratios,
                        aspects,
                        scaled,
                        condition,
                        minimum_angle,
                        maximum_angle,
                        warpage,
                        metric,
                        valid,
                    )
                ],
            }
        )


class _BlockQuality(NamedTuple):
    measure: Array
    mean_ratio: Array
    aspect: Array
    radius_ratio: Array
    scaled_jacobian: Array
    minimum_angle: Array
    maximum_angle: Array
    condition: Array
    sliver: Array
    warpage: Array
    metric_quality: Array
    sampled_valid: Array


# Static reference tables -----------------------------------------------------------


def _ideal_vertices(kind: str, arity: int, /) -> np.ndarray:
    root3 = np.sqrt(3.0)
    match kind:
        case "triangle":
            return np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 0.5 * root3)))
        case "polygon":
            angles = 2.0 * np.pi * np.arange(arity) / arity
            return np.stack((np.cos(angles), np.sin(angles)), axis=1)
        case "tetrahedron":
            return np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.5, 0.5 * root3, 0.0),
                    (0.5, root3 / 6.0, np.sqrt(2.0 / 3.0)),
                )
            )
        case "prism":
            base = _ideal_vertices("triangle", 3)
            return np.concatenate(
                (
                    np.pad(base, ((0, 0), (0, 1))),
                    np.pad(base, ((0, 0), (0, 1)), constant_values=1.0),
                )
            )
        case "pyramid":
            vertices = np.asarray(reference_cell_topology("pyramid").vertices)
            vertices[4, 2] = np.sqrt(0.5)
            return vertices
        case "quadrilateral" | "hexahedron":
            return np.asarray(reference_cell_topology(kind).vertices)
        case _:
            raise ValueError(f"No ideal reference shape for {kind!r}.")


def _faces(kind: str, arity: int, /) -> tuple[tuple[int, ...], ...]:
    if kind == "polygon":
        return (tuple(range(arity)),)
    return tuple(reference_cell_topology(kind).entities[2])


def _edges(kind: str, arity: int, /) -> tuple[tuple[int, int], ...]:
    if kind == "polygon":
        return tuple((index, (index + 1) % arity) for index in range(arity))
    return tuple(
        (int(start), int(stop))
        for start, stop in reference_cell_topology(kind).entities[1]
    )


def _square_determinant(matrix: Any, /) -> Any:
    size = matrix.shape[-1]
    if size == 1:
        return matrix[..., 0, 0]
    if size == 2:
        return (
            matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
        )
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def _adjugate_trace(matrix: Any, /) -> Any:
    if matrix.shape[-1] == 2:
        return matrix[..., 0, 0] + matrix[..., 1, 1]
    return (
        matrix[..., 0, 0] * matrix[..., 1, 1]
        - matrix[..., 0, 1] * matrix[..., 1, 0]
        + matrix[..., 0, 0] * matrix[..., 2, 2]
        - matrix[..., 0, 2] * matrix[..., 2, 0]
        + matrix[..., 1, 1] * matrix[..., 2, 2]
        - matrix[..., 1, 2] * matrix[..., 2, 1]
    )


@dataclass(frozen=True)
class _CornerTable:
    """Corner Jacobian frames ordered to be positive on the reference cell."""

    vertices: np.ndarray
    neighbors: np.ndarray
    ideal_inverse: np.ndarray
    ideal_determinant: np.ndarray
    ideal_scaled: np.ndarray


@cache
def _corner_table(kind: str, arity: int, /) -> _CornerTable:
    ideal = _ideal_vertices(kind, arity)
    if kind == "polygon":
        vertices = np.arange(arity)
        neighbors = np.stack(((vertices + 1) % arity, (vertices - 1) % arity), axis=1)
    else:
        vertices, neighbors = reference_corner_frames(kind)
    frames = np.swapaxes(ideal[neighbors] - ideal[vertices][:, None, :], -1, -2)
    determinant = np.linalg.det(frames)
    if np.any(determinant <= 0.0):
        raise ValueError(f"Reference corner frames of {kind!r} are not positive.")
    identity = np.broadcast_to(np.eye(frames.shape[-1]), frames.shape)
    inverse = np.linalg.solve(frames, identity)
    scaled = determinant / np.prod(np.linalg.norm(frames, axis=-2), axis=-1)
    return _CornerTable(vertices, neighbors, inverse, determinant, scaled)


@dataclass(frozen=True)
class _DihedralTable:
    """Edge samples ``(i, j, k, l)``: interior angle about i->j from k to l."""

    samples: np.ndarray


@cache
def _dihedral_table(kind: str, /) -> _DihedralTable:
    ideal = _ideal_vertices(kind, 0)
    faces = _faces(kind, 0)
    samples = []
    for start, stop in _edges(kind, 0):
        incident = [face for face in faces if start in face and stop in face]
        for origin, target in ((start, stop), (stop, start)):
            others = []
            for face in incident:
                position = face.index(origin)
                before = face[position - 1]
                after = face[(position + 1) % len(face)]
                others.append(after if before == target else before)
            first, second = others
            edge = ideal[target] - ideal[origin]
            cross = np.cross(ideal[first] - ideal[origin], ideal[second] - ideal[origin])
            if np.dot(cross, edge) < 0.0:
                first, second = second, first
            samples.append((origin, target, first, second))
    return _DihedralTable(np.asarray(samples))


def _first_order_space(kind: str, points: np.ndarray, /) -> Any:
    """Values and gradients of a spanning set of the first-order geometry space."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    one, zero = np.ones_like(x), np.zeros_like(x)
    match kind:
        case "hexahedron":
            values = (one, x, y, z, x * y, y * z, x * z, x * y * z)
            gradients = (
                (zero, zero, zero),
                (one, zero, zero),
                (zero, one, zero),
                (zero, zero, one),
                (y, x, zero),
                (zero, z, y),
                (z, zero, x),
                (y * z, x * z, x * y),
            )
        case "prism":
            values = (one, x, y, z, x * z, y * z)
            gradients = (
                (zero, zero, zero),
                (one, zero, zero),
                (zero, one, zero),
                (zero, zero, one),
                (z, zero, x),
                (zero, z, y),
            )
        case "pyramid":
            # Rational pyramid space: the apex term is (x - 1/2)(y - 1/2) / (1 - z),
            # which vanishes at the apex (x = y = 1/2, z = 1).
            scale = np.where(z < 1.0, 1.0 - z, 1.0)
            first, second = x - 0.5, y - 0.5
            values = (one, x, y, z, first * second / scale)
            gradients = (
                (zero, zero, zero),
                (one, zero, zero),
                (zero, one, zero),
                (zero, zero, one),
                (second / scale, first / scale, first * second / scale**2),
            )
        case _:
            raise ValueError(f"No first-order geometry space for {kind!r}.")
    return np.stack(values, axis=1), np.stack(
        [np.stack(row, axis=-1) for row in gradients], axis=1
    )


@dataclass(frozen=True)
class _MeasureRule:
    gradients: np.ndarray
    weights: np.ndarray


@cache
def _measure_rule(kind: str, /) -> _MeasureRule:
    """Exact quadrature of the multilinear Jacobian determinant.

    The nodal gradients are host NumPy data: the rule may first be requested
    under a JAX transformation, where a JAX tabulation would be traced.
    """

    gauss = np.asarray((0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)))
    gauss_weights = np.asarray((0.5, 0.5))
    match kind:
        case "hexahedron":
            points = np.stack(np.meshgrid(gauss, gauss, gauss, indexing="ij"), -1)
            points = points.reshape(-1, 3)
            weights = np.prod(
                np.stack(
                    np.meshgrid(
                        gauss_weights, gauss_weights, gauss_weights, indexing="ij"
                    ),
                    -1,
                ),
                axis=-1,
            ).reshape(-1)
        case "prism":
            points = np.stack(
                (np.full(2, 1.0 / 3.0), np.full(2, 1.0 / 3.0), gauss), axis=1
            )
            weights = 0.5 * gauss_weights
        case "pyramid":
            collapsed = np.stack(np.meshgrid(gauss, gauss, gauss, indexing="ij"), -1)
            collapsed = collapsed.reshape(-1, 3)
            height = collapsed[:, 2:3]
            points = np.concatenate(
                (collapsed[:, :2] * (1.0 - height) + 0.5 * height, height), axis=1
            )
            weights = np.full(8, 0.125) * (1.0 - collapsed[:, 2]) ** 2
        case _:
            raise ValueError(f"No quadrature measure rule for {kind!r}.")
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    vandermonde, _ = _first_order_space(kind, vertices)
    _, spanning = _first_order_space(kind, points)
    count = vandermonde.shape[0]
    # Nodal gradients solve V^T G_nodal = G_spanning for every point and axis.
    nodal = np.linalg.solve(
        vandermonde.T, np.moveaxis(spanning, 1, 0).reshape((count, -1))
    )
    gradients = np.moveaxis(nodal.reshape((count, points.shape[0], 3)), 0, 1)
    return _MeasureRule(gradients, weights)


# Quality kernels -----------------------------------------------------------------


def _tiny(values: Array, /) -> Array:
    # ty: ignore[invalid-return-type]
    return jnp.finfo(values.dtype).tiny


def _angle(sine: Array, cosine: Array, /) -> Array:
    angle = jnp.arctan2(sine, cosine)
    return jnp.where(angle < 0.0, angle + 2.0 * jnp.pi, angle)


def _vector_area(points: Array, /) -> Array:
    """Vector area (Newell) of cyclic 2-cells, axis -2 cycling vertices."""

    center = jnp.mean(points, axis=-2, keepdims=True)
    shifted = points - center
    return 0.5 * jnp.sum(jnp.cross(shifted, jnp.roll(shifted, -1, axis=-2)), axis=-2)


def _edge_aspect(
    points: Array, edges: tuple[tuple[int, int], ...], /
) -> tuple[Array, Array]:
    index = np.asarray(edges)
    lengths = jnp.linalg.norm(points[:, index[:, 1]] - points[:, index[:, 0]], axis=-1)
    return jnp.max(lengths, axis=1) / jnp.maximum(
        jnp.min(lengths, axis=1), _tiny(points)
    ), lengths


def _fan_warpage(points: Array, normal: Array, /) -> Array:
    """Largest angle between centroid-fan triangle normals and ``normal``."""

    center = jnp.mean(points, axis=-2, keepdims=True)
    shifted = points - center
    fan = jnp.cross(shifted, jnp.roll(shifted, -1, axis=-2))
    unit = normal[..., None, :]
    sine = jnp.linalg.norm(jnp.cross(fan, unit), axis=-1)
    cosine = jnp.sum(fan * unit, axis=-1)
    return jnp.max(jnp.arctan2(sine, cosine), axis=-1)


def _corner_measures(
    points: Array,
    table: _CornerTable,
    metric: Array | None,
    unit_normal: Array | None,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Corner samples: signed determinant, scaled Jacobian, mean ratio,
    condition number, and metric mean ratio, each shaped (cells, corners)."""

    frames = jnp.swapaxes(
        points[:, table.neighbors] - points[:, table.vertices][:, :, None, :], -1, -2
    )
    dimension = frames.shape[-1]
    if unit_normal is None:
        determinant = _square_determinant(frames)
    else:
        cross = jnp.cross(frames[..., 0], frames[..., 1])
        determinant = jnp.sum(cross * unit_normal[:, None, :], axis=-1)
    lengths = jnp.linalg.norm(frames, axis=-2)
    scaled = (
        determinant
        / jnp.maximum(jnp.prod(lengths, axis=-1), _tiny(points))
        / table.ideal_scaled
    )
    target = ein.contract("ckad,kde->ckae", frames, table.ideal_inverse, backend="jax")
    gram = ein.contract("ckad,ckae->ckde", target, target, backend="jax")
    positive = determinant > 0.0
    gram_determinant = jnp.maximum(_square_determinant(gram), _tiny(points))
    trace = jnp.trace(gram, axis1=-2, axis2=-1)
    ratio = jnp.where(
        positive,
        dimension
        * gram_determinant ** (1.0 / dimension)
        / jnp.maximum(trace, _tiny(points)),
        0.0,
    )
    condition = jnp.where(
        positive,
        jnp.sqrt(trace * _adjugate_trace(gram) / gram_determinant) / dimension,
        jnp.inf,
    )
    if metric is None:
        metric_ratio = jnp.full(ratio.shape, jnp.nan, dtype=points.dtype)
    else:
        metric_gram = ein.contract(
            "ckad,cab,ckbe->ckde", target, metric, target, backend="jax"
        )
        metric_determinant = jnp.maximum(_square_determinant(metric_gram), _tiny(points))
        metric_ratio = jnp.where(
            positive,
            dimension
            * metric_determinant ** (1.0 / dimension)
            / jnp.maximum(jnp.trace(metric_gram, axis1=-2, axis2=-1), _tiny(points)),
            0.0,
        )
    return determinant, scaled, ratio, condition, metric_ratio


def _interval_quality(points: Array, metric: Array | None, /) -> _BlockQuality:
    length = jnp.linalg.norm(points[:, 1] - points[:, 0], axis=-1)
    one = jnp.ones_like(length)
    nan = jnp.full_like(length, jnp.nan)
    return _BlockQuality(
        length,
        one,
        one,
        nan,
        one,
        nan,
        nan,
        one,
        nan,
        jnp.zeros_like(length),
        nan if metric is None else one,
        jnp.isfinite(length) & (length > 0.0),
    )


def _surface_quality(kind: str, points: Array, metric: Array | None, /) -> _BlockQuality:
    """Triangles, quadrilaterals, and polygons in 2-D or embedded in 3-D."""

    arity = points.shape[1]
    table = _corner_table(kind, arity)
    embedded = points.shape[-1] == 3
    if embedded:
        area_vector = _vector_area(points)
        area = jnp.linalg.norm(area_vector, axis=-1)
        unit_normal = area_vector / jnp.maximum(area, _tiny(points))[:, None]
        measure = area
        warpage = (
            jnp.zeros_like(area)
            if kind == "triangle"
            else _fan_warpage(points, unit_normal)
        )
    else:
        unit_normal = None
        shifted = points - jnp.mean(points, axis=1, keepdims=True)
        following = jnp.roll(shifted, -1, axis=1)
        fan = shifted[..., 0] * following[..., 1] - shifted[..., 1] * following[..., 0]
        measure = 0.5 * jnp.sum(fan, axis=1)
        warpage = jnp.zeros_like(measure)
    determinant, scaled, ratio, condition, metric_ratio = _corner_measures(
        points, table, metric, unit_normal
    )
    frames = points[:, table.neighbors] - points[:, table.vertices][:, :, None, :]
    cosine = jnp.sum(frames[:, :, 0] * frames[:, :, 1], axis=-1)
    angles = _angle(determinant, cosine)
    aspect, lengths = _edge_aspect(points, _edges(kind, arity))
    if kind == "triangle":
        magnitude = jnp.abs(measure)
        perimeter = jnp.sum(lengths, axis=1)
        radius = (
            16.0
            * magnitude**2
            / jnp.maximum(perimeter * jnp.prod(lengths, axis=1), _tiny(points))
        )
        valid = measure > 0.0
    elif kind == "quadrilateral":
        radius = jnp.full_like(measure, jnp.nan)
        valid = jnp.all(determinant > 0.0, axis=1)
    else:
        radius = jnp.full_like(measure, jnp.nan)
        center = jnp.mean(points, axis=1, keepdims=True)
        shifted = points - center
        following = jnp.roll(shifted, -1, axis=1)
        if embedded:
            star = jnp.sum(
                # ty: ignore[not-subscriptable]
                jnp.cross(shifted, following) * unit_normal[:, None, :],
                axis=-1,
            )
        else:
            star = (
                shifted[..., 0] * following[..., 1] - shifted[..., 1] * following[..., 0]
            )
        valid = jnp.all(star > 0.0, axis=1)
    return _BlockQuality(
        measure,
        jnp.min(ratio, axis=1),
        aspect,
        radius,
        jnp.min(scaled, axis=1),
        jnp.min(angles, axis=1),
        jnp.max(angles, axis=1),
        jnp.max(condition, axis=1),
        jnp.full_like(measure, jnp.nan),
        warpage,
        jnp.min(metric_ratio, axis=1),
        valid & jnp.isfinite(measure),
    )


def _dihedral_angles(points: Array, samples: np.ndarray, /) -> Array:
    origin = points[:, samples[:, 0]]
    edge = points[:, samples[:, 1]] - origin
    first = points[:, samples[:, 2]] - origin
    second = points[:, samples[:, 3]] - origin
    return _dihedral_vectors(edge, first, second)


def _dihedral_vectors(edge: Array, first: Array, second: Array, /) -> Array:
    unit = edge / jnp.maximum(jnp.linalg.norm(edge, axis=-1, keepdims=True), _tiny(edge))
    sine = jnp.sum(jnp.cross(first, second) * unit, axis=-1)
    cosine = jnp.sum(first * second, axis=-1) - jnp.sum(first * unit, axis=-1) * jnp.sum(
        second * unit, axis=-1
    )
    return _angle(sine, cosine)


def _volume_measure(kind: str, points: Array, /) -> Array:
    if kind == "tetrahedron":
        frames = jnp.swapaxes(points[:, 1:] - points[:, :1], -1, -2)
        return _square_determinant(frames) / 6.0
    rule = _measure_rule(kind)
    jacobian = ein.contract("qnk,cna->cqak", rule.gradients, points, backend="jax")
    return jnp.sum(_square_determinant(jacobian) * rule.weights, axis=1)


def _tetrahedron_shape(points: Array, volume: Array, lengths: Array, /) -> Any:
    faces = np.asarray(reference_cell_topology("tetrahedron").entities[2])
    face_areas = 0.5 * jnp.linalg.norm(
        jnp.cross(
            points[:, faces[:, 1]] - points[:, faces[:, 0]],
            points[:, faces[:, 2]] - points[:, faces[:, 0]],
        ),
        axis=-1,
    )
    # Opposite edge pairs of the reference edge order (01,12,20,03,13,23).
    first = lengths[:, 0] * lengths[:, 5]
    second = lengths[:, 1] * lengths[:, 3]
    third = lengths[:, 2] * lengths[:, 4]
    product = (
        (first + second + third)
        * (first + second - third)
        * (first - second + third)
        * (-first + second + third)
    )
    radius = (
        216.0
        * volume**2
        / jnp.maximum(
            jnp.sum(face_areas, axis=1) * jnp.sqrt(jnp.maximum(product, _tiny(points))),
            _tiny(points),
        )
    )
    shortest = jnp.min(lengths, axis=1)
    sliver = 6.0 * jnp.sqrt(2.0) * volume / jnp.maximum(shortest**3, _tiny(points))
    return radius, sliver


def _solid_quality(kind: str, points: Array, metric: Array | None, /) -> _BlockQuality:
    table = _corner_table(kind, points.shape[1])
    determinant, scaled, ratio, condition, metric_ratio = _corner_measures(
        points, table, metric, None
    )
    measure = _volume_measure(kind, points)
    aspect, lengths = _edge_aspect(points, _edges(kind, points.shape[1]))
    angles = _dihedral_angles(points, _dihedral_table(kind).samples)
    quadrilaterals = tuple(face for face in _faces(kind, 0) if len(face) == 4)
    if quadrilaterals:
        face_points = points[:, np.asarray(quadrilaterals)]
        area = _vector_area(face_points)
        unit = area / jnp.maximum(
            jnp.linalg.norm(area, axis=-1, keepdims=True), _tiny(points)
        )
        warpage = jnp.max(_fan_warpage(face_points, unit), axis=1)
    else:
        warpage = jnp.zeros_like(measure)
    if kind == "tetrahedron":
        radius, sliver = _tetrahedron_shape(points, measure, lengths)
    else:
        radius = jnp.full_like(measure, jnp.nan)
        sliver = jnp.full_like(measure, jnp.nan)
    return _BlockQuality(
        measure,
        jnp.min(ratio, axis=1),
        aspect,
        radius,
        jnp.min(scaled, axis=1),
        jnp.min(angles, axis=1),
        jnp.max(angles, axis=1),
        jnp.max(condition, axis=1),
        sliver,
        warpage,
        jnp.min(metric_ratio, axis=1),
        jnp.all(determinant > 0.0, axis=1) & (measure > 0.0) & jnp.isfinite(measure),
    )


def _standard_block_quality(
    kind: str, points: Array, metric: Array | None, /
) -> _BlockQuality:
    match kind:
        case "interval":
            return _interval_quality(points, metric)
        case "triangle" | "quadrilateral" | "polygon":
            return _surface_quality(kind, points, metric)
        case "tetrahedron" | "hexahedron" | "prism" | "pyramid":
            return _solid_quality(kind, points, metric)
        case _:
            raise ValueError(f"No native quality implementation for {kind!r}.")


def _mapped_face_warpage(
    kind: str,
    element: CellGeometryElement,
    coordinates: Array,
    /,
) -> Array:
    """Sample the actual mapped face normals, including curved interior faces."""
    source = _require_scalar_coordinate_element(element, "Mapped quality")
    dimension, ambient = source.topological_dimension, coordinates.shape[-1]
    if dimension < 2 or ambient < 3:
        return jnp.zeros((coordinates.shape[0],), dtype=coordinates.dtype)
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    faces = (
        _faces(kind, vertices.shape[0])
        if dimension == 3
        else (tuple(range(vertices.shape[0])),)
    )
    angles = []
    for face in faces:
        points = vertices[np.asarray(face, dtype=np.int64)]
        # Edge midpoints stay in the owning face chart without asking a
        # collapsed pyramid apex for a nonexistent unique tangent plane.
        probes = np.concatenate(
            (
                np.mean(points, axis=0, keepdims=True),
                0.5 * (points + np.roll(points, -1, axis=0)),
            )
        )
        charge_native_geometry_queries(
            coordinates.shape[0] * probes.shape[0],
            work_units=coordinates.shape[0] * probes.shape[0],
        )
        _, gradients = source.tabulate(jnp.asarray(probes, dtype=coordinates.dtype))
        jacobian = ein.contract("qir,cia->cqar", gradients, coordinates, backend="jax")
        directions = np.stack((points[1] - points[0], points[-1] - points[0]), axis=1)
        tangents = ein.contract("cqar,rs->cqas", jacobian, directions, backend="jax")
        normals = jnp.cross(tangents[..., 0], tangents[..., 1])
        reference = normals[:, :1]
        sine = jnp.linalg.norm(jnp.cross(reference, normals[:, 1:]), axis=-1)
        cosine = jnp.sum(reference * normals[:, 1:], axis=-1)
        angles.append(jnp.max(jnp.arctan2(sine, cosine), axis=1))
    return jnp.max(jnp.stack(angles, axis=1), axis=1)


def _coordinate_total_order(element: CellGeometryElement, /) -> int:
    """Static source-space bound, including affine mixing of tensor roots."""
    source = _require_scalar_coordinate_element(element, "Mapped quality order")
    if isinstance(source, RestrictedCellGeometryElement):
        return _coordinate_total_order(source.source_element)
    if isinstance(source, LayerColumnCellGeometryElement):
        return max(2, _coordinate_total_order(source.wall_element) + 1)
    if isinstance(source, SplineCellGeometryElement):
        return source.u_degree + source.v_degree
    if source.cell_kind in (
        "interval",
        "triangle",
        "tetrahedron",
    ) or source.cell_kind.startswith("simplex:"):
        return source.degree
    return (
        2 if source.cell_kind == "prism" else source.topological_dimension
    ) * source.degree


def _mapped_boundary_angles(
    kind: str,
    element: CellGeometryElement,
    coordinates: Array,
    /,
) -> tuple[Array, Array]:
    """Actual corner tangents in 2D and sampled boundary-edge dihedrals in 3D."""
    source = _require_scalar_coordinate_element(element, "Mapped boundary quality")
    dimension = source.topological_dimension
    if dimension == 1:
        empty = jnp.full((coordinates.shape[0],), jnp.nan, dtype=coordinates.dtype)
        return empty, empty
    vertices = jnp.asarray(
        reference_cell_topology(kind).vertices, dtype=coordinates.dtype
    )
    if dimension == 2:
        charge_native_geometry_queries(
            coordinates.shape[0] * vertices.shape[0],
            work_units=coordinates.shape[0] * vertices.shape[0],
        )
        _, gradients = source.tabulate(vertices)
        jacobian = ein.contract("qir,cia->cqar", gradients, coordinates, backend="jax")
        directions = jnp.stack(
            (
                jnp.roll(vertices, -1, axis=0) - vertices,
                jnp.roll(vertices, 1, axis=0) - vertices,
            ),
            axis=-1,
        )
        tangents = ein.contract("cqar,qrs->cqas", jacobian, directions, backend="jax")
        cosine = jnp.sum(tangents[..., 0] * tangents[..., 1], axis=-1)
        if coordinates.shape[-1] == 2:
            sine = determinant_small_linear(SmallLinearSolvePlan(2), tangents)
        else:
            normal = jnp.cross(jacobian[..., 0], jacobian[..., 1])
            normal = normal / jnp.maximum(
                jnp.linalg.norm(normal, axis=-1, keepdims=True), _tiny(coordinates)
            )
            sine = jnp.sum(
                jnp.cross(tangents[..., 0], tangents[..., 1]) * normal, axis=-1
            )
        angles = _angle(sine, cosine)
        return jnp.min(angles, axis=1), jnp.max(angles, axis=1)
    samples = _dihedral_table(kind).samples
    first, last = vertices[samples[:, 0]], vertices[samples[:, 1]]
    axis, _ = _degree_aware_reference_rule("interval", 2)
    probes = (first[:, None] * (1.0 - axis[None]) + last[:, None] * axis[None]).reshape(
        -1, dimension
    )
    directions = jnp.stack(
        (last - first, vertices[samples[:, 2]] - first, vertices[samples[:, 3]] - first),
        axis=-1,
    )
    directions = jnp.repeat(directions, axis.shape[0], axis=0)
    charge_native_geometry_queries(
        coordinates.shape[0] * probes.shape[0],
        work_units=coordinates.shape[0] * probes.shape[0],
    )
    _, gradients = source.tabulate(probes)
    low = jnp.full((coordinates.shape[0],), jnp.inf, dtype=coordinates.dtype)
    high = jnp.full((coordinates.shape[0],), -jnp.inf, dtype=coordinates.dtype)

    def sample(
        bounds: tuple[Array, Array],
        probe: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        gradient, reference = probe
        jacobian = ein.contract("ir,cia->car", gradient, coordinates, backend="jax")
        vectors = jacobian @ reference
        angle = _dihedral_vectors(vectors[..., 0], vectors[..., 1], vectors[..., 2])
        return (jnp.minimum(bounds[0], angle), jnp.maximum(bounds[1], angle)), None

    result, _ = jax.lax.scan(sample, (low, high), (gradients, directions))
    return result


def _mapped_block_quality(
    kind: str,
    element: CellGeometryElement,
    coordinates: Array,
    metric: Array | None,
    /,
) -> _BlockQuality:
    """Measure the actual coordinate map and its sampled Jacobian metrics.

    Jacobian images of the canonical reference shape define local metric
    quality; they are never a replacement coordinate source or a fitted cell.
    Positive-weight source quadrature is exact for polynomial signed volume
    under its declared degree bound. Rational maps retain sampled integration
    semantics, independently of their owning whole-map validity certificate.
    """
    source = _require_scalar_coordinate_element(element, "Mapped quality")
    dimension = source.topological_dimension
    order = (
        _coordinate_total_order(source)
        if isinstance(source, RestrictedCellGeometryElement)
        else source.degree
    )
    count = max(2, (dimension * order + 1) // 2)
    rule_degree = max(0, count - (2 if kind == "tetrahedron" else 1))
    probes, weights = _degree_aware_reference_rule(kind, rule_degree)
    charge_native_geometry_queries(
        coordinates.shape[0] * probes.shape[0],
        work_units=coordinates.shape[0] * probes.shape[0],
    )
    _, gradients = source.tabulate(probes)
    plan = SmallLinearSolvePlan(dimension)
    reference = jnp.asarray(
        reference_cell_topology(kind).vertices, dtype=coordinates.dtype
    )
    count = coordinates.shape[0]
    low = jnp.full((count,), jnp.inf, dtype=coordinates.dtype)
    high = jnp.full((count,), -jnp.inf, dtype=coordinates.dtype)
    zero = jnp.zeros((count,), dtype=coordinates.dtype)
    initial = _BlockQuality(
        zero,
        low,
        high,
        low,
        low,
        zero,
        zero,
        high,
        low,
        zero,
        low,
        jnp.ones((count,), dtype=jnp.bool_),
    )

    def sample(
        total: _BlockQuality,
        probe: tuple[Array, Array],
    ) -> tuple[_BlockQuality, None]:
        gradient, weight = probe
        jacobian = ein.contract("ir,cia->car", gradient, coordinates, backend="jax")
        if coordinates.shape[-1] == dimension:
            density = determinant_small_linear(plan, jacobian)
        else:
            gram = ein.contract("car,cas->crs", jacobian, jacobian, backend="jax")
            density = jnp.sqrt(determinant_small_linear(plan, gram))
        metric_vertices = ein.contract("car,vr->cva", jacobian, reference, backend="jax")
        quality = _standard_block_quality(kind, metric_vertices, metric)
        return _BlockQuality(
            total.measure + weight * density,
            jnp.minimum(total.mean_ratio, quality.mean_ratio),
            jnp.maximum(total.aspect, quality.aspect),
            jnp.minimum(total.radius_ratio, quality.radius_ratio),
            jnp.minimum(total.scaled_jacobian, quality.scaled_jacobian),
            total.minimum_angle,
            total.maximum_angle,
            jnp.maximum(total.condition, quality.condition),
            jnp.minimum(total.sliver, quality.sliver),
            total.warpage,
            jnp.minimum(total.metric_quality, quality.metric_quality),
            total.sampled_valid
            & quality.sampled_valid
            & (density > 0.0)
            & jnp.isfinite(density),
        ), None

    # Sample lanes reduce directly to per-cell outputs: no cell×sample×corner
    # metric bank or sample-by-cell Jacobian bank is materialized.
    measured, _ = jax.lax.scan(sample, initial, (gradients, weights))
    minimum_angle, maximum_angle = _mapped_boundary_angles(kind, source, coordinates)
    return measured._replace(
        minimum_angle=minimum_angle,
        maximum_angle=maximum_angle,
        warpage=_mapped_face_warpage(kind, source, coordinates),
        sampled_valid=measured.sampled_valid
        & (measured.measure > 0.0)
        & jnp.isfinite(measured.measure),
    )


def _polyhedral_quality(
    tables: PolyhedralStarTables,
    cells: np.ndarray,
    coordinates: Array,
    metric: Array | None,
    /,
) -> _BlockQuality:
    """Segment-reduced polyhedral quality of connectivity rows ``cells``."""

    count = cells.size
    cell_total = tables.cell_vertex_counts.size
    face_total = tables.face_sizes.size
    center = (
        jax.ops.segment_sum(
            coordinates[tables.cell_vertex_values], tables.cell_vertex_cell, cell_total
        )
        / jnp.asarray(tables.cell_vertex_counts, dtype=coordinates.dtype)[:, None]
    )
    corner_points = coordinates[tables.face_corner_vertex]
    face_center = (
        jax.ops.segment_sum(corner_points, tables.face_corner_face, face_total)
        / jnp.asarray(tables.face_sizes, dtype=coordinates.dtype)[:, None]
    )
    face_area = 0.5 * jax.ops.segment_sum(
        jnp.cross(
            corner_points - face_center[tables.face_corner_face],
            coordinates[tables.face_corner_next] - face_center[tables.face_corner_face],
        ),
        tables.face_corner_face,
        face_total,
    )
    selected = np.flatnonzero(np.isin(tables.star_cell, cells))
    local = np.searchsorted(cells, tables.star_cell[selected])
    star_face = tables.star_face[selected]
    first = coordinates[tables.star_first[selected]]
    second = coordinates[tables.star_second[selected]]
    tetrahedra = jnp.stack(
        (
            center[tables.star_cell[selected]],
            face_center[star_face],
            first,
            second,
        ),
        axis=1,
    )
    star_metric = None if metric is None else metric[local]
    tetra = _solid_quality("tetrahedron", tetrahedra, star_metric)
    measure = jax.ops.segment_sum(tetra.measure, local, count)
    edge_lengths = jnp.linalg.norm(second - first, axis=-1)
    aspect = jax.ops.segment_max(edge_lengths, local, count) / jnp.maximum(
        jax.ops.segment_min(edge_lengths, local, count), _tiny(coordinates)
    )
    outward = jnp.asarray(tables.star_sign[selected], dtype=coordinates.dtype)[:, None]
    unit = face_area[star_face] * outward
    unit = unit / jnp.maximum(
        jnp.linalg.norm(unit, axis=-1, keepdims=True), _tiny(coordinates)
    )
    fan = jnp.cross(first - face_center[star_face], second - face_center[star_face])
    fan_angle = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(fan, unit), axis=-1), jnp.sum(fan * unit, axis=-1)
    )
    pairs = tables.edge_pairs[np.all(np.isin(tables.edge_pairs, selected), axis=1)]
    pair_first = np.searchsorted(selected, pairs[:, 0])
    pair_second = np.searchsorted(selected, pairs[:, 1])
    direction = second[pair_first] - first[pair_first]
    direction = direction / jnp.maximum(
        jnp.linalg.norm(direction, axis=-1, keepdims=True), _tiny(coordinates)
    )
    normal_first = unit[pair_first]
    normal_second = unit[pair_second]
    dihedral = jnp.pi - jnp.arctan2(
        jnp.sum(jnp.cross(normal_first, normal_second) * direction, axis=-1),
        jnp.sum(normal_first * normal_second, axis=-1),
    )
    pair_cell = local[pair_first]
    return _BlockQuality(
        measure,
        jax.ops.segment_min(tetra.mean_ratio, local, count),
        aspect,
        jnp.full((count,), jnp.nan, dtype=coordinates.dtype),
        jax.ops.segment_min(tetra.scaled_jacobian, local, count),
        jax.ops.segment_min(dihedral, pair_cell, count),
        jax.ops.segment_max(dihedral, pair_cell, count),
        jax.ops.segment_max(tetra.condition, local, count),
        jnp.full((count,), jnp.nan, dtype=coordinates.dtype),
        jax.ops.segment_max(fan_angle, local, count),
        (
            jnp.full((count,), jnp.nan, dtype=coordinates.dtype)
            if metric is None
            else jax.ops.segment_min(tetra.metric_quality, local, count)
        ),
        (jax.ops.segment_min(tetra.sampled_valid.astype(jnp.int32), local, count) > 0)
        & (measure > 0.0),
    )


def _vertex_metric(mesh: CellMesh, metric: MeshMetricField | None, /) -> Array | None:
    if metric is None:
        return None
    if not isinstance(metric, MeshMetricField):
        raise TypeError("metric must be MeshMetricField or None.")
    dimension = mesh.ambient_dimension
    if metric.values.shape != (mesh.coordinates.shape[0], dimension, dimension):
        raise ValueError("Quality metric must be one ambient tensor per mesh vertex.")
    scope = metric.scope
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.entity_set_id != mesh.entity_set(0).entity_set_id
    ):
        raise ValueError("Quality metric must bind the exact owning mesh vertex scope.")
    return eqx.error_if(
        metric.values,
        ~jnp.all(metric.scope.entity_ids == mesh.vertex_global_ids),
        "Quality metric must bind the actual source mesh vertex identities.",
    )


def _cell_metric(values: Array | None, rows: Array, /) -> Array | None:
    if values is None:
        return None
    weights = jnp.full(rows.shape, 1.0 / rows.shape[1], dtype=values.dtype)
    return interpolate_mesh_metric(values[rows], weights)


def _vertex_geometry_coordinates(
    mesh: CellMesh,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    bank: Array,
    /,
) -> Array:
    """Join vertex-source banks through declared cell incidences, not proximity."""
    if mesh.coordinates.shape[0] == 0:
        return mesh.coordinates
    vertices, source_rows, masks = [], [], []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if isinstance(element, CellVertexGeometryElement):
            vertices.append(block.vertices.reshape(-1))
            source_rows.append(route.reshape(-1))
            masks.append(block.vertex_valid.reshape(-1))
    if not vertices:
        return mesh.coordinates
    vertex = jnp.concatenate(vertices)
    rows = jnp.concatenate(source_rows)
    valid = jnp.concatenate(masks)
    safe_vertex, safe_row = jnp.maximum(vertex, 0), jnp.maximum(rows, 0)
    sentinel = jnp.asarray(bank.shape[0], dtype=rows.dtype)
    first = jax.ops.segment_min(
        jnp.where(valid, rows, sentinel),
        safe_vertex,
        mesh.coordinates.shape[0],
    )
    selected = bank[jnp.minimum(first, bank.shape[0] - 1)]
    points = jnp.where((first < sentinel)[:, None], selected, mesh.coordinates)
    shared = jnp.all(~valid | jnp.all(bank[safe_row] == points[safe_vertex], axis=-1))
    return eqx.error_if(
        points, ~shared, "Shared vertex coordinate sources disagree exactly."
    )


def evaluate_cell_quality(
    mesh: CellMesh,
    coordinates: ArrayLike | None = None,
    /,
    *,
    metric: MeshMetricField | None = None,
    geometry: CellGeometrySpec | None = None,
) -> CellQualityEvaluation:
    """Evaluate actual source-map quality or explicitly straight-corner quality.

    ``geometry`` retains its owning coordinate coefficients and tabulation,
    including high-order, restricted, composed and rational maps. The separate
    ``coordinates`` override deliberately evaluates a straight-corner mesh;
    supplying both authorities is ambiguous and refused.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if geometry is not None and not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec or None.")
    if geometry is not None and coordinates is not None:
        raise ValueError(
            "Quality requires one owning geometry or one explicit corner-coordinate override, not both."
        )
    points = mesh.coordinates if coordinates is None else jnp.asarray(coordinates)
    if points.shape != mesh.coordinates.shape:
        raise ValueError("Quality coordinates must preserve the mesh coordinate shape.")
    if not mesh.blocks and (
        mesh.storage is None
        or mesh.storage.global_entity_counts[mesh.topological_dimension] <= 0
        or points.shape[0] != 0
        or any(entities.count != 0 for entities in mesh.topology.entity_sets)
    ):
        raise ValueError(
            "An empty sampled cell set requires an actual zero-resident distributed mesh."
        )
    resolved = None if geometry is None else geometry.resolve(mesh)
    if resolved is not None:
        elements, routes, bank = resolved
        points = _vertex_geometry_coordinates(mesh, elements, routes, bank)
    vertex_metric = _vertex_metric(mesh, metric)
    tables = (
        # ty: ignore[invalid-argument-type]
        polyhedral_star_tables(mesh.connectivity)
        if any(block.cell_kind == "polyhedron" for block in mesh.blocks)
        else None
    )
    blocks = []
    offsets = [0]
    for block_index, block in enumerate(mesh.blocks):
        start = offsets[-1]
        stop = start + block.cell_count
        if block.cell_kind == "polyhedron":
            cells = np.arange(start, stop)
            cell_metric = (
                None
                if vertex_metric is None
                # ty: ignore[invalid-argument-type]
                else _polyhedral_cell_metric(vertex_metric, tables, cells)
            )
            # ty: ignore[invalid-argument-type]
            blocks.append(_polyhedral_quality(tables, cells, points, cell_metric))
        else:
            rows = block.vertices
            cell_metric = _cell_metric(vertex_metric, rows)
            if resolved is None:
                blocks.append(
                    _standard_block_quality(block.cell_kind, points[rows], cell_metric)
                )
            else:
                element = resolved[0][block_index]
                values = resolved[2][resolved[1][block_index]]
                if isinstance(element, CellVertexGeometryElement):
                    blocks.append(
                        _standard_block_quality(block.cell_kind, values, cell_metric)
                    )
                else:
                    blocks.append(
                        _mapped_block_quality(
                            block.cell_kind, element, values, cell_metric
                        )
                    )
        offsets.append(stop)
    if blocks:
        fields = _BlockQuality(
            *(jnp.concatenate(values) for values in zip(*blocks, strict=True))
        )
    else:
        empty = jnp.empty((0,), dtype=points.dtype)
        fields = _BlockQuality(
            measure=empty,
            mean_ratio=empty,
            aspect=empty,
            radius_ratio=empty,
            scaled_jacobian=empty,
            minimum_angle=empty,
            maximum_angle=empty,
            condition=empty,
            sliver=empty,
            warpage=empty,
            metric_quality=empty,
            sampled_valid=jnp.empty((0,), dtype=jnp.bool_),
        )
    evaluation_id = canonical_fingerprint(
        {
            "kind": "cell-quality-evaluation",
            "topology": mesh.topology_id,
            "block_ids": [block.block_id for block in mesh.blocks],
            "metric": None if metric is None else metric.metric_id,
            "geometry_layout": None if geometry is None else geometry.geometry_layout_id,
        }
    )
    return CellQualityEvaluation(
        cell_global_ids=(
            jnp.concatenate(tuple(block.global_ids for block in mesh.blocks))
            if mesh.blocks
            else jnp.empty((0,), dtype=jnp.int64)
        ),
        measures=fields.measure,
        mean_ratios=fields.mean_ratio,
        aspect_ratios=fields.aspect,
        radius_ratios=fields.radius_ratio,
        scaled_jacobian=fields.scaled_jacobian,
        minimum_angle=fields.minimum_angle,
        maximum_angle=fields.maximum_angle,
        condition_number=fields.condition,
        sliver_measures=fields.sliver,
        warpage=fields.warpage,
        metric_quality=fields.metric_quality,
        sampled_valid=fields.sampled_valid,
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=tuple(offsets),
        topology_id=mesh.topology_id,
        geometry_layout_id=None if geometry is None else geometry.geometry_layout_id,
        evaluation_id=evaluation_id,
        metric=metric,
    )


def _polyhedral_cell_metric(
    values: Array, tables: PolyhedralStarTables, cells: np.ndarray, /
) -> Array:
    selected = np.isin(tables.cell_vertex_cell, cells)
    owners = np.searchsorted(cells, tables.cell_vertex_cell[selected])
    counts = np.bincount(owners, minlength=cells.size)
    width = int(np.max(counts))
    rank = np.arange(owners.size) - np.repeat(np.cumsum(counts) - counts, counts)
    rows = np.zeros((cells.size, width), dtype=np.int64)
    weights = np.zeros((cells.size, width), dtype=np.float64)
    rows[owners, rank] = tables.cell_vertex_values[selected]
    weights[owners, rank] = 1.0 / counts[owners]
    return interpolate_mesh_metric(values[rows], jnp.asarray(weights, dtype=values.dtype))


def summarize_cell_quality(
    evaluation: CellQualityEvaluation,
    /,
    *,
    mesh: CellMesh | None = None,
) -> CellQualityReport:
    return CellQualityReport(evaluation, mesh=mesh)


# Straight sweep layers -------------------------------------------------------------


class SweptLayerQualityEvaluation(StrictModule, NonTrainableState):
    """Measured straight-sweep quality for one exact layer schedule."""

    measured_thicknesses: Array
    measured_growth_rates: Array
    maximum_thickness_residual: float = eqx.field(static=True)
    maximum_alignment_residual: float = eqx.field(static=True)
    maximum_interface_residual: float = eqx.field(static=True)
    valid: bool = eqx.field(static=True)


def evaluate_swept_layer_quality(
    points: ArrayLike,
    prisms: ArrayLike,
    layer_indices: ArrayLike,
    origin: ArrayLike,
    unit_direction: ArrayLike,
    requested_thicknesses: ArrayLike,
    /,
) -> SweptLayerQualityEvaluation:
    """Measure thickness, growth, axial alignment, and interface placement."""

    coordinates = np.asarray(points, dtype=np.float64)
    cells = np.asarray(prisms, dtype=np.int32)
    indices = np.asarray(layer_indices, dtype=np.int32)
    anchor = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(unit_direction, dtype=np.float64)
    requested = np.asarray(requested_thicknesses, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Sweep quality points must have shape (n, 3).")
    if (
        cells.ndim != 2
        or cells.shape[1] != 6
        or indices.shape != cells.shape[:1]
        or np.any(cells < 0)
        or np.any(cells >= coordinates.shape[0])
    ):
        raise ValueError(
            "Sweep quality requires one valid six-node prism index per cell."
        )
    if requested.ndim != 1 or requested.size == 0 or np.any(requested <= 0.0):
        raise ValueError("Requested sweep thicknesses must be one positive vector.")
    if np.any(indices < 0) or np.any(indices >= requested.size):
        raise ValueError("Sweep layer indices are outside the requested schedule.")
    norm = float(np.linalg.norm(direction))
    if (
        anchor.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(norm)
        or norm <= 0.0
    ):
        raise ValueError(
            "Sweep quality requires a finite nonzero three-vector direction."
        )
    unit = direction / norm
    values = coordinates[cells]
    axial_edges = values[:, 3:] - values[:, :3]
    projected_edges = axial_edges @ unit
    edge_lengths = np.linalg.norm(axial_edges, axis=2)
    transverse = axial_edges - projected_edges[:, :, None] * unit
    alignment = np.linalg.norm(transverse, axis=2) / np.maximum(
        edge_lengths, np.finfo(np.float64).tiny
    )
    levels = np.concatenate(([0.0], np.cumsum(requested)))
    projected_vertices = (values - anchor) @ unit
    expected_lower = levels[indices]
    expected_upper = levels[indices + 1]
    interface_residual = float(
        max(
            np.max(np.abs(projected_vertices[:, :3] - expected_lower[:, None])),
            np.max(np.abs(projected_vertices[:, 3:] - expected_upper[:, None])),
        )
    )
    measured = np.full(requested.shape, np.nan, dtype=np.float64)
    for layer in range(requested.size):
        selected = indices == layer
        if np.any(selected):
            measured[layer] = float(np.mean(projected_edges[selected]))
    growth = (
        measured[1:] / measured[:-1]
        if measured.size > 1
        else np.empty((0,), dtype=np.float64)
    )
    thickness_residual = float(np.max(np.abs(projected_edges - requested[indices, None])))
    alignment_residual = float(np.max(alignment))
    valid = bool(
        np.all(np.isfinite(measured))
        and np.all(measured > 0.0)
        and np.isfinite(thickness_residual)
        and np.isfinite(alignment_residual)
        and np.isfinite(interface_residual)
    )
    return SweptLayerQualityEvaluation(
        measured_thicknesses=jnp.asarray(measured),
        measured_growth_rates=jnp.asarray(growth),
        maximum_thickness_residual=thickness_residual,
        maximum_alignment_residual=alignment_residual,
        maximum_interface_residual=interface_residual,
        valid=valid,
    )


# Finite-volume face quality ------------------------------------------------------


class FiniteVolumeQualityEvaluation(StrictModule, NonTrainableState):
    """Interior-face skewness and non-orthogonality of a full-dimensional mesh.

    ``non_orthogonality`` is the angle (radians) between the owner-to-neighbor
    centroid vector and the face normal oriented out of the owner; values above
    pi/2 mean the neighbor centroid lies behind the face. ``skewness`` is the
    distance from the face centroid to the centroid line's face intersection,
    divided by the centroid distance. Owners are the lower cell entity rows.
    """

    face_global_ids: Array
    owner_cell_global_ids: Array
    neighbor_cell_global_ids: Array
    skewness: Array
    non_orthogonality: Array
    topology_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


def _block_centroids(kind: str, points: Array, /) -> tuple[Array, Array]:
    """Measure-weighted centroids of standard cells via star decomposition."""

    if kind in ("triangle", "quadrilateral", "polygon"):
        center = jnp.mean(points, axis=1, keepdims=True)
        following = jnp.roll(points, -1, axis=1)
        first = points - center
        second = following - center
        areas = 0.5 * (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
        centroids = (center + points + following) / 3.0
        return jnp.sum(areas[..., None] * centroids, axis=1), jnp.sum(areas, axis=1)
    center = jnp.mean(points, axis=1)
    volumes = []
    moments = []
    for face in _faces(kind, 0):
        loop = points[:, np.asarray(face)]
        face_center = jnp.mean(loop, axis=1)
        following = jnp.roll(loop, -1, axis=1)
        frames = jnp.stack(
            (
                jnp.broadcast_to((face_center - center)[:, None], loop.shape),
                loop - center[:, None],
                following - center[:, None],
            ),
            axis=-1,
        )
        volume = _square_determinant(frames) / 6.0
        centroid = (center[:, None] + face_center[:, None] + loop + following) / 4.0
        volumes.append(volume)
        moments.append(jnp.sum(volume[..., None] * centroid, axis=1))
    total = jnp.sum(jnp.concatenate(volumes, axis=1), axis=1)
    # ty: ignore[invalid-return-type]
    return sum(moments), total


def _polyhedral_centroids(
    tables: PolyhedralStarTables, cells: np.ndarray, coordinates: Array, /
) -> tuple[Array, Array]:
    cell_total = tables.cell_vertex_counts.size
    center = (
        jax.ops.segment_sum(
            coordinates[tables.cell_vertex_values], tables.cell_vertex_cell, cell_total
        )
        / jnp.asarray(tables.cell_vertex_counts, dtype=coordinates.dtype)[:, None]
    )
    face_center = (
        jax.ops.segment_sum(
            coordinates[tables.face_corner_vertex],
            tables.face_corner_face,
            tables.face_sizes.size,
        )
        / jnp.asarray(tables.face_sizes, dtype=coordinates.dtype)[:, None]
    )
    selected = np.flatnonzero(np.isin(tables.star_cell, cells))
    local = np.searchsorted(cells, tables.star_cell[selected])
    apex = center[tables.star_cell[selected]]
    base = face_center[tables.star_face[selected]]
    first = coordinates[tables.star_first[selected]]
    second = coordinates[tables.star_second[selected]]
    volume = (
        _square_determinant(
            jnp.stack((base - apex, first - apex, second - apex), axis=-1)
        )
        / 6.0
    )
    centroid = (apex + base + first + second) / 4.0
    return (
        jax.ops.segment_sum(volume[:, None] * centroid, local, cells.size),
        jax.ops.segment_sum(volume, local, cells.size),
    )


def _facet_geometry(mesh: CellMesh, points: Array, /) -> tuple[Array, Array]:
    """Facet centroids and area-weighted normals in facet entity row order."""

    connectivity = mesh.connectivity
    if mesh.topological_dimension == 2:
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError(
                "Planar finite-volume quality requires polygonal connectivity."
            )
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        start = points[edges[:, 0]]
        stop = points[edges[:, 1]]
        tangent = stop - start
        return 0.5 * (start + stop), jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    if isinstance(connectivity, PolyhedralConnectivity):
        tables = polyhedral_star_tables(connectivity)
        face_count = tables.face_sizes.size
        corners = points[tables.face_corner_vertex]
        following = points[tables.face_corner_next]
        center = (
            jax.ops.segment_sum(corners, tables.face_corner_face, face_count)
            / (jnp.asarray(tables.face_sizes, dtype=points.dtype)[:, None])
        )
        fan = 0.5 * jnp.cross(
            corners - center[tables.face_corner_face],
            following - center[tables.face_corner_face],
        )
        fan_area = jnp.linalg.norm(fan, axis=-1)
        fan_center = (center[tables.face_corner_face] + corners + following) / 3.0
        area = jax.ops.segment_sum(fan_area, tables.face_corner_face, face_count)
        centroid = (
            jax.ops.segment_sum(
                fan_area[:, None] * fan_center, tables.face_corner_face, face_count
            )
            / jnp.maximum(area, _tiny(points))[:, None]
        )
        return centroid, jax.ops.segment_sum(fan, tables.face_corner_face, face_count)
    if not isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
        raise TypeError("Finite-volume quality requires volume connectivity.")
    loops = points[np.asarray(connectivity.faces, dtype=np.int64)]
    center = jnp.mean(loops, axis=1, keepdims=True)
    following = jnp.roll(loops, -1, axis=1)
    fan = 0.5 * jnp.cross(loops - center, following - center)
    fan_area = jnp.linalg.norm(fan, axis=-1)
    fan_center = (center + loops + following) / 3.0
    centroid = (
        jnp.sum(fan_area[..., None] * fan_center, axis=1)
        / jnp.maximum(jnp.sum(fan_area, axis=1), _tiny(points))[:, None]
    )
    return centroid, jnp.sum(fan, axis=1)


def evaluate_finite_volume_quality(mesh: CellMesh, /) -> FiniteVolumeQualityEvaluation:
    """Evaluate skewness and non-orthogonality on every interior face."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    dimension = mesh.topological_dimension
    if dimension not in (2, 3) or mesh.ambient_dimension != dimension:
        raise ValueError(
            "Finite-volume quality requires full-dimensional 2-D or 3-D cells."
        )
    points = mesh.coordinates
    tables = (
        polyhedral_star_tables(mesh.connectivity)
        if isinstance(mesh.connectivity, PolyhedralConnectivity)
        else None
    )
    moments = []
    volumes = []
    cursor = 0
    for block in mesh.blocks:
        if block.cell_kind == "polyhedron":
            moment, volume = _polyhedral_centroids(
                # ty: ignore[invalid-argument-type]
                tables,
                np.arange(cursor, cursor + block.cell_count),
                points,
            )
        else:
            moment, volume = _block_centroids(
                block.cell_kind, points[np.asarray(block.vertices, dtype=np.int64)]
            )
        moments.append(moment)
        volumes.append(volume)
        cursor += block.cell_count
    volume = jnp.concatenate(volumes)
    centroids = jnp.concatenate(moments) / jnp.where(volume == 0.0, 1.0, volume)[:, None]
    face_centroids, face_normals = _facet_geometry(mesh, points)
    relation = mesh.topology.incidences[-1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    facets = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    cells = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    order = np.lexsort((cells, facets))
    facets = facets[order]
    cells = cells[order]
    paired = np.flatnonzero(facets[1:] == facets[:-1])
    face = facets[paired]
    owner = cells[paired]
    neighbor = cells[paired + 1]
    delta = centroids[neighbor] - centroids[owner]
    normal = face_normals[face]
    offset = face_centroids[face] - centroids[owner]
    normal = jnp.where(
        jnp.sum(normal * offset, axis=-1, keepdims=True) < 0.0, -normal, normal
    )
    projection = jnp.sum(delta * normal, axis=-1)
    if dimension == 2:
        sine = jnp.abs(delta[:, 0] * normal[:, 1] - delta[:, 1] * normal[:, 0])
    else:
        sine = jnp.linalg.norm(jnp.cross(delta, normal), axis=-1)
    non_orthogonality = jnp.arctan2(sine, projection)
    parameter = jnp.sum(offset * normal, axis=-1) / jnp.where(
        projection == 0.0, jnp.inf, projection
    )
    intersection = centroids[owner] + parameter[:, None] * delta
    distance = jnp.linalg.norm(delta, axis=-1)
    skewness = jnp.linalg.norm(
        face_centroids[face] - intersection, axis=-1
    ) / jnp.maximum(distance, _tiny(points))
    skewness = jnp.where(projection == 0.0, jnp.inf, skewness)
    facet_ids = np.asarray(mesh.entity_set(dimension - 1).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    return FiniteVolumeQualityEvaluation(
        face_global_ids=jnp.asarray(facet_ids[face]),
        owner_cell_global_ids=jnp.asarray(cell_ids[owner]),
        neighbor_cell_global_ids=jnp.asarray(cell_ids[neighbor]),
        skewness=skewness,
        non_orthogonality=non_orthogonality,
        topology_id=mesh.topology_id,
        evaluation_id=canonical_fingerprint(
            {
                "kind": "finite-volume-quality-evaluation",
                "mesh": mesh.mesh_id,
            }
        ),
    )


__all__ = [
    "CellQualityEvaluation",
    "CellQualityReport",
    "FiniteVolumeQualityEvaluation",
    "evaluate_cell_quality",
    "evaluate_finite_volume_quality",
    "summarize_cell_quality",
]
