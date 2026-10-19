#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable sampled cell quality and finite-volume face quality.

NumPy prepares reference quadrature and incidence routes. Stable module-level
compiled kernels keep coordinate coefficients and metric tensors dynamic, and
sample cells in bounded power-of-two worksets. Host query admission precedes
volume sampling, boundary sampling, and each face sample in reference order.
Coordinate maps supply signed measures and sampled Jacobian metric quality;
explicit corner-coordinate evaluations retain straight-corner/star quality.
Neither sampling route proves validity: the Bernstein certificate in
``phydrax.discretization._cell_geometry_validity`` is the validity owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import assert_never, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_geometry_validity as cell_validity_owner
import phydrax.discretization._coordinate_budget as coordinate_budget_owner
import phydrax.discretization.fem._reference_rules as reference_rules_owner
import phydrax.ein as ein
import phydrax.linalg._small_batched as small_linear_owner
import phydrax.meshing._metric as metric_owner
import phydrax.meshing._scope as scope_owner
from phydrax.discretization._reference_cell_types import CellGeometryElement

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_table_cache import host_table_cache, logical_table_bytes
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._adaptive_simplex import adaptive_simplex_bucket
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._motion_validity import reference_corner_frames
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference_types import FiniteElementSpec
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..typing import checked


type _QualityRule = tuple[np.ndarray, np.ndarray]
# One MiB admits eight measured hot sets: volume 124704, boundary 20496,
# and face 3328 logical bytes for the supported reference-cell workloads.
_QUALITY_RULE_CACHE_BYTES = 1048576

type _QualityGeometry = tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]
type _ScalarCoordinateElement = (
    FiniteElementSpec
    | cell_geometry_owner.BarycentricCellGeometryElement
    | cell_geometry_owner.RestrictedCellGeometryElement
    | cell_geometry_owner.PolynomialComposedCellGeometryElement
    | cell_geometry_owner.RationalComposedCellGeometryElement
    | cell_geometry_owner.SplineCellGeometryElement
    | cell_geometry_owner.LayerColumnCellGeometryElement
)


class CellQualityEvaluation(StrictModule):
    """Per-cell sampled quality in mesh block order.

    Angles are radians. Without ``geometry_layout_id``, every shape statistic
    uses straight corner frames (or vertex-star simplices). With a scalar
    coordinate map, measures integrate its signed Jacobian; shape,
    radius/sliver, condition, and metric statistics sample its Jacobian images
    of the reference shape. Boundary angles use source-map corner tangents or
    edge dihedrals; warpage samples mapped face normals.
    Vertex-defined polygon/polyhedron maps retain their star convention.
    Rational/embedded measures are sampled quadrature. Integration-error bounds
    and whole-map validity belong to separate certificates. ``metric_quality``
    is NaN without a metric; radius/sliver values are NaN for nonsimplicial kinds.
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
    metric: metric_owner.MeshMetricField | None


def _nan_extreme(values: np.ndarray, minimum: bool, /) -> float:
    finite = values[~np.isnan(values)]
    if not finite.size:
        return float("nan")
    return float(np.min(finite) if minimum else np.max(finite))


class CellQualityReport(StrictModule, NonTrainableState):
    """Summary of source-map or explicitly straight-corner sampled quality.

    The evaluation's geometry layout distinguishes coordinate-map measurements
    from corner-only measurements. This is quality evidence only: mapped-cell
    validity is owned by the Bernstein validity certificate and global embedding
    by geometry mesh certificates.
    ``sampled_invalid_cell_global_ids`` lists failed sampled Jacobian frames.
    Empty sampled arrays are admitted only with a zero-resident distributed
    mesh. Their extrema are neutral reductions over an empty
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
        host_fields = jax.device_get(
            (
                evaluation.measures,
                evaluation.mean_ratios,
                evaluation.aspect_ratios,
                evaluation.scaled_jacobian,
                evaluation.condition_number,
                evaluation.minimum_angle,
                evaluation.maximum_angle,
                evaluation.warpage,
                evaluation.metric_quality,
                evaluation.sampled_valid,
                evaluation.cell_global_ids,
            )
        )
        (
            measures,
            ratios,
            aspects,
            scaled,
            condition,
            minimum_angle,
            maximum_angle,
            warpage,
            metric,
        ) = (np.asarray(value, dtype=np.float64) for value in host_fields[:9])
        valid = np.asarray(host_fields[9], dtype=np.bool_)
        identifiers = np.asarray(host_fields[10], dtype=np.int64)
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
                    "quality.evaluation.cell_global_ids must be nonempty unless mesh has zero resident cells and positive global count; got empty"
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
        self.minimum_angle = _nan_extreme(minimum_angle, True)
        self.maximum_angle = _nan_extreme(maximum_angle, False)
        self.maximum_warpage = float(np.max(warpage, initial=-np.inf))
        self.minimum_metric_quality = _nan_extreme(metric, True)
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


class BlockQuality(NamedTuple):
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


def ideal_vertices(kind: str, arity: int, /) -> np.ndarray:
    root3 = np.sqrt(3.0)
    match kind:
        case "triangle":
            return np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5 * root3)), dtype=np.float64
            )
        case "polygon":
            angles = 2.0 * np.pi * np.arange(arity, dtype=np.float64) / arity
            return np.stack((np.cos(angles), np.sin(angles)), axis=1)
        case "tetrahedron":
            return np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.5, 0.5 * root3, 0.0),
                    (0.5, root3 / 6.0, np.sqrt(2.0 / 3.0)),
                ),
                dtype=np.float64,
            )
        case "prism":
            base = ideal_vertices("triangle", 3)
            return np.concatenate(
                (
                    np.pad(base, ((0, 0), (0, 1))),
                    np.pad(base, ((0, 0), (0, 1)), constant_values=1.0),
                )
            )
        case "pyramid":
            vertices = np.asarray(
                reference_cell_topology("pyramid").vertices, dtype=np.float64
            )
            vertices[4, 2] = np.sqrt(0.5)
            return vertices
        case "quadrilateral" | "hexahedron":
            return np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        case _:
            raise ValueError(
                f"quality.cell_kind must have an ideal reference shape; got {kind!r}"
            )


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


def _square_determinant(matrix: Array, /) -> Array:
    return small_linear_owner.cofactor_determinant(matrix)


def _adjugate_trace(matrix: Array, /) -> Array:
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


@host_table_cache(
    max_entries=256,
    max_bytes=1048576,
    weight=lambda value: logical_table_bytes(
        (
            value.vertices,
            value.neighbors,
            value.ideal_inverse,
            value.ideal_determinant,
            value.ideal_scaled,
        )
    ),
)
def corner_table(kind: str, arity: int, /) -> _CornerTable:
    ideal = ideal_vertices(kind, arity)
    if kind == "polygon":
        vertices = np.arange(arity, dtype=np.int64)
        neighbors = np.stack(((vertices + 1) % arity, (vertices - 1) % arity), axis=1)
    else:
        vertices, neighbors = reference_corner_frames(kind)
    frames = np.swapaxes(ideal[neighbors] - ideal[vertices][:, None, :], -1, -2)
    determinant = np.linalg.det(frames)
    if np.any(determinant <= 0.0):
        raise ValueError(
            f"quality.reference_corner_determinants must be positive; got {kind!r} with minimum {np.min(determinant)}"
        )
    identity = np.broadcast_to(np.eye(frames.shape[-1], dtype=np.float64), frames.shape)
    inverse = np.linalg.solve(frames, identity)
    scaled = determinant / np.prod(np.linalg.norm(frames, axis=-2), axis=-1)
    return _CornerTable(vertices, neighbors, inverse, determinant, scaled)


@dataclass(frozen=True)
class _DihedralTable:
    """Edge samples ``(i, j, k, l)``: interior angle about i->j from k to l."""

    samples: np.ndarray


@cache
def _dihedral_table(kind: str, /) -> _DihedralTable:
    ideal = ideal_vertices(kind, 0)
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
    return _DihedralTable(np.asarray(samples, dtype=np.int64))


def _first_order_space(kind: str, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
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
            raise ValueError(
                f"quality.cell_kind must have a first-order geometry space; got {kind!r}"
            )
    return np.stack(values, axis=1), np.stack(
        [np.stack(row, axis=-1) for row in gradients], axis=1
    )


@dataclass(frozen=True)
class _MeasureRule:
    gradients: np.ndarray
    weights: np.ndarray


@cache
def measure_rule(kind: str, /) -> _MeasureRule:
    """Prepare gradients and weights for first-order determinant integration.

    The nodal gradients are host NumPy data: the rule may first be requested
    under a JAX transformation, where a JAX tabulation would be traced.
    Only hexahedron, prism, and pyramid are admitted; their three immutable
    host tables retain 2928 logical payload bytes in total.
    """

    gauss = np.asarray(
        (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)), dtype=np.float64
    )
    gauss_weights = np.asarray((0.5, 0.5), dtype=np.float64)
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
                (
                    np.full(2, 1.0 / 3.0, dtype=np.float64),
                    np.full(2, 1.0 / 3.0, dtype=np.float64),
                    gauss,
                ),
                axis=1,
            )
            weights = 0.5 * gauss_weights
        case "pyramid":
            collapsed = np.stack(np.meshgrid(gauss, gauss, gauss, indexing="ij"), -1)
            collapsed = collapsed.reshape(-1, 3)
            height = collapsed[:, 2:3]
            points = np.concatenate(
                (collapsed[:, :2] * (1.0 - height) + 0.5 * height, height), axis=1
            )
            weights = np.full(8, 0.125, dtype=np.float64) * (1.0 - collapsed[:, 2]) ** 2
        case _:
            raise ValueError(
                f"quality.cell_kind must have a quadrature measure rule; got {kind!r}"
            )
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    vandermonde, _ = _first_order_space(kind, vertices)
    _, spanning = _first_order_space(kind, points)
    count = vandermonde.shape[0]
    # Nodal gradients solve V^T G_nodal = G_spanning for every point and axis.
    nodal = np.linalg.solve(
        vandermonde.T, np.moveaxis(spanning, 1, 0).reshape((count, -1))
    )
    gradients = np.moveaxis(nodal.reshape((count, points.shape[0], 3)), 0, 1)
    gradients.flags.writeable = False
    weights.flags.writeable = False
    return _MeasureRule(gradients, weights)


# Quality kernels -----------------------------------------------------------------


def _tiny(values: Array, /) -> Array:
    return jnp.asarray(jnp.finfo(values.dtype).tiny, dtype=values.dtype)


def _angle(sine: Array, cosine: Array, /) -> Array:
    angle = jnp.arctan2(sine, cosine)
    return jnp.where(angle < 0.0, angle + 2.0 * jnp.pi, angle)


def _centered_cell_points(points: Array, /) -> Array:
    """Center cyclic cell coordinates after removing the first-vertex translation."""
    relative = points - points[..., :1, :]
    # Averaging absolute coordinates lets fused centroid subtraction lose
    # local cell bits at large translations. The centroid-fan convention is
    # unchanged when its average is formed in this translated local frame.
    return relative - jnp.mean(relative, axis=-2, keepdims=True)


def _vector_area(points: Array, /) -> Array:
    """Vector area (Newell) of cyclic 2-cells, axis -2 cycling vertices."""

    shifted = _centered_cell_points(points)
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

    shifted = _centered_cell_points(points)
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
        / table.ideal_scaled[None, :]
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


def _interval_quality(points: Array, metric: Array | None, /) -> BlockQuality:
    length = jnp.linalg.norm(points[:, 1] - points[:, 0], axis=-1)
    one = jnp.ones_like(length)
    nan = jnp.full_like(length, jnp.nan)
    return BlockQuality(
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


def _surface_quality(kind: str, points: Array, metric: Array | None, /) -> BlockQuality:
    """Triangles, quadrilaterals, and polygons in 2-D or embedded in 3-D."""

    arity = points.shape[1]
    table = corner_table(kind, arity)
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
        shifted = _centered_cell_points(points)
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
        shifted = _centered_cell_points(points)
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
    return BlockQuality(
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
    rule = measure_rule(kind)
    jacobian = ein.contract("qnk,cna->cqak", rule.gradients, points, backend="jax")
    return jnp.sum(_square_determinant(jacobian) * rule.weights, axis=1)


def _tetrahedron_shape(
    points: Array, volume: Array, lengths: Array, /
) -> tuple[Array, Array]:
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


def _solid_quality(kind: str, points: Array, metric: Array | None, /) -> BlockQuality:
    table = corner_table(kind, points.shape[1])
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
    return BlockQuality(
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


@eqx.filter_jit
@checked
def standard_block_quality(
    kind: str, points: Array, metric: Array | None, /
) -> BlockQuality:
    """Evaluate straight-cell quality in bounded power-of-two lane worksets."""

    def evaluate(values: Array, tensor: Array | None) -> BlockQuality:
        return _standard_block_quality(kind, values, tensor)

    return _quality_worksets(evaluate, points, metric)


def _quality_worksets(
    evaluate: Callable[[Array, Array | None], BlockQuality],
    points: Array,
    metric: Array | None,
    /,
) -> BlockQuality:
    count = points.shape[0]
    if count == 0:
        return evaluate(points, metric)
    capacity = adaptive_simplex_bucket(min(count, 256), 1.0)
    lanes = jnp.arange(capacity, dtype=jnp.int32)
    starts = jnp.arange((count + capacity - 1) // capacity, dtype=jnp.int32) * capacity

    def sample(start: Array) -> BlockQuality:
        rows = start + lanes
        valid = rows < count
        safe = jnp.minimum(rows, count - 1)
        return _masked_quality_workset(
            evaluate, points[safe], None if metric is None else metric[safe], valid
        )

    quality = jax.lax.map(sample, starts)
    return jax.tree_util.tree_map(lambda field: field.reshape(-1)[:count], quality)


def _masked_quality_workset(
    evaluate: Callable[[Array, Array | None], BlockQuality],
    points: Array,
    metric: Array | None,
    valid: Array,
    /,
) -> BlockQuality:
    """Exclude inactive workset lanes from results and coordinate/metric adjoints."""
    values = jnp.where(valid[:, None, None], points, jax.lax.stop_gradient(points))
    tensor = (
        None
        if metric is None
        else jnp.where(valid[:, None, None], metric, jax.lax.stop_gradient(metric))
    )
    quality = evaluate(values, tensor)
    return jax.tree_util.tree_map(
        lambda field: jnp.where(valid, field, jnp.zeros_like(field)), quality
    )


def _standard_block_quality(
    kind: str, points: Array, metric: Array | None, /
) -> BlockQuality:
    match kind:
        case "interval":
            return _interval_quality(points, metric)
        case "triangle" | "quadrilateral" | "polygon":
            return _surface_quality(kind, points, metric)
        case "tetrahedron" | "hexahedron" | "prism" | "pyramid":
            return _solid_quality(kind, points, metric)
        case _:
            raise ValueError(
                f"quality.cell_kind must name a supported reference cell; got {kind!r}"
            )


def _count_quality_queries(
    budget: coordinate_budget_owner.CoordinateEnclosureBudget,
    count: int,
    /,
) -> None:
    """Count the sampled source evaluations without altering coefficient counters."""

    ticket = budget.ticket
    if count > ticket.remaining_work_units or count > ticket.remaining_geometry_queries:
        ticket.refuse(count, queries_upper=count)
    ticket.consume(count, geometry_queries=count)


class _MappedQualityPlan(StrictModule):
    """Reference quadrature, boundary tangents, and face-normal sample routes."""

    volume: _QualityRule
    boundary: _QualityRule | None
    faces: tuple[_QualityRule, ...]

    @property
    def query_count(self) -> int:
        return (
            self.volume[0].shape[0]
            + (0 if self.boundary is None else self.boundary[0].shape[0])
            + sum(probes.shape[0] for probes, _ in self.faces)
        )


def _mapped_quality_rule(kind: str, element: CellGeometryElement, /) -> _QualityRule:
    source = cell_geometry_owner.require_scalar_coordinate_element(
        element, "Mapped quality"
    )
    match source:
        case cell_geometry_owner.RestrictedCellGeometryElement():
            order = _coordinate_total_order(source)
        case (
            FiniteElementSpec()
            | cell_geometry_owner.BarycentricCellGeometryElement()
            | cell_geometry_owner.PolynomialComposedCellGeometryElement()
            | cell_geometry_owner.RationalComposedCellGeometryElement()
            | cell_geometry_owner.SplineCellGeometryElement()
            | cell_geometry_owner.LayerColumnCellGeometryElement()
        ):
            order = source.degree
        case _:
            assert_never(source)
    count = max(2, (source.topological_dimension * order + 1) // 2)
    rule_degree = max(0, count - (2 if kind == "tetrahedron" else 1))
    return _volume_rule(kind, rule_degree)


@host_table_cache(max_entries=256, max_bytes=_QUALITY_RULE_CACHE_BYTES)
def _volume_rule(kind: str, degree: int, /) -> _QualityRule:
    """Retain immutable degree-dependent host quadrature under a byte ceiling."""
    probes, weights = reference_rules_owner.host_degree_aware_reference_rule(kind, degree)
    probes.flags.writeable = False
    weights.flags.writeable = False
    return probes, weights


def _mapped_face_warpage(
    plan: _MappedQualityPlan,
    source: _ScalarCoordinateElement,
    coordinates: Array,
    /,
    *,
    budget: coordinate_budget_owner.CoordinateEnclosureBudget,
) -> Array:
    """Sample coordinate-map face normals in prepared reference-face order."""
    if not plan.faces:
        return _zero_face_warpage(coordinates)
    angles = []
    for probes, directions in plan.faces:
        _count_quality_queries(budget, coordinates.shape[0] * probes.shape[0])
        angles.append(_mapped_face_sample(source, coordinates, probes, directions))
    return _maximum_face_warpage(tuple(angles))


def _zero_face_warpage(coordinates: Array, /) -> Array:
    return jax.device_put(np.zeros((coordinates.shape[0],), dtype=coordinates.dtype))


@eqx.filter_jit
def _maximum_face_warpage(angles: tuple[Array, ...], /) -> Array:
    return jnp.max(jnp.stack(angles, axis=1), axis=1)


@eqx.filter_jit
def _mapped_face_sample(
    source: _ScalarCoordinateElement,
    coordinates: Array,
    probes: np.ndarray,
    directions: np.ndarray,
    /,
) -> Array:
    _, gradients = source.tabulate(jnp.asarray(probes, dtype=coordinates.dtype))
    jacobian = ein.contract("qir,cia->cqar", gradients, coordinates, backend="jax")
    tangents = ein.contract("cqar,rs->cqas", jacobian, directions, backend="jax")
    normals = jnp.cross(tangents[..., 0], tangents[..., 1])
    reference = normals[:, :1]
    sine = jnp.linalg.norm(jnp.cross(reference, normals[:, 1:]), axis=-1)
    cosine = jnp.sum(reference * normals[:, 1:], axis=-1)
    return jnp.max(jnp.arctan2(sine, cosine), axis=1)


def _coordinate_total_order(element: CellGeometryElement, /) -> int:
    """Static source-space bound, including affine mixing of tensor roots."""
    source = cell_geometry_owner.require_scalar_coordinate_element(
        element, "Mapped quality order"
    )
    match source:
        case cell_geometry_owner.RestrictedCellGeometryElement():
            return _coordinate_total_order(source.source_element)
        case cell_geometry_owner.LayerColumnCellGeometryElement():
            return max(2, _coordinate_total_order(source.wall_element) + 1)
        case cell_geometry_owner.SplineCellGeometryElement():
            return source.u_degree + source.v_degree
        case (
            FiniteElementSpec()
            | cell_geometry_owner.BarycentricCellGeometryElement()
            | cell_geometry_owner.PolynomialComposedCellGeometryElement()
            | cell_geometry_owner.RationalComposedCellGeometryElement()
        ):
            if source.cell_kind in (
                "interval",
                "triangle",
                "tetrahedron",
            ) or source.cell_kind.startswith("simplex:"):
                return source.degree
            return (
                2 if source.cell_kind == "prism" else source.topological_dimension
            ) * source.degree
        case _:
            assert_never(source)


def _mapped_boundary_angles(
    plan: _MappedQualityPlan,
    source: _ScalarCoordinateElement,
    coordinates: Array,
    /,
    *,
    budget: coordinate_budget_owner.CoordinateEnclosureBudget,
) -> tuple[Array, Array]:
    """Measure source-map corner tangents and boundary-edge dihedrals."""
    if plan.boundary is None:
        return _missing_boundary_angles(coordinates)
    probes, directions = plan.boundary
    _count_quality_queries(budget, coordinates.shape[0] * probes.shape[0])
    return _mapped_boundary_samples(source, coordinates, probes, directions)


@host_table_cache(max_entries=256, max_bytes=_QUALITY_RULE_CACHE_BYTES)
def _boundary_rule(kind: str, /) -> _QualityRule | None:
    """Retain host tangent probes without caching geometry or query charges."""
    dimension = reference_cell_topology(kind).dimension
    if dimension == 1:
        return None
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    if dimension == 2:
        probes = vertices
        directions = np.stack(
            (
                np.roll(vertices, -1, axis=0) - vertices,
                np.roll(vertices, 1, axis=0) - vertices,
            ),
            axis=-1,
        )
    else:
        samples = _dihedral_table(kind).samples
        first, last = vertices[samples[:, 0]], vertices[samples[:, 1]]
        axis, _ = reference_rules_owner.host_degree_aware_reference_rule("interval", 2)
        probes = (
            first[:, None] * (1.0 - axis[None]) + last[:, None] * axis[None]
        ).reshape(-1, dimension)
        directions = np.stack(
            (
                last - first,
                vertices[samples[:, 2]] - first,
                vertices[samples[:, 3]] - first,
            ),
            axis=-1,
        )
        directions = np.repeat(directions, axis.shape[0], axis=0)
    probes.flags.writeable = False
    directions.flags.writeable = False
    return probes, directions


@host_table_cache(max_entries=256, max_bytes=_QUALITY_RULE_CACHE_BYTES)
def _face_rules(kind: str, ambient: int, /) -> tuple[_QualityRule, ...]:
    """Retain immutable face probes and directions under a byte ceiling."""
    topology = reference_cell_topology(kind)
    if topology.dimension < 2 or ambient < 3:
        return ()
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    faces = (
        _faces(kind, vertices.shape[0])
        if topology.dimension == 3
        else (tuple(range(vertices.shape[0])),)
    )
    rules = []
    for face in faces:
        points = vertices[np.asarray(face, dtype=np.int64)]
        # Edge midpoints avoid the collapsed pyramid apex, whose tangent
        # plane is not unique.
        probes = np.concatenate(
            (
                np.mean(points, axis=0, keepdims=True),
                0.5 * (points + np.roll(points, -1, axis=0)),
            )
        )
        directions = np.stack((points[1] - points[0], points[-1] - points[0]), axis=1)
        probes.flags.writeable = False
        directions.flags.writeable = False
        rules.append((probes, directions))
    return tuple(rules)


@eqx.filter_jit
def _missing_boundary_angles(coordinates: Array, /) -> tuple[Array, Array]:
    empty = jnp.full((coordinates.shape[0],), jnp.nan, dtype=coordinates.dtype)
    return empty, empty


def _mapped_boundary_samples(
    source: _ScalarCoordinateElement,
    coordinates: Array,
    probes: np.ndarray,
    directions: np.ndarray,
    /,
) -> tuple[Array, Array]:
    return _mapped_boundary_values(
        cell_geometry_owner.coordinate_tabulation(source),
        source.topological_dimension,
        coordinates,
        probes,
        directions,
    )


@eqx.filter_jit
def _mapped_boundary_values(
    source: cell_geometry_owner.ScalarCoordinateTabulation,
    dimension: int,
    coordinates: Array,
    probes: np.ndarray,
    directions: np.ndarray,
    /,
) -> tuple[Array, Array]:
    _, gradients = source.tabulate(jnp.asarray(probes, dtype=coordinates.dtype))
    direction_values = jnp.asarray(directions, dtype=coordinates.dtype)
    if dimension == 2:
        jacobian = ein.contract("qir,cia->cqar", gradients, coordinates, backend="jax")
        tangents = ein.contract(
            "cqar,qrs->cqas", jacobian, direction_values, backend="jax"
        )
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

    result, _ = jax.lax.scan(sample, (low, high), (gradients, direction_values))
    return result


def _mapped_block_quality(
    kind: str,
    element: CellGeometryElement,
    coordinates: Array,
    metric: Array | None,
    /,
    *,
    budget: coordinate_budget_owner.CoordinateEnclosureBudget,
    rule: _MappedQualityPlan,
) -> BlockQuality:
    """Measure the coordinate map and its sampled Jacobian metrics.

    Jacobian images of the reference shape define local metric quality.
    Positive-weight quadrature integrates polynomial signed volume under its
    declared degree bound. Rational maps retain sampled integration semantics,
    independently of their whole-map validity certificate.
    """
    source = cell_geometry_owner.require_scalar_coordinate_element(
        element, "Mapped quality"
    )
    _count_quality_queries(budget, coordinates.shape[0] * rule.volume[0].shape[0])
    measured = _mapped_sample_worksets(kind, source, coordinates, metric, rule.volume)
    minimum_angle, maximum_angle = _mapped_boundary_angles(
        rule, source, coordinates, budget=budget
    )
    warpage = _mapped_face_warpage(rule, source, coordinates, budget=budget)
    return _finish_mapped_quality(measured, minimum_angle, maximum_angle, warpage)


def _mapped_sample_worksets(
    kind: str,
    source: _ScalarCoordinateElement,
    coordinates: Array,
    metric: Array | None,
    rule: _QualityRule,
    /,
) -> BlockQuality:
    return _mapped_sample_values(
        kind,
        cell_geometry_owner.coordinate_tabulation(source),
        source.topological_dimension,
        coordinates,
        metric,
        rule,
    )


@eqx.filter_jit
def _mapped_sample_values(
    kind: str,
    source: cell_geometry_owner.ScalarCoordinateTabulation,
    dimension: int,
    coordinates: Array,
    metric: Array | None,
    rule: _QualityRule,
    /,
) -> BlockQuality:
    def evaluate(values: Array, tensor: Array | None) -> BlockQuality:
        return _mapped_cell_samples(kind, source, dimension, values, tensor, rule)

    return _quality_worksets(evaluate, coordinates, metric)


def _mapped_cell_samples(
    kind: str,
    source: cell_geometry_owner.ScalarCoordinateTabulation,
    dimension: int,
    coordinates: Array,
    metric: Array | None,
    rule: _QualityRule,
    /,
) -> BlockQuality:
    probes, weights = rule
    _, gradients = source.tabulate(jnp.asarray(probes, dtype=coordinates.dtype))
    plan = SmallLinearSolvePlan(dimension)
    reference = jnp.asarray(
        reference_cell_topology(kind).vertices, dtype=coordinates.dtype
    )
    count = coordinates.shape[0]
    low = jnp.full((count,), jnp.inf, dtype=coordinates.dtype)
    high = jnp.full((count,), -jnp.inf, dtype=coordinates.dtype)
    zero = jnp.zeros((count,), dtype=coordinates.dtype)
    initial = BlockQuality(
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
        total: BlockQuality,
        probe: tuple[Array, Array],
    ) -> tuple[BlockQuality, None]:
        gradient, weight = probe
        jacobian = ein.contract("ir,cia->car", gradient, coordinates, backend="jax")
        if coordinates.shape[-1] == dimension:
            density = determinant_small_linear(plan, jacobian)
        else:
            gram = ein.contract("car,cas->crs", jacobian, jacobian, backend="jax")
            density = jnp.sqrt(determinant_small_linear(plan, gram))
        metric_vertices = ein.contract("car,vr->cva", jacobian, reference, backend="jax")
        quality = standard_block_quality(kind, metric_vertices, metric)
        return BlockQuality(
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
    measured, _ = jax.lax.scan(
        sample, initial, (gradients, jnp.asarray(weights, dtype=coordinates.dtype))
    )
    return measured


@eqx.filter_jit
def _finish_mapped_quality(
    measured: BlockQuality,
    minimum_angle: Array,
    maximum_angle: Array,
    warpage: Array,
    /,
) -> BlockQuality:
    return measured._replace(
        minimum_angle=minimum_angle,
        maximum_angle=maximum_angle,
        warpage=warpage,
        sampled_valid=measured.sampled_valid
        & (measured.measure > 0.0)
        & jnp.isfinite(measured.measure),
    )


class _PolyhedralQualityPlan(StrictModule):
    """Host-prepared incidence routes for one polyhedral block."""

    cell_vertex_values: np.ndarray
    cell_vertex_cell: np.ndarray
    cell_vertex_counts: np.ndarray
    face_corner_vertex: np.ndarray
    face_corner_face: np.ndarray
    face_corner_next: np.ndarray
    face_sizes: np.ndarray
    star_cell: np.ndarray
    star_face: np.ndarray
    star_first: np.ndarray
    star_second: np.ndarray
    star_sign: np.ndarray
    local: np.ndarray
    pair_first: np.ndarray
    pair_second: np.ndarray
    count: int = eqx.field(static=True)


def _prepare_polyhedral_quality(
    tables: cell_validity_owner.PolyhedralStarTables, cells: np.ndarray, /
) -> _PolyhedralQualityPlan:
    selected = np.flatnonzero(np.isin(tables.star_cell, cells))
    pairs = tables.edge_pairs[np.all(np.isin(tables.edge_pairs, selected), axis=1)]
    return _PolyhedralQualityPlan(
        cell_vertex_values=tables.cell_vertex_values,
        cell_vertex_cell=tables.cell_vertex_cell,
        cell_vertex_counts=tables.cell_vertex_counts,
        face_corner_vertex=tables.face_corner_vertex,
        face_corner_face=tables.face_corner_face,
        face_corner_next=tables.face_corner_next,
        face_sizes=tables.face_sizes,
        star_cell=tables.star_cell[selected],
        star_face=tables.star_face[selected],
        star_first=tables.star_first[selected],
        star_second=tables.star_second[selected],
        star_sign=tables.star_sign[selected],
        local=np.searchsorted(cells, tables.star_cell[selected]),
        pair_first=np.searchsorted(selected, pairs[:, 0]),
        pair_second=np.searchsorted(selected, pairs[:, 1]),
        count=cells.size,
    )


def _polyhedral_quality(
    tables: cell_validity_owner.PolyhedralStarTables,
    cells: np.ndarray,
    coordinates: Array,
    metric: Array | None,
    /,
) -> BlockQuality:
    """Segment-reduce quality using host-prepared block incidence routes."""
    return _polyhedral_quality_kernel(
        _prepare_polyhedral_quality(tables, cells), coordinates, metric
    )


@eqx.filter_jit
def _polyhedral_quality_kernel(
    tables: _PolyhedralQualityPlan,
    coordinates: Array,
    metric: Array | None,
    /,
) -> BlockQuality:
    count = tables.count
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
    local = tables.local
    star_face = tables.star_face
    first = coordinates[tables.star_first]
    second = coordinates[tables.star_second]
    tetrahedra = jnp.stack(
        (
            center[tables.star_cell],
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
    outward = jnp.asarray(tables.star_sign, dtype=coordinates.dtype)[:, None]
    unit = face_area[star_face] * outward
    unit = unit / jnp.maximum(
        jnp.linalg.norm(unit, axis=-1, keepdims=True), _tiny(coordinates)
    )
    fan = jnp.cross(first - face_center[star_face], second - face_center[star_face])
    fan_angle = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(fan, unit), axis=-1), jnp.sum(fan * unit, axis=-1)
    )
    pair_first = tables.pair_first
    pair_second = tables.pair_second
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
    return BlockQuality(
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


def _vertex_metric(
    mesh: CellMesh, metric: metric_owner.MeshMetricField | None, /
) -> Array | None:
    if metric is None:
        return None
    dimension = mesh.ambient_dimension
    if metric.values.shape != (mesh.coordinates.shape[0], dimension, dimension):
        raise ValueError(
            f"quality.metric.values must have shape {(mesh.coordinates.shape[0], dimension, dimension)}; got {metric.values.shape}"
        )
    scope = metric.scope
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not scope_owner.MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.entity_set_id != mesh.entity_set(0).entity_set_id
    ):
        raise ValueError(
            f"quality.metric.scope must bind mesh vertices; got {scope.scope_id}"
        )
    return _validated_vertex_metric(
        metric.values, metric.scope.entity_ids, mesh.vertex_global_ids
    )


@eqx.filter_jit
def _validated_vertex_metric(
    values: Array, members: Array, identifiers: Array, /
) -> Array:
    """Admit vertex identities in the metric bank's declared capacity."""
    return eqx.error_if(
        values,
        ~jnp.all(members == identifiers),
        "quality.metric.scope.entity_ids must match mesh.vertex_global_ids; got different identities",
    )


@eqx.filter_jit
def _cell_metric(values: Array, rows: Array, /) -> Array:
    weights = jnp.full(rows.shape, 1.0 / rows.shape[1], dtype=values.dtype)
    return metric_owner.interpolate_mesh_metric(values[rows], weights)


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
    vertices: list[Array] = []
    source_rows: list[Array] = []
    masks: list[Array] = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if isinstance(
            element,
            cell_geometry_owner.CellVertexGeometryElement,
        ):
            vertices.append(block.vertices)
            source_rows.append(route)
            masks.append(block.vertex_valid)
    if not vertices:
        return mesh.coordinates
    return _vertex_geometry_coordinate_values(
        mesh.coordinates, tuple(vertices), tuple(source_rows), tuple(masks), bank
    )


@eqx.filter_jit
def _vertex_geometry_coordinate_values(
    coordinates: Array,
    vertices: tuple[Array, ...],
    source_rows: tuple[Array, ...],
    masks: tuple[Array, ...],
    bank: Array,
    /,
) -> Array:
    """Join fixed-capacity incidence arrays without source or mesh metadata."""
    vertex = jnp.concatenate(tuple(values.reshape(-1) for values in vertices))
    rows = jnp.concatenate(tuple(values.reshape(-1) for values in source_rows))
    valid = jnp.concatenate(tuple(values.reshape(-1) for values in masks))
    safe_vertex, safe_row = jnp.maximum(vertex, 0), jnp.maximum(rows, 0)
    sentinel = jnp.asarray(bank.shape[0], dtype=rows.dtype)
    first = jax.ops.segment_min(
        jnp.where(valid, rows, sentinel),
        safe_vertex,
        coordinates.shape[0],
    )
    selected = bank[jnp.minimum(first, bank.shape[0] - 1)]
    points = jnp.where((first < sentinel)[:, None], selected, coordinates)
    shared = jnp.all(~valid | jnp.all(bank[safe_row] == points[safe_vertex], axis=-1))
    return eqx.error_if(
        points,
        ~shared,
        "quality.geometry.coordinates must agree at shared vertices; got different source values",
    )


def _quality_query_plan(
    mesh: CellMesh, resolved: _QualityGeometry | None, /
) -> tuple[tuple[_MappedQualityPlan | None, ...], int]:
    """Prepare quadrature once and bound all sampled source evaluations."""
    if resolved is None:
        return (None,) * len(mesh.blocks), 0
    rules: list[_MappedQualityPlan | None] = []
    queries = 0
    for block, element in zip(mesh.blocks, resolved[0], strict=True):
        if block.cell_kind == "polyhedron" or isinstance(
            element,
            cell_geometry_owner.CellVertexGeometryElement,
        ):
            rules.append(None)
            continue
        rule = _MappedQualityPlan(
            volume=_mapped_quality_rule(block.cell_kind, element),
            boundary=_boundary_rule(block.cell_kind),
            faces=_face_rules(block.cell_kind, mesh.ambient_dimension),
        )
        rules.append(rule)
        queries += block.cell_count * rule.query_count
    return tuple(rules), queries


@checked
def evaluate_cell_quality(
    mesh: CellMesh,
    coordinates: ArrayLike | None = None,
    /,
    *,
    metric: metric_owner.MeshMetricField | None = None,
    geometry: CellGeometrySpec | None = None,
    budget: coordinate_budget_owner.CoordinateEnclosureBudget | None = None,
) -> CellQualityEvaluation:
    """Evaluate coordinate-map quality or explicit straight-corner quality.

    ``geometry`` retains its coordinate coefficients and tabulation,
    including high-order, restricted, composed and rational maps. The separate
    ``coordinates`` override deliberately evaluates a straight-corner mesh;
    supplying both authorities is ambiguous and refused.
    """

    if geometry is not None and coordinates is not None:
        raise ValueError(
            "quality.geometry must be absent when coordinates are supplied; got both"
        )
    points = mesh.coordinates if coordinates is None else jnp.asarray(coordinates)
    if points.shape != mesh.coordinates.shape:
        raise ValueError(
            f"quality.coordinates must have shape {mesh.coordinates.shape}; got {points.shape}"
        )
    if not mesh.blocks and (
        mesh.storage is None
        or mesh.storage.global_entity_counts[mesh.topological_dimension] <= 0
        or points.shape[0] != 0
        or any(entities.count != 0 for entities in mesh.topology.entity_sets)
    ):
        raise ValueError(
            "quality.mesh.blocks must be nonempty unless mesh has zero resident cells and positive global count; got empty"
        )
    resolved = None if geometry is None else geometry.resolve(mesh, budget=budget)
    if resolved is not None:
        elements, routes, bank = resolved
        points = _vertex_geometry_coordinates(mesh, elements, routes, bank)
    rules, query_upper = _quality_query_plan(mesh, resolved)
    if budget is None:
        ticket = HostBatchTicket(query_upper, queries_upper=query_upper)
        view = coordinate_budget_owner.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=query_upper,
            maximum_memory_bytes=0,
        )
        with ticket, view.activate():
            return _evaluated_cell_quality(
                mesh, points, geometry, metric, resolved, rules, budget=view
            )
    return _evaluated_cell_quality(
        mesh, points, geometry, metric, resolved, rules, budget=budget
    )


def _evaluated_cell_quality(
    mesh: CellMesh,
    points: Array,
    geometry: CellGeometrySpec | None,
    metric: metric_owner.MeshMetricField | None,
    resolved: _QualityGeometry | None,
    rules: tuple[_MappedQualityPlan | None, ...],
    /,
    *,
    budget: coordinate_budget_owner.CoordinateEnclosureBudget,
) -> CellQualityEvaluation:
    vertex_metric = _vertex_metric(mesh, metric)
    if any(block.cell_kind == "polyhedron" for block in mesh.blocks):
        match mesh.connectivity:
            case PolyhedralConnectivity() as connectivity:
                tables = cell_validity_owner.polyhedral_star_tables(connectivity)
            case _:
                raise ValueError(
                    "mesh.connectivity must be polyhedral for polyhedron blocks; got "
                    + type(mesh.connectivity).__name__
                )
    else:
        tables = None
    blocks = []
    offsets = [0]
    for block_index, block in enumerate(mesh.blocks):
        start = offsets[-1]
        stop = start + block.cell_count
        if block.cell_kind == "polyhedron":
            cells = np.arange(start, stop, dtype=np.int64)
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
            cell_metric = (
                None if vertex_metric is None else _cell_metric(vertex_metric, rows)
            )
            if resolved is None:
                blocks.append(
                    standard_block_quality(
                        block.cell_kind, _gather_coordinates(points, rows), cell_metric
                    )
                )
            else:
                element = resolved[0][block_index]
                values = _gather_coordinates(resolved[2], resolved[1][block_index])
                if isinstance(
                    element,
                    cell_geometry_owner.CellVertexGeometryElement,
                ):
                    blocks.append(
                        standard_block_quality(block.cell_kind, values, cell_metric)
                    )
                else:
                    rule = rules[block_index]
                    if rule is None:
                        raise RuntimeError(
                            "quality.mapped_rule must contain prepared quadrature; got None"
                        )
                    blocks.append(
                        _mapped_block_quality(
                            block.cell_kind,
                            element,
                            values,
                            cell_metric,
                            budget=budget,
                            rule=rule,
                        )
                    )
        offsets.append(stop)
    if blocks:
        fields = _join_block_quality(tuple(blocks))
    else:
        fields = _empty_block_quality(points)
    return _quality_evaluation(mesh, geometry, metric, tuple(offsets), fields)


@eqx.filter_jit
def _empty_block_quality(points: Array, /) -> BlockQuality:
    empty = jnp.empty((0,), dtype=points.dtype)
    return BlockQuality(
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


def _quality_evaluation(
    mesh: CellMesh,
    geometry: CellGeometrySpec | None,
    metric: metric_owner.MeshMetricField | None,
    offsets: tuple[int, ...],
    fields: BlockQuality,
    /,
) -> CellQualityEvaluation:
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
        cell_global_ids=_join_global_ids(
            tuple(block.global_ids for block in mesh.blocks)
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
        block_offsets=offsets,
        topology_id=mesh.topology_id,
        geometry_layout_id=None if geometry is None else geometry.geometry_layout_id,
        evaluation_id=evaluation_id,
        metric=metric,
    )


@eqx.filter_jit
def _join_global_ids(identifiers: tuple[Array, ...], /) -> Array:
    return (
        jnp.concatenate(identifiers) if identifiers else jnp.empty((0,), dtype=jnp.int64)
    )


def _polyhedral_cell_metric(
    values: Array,
    tables: cell_validity_owner.PolyhedralStarTables,
    cells: np.ndarray,
    /,
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
    return _interpolated_metric(values, rows, weights)


@eqx.filter_jit
def _interpolated_metric(
    values: Array, rows: np.ndarray, weights: np.ndarray, /
) -> Array:
    return metric_owner.interpolate_mesh_metric(
        values[rows], jnp.asarray(weights, dtype=values.dtype)
    )


@eqx.filter_jit
def _gather_coordinates(points: Array, rows: Array, /) -> Array:
    return points[rows]


@eqx.filter_jit
def _join_block_quality(blocks: tuple[BlockQuality, ...], /) -> BlockQuality:
    return BlockQuality(
        *(jnp.concatenate(values) for values in zip(*blocks, strict=True))
    )


@checked
def summarize_cell_quality(
    evaluation: CellQualityEvaluation,
    /,
    *,
    mesh: CellMesh | None = None,
) -> CellQualityReport:
    return CellQualityReport(evaluation, mesh=mesh)


# Straight sweep layers -------------------------------------------------------------


class SweptLayerQualityEvaluation(StrictModule, NonTrainableState):
    """Measured straight-sweep quality for one prescribed layer schedule."""

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
        raise ValueError(
            f"quality.points must have shape (n, 3); got {coordinates.shape}"
        )
    if (
        cells.ndim != 2
        or cells.shape[1] != 6
        or indices.shape != cells.shape[:1]
        or np.any(cells < 0)
        or np.any(cells >= coordinates.shape[0])
    ):
        raise ValueError(
            f"quality.prisms must have six valid vertex rows per cell and one layer index; got shapes {(cells.shape, indices.shape)}"
        )
    if requested.ndim != 1 or requested.size == 0 or np.any(requested <= 0.0):
        raise ValueError(
            f"quality.requested_thicknesses must be a nonempty positive vector; got {requested}"
        )
    if np.any(indices < 0) or np.any(indices >= requested.size):
        raise ValueError(
            f"quality.layer_indices must lie in [0, {requested.size}); got {indices}"
        )
    norm = float(np.linalg.norm(direction))
    if (
        anchor.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(norm)
        or norm <= 0.0
    ):
        raise ValueError(
            f"quality.unit_direction must be a three-vector with finite positive norm and quality.origin of shape (3,); got direction_shape={direction.shape}, origin_shape={anchor.shape}, direction_norm={norm}"
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


@eqx.filter_jit
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
    return sum(moments, jnp.zeros_like(moments[0])), total


def _polyhedral_centroids(
    tables: cell_validity_owner.PolyhedralStarTables,
    cells: np.ndarray,
    coordinates: Array,
    /,
) -> tuple[Array, Array]:
    return _polyhedral_centroid_kernel(
        _prepare_polyhedral_quality(tables, cells), coordinates
    )


@eqx.filter_jit
def _polyhedral_centroid_kernel(
    tables: _PolyhedralQualityPlan, coordinates: Array, /
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
    local = tables.local
    apex = center[tables.star_cell]
    base = face_center[tables.star_face]
    first = coordinates[tables.star_first]
    second = coordinates[tables.star_second]
    volume = (
        _square_determinant(
            jnp.stack((base - apex, first - apex, second - apex), axis=-1)
        )
        / 6.0
    )
    centroid = (apex + base + first + second) / 4.0
    return (
        jax.ops.segment_sum(volume[:, None] * centroid, local, tables.count),
        jax.ops.segment_sum(volume, local, tables.count),
    )


def _facet_geometry(mesh: CellMesh, points: Array, /) -> tuple[Array, Array]:
    """Facet centroids and area-weighted normals in facet entity row order."""

    connectivity = mesh.connectivity
    match connectivity:
        case PolygonalConnectivity() if mesh.topological_dimension == 2:
            return _planar_facet_geometry(points, connectivity.edges)
        case PolyhedralConnectivity():
            tables = cell_validity_owner.polyhedral_star_tables(connectivity)
            return _polyhedral_facet_geometry(
                points,
                tables.face_corner_vertex,
                tables.face_corner_next,
                tables.face_corner_face,
                tables.face_sizes,
            )
        case TetrahedralConnectivity() | HexahedralConnectivity():
            return _loop_facet_geometry(points, connectivity.faces)
        case _:
            raise TypeError(
                f"quality.mesh.connectivity must describe planar or volume facets; got {type(connectivity).__name__}"
            )


@eqx.filter_jit
def _planar_facet_geometry(points: Array, edges: Array, /) -> tuple[Array, Array]:
    start = points[edges[:, 0]]
    stop = points[edges[:, 1]]
    tangent = stop - start
    return 0.5 * (start + stop), jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=1)


@eqx.filter_jit
def _polyhedral_facet_geometry(
    points: Array,
    corner_vertex: np.ndarray,
    corner_next: np.ndarray,
    corner_face: np.ndarray,
    face_sizes: np.ndarray,
    /,
) -> tuple[Array, Array]:
    face_count = face_sizes.size
    corners = points[corner_vertex]
    following = points[corner_next]
    center = (
        jax.ops.segment_sum(corners, corner_face, face_count)
        / jnp.asarray(face_sizes, dtype=points.dtype)[:, None]
    )
    fan = 0.5 * jnp.cross(corners - center[corner_face], following - center[corner_face])
    fan_area = jnp.linalg.norm(fan, axis=-1)
    fan_center = (center[corner_face] + corners + following) / 3.0
    area = jax.ops.segment_sum(fan_area, corner_face, face_count)
    centroid = (
        jax.ops.segment_sum(fan_area[:, None] * fan_center, corner_face, face_count)
        / jnp.maximum(area, _tiny(points))[:, None]
    )
    return centroid, jax.ops.segment_sum(fan, corner_face, face_count)


@eqx.filter_jit
def _loop_facet_geometry(points: Array, faces: Array, /) -> tuple[Array, Array]:
    loops = points[faces]
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


@eqx.filter_jit
def _measure_centroids(
    moments: tuple[Array, ...], volumes: tuple[Array, ...], /
) -> Array:
    volume = jnp.concatenate(volumes)
    return jnp.concatenate(moments) / jnp.where(volume == 0.0, 1.0, volume)[:, None]


@eqx.filter_jit
def _finite_volume_fields(
    centroids: Array,
    face_centroids: Array,
    face_normals: Array,
    face: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
    /,
) -> tuple[Array, Array]:
    delta = centroids[neighbor] - centroids[owner]
    normal = face_normals[face]
    offset = face_centroids[face] - centroids[owner]
    normal = jnp.where(
        jnp.sum(normal * offset, axis=-1, keepdims=True) < 0.0, -normal, normal
    )
    projection = jnp.sum(delta * normal, axis=-1)
    if centroids.shape[-1] == 2:
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
    ) / jnp.maximum(distance, _tiny(centroids))
    return jnp.where(projection == 0.0, jnp.inf, skewness), non_orthogonality


@checked
def evaluate_finite_volume_quality(mesh: CellMesh, /) -> FiniteVolumeQualityEvaluation:
    """Evaluate skewness and non-orthogonality on every interior face."""

    dimension = mesh.topological_dimension
    if dimension not in (2, 3) or mesh.ambient_dimension != dimension:
        raise ValueError(
            f"quality.mesh must contain full-dimensional 2-D or 3-D cells; got dimensions {(dimension, mesh.ambient_dimension)}"
        )
    points = mesh.coordinates
    tables = (
        cell_validity_owner.polyhedral_star_tables(mesh.connectivity)
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
                np.arange(cursor, cursor + block.cell_count, dtype=np.int64),
                points,
            )
        else:
            moment, volume = _block_centroids(
                block.cell_kind, _gather_coordinates(points, block.vertices)
            )
        moments.append(moment)
        volumes.append(volume)
        cursor += block.cell_count
    centroids = _measure_centroids(tuple(moments), tuple(volumes))
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
    skewness, non_orthogonality = _finite_volume_fields(
        centroids, face_centroids, face_normals, face, owner, neighbor
    )
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
