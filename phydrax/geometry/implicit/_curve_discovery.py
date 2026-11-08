#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ..._bvh import bvh_overlap_pairs_host, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._tensor_support import PreparedTensorGrid
from ...typing import checked, parse
from .._certificate import FieldRegularity, SignReliability, ZeroSetAccuracy
from .._contracts import CompiledGeometry, GeometryKind
from ..design._schema import DesignState, ParameterSchema
from ..simplicial._regions import SegmentMesh
from ._discovery import _isolate_roots
from ._policy import AdaptiveImplicitSurfacePolicy, ImplicitSurfacePolicy
from ._projection import (
    _field_and_gradient,
    ImplicitPointProjectionEvidence,
    ImplicitPointProjectionPlan,
)


if TYPE_CHECKING:
    from ._adaptive_discovery import _FieldEnclosures


_DEFAULT_CURVE_POLICY = ImplicitSurfacePolicy()
_CELL_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))
_CELL_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


class ImplicitCurveEvidence(StrictModule):
    projection: ImplicitPointProjectionEvidence
    minimum_edge_length: jax.Array
    minimum_orientation_margin: jax.Array
    intersection_free: jax.Array
    finite: jax.Array
    accepted: jax.Array
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ImplicitCurveRealization(StrictModule):
    proposed_vertices: jax.Array
    vertices: jax.Array
    edges: jax.Array
    evidence: ImplicitCurveEvidence
    source_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> jax.Array:
        return self.evidence.accepted

    @property
    def refresh_required(self) -> jax.Array:
        return ~self.evidence.accepted | self.evidence.projection.refresh_required

    def to_segment_mesh(self, /) -> SegmentMesh:
        if not bool(np.asarray(self.accepted)):
            raise ValueError("Only an accepted implicit curve can be materialized.")
        return SegmentMesh(
            np.asarray(self.vertices),
            np.asarray(self.edges),
            source_id=self.source_id,
        )


def _cross_2d(first: jax.Array, second: jax.Array) -> jax.Array:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _segment_pair_intersects(
    first: jax.Array, second: jax.Array, tolerance: jax.Array | float
) -> jax.Array:
    first_start, first_stop = first
    second_start, second_stop = second
    first_direction = first_stop - first_start
    second_direction = second_stop - second_start
    a = _cross_2d(first_direction, second_start - first_start)
    b = _cross_2d(first_direction, second_stop - first_start)
    c = _cross_2d(second_direction, first_start - second_start)
    d = _cross_2d(second_direction, first_stop - second_start)
    return (a * b < -(tolerance**2)) & (c * d < -(tolerance**2))


class ImplicitCurvePlan(StrictModule):
    """Fixed segment topology with differentiable normal-gauge vertex refresh."""

    projection: ImplicitPointProjectionPlan
    base_vertices: jax.Array
    edges: jax.Array
    base_tangents: jax.Array
    intersection_pairs: jax.Array
    schema: ParameterSchema = eqx.field(static=True)
    policy: ImplicitSurfacePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        projection: ImplicitPointProjectionPlan,
        vertices: npt.ArrayLike,
        edges: npt.ArrayLike,
        intersection_pairs: npt.ArrayLike,
        /,
        *,
        policy: ImplicitSurfacePolicy,
        source_id: str,
    ) -> None:
        vertices_ = np.asarray(vertices, dtype=np.float64)
        edges_ = np.asarray(edges, dtype=np.int32)
        pairs_ = np.asarray(intersection_pairs, dtype=np.int32).reshape((-1, 2))
        vectors = vertices_[edges_[:, 1]] - vertices_[edges_[:, 0]]
        lengths = np.linalg.norm(vectors, axis=-1)
        if np.any(lengths <= 0.0):
            raise ValueError("Implicit curve contains a degenerate segment.")
        topology_id = canonical_fingerprint(
            {
                "kind": "implicit-curve-topology",
                "source": source_id,
                "edges": array_tree_fingerprint(edges_),
            }
        )
        self.projection = projection
        self.base_vertices = jnp.asarray(vertices_)
        self.edges = jnp.asarray(edges_)
        self.base_tangents = jnp.asarray(vectors / lengths[:, None])
        self.intersection_pairs = jnp.asarray(pairs_)
        self.schema = geometry.schema
        self.policy = policy
        self.source_id = source_id
        self.topology_id = topology_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-curve-plan",
                "topology": topology_id,
                "projection": projection.plan_id,
                "policy": repr(policy),
            }
        )

    def realize(self, state: DesignState, /) -> ImplicitCurveRealization:
        if not isinstance(state, DesignState) or state.schema != self.schema:
            raise ValueError("Implicit curve state must use the discovery schema.")
        projection = self.projection.realize(state)
        proposed = projection.proposed_points
        segments = proposed[self.edges]
        vectors = segments[:, 1] - segments[:, 0]
        lengths = jnp.linalg.norm(vectors, axis=-1)
        tangents = vectors / jnp.maximum(
            lengths[:, None],
            jnp.finfo(vectors.dtype).tiny,
        )
        orientation_margin = jnp.min(jnp.sum(tangents * self.base_tangents, axis=-1))
        tolerance = jnp.asarray(
            self.policy.minimum_face_area,
            dtype=lengths.dtype,
        )
        if self.intersection_pairs.shape[0]:
            selected = segments[self.intersection_pairs]
            intersections = jax.vmap(
                lambda pair: _segment_pair_intersects(pair[0], pair[1], tolerance)
            )(selected)
            intersection_free = ~jnp.any(intersections)
        else:
            intersection_free = jnp.asarray(True)
        minimum_length = jnp.min(lengths)
        finite = jnp.all(jnp.isfinite(proposed)) & jnp.all(jnp.isfinite(lengths))
        accepted = (
            projection.accepted
            & finite
            & (minimum_length > tolerance)
            & (orientation_margin > 0.0)
            & intersection_free
        )
        safe = jnp.where(accepted, proposed, self.base_vertices)
        evidence = ImplicitCurveEvidence(
            projection.evidence,
            minimum_length,
            orientation_margin,
            intersection_free,
            finite,
            accepted,
            self.topology_id,
            self.plan_id,
        )
        return ImplicitCurveRealization(
            proposed,
            safe,
            self.edges,
            evidence,
            self.source_id,
        )


def _edge_key(
    cell_i: int, cell_j: int, first: int, second: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    first_offset = _CELL_CORNERS[first]
    second_offset = _CELL_CORNERS[second]
    first_index = (cell_i + first_offset[0], cell_j + first_offset[1])
    second_index = (cell_i + second_offset[0], cell_j + second_offset[1])
    lower, upper = sorted((first_index, second_index))
    return lower, upper


def discover_implicit_curve(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: ImplicitSurfacePolicy = _DEFAULT_CURVE_POLICY,
    source_id: str,
) -> ImplicitCurvePlan:
    """Discover an oriented closed planar zero contour and freeze its segment topology."""
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    if not isinstance(grid, PreparedTensorGrid):
        raise TypeError("grid must be PreparedTensorGrid.")
    if not isinstance(policy, ImplicitSurfacePolicy):
        raise TypeError("policy must be ImplicitSurfacePolicy.")
    if not source_id:
        raise ValueError("source_id must be non-empty.")
    if geometry.ambient_dimension != 2 or geometry.kind is not GeometryKind.REGION:
        raise ValueError("Implicit curve discovery requires a two-dimensional region.")
    if len(grid.structured_axes) != 2:
        raise ValueError("Implicit curve discovery requires a two-dimensional grid.")
    if any(axis.periodic for axis in grid.structured_axes):
        raise ValueError("Implicit curve discovery requires nonperiodic axes.")
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError("Implicit curve discovery geometry must be valid.")
    certificate = geometry.field_certificate
    if certificate.sign_reliability is not SignReliability.RELIABLE:
        raise ValueError("Implicit curve discovery requires reliable field sign.")
    if (
        certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
        and not policy.allow_approximate_zero_set
    ):
        raise ValueError("Approximate zero sets require explicit policy approval.")
    if (
        certificate.regularity is FieldRegularity.NONSMOOTH
        and not policy.allow_nonsmooth_field
    ):
        raise ValueError("Nonsmooth fields require explicit selected-branch approval.")

    axes = tuple(
        np.asarray(axis.point_coordinates, dtype=np.float64)
        for axis in grid.structured_axes
    )
    first, second = np.meshgrid(*axes, indexing="ij")
    lattice_points = np.stack((first, second), axis=-1)
    values = np.asarray(
        geometry.boundary_field(jnp.asarray(lattice_points)),
        dtype=np.float64,
    )
    root_tolerance = float(policy.projection.root_tolerance)
    zero_tolerance = float(policy.lattice_zero_tolerance)
    if np.any(~np.isfinite(values)):
        raise ValueError("Implicit curve lattice values must be finite.")
    if np.any(np.abs(values) <= zero_tolerance):
        raise ValueError(
            "Implicit curve discovery refuses lattice vertices on the zero set."
        )

    vertices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    edge_vertices: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    segments: list[tuple[int, int]] = []
    for cell_i in range(axes[0].size - 1):
        for cell_j in range(axes[1].size - 1):
            corner_indices = tuple(
                (cell_i + offset[0], cell_j + offset[1]) for offset in _CELL_CORNERS
            )
            corner_values = np.asarray(
                [values[index] for index in corner_indices],
                dtype=np.float64,
            )
            crossings = tuple(
                edge
                for edge, (start, stop) in enumerate(_CELL_EDGES)
                if (corner_values[start] < 0.0) != (corner_values[stop] < 0.0)
            )
            if not crossings:
                continue
            if len(crossings) not in (2, 4):
                raise ValueError("Implicit curve cell has an invalid crossing count.")

            crossing_vertices: dict[int, int] = {}
            for edge in crossings:
                start, stop = _CELL_EDGES[edge]
                key = _edge_key(cell_i, cell_j, start, stop)
                vertex = edge_vertices.get(key)
                if vertex is None:
                    vertex = len(vertices)
                    vertices.append(key)
                    edge_vertices[key] = vertex
                crossing_vertices[edge] = vertex

            if len(crossings) == 2:
                pairings = ((crossings[0], crossings[1]),)
            else:
                center = np.asarray(
                    (
                        0.5 * (axes[0][cell_i] + axes[0][cell_i + 1]),
                        0.5 * (axes[1][cell_j] + axes[1][cell_j + 1]),
                    ),
                    dtype=np.float64,
                )
                center_value = float(
                    np.asarray(geometry.boundary_field(jnp.asarray(center)))
                )
                if abs(center_value) <= zero_tolerance:
                    raise ValueError(
                        "Implicit curve ambiguity decider landed on the zero set."
                    )
                if (center_value < 0.0) == (corner_values[0] < 0.0):
                    pairings = ((0, 3), (1, 2))
                else:
                    pairings = ((0, 1), (2, 3))

            for first_edge, second_edge in pairings:
                segments.append(
                    (crossing_vertices[first_edge], crossing_vertices[second_edge])
                )

    if not vertices or not segments:
        raise ValueError("Implicit curve discovery found no closed zero contour.")
    edge_keys = np.asarray(vertices, dtype=np.int64)
    lower_key = tuple(edge_keys[:, 0].T)
    upper_key = tuple(edge_keys[:, 1].T)
    roots, residuals = _isolate_roots(
        geometry.kernel,
        geometry.state,
        jnp.asarray(lattice_points[lower_key]),
        jnp.asarray(lattice_points[upper_key]),
        jnp.asarray(values[lower_key]),
        jnp.asarray(values[upper_key]),
        jnp.asarray(root_tolerance, dtype=jnp.float64),
    )
    if np.any(np.asarray(residuals) > root_tolerance):
        raise ValueError("Implicit root isolation did not meet root_tolerance.")
    vertices_array = np.asarray(roots, dtype=np.float64)
    segment_array = np.asarray(segments, dtype=np.int32)
    tangent = vertices_array[segment_array[:, 1]] - vertices_array[segment_array[:, 0]]
    right_normal = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    _, gradients = _field_and_gradient(
        geometry.kernel,
        geometry.state,
        jnp.asarray(np.mean(vertices_array[segment_array], axis=1)),
    )
    flip = np.sum(right_normal * np.asarray(gradients, dtype=np.float64), axis=1) < 0.0
    segment_array = np.where(flip[:, None], segment_array[:, ::-1], segment_array)
    degree = np.bincount(segment_array.reshape((-1,)), minlength=len(vertices))
    if np.any(degree != 2):
        raise ValueError("Implicit curve discovery produced a nonmanifold open topology.")
    base_segments = vertices_array[segment_array]
    tolerance = policy.projection.root_tolerance
    box_min = np.min(base_segments, axis=1) - tolerance
    box_max = np.max(base_segments, axis=1) + tolerance
    diagonal = float(np.max(np.linalg.norm(box_max - box_min, axis=1)))
    padding = tolerance * (1.0 + 2.0 * diagonal)
    hierarchy = prepare_bvh(box_min, box_max, dtype=jnp.float64)
    first, second = bvh_overlap_pairs_host(
        hierarchy,
        hierarchy,
        include_touching=True,
        absolute_tolerance=padding,
    )
    ordered = first < second
    first = first[ordered]
    second = second[ordered]
    shares_vertex = np.any(
        segment_array[first][:, :, None] == segment_array[second][:, None, :],
        axis=(1, 2),
    )
    intersection_pairs = np.stack(
        (first[~shares_vertex], second[~shares_vertex]), axis=1
    ).astype(np.int32)
    if intersection_pairs.shape[0]:
        intersections = jax.vmap(
            lambda pair: _segment_pair_intersects(
                pair[0],
                pair[1],
                policy.minimum_face_area,
            )
        )(jnp.asarray(base_segments[intersection_pairs]))
        if bool(np.asarray(jnp.any(intersections))):
            raise ValueError("Implicit curve discovery produced a self-intersection.")
    minimum_spacing = min(float(np.min(np.diff(axis))) for axis in axes)
    projection = ImplicitPointProjectionPlan(
        geometry,
        vertices_array,
        policy.projection.trust_fraction * minimum_spacing,
        policy=policy.projection,
        source_id=source_id,
    )
    return ImplicitCurvePlan(
        geometry,
        projection,
        vertices_array,
        segment_array,
        intersection_pairs,
        policy=policy,
        source_id=source_id,
    )


ParametricChartOutcome: TypeAlias = Literal[
    "resolved", "refine", "tangency", "lattice_vertex", "binary64_resolution"
]
"""Decided outcome of a parametric-chart crease discovery.

``resolved``: every face cell is certified (no crease, or one monotone arc
with one root per crossed edge). ``refine``: uncertified cells hold no
candidate tangency, so a finer chart lattice (``required_level``) decides
them. ``tangency``: a piece of the face where the value and both chart
partial derivatives enclose zero survives bisection to ``root_tolerance``,
so a tangential contact of the zero set with the chart surface cannot be
excluded (a certified unresolved event, never meshed). ``lattice_vertex``:
a chart lattice vertex has a value enclosure containing zero; the vertex is
kept by every dyadic refinement, so it is a decided refusal.
``binary64_resolution``: uncertified cells reach ``root_tolerance``.
"""


@final
class ParametricChartCurve(StrictModule, NonTrainableState):
    """Certified zero set of a field restricted to one face of a box chart domain.

    The field is a three-dimensional region field composed with a parametric
    chart map (for example a TPMS field on an envelope parametrization), and
    the face ``axis = domain[side, axis]`` is the parameter domain of one
    envelope surface chart. The face is enclosed by a dyadic quadtree to
    ``target_level``; ``leaf_*`` rows are its leaves with class ``-1``
    (inside, value enclosure negative), ``+1`` (outside) or ``0`` (a
    target-level band cell). A band cell is certified when its corners share
    a sign proved on the whole cell (no crease), or when one chart partial
    derivative excludes zero on the cell (the crease is a graph, with no
    closed loop or tangency) and every crossed edge has a monotone field
    along it (one root). Its crease segment joins the two edge roots.

    Edges and corners are named by three-dimensional corner ids at
    ``target_level`` (lexicographically ordered edge endpoints), the same
    identities as the volume discovery lattice, and roots are isolated by the
    discovery root kernel, so ``root_*`` is the crease root bank shared with
    clipped volume discovery. ``segments`` are unordered root-row pairs.
    """

    domain: np.ndarray
    leaf_levels: np.ndarray
    leaf_indices: np.ndarray
    leaf_classes: np.ndarray
    corner_ids: np.ndarray
    corner_signs: np.ndarray
    root_edge_ids: np.ndarray
    root_points: np.ndarray
    root_errors: np.ndarray
    root_issues: np.ndarray
    root_brackets: np.ndarray
    root_signs: np.ndarray
    segments: np.ndarray
    segment_cells: np.ndarray
    unresolved_cells: np.ndarray
    event_boxes: np.ndarray
    axis: int = eqx.field(static=True)
    side: int = eqx.field(static=True)
    target_level: int = eqx.field(static=True)
    required_level: int = eqx.field(static=True)
    outcome: ParametricChartOutcome = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        domain: np.ndarray,
        axis: int,
        side: int,
        target_level: int,
        arrays: dict[str, np.ndarray],
        required_level: int,
        outcome: ParametricChartOutcome,
        source_id: str,
    ) -> None:
        def frozen(name: str, dtype: type, shape: tuple[int, ...]) -> np.ndarray:
            value = np.array(arrays[name], dtype=dtype, copy=True).reshape(shape)
            value.setflags(write=False)
            return value

        box = np.array(domain, dtype=np.float64, copy=True).reshape((2, 3))
        box.setflags(write=False)
        self.domain = box
        self.leaf_levels = frozen("leaf_levels", np.int64, (-1,))
        self.leaf_indices = frozen("leaf_indices", np.int64, (-1, 2))
        self.leaf_classes = frozen("leaf_classes", np.int8, (-1,))
        self.corner_ids = frozen("corner_ids", np.int64, (-1, 3))
        self.corner_signs = frozen("corner_signs", np.int8, (-1,))
        self.root_edge_ids = frozen("root_edge_ids", np.int64, (-1, 2, 3))
        self.root_points = frozen("root_points", np.float64, (-1, 3))
        self.root_errors = frozen("root_errors", np.float64, (-1,))
        self.root_issues = frozen("root_issues", np.int64, (-1,))
        self.root_brackets = frozen("root_brackets", np.float64, (-1, 2, 3))
        self.root_signs = frozen("root_signs", np.int8, (-1, 2))
        self.segments = frozen("segments", np.int64, (-1, 2))
        self.segment_cells = frozen("segment_cells", np.int64, (-1, 2))
        self.unresolved_cells = frozen("unresolved_cells", np.int64, (-1, 2))
        self.event_boxes = frozen("event_boxes", np.float64, (-1, 2, 3))
        if axis not in (0, 1, 2) or side not in (0, 1):
            raise ValueError("A chart face needs axis in 0..2 and side in 0..1.")
        self.axis = int(axis)
        self.side = int(side)
        self.target_level = int(target_level)
        self.required_level = int(required_level)
        self.outcome = parse(outcome, ParametricChartOutcome, "outcome")
        self.source_id = source_id

    @property
    def resolved(self) -> bool:
        return self.outcome == "resolved"

    @property
    def chart_axes(self) -> tuple[int, int]:
        """The two free parameter axes of the face, in increasing order."""
        first, second = (other for other in range(3) if other != self.axis)
        return first, second


def _face_box(
    domain: np.ndarray,
    axis: int,
    side: int,
    level: int,
    indices: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward-rounded flat boxes of face cells ``indices`` at ``level``."""
    from ._adaptive_discovery import _grid_coordinate_fraction, _outward_coordinate

    count = indices.shape[0]
    lower = np.empty((count, 3), dtype=np.float64)
    upper = np.empty((count, 3), dtype=np.float64)
    lower[:, axis] = upper[:, axis] = domain[side, axis]
    free = [other for other in range(3) if other != axis]
    for column, target in enumerate(free):
        values, inverse = np.unique(indices[:, column], return_inverse=True)
        low = [
            _outward_coordinate(
                _grid_coordinate_fraction(
                    int(i), level, domain[0, target], domain[1, target]
                ),
                lower=True,
            )
            for i in values
        ]
        high = [
            _outward_coordinate(
                _grid_coordinate_fraction(
                    int(i) + 1, level, domain[0, target], domain[1, target]
                ),
                lower=False,
            )
            for i in values
        ]
        lower[:, target] = np.asarray(low, dtype=np.float64)[inverse.reshape(-1)]
        upper[:, target] = np.asarray(high, dtype=np.float64)[inverse.reshape(-1)]
    return lower, upper


def _excludes_zero(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    return (lower > 0.0) | (upper < 0.0)


def _tangency_search(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    free: tuple[int, int],
    tolerance: float,
    /,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Bisect uncertified face boxes until each piece excludes a tangency.

    A piece is decided when its value enclosure excludes zero or one chart
    partial derivative does (the crease there is a regular graph, resolvable
    by a finer lattice). Returns the deepest bisection depth needed and the
    pieces that reached ``tolerance`` (or binary64) undecided: candidate
    tangential contacts.
    """
    from ._adaptive_discovery import _norm_upper_rows

    depth = 0
    low, high = lower.copy(), upper.copy()
    stuck_low: list[np.ndarray] = []
    stuck_high: list[np.ndarray] = []
    while low.shape[0]:
        evidence = bounds.boxes(low, high)
        value = _excludes_zero(evidence.value_lower, evidence.value_upper)
        slope = np.any(
            _excludes_zero(
                evidence.gradient_lower[:, list(free)],
                evidence.gradient_upper[:, list(free)],
            ),
            axis=1,
        )
        open_rows = ~(value | slope)
        low, high = low[open_rows], high[open_rows]
        if not low.shape[0]:
            break
        middle = 0.5 * low + 0.5 * high
        splittable = np.zeros(low.shape, dtype=np.bool_)
        for target in free:
            splittable[:, target] = (middle[:, target] > low[:, target]) & (
                middle[:, target] < high[:, target]
            )
        small = _norm_upper_rows(high - low) <= tolerance
        terminal = small | ~np.all(splittable[:, list(free)], axis=1)
        stuck_low.append(low[terminal])
        stuck_high.append(high[terminal])
        low, high, middle = low[~terminal], high[~terminal], middle[~terminal]
        if not low.shape[0]:
            break
        depth += 1
        children_low: list[np.ndarray] = []
        children_high: list[np.ndarray] = []
        first, second = free
        for take_first in (False, True):
            for take_second in (False, True):
                child_low = low.copy()
                child_high = high.copy()
                for target, upper_half in ((first, take_first), (second, take_second)):
                    if upper_half:
                        child_low[:, target] = middle[:, target]
                    else:
                        child_high[:, target] = middle[:, target]
                children_low.append(child_low)
                children_high.append(child_high)
        low = np.concatenate(children_low)
        high = np.concatenate(children_high)
    stuck = np.concatenate(stuck_low) if stuck_low else np.empty((0, 3), dtype=np.float64)
    stuck_upper = (
        np.concatenate(stuck_high) if stuck_high else np.empty((0, 3), dtype=np.float64)
    )
    return depth, stuck, stuck_upper


class _CurvatureEnclosures:
    """Outward interval enclosure of the Hessian diagonal of a region field."""

    def __init__(self, geometry: CompiledGeometry, /) -> None:
        from .._interval_enclosure import prepare_interval_function

        kernel, state = geometry.kernel, geometry.state

        def scalar(point: jax.Array) -> jax.Array:
            return kernel.boundary_field(state, point[None, :])[0]

        def second(point: jax.Array, direction: jax.Array) -> jax.Array:
            def slope(at: jax.Array) -> jax.Array:
                return jax.jvp(scalar, (at,), (direction,))[1]

            return jax.jvp(slope, (point,), (direction,))[1]

        def diagonal(point: jax.Array) -> jax.Array:
            return jnp.stack([second(point, jnp.asarray(unit)) for unit in np.eye(3)])

        self._program = prepare_interval_function(diagonal, 3)

    def boxes(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        low, high = self._program.evaluate(lower, upper)
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def _edges_by_curvature(
    bounds: _FieldEnclosures,
    curvature: _CurvatureEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    direction: int,
    end_signs: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    """Root-count proofs of grid edges from a strictly signed second derivative.

    With ``f''`` of one strict sign along an edge, ends of opposite sign
    bracket exactly one root, and ends of sign ``s`` with ``s f''`` negative
    leave none (``s f`` is concave, so it stays above its end values). For
    ``s f`` convex, each piece is proved free of roots by its value enclosure
    or by the Taylor bound ``s f >= s f(c) - f'(c)^2 / (2 m)`` with
    ``m = min s f''``; other pieces are bisected until proved or until a piece
    reaches ``tolerance`` or binary64 (the edge stays unproved). A piece centre
    with certified ``s f(c) < 0`` decides the edge unproved at once: the edge
    then holds two roots, and bisecting the dip between them to ``tolerance``
    would double its pieces at every level.
    """
    count = lower.shape[0]
    proved = np.zeros(count, dtype=np.bool_)
    low_curvature, high_curvature = curvature.boxes(lower, upper)
    convex = low_curvature[:, direction] > 0.0
    concave = high_curvature[:, direction] < 0.0
    signed = convex | concave
    opposite = end_signs[:, 0] != end_signs[:, 1]
    sign = end_signs[:, 0].astype(np.float64)
    proved[signed & opposite] = True
    hump = signed & ~opposite & np.where(sign > 0, convex, concave)
    proved[signed & ~opposite & ~hump] = True
    rows = np.flatnonzero(hump)
    owner = rows.copy()
    low, high = lower[rows].copy(), upper[rows].copy()
    failed = np.zeros(count, dtype=np.bool_)
    while owner.size:
        piece_sign = sign[owner]
        evidence = bounds.boxes(low, high)
        value_low = np.where(piece_sign > 0, evidence.value_lower, -evidence.value_upper)
        curve_low, curve_high = curvature.boxes(low, high)
        minimum = np.where(
            piece_sign > 0, curve_low[:, direction], -curve_high[:, direction]
        )
        middle = low.copy()
        middle[:, direction] = 0.5 * low[:, direction] + 0.5 * high[:, direction]
        center = bounds.boxes(middle, middle)
        center_low = np.where(piece_sign > 0, center.value_lower, -center.value_upper)
        slope = np.maximum(
            np.abs(center.gradient_lower[:, direction]),
            np.abs(center.gradient_upper[:, direction]),
        )
        drop = np.nextafter(
            np.nextafter(slope * slope, np.inf)
            / np.nextafter(2.0 * np.where(minimum > 0.0, minimum, 1.0), -np.inf),
            np.inf,
        )
        taylor = np.nextafter(center_low - drop, -np.inf)
        settled = (value_low > 0.0) | ((minimum > 0.0) & (taylor > 0.0))
        failed[owner[minimum <= 0.0]] = True
        dip = np.where(piece_sign > 0, center.value_upper, -center.value_lower) < 0.0
        failed[owner[dip]] = True
        splittable = (middle[:, direction] > low[:, direction]) & (
            middle[:, direction] < high[:, direction]
        )
        small = high[:, direction] - low[:, direction] <= tolerance
        failed[owner[~settled & (~splittable | small)]] = True
        keep = ~settled & splittable & ~small & ~failed[owner]
        owner, low, high, middle = owner[keep], low[keep], high[keep], middle[keep]
        left_high = high.copy()
        left_high[:, direction] = middle[:, direction]
        right_low = low.copy()
        right_low[:, direction] = middle[:, direction]
        owner = np.concatenate((owner, owner))
        low = np.concatenate((low, right_low))
        high = np.concatenate((left_high, high))
    proved[rows] = ~failed[rows]
    return proved


def _proved_cell_edges(
    bounds: _FieldEnclosures,
    curvature: _CurvatureEnclosures,
    corners: np.ndarray,
    signs: np.ndarray,
    monotone: np.ndarray,
    edge_axis: tuple[int, int, int, int],
    first: int,
    target_level: int,
    domain: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    """Root-count proofs of the four edges of face cells.

    Edge ``k`` joins cell corners ``k`` and ``k + 1 (mod 4)``. An edge is
    proved when its root count is the one its end signs imply (one for
    opposite signs, none for equal signs): by a monotone cell or edge, a value
    enclosure excluding zero, a strictly signed second derivative
    (``_edges_by_curvature``), or a subdivision count
    (``_edges_by_subdivision``).
    """
    from ._adaptive_packed_polygon import _grid_points

    proved = np.ones((corners.shape[0], 4), dtype=np.bool_)
    for edge in range(4):
        start, stop = edge, (edge + 1) % 4
        along = 0 if edge_axis[edge] == first else 1
        check = np.flatnonzero(~monotone[:, along])
        if not check.size:
            continue
        ends = corners[check][:, (start, stop)]
        points = _grid_points(ends.reshape(-1, 3), target_level, domain).reshape(-1, 2, 3)
        edge_low = np.minimum(points[:, 0], points[:, 1])
        edge_high = np.maximum(points[:, 0], points[:, 1])
        evidence = bounds.boxes(edge_low, edge_high)
        no_root = _excludes_zero(evidence.value_lower, evidence.value_upper)
        one_root = _excludes_zero(
            evidence.gradient_lower[:, edge_axis[edge]],
            evidence.gradient_upper[:, edge_axis[edge]],
        )
        settled = no_root | one_root
        pending = np.flatnonzero(~settled)
        if pending.size:
            settled[pending] = _edges_by_curvature(
                bounds,
                curvature,
                edge_low[pending],
                edge_high[pending],
                edge_axis[edge],
                signs[check[pending]][:, (start, stop)],
                tolerance,
            )
        pending = np.flatnonzero(~settled)
        if pending.size:
            settled[pending] = _edges_by_subdivision(
                bounds,
                edge_low[pending],
                edge_high[pending],
                edge_axis[edge],
                signs[check[pending]][:, (start, stop)],
                tolerance,
            )
        proved[check, edge] = settled
    return proved


def _edges_by_subdivision(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    direction: int,
    end_signs: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    """Count the roots of grid edges exactly by subdivision.

    Each piece is decided when its value enclosure excludes zero (no root) or
    its derivative along the edge excludes zero with certified nonzero signs
    at both piece ends (one root iff those signs differ). Undecided pieces are
    bisected; an edge stays unproved when a piece reaches ``tolerance`` or
    binary64 undecided, or as soon as its counted roots exceed the count its
    end signs imply. Away from a double root every piece is decided at a
    finite depth, so only pieces at a tangential root reach ``tolerance``.
    """
    count = lower.shape[0]
    expected = (end_signs[:, 0] != end_signs[:, 1]).astype(np.int64)
    roots = np.zeros(count, dtype=np.int64)
    failed = np.zeros(count, dtype=np.bool_)
    owner = np.arange(count)
    low, high = lower.copy(), upper.copy()

    def certified_signs(points: np.ndarray, /) -> np.ndarray:
        point_lower, point_upper = bounds.point_values(points)
        return np.where(point_lower > 0.0, 1, np.where(point_upper < 0.0, -1, 0))

    while owner.size:
        evidence = bounds.boxes(low, high)
        empty = _excludes_zero(evidence.value_lower, evidence.value_upper)
        monotone = ~empty & _excludes_zero(
            evidence.gradient_lower[:, direction], evidence.gradient_upper[:, direction]
        )
        stop = low.copy()
        stop[:, direction] = high[:, direction]
        start_sign = np.zeros(owner.size, dtype=np.int64)
        stop_sign = np.zeros(owner.size, dtype=np.int64)
        rows = np.flatnonzero(monotone)
        if rows.size:
            start_sign[rows] = certified_signs(low[rows])
            stop_sign[rows] = certified_signs(stop[rows])
        counted = monotone & (start_sign != 0) & (stop_sign != 0)
        np.add.at(roots, owner[counted], (start_sign != stop_sign)[counted])
        failed |= roots > expected
        open_rows = ~(empty | counted) & ~failed[owner]
        owner, low, high = owner[open_rows], low[open_rows], high[open_rows]
        middle = 0.5 * low[:, direction] + 0.5 * high[:, direction]
        splittable = (middle > low[:, direction]) & (middle < high[:, direction])
        small = high[:, direction] - low[:, direction] <= tolerance
        failed[owner[small | ~splittable]] = True
        keep = ~failed[owner]
        owner, low, high, middle = owner[keep], low[keep], high[keep], middle[keep]
        left_high = high.copy()
        left_high[:, direction] = middle
        right_low = low.copy()
        right_low[:, direction] = middle
        owner = np.concatenate((owner, owner))
        low = np.concatenate((low, right_low))
        high = np.concatenate((left_high, high))
    return ~failed & (roots == expected)


def _directional_enclosure(
    gradient_lower: np.ndarray, gradient_upper: np.ndarray, directions: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward enclosure of ``d . grad`` from two gradient interval columns."""
    left = directions * gradient_lower
    right = directions * gradient_upper
    zero = directions == 0.0
    low = np.where(zero, 0.0, np.nextafter(np.minimum(left, right), -np.inf))
    high = np.where(zero, 0.0, np.nextafter(np.maximum(left, right), np.inf))
    return (
        np.nextafter(low[:, 0] + low[:, 1], -np.inf),
        np.nextafter(high[:, 0] + high[:, 1], np.inf),
    )


def _directional_arc_certificate(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    free: tuple[int, int],
    directions: np.ndarray,
    sign: int,
    tolerance: float,
    /,
) -> np.ndarray:
    """Prove ``sign * d . grad F > 0`` wherever ``F`` may vanish in each face box.

    Pieces are bisected in the chart axes until their value enclosure
    excludes zero or their directional-derivative enclosure has the required
    strict sign. A box fails when a piece reaches ``tolerance`` or binary64
    undecided, or when a piece of strictly opposite slope has corners of both
    certified signs (it then holds a zero of the wrong slope).
    """
    from ._adaptive_discovery import _norm_upper_rows

    count = lower.shape[0]
    failed = np.zeros(count, dtype=np.bool_)
    owner = np.arange(count)
    low, high = lower.copy(), upper.copy()
    first, second = free
    corner_masks = ((False, False), (True, False), (False, True), (True, True))
    while owner.size:
        evidence = bounds.boxes(low, high)
        value_free = _excludes_zero(evidence.value_lower, evidence.value_upper)
        slope_low, slope_high = _directional_enclosure(
            evidence.gradient_lower[:, list(free)],
            evidence.gradient_upper[:, list(free)],
            directions[owner],
        )
        right = slope_low > 0.0 if sign > 0 else slope_high < 0.0
        wrong = slope_high < 0.0 if sign > 0 else slope_low > 0.0
        open_rows = ~(value_free | right)
        suspects = np.flatnonzero(open_rows & wrong)
        if suspects.size:
            points = np.repeat(low[suspects][:, None, :], 4, axis=1)
            for corner, (take_first, take_second) in enumerate(corner_masks):
                if take_first:
                    points[:, corner, first] = high[suspects, first]
                if take_second:
                    points[:, corner, second] = high[suspects, second]
            point_lower, point_upper = bounds.point_values(points.reshape(-1, 3))
            positive = np.any((point_lower > 0.0).reshape(-1, 4), axis=1)
            negative = np.any((point_upper < 0.0).reshape(-1, 4), axis=1)
            failed[owner[suspects[positive & negative]]] = True
        open_rows &= ~failed[owner]
        owner, low, high = owner[open_rows], low[open_rows], high[open_rows]
        if not owner.size:
            break
        middle = 0.5 * low + 0.5 * high
        splittable = np.all(
            (middle[:, list(free)] > low[:, list(free)])
            & (middle[:, list(free)] < high[:, list(free)]),
            axis=1,
        )
        small = _norm_upper_rows(high - low) <= tolerance
        failed[owner[small | ~splittable]] = True
        keep = ~failed[owner]
        owner, low, high, middle = owner[keep], low[keep], high[keep], middle[keep]
        children_owner: list[np.ndarray] = []
        children_low: list[np.ndarray] = []
        children_high: list[np.ndarray] = []
        for take_first, take_second in corner_masks:
            child_low, child_high = low.copy(), high.copy()
            for target, upper_half in ((first, take_first), (second, take_second)):
                if upper_half:
                    child_low[:, target] = middle[:, target]
                else:
                    child_high[:, target] = middle[:, target]
            children_owner.append(owner)
            children_low.append(child_low)
            children_high.append(child_high)
        owner = np.concatenate(children_owner)
        low = np.concatenate(children_low)
        high = np.concatenate(children_high)
    return ~failed


def _locally_certified_cells(
    bounds: _FieldEnclosures,
    lower: np.ndarray,
    upper: np.ndarray,
    free: tuple[int, int],
    tolerance: float,
    /,
) -> np.ndarray:
    """Local topology certificate of face cells with proved edges and at most two crossings.

    If ``s d . grad F > 0`` at every zero of ``F`` in the closed cell, every
    line along ``d`` meets the zero set at most once and transversally, so
    each zero-set component in the cell is a graph over an interval of the
    orthogonal coordinate whose two distinct endpoints are boundary roots.
    With the edge root counts proved, a cell with two crossed edges holds
    exactly one arc joining them and a cell without crossings holds none.
    Candidates ``(d, s)`` are the normalized gradient at the cell centre
    (``s = 1``), then both signs of the chart axes and diagonals; each is
    proved by subdivision of the cell alone (``_directional_arc_certificate``).
    """
    count = lower.shape[0]
    proved = np.zeros(count, dtype=np.bool_)
    middle = 0.5 * lower + 0.5 * upper
    center = bounds.boxes(middle, middle)
    gradient = (
        0.5 * center.gradient_lower[:, list(free)]
        + 0.5 * (center.gradient_upper[:, list(free)])
    )
    norm = np.linalg.norm(gradient, axis=1)
    usable = np.isfinite(norm) & (norm > 0.0)
    candidates: list[tuple[np.ndarray, int, np.ndarray]] = [
        (gradient / np.where(usable, norm, 1.0)[:, None], 1, usable)
    ]
    diagonal = math.sqrt(0.5)
    for direction in (
        (1.0, 0.0),
        (0.0, 1.0),
        (diagonal, diagonal),
        (diagonal, -diagonal),
    ):
        for sign in (1, -1):
            candidates.append(
                (
                    np.tile(np.asarray(direction, dtype=np.float64), (count, 1)),
                    sign,
                    np.ones(count, dtype=np.bool_),
                )
            )
    for directions, sign, allowed in candidates:
        rows = np.flatnonzero(~proved & allowed)
        if not rows.size:
            continue
        proved[rows] = _directional_arc_certificate(
            bounds, lower[rows], upper[rows], free, directions[rows], sign, tolerance
        )
    return proved


def _discover_chart_face(
    bounds: _FieldEnclosures,
    curvature: _CurvatureEnclosures,
    domain: np.ndarray,
    axis: int,
    side: int,
    target_level: int,
    root_tolerance: float,
    source_id: str,
    /,
) -> ParametricChartCurve:
    """Parametric-chart crease discovery on one face with prepared enclosures."""
    from ._adaptive_discovery import (
        _norm_upper_rows,
        _prove_constant_signs,
        _root_on_edges,
    )
    from ._adaptive_packed_polygon import _grid_points

    span = 1 << target_level
    first, second = (other for other in range(3) if other != axis)
    free = (first, second)
    leaf_levels: list[np.ndarray] = []
    leaf_indices: list[np.ndarray] = []
    leaf_classes: list[np.ndarray] = []
    frontier = np.zeros((1, 2), dtype=np.int64)
    band = np.empty((0, 2), dtype=np.int64)
    band_evidence = None
    for level in range(target_level + 1):
        lower, upper = _face_box(domain, axis, side, level, frontier)
        evidence = bounds.boxes(lower, upper)
        inside = evidence.value_upper < 0.0
        outside = evidence.value_lower > 0.0
        decided = inside | outside
        leaf_levels.append(np.full(int(np.count_nonzero(decided)), level, np.int64))
        leaf_indices.append(frontier[decided])
        leaf_classes.append(np.where(inside[decided], -1, 1).astype(np.int8))
        if level == target_level:
            band = frontier[~decided]
            band_evidence = (
                lower[~decided],
                upper[~decided],
                evidence.gradient_lower[~decided],
                evidence.gradient_upper[~decided],
            )
            break
        parents = frontier[~decided]
        frontier = (
            2 * parents[:, None, :]
            + np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=np.int64)[None]
        ).reshape(-1, 2)
    assert band_evidence is not None
    band_lower, band_upper, gradient_lower, gradient_upper = band_evidence
    leaf_levels.append(np.full(band.shape[0], target_level, np.int64))
    leaf_indices.append(band)
    leaf_classes.append(np.zeros(band.shape[0], dtype=np.int8))

    # Face corners as three-dimensional corner ids at the target level.
    offsets = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int64)
    cell_corners = np.zeros((band.shape[0], 4, 3), dtype=np.int64)
    cell_corners[:, :, axis] = 0 if side == 0 else span
    cell_corners[:, :, first] = band[:, None, 0] + offsets[None, :, 0]
    cell_corners[:, :, second] = band[:, None, 1] + offsets[None, :, 1]
    corner_ids, corner_rows = np.unique(
        cell_corners.reshape(-1, 3), axis=0, return_inverse=True
    )
    corner_rows = corner_rows.reshape(-1, 4)
    corner_points = _grid_points(corner_ids, target_level, domain)
    point_lower, point_upper = bounds.point_values(corner_points)
    corner_signs = np.zeros(corner_ids.shape[0], dtype=np.int8)
    corner_signs[point_lower > 0.0] = 1
    corner_signs[point_upper < 0.0] = -1
    cell_signs = corner_signs[corner_rows]

    unresolved = np.zeros(band.shape[0], dtype=np.bool_)
    empty = np.zeros(band.shape[0], dtype=np.bool_)
    zero_vertex = np.any(cell_signs == 0, axis=1)
    same = ~zero_vertex & np.all(cell_signs == cell_signs[:, :1], axis=1)
    same_rows = np.flatnonzero(same)
    if same_rows.size:
        proved = _prove_constant_signs(
            bounds,
            band_lower[same_rows],
            band_upper[same_rows],
            cell_signs[same_rows, 0],
            gradient_lower[same_rows],
            gradient_upper[same_rows],
        )
        empty[same_rows[proved]] = True
        unresolved[same_rows[~proved]] = True
    mixed = ~zero_vertex & ~same
    edge_axis = (first, second, first, second)
    monotone = _excludes_zero(
        gradient_lower[:, list(free)], gradient_upper[:, list(free)]
    )
    edges_proved = np.zeros(band.shape[0], dtype=np.bool_)
    crease_cell = np.zeros(band.shape[0], dtype=np.bool_)

    def prove_edges(rows: np.ndarray, /) -> None:
        edges_proved[rows] = np.all(
            _proved_cell_edges(
                bounds,
                curvature,
                cell_corners[rows],
                cell_signs[rows],
                monotone[rows],
                edge_axis,
                first,
                target_level,
                domain,
                root_tolerance,
            ),
            axis=1,
        )

    mixed_rows = np.flatnonzero(mixed)
    if mixed_rows.size:
        prove_edges(mixed_rows)
        certified = np.any(monotone[mixed_rows], axis=1) & edges_proved[mixed_rows]
        crease_cell[mixed_rows[certified]] = True
        unresolved[mixed_rows[~certified]] = True
    # Only the cells left unresolved are subdivided for the local certificate;
    # the lattice is refined only if one of them still fails.
    retry = np.flatnonzero(unresolved)
    if retry.size:
        pending_same = retry[~mixed[retry]]
        if pending_same.size:
            prove_edges(pending_same)
        signs = cell_signs[retry]
        crossings = np.count_nonzero(signs != np.roll(signs, -1, axis=1), axis=1)
        candidates = retry[edges_proved[retry] & (crossings <= 2)]
        if candidates.size:
            proved = candidates[
                _locally_certified_cells(
                    bounds,
                    band_lower[candidates],
                    band_upper[candidates],
                    free,
                    root_tolerance,
                )
            ]
            unresolved[proved] = False
            crease_cell[proved[mixed[proved]]] = True
            empty[proved[~mixed[proved]]] = True

    event_lower: list[np.ndarray] = []
    event_upper: list[np.ndarray] = []
    required_level = target_level
    if np.any(zero_vertex):
        outcome: ParametricChartOutcome = "lattice_vertex"
        event_lower.append(band_lower[zero_vertex])
        event_upper.append(band_upper[zero_vertex])
    elif np.any(unresolved):
        rows = np.flatnonzero(unresolved)
        depth, stuck_lower, stuck_upper = _tangency_search(
            bounds, band_lower[rows], band_upper[rows], free, root_tolerance
        )
        if stuck_lower.shape[0]:
            outcome = "tangency"
            event_lower.append(stuck_lower)
            event_upper.append(stuck_upper)
        elif (
            float(np.max(_norm_upper_rows(band_upper[rows] - band_lower[rows])))
            <= root_tolerance
        ):
            outcome = "binary64_resolution"
            event_lower.append(band_lower[rows])
            event_upper.append(band_upper[rows])
        else:
            outcome = "refine"
            required_level = target_level + max(depth, 1)
    else:
        outcome = "resolved"

    # Crease root bank: the two crossed edges of every certified crease cell,
    # in cell-edge order, each oriented from its lexicographically lower corner
    # (the corners differ by one in a single coordinate).
    crossing_rows = np.flatnonzero(crease_cell)
    if crossing_rows.size:
        starts = np.asarray((0, 1, 2, 3), dtype=np.int64)
        stops = np.asarray((1, 2, 3, 0), dtype=np.int64)
        signs = cell_signs[crossing_rows]
        crossed = signs[:, starts] != signs[:, stops]
        if np.any(np.count_nonzero(crossed, axis=1) != 2):
            raise RuntimeError("A certified crease cell must cross exactly two edges.")
        row_index, edge_index = np.nonzero(crossed)
        cells = crossing_rows[row_index]
        ends = np.stack(
            (
                cell_corners[cells, starts[edge_index]],
                cell_corners[cells, stops[edge_index]],
            ),
            axis=1,
        )
        end_signs = np.stack(
            (signs[row_index, starts[edge_index]], signs[row_index, stops[edge_index]]),
            axis=1,
        )
        swap = np.sum(ends[:, 1], axis=1) < np.sum(ends[:, 0], axis=1)
        ends = np.where(swap[:, None, None], ends[:, ::-1], ends)
        end_signs = np.where(swap[:, None], end_signs[:, ::-1], end_signs)
        root_edge_ids, first_rows, segment_roots = np.unique(
            np.ascontiguousarray(ends.reshape(-1, 6), dtype=np.int64),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        root_edge_ids = root_edge_ids.reshape(-1, 2, 3)
        segments = segment_roots.reshape(-1, 2)
        endpoint_points = _grid_points(
            root_edge_ids.reshape(-1, 3), target_level, domain
        ).reshape(-1, 2, 3)
        endpoint_signs = np.ascontiguousarray(end_signs[first_rows], dtype=np.int8)
        records = _root_on_edges(bounds, endpoint_points, endpoint_signs, root_tolerance)
        root_points = np.stack([record[0] for record in records])
        root_errors = np.asarray([record[1] for record in records], dtype=np.float64)
        root_issues = np.asarray([record[2] for record in records], dtype=np.int64)
        root_brackets = np.stack([np.stack((record[3], record[4])) for record in records])
        root_signs = np.asarray(
            [(record[5], record[6]) for record in records], dtype=np.int8
        )
        if np.any(root_issues != 0) and outcome == "resolved":
            outcome = "binary64_resolution"
    else:
        root_edge_ids = np.empty((0, 2, 3), dtype=np.int64)
        segments = np.empty((0, 2), dtype=np.int64)
        root_points = np.empty((0, 3), dtype=np.float64)
        root_errors = np.empty((0,), dtype=np.float64)
        root_issues = np.empty((0,), dtype=np.int64)
        root_brackets = np.empty((0, 2, 3), dtype=np.float64)
        root_signs = np.empty((0, 2), dtype=np.int8)
    arrays = {
        "leaf_levels": np.concatenate(leaf_levels),
        "leaf_indices": np.concatenate(leaf_indices),
        "leaf_classes": np.concatenate(leaf_classes),
        "corner_ids": corner_ids,
        "corner_signs": corner_signs,
        "root_edge_ids": root_edge_ids,
        "root_points": root_points,
        "root_errors": root_errors,
        "root_issues": root_issues,
        "root_brackets": root_brackets,
        "root_signs": root_signs,
        "segments": segments,
        "segment_cells": band[crossing_rows],
        "unresolved_cells": band[unresolved | zero_vertex],
        "event_boxes": (
            np.stack((np.concatenate(event_lower), np.concatenate(event_upper)), axis=1)
            if event_lower
            else np.empty((0, 2, 3), dtype=np.float64)
        ),
    }
    return ParametricChartCurve(
        domain=domain,
        axis=axis,
        side=side,
        target_level=target_level,
        arrays=arrays,
        required_level=required_level,
        outcome=outcome,
        source_id=source_id,
    )


@checked
def discover_parametric_chart_curve(
    geometry: CompiledGeometry,
    /,
    *,
    domain: npt.ArrayLike,
    axis: int,
    side: int,
    target_level: int,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    source_id: str,
) -> ParametricChartCurve:
    """Discover the certified zero set of ``geometry`` on one face of ``domain``.

    Parametric-chart mode of two-dimensional implicit curve discovery: the
    three-dimensional region field is evaluated on the face
    ``x[axis] = domain[side, axis]``, whose two free coordinates are the
    chart parameters of an envelope surface. Certification uses rigorous
    interval value and gradient enclosures (``policy.enclosure`` must be
    ``"interval"``) and ``policy.root_tolerance``; see
    ``ParametricChartCurve`` and ``ParametricChartOutcome``.
    """
    from ._adaptive_discovery import _FieldEnclosures, _validate_discovery_inputs

    _validate_discovery_inputs(geometry, policy, source_id)
    if policy.enclosure != "interval":
        raise ValueError("Parametric-chart crease discovery needs interval enclosures.")
    box = np.asarray(domain, dtype=np.float64)
    if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] >= box[1]):
        raise ValueError("domain must be a finite ordered (2, 3) box.")
    if isinstance(target_level, bool) or not 0 <= int(target_level) <= 30:
        raise ValueError("target_level must be an integer in [0, 30].")
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    if not bounds.gradient_rigorous:
        raise ValueError("Parametric-chart crease discovery needs gradient enclosures.")
    return _discover_chart_face(
        bounds,
        _CurvatureEnclosures(geometry),
        box,
        int(axis),
        int(side),
        int(target_level),
        float(policy.root_tolerance),
        source_id,
    )


__all__ = [
    "discover_implicit_curve",
    "discover_parametric_chart_curve",
    "ImplicitCurveEvidence",
    "ImplicitCurvePlan",
    "ImplicitCurveRealization",
    "ParametricChartCurve",
    "ParametricChartOutcome",
]
