#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native Booleans of closed oriented triangle surfaces.

All operands join one exact arrangement over their original source features;
there are no rounded intermediate binary Booleans. Fragments of one operand
form components bounded by contact curves, and the native arrangement decides
each component's membership in every other operand exactly: the winding
number is counted along a symbolically perturbed ray from the centroid of the
component's implicit corners with filtered and exact dyadic predicates.
Coplanar coverage uses native exact orientation and emits a region once, from
the lowest-index contributing original operand. Empty and disconnected results
are legitimate; canceled subexpressions cannot own surviving face properties.
"""

from __future__ import annotations

from enum import StrEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike
from scipy.sparse.csgraph import connected_components

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    ARRANGEMENT_UNCLASSIFIED,
    ArrangementClassification,
    MeshcoreStatus,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...typing import Dim, HostBool, HostFloat64, HostInt64, parse, Scope
from ._arrangement import (
    _arrange,
    _require_published_embedded,
    SurfaceArrangement,
    SurfaceArrangementEvidence,
    SurfaceArrangementLimits,
)
from ._contracts import SurfaceMetadata
from ._model import SurfaceModel


class SurfaceBooleanOperation(StrEnum):
    """Boolean of closed solids; ``DIFFERENCE`` removes later operands from the first."""

    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


@final
class _RegionExpression(StrictModule, NonTrainableState):
    """Immutable original-operand CSG tree, independent of rounded output."""

    operation: SurfaceBooleanOperation | None = eqx.field(static=True)
    operand_index: int | None = eqx.field(static=True)
    children: tuple[_RegionExpression, ...] = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        operation: SurfaceBooleanOperation | None = None,
        operand_index: int | None = None,
        children: tuple[_RegionExpression, ...] = (),
    ) -> None:
        if not isinstance(children, tuple):
            raise TypeError("CSG children must be an immutable tuple.")
        if operand_index is not None and (
            isinstance(operand_index, bool) or not isinstance(operand_index, int)
        ):
            raise TypeError("An original-operand index must be an integer.")
        if operation is None:
            if operand_index is None or operand_index < 0 or children:
                raise ValueError(
                    "An original-operand leaf requires one nonnegative index."
                )
        elif operand_index is not None or len(children) < 2:
            raise ValueError("A CSG operation requires two or more child expressions.")
        if not all(isinstance(child, _RegionExpression) for child in children):
            raise TypeError("CSG children must be original-feature region expressions.")
        operation_ = (
            None
            if operation is None
            else parse(operation, SurfaceBooleanOperation, "operation")
        )
        self.operation = operation_
        self.operand_index = operand_index
        self.children = children
        self.expression_id = canonical_fingerprint(
            {
                "kind": "original-feature-csg-expression",
                "operation": None if operation_ is None else operation_.value,
                "operand_index": operand_index,
                "children": tuple(child.expression_id for child in children),
            }
        )


def _shift_expression(expression: _RegionExpression, offset: int, /) -> _RegionExpression:
    index = expression.operand_index
    if index is not None:
        return _RegionExpression(operand_index=index + offset)
    return _RegionExpression(
        operation=expression.operation,
        children=tuple(_shift_expression(child, offset) for child in expression.children),
    )


def _expression_region(
    expression: _RegionExpression, occupancy: np.ndarray, /
) -> np.ndarray:
    index = expression.operand_index
    if index is not None:
        if index >= occupancy.shape[1]:
            raise ValueError("CSG expression references a missing original operand.")
        return occupancy[:, index]
    operation = expression.operation
    if operation is None:
        raise ValueError("A CSG operation node is missing its operation.")
    return _region(
        operation,
        np.stack(
            tuple(_expression_region(child, occupancy) for child in expression.children),
            axis=1,
        ),
    )


SurfaceBooleanStatus: TypeAlias = Literal[
    "open_operand",
    "nonmanifold_operand",
    "operand_not_solid",
    "limit_exceeded",
]

# Operand validation only: winding numbers of a closed surface are integers off
# the surface, and a nested-shell check farther than this from an integer is
# refused rather than trusted.  Fragment membership is decided exactly.
_WINDING_MARGIN = 1.0e-2
_WINDING_BLOCK = 1 << 20
_BARYCENTRIC_PLAN = SmallLinearSolvePlan(2)


class _ResultVertexDim(Dim):
    """Vertices of a Boolean result."""


class _ResultTriangleDim(Dim):
    """Triangles of a Boolean result."""


class SurfaceBooleanError(ValueError):
    """Fail-closed Boolean refusal naming the offending operand."""

    def __init__(
        self,
        status: SurfaceBooleanStatus,
        message: str,
        /,
        *,
        operand: int | None = None,
    ) -> None:
        self.status: SurfaceBooleanStatus = parse(status, SurfaceBooleanStatus, "status")
        self.operand = operand
        super().__init__(f"{message} (status {self.status})")


@final
class SurfaceClosureEvidence(StrictModule, NonTrainableState):
    """Watertightness and manifoldness of one oriented triangle surface.

    ``boundary_edges`` counts unpaired directed edges (an edge used ``p`` times
    in one direction and ``q`` in the other contributes ``|p - q|``);
    ``nonmanifold_edges`` counts edges not used exactly once in each direction;
    ``nonmanifold_vertices`` counts vertices whose incident triangles form more
    than one edge-connected fan.  ``signed_volume`` is the divergence-theorem
    volume of the oriented triangles.
    """

    triangle_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    boundary_edges: int = eqx.field(static=True)
    nonmanifold_edges: int = eqx.field(static=True)
    nonmanifold_vertices: int = eqx.field(static=True)
    signed_volume: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        triangle_count: int,
        component_count: int,
        boundary_edges: int,
        nonmanifold_edges: int,
        nonmanifold_vertices: int,
        signed_volume: float,
    ) -> None:
        counts = (
            int(triangle_count),
            int(component_count),
            int(boundary_edges),
            int(nonmanifold_edges),
            int(nonmanifold_vertices),
        )
        if min(counts) < 0:
            raise ValueError("Surface closure counts must be nonnegative.")
        volume = float(signed_volume)
        if not np.isfinite(volume):
            raise ValueError("signed_volume must be finite.")
        (
            self.triangle_count,
            self.component_count,
            self.boundary_edges,
            self.nonmanifold_edges,
            self.nonmanifold_vertices,
        ) = counts
        self.signed_volume = volume
        self.evidence_id = canonical_fingerprint(
            {"kind": "surface-closure-evidence", "counts": counts, "volume": volume}
        )

    @property
    def closed(self) -> bool:
        """Every directed edge is matched by the reversed edge."""

        return self.boundary_edges == 0

    @property
    def edge_manifold(self) -> bool:
        """Every edge is used exactly once in each direction."""

        return self.nonmanifold_edges == 0

    @property
    def vertex_manifold(self) -> bool:
        """Every vertex star is a single fan."""

        return self.nonmanifold_vertices == 0


def _undirected_edges(
    triangles: np.ndarray, vertex_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge keys ``low * n + high``, owning triangles and directions of all sides."""

    directed = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    owners = np.tile(np.arange(triangles.shape[0], dtype=np.int64), 3)
    ordered = np.sort(directed, axis=1)
    keys = ordered[:, 0] * vertex_count + ordered[:, 1]
    return keys, owners, directed[:, 0] < directed[:, 1]


def _linked_components(
    keys: np.ndarray, owners: np.ndarray, eligible: np.ndarray, count: int, /
) -> np.ndarray:
    """Components of ``count`` nodes joined when they share an eligible key."""

    order = np.lexsort((owners, keys))
    order = order[eligible[order]]
    same = keys[order][1:] == keys[order][:-1]
    rows = owners[order][:-1][same]
    columns = owners[order][1:][same]
    graph = sp.coo_matrix(
        (np.ones((rows.size,), dtype=np.int8), (rows, columns)), shape=(count, count)
    )
    return connected_components(graph, directed=False)[1].astype(np.int64)


def _signed_volumes(vertices: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    corners = vertices[triangles]
    return np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0


def _closure(
    vertices: np.ndarray, triangles: np.ndarray, /
) -> tuple[SurfaceClosureEvidence, np.ndarray]:
    """Closure evidence and edge-connected triangle components of a surface."""

    if triangles.shape[0] == 0:
        empty = SurfaceClosureEvidence(
            triangle_count=0,
            component_count=0,
            boundary_edges=0,
            nonmanifold_edges=0,
            nonmanifold_vertices=0,
            signed_volume=0.0,
        )
        return empty, np.zeros((0,), dtype=np.int64)
    count = vertices.shape[0]
    keys, owners, forward = _undirected_edges(triangles, count)
    _, inverse = np.unique(keys, return_inverse=True)
    forward_uses = np.bincount(inverse, weights=forward.astype(np.float64))
    backward_uses = np.bincount(inverse, weights=(~forward).astype(np.float64))
    labels = _linked_components(keys, owners, np.ones_like(forward), triangles.shape[0])
    # Fans: the link of vertex c in triangle (c, a, b) is the edge (a, b); link
    # nodes (c, a) and (c, b) are joined, and each connected link is one fan.
    centers = triangles.reshape(-1)
    link_first = triangles[:, (1, 2, 0)].reshape(-1)
    link_second = triangles[:, (2, 0, 1)].reshape(-1)
    nodes, node_ids = np.unique(
        np.concatenate((centers * count + link_first, centers * count + link_second)),
        return_inverse=True,
    )
    node_ids = node_ids.reshape(2, -1)
    link = sp.coo_matrix(
        (np.ones((centers.size,), dtype=np.int8), (node_ids[0], node_ids[1])),
        shape=(nodes.size, nodes.size),
    )
    fans = connected_components(link, directed=False)[1]
    fan_keys = np.unique(np.stack((nodes // count, fans), axis=1), axis=0)
    evidence = SurfaceClosureEvidence(
        triangle_count=triangles.shape[0],
        component_count=int(np.max(labels, initial=-1)) + 1,
        boundary_edges=int(np.sum(np.abs(forward_uses - backward_uses))),
        nonmanifold_edges=int(np.sum((forward_uses != 1) | (backward_uses != 1))),
        nonmanifold_vertices=int(np.sum(np.bincount(fan_keys[:, 0]) > 1)),
        signed_volume=float(np.sum(_signed_volumes(vertices, triangles))),
    )
    return evidence, labels


def _winding_numbers(
    points: np.ndarray, triangles: np.ndarray, budget: list[int], /
) -> np.ndarray:
    """Generalized winding numbers by van Oosterom-Strackee solid angles.

    ``budget`` holds the remaining point-triangle evaluations and is consumed.
    """

    work = points.shape[0] * triangles.shape[0]
    if work > budget[0]:
        raise SurfaceBooleanError(
            "limit_exceeded",
            "Boolean classification exceeds maximum_winding_evaluations.",
        )
    budget[0] -= work
    values = np.zeros((points.shape[0],), dtype=np.float64)
    step = max(1, _WINDING_BLOCK // max(1, triangles.shape[0]))
    for start in range(0, points.shape[0], step):
        relative = triangles[None] - points[start : start + step, None, None, :]
        first, second, third = relative[:, :, 0], relative[:, :, 1], relative[:, :, 2]
        lengths = np.linalg.norm(relative, axis=-1)
        numerator = np.sum(first * np.cross(second, third), axis=-1)
        denominator = (
            lengths[..., 0] * lengths[..., 1] * lengths[..., 2]
            + np.sum(first * second, axis=-1) * lengths[..., 2]
            + np.sum(second * third, axis=-1) * lengths[..., 0]
            + np.sum(third * first, axis=-1) * lengths[..., 1]
        )
        values[start : start + step] = np.sum(
            np.arctan2(numerator, denominator), axis=1
        ) / (2.0 * np.pi)
    return values


def _operand_arrays(
    surface: SurfaceModel, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates, triangles and cell global ids of an operand's triangle blocks."""

    blocks = surface.mesh.blocks
    return (
        np.asarray(surface.mesh.coordinates, dtype=np.float64),
        np.concatenate([np.asarray(block.vertices, dtype=np.int64) for block in blocks]),
        np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in blocks]
        ),
    )


def _require_solid(
    vertices: np.ndarray, triangles: np.ndarray, index: int, budget: list[int], /
) -> None:
    """Refuse operands that do not bound a solid with winding numbers in {0, 1}.

    Off its own surface, a closed component with positive volume contributes
    winding ``1`` inside and ``0`` outside; a negative one ``0`` and ``-1``.
    The winding number of the other components at a point of one component is
    therefore required to be ``0`` for a positive and ``1`` for a negative one.
    """

    evidence, labels = _closure(vertices, triangles)
    if not evidence.closed:
        raise SurfaceBooleanError(
            "open_operand",
            f"Volumetric Boolean operands must be closed; operand {index} has "
            f"{evidence.boundary_edges} unpaired directed edges. Open sheets are "
            "split with arrange_triangle_surfaces.",
            operand=index,
        )
    if not evidence.edge_manifold:
        raise SurfaceBooleanError(
            "nonmanifold_operand",
            f"Boolean operand {index} has {evidence.nonmanifold_edges} edges not "
            "shared by exactly two consistently oriented triangles.",
            operand=index,
        )
    volumes = np.bincount(labels, weights=_signed_volumes(vertices, triangles))
    if np.any(volumes == 0.0) or np.sum(volumes) <= 0.0:
        raise SurfaceBooleanError(
            "operand_not_solid",
            f"Boolean operand {index} does not enclose a positive oriented volume.",
            operand=index,
        )
    if volumes.size == 1:
        return
    corners = vertices[triangles]
    areas = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    for component in range(volumes.size):
        members = labels == component
        representative = np.flatnonzero(members)[np.argmax(areas[members])]
        winding = _winding_numbers(
            np.mean(corners[representative], axis=0)[None],
            corners[~members],
            budget,
        )[0]
        expected = 0.0 if volumes[component] > 0.0 else 1.0
        if abs(winding - expected) > _WINDING_MARGIN:
            raise SurfaceBooleanError(
                "operand_not_solid",
                f"Boolean operand {index} has nested shells with inconsistent "
                "orientation (winding numbers outside {0, 1}).",
                operand=index,
            )


def _classify(
    arrangement: SurfaceArrangement, classification: ArrangementClassification, /
) -> tuple[np.ndarray, int]:
    """Exact inside flags ``(fragments, operands)`` and the component count.

    A fragment has a membership in every operand other than its own and those
    it is coplanar with; coplanar fragments follow the coverage orientation.
    """

    windings = classification.component_windings[classification.fragment_components]
    classified = windings != ARRANGEMENT_UNCLASSIFIED
    rows = np.arange(arrangement.triangles.shape[0])
    expected = arrangement.coincident_face < 0
    expected[rows, arrangement.source_surface] = False
    if np.any(classified != expected):
        raise RuntimeError(
            "Native fragment components disagree with the exact coplanar coverage."
        )
    invalid = classified & (windings != 0) & (windings != 1)
    if np.any(invalid):
        raise SurfaceBooleanError(
            "operand_not_solid",
            "A Boolean operand has exact winding numbers outside {0, 1}.",
            operand=int(np.flatnonzero(np.any(invalid, axis=0))[0]),
        )
    return windings == 1, classification.component_windings.shape[0]


def _region(operation: SurfaceBooleanOperation, occupancy: np.ndarray, /) -> np.ndarray:
    match operation:
        case SurfaceBooleanOperation.UNION:
            return np.any(occupancy, axis=1)
        case SurfaceBooleanOperation.INTERSECTION:
            return np.all(occupancy, axis=1)
        case SurfaceBooleanOperation.DIFFERENCE:
            return occupancy[:, 0] & ~np.any(occupancy[:, 1:], axis=1)
        case _:
            raise ValueError(f"Unknown surface Boolean operation {operation!r}.")


def _boundary_sources(
    expression: _RegionExpression,
    behind: np.ndarray,
    front: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep ancestry only through subexpressions whose region actually changes.

    For example, coincident A and B cancel in ``(A - B) union C``. Their
    geometrically covering faces must not steal C's properties just because
    their original operand indices precede C.
    """
    sides: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pending: list[tuple[_RegionExpression, bool]] = [(expression, False)]
    while pending:
        node, expanded = pending.pop()
        key = node.expression_id
        if key in sides:
            continue
        index = node.operand_index
        if index is not None:
            sides[key] = behind[:, index], front[:, index]
            continue
        operation = node.operation
        if operation is None:
            raise ValueError("A CSG operation node is missing its operation.")
        if not expanded:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node.children))
            continue
        child_sides = tuple(sides[child.expression_id] for child in node.children)
        sides[key] = (
            _region(
                operation, np.stack(tuple(value[0] for value in child_sides), axis=1)
            ),
            _region(
                operation, np.stack(tuple(value[1] for value in child_sides), axis=1)
            ),
        )
    eligible = np.zeros_like(behind)
    paths: list[tuple[_RegionExpression, np.ndarray]] = [
        (expression, np.ones((behind.shape[0],), dtype=np.bool_))
    ]
    while paths:
        node, active = paths.pop()
        node_behind, node_front = sides[node.expression_id]
        varying = active & (node_behind != node_front)
        index = node.operand_index
        if index is not None:
            eligible[:, index] |= varying
        else:
            paths.extend((child, varying) for child in node.children)
    region_behind, region_front = sides[expression.expression_id]
    return region_behind, region_front, eligible


def _emission(
    arrangement: SurfaceArrangement,
    inside: np.ndarray,
    expression: _RegionExpression,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    orientation = arrangement.coincident_orientation
    behind = np.where(orientation == 0, inside, orientation > 0)
    front = np.where(orientation == 0, inside, orientation < 0)
    rows = np.arange(arrangement.triangles.shape[0])
    behind[rows, arrangement.source_surface] = True
    front[rows, arrangement.source_surface] = False
    region_behind, region_front, eligible = _boundary_sources(expression, behind, front)
    duplicate = np.any(
        (orientation != 0)
        & eligible
        & (
            np.arange(orientation.shape[1])[None, :] < arrangement.source_surface[:, None]
        ),
        axis=1,
    )
    emitted = np.flatnonzero(
        (region_behind != region_front)
        & eligible[rows, arrangement.source_surface]
        & ~duplicate
    )
    return emitted, ~region_behind[emitted]


def _barycentric(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    """Barycentric weights ``(n, 3)`` of points projected onto their triangles."""

    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    offset = points - triangles[:, 0]
    gram = np.stack(
        (
            np.stack((np.sum(first * first, 1), np.sum(first * second, 1)), axis=-1),
            np.stack((np.sum(first * second, 1), np.sum(second * second, 1)), axis=-1),
        ),
        axis=-2,
    )
    right = np.stack((np.sum(first * offset, 1), np.sum(second * offset, 1)), axis=-1)
    solved = solve_small_linear(_BARYCENTRIC_PLAN, gram, right)
    if not np.all(np.asarray(solved.successful)):
        # Degenerate source faces are refused by the arrangement beforehand.
        raise RuntimeError("Source-face barycentric weights met a degenerate face.")
    weights = np.asarray(solved.value, dtype=np.float64)
    return np.concatenate((1.0 - np.sum(weights, axis=1, keepdims=True), weights), 1)


@final
class SurfaceBooleanResult(StrictModule, NonTrainableState):
    """Oriented Boolean output with exact source-face ancestry.

    Triangle ``t`` lies in the face with cell global id
    ``source_cell_global_ids[t]`` of operand ``source_operand[t]``, whose
    explicit original model identity is ``operand_ids[source_operand[t]]``.
    Its orientation is reversed when ``reversed[t]``. ``source_vertices[t]`` are that face's
    operand vertex indices and ``source_barycentric[t, c]`` the weights of
    corner ``c`` in it.  ``surface`` carries the same triangles and vertices
    (cell and vertex global ids equal their row) for a nonempty edge-manifold
    result and is ``None`` otherwise; ``closure`` states why.
    Original models remain in the CSG expression even when a subexpression
    cancels, but ancestry passes only through changing region boundaries.
    ``classified_components`` fragment components were classified exactly;
    ``classification_pair_scans`` representative/triangle pairs and
    ``classification_matrix_entries`` winding entries were the admitted
    classification work (operand validation work is not included).
    """

    __strict_contract__ = True

    vertices: HostFloat64[_ResultVertexDim, Literal[3]]
    vertex_bounds: HostFloat64[_ResultVertexDim]
    triangles: HostInt64[_ResultTriangleDim, Literal[3]]
    source_operand: HostInt64[_ResultTriangleDim]
    source_cell_global_ids: HostInt64[_ResultTriangleDim]
    source_vertices: HostInt64[_ResultTriangleDim, Literal[3]]
    source_barycentric: HostFloat64[_ResultTriangleDim, Literal[3], Literal[3]]
    reversed: HostBool[_ResultTriangleDim]
    surface: SurfaceModel | None
    arrangement: SurfaceArrangementEvidence
    closure: SurfaceClosureEvidence
    source_models: tuple[SurfaceModel, ...]
    region_expression: _RegionExpression = eqx.field(static=True)
    operation: SurfaceBooleanOperation = eqx.field(static=True)
    operand_ids: tuple[str, ...] = eqx.field(static=True)
    classified_components: int = eqx.field(static=True)
    classification_pair_scans: int = eqx.field(static=True)
    classification_matrix_entries: int = eqx.field(static=True)
    publication_candidate_pairs: int = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        operation: SurfaceBooleanOperation,
        vertices: ArrayLike,
        vertex_bounds: ArrayLike,
        source_models: tuple[SurfaceModel, ...],
        region_expression: _RegionExpression,
        triangles: ArrayLike,
        source_operand: ArrayLike,
        source_cell_global_ids: ArrayLike,
        source_vertices: ArrayLike,
        source_barycentric: ArrayLike,
        reversed: ArrayLike,
        surface: SurfaceModel | None,
        arrangement: SurfaceArrangementEvidence,
        closure: SurfaceClosureEvidence,
        classified_components: int,
        classification_pair_scans: int,
        classification_matrix_entries: int,
        publication_candidate_pairs: int,
    ) -> None:
        scope = Scope()
        if (
            not isinstance(source_models, tuple)
            or len(source_models) < 2
            or not all(isinstance(model, SurfaceModel) for model in source_models)
        ):
            raise TypeError(
                "source_models must retain an immutable tuple of original SurfaceModel values."
            )
        if not isinstance(region_expression, _RegionExpression):
            raise TypeError(
                "region_expression must preserve the original CSG expression."
            )
        operand_ids = tuple(model.model_id for model in source_models)
        _expression_region(
            region_expression, np.zeros((0, len(source_models)), dtype=np.bool_)
        )
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[_ResultVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        bounds = parse(
            np.asarray(vertex_bounds, dtype=np.float64),
            HostFloat64[_ResultVertexDim],
            "vertex_bounds",
            scope=scope,
        )
        faces = parse(
            np.asarray(triangles, dtype=np.int64),
            HostInt64[_ResultTriangleDim, Literal[3]],
            "triangles",
            scope=scope,
        )
        operands = parse(
            np.asarray(source_operand, dtype=np.int64),
            HostInt64[_ResultTriangleDim],
            "source_operand",
            scope=scope,
        )
        cells = parse(
            np.asarray(source_cell_global_ids, dtype=np.int64),
            HostInt64[_ResultTriangleDim],
            "source_cell_global_ids",
            scope=scope,
        )
        corners = parse(
            np.asarray(source_vertices, dtype=np.int64),
            HostInt64[_ResultTriangleDim, Literal[3]],
            "source_vertices",
            scope=scope,
        )
        weights = parse(
            np.asarray(source_barycentric, dtype=np.float64),
            HostFloat64[_ResultTriangleDim, Literal[3], Literal[3]],
            "source_barycentric",
            scope=scope,
        )
        flipped = parse(
            np.asarray(reversed, dtype=np.bool_),
            HostBool[_ResultTriangleDim],
            "reversed",
            scope=scope,
        )
        operation_ = parse(operation, SurfaceBooleanOperation, "operation")
        if surface is not None and not isinstance(surface, SurfaceModel):
            raise TypeError("surface must be a SurfaceModel or None.")
        if not isinstance(arrangement, SurfaceArrangementEvidence):
            raise TypeError("arrangement must be SurfaceArrangementEvidence.")
        if not isinstance(closure, SurfaceClosureEvidence):
            raise TypeError("closure must be SurfaceClosureEvidence.")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            raise ValueError("Boolean vertices and barycentric weights must be finite.")
        if np.any(faces < 0) or np.any(faces >= points.shape[0]):
            raise ValueError("Boolean triangles index missing vertices.")
        if (
            len(operand_ids) < 2
            or np.any(operands < 0)
            or np.any(operands >= len(operand_ids))
            or np.any(corners < 0)
        ):
            raise ValueError(
                "Boolean ancestry names an explicit original operand and its vertices."
            )
        if not np.all(np.isfinite(bounds)) or np.any(bounds < 0.0):
            raise ValueError("Boolean publication bounds must be finite and nonnegative.")
        if (surface is None) == (faces.shape[0] > 0 and closure.edge_manifold):
            raise ValueError(
                "A Boolean surface is published exactly for nonempty edge-manifold "
                "results."
            )
        self.vertices = points
        self.vertex_bounds = bounds
        self.operand_ids = operand_ids
        self.source_models = source_models
        self.region_expression = region_expression
        self.triangles = faces
        self.source_operand = operands
        self.source_cell_global_ids = cells
        self.source_vertices = corners
        self.source_barycentric = weights
        self.reversed = flipped
        self.surface = surface
        self.arrangement = arrangement
        self.closure = closure
        self.operation = operation_
        self.classified_components = int(classified_components)
        self.classification_pair_scans = int(classification_pair_scans)
        self.classification_matrix_entries = int(classification_matrix_entries)
        self.publication_candidate_pairs = publication_candidate_pairs
        self.result_id = canonical_fingerprint(
            {
                "kind": "surface-boolean-result",
                "operation": operation_.value,
                "arrays": array_tree_fingerprint(
                    (points, bounds, faces, operands, cells, corners, weights, flipped)
                ),
                "operand_ids": operand_ids,
                "region_expression": region_expression.expression_id,
                "arrangement_id": arrangement.evidence_id,
                "closure_id": closure.evidence_id,
                "surface_id": None if surface is None else surface.model_id,
                "classified_components": self.classified_components,
                "classification_pair_scans": self.classification_pair_scans,
                "classification_matrix_entries": self.classification_matrix_entries,
                "publication_candidate_pairs": publication_candidate_pairs,
            }
        )

    @property
    def empty(self) -> bool:
        """Whether the Boolean result has no triangles."""

        return self.triangles.shape[0] == 0

    def source_corner_values(
        self,
        first_values: ArrayLike,
        second_values: ArrayLike,
        /,
        *,
        operand_values: tuple[ArrayLike, ...] = (),
    ) -> np.ndarray:
        """Transfer per-vertex operand values to result triangle corners.

        Each corner takes the barycentric interpolation of its own source face,
        so values stay discontinuous across cut curves; affine fields are
        reproduced exactly.  Returns ``(triangles, 3, *component_shape)``.
        """

        values = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (first_values, second_values, *operand_values)
        )
        if len(values) != len(self.operand_ids) or any(
            value.ndim == 0 or value.shape[1:] != values[0].shape[1:] for value in values
        ):
            raise ValueError(
                "Operand values need one row per vertex and one shape for every operand."
            )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Operand values must be finite.")
        component = values[0].shape[1:]
        result = np.zeros((self.triangles.shape[0], 3, *component), dtype=np.float64)
        for operand, value in enumerate(values):
            rows = self.source_operand == operand
            if np.any(self.source_vertices[rows] >= value.shape[0]):
                raise ValueError(f"Operand {operand} values miss source vertices.")
            source = value[self.source_vertices[rows]]
            weights = self.source_barycentric[rows]
            result[rows] = np.sum(
                weights.reshape(*weights.shape, *(1,) * len(component)) * source[:, None],
                axis=2,
            )
        return result


def _output_surface(
    operands: tuple[SurfaceModel, ...],
    operation: SurfaceBooleanOperation,
    vertices: np.ndarray,
    triangles: np.ndarray,
    source_operand: np.ndarray,
    source_rows: np.ndarray,
    /,
) -> SurfaceModel:
    tags = tuple(
        operands[operand].metadata.cell_tags[row]
        if operands[operand].metadata.cell_tags
        else operands[operand].metadata.source_id
        for operand, row in zip(
            source_operand.tolist(), source_rows.tolist(), strict=True
        )
    )
    source_id = canonical_fingerprint(
        {
            "kind": "native-surface-boolean",
            "operation": operation.value,
            "operands": tuple(operand.model_id for operand in operands),
            "arrays": array_tree_fingerprint((vertices, triangles)),
        }
    )
    return SurfaceModel.from_triangles(
        vertices,
        triangles,
        SurfaceMetadata(
            source_id=source_id,
            source_revision="0",
            coordinate_contract=operands[0].metadata.coordinate_contract,
            provenance=(
                "native-surface-boolean",
                operation.value,
                *(operand.model_id for operand in operands),
            ),
            cell_tags=tags,
        ),
        vertex_global_ids=np.arange(vertices.shape[0], dtype=np.int64),
        cell_global_ids=np.arange(triangles.shape[0], dtype=np.int64),
    )


def surface_boolean(
    first: SurfaceModel | SurfaceBooleanResult,
    second: SurfaceModel | SurfaceBooleanResult,
    operation: SurfaceBooleanOperation,
    /,
    *,
    operands: tuple[SurfaceModel | SurfaceBooleanResult, ...] = (),
    limits: SurfaceArrangementLimits | None = None,
) -> SurfaceBooleanResult:
    """Union, intersection or first-minus-all-later difference of closed solids.

    Operands must be closed, edge-manifold, embedded and outward oriented
    (winding numbers in {0, 1}); open or inconsistent operands are refused with
    :class:`SurfaceBooleanError`.  Arrangement refusals raise
    :class:`SurfaceArrangementError`.  Input selections and interfaces are not
    inherited; per-face cell tags follow source-face ancestry.
    Chained results retain their original operands and region expression:
    the next operation recomputes one exact arrangement of those originals,
    rather than treating rounded published coordinates as exact new sources.
    Consequently publication error is bounded once, not silently accumulated,
    and source-face ancestry always reaches the original model.
    """

    originals: list[SurfaceModel] = []
    expressions: list[_RegionExpression] = []
    for operand in (first, second, *operands):
        if isinstance(operand, SurfaceModel):
            expressions.append(_RegionExpression(operand_index=len(originals)))
            originals.append(operand)
        elif isinstance(operand, SurfaceBooleanResult):
            expressions.append(
                _shift_expression(operand.region_expression, len(originals))
            )
            originals.extend(operand.source_models)
        else:
            raise TypeError(
                "Boolean operands must be SurfaceModel or SurfaceBooleanResult values."
            )
    operands_ = tuple(originals)
    operation_ = parse(operation, SurfaceBooleanOperation, "operation")
    expression = _RegionExpression(operation=operation_, children=tuple(expressions))
    limits_ = SurfaceArrangementLimits() if limits is None else limits
    if not isinstance(limits_, SurfaceArrangementLimits):
        raise TypeError("limits must be SurfaceArrangementLimits or None.")
    if any(
        surface.metadata.coordinate_contract.spatial_id
        != operands_[0].metadata.coordinate_contract.spatial_id
        for surface in operands_
    ):
        raise ValueError(
            "Boolean operands require identical spatial coordinate contracts."
        )
    arrays = tuple(_operand_arrays(surface) for surface in operands_)
    budget = [limits_.maximum_winding_evaluations]
    for index, (vertices, triangles, _) in enumerate(arrays):
        _require_solid(vertices, triangles, index, budget)
    arrangement, classification = _arrange(
        tuple((points, faces) for points, faces, _ in arrays), limits_, budget[0]
    )
    if classification is None:
        raise RuntimeError("The native arrangement omitted the requested classification.")
    if classification.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        raise SurfaceBooleanError(
            "limit_exceeded",
            "Boolean classification needs "
            f"{classification.matrix_entries} matrix entries and at least "
            f"{classification.pair_scans} pair scans beyond maximum_winding_evaluations.",
        )
    inside, components = _classify(arrangement, classification)
    rows, flipped = _emission(arrangement, inside, expression)
    emitted = arrangement.triangles[rows]
    emitted[flipped] = emitted[flipped][:, (0, 2, 1)]
    used, first_use = np.unique(emitted.reshape(-1), return_index=True)
    ordered = used[np.argsort(first_use, kind="stable")]
    renumber = np.zeros((arrangement.vertices.shape[0],), dtype=np.int64)
    renumber[ordered] = np.arange(ordered.size)
    vertices = arrangement.vertices[ordered].reshape(-1, 3)
    triangles = renumber[emitted].reshape(-1, 3)
    source_operand = arrangement.source_surface[rows]
    source_rows = arrangement.source_face[rows]
    source_vertices = np.zeros((rows.size, 3), dtype=np.int64)
    source_cells = np.zeros((rows.size,), dtype=np.int64)
    for operand, (points, faces, cell_ids) in enumerate(arrays):
        selected = source_operand == operand
        source_vertices[selected] = faces[source_rows[selected]]
        source_cells[selected] = cell_ids[source_rows[selected]]
    source_corners = np.zeros((rows.size, 3, 3), dtype=np.float64)
    for operand, (points, _, _) in enumerate(arrays):
        selected = source_operand == operand
        source_corners[selected] = points[source_vertices[selected]]
    barycentric = _barycentric(
        vertices[triangles].reshape(-1, 3), np.repeat(source_corners, 3, axis=0)
    ).reshape(-1, 3, 3)
    remaining_pairs = limits_.maximum_candidate_pairs - (
        arrangement.evidence.self_candidate_pairs
        + arrangement.evidence.candidate_pairs
        + arrangement.evidence.publication_candidate_pairs
    )
    # Offending faces of a refused publication are result triangle rows.
    publication_pairs = (
        _require_published_embedded(
            vertices,
            triangles,
            np.arange(rows.size, dtype=np.int64),
            remaining_pairs,
            surface=None,
        )
        if rows.size
        else 0
    )
    closure, _ = _closure(vertices, triangles)
    surface = (
        _output_surface(
            operands_, operation_, vertices, triangles, source_operand, source_rows
        )
        if rows.size and closure.edge_manifold
        else None
    )
    return SurfaceBooleanResult(
        operation=operation_,
        vertices=vertices,
        vertex_bounds=arrangement.vertex_bounds[ordered],
        source_models=operands_,
        region_expression=expression,
        triangles=triangles,
        source_operand=source_operand,
        source_cell_global_ids=source_cells,
        source_vertices=source_vertices,
        source_barycentric=barycentric,
        reversed=flipped,
        surface=surface,
        arrangement=arrangement.evidence,
        closure=closure,
        classified_components=components,
        classification_pair_scans=classification.pair_scans,
        classification_matrix_entries=classification.matrix_entries,
        publication_candidate_pairs=publication_pairs,
    )


__all__ = [
    "surface_boolean",
    "SurfaceBooleanError",
    "SurfaceBooleanOperation",
    "SurfaceBooleanResult",
    "SurfaceBooleanStatus",
    "SurfaceClosureEvidence",
]
