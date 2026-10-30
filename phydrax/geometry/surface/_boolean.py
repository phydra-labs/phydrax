#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-independent Booleans of closed oriented triangle surfaces.

The operation is performed on one joint native triangle arrangement. It retains
ancestry to the input ``SurfaceModel`` faces and does not claim to operate on
underlying CAD curves, faces, or feature history.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias

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
from ...typing import checked, parse
from ._arrangement import (
    arrange_classified_surfaces,
    require_published_embedded,
    SurfaceArrangement,
    SurfaceArrangementEvidence,
)
from ._contracts import SurfaceMetadata
from ._model import _surface_connectivity, SurfaceModel


class SurfaceBooleanOperation(StrEnum):
    """Volumetric Boolean operation over oriented triangle surfaces."""

    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


SurfaceBooleanStatus: TypeAlias = Literal[
    "open_operand",
    "nonmanifold_operand",
    "operand_not_solid",
    "classification_unresolved",
]

_WINDING_MARGIN = 1.0e-2
_WINDING_PAIR_BLOCK = 1 << 18


class SurfaceBooleanError(ValueError):
    """Fail-closed Boolean refusal naming the offending operand when known."""

    def __init__(
        self,
        status: SurfaceBooleanStatus,
        message: str,
        /,
        *,
        operand: int | None = None,
    ) -> None:
        status = parse(status, SurfaceBooleanStatus, "status")
        self.status = status
        self.operand = operand
        super().__init__(f"{message} (status {status})")


class SurfaceClosureEvidence(StrictModule, NonTrainableState):
    """Directed-edge closure, edge/vertex manifoldness, and oriented volume."""

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
        counts = tuple(
            int(value)
            for value in (
                triangle_count,
                component_count,
                boundary_edges,
                nonmanifold_edges,
                nonmanifold_vertices,
            )
        )
        volume = float(signed_volume)
        if min(counts) < 0 or not np.isfinite(volume):
            raise ValueError("Closure counts must be nonnegative and volume finite.")
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
        return self.boundary_edges == 0

    @property
    def edge_manifold(self) -> bool:
        return self.nonmanifold_edges == 0

    @property
    def vertex_manifold(self) -> bool:
        return self.nonmanifold_vertices == 0


class _RegionExpression(StrictModule, NonTrainableState):
    """Original source-operand CSG tree, independent of rounded output."""

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
        if operand_index is not None and (
            isinstance(operand_index, bool) or not isinstance(operand_index, int)
        ):
            raise TypeError("operand_index must be an integer or None.")
        if operation is None:
            if operand_index is None or operand_index < 0 or children:
                raise ValueError("A CSG leaf requires one nonnegative operand index.")
        elif operand_index is not None or len(children) < 2:
            raise ValueError("A CSG operation requires at least two child expressions.")
        if not isinstance(children, tuple) or not all(
            isinstance(child, _RegionExpression) for child in children
        ):
            raise TypeError("CSG children must be an immutable tuple of expressions.")
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
                "kind": "source-feature-csg-expression",
                "operation": None if operation_ is None else operation_.value,
                "operand_index": operand_index,
                "children": tuple(child.expression_id for child in children),
            }
        )


def _region(operation: SurfaceBooleanOperation, occupancy: np.ndarray, /) -> np.ndarray:
    match operation:
        case SurfaceBooleanOperation.UNION:
            return np.any(occupancy, axis=1)
        case SurfaceBooleanOperation.INTERSECTION:
            return np.all(occupancy, axis=1)
        case SurfaceBooleanOperation.DIFFERENCE:
            return occupancy[:, 0] & ~np.any(occupancy[:, 1:], axis=1)
        case _:
            raise ValueError(f"Unsupported surface Boolean operation {operation!r}.")


def _expression_region(
    expression: _RegionExpression, occupancy: np.ndarray, /
) -> np.ndarray:
    index = expression.operand_index
    if index is not None:
        if index >= occupancy.shape[1]:
            raise ValueError("CSG expression references an undeclared source operand.")
        return occupancy[:, index]
    operation = expression.operation
    if operation is None:
        raise ValueError("CSG operation node is missing its operation.")
    return _region(
        operation,
        np.stack(
            tuple(_expression_region(child, occupancy) for child in expression.children),
            axis=1,
        ),
    )


def _shift_expression(expression: _RegionExpression, offset: int, /) -> _RegionExpression:
    index = expression.operand_index
    if index is not None:
        return _RegionExpression(operand_index=index + offset)
    return _RegionExpression(
        operation=expression.operation,
        children=tuple(_shift_expression(child, offset) for child in expression.children),
    )


def _linked_components(keys: np.ndarray, owners: np.ndarray, count: int, /) -> np.ndarray:
    order = np.lexsort((owners, keys))
    same = keys[order][1:] == keys[order][:-1]
    rows = owners[order][:-1][same]
    columns = owners[order][1:][same]
    graph = sp.coo_matrix(
        (np.ones((rows.size,), dtype=np.int8), (rows, columns)),
        shape=(count, count),
    )
    return connected_components(graph, directed=False, return_labels=True)[1].astype(
        np.int64, copy=False
    )


def _component_volumes(
    vertices: np.ndarray, triangles: np.ndarray, labels: np.ndarray, /
) -> np.ndarray:
    component_count = int(np.max(labels, initial=-1)) + 1
    volumes = np.zeros((component_count,), dtype=np.float64)
    for component in range(component_count):
        rows = np.flatnonzero(labels == component)
        corners = vertices[triangles[rows]]
        reference = np.mean(vertices[np.unique(triangles[rows])], axis=0)
        relative = corners - reference
        face_volumes = (
            np.einsum(
                "ij,ij->i", relative[:, 0], np.cross(relative[:, 1], relative[:, 2])
            )
            / 6.0
        )
        volumes[component] = np.sum(face_volumes)
    return volumes


def _closure(
    vertices: np.ndarray, triangles: np.ndarray, /
) -> tuple[SurfaceClosureEvidence, np.ndarray, np.ndarray]:
    if triangles.shape[0] == 0:
        empty = SurfaceClosureEvidence(
            triangle_count=0,
            component_count=0,
            boundary_edges=0,
            nonmanifold_edges=0,
            nonmanifold_vertices=0,
            signed_volume=0.0,
        )
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    vertex_count = vertices.shape[0]
    directed = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
        axis=0,
    )
    owners = np.tile(np.arange(triangles.shape[0], dtype=np.int64), 3)
    ordered_edges = np.sort(directed, axis=1)
    keys = ordered_edges[:, 0] * np.int64(vertex_count) + ordered_edges[:, 1]
    _, inverse = np.unique(keys, return_inverse=True)
    forward = directed[:, 0] < directed[:, 1]
    forward_uses = np.bincount(inverse, weights=forward.astype(np.float64))
    backward_uses = np.bincount(inverse, weights=(~forward).astype(np.float64))
    labels = _linked_components(keys, owners, triangles.shape[0])
    volumes = _component_volumes(vertices, triangles, labels)
    # The link of vertex c in (c,a,b) contains edge (a,b). Its connected
    # components are the distinct incident triangle fans at c.
    centers = triangles.reshape(-1)
    link_first = triangles[:, (1, 2, 0)].reshape(-1)
    link_second = triangles[:, (2, 0, 1)].reshape(-1)
    nodes, node_ids = np.unique(
        np.concatenate(
            (centers * vertex_count + link_first, centers * vertex_count + link_second)
        ),
        return_inverse=True,
    )
    node_ids = node_ids.reshape(2, -1)
    link = sp.coo_matrix(
        (np.ones((centers.size,), dtype=np.int8), (node_ids[0], node_ids[1])),
        shape=(nodes.size, nodes.size),
    )
    fans = connected_components(link, directed=False, return_labels=True)[1]
    fan_keys = np.unique(np.stack((nodes // vertex_count, fans), axis=1), axis=0)
    fan_counts = np.bincount(fan_keys[:, 0], minlength=vertex_count)
    evidence = SurfaceClosureEvidence(
        triangle_count=triangles.shape[0],
        component_count=int(np.max(labels, initial=-1)) + 1,
        boundary_edges=int(np.sum(np.abs(forward_uses - backward_uses))),
        nonmanifold_edges=int(np.sum((forward_uses != 1) | (backward_uses != 1))),
        nonmanifold_vertices=int(np.sum(fan_counts > 1)),
        signed_volume=float(np.sum(volumes)),
    )
    return evidence, labels, volumes


def _winding_numbers(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    """Generalized winding numbers, evaluated in complete bounded blocks."""
    result = np.zeros((points.shape[0],), dtype=np.float64)
    if points.shape[0] == 0 or triangles.shape[0] == 0:
        return result
    point_block = max(1, min(points.shape[0], _WINDING_PAIR_BLOCK))
    triangle_block = max(1, _WINDING_PAIR_BLOCK // point_block)
    for point_start in range(0, points.shape[0], point_block):
        point_stop = min(point_start + point_block, points.shape[0])
        accum = np.zeros((point_stop - point_start,), dtype=np.float64)
        sample = points[point_start:point_stop]
        for triangle_start in range(0, triangles.shape[0], triangle_block):
            corners = triangles[triangle_start : triangle_start + triangle_block]
            relative = corners[None, :, :, :] - sample[:, None, None, :]
            lengths = np.linalg.norm(relative, axis=-1)
            unit = relative / np.where(lengths > 0.0, lengths, 1.0)[..., None]
            first, second, third = unit[:, :, 0], unit[:, :, 1], unit[:, :, 2]
            numerator = np.sum(first * np.cross(second, third), axis=-1)
            denominator = (
                1.0
                + np.sum(first * second, axis=-1)
                + np.sum(second * third, axis=-1)
                + np.sum(third * first, axis=-1)
            )
            angles = np.arctan2(numerator, denominator)
            angles[np.any(lengths == 0.0, axis=-1)] = np.nan
            accum += np.sum(angles, axis=1)
        result[point_start:point_stop] = accum / (2.0 * np.pi)
    return result


def _operand_arrays(
    surface: SurfaceModel, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    connectivity = _surface_connectivity(surface.mesh)
    return (
        np.asarray(surface.mesh.coordinates, dtype=np.float64),
        np.asarray(connectivity.cell_vertices, dtype=np.int64)[:, :3],
        np.asarray(surface.mesh.entity_set(2).entity_ids, dtype=np.int64),
    )


def _require_solid(vertices: np.ndarray, triangles: np.ndarray, index: int, /) -> None:
    evidence, labels, volumes = _closure(vertices, triangles)
    if not evidence.closed:
        raise SurfaceBooleanError(
            "open_operand",
            f"operand {index} has unpaired directed boundary edges ({evidence.boundary_edges})",
            operand=index,
        )
    if not evidence.edge_manifold or not evidence.vertex_manifold:
        raise SurfaceBooleanError(
            "nonmanifold_operand",
            f"operand {index} is not an edge- and vertex-manifold triangle surface",
            operand=index,
        )
    total_volume = float(np.sum(volumes))
    if (
        not np.all(np.isfinite(volumes))
        or np.any(volumes == 0.0)
        or not np.isfinite(total_volume)
        or total_volume <= 0.0
    ):
        raise SurfaceBooleanError(
            "operand_not_solid",
            f"operand {index} must enclose positive nonzero oriented volume",
            operand=index,
        )
    if volumes.size <= 1:
        return
    corners = vertices[triangles]
    twice_area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    for component, volume in enumerate(volumes):
        members = labels == component
        member_rows = np.flatnonzero(members)
        representative = member_rows[int(np.argmax(twice_area[member_rows]))]
        point = np.mean(corners[representative], axis=0, keepdims=True)
        others = corners[~members]
        winding = float(_winding_numbers(point, others)[0])
        expected = 0.0 if volume > 0.0 else 1.0
        if not np.isfinite(winding) or abs(winding - expected) > _WINDING_MARGIN:
            raise SurfaceBooleanError(
                "operand_not_solid",
                f"operand {index} has an invalid nested-shell winding "
                f"({winding!r}, expected {expected:g})",
                operand=index,
            )


def _classify(
    arrangement: SurfaceArrangement, classification: ArrangementClassification, /
) -> tuple[np.ndarray, int]:
    if classification.status is MeshcoreStatus.UNRESOLVED:
        raise SurfaceBooleanError(
            "classification_unresolved",
            "native meshcore could not resolve every fragment membership",
        )
    if classification.status is not MeshcoreStatus.OK:
        raise RuntimeError("Native arrangement returned incomplete component membership.")
    windings = classification.component_windings[classification.fragment_components]
    classified = windings != ARRANGEMENT_UNCLASSIFIED
    rows = np.arange(arrangement.triangles.shape[0])
    expected = arrangement.coincident_face < 0
    expected[rows, arrangement.source_surface] = False
    if np.any(classified != expected):
        raise RuntimeError(
            "Native component classification disagrees with coplanar coverage."
        )
    invalid = classified & (windings != 0) & (windings != 1)
    if np.any(invalid):
        first_operand = int(np.flatnonzero(np.any(invalid, axis=0))[0])
        raise SurfaceBooleanError(
            "operand_not_solid",
            "classified source winding counts are not in {0, 1}",
            operand=first_operand,
        )
    return windings == 1, int(classification.component_windings.shape[0])


def _boundary_sources(
    expression: _RegionExpression,
    behind: np.ndarray,
    front: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate ancestry only through subexpressions that change region."""
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
            raise ValueError("CSG operation node is missing its operation.")
        if not expanded:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node.children))
            continue
        child_sides = tuple(sides[child.expression_id] for child in node.children)
        sides[key] = (
            _region(operation, np.stack(tuple(item[0] for item in child_sides), axis=1)),
            _region(operation, np.stack(tuple(item[1] for item in child_sides), axis=1)),
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
    return (*sides[expression.expression_id], eligible)


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
    """Barycentric weights for points in their nondegenerate source triangles."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    offset = points - triangles[:, 0]
    scales = np.maximum(np.max(np.abs(first), axis=1), np.max(np.abs(second), axis=1))
    if np.any(scales == 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError(
            "Boolean ancestry contains a numerically degenerate source face."
        )
    first = first / scales[:, None]
    second = second / scales[:, None]
    offset = offset / scales[:, None]
    d00 = np.sum(first * first, axis=1)
    d01 = np.sum(first * second, axis=1)
    d11 = np.sum(second * second, axis=1)
    d20 = np.sum(offset * first, axis=1)
    d21 = np.sum(offset * second, axis=1)
    denominator = d00 * d11 - d01 * d01
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise ValueError("Boolean ancestry source barycentrics are not representable.")
    second_weight = (d11 * d20 - d01 * d21) / denominator
    third_weight = (d00 * d21 - d01 * d20) / denominator
    weights = np.stack(
        (1.0 - second_weight - third_weight, second_weight, third_weight), axis=1
    )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Boolean ancestry produced nonfinite source barycentrics.")
    return weights


class SurfaceBooleanResult(StrictModule, NonTrainableState):
    """Boolean triangle output with source model, face, and corner ancestry."""

    vertices: np.ndarray
    vertex_bounds: np.ndarray
    triangles: np.ndarray
    source_operand: np.ndarray
    source_cell_global_ids: np.ndarray
    source_vertices: np.ndarray
    source_barycentric: np.ndarray
    reversed: np.ndarray
    surface: SurfaceModel | None
    arrangement: SurfaceArrangementEvidence
    closure: SurfaceClosureEvidence
    source_models: tuple[SurfaceModel, ...]
    region_expression: _RegionExpression = eqx.field(static=True)
    operation: SurfaceBooleanOperation = eqx.field(static=True)
    operand_ids: tuple[str, ...] = eqx.field(static=True)
    classified_components: int = eqx.field(static=True)
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
    ) -> None:
        if (
            not isinstance(source_models, tuple)
            or len(source_models) < 2
            or not all(isinstance(model, SurfaceModel) for model in source_models)
        ):
            raise TypeError(
                "source_models must be a tuple of at least two SurfaceModel values."
            )
        if not isinstance(region_expression, _RegionExpression):
            raise TypeError("region_expression must be a source CSG expression.")
        if not isinstance(arrangement, SurfaceArrangementEvidence) or not isinstance(
            closure, SurfaceClosureEvidence
        ):
            raise TypeError("arrangement and closure evidence have incompatible types.")
        points = np.asarray(vertices, dtype=np.float64)
        bounds = np.asarray(vertex_bounds, dtype=np.float64)
        faces = np.asarray(triangles, dtype=np.int64)
        operands = np.asarray(source_operand, dtype=np.int64)
        cells = np.asarray(source_cell_global_ids, dtype=np.int64)
        corners = np.asarray(source_vertices, dtype=np.int64)
        weights = np.asarray(source_barycentric, dtype=np.float64)
        flipped = np.asarray(reversed, dtype=np.bool_)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or bounds.shape != (points.shape[0],)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or operands.shape != (faces.shape[0],)
            or cells.shape != operands.shape
            or corners.shape != faces.shape
            or weights.shape != (faces.shape[0], 3, 3)
            or flipped.shape != operands.shape
        ):
            raise ValueError("SurfaceBooleanResult arrays have inconsistent shapes.")
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.0)
            or not np.all(np.isfinite(weights))
            or np.any(faces < 0)
            or np.any(faces >= points.shape[0])
            or np.any(operands < 0)
            or np.any(operands >= len(source_models))
            or np.any(cells < 0)
            or np.any(corners < 0)
        ):
            raise ValueError("SurfaceBooleanResult contains invalid output or ancestry.")
        if (surface is None) == (faces.shape[0] > 0 and closure.edge_manifold):
            raise ValueError(
                "surface must be present exactly for nonempty edge-manifold output."
            )
        operation_ = parse(operation, SurfaceBooleanOperation, "operation")
        operand_ids = tuple(model.model_id for model in source_models)
        _expression_region(
            region_expression, np.zeros((0, len(source_models)), dtype=np.bool_)
        )
        self.vertices = points
        self.vertex_bounds = bounds
        self.triangles = faces
        self.source_operand = operands
        self.source_cell_global_ids = cells
        self.source_vertices = corners
        self.source_barycentric = weights
        self.reversed = flipped
        self.surface = surface
        self.arrangement = arrangement
        self.closure = closure
        self.source_models = source_models
        self.region_expression = region_expression
        self.operation = operation_
        self.operand_ids = operand_ids
        self.classified_components = int(classified_components)
        self.result_id = canonical_fingerprint(
            {
                "kind": "surface-boolean-result",
                "operation": operation_.value,
                "arrays": array_tree_fingerprint(
                    (points, bounds, faces, operands, cells, corners, weights, flipped)
                ),
                "operand_ids": operand_ids,
                "region_expression_id": region_expression.expression_id,
                "arrangement_evidence_id": arrangement.evidence_id,
                "closure_evidence_id": closure.evidence_id,
                "surface_id": None if surface is None else surface.model_id,
                "classified_components": self.classified_components,
            }
        )

    @property
    def empty(self) -> bool:
        return self.triangles.shape[0] == 0

    def source_corner_values(
        self,
        first_values: ArrayLike,
        second_values: ArrayLike,
        /,
        *,
        operand_values: tuple[ArrayLike, ...] = (),
    ) -> np.ndarray:
        """Interpolate each result corner on its declared source triangle.

        Values remain discontinuous across cut curves and affine fields are
        reproduced from the original surface vertices.
        """
        if not isinstance(operand_values, tuple):
            raise TypeError("operand_values must be an immutable tuple.")
        values = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (first_values, second_values, *operand_values)
        )
        if len(values) != len(self.operand_ids) or any(
            value.ndim == 0 or value.shape[1:] != values[0].shape[1:] for value in values
        ):
            raise ValueError("operand values must align with every source operand.")
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("operand values must be finite.")
        component_shape = values[0].shape[1:]
        result = np.zeros(
            (self.triangles.shape[0], 3, *component_shape), dtype=np.float64
        )
        for operand, value in enumerate(values):
            rows = self.source_operand == operand
            if np.any(self.source_vertices[rows] >= value.shape[0]):
                raise ValueError(
                    f"operand {operand} values omit a required source vertex."
                )
            source = value[self.source_vertices[rows]]
            weights = self.source_barycentric[rows]
            result[rows] = np.sum(
                weights.reshape(*weights.shape, *(1,) * len(component_shape))
                * source[:, None],
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
            "kind": "native-triangle-surface-boolean",
            "operation": operation.value,
            "operands": tuple(model.model_id for model in operands),
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
                "native-triangle-surface-boolean",
                operation.value,
                *(model.model_id for model in operands),
            ),
            cell_tags=tags,
        ),
        vertex_global_ids=np.arange(vertices.shape[0], dtype=np.int64),
        cell_global_ids=np.arange(triangles.shape[0], dtype=np.int64),
    )


@checked
def surface_boolean(
    first: SurfaceModel | SurfaceBooleanResult,
    second: SurfaceModel | SurfaceBooleanResult,
    operation: SurfaceBooleanOperation,
    /,
    *,
    operands: tuple[SurfaceModel | SurfaceBooleanResult, ...] = (),
) -> SurfaceBooleanResult:
    """Compute union, intersection, or first-minus-later triangle surfaces.

    Operands must be closed, embedded, consistently oriented triangle meshes
    bounding a solid with winding counts in ``{0, 1}``. No orientation repair,
    CAD-source reconstruction, optional-engine fallback, or candidate quota is
    applied. Output properties retain source-face ancestry; source selections
    and interfaces require an explicit transfer.
    """
    if not isinstance(operands, tuple):
        raise TypeError("operands must be an immutable tuple.")
    source_models: list[SurfaceModel] = []
    expressions: list[_RegionExpression] = []
    for operand in (first, second, *operands):
        if isinstance(operand, SurfaceModel):
            expressions.append(_RegionExpression(operand_index=len(source_models)))
            source_models.append(operand)
        elif isinstance(operand, SurfaceBooleanResult):
            expressions.append(
                _shift_expression(operand.region_expression, len(source_models))
            )
            source_models.extend(operand.source_models)
        else:
            raise TypeError(
                "Boolean operands must be SurfaceModel or SurfaceBooleanResult values."
            )
    models = tuple(source_models)
    operation_ = parse(operation, SurfaceBooleanOperation, "operation")
    expression = _RegionExpression(operation=operation_, children=tuple(expressions))
    spatial_ids = tuple(model.metadata.coordinate_contract.spatial_id for model in models)
    if any(identity != spatial_ids[0] for identity in spatial_ids[1:]):
        raise ValueError(
            "Boolean operand coordinate contracts must share one spatial frame."
        )
    arrays = tuple(_operand_arrays(model) for model in models)
    for index, (vertices, faces, _) in enumerate(arrays):
        _require_solid(vertices, faces, index)
    arrangement, classification = arrange_classified_surfaces(
        tuple((vertices, faces) for vertices, faces, _ in arrays)
    )
    inside, component_count = _classify(arrangement, classification)
    rows, flipped = _emission(arrangement, inside, expression)
    emitted = arrangement.triangles[rows].copy()
    emitted[flipped] = emitted[flipped][:, (0, 2, 1)]
    if rows.size:
        used, first_use = np.unique(emitted.reshape(-1), return_index=True)
        ordered = used[np.argsort(first_use, kind="stable")]
        renumber = np.full((arrangement.vertices.shape[0],), -1, dtype=np.int64)
        renumber[ordered] = np.arange(ordered.size, dtype=np.int64)
        vertices = arrangement.vertices[ordered]
        triangles = renumber[emitted]
    else:
        ordered = np.empty((0,), dtype=np.int64)
        vertices = np.empty((0, 3), dtype=np.float64)
        triangles = np.empty((0, 3), dtype=np.int64)
    source_operand = arrangement.source_surface[rows]
    source_rows = arrangement.source_face[rows]
    source_vertices = np.empty((rows.size, 3), dtype=np.int64)
    source_cells = np.empty((rows.size,), dtype=np.int64)
    source_corners = np.empty((rows.size, 3, 3), dtype=np.float64)
    for index, (points, faces, cell_ids) in enumerate(arrays):
        selected = source_operand == index
        if not np.any(selected):
            continue
        source_vertices[selected] = faces[source_rows[selected]]
        source_cells[selected] = cell_ids[source_rows[selected]]
        source_corners[selected] = points[source_vertices[selected]]
    barycentric = _barycentric(
        vertices[triangles].reshape((-1, 3)),
        np.repeat(source_corners, 3, axis=0),
    ).reshape((-1, 3, 3))
    if rows.size:
        require_published_embedded(
            vertices,
            triangles,
            np.arange(rows.size, dtype=np.int64),
            surface=None,
        )
    closure, _, _ = _closure(vertices, triangles)
    result_surface = (
        _output_surface(
            models, operation_, vertices, triangles, source_operand, source_rows
        )
        if rows.size and closure.edge_manifold
        else None
    )
    return SurfaceBooleanResult(
        operation=operation_,
        vertices=vertices,
        vertex_bounds=arrangement.vertex_bounds[ordered],
        source_models=models,
        region_expression=expression,
        triangles=triangles,
        source_operand=source_operand,
        source_cell_global_ids=source_cells,
        source_vertices=source_vertices,
        source_barycentric=barycentric,
        reversed=flipped,
        surface=result_surface,
        arrangement=arrangement.evidence,
        closure=closure,
        classified_components=component_count,
    )


__all__ = [
    "surface_boolean",
    "SurfaceBooleanError",
    "SurfaceBooleanOperation",
    "SurfaceBooleanResult",
    "SurfaceBooleanStatus",
    "SurfaceClosureEvidence",
]
