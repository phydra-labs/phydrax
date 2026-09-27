#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh element extraction, native quality certification, and curved geometry maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...discretization import CellGeometrySpec, CellMesh, lagrange_element
from ...discretization._reference_cell import reference_cell_topology
from .._contracts import MeshingFailure, MeshingFailureCategory
from .._trace import MeshingStageKind


_BLOCK_NAMES = {
    "triangle": "triangles",
    "quadrilateral": "quadrilaterals",
    "tetrahedron": "tetrahedra",
    "prism": "prisms",
    "hexahedron": "hexahedra",
}
_GMSH_FAMILIES = {
    "Triangle": "triangle",
    "Quadrilateral": "quadrilateral",
    "Tetrahedron": "tetrahedron",
    "Prism": "prism",
    "Hexahedron": "hexahedron",
}


@dataclass(frozen=True, slots=True)
class _ElementRows:
    tags: np.ndarray
    vertices: np.ndarray
    entity_tags: np.ndarray
    element_type: int
    cell_kind: str
    corner_count: int

    @property
    def block_name(self) -> str:
        return _BLOCK_NAMES[self.cell_kind]


def _element_rows(
    gmsh: Any, dimension: int, geometry_order: int, /
) -> tuple[_ElementRows, ...]:
    records: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    properties = {}
    for _, entity_tag in sorted(gmsh.model.getEntities(dimension)):
        element_types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(
            dimension, entity_tag
        )
        for element_type, tag_values, node_values in zip(
            element_types, tag_blocks, node_blocks, strict=True
        ):
            element_type = int(element_type)
            name, _, order, count, _, corners = gmsh.model.mesh.getElementProperties(
                element_type
            )
            family = name.split()[0]
            if family not in _GMSH_FAMILIES or int(order) != geometry_order:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    f"Gmsh returned unsupported element {name!r}; no elements may be discarded.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            tags = np.asarray(tag_values, dtype=np.int64)
            nodes = np.asarray(node_values, dtype=np.int64).reshape((-1, int(count)))
            records.setdefault(element_type, []).append(
                (tags, nodes, np.full(tags.shape, entity_tag, dtype=np.int64))
            )
            properties[element_type] = (_GMSH_FAMILIES[family], int(corners))
    if not records:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Gmsh returned no dimension-{dimension} elements.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    result = []
    for element_type, chunks in records.items():
        tags, nodes, entities = (
            np.concatenate(values) for values in zip(*chunks, strict=True)
        )
        order = np.argsort(tags, kind="stable")
        kind, corners = properties[element_type]
        result.append(
            _ElementRows(
                tags[order], nodes[order], entities[order], element_type, kind, corners
            )
        )
    key = (
        (lambda rows: (rows.corner_count, rows.block_name))
        if dimension == 2
        else (lambda rows: rows.block_name)
    )
    return tuple(sorted(result, key=key))


def _local_connectivity(node_tags: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    locations = np.searchsorted(node_tags, values)
    if np.any(locations >= node_tags.size) or not np.array_equal(
        node_tags[locations], values
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh element connectivity references an undeclared node tag.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return locations.astype(np.int32, copy=False)


def _curve_corner_rows(
    gmsh: Any, geometry_order: int, /
) -> tuple[np.ndarray, np.ndarray]:
    node_chunks = []
    entity_chunks = []
    for _, curve in sorted(gmsh.model.getEntities(1)):
        element_types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(1, curve)
        for element_type, tag_values, node_values in zip(
            element_types, tag_blocks, node_blocks, strict=True
        ):
            name, dimension, order, count, _, corners = (
                gmsh.model.mesh.getElementProperties(int(element_type))
            )
            tags = np.asarray(tag_values, dtype=np.int64)
            if (
                name.split()[0] != "Line"
                or int(dimension) != 1
                or int(order) != geometry_order
                or int(corners) != 2
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    f"Gmsh returned unsupported planar curve element {name!r}.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            nodes = np.asarray(node_values, dtype=np.int64).reshape((-1, int(count)))
            node_chunks.append(nodes[:, :2])
            entity_chunks.append(np.full(tags.shape, curve, dtype=np.int64))
    if not node_chunks:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh returned no planar curve elements.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return np.concatenate(node_chunks), np.concatenate(entity_chunks)


def _evaluate_element_maps(
    gmsh: Any,
    element_type: int,
    element_points: np.ndarray,
    local_points: np.ndarray,
    /,
) -> np.ndarray:
    """Evaluate Gmsh's own Lagrange maps at points of its reference element."""
    _, _, _, count, _, _ = gmsh.model.mesh.getElementProperties(element_type)
    reference = np.asarray(local_points, dtype=np.float64)
    # Gmsh always consumes (u, v, w) triples, independent of element dimension.
    padded = np.zeros((reference.shape[0], 3), dtype=np.float64)
    padded[:, : reference.shape[1]] = reference
    _, basis, _ = gmsh.model.mesh.getBasisFunctions(
        element_type, padded.reshape(-1).tolist(), "Lagrange"
    )
    values = np.asarray(basis, dtype=np.float64).reshape((-1, int(count)))
    if values.shape[0] != reference.shape[0]:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh basis tabulation does not match the requested reference points.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return values @ element_points


@dataclass(frozen=True, slots=True)
class _NativeQuality:
    minimum_jacobian: float
    minimum_sicn: float
    mean_sicn: float
    minimum_sige: float
    mean_sige: float

    @property
    def issues(self) -> tuple[str, ...]:
        return (
            ("native_scaled_inverse_condition",) if self.minimum_sicn <= 0.0 else ()
        ) + (
            ("native_scaled_inverse_gradient_error",) if self.minimum_sige <= 0.0 else ()
        )

    @property
    def achieved(self) -> tuple[tuple[str, float], ...]:
        return (
            ("gmsh_minimum_scaled_inverse_condition", self.minimum_sicn),
            ("gmsh_mean_scaled_inverse_condition", self.mean_sicn),
            ("gmsh_minimum_scaled_inverse_gradient_error", self.minimum_sige),
            ("gmsh_mean_scaled_inverse_gradient_error", self.mean_sige),
        )


def _audit_jacobians(gmsh: Any, rows: tuple[_ElementRows, ...], /) -> float:
    """Audit the curved map with Gmsh's adaptive determinant extrema, not corners."""
    minimum = np.inf
    for block in rows:
        determinants = np.asarray(
            gmsh.model.mesh.getElementQualities(block.tags, "minDetJac"), dtype=np.float64
        )
        invalid = ~np.isfinite(determinants) | (determinants <= 0.0)
        if determinants.shape != block.tags.shape or np.any(invalid):
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "Gmsh curved-element minimum Jacobian determinant is nonpositive or unavailable.",
                stage=MeshingStageKind.GEOMETRY_AUDIT.value,
                entity_ids=tuple(block.tags[invalid])
                if determinants.shape == block.tags.shape
                else (),
            )
        minimum = min(minimum, float(np.min(determinants)))
    return minimum


def _native_quality(gmsh: Any, rows: tuple[_ElementRows, ...], /) -> _NativeQuality:
    """Certify curved maps with Gmsh's own per-element quality measures.

    `minSICN` and `minSIGE` are Gmsh's scaled inverse condition number and
    inverse gradient error, bounded over the complete curved map.
    """
    minimum = _audit_jacobians(gmsh, rows)
    measures: dict[str, list[np.ndarray]] = {"minSICN": [], "minSIGE": []}
    for block in rows:
        for name, values in measures.items():
            quality = np.asarray(
                gmsh.model.mesh.getElementQualities(block.tags, name), dtype=np.float64
            )
            if quality.shape != block.tags.shape or not np.all(np.isfinite(quality)):
                raise MeshingFailure(
                    MeshingFailureCategory.AUDIT_FAILED,
                    f"Gmsh native {name} element quality is unavailable.",
                    stage=MeshingStageKind.QUALITY_EVALUATION.value,
                )
            values.append(quality)
    sicn = np.concatenate(measures["minSICN"])
    sige = np.concatenate(measures["minSIGE"])
    return _NativeQuality(
        minimum,
        float(np.min(sicn)),
        float(np.mean(sicn)),
        float(np.min(sige)),
        float(np.mean(sige)),
    )


def _reference_permutation(
    gmsh: Any, rows: _ElementRows, element: Any, /
) -> np.ndarray | None:
    """Map Gmsh reference nodes onto canonical element nodes when they coincide.

    This maps actual Gmsh reference nodes, not meshio's distinct wedge/hex
    ordering. `None` means the node sets differ (for example warp-and-blend
    simplex nodes above order two) and the map must be resampled instead.
    """
    _, dimension, _, count, coordinates, _ = gmsh.model.mesh.getElementProperties(
        rows.element_type
    )
    source = np.asarray(coordinates, dtype=np.float64).reshape(
        (int(count), int(dimension))
    )
    if rows.cell_kind in ("quadrilateral", "hexahedron"):
        source = 0.5 * (source + 1.0)
    elif rows.cell_kind == "prism":
        source[:, 2] = 0.5 * (source[:, 2] + 1.0)
    target = np.asarray(element.reference_nodes, dtype=np.float64)
    if source.shape != target.shape:
        return None
    matches = np.max(np.abs(target[:, None] - source[None]), axis=-1) <= 2.0e-12
    if not np.all(np.sum(matches, axis=1) == 1):
        return None
    return np.argmax(matches, axis=1).astype(np.int32)


@dataclass(frozen=True, slots=True)
class _CanonicalElementNodes:
    """Reference simplex nodes grouped by the reference entity carrying them."""

    barycentric: np.ndarray
    entities: tuple[tuple[int, tuple[int, ...], np.ndarray], ...]
    canonical: dict[int, np.ndarray]


def _canonical_element_nodes(element: Any, dimension: int, /) -> _CanonicalElementNodes:
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    barycentric = np.concatenate(
        (1.0 - np.sum(reference, axis=1, keepdims=True), reference), axis=1
    )
    topology = reference_cell_topology(element.cell_kind)
    support = barycentric > 1.0e-12
    entities = []
    canonical: dict[int, np.ndarray] = {}
    covered = np.zeros((reference.shape[0],), dtype=np.bool_)
    for entity_dimension in range(dimension + 1):
        for entity in topology.entities[entity_dimension]:
            vertices = tuple(int(value) for value in entity)
            mask = np.zeros((dimension + 1,), dtype=np.bool_)
            mask[list(vertices)] = True
            dofs = np.flatnonzero(np.all(support == mask[None, :], axis=1))
            covered[dofs] = True
            entities.append((entity_dimension, vertices, dofs))
            if entity_dimension not in canonical:
                canonical[entity_dimension] = barycentric[dofs][:, list(vertices)]
    if not np.all(covered):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Canonical simplex geometry nodes do not partition by reference entity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return _CanonicalElementNodes(barycentric, tuple(entities), canonical)


def _resampled_block(
    gmsh: Any,
    rows: _ElementRows,
    element: Any,
    cell_corners: np.ndarray,
    gmsh_corners: np.ndarray,
    element_points: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample Gmsh's curved simplex maps at canonical nodes with shared-entity keys.

    Both node sets span the same complete polynomial space, so resampling the
    Gmsh map reproduces it exactly. A node key is its carrying entity's sorted
    global vertices plus the index of its entity-local barycentric position in
    the canonical reference entity, which is orientation independent.
    """
    dimension = gmsh_corners.shape[1] - 1
    matches = cell_corners[:, :, None] == gmsh_corners[:, None, :]
    if not np.all(np.sum(matches, axis=2) == 1):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Canonical cells do not permute their Gmsh element corners.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    permutation = np.argmax(matches, axis=2)
    nodes = _canonical_element_nodes(element, dimension)
    count = nodes.barycentric.shape[0]
    values = np.empty((cell_corners.shape[0], count, 3), dtype=np.float64)
    groups, inverse = np.unique(permutation, axis=0, return_inverse=True)
    for group_index, group in enumerate(groups):
        selected = np.flatnonzero(inverse.reshape(-1) == group_index)
        gmsh_barycentric = np.empty_like(nodes.barycentric)
        gmsh_barycentric[:, group] = nodes.barycentric
        values[selected] = _evaluate_element_maps(
            gmsh, rows.element_type, element_points[selected], gmsh_barycentric[:, 1:]
        )
    keys = np.full((cell_corners.shape[0], count, dimension + 3), -1, dtype=np.int64)
    for entity_dimension, vertices, dofs in nodes.entities:
        if not dofs.size:
            continue
        global_vertices = cell_corners[:, list(vertices)]
        order = np.argsort(global_vertices, axis=1, kind="stable")
        local = nodes.barycentric[dofs][:, list(vertices)]
        oriented = np.take_along_axis(
            np.broadcast_to(local[None], (order.shape[0], *local.shape)),
            np.broadcast_to(order[:, None, :], (order.shape[0], *local.shape)),
            axis=2,
        )
        canonical = nodes.canonical[entity_dimension]
        distances = np.max(
            np.abs(oriented[:, :, None, :] - canonical[None, None, :, :]), axis=-1
        )
        index = np.argmin(distances, axis=2)
        if np.any(np.min(distances, axis=2) > 1.0e-9):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Canonical simplex geometry nodes are not symmetric under entity permutation.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        keys[:, dofs, 0] = entity_dimension
        keys[:, dofs, 1 : entity_dimension + 2] = np.sort(global_vertices, axis=1)[
            :, None, :
        ]
        keys[:, dofs, -1] = index
    return keys, values


@dataclass(frozen=True, slots=True)
class _CurvedGeometry:
    geometry: CellGeometrySpec
    conformity_residual: float | None


def _cell_geometry(
    gmsh: Any,
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    top_vertices: dict[str, np.ndarray],
    corner_nodes: np.ndarray,
    points: np.ndarray,
    output_points: np.ndarray,
    to_output: Any,
    geometry_order: int,
    flip_surface_routes: bool,
    /,
) -> _CurvedGeometry:
    if geometry_order == 1:
        return _CurvedGeometry(CellGeometrySpec.affine(mesh), None)
    elements = {}
    permutations = {}
    for block in rows:
        element = lagrange_element(block.cell_kind, geometry_order)
        elements[block.block_name] = element
        permutations[block.block_name] = _reference_permutation(gmsh, block, element)
    if all(value is not None for value in permutations.values()):
        return _CurvedGeometry(
            _routed_geometry(
                mesh,
                rows,
                row_orders,
                top_vertices,
                corner_nodes,
                output_points,
                elements,
                # ty: ignore[invalid-argument-type]
                permutations,
                flip_surface_routes,
            ),
            None,
        )
    if any(block.cell_kind not in ("triangle", "tetrahedron") for block in rows):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Resampled curved geometry requires simplex cells.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    key_chunks = []
    value_chunks = []
    route_shapes = []
    for block in rows:
        ordered = top_vertices[block.block_name][row_orders[block.block_name]]
        cell_corners = np.asarray(mesh.block(block.block_name).vertices, dtype=np.int64)
        gmsh_corners = np.full((ordered.shape[0], block.corner_count), -1, dtype=np.int64)
        lookup = np.full((points.shape[0],), -1, dtype=np.int64)
        lookup[corner_nodes] = np.arange(corner_nodes.size, dtype=np.int64)
        gmsh_corners[:] = lookup[ordered[:, : block.corner_count]]
        keys, values = _resampled_block(
            gmsh,
            block,
            elements[block.block_name],
            cell_corners,
            gmsh_corners,
            points[ordered],
        )
        key_chunks.append(keys.reshape((-1, keys.shape[-1])))
        value_chunks.append(values.reshape((-1, 3)))
        route_shapes.append((block.block_name, keys.shape[:2]))
    keys = np.concatenate(key_chunks)
    values = np.concatenate(value_chunks)
    unique, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    representative = values[first]
    residual = float(
        np.max(np.linalg.norm(values - representative[inverse], axis=1), initial=0.0)
    )
    scale = max(float(np.max(np.abs(points), initial=0.0)), 1.0)
    if residual > 1.0e-9 * scale:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh curved maps disagree on shared high-order entity traces.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    coordinates = to_output(representative)
    ordering = np.lexsort(
        tuple(
            coordinates[:, column] for column in range(coordinates.shape[1] - 1, -1, -1)
        )
    )
    position = np.empty((unique.shape[0],), dtype=np.int32)
    position[ordering] = np.arange(unique.shape[0], dtype=np.int32)
    routes = {}
    cursor = 0
    for name, shape in route_shapes:
        stop = cursor + shape[0] * shape[1]
        routes[name] = position[inverse[cursor:stop]].reshape(shape)
        cursor = stop
    return _CurvedGeometry(
        CellGeometrySpec(elements, routes, coordinates[ordering]), residual
    )


def _routed_geometry(
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    top_vertices: dict[str, np.ndarray],
    corner_nodes: np.ndarray,
    output_points: np.ndarray,
    elements: dict,
    permutations: dict[str, np.ndarray],
    flip_surface_routes: bool,
    /,
) -> CellGeometrySpec:
    """Route canonical geometry nodes directly to coinciding Gmsh nodes."""
    geometry_ordering = np.lexsort(
        tuple(
            output_points[:, column]
            for column in range(output_points.shape[1] - 1, -1, -1)
        )
    )
    point_to_geometry = np.empty((output_points.shape[0],), dtype=np.int32)
    point_to_geometry[geometry_ordering] = np.arange(
        output_points.shape[0], dtype=np.int32
    )
    routes = {}
    for block in rows:
        element = elements[block.block_name]
        route = top_vertices[block.block_name][row_orders[block.block_name]][
            :, permutations[block.block_name]
        ]
        if flip_surface_routes:
            expected = corner_nodes[
                np.asarray(mesh.block(block.block_name).vertices, dtype=np.int32)
            ]
            flipped = np.any(route[:, :3] != expected, axis=1)
            if np.any(flipped):
                reference = np.asarray(element.reference_nodes)
                matches = (
                    np.max(np.abs(reference[:, None] - reference[None, :, ::-1]), axis=-1)
                    <= 2.0e-12
                )
                route[flipped] = route[flipped][:, np.argmax(matches, axis=1)]
        routes[block.block_name] = point_to_geometry[route]
    return CellGeometrySpec(elements, routes, output_points[geometry_ordering])
