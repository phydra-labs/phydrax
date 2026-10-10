#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Curving constraints transported through actual prepared prism columns."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..discretization.fem import FiniteElementSpec
from ._boundary_layer import BoundaryLayerMesh


class LayerCurvingNodes(NamedTuple):
    """Each prepared column node and its original-wall displacement owner."""

    owners: np.ndarray
    nodes: np.ndarray


def _column_roots(layers: BoundaryLayerMesh, /) -> tuple[dict[int, int], dict[int, int]]:
    """Trace source vertex IDs through the prepared layer's directed prism edges."""
    identifiers = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    wall = np.asarray(layers.wall_vertices, dtype=np.int64)
    roots = {
        int(identifiers[vertex]): int(identifiers[vertex])
        for vertex in wall
        if vertex >= 0
    }
    depths = dict.fromkeys(roots, 0)
    below: dict[int, int] = {}
    for block in layers.mesh.blocks:
        if block.cell_kind != "prism":
            raise ValueError("Layer-preserving curving requires prepared prism columns.")
        for vertices in identifiers[np.asarray(block.vertices, dtype=np.int64)]:
            for first, last in zip(
                vertices[:3].tolist(), vertices[3:].tolist(), strict=True
            ):
                if below.setdefault(last, first) != first:
                    raise ValueError(
                        "A prepared layer vertex has two column predecessors."
                    )
    for identifier in identifiers.tolist():
        current = identifier
        chain: list[int] = []
        visited: set[int] = set()
        while current not in roots:
            if current in visited or current not in below:
                raise ValueError(
                    "A prepared layer vertex has no unique original-wall column."
                )
            chain.append(current)
            visited.add(current)
            current = below[current]
        for vertex in reversed(chain):
            roots[vertex] = roots[current]
            depths[vertex] = depths[current] + 1
            current = vertex
    return roots, depths


def _source_layer_cells(
    mesh: CellMesh,
    layers: BoundaryLayerMesh,
    /,
) -> tuple[dict[int, tuple[int, ...]], np.ndarray]:
    """Require the exact source vertex registry and its unchanged numerical view."""
    source_vertices = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    source_cells = {
        int(identifier): tuple(source_vertices[vertices].tolist())
        for block in layers.mesh.blocks
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = {identifier: row for row, identifier in enumerate(identifiers.tolist())}
    if any(identifier not in rows for identifier in source_vertices.tolist()):
        raise ValueError("The carrier omits prepared layer vertex identities.")
    inherited = np.asarray(
        [rows[identifier] for identifier in source_vertices.tolist()], dtype=np.int64
    )
    if not np.array_equal(
        np.asarray(mesh.coordinates, dtype=np.float64)[inherited].view(np.uint64),
        np.asarray(layers.mesh.coordinates, dtype=np.float64).view(np.uint64),
    ):
        raise ValueError("The carrier changed the prepared source column coordinates.")
    return source_cells, identifiers


def prepare_layer_curving_nodes(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    layers: BoundaryLayerMesh,
    /,
) -> LayerCurvingNodes:
    """Bind nodal displacement groups by source cells and integer reference weights.

    A grouping is never inferred from near coordinates, block names, or a cell's
    apparent height. Every layer cell and column comes from the actual prepared
    layer source and must occur unchanged in the combined carrier.
    """
    if not isinstance(layers, BoundaryLayerMesh):
        raise TypeError("layers must be the actual prepared BoundaryLayerMesh.")
    roots, depths = _column_roots(layers)
    source_cells, identifiers = _source_layer_cells(mesh, layers)
    groups: dict[tuple[tuple[int, int], ...], set[int]] = {}
    owners: dict[tuple[tuple[int, int], ...], int] = {}
    seen: set[int] = set()
    elements, routes, _ = geometry.resolve(mesh)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        selected = [
            row
            for row, identifier in enumerate(np.asarray(block.global_ids).tolist())
            if identifier in source_cells
        ]
        if not selected:
            continue
        if block.cell_kind != "prism" or not isinstance(element, FiniteElementSpec):
            raise ValueError(
                "Prepared prism cells require the canonical nodal prism coordinate map."
            )
        nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        degree = element.degree
        barycentric = np.stack(
            (1.0 - nodes[:, 0] - nodes[:, 1], nodes[:, 0], nodes[:, 1]), axis=1
        )
        weights = np.rint(degree * barycentric).astype(np.int64)
        if not np.allclose(weights / degree, barycentric, rtol=0.0, atol=1e-14):
            raise ValueError(
                "Layer coordinate nodes must use their declared integer barycentric lattice."
            )
        for row in selected:
            identifier = int(np.asarray(block.global_ids)[row])
            vertices = identifiers[np.asarray(block.vertices)[row]].tolist()
            if set(vertices) != set(source_cells[identifier]):
                raise ValueError(
                    "The carrier changed a prepared layer cell's source vertex identities."
                )
            if any(
                roots[vertices[first]] != roots[vertices[last]]
                for first, last in zip(range(3), range(3, 6), strict=True)
            ):
                raise ValueError(
                    "The carrier changed a prepared layer cell's directed column pairs."
                )
            seen.add(identifier)
            bottom_depths = np.asarray(
                [depths[vertex] for vertex in vertices[:3]], dtype=np.int64
            )
            top_depths = np.asarray(
                [depths[vertex] for vertex in vertices[3:]], dtype=np.int64
            )
            if np.unique(bottom_depths).size != 1 or np.unique(top_depths).size != 1:
                raise ValueError(
                    "A prepared layer face mixes distinct physical intervals."
                )
            for node, weight, axial in zip(
                np.asarray(route[row], dtype=np.int64).tolist(),
                weights.tolist(),
                nodes[:, 2].tolist(),
                strict=True,
            ):
                key = tuple(
                    sorted(
                        (roots[vertex], int(value))
                        for vertex, value in zip(vertices[:3], weight, strict=True)
                        if value
                    )
                )
                groups.setdefault(key, set()).add(node)
                depth = (1.0 - axial) * bottom_depths[0] + axial * top_depths[0]
                if depth == 0.0 and owners.setdefault(key, node) != node:
                    raise ValueError(
                        "A prepared column has two distinct original-wall coordinate nodes."
                    )
    if seen != set(source_cells) or set(groups) != set(owners):
        raise ValueError(
            "Layer-preserving curving requires every prepared cell and its original-wall nodes."
        )
    pairs = sorted(
        (node, owners[key]) for key, members in groups.items() for node in members
    )
    node_owners: dict[int, int] = {}
    for node, owner in pairs:
        if node_owners.setdefault(node, owner) != owner:
            raise ValueError(
                "Shared layer coordinate nodes disagree on original-wall displacement ownership."
            )
    return LayerCurvingNodes(
        np.asarray([owner for _, owner in sorted(node_owners.items())], dtype=np.int64),
        np.asarray(sorted(node_owners), dtype=np.int64),
    )


def transport_layer_curvature(
    coordinates: np.ndarray,
    reference: np.ndarray,
    plan: LayerCurvingNodes,
    /,
) -> None:
    """Apply the same wall displacement at every axial level of a source column.

    The transported polynomial displacement is independent of the axial
    coordinate. The original physical column vectors therefore survive up to
    float64 coefficient roundoff, rather than only at the corner vertices.
    """
    displacement = coordinates[plan.owners] - reference[plan.owners]
    coordinates[plan.nodes] = reference[plan.nodes] + displacement


def prepare_layer_column_geometry(
    reference_mesh: CellMesh,
    layers: BoundaryLayerMesh,
    wall_mesh: CellMesh,
    wall_geometry: CellGeometrySpec,
    upper_geometry: CellGeometrySpec,
    upper_vertex_ids: np.ndarray,
    /,
    *,
    fiber_graph: bool = False,
) -> CellGeometrySpec:
    """Author exact source roots using original profiles and actual column banks.

    ``upper_vertex_ids`` explicitly identifies the upper root-complex vertex
    above each ``wall_mesh`` vertex. Reference charts never establish this
    identity by proximity. Actual interval endpoint coordinates and original
    profile controls remain separate dynamic banks; their differences are
    symbolic basis terms, not rounded intermediate coordinates.
    """
    from ..discretization._cell_geometry import LayerColumnCellGeometryElement

    roots, _ = _column_roots(layers)
    layer_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    physical_rows = {identifier: row for row, identifier in enumerate(layer_ids.tolist())}
    wall_ids = np.asarray(wall_mesh.vertex_global_ids, dtype=np.int64)
    upper_ids = np.asarray(upper_vertex_ids)
    if (
        upper_ids.shape != wall_ids.shape
        or upper_ids.dtype.kind not in "iu"
        or np.unique(upper_ids).size != upper_ids.size
    ):
        raise ValueError(
            "Original upper profiles require one distinct authored root vertex per original wall vertex."
        )
    top_roots = dict(zip(upper_ids.tolist(), wall_ids.tolist(), strict=True))
    if set(top_roots) & set(physical_rows):
        raise ValueError(
            "Upper profile root vertices cannot alias actual prepared layer vertex identities."
        )
    lower_elements, lower_routes, _ = wall_geometry.resolve(wall_mesh)
    upper_elements, upper_routes, _ = upper_geometry.resolve(wall_mesh)
    profiles: dict[tuple[int, ...], tuple[FiniteElementSpec, np.ndarray, np.ndarray]] = {}
    lower_start = layers.mesh.coordinates.shape[0]
    upper_start = lower_start + wall_geometry.coordinates.shape[0]
    for block, lower, upper, below, above in zip(
        wall_mesh.blocks,
        lower_elements,
        upper_elements,
        lower_routes,
        upper_routes,
        strict=True,
    ):
        if (
            block.cell_kind != "triangle"
            or not isinstance(lower, FiniteElementSpec)
            or not isinstance(upper, FiniteElementSpec)
            or lower.element_id != upper.element_id
        ):
            raise ValueError(
                "Original endpoint profiles require one canonical nodal triangle source basis."
            )
        for vertices, bottom, top in zip(
            np.asarray(block.vertices), np.asarray(below), np.asarray(above), strict=True
        ):
            key = tuple(wall_ids[vertices].tolist())
            if key in profiles:
                raise ValueError(
                    "An original wall column cannot own two endpoint profile cells."
                )
            profiles[key] = (lower, lower_start + bottom, upper_start + top)
    elements, routes = {}, {}
    reference_ids = np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64)
    actual_cells = {
        int(identifier): tuple(layer_ids[vertices].tolist())
        for block in layers.mesh.blocks
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    seen = set()
    for block in reference_mesh.blocks:
        if block.cell_kind != "prism":
            raise ValueError(
                "Source-controlled column roots must be independently authored prisms."
            )
        rows = []
        prototype = None
        for identifier, vertices in zip(
            np.asarray(block.global_ids).tolist(), np.asarray(block.vertices), strict=True
        ):
            identifiers = reference_ids[vertices].tolist()
            if any(vertex not in roots for vertex in identifiers[:3]):
                raise ValueError("A root column has no original physical wall ancestry.")
            key = tuple(roots[vertex] for vertex in identifiers[:3])
            if key not in profiles:
                raise ValueError(
                    "A root column changed its original wall profile's directed corner order."
                )
            element, bottom_profile, top_profile = profiles[key]
            if prototype is not None and prototype.element_id != element.element_id:
                raise ValueError(
                    "One column root block requires one original endpoint profile source basis."
                )
            prototype = element
            corner_dofs = np.asarray(
                [dofs[0] for dofs in element.entity_dofs[0]], dtype=np.int64
            )
            if identifier in actual_cells:
                if tuple(identifiers) != actual_cells[identifier]:
                    raise ValueError(
                        "Reference roots changed an actual physical layer cell."
                    )
                seen.add(identifier)
                top_profile = bottom_profile
                actual_top = np.asarray(
                    [physical_rows[vertex] for vertex in identifiers[3:]], dtype=np.int64
                )
            else:
                if any(
                    top_roots.get(vertex) != root
                    for vertex, root in zip(identifiers[3:], key, strict=True)
                ):
                    raise ValueError(
                        "Core root profiles require their explicitly authored original upper vertex correspondence."
                    )
                actual_top = top_profile[corner_dofs]
            actual_bottom = np.asarray(
                [physical_rows[vertex] for vertex in identifiers[:3]], dtype=np.int64
            )
            rows.append(
                np.concatenate(
                    (
                        bottom_profile,
                        top_profile,
                        actual_bottom,
                        actual_top,
                        bottom_profile[corner_dofs],
                        top_profile[corner_dofs],
                    )
                )
            )
        if prototype is None:
            raise ValueError(
                "Source-controlled column root blocks must contain actual authored cells."
            )
        elements[block.name] = LayerColumnCellGeometryElement(
            prototype, fiber_graph=fiber_graph
        )
        routes[block.name] = np.stack(rows)
    if seen != set(actual_cells):
        raise ValueError(
            "An independently authored root complex must retain every actual prepared physical layer cell."
        )
    coordinates = np.concatenate(
        (
            np.asarray(layers.mesh.coordinates, dtype=np.float64),
            np.asarray(wall_geometry.coordinates, dtype=np.float64),
            np.asarray(upper_geometry.coordinates, dtype=np.float64),
        )
    )
    geometry = CellGeometrySpec(elements, routes, coordinates)
    if fiber_graph:
        source = geometry.source_coordinates()
        resolved, resolved_routes, _ = geometry.resolve(reference_mesh)
        for element, route in zip(resolved, resolved_routes, strict=True):
            if not isinstance(element, LayerColumnCellGeometryElement):
                raise TypeError(
                    "Layer fiber graphs require their owning column elements."
                )
            for row in np.asarray(route, dtype=np.int64):
                element.fiber_reference_expressions(
                    tuple(source[int(index)] for index in row)
                )
    return geometry
