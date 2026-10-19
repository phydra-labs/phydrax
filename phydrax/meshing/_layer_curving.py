#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Curving constraints transported through realized prepared prism columns."""

from __future__ import annotations

from typing import final

import numpy as np

import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.meshing._boundary_layer as _dependency_phydrax_meshing__boundary_layer

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization.fem import FiniteElementSpec
from ..typing import checked


@final
class LayerCurvingNodes(StrictModule, NonTrainableState):
    """Each prepared column node and its source-wall displacement owner."""

    owners: np.ndarray
    nodes: np.ndarray


@checked
def column_roots(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh, /
) -> tuple[dict[int, int], dict[int, int]]:
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
            raise ValueError(
                "BoundaryLayerMesh.mesh must provide prepared prism columns; got missing or incompatible prepared prism columns"
            )
        for vertices in identifiers[np.asarray(block.vertices, dtype=np.int64)]:
            for first, last in zip(
                vertices[:3].tolist(), vertices[3:].tolist(), strict=True
            ):
                if below.setdefault(last, first) != first:
                    raise ValueError(
                        "BoundaryLayerMesh.mesh must retain prepared prisms with unique directed wall-column ancestry; got a prepared layer vertex has two column predecessors"
                    )
    for identifier in identifiers.tolist():
        current = identifier
        chain: list[int] = []
        visited: set[int] = set()
        while current not in roots:
            if current in visited or current not in below:
                raise ValueError(
                    "BoundaryLayerMesh.mesh must retain prepared prisms with unique directed wall-column ancestry; got a prepared layer vertex has no unique source-wall column"
                )
            chain.append(current)
            visited.add(current)
            current = below[current]
        for vertex in reversed(chain):
            roots[vertex] = roots[current]
            depths[vertex] = depths[current] + 1
            current = vertex
    return roots, depths


@checked
def source_layer_cells(
    mesh: CellMesh,
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    /,
) -> tuple[dict[int, tuple[int, ...]], np.ndarray]:
    """Require the source vertex registry and its unchanged numerical view."""
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
        raise ValueError(
            "source_layer_cells.mesh must retain the prepared source vertex IDs and bitwise coordinates; got the carrier omits prepared layer vertex identities"
        )
    inherited = np.asarray(
        [rows[identifier] for identifier in source_vertices.tolist()], dtype=np.int64
    )
    if not np.array_equal(
        np.asarray(mesh.coordinates, dtype=np.float64)[inherited].view(np.uint64),
        np.asarray(layers.mesh.coordinates, dtype=np.float64).view(np.uint64),
    ):
        raise ValueError(
            "source_layer_cells.mesh must retain the prepared source vertex IDs and bitwise coordinates; got the carrier changed the prepared source column coordinates"
        )
    return source_cells, identifiers


@final
class LayerCurvingGroups(StrictModule, NonTrainableState):
    groups: dict[tuple[tuple[int, int], ...], set[int]]
    owners: dict[tuple[tuple[int, int], ...], int]
    seen: set[int]


@final
class PrismNodeLattice(StrictModule, NonTrainableState):
    nodes: np.ndarray
    weights: np.ndarray


def _prism_node_lattice(element: FiniteElementSpec) -> PrismNodeLattice:
    """Bind nodal barycentric weights to the declared integer lattice."""
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    degree = element.degree
    barycentric = np.stack(
        (1.0 - nodes[:, 0] - nodes[:, 1], nodes[:, 0], nodes[:, 1]), axis=1
    )
    weights = np.rint(degree * barycentric).astype(np.int64)
    if not np.allclose(weights / degree, barycentric, rtol=0.0, atol=1e-14):
        raise ValueError(
            "FiniteElementSpec.reference_nodes must use their declared integer barycentric lattice; got incompatible Layer coordinate nodes"
        )
    return PrismNodeLattice(nodes, weights)


def _prepared_column_depths(
    identifier: int,
    vertices: list[int],
    source_cells: dict[int, tuple[int, ...]],
    roots: dict[int, int],
    depths: dict[int, int],
    seen: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Require unchanged source corners, directed pairs, and unmixed intervals."""
    if set(vertices) != set(source_cells[identifier]):
        raise ValueError(
            "prepare_layer_curving_nodes.mesh must retain source cell corners, directed pairs, and unmixed physical intervals; got the carrier changed a prepared layer cell's source vertex identities"
        )
    if any(
        roots[vertices[first]] != roots[vertices[last]]
        for first, last in zip(range(3), range(3, 6), strict=True)
    ):
        raise ValueError(
            "prepare_layer_curving_nodes.mesh must retain source cell corners, directed pairs, and unmixed physical intervals; got the carrier changed a prepared layer cell's directed column pairs"
        )
    seen.add(identifier)
    bottom_depths = np.asarray(
        [depths[vertex] for vertex in vertices[:3]], dtype=np.int64
    )
    top_depths = np.asarray([depths[vertex] for vertex in vertices[3:]], dtype=np.int64)
    if np.unique(bottom_depths).size != 1 or np.unique(top_depths).size != 1:
        raise ValueError(
            "prepare_layer_curving_nodes.mesh must retain source cell corners, directed pairs, and unmixed physical intervals; got a prepared layer face mixes distinct physical intervals"
        )
    return bottom_depths, top_depths


def _collect_layer_node_groups(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    roots: dict[int, int],
    depths: dict[int, int],
    source_cells: dict[int, tuple[int, ...]],
    identifiers: np.ndarray,
) -> LayerCurvingGroups:
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
        match element:
            case FiniteElementSpec() if block.cell_kind == "prism":
                pass
            case _:
                raise ValueError(
                    "prepare_layer_curving_nodes.geometry must provide the standard nodal prism coordinate map; got missing or incompatible the standard nodal prism coordinate map"
                )
        lattice = _prism_node_lattice(element)
        nodes, weights = lattice.nodes, lattice.weights
        for row in selected:
            identifier = int(np.asarray(block.global_ids)[row])
            vertices = identifiers[np.asarray(block.vertices)[row]].tolist()
            bottom_depths, top_depths = _prepared_column_depths(
                identifier, vertices, source_cells, roots, depths, seen
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
                        "prepare_layer_curving_nodes.geometry must provide a nodal prism map with one source-wall displacement owner per column; got a prepared column has two distinct source-wall coordinate nodes"
                    )
    return LayerCurvingGroups(groups, owners, seen)


@checked
def prepare_layer_curving_nodes(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    /,
) -> LayerCurvingNodes:
    """Bind nodal displacement groups by source cells and integer reference weights.

    A grouping is never inferred from near coordinates, block names, or a cell's
    apparent height. Every layer cell and column comes from the realized prepared
    layer source and must occur unchanged in the combined carrier.
    """
    roots, depths = column_roots(layers)
    source_cells, identifiers = source_layer_cells(mesh, layers)
    collected = _collect_layer_node_groups(
        mesh, geometry, roots, depths, source_cells, identifiers
    )
    groups, owners, seen = collected.groups, collected.owners, collected.seen
    if seen != set(source_cells) or set(groups) != set(owners):
        raise ValueError(
            "prepare_layer_curving_nodes.geometry must provide every prepared cell and its source-wall nodes; got missing or incompatible every prepared cell and its source-wall nodes"
        )
    pairs = sorted(
        (node, owners[key]) for key, members in groups.items() for node in members
    )
    node_owners: dict[int, int] = {}
    for node, owner in pairs:
        if node_owners.setdefault(node, owner) != owner:
            raise ValueError(
                "prepare_layer_curving_nodes.geometry must retain every source cell and unambiguous source-wall displacement ownership; got shared layer coordinate nodes disagree on source-wall displacement ownership"
            )
    return LayerCurvingNodes(
        np.asarray([owner for _, owner in sorted(node_owners.items())], dtype=np.int64),
        np.asarray(sorted(node_owners), dtype=np.int64),
    )


@checked
def transport_layer_curvature(
    coordinates: np.ndarray,
    reference: np.ndarray,
    plan: LayerCurvingNodes,
    /,
) -> None:
    """Apply the same wall displacement at every axial level of a source column.

    The transported polynomial displacement is independent of the axial
    coordinate. The source physical column vectors therefore survive up to
    float64 coefficient roundoff, rather than only at the corner vertices.
    """
    displacement = coordinates[plan.owners] - reference[plan.owners]
    coordinates[plan.nodes] = reference[plan.nodes] + displacement


@final
class LayerEndpointProfiles(StrictModule, NonTrainableState):
    roots: dict[int, int]
    physical_rows: dict[int, int]
    top_roots: dict[int, int]
    profiles: dict[tuple[int, ...], tuple[FiniteElementSpec, np.ndarray, np.ndarray]]


def _prepare_layer_endpoint_profiles(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    wall_mesh: CellMesh,
    wall_geometry: CellGeometrySpec,
    upper_geometry: CellGeometrySpec,
    upper_vertex_ids: np.ndarray,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerEndpointProfiles:
    """Bind endpoint profiles by authored upper IDs and directed wall corners."""
    roots, _ = column_roots(layers)
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
            "prepare_layer_column_geometry.upper_vertex_ids must provide one distinct authored root vertex per source wall vertex; got missing or incompatible one distinct authored root vertex per source wall vertex"
        )
    top_roots = dict(zip(upper_ids.tolist(), wall_ids.tolist(), strict=True))
    if set(top_roots) & set(physical_rows):
        raise ValueError(
            "prepare_layer_column_geometry.upper_vertex_ids must bind distinct upper roots to compatible directed wall profiles; got upper profile root vertices cannot alias realized prepared layer vertex identities"
        )
    lower_elements, lower_routes, _ = wall_geometry.resolve(wall_mesh, budget=budget)
    upper_elements, upper_routes, _ = upper_geometry.resolve(wall_mesh, budget=budget)
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
        match (lower, upper):
            case (FiniteElementSpec() as lower, FiniteElementSpec() as upper) if (
                block.cell_kind == "triangle" and lower.element_id == upper.element_id
            ):
                pass
            case _:
                raise ValueError(
                    "prepare_layer_column_geometry.wall_geometry must share a nodal triangle basis with upper_geometry; got mismatched endpoint bases"
                )
        for vertices, bottom, top in zip(
            np.asarray(block.vertices), np.asarray(below), np.asarray(above), strict=True
        ):
            key = tuple(wall_ids[vertices].tolist())
            if key in profiles:
                raise ValueError(
                    "prepare_layer_column_geometry.wall_geometry must bind one endpoint profile per directed wall cell; got duplicate profile ownership"
                )
            profiles[key] = (lower, lower_start + bottom, upper_start + top)
    return LayerEndpointProfiles(roots, physical_rows, top_roots, profiles)


@final
class LayerColumnRoutes(StrictModule, NonTrainableState):
    elements: dict[
        str, _owner_phydrax_discretization__cell_geometry.LayerColumnCellGeometryElement
    ]
    routes: dict[str, np.ndarray]


def _column_physical_controls(
    identifier: int,
    identifiers: list[int],
    key: tuple[int, ...],
    element: FiniteElementSpec,
    bottom_profile: np.ndarray,
    top_profile: np.ndarray,
    actual_cells: dict[int, tuple[int, ...]],
    seen: set[int],
    physical_rows: dict[int, int],
    top_roots: dict[int, int],
) -> np.ndarray:
    """Keep realized layer intervals distinct from explicitly authored upper roots."""
    corner_dofs = np.asarray([dofs[0] for dofs in element.entity_dofs[0]], dtype=np.int64)
    if identifier in actual_cells:
        if tuple(identifiers) != actual_cells[identifier]:
            raise ValueError(
                "prepare_layer_column_geometry.reference_mesh must retain physical layer cells and authored upper-root correspondence; got reference roots changed an realized physical layer cell"
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
                "prepare_layer_column_geometry.reference_mesh must provide their explicitly authored source upper vertex correspondence; got missing or incompatible their explicitly authored source upper vertex correspondence"
            )
        actual_top = top_profile[corner_dofs]
    actual_bottom = np.asarray(
        [physical_rows[vertex] for vertex in identifiers[:3]], dtype=np.int64
    )
    return np.concatenate(
        (
            bottom_profile,
            top_profile,
            actual_bottom,
            actual_top,
            bottom_profile[corner_dofs],
            top_profile[corner_dofs],
        )
    )


def _prepare_layer_column_routes(
    reference_mesh: CellMesh,
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    endpoints: LayerEndpointProfiles,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    fiber_graph: bool,
) -> LayerColumnRoutes:
    """Bind every authored prism to its endpoint and physical interval banks."""
    roots, physical_rows = endpoints.roots, endpoints.physical_rows
    top_roots, profiles = endpoints.top_roots, endpoints.profiles
    layer_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
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
                "prepare_layer_column_geometry.reference_mesh must be independently authored prisms; got incompatible Source-controlled column roots"
            )
        rows = []
        prototype = None
        for identifier, vertices in zip(
            np.asarray(block.global_ids).tolist(), np.asarray(block.vertices), strict=True
        ):
            identifiers = reference_ids[vertices].tolist()
            if any(vertex not in roots for vertex in identifiers[:3]):
                raise ValueError(
                    "prepare_layer_column_geometry.reference_mesh must retain every authored prism and its directed endpoint profile basis; got a root column has no source physical wall ancestry"
                )
            key = tuple(roots[vertex] for vertex in identifiers[:3])
            if key not in profiles:
                raise ValueError(
                    "prepare_layer_column_geometry.reference_mesh must retain every authored prism and its directed endpoint profile basis; got a root column changed its source wall profile's directed corner order"
                )
            element, bottom_profile, top_profile = profiles[key]
            if prototype is not None and prototype.element_id != element.element_id:
                raise ValueError(
                    "prepare_layer_column_geometry.reference_mesh must provide one source endpoint profile source basis; got missing or incompatible one source endpoint profile source basis"
                )
            prototype = element
            rows.append(
                _column_physical_controls(
                    identifier,
                    identifiers,
                    key,
                    element,
                    bottom_profile,
                    top_profile,
                    actual_cells,
                    seen,
                    physical_rows,
                    top_roots,
                )
            )
        if prototype is None:
            raise ValueError(
                "prepare_layer_column_geometry.reference_mesh must contain realized authored cells; got incompatible Source-controlled column root blocks"
            )
        elements[block.name] = (
            _owner_phydrax_discretization__cell_geometry.LayerColumnCellGeometryElement(
                prototype,
                fiber_graph=fiber_graph,
                budget=budget,
            )
        )
        routes[block.name] = np.stack(rows)
    if seen != set(actual_cells):
        raise ValueError(
            "prepare_layer_column_geometry.reference_mesh must retain every realized prepared physical layer cell; got incompatible independently authored root complex"
        )
    return LayerColumnRoutes(elements, routes)


@checked
def prepare_layer_column_geometry(
    reference_mesh: CellMesh,
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    wall_mesh: CellMesh,
    wall_geometry: CellGeometrySpec,
    upper_geometry: CellGeometrySpec,
    upper_vertex_ids: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    fiber_graph: bool = False,
) -> CellGeometrySpec:
    """Author source roots using source profiles and realized column banks.

    ``upper_vertex_ids`` explicitly identifies the upper root-complex vertex
    above each ``wall_mesh`` vertex. Reference charts never establish this
    identity by proximity. Realized interval endpoint coordinates and source
    profile controls remain separate dynamic banks; their differences are
    symbolic basis terms, not rounded intermediate coordinates.
    """

    endpoints = _prepare_layer_endpoint_profiles(
        layers, wall_mesh, wall_geometry, upper_geometry, upper_vertex_ids, budget
    )
    prepared = _prepare_layer_column_routes(
        reference_mesh, layers, endpoints, budget, fiber_graph
    )
    elements, routes = prepared.elements, prepared.routes
    coordinates = np.concatenate(
        (
            np.asarray(layers.mesh.coordinates, dtype=np.float64),
            np.asarray(wall_geometry.coordinates, dtype=np.float64),
            np.asarray(upper_geometry.coordinates, dtype=np.float64),
        )
    )
    geometry = CellGeometrySpec(elements, routes, coordinates, budget=budget)
    if fiber_graph:
        source = geometry.source_coordinates(budget=budget)
        resolved, resolved_routes, _ = geometry.resolve(reference_mesh, budget=budget)
        for element, route in zip(resolved, resolved_routes, strict=True):
            match element:
                case _owner_phydrax_discretization__cell_geometry.LayerColumnCellGeometryElement():
                    pass
                case _:
                    raise TypeError(
                        "prepare_layer_column_geometry.reference_mesh must provide their defining column elements; got missing or incompatible their defining column elements"
                    )
            for row in np.asarray(route, dtype=np.int64):
                element.fiber_reference_expressions(
                    tuple(source[int(index)] for index in row),
                    budget=budget,
                )
    return geometry
