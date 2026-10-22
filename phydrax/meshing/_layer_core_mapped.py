#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lift an realized reference partition through independently authored source roots."""

from __future__ import annotations

import dataclasses as _owner_dataclasses
from contextlib import ExitStack
from typing import final

import numpy as np

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _dependency_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._periodic_topology as _owner_phydrax_discretization__periodic_topology
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.geometry._source_fidelity as _owner_phydrax_geometry__mesh_certificates
import phydrax.meshing._boundary_layer as _dependency_phydrax_meshing__boundary_layer
import phydrax.meshing._canonical as _owner_phydrax_meshing__canonical
import phydrax.meshing._certification_schedule as _leaf_phydrax_meshing__certification_schedule
import phydrax.meshing._controls as _owner_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._layer_core as _owner_phydrax_meshing__layer_core
import phydrax.meshing._layer_core_controls as _owner_phydrax_meshing__layer_core_controls
import phydrax.meshing._layer_core_publication as layer_publication
import phydrax.meshing._layer_core_resources as _dependency_phydrax_meshing__layer_core_resources
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._layer_curving as _owner_phydrax_meshing__layer_curving
import phydrax.meshing._layer_source_certification as _owner_phydrax_meshing__layer_source_certification
import phydrax.meshing._mapped_reference_association as _owner_phydrax_meshing__mapped_reference_association
import phydrax.meshing._measurements as _owner_phydrax_meshing__measurements
import phydrax.meshing._organization_records as _owner_phydrax_meshing__organization
import phydrax.meshing._reference_root_composition as _dependency_phydrax_meshing__reference_root_composition
import phydrax.meshing._result as _owner_phydrax_meshing__result
import phydrax.meshing._scope as _owner_phydrax_meshing_scope
import phydrax.meshing._sizing as _owner_phydrax_meshing__sizing
import phydrax.meshing._trace as _owner_phydrax_meshing__trace
import phydrax.meshing.providers._native_publication as _owner_phydrax_meshing_providers__native_publication
import phydrax.meshing.providers._native_route_checks as route_checks
import phydrax.meshing.providers._native_sources as sources_owner
from phydrax.discretization._reference_cell_types import CellGeometryElement

from .._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization.fem import FiniteElementSpec
from ..geometry._rational import rational_vector
from ..typing import checked
from ._contracts import MeshingProviderInfo, VolumeMeshingSpec
from ._family_execution import family_route_refusals
from ._measurements import NativeMeshingPhaseRecorder
from ._result import CellMeshingResult
from ._scope import mesh_entity_scope


@final
class LayerReferenceComposition(StrictModule, NonTrainableState):
    mesh: CellMesh
    geometry: CellGeometrySpec
    reference_mesh: CellMesh
    cell_order: np.ndarray
    cell_regions: np.ndarray


@checked
def validate_layer_reference_columns(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    reference_layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Prove the entire source physical interval vectors, not sampled heights."""
    if not np.array_equal(
        layers.mesh.vertex_global_ids, reference_layers.mesh.vertex_global_ids
    ):
        raise ValueError(
            "NativeLayerCoreSource.reference_source must retain the realized physical source vertex registry; got incompatible Reference layer columns"
        )
    if not np.array_equal(
        layers.layer_index, reference_layers.layer_index
    ) or not np.array_equal(
        layers.column_index,
        reference_layers.column_index,
    ):
        raise ValueError(
            "NativeLayerCoreSource.reference_source must retain every realized physical interval and column identity; got incompatible Reference layer columns"
        )
    reference_cells = {
        int(identifier): tuple(
            np.asarray(reference_layers.mesh.vertex_global_ids)[vertices].tolist()
        )
        for block in reference_layers.mesh.blocks
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    elements, routes, _ = domain.source_geometry.resolve(
        domain.reference_mesh, budget=budget
    )
    values = domain.source_geometry.source_coordinates(budget=budget)
    roots = {
        int(identifier): (
            block,
            element,
            np.asarray(route, dtype=np.int64)[row],
            vertices,
        )
        for block, element, route in zip(
            domain.reference_mesh.blocks, elements, routes, strict=True
        )
        for row, (identifier, vertices) in enumerate(
            zip(np.asarray(block.global_ids), np.asarray(block.vertices), strict=True)
        )
    }
    physical_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    root_vertex_ids = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    affine = (
        _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
            "prism", 1
        )
    )
    for block in layers.mesh.blocks:
        if block.cell_kind != "prism":
            raise ValueError(
                "NativeLayerCoreSource.reference_source must provide realized prepared prism columns; got missing or incompatible realized prepared prism columns"
            )
        for identifier, vertices in zip(
            np.asarray(block.global_ids).tolist(), np.asarray(block.vertices), strict=True
        ):
            work.charge(1)
            if identifier not in roots or identifier not in reference_cells:
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must own its independently declared reference root; got incompatible physical source layer cell"
                )
            root_block, element, route, root_vertices = roots[identifier]
            expected = tuple(physical_ids[vertices].tolist())
            if (
                root_block.cell_kind != "prism"
                or reference_cells[identifier] != expected
                or tuple(root_vertex_ids[root_vertices].tolist()) != expected
            ):
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must retain source vertex IDs, prism cells, intervals, and polynomial column vectors; got reference roots changed an realized physical column's directed source corners"
                )
            if not np.array_equal(
                np.asarray(domain.reference_mesh.coordinates)[root_vertices].view(
                    np.uint64
                ),
                np.asarray(reference_layers.mesh.coordinates)[vertices].view(np.uint64),
            ):
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must retain their realized authored reference column coordinates; got incompatible Source layer roots"
                )
            local = tuple(values[index] for index in route)
            physical = np.asarray(layers.mesh.coordinates, dtype=np.float64)[vertices]
            images = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
                element, local, budget=budget
            )
            if images != tuple(rational_vector(point) for point in physical):
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must retain source vertex IDs, prism cells, intervals, and polynomial column vectors; got reference source roots changed the source physical column corners"
                )
            source = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
                element, local, budget=budget
            )
            original = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
                affine, physical, budget=budget
            )
            if source is None or original is None:
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must provide their source polynomial coordinate expressions; got missing or incompatible their source polynomial coordinate expressions"
                )
            work.charge(sum(len(value) for value in (*source, *original)))
            if any(
                _dependency_phydrax_discretization__coordinate_enclosure.derivative(
                    _dependency_phydrax_discretization__coordinate_enclosure.add(
                        current,
                        _dependency_phydrax_discretization__coordinate_enclosure.scale(
                            initial, -1, budget=budget
                        ),
                        budget=budget,
                    ),
                    2,
                    budget=budget,
                )
                for current, initial in zip(source, original, strict=True)
            ):
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must retain source vertex IDs, prism cells, intervals, and polynomial column vectors; got mapped layer curvature changed an entire source physical interval vector"
                )
    if set(reference_cells) != set(
        np.asarray(layers.mesh.entity_set(3).entity_ids).tolist()
    ):
        raise ValueError(
            "NativeLayerCoreSource.reference_source must retain source vertex IDs, prism cells, intervals, and polynomial column vectors; got the reference layer source added or omitted an source physical column cell"
        )


@checked
def require_reference_root_sheets(
    reference: sources_owner.NativeLayerCoreSource,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    /,
) -> None:
    """Require realized core-root incidence sheets, even within one material."""

    mapping, _, polygons, points = layer_topology.prepare_identity(
        reference.layers,
        reference.complex,
        reference.vertex_layer_ids,
        reference.cap_polygon_ids,
        work=work,
    )
    layer_ids = np.asarray(reference.layers.mesh.vertex_global_ids, dtype=np.int64)
    extra = points.shape[0] - layer_ids.size
    next_vertex = int(np.max(layer_ids)) + 1
    if extra and next_vertex + extra - 1 > np.iinfo(np.int64).max:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "NativeLayerCoreSource.reference_source must retain the root vertex registry and every internal root incidence sheet; got the reference layer vertex registry has no room for its source core source identities",
        )
    combined_ids = np.concatenate(
        (layer_ids, next_vertex + np.arange(extra, dtype=np.int64))
    )
    constrained = {tuple(sorted(row.tolist())) for row in combined_ids[mapping[polygons]]}
    cells = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in domain.reference_mesh.blocks
        ]
    )
    sorted_cells = np.sort(cells)
    regions = np.empty(cells.size, dtype=np.int64)
    regions[np.searchsorted(sorted_cells, cells)] = domain.cell_regions
    layer_cells = set(np.asarray(reference.layers.mesh.entity_set(3).entity_ids).tolist())
    vertices = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    for incidents in layer_topology.faces(
        domain.reference_mesh, regions, work=work
    ).values():
        if len(incidents) != 2 or any(
            int(sorted_cells[face.cell]) in layer_cells for face in incidents
        ):
            continue
        for triangle in layer_topology.triangles(incidents[0].vertices):
            work.charge(1)
            key = tuple(sorted(vertices[np.asarray(triangle, dtype=np.int64)].tolist()))
            if key not in constrained:
                raise ValueError(
                    "NativeLayerCoreSource.reference_source must retain every source internal core-root incidence sheet, including equal-material sheets; got incompatible Native reference fill"
                )


@final
class LayerReferenceRoots(StrictModule, NonTrainableState):
    """Source root inventory and source coefficient bank."""

    cells: list[tuple[CellBlock, CellGeometryElement, np.ndarray, int, int]]
    coordinates: (
        _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank
    )
    points: np.ndarray
    vertex_ids: np.ndarray


def _prepare_layer_reference_roots(
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerReferenceRoots:
    source_elements, source_routes, _ = domain.source_geometry.resolve(
        domain.reference_mesh,
        budget=budget,
    )
    source_coordinates = domain.source_geometry.source_coordinates(budget=budget)
    reference_points = np.asarray(domain.reference_mesh.coordinates, dtype=np.float64)
    source_ids = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    roots: list[tuple[CellBlock, CellGeometryElement, np.ndarray, int, int]] = []
    source_offset = 0
    for block, element, route in zip(
        domain.reference_mesh.blocks, source_elements, source_routes, strict=True
    ):
        match block:
            case CellBlock():
                pass
            case _:
                raise TypeError(
                    "MappedReferenceDomain.reference_mesh must provide standard fixed-arity CellBlock reference cells; got missing or incompatible standard fixed-arity CellBlock reference cells"
                )
        for row in range(block.cell_count):
            roots.append(
                (
                    block,
                    element,
                    np.asarray(route, dtype=np.int64),
                    row,
                    int(domain.cell_regions[source_offset + row]),
                )
            )
        source_offset += block.cell_count
    return LayerReferenceRoots(roots, source_coordinates, reference_points, source_ids)


def _unique_layer_root_chart(
    target: np.ndarray,
    region: int,
    roots: list[tuple[CellBlock, CellGeometryElement, np.ndarray, int, int]],
    reference_points: np.ndarray,
    chart: FiniteElementSpec,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
    """Select the unique entire source chart without altering root traversal order."""
    candidates: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    for parent, (
        root_block,
        source_element,
        route,
        root_row,
        root_region,
    ) in enumerate(roots):
        work.charge(1)
        if region != root_region:
            continue
        root_vertices = np.asarray(root_block.vertices[root_row], dtype=np.int64)
        corners = reference_points[root_vertices]
        work.ticket.consume(geometry_queries=1)
        if np.any(np.min(target, axis=0) < np.min(corners, axis=0)) or np.any(
            np.max(target, axis=0) > np.max(corners, axis=0)
        ):
            continue
        controls = _dependency_phydrax_meshing__reference_root_composition.exact_reference_chart_controls(
            target, corners, root_block.cell_kind, chart, budget=budget
        )
        if controls is not None:
            candidates.append((parent, controls))
    if len(candidates) != 1:
        raise ValueError(
            "MappedReferenceDomain.reference_mesh must provide a unique entire root chart for each reference target cell; got an realized reference cell has no unique entire source-root chart; preserve all internal root incidence sheets during fill"
        )
    return candidates[0]


def _assemble_layer_reference_mesh(
    reference_mesh: CellMesh,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    regions: np.ndarray,
    physical_points: np.ndarray,
    groups: dict[str, list[tuple[int, np.ndarray, int, int, np.ndarray, np.ndarray]]],
    elements: dict[
        str,
        _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
    ],
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerReferenceComposition:
    """Publish sorted chart blocks with their root restriction ancestry."""
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    blocks, routes, parents, parent_vertices, order = [], {}, {}, {}, []
    for name in sorted(groups):
        cells = sorted(groups[name], key=lambda cell: cell[2])
        blocks.append(
            CellBlock(
                name,
                elements[name].cell_kind,
                np.stack([cell[1] for cell in cells]),
                global_ids=np.asarray([cell[2] for cell in cells], dtype=np.int64),
            )
        )
        routes[name] = np.stack([cell[5] for cell in cells])
        parents[name] = np.asarray([cell[3] for cell in cells], dtype=np.int64)
        parent_vertices[name] = np.stack([cell[4] for cell in cells])
        order.extend(cell[0] for cell in cells)
    ancestry = (
        _dependency_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
            _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                domain.source_geometry
            ),
            domain.reference_mesh.topology_id,
            parents,
            parent_vertices,
        )
    )
    geometry = CellGeometrySpec(
        elements,
        routes,
        domain.source_geometry.coordinates,
        restriction_source=ancestry,
        periodic_source=domain.source_geometry.periodic_source,
        budget=budget,
    )
    reference = CellMesh(
        points,
        tuple(blocks),
        vertex_global_ids=reference_mesh.vertex_global_ids,
        numeric_version=reference_mesh.numeric_version,
    )
    mesh = reference.with_coordinates(
        physical_points, numeric_version=domain.source_revision
    )
    cell_order = np.asarray(order, dtype=np.int64)
    return LayerReferenceComposition(
        mesh, geometry, reference, cell_order, regions[cell_order]
    )


@checked
def compose_layer_reference_mesh(
    reference_mesh: CellMesh,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    cell_regions: np.ndarray,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerReferenceComposition:
    """Bind every realized cell to its unique entire reference-root chart.

    Native fill must retain every internal source-root incidence sheet, including
    sheets whose two material labels agree. A target crossing a sheet has no
    containing root and is refused; no physical inverse or nearest-root choice
    can conceal missing reference partition constraints.
    """
    if reference_mesh.topological_dimension != 3 or reference_mesh.ambient_dimension != 3:
        raise ValueError(
            "compose_layer_reference_mesh.reference_mesh must provide a three-dimensional reference partition; got missing or incompatible a three-dimensional reference partition"
        )
    count = sum(block.cell_count for block in reference_mesh.blocks)
    regions = np.asarray(cell_regions)
    if regions.shape != (count,) or regions.dtype.kind not in "iu":
        raise ValueError(
            "compose_layer_reference_mesh.reference_mesh must bind every realized cell in block order; got incompatible Layer reference regions"
        )
    inventory = _prepare_layer_reference_roots(domain, budget)
    roots, source_coordinates = inventory.cells, inventory.coordinates
    reference_points, source_ids = inventory.points, inventory.vertex_ids
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    groups: dict[str, list[tuple[int, np.ndarray, int, int, np.ndarray, np.ndarray]]] = {}
    elements: dict[
        str,
        _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
    ] = {}
    exact_points: dict[
        int,
        tuple[_dependency_phydrax_discretization__coordinate_enclosure.ExactScalar, ...],
    ] = {}
    budget.reserve(0, points.size * np.dtype(np.float64).itemsize)
    physical_points = np.empty(points.shape, dtype=np.float64)
    position = 0
    for block in reference_mesh.blocks:
        match block:
            case CellBlock():
                pass
            case _:
                raise TypeError(
                    "compose_layer_reference_mesh.reference_mesh must provide standard fixed-arity CellBlock reference cells; got missing or incompatible standard fixed-arity CellBlock reference cells"
                )
        chart = (
            _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                block.cell_kind, 1
            )
        )
        for target_row in range(block.cell_count):
            identifier = int(np.asarray(block.global_ids)[target_row])
            vertices = np.asarray(block.vertices[target_row], dtype=np.int64)
            parent, (numerators, denominators) = _unique_layer_root_chart(
                points[vertices],
                int(regions[position]),
                roots,
                reference_points,
                chart,
                work,
                budget,
            )
            root_block, source_element, route, root_row, _ = roots[parent]
            source_element = _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
                source_element, "Layer source root"
            )
            element = _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement(
                source_element,
                chart,
                numerators,
                denominators,
                budget=budget,
            )
            name = f"root-chart:{element.element_id}"
            elements[name] = element
            source_route = np.asarray(route[root_row], dtype=np.int64)
            root_vertices = np.asarray(root_block.vertices[root_row], dtype=np.int64)
            groups.setdefault(name, []).append(
                (
                    position,
                    vertices,
                    identifier,
                    int(np.asarray(root_block.global_ids)[root_row]),
                    source_ids[root_vertices],
                    source_route,
                )
            )
            images = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
                element,
                tuple(source_coordinates[int(index)] for index in source_route.tolist()),
                budget=budget,
            )
            if images is None:
                raise ValueError(
                    "compose_layer_reference_mesh.reference_mesh must retain a three-dimensional reference partition with region-bound root charts; got a layer root composition lacks its source coordinate expression"
                )
            work.charge(len(images))
            for vertex, image in zip(vertices.tolist(), images, strict=True):
                previous = exact_points.get(vertex)
                if previous is not None and previous != image:
                    raise ValueError(
                        "compose_layer_reference_mesh.reference_mesh must retain a three-dimensional reference partition with region-bound root charts; got independent layer source charts disagree exactly at a shared reference vertex"
                    )
                exact_points[vertex] = image
                physical_points[vertex] = (
                    _dependency_phydrax_discretization__coordinate_enclosure.rounded_point(
                        image
                    )
                )
            position += 1
    if len(exact_points) != points.shape[0]:
        raise ValueError(
            "compose_layer_reference_mesh.reference_mesh must retain a three-dimensional reference partition with region-bound root charts; got the reference partition has a vertex without an source-root chart"
        )
    return _assemble_layer_reference_mesh(
        reference_mesh, domain, regions, physical_points, groups, elements, budget
    )


def _mapped_organization_records(
    construction: layer_topology.LayerCoreConstruction, mesh: CellMesh, /
) -> int:
    """Correspondence rows per degree plus every remapped scope and attribute row."""
    return (
        sum(
            2 * construction.mesh.entity_set(degree).count + mesh.entity_set(degree).count
            for degree in range(4)
        )
        + sum(
            np.asarray(item.scope.entity_ids).size
            for item in (*construction.zones, *construction.patches, *construction.labels)
        )
        + 3
        * sum(
            np.asarray(attribute.scope.entity_ids).size
            for attribute in construction.attributes
        )
    )


def _mapped_layer_organization(
    construction: layer_topology.LayerCoreConstruction,
    mesh: CellMesh,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> tuple[
    tuple[_owner_phydrax_meshing__organization.MeshZone, ...],
    tuple[_owner_phydrax_meshing__organization.MeshPatch, ...],
    tuple[_owner_phydrax_meshing__organization.MeshLabel, ...],
    tuple[_owner_phydrax_meshing__organization.MeshAttribute, ...],
]:

    correspondences = {}
    for degree in range(4):
        old = np.asarray(construction.mesh.entity_set(degree).entity_ids, dtype=np.int64)
        new = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
        work.charge(2 * old.size + new.size)
        if degree in (0, 3):
            # Composition retains explicit vertex/cell identities; only edge
            # and face registries are rebuilt from directed incidence.
            if not np.array_equal(np.sort(old), np.sort(new)):
                raise ValueError(
                    "LayerCoreConstruction.mesh must retain every explicit source vertex and cell identity; got incompatible Mapped organization"
                )
            correspondences[degree] = dict(zip(old.tolist(), old.tolist(), strict=True))
            continue
        targets = dict(
            zip(
                _owner_phydrax_meshing__canonical.entity_vertex_keys(mesh, degree),
                new.tolist(),
                strict=True,
            )
        )
        correspondences[degree] = {
            identifier: targets[key]
            for identifier, key in zip(
                old.tolist(),
                _owner_phydrax_meshing__canonical.entity_vertex_keys(
                    construction.mesh, degree
                ),
                strict=True,
            )
        }

    def scope(
        original: _owner_phydrax_meshing_scope.MeshingScope,
    ) -> _owner_phydrax_meshing_scope.MeshingScope:
        degree = original.entity_dimension
        identifiers = np.asarray(original.entity_ids)
        work.charge(identifiers.size)
        return mesh_entity_scope(
            mesh,
            degree,
            np.asarray(
                [
                    correspondences[degree][identifier]
                    for identifier in identifiers.tolist()
                ],
                dtype=np.int64,
            ),
        )

    zones = tuple(
        _owner_phydrax_meshing__organization.MeshZone(
            zone.name,
            zone.role,
            scope(zone.scope),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for zone in construction.zones
    )
    zone_ids = {
        old.zone_id: new.zone_id
        for old, new in zip(construction.zones, zones, strict=True)
    }
    patches = tuple(
        _owner_phydrax_meshing__organization.MeshPatch(
            patch.name,
            scope(patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(
                zone_ids[identifier] for identifier in patch.adjacent_zone_ids
            ),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
        )
        for patch in construction.patches
    )
    labels = tuple(
        _owner_phydrax_meshing__organization.MeshLabel(label.name, scope(label.scope))
        for label in construction.labels
    )
    attributes = []
    for attribute in construction.attributes:
        identifiers = np.asarray(attribute.scope.entity_ids)
        work.charge(2 * identifiers.size)
        renamed = np.asarray(
            [
                correspondences[attribute.scope.entity_dimension][identifier]
                for identifier in identifiers.tolist()
            ],
            dtype=np.int64,
        )
        attributes.append(
            _owner_phydrax_meshing__organization.MeshAttribute(
                attribute.name,
                attribute.role,
                scope(attribute.scope),
                np.asarray(attribute.values)[np.argsort(renamed, kind="stable")],
                unit=attribute.unit,
            )
        )
    return zones, patches, labels, tuple(attributes)


def _compose_mapped_layer_carrier(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_topology.LayerCoreConstruction,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerReferenceComposition:
    """Lift source roots, bind quotient ancestry, and preserve physical columns."""
    composed = compose_layer_reference_mesh(
        construction.mesh,
        domain,
        construction.cell_regions,
        work,
        budget=budget,
    )
    mesh = composed.mesh
    periodic = source.layers.mesh.periodic_topology
    reference_periodic = construction.mesh.periodic_topology
    if (periodic is None) != (reference_periodic is None):
        raise ValueError(
            "NativeLayerCoreSource.layers must retain the same declared periodic column ancestry; got incompatible Physical and reference layer sources"
        )
    if periodic is not None and reference_periodic is not None:
        topology = _owner_phydrax_discretization__periodic_topology.PeriodicMeshTopology(
            mesh,
            periodic.cell,
            reference_periodic.vertex_representatives,
            reference_periodic.vertex_shifts,
        )
        mesh = CellMesh(
            mesh.coordinates,
            mesh.blocks,
            vertex_global_ids=mesh.vertex_global_ids,
            numeric_version=mesh.numeric_version,
            periodic_topology=topology,
        )

    _owner_phydrax_meshing__layer_curving.source_layer_cells(mesh, source.layers)
    return LayerReferenceComposition(
        mesh,
        composed.geometry,
        composed.reference_mesh,
        composed.cell_order,
        composed.cell_regions,
    )


@final
class MappedLayerCompliance(StrictModule, NonTrainableState):
    size: _owner_phydrax_meshing__sizing.UniformSizeControl
    report: _owner_phydrax_meshing__result.MeshingComplianceReport


def _mapped_core_size_bounds(
    source: sources_owner.NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> tuple[float, ...]:
    return tuple(
        control.core_maximum_size
        for control in (*specification.layer_controls, source.layers.control)
        if control.core_maximum_size is not None
    )


def _mapped_compliance_records(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    mesh: CellMesh,
    /,
) -> int:
    """Vertex rows, edge keys, lengths and growth; core rows, edges and lengths."""
    records = mesh.coordinates.shape[0] + 3 * mesh.entity_set(1).count
    if _mapped_core_size_bounds(source, specification):
        records += 16 * sum(
            block.cell_count for block in mesh.blocks if block.cell_kind == "tetrahedron"
        )
    return records


def _mapped_layer_compliance(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    construction: layer_topology.LayerCoreConstruction,
    mesh: CellMesh,
    preparation: layer_publication.LayerRoutePreparation,
) -> MappedLayerCompliance:
    """Measure physical size and retain reference construction work separately."""
    periodic_requested, periodic_achieved = (
        preparation.periodic_requested,
        preparation.periodic_achieved,
    )
    work = preparation.work
    size = specification.size_controls[0]
    match size:
        case _owner_phydrax_meshing__sizing.UniformSizeControl():
            pass
        case _:
            raise TypeError(
                "VolumeMeshingSpec.size_controls must provide its source uniform physical size control; got missing or incompatible its source uniform physical size control"
            )
    work.charge(mesh.coordinates.shape[0] + 3 * mesh.entity_set(1).count)
    lengths, growth = route_checks.edge_size_evidence(
        np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                mesh, 1
            ),
            dtype=np.int64,
        ),
    )
    requested, achieved, issues = route_checks.uniform_size_compliance(
        size, specification.size_compliance, lengths, growth
    )
    bounds = _mapped_core_size_bounds(source, specification)
    if bounds:
        cap = min(bounds)
        vertices = np.asarray(mesh.coordinates, dtype=np.float64)
        core_cells = np.concatenate(
            [
                np.asarray(block.vertices, dtype=np.int64)
                for block in mesh.blocks
                if block.cell_kind == "tetrahedron"
            ]
        )
        work.charge(core_cells.size + 6 * core_cells.shape[0])
        core_edges = route_checks.unique_edges(core_cells, "tetrahedron")
        work.charge(core_edges.shape[0])
        maximum = float(
            np.max(
                np.linalg.norm(
                    vertices[core_edges[:, 1]] - vertices[core_edges[:, 0]],
                    axis=1,
                )
            )
        )
        requested.append(("core_maximum_size", cap))
        achieved.append(("core_maximum_edge", maximum))
        if (
            maximum
            > cap
            + specification.size_compliance.absolute_tolerance
            + specification.size_compliance.relative_tolerance * cap
        ):
            issues.append("core_maximum_size")
    achieved.extend(
        (f"reference:construction:{name}", value)
        for name, value in construction.core.construction_counters
    )
    achieved.extend(
        (
            ("reference:native:work_units", construction.core.work_units),
            ("reference:native:steiner_points", construction.core.steiner_points),
        )
    )
    evidence = source.layers.evidence
    compliance = _owner_phydrax_meshing__result.MeshingComplianceReport(
        specification.specification_id,
        requested=(
            *requested,
            *periodic_requested,
            *(
                (f"layer:{index}:thickness", value)
                for index, value in enumerate(evidence.requested_thicknesses)
            ),
        ),
        achieved=(
            *achieved,
            *periodic_achieved,
            ("original_polynomial_column_vectors", 1),
            ("fixed_cap_vertices_bitwise", 1),
            ("fixed_reference_root_sheets", 1),
            ("layer_core_prepublication_work_units", work.work_units),
            *(
                (f"layer:{index}:mean_thickness", float(value))
                for index, value in enumerate(np.asarray(evidence.achieved_thicknesses))
                if np.isfinite(value)
            ),
        ),
        issues=tuple(issues),
    )
    return MappedLayerCompliance(size, compliance)


@checked
def execute_mapped_layer_core_route(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    prepared: layer_publication.PreparedMappedLayer,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Native reference fill followed by source-root, column-preserving curving."""
    with ExitStack() as resources:
        reference, domain = source.reference_source, source.mapped_domain
        reference_prepared = prepared.reference_prepared
        if reference is None or domain is None or reference_prepared is None:
            raise ValueError(
                "execute_mapped_layer_core_route.source must provide the independently prepared source roots and reference partition; got missing or incompatible the independently prepared source roots and reference partition"
            )
        started = scope.operation_started
        limits = specification.limits
        preparation = layer_publication.prepare_layer_route(
            source,
            specification,
            prepared.source_work_units,
            scope,
            resources,
            started=started,
        )
        work = preparation.work
        audit_policy = preparation.policy.audit_policy
        construction = _owner_phydrax_meshing__layer_core.generate_layer_core(
            reference.layers,
            reference.complex,
            reference_prepared.core_specification,
            prepared.schedule,
            validity_policy=audit_policy.validity_policy,
            scope=scope,
            vertex_layer_ids=reference.vertex_layer_ids,
            cap_polygon_ids=reference.cap_polygon_ids,
            layer_regions=reference.layer_regions,
            source_id=reference.source_id,
            source_revision=reference.source_revision,
            input_id=plan_id,
            source_binding=reference,
            source_work_units=work.work_units,
            record_phase=record_phase,
        )
        work.work_units = construction.work_units
        allowance = scope.remaining()
        with family_route_refusals(
            limits,
            _owner_phydrax_meshing__trace.MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        ):
            ticket = resources.enter_context(
                scope.batch(
                    min(limits.maximum_work_units, allowance.remaining_work_units),
                    queries_upper=min(
                        limits.maximum_geometry_queries,
                        allowance.remaining_geometry_queries,
                    ),
                    bytes_upper=min(
                        limits.maximum_scratch_bytes, allowance.remaining_managed_bytes
                    ),
                )
            )
        work.ticket = ticket
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        resources.enter_context(budget.activate())
        with _owner_phydrax_meshing__measurements.measure_phase(record_phase, "curving"):
            composed = _compose_mapped_layer_carrier(
                source, construction, domain, work, budget
            )
            mesh = composed.mesh
        with _owner_phydrax_meshing__measurements.measure_phase(
            record_phase, "geometry_association"
        ):
            associations = _owner_phydrax_meshing__mapped_reference_association.mapped_reference_associations(
                domain,
                mesh,
                composed.geometry,
                budget=budget,
                maximum_support_queries=min(
                    limits.maximum_geometry_queries, limits.maximum_work_units
                ),
            )
        # The coordinate phases settle; their caches stay reserved for publication,
        # and host organization, controls, compliance and fidelity requests are
        # admitted from carrier counts before they allocate.
        resources.close()
        resources.enter_context(
            budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size))
        )
        host_upper = (
            _mapped_organization_records(construction, mesh)
            + _mapped_compliance_records(source, specification, mesh)
            + _owner_phydrax_meshing__layer_core.layer_publication_work_upper(
                source,
                specification,
                _owner_dataclasses.replace(construction, mesh=mesh),
            )
        )
        with _dependency_phydrax_meshing__layer_core_resources.admit_layer_phase(
            scope,
            limits,
            work,
            _owner_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION,
            work_upper=host_upper,
        ):
            with _owner_phydrax_meshing__measurements.measure_phase(
                record_phase, "organization"
            ):
                zones, patches, labels, attributes = _mapped_layer_organization(
                    construction, mesh, work=work
                )
                physical = _owner_dataclasses.replace(
                    construction,
                    mesh=mesh,
                    domain=domain,
                    cell_regions=composed.cell_regions,
                    zones=zones,
                    patches=patches,
                    labels=labels,
                    attributes=attributes,
                )
                zones, patches = (
                    _owner_phydrax_meshing__layer_core_controls.compose_layer_controls(
                        source, specification, physical, work=work
                    )
                )
            with _owner_phydrax_meshing__measurements.measure_phase(
                record_phase, "compliance"
            ):
                measured = _mapped_layer_compliance(
                    source, specification, construction, mesh, preparation
                )
                size, compliance = measured.size, measured.report
            scoped = _owner_phydrax_meshing__layer_source_certification.layer_source_fidelity_requests(
                source, specification, physical, work=work
            )
        fidelity = source.fidelity_source
        if fidelity is None:
            raise ValueError(
                "execute_mapped_layer_core_route.source must retain its independent source physical geometry query; got incompatible Mapped layer/core publication"
            )

        match fidelity:
            case _owner_phydrax_geometry__mesh_certificates.MappedDomainBoundarySource():
                # The source owns immutable root-region facts; certification binds the
                # same mapped boundary to this publication's realized cell inventory.
                fidelity = fidelity.for_target_regions(composed.cell_regions)
            case _:
                pass

        tolerance = min(
            (
                feature.maximum_deviation
                for feature in specification.protected_features
                if feature.feature_kind
                is _owner_phydrax_meshing__controls.FeatureKind.SURFACE
                and feature.scope.scope_id == specification.boundary_scope.scope_id
            ),
            default=size.target_size,
        )
        request = _owner_phydrax_meshing_providers__native_publication.NativeCertificationRequest(
            _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
                "mapped_volume"
            ),
            source.source_id,
            source.source_revision,
            limits,
            domain=domain,
            cell_regions=composed.cell_regions,
            fidelity_source=fidelity,
            fidelity_tolerance=tolerance,
            scoped_fidelity=scoped,
        )
        route_checks.check_deadline(
            started, limits, _owner_phydrax_meshing__trace.MeshingStageKind.CERTIFICATION
        )
        publication = layer_publication.LayerCorePublication(
            mesh=mesh,
            compliance=compliance,
            stages=construction.stages,
            provenance={
                "kind": "native-mapped-layer-core",
                "plan": plan_id,
                "specification": specification.specification_id,
                "source": source.binding_id,
                "reference": reference.binding_id,
                "mapped_domain": domain.domain_id,
            },
            certification=request,
            geometry=composed.geometry,
            organization=_owner_phydrax_meshing_providers__native_publication.NativePublicationOrganization(
                zones=zones,
                patches=patches,
                labels=labels,
                attributes=attributes,
                associations=associations,
            ),
        )
        return layer_publication.publish_layer_route(
            publication,
            preparation,
            coordinate_contract,
            provider,
            scope,
            workspace,
            budget=budget,
            record_phase=record_phase,
        )
