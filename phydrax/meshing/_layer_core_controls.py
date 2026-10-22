#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Retained source region/patch requests on the full immutable hybrid volume."""

from __future__ import annotations

from typing import final

import numpy as np

import phydrax.discretization as _owner_phydrax_discretization
import phydrax.geometry._meshing_domain as _dependency_phydrax_geometry__meshing_domain
import phydrax.meshing._layer_core_resources as _dependency_phydrax_meshing__layer_core_resources
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing.providers._native_sources as sources_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._contracts import VolumeMeshingSpec
from ._scope import mesh_entity_scope


@checked
def layer_control_issues(
    source: sources_owner.NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues: list[str] = []
    controls = {}
    for control in specification.region_controls:
        if (
            control.scope.entity_dimension != 3
            or control.region_name not in source.region_ids
            or not np.array_equal(
                np.asarray(control.scope.entity_ids),
                [source.region_ids.index(control.region_name)],
            )
        ):
            issues.append(
                "region controls bound to their authoritative layer/core region"
            )
            continue
        if not control.meshing_enabled:
            issues.append(
                "disabled regions represented as void, not occupied layer/core cells"
            )
        previous = controls.get(control.region_name)
        if previous is not None and (previous.material_id, previous.role) != (
            control.material_id,
            control.role,
        ):
            issues.append("contradictory layer/core material and role controls")
        controls[control.region_name] = control
        for seed in specification.region_seeds:
            if seed.region_name == control.region_name and (
                seed.material_id,
                seed.role,
            ) != (control.material_id, control.role):
                issues.append("contradictory layer/core region seed material and role")
    boundary = specification.boundary_scope
    allowed = np.asarray(boundary.entity_ids, dtype=np.int64)
    for patch in specification.patch_controls:
        if (
            patch.scope.entity_dimension != 2
            or np.setdiff1d(np.asarray(patch.scope.entity_ids), allowed).size
        ):
            issues.append("patch controls bound to authoritative supplied source faces")
        if not set(patch.adjacent_region_names) <= set(source.region_ids):
            issues.append("patch controls naming authoritative layer/core regions")
    return issues


def _wall_face_sources(source: sources_owner.NativeLayerCoreSource, /) -> dict[int, int]:
    association = source.layers.wall_association
    if association is None:
        return {}
    indices = np.asarray(association.source_indices, dtype=np.int64)
    domain = source.source_domain
    match domain:
        case _dependency_phydrax_geometry__meshing_domain.MeshingDomain():
            slots = dict(
                zip(
                    zip(
                        domain.source_indices[2],
                        domain.source_occurrences[2],
                        strict=True,
                    ),
                    domain.scope_indices(2),
                    strict=True,
                )
            )
            indices = np.asarray(
                [
                    slots[key]
                    for key in zip(
                        indices.tolist(),
                        association.source_entities.occurrence_paths(),
                        strict=True,
                    )
                ],
                dtype=np.int64,
            )
        case _:
            pass
    return dict(
        zip(
            np.asarray(association.target_global_ids, dtype=np.int64).tolist(),
            indices.tolist(),
            strict=True,
        )
    )


@final
class LateralColumnRoots(StrictModule, NonTrainableState):
    roots: dict[int, int]


@final
class LateralFaceOwners(StrictModule, NonTrainableState):
    edge_owners: dict[tuple[int, int], int]


@final
class LateralFaceSources(StrictModule, NonTrainableState):
    sources: dict[tuple[int, ...], int]


def _prepare_lateral_column_roots(
    source: sources_owner.NativeLayerCoreSource,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> LateralColumnRoots:
    """Trace directed layer incidence to unique source wall vertices."""
    layers = source.layers
    wall = np.asarray(layers.wall_vertices, dtype=np.int64)
    roots = {int(vertex): int(vertex) for vertex in wall if vertex >= 0}
    predecessors: dict[int, int] = {}
    for block in layers.mesh.blocks:
        match block:
            case _owner_phydrax_discretization.CellBlock(
                cell_kind="prism" | "hexahedron"
            ):
                pass
            case _:
                raise ValueError(
                    "BoundaryLayerMesh.mesh must provide their realized directed prism or hexahedron column incidence; got missing or incompatible their realized directed prism or hexahedron column incidence"
                )
        corners = block.vertices.shape[1] // 2
        for row in np.asarray(block.vertices, dtype=np.int64).tolist():
            work.charge(corners)
            for bottom, top in zip(row[:corners], row[corners:], strict=True):
                if predecessors.setdefault(top, bottom) != bottom:
                    raise ValueError(
                        "BoundaryLayerMesh.mesh must have unique directed wall-column ancestry; got a lateral layer vertex has conflicting source column predecessors"
                    )
    for vertex in predecessors:
        current = vertex
        chain = []
        seen = set()
        while current not in roots:
            work.charge(1)
            if current in seen or current not in predecessors:
                raise ValueError(
                    "BoundaryLayerMesh.mesh must have unique directed wall-column ancestry; got a lateral layer trace lacks unique source wall-column ancestry"
                )
            seen.add(current)
            chain.append(current)
            current = predecessors[current]
        for member in chain:
            roots[member] = roots[current]
    return LateralColumnRoots(roots)


def _prepare_lateral_face_owners(
    source: sources_owner.NativeLayerCoreSource,
    mapping: np.ndarray,
    polygons: np.ndarray,
    owners: np.ndarray,
    roots: dict[int, int],
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> LateralFaceOwners:
    """Bind source-face ownership along boundary cap-edge orbits."""
    cap_edges: dict[tuple[int, int], int] = {}
    for polygon in mapping[
        polygons[np.asarray(source.cap_polygon_ids, dtype=np.int64)]
    ].tolist():
        for first, last in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
            work.charge(1)
            key = tuple(sorted((first, last)))
            cap_edges[key] = cap_edges.get(key, 0) + 1
    boundary_edges = {key for key, count in cap_edges.items() if count == 1}
    edge_owners: dict[tuple[int, int], int] = {}
    for polygon, owner in zip(mapping[polygons].tolist(), owners.tolist(), strict=True):
        if owner < 0:
            continue
        for first, last in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
            work.charge(1)
            edge = tuple(sorted((first, last)))
            if edge not in boundary_edges:
                continue
            root_edge = (min(roots[first], roots[last]), max(roots[first], roots[last]))
            if edge_owners.setdefault(root_edge, owner) != owner:
                raise ValueError(
                    "NativeLayerCoreSource.cap_polygon_ids must bind one source-face owner per boundary cap edge; got one source cap edge cannot claim conflicting lateral source faces"
                )
    return LateralFaceOwners(edge_owners)


def _exterior_lateral_edge(
    vertices: np.ndarray,
    facet: tuple[int, ...],
    corners: int,
    faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
) -> tuple[tuple[int, ...], tuple[int, int]] | None:
    """Select an exterior lateral quad and require its realized lower column edge."""
    if (
        len(facet) != 4
        or all(index < corners for index in facet)
        or all(index >= corners for index in facet)
    ):
        return None
    key = tuple(sorted(vertices[np.asarray(facet, dtype=np.int64)].tolist()))
    if len(faces[key]) != 1:
        return None
    bottom = [int(vertices[index]) for index in facet if index < corners]
    if len(bottom) != 2:
        raise ValueError(
            "BoundaryLayerMesh.mesh must have a lower column edge on each exterior lateral quad; got an exterior lateral reference facet lacks its realized column edge"
        )
    return key, (bottom[0], bottom[1])


def _layer_lateral_face_sources(
    source: sources_owner.NativeLayerCoreSource,
    mapping: np.ndarray,
    polygons: np.ndarray,
    owners: np.ndarray,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    /,
) -> LateralFaceSources:
    """Transport cap-edge source ownership along realized reference-cell columns."""

    layers = source.layers
    roots = _prepare_lateral_column_roots(source, work).roots
    edge_owners = _prepare_lateral_face_owners(
        source, mapping, polygons, owners, roots, work
    ).edge_owners
    faces = layer_topology.faces(
        layers.mesh, np.asarray(source.layer_regions, dtype=np.int64), work=work
    )
    cells = np.sort(np.asarray(layers.mesh.entity_set(3).entity_ids, dtype=np.int64))
    intervals = np.asarray(layers.layer_index, dtype=np.int64)
    columns = np.asarray(layers.column_index, dtype=np.int64)
    claims: dict[tuple[int, tuple[int, int]], tuple[int, set[int]]] = {}
    output: dict[tuple[int, ...], int] = {}
    for block in layers.mesh.blocks:
        match block:
            case _owner_phydrax_discretization.CellBlock():
                pass
            case _:
                raise TypeError(
                    "NativeLayerCoreSource.layers must provide standard fixed reference-cell incidence; got missing or incompatible standard fixed reference-cell incidence"
                )
        corners = block.vertices.shape[1] // 2
        topology = _owner_phydrax_discretization.reference_cell_topology(block.cell_kind)
        for cell in range(block.cell_count):
            identifier = int(np.asarray(block.global_ids)[cell])
            position = int(np.searchsorted(cells, identifier))
            vertices = np.asarray(block.vertices[cell], dtype=np.int64)
            for facet in topology.entities[2]:
                selected = _exterior_lateral_edge(vertices, facet, corners, faces)
                if selected is None:
                    continue
                key, bottom = selected
                first, last = roots[bottom[0]], roots[bottom[1]]
                root_edge = (min(first, last), max(first, last))
                if root_edge not in edge_owners:
                    raise ValueError(
                        "NativeLayerCoreSource.layers must retain source-face ancestry for every lateral physical interval; got an exterior lateral layer column lacks an source cap-edge source-face owner"
                    )
                owner = edge_owners[root_edge]
                column = int(columns[position])
                claim = claims.setdefault((column, root_edge), (owner, set()))
                if claim[0] != owner or int(intervals[position]) in claim[1]:
                    raise ValueError(
                        "NativeLayerCoreSource.layers must retain source-face ancestry for every lateral physical interval; got source lateral column intervals have conflicting or duplicate source-face ancestry"
                    )
                claim[1].add(int(intervals[position]))
                output[key] = owner
                work.charge(1)
    for _, carried in claims.values():
        if carried != set(range(max(carried) + 1)):
            raise ValueError(
                "NativeLayerCoreSource.layers must retain source-face ancestry for every lateral physical interval; got a lateral source trace omitted a recorded physical column interval"
            )
    return LateralFaceSources(output)


@checked
def layer_source_face_ancestry(
    source: sources_owner.NativeLayerCoreSource,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> dict[tuple[int, ...], int]:
    """source face identity by combined-mesh vertex-row key."""

    mapping, _, polygons, _ = layer_topology.prepare_identity(
        source.layers,
        source.complex,
        source.vertex_layer_ids,
        source.cap_polygon_ids,
        work=work,
    )
    owners = source.complex.polygon_facets
    if source.core_facet_source_ids is not None:
        owners = source.core_facet_source_ids[owners]
    work.charge(polygons.shape[0])
    authored = {
        tuple(sorted(mapping[row].tolist())): int(owner)
        for row, owner in zip(polygons, owners, strict=True)
        if owner >= 0
    }
    wall_sources = _wall_face_sources(source)
    for block in source.layers.source_wall.blocks:
        for row, identifier in zip(
            np.asarray(block.vertices, dtype=np.int64),
            np.asarray(block.global_ids, dtype=np.int64),
            strict=True,
        ):
            work.charge(1)
            mapped = np.asarray(source.layers.wall_vertices, dtype=np.int64)[row]
            if np.all(mapped >= 0) and int(identifier) in wall_sources:
                key = tuple(sorted(mapped.tolist()))
                owner = wall_sources[int(identifier)]
                previous = authored.get(key)
                if previous is not None and previous != owner:
                    raise ValueError(
                        "NativeLayerCoreSource.layers must bind each physical face to one source stratum; got one physical face cannot claim conflicting source strata"
                    )
                authored[key] = owner
    if source.core_facet_source_ids is not None:
        for key, owner in _layer_lateral_face_sources(
            source, mapping, polygons, owners, work
        ).sources.items():
            previous = authored.get(key)
            if previous is not None and previous != owner:
                raise ValueError(
                    "NativeLayerCoreSource.layers must bind each physical face to one source stratum; got one lateral physical face cannot claim conflicting source strata"
                )
            authored[key] = owner
    return authored


@checked
def compose_layer_controls(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    construction: layer_topology.LayerCoreConstruction,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> tuple[
    tuple[_dependency_phydrax_meshing__organization.MeshZone, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...],
]:

    mesh = construction.mesh
    controls = {control.region_name: control for control in specification.region_controls}
    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    work.charge(len(construction.zones))
    zones = tuple(
        _dependency_phydrax_meshing__organization.MeshZone(
            zone.name,
            _dependency_phydrax_meshing__organization.MeshZoneRole.REGION,
            zone.scope,
            material_id=controls[zone.name].material_id
            if zone.name in controls
            else seeds[zone.name].material_id
            if zone.name in seeds
            else None,
            region_role=controls[zone.name].role
            if zone.name in controls
            else seeds[zone.name].role
            if zone.name in seeds
            else None,
        )
        for zone in construction.zones
    )
    if not specification.patch_controls:
        return zones, construction.patches
    authored = layer_source_face_ancestry(source, work=work)
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    work.charge(2 * cells.size + mesh.coordinates.shape[0] + mesh.entity_set(2).count)
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    regions = np.empty(cells.size, dtype=np.int64)
    regions[np.searchsorted(cells, block_ids)] = construction.cell_regions
    incidents = layer_topology.faces(mesh, regions, work=work)
    face_ids = dict(
        zip(
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                mesh, 2
            ),
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    patches = list(construction.patches)
    for control in specification.patch_controls:
        work.charge(len(authored))
        selected = set(np.asarray(control.scope.entity_ids, dtype=np.int64).tolist())
        keys = [
            key
            for key, owner in authored.items()
            if owner in selected and key in incidents
        ]
        represented = {authored[key] for key in keys}
        if control.required and represented != selected:
            raise ValueError(
                "VolumeMeshingSpec.patch_controls must retain requested source facets, material adjacency, and unique patch names; got a required source patch lost an declared wall or remaining PLC facet"
            )
        for key in keys:
            work.charge(1)
            adjacent = {source.region_ids[face.region] for face in incidents[key]}
            if adjacent != set(control.adjacent_region_names):
                raise ValueError(
                    "VolumeMeshingSpec.patch_controls must retain requested source facets, material adjacency, and unique patch names; got a source patch contradicts independently measured layer/core material adjacency"
                )
        if keys:
            if any(patch.name == control.name for patch in patches):
                raise ValueError(
                    "VolumeMeshingSpec.patch_controls must retain requested source facets, material adjacency, and unique patch names; got requested patch name collides with an immutable construction patch"
                )
            patches.append(
                _dependency_phydrax_meshing__organization.MeshPatch(
                    control.name,
                    mesh_entity_scope(
                        mesh,
                        2,
                        np.asarray([face_ids[key] for key in keys], dtype=np.int64),
                    ),
                    connected=False,
                )
            )
    return zones, tuple(patches)


__all__ = ["layer_control_issues", "compose_layer_controls"]
