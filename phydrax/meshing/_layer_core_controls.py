#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Retained source region and patch controls on the full hybrid volume."""

from __future__ import annotations

from typing import final

import numpy as np

import phydrax.discretization as discretization_owner
import phydrax.geometry._meshing_domain as meshing_domain_owner
import phydrax.meshing._canonical as canonical_owner
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._organization as organization_owner
import phydrax.meshing.providers._native_sources as sources_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._contracts import VolumeMeshingSpec


@checked
def layer_control_issues(
    source: sources_owner.NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Report source controls that cannot be enforced without changing fixed layers."""
    issues: list[str] = []
    controls = {}
    source_region_entity_ids = dict(
        zip(source.region_ids, source.region_entity_ids, strict=True)
    )
    for control in specification.region_controls:
        if (
            control.scope.entity_dimension != 3
            or control.scope.entity_set_id != source.scope_entity_set_id(3)
            or control.region_name not in source_region_entity_ids
            or not np.array_equal(
                np.asarray(control.scope.entity_ids),
                np.asarray(
                    [source_region_entity_ids[control.region_name]], dtype=np.int64
                ),
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
    boundary_ids = np.asarray(specification.boundary_scope.entity_ids, dtype=np.int64)
    for patch in specification.patch_controls:
        patch_ids = np.asarray(patch.scope.entity_ids, dtype=np.int64)
        if (
            (
                patch.scope.source_id,
                patch.scope.source_revision,
                patch.scope.entity_kind,
                patch.scope.entity_set_id,
            )
            != (
                specification.boundary_scope.source_id,
                specification.boundary_scope.source_revision,
                specification.boundary_scope.entity_kind,
                specification.boundary_scope.entity_set_id,
            )
            or patch.scope.entity_dimension != 2
            or np.setdiff1d(patch_ids, boundary_ids).size
        ):
            issues.append("patch controls bound to authoritative supplied source faces")
        if not set(patch.adjacent_region_names) <= set(source.region_ids):
            issues.append("patch controls naming authoritative layer/core regions")
    return issues


@checked
def core_facet_scope_ids(source: sources_owner.NativeLayerCoreSource, /) -> np.ndarray:
    """Map stable PLC facet source IDs to exact original source-scope IDs."""
    if source.source_boundary_scope is None:
        return np.asarray(source.complex.facet_entity_ids, dtype=np.int64)
    domain = source.source_domain
    if not isinstance(domain, meshing_domain_owner.MeshingDomain):
        raise ValueError(
            "Original core facet controls require an occurrence-aware MeshingDomain."
        )
    identities = source.core_facet_source_ids
    paths = source.core_facet_source_occurrence_paths
    if (
        len(identities) != source.complex.facet_count
        or len(paths) != source.complex.facet_count
    ):
        raise ValueError(
            "PLC facet source identity and occurrence banks must cover every facet."
        )
    scope_ids = np.asarray(domain.scope_entity_ids(2), dtype=np.int64)
    result = np.full((len(identities),), -1, dtype=np.int64)
    for facet, (identity, path) in enumerate(zip(identities, paths, strict=True)):
        if identity is None:
            if path is not None:
                raise ValueError(
                    "An unbound PLC facet cannot declare a source occurrence."
                )
            continue
        if path is None:
            raise ValueError("A bound PLC facet must retain its source occurrence path.")
        slot = domain.resolve_entity_id(2, identity, occurrence_path=path)
        if slot < 0 or slot >= scope_ids.size:
            raise ValueError(
                "A PLC facet resolved outside the retained source-domain inventory."
            )
        result[facet] = scope_ids[slot]
    allowed = np.asarray(source.source_boundary_scope.entity_ids, dtype=np.int64)
    if np.setdiff1d(result[result >= 0], allowed).size:
        raise ValueError(
            "PLC facet source identities escaped the retained boundary scope."
        )
    return result


def _wall_face_sources(source: sources_owner.NativeLayerCoreSource, /) -> dict[int, int]:
    """Map exact front-row source-wall faces to their authored scope IDs."""
    layer = source.layers
    kinds = tuple(layer.front_row_source_entity_kinds)
    identifiers = np.asarray(layer.front_row_source_global_ids, dtype=np.int64)
    if identifiers.shape != (len(kinds),):
        raise ValueError("Front-row source face identities have incompatible extents.")
    wall_face_ids = set(
        np.asarray(layer.source_wall.entity_set(2).entity_ids, dtype=np.int64).tolist()
    )
    if source.source_boundary_scope is None:
        allowed = set(
            np.asarray(source.complex.facet_entity_ids, dtype=np.int64).tolist()
        )
    else:
        allowed = set(
            np.asarray(source.source_boundary_scope.entity_ids, dtype=np.int64).tolist()
        )
    result: dict[int, int] = {}
    for raw_identifier, kind in zip(identifiers.tolist(), kinds, strict=True):
        if kind != "face":
            continue
        identifier = int(raw_identifier)
        if identifier not in wall_face_ids:
            raise ValueError(
                "A front-row face is absent from the exact source wall mesh."
            )
        if identifier not in allowed:
            if source.source_boundary_scope is None:
                continue
            raise ValueError("A source wall face escaped its retained boundary scope.")
        previous = result.setdefault(identifier, identifier)
        if previous != identifier:
            raise ValueError("One source wall face has conflicting front-row ancestry.")
    return result


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
) -> LateralColumnRoots:
    """Trace directed layer incidence to unique source wall vertices."""
    layers = source.layers
    wall = np.asarray(layers.wall_vertices, dtype=np.int64)
    roots = {int(vertex): int(vertex) for vertex in wall if vertex >= 0}
    predecessors: dict[int, int] = {}
    for block in layers.mesh.blocks:
        if block.cell_kind not in ("prism", "hexahedron"):
            raise ValueError(
                "BoundaryLayerMesh must retain directed prism or hexahedron column incidence."
            )
        corners = block.vertices.shape[1] // 2
        for row in np.asarray(block.vertices, dtype=np.int64).tolist():
            for bottom, top in zip(row[:corners], row[corners:], strict=True):
                if predecessors.setdefault(top, bottom) != bottom:
                    raise ValueError(
                        "A layer vertex has conflicting source wall-column predecessors."
                    )
    for vertex in predecessors:
        current = vertex
        chain: list[int] = []
        seen: set[int] = set()
        while current not in roots:
            if current in seen or current not in predecessors:
                raise ValueError(
                    "A lateral layer trace lacks unique source wall-column ancestry."
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
    /,
) -> LateralFaceOwners:
    """Bind source-face ownership along boundary cap-edge orbits."""
    cap_edges: dict[tuple[int, int], int] = {}
    for polygon in mapping[polygons[source.cap_polygon_ids]].tolist():
        for first, last in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
            key = tuple(sorted((first, last)))
            cap_edges[key] = cap_edges.get(key, 0) + 1
    boundary_edges = {key for key, count in cap_edges.items() if count == 1}
    edge_owners: dict[tuple[int, int], int] = {}
    for polygon, owner in zip(mapping[polygons].tolist(), owners.tolist(), strict=True):
        if owner < 0:
            continue
        for first, last in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
            edge = tuple(sorted((first, last)))
            if edge not in boundary_edges:
                continue
            root_edge = (min(roots[first], roots[last]), max(roots[first], roots[last]))
            if edge_owners.setdefault(root_edge, owner) != owner:
                raise ValueError(
                    "One source cap edge cannot claim conflicting lateral source faces."
                )
    return LateralFaceOwners(edge_owners)


def _exterior_lateral_edge(
    vertices: np.ndarray,
    facet: tuple[int, ...],
    corners: int,
    faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    /,
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
        raise ValueError("An exterior lateral facet lacks its realized column edge.")
    return key, (bottom[0], bottom[1])


def _layer_lateral_face_sources(
    source: sources_owner.NativeLayerCoreSource,
    mapping: np.ndarray,
    polygons: np.ndarray,
    owners: np.ndarray,
    /,
) -> LateralFaceSources:
    """Carry exact source-face ownership along realized reference-cell columns."""
    layers = source.layers
    roots = _prepare_lateral_column_roots(source).roots
    edge_owners = _prepare_lateral_face_owners(
        source, mapping, polygons, owners, roots
    ).edge_owners
    layer_regions = np.asarray(source.layer_regions, dtype=np.int64)
    faces = layer_topology.faces(layers.mesh, layer_regions)
    cells = np.sort(np.asarray(layers.mesh.entity_set(3).entity_ids, dtype=np.int64))
    intervals = np.asarray(layers.layer_index, dtype=np.int64)
    columns = np.asarray(layers.column_index, dtype=np.int64)
    claims: dict[tuple[int, tuple[int, int]], tuple[int, set[int]]] = {}
    output: dict[tuple[int, ...], int] = {}
    for block in layers.mesh.blocks:
        corners = block.vertices.shape[1] // 2
        topology = discretization_owner.reference_cell_topology(block.cell_kind)
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
                        "An exterior lateral layer column lacks a source cap-edge owner."
                    )
                owner = edge_owners[root_edge]
                column = int(columns[position])
                claim = claims.setdefault((column, root_edge), (owner, set()))
                if claim[0] != owner or int(intervals[position]) in claim[1]:
                    raise ValueError(
                        "Source lateral column intervals have conflicting or duplicate face ancestry."
                    )
                claim[1].add(int(intervals[position]))
                output[key] = owner
    for _, carried in claims.values():
        if carried != set(range(max(carried) + 1)):
            raise ValueError("A lateral source trace omitted a physical column interval.")
    return LateralFaceSources(output)


@checked
def layer_source_face_ancestry(
    source: sources_owner.NativeLayerCoreSource, /
) -> dict[tuple[int, ...], int]:
    """Map combined mesh face keys to original source boundary entity IDs."""
    mapping, _, polygons, _ = layer_topology.prepare_identity(
        source.layers,
        source.complex,
        source.vertex_layer_ids,
        source.cap_polygon_ids,
    )
    owners = core_facet_scope_ids(source)[
        np.asarray(source.complex.polygon_facets, dtype=np.int64)
    ]
    authored = {
        tuple(sorted(mapping[row].tolist())): int(owner)
        for row, owner in zip(polygons, owners, strict=True)
        if owner >= 0
    }
    wall_sources = _wall_face_sources(source)
    wall_vertices = np.asarray(source.layers.wall_vertices, dtype=np.int64)
    for block in source.layers.source_wall.blocks:
        for row, identifier in zip(
            np.asarray(block.vertices, dtype=np.int64),
            np.asarray(block.global_ids, dtype=np.int64),
            strict=True,
        ):
            mapped = wall_vertices[row]
            if np.all(mapped >= 0) and int(identifier) in wall_sources:
                key = tuple(sorted(mapped.tolist()))
                owner = wall_sources[int(identifier)]
                previous = authored.get(key)
                if previous is not None and previous != owner:
                    raise ValueError(
                        "One physical face cannot claim conflicting source strata."
                    )
                authored[key] = owner
    for key, owner in _layer_lateral_face_sources(
        source, mapping, polygons, owners
    ).sources.items():
        previous = authored.get(key)
        if previous is not None and previous != owner:
            raise ValueError(
                "One lateral physical face cannot claim conflicting source strata."
            )
        authored[key] = owner
    return authored


@checked
def compose_layer_controls(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    construction: layer_topology.LayerCoreConstruction,
    /,
) -> tuple[
    tuple[organization_owner.MeshZone, ...],
    tuple[organization_owner.MeshPatch, ...],
]:
    """Apply source material metadata and patch controls after full composition."""
    mesh = construction.mesh
    controls = {control.region_name: control for control in specification.region_controls}
    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    zones = tuple(
        organization_owner.MeshZone(
            zone.name,
            organization_owner.MeshZoneRole.REGION,
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
    authored = layer_source_face_ancestry(source)
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    regions = np.asarray(construction.cell_regions, dtype=np.int64)
    if regions.shape != cells.shape:
        raise ValueError("Combined region identity must cover every source patch cell.")
    incidents = layer_topology.faces(mesh, regions)
    face_ids = dict(
        zip(
            canonical_owner._entity_vertex_keys(mesh, 2),
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    patches = list(construction.patches)
    for control in specification.patch_controls:
        selected = set(np.asarray(control.scope.entity_ids, dtype=np.int64).tolist())
        keys = [
            key
            for key, owner in authored.items()
            if owner in selected and key in incidents
        ]
        represented = {authored[key] for key in keys}
        if control.required and represented != selected:
            raise ValueError(
                "A required source patch lost a declared wall or remaining PLC facet."
            )
        for key in keys:
            adjacent = {source.region_ids[face.region] for face in incidents[key]}
            if adjacent != set(control.adjacent_region_names):
                raise ValueError(
                    "A source patch contradicts measured layer/core material adjacency."
                )
        if keys:
            if any(patch.name == control.name for patch in patches):
                raise ValueError(
                    "A requested patch name collides with an immutable construction patch."
                )
            patches.append(
                organization_owner.MeshPatch(
                    control.name,
                    layer_topology.mesh_entity_scope(
                        mesh,
                        2,
                        np.asarray([face_ids[key] for key in keys], dtype=np.int64),
                    ),
                    connected=False,
                )
            )
    return zones, tuple(patches)


__all__ = [
    "compose_layer_controls",
    "core_facet_scope_ids",
    "layer_control_issues",
    "layer_source_face_ancestry",
]
