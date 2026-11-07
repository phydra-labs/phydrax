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
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..typing import checked
from ._contracts import VolumeMeshingSpec
from ._controls import BoundaryLayerControl, BoundaryLayerRoute
from ._scope import MeshingEntityKind, MeshingScope


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
    scope_by_key = {
        domain.source_entity_key_for_scope(2, int(identifier)): int(identifier)
        for identifier in scope_ids
    }
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
        domain.resolve_entity_id(2, identity, occurrence_path=path)
        scope_id = scope_by_key.get((identity, tuple(path)))
        if scope_id is None:
            raise ValueError("A PLC facet escaped the retained source-domain inventory.")
        result[facet] = scope_id
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
    charts = source.source_wall_charts
    chart_sources: dict[int, int] = {}
    if charts is not None:
        charts.require_root(layer.source_wall, charts.geometry)
        scope_by_key = {
            charts.domain.source_entity_key_for_scope(2, int(identifier)): int(identifier)
            for identifier in charts.domain.scope_entity_ids(2)
        }
        chart_sources = {
            int(cell): scope_by_key[(
                charts.domain.entity_id(2, int(patch)),
                tuple(charts.domain.source_occurrences[2][int(patch)]),
            )]
            for cell, patch in zip(charts.cell_global_ids, charts.root_patches, strict=True)
        }
    if source.source_boundary_scope is None:
        allowed = set(
            np.asarray(
                source.complex.facet_entity_ids
                if charts is None else charts.domain.scope_entity_ids(2),
                dtype=np.int64,
            ).tolist()
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
        owner = chart_sources.get(identifier) if charts is not None else identifier
        if owner is None or owner not in allowed:
            if source.source_boundary_scope is None and charts is None:
                continue
            raise ValueError("A source wall face escaped its retained boundary scope.")
        previous = result.setdefault(identifier, owner)
        if previous != owner:
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
    cap_polygons = set(np.asarray(source.cap_polygon_ids, dtype=np.int64).tolist())
    for index, (polygon, owner) in enumerate(
        zip(mapping[polygons].tolist(), owners.tolist(), strict=True)
    ):
        if index in cap_polygons or owner < 0:
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


def _conforming_cap_control_keys(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_topology.LayerCoreConstruction,
    incidents: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    /,
) -> tuple[set[int], dict[tuple[int, ...], set[int]]]:
    """Prove the complete original cap and realized interface have one exact cover."""
    from fractions import Fraction

    from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from ._layer_core_subdivision import require_overlap_source_provenance
    from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions

    carrier = construction.overlap_exact_geometry
    if not isinstance(carrier, ExactPlcCellGeometrySource):
        raise ValueError("Conforming cap controls require exact source-bound geometry.")
    require_overlap_source_provenance(source, construction)
    polygon_facets = np.asarray(source.complex.polygon_facets, dtype=np.int64)
    cap_ids = np.asarray(source.cap_polygon_ids, dtype=np.int64)
    scopes = core_facet_scope_ids(source)
    selected = {int(scopes[int(polygon_facets[int(row)])]) for row in cap_ids}
    if not selected or any(value < 0 for value in selected):
        raise ValueError("Every original cap facet needs its declared source scope.")
    offsets = np.asarray(source.complex.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(source.complex.polygon_vertices, dtype=np.int64)
    source_points = np.asarray(source.complex.vertices, dtype=np.float64)
    original = []
    for polygon in cap_ids:
        row = vertices[offsets[int(polygon)]:offsets[int(polygon) + 1]]
        if row.shape != (3,):
            raise ValueError("Conforming cap controls require original triangular facets.")
        original.append(tuple(
            tuple(Fraction.from_float(float(value)) for value in source_points[int(vertex)])
            for vertex in row
        ))
    ids = np.sort(np.asarray(construction.mesh.entity_set(3).entity_ids, dtype=np.int64))
    core = set(np.asarray(construction.core_cell_ids, dtype=np.int64).tolist())
    keys = []
    target = []
    for key, owners in incidents.items():
        if len(owners) != 2:
            continue
        is_core = tuple(int(ids[owner.cell]) in core for owner in owners)
        if is_core[0] == is_core[1]:
            continue
        if len(key) != 3:
            raise ValueError("Conforming material-contact faces must be triangular.")
        keys.append(key)
        target.append(tuple(carrier.exact_coordinates[int(vertex)] for vertex in key))
    pieces = exact_coplanar_triangle_partitions(tuple(original), tuple(target))
    owners_by_target: dict[tuple[int, ...], set[int]] = {key: set() for key in keys}
    source_scopes = [int(scopes[int(polygon_facets[int(row)])]) for row in cap_ids]
    for piece in pieces:
        owners_by_target[keys[piece.target_row]].add(source_scopes[piece.source_row])
    if any(not value for value in owners_by_target.values()):
        raise ValueError("Every conforming cap face must bind original source facets.")
    return selected, owners_by_target


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
    authored = {} if source.conforming_execution else layer_source_face_ancestry(source)
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
    cap_scope: set[int] = set()
    cap_owners: dict[tuple[int, ...], set[int]] = {}
    if source.conforming_execution:
        cap_scope, cap_owners = _conforming_cap_control_keys(source, construction, incidents)
    patches = list(construction.patches)
    for control in specification.patch_controls:
        selected = set(np.asarray(control.scope.entity_ids, dtype=np.int64).tolist())
        if source.conforming_execution and selected <= cap_scope:
            if any(owners & selected and not owners <= selected for owners in cap_owners.values()):
                raise ValueError("A source patch cuts a realized face and cannot be a whole-face MeshPatch.")
            keys = [key for key, owners in cap_owners.items() if owners <= selected]
            represented = set().union(*(cap_owners[key] for key in keys)) if keys else set()
        else:
            if source.conforming_execution:
                authored = layer_source_face_ancestry(source)
            keys = [
                key for key, owner in authored.items()
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


@checked
def wall_layer_control(
    control: BoundaryLayerControl, charts: SurfaceSourceCharts, /
) -> BoundaryLayerControl:
    """Resolve a source-face layer control to the cells of one charted wall mesh.

    Each chart binds one wall cell to exactly one source face, so the selected
    cells are the exact preimage of ``control.wall_scope``. Schedule and policies
    are unchanged; only the wall scope moves from source faces to mesh cells.
    """
    charts.require_current()
    domain = charts.domain
    wall = charts.mesh
    scope = control.wall_scope
    volume = control.volume_scope
    if control.route is not BoundaryLayerRoute.ADVANCING:
        raise ValueError("Charted source walls realize ADVANCING layer controls only.")
    if (
        scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 2
        or (scope.source_id, scope.source_revision)
        != (domain.source_id, domain.source_revision)
        or scope.entity_set_id != domain.scope_entity_set_id(2)
    ):
        raise ValueError("The layer wall scope must name faces of the charted source.")
    if (
        volume is None
        or volume.entity_set_id != domain.scope_entity_set_id(3)
        or not np.array_equal(
            np.asarray(volume.entity_ids, dtype=np.int64),
            np.sort(np.asarray(domain.scope_entity_ids(3), dtype=np.int64)),
        )
    ):
        raise ValueError(
            "Charted source walls grow layers into the whole source solid only."
        )
    patches = {
        domain.resolve_entity_id(2, entity_id, occurrence_path=occurrence)
        for entity_id, occurrence in (
            domain.source_entity_key_for_scope(2, int(identifier))
            for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist()
        )
    }
    cells = np.asarray(charts.cell_global_ids, dtype=np.int64)
    selected = cells[
        np.isin(np.asarray(charts.root_patches, dtype=np.int64), list(patches))
    ]
    if not selected.size:
        raise ValueError("The layer wall faces bind no charted wall cells.")
    return BoundaryLayerControl(
        MeshingScope(
            wall.mesh_id,
            wall.numeric_version,
            MeshingEntityKind.MESH,
            2,
            wall.entity_set(2).entity_set_id,
            np.sort(selected),
        ),
        control.schedule,
        route=BoundaryLayerRoute.ADVANCING,
        collision=control.collision,
        corner=control.corner,
        feature_angle=control.feature_angle,
        minimum_thickness_fraction=control.minimum_thickness_fraction,
        growth_rate_bounds=control.growth_rate_bounds,
        maximum_corner_stretch=control.maximum_corner_stretch,
        smoothing_iterations=control.smoothing_iterations,
        core_maximum_size=control.core_maximum_size,
    )


def _mapped_wall_source_cells(
    source: sources_owner.NativeLayerCoreSource, /
) -> tuple[
    sources_owner.NativePlcSource | sources_owner.NativeMappedHexSource,
    dict[int, list[int]],
]:
    """Bind original wall cells to full-source facets by explicit root IDs."""
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource

    query = source.fidelity_source
    domain = source.mapped_domain
    if not isinstance(query, MappedReferenceBoundarySource) or domain is None:
        raise ValueError("Mapped wall controls require their original mapped boundary query.")
    authority = query.authority
    if not isinstance(
        authority, (sources_owner.NativePlcSource, sources_owner.NativeMappedHexSource)
    ):
        raise TypeError("Mapped wall controls require their original full PLC authority.")
    query.require_current()
    sources_owner._mapped_layer_core_facet_relation(source, authority)
    relation = authority.root_relation
    root_ids = domain.reference_domain_vertex_root_ids
    if relation is None or root_ids is None:
        raise ValueError("Mapped wall controls require declared source-root vertex ancestry.")
    complex_ = authority.complex
    polygon_offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    full_roots = np.asarray(relation.vertex_root_global_ids, dtype=np.int64)
    facet_entities = np.asarray(complex_.facet_entity_ids, dtype=np.int64)
    selected_keys = set(query.selected_source_keys(2))
    facet_by_triangle: dict[tuple[int, ...], int] = {}
    for row, facet_value in enumerate(polygon_facets):
        facet = int(facet_value)
        key = (complex_.facet_ids[facet], complex_.facet_occurrence_paths[facet])
        if key not in selected_keys:
            continue
        vertices = polygon_vertices[polygon_offsets[row] : polygon_offsets[row + 1]]
        if vertices.size != 3:
            raise ValueError("Mapped source wall facets must be triangles.")
        triangle = tuple(sorted(int(value) for value in full_roots[vertices]))
        if triangle in facet_by_triangle:
            raise ValueError("Mapped source wall has ambiguous full-source facets.")
        facet_by_triangle[triangle] = int(facet_entities[facet])
    wall = source.layers.source_wall
    wall_rows = np.asarray(source.layers.wall_vertices, dtype=np.int64)
    if wall_rows.shape != (wall.coordinates.shape[0],):
        raise ValueError("Mapped wall source ancestry does not cover wall vertices.")
    domain_roots = np.asarray(root_ids, dtype=np.int64)
    result: dict[int, list[int]] = {}
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Mapped source wall controls require triangular wall cells.")
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True):
            mapped = wall_rows[np.asarray(vertices, dtype=np.int64)]
            if np.any(mapped < 0) or np.any(mapped >= domain_roots.size):
                continue
            triangle = tuple(sorted(int(value) for value in domain_roots[mapped]))
            facet = facet_by_triangle.get(triangle)
            if facet is None:
                raise ValueError("A source wall cell lacks its exact full-source facet.")
            result.setdefault(facet, []).append(int(identifier))
    if not result:
        raise ValueError("Mapped source contains no bound wall cells.")
    return authority, result


def mapped_wall_source_scope(
    source: sources_owner.NativeLayerCoreSource, /
) -> MeshingScope:
    """Return the full-source face scope exactly represented by the layer wall."""
    authority, cells = _mapped_wall_source_cells(source)
    return MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        authority.scope_entity_set_id(2),
        np.asarray(sorted(cells), dtype=np.int64),
    )


def _mapped_wall_layer_control(
    source: sources_owner.NativeLayerCoreSource, control: BoundaryLayerControl, /
) -> BoundaryLayerControl:
    authority, cells = _mapped_wall_source_cells(source)
    scope = control.wall_scope
    volume = control.volume_scope
    if (
        control.route is not BoundaryLayerRoute.ADVANCING
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 2
        or (scope.source_id, scope.source_revision) != (source.source_id, source.source_revision)
        or scope.entity_set_id != authority.scope_entity_set_id(2)
    ):
        raise ValueError("Mapped wall control must bind the original full-source faces.")
    if (
        volume is None
        or volume.entity_set_id != source.scope_entity_set_id(3)
        or not np.array_equal(
            np.sort(np.asarray(volume.entity_ids, dtype=np.int64)),
            np.sort(np.asarray(source.region_entity_ids, dtype=np.int64)[
                np.unique(np.asarray(source.layer_regions, dtype=np.int64))
            ]),
        )
    ):
        raise ValueError("Mapped wall control must bind its actual layer material regions.")
    selected_ids = np.asarray(scope.entity_ids, dtype=np.int64).tolist()
    if not selected_ids or any(int(identifier) not in cells for identifier in selected_ids):
        raise ValueError("Mapped wall scope selects faces outside the authored layer wall.")
    selected = np.asarray(
        sorted(cell for identifier in selected_ids for cell in cells[int(identifier)]),
        dtype=np.int64,
    )
    wall = source.layers.source_wall
    return BoundaryLayerControl(
        MeshingScope(
            wall.mesh_id, wall.numeric_version, MeshingEntityKind.MESH, 2,
            wall.entity_set(2).entity_set_id, selected,
        ),
        control.schedule,
        route=BoundaryLayerRoute.ADVANCING,
        collision=control.collision,
        corner=control.corner,
        feature_angle=control.feature_angle,
        minimum_thickness_fraction=control.minimum_thickness_fraction,
        growth_rate_bounds=control.growth_rate_bounds,
        maximum_corner_stretch=control.maximum_corner_stretch,
        smoothing_iterations=control.smoothing_iterations,
        core_maximum_size=control.core_maximum_size,
    )


@checked
def realized_layer_control(
    source: sources_owner.NativeLayerCoreSource, control: BoundaryLayerControl, /
) -> BoundaryLayerControl:
    """Return the mesh-scoped control an accepted layer realization must carry."""
    if control.wall_scope.entity_kind is MeshingEntityKind.MESH:
        return control
    charts = source.source_wall_charts
    if charts is None:
        if source.mapped_domain is not None:
            return _mapped_wall_layer_control(source, control)
        raise ValueError(
            "Source-face layer controls require the retained charted layer wall."
        )
    charts.require_root(source.layers.source_wall, charts.geometry)
    return wall_layer_control(control, charts)


__all__ = [
    "compose_layer_controls",
    "core_facet_scope_ids",
    "layer_control_issues",
    "layer_source_face_ancestry",
    "mapped_wall_source_scope",
    "realized_layer_control",
    "wall_layer_control",
]
