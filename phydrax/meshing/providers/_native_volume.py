#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare constrained PLC fills for canonical native acceptance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import combinations
from typing import final

import equinox as eqx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._topology import _row_order, row_positions, sorted_unique_rows
from ...geometry._affine_plc_boundary import AffinePlcBoundarySource
from ...geometry._affine_plc_domain import AffinePlcDomain
from ...geometry._affine_plc_geometry import map_affine_plc_mesh
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ...geometry._planar_coverage import PlcSourcePointWitness
from ...geometry._source_boundary_records import SourceBoundaryChartWitness
from ...typing import checked
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import VolumeFillStrategy, VolumeMeshingSpec
from .._controls import FeatureKind, RegionControl, RegionSeed
from .._input_forced_quality import (
    input_forced_dihedral_exemptions,
    input_forced_radius_edge_exemptions,
)
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._organization import (
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionBoundaryEvidence,
)
from .._result import MeshingComplianceReport
from .._scope import _SourceEntityScopes, MeshingEntityKind, MeshingScope
from .._sizing import (
    edge_length_measures,
    S3_BOUNDARY_EDGE_SIZE,
    size_statistic,
    SizeControlStrength,
    UniformSizeControl,
)
from .._volume_generation import (
    _shape_targets_gated,
    _soft_shape_target_evidence,
    _split_shape_target_rows,
    generate_plc_volume,
    NativeVolumeSchedule,
    object_rows,
    VolumeConstruction,
)
from ._native_affine_plc_witness import affine_plc_boundary_chart_witness
from ._native_options import NativeMeshingOptions
from ._native_plc_geometry import (
    mapped_plc_boundary_chart_witness,
    mapped_plc_edge_chart_witness,
    mapped_plc_point_witness,
    mapped_plc_reference_map,
    physical_plc_tetrahedra,
    plc_curve_chart_witness,
    simplex_volume_entities,
)
from ._native_sources import NativePlcSource


def _source_binding_issues(
    source: NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    scope = specification.boundary_scope
    scope_ids = np.asarray(scope.entity_ids, dtype=np.int64)
    expected_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
    if (
        scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 2
        or scope.entity_set_id != source.scope_entity_set_id(2)
        or not np.array_equal(np.sort(scope_ids), np.sort(expected_ids))
    ):
        return ["a boundary scope of every stable PLC facet on this source revision"]
    return []


def _rows_from_stable_ids(
    entity_ids: np.ndarray, stable_ids: tuple[int, ...], name: str, /
) -> np.ndarray:
    rows = {int(identifier): row for row, identifier in enumerate(stable_ids)}
    values = np.asarray(entity_ids, dtype=np.int64)
    if any(int(identifier) not in rows for identifier in values):
        raise ValueError(f"{name} scope contains an undeclared stable source entity ID.")
    return np.asarray([rows[int(identifier)] for identifier in values], dtype=np.int64)


def _target_support(
    specification: VolumeMeshingSpec,
    /,
    *,
    maximum_geometry_order: int = 1,
) -> list[str]:
    target = specification.target
    families = target.cell_families
    issues: list[str] = []
    if set((*families.required, *families.preferred)) != {"tetrahedron"}:
        issues.append("a tetrahedral volume target")
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or not 1 <= target.geometry_order <= maximum_geometry_order
        or families.allowed_transitions
        or families.allow_mixed
    ):
        issues.append(
            "affine tetrahedra in ambient dimension three"
            if maximum_geometry_order == 1
            else f"tetrahedra of geometry order at most {maximum_geometry_order} "
            "in ambient dimension three"
        )
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("the simplex volume-fill strategy")
    return issues


def _control_support(
    source: NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    complex_ = source.complex
    issues: list[str] = []
    binding = (
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
    )
    facet_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append("uniform size controls on PLC facets")
            continue
        scope = control.scope
        identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
        if (
            (scope.source_id, scope.source_revision, scope.entity_kind) != binding
            or scope.entity_dimension != 2
            or scope.entity_set_id != source.scope_entity_set_id(2)
            or np.setdiff1d(identifiers, facet_ids).size
        ):
            issues.append("uniform size controls bound to stable PLC facet IDs")
    for feature in specification.protected_features:
        scope = feature.scope
        identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
        if (scope.source_id, scope.source_revision, scope.entity_kind) != binding:
            issues.append("protected features bound to this PLC source revision")
            continue
        match scope.entity_dimension, feature.feature_kind:
            case 0, FeatureKind.CORNER:
                stable_ids = np.asarray(source.scope_entity_ids(0), dtype=np.int64)
            case 1, FeatureKind.CURVE:
                stable_ids = np.asarray(complex_.plc_edge_entity_ids, dtype=np.int64)
            case 2, FeatureKind.SURFACE | FeatureKind.MATERIAL_INTERFACE:
                stable_ids = facet_ids
            case _:
                issues.append("protected PLC vertices, curves, surfaces or interfaces")
                continue
        if (
            scope.entity_set_id != source.scope_entity_set_id(scope.entity_dimension)
            or np.setdiff1d(identifiers, stable_ids).size
        ):
            issues.append(
                "protected features inside the declared stable PLC source strata"
            )
        elif feature.feature_kind is FeatureKind.MATERIAL_INTERFACE:
            rows = _rows_from_stable_ids(identifiers, complex_.facet_entity_ids, "facet")
            if any(
                complex_.facet_regions[int(row), 0] < 0
                or complex_.facet_regions[int(row), 1] < 0
                or complex_.facet_regions[int(row), 0]
                == complex_.facet_regions[int(row), 1]
                for row in rows.tolist()
            ):
                issues.append(
                    "material-interface features selecting only separating PLC facets"
                )
    region_ids = np.asarray(complex_.region_entity_ids, dtype=np.int64)
    controlled_rows: set[int] = set()
    for control in specification.region_controls:
        scope = control.scope
        identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
        if (
            (scope.source_id, scope.source_revision, scope.entity_kind) != binding
            or scope.entity_dimension != 3
            or scope.entity_set_id != source.scope_entity_set_id(3)
            or identifiers.size != 1
            or not control.meshing_enabled
        ):
            issues.append("enabled region controls bound to one stable PLC region ID")
            continue
        try:
            matching_rows = _rows_from_stable_ids(
                identifiers, tuple(int(value) for value in region_ids.tolist()), "region"
            )
        except ValueError:
            issues.append("region controls bound to declared PLC region IDs")
            continue
        if matching_rows.size != 1:
            issues.append("region controls bound to one declared PLC region")
            continue
        row = int(matching_rows[0])
        if complex_.region_ids[row] != control.region_name:
            issues.append("region control names matching its scoped PLC region")
        if row in controlled_rows:
            issues.append("one consistent region control per scoped PLC region")
        controlled_rows.add(row)
    for seed in specification.region_seeds:
        matching_rows = [
            row
            for row, name in enumerate(complex_.region_ids)
            if name == seed.region_name
        ]
        if len(matching_rows) != 1:
            issues.append("region seeds naming one unambiguous declared PLC region")
    for seed in specification.hole_seeds:
        if (
            seed.scope.source_id != source.source_id
            or seed.scope.source_revision != source.source_revision
            or seed.scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or seed.scope.entity_set_id
            != source.scope_entity_set_id(seed.scope.entity_dimension)
        ):
            issues.append("hole seeds bound to this PLC source revision")
    for control in specification.patch_controls:
        identifiers = np.asarray(control.scope.entity_ids, dtype=np.int64)
        if (
            control.scope.source_id != source.source_id
            or control.scope.source_revision != source.source_revision
            or control.scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or control.scope.entity_dimension != 2
            or control.scope.entity_set_id != source.scope_entity_set_id(2)
            or np.setdiff1d(identifiers, facet_ids).size
        ):
            issues.append("patch controls bound to stable PLC facet IDs")
            continue
        rows = _rows_from_stable_ids(identifiers, complex_.facet_entity_ids, "facet")
        for facet in rows.tolist():
            adjacency = {
                complex_.region_ids[int(region)]
                for region in complex_.facet_regions[facet]
                if region >= 0
            }
            if adjacency != set(control.adjacent_region_names):
                issues.append("patch control adjacency matching PLC facet incidence")
    if specification.periodic_constraints:
        issues.append("periodic constraints")
    if specification.layer_controls:
        issues.append("boundary-layer controls")
    return issues


@checked
def volume_support_issues(
    source: NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Return unmet constrained-volume requirements in stable validation order."""
    issues = [
        *_target_support(specification),
        *_source_binding_issues(source, specification),
        *_control_support(source, specification),
    ]
    if source.mapped_domain is not None:
        if source.complex.geometry is None:
            issues.append("the actual mapped PLC source coordinate map")
        if source.root_relation is None:
            issues.append("a source-bound mapped PLC root entity relation")
        tolerance = (
            specification.maximum_deviation
            if specification.maximum_deviation is not None
            else source.maximum_deviation
        )
        if tolerance is None:
            issues.append("an explicit or source-declared mapped PLC deviation bound")
        if (
            specification.maximum_deviation is not None
            and source.maximum_deviation is not None
            and specification.maximum_deviation > source.maximum_deviation
        ):
            issues.append("a mapped PLC deviation bound no weaker than source authority")
    return issues


def _source_region_scopes(
    source: NativePlcSource,
    owner: _SourceEntityScopes,
    /,
) -> tuple[tuple[str, tuple[str, ...], MeshingScope], ...]:
    complex_ = source.complex
    owner_ids = np.asarray(owner.scope_entity_ids(3), dtype=np.int64)
    owner_ids_by_key: dict[tuple[str, tuple[str, ...]], int] = {}
    for identifier in owner_ids.tolist():
        key = owner.source_entity_key_for_scope(3, int(identifier))
        if key in owner_ids_by_key:
            raise ValueError("Source region keys must resolve to unique owner scope IDs.")
        owner_ids_by_key[key] = int(identifier)
    region_ids = np.asarray(complex_.region_entity_ids, dtype=np.int64)
    if region_ids.shape != (len(complex_.region_ids),) or len(
        complex_.region_occurrence_paths
    ) != len(complex_.region_ids):
        raise ValueError("PLC source region identities and occurrence paths must align.")
    records = []
    for row, identifier in enumerate(region_ids.tolist()):
        region_id, occurrence_path = source.source_entity_key_for_scope(
            3, int(identifier)
        )
        expected_path = tuple(complex_.region_occurrence_paths[row])
        if (region_id, occurrence_path) != (complex_.region_ids[row], expected_path):
            raise ValueError("PLC region key differs from its declared occurrence path.")
        try:
            owner_identifier = owner_ids_by_key[(region_id, occurrence_path)]
        except KeyError as error:
            raise ValueError(
                "PLC region occurrence is absent from the authoritative source scope."
            ) from error
        scope = MeshingScope(
            owner.source_id,
            owner.source_revision,
            MeshingEntityKind.GEOMETRY,
            3,
            owner.scope_entity_set_id(3),
            np.asarray((owner_identifier,), dtype=np.int64),
        )
        records.append((region_id, occurrence_path, scope))
    return tuple(records)


def _region_controls_by_row(
    region_scopes: tuple[tuple[str, tuple[str, ...], MeshingScope], ...],
    controls: tuple[RegionControl, ...],
    /,
) -> dict[int, RegionControl]:
    rows_by_scope = {
        scope.scope_id: row for row, (_, _, scope) in enumerate(region_scopes)
    }
    result: dict[int, RegionControl] = {}
    for control in controls:
        row = rows_by_scope.get(control.scope.scope_id)
        if row is None:
            raise ValueError(
                "Region control scope does not identify an authored source region."
            )
        if row in result:
            raise ValueError("A source region may have only one explicit region control.")
        result[row] = control
    return result


def _region_seeds_by_row(
    region_scopes: tuple[tuple[str, tuple[str, ...], MeshingScope], ...],
    seeds: tuple[RegionSeed, ...],
    /,
) -> dict[int, RegionSeed]:
    result: dict[int, RegionSeed] = {}
    for seed in seeds:
        matches = [
            row
            for row, (region_id, _, _) in enumerate(region_scopes)
            if region_id == seed.region_name
        ]
        if len(matches) != 1:
            raise ValueError("A region seed must name one unambiguous source region.")
        row = matches[0]
        if row in result:
            raise ValueError("A source region may have only one region seed.")
        result[row] = seed
    return result


def _region_zones(
    construction: VolumeConstruction,
    region_scopes: tuple[tuple[str, tuple[str, ...], MeshingScope], ...],
    controls: dict[int, RegionControl],
    seeds: dict[int, RegionSeed],
    /,
) -> tuple[tuple[MeshZone, ...], dict[int, str]]:
    mesh = construction.mesh
    cell_set = mesh.entity_set(3)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    if (
        len(construction.region_ids) != len(region_scopes)
        or len(construction.region_occurrence_paths) != len(region_scopes)
        or any(
            (construction.region_ids[row], construction.region_occurrence_paths[row])
            != (region_id, occurrence_path)
            for row, (region_id, occurrence_path, _) in enumerate(region_scopes)
        )
    ):
        raise ValueError(
            "Volume region assignments differ from their source occurrences."
        )
    multiplicity = {
        name: construction.region_ids.count(name) for name in set(construction.region_ids)
    }
    zones: list[MeshZone] = []
    zone_ids: dict[int, str] = {}
    for region, name in enumerate(construction.region_ids):
        source_scope = region_scopes[region][2]
        selected = cell_ids[construction.cell_regions == region]
        if selected.size == 0:
            raise ValueError(f"PLC region {name!r} has no volume cells.")
        control = controls.get(region)
        seed = seeds.get(region)
        role = (
            control.role
            if control is not None
            else seed.role
            if seed is not None
            else None
        )
        material = (
            control.material_id
            if control is not None
            else seed.material_id
            if seed is not None
            else None
        )
        zone_name = name if multiplicity[name] == 1 else f"{name}@{source_scope.scope_id}"
        zone = MeshZone(
            zone_name,
            MeshZoneRole.REGION,
            _mesh_scope(mesh, 3, selected),
            material_id=material,
            region_role=role,
        )
        zones.append(zone)
        zone_ids[region] = zone.zone_id
    return tuple(zones), zone_ids


def _boundary_face_entities(
    mesh: CellMesh, construction: VolumeConstruction, /
) -> np.ndarray:
    _, connectivity_faces = simplex_volume_entities(mesh)
    identifiers = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if connectivity_faces.shape[0] != identifiers.size:
        raise ValueError("Canonical face incidence and face entity IDs are misaligned.")
    rows = _rows_of_simplices(
        connectivity_faces, np.asarray(construction.boundary_faces, dtype=np.int64)
    )
    if np.any(rows < 0):
        raise ValueError(
            "A recovered boundary face is absent from canonical mesh incidence."
        )
    return identifiers[rows]


def _edge_entities(mesh: CellMesh, edges: np.ndarray, /) -> np.ndarray:
    """Entity IDs of the mesh edges with the given vertex pairs."""
    connectivity_edges, _ = simplex_volume_entities(mesh)
    identifiers = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    if connectivity_edges.shape[0] != identifiers.size:
        raise ValueError("Canonical edge incidence and edge entity IDs are misaligned.")
    rows = _rows_of_simplices(connectivity_edges, edges)
    if np.any(rows < 0):
        raise ValueError("A constrained segment is absent from canonical mesh edges.")
    return identifiers[rows]


def _faces_connected(faces: np.ndarray, /) -> bool:
    """Whether the faces form one component through shared vertex pairs."""
    count = faces.shape[0]
    if count <= 1:
        return True
    pairs = np.asarray(tuple(combinations(range(faces.shape[1]), 2)), dtype=np.int64)
    edges = np.sort(faces[:, pairs], axis=2).reshape((-1, 2))
    owners = np.repeat(np.arange(count, dtype=np.int64), pairs.shape[0])
    order = _row_order(edges)
    ordered = edges[order]
    same = np.all(ordered[1:] == ordered[:-1], axis=1)
    first = owners[order][:-1][same]
    second = owners[order][1:][same]
    graph = coo_matrix(
        (np.ones((first.size,), dtype=np.int8), (first, second)), shape=(count, count)
    )
    components, _ = connected_components(graph, directed=False)
    return components == 1


def _patches_and_labels(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    construction: VolumeConstruction,
    zone_ids: dict[int, str],
    region_scopes: tuple[tuple[str, tuple[str, ...], MeshingScope], ...],
    controls: dict[int, RegionControl],
    seeds: dict[int, RegionSeed],
    /,
) -> tuple[
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[RegionBoundaryEvidence, ...],
]:
    mesh = construction.mesh
    boundary_face_ids = _boundary_face_entities(mesh, construction)
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    face_facets = np.asarray(construction.face_source_facets, dtype=np.int64)
    facet_regions = np.asarray(source.complex.facet_regions, dtype=np.int64)
    if facet_regions.shape != (len(construction.facet_ids), 2):
        raise ValueError("PLC facet region incidence must align with source facets.")
    patches: list[MeshPatch] = []
    region_patch_sides: list[list[tuple[str, int]]] = [
        [] for _ in range(len(region_scopes))
    ]
    # Face rows of every facet, ascending, by one stable sort.
    facet_order = np.argsort(face_facets, kind="stable")
    facet_bounds = np.searchsorted(
        face_facets[facet_order],
        np.arange(len(construction.facet_ids) + 1, dtype=np.int64),
    )
    for facet, identity in enumerate(construction.facet_ids):
        selected_rows = facet_order[facet_bounds[facet] : facet_bounds[facet + 1]]
        selected = boundary_face_ids[selected_rows]
        if selected.size == 0:
            raise ValueError(f"PLC facet {identity!r} has no constrained mesh faces.")
        first = int(facet_regions[facet, 0])
        second = int(facet_regions[facet, 1])
        sides = (first, second)
        if any(region < -1 or region >= len(region_scopes) for region in sides):
            raise ValueError("PLC facet incidence names an undeclared source region.")
        source_pair = (
            None if first < 0 else region_scopes[first][0],
            None if second < 0 else region_scopes[second][0],
        )
        source_paths = (
            None if first < 0 else region_scopes[first][1],
            None if second < 0 else region_scopes[second][1],
        )
        adjacent = tuple(
            dict.fromkeys(zone_ids[region] for region in sides if region >= 0)
        )
        path = construction.facet_occurrence_paths[facet]
        patch = MeshPatch(
            f"source-facet:{identity}:{'/'.join(path)}",
            _mesh_scope(mesh, 2, selected),
            connected=_faces_connected(boundary_faces[selected_rows]),
            adjacent_zone_ids=adjacent,
            source_adjacent_region_ids=source_pair,
            source_adjacent_region_occurrence_paths=source_paths,
        )
        patches.append(patch)
        if sides[0] != sides[1]:
            for side_index, region in enumerate(sides):
                if region >= 0:
                    region_patch_sides[region].append(
                        (patch.patch_id, 1 if side_index == 0 else -1)
                    )
    for control in specification.patch_controls:
        selected_facets = _rows_from_stable_ids(
            np.asarray(control.scope.entity_ids, dtype=np.int64),
            source.complex.facet_entity_ids,
            "facet",
        )
        selected_rows = np.flatnonzero(np.isin(face_facets, selected_facets))
        adjacent_rows = np.unique(facet_regions[selected_facets])
        patches.append(
            MeshPatch(
                control.name,
                _mesh_scope(mesh, 2, boundary_face_ids[selected_rows]),
                connected=_faces_connected(boundary_faces[selected_rows]),
                adjacent_zone_ids=tuple(
                    zone_ids[row] for row in adjacent_rows.tolist() if row >= 0
                ),
            )
        )

    face_sides = facet_regions[face_facets]
    distinct = face_sides[:, 0] != face_sides[:, 1]
    interfaces = boundary_face_ids[distinct & np.all(face_sides >= 0, axis=1)]
    sheets = boundary_face_ids[~distinct]
    labels: list[MeshLabel] = []
    region_evidence: list[RegionBoundaryEvidence] = []
    for region, (region_id, occurrence_path, source_scope) in enumerate(region_scopes):
        selected = np.unique(
            boundary_face_ids[distinct & np.any(face_sides == region, axis=1)]
        )
        if selected.size == 0 or not region_patch_sides[region]:
            raise ValueError(
                f"PLC region occurrence {region_id!r} has no oriented boundary patches."
            )
        label = MeshLabel(
            f"region-boundary:{source_scope.scope_id}",
            _mesh_scope(mesh, 2, selected),
        )
        labels.append(label)
        control = controls.get(region)
        seed = seeds.get(region)
        region_evidence.append(
            RegionBoundaryEvidence(
                mesh,
                source_scope,
                region_id,
                None if control is None else control.control_id,
                control.material_id
                if control is not None
                else None
                if seed is None
                else seed.material_id,
                control.role
                if control is not None
                else None
                if seed is None
                else seed.role,
                label,
                tuple(region_patch_sides[region]),
                source_occurrence_path=occurrence_path,
            )
        )
    for name, selected in (
        ("material_interface", interfaces),
        ("internal_sheet", sheets),
    ):
        if selected.size:
            labels.append(MeshLabel(name, _mesh_scope(mesh, 2, selected)))
    explicit = source.complex.segments.shape[0]
    internal = (
        np.asarray(construction.source_segment_complex_rows, dtype=np.int64)[
            np.asarray(construction.segment_source_edges, dtype=np.int64)
        ]
        < explicit
    )
    if np.any(internal):
        labels.append(
            MeshLabel(
                "internal_curve",
                _mesh_scope(
                    mesh,
                    1,
                    _edge_entities(
                        mesh, np.asarray(construction.segments, dtype=np.int64)[internal]
                    ),
                ),
            )
        )
    return tuple(patches), tuple(labels), tuple(region_evidence)


def _tetrahedron_regions(
    construction: VolumeConstruction, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = construction.mesh
    if (
        len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind not in ("simplex:3", "tetrahedron")
        or mesh.blocks[0].arity != 4
    ):
        raise ValueError("Native PLC region associations require one tetrahedron block.")
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    if np.asarray(construction.cell_regions).shape != (cells.shape[0],):
        raise ValueError(
            "Native PLC cell regions must align with tetrahedron block rows."
        )
    edge_rows, face_rows = simplex_volume_entities(mesh)
    return cells, edge_rows, face_rows


def _source_identity_maps(
    source: NativePlcSource, /
) -> tuple[
    dict[tuple[str, tuple[str, ...]], int],
    dict[tuple[str, tuple[str, ...]], int],
    dict[tuple[str, tuple[str, ...]], int],
    dict[tuple[str, tuple[str, ...]], int],
]:
    complex_ = source.complex
    banks = (
        (
            complex_.vertex_ids,
            complex_.vertex_occurrence_paths,
            complex_.vertex_entity_ids,
        ),
        (
            complex_.plc_edge_ids,
            complex_.plc_edge_occurrence_paths,
            complex_.plc_edge_entity_ids,
        ),
        (
            complex_.facet_ids,
            complex_.facet_occurrence_paths,
            complex_.facet_entity_ids,
        ),
        (
            complex_.region_ids,
            complex_.region_occurrence_paths,
            complex_.region_entity_ids,
        ),
    )
    maps: list[dict[tuple[str, tuple[str, ...]], int]] = []
    for identifiers, paths, scope_ids in banks:
        maps.append(
            {
                (identifier, path): int(scope_id)
                for identifier, path, scope_id in zip(
                    identifiers, paths, scope_ids, strict=True
                )
            }
        )
    return maps[0], maps[1], maps[2], maps[3]


def _source_key(
    key_for_scope: Callable[[int, int], tuple[str, tuple[str, ...]]],
    source_maps: tuple[dict[tuple[str, tuple[str, ...]], int], ...],
    dimension: int,
    identifier: str,
    occurrence_path: tuple[str, ...],
    /,
) -> str:
    try:
        entity_id = source_maps[dimension][(identifier, occurrence_path)]
    except (IndexError, KeyError) as error:
        raise ValueError(
            "A PLC source association must resolve one authored ID/occurrence pair."
        ) from error
    key, path = key_for_scope(dimension, entity_id)
    if path != occurrence_path:
        raise ValueError(
            "PLC source scope resolution changed its authored occurrence path."
        )
    return key


def _source_key_table(
    source: NativePlcSource,
    source_maps: tuple[dict[tuple[str, tuple[str, ...]], int], ...],
    dimension: int,
    /,
) -> dict[tuple[str, tuple[str, ...]], str]:
    """Keys of one dimension's authored identities whose scope row keeps the path.

    The table equals :func:`_source_key` on every pair it holds; a pair it
    lacks resolves through :func:`_source_key`, which refuses it.
    """
    identities = tuple(source_maps[dimension])
    bank = np.asarray(source.scope_entity_ids(dimension), dtype=np.int64)
    if not identities or bank.size == 0:
        return {}
    entity_ids = np.fromiter(
        source_maps[dimension].values(), dtype=np.int64, count=len(identities)
    )
    positions = np.minimum(np.searchsorted(bank, entity_ids), bank.size - 1)
    found = bank[positions] == entity_ids
    keys = source.scope_entity_key_banks[dimension]
    return {
        identity: keys[position][0]
        for identity, position, present in zip(
            identities, positions.tolist(), found.tolist(), strict=True
        )
        if present and keys[position][1] == identity[1]
    }


def _face_order_sign(
    source: tuple[int, int, int], target: tuple[int, int, int], /
) -> int:
    """Permutation parity from an oriented recovered triangle to its mesh order."""
    if len(set(source)) != 3 or set(source) != set(target):
        raise ValueError("Recovered and canonical face vertex identities must agree.")
    order = tuple(source.index(vertex) for vertex in target)
    inversions = sum(
        order[left] > order[right] for left, right in ((0, 1), (0, 2), (1, 2))
    )
    return -1 if inversions % 2 else 1


_TRIANGLE_EDGES = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)


def _rows_of_simplices(table: np.ndarray, query: np.ndarray, /) -> np.ndarray:
    """Row of the unique simplices ``table`` with each ``query`` vertex set, or -1."""
    return row_positions(
        np.sort(np.asarray(table, dtype=np.int64), axis=1),
        np.sort(np.asarray(query, dtype=np.int64), axis=1),
    )


def _cell_entity_rows(
    mesh: CellMesh, cells: np.ndarray, entities: np.ndarray, degree: int, /
) -> np.ndarray:
    """Row of ``entities`` of every cell's ``degree`` faces, cell-major, or -1.

    A simplicial connectivity's cell routes are taken once every routed row is
    checked to be exactly the cell's sorted face; other connectivities search.
    """
    local = np.asarray(tuple(combinations(range(4), degree + 1)), dtype=np.int64)
    faces = np.sort(cells[:, local], axis=-1)
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        routes = np.asarray(connectivity.cell_entities[degree], dtype=np.int64)
        if routes.shape == faces.shape[:2] and np.array_equal(entities[routes], faces):
            return routes.reshape((-1,))
    return _rows_of_simplices(entities, faces.reshape((-1, degree + 1)))


def _single_region(rows: np.ndarray, values: np.ndarray, count: int, /) -> np.ndarray:
    """The one value each row carries, -1 for rows with none or several."""
    sentinel = np.iinfo(np.int64).max
    lower = np.full((count,), sentinel, dtype=np.int64)
    upper = np.full((count,), -1, dtype=np.int64)
    valid = rows >= 0
    np.minimum.at(lower, rows[valid], values[valid])
    np.maximum.at(upper, rows[valid], values[valid])
    return np.where(lower == upper, lower, -1)


def _face_order_signs(source: np.ndarray, target: np.ndarray, /) -> np.ndarray:
    """Permutation parities from oriented recovered triangles to mesh face order."""
    position = np.argmax(target[:, :, None] == source[:, None, :], axis=2)
    inversions = (
        (position[:, 0] > position[:, 1]).astype(np.int8)
        + (position[:, 0] > position[:, 2])
        + (position[:, 1] > position[:, 2])
    )
    return np.where(inversions % 2 == 1, -1, 1).astype(np.int8)


def _segment_edge_orientations(
    construction: VolumeConstruction, edges: np.ndarray, source_rows: np.ndarray, /
) -> np.ndarray:
    """Orientation of each constrained edge along its source segment's parameter."""
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)[edges]
    endpoints = np.asarray(construction.source_segment_vertices, dtype=np.int64)[
        source_rows
    ]
    on_segment = (np.asarray(construction.witness_strata)[edges] == 1) & (
        np.asarray(construction.witness_entities, dtype=np.int64)[edges]
        == source_rows[:, None]
    )
    first = input_vertices == endpoints[:, :1]
    second = input_vertices == endpoints[:, 1:]
    if not np.all(first | second | on_segment):
        raise ValueError("Constrained PLC edge endpoints lost exact source parameters.")
    parameters = np.where(
        first,
        0.0,
        np.where(
            second,
            1.0,
            np.asarray(construction.witness_parameters, dtype=np.float64)[edges, 0],
        ),
    )
    return np.where(parameters[:, 1] > parameters[:, 0], 1, -1).astype(np.int8)


def _source_associations(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> tuple[GeometryAssociation, ...]:
    mesh = construction.mesh
    complex_ = source.complex
    source_maps = _source_identity_maps(source)
    # One authored ID/occurrence pair resolves to one key; the mesh refers to
    # each source entity from many simplices. Each dimension's pairs resolve
    # in one pass; the resolver is bound once for the refusals: an Equinox
    # module method lookup traverses the whole source pytree.
    key_for_scope = source.source_entity_key_for_scope
    key_tables: dict[int, dict[tuple[str, tuple[str, ...]], str]] = {}

    def _cached_source_key(
        dimension: int, identifier: str, occurrence_path: tuple[str, ...], /
    ) -> str:
        table = key_tables.get(dimension)
        if table is None:
            table = _source_key_table(source, source_maps, dimension)
            key_tables[dimension] = table
        key = table.get((identifier, occurrence_path))
        if key is None:
            key = _source_key(
                key_for_scope, source_maps, dimension, identifier, occurrence_path
            )
        return key

    vertex_set, edge_set, face_set, cell_set = (
        mesh.entity_set(dimension) for dimension in (0, 1, 2, 3)
    )
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    roles_by_dimension = object_rows(
        (
            GeometrySourceEntityRole.VERTEX,
            GeometrySourceEntityRole.EDGE,
            GeometrySourceEntityRole.FACET,
            GeometrySourceEntityRole.REGION,
        )
    )
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    witness_strata = np.asarray(construction.witness_strata, dtype=np.int8)
    if (
        input_vertices.shape != vertex_ids.shape
        or witness_strata.shape != vertex_ids.shape
        or np.any(input_vertices < -1)
        or np.any(input_vertices >= len(complex_.vertex_ids))
    ):
        raise ValueError(
            "PLC vertex ancestry must align with the recovered mesh vertex bank."
        )
    # Source vertices are dimension 0; inserted vertices take their witness
    # stratum, a declared-plane point (stratum 3) is a facet point and an
    # interior point (stratum 0) a region point.
    inserted = input_vertices < 0
    if np.any(inserted & ((witness_strata < 0) | (witness_strata > 3))):
        raise ValueError("Inserted PLC vertices must retain one source-stratum witness.")
    vertex_dimensions = np.where(
        inserted,
        np.asarray((3, 1, 2, 2), dtype=np.int64)[np.clip(witness_strata, 0, 3)],
        0,
    )
    vertex_keys = tuple(
        _cached_source_key(int(dimension), identity, path)
        for dimension, identity, path in zip(
            vertex_dimensions.tolist(),
            construction.vertex_source_ids,
            construction.vertex_source_occurrence_paths,
            strict=True,
        )
    )
    facet_rows_by_key = {
        (identity, path): row
        for row, (identity, path) in enumerate(
            zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)
        )
    }

    def _facet_region_sides(facet: int, /) -> tuple[str | None, str | None]:
        region_sides: list[str | None] = []
        for region in np.asarray(complex_.facet_regions[facet], dtype=np.int64).tolist():
            if region < 0:
                region_sides.append(None)
            else:
                region_row = int(region)
                region_sides.append(
                    _cached_source_key(
                        3,
                        complex_.region_ids[region_row],
                        complex_.region_occurrence_paths[region_row],
                    )
                )
        return region_sides[0], region_sides[1]

    vertex_region_sides = tuple(
        _facet_region_sides(facet_rows_by_key[(identity, path)])
        if dimension == 2
        else (None, None)
        for dimension, identity, path in zip(
            vertex_dimensions.tolist(),
            construction.vertex_source_ids,
            construction.vertex_source_occurrence_paths,
            strict=True,
        )
    )
    vertex_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        vertex_set.entity_set_id,
        vertex_ids,
        vertex_keys,
        construction.witness_deviations,
        exact=bool(np.all(construction.witness_deviations == 0.0)),
        source_roles=tuple(roles_by_dimension[vertex_dimensions].tolist()),
        source_occurrence_paths=construction.vertex_source_occurrence_paths,
        source_region_sides=vertex_region_sides,
    )

    cells, connectivity_edges, connectivity_faces = _tetrahedron_regions(construction)
    deviations = np.asarray(construction.witness_deviations, dtype=np.float64)
    cell_regions = np.asarray(construction.cell_regions, dtype=np.int64)
    region_keys = object_rows(
        [
            _cached_source_key(3, identifier, path)
            for identifier, path in zip(
                construction.region_ids,
                construction.region_occurrence_paths,
                strict=True,
            )
        ]
    )
    region_paths = object_rows(construction.region_occurrence_paths)
    facet_keys = object_rows(
        [
            _cached_source_key(2, identifier, path)
            for identifier, path in zip(
                construction.facet_ids, construction.facet_occurrence_paths, strict=True
            )
        ]
    )
    facet_paths = object_rows(construction.facet_occurrence_paths)
    facet_sides = object_rows(
        [_facet_region_sides(facet) for facet in range(len(construction.facet_ids))]
    )
    none_sides = object_rows([(None, None)])

    # Edges: a constrained segment, else the one source facet whose recovered
    # faces contain it, else the one region of its cells.
    edge_count = connectivity_edges.shape[0]
    edge_region = _single_region(
        _cell_entity_rows(mesh, cells, connectivity_edges, 1),
        np.repeat(cell_regions, 6),
        edge_count,
    )
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    face_facets = np.asarray(construction.face_source_facets, dtype=np.int64)
    facet_edge_rows = _rows_of_simplices(
        connectivity_edges, boundary_faces[:, _TRIANGLE_EDGES].reshape((-1, 2))
    )
    edge_facet = _single_region(
        facet_edge_rows,
        np.repeat(face_facets, _TRIANGLE_EDGES.shape[0]),
        edge_count,
    )
    segment_edges = np.asarray(construction.segments, dtype=np.int64).reshape((-1, 2))
    segment_rows = _rows_of_simplices(connectivity_edges, segment_edges)
    if np.any(segment_rows < 0) or np.any(facet_edge_rows < 0):
        raise ValueError("Constrained PLC edges must be canonical mesh edges.")
    edge_segment = np.full((edge_count,), -1, dtype=np.int64)
    edge_segment[segment_rows] = np.arange(segment_rows.size, dtype=np.int64)
    constrained_edges = edge_segment >= 0
    facet_edges = ~constrained_edges & (edge_facet >= 0)
    region_edges = ~constrained_edges & ~facet_edges
    if np.any(region_edges & (edge_region < 0)):
        raise ValueError("Every PLC mesh edge must have one source stratum or region.")
    segment_keys = object_rows(
        [
            _cached_source_key(1, identifier, path)
            for identifier, path in zip(
                construction.segment_source_ids,
                construction.segment_source_occurrence_paths,
                strict=True,
            )
        ]
    )
    segment_paths = object_rows(construction.segment_source_occurrence_paths)
    selected_segments = edge_segment[constrained_edges]
    selected_facets = edge_facet[facet_edges]
    selected_regions = edge_region[region_edges]
    edge_keys = np.empty((edge_count,), dtype=object)
    edge_keys[constrained_edges] = segment_keys[selected_segments]
    edge_keys[facet_edges] = facet_keys[selected_facets]
    edge_keys[region_edges] = region_keys[selected_regions]
    edge_path_rows = np.empty((edge_count,), dtype=object)
    edge_path_rows[constrained_edges] = segment_paths[selected_segments]
    edge_path_rows[facet_edges] = facet_paths[selected_facets]
    edge_path_rows[region_edges] = region_paths[selected_regions]
    edge_side_rows = np.repeat(none_sides, edge_count)
    edge_side_rows[facet_edges] = facet_sides[selected_facets]
    # np.full would turn the string enum into str; slice assignment keeps it.
    edge_role_rows = np.empty((edge_count,), dtype=object)
    edge_role_rows[:] = GeometrySourceEntityRole.REGION
    edge_role_rows[constrained_edges] = GeometrySourceEntityRole.EDGE
    edge_role_rows[facet_edges] = GeometrySourceEntityRole.FACET
    edge_residual_values = np.where(
        region_edges, 0.0, np.max(deviations[connectivity_edges], axis=1)
    )
    edge_orientations = np.zeros((edge_count,), dtype=np.int8)
    edge_orientations[constrained_edges] = _segment_edge_orientations(
        construction,
        connectivity_edges[constrained_edges],
        np.asarray(construction.segment_source_edges, dtype=np.int64)[selected_segments],
    )
    edge_sources = edge_keys.tolist()
    edge_paths = edge_path_rows.tolist()
    edge_roles = edge_role_rows.tolist()
    edge_residuals = edge_residual_values
    edge_region_sides = edge_side_rows.tolist()

    # Faces: a recovered boundary face of its source facet, else the one
    # region of its cells.
    face_count = connectivity_faces.shape[0]
    face_region = _single_region(
        _cell_entity_rows(mesh, cells, connectivity_faces, 2),
        np.repeat(cell_regions, 4),
        face_count,
    )
    boundary_rows = _rows_of_simplices(connectivity_faces, boundary_faces)
    if np.any(boundary_rows < 0):
        raise ValueError("Recovered PLC faces must be canonical mesh faces.")
    constrained_faces = np.zeros((face_count,), dtype=np.bool_)
    constrained_faces[boundary_rows] = True
    region_faces = ~constrained_faces
    if np.any(region_faces & (face_region < 0)):
        raise ValueError("Every PLC mesh face must have one source facet or region.")
    boundary_keys = object_rows(
        [
            _cached_source_key(2, identifier, path)
            for identifier, path in zip(
                construction.face_source_ids,
                construction.face_source_occurrence_paths,
                strict=True,
            )
        ]
    )
    face_keys = np.empty((face_count,), dtype=object)
    face_keys[boundary_rows] = boundary_keys
    face_keys[region_faces] = region_keys[face_region[region_faces]]
    face_path_rows = np.empty((face_count,), dtype=object)
    face_path_rows[boundary_rows] = object_rows(
        list(construction.face_source_occurrence_paths)
    )
    face_path_rows[region_faces] = region_paths[face_region[region_faces]]
    face_side_rows = np.repeat(none_sides, face_count)
    face_side_rows[boundary_rows] = facet_sides[face_facets]
    face_role_rows = np.empty((face_count,), dtype=object)
    face_role_rows[:] = GeometrySourceEntityRole.REGION
    face_role_rows[boundary_rows] = GeometrySourceEntityRole.FACET
    face_residual_values = np.zeros((face_count,), dtype=np.float64)
    face_residual_values[boundary_rows] = np.max(deviations[boundary_faces], axis=1)
    face_orientations = np.zeros((face_count,), dtype=np.int8)
    face_orientations[boundary_rows] = np.asarray(
        construction.face_orientations, dtype=np.int8
    ) * _face_order_signs(boundary_faces, connectivity_faces[boundary_rows])
    face_sources = face_keys.tolist()
    face_paths = face_path_rows.tolist()
    face_roles = face_role_rows.tolist()
    face_residuals = face_residual_values
    face_region_sides = face_side_rows.tolist()

    cell_sources = tuple(region_keys[cell_regions].tolist())
    cell_paths = tuple(region_paths[cell_regions].tolist())
    cell_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        cell_set.entity_set_id,
        cell_ids,
        cell_sources,
        np.zeros((cell_ids.size,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.REGION,) * cell_ids.size,
        source_occurrence_paths=cell_paths,
    )
    edge_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        edge_set.entity_set_id,
        edge_ids,
        tuple(edge_sources),
        np.asarray(edge_residuals, dtype=np.float64),
        exact=bool(np.all(np.asarray(edge_residuals, dtype=np.float64) == 0.0)),
        source_roles=tuple(edge_roles),
        source_occurrence_paths=tuple(edge_paths),
        source_region_sides=tuple(edge_region_sides),
        orientations=edge_orientations,
    )
    face_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        face_set.entity_set_id,
        face_ids,
        tuple(face_sources),
        np.asarray(face_residuals, dtype=np.float64),
        exact=bool(np.all(np.asarray(face_residuals, dtype=np.float64) == 0.0)),
        source_roles=tuple(face_roles),
        source_occurrence_paths=tuple(face_paths),
        source_region_sides=tuple(face_region_sides),
        orientations=face_orientations,
    )
    return vertex_association, edge_association, face_association, cell_association


def _plc_source_point_witness(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> PlcSourcePointWitness | None:
    """Exact source points of a PLC mesh whose boundary received source points.

    Absent unless refinement inserted a segment, facet or plane point; an
    unsplit boundary consists of source vertices, whose carriers are exact.
    """
    domain = construction.domain
    if not isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain)):
        return None
    input_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    strata = np.asarray(construction.witness_strata, dtype=np.int64)
    inserted = (input_vertices < 0) & (strata > 0)
    if not np.any(inserted) and not isinstance(domain, AffinePlcDomain):
        return None
    kinds = np.where(input_vertices >= 0, 0, np.where(strata > 0, strata, -1))
    entities = np.where(
        input_vertices >= 0,
        input_vertices,
        np.where(strata > 0, np.asarray(construction.witness_entities), -1),
    )
    parameters = np.where(
        inserted[:, None], np.asarray(construction.witness_parameters), 0.0
    )
    face_facets = np.asarray(construction.source_face_facets, dtype=np.int64)
    complex_ = source.complex
    sides = np.asarray(complex_.facet_regions, dtype=np.int64)[face_facets]
    if complex_.facet_planes is None:
        planes = np.full((face_facets.size, 3, 3), np.nan, dtype=np.float64)
        tolerances = np.zeros((face_facets.size,), dtype=np.float64)
    else:
        planes = np.asarray(complex_.facet_planes, dtype=np.float64)[face_facets]
        tolerances = np.asarray(complex_.facet_plane_tolerances, dtype=np.float64)[
            face_facets
        ]
    return PlcSourcePointWitness(
        np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64),
        kinds,
        entities,
        parameters,
        segments=construction.source_segment_vertices,
        faces=construction.source_faces,
        face_groups=face_facets,
        face_planes=planes,
        face_plane_tolerances=tolerances,
        domain_face_rows=np.flatnonzero(sides[:, 0] != sides[:, 1]),
        domain_id=domain.domain_id,
        vertex_deviations=None,
    )


def _organization(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    construction: VolumeConstruction,
    /,
    *,
    source_scope_owner: _SourceEntityScopes | None = None,
    region_controls: tuple[RegionControl, ...] | None = None,
) -> NativeMeshOrganization:
    owner = source if source_scope_owner is None else source_scope_owner
    region_scopes = _source_region_scopes(source, owner)
    controls = _region_controls_by_row(
        region_scopes,
        specification.region_controls if region_controls is None else region_controls,
    )
    seeds = _region_seeds_by_row(region_scopes, specification.region_seeds)
    zones, zone_ids = _region_zones(construction, region_scopes, controls, seeds)
    patches, labels, region_boundary_evidence = _patches_and_labels(
        source,
        specification,
        construction,
        zone_ids,
        region_scopes,
        controls,
        seeds,
    )
    return NativeMeshOrganization(
        patches=patches,
        zones=zones,
        labels=labels,
        associations=_source_associations(source, construction),
        region_boundary_evidence=region_boundary_evidence,
    )


def _mesh_scope(
    mesh: CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        np.asarray(identifiers, dtype=np.int64),
    )


@final
class PreparedPlcVolume(StrictModule, NonTrainableState):
    """Prepared native PLC request whose execution returns construction data.

    ``carrier`` is an optional kept conforming tetrahedralization (vertices,
    tetrahedra) of a one-region triangle PLC, refined in place instead of a
    fresh constraint recovery.
    """

    source: NativePlcSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    schedule: NativeVolumeSchedule
    global_affine: np.ndarray | None
    affine_source: CompartmentMeshingSource | None
    carrier: tuple[np.ndarray, np.ndarray] | None
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        schedule: NativeVolumeSchedule,
        /,
        *,
        affine_source: CompartmentMeshingSource | None = None,
        carrier: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if (
            source.mapped_domain is not None
            and source.mapped_domain.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Mapped PLC source and target use different spatial coordinate contracts."
            )
        global_affine = None
        if affine_source is not None:
            affine_source.validate_source_integrity()
            if (affine_source.source_id, affine_source.source_revision) != (
                source.source_id,
                source.source_revision,
            ):
                raise ValueError(
                    "Global affine source must bind the exact PLC source revision."
                )
            global_affine = np.asarray(
                affine_source.labels.asset.spatial_affine.matrix, dtype=np.float64
            )
        if global_affine is not None:
            global_affine = np.array(global_affine, dtype=np.float64, copy=True)
            if (
                global_affine.shape != (4, 4)
                or not np.all(np.isfinite(global_affine))
                or not np.array_equal(global_affine[3], (0.0, 0.0, 0.0, 1.0))
                or source.mapped_domain is not None
            ):
                raise ValueError(
                    "Global affine PLC mapping requires one finite affine matrix and no rooted map."
                )
            global_affine.setflags(write=False)
        if carrier is not None:
            if source.mapped_domain is not None or global_affine is not None:
                raise ValueError(
                    "A kept PLC carrier is refined only in its Cartesian frame."
                )
            carrier = (
                np.array(carrier[0], dtype=np.float64, copy=True),
                np.array(carrier[1], dtype=np.int64, copy=True),
            )
            for values in carrier:
                values.setflags(write=False)
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.schedule = schedule
        self.global_affine = global_affine
        self.affine_source = affine_source
        self.carrier = carrier
        preparation: dict[str, object] = {
            "kind": "native-plc-volume-preparation",
            "source": source.binding_id,
            "specification": specification.specification_id,
            "schedule": schedule.schedule_id,
            "global_affine": None if global_affine is None else global_affine.tolist(),
        }
        if carrier is not None:
            preparation["kept_carrier"] = array_tree_fingerprint(carrier)
        self.prepared_id = canonical_fingerprint(preparation)

    def execute(self) -> NativeMeshConstruction:
        mapped_geometry = (
            None
            if self.source.mapped_domain is None
            else self.source.mapped_domain.source_geometry
        )
        source_map_id = (
            ""
            if mapped_geometry is None or mapped_geometry.source_map_id is None
            else mapped_geometry.source_map_id
        )
        if self.global_affine is not None:
            source_map_id = canonical_fingerprint(
                {
                    "kind": "global-affine-plc-map",
                    "source": self.source.source_id,
                    "revision": self.source.source_revision,
                    "matrix": self.global_affine.tolist(),
                }
            )
        construction = generate_plc_volume(
            self.source.complex,
            self.specification,
            self.schedule,
            source_id=self.source.source_id,
            source_revision=self.source.source_revision,
            source_map_id=source_map_id,
            frame_id=self.coordinate_contract.spatial_id,
            source_domain=self.source.mapped_domain,
            global_affine=self.global_affine,
            affine_source=self.affine_source,
            carrier=self.carrier,
        )
        source_point_witness = _plc_source_point_witness(self.source, construction)
        if self.global_affine is None:
            physical = physical_plc_tetrahedra(self.source, construction)
            geometry = physical.geometry
            physical_points = physical.points
            reference_map = mapped_plc_reference_map(self.source, construction)
        else:
            if not isinstance(construction.domain, AffinePlcDomain):
                raise RuntimeError(
                    "Global affine PLC construction lost its source domain."
                )
            physical_mesh, geometry = map_affine_plc_mesh(
                construction.domain, construction.mesh, witness=source_point_witness
            )
            construction = replace(
                construction,
                mesh=physical_mesh,
                face_orientations=(
                    -construction.face_orientations
                    if construction.domain.determinant < 0
                    else construction.face_orientations
                ),
            )
            physical_points = np.asarray(physical_mesh.coordinates, dtype=np.float64)
            reference_map = None
        compliance = _volume_compliance(
            self.source, self.specification, self.schedule, construction, physical_points
        )
        fidelity_source: (
            MappedReferenceBoundarySource | AffinePlcBoundarySource | None
        ) = None
        fidelity_tolerance: float | None = None
        fidelity_witness: SourceBoundaryChartWitness | None = None
        scoped_fidelity = []
        if self.source.mapped_domain is not None:
            candidate_query = self.source.boundary_query()
            if not isinstance(candidate_query, MappedReferenceBoundarySource):
                raise TypeError(
                    "Mapped PLC volume coverage requires its exact root boundary query."
                )
            fidelity_source = candidate_query
            fidelity_witness = mapped_plc_boundary_chart_witness(
                self.source, construction, geometry, fidelity_source
            )
            fidelity_tolerance = (
                self.specification.maximum_deviation
                if self.specification.maximum_deviation is not None
                else self.source.maximum_deviation
            )
            if fidelity_tolerance is None:
                raise ValueError(
                    "Mapped PLC filling requires an explicit or source-declared physical tolerance."
                )
            if (
                self.source.maximum_deviation is not None
                and fidelity_tolerance > self.source.maximum_deviation
            ):
                raise ValueError(
                    "The requested mapped PLC tolerance exceeds its source-declared bound."
                )
            for feature in self.specification.protected_features:
                if not feature.hard:
                    continue
                identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
                if feature.feature_kind in (
                    FeatureKind.SURFACE,
                    FeatureKind.MATERIAL_INTERFACE,
                ):
                    keys = tuple(
                        self.source.source_entity_key_for_scope(2, int(identifier))
                        for identifier in identifiers.tolist()
                    )
                    candidate_feature_query = self.source.boundary_query(
                        face_scope_keys=keys
                    )
                    if not isinstance(
                        candidate_feature_query, MappedReferenceBoundarySource
                    ):
                        raise TypeError(
                            "Mapped facet features require their exact root chart query."
                        )
                    feature_query = candidate_feature_query
                    feature_witness = mapped_plc_boundary_chart_witness(
                        self.source, construction, geometry, feature_query
                    )
                elif feature.feature_kind is FeatureKind.CURVE:
                    keys = tuple(
                        self.source.source_entity_key_for_scope(1, int(identifier))
                        for identifier in identifiers.tolist()
                    )
                    candidate_feature_query = self.source.boundary_query(
                        edge_scope_keys=keys
                    )
                    if not isinstance(
                        candidate_feature_query, MappedReferenceBoundarySource
                    ):
                        raise TypeError(
                            "Mapped curve features require their exact root chart query."
                        )
                    feature_query = candidate_feature_query
                    feature_witness = mapped_plc_edge_chart_witness(
                        self.source, construction, geometry, feature_query
                    )
                elif feature.feature_kind is FeatureKind.CORNER:
                    feature_query = fidelity_source
                    feature_witness = mapped_plc_point_witness(
                        self.source, construction, geometry, feature_query, identifiers
                    )
                else:
                    raise ValueError(
                        f"Mapped PLC feature {feature.feature_id} has no exact source-chart witness route."
                    )
                scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
        elif isinstance(construction.domain, AffinePlcDomain):
            face_facets = np.asarray(construction.source_face_facets, dtype=np.int64)
            sides = np.asarray(self.source.complex.facet_regions, dtype=np.int64)[
                face_facets
            ]
            fidelity_source = AffinePlcBoundarySource(
                self.source, construction.domain, face_facets[sides[:, 0] != sides[:, 1]]
            )
            fidelity_witness = affine_plc_boundary_chart_witness(
                self.source, fidelity_source, construction, construction.mesh, geometry
            )
            fidelity_tolerance = (
                self.specification.maximum_deviation
                if self.specification.maximum_deviation is not None
                else 0.0
            )
        if self.source.mapped_domain is None and self.global_affine is None:
            for feature in self.specification.protected_features:
                if feature.hard and feature.feature_kind is FeatureKind.CURVE:
                    feature_query, feature_witness = plc_curve_chart_witness(
                        self.source,
                        construction,
                        geometry,
                        np.asarray(feature.scope.entity_ids, dtype=np.int64),
                    )
                    scoped_fidelity.append(
                        (feature_query, feature_witness, feature.maximum_deviation)
                    )
        schedule = MeshCertificationSchedule(
            "mapped_volume"
            if self.source.mapped_domain is not None or self.global_affine is not None
            else "volume_plc"
        )
        certification = NativeCertificationRequest(
            schedule,
            self.source.source_id,
            self.source.source_revision,
            domain=construction.domain,
            cell_regions=construction.cell_regions,
            fidelity_source=fidelity_source,
            fidelity_tolerance=fidelity_tolerance,
            fidelity_witness=fidelity_witness,
            protected_features=self.specification.protected_features,
            scoped_fidelity=tuple(scoped_fidelity),
            feature_source=self.source if self.specification.protected_features else None,
            source_coverage_witness=source_point_witness,
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=_organization(self.source, self.specification, construction),
            stages=construction.stages,
            reference_maps=() if reference_map is None else (reference_map,),
            source_mesh=(
                None
                if self.source.mapped_domain is None
                else self.source.mapped_domain.reference_mesh
            ),
            source_geometry=(
                None
                if self.source.mapped_domain is None
                else self.source.mapped_domain.source_geometry
            ),
        )


def _volume_compliance(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    construction: VolumeConstruction,
    physical_points: np.ndarray,
    /,
) -> MeshingComplianceReport:
    points = np.asarray(physical_points, dtype=np.float64)
    requested: list[tuple[str, float]] = []
    if schedule.radius_edge_bound is not None:
        requested.append(("radius_edge_bound", schedule.radius_edge_bound))
    requested.append(("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees))
    achieved: list[tuple[str, float]] = [
        ("refinement_status", float(int(construction.refinement_status))),
        ("improvement_status", float(int(construction.improvement_status))),
        ("refinement_unmet_count", float(len(construction.refinement_unmet))),
        ("improvement_unmet_count", float(len(construction.improvement_unmet))),
        (
            "refinement_inserted_vertices",
            float(construction.refinement_inserted_vertices),
        ),
        ("final_quality_unmet_count", float(len(construction.final_quality_unmet))),
    ]
    # Approved input-forced exemptions: only a dihedral row forced by an
    # authored sub-target ridge, or a radius-edge row forced by an authored
    # facet corner, is evidence instead of a finding. A soft schedule first
    # moves every unmet shape row out of the gate into soft evidence.
    schedule_rows, soft_rows = _split_shape_target_rows(
        schedule, construction.final_quality_schedule_unmet
    )
    exemptions = input_forced_dihedral_exemptions(
        source.complex,
        construction,
        points,
        schedule.minimum_dihedral_degrees,
        schedule_rows,
    )
    radius_exemptions = input_forced_radius_edge_exemptions(
        source.complex,
        construction,
        points,
        schedule.radius_edge_bound,
        schedule_rows,
    )
    exempt_rows = {exemption.row for exemption in exemptions} | {
        exemption.row for exemption in radius_exemptions
    }
    if exemptions:
        achieved.append(("input_forced_dihedral_exemption_count", float(len(exemptions))))
    ridge_angles: dict[int, float] = {}
    for exemption in exemptions:
        for ridge in exemption.ridges:
            ridge_angles[ridge.entity_id] = ridge.input_angle_degrees
            achieved.append(
                (
                    f"input_forced_dihedral:cell:{exemption.cell_id}:ridge:"
                    f"{ridge.entity_id}:achieved_degrees",
                    exemption.achieved_degrees,
                )
            )
    achieved.extend(
        (f"input_forced_dihedral:ridge:{entity_id}:input_angle_degrees", angle)
        for entity_id, angle in sorted(ridge_angles.items())
    )
    if radius_exemptions:
        achieved.append(
            ("input_forced_radius_edge_exemption_count", float(len(radius_exemptions)))
        )
    corner_evidence: dict[str, tuple[float, float]] = {}
    for radius_exemption in radius_exemptions:
        for corner in radius_exemption.corners:
            corner_key = (
                f"vertex:{corner.vertex_entity_id}:edges:"
                f"{corner.edge_entity_ids[0]}:{corner.edge_entity_ids[1]}"
            )
            corner_evidence[corner_key] = (
                corner.input_angle_degrees,
                corner.forced_ratio,
            )
            prefix = (
                f"input_forced_radius_edge:cell:{radius_exemption.cell_id}:{corner_key}"
            )
            achieved.extend(
                (
                    (f"{prefix}:face_ratio", corner.face_ratio),
                    (f"{prefix}:achieved_ratio", radius_exemption.achieved_ratio),
                )
            )
    for corner_key, (angle, forced_ratio) in sorted(corner_evidence.items()):
        achieved.extend(
            (
                (f"input_forced_radius_edge:{corner_key}:input_angle_degrees", angle),
                (f"input_forced_radius_edge:{corner_key}:forced_ratio", forced_ratio),
            )
        )
    if not _shape_targets_gated(schedule):
        achieved.extend(_soft_shape_target_evidence(construction, soft_rows))
    issues: list[str] = []
    issues.extend(
        f"native_final_quality:{criterion}"
        for row, (criterion, _reason, _cell_id, _value) in enumerate(schedule_rows)
        if row not in exempt_rows
    )
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append(f"size_control:{control.control_id}:unsupported")
            continue
        selected_facets = _rows_from_stable_ids(
            np.asarray(control.scope.entity_ids, dtype=np.int64),
            source.complex.facet_entity_ids,
            "facet",
        )
        selected_faces = np.flatnonzero(
            np.isin(construction.face_source_facets, selected_facets)
        )
        if selected_faces.size == 0:
            issues.append(f"size_control:{control.control_id}:no_boundary_faces")
            continue
        faces = np.asarray(construction.boundary_faces, dtype=np.int64)[selected_faces]
        edges = sorted_unique_rows(
            np.sort(faces[:, ((0, 1), (1, 2), (2, 0))], axis=2).reshape((-1, 2))
        )
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        measures = edge_length_measures(points, edges)
        policy = specification.size_compliance
        declaration = S3_BOUNDARY_EDGE_SIZE
        for statistic in policy.target_statistics:
            achieved_value = size_statistic(lengths, statistic)
            key = f"size:{control.control_id}:{statistic}"
            requested.append((key, declaration.bound_factor * control.target_size))
            achieved.append((key, achieved_value))
            if control.strength is SizeControlStrength.HARD and declaration.exceeds(
                measures, statistic, control.target_size, policy
            ):
                issues.append(key)
        if control.maximum_size is not None:
            achieved_maximum = float(np.max(lengths, initial=0.0))
            key = f"size:{control.control_id}:maximum"
            requested.append((key, control.maximum_size))
            achieved.append((key, achieved_maximum))
            if (
                control.strength is SizeControlStrength.HARD
                and achieved_maximum > control.maximum_size
            ):
                issues.append(key)
    for feature in specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        if feature.scope.entity_dimension == 2:
            rows = _rows_from_stable_ids(
                identifiers, source.complex.facet_entity_ids, "facet"
            )
            face_mask = np.isin(construction.face_source_facets, rows)
            vertices = np.unique(construction.boundary_faces[face_mask])
        elif feature.scope.entity_dimension == 1:
            complex_rows = _rows_from_stable_ids(
                identifiers, source.complex.plc_edge_entity_ids, "PLC edge"
            )
            selected_rows = {
                int(complex_row): local_row
                for local_row, complex_row in enumerate(
                    construction.source_segment_complex_rows.tolist()
                )
            }
            try:
                rows = np.asarray(
                    [selected_rows[int(row)] for row in complex_rows.tolist()],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    "Protected PLC edge is absent from the selected source-edge bank."
                ) from error
            segment_mask = np.isin(construction.segment_source_edges, rows)
            vertices = np.unique(construction.segments[segment_mask])
        elif feature.scope.entity_dimension == 0:
            rows = _rows_from_stable_ids(
                identifiers, source.complex.vertex_entity_ids, "vertex"
            )
            vertices = np.flatnonzero(np.isin(construction.input_vertices, rows))
        else:
            continue
        achieved_bound = float(
            np.max(construction.witness_deviations[vertices], initial=0.0)
        )
        key = f"protected:{feature.feature_id}:maximum_deviation"
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, achieved_bound))
        if feature.hard and achieved_bound > feature.maximum_deviation:
            issues.append(key)
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


@checked
def prepare_native_plc_volume(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedPlcVolume:
    if options.route != "plc_tetrahedral":
        raise ValueError(
            "Native PLC volume preparation requires the plc_tetrahedral route."
        )
    issues = volume_support_issues(source, specification)
    if issues:
        raise ValueError("Unsupported native PLC volume request: " + "; ".join(issues))
    schedule = options.volume_schedule
    if schedule is None:
        raise ValueError(
            "Native PLC volume route requires its numerical volume schedule."
        )
    return PreparedPlcVolume(
        source,
        specification,
        coordinate_contract,
        schedule,
    )


__all__ = [
    "PreparedPlcVolume",
    "prepare_native_plc_volume",
    "volume_support_issues",
]
