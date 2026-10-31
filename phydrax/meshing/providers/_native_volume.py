#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare constrained PLC fills for canonical native acceptance."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import MeshcoreStatus
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh
from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource
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
from .._sizing import SizeControlStrength, UniformSizeControl
from .._volume_generation import (
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from ._native_options import NativeMeshingOptions
from ._native_plc_geometry import (
    mapped_plc_boundary_chart_witness,
    mapped_plc_edge_chart_witness,
    mapped_plc_point_witness,
    mapped_plc_reference_map,
    physical_plc_tetrahedra,
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
) -> list[str]:
    target = specification.target
    families = target.cell_families
    issues: list[str] = []
    if set((*families.required, *families.preferred)) != {"tetrahedron"}:
        issues.append("a tetrahedral volume target")
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or families.allowed_transitions
        or families.allow_mixed
    ):
        issues.append("affine tetrahedra in ambient dimension three")
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
    if not isinstance(source, NativePlcSource):
        raise TypeError("source must be NativePlcSource.")
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
    rows = {
        tuple(sorted(face)): int(identifier)
        for face, identifier in zip(
            connectivity_faces.tolist(), identifiers.tolist(), strict=True
        )
    }
    if len(rows) != identifiers.size:
        raise ValueError("Canonical volume faces do not have unique vertex incidence.")
    try:
        return np.asarray(
            [
                rows[tuple(sorted(face))]
                for face in np.asarray(
                    construction.boundary_faces, dtype=np.int64
                ).tolist()
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "A recovered boundary face is absent from canonical mesh incidence."
        ) from error


def _edge_entity_ids(mesh: CellMesh, /) -> dict[tuple[int, int], int]:
    connectivity_edges, _ = simplex_volume_entities(mesh)
    identifiers = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    if connectivity_edges.shape[0] != identifiers.size:
        raise ValueError("Canonical edge incidence and edge entity IDs are misaligned.")
    rows = {
        tuple(sorted(edge)): int(identifier)
        for edge, identifier in zip(
            connectivity_edges.tolist(), identifiers.tolist(), strict=True
        )
    }
    if len(rows) != identifiers.size:
        raise ValueError("Canonical volume edges do not have unique vertex incidence.")
    return rows


def _faces_connected(faces: np.ndarray, /) -> bool:
    if faces.shape[0] <= 1:
        return True
    edge_rows: dict[tuple[int, int], list[int]] = {}
    for row, face in enumerate(faces.tolist()):
        for pair in combinations(face, 2):
            edge_rows.setdefault(tuple(sorted(pair)), []).append(row)
    parents = list(range(faces.shape[0]))

    def find(row: int, /) -> int:
        while parents[row] != row:
            parents[row] = parents[parents[row]]
            row = parents[row]
        return row

    for incident in edge_rows.values():
        if len(incident) < 2:
            continue
        root = find(incident[0])
        for row in incident[1:]:
            other = find(row)
            if other != root:
                parents[other] = root
    return len({find(row) for row in range(faces.shape[0])}) == 1


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
    facet_faces: dict[int, np.ndarray] = {}
    region_boundary_faces: list[list[int]] = [[] for _ in range(len(region_scopes))]
    region_patch_sides: list[list[tuple[str, int]]] = [
        [] for _ in range(len(region_scopes))
    ]
    for facet, identity in enumerate(construction.facet_ids):
        selected_mask = face_facets == facet
        selected = boundary_face_ids[selected_mask]
        if selected.size == 0:
            raise ValueError(f"PLC facet {identity!r} has no constrained mesh faces.")
        facet_faces[facet] = selected
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
            connected=_faces_connected(boundary_faces[selected_mask]),
            adjacent_zone_ids=adjacent,
            source_adjacent_region_ids=source_pair,
            source_adjacent_region_occurrence_paths=source_paths,
        )
        patches.append(patch)
        if sides[0] != sides[1]:
            for side_index, region in enumerate(sides):
                if region < 0:
                    continue
                region_boundary_faces[region].extend(int(value) for value in selected)
                region_patch_sides[region].append(
                    (patch.patch_id, 1 if side_index == 0 else -1)
                )
    for control in specification.patch_controls:
        selected_facets = _rows_from_stable_ids(
            np.asarray(control.scope.entity_ids, dtype=np.int64),
            source.complex.facet_entity_ids,
            "facet",
        )
        selected = np.concatenate([facet_faces[int(facet)] for facet in selected_facets])
        selected_rows = np.flatnonzero(np.isin(face_facets, selected_facets))
        adjacent_rows = sorted(
            {
                int(region)
                for facet in selected_facets.tolist()
                for region in facet_regions[int(facet)]
                if int(region) >= 0
            }
        )
        patches.append(
            MeshPatch(
                control.name,
                _mesh_scope(mesh, 2, selected),
                connected=_faces_connected(boundary_faces[selected_rows]),
                adjacent_zone_ids=tuple(zone_ids[row] for row in adjacent_rows),
            )
        )

    interfaces: list[int] = []
    sheets: list[int] = []
    for face_row, facet in enumerate(face_facets.tolist()):
        positive, negative = facet_regions[facet]
        if positive == negative:
            sheets.append(int(boundary_face_ids[face_row]))
        elif positive >= 0 and negative >= 0:
            interfaces.append(int(boundary_face_ids[face_row]))
    labels: list[MeshLabel] = []
    region_evidence: list[RegionBoundaryEvidence] = []
    for region, (region_id, occurrence_path, source_scope) in enumerate(region_scopes):
        selected = np.unique(np.asarray(region_boundary_faces[region], dtype=np.int64))
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
        if selected:
            labels.append(
                MeshLabel(
                    name, _mesh_scope(mesh, 2, np.asarray(selected, dtype=np.int64))
                )
            )
    edge_ids_by_key = _edge_entity_ids(mesh)
    explicit = source.complex.segments.shape[0]
    internal = np.asarray(
        [
            edge_ids_by_key[tuple(sorted(edge))]
            for edge, source_edge in zip(
                construction.segments.tolist(),
                construction.segment_source_edges.tolist(),
                strict=True,
            )
            if construction.source_segment_complex_rows[source_edge] < explicit
        ],
        dtype=np.int64,
    )
    if internal.size:
        labels.append(MeshLabel("internal_curve", _mesh_scope(mesh, 1, internal)))
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


def _source_associations(
    source: NativePlcSource, construction: VolumeConstruction, /
) -> tuple[GeometryAssociation, ...]:
    mesh = construction.mesh
    complex_ = source.complex
    source_maps = _source_identity_maps(source)
    # One authored ID/occurrence pair resolves to one key; the mesh refers to
    # each source entity from many simplices. The resolver is bound once: an
    # Equinox module method lookup traverses the whole source pytree.
    key_for_scope = source.source_entity_key_for_scope
    source_keys: dict[tuple[int, str, tuple[str, ...]], str] = {}

    def _cached_source_key(
        dimension: int, identifier: str, occurrence_path: tuple[str, ...], /
    ) -> str:
        cache_key = (dimension, identifier, occurrence_path)
        key = source_keys.get(cache_key)
        if key is None:
            key = _source_key(
                key_for_scope, source_maps, dimension, identifier, occurrence_path
            )
            source_keys[cache_key] = key
        return key

    vertex_set, edge_set, face_set, cell_set = (
        mesh.entity_set(dimension) for dimension in (0, 1, 2, 3)
    )
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    roles_by_dimension = {
        0: GeometrySourceEntityRole.VERTEX,
        1: GeometrySourceEntityRole.EDGE,
        2: GeometrySourceEntityRole.FACET,
        3: GeometrySourceEntityRole.REGION,
    }
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
    vertex_dimensions_list: list[int] = []
    for input_vertex, witness_stratum in zip(
        input_vertices.tolist(), witness_strata.tolist(), strict=True
    ):
        if input_vertex >= 0:
            vertex_dimensions_list.append(0)
        elif witness_stratum in (1, 2):
            vertex_dimensions_list.append(witness_stratum)
        elif witness_stratum == 3:
            # A point on its facet's declared exact plane is a facet point.
            vertex_dimensions_list.append(2)
        elif witness_stratum == 0:
            vertex_dimensions_list.append(3)
        else:
            raise ValueError(
                "Inserted PLC vertices must retain one source-stratum witness."
            )
    vertex_dimensions = np.asarray(vertex_dimensions_list, dtype=np.int64)
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
        source_roles=tuple(
            roles_by_dimension[int(dimension)] for dimension in vertex_dimensions
        ),
        source_occurrence_paths=construction.vertex_source_occurrence_paths,
        source_region_sides=vertex_region_sides,
    )

    cells, connectivity_edges, connectivity_faces = _tetrahedron_regions(construction)
    edge_regions: dict[tuple[int, int], set[int]] = {}
    face_regions: dict[tuple[int, ...], set[int]] = {}
    for cell, region in zip(
        cells.tolist(), construction.cell_regions.tolist(), strict=True
    ):
        for pair in combinations(cell, 2):
            edge_regions.setdefault(tuple(sorted(pair)), set()).add(region)
        for triple in combinations(cell, 3):
            face_regions.setdefault(tuple(sorted(triple)), set()).add(region)

    segment_source_by_edge: dict[
        tuple[int, int], tuple[str, tuple[str, ...], float, int]
    ] = {}
    for edge, source_id, path, source_row in zip(
        construction.segments.tolist(),
        construction.segment_source_ids,
        construction.segment_source_occurrence_paths,
        construction.segment_source_edges.tolist(),
        strict=True,
    ):
        key = tuple(sorted(edge))
        residual = float(
            np.max(construction.witness_deviations[np.asarray(edge, dtype=np.int64)])
        )
        segment_source_by_edge[key] = (source_id, path, residual, source_row)
    face_source_by_face: dict[
        tuple[int, int, int], tuple[str, tuple[str, ...], float, int, int]
    ] = {}
    for face, source_id, path, facet, orientation in zip(
        construction.boundary_faces.tolist(),
        construction.face_source_ids,
        construction.face_source_occurrence_paths,
        construction.face_source_facets.tolist(),
        construction.face_orientations.tolist(),
        strict=True,
    ):
        key = tuple(sorted(face))
        residual = float(
            np.max(construction.witness_deviations[np.asarray(face, dtype=np.int64)])
        )
        face_source_by_face[key] = (source_id, path, residual, facet, orientation)
    face_facet_edges: dict[tuple[int, int], set[int]] = {}
    for face, facet in zip(
        construction.boundary_faces.tolist(),
        construction.face_source_facets.tolist(),
        strict=True,
    ):
        for pair in combinations(face, 2):
            face_facet_edges.setdefault(tuple(sorted(pair)), set()).add(facet)

    edge_sources: list[str] = []
    edge_paths: list[tuple[str, ...]] = []
    edge_roles: list[GeometrySourceEntityRole] = []
    edge_residuals: list[float] = []
    edge_region_sides: list[tuple[str | None, str | None]] = []
    edge_orientations = np.zeros((edge_ids.size,), dtype=np.int8)
    for row, edge in enumerate(connectivity_edges.tolist()):
        key = tuple(sorted(edge))
        constrained = segment_source_by_edge.get(key)
        if constrained is not None:
            local_id, path, residual, source_row = constrained
            edge_sources.append(_cached_source_key(1, local_id, path))
            edge_paths.append(path)
            edge_roles.append(GeometrySourceEntityRole.EDGE)
            edge_region_sides.append((None, None))
            edge_residuals.append(residual)
            source_endpoints = construction.source_segment_vertices[source_row]
            parameters: list[float] = []
            for vertex in edge:
                input_vertex = int(construction.input_vertices[vertex])
                if input_vertex == int(source_endpoints[0]):
                    parameters.append(0.0)
                elif input_vertex == int(source_endpoints[1]):
                    parameters.append(1.0)
                elif (
                    construction.witness_strata[vertex] == 1
                    and construction.witness_entities[vertex] == source_row
                ):
                    parameters.append(float(construction.witness_parameters[vertex, 0]))
                else:
                    raise ValueError(
                        "Constrained PLC edge endpoints lost exact source parameters."
                    )
            edge_orientations[row] = 1 if parameters[1] > parameters[0] else -1
            continue
        facets = face_facet_edges.get(key, set())
        if len(facets) == 1:
            facet = next(iter(facets))
            local_id = construction.facet_ids[facet]
            path = construction.facet_occurrence_paths[facet]
            edge_sources.append(_cached_source_key(2, local_id, path))
            edge_paths.append(path)
            edge_roles.append(GeometrySourceEntityRole.FACET)
            edge_residuals.append(
                float(
                    np.max(
                        construction.witness_deviations[np.asarray(edge, dtype=np.int64)]
                    )
                )
            )
            edge_region_sides.append(_facet_region_sides(facet))
            continue
        regions = edge_regions.get(key, set())
        if len(regions) != 1:
            raise ValueError(
                "Every PLC mesh edge must have one source stratum or region."
            )
        region = next(iter(regions))
        local_id = construction.region_ids[region]
        path = construction.region_occurrence_paths[region]
        edge_sources.append(_cached_source_key(3, local_id, path))
        edge_paths.append(path)
        edge_roles.append(GeometrySourceEntityRole.REGION)
        edge_region_sides.append((None, None))
        edge_residuals.append(0.0)

    face_sources: list[str] = []
    face_paths: list[tuple[str, ...]] = []
    face_roles: list[GeometrySourceEntityRole] = []
    face_residuals: list[float] = []
    face_orientations = np.zeros((face_ids.size,), dtype=np.int8)
    face_region_sides: list[tuple[str | None, str | None]] = []
    for row, face in enumerate(connectivity_faces.tolist()):
        key = tuple(sorted(face))
        constrained = face_source_by_face.get(key)
        if constrained is not None:
            local_id, path, residual, facet, orientation = constrained
            face_sources.append(_cached_source_key(2, local_id, path))
            face_paths.append(path)
            face_roles.append(GeometrySourceEntityRole.FACET)
            face_residuals.append(residual)
            face_orientations[row] = orientation
            face_region_sides.append(_facet_region_sides(facet))
            continue
        regions = face_regions.get(key, set())
        if len(regions) != 1:
            raise ValueError("Every PLC mesh face must have one source facet or region.")
        region = next(iter(regions))
        local_id = construction.region_ids[region]
        path = construction.region_occurrence_paths[region]
        face_sources.append(_cached_source_key(3, local_id, path))
        face_paths.append(path)
        face_roles.append(GeometrySourceEntityRole.REGION)
        face_residuals.append(0.0)
        face_region_sides.append((None, None))

    cell_sources = tuple(
        _cached_source_key(
            3,
            construction.region_ids[int(region)],
            construction.region_occurrence_paths[int(region)],
        )
        for region in construction.cell_regions
    )
    cell_paths = tuple(
        construction.region_occurrence_paths[int(region)]
        for region in construction.cell_regions
    )
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
    """Prepared native PLC request whose execution returns construction data."""

    source: NativePlcSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    schedule: NativeVolumeSchedule
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        if (
            source.mapped_domain is not None
            and source.mapped_domain.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Mapped PLC source and target use different spatial coordinate contracts."
            )
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.schedule = schedule
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-plc-volume-preparation",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )

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
        construction = generate_plc_volume(
            self.source.complex,
            self.specification,
            self.schedule,
            source_id=self.source.source_id,
            source_revision=self.source.source_revision,
            source_map_id=source_map_id,
            frame_id=self.coordinate_contract.spatial_id,
            source_domain=self.source.mapped_domain,
        )
        physical = physical_plc_tetrahedra(self.source, construction)
        geometry = physical.geometry
        reference_map = mapped_plc_reference_map(self.source, construction)
        compliance = _volume_compliance(
            self.source, self.specification, self.schedule, construction, physical.points
        )
        fidelity_source: MappedReferenceBoundarySource | None = None
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
        schedule = MeshCertificationSchedule(
            "mapped_volume" if self.source.mapped_domain is not None else "volume_plc"
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
    requested: list[tuple[str, float]] = [
        ("radius_edge_bound", schedule.radius_edge_bound),
        ("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees),
    ]
    achieved: list[tuple[str, float]] = [
        ("refinement_status", float(int(construction.refinement_status))),
        ("improvement_status", float(int(construction.improvement_status))),
        ("refinement_unmet_count", float(len(construction.refinement_unmet))),
        ("improvement_unmet_count", float(len(construction.improvement_unmet))),
    ]
    issues: list[str] = []
    if construction.improvement_status is not MeshcoreStatus.OK:
        issues.append("native_improvement_status")
    if construction.improvement_unmet:
        issues.extend(
            f"native_improvement:{criterion}"
            for criterion, _reason, _cell_id, _value in construction.improvement_unmet
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
        edge_rows = {
            tuple(sorted((int(face[k]), int(face[(k + 1) % 3]))))
            for face in construction.boundary_faces[selected_faces].tolist()
            for k in range(3)
        }
        edges = np.asarray(sorted(edge_rows), dtype=np.int64)
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        policy = specification.size_compliance
        for statistic in policy.target_statistics:
            quantile = 0.5 if statistic == "p50" else 0.95
            achieved_value = float(np.quantile(lengths, quantile))
            key = f"size:{control.control_id}:{statistic}"
            requested.append((key, control.target_size))
            achieved.append((key, achieved_value))
            allowed = (
                control.target_size * (1.0 + policy.relative_tolerance)
                + policy.absolute_tolerance
            )
            if control.strength is SizeControlStrength.HARD and achieved_value > allowed:
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
