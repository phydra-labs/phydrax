#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolve authenticated CAD face uses to native one-sided trace domains."""

from __future__ import annotations

import numpy as np

from .._fingerprint import canonical_fingerprint
from ..discretization import CellMesh, IntegrationDomain
from ..discretization._views import FieldTraceSide
from ..discretization.fem._generic import FiniteElementDiscretization
from ..discretization.finite_volume._triangle_fv import TriangleFiniteVolumeDiscretization
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from ..geometry._source_identity import source_entity_id
from ..geometry.brep._differentiable import FixedTopologyBRepSource
from ..geometry.brep._model import BRepModel
from ..geometry.brep._partition import _entity_occurrence_id
from ..geometry.brep._source import BRepSource
from ._association import _incidence_pairs, _ordered_vertices, GeometryAssociation
from ._interface_binding import _outward_relation, _witness_rows
from ._organization import MeshZoneRole
from ._result import CellMeshingResult
from ._scope import (
    _geometry_model,
    _source_scope_owner,
    _SourceEntityScopes,
    MeshingEntityKind,
    MeshingScope,
)


def _association(
    result: CellMeshingResult, dimension: int, source: _SourceEntityScopes, /
) -> GeometryAssociation:
    mesh = result.mesh
    matches = tuple(
        value
        for value in result.associations
        if value.source_id == source.source_id
        and value.source_revision == source.source_revision
        and value.target_entity_set_id == mesh.entity_set(dimension).entity_set_id
    )
    if len(matches) != 1:
        raise ValueError("A unique source association for each mesh stratum is required.")
    return matches[0]


def _cell_incidence(mesh: CellMesh, /) -> dict[int, tuple[int, ...]]:
    vertices = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    cells = np.asarray(
        mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
    )
    pairs = _incidence_pairs(mesh, 0, mesh.topological_dimension)
    result: dict[int, list[int]] = {int(cell): [] for cell in cells}
    for vertex, cell in pairs:
        result[int(cells[cell])].append(int(vertices[vertex]))
    return {cell: tuple(sorted(members)) for cell, members in result.items()}


def _require_same_mesh(source: CellMesh, target: CellMesh, /) -> None:
    if source.numeric_version != target.numeric_version:
        raise ValueError("Discretization and result must bind the same mesh revision.")
    source_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    source_order = np.argsort(source_ids)
    target_order = np.argsort(target_ids)
    if (
        source.topological_dimension != target.topological_dimension
        or not np.array_equal(source_ids[source_order], target_ids[target_order])
        or not np.array_equal(
            np.asarray(source.coordinates)[source_order],
            np.asarray(target.coordinates)[target_order],
        )
        or _cell_incidence(source) != _cell_incidence(target)
    ):
        raise ValueError(
            "Discretization must preserve the exact mesh vertex and cell identities."
        )


def _subset_domain(
    base: IntegrationDomain, rows: np.ndarray, selection_id: str, /
) -> IntegrationDomain:
    return IntegrationDomain(
        base.kind,
        np.asarray(base.entity_indices)[rows],
        base.support_id,
        base.entity_set_id,
        owner_cells=np.asarray(base.owner_cells)[rows],
        neighbor_cells=np.asarray(base.neighbor_cells)[rows],
        owner_local_entities=np.asarray(base.owner_local_entities)[rows],
        neighbor_local_entities=np.asarray(base.neighbor_local_entities)[rows],
        neighbor_trace_permutations=np.asarray(base.neighbor_trace_permutations)[rows],
        periodic_face_mask=np.asarray(base.periodic_face_mask)[rows],
        selection_id=selection_id,
    )


def resolve_sided_integration_domains(
    result: CellMeshingResult,
    source: BRepModel | BRepSource | FixedTopologyBRepSource,
    scope: MeshingScope,
    discretization: FiniteElementDiscretization
    | TriangleFiniteVolumeDiscretization
    | UnstructuredFiniteVolumeDiscretization,
    /,
    *,
    tolerance: float,
) -> tuple[tuple[IntegrationDomain, FieldTraceSide], ...]:
    """Return ``(domain, side)`` routes for each selected exact CAD face use.

    Routes retain the native interior/exterior classification and partition by the
    incident cell's owner/neighbor position. Each CAD use has its own domain
    identity, even when two uses project to the same mesh facets. All required
    source, region, patch, association and topology evidence must be present.
    ``tolerance`` is the finite non-negative maximum source association residual.
    """
    if not isinstance(result, CellMeshingResult) or not isinstance(scope, MeshingScope):
        raise TypeError("Expected a CellMeshingResult and a MeshingScope.")
    if not isinstance(
        discretization,
        (
            FiniteElementDiscretization,
            TriangleFiniteVolumeDiscretization,
            UnstructuredFiniteVolumeDiscretization,
        ),
    ):
        raise TypeError("Sided routes require a native mesh FE or FV discretization.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    model = _geometry_model(source)
    MeshingScope.from_stored_fields(scope, source_model=model)
    owner = _source_scope_owner(source)
    mesh = result.mesh
    prepared_mesh = discretization.mesh
    if prepared_mesh is None:
        raise ValueError("Discretization must retain its source mesh.")
    _require_same_mesh(mesh, prepared_mesh)
    if (
        scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != mesh.topological_dimension - 1
        or (scope.source_id, scope.source_revision)
        != (owner.source_id, owner.source_revision)
        or scope.entity_set_id != owner.scope_entity_set_id(scope.entity_dimension)
        or not scope.side_records
        or scope.unqualified_entity_ids
    ):
        raise ValueError(
            "A purely sided scope of the exact source facet stratum is required."
        )
    top = mesh.topological_dimension
    if top != 3:
        raise ValueError("CAD solid side routes require a three-dimensional volume mesh.")
    facets = _association(result, top - 1, owner)
    cells = _association(result, top, owner)
    incidence = model.geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    )
    entities = {
        _entity_occurrence_id(record.member): record.member for record in incidence
    }
    patches = {patch.patch_id: patch for patch in result.patches}
    zones = {
        zone.zone_id: zone for zone in result.zones if zone.role is MeshZoneRole.REGION
    }
    facet_ids = np.asarray(mesh.entity_set(top - 1).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(top).entity_ids, dtype=np.int64)
    source_faces = _ordered_vertices(mesh, top - 1)
    if source_faces is None:
        raise ValueError("Sided routes require simplex facet incidence.")
    if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
        if discretization.connectivity is None or not hasattr(
            discretization.connectivity, "faces"
        ):
            raise ValueError("FV sided routes require tetrahedral face incidence.")
        target_faces = np.asarray(discretization.connectivity.faces, dtype=np.int64)
        prepared_cell_ids = np.asarray(discretization.cell_global_ids, dtype=np.int64)
    else:
        target_faces = _ordered_vertices(prepared_mesh, top - 1)
        if target_faces is None:
            raise ValueError("Sided routes require simplex facet incidence.")
        prepared_cell_ids = np.asarray(
            prepared_mesh.entity_set(top).entity_ids, dtype=np.int64
        )
    source_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    target_vertices = np.asarray(prepared_mesh.vertex_global_ids, dtype=np.int64)
    face_lookup = {
        tuple(sorted(int(value) for value in target_vertices[face])): row
        for row, face in enumerate(target_faces)
    }
    if len(face_lookup) != len(target_faces):
        raise ValueError("Prepared mesh has ambiguous facet incidence.")
    routes: list[tuple[IntegrationDomain, FieldTraceSide]] = []
    for record in scope.side_records:
        solid = entities.get(record.solid.occurrence_id)
        if (
            solid is None
            or solid.kind != "solid"
            or solid.index != record.solid_index
            or solid.occurrence_path != record.region_path
        ):
            raise ValueError("The selected solid occurrence is absent from the source.")
        solid_key = source_entity_id(
            model.source_id,
            model.source_revision,
            "solid",
            solid.index,
            occurrence_path=solid.occurrence_path,
        )
        region_matches = [
            e
            for e in result.region_boundary_evidence
            if e.source_scope.source_id == scope.source_id
            and e.source_scope.source_revision == scope.source_revision
            and e.source_scope.entity_set_id == owner.scope_entity_set_id(top)
            and owner.source_entity_key_for_scope(
                top, int(np.asarray(e.source_scope.entity_ids)[0])
            )
            == (solid_key, solid.occurrence_path)
            and e.target_mesh_id == mesh.mesh_id
            and e.target_topology_id == mesh.topology_id
        ]
        if len(region_matches) != 1:
            raise ValueError(
                "Selected solid needs unique current region boundary evidence."
            )
        evidence = region_matches[0]
        face_key, face_path = owner.source_entity_key_for_scope(top - 1, record.entity_id)
        selected = np.asarray(
            [
                i
                for i, key in enumerate(facets.source_entity_ids)
                if key == face_key and facets.source_occurrence_paths[i] == face_path
            ],
            dtype=np.int64,
        )
        if selected.size == 0:
            raise ValueError("Selected CAD face has no associated mesh facets.")
        selected_ids = np.asarray(facets.target_global_ids, dtype=np.int64)[selected]
        if not np.all(
            np.isin(selected_ids, np.asarray(evidence.boundary_label.scope.entity_ids))
        ):
            raise ValueError("Selected facets are not in the incident region boundary.")
        matching_patches = [
            patches[pid]
            for pid, sign in evidence.patch_sides
            if pid in patches
            and np.any(np.isin(selected_ids, np.asarray(patches[pid].scope.entity_ids)))
        ]
        if not matching_patches:
            raise ValueError("Selected face lacks oriented region patch evidence.")
        covered = set()
        zone_candidates = set()
        for patch in matching_patches:
            sign = dict(evidence.patch_sides)[patch.patch_id]
            pair, paths = (
                patch.source_adjacent_region_ids,
                patch.source_adjacent_region_occurrence_paths,
            )
            if pair is None or paths is None:
                raise ValueError("Patch lacks exact adjacent region occurrences.")
            index = 0 if sign == 1 else 1
            if (pair[index], paths[index]) != (
                evidence.source_region_id,
                evidence.source_occurrence_path,
            ):
                raise ValueError("Patch side does not match its region evidence.")
            covered.update(
                int(x) for x in np.asarray(patch.scope.entity_ids) if x in selected_ids
            )
            zone_candidates.update(patch.adjacent_zone_ids)
        if covered != set(selected_ids.tolist()):
            raise ValueError("Region patches do not cover every selected facet.")
        region_zones = []
        for zid in zone_candidates:
            zone = zones.get(zid)
            if (
                zone is None
                or zone.scope.entity_set_id != mesh.entity_set(top).entity_set_id
            ):
                continue
            ids = np.asarray(zone.scope.entity_ids, dtype=np.int64)
            rows = cells.target_rows(ids)
            if (
                np.all(np.asarray(cells.resolved)[rows])
                and not np.any(np.asarray(cells.ambiguous)[rows])
                and all(
                    cells.source_entity_ids[i] == solid_key
                    and cells.source_occurrence_paths[i] == solid.occurrence_path
                    for i in rows
                )
            ):
                region_zones.append(zone)
        if len(region_zones) != 1:
            raise ValueError(
                "Selected side needs a unique associated incident region zone."
            )
        zone = region_zones[0]
        if any(zone.zone_id not in patch.adjacent_zone_ids for patch in matching_patches):
            raise ValueError("Each source patch must name the incident region zone.")
        facet_scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            top - 1,
            mesh.entity_set(top - 1).entity_set_id,
            selected_ids,
        )
        _witness_rows(result, facets, facet_scope, (face_key,), tolerance)
        if not np.all(np.asarray(facets.resolved)[selected]) or np.any(
            np.asarray(facets.ambiguous)[selected]
        ):
            raise ValueError("Selected facet association is unresolved or ambiguous.")
        orientations = np.asarray(facets.orientations)[selected]
        if np.any(orientations == 0):
            raise ValueError("Selected facets have no source orientation evidence.")
        facet_rows = np.flatnonzero(np.isin(facet_ids, selected_ids))
        region_rows = np.flatnonzero(np.isin(cell_ids, np.asarray(zone.scope.entity_ids)))
        assoc_rows = facets.target_rows(facet_ids[facet_rows])
        if np.any(
            np.asarray(facets.orientations)[assoc_rows]
            * _outward_relation(mesh, facet_rows, region_rows)
            != record.orientation
        ):
            raise ValueError(
                "CAD face use orientation disagrees with the incident mesh region."
            )
        try:
            prepared_facet_rows = np.asarray(
                [
                    face_lookup[
                        tuple(
                            sorted(
                                int(value) for value in source_vertices[source_faces[row]]
                            )
                        )
                    ]
                    for row in facet_rows
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "Prepared mesh does not preserve selected facet incidence."
            ) from exc
        prepared_region_rows = np.flatnonzero(
            np.isin(prepared_cell_ids, np.asarray(zone.scope.entity_ids))
        )
        if (
            prepared_facet_rows.size != selected_ids.size
            or prepared_region_rows.size != zone.scope.entity_ids.shape[0]
        ):
            raise ValueError(
                "Prepared mesh does not preserve the selected entity identities."
            )
        found = set()
        for kind in ("exterior_facet", "interior_facet"):
            base = discretization.integration_domain(kind)
            indices = np.asarray(base.entity_indices, dtype=np.int64)
            positions = np.flatnonzero(np.isin(indices, prepared_facet_rows))
            if not positions.size:
                continue
            if np.any(np.asarray(base.periodic_face_mask)[positions]):
                raise ValueError("Periodic facets do not have a single shared embedding.")
            owner_members = np.isin(
                np.asarray(base.owner_cells)[positions], prepared_region_rows
            )
            neighbor_members = np.isin(
                np.asarray(base.neighbor_cells)[positions], prepared_region_rows
            )
            if np.any(owner_members == neighbor_members):
                raise ValueError(
                    "Every selected facet must have exactly one incident region cell."
                )
            for side, mask in (("owner", owner_members), ("neighbor", neighbor_members)):
                subset = positions[mask]
                if subset.size:
                    identity = canonical_fingerprint(
                        {
                            "kind": "cad-sided-integration-selection",
                            "scope": scope.scope_id,
                            "face": record.face.occurrence_id,
                            "solid": record.solid.occurrence_id,
                            "side": side,
                            "domain": kind,
                        }
                    )
                    routes.append((_subset_domain(base, subset, identity), side))
                    found.update(indices[subset].tolist())
        if found != set(prepared_facet_rows.tolist()):
            raise ValueError(
                "The prepared discretization does not route all selected facets."
            )
    return tuple(routes)


__all__ = ["resolve_sided_integration_domains"]
