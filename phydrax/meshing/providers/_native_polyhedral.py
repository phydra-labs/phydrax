from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import isfinite
from typing import final

import equinox as eqx
import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import PeriodicConstraint
from phydrax.meshing._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from phydrax.meshing._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from phydrax.meshing._plc_mapped_support import classify_plc_carriers
from phydrax.meshing._polyhedral_generation import (
    effective_polyhedral_schedule,
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
)
from phydrax.meshing._result import MeshingComplianceReport
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_periodic_constraints import (
    validate_periodic_mesh_constraints,
)
from phydrax.meshing.providers._native_sources import (
    NativePlcSource,
    NativePolyhedralSource,
)
from phydrax.typing import checked


def _periodic_control_issues(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> tuple[str, ...]:
    if not specification.periodic_constraints:
        return ()
    issues: list[str] = []
    has_facet_pair = False
    for constraint in specification.periodic_constraints:
        if not isinstance(constraint, PeriodicConstraint):
            issues.append("typed periodic constraints")
            continue
        source_scope, target_scope = constraint.source_scope, constraint.target_scope
        degree = source_scope.entity_dimension
        if (
            degree not in (0, 1, 2)
            or target_scope.entity_dimension != degree
            or any(
                scope.entity_kind is not MeshingEntityKind.GEOMETRY
                or (scope.source_id, scope.source_revision)
                != (source.source_id, source.source_revision)
                or scope.entity_set_id != source.scope_entity_set_id(degree)
                or np.setdiff1d(
                    np.asarray(scope.entity_ids, dtype=np.int64),
                    source.scope_entity_ids(degree),
                ).size
                for scope in (source_scope, target_scope)
            )
        ):
            issues.append("source-bound PLC periodic entity scopes of matching degree")
            continue
        transform = np.asarray(constraint.transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.array_equal(
            transform[3], np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        ):
            issues.append("exact homogeneous 3D periodic transforms")
            continue
        has_facet_pair |= degree == 2
    if not has_facet_pair:
        issues.append("at least one periodic PLC facet pair")
    return tuple(dict.fromkeys(issues))


def _rebind_periodic_constraints(
    source: NativePolyhedralSource,
    mesh: CellMesh,
    associations: tuple[GeometryAssociation, ...],
    specification: VolumeMeshingSpec,
    /,
) -> tuple[PeriodicConstraint, ...]:
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
    )
    bound: list[PeriodicConstraint] = []
    for constraint in specification.periodic_constraints:
        degree = constraint.source_scope.entity_dimension
        association = associations[degree]

        def output_ids(scope: MeshingScope, /) -> np.ndarray:
            source_keys = {
                source.source_entity_key_for_scope(degree, int(identifier))
                for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist()
            }
            selected = [
                int(target_id)
                for target_id, source_id, role, path in zip(
                    np.asarray(association.target_global_ids, dtype=np.int64),
                    association.source_entity_ids,
                    association.source_roles,
                    association.source_occurrence_paths,
                    strict=True,
                )
                if role is roles[degree] and (source_id, path) in source_keys
            ]
            if not selected:
                raise ValueError(
                    "A periodic PLC control scope has no associated output entities."
                )
            return np.asarray(sorted(selected), dtype=np.int64)

        entity_set_id = mesh.entity_set(degree).entity_set_id
        source_scope = MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.MESH,
            degree,
            entity_set_id,
            output_ids(constraint.source_scope),
        )
        target_scope = MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.MESH,
            degree,
            entity_set_id,
            output_ids(constraint.target_scope),
        )
        bound.append(
            PeriodicConstraint(
                source_scope,
                target_scope,
                constraint.transform,
                tolerance=constraint.tolerance,
                conforming_required=constraint.conforming_required,
            )
        )
    return tuple(bound)


def polyhedral_support_issues(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> tuple[str, ...]:
    """Admit only an explicitly authored PLC and implemented power semantics."""
    issues: list[str] = []
    if not isinstance(source, NativePolyhedralSource):
        issues.append(
            "a NativePolyhedralSource with authored power sites and stable site IDs"
        )
    target = specification.target
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
    ):
        issues.append("affine-face polyhedra in three ambient dimensions")
    families = set((*target.cell_families.required, *target.cell_families.preferred))
    if families != {"polyhedron"}:
        issues.append("an exclusively polyhedral cell-family request")
    if specification.fill_strategy is not VolumeFillStrategy.POLYHEDRAL:
        issues.append("the polyhedral fill strategy")
    scope = specification.boundary_scope
    facet_scope_ids = source.scope_entity_ids(2)
    if (
        scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 2
        or (scope.source_id, scope.source_revision)
        != (source.source_id, source.source_revision)
        or scope.entity_set_id != source.scope_entity_set_id(2)
        or not np.array_equal(
            np.sort(np.asarray(scope.entity_ids, dtype=np.int64)),
            np.sort(facet_scope_ids),
        )
    ):
        issues.append("the whole source PLC facet boundary on its bound revision")
    mapped_source = (
        source.mapped_domain is not None or source.complex.geometry is not None
    )
    if mapped_source:
        domain = source.mapped_domain
        if domain is None:
            issues.append("a declared MappedReferenceDomain for the PLC coordinate map")
        else:
            if (
                source.complex.geometry is None
                or cell_geometry_id(source.complex.geometry)
                != domain.reference_geometry_id
            ):
                issues.append(
                    "the exact PLC coordinate map bound to its mapped root geometry"
                )
            query = (
                source.boundary_query()
                if isinstance(source, NativePlcSource)
                else source.fidelity_source
            )
            authority = (
                query.authority
                if isinstance(query, MappedReferenceBoundarySource)
                else None
            )
            if (
                not isinstance(query, MappedReferenceBoundarySource)
                or not isinstance(authority, NativePlcSource)
                or (query.source_id, query.source_revision)
                != (source.source_id, source.source_revision)
                or authority.domain.domain_id != domain.domain_id
                or authority.complex.complex_id != source.complex.complex_id
            ):
                issues.append("an independent mapped PLC root-chart boundary witness")
        requested_tolerance = specification.maximum_deviation
        declared_tolerance = source.maximum_deviation
        if requested_tolerance is None and declared_tolerance is None:
            issues.append("an explicit physical source-deviation tolerance")
        if (
            requested_tolerance is not None
            and declared_tolerance is not None
            and requested_tolerance != declared_tolerance
        ):
            issues.append("non-contradictory source and request deviation tolerances")
    if source.complex.geometry is not None and source.mapped_domain is None:
        issues.append("a mapped PLC carrier bound to its MappedReferenceDomain")
    for feature in specification.protected_features:
        dimension = feature.scope.entity_dimension
        if (
            dimension not in (0, 1, 2)
            or feature.scope.source_id != source.source_id
            or feature.scope.source_revision != source.source_revision
            or feature.scope.entity_set_id != source.scope_entity_set_id(dimension)
            or not np.isin(
                np.asarray(feature.scope.entity_ids, dtype=np.int64),
                source.scope_entity_ids(dimension),
            ).all()
        ):
            issues.append("source-bound protected PLC vertex, edge, or facet features")
    uniform_controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if len(uniform_controls) != len(specification.size_controls):
        issues.append("uniform Euclidean size controls")
    if any(control.maximum_growth_rate is not None for control in uniform_controls):
        issues.append("uniform size controls without a nonconstant growth-rate request")
    if specification.layer_controls:
        issues.append(
            "a layer/core composition owner for requested boundary-layer controls"
        )
    issues.extend(_periodic_control_issues(source, specification))
    return tuple(issues)


def _mesh_scope(
    mesh: CellMesh, dimension: int, entity_ids: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        entity_ids,
    )


def _association(
    source: NativePlcSource | NativePolyhedralSource,
    mesh: CellMesh,
    dimension: int,
    target_ids: np.ndarray,
    source_ids: Sequence[str],
    roles: Sequence[GeometrySourceEntityRole],
    residuals: np.ndarray,
    region_sides: Sequence[tuple[str | None, str | None]],
    source_occurrence_paths: Sequence[tuple[str, ...]],
    /,
) -> GeometryAssociation:
    entity_set = mesh.entity_set(dimension)
    values = np.asarray(residuals, dtype=np.float64)
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        entity_set.entity_set_id,
        target_ids,
        tuple(source_ids),
        values,
        exact=bool(np.all(values == 0.0)),
        source_roles=tuple(roles),
        source_region_sides=tuple(region_sides),
        source_occurrence_paths=tuple(source_occurrence_paths),
    )


def _vertex_association(
    source: NativePlcSource | NativePolyhedralSource,
    construction: PolyhedralConstruction,
    /,
) -> tuple[GeometryAssociation, np.ndarray, tuple[GeometrySourceEntityRole, ...]]:
    carrier = construction.carrier
    local_carriers = np.asarray(
        construction.restricted_cells.vertex_carriers, dtype=np.int64
    )
    carrier_global_ids = np.asarray(carrier.mesh.vertex_global_ids, dtype=np.int64)
    if np.any(local_carriers >= carrier_global_ids.size):
        raise ValueError(
            "Power vertex carrier refers outside the recovered PLC point bank."
        )
    native_carriers = np.full(local_carriers.shape, -1, dtype=np.int64)
    valid = local_carriers >= 0
    native_carriers[valid] = carrier_global_ids[local_carriers[valid]]
    ancestry = classify_plc_carriers(carrier, native_carriers)
    role_by_dimension = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    dimensions = np.asarray(ancestry.dimensions, dtype=np.int64)
    if np.any((dimensions < 0) | (dimensions > 3)):
        raise ValueError("Power vertex carrier ancestry has an unsupported PLC stratum.")
    roles = tuple(role_by_dimension[int(value)] for value in dimensions)
    facet_rows = {
        identifier: row for row, identifier in enumerate(source.complex.facet_ids)
    }
    region_sides: list[tuple[str | None, str | None]] = []
    for role, identifier in zip(roles, ancestry.source_entity_ids, strict=True):
        if role is GeometrySourceEntityRole.FACET:
            facet = facet_rows[identifier]
            minus, plus = map(int, np.asarray(source.complex.facet_regions)[facet])
            region_sides.append(
                (
                    None if minus < 0 else source.complex.region_ids[minus],
                    None if plus < 0 else source.complex.region_ids[plus],
                )
            )
        else:
            region_sides.append((None, None))
    association = _association(
        source,
        construction.mesh,
        0,
        np.asarray(construction.mesh.entity_set(0).entity_ids, dtype=np.int64),
        ancestry.source_entity_ids,
        roles,
        np.asarray(ancestry.deviations, dtype=np.float64),
        region_sides,
        ancestry.source_occurrence_paths,
    )
    return association, native_carriers, roles


def _carrier_segment_rows(
    construction: PolyhedralConstruction,
    /,
) -> dict[tuple[int, int], tuple[str, tuple[str, ...]]]:
    mesh_vertex_ids = np.asarray(
        construction.carrier.mesh.vertex_global_ids, dtype=np.int64
    )
    segments = np.asarray(construction.carrier.segments, dtype=np.int64)
    return {
        (
            min(int(mesh_vertex_ids[first]), int(mesh_vertex_ids[second])),
            max(int(mesh_vertex_ids[first]), int(mesh_vertex_ids[second])),
        ): (source_id, tuple(path))
        for (first, second), source_id, path in zip(
            segments.tolist(),
            construction.carrier.segment_source_ids,
            construction.carrier.segment_source_occurrence_paths,
            strict=True,
        )
    }


def _edge_association(
    source: NativePlcSource | NativePolyhedralSource,
    construction: PolyhedralConstruction,
    vertex_association: GeometryAssociation,
    vertex_carriers: np.ndarray,
    vertex_roles: tuple[GeometrySourceEntityRole, ...],
    /,
) -> GeometryAssociation:
    mesh = construction.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral source association requires polyhedral connectivity.")
    edges = np.asarray(connectivity.edges, dtype=np.int64)
    vertex_ids = vertex_association.source_entity_ids
    vertex_paths = vertex_association.source_occurrence_paths
    vertex_dims = tuple(vertex_roles)
    vertex_residuals = np.asarray(vertex_association.residuals, dtype=np.float64)
    segment_by_carriers = _carrier_segment_rows(construction)
    face_edge_offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int64)
    face_edge_values = np.asarray(connectivity.face_edge_values, dtype=np.int64)
    face_regions: list[set[int]] = [set() for _ in range(connectivity.edge_count)]
    cell_regions = np.asarray(construction.cell_regions, dtype=np.int64)
    for face, (start, stop) in enumerate(
        zip(face_edge_offsets[:-1], face_edge_offsets[1:], strict=True)
    ):
        owners = (
            int(connectivity.face_owner[face]),
            int(connectivity.face_neighbor[face]),
        )
        adjacent = {int(cell_regions[cell]) for cell in owners if cell >= 0}
        for edge in face_edge_values[start:stop]:
            face_regions[int(edge)].update(adjacent)

    complex_ = source.complex
    facet_keys = tuple(
        zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)
    )
    facet_vertex_keys: dict[
        tuple[str, tuple[str, ...]], set[tuple[str, tuple[str, ...]]]
    ] = {key: set() for key in facet_keys}
    vertex_facet_keys: dict[
        tuple[str, tuple[str, ...]], set[tuple[str, tuple[str, ...]]]
    ] = {}
    polygon_owners = np.asarray(complex_.polygon_facets, dtype=np.int64)
    polygon_offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    for polygon, facet_row in enumerate(polygon_owners.tolist()):
        facet_key = facet_keys[int(facet_row)]
        first, last = polygon_offsets[polygon : polygon + 2]
        for vertex in polygon_vertices[first:last].tolist():
            vertex_key = (
                complex_.vertex_ids[int(vertex)],
                tuple(complex_.vertex_occurrence_paths[int(vertex)]),
            )
            facet_vertex_keys[facet_key].add(vertex_key)
            vertex_facet_keys.setdefault(vertex_key, set()).add(facet_key)

    source_edge_by_endpoints: dict[
        frozenset[tuple[str, tuple[str, ...]]], tuple[str, tuple[str, ...]]
    ] = {}
    source_edge_endpoints: dict[
        tuple[str, tuple[str, ...]], frozenset[tuple[str, tuple[str, ...]]]
    ] = {}
    for vertices, edge_id, occurrence in zip(
        complex_.plc_edge_vertices,
        complex_.plc_edge_ids,
        complex_.plc_edge_occurrence_paths,
        strict=True,
    ):
        endpoints = frozenset(
            (
                complex_.vertex_ids[int(vertex)],
                tuple(complex_.vertex_occurrence_paths[int(vertex)]),
            )
            for vertex in vertices
        )
        record = (edge_id, tuple(occurrence))
        old = source_edge_by_endpoints.get(endpoints)
        if old is not None and old != record:
            raise ValueError(
                "PLC edge endpoint ancestry is ambiguous across occurrences."
            )
        source_edge_by_endpoints[endpoints] = record
        source_edge_endpoints[record] = endpoints

    source_ids: list[str] = []
    source_paths: list[tuple[str, ...]] = []
    roles: list[GeometrySourceEntityRole] = []
    residuals = np.zeros((edges.shape[0],), dtype=np.float64)
    region_sides: list[tuple[str | None, str | None]] = []
    region_paths = tuple(complex_.region_occurrence_paths)
    for edge in range(edges.shape[0]):
        first, second = int(edges[edge, 0]), int(edges[edge, 1])
        first_carrier = set(map(int, vertex_carriers[first][vertex_carriers[first] >= 0]))
        second_carrier = set(
            map(int, vertex_carriers[second][vertex_carriers[second] >= 0])
        )
        carrier_union = tuple(sorted(first_carrier | second_carrier))
        segment_record = (
            segment_by_carriers.get((carrier_union[0], carrier_union[1]))
            if len(carrier_union) == 2
            else None
        )
        first_key = (vertex_ids[first], tuple(vertex_paths[first]))
        second_key = (vertex_ids[second], tuple(vertex_paths[second]))
        if segment_record is not None:
            role = GeometrySourceEntityRole.EDGE
            identifier, occurrence_path = segment_record
        else:
            first_role, second_role = vertex_dims[first], vertex_dims[second]
            first_id, first_path = first_key
            second_id, second_path = second_key
            identifier = ""
            occurrence_path = ()
            role = GeometrySourceEntityRole.REGION
            if (
                first_role is second_role
                and first_id == second_id
                and first_path == second_path
                and first_role is not GeometrySourceEntityRole.VERTEX
            ):
                role, identifier, occurrence_path = first_role, first_id, first_path
            elif (
                first_role is GeometrySourceEntityRole.VERTEX
                and second_role is GeometrySourceEntityRole.VERTEX
            ):
                record = source_edge_by_endpoints.get(frozenset((first_key, second_key)))
                if record is not None:
                    role, identifier, occurrence_path = (
                        GeometrySourceEntityRole.EDGE,
                        record[0],
                        record[1],
                    )
                else:
                    common_facets = vertex_facet_keys.get(
                        first_key, set()
                    ) & vertex_facet_keys.get(second_key, set())
                    if len(common_facets) == 1:
                        facet_key = next(iter(common_facets))
                        role, identifier, occurrence_path = (
                            GeometrySourceEntityRole.FACET,
                            facet_key[0],
                            facet_key[1],
                        )
            elif (
                first_role is GeometrySourceEntityRole.VERTEX
                and second_role is GeometrySourceEntityRole.FACET
                and first_key in facet_vertex_keys.get((second_id, second_path), set())
            ):
                role, identifier, occurrence_path = (
                    GeometrySourceEntityRole.FACET,
                    second_id,
                    second_path,
                )
            elif (
                second_role is GeometrySourceEntityRole.VERTEX
                and first_role is GeometrySourceEntityRole.FACET
                and second_key in facet_vertex_keys.get((first_id, first_path), set())
            ):
                role, identifier, occurrence_path = (
                    GeometrySourceEntityRole.FACET,
                    first_id,
                    first_path,
                )
            elif (
                first_role is GeometrySourceEntityRole.EDGE
                and second_role is GeometrySourceEntityRole.VERTEX
            ):
                record = (first_id, first_path)
                if second_key in source_edge_endpoints.get(record, frozenset()):
                    role, identifier, occurrence_path = (
                        GeometrySourceEntityRole.EDGE,
                        first_id,
                        first_path,
                    )
            elif (
                second_role is GeometrySourceEntityRole.EDGE
                and first_role is GeometrySourceEntityRole.VERTEX
            ):
                record = (second_id, second_path)
                if first_key in source_edge_endpoints.get(record, frozenset()):
                    role, identifier, occurrence_path = (
                        GeometrySourceEntityRole.EDGE,
                        second_id,
                        second_path,
                    )
            if not identifier:
                adjacent_regions = face_regions[edge]
                if len(adjacent_regions) == 1:
                    region = next(iter(adjacent_regions))
                    role = GeometrySourceEntityRole.REGION
                    identifier = complex_.region_ids[region]
                    occurrence_path = region_paths[region]
                else:
                    raise MeshingFailure(
                        MeshingFailureCategory.ASSOCIATION_FAILED,
                        "A polyhedral mesh edge lacks unique PLC source ancestry.",
                        provider_code="native_polyhedral_association",
                        stage="geometry_association",
                        entity_ids=(int(np.asarray(connectivity.edge_global_ids)[edge]),),
                    )
        source_ids.append(identifier)
        source_paths.append(occurrence_path)
        roles.append(role)
        residuals[edge] = max(vertex_residuals[first], vertex_residuals[second])
        adjacent_regions = sorted(face_regions[edge])
        region_sides.append(
            (
                None
                if not adjacent_regions
                else complex_.region_ids[adjacent_regions[0]],
                None
                if len(adjacent_regions) < 2
                else complex_.region_ids[adjacent_regions[1]],
            )
        )
    return _association(
        source,
        mesh,
        1,
        np.asarray(connectivity.edge_global_ids, dtype=np.int64),
        source_ids,
        roles,
        residuals,
        region_sides,
        source_paths,
    )


def _face_association(
    source: NativePlcSource | NativePolyhedralSource,
    construction: PolyhedralConstruction,
    vertex_association: GeometryAssociation,
    /,
) -> GeometryAssociation:
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral source association requires polyhedral connectivity.")
    regions = np.asarray(construction.cell_regions, dtype=np.int64)
    owners = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    residuals_by_vertex = np.asarray(vertex_association.residuals, dtype=np.float64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    source_ids: list[str] = []
    roles: list[GeometrySourceEntityRole] = []
    residuals = np.zeros((connectivity.face_count,), dtype=np.float64)
    region_sides: list[tuple[str | None, str | None]] = []
    source_paths: list[tuple[str, ...]] = []
    for face in range(connectivity.face_count):
        facet = int(construction.face_source_facets[face])
        if facet >= 0:
            source_id = construction.face_source_ids[face]
            if source_id is None or source_id != source.complex.facet_ids[facet]:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    "Power face source identity must match its PLC facet ancestry.",
                    provider_code="native_polyhedral_association",
                    stage="geometry_association",
                )
            left, right = map(int, np.asarray(source.complex.facet_regions)[facet])
            sides = (
                None if left < 0 else source.complex.region_ids[left],
                None if right < 0 else source.complex.region_ids[right],
            )
            role = GeometrySourceEntityRole.FACET
            occurrence_path = tuple(construction.face_source_occurrence_paths[face])
        else:
            left_cell, right_cell = int(owners[face]), int(neighbors[face])
            region = int(regions[left_cell])
            if right_cell >= 0 and regions[right_cell] != region:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    "A material interface must retain its PLC facet witness.",
                    provider_code="native_polyhedral_association",
                    stage="geometry_association",
                )
            source_id = source.complex.region_ids[region]
            sides = (source_id, source_id)
            role = GeometrySourceEntityRole.REGION
            occurrence_path = tuple(source.complex.region_occurrence_paths[region])
        first, last = face_offsets[face : face + 2]
        residuals[face] = float(np.max(residuals_by_vertex[face_vertices[first:last]]))
        source_ids.append(source_id)
        roles.append(role)
        region_sides.append(sides)
        source_paths.append(occurrence_path)
    return _association(
        source,
        construction.mesh,
        2,
        np.asarray(connectivity.face_global_ids, dtype=np.int64),
        source_ids,
        roles,
        residuals,
        region_sides,
        source_paths,
    )


def _cell_association(
    source: NativePlcSource | NativePolyhedralSource,
    construction: PolyhedralConstruction,
    /,
) -> GeometryAssociation:
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral source association requires polyhedral connectivity.")
    source_ids = tuple(
        source.complex.region_ids[int(region)] for region in construction.cell_regions
    )
    sides = tuple((identifier, identifier) for identifier in source_ids)
    source_paths = tuple(
        source.complex.region_occurrence_paths[int(region)]
        for region in construction.cell_regions
    )
    return _association(
        source,
        construction.mesh,
        3,
        np.asarray(connectivity.cell_global_ids, dtype=np.int64),
        source_ids,
        tuple(GeometrySourceEntityRole.REGION for _ in source_ids),
        np.zeros((len(source_ids),), dtype=np.float64),
        sides,
        source_paths,
    )


def _polyhedral_associations(
    source: NativePlcSource | NativePolyhedralSource,
    construction: PolyhedralConstruction,
    /,
) -> tuple[GeometryAssociation, ...]:
    vertex, carriers, roles = _vertex_association(source, construction)
    edge = _edge_association(source, construction, vertex, carriers, roles)
    face = _face_association(source, construction, vertex)
    cell = _cell_association(source, construction)
    return vertex, edge, face, cell


def _require_protected_feature_witnesses(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    associations: tuple[GeometryAssociation, ...],
    /,
) -> None:
    for feature in specification.protected_features:
        dimension = feature.scope.entity_dimension
        expected = {
            source.source_entity_key_for_scope(dimension, int(identifier))
            for identifier in np.asarray(feature.scope.entity_ids, dtype=np.int64)
        }
        association = associations[dimension]
        matched = np.asarray(
            [
                (identifier, path) in expected
                for identifier, path in zip(
                    association.source_entity_ids,
                    association.source_occurrence_paths,
                    strict=True,
                )
            ],
            dtype=np.bool_,
        )
        if not np.any(matched):
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                f"Protected feature {feature.feature_id!r} has no polyhedral descendants.",
                provider_code="native_polyhedral_protected_feature",
                stage="geometry_association",
                entity_ids=tuple(
                    int(value) for value in np.asarray(feature.scope.entity_ids)
                ),
            )
        residuals = np.asarray(association.residuals, dtype=np.float64)
        failed = matched & (residuals > feature.maximum_deviation)
        if np.any(failed):
            target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Protected feature {feature.feature_id!r} exceeds its source deviation.",
                provider_code="native_polyhedral_protected_feature",
                stage="geometry_association",
                entity_ids=tuple(int(value) for value in target_ids[failed]),
            )


def _size_scope_edges(
    construction: PolyhedralConstruction,
    control: UniformSizeControl,
    /,
) -> np.ndarray:
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral size compliance requires polyhedral connectivity.")
    identifiers = np.asarray(control.scope.entity_ids, dtype=np.int64)
    if control.scope.entity_dimension == 2:
        selected_faces = np.flatnonzero(
            np.isin(construction.face_source_facets, identifiers)
        )
        face_offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int64)
        face_edges = np.asarray(connectivity.face_edge_values, dtype=np.int64)
        return (
            np.unique(
                np.concatenate(
                    [
                        face_edges[face_offsets[face] : face_offsets[face + 1]]
                        for face in selected_faces
                    ]
                )
            )
            if selected_faces.size
            else np.zeros((0,), dtype=np.int64)
        )
    if control.scope.entity_dimension == 3:
        cells = np.flatnonzero(np.isin(construction.cell_regions, identifiers))
        cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        face_edge_offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int64)
        face_edges = np.asarray(connectivity.face_edge_values, dtype=np.int64)
        selected_faces = (
            np.unique(
                np.concatenate(
                    [
                        cell_faces[cell_face_offsets[cell] : cell_face_offsets[cell + 1]]
                        for cell in cells
                    ]
                )
            )
            if cells.size
            else np.zeros((0,), dtype=np.int64)
        )
        return (
            np.unique(
                np.concatenate(
                    [
                        face_edges[face_edge_offsets[face] : face_edge_offsets[face + 1]]
                        for face in selected_faces
                    ]
                )
            )
            if selected_faces.size
            else np.zeros((0,), dtype=np.int64)
        )
    raise MeshingFailure(
        MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        "Polyhedral uniform size controls must scope PLC facets or regions.",
        provider_code="native_polyhedral_compliance",
        stage="preparation",
    )


def _sqrt_lower(value: Fraction, /) -> float:
    if value <= 0:
        return 0.0
    represented = float(value)
    if not isfinite(represented):
        raise ValueError("Exact edge length is not representable in binary64.")
    result = float(np.sqrt(represented))
    while Fraction(result) ** 2 > value:
        result = float(np.nextafter(result, -np.inf))
    return result


def _sqrt_upper(value: Fraction, /) -> float:
    if value < 0:
        raise ValueError("Squared edge length must be nonnegative.")
    if value == 0:
        return 0.0
    represented = float(value)
    if represented == 0.0 or not isfinite(represented):
        raise ValueError("Exact edge length is not representable in binary64.")
    result = float(np.nextafter(np.sqrt(represented), np.inf))
    if not isfinite(result):
        raise ValueError("Exact edge length is not representable in binary64.")
    while Fraction(result) ** 2 < value:
        result = float(np.nextafter(result, np.inf))
        if not isfinite(result):
            raise ValueError("Exact edge length is not representable in binary64.")
    return result


def _polyhedral_compliance(
    construction: PolyhedralConstruction,
    specification: VolumeMeshingSpec,
    /,
) -> MeshingComplianceReport:
    coordinates = construction.coordinate_source.source_coordinates()
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral compliance requires polyhedral connectivity.")
    edge_rows = np.asarray(connectivity.edges, dtype=np.int64)
    lower_lengths = np.empty((edge_rows.shape[0],), dtype=np.float64)
    upper_lengths = np.empty_like(lower_lengths)
    for edge in range(edge_rows.shape[0]):
        first, second = int(edge_rows[edge, 0]), int(edge_rows[edge, 1])
        delta = tuple(
            coordinates[int(first)][axis] - coordinates[int(second)][axis]
            for axis in range(3)
        )
        squared = sum((value * value for value in delta), Fraction(0))
        lower_lengths[edge] = _sqrt_lower(squared)
        upper_lengths[edge] = _sqrt_upper(squared)
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    policy = specification.size_compliance
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append(f"unsupported_size_control:{control.control_id}")
            continue
        rows = _size_scope_edges(construction, control)
        if not rows.size:
            issues.append(f"empty_size_scope:{control.control_id}")
            continue
        lower, upper = lower_lengths[rows], upper_lengths[rows]
        requested.append((f"size:{control.control_id}:target", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"size:{control.control_id}:minimum", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"size:{control.control_id}:maximum", control.maximum_size))
        for statistic in policy.target_statistics:
            quantile = 0.5 if statistic == "p50" else 0.95
            lower_value = float(np.quantile(lower, quantile))
            upper_value = float(np.quantile(upper, quantile))
            achieved.append(
                (f"size:{control.control_id}:{statistic}_edge_lower", lower_value)
            )
            achieved.append(
                (f"size:{control.control_id}:{statistic}_edge_upper", upper_value)
            )
            if control.strength is SizeControlStrength.HARD:
                tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(
                    control.target_size
                )
                if (
                    lower_value < control.target_size - tolerance
                    or upper_value > control.target_size + tolerance
                ):
                    issues.append(f"size:{control.control_id}:{statistic}")
        if control.minimum_size is not None:
            achieved.append(
                (f"size:{control.control_id}:minimum_edge", float(np.min(lower)))
            )
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(
                control.minimum_size
            )
            if (
                control.strength is SizeControlStrength.HARD
                and np.min(lower) < control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
        if control.maximum_size is not None:
            achieved.append(
                (f"size:{control.control_id}:maximum_edge", float(np.max(upper)))
            )
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(
                control.maximum_size
            )
            if (
                control.strength is SizeControlStrength.HARD
                and np.max(upper) > control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
    if any(block.cell_kind != "polyhedron" for block in construction.mesh.blocks):
        issues.append("cell_family")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _face_group_connected(
    connectivity: PolyhedralConnectivity,
    faces: np.ndarray,
    /,
) -> bool:
    if faces.size <= 1:
        return True
    parent = np.arange(faces.size, dtype=np.int64)
    local = {int(face): row for row, face in enumerate(faces)}
    offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_edge_values, dtype=np.int64)
    first_by_edge: dict[int, int] = {}
    for face in faces:
        row = local[int(face)]
        for edge in values[offsets[int(face)] : offsets[int(face) + 1]]:
            edge_row = int(edge)
            prior = first_by_edge.get(edge_row)
            if prior is None:
                first_by_edge[edge_row] = row
            else:
                a, b = prior, row
                while parent[a] != a:
                    a = int(parent[a])
                while parent[b] != b:
                    b = int(parent[b])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = set()
    for row in range(parent.size):
        current = row
        while parent[current] != current:
            current = int(parent[current])
        roots.add(current)
    return len(roots) == 1


def _polyhedral_organization(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    construction: PolyhedralConstruction,
    associations: tuple[GeometryAssociation, ...],
    /,
) -> NativeMeshOrganization:
    mesh, connectivity = construction.mesh, construction.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral organization requires polyhedral connectivity.")
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    edge_ids = np.asarray(connectivity.edge_global_ids, dtype=np.int64)
    regions = source.complex.region_ids
    controls = {control.region_name: control for control in specification.region_controls}
    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    zones: list[MeshZone] = []
    for region, name in enumerate(regions):
        members = cell_ids[construction.cell_regions == region]
        if not members.size:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                f"PLC region {name!r} has no restricted-power cells.",
                provider_code="native_polyhedral_organization",
                stage="organization",
            )
        control, seed = controls.get(name), seeds.get(name)
        zones.append(
            MeshZone(
                name,
                MeshZoneRole.REGION,
                _mesh_scope(mesh, 3, members),
                material_id=(
                    control.material_id
                    if control is not None
                    else seed.material_id
                    if seed is not None
                    else None
                ),
                region_role=(
                    control.role
                    if control is not None
                    else seed.role
                    if seed is not None
                    else None
                ),
            )
        )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    facets = construction.face_source_facets
    patches: list[MeshPatch] = []
    for facet in sorted(set(int(value) for value in facets if value >= 0)):
        rows = np.flatnonzero(facets == facet)
        pair = np.asarray(source.complex.facet_regions, dtype=np.int64)[facet]
        adjacent = tuple(zone_ids[regions[int(region)]] for region in pair if region >= 0)
        patch_id = source.complex.facet_ids[facet]
        patches.append(
            MeshPatch(
                f"facet:{patch_id}",
                _mesh_scope(mesh, 2, face_ids[rows]),
                connected=_face_group_connected(connectivity, rows),
                adjacent_zone_ids=adjacent,
            )
        )
    labels: list[MeshLabel] = []
    for name, rows in (
        ("interface", construction.interface_faces),
        ("internal_sheet", construction.sheet_faces),
    ):
        if rows.size:
            labels.append(MeshLabel(name, _mesh_scope(mesh, 2, face_ids[rows])))
    edge_association = associations[1]
    explicit_segments = set(source.complex.segment_ids)
    curve_rows = np.asarray(
        [
            row
            for row, (identifier, role) in enumerate(
                zip(
                    edge_association.source_entity_ids,
                    edge_association.source_roles,
                    strict=True,
                )
            )
            if role is GeometrySourceEntityRole.EDGE and identifier in explicit_segments
        ],
        dtype=np.int64,
    )
    if curve_rows.size:
        labels.append(
            MeshLabel("internal_curve", _mesh_scope(mesh, 1, edge_ids[curve_rows]))
        )
    site_weights = construction.site_weights[construction.cell_sites]
    site_parent_rows = construction.site_parents[construction.cell_sites]
    attributes = [
        MeshAttribute(
            "power_site_index",
            MeshAttributeRole.MARKER,
            _mesh_scope(mesh, 3, cell_ids),
            construction.cell_sites,
        ),
        MeshAttribute(
            "power_component_index",
            MeshAttributeRole.MARKER,
            _mesh_scope(mesh, 3, cell_ids),
            construction.component_ids,
        ),
        MeshAttribute(
            "power_site_parent_index",
            MeshAttributeRole.MARKER,
            _mesh_scope(mesh, 3, cell_ids),
            site_parent_rows,
        ),
        MeshAttribute(
            "power_site_weight",
            MeshAttributeRole.USER,
            _mesh_scope(mesh, 3, cell_ids),
            site_weights,
        ),
    ]
    for site, site_id in enumerate(construction.site_ids):
        cells = np.flatnonzero(construction.cell_sites == site)
        if not cells.size:
            continue
        cell_scope = _mesh_scope(mesh, 3, cell_ids[cells])
        labels.append(MeshLabel(f"power-site:{site_id}", cell_scope))
        parent_id = construction.site_parent_ids[site]
        if parent_id is not None:
            labels.append(
                MeshLabel(f"power-site-parent:{site_id}:{parent_id}", cell_scope)
            )
        for component in np.unique(construction.component_ids[cells]):
            component_cells = cells[construction.component_ids[cells] == component]
            labels.append(
                MeshLabel(
                    f"power-component:{site_id}:{int(component)}",
                    _mesh_scope(mesh, 3, cell_ids[component_cells]),
                )
            )
    for control in specification.patch_controls:
        requested_facets = np.asarray(control.scope.entity_ids, dtype=np.int64)
        rows = face_ids[np.isin(facets, requested_facets)]
        if control.required and not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Required patch {control.name!r} has no restricted-power faces.",
                provider_code="native_polyhedral_organization",
                stage="organization",
                entity_ids=tuple(map(int, requested_facets)),
            )
        if rows.size:
            patches.append(
                MeshPatch(
                    control.name,
                    _mesh_scope(mesh, 2, rows),
                    adjacent_zone_ids=tuple(
                        zone_ids[name] for name in control.adjacent_region_names
                    ),
                )
            )
    return NativeMeshOrganization(
        patches=tuple(patches),
        zones=tuple(zones),
        labels=tuple(labels),
        attributes=tuple(attributes),
        associations=associations,
    )


@final
class _PreparedNativePolyhedral(StrictModule, NonTrainableState):
    source: NativePlcSource | NativePolyhedralSource
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    schedule: NativePolyhedralSchedule = eqx.field(static=True)

    def __init__(
        self,
        source: NativePlcSource | NativePolyhedralSource,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        schedule: NativePolyhedralSchedule,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.schedule = schedule

    def execute(self, /) -> NativeMeshConstruction:
        construction = generate_polyhedral_volume(
            self.source,
            self.specification,
            self.schedule,
            volume_schedule=self.options.volume_schedule,
            frame_id=self.coordinate_contract.spatial_id,
        )
        compliance = _polyhedral_compliance(construction, self.specification)
        mapped_source = (
            self.source.mapped_domain is not None
            or self.source.complex.geometry is not None
        )
        tolerance = (
            self.specification.maximum_deviation
            if self.specification.maximum_deviation is not None
            else self.source.maximum_deviation
        )
        fidelity_source = (
            self.source.fidelity_source
            if isinstance(self.source, NativePolyhedralSource)
            else None
        )
        if mapped_source and isinstance(self.source, NativePlcSource):
            fidelity_source = self.source.boundary_query()
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("mapped_volume" if mapped_source else "volume_plc"),
            self.source.source_id,
            self.source.source_revision,
            domain=construction.carrier.domain,
            cell_regions=construction.cell_regions,
            fidelity_source=fidelity_source if mapped_source else None,
            fidelity_tolerance=tolerance if mapped_source else None,
        )
        associations = _polyhedral_associations(self.source, construction)
        if self.specification.periodic_constraints:
            if not isinstance(self.source, NativePolyhedralSource):
                raise TypeError(
                    "Periodic restricted-power generation requires NativePolyhedralSource."
                )
            rebound = _rebind_periodic_constraints(
                self.source,
                construction.mesh,
                associations,
                self.specification,
            )
            validate_periodic_mesh_constraints(
                construction.mesh,
                self.source.source_id,
                self.source.source_revision,
                rebound,
            )
        _require_protected_feature_witnesses(
            self.source, self.specification, associations
        )
        organization = _polyhedral_organization(
            self.source, self.specification, construction, associations
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=construction.geometry,
            organization=organization,
        )


@checked
def prepare_native_polyhedral(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativePolyhedral:
    issues = polyhedral_support_issues(source, specification)
    if options.route != "plc_restricted_power":
        issues = (*issues, "the plc_restricted_power route")
    if issues:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(issues),
            provider_code="native_polyhedral_preflight",
            stage="source_inspection",
        )
    if options.polyhedral_schedule is None:
        raise RuntimeError("The polyhedral route must own its numerical schedule.")
    schedule = effective_polyhedral_schedule(specification, options.polyhedral_schedule)
    return _PreparedNativePolyhedral(
        source, specification, options, coordinate_contract, schedule
    )


__all__ = ["prepare_native_polyhedral", "polyhedral_support_issues"]
