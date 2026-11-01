#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native transfinite-block and profile-sweep construction routes."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import final

import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._cell_complex import PolygonalConnectivity, PolyhedralConnectivity
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._reference_cell import reference_cell_topology
from ...discretization.fem._reference import FiniteElementSpec
from ...geometry._curve_boundary import ParametricCurveBoundarySource
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
)
from ...typing import checked
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import TransfiniteCurveControl
from .._multiblock import (
    glue_structured_blocks,
    logical_face_vertices,
    MultiblockConstruction,
)
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshZone,
    MeshZoneRole,
)
from .._result import MeshingComplianceReport
from .._scope import _SourceEntityScopes, MeshingEntityKind, MeshingScope
from .._sizing import SizeControlStrength, UniformSizeControl
from .._structured import generate_structured_block, StructuredConstruction
from .._sweep import generate_sweep, SweepConstruction
from ._native_options import NativeMeshingOptions
from ._native_schedules import NativeStructuredSchedule
from ._native_sources import NativeStructuredSource, NativeSweepSource


type _SweepExactPoint3 = tuple[Fraction, Fraction, Fraction]
type _SweepFaceKey = tuple[int, ...]
type _SweepFaceSide = tuple[int, int, int, int]


def _specification_scope(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec, /
) -> MeshingScope:
    return (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )


def _family_issues(
    actual: set[str],
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> list[str]:
    policy = specification.target.cell_families
    issues: list[str] = []
    if not set(policy.required).issubset(actual):
        issues.append("required_cell_families")
    admitted = set((*policy.required, *policy.preferred, *policy.allowed_transitions))
    if not actual.issubset(admitted):
        issues.append("undeclared_cell_families")
    if not policy.allow_mixed and len(actual) != 1:
        issues.append("pure_family_policy")
    return issues


def _source_scope_keys(
    source: NativeStructuredSource | NativeSweepSource,
    scope: MeshingScope,
    /,
) -> tuple[str, ...] | None:
    if scope.entity_kind is not MeshingEntityKind.GEOMETRY:
        return None
    for candidate in (source, source.domain, source.fidelity_source):
        if not isinstance(candidate, _SourceEntityScopes):
            continue
        if (candidate.source_id, candidate.source_revision) != (
            scope.source_id,
            scope.source_revision,
        ):
            continue
        if candidate.scope_entity_set_id(scope.entity_dimension) != scope.entity_set_id:
            continue
        identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
        bank_ids = np.asarray(
            candidate.scope_entity_ids(scope.entity_dimension), dtype=np.int64
        )
        if not np.all(np.isin(identifiers, bank_ids)):
            return None
        keys = []
        for identifier in identifiers:
            entity_id, occurrences = candidate.source_entity_key_for_scope(
                scope.entity_dimension, int(identifier)
            )
            if not entity_id.strip() or any(not path for path in occurrences):
                return None
            keys.append(entity_id)
        return tuple(keys)
    return None


def _source_scope_is_bound(
    source: NativeStructuredSource | NativeSweepSource,
    scope: MeshingScope,
    /,
) -> bool:
    return _source_scope_keys(source, scope) is not None


def structured_support_issues(
    source: NativeStructuredSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> list[str]:
    """Return structured-route semantics that its declared blocks cannot honor."""
    issues: list[str] = []
    if source.maximum_deviation is None and specification.maximum_deviation is None:
        issues.append("an explicit source-fidelity tolerance")
    target = specification.target
    expected_dimension = target.topological_dimension
    expected_fill = VolumeFillStrategy.MULTIZONE
    if isinstance(specification, VolumeMeshingSpec) and specification.fill_strategy is not expected_fill:
        issues.append("multizone volume fill strategy")
    if expected_dimension not in (2, 3):
        issues.append("two- or three-dimensional structured blocks")
    if isinstance(specification, VolumeMeshingSpec) and expected_dimension == 3 and not isinstance(
        source.domain, PiecewiseLinearDomain
    ):
        issues.append("an authoritative source volume domain for coverage")
    if expected_dimension == target.ambient_dimension == 2 and (
        not isinstance(source.domain, PiecewiseLinearDomain)
        or source.domain.ambient_dimension != 2
    ):
        issues.append("an authoritative planar source domain for coverage")
    if any(len(block.intervals) != expected_dimension for block in source.blocks):
        issues.append("block dimensions matching the target")
    family = "quadrilateral" if expected_dimension == 2 else "hexahedron"
    issues.extend(_family_issues({family}, specification))
    scope = _specification_scope(specification)
    if _source_scope_keys(source, scope) is None:
        issues.append("a source-owned entity bank for the structured scope")
    if any(not interface.conforming for interface in source.interfaces):
        issues.append("conforming interfaces for one canonical mesh topology")
    if any(not isinstance(control, UniformSizeControl) for control in specification.size_controls):
        issues.append("uniform physical size controls on the complete structured scope")
    if any(
        isinstance(control, UniformSizeControl)
        and (
            control.scope.entity_set_id != scope.entity_set_id
            or not np.array_equal(
                np.asarray(control.scope.entity_ids),
                np.asarray(scope.entity_ids),
            )
            or not _source_scope_is_bound(source, control.scope)
        )
        for control in specification.size_controls
    ):
        issues.append("uniform size controls bound to the complete source scope")
    if specification.periodic_constraints or specification.layer_controls:
        issues.append("periodic and layer controls outside the explicit block declaration")
    if isinstance(specification, SurfaceMeshingSpec):
        if specification.region_controls or specification.patch_controls:
            issues.append("volume-region and facet controls on a surface-only block request")
        if specification.protected_features:
            issues.append("protected feature scopes not bound to a transfinite source boundary")
        if specification.maximum_normal_deviation is not None:
            issues.append("normal-deviation compliance not represented by the declared block contract")
        if specification.background_metric is not None:
            issues.append("background metrics not represented by the declared block contract")
    if isinstance(specification, VolumeMeshingSpec):
        if any(
            _source_scope_keys(source, control.scope) is None
            for control in specification.region_controls
        ):
            issues.append("source-owned entity banks for structured region controls")
        if specification.patch_controls or specification.protected_features:
            issues.append("volume facet/feature scopes not bound to transfinite source boundaries")
        if specification.region_seeds or specification.hole_seeds:
            issues.append("region and hole decomposition outside the declared block partition")
    return list(dict.fromkeys(issues))


def sweep_support_issues(
    source: NativeSweepSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    """Return volume controls that the exact declared sweep cannot realize."""
    issues: list[str] = []
    if source.maximum_deviation is None and specification.maximum_deviation is None:
        issues.append("an explicit source-fidelity tolerance")
    target = specification.target
    if specification.fill_strategy is not VolumeFillStrategy.SWEEP:
        issues.append("sweep volume fill strategy")
    if target.topological_dimension != 3 or target.ambient_dimension != 3:
        issues.append("three-dimensional sweep cells in three-dimensional coordinates")
    if not isinstance(source.domain, (PiecewiseLinearDomain, MappedReferenceDomain)):
        issues.append("an authoritative swept source volume domain for coverage")
    if isinstance(source.domain, PiecewiseLinearDomain):
        query = source.fidelity_source
        if source.domain_vertex_map is None:
            issues.append("an explicit source-domain vertex/profile-vertex/station correspondence")
        if (
            not isinstance(query, PiecewiseLinearBoundarySource)
            or query.domain.domain_id != source.domain.domain_id
            or query.entity_dimension != 2
            or not query.is_full_scope
        ):
            issues.append("a complete PWL facet-chart query bound to the swept source domain")
    if isinstance(source.domain, MappedReferenceDomain) and source.root_correspondence is None:
        issues.append("root-cell/reference-map correspondence for the mapped sweep")
    actual = {
        "prism" if block.cell_kind == "triangle" else "hexahedron"
        for block in source.profile.blocks
        if block.cell_kind in ("triangle", "quadrilateral")
    }
    if len(actual) == 0 or any(
        block.cell_kind not in ("triangle", "quadrilateral") for block in source.profile.blocks
    ):
        issues.append("triangle-to-prism or quadrilateral-to-hexahedron profile cells")
    issues.extend(_family_issues(actual, specification))
    scope = specification.boundary_scope
    if _source_scope_keys(source, scope) is None:
        issues.append("a source-owned entity bank for the sweep boundary scope")
    if (scope.source_id, scope.source_revision) != (source.source_id, source.source_revision):
        issues.append("the declared source revision and sweep boundary scope")
    if any(not isinstance(control, UniformSizeControl) for control in specification.size_controls):
        issues.append("uniform physical size controls on the complete swept scope")
    if any(
        isinstance(control, UniformSizeControl)
        and (
            control.scope.entity_set_id != scope.entity_set_id
            or not np.array_equal(
                np.asarray(control.scope.entity_ids),
                np.asarray(scope.entity_ids),
            )
            or not _source_scope_is_bound(source, control.scope)
        )
        for control in specification.size_controls
    ):
        issues.append("sweep size controls bound to the complete source scope")
    if specification.periodic_constraints or specification.layer_controls:
        issues.append("periodic and layer controls outside the explicit sweep map")
    if any(
        _source_scope_keys(source, control.scope) is None
        for control in specification.region_controls
    ):
        issues.append("source-owned entity banks for sweep region controls")
    if specification.patch_controls or specification.protected_features:
        issues.append("volume facet/feature scopes not bound to the swept source profile")
    if specification.region_seeds or specification.hole_seeds:
        issues.append("region and hole decomposition outside the declared swept profile")
    has_non_polynomial_profile = False
    has_profile_order_mismatch = False
    for element in source.profile_geometry.elements:
        if not isinstance(element, FiniteElementSpec):
            has_non_polynomial_profile = True
            continue
        if element.degree != target.geometry_order:
            has_profile_order_mismatch = True
    if has_non_polynomial_profile:
        issues.append("polynomial H1 profile coordinate maps")
    if has_profile_order_mismatch:
        issues.append("profile geometry order equal to the declared target order")
    return list(dict.fromkeys(issues))


def _mesh_scope(
    mesh: CellMesh,
    dimension: int,
    entity_ids: ArrayLike,
    /,
) -> MeshingScope:
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        np.asarray(entity_ids, dtype=np.int64),
    )


def _entity_vertex_rows(mesh: CellMesh, dimension: int, /) -> tuple[np.ndarray, ...]:
    connectivity = mesh.connectivity
    if dimension == 1:
        if isinstance(connectivity, PolygonalConnectivity):
            values = np.asarray(connectivity.edges, dtype=np.int64)
        elif isinstance(connectivity, HexahedralConnectivity):
            values = np.asarray(connectivity.edges, dtype=np.int64)
        elif isinstance(connectivity, PolyhedralConnectivity):
            values = np.asarray(connectivity.edges, dtype=np.int64)
        else:
            raise ValueError("The structured/sweep route requires explicit edge incidence.")
        return tuple(values[row] for row in range(values.shape[0]))
    if dimension == 2:
        if isinstance(connectivity, HexahedralConnectivity):
            values = np.asarray(connectivity.faces, dtype=np.int64)
            return tuple(values[row] for row in range(values.shape[0]))
        if isinstance(connectivity, PolyhedralConnectivity):
            offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            return tuple(values[offsets[row] : offsets[row + 1]] for row in range(offsets.size - 1))
    raise ValueError(f"No vertex incidence is available for mesh entity dimension {dimension}.")


def _sweep_face_side_lookup(
    mesh: CellMesh, /
) -> tuple[
    tuple[np.ndarray, ...],
    np.ndarray,
    dict[_SweepFaceKey, list[_SweepFaceSide]],
]:
    faces = _entity_vertex_rows(mesh, 2)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if len(faces) != face_ids.size:
        raise ValueError("Sweep face vertices must align with stable face identities.")
    face_keys = {
        tuple(sorted(int(vertex) for vertex in face)): row
        for row, face in enumerate(faces)
    }
    if len(face_keys) != len(faces):
        raise ValueError("Sweep output face vertex sets must be unique.")
    sides: dict[_SweepFaceKey, list[_SweepFaceSide]] = {}
    for block_index, block in enumerate(mesh.blocks):
        topology = reference_cell_topology(block.cell_kind)
        local_faces = tuple(
            tuple(int(vertex) for vertex in face)
            for face in topology.entities[2]
        )
        cells = np.asarray(block.vertices, dtype=np.int64)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        if cell_ids.shape != (cells.shape[0],):
            raise ValueError("Sweep block cell identities must align with cell rows.")
        for cell_row in range(cells.shape[0]):
            for local_face, face_vertices in enumerate(local_faces):
                key = tuple(
                    sorted(int(cells[cell_row, vertex]) for vertex in face_vertices)
                )
                if key not in face_keys:
                    raise ValueError("Sweep reference faces do not match the output face bank.")
                sides.setdefault(key, []).append(
                    (int(cell_ids[cell_row]), block_index, cell_row, local_face)
                )
    return faces, face_ids, sides


def _sweep_piecewise_boundary_witness(
    source: NativeSweepSource,
    construction: SweepConstruction,
    /,
) -> SourceBoundaryChartWitness:
    domain = source.domain
    query = source.fidelity_source
    if not isinstance(domain, PiecewiseLinearDomain) or not isinstance(
        query, PiecewiseLinearBoundarySource
    ):
        raise TypeError("PWL sweep witnesses require a PiecewiseLinearDomain and its PWL query.")
    if (
        query.domain.domain_id != domain.domain_id
        or query.entity_dimension != 2
        or not query.is_full_scope
        or source.domain_vertex_map is None
    ):
        raise ValueError(
            "PWL sweep witnesses require full source facets and explicit domain-vertex ancestry."
        )
    query.require_current()
    mesh = construction.mesh
    geometry = construction.geometry
    target_face_rows, target_face_ids, face_sides = _sweep_face_side_lookup(mesh)
    face_rows_by_triangle: dict[tuple[int, int, int], set[int]] = {}
    for face_row, face in enumerate(target_face_rows):
        face_vertices = tuple(int(vertex) for vertex in face)
        for first, second, third in combinations(face_vertices, 3):
            lower = min(first, second, third)
            upper = max(first, second, third)
            middle = first + second + third - lower - upper
            triangle_key = (lower, middle, upper)
            face_rows = face_rows_by_triangle.get(triangle_key)
            if face_rows is None:
                face_rows = set()
                face_rows_by_triangle[triangle_key] = face_rows
            face_rows.add(face_row)

    source_vertex_ids = np.asarray(source.profile.vertex_global_ids, dtype=np.int64)
    profile_row_by_id = {
        int(identifier): row for row, identifier in enumerate(source_vertex_ids)
    }
    output_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    output_row_by_id = {
        int(identifier): row for row, identifier in enumerate(output_vertex_ids)
    }
    layer_vertex_ids = np.asarray(construction.layer_vertex_ids, dtype=np.int64)
    domain_vertex_to_output_row: list[int] = []
    for profile_id, station in source.domain_vertex_map:
        profile_row = profile_row_by_id.get(profile_id)
        if profile_row is None or not 0 <= station < layer_vertex_ids.shape[0]:
            raise ValueError("PWL domain vertex ancestry leaves the retained sweep map.")
        output_id = int(layer_vertex_ids[station, profile_row])
        output_row = output_row_by_id.get(output_id)
        if output_row is None:
            raise ValueError("PWL domain vertex ancestry names an absent sweep vertex.")
        domain_vertex_to_output_row.append(output_row)

    reference_rows: list[tuple[_SweepExactPoint3, ...]] = []
    source_parameter_rows: list[
        tuple[tuple[Fraction, Fraction], ...]
    ] = []
    witness_cell_ids: list[int] = []
    witness_face_ids: list[int] = []
    witness_local_faces: list[int] = []
    chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    orientations: list[int] = []
    standard_source_triangle: tuple[
        tuple[Fraction, Fraction],
        tuple[Fraction, Fraction],
        tuple[Fraction, Fraction],
    ] = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for query_row, facet_row_value in enumerate(query.source_rows.tolist()):
        facet_row = int(facet_row_value)
        facet_vertices = tuple(
            int(vertex)
            for vertex in np.asarray(domain.facets[facet_row], dtype=np.int64)
        )
        target_vertices = tuple(
            domain_vertex_to_output_row[vertex] for vertex in facet_vertices
        )
        triangle_key = tuple(sorted(target_vertices))
        matching_faces = face_rows_by_triangle.get(triangle_key, set())
        if len(matching_faces) != 1:
            raise ValueError(
                "Each PWL source facet must map to one explicit sweep target face."
            )
        face_row = next(iter(matching_faces))
        face_loop = target_face_rows[face_row]
        face_key = tuple(sorted(int(vertex) for vertex in face_loop))
        sides = face_sides.get(face_key, ())
        if not sides:
            raise ValueError("A PWL source facet has no sweep target-cell side.")
        cell_id, block_index, cell_row, local_face = min(
            sides, key=lambda value: (value[0], value[1], value[3])
        )
        block = mesh.blocks[block_index]
        cell_vertices = np.asarray(block.vertices[cell_row], dtype=np.int64)
        local_vertex_by_row = {
            int(vertex_row): local_vertex
            for local_vertex, vertex_row in enumerate(cell_vertices.tolist())
        }
        reference = reference_cell_topology(block.cell_kind)
        reference_vertices = np.asarray(reference.vertices, dtype=np.float64)
        local_faces = tuple(
            tuple(int(vertex) for vertex in face)
            for face in reference.entities[2]
        )
        local_face_vertices = local_faces[local_face]
        if len(local_face_vertices) == 3:
            face_chart_corners = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
        elif len(local_face_vertices) == 4:
            face_chart_corners = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(1)),
            )
        else:
            raise ValueError("PWL sweep charts require triangular or quadrilateral target faces.")
        chart_by_local_vertex = dict(
            zip(local_face_vertices, face_chart_corners, strict=True)
        )
        target_reference_triangle: list[_SweepExactPoint3] = []
        target_chart_triangle = []
        for vertex_row in target_vertices:
            local_vertex = local_vertex_by_row.get(vertex_row)
            if local_vertex is None or local_vertex not in chart_by_local_vertex:
                raise ValueError("PWL source facet vertices do not match the target face.")
            target_reference_triangle.append(
                (
                    Fraction.from_float(
                        float(reference_vertices[local_vertex, 0])
                    ),
                    Fraction.from_float(
                        float(reference_vertices[local_vertex, 1])
                    ),
                    Fraction.from_float(
                        float(reference_vertices[local_vertex, 2])
                    ),
                )
            )
            target_chart_triangle.append(chart_by_local_vertex[local_vertex])
        first, second, third = target_chart_triangle
        signed_area = (
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )
        if signed_area == 0:
            raise ValueError("PWL sweep target chart is degenerate.")
        reference_rows.append(tuple(target_reference_triangle))
        source_parameter_rows.append(standard_source_triangle)
        witness_cell_ids.append(cell_id)
        witness_face_ids.append(int(target_face_ids[face_row]))
        witness_local_faces.append(local_face)
        chart_ids.append(query.chart_ids[query_row])
        source_entity_ids.append(query.source_entity_keys[query_row])
        occurrence_paths.append(query.source_occurrence_paths[query_row])
        orientations.append(1 if signed_area > 0 else -1)

    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(witness_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(witness_face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(witness_local_faces, dtype=np.int32),
        target_reference_vertices=np.asarray(reference_rows, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(reference_rows),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_parameter_rows, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _selected_entities(
    mesh: CellMesh,
    dimension: int,
    vertices: np.ndarray,
    /,
) -> np.ndarray:
    rows = _entity_vertex_rows(mesh, dimension)
    entity_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    selected = [
        int(entity_ids[index])
        for index, row in enumerate(rows)
        if row.size and np.all(np.isin(row, vertices))
    ]
    return np.asarray(selected, dtype=np.int64)


def _structured_curve_witness(
    source: NativeStructuredSource,
    constructions: tuple[StructuredConstruction, ...],
    combined: MultiblockConstruction,
    /,
) -> SourceBoundaryChartWitness | None:
    query = source.fidelity_source
    if combined.mesh.topological_dimension != 2:
        return None
    if not isinstance(query, ParametricCurveBoundarySource):
        raise ValueError("Structured surface fidelity requires authored curve charts.")
    mesh = combined.mesh
    source_blocks = {block.name: block for block in source.blocks}
    local_blocks = {item.mesh.blocks[0].name: item for item in constructions}
    interfaces = {
        face
        for interface in source.interfaces
        for face in (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        )
    }
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edges = {
        tuple(sorted(map(int, vertices))): int(identifier)
        for vertices, identifier in zip(_entity_vertex_rows(mesh, 1), edge_ids, strict=True)
    }
    query_rows = {entity: row for row, entity in enumerate(query.curve_entity_ids)}
    cell_ids, facet_ids, local_facets = [], [], []
    references, parameters, charts, entities, occurrences, orientations = [], [], [], [], [], []
    for block in mesh.blocks:
        construction = local_blocks[block.name]
        declared = source_blocks[block.name]
        topology = reference_cell_topology(block.cell_kind)
        reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
        local_cells = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
        for row, vertices in enumerate(np.asarray(block.vertices, dtype=np.int64)):
            logical = np.stack(
                np.unravel_index(local_cells[row], construction.logical_shape, order="C"),
                axis=-1,
            )
            for facet, corners in enumerate(topology.entities[1]):
                corner_rows = np.asarray(corners, dtype=np.int64)
                for axis in range(2):
                    values = logical[corner_rows, axis]
                    if not np.all(values == values[0]) or int(values[0]) not in (
                        0, construction.logical_shape[axis] - 1
                    ):
                        continue
                    side = int(values[0] != 0)
                    face = 2 * axis + side
                    if (block.name, face) in interfaces:
                        continue
                    control = declared.boundaries[face]
                    if not isinstance(control, TransfiniteCurveControl):
                        raise TypeError("Structured surface boundaries require curve controls.")
                    if control.source_id not in query_rows:
                        raise ValueError("A structured exterior curve lacks its authored source chart.")
                    source_row = query_rows[control.source_id]
                    lower, upper = map(Fraction, control.parameter_range)
                    fractions = tuple(
                        Fraction(int(value), construction.logical_shape[1 - axis] - 1)
                        for value in logical[corner_rows, 1 - axis]
                    )
                    if control.reversed:
                        fractions = tuple(1 - value for value in fractions)
                    parameter = tuple((lower + (upper - lower) * value,) for value in fractions)
                    cell_ids.append(int(block.global_ids[row]))
                    facet_ids.append(edges[tuple(sorted(map(int, vertices[corner_rows])))])
                    local_facets.append(facet)
                    references.append(reference_vertices[corner_rows])
                    parameters.append(parameter)
                    charts.append(query.chart_ids[source_row])
                    entities.append(query.curve_entity_ids[source_row])
                    occurrences.append(query.occurrence_paths[source_row])
                    orientations.append(1 if parameter[1] > parameter[0] else -1)
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(combined.geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.query_id,
        chart_dimension=1,
        reference_dimension=2,
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(facet_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_facets, dtype=np.int32),
        target_reference_vertices=np.asarray(references, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            tuple(
                tuple(tuple(Fraction(float(value)) for value in point) for point in chart)
                for chart in references
            )
        ),
        source_chart_ids=tuple(charts),
        source_entity_ids=tuple(entities),
        occurrence_paths=tuple(occurrences),
        source_parameters=np.asarray(parameters, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(tuple(parameters)),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _structured_organization(
    source: NativeStructuredSource,
    constructions: tuple[StructuredConstruction, ...],
    combined: MultiblockConstruction,
    fidelity_witness: SourceBoundaryChartWitness | None,
    /,
) -> NativeMeshOrganization:
    mesh = combined.mesh
    block_vertices = dict(combined.block_vertex_ids)
    source_blocks = {block.name: block for block in source.blocks}
    constructions_by_name = {
        item.mesh.blocks[0].name: item for item in constructions
    }
    output_blocks = {block.name: block for block in mesh.blocks}
    labels: list[MeshLabel] = []
    attributes: list[MeshAttribute] = []
    for name, construction in constructions_by_name.items():
        source_block = source_blocks[name]
        output_block = output_blocks[name]
        cell_scope = _mesh_scope(mesh, mesh.topological_dimension, output_block.global_ids)
        labels.append(MeshLabel(f"structured-block:{source_block.block_id}", cell_scope))
        local_vertices = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
        logical_indices = np.stack(
            np.unravel_index(local_vertices, construction.logical_shape, order="C"),
            axis=-1,
        ).astype(np.int32)
        attributes.append(
            MeshAttribute(
                f"structured-block:{source_block.block_id}:logical-cell-vertices",
                MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                cell_scope,
                logical_indices,
            )
        )
        vertex_map = np.asarray(block_vertices[name], dtype=np.int64)
        scoped_vertices = np.unique(vertex_map)
        local_for_global = np.empty((scoped_vertices.size,), dtype=np.int64)
        positions = {int(identifier): index for index, identifier in enumerate(vertex_map)}
        for row, identifier in enumerate(scoped_vertices):
            local_for_global[row] = positions[int(identifier)]
        vertex_scope = _mesh_scope(mesh, 0, scoped_vertices)
        logical_points = np.stack(
            np.unravel_index(local_for_global, construction.logical_shape, order="C"),
            axis=-1,
        ).astype(np.int32)
        attributes.append(
            MeshAttribute(
                f"structured-block:{source_block.block_id}:logical-vertices",
                MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                vertex_scope,
                logical_points,
            )
        )
        boundary_dimension = mesh.topological_dimension - 1
        for face, source_id in enumerate(construction.boundary_source_ids):
            local_face = logical_face_vertices(construction.logical_shape, face).reshape(-1)
            global_face = vertex_map[local_face]
            selected = _selected_entities(mesh, boundary_dimension, global_face)
            if selected.size:
                labels.append(
                    MeshLabel(
                        f"structured-boundary:{source_block.block_id}:{face}:{source_id}",
                        _mesh_scope(mesh, boundary_dimension, selected),
                    )
                )
    for interface in source.interfaces:
        first = constructions_by_name[interface.first_block]
        face = logical_face_vertices(first.logical_shape, interface.first_face).reshape(-1)
        vertices = np.asarray(block_vertices[interface.first_block], dtype=np.int64)[face]
        selected = _selected_entities(mesh, mesh.topological_dimension - 1, vertices)
        if selected.size:
            labels.append(
                MeshLabel(
                    f"structured-interface:{interface.control_id}",
                    _mesh_scope(mesh, mesh.topological_dimension - 1, selected),
                )
            )
    associations: tuple[GeometryAssociation, ...] = ()
    if fidelity_witness is not None:
        query = source.fidelity_source
        if not isinstance(query, ParametricCurveBoundarySource):
            raise TypeError("Structured curve associations require their chart query.")
        curves = dict(zip(query.chart_ids, query.curves, strict=True))
        cells = {
            int(identifier): (block, row)
            for block in mesh.blocks
            for row, identifier in enumerate(block.global_ids)
        }
        edge_rows = dict(
            zip(map(int, mesh.entity_set(1).entity_ids), _entity_vertex_rows(mesh, 1), strict=True)
        )
        intervals = tuple(
            (
                min(Fraction(*chart[0][0]), Fraction(*chart[1][0])),
                max(Fraction(*chart[0][0]), Fraction(*chart[1][0])),
            )
            for chart in fidelity_witness.source_parameter_rational_tokens
        )
        edge_orientations = []
        residuals = []
        for identifier, edge_id, facet, chart, parameters in zip(
            fidelity_witness.cell_global_ids,
            fidelity_witness.entity_global_ids,
            fidelity_witness.local_entity_indices,
            fidelity_witness.source_chart_ids,
            fidelity_witness.source_parameters,
            strict=True,
        ):
            block, row = cells[int(identifier)]
            corners = reference_cell_topology(block.cell_kind).entities[1][int(facet)]
            target = np.asarray(mesh.coordinates)[
                np.asarray(block.vertices)[row, np.asarray(corners)]
            ]
            source_points = np.asarray(curves[chart].evaluate(parameters[:, 0]))
            residuals.append(float(np.max(np.linalg.norm(target - source_points, axis=-1))))
            direction = 1 if parameters[1, 0] > parameters[0, 0] else -1
            if int(block.vertices[row, corners[0]]) != int(edge_rows[int(edge_id)][0]):
                direction = -direction
            edge_orientations.append(direction)
        associations = (
            GeometryAssociation(
                GeometryAssociationKind.CURVE,
                source.source_id,
                source.source_revision,
                mesh.entity_set(1).entity_set_id,
                fidelity_witness.entity_global_ids,
                fidelity_witness.source_entity_ids,
                np.asarray(residuals, dtype=np.float64),
                exact=False,
                source_roles=(GeometrySourceEntityRole.EDGE,) * len(residuals),
                source_occurrence_paths=fidelity_witness.occurrence_paths,
                parameters=np.asarray(intervals, dtype=np.float64),
                source_parameter_intervals=intervals,
                orientations=np.asarray(edge_orientations, dtype=np.int8),
            ),
        )
    return NativeMeshOrganization(
        labels=tuple(labels),
        attributes=tuple(attributes),
        associations=associations,
    )


def _sweep_organization(
    source: NativeSweepSource,
    specification: VolumeMeshingSpec,
    construction: SweepConstruction,
    /,
) -> NativeMeshOrganization:
    mesh = construction.mesh
    layers = construction.stations.size - 1
    source_vertices = construction.source_vertex_ids
    parent_cells = dict(construction.cell_parent_ids)
    source_blocks = {block.name: block for block in source.profile.blocks}
    output_blocks = {block.name: block for block in mesh.blocks}
    labels: list[MeshLabel] = []
    attributes: list[MeshAttribute] = []
    for source_name, source_block in source_blocks.items():
        name = f"sweep:{source_name}"
        output_block = output_blocks[name]
        cell_scope = _mesh_scope(mesh, 3, output_block.global_ids)
        labels.append(MeshLabel(f"sweep-profile-block:{construction.source_mesh_id}:{source_name}", cell_scope))
        profile_corners = source_vertices[np.asarray(source_block.vertices, dtype=np.int64)]
        layered_corners = np.tile(profile_corners, (layers, 1))
        corner_ancestry = np.concatenate((layered_corners, layered_corners), axis=1)
        attributes.extend(
            (
                MeshAttribute(
                    f"sweep:{source_name}:source-profile-cell-id",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    cell_scope,
                    parent_cells[name],
                ),
                MeshAttribute(
                    f"sweep:{source_name}:source-profile-vertex-id",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    cell_scope,
                    corner_ancestry,
                ),
                MeshAttribute(
                    f"sweep:{source_name}:layer-index",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    cell_scope,
                    np.repeat(np.arange(layers, dtype=np.int32), source_block.cell_count),
                ),
                MeshAttribute(
                    f"sweep:{source_name}:station-interval",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    cell_scope,
                    np.repeat(
                        np.stack((construction.stations[:-1], construction.stations[1:]), axis=-1),
                        source_block.cell_count,
                        axis=0,
                    ),
                ),
            )
        )
    vertex_count = mesh.coordinates.shape[0]
    vertex_scope = _mesh_scope(mesh, 0, np.asarray(mesh.vertex_global_ids, dtype=np.int64))
    node_layers = vertex_count // source_vertices.size
    attributes.extend(
        (
            MeshAttribute(
                f"sweep:{construction.source_mesh_id}:source-profile-vertex-id",
                MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                vertex_scope,
                np.tile(source_vertices, node_layers),
            ),
            MeshAttribute(
                f"sweep:{construction.control_id}:station-index",
                MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                vertex_scope,
                np.repeat(np.arange(node_layers, dtype=np.int32), source_vertices.size),
            ),
        )
    )
    if source.control.closed:
        seam_vertices = np.asarray(construction.layer_vertex_ids[0], dtype=np.int64)
        seam_scope = _mesh_scope(mesh, 0, seam_vertices)
        labels.append(MeshLabel(f"sweep-periodic-seam:{construction.control_id}", seam_scope))
        seam_faces = _selected_entities(mesh, 2, seam_vertices)
        if seam_faces.size:
            labels.append(
                MeshLabel(
                    f"sweep-periodic-seam-facets:{construction.control_id}",
                    _mesh_scope(mesh, 2, seam_faces),
                )
            )
    zones = _region_zones(source, specification, mesh)
    return NativeMeshOrganization(
        labels=tuple(labels),
        attributes=tuple(attributes),
        zones=zones,
    )


def _domain_region_ids(
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    /,
) -> tuple[str, ...]:
    if isinstance(domain, PiecewiseLinearDomain):
        return tuple(domain.region_ids)
    if isinstance(domain, MappedReferenceDomain):
        return tuple(domain.reference_domain.region_ids)
    return ()


def _reference_corner_indices(
    element: FiniteElementSpec, cell_kind: str, /
) -> tuple[int, ...]:
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    corners: list[int] = []
    for vertex in reference_cell_topology(cell_kind).vertices:
        found = np.flatnonzero(np.all(nodes == np.asarray(vertex, dtype=np.float64), axis=1))
        if found.size != 1:
            raise ValueError("The structured coordinate element must contain every cell corner exactly once.")
        corners.append(int(found[0]))
    return tuple(corners)


def _quadrature_edge_length(
    element: FiniteElementSpec,
    dofs: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    /,
    *,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    def integrate(lower: float, upper: float) -> float:
        parameter = lower + (nodes + 1.0) * (upper - lower) * 0.5
        points = start[None, :] + parameter[:, None] * direction[None, :]
        _values, gradients = element.tabulate(points)
        jacobian = np.einsum("qjd,jk->qkd", np.asarray(gradients), dofs)
        tangent = np.einsum("qkd,d->qk", jacobian, direction)
        speed = np.linalg.norm(tangent, axis=-1)
        return float(np.dot(weights, speed) * (upper - lower) * 0.5)

    total = 0.0
    error_total = 0.0
    pending = [(0.0, 1.0, integrate(0.0, 1.0))]
    epsilon = np.finfo(np.float64).eps
    while pending:
        lower, upper, whole = pending.pop()
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint == lower or midpoint == upper:
            raise ValueError("Structured physical edge length did not reach floating-point convergence.")
        left = integrate(lower, midpoint)
        right = integrate(midpoint, upper)
        refined = left + right
        defect = abs(refined - whole)
        if not np.isfinite(refined) or not np.isfinite(defect):
            raise ValueError("Structured physical edge length is nonfinite.")
        if defect <= 32.0 * epsilon * max(abs(refined), np.finfo(np.float64).tiny):
            total += refined
            error_total += defect
        else:
            pending.append((midpoint, upper, right))
            pending.append((lower, midpoint, left))
    return total, error_total


def _physical_edge_lengths(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    elements, routes, coordinates = geometry.resolve(mesh)
    points = np.asarray(coordinates, dtype=np.float64)
    mesh_points = np.asarray(mesh.coordinates, dtype=np.float64)
    lengths: dict[tuple[int, int], float] = {}
    maximum_error = 0.0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, FiniteElementSpec):
            raise ValueError("Native structured and sweep sizing requires polynomial coordinate maps.")
        topology = reference_cell_topology(block.cell_kind)
        corner_indices = _reference_corner_indices(element, block.cell_kind)
        quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(element.degree + 2)
        reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
        route_rows = np.asarray(route, dtype=np.int64)
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)
        for vertex_row, coordinate_route in zip(cell_vertices, route_rows, strict=True):
            local_dofs = points[coordinate_route]
            if not np.array_equal(local_dofs[list(corner_indices)], mesh_points[vertex_row]):
                raise ValueError("Coordinate geometry must retain the exact mesh cell corners.")
            for first_corner, second_corner in topology.entities[1]:
                first_vertex = int(vertex_row[first_corner])
                second_vertex = int(vertex_row[second_corner])
                key = (min(first_vertex, second_vertex), max(first_vertex, second_vertex))
                if key in lengths:
                    continue
                start = reference_vertices[first_corner]
                direction = reference_vertices[second_corner] - start
                length, error = _quadrature_edge_length(
                    element,
                    local_dofs,
                    start,
                    direction,
                    nodes=quadrature_nodes,
                    weights=quadrature_weights,
                )
                if not np.isfinite(length) or length <= 0.0:
                    raise ValueError(f"Structured physical edge {key} has nonpositive length {length}.")
                lengths[key] = length
                maximum_error = max(maximum_error, error)
    ordered_edges = tuple(sorted(lengths))
    edge_rows = np.asarray(ordered_edges, dtype=np.int64)
    edge_lengths = np.asarray([lengths[edge] for edge in ordered_edges], dtype=np.float64)
    return edge_rows, edge_lengths, maximum_error


def _minimum_surface_angle(mesh: CellMesh, geometry: CellGeometrySpec, /) -> float:
    """Measure corner angles from the realized coordinate-map edge tangents."""
    elements, routes, coordinates = geometry.resolve(mesh)
    points = np.asarray(coordinates, dtype=np.float64)
    minimum = np.inf
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, FiniteElementSpec):
            raise ValueError("Structured corner angles require polynomial coordinate maps.")
        topology = reference_cell_topology(block.cell_kind)
        vertices = np.asarray(topology.vertices, dtype=np.float64)
        _, gradients = element.tabulate(vertices)
        jacobians = np.einsum(
            "vjd,cjk->cvkd",
            np.asarray(gradients),
            points[np.asarray(route, dtype=np.int64)],
        )
        for corner, vertex in enumerate(vertices):
            neighbors = [
                second if first == corner else first
                for first, second in topology.entities[1]
                if corner in (first, second)
            ]
            if len(neighbors) != 2:
                raise ValueError("Surface corner quality requires two incident edges.")
            directions = vertices[neighbors] - vertex
            tangents = np.einsum("ckd,ed->cek", jacobians[:, corner], directions)
            lengths = np.linalg.norm(tangents, axis=-1)
            if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
                raise ValueError("Surface corner quality requires nonzero finite tangents.")
            cosine = np.sum(tangents[:, 0] * tangents[:, 1], axis=-1) / np.prod(
                lengths, axis=-1
            )
            angles = np.arccos(np.clip(cosine, -1.0, 1.0))
            minimum = min(minimum, float(np.min(angles)))
    if not np.isfinite(minimum):
        raise ValueError("Surface corner quality requires nonempty coordinate geometry.")
    return minimum


def _size_compliance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> MeshingComplianceReport:
    target = specification.target
    actual = {block.cell_kind for block in mesh.blocks}
    issues = _family_issues(actual, specification)
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = [
        (f"family:{kind}:cells", float(sum(block.cell_count for block in mesh.blocks if block.cell_kind == kind)))
        for kind in sorted(actual)
    ]
    if (mesh.topological_dimension, mesh.ambient_dimension) != (
        target.topological_dimension,
        target.ambient_dimension,
    ):
        issues.append("target_dimensions")
    if specification.size_controls:
        edges, lengths, quadrature_error = _physical_edge_lengths(mesh, geometry)
        vertex_count = mesh.coordinates.shape[0]
        minimum = np.full((vertex_count,), np.inf, dtype=np.float64)
        maximum = np.zeros((vertex_count,), dtype=np.float64)
        np.minimum.at(minimum, edges[:, 0], lengths)
        np.minimum.at(minimum, edges[:, 1], lengths)
        np.maximum.at(maximum, edges[:, 0], lengths)
        np.maximum.at(maximum, edges[:, 1], lengths)
        growth = float(np.max(maximum / minimum, initial=1.0))
        achieved.extend(
            (
                ("physical_edge_length_minimum", float(np.min(lengths))),
                ("physical_edge_length_maximum", float(np.max(lengths))),
                ("physical_edge_length_quadrature_error", quadrature_error),
                ("physical_edge_growth_ratio", growth),
            )
        )
        policy = specification.size_compliance
        for control in specification.size_controls:
            if not isinstance(control, UniformSizeControl):
                raise TypeError("Structured/sweep compliance requires UniformSizeControl values.")
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * control.target_size
            statistics = {
                "p50": float(np.quantile(lengths, 0.50)),
                "p95": float(np.quantile(lengths, 0.95)),
            }
            for statistic in policy.target_statistics:
                measured = statistics[statistic]
                requested_name = f"{control.control_id}:{statistic}"
                requested.append((requested_name, control.target_size))
                achieved.append((requested_name, measured))
                if control.strength is SizeControlStrength.HARD and abs(measured - control.target_size) > tolerance:
                    issues.append(requested_name)
            if control.minimum_size is not None:
                requested.append((f"{control.control_id}:minimum_size", control.minimum_size))
                achieved.append((f"{control.control_id}:minimum_edge", float(np.min(lengths))))
                if control.strength is SizeControlStrength.HARD and np.min(lengths) < control.minimum_size - tolerance:
                    issues.append(f"{control.control_id}:minimum_size")
            if control.maximum_size is not None:
                requested.append((f"{control.control_id}:maximum_size", control.maximum_size))
                achieved.append((f"{control.control_id}:maximum_edge", float(np.max(lengths))))
                if control.strength is SizeControlStrength.HARD and np.max(lengths) > control.maximum_size + tolerance:
                    issues.append(f"{control.control_id}:maximum_size")
            if control.maximum_growth_rate is not None:
                requested.append((f"{control.control_id}:maximum_growth_rate", control.maximum_growth_rate))
                achieved.append((f"{control.control_id}:observed_growth_rate", growth))
                growth_tolerance = (
                    policy.relative_tolerance * control.maximum_growth_rate
                    + 32.0 * np.finfo(np.float64).eps * control.maximum_growth_rate
                )
                if control.strength is SizeControlStrength.HARD and growth > control.maximum_growth_rate + growth_tolerance:
                    issues.append(f"{control.control_id}:maximum_growth_rate")
    if isinstance(specification, SurfaceMeshingSpec) and specification.quality_target is not None:
        quality = specification.quality_target
        minimum_angle = _minimum_surface_angle(mesh, geometry)
        requested.append(("minimum_angle", quality.minimum_angle))
        achieved.append(("minimum_angle", minimum_angle))
        if quality.hard and minimum_angle < quality.minimum_angle:
            issues.append("minimum_angle")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(dict.fromkeys(issues)),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _cell_regions(
    source: NativeStructuredSource | NativeSweepSource,
    mesh: CellMesh,
    /,
) -> np.ndarray | None:
    region_ids = _domain_region_ids(source.domain)
    if not region_ids:
        return None
    block_regions = dict(source.block_regions)
    names = tuple(block.name for block in mesh.blocks)
    if not block_regions and len(region_ids) == 1:
        block_regions = {name: region_ids[0] for name in names}
    if set(block_regions) != set(names) or any(region not in region_ids for region in block_regions.values()):
        raise ValueError("Every structured output block must have one declared source-domain region.")
    return np.concatenate(
        tuple(
            np.full((block.cell_count,), region_ids.index(block_regions[name]), dtype=np.int64)
            for name, block in zip(names, mesh.blocks, strict=True)
        )
    )


def _region_zones(
    source: NativeStructuredSource | NativeSweepSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    mesh: CellMesh,
    /,
) -> tuple[MeshZone, ...]:
    if mesh.topological_dimension != 3:
        return ()
    regions = _domain_region_ids(source.domain)
    if not regions:
        return ()
    block_regions = dict(source.block_regions)
    names = tuple(block.name for block in mesh.blocks)
    if not block_regions and len(regions) == 1:
        block_regions = {name: regions[0] for name in names}
    controls = {control.region_name: control for control in specification.region_controls}
    zones: list[MeshZone] = []
    for region in regions:
        ids = tuple(
            int(identifier)
            for name, block in zip(names, mesh.blocks, strict=True)
            if block_regions.get(name) == region
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        )
        if not ids:
            continue
        control = controls.get(region)
        zones.append(
            MeshZone(
                region,
                MeshZoneRole.REGION,
                _mesh_scope(mesh, 3, np.asarray(ids, dtype=np.int64)),
                material_id=None if control is None else control.material_id,
                region_role=None if control is None else control.role,
            )
        )
    return tuple(zones)




def _certification_request(
    source: NativeStructuredSource | NativeSweepSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    mesh: CellMesh,
    /,
    *,
    fidelity_witness: SourceBoundaryChartWitness | None = None,
) -> NativeCertificationRequest:
    if mesh.topological_dimension == 2:
        schedule_name = "planar" if mesh.ambient_dimension == 2 else "surface"
    elif isinstance(source.domain, PiecewiseLinearDomain):
        schedule_name = "volume_plc"
    elif isinstance(source.domain, MappedReferenceDomain):
        schedule_name = "mapped_volume"
    else:
        raise ValueError("Volume construction requires an authoritative source-domain coverage route.")
    junction_vertices = None
    if schedule_name == "planar":
        domain = source.domain
        if not isinstance(domain, PiecewiseLinearDomain):
            raise ValueError("Planar structured certification requires its source domain.")
        incidence = np.bincount(
            np.asarray(domain.facets, dtype=np.int64).reshape(-1),
            minlength=domain.vertices.shape[0],
        )
        junction_ids: list[int] = []
        for point in np.asarray(domain.vertices)[incidence >= 3]:
            # This checks the image of an already authored source junction; it
            # neither welds vertices nor creates an identity from rounded proximity.
            rows = np.flatnonzero(np.all(np.asarray(mesh.coordinates) == point, axis=1))
            if rows.size != 1:
                raise ValueError("An authored planar junction lacks one exact mesh image.")
            junction_ids.append(int(mesh.vertex_global_ids[rows[0]]))
        junction_vertices = np.asarray(junction_ids, dtype=np.int64)
    return NativeCertificationRequest(
        MeshCertificationSchedule(schedule_name),
        source.source_id,
        source.source_revision,
        domain=source.domain,
        cell_regions=_cell_regions(source, mesh),
        fidelity_source=source.fidelity_source,
        fidelity_tolerance=(
            source.maximum_deviation
            if source.maximum_deviation is not None
            else specification.maximum_deviation
        ),
        fidelity_witness=fidelity_witness,
        feature_source=source,
        junction_vertices=junction_vertices,
        protected_features=specification.protected_features,
    )


@final
class PreparedStructured(StrictModule):
    source: NativeStructuredSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    schedule: NativeStructuredSchedule
    coordinate_contract: SpatialCoordinateContract
    source_binding_id: str
    prepared_id: str

    @checked
    def __init__(
        self,
        source: NativeStructuredSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        schedule: NativeStructuredSchedule,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        issues = structured_support_issues(source, specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(issues),
                provider_code="native_structured_preflight",
                stage="source_inspection",
            )
        self.source = source
        self.specification = specification
        self.schedule = schedule
        self.coordinate_contract = coordinate_contract
        self.source_binding_id = source.binding_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-structured",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )

    def execute(self, /) -> NativeMeshConstruction:
        source, specification = self.source, self.specification
        constructions = tuple(
            generate_structured_block(
                block,
                geometry_degree=specification.target.geometry_order,
                optimization_steps=self.schedule.optimization_steps,
            )
            for block in source.blocks
        )
        combined = glue_structured_blocks(constructions, source.interfaces)
        compliance = _size_compliance(combined.mesh, combined.geometry, specification)
        fidelity_witness = _structured_curve_witness(source, constructions, combined)
        organization = _structured_organization(
            source, constructions, combined, fidelity_witness
        )
        organization = NativeMeshOrganization(
            patches=organization.patches,
            zones=_region_zones(source, specification, combined.mesh),
            labels=organization.labels,
            attributes=organization.attributes,
            associations=organization.associations,
            region_boundary_evidence=organization.region_boundary_evidence,
            surface_source=organization.surface_source,
        )
        return NativeMeshConstruction(
            combined.mesh,
            self.coordinate_contract,
            compliance,
            _certification_request(
                source,
                specification,
                combined.mesh,
                fidelity_witness=fidelity_witness,
            ),
            geometry=combined.geometry,
            organization=organization,
            planar_embedding=(
                specification.planar_embedding
                if isinstance(specification, SurfaceMeshingSpec)
                else None
            ),
        )


@final
class PreparedSweep(StrictModule):
    source: NativeSweepSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    source_binding_id: str
    prepared_id: str

    @checked
    def __init__(
        self,
        source: NativeSweepSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        issues = sweep_support_issues(source, specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(issues),
                provider_code="native_sweep_preflight",
                stage="source_inspection",
            )
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.source_binding_id = source.binding_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-sweep",
                "source": source.binding_id,
                "specification": specification.specification_id,
            }
        )

    def execute(self, /) -> NativeMeshConstruction:
        source, specification = self.source, self.specification
        construction = generate_sweep(
            source.profile,
            source.control,
            name="sweep",
            profile_geometry=source.profile_geometry,
            root_correspondence=source.root_correspondence,
        )
        fidelity_witness = (
            _sweep_piecewise_boundary_witness(source, construction)
            if isinstance(source.domain, PiecewiseLinearDomain)
            else None
        )
        compliance = _size_compliance(construction.mesh, construction.geometry, specification)
        organization = _sweep_organization(source, specification, construction)
        organization = NativeMeshOrganization(
            patches=organization.patches,
            zones=_region_zones(source, specification, construction.mesh),
            labels=organization.labels,
            attributes=organization.attributes,
            associations=organization.associations,
            region_boundary_evidence=organization.region_boundary_evidence,
            surface_source=organization.surface_source,
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            _certification_request(
                source,
                specification,
                construction.mesh,
                fidelity_witness=fidelity_witness,
            ),
            geometry=construction.geometry,
            organization=organization,
            source_mesh=source.profile,
            source_geometry=source.profile_geometry,
            reference_maps=construction.reference_map,
        )


@checked
def prepare_native_structured(
    source: NativeStructuredSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedStructured:
    if options.route != "structured_transfinite":
        raise ValueError("options.route must be structured_transfinite for structured preparation.")
    if options.structured_schedule is None:
        raise ValueError("structured_transfinite requires its declared optimization schedule.")
    return PreparedStructured(source, specification, options.structured_schedule, coordinate_contract)


@checked
def prepare_native_sweep(
    source: NativeSweepSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedSweep:
    if options.route != "sweep":
        raise ValueError("options.route must be sweep for sweep preparation.")
    if options.structured_schedule is not None:
        raise ValueError("Sweep does not consume a structured optimization schedule.")
    return PreparedSweep(source, specification, coordinate_contract)


__all__ = [
    "PreparedStructured",
    "PreparedSweep",
    "structured_support_issues",
    "sweep_support_issues",
    "prepare_native_structured",
    "prepare_native_sweep",
]
