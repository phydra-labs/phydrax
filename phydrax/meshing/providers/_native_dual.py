# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Source-bound quad and hex duals with direct mapped-grid routes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import final

import numpy as np

import phydrax.geometry._source_fidelity as source_fidelity_owner

from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellBlock, CellGeometrySpec, CellMesh
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._cell_geometry import ScalarCoordinateElement
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_locator import CellLocationStatus
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    compose_expression,
    constant,
    coordinate_map_expressions,
    CoordinateSourceBank,
    derivative,
    evaluate_expression,
    expression_equal,
    RationalPolynomial,
    scale,
)
from ...discretization._exact_algebraic_geometry import rational_coefficient
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._mapped_locator import PreparedMappedCellLocator
from ...discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    ExactReferenceChartExpression,
)
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._curve_boundary import ParametricCurveBoundarySource
from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from ...geometry._source_identity import source_entity_id
from ...meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from ...meshing._certification_schedule import MeshCertificationSchedule
from ...meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from ...meshing._controls import (
    FeatureKind,
    ProtectedFeature,
)
from ...meshing._hex_cut_generation import (
    BalancedCutHexConstruction,
    generate_balanced_cut_hexes,
)
from ...meshing._hex_cut_preparation import prepare_balanced_cut_grid
from ...meshing._hex_dominant import extract_hex_dominant, HexDominantConstruction
from ...meshing._hex_frame_map import (
    generate_frame_hexes,
    prepare_frame_hex_grid,
    prepare_source_block_connection,
    prepare_source_hex_frame,
)
from ...meshing._hex_frame_records import (
    PreparedSourceHexFrame,
    SourceBlockConnection,
)
from ...meshing._hex_generation import (
    extract_volume_hexes,
    generate_integer_grid_hexes,
    mapped_source_lipschitz_bound,
    MappedGridHexConstruction,
    realize_mapped_grid_hexes,
)
from ...meshing._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from ...meshing._organization import (
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionRole,
)
from ...meshing._quad_generation import (
    extract_surface_quads,
    prepare_surface_cross_field,
)
from ...meshing._result import MeshingComplianceReport
from ...meshing._scope import MeshingEntityKind, MeshingScope
from ...meshing._sizing import SizeControlStrength, UniformSizeControl
from ...meshing.providers._native_options import NativeMeshingOptions
from ...meshing.providers._native_planar import (
    _planar_curve_source_query,
    _PreparedPlanarRoute,
    planar_support_issues,
    prepare_native_planar,
)
from ...meshing.providers._native_planar_domain import PreparedPlanarDomain
from ...meshing.providers._native_sources import (
    NativeMappedHexSource,
    NativePlanarSource,
    NativePlcSource,
)
from ...meshing.providers._native_volume import (
    prepare_native_plc_volume,
    PreparedPlcVolume,
    volume_support_issues,
)
from ...typing import checked
from .._dual_extraction import DualExtraction


@dataclass(frozen=True, slots=True)
class _MappedRootChart:
    source_cell_id: int
    source_parent_row: int
    source_region: int
    source_occurrence_path: tuple[str, ...]
    source_reference_exact_vertices: tuple[tuple[Fraction, ...], ...]
    source_reference_chart_arguments: tuple[ExactReferenceChartExpression, ...]


def _mapped_reference_cell_charts(
    source: NativeMappedHexSource,
    construction: MappedGridHexConstruction,
    /,
) -> dict[int, _MappedRootChart]:
    """Bind each target hex to its retained exact root-local chart and occurrence."""
    mesh = construction.mesh
    blocks = tuple(mesh.blocks)
    maps = tuple(construction.reference_maps)
    flat_target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in blocks)
    )
    parent_rows = np.asarray(construction.source_parent_rows, dtype=np.int64)
    target_regions = np.asarray(construction.cell_regions, dtype=np.int64)
    root_regions = np.asarray(source.domain.cell_regions, dtype=np.int64)
    if (
        len(maps) != len(blocks)
        or parent_rows.shape != flat_target_ids.shape
        or target_regions.shape != flat_target_ids.shape
        or root_regions.shape != (len(source.domain.root_cell_ids),)
        or len(source.domain.root_cell_occurrence_paths)
        != len(source.domain.root_cell_ids)
    ):
        raise ValueError("Mapped target blocks lost their exact root-row ancestry banks.")
    records: dict[int, _MappedRootChart] = {}
    offset = 0
    for block, witnesses in zip(blocks, maps, strict=True):
        block_ids = np.asarray(block.global_ids, dtype=np.int64)
        target_ids = np.asarray(witnesses.target_cell_ids, dtype=np.int64)
        root_ids = np.asarray(witnesses.source_cell_ids, dtype=np.int64)
        exact_vertices = witnesses.source_reference_exact_vertices
        chart_arguments = witnesses.source_reference_chart_arguments
        if (
            block.cell_kind != "hexahedron"
            or witnesses.target_cell_kind != "hexahedron"
            or witnesses.source_cell_kind != "hexahedron"
            or not np.array_equal(target_ids, block_ids)
            or root_ids.shape != block_ids.shape
            or exact_vertices is None
            or len(exact_vertices) != block_ids.size
            or chart_arguments is None
            or len(chart_arguments) != block_ids.size
        ):
            raise ValueError(
                "Mapped target hexes lack exact root-local reference chart witnesses."
            )
        for row, (target_id, root_id) in enumerate(
            zip(block_ids.tolist(), root_ids.tolist(), strict=True)
        ):
            flat_row = offset + row
            source_parent_row = int(parent_rows[flat_row])
            if (
                source_parent_row < 0
                or source_parent_row >= len(source.domain.root_cell_ids)
                or int(source.domain.root_cell_ids[source_parent_row]) != int(root_id)
            ):
                raise ValueError(
                    "Mapped root chart changes its authored source cell identity."
                )
            source_region = int(root_regions[source_parent_row])
            if int(target_regions[flat_row]) != source_region:
                raise ValueError(
                    "Mapped root chart changes its material region ancestry."
                )
            if int(target_id) in records:
                raise ValueError("Mapped target cell IDs must be unique across blocks.")
            records[int(target_id)] = _MappedRootChart(
                int(root_id),
                source_parent_row,
                source_region,
                tuple(source.domain.root_cell_occurrence_paths[source_parent_row]),
                tuple(
                    tuple(Fraction(value) for value in point)
                    for point in exact_vertices[row]
                ),
                tuple(chart_arguments[row]),
            )
        offset += block_ids.size
    if offset != flat_target_ids.size or len(records) != flat_target_ids.size:
        raise ValueError(
            "Mapped root charts must cover every target hexahedron exactly once."
        )
    return records


def _mapped_root_reference_corner(
    control: _MappedRootChart,
    local_vertex: int,
    local_point: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...]:
    mapped = tuple(
        rational_coefficient(
            evaluate_expression(expression, local_point), role="mapped root corner"
        )
        for expression in control.source_reference_chart_arguments
    )
    if mapped != control.source_reference_exact_vertices[local_vertex]:
        raise ValueError(
            "Mapped root chart expression disagrees with its exact corner bank."
        )
    return mapped


def _triangle_scaffold(
    specification: SurfaceMeshingSpec,
    /,
    *,
    capture_soft_feature_witnesses: bool = False,
) -> SurfaceMeshingSpec:
    target = CellMeshingTarget(
        2,
        specification.target.ambient_dimension,
        CellFamilyPolicy(required=("triangle",)),
        geometry_order=1,
        require_conforming=specification.target.require_conforming,
    )
    protected_features = tuple(
        feature
        if feature.hard or not capture_soft_feature_witnesses
        else ProtectedFeature(
            feature.scope,
            feature.feature_kind,
            maximum_deviation=feature.maximum_deviation,
            hard=True,
        )
        for feature in specification.protected_features
    )
    return SurfaceMeshingSpec(
        target,
        specification.scope,
        planar_embedding=specification.planar_embedding,
        size_controls=specification.size_controls,
        protected_features=protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        periodic_constraints=specification.periodic_constraints,
        layer_controls=specification.layer_controls,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        maximum_normal_deviation=specification.maximum_normal_deviation,
        deterministic=specification.deterministic,
        quality_target=specification.quality_target,
        background_metric=specification.background_metric,
    )


def _tetrahedron_scaffold(specification: VolumeMeshingSpec, /) -> VolumeMeshingSpec:
    target = CellMeshingTarget(
        3,
        3,
        CellFamilyPolicy(required=("tetrahedron",)),
        geometry_order=1,
        require_conforming=specification.target.require_conforming,
    )
    return VolumeMeshingSpec(
        target,
        specification.boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=specification.size_controls,
        protected_features=specification.protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=specification.layer_controls,
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


def dual_hex_support_issues(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    if not isinstance(source, NativePlcSource):
        raise TypeError("source must be NativePlcSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    issues: list[str] = []
    if options.route != "plc_dual_hex":
        issues.append("the plc_dual_hex native route")
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or set((*families.required, *families.preferred)) != {"hexahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append("a pure required hexahedral target")
    if options.volume_schedule is None:
        issues.append("a native tetrahedral scaffold schedule")
    if source.mapped_domain is not None and source.complex.geometry is None:
        issues.append("a source-coordinate map for the mapped PLC tetrahedral scaffold")
    if not issues:
        issues.extend(volume_support_issues(source, _tetrahedron_scaffold(specification)))
    return list(dict.fromkeys(issues))


def quad_support_issues(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    """Return unsupported source or family obligations for planar quad construction."""
    if not isinstance(source, NativePlanarSource):
        raise TypeError("source must be NativePlanarSource.")
    if not isinstance(specification, SurfaceMeshingSpec):
        raise TypeError("specification must be SurfaceMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    issues: list[str] = []
    if options.route != "planar_dual_quad":
        issues.append("the planar_dual_quad native route")
    families = specification.target.cell_families
    requested = set((*families.required, *families.preferred))
    if (
        "quadrilateral" not in requested
        or set(families.required) - {"quadrilateral"}
        or requested - {"quadrilateral", "triangle"}
        or set(families.allowed_transitions) - {"quadrilateral", "triangle"}
    ):
        issues.append("a quadrilateral target with no required non-quad transitions")
    if specification.target.geometry_order != 1:
        issues.append("affine planar source geometry")
    if specification.target.ambient_dimension != 2:
        issues.append("a planar source embedded in two dimensions")
    if specification.quality_target is not None:
        issues.append("no separately declared global quality target")
    if specification.background_metric is not None:
        issues.append("uniform or source-scoped size controls")
    if specification.periodic_constraints:
        issues.append("no periodic quotient constraints")
    if specification.layer_controls:
        issues.append("no boundary-layer construction")
    scaffold = _triangle_scaffold(specification)
    issues.extend(planar_support_issues(source, scaffold))
    return list(dict.fromkeys(issues))


def _association_for_dimension(
    organization: NativeMeshOrganization,
    mesh: CellMesh,
    dimension: int,
    /,
) -> GeometryAssociation:
    """Return the single association that covers one entity dimension."""
    entity_set = mesh.entity_set(dimension)
    matches = tuple(
        association
        for association in organization.associations
        if association.target_entity_set_id == entity_set.entity_set_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"Scaffold needs exactly one source association for dimension {dimension}."
        )
    return matches[0]


def _association_rows(
    organization: NativeMeshOrganization,
    mesh: CellMesh,
    dimension: int,
    /,
) -> dict[int, tuple[GeometryAssociation, int]]:
    """Map entity IDs to their association row across kind-split associations."""
    entity_set = mesh.entity_set(dimension)
    rows: dict[int, tuple[GeometryAssociation, int]] = {}
    for association in organization.associations:
        if association.target_entity_set_id != entity_set.entity_set_id:
            continue
        identifiers = np.asarray(association.target_global_ids, dtype=np.int64)
        for row, identifier in enumerate(identifiers.tolist()):
            if identifier in rows:
                raise ValueError("Scaffold source associations repeat a target entity.")
            rows[identifier] = (association, row)
    if not rows:
        raise ValueError(f"Scaffold has no source association for dimension {dimension}.")
    return rows


def _entity_source_ids(
    organization: NativeMeshOrganization,
    mesh: CellMesh,
    dimension: int,
    /,
) -> tuple[str, ...]:
    """Source entity IDs aligned with the mesh entity-set order."""
    rows = _association_rows(organization, mesh, dimension)
    result: list[str] = []
    for identifier in np.asarray(
        mesh.entity_set(dimension).entity_ids, dtype=np.int64
    ).tolist():
        record = rows.get(identifier)
        if record is None:
            raise ValueError("A scaffold entity lacks its source association.")
        association, row = record
        result.append(association.source_entity_ids[row])
    return tuple(result)


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
        identifiers,
    )


def _feature_scaffold_edges(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    scaffold: NativeMeshConstruction,
    /,
) -> np.ndarray:
    requested = {
        source_entity_id(
            source.source_id,
            source.source_revision,
            "edge",
            int(identifier),
        )
        for feature in specification.protected_features
        if feature.feature_kind is FeatureKind.CURVE
        for identifier in np.asarray(feature.scope.entity_ids, dtype=np.int64).tolist()
    }
    if not requested:
        return np.empty((0,), dtype=np.int64)
    edge_sources = _entity_source_ids(scaffold.organization, scaffold.mesh, 1)
    return np.asarray(
        [row for row, source_id in enumerate(edge_sources) if source_id in requested],
        dtype=np.int64,
    )


def _entity_vertex_rows(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if dimension == 0:
        vertex_ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
        rows_by_id = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(mesh.vertex_global_ids, dtype=np.int64)
            )
        }
        if len(rows_by_id) != vertex_ids.size:
            raise ValueError("Dual vertex IDs do not bijectively bind coordinate rows.")
        return np.asarray(
            [[rows_by_id[int(identifier)]] for identifier in vertex_ids.tolist()],
            dtype=np.int64,
        )
    if isinstance(connectivity, SimplicialConnectivity):
        if dimension >= len(connectivity.entities):
            raise ValueError("Requested simplex entity dimension is absent.")
        rows = np.asarray(connectivity.entities[dimension], dtype=np.int64)
    elif dimension == 1 and isinstance(
        connectivity,
        (
            PolygonalConnectivity,
            PolyhedralConnectivity,
            TetrahedralConnectivity,
            HexahedralConnectivity,
        ),
    ):
        rows = np.asarray(connectivity.edges, dtype=np.int64)
    elif dimension == 2 and isinstance(
        connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
    ):
        rows = np.asarray(connectivity.faces, dtype=np.int64)
    elif dimension == 2 and isinstance(connectivity, PolygonalConnectivity):
        valid = np.asarray(connectivity.cell_vertex_valid, dtype=np.bool_)
        rows = np.where(valid, np.asarray(connectivity.cell_vertices, dtype=np.int64), -1)
    elif dimension == 2 and isinstance(connectivity, PolyhedralConnectivity):
        # Mixed families carry triangle and quadrilateral faces; shorter rows
        # are right-padded with -1 and every consumer skips the padding.
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        counts = np.diff(offsets)
        values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        width = int(np.max(counts)) if counts.size else 0
        rows = np.full((counts.size, width), -1, dtype=np.int64)
        columns = np.arange(values.size, dtype=np.int64) - np.repeat(offsets[:-1], counts)
        rows[np.repeat(np.arange(counts.size, dtype=np.int64), counts), columns] = values
    else:
        raise ValueError(
            "Dual route has no canonical vertex rows for this entity dimension."
        )
    identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[0] != identifiers.size:
        raise ValueError("Dual entity vertex rows and stable IDs are misaligned.")
    return rows


def _cell_entity_incidence(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        return np.asarray(connectivity.cell_entities[dimension], dtype=np.int64)
    if dimension == 1 and isinstance(
        connectivity, (PolygonalConnectivity, HexahedralConnectivity)
    ):
        return np.asarray(connectivity.cell_edges, dtype=np.int64)
    if dimension == 2 and isinstance(
        connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
    ):
        return np.asarray(connectivity.cell_faces, dtype=np.int64)
    raise ValueError(
        "Dual route has no canonical cell-to-entity incidence for this dimension."
    )


def _quad_organization(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    scaffold: NativeMeshConstruction,
    mesh: CellMesh,
    supports: np.ndarray,
    /,
) -> NativeMeshOrganization:
    base_mesh = scaffold.mesh
    base_edges = np.sort(_entity_vertex_rows(base_mesh, 1), axis=1)
    base_edge_ids = np.asarray(base_mesh.entity_set(1).entity_ids, dtype=np.int64)
    base_vertex_ids = np.asarray(base_mesh.vertex_global_ids, dtype=np.int64)
    base_edge_row = {
        tuple(map(int, edge)): row for row, edge in enumerate(base_edges.tolist())
    }
    # The planar scaffold splits each dimension by association kind: curve rows
    # carry exact parameters, piecewise-linear rows carry vertex/region strata.
    base_edge_rows = _association_rows(scaffold.organization, base_mesh, 1)
    base_vertex_rows = _association_rows(scaffold.organization, base_mesh, 0)
    region_id = source.region_entity_id
    region_path = source.source_entity_key_for_scope(
        2, int(source.scope_entity_ids(2)[0])
    )[1]
    target_edges = np.sort(_entity_vertex_rows(mesh, 1), axis=1)
    target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    vertex_row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        )
    }

    vertex_source_ids: list[str] = []
    vertex_roles: list[GeometrySourceEntityRole | None] = []
    vertex_residuals: list[float] = []
    vertex_parameters = np.full((target_vertex_ids.size, 2), np.nan, dtype=np.float64)
    vertex_points: list[Fraction | None] = []
    vertex_paths: list[tuple[str, ...]] = []
    for output_row, identifier in enumerate(target_vertex_ids.tolist()):
        vertex = vertex_row_by_id[int(identifier)]
        support = tuple(
            int(value) for value in supports[vertex].tolist() if int(value) >= 0
        )
        if len(support) == 1:
            record = base_vertex_rows.get(int(base_vertex_ids[support[0]]))
            if record is not None:
                association, row = record
                vertex_source_ids.append(association.source_entity_ids[row])
                vertex_roles.append(association.source_roles[row])
                vertex_residuals.append(float(np.asarray(association.residuals)[row]))
                vertex_parameters[output_row] = np.asarray(
                    association.parameters, dtype=np.float64
                )[row]
                vertex_points.append(association.source_parameter_points[row])
                vertex_paths.append(association.source_occurrence_paths[row])
                continue
        edge_row = (
            base_edge_row.get(tuple(sorted(support))) if len(support) == 2 else None
        )
        record = (
            None
            if edge_row is None
            else base_edge_rows.get(int(base_edge_ids[int(edge_row)]))
        )
        if record is not None:
            association, row = record
            interval = association.source_parameter_intervals[row]
            if interval is not None:
                vertex_source_ids.append(association.source_entity_ids[row])
                vertex_roles.append(GeometrySourceEntityRole.EDGE)
                vertex_residuals.append(float(np.asarray(association.residuals)[row]))
                parameter = (interval[0] + interval[1]) / 2
                vertex_parameters[output_row, 0] = float(parameter)
                vertex_points.append(parameter)
                vertex_paths.append(association.source_occurrence_paths[row])
                continue
        vertex_source_ids.append(region_id)
        vertex_roles.append(GeometrySourceEntityRole.REGION)
        vertex_residuals.append(0.0)
        vertex_points.append(None)
        vertex_paths.append(region_path)

    target_edge_sources: list[str] = []
    target_edge_roles: list[GeometrySourceEntityRole | None] = []
    target_edge_residuals: list[float] = []
    target_edge_parameters = np.full((target_edge_ids.size, 2), np.nan, dtype=np.float64)
    target_edge_orientations = np.zeros((target_edge_ids.size,), dtype=np.int8)
    target_edge_intervals: list[tuple[Fraction, Fraction] | None] = []
    target_edge_paths: list[tuple[str, ...]] = []
    for row, edge in enumerate(target_edges.tolist()):
        endpoint_supports = tuple(
            tuple(
                int(value) for value in supports[int(vertex)].tolist() if int(value) >= 0
            )
            for vertex in edge
        )
        union = tuple(sorted(set(endpoint_supports[0]) | set(endpoint_supports[1])))
        base_row = base_edge_row.get(union) if len(union) == 2 else None
        record = (
            None
            if base_row is None
            else base_edge_rows.get(int(base_edge_ids[int(base_row)]))
        )
        interval = (
            None if record is None else record[0].source_parameter_intervals[record[1]]
        )
        if record is not None and base_row is not None and interval is not None:
            association, association_row = record
            base_pair = tuple(map(int, base_edges[int(base_row)].tolist()))
            orientation = int(
                np.asarray(association.orientations, dtype=np.int8)[association_row]
            )
            directed_interval = interval if orientation >= 0 else interval[::-1]

            def endpoint_parameter(endpoint_support: tuple[int, ...]) -> Fraction:
                if len(endpoint_support) == 1:
                    if endpoint_support[0] == base_pair[0]:
                        return directed_interval[0]
                    if endpoint_support[0] == base_pair[1]:
                        return directed_interval[1]
                if set(endpoint_support) == set(base_pair):
                    return (directed_interval[0] + directed_interval[1]) / 2
                raise ValueError("Quad edge endpoint has no exact source-edge parameter.")

            first_parameter = endpoint_parameter(endpoint_supports[0])
            last_parameter = endpoint_parameter(endpoint_supports[1])
            target_edge_sources.append(association.source_entity_ids[association_row])
            target_edge_roles.append(GeometrySourceEntityRole.EDGE)
            target_edge_residuals.append(
                float(np.asarray(association.residuals)[association_row])
            )
            target_edge_parameters[row] = (
                float(first_parameter),
                float(last_parameter),
            )
            target_edge_orientations[row] = 1 if last_parameter > first_parameter else -1
            # Intervals are increasing; direction lives in the edge orientation.
            target_edge_intervals.append(
                (
                    min(first_parameter, last_parameter),
                    max(first_parameter, last_parameter),
                )
            )
            target_edge_paths.append(association.source_occurrence_paths[association_row])
        else:
            target_edge_sources.append(region_id)
            target_edge_roles.append(GeometrySourceEntityRole.REGION)
            target_edge_residuals.append(0.0)
            target_edge_intervals.append(None)
            target_edge_paths.append(region_path)

    vertex_set = mesh.entity_set(0)
    edge_set = mesh.entity_set(1)
    cell_set = mesh.entity_set(2)

    def kind_split(
        entity_set_id: str,
        identifiers: np.ndarray,
        curve_rows: np.ndarray,
        sources: list[str],
        roles: list[GeometrySourceEntityRole | None],
        residuals: list[float],
        parameters: np.ndarray,
        paths: list[tuple[str, ...]],
        /,
        *,
        intervals: list[tuple[Fraction, Fraction] | None] | None = None,
        points: list[Fraction | None] | None = None,
        orientations: np.ndarray | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        """Curve rows keep exact parameters; the rest are piecewise-linear strata."""
        result: list[GeometryAssociation] = []
        residual_array = np.asarray(residuals, dtype=np.float64)
        for kind, rows in (
            (GeometryAssociationKind.CURVE, np.flatnonzero(curve_rows)),
            (GeometryAssociationKind.PIECEWISE_LINEAR, np.flatnonzero(~curve_rows)),
        ):
            if not rows.size:
                continue
            selected = rows.tolist()
            curve = kind is GeometryAssociationKind.CURVE
            result.append(
                GeometryAssociation(
                    kind,
                    source.source_id,
                    source.source_revision,
                    entity_set_id,
                    identifiers[rows],
                    tuple(sources[row] for row in selected),
                    residual_array[rows],
                    exact=bool(np.all(residual_array[rows] == 0.0)),
                    source_roles=tuple(roles[row] for row in selected),
                    parameters=parameters[rows] if curve else None,
                    orientations=(
                        None if orientations is None or not curve else orientations[rows]
                    ),
                    source_parameter_intervals=(
                        None
                        if intervals is None or not curve
                        else tuple(intervals[row] for row in selected)
                    ),
                    source_parameter_points=(
                        None
                        if points is None or not curve
                        else tuple(points[row] for row in selected)
                    ),
                    source_occurrence_paths=tuple(paths[row] for row in selected),
                )
            )
        return tuple(result)

    vertex_associations = kind_split(
        vertex_set.entity_set_id,
        target_vertex_ids,
        np.asarray([value is not None for value in vertex_points], dtype=np.bool_),
        vertex_source_ids,
        vertex_roles,
        vertex_residuals,
        vertex_parameters,
        vertex_paths,
        points=vertex_points,
    )
    edge_associations = kind_split(
        edge_set.entity_set_id,
        target_edge_ids,
        np.asarray(
            [value is not None for value in target_edge_intervals], dtype=np.bool_
        ),
        target_edge_sources,
        target_edge_roles,
        target_edge_residuals,
        target_edge_parameters,
        target_edge_paths,
        intervals=target_edge_intervals,
        orientations=target_edge_orientations,
    )
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    cell_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        cell_set.entity_set_id,
        cell_ids,
        (region_id,) * cell_ids.size,
        np.zeros((cell_ids.size,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.REGION,) * cell_ids.size,
    )
    labels = [MeshLabel("region:0", _mesh_scope(mesh, 2, cell_ids))]
    prepared_edge_ids = np.asarray(prepared.source_edge_ids, dtype=np.int64)
    prepared_loops = np.asarray(prepared.source_loops, dtype=np.int64)
    source_edge_names = tuple(
        source_entity_id(source.source_id, source.source_revision, "edge", int(value))
        for value in prepared_edge_ids.tolist()
    )
    source_loop_by_name = {
        name: int(loop)
        for name, loop in zip(source_edge_names, prepared_loops.tolist(), strict=True)
    }
    for loop in sorted(set(prepared_loops.tolist()) - {-1}):
        selected = np.asarray(
            [
                int(identifier)
                for identifier, name in zip(
                    target_edge_ids.tolist(),
                    target_edge_sources,
                    strict=True,
                )
                if source_loop_by_name.get(name) == loop
            ],
            dtype=np.int64,
        )
        if selected.size:
            labels_scope = _mesh_scope(mesh, 1, selected)
            labels.append(MeshLabel(f"loop:{loop}", labels_scope))
    embedded = np.asarray(
        [
            int(identifier)
            for identifier, name in zip(
                target_edge_ids.tolist(), target_edge_sources, strict=True
            )
            if source_loop_by_name.get(name) == -1
        ],
        dtype=np.int64,
    )
    if embedded.size:
        labels.append(MeshLabel("embedded", _mesh_scope(mesh, 1, embedded)))
    for feature in specification.protected_features:
        selected_ids = set(np.asarray(feature.scope.entity_ids, dtype=np.int64).tolist())
        if feature.feature_kind is FeatureKind.CURVE:
            selected_sources = {
                source_entity_id(source.source_id, source.source_revision, "edge", value)
                for value in selected_ids
            }
            targets = np.asarray(
                [
                    int(identifier)
                    for identifier, name in zip(
                        target_edge_ids.tolist(),
                        target_edge_sources,
                        strict=True,
                    )
                    if name in selected_sources
                ],
                dtype=np.int64,
            )
            if targets.size:
                labels.append(
                    MeshLabel(
                        f"feature:{feature.feature_id}", _mesh_scope(mesh, 1, targets)
                    )
                )
        elif feature.feature_kind is FeatureKind.CORNER:
            selected_sources = {
                source_entity_id(
                    source.source_id, source.source_revision, "vertex", value
                )
                for value in selected_ids
            }
            targets = np.asarray(
                [
                    int(identifier)
                    for identifier, name in zip(
                        target_vertex_ids.tolist(),
                        vertex_source_ids,
                        strict=True,
                    )
                    if name in selected_sources
                ],
                dtype=np.int64,
            )
            if targets.size:
                labels.append(
                    MeshLabel(
                        f"feature:{feature.feature_id}", _mesh_scope(mesh, 0, targets)
                    )
                )
    patches = tuple(
        MeshPatch(f"loop:{label.name.split(':', 1)[1]}", label.scope)
        for label in labels
        if label.name.startswith("loop:")
    )
    return NativeMeshOrganization(
        patches=patches,
        labels=tuple(labels),
        associations=(cell_association, *edge_associations, *vertex_associations),
    )


def _uniform_size_controls(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec, /
) -> tuple[UniformSizeControl, ...]:
    """Retain the sizing contract admitted by the native scaffold routes."""
    controls: list[UniformSizeControl] = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise ValueError("This native dual route requires uniform size controls.")
        controls.append(control)
    return tuple(controls)


def _quad_compliance(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    mesh: CellMesh,
    organization: NativeMeshOrganization,
    /,
    *,
    soft_feature_outcomes: tuple[tuple[ProtectedFeature, float, bool], ...] = (),
) -> MeshingComplianceReport:
    edge_pairs = np.sort(_entity_vertex_rows(mesh, 1), axis=1)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    lengths = np.linalg.norm(points[edge_pairs[:, 1]] - points[edge_pairs[:, 0]], axis=1)
    edge_source_names = _entity_source_ids(organization, mesh, 1)
    vertex_source_names = _entity_source_ids(organization, mesh, 0)
    source_edge_rows = {
        source_entity_id(
            source.source_id, source.source_revision, "edge", int(value)
        ): row
        for row, value in enumerate(
            np.asarray(prepared.source_edge_ids, dtype=np.int64).tolist()
        )
    }
    edge_sources = np.asarray(
        [source_edge_rows.get(name, -1) for name in edge_source_names],
        dtype=np.int64,
    )
    vertex_sources = np.asarray(
        [source_edge_rows.get(name, -1) for name in vertex_source_names],
        dtype=np.int64,
    )
    midpoint = 0.5 * (points[edge_pairs[:, 0]] + points[edge_pairs[:, 1]])
    samples = np.concatenate(
        (points[edge_pairs[:, 0]], points[edge_pairs[:, 1]], midpoint), axis=0
    )
    sample_sources = np.concatenate(
        (vertex_sources[edge_pairs[:, 0]], vertex_sources[edge_pairs[:, 1]], edge_sources)
    )
    local_sizes = np.min(
        np.asarray(prepared.sizes(samples, point_sources=sample_sources)).reshape(
            (3, edge_pairs.shape[0])
        ),
        axis=0,
    )
    vertex_sizes = np.asarray(prepared.sizes(points, point_sources=vertex_sources))
    growth = 1.0 + float(
        np.max(
            np.abs(vertex_sizes[edge_pairs[:, 1]] - vertex_sizes[edge_pairs[:, 0]])
            / lengths,
            initial=0.0,
        )
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    policy = specification.size_compliance
    for control in _uniform_size_controls(specification):
        key = f"size:{control.control_id}"
        if control.scope.entity_dimension == 1:
            selected = np.isin(
                np.asarray(prepared.source_edge_ids, dtype=np.int64)[
                    edge_sources.clip(min=0)
                ],
                np.asarray(control.scope.entity_ids, dtype=np.int64),
            ) & (edge_sources >= 0)
            rows = np.flatnonzero(selected)
        else:
            rows = np.arange(edge_pairs.shape[0], dtype=np.int64)
        selected_lengths = lengths[rows]
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        if control.maximum_growth_rate is not None:
            requested.append((f"{key}:maximum_growth_rate", control.maximum_growth_rate))
        if not selected_lengths.size:
            issues.append(f"missing_controlled_edges:{control.control_id}")
            continue
        statistics = {
            name: float(np.quantile(selected_lengths, 0.5 if name == "p50" else 0.95))
            for name in policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(selected_lengths))),
                (f"{key}:maximum_edge", float(np.max(selected_lengths))),
                (
                    f"{key}:maximum_edge_to_local_size",
                    float(np.max(selected_lengths / local_sizes[rows])),
                ),
                (f"{key}:size_field_growth_rate", growth),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if (
                control.minimum_size is not None
                and float(np.min(selected_lengths)) < control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            if (
                control.maximum_size is not None
                and float(np.max(selected_lengths)) > control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
            if np.any(
                selected_lengths
                > local_sizes[rows]
                + policy.absolute_tolerance
                + policy.relative_tolerance * local_sizes[rows]
            ):
                issues.append(f"local_size:{control.control_id}")
            if (
                control.maximum_growth_rate is not None
                and growth
                > control.maximum_growth_rate
                + policy.absolute_tolerance
                + policy.relative_tolerance * control.maximum_growth_rate
            ):
                issues.append(f"maximum_growth_rate:{control.control_id}")
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
    for feature, deviation_upper, warning in soft_feature_outcomes:
        achieved.extend(
            (
                (f"soft_feature:{feature.feature_id}:deviation_upper", deviation_upper),
                (f"soft_feature:{feature.feature_id}:warning", float(warning)),
            )
        )
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _planar_scaffold_feature_witness(
    source: NativePlanarSource,
    feature: ProtectedFeature,
    scaffold: NativeMeshConstruction,
    query: ParametricCurveBoundarySource,
    witness_type: type[SourceBoundaryChartWitness] | type[SourceBoundaryPointWitness],
    /,
) -> tuple[
    ParametricCurveBoundarySource, SourceBoundaryChartWitness | SourceBoundaryPointWitness
]:
    expected_keys = {
        source.source_entity_key_for_scope(
            feature.scope.entity_dimension, int(identifier)
        )
        for identifier in np.asarray(feature.scope.entity_ids, dtype=np.int64).tolist()
    }
    matches = {
        witness.witness_id: (source_query, witness)
        for source_query, witness, _ in scaffold.certification.scoped_fidelity
        if isinstance(source_query, ParametricCurveBoundarySource)
        and source_query.query_id == query.query_id
        and isinstance(witness, witness_type)
        and set(zip(witness.source_entity_ids, witness.occurrence_paths, strict=True))
        == expected_keys
    }
    if len(matches) != 1:
        raise ValueError("Planar scaffold omitted unique exact feature-chart ancestry.")
    return next(iter(matches.values()))


def _dual_quad_curve_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    extraction: DualExtraction,
    query: ParametricCurveBoundarySource,
    source_witness: SourceBoundaryChartWitness,
    /,
) -> SourceBoundaryChartWitness:
    source_mesh = extraction.source
    if (
        source_witness.chart_dimension != 1
        or source_witness.reference_dimension != 2
        or source_witness.source_query_id != query.query_id
        or source_witness.source_id != query.source_id
        or source_witness.source_revision != query.source_revision
    ):
        raise ValueError("Dual curve conversion changed its retained source chart query.")
    source_cell_rows: dict[int, tuple[CellBlock, int]] = {}
    for block in source_mesh.blocks:
        if not isinstance(block, CellBlock) or block.cell_kind != "triangle":
            raise ValueError("Dual curve witnesses require a triangular scaffold.")
        for row, cell_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            source_cell_rows[int(cell_id)] = (block, row)
    source_cell_ids = np.asarray(source_mesh.entity_set(2).entity_ids, dtype=np.int64)
    source_cell_edges = _cell_entity_incidence(source_mesh, 1)
    source_cell_row_by_id = {
        int(identifier): row for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_edge_ids = np.asarray(source_mesh.entity_set(1).entity_ids, dtype=np.int64)
    source_edge_rows = _entity_vertex_rows(source_mesh, 1)
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    source_edge_row_by_id = {
        int(identifier): row for row, identifier in enumerate(source_edge_ids.tolist())
    }
    source_parameters_by_edge: dict[
        int, tuple[dict[int, Fraction], str, str, tuple[str, ...]]
    ] = {}
    triangle_topology = reference_cell_topology("triangle")
    reference_vertices = np.asarray(triangle_topology.vertices, dtype=np.float64)
    for row, source_edge_id_raw in enumerate(
        np.asarray(source_witness.entity_global_ids, dtype=np.int64).tolist()
    ):
        source_edge_id = int(source_edge_id_raw)
        source_cell_id = int(source_witness.cell_global_ids[row])
        local_edge = int(source_witness.local_entity_indices[row])
        block_row = source_cell_rows.get(source_cell_id)
        source_edge_row = source_edge_row_by_id.get(source_edge_id)
        if block_row is None or source_edge_row is None:
            raise ValueError(
                "Planar curve witness names an absent scaffold cell or edge."
            )
        block, cell_row = block_row
        local_edges = tuple(
            tuple(int(value) for value in edge) for edge in triangle_topology.entities[1]
        )
        if local_edge < 0 or local_edge >= len(local_edges):
            raise ValueError(
                "Planar curve witness names an absent scaffold reference edge."
            )
        edge_vertices = local_edges[local_edge]
        reference_edge = reference_vertices[np.asarray(edge_vertices, dtype=np.int64)]
        vertex_parameters: dict[int, Fraction] = {}
        cell_vertices = np.asarray(block.vertices[cell_row], dtype=np.int64)
        exact_target_chart = source_witness.target_reference_chart(row)
        exact_source_chart = source_witness.source_parameter_chart(row)
        for endpoint, target_reference in enumerate(exact_target_chart):
            local_target_reference = np.asarray(
                tuple(float(value) for value in target_reference), dtype=np.float64
            )
            local_matches = np.flatnonzero(
                np.all(reference_edge == local_target_reference[None, :], axis=1)
            )
            if local_matches.size != 1:
                raise ValueError(
                    "Planar curve witness changes an exact scaffold edge endpoint."
                )
            source_vertex_row = int(cell_vertices[edge_vertices[int(local_matches[0])]])
            source_vertex_id = int(source_vertex_ids[source_vertex_row])
            vertex_parameters[source_vertex_id] = exact_source_chart[endpoint][0]
        edge_cell_position = source_cell_row_by_id.get(source_cell_id)
        if edge_cell_position is None:
            raise ValueError(
                "Planar source cell is absent from canonical scaffold incidence."
            )
        if int(source_cell_edges[edge_cell_position, local_edge]) != source_edge_row:
            raise ValueError(
                "Planar curve witness changes its exact scaffold edge incidence."
            )
        endpoints = tuple(
            int(source_vertex_ids[int(value)])
            for value in source_edge_rows[source_edge_row]
        )
        if set(vertex_parameters) != set(endpoints):
            raise ValueError(
                "Planar curve witness omits an authored source-edge endpoint."
            )
        ancestry = (
            vertex_parameters,
            source_witness.source_chart_ids[row],
            source_witness.source_entity_ids[row],
            source_witness.occurrence_paths[row],
        )
        if source_edge_id in source_parameters_by_edge:
            raise ValueError(
                "Planar curve witness duplicates a scaffold source-edge interval."
            )
        source_parameters_by_edge[source_edge_id] = ancestry

    target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    target_edge_rows = _entity_vertex_rows(mesh, 1)
    target_cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    target_cell_edges = _cell_entity_incidence(mesh, 1)
    target_cell_row_by_id = {
        int(identifier): row for row, identifier in enumerate(target_cell_ids.tolist())
    }
    source_edge_lineage = np.asarray(extraction.source_edge_ids, dtype=np.int64)
    parameter_numerators = np.asarray(
        extraction.source_edge_parameter_numerators, dtype=np.int64
    )
    parameter_denominators = np.asarray(
        extraction.source_edge_parameter_denominators, dtype=np.int64
    )
    if (
        target_edge_rows.shape != (target_edge_ids.size, 2)
        or target_cell_edges.shape[0] != target_cell_ids.size
        or source_edge_lineage.shape != target_edge_ids.shape
        or parameter_numerators.shape != (target_edge_ids.size, 2)
        or parameter_denominators.shape != (target_edge_ids.size, 2)
    ):
        raise ValueError(
            "Quad extraction edge ancestry is not aligned with target topology."
        )

    witness_cells: list[int] = []
    witness_edges: list[int] = []
    witness_local_edges: list[int] = []
    target_references: list[np.ndarray] = []
    target_reference_rational_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    source_chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrences: list[tuple[str, ...]] = []
    parameters: list[tuple[float, float]] = []
    source_parameter_rational_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    orientations: list[int] = []
    quad_topology = reference_cell_topology("quadrilateral")
    local_edges = tuple(
        tuple(int(value) for value in edge) for edge in quad_topology.entities[1]
    )
    target_reference_vertices = np.asarray(quad_topology.vertices, dtype=np.float64)
    seen_target_edge_ids: set[int] = set()
    for block in mesh.blocks:
        if block.cell_kind != "quadrilateral":
            raise ValueError(
                "Dual curve witness target must contain quadrilateral cells."
            )
        block_cells = np.asarray(block.global_ids, dtype=np.int64)
        block_vertices = np.asarray(block.vertices, dtype=np.int64)
        for cell_row, cell_id in enumerate(block_cells.tolist()):
            cell_position = target_cell_row_by_id.get(int(cell_id))
            if cell_position is None:
                raise ValueError(
                    "Quad target cell is absent from its canonical cell bank."
                )
            for local_edge, local_vertices in enumerate(local_edges):
                if local_edge >= target_cell_edges.shape[1]:
                    raise ValueError("Quad target edge incidence is truncated.")
                target_edge_row = int(target_cell_edges[cell_position, local_edge])
                target_edge_id = int(target_edge_ids[target_edge_row])
                if target_edge_id in seen_target_edge_ids:
                    continue
                source_edge_id = int(source_edge_lineage[target_edge_row])
                source_ancestry = source_parameters_by_edge.get(source_edge_id)
                if source_edge_id < 0 or source_ancestry is None:
                    continue
                seen_target_edge_ids.add(target_edge_id)
                source_vertex_parameters, chart_id, entity_id, occurrence = (
                    source_ancestry
                )
                source_edge_row = source_edge_row_by_id[source_edge_id]
                source_edge_endpoints = tuple(
                    int(source_vertex_ids[int(value)])
                    for value in source_edge_rows[source_edge_row]
                )
                source_start = source_vertex_parameters[source_edge_endpoints[0]]
                source_stop = source_vertex_parameters[source_edge_endpoints[1]]
                edge_vertex_rows = tuple(
                    int(value) for value in target_edge_rows[target_edge_row]
                )
                edge_fractions = {
                    vertex_row: Fraction(
                        int(parameter_numerators[target_edge_row, endpoint]),
                        int(parameter_denominators[target_edge_row, endpoint]),
                    )
                    for endpoint, vertex_row in enumerate(edge_vertex_rows)
                }
                target_local_vertices = tuple(
                    int(value)
                    for value in block_vertices[cell_row, np.asarray(local_vertices)]
                )
                if set(target_local_vertices) != set(edge_vertex_rows):
                    raise ValueError(
                        "Quad target local edge changes its target edge incidence."
                    )
                edge_parameters = tuple(
                    source_start
                    + edge_fractions[vertex_row] * (source_stop - source_start)
                    for vertex_row in target_local_vertices
                )
                if edge_parameters[0] == edge_parameters[1]:
                    raise ValueError("Quad target source interval has zero width.")
                witness_cells.append(int(cell_id))
                witness_edges.append(int(target_edge_ids[target_edge_row]))
                witness_local_edges.append(local_edge)
                target_reference_row = target_reference_vertices[
                    np.asarray(local_vertices, dtype=np.int64)
                ]
                target_references.append(target_reference_row)
                target_reference_rational_rows.append(
                    tuple(
                        tuple(Fraction.from_float(float(value)) for value in point)
                        for point in target_reference_row
                    )
                )
                source_chart_ids.append(chart_id)
                source_entity_ids.append(entity_id)
                occurrences.append(tuple(occurrence))
                parameters.append((float(edge_parameters[0]), float(edge_parameters[1])))
                source_parameter_rational_rows.append(
                    tuple((value,) for value in edge_parameters)
                )
                orientations.append(1 if edge_parameters[1] > edge_parameters[0] else -1)
    if not witness_cells:
        raise ValueError(
            "Quad extraction produced no target edge for the requested source curve."
        )
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=source_witness.source_query_id,
        chart_dimension=1,
        reference_dimension=2,
        cell_global_ids=np.asarray(witness_cells, dtype=np.int64),
        entity_global_ids=np.asarray(witness_edges, dtype=np.int64),
        local_entity_indices=np.asarray(witness_local_edges, dtype=np.int32),
        target_reference_vertices=np.asarray(target_references, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            target_reference_rational_rows
        ),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrences),
        source_parameters=np.asarray(parameters, dtype=np.float64)[:, :, None],
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    source_fidelity_owner.validate_source_fidelity_witness(mesh, geometry, query, witness)
    return witness


def _dual_quad_point_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    extraction: DualExtraction,
    query: ParametricCurveBoundarySource,
    source_witness: SourceBoundaryPointWitness,
    /,
) -> SourceBoundaryPointWitness:
    source_mesh = extraction.source
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    supports = np.asarray(extraction.vertex_supports, dtype=np.int64)
    target_ids: list[int] = []
    for source_id in source_witness.vertex_global_ids.tolist():
        source_row = int(np.searchsorted(source_vertex_ids, source_id))
        source_rows = np.flatnonzero(source_vertex_ids == source_id)
        if source_rows.size != 1:
            raise ValueError(
                "Planar point witness names an absent source scaffold vertex."
            )
        source_row = int(source_rows[0])
        matches = np.flatnonzero(
            (supports[:, 0] == source_row) & np.all(supports[:, 1:] < 0, axis=1)
        )
        if matches.size != 1:
            raise ValueError(
                "Quad extraction does not retain one exact vertex for a protected corner."
            )
        target_ids.append(int(target_vertex_ids[int(matches[0])]))
    witness = SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=source_witness.source_query_id,
        vertex_global_ids=np.asarray(target_ids, dtype=np.int64),
        source_entity_ids=source_witness.source_entity_ids,
        occurrence_paths=source_witness.occurrence_paths,
    )
    source_fidelity_owner.validate_source_fidelity_witness(mesh, geometry, query, witness)
    return witness


def _dual_quad_feature_witness(
    source: NativePlanarSource,
    prepared: PreparedPlanarDomain,
    scaffold: NativeMeshConstruction,
    extraction: DualExtraction,
    feature: ProtectedFeature,
    /,
) -> tuple[
    ParametricCurveBoundarySource, SourceBoundaryChartWitness | SourceBoundaryPointWitness
]:
    geometry = extraction.geometry
    if geometry is None:
        raise ValueError("Quad feature ancestry requires the extracted coordinate map.")
    feature_ids = np.asarray(feature.scope.entity_ids, dtype=np.int64)
    if feature.feature_kind is FeatureKind.CURVE:
        query = _planar_curve_source_query(prepared, feature_ids)
        witness_type = SourceBoundaryChartWitness
    elif feature.feature_kind is FeatureKind.CORNER:
        query = _planar_curve_source_query(prepared)
        witness_type = SourceBoundaryPointWitness
    else:
        raise ValueError("Dual quads require authored curve or corner feature ancestry.")
    root_query, root_witness = _planar_scaffold_feature_witness(
        source, feature, scaffold, query, witness_type
    )
    if isinstance(root_witness, SourceBoundaryChartWitness):
        witness = _dual_quad_curve_witness(
            extraction.mesh, geometry, extraction, root_query, root_witness
        )
    else:
        witness = _dual_quad_point_witness(
            extraction.mesh, geometry, extraction, root_query, root_witness
        )
    return root_query, witness


def _cell_entity_rows(mesh: CellMesh, dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if dimension == 3:
        rows_by_id = {
            int(identifier): np.asarray(cell, dtype=np.int64)
            for block in mesh.blocks
            for identifier, cell in zip(
                np.asarray(block.global_ids, dtype=np.int64).tolist(),
                np.asarray(block.vertices, dtype=np.int64),
                strict=True,
            )
        }
        if len(rows_by_id) != identifiers.size or set(rows_by_id) != set(
            map(int, identifiers.tolist())
        ):
            raise ValueError(
                "Dual volume cell IDs do not bijectively bind connectivity rows."
            )
        width = max(block.vertices.shape[1] for block in mesh.blocks)
        rows = np.full((identifiers.size, width), -1, dtype=np.int64)
        for row, identifier in enumerate(identifiers.tolist()):
            cell = rows_by_id[int(identifier)]
            rows[row, : cell.size] = cell
    else:
        rows = _entity_vertex_rows(mesh, dimension)
    if rows.shape[0] != identifiers.size:
        raise ValueError("Cell entity rows and stable entity IDs must have equal extent.")
    return identifiers, rows


def _dual_hex_organization(
    source: NativePlcSource | NativeMappedHexSource,
    base: NativeMeshConstruction,
    mesh: CellMesh,
    extraction: DualExtraction | HexDominantConstruction | BalancedCutHexConstruction,
    /,
    *,
    association_kind: GeometryAssociationKind = GeometryAssociationKind.PIECEWISE_LINEAR,
) -> NativeMeshOrganization:
    base_mesh = base.mesh
    supports = np.asarray(extraction.vertex_supports, dtype=np.int64)
    base_cell_ids = np.asarray(base_mesh.blocks[0].global_ids, dtype=np.int64)
    base_entity_data = {
        dimension: _cell_entity_rows(base_mesh, dimension) for dimension in range(4)
    }
    target_entity_data = {
        dimension: _cell_entity_rows(mesh, dimension) for dimension in range(4)
    }
    base_entity_lookup: dict[int, dict[tuple[int, ...], int]] = {}
    base_associations: dict[int, GeometryAssociation] = {}
    base_assoc_rows: dict[int, dict[int, int]] = {}
    for dimension in range(4):
        identifiers, rows = base_entity_data[dimension]
        if dimension == 0:
            base_entity_lookup[dimension] = {
                (int(entity[0]),): row for row, entity in enumerate(rows.tolist())
            }
        else:
            base_entity_lookup[dimension] = {
                tuple(sorted(int(value) for value in vertices if value >= 0)): row
                for row, vertices in enumerate(rows.tolist())
            }
        association = _association_for_dimension(base.organization, base_mesh, dimension)
        base_associations[dimension] = association
        base_assoc_rows[dimension] = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(association.target_global_ids, dtype=np.int64).tolist()
            )
        }

    parent_rows = np.asarray(
        [
            int(extraction.parent_cells[int(extraction.parent_offsets[row])])
            for row in range(mesh.connectivity.cell_count)
        ],
        dtype=np.int64,
    )
    target_parent_global_ids = base_cell_ids[parent_rows]
    source_cell_association = base_associations[3]
    source_cell_assoc_rows = base_assoc_rows[3]
    source_cell_association_rows = np.asarray(
        [
            source_cell_assoc_rows[int(identifier)]
            for identifier in target_parent_global_ids
        ],
        dtype=np.int64,
    )
    fallback_region_ids = tuple(
        source_cell_association.source_entity_ids[row]
        for row in source_cell_association_rows.tolist()
    )

    def choose_source_entry(
        support: tuple[int, ...],
        parent_row: int,
        /,
    ) -> tuple[int, int] | None:
        dimension = len(support) - 1
        if 0 <= dimension <= 3:
            entity_row = base_entity_lookup[dimension].get(
                (support[0],) if dimension == 0 else support
            )
            if entity_row is not None:
                entity_id = int(base_entity_data[dimension][0][entity_row])
                association_row = base_assoc_rows[dimension].get(entity_id)
                if association_row is not None:
                    return dimension, association_row
        parent_id = int(base_cell_ids[parent_row])
        row = base_assoc_rows[3].get(parent_id)
        return None if row is None else (3, row)

    output_associations: dict[int, GeometryAssociation] = {}
    for dimension in range(4):
        target_ids, entity_rows = target_entity_data[dimension]
        source_ids: list[str] = []
        roles: list[GeometrySourceEntityRole | None] = []
        residuals: list[float] = []
        occurrence_paths: list[tuple[str, ...]] = []
        region_sides: list[tuple[str | None, str | None]] = []
        for entity_row, entity in enumerate(entity_rows.tolist()):
            parent_row = (
                int(parent_rows[entity_row]) if dimension == 3 else int(parent_rows[0])
            )
            if dimension == 0:
                support = tuple(
                    int(value)
                    for value in supports[int(entity[0])].tolist()
                    if int(value) >= 0
                )
            else:
                support = tuple(
                    sorted(
                        {
                            int(value)
                            for vertex in entity
                            if vertex >= 0
                            for value in supports[int(vertex)].tolist()
                            if int(value) >= 0
                        }
                    )
                )
            selected = choose_source_entry(support, parent_row)
            if selected is None:
                source_ids.append(fallback_region_ids[parent_row])
                roles.append(GeometrySourceEntityRole.REGION)
                residuals.append(0.0)
                occurrence_paths.append(())
                region_sides.append((None, None))
                continue
            source_dimension, base_row = selected
            association = base_associations[source_dimension]
            source_ids.append(association.source_entity_ids[base_row])
            roles.append(association.source_roles[base_row])
            residuals.append(float(np.asarray(association.residuals)[base_row]))
            occurrence_paths.append(association.source_occurrence_paths[base_row])
            region_sides.append(association.source_region_sides[base_row])
        entity_set = mesh.entity_set(dimension)
        output_associations[dimension] = GeometryAssociation(
            association_kind,
            source.source_id,
            source.source_revision,
            entity_set.entity_set_id,
            target_ids,
            tuple(source_ids),
            np.asarray(residuals, dtype=np.float64),
            exact=bool(np.all(np.asarray(residuals, dtype=np.float64) == 0.0)),
            source_roles=tuple(roles),
            source_occurrence_paths=tuple(occurrence_paths),
            source_region_sides=tuple(region_sides),
        )

    target_cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    zones = []
    # Zone IDs are bound to their mesh scope, so patch adjacency must name the
    # rebuilt output zones rather than the scaffold zones.
    zone_ids: dict[str, str] = {}
    for zone in base.organization.zones:
        source_zone_cells = set(
            np.asarray(zone.scope.entity_ids, dtype=np.int64).tolist()
        )
        selected = target_cell_ids[
            np.isin(
                target_parent_global_ids,
                np.asarray(tuple(source_zone_cells), dtype=np.int64),
            )
        ]
        if selected.size:
            output_zone = MeshZone(
                zone.name,
                zone.role,
                _mesh_scope(mesh, 3, selected),
                material_id=zone.material_id,
                region_role=zone.region_role,
            )
            zone_ids[zone.zone_id] = output_zone.zone_id
            zones.append(output_zone)

    labels = []
    for label in base.organization.labels:
        dimension = label.scope.entity_dimension
        if dimension not in output_associations:
            continue
        base_association = base_associations[dimension]
        selected_source_ids = {
            base_association.source_entity_ids[
                base_assoc_rows[dimension][int(identifier)]
            ]
            for identifier in np.asarray(label.scope.entity_ids, dtype=np.int64).tolist()
            if int(identifier) in base_assoc_rows[dimension]
        }
        target_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        target_association = output_associations[dimension]
        selected = np.asarray(
            [
                int(identifier)
                for identifier, source_id in zip(
                    target_ids.tolist(), target_association.source_entity_ids, strict=True
                )
                if source_id in selected_source_ids
            ],
            dtype=np.int64,
        )
        if selected.size:
            labels.append(MeshLabel(label.name, _mesh_scope(mesh, dimension, selected)))

    patches = []
    for patch in base.organization.patches:
        dimension = patch.scope.entity_dimension
        if dimension not in output_associations:
            continue
        base_association = base_associations[dimension]
        selected_source_ids = {
            base_association.source_entity_ids[
                base_assoc_rows[dimension][int(identifier)]
            ]
            for identifier in np.asarray(patch.scope.entity_ids, dtype=np.int64).tolist()
            if int(identifier) in base_assoc_rows[dimension]
        }
        target_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        target_association = output_associations[dimension]
        selected = np.asarray(
            [
                int(identifier)
                for identifier, source_id in zip(
                    target_ids.tolist(), target_association.source_entity_ids, strict=True
                )
                if source_id in selected_source_ids
            ],
            dtype=np.int64,
        )
        if selected.size:
            if any(value not in zone_ids for value in patch.adjacent_zone_ids):
                raise ValueError(
                    "Dual patch adjacency names a scaffold zone without output cells."
                )
            patches.append(
                MeshPatch(
                    patch.name,
                    _mesh_scope(mesh, dimension, selected),
                    connected=patch.connected,
                    adjacent_zone_ids=tuple(
                        zone_ids[value] for value in patch.adjacent_zone_ids
                    ),
                    source_adjacent_region_ids=patch.source_adjacent_region_ids,
                    source_adjacent_region_occurrence_paths=(
                        patch.source_adjacent_region_occurrence_paths
                    ),
                )
            )
    return NativeMeshOrganization(
        boundary=base.organization.boundary,
        patches=tuple(patches),
        zones=tuple(zones),
        labels=tuple(labels),
        associations=tuple(output_associations[dimension] for dimension in (3, 2, 1, 0)),
        surface_source=base.organization.surface_source,
    )


def _dual_hex_compliance(
    source: NativePlcSource | NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    mesh: CellMesh,
    organization: NativeMeshOrganization,
    /,
) -> MeshingComplianceReport:
    pairs = np.sort(_entity_vertex_rows(mesh, 1), axis=1)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    lengths = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
    edge_association = _association_for_dimension(organization, mesh, 1)
    edge_source_ids = edge_association.source_entity_ids
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    policy = specification.size_compliance
    for control in _uniform_size_controls(specification):
        key = f"size:{control.control_id}"
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        if control.scope.entity_dimension != 2:
            issues.append(f"unsupported_size_scope:{control.control_id}")
            continue
        selected_source_ids = {
            source.source_entity_key_for_scope(2, int(identifier))[0]
            for identifier in np.asarray(
                control.scope.entity_ids, dtype=np.int64
            ).tolist()
        }
        selected = np.asarray(
            [
                row
                for row, source_id in enumerate(edge_source_ids)
                if source_id in selected_source_ids
            ],
            dtype=np.int64,
        )
        selected_lengths = lengths[selected]
        if not selected_lengths.size:
            issues.append(f"missing_controlled_edges:{control.control_id}")
            continue
        statistics = {
            name: float(np.quantile(selected_lengths, 0.5 if name == "p50" else 0.95))
            for name in policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(selected_lengths))),
                (f"{key}:maximum_edge", float(np.max(selected_lengths))),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if (
                control.minimum_size is not None
                and float(np.min(selected_lengths)) < control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            if (
                control.maximum_size is not None
                and float(np.max(selected_lengths)) > control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
    if specification.maximum_deviation is not None:
        requested.append(("source:maximum_deviation", specification.maximum_deviation))
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


type _FacePoint = tuple[Fraction, Fraction]


def _face_cross(first: _FacePoint, second: _FacePoint, /) -> Fraction:
    return first[0] * second[1] - first[1] * second[0]


def _face_area2(points: tuple[_FacePoint, ...], /) -> Fraction:
    return sum(
        (
            _face_cross(points[row], points[(row + 1) % len(points)])
            for row in range(len(points))
        ),
        Fraction(0),
    )


def _face_barycentric(
    point: _FacePoint, triangle: tuple[_FacePoint, ...], /
) -> tuple[Fraction, Fraction, Fraction]:
    first = (triangle[1][0] - triangle[0][0], triangle[1][1] - triangle[0][1])
    second = (triangle[2][0] - triangle[0][0], triangle[2][1] - triangle[0][1])
    delta = (point[0] - triangle[0][0], point[1] - triangle[0][1])
    determinant = _face_cross(first, second)
    if determinant == 0:
        raise ValueError("Restricted boundary chart triangle is singular.")
    s = _face_cross(delta, second) / determinant
    t = _face_cross(first, delta) / determinant
    return 1 - s - t, s, t


def _clip_convex_by_triangle(
    subject: tuple[_FacePoint, ...], triangle: tuple[_FacePoint, ...], /
) -> tuple[_FacePoint, ...]:
    """Exactly intersect a convex polygon with a triangle (Sutherland-Hodgman)."""
    clip = triangle if _face_area2(triangle) > 0 else tuple(reversed(triangle))
    polygon = list(subject)
    for edge_row in range(3):
        if not polygon:
            break
        edge_start = clip[edge_row]
        edge_end = clip[(edge_row + 1) % 3]
        direction = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
        result: list[_FacePoint] = []
        start = polygon[-1]
        start_gap = _face_cross(
            direction, (start[0] - edge_start[0], start[1] - edge_start[1])
        )
        for end in polygon:
            end_gap = _face_cross(
                direction, (end[0] - edge_start[0], end[1] - edge_start[1])
            )
            if (start_gap >= 0) != (end_gap >= 0):
                fraction = start_gap / (start_gap - end_gap)
                crossing = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                if not result or result[-1] != crossing:
                    result.append(crossing)
            if end_gap >= 0 and (not result or result[-1] != end):
                result.append(end)
            start, start_gap = end, end_gap
        if len(result) > 1 and result[0] == result[-1]:
            result.pop()
        polygon = result
    changed = True
    while changed and len(polygon) >= 3:
        changed = False
        for row in range(len(polygon)):
            previous, current = polygon[row - 1], polygon[row]
            following = polygon[(row + 1) % len(polygon)]
            if (
                _face_cross(
                    (current[0] - previous[0], current[1] - previous[1]),
                    (following[0] - current[0], following[1] - current[1]),
                )
                == 0
            ):
                polygon.pop(row)
                changed = True
                break
    return tuple(polygon)


def _reference_face_frame(
    cell_kind: str, local_face: int, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
    topology = reference_cell_topology(cell_kind)
    vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in topology.vertices
    )
    face = tuple(int(value) for value in topology.entities[2][local_face])
    origin = vertices[face[0]]
    first = tuple(vertices[face[1]][axis] - origin[axis] for axis in range(3))
    other = vertices[face[2]] if len(face) == 3 else vertices[face[3]]
    second = tuple(other[axis] - origin[axis] for axis in range(3))
    return origin, first, second


def _reference_face_coordinates(
    frame: tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]],
    point: tuple[Fraction, ...],
    /,
) -> _FacePoint | None:
    """Exact face-frame coordinates of a reference point, or None off the face."""
    origin, first, second = frame
    for left in range(3):
        for right in range(left + 1, 3):
            determinant = first[left] * second[right] - first[right] * second[left]
            if determinant == 0:
                continue
            dx, dy = point[left] - origin[left], point[right] - origin[right]
            a = (dx * second[right] - dy * second[left]) / determinant
            b = (first[left] * dy - first[right] * dx) / determinant
            if any(
                origin[axis] + a * first[axis] + b * second[axis] != point[axis]
                for axis in range(3)
            ):
                return None
            return a, b
    raise ValueError("Reference face frame is singular.")


type _ReferencePoint = tuple[Fraction, ...]

# Largest per-axis refinement of one twisted quadrilateral face piece bank.
_MAXIMUM_FACE_SUBDIVISIONS = 64


def _bilinear_point(
    corners: tuple[_ReferencePoint, ...], a: Fraction, b: Fraction, /
) -> _ReferencePoint:
    weights = ((1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b)
    return tuple(
        sum(
            (
                weight * corner[axis]
                for weight, corner in zip(weights, corners, strict=True)
            ),
            Fraction(0),
        )
        for axis in range(len(corners[0]))
    )


def _require_bilinear_face_chart(
    chart: tuple[ExactReferenceChartExpression, ...] | None,
    reference: tuple[_ReferencePoint, ...],
    parent: tuple[_ReferencePoint, ...],
    /,
) -> None:
    """Prove a child chart restricts to its quadrilateral face bilinearly."""
    if any(
        reference[2][axis] != reference[1][axis] + reference[3][axis] - reference[0][axis]
        for axis in range(3)
    ):
        raise ValueError("Reference quadrilateral faces must be parallelograms.")
    if chart is None:
        # Corner-interpolating hexahedron and pyramid maps are bilinear on
        # their quadrilateral faces.
        return
    a, b = axes(2)
    face_parameters = tuple(
        add(
            add(
                constant(reference[0][axis], 2),
                scale(a, reference[1][axis] - reference[0][axis]),
            ),
            scale(b, reference[3][axis] - reference[0][axis]),
        )
        for axis in range(3)
    )
    twist_monomial: dict[tuple[int, ...], Fraction] = {(1, 1): Fraction(1)}
    for axis, expression in enumerate(chart):
        bilinear = add(
            add(
                add(
                    constant(parent[0][axis], 2),
                    scale(a, parent[1][axis] - parent[0][axis]),
                ),
                scale(b, parent[3][axis] - parent[0][axis]),
            ),
            scale(
                twist_monomial,
                parent[0][axis] - parent[1][axis] + parent[2][axis] - parent[3][axis],
            ),
        )
        if not expression_equal(
            compose_expression(expression, face_parameters), bilinear
        ):
            raise ValueError(
                "A restricted quadrilateral boundary face is not bilinear in its parent."
            )


def _cell_map_lipschitz_squared(
    element: ScalarCoordinateElement,
    local: CoordinateSourceBank,
    row: int,
    /,
) -> Fraction:
    """Bound the squared spectral norm of a simplex cell map's Jacobian.

    Each exact Jacobian entry is enclosed by its largest absolute Bernstein
    coefficient on the reference simplex; their squared sum bounds the
    Frobenius, hence spectral, norm.
    """
    total = Fraction(0)
    for expression in coordinate_map_expressions(element, local, cell_row=row):
        if isinstance(expression, RationalPolynomial):
            raise ValueError(
                "Twisted restricted boundary faces require polynomial scaffold maps."
            )
        for axis in range(3):
            controls = tuple(
                rational_coefficient(value, role="Scaffold Jacobian control")
                for value in bernstein_coefficients(
                    derivative(expression, axis), "simplex", 3
                )
            )
            largest = max((abs(value) for value in controls), default=Fraction(0))
            total += largest * largest
    return total


def _bilinear_face_subdivisions(
    lipschitz_squared: Fraction, twist_squared: Fraction, tolerance: float | None, /
) -> int:
    """Smallest dyadic refinement keeping twisted-face pieces within tolerance/2.

    On a triangle of an ``n``-by-``n`` split, a bilinear face departs from its
    straight interpolant by at most ``|twist| / (4 n**2)`` in parent reference
    coordinates, so the physical correspondence error is at most
    ``L |twist| / (4 n**2)``.
    """
    if tolerance is None or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "Twisted restricted boundary faces require a positive fidelity tolerance."
        )
    bound = Fraction.from_float(float(tolerance))
    subdivisions = 1
    while lipschitz_squared * twist_squared > 4 * subdivisions**4 * bound * bound:
        subdivisions *= 2
        if subdivisions > _MAXIMUM_FACE_SUBDIVISIONS:
            raise ValueError(
                "A twisted restricted boundary face needs more than "
                f"{_MAXIMUM_FACE_SUBDIVISIONS} subdivisions per axis to meet its "
                "fidelity tolerance."
            )
    return subdivisions


def _quad_face_pieces(
    reference: tuple[_ReferencePoint, ...],
    parent: tuple[_ReferencePoint, ...],
    frame: tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]],
    subdivisions: int,
    /,
) -> tuple[tuple[tuple[_FacePoint, ...], tuple[_ReferencePoint, ...]], ...]:
    """Split a bilinear quadrilateral face into straight parent-face triangles.

    Corners of every piece are exact images of the bilinear face map; outer
    edges are affine, so the straight pieces tile the parent face exactly.
    """
    planar_face = tuple(_reference_face_coordinates(frame, point) for point in parent)
    if any(point is None for point in planar_face):
        raise ValueError("Restricted quadrilateral face leaves its parent face.")
    area = _face_area2(tuple(point for point in planar_face if point is not None))
    pieces: list[tuple[tuple[_FacePoint, ...], tuple[_ReferencePoint, ...]]] = []
    step = Fraction(1, subdivisions)
    for first in range(subdivisions):
        for second in range(subdivisions):
            parameters = (
                (first * step, second * step),
                ((first + 1) * step, second * step),
                ((first + 1) * step, (second + 1) * step),
                (first * step, (second + 1) * step),
            )
            references = tuple(_bilinear_point(reference, a, b) for a, b in parameters)
            planar = tuple(
                _reference_face_coordinates(frame, _bilinear_point(parent, a, b))
                for a, b in parameters
            )
            if any(point is None for point in planar):
                raise ValueError("Restricted quadrilateral face leaves its parent face.")
            corners = tuple(point for point in planar if point is not None)
            split = next(
                (
                    candidate
                    for candidate in (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3)))
                    if all(
                        _face_area2(tuple(corners[index] for index in triangle)) * area
                        > 0
                        for triangle in candidate
                    )
                ),
                None,
            )
            if split is None:
                raise ValueError(
                    "Restricted quadrilateral boundary face has no positive "
                    "triangle split in its parent face."
                )
            pieces.extend(
                (
                    tuple(corners[index] for index in triangle),
                    tuple(references[index] for index in triangle),
                )
                for triangle in split
            )
    return tuple(pieces)


def _restricted_boundary_chart_witness(
    scaffold_mesh: CellMesh,
    scaffold_geometry: CellGeometrySpec,
    scaffold_witness: SourceBoundaryChartWitness,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    tolerance: float | None,
    /,
) -> SourceBoundaryChartWitness:
    """Compose a scaffold boundary chart witness through exact child-in-parent maps.

    Every output face on a witnessed scaffold face is split into reference
    triangles whose exact parent-reference corner images span straight
    triangles of that face; these are clipped exactly against the scaffold's
    source pieces, so target and source pieces partition the same face.
    Twisted (non-parallelogram) bilinear quadrilateral faces are refined until
    the straight correspondence stays within half the fidelity tolerance.
    """
    if scaffold_witness.chart_dimension != 2 or scaffold_witness.reference_dimension != 3:
        raise ValueError(
            "Restricted boundary witnesses compose only volume face chart witnesses."
        )
    restriction = geometry.restriction_source
    if restriction is None:
        raise ValueError(
            "A restricted boundary witness requires the output's exact parent restriction."
        )
    scaffold_kinds = {block.cell_kind for block in scaffold_mesh.blocks}
    if len(scaffold_kinds) != 1:
        raise ValueError("Restricted boundary witnesses require one scaffold cell kind.")
    scaffold_kind = next(iter(scaffold_kinds))
    scaffold_topology = reference_cell_topology(scaffold_kind)
    scaffold_frames = tuple(
        _reference_face_frame(scaffold_kind, local)
        for local in range(len(scaffold_topology.entities[2]))
    )
    pieces_by_face: dict[
        tuple[int, int],
        list[tuple[tuple[_FacePoint, ...], tuple[_FacePoint, ...], int]],
    ] = {}
    scaffold_ids = np.asarray(scaffold_witness.cell_global_ids, dtype=np.int64)
    scaffold_locals = np.asarray(scaffold_witness.local_entity_indices, dtype=np.int64)
    for row in range(scaffold_ids.size):
        local = int(scaffold_locals[row])
        target = tuple(
            tuple(Fraction(numerator, denominator) for numerator, denominator in point)
            for point in scaffold_witness.target_reference_rational_tokens[row]
        )
        planar = tuple(
            _reference_face_coordinates(scaffold_frames[local], point) for point in target
        )
        if any(point is None for point in planar):
            raise ValueError("Scaffold chart witness leaves its declared parent face.")
        source_points = tuple(
            (Fraction(*point[0]), Fraction(*point[1]))
            for point in scaffold_witness.source_parameter_rational_tokens[row]
        )
        pieces_by_face.setdefault((int(scaffold_ids[row]), local), []).append(
            (tuple(point for point in planar if point is not None), source_points, row)
        )

    reference_maps = dict(
        zip(restriction.block_names, restriction.parent_reference_maps, strict=True)
    )
    facets = source_fidelity_owner._volume_boundary_facet_lookup(mesh)
    scaffold_elements, scaffold_routes, _ = scaffold_geometry.resolve(scaffold_mesh)
    scaffold_coordinates = scaffold_geometry.source_coordinates()
    scaffold_rows = {
        int(global_id): (block_index, row)
        for block_index, block in enumerate(scaffold_mesh.blocks)
        for row, global_id in enumerate(np.asarray(block.global_ids).tolist())
    }
    lipschitz_squared: dict[int, Fraction] = {}

    def scaffold_lipschitz_squared(cell_id: int, /) -> Fraction:
        cached = lipschitz_squared.get(cell_id)
        if cached is not None:
            return cached
        block_index, row = scaffold_rows[cell_id]
        scaffold_kind = scaffold_mesh.blocks[block_index].cell_kind
        if reference_cell_topology(scaffold_kind).family != "tetrahedron":
            raise ValueError(
                "Twisted restricted boundary faces require tetrahedral scaffold cells."
            )
        element = scaffold_elements[block_index]
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError("Scaffold cells require scalar coordinate elements.")
        route = np.asarray(scaffold_routes[block_index][row], dtype=np.int64)
        value = _cell_map_lipschitz_squared(
            element,
            tuple(scaffold_coordinates[int(index)] for index in route.tolist()),
            row,
        )
        lipschitz_squared[cell_id] = value
        return value

    cell_ids: list[int] = []
    face_ids: list[int] = []
    local_faces: list[int] = []
    target_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    chart_ids: list[str] = []
    entity_ids: list[str] = []
    occurrences: list[tuple[str, ...]] = []
    parameter_rows: list[tuple[tuple[Fraction, Fraction], ...]] = []
    orientations: list[int] = []
    for block in mesh.blocks:
        witness = reference_maps.get(block.name)
        if witness is None:
            raise ValueError(
                f"Output block {block.name!r} lacks the exact parent reference map "
                "needed to restrict the scaffold boundary witness."
            )
        parent_corners = exact_reference_cell_vertices(witness)
        parent_ids = np.asarray(witness.source_cell_ids, dtype=np.int64)
        target_ids = np.asarray(witness.target_cell_ids, dtype=np.int64)
        if not np.array_equal(target_ids, np.arange(block.cell_count, dtype=np.int64)):
            raise ValueError("Output parent reference maps must cover block rows.")
        topology = reference_cell_topology(block.cell_kind)
        reference_vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in topology.vertices
        )
        global_ids = np.asarray(block.global_ids, dtype=np.int64)
        for row in range(block.cell_count):
            parent_id = int(parent_ids[row])
            corners = parent_corners[row]
            for local_face, face in enumerate(topology.entities[2]):
                face_vertices = tuple(int(value) for value in face)
                parent_face = None
                planar_corners: tuple[_FacePoint, ...] = ()
                for scaffold_local, frame in enumerate(scaffold_frames):
                    if (parent_id, scaffold_local) not in pieces_by_face:
                        continue
                    projected = tuple(
                        _reference_face_coordinates(frame, corners[index])
                        for index in face_vertices
                    )
                    if all(point is not None for point in projected):
                        parent_face = scaffold_local
                        planar_corners = tuple(
                            point for point in projected if point is not None
                        )
                        break
                if parent_face is None:
                    continue
                cell_id = int(global_ids[row])
                facet = facets.get((cell_id, local_face))
                if facet is None:
                    raise ValueError(
                        "Restricted boundary face is absent from target facet routes."
                    )
                frame = _reference_face_frame(block.cell_kind, local_face)
                face_reference = tuple(
                    reference_vertices[index] for index in face_vertices
                )
                if len(face_vertices) == 3:
                    face_pieces = ((planar_corners, face_reference),)
                else:
                    face_parent = tuple(
                        tuple(
                            rational_coefficient(value, role="Parent face corner")
                            for value in corners[index]
                        )
                        for index in face_vertices
                    )
                    twist = tuple(
                        face_parent[0][axis]
                        - face_parent[1][axis]
                        + face_parent[2][axis]
                        - face_parent[3][axis]
                        for axis in range(3)
                    )
                    subdivisions = 1
                    if any(twist):
                        _require_bilinear_face_chart(
                            None
                            if witness.source_reference_chart_arguments is None
                            else witness.source_reference_chart_arguments[row],
                            face_reference,
                            face_parent,
                        )
                        subdivisions = _bilinear_face_subdivisions(
                            scaffold_lipschitz_squared(parent_id),
                            sum((value * value for value in twist), Fraction(0)),
                            tolerance,
                        )
                    face_pieces = _quad_face_pieces(
                        face_reference,
                        face_parent,
                        scaffold_frames[parent_face],
                        subdivisions,
                    )
                for straight, reference in face_pieces:
                    for pieces, source_points, scaffold_row in pieces_by_face[
                        (parent_id, parent_face)
                    ]:
                        clipped = _clip_convex_by_triangle(straight, pieces)
                        for fan in range(1, len(clipped) - 1):
                            piece = (clipped[0], clipped[fan], clipped[fan + 1])
                            if _face_area2(piece) == 0:
                                continue
                            target_triangle = []
                            parameters = []
                            for point in piece:
                                weights = _face_barycentric(point, straight)
                                target_triangle.append(
                                    tuple(
                                        sum(
                                            (
                                                weight * corner[axis]
                                                for weight, corner in zip(
                                                    weights, reference, strict=True
                                                )
                                            ),
                                            Fraction(0),
                                        )
                                        for axis in range(3)
                                    )
                                )
                                source_weights = _face_barycentric(point, pieces)
                                parameters.append(
                                    (
                                        sum(
                                            (
                                                weight * value[0]
                                                for weight, value in zip(
                                                    source_weights,
                                                    source_points,
                                                    strict=True,
                                                )
                                            ),
                                            Fraction(0),
                                        ),
                                        sum(
                                            (
                                                weight * value[1]
                                                for weight, value in zip(
                                                    source_weights,
                                                    source_points,
                                                    strict=True,
                                                )
                                            ),
                                            Fraction(0),
                                        ),
                                    )
                                )
                            local_target = tuple(
                                _reference_face_coordinates(frame, point)
                                for point in target_triangle
                            )
                            if any(point is None for point in local_target):
                                raise ValueError(
                                    "Restricted boundary piece leaves its target face."
                                )
                            target_sign = _face_area2(
                                tuple(
                                    point for point in local_target if point is not None
                                )
                            )
                            source_sign = _face_area2(tuple(parameters))
                            if target_sign == 0 or source_sign == 0:
                                raise ValueError(
                                    "Restricted boundary chart piece is degenerate."
                                )
                            cell_ids.append(cell_id)
                            face_ids.append(int(facet[0]))
                            local_faces.append(local_face)
                            target_rows.append(tuple(target_triangle))
                            chart_ids.append(
                                scaffold_witness.source_chart_ids[scaffold_row]
                            )
                            entity_ids.append(
                                scaffold_witness.source_entity_ids[scaffold_row]
                            )
                            occurrences.append(
                                tuple(scaffold_witness.occurrence_paths[scaffold_row])
                            )
                            parameter_rows.append(tuple(parameters))
                            orientations.append(
                                1 if target_sign * source_sign > 0 else -1
                            )
    if not cell_ids:
        raise ValueError("Restricted boundary witness produced no output face pieces.")
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=scaffold_witness.source_id,
        source_revision=scaffold_witness.source_revision,
        source_query_id=scaffold_witness.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_faces, dtype=np.int32),
        target_reference_vertices=np.asarray(
            [[[float(value) for value in point] for point in row] for row in target_rows],
            dtype=np.float64,
        ),
        target_reference_rational_tokens=rational_chart_tokens(target_rows),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(occurrences),
        source_parameters=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in parameter_rows
            ],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=rational_chart_tokens(parameter_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    return witness


def _restricted_certification(
    scaffold: NativeMeshConstruction,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: np.ndarray,
    /,
) -> NativeCertificationRequest:
    """Carry the scaffold certification request onto an exact restriction of it."""
    request = scaffold.certification
    scaffold_geometry = (
        CellGeometrySpec.affine(scaffold.mesh)
        if scaffold.geometry is None
        else scaffold.geometry
    )
    fidelity_witness = request.fidelity_witness
    if isinstance(fidelity_witness, SourceBoundaryChartWitness):
        fidelity_witness = _restricted_boundary_chart_witness(
            scaffold.mesh,
            scaffold_geometry,
            fidelity_witness,
            mesh,
            geometry,
            request.fidelity_tolerance,
        )
    elif fidelity_witness is not None:
        raise ValueError(
            "Restricted outputs compose only volume face chart fidelity witnesses."
        )
    scoped = []
    for query, witness, bound in request.scoped_fidelity:
        if not isinstance(witness, SourceBoundaryChartWitness) or (
            witness.chart_dimension != 2
        ):
            raise ValueError(
                "Restricted outputs compose only face chart feature witnesses."
            )
        scoped.append(
            (
                query,
                _restricted_boundary_chart_witness(
                    scaffold.mesh, scaffold_geometry, witness, mesh, geometry, bound
                ),
                bound,
            )
        )
    return replace(
        request,
        cell_regions=np.asarray(cell_regions, dtype=np.int64),
        fidelity_witness=fidelity_witness,
        scoped_fidelity=tuple(scoped),
    )


@final
class _PreparedNativeDualHex(StrictModule, NonTrainableState):
    source: NativePlcSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    tetrahedral_route: PreparedPlcVolume

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        tetrahedral_route: PreparedPlcVolume,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.tetrahedral_route = tetrahedral_route

    def execute(self, /) -> NativeMeshConstruction:
        scaffold = self.tetrahedral_route.execute()
        if scaffold.certification.cell_regions is None:
            raise ValueError(
                "Native PLC tetrahedral scaffold must retain source cell regions."
            )
        extraction = extract_volume_hexes(
            scaffold.mesh,
            source_geometry=scaffold.geometry,
        )
        geometry = extraction.geometry
        if geometry is None:
            raise ValueError(
                "Native dual hexes must retain their extracted coordinate map."
            )
        parent_rows = np.asarray(extraction.parent_cells, dtype=np.int64)
        regions = np.asarray(scaffold.certification.cell_regions, dtype=np.int64)[
            parent_rows
        ]
        organization = _dual_hex_organization(
            self.source, scaffold, extraction.mesh, extraction
        )
        compliance = _dual_hex_compliance(
            self.source, self.specification, extraction.mesh, organization
        )
        certification = replace(
            _restricted_certification(scaffold, extraction.mesh, geometry, regions),
            protected_features=self.specification.protected_features,
            feature_source=self.source if self.specification.protected_features else None,
        )
        return NativeMeshConstruction(
            extraction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
            stages=scaffold.stages,
            source_mesh=scaffold.mesh,
            source_geometry=scaffold.geometry,
        )


@checked
def prepare_native_dual_hex(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeDualHex:
    issues = dual_hex_support_issues(source, specification, options)
    if issues:
        raise ValueError("Unsupported native dual-hex request: " + "; ".join(issues))
    schedule = options.volume_schedule
    if schedule is None:
        raise RuntimeError(
            "Native dual-hex route requires a tetrahedral scaffold schedule."
        )
    tetrahedral_options = NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=schedule,
    )
    tetrahedral_route = prepare_native_plc_volume(
        source,
        _tetrahedron_scaffold(specification),
        tetrahedral_options,
        coordinate_contract,
    )
    return _PreparedNativeDualHex(
        source,
        specification,
        coordinate_contract,
        tetrahedral_route,
    )


@final
class _PreparedNativeDualQuad(StrictModule, NonTrainableState):
    source: NativePlanarSource
    specification: SurfaceMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    planar_route: _PreparedPlanarRoute

    def __init__(
        self,
        source: NativePlanarSource,
        specification: SurfaceMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        planar_route: _PreparedPlanarRoute,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.planar_route = planar_route

    def execute(self, /) -> NativeMeshConstruction:
        source, specification = self.source, self.specification
        scaffold = self.planar_route.execute()
        feature_edges = _feature_scaffold_edges(source, specification, scaffold)
        field = prepare_surface_cross_field(scaffold.mesh, feature_edges=feature_edges)
        extraction = extract_surface_quads(
            scaffold.mesh,
            cross_field=field,
            source_geometry=scaffold.geometry,
            unpaired_triangles="subdivide",
        )
        geometry = extraction.geometry
        if geometry is None:
            raise ValueError(
                "Native dual quads must retain their extracted coordinate map."
            )
        organization = _quad_organization(
            source,
            specification,
            self.planar_route.prepared,
            scaffold,
            extraction.mesh,
            np.asarray(extraction.vertex_supports, dtype=np.int64),
        )
        feature_witnesses = tuple(
            _dual_quad_feature_witness(
                source,
                self.planar_route.prepared,
                scaffold,
                extraction,
                feature,
            )
            for feature in specification.protected_features
        )
        scoped_fidelity = tuple(
            (query, witness, feature.maximum_deviation)
            for feature, (query, witness) in zip(
                specification.protected_features, feature_witnesses, strict=True
            )
            if feature.hard
        )
        soft_feature_outcomes = []
        for feature, (query, witness) in zip(
            specification.protected_features, feature_witnesses, strict=True
        ):
            if feature.hard:
                continue
            certificate = source_fidelity_owner.certify_source_fidelity(
                extraction.mesh,
                geometry,
                query,
                tolerance=feature.maximum_deviation,
                witness=witness,
            )
            deviation_upper = max(
                certificate.mesh_to_source_upper,
                certificate.source_to_mesh_upper,
            )
            warning = (
                certificate.status != "certified"
                or deviation_upper > feature.maximum_deviation
            )
            soft_feature_outcomes.append((feature, deviation_upper, warning))
        compliance = _quad_compliance(
            source,
            specification,
            self.planar_route.prepared,
            extraction.mesh,
            organization,
            soft_feature_outcomes=tuple(soft_feature_outcomes),
        )
        if scaffold.certification.cell_regions is None:
            raise ValueError(
                "Planar triangle scaffold must retain source region assignments."
            )
        source_regions = np.asarray(scaffold.certification.cell_regions, dtype=np.int64)
        parent_offsets = np.asarray(extraction.parent_offsets, dtype=np.int64)
        parent_cells = np.asarray(extraction.parent_cells, dtype=np.int64)
        if source_regions.shape != (scaffold.mesh.connectivity.cell_count,):
            raise ValueError(
                "Planar scaffold regions do not align with its source cells."
            )
        regions: list[int] = []
        for first, last in zip(parent_offsets[:-1], parent_offsets[1:], strict=True):
            parent_rows = parent_cells[int(first) : int(last)]
            if parent_rows.size == 0:
                raise ValueError(
                    "Quad cell lost its exact source parent-region ancestry."
                )
            parent_regions = source_regions[parent_rows]
            unique_regions = np.unique(parent_regions)
            if unique_regions.size != 1:
                raise ValueError(
                    "Quad conversion may not merge cells across source regions."
                )
            regions.append(int(unique_regions[0]))
        certification = replace(
            scaffold.certification,
            cell_regions=np.asarray(regions, dtype=np.int64),
            protected_features=specification.protected_features,
            maximum_normal_deviation=specification.maximum_normal_deviation,
            scoped_fidelity=scoped_fidelity,
            feature_source=source,
        )
        return NativeMeshConstruction(
            extraction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
            stages=scaffold.stages,
            source_mesh=scaffold.mesh,
            source_geometry=scaffold.geometry,
            planar_embedding=specification.planar_embedding,
        )


@checked
def prepare_native_dual_quad(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeDualQuad:
    issues = quad_support_issues(source, specification, options)
    if issues:
        raise ValueError("Unsupported native dual-quad request: " + "; ".join(issues))
    scaffold_specification = _triangle_scaffold(
        specification, capture_soft_feature_witnesses=True
    )
    planar_route = prepare_native_planar(
        source, scaffold_specification, options, coordinate_contract
    )
    return _PreparedNativeDualQuad(
        source, specification, coordinate_contract, planar_route
    )


def hex_dominant_support_issues(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    if not isinstance(source, NativePlcSource):
        raise TypeError("source must be NativePlcSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    issues: list[str] = []
    if options.route != "plc_hex_dominant":
        issues.append("the plc_hex_dominant native route")
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or "hexahedron" not in families.required
        or not families.allow_mixed
        or not {"pyramid", "tetrahedron"} <= set(families.allowed_transitions)
    ):
        issues.append(
            "a mixed hexahedral target permitting pyramid and tetrahedron transitions"
        )
    if (
        specification.fill_strategy
        is not VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION
    ):
        issues.append("the declared hex-dominant simplex-transition fill strategy")
    if source.mapped_domain is not None and source.complex.geometry is None:
        issues.append("a source-coordinate map for the mapped PLC tetrahedral scaffold")
    if options.hex_core_fraction is None or not 0.0 < options.hex_core_fraction < 1.0:
        issues.append("a hex-core fraction strictly between zero and one")
    if options.volume_schedule is None:
        issues.append("a native tetrahedral scaffold schedule")
    if not issues:
        issues.extend(volume_support_issues(source, _tetrahedron_scaffold(specification)))
    return list(dict.fromkeys(issues))


def _hex_core_quality(mesh: CellMesh, /) -> tuple[int, ...]:
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    identifiers = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    ranked = []
    for row, cell in enumerate(cells):
        local = points[cell]
        edges = np.asarray(
            [
                local[right] - local[left]
                for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ],
            dtype=np.float64,
        )
        determinant = abs(
            float(np.linalg.det(np.stack((edges[0], edges[1], edges[2]), axis=-1)))
        )
        volume = determinant / 6.0
        edge_square_sum = float(np.sum(edges * edges))
        if volume <= 0.0 or edge_square_sum <= 0.0:
            raise ValueError(
                "Hex-dominant core selection encountered a degenerate source tet."
            )
        quality = 12.0 * (3.0 * volume) ** (2.0 / 3.0) / edge_square_sum
        ranked.append((-quality, int(identifiers[row]), row))
    return tuple(row for _, _, row in sorted(ranked))


@final
class _PreparedNativeHexDominant(StrictModule, NonTrainableState):
    source: NativePlcSource
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    tetrahedral_route: PreparedPlcVolume

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        tetrahedral_route: PreparedPlcVolume,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.tetrahedral_route = tetrahedral_route

    def execute(self, /) -> NativeMeshConstruction:
        scaffold = self.tetrahedral_route.execute()
        if scaffold.certification.cell_regions is None:
            raise ValueError(
                "Native PLC tetrahedral scaffold must retain source cell regions."
            )
        cell_ids = np.asarray(scaffold.mesh.blocks[0].global_ids, dtype=np.int64)
        fraction = self.options.hex_core_fraction
        if fraction is None:
            raise ValueError("Prepared hex-dominant route lost its core fraction.")
        count = int(np.floor(fraction * cell_ids.size + 0.5))
        count = min(cell_ids.size - 1, max(1, count))
        ordered_rows = _hex_core_quality(scaffold.mesh)
        core_ids = cell_ids[np.asarray(ordered_rows[:count], dtype=np.int64)]
        construction = extract_hex_dominant(
            scaffold.mesh,
            core_ids,
            source_geometry=scaffold.geometry,
        )
        parent_rows = np.asarray(construction.parent_cells, dtype=np.int64)
        regions = np.asarray(scaffold.certification.cell_regions, dtype=np.int64)[
            parent_rows
        ]
        organization = _dual_hex_organization(
            self.source,
            scaffold,
            construction.mesh,
            construction,
        )
        compliance = _dual_hex_compliance(
            self.source, self.specification, construction.mesh, organization
        )
        certification = _restricted_certification(
            scaffold, construction.mesh, construction.geometry, regions
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=construction.geometry,
            organization=organization,
            stages=scaffold.stages,
            source_mesh=scaffold.mesh,
            source_geometry=scaffold.geometry,
        )


@checked
def prepare_native_hex_dominant(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeHexDominant:
    issues = hex_dominant_support_issues(source, specification, options)
    if issues:
        raise ValueError("Unsupported native hex-dominant request: " + "; ".join(issues))
    schedule = options.volume_schedule
    if schedule is None:
        raise RuntimeError(
            "Native hex-dominant route requires a tetrahedral scaffold schedule."
        )
    tetrahedral_options = NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=schedule,
    )
    tetrahedral_route = prepare_native_plc_volume(
        source,
        _tetrahedron_scaffold(specification),
        tetrahedral_options,
        coordinate_contract,
    )
    return _PreparedNativeHexDominant(
        source,
        specification,
        options,
        coordinate_contract,
        tetrahedral_route,
    )


def balanced_hex_support_issues(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    if not isinstance(source, NativePlcSource):
        raise TypeError("source must be NativePlcSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    issues: list[str] = []
    if options.route != "plc_balanced_grid_hex":
        issues.append("the plc_balanced_grid_hex native route")
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"hexahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append("a conforming affine pure-hexahedral target")
    if source.mapped_domain is not None and source.complex.geometry is None:
        issues.append("a source-coordinate map for the mapped PLC tetrahedral scaffold")
    if options.volume_schedule is None:
        issues.append("a native tetrahedral scaffold schedule")
    if options.grid_schedule is None or options.grid_schedule.route != "balanced_grid":
        issues.append("a balanced-grid integer schedule")
    if specification.periodic_constraints:
        issues.append("no periodic quotient constraints")
    if specification.layer_controls:
        issues.append("no boundary-layer construction")
    if len(specification.size_controls) != 1 or not isinstance(
        specification.size_controls[0], UniformSizeControl
    ):
        issues.append(
            "an explicit physical resolution (one global uniform size control) for the partial-cell lattice"
        )
    else:
        control = specification.size_controls[0]
        facet_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
        if (
            control.scope.source_id != source.source_id
            or control.scope.source_revision != source.source_revision
            or control.scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or control.scope.entity_dimension != 2
            or not np.array_equal(
                np.sort(np.asarray(control.scope.entity_ids, dtype=np.int64)),
                np.sort(facet_ids),
            )
        ):
            issues.append("a global uniform size control over every PLC facet")
    if not issues:
        issues.extend(volume_support_issues(source, _tetrahedron_scaffold(specification)))
    return list(dict.fromkeys(issues))


@final
class _PreparedNativeBalancedGridHex(StrictModule, NonTrainableState):
    source: NativePlcSource
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    tetrahedral_route: PreparedPlcVolume

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        tetrahedral_route: PreparedPlcVolume,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.tetrahedral_route = tetrahedral_route

    def execute(self, /) -> NativeMeshConstruction:
        scaffold = self.tetrahedral_route.execute()
        if scaffold.certification.cell_regions is None:
            raise ValueError(
                "Native PLC tetrahedral scaffold must retain source cell regions."
            )
        control = _uniform_size_controls(self.specification)[0]
        schedule = self.options.grid_schedule
        if schedule is None:
            raise RuntimeError("Prepared balanced-grid route lost its grid schedule.")
        plan = prepare_balanced_cut_grid(
            self.source.complex,
            scaffold.mesh,
            schedule,
            control.target_size,
            source_id=self.source.source_id,
            source_revision=self.source.source_revision,
        )
        construction = generate_balanced_cut_hexes(
            self.source.complex,
            scaffold.mesh,
            scaffold.certification.cell_regions,
            plan,
            schedule,
            source_geometry=scaffold.geometry,
        )
        organization = _dual_hex_organization(
            self.source,
            scaffold,
            construction.mesh,
            construction,
        )
        compliance = _dual_hex_compliance(
            self.source, self.specification, construction.mesh, organization
        )
        certification = _restricted_certification(
            scaffold,
            construction.mesh,
            construction.geometry,
            np.asarray(construction.cell_regions, dtype=np.int64),
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=construction.geometry,
            organization=organization,
            stages=scaffold.stages,
            source_mesh=scaffold.mesh,
            source_geometry=scaffold.geometry,
        )


@checked
def prepare_native_balanced_grid_hex(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeBalancedGridHex:
    issues = balanced_hex_support_issues(source, specification, options)
    if issues:
        raise ValueError(
            "Unsupported native balanced-cut hex request: " + "; ".join(issues)
        )
    schedule = options.volume_schedule
    if schedule is None:
        raise RuntimeError(
            "Native balanced-cut route requires a tetrahedral scaffold schedule."
        )
    tetrahedral_options = NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=schedule,
    )
    tetrahedral_route = prepare_native_plc_volume(
        source,
        _tetrahedron_scaffold(specification),
        tetrahedral_options,
        coordinate_contract,
    )
    return _PreparedNativeBalancedGridHex(
        source,
        specification,
        options,
        coordinate_contract,
        tetrahedral_route,
    )


def frame_hex_support_issues(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    if not isinstance(source, NativePlcSource):
        raise TypeError("source must be NativePlcSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    issues: list[str] = []
    if options.route != "plc_frame_grid_hex":
        issues.append("the plc_frame_grid_hex native route")
    families = specification.target.cell_families
    if (
        specification.target.topological_dimension != 3
        or specification.target.ambient_dimension != 3
        or specification.target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"hexahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append("a conforming affine pure-hexahedral target")
    if source.complex.geometry is not None or source.mapped_domain is not None:
        issues.append("a piecewise-linear PLC source map")
    if options.volume_schedule is None:
        issues.append("a native tetrahedral scaffold schedule")
    if options.grid_schedule is None or options.grid_schedule.route != "frame_grid":
        issues.append("a native frame-field integer schedule")
    if specification.periodic_constraints:
        issues.append("no periodic quotient constraints")
    if specification.layer_controls:
        issues.append("no boundary-layer construction")
    if len(specification.size_controls) != 1 or not isinstance(
        specification.size_controls[0], UniformSizeControl
    ):
        issues.append(
            "an explicit physical resolution (one global uniform size control) for the frame integer grid"
        )
    else:
        control = specification.size_controls[0]
        facet_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
        if (
            control.scope.source_id != source.source_id
            or control.scope.source_revision != source.source_revision
            or control.scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or control.scope.entity_dimension != 2
            or not np.array_equal(
                np.sort(np.asarray(control.scope.entity_ids, dtype=np.int64)),
                np.sort(facet_ids),
            )
        ):
            issues.append("a global uniform size control over every PLC facet")
    if not issues:
        issues.extend(volume_support_issues(source, _tetrahedron_scaffold(specification)))
    return list(dict.fromkeys(issues))


@final
class _PreparedNativeFrameGridHex(StrictModule, NonTrainableState):
    source: NativePlcSource
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    tetrahedral_route: PreparedPlcVolume

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        tetrahedral_route: PreparedPlcVolume,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.tetrahedral_route = tetrahedral_route

    def execute(self, /) -> NativeMeshConstruction:
        scaffold = self.tetrahedral_route.execute()
        if scaffold.certification.cell_regions is None:
            raise ValueError(
                "Native PLC tetrahedral scaffold must retain source cell regions."
            )
        control = _uniform_size_controls(self.specification)[0]
        schedule = self.options.grid_schedule
        if schedule is None:
            raise RuntimeError("Prepared frame-grid route lost its integer schedule.")
        prepared = prepare_frame_hex_grid(
            self.source.complex,
            scaffold.mesh,
            schedule,
            control.target_size,
            source_id=self.source.source_id,
            source_revision=self.source.source_revision,
        )
        construction = generate_frame_hexes(
            self.source.complex,
            scaffold.mesh,
            scaffold.certification.cell_regions,
            prepared,
            schedule,
            source_geometry=scaffold.geometry,
        )
        organization = _dual_hex_organization(
            self.source,
            scaffold,
            construction.mesh,
            construction,
        )
        compliance = _dual_hex_compliance(
            self.source, self.specification, construction.mesh, organization
        )
        certification = _restricted_certification(
            scaffold,
            construction.mesh,
            construction.geometry,
            np.asarray(construction.cell_regions, dtype=np.int64),
        )
        return NativeMeshConstruction(
            construction.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=construction.geometry,
            organization=organization,
            stages=scaffold.stages,
            source_mesh=scaffold.mesh,
            source_geometry=scaffold.geometry,
        )


@checked
def prepare_native_frame_grid_hex(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeFrameGridHex:
    issues = frame_hex_support_issues(source, specification, options)
    if issues:
        raise ValueError(
            "Unsupported native frame-grid hex request: " + "; ".join(issues)
        )
    volume_schedule = options.volume_schedule
    if volume_schedule is None:
        raise RuntimeError(
            "Native frame-grid route requires a tetrahedral scaffold schedule."
        )
    tetrahedral_options = NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=volume_schedule,
    )
    tetrahedral_route = prepare_native_plc_volume(
        source,
        _tetrahedron_scaffold(specification),
        tetrahedral_options,
        coordinate_contract,
    )
    return _PreparedNativeFrameGridHex(
        source,
        specification,
        options,
        coordinate_contract,
        tetrahedral_route,
    )


def _mapped_reference_boundary_matches(source: NativeMappedHexSource, /) -> bool:
    """Require the source-owned declared PLC/root relation to revalidate."""
    try:
        source.validate_restored()
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        return False
    return True


def _mapped_grid_hex_support_issues(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    route: str,
    grid_route: str,
    /,
) -> list[str]:
    if not isinstance(source, NativeMappedHexSource):
        raise TypeError("source must be NativeMappedHexSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    issues: list[str] = []
    domain = source.domain
    mesh = domain.reference_mesh
    if options.route != route:
        issues.append(f"the {route} native route")
    families = specification.target.cell_families
    if (
        specification.target.topological_dimension != 3
        or specification.target.ambient_dimension != 3
        or set((*families.required, *families.preferred)) != {"hexahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append("a conforming pure-hexahedral target")
    if options.grid_schedule is None or options.grid_schedule.route != grid_route:
        issues.append(f"a {grid_route} integer schedule")
    try:
        source.validate_restored()
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        issues.append("the declared PLC-to-mapped-root entity relation")
    if (domain.source_id, domain.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        issues.append("a mapped root domain bound to the mapped source revision")
    if domain.reference_domain.source_id != source.source_id:
        issues.append(
            "a reference piecewise-linear domain bound to the mapped source identity"
        )
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not mesh.blocks
        or any(block.cell_kind != "hexahedron" for block in mesh.blocks)
    ):
        issues.append("an all-hexahedral three-dimensional mapped root mesh")
    else:
        elements, _, _ = domain.source_geometry.resolve(mesh)
        if any(
            not isinstance(element, ScalarCoordinateElement)
            or element.cell_kind != "hexahedron"
            or int(element.degree) != specification.target.geometry_order
            for element in elements
        ):
            issues.append(
                "root coordinate elements whose exact degree matches the target geometry order"
            )
    if (
        source.reference.complex.geometry is not None
        or source.reference.mapped_domain is not None
    ):
        issues.append("a reference PLC without a second coordinate map")
    if source.reference.complex.boundary != "conforming":
        issues.append("a conforming reference PLC boundary")
    if not _mapped_reference_boundary_matches(source):
        issues.append(
            "reference PLC facets matching the mapped root-domain boundary strata"
        )
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append(
            "a simplex fill declaration for the direct mapped tensor-grid route"
        )
    if specification.periodic_constraints:
        issues.append("no periodic quotient constraints")
    if specification.layer_controls:
        issues.append("no boundary-layer construction")
    if specification.hole_seeds:
        issues.append("no hole seeds outside the declared mapped root domain")
    issues.extend(_mapped_root_control_issues(source, specification))
    if any(
        seed.region_name not in domain.region_ids for seed in specification.region_seeds
    ):
        issues.append("region seeds naming regions declared by the mapped root domain")
    boundary = specification.boundary_scope
    reference_facets = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
    if (
        boundary.source_id != source.source_id
        or boundary.source_revision != source.source_revision
        or boundary.entity_kind is not MeshingEntityKind.GEOMETRY
        or boundary.entity_dimension != 2
        or boundary.entity_set_id != source.scope_entity_set_id(2)
        or not np.array_equal(
            np.sort(np.asarray(boundary.entity_ids, dtype=np.int64)),
            np.sort(reference_facets),
        )
    ):
        issues.append("a boundary scope drawn from the mapped source facet relation")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append(
                "uniform size controls addressable over mapped reference facets"
            )
            continue
        control_ids = np.asarray(control.scope.entity_ids, dtype=np.int64)
        if (
            control.scope.source_id != source.source_id
            or control.scope.source_revision != source.source_revision
            or control.scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or control.scope.entity_dimension != 2
            or control.scope.entity_set_id != source.scope_entity_set_id(2)
            or not np.array_equal(
                np.sort(control_ids),
                np.sort(np.asarray(boundary.entity_ids, dtype=np.int64)),
            )
        ):
            issues.append(
                f"a mapped-reference facet scope for size control {control.control_id}"
            )
    if not specification.size_controls:
        issues.append("a physical uniform size target to determine mapped grid spacing")
    return list(dict.fromkeys(issues))


def mapped_balanced_grid_hex_support_issues(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    return _mapped_grid_hex_support_issues(
        source, specification, options, "mapped_balanced_grid_hex", "balanced_grid"
    )


def mapped_frame_grid_hex_support_issues(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    return _mapped_grid_hex_support_issues(
        source, specification, options, "mapped_frame_grid_hex", "frame_grid"
    )


def _mapped_domain_face_data(
    source: NativeMappedHexSource, /
) -> tuple[
    tuple[int, ...],
    dict[int, tuple[str | None, str | None]],
    dict[
        int,
        tuple[tuple[str, ...] | None, tuple[str, ...] | None],
    ],
]:
    complex_ = source.reference.complex
    relation = source.root_relation
    source_facets = np.asarray(relation.polygon_source_facet_rows, dtype=np.int64)
    root_face_ids = np.asarray(relation.polygon_root_face_global_ids, dtype=np.int64)
    if source_facets.shape != root_face_ids.shape:
        raise ValueError("Mapped PLC/root facet relation is misaligned.")
    sides_by_face: dict[int, tuple[str | None, str | None]] = {}
    paths_by_face: dict[int, tuple[tuple[str, ...] | None, tuple[str, ...] | None]] = {}
    for facet_row, root_face_id in zip(
        source_facets.tolist(), root_face_ids.tolist(), strict=True
    ):
        incidence = np.asarray(complex_.facet_regions, dtype=np.int64)[facet_row][::-1]
        first_region = int(incidence[0])
        second_region = int(incidence[1])
        sides = (
            None if first_region < 0 else complex_.region_ids[first_region],
            None if second_region < 0 else complex_.region_ids[second_region],
        )
        occurrence_paths = (
            None if first_region < 0 else complex_.region_occurrence_paths[first_region],
            None
            if second_region < 0
            else complex_.region_occurrence_paths[second_region],
        )
        previous_sides = sides_by_face.get(int(root_face_id))
        previous_paths = paths_by_face.get(int(root_face_id))
        if previous_sides is not None and (
            previous_sides != sides or previous_paths != occurrence_paths
        ):
            raise ValueError(
                "Mapped source facets on one root face disagree on oriented region occurrences."
            )
        sides_by_face[int(root_face_id)] = sides
        paths_by_face[int(root_face_id)] = occurrence_paths
    return tuple(sorted(sides_by_face)), sides_by_face, paths_by_face


def _mapped_root_control_issues(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    complex_ = source.reference.complex
    relation = source.root_relation
    issues: list[str] = []
    region_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(complex_.region_entity_ids, dtype=np.int64).tolist()
        )
    }
    facet_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(complex_.facet_entity_ids, dtype=np.int64).tolist()
        )
    }
    segment_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(complex_.plc_edge_entity_ids, dtype=np.int64).tolist()
        )
    }
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(complex_.vertex_entity_ids, dtype=np.int64).tolist()
        )
    }
    valid_scope = lambda scope, dimension: (
        scope.source_id == source.source_id
        and scope.source_revision == source.source_revision
        and scope.entity_kind is MeshingEntityKind.GEOMETRY
        and scope.entity_dimension == dimension
        and scope.entity_set_id == source.scope_entity_set_id(dimension)
    )
    relation_facets = np.asarray(relation.polygon_source_facet_rows, dtype=np.int64)
    relation_faces = np.asarray(relation.polygon_root_face_global_ids, dtype=np.int64)
    facets_by_root_face: dict[int, set[int]] = {}
    for facet_row, root_face_id in zip(
        relation_facets.tolist(), relation_faces.tolist(), strict=True
    ):
        facets_by_root_face.setdefault(int(root_face_id), set()).add(int(facet_row))
    _, sides_by_root_face, _ = _mapped_domain_face_data(source)

    for control in specification.region_controls:
        if not valid_scope(control.scope, 3):
            issues.append(f"mapped source region scope for {control.control_id}")
            continue
        identifiers = tuple(
            int(value)
            for value in np.asarray(control.scope.entity_ids, dtype=np.int64).tolist()
        )
        names = {
            complex_.region_ids[region_rows[value]]
            for value in identifiers
            if value in region_rows
        }
        if (
            len(names) != 1
            or control.region_name not in names
            or control.region_name not in source.domain.region_ids
        ):
            issues.append(f"mapped source region identity for {control.control_id}")
        if not control.meshing_enabled:
            issues.append(
                f"disabled root region cannot be omitted by {control.control_id}"
            )

    for control in specification.patch_controls:
        if not valid_scope(control.scope, 2):
            issues.append(f"mapped source patch scope for {control.control_id}")
            continue
        identifiers = tuple(
            int(value)
            for value in np.asarray(control.scope.entity_ids, dtype=np.int64).tolist()
        )
        selected_facets = {
            facet_rows[value] for value in identifiers if value in facet_rows
        }
        if len(selected_facets) != len(identifiers):
            issues.append(
                f"mapped source patch names an absent facet for {control.control_id}"
            )
            continue
        selected_root_faces = {
            int(root_face)
            for facet, root_face in zip(
                relation_facets.tolist(), relation_faces.tolist(), strict=True
            )
            if int(facet) in selected_facets
        }
        if control.required and not selected_root_faces:
            issues.append(
                f"required mapped source patch is absent for {control.control_id}"
            )
            continue
        for root_face in selected_root_faces:
            if not facets_by_root_face[root_face].issubset(selected_facets):
                issues.append(
                    f"mapped patch {control.control_id} splits a root-face source chart"
                )
                break
            adjacent = tuple(
                sorted(
                    value for value in sides_by_root_face[root_face] if value is not None
                )
            )
            if adjacent != control.adjacent_region_names:
                issues.append(
                    f"mapped patch {control.control_id} changes oriented region incidence"
                )
                break

    feature_dimensions = {
        FeatureKind.CORNER: 0,
        FeatureKind.CURVE: 1,
        FeatureKind.SURFACE: 2,
        FeatureKind.MATERIAL_INTERFACE: 2,
    }
    for feature in specification.protected_features:
        dimension = feature_dimensions.get(feature.feature_kind)
        if dimension is None or not valid_scope(feature.scope, dimension):
            issues.append(
                f"mapped protected-feature source scope for {feature.feature_id}"
            )
            continue
        identifiers = tuple(
            int(value)
            for value in np.asarray(feature.scope.entity_ids, dtype=np.int64).tolist()
        )
        rows_by_dimension = {0: vertex_rows, 1: segment_rows, 2: facet_rows}
        row_map = rows_by_dimension[dimension]
        selected = {row_map[value] for value in identifiers if value in row_map}
        if len(selected) != len(identifiers):
            issues.append(
                f"mapped protected feature names an absent source entity for {feature.feature_id}"
            )
            continue
        if dimension == 1:
            relation_dimensions = np.asarray(
                relation.segment_root_dimensions, dtype=np.int8
            )
            if any(relation_dimensions[row] != 1 for row in selected):
                issues.append(
                    f"mapped curve feature {feature.feature_id} is not a root-edge stratum"
                )
        if dimension == 2:
            selected_root_faces = {
                int(root_face)
                for facet, root_face in zip(
                    relation_facets.tolist(), relation_faces.tolist(), strict=True
                )
                if int(facet) in selected
            }
            if any(
                not facets_by_root_face[root_face].issubset(selected)
                for root_face in selected_root_faces
            ):
                issues.append(
                    f"mapped surface feature {feature.feature_id} splits a root-face chart"
                )
            if feature.feature_kind is FeatureKind.MATERIAL_INTERFACE and any(
                len(
                    tuple(
                        value
                        for value in sides_by_root_face[root_face]
                        if value is not None
                    )
                )
                != 2
                or len(
                    set(
                        value
                        for value in sides_by_root_face[root_face]
                        if value is not None
                    )
                )
                != 2
                for root_face in selected_root_faces
            ):
                issues.append(
                    f"mapped material-interface feature {feature.feature_id} is not an interface"
                )
    return list(dict.fromkeys(issues))


def _validate_mapped_region_seeds(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    /,
) -> None:
    if not specification.region_seeds:
        return
    domain = source.domain
    source.validate_restored()
    if (domain.source_id, domain.source_revision) != (
        source.source_id,
        source.source_revision,
    ) or domain.reference_domain.source_id != source.source_id:
        raise ValueError(
            "Mapped region seeds require the exact source-bound reference domain."
        )
    blocks = domain.reference_mesh.blocks
    cell_count = sum(block.cell_count for block in blocks)
    root_cell_ids = domain.root_cell_ids
    occurrences = domain.root_cell_occurrence_paths
    cell_regions = np.asarray(domain.cell_regions, dtype=np.int64)
    if (
        len(root_cell_ids) != cell_count
        or len(occurrences) != cell_count
        or cell_regions.shape != (cell_count,)
    ):
        raise ValueError(
            "Mapped source root-cell identity, occurrence, and region banks are misaligned."
        )
    offsets = np.cumsum(
        np.asarray((0, *(block.cell_count for block in blocks)), dtype=np.int64)
    )
    for block_index, block in enumerate(blocks):
        start, stop = (int(offsets[block_index]), int(offsets[block_index + 1]))
        if not np.array_equal(
            np.asarray(root_cell_ids[start:stop], dtype=np.int64),
            np.asarray(block.global_ids, dtype=np.int64),
        ):
            raise ValueError(
                "Mapped seed locator cells do not match source root-cell identities."
            )
    locators = tuple(
        PreparedMappedCellLocator.from_cell_geometry(
            domain.reference_mesh,
            domain.source_geometry,
            block_index,
        )
        for block_index in range(len(blocks))
    )
    for seed in specification.region_seeds:
        expected_region = seed.region_name
        domain.region_ids.index(expected_region)
        located_cell_keys: set[tuple[str, str, int, tuple[str, ...], str]] = set()
        point = np.asarray(seed.point, dtype=np.float64)[None, :]
        for block_index, (block, locator) in enumerate(
            zip(blocks, locators, strict=True)
        ):
            result = locator.locate(point)
            status = int(np.asarray(result.status)[0])
            complete = bool(np.asarray(result.candidates_complete)[0])
            successful = bool(np.asarray(result.successful)[0])
            # A complete candidate search proves OUTSIDE; only LOCATED rows
            # additionally carry a successful inverse.
            if (
                status
                not in (
                    int(CellLocationStatus.OUTSIDE),
                    int(CellLocationStatus.LOCATED),
                )
                or not complete
                or (status == int(CellLocationStatus.LOCATED) and not successful)
            ):
                raise ValueError(
                    "Mapped region-seed admission requires a complete resolved source-map inverse."
                )
            if status == int(CellLocationStatus.OUTSIDE):
                continue
            count = int(np.asarray(result.candidate_count)[0])
            rows = np.asarray(result.candidate_cells, dtype=np.int64)[0, :count]
            if np.any(rows < 0) or np.any(rows >= block.cell_count):
                raise ValueError(
                    "Mapped region-seed candidates leave their source block."
                )
            for row in rows.tolist():
                global_row = int(offsets[block_index]) + int(row)
                region_index = int(cell_regions[global_row])
                if region_index < 0 or region_index >= len(domain.region_ids):
                    raise ValueError(
                        "Mapped region-seed candidate has no source region binding."
                    )
                located_cell_keys.add(
                    (
                        source.source_id,
                        source.source_revision,
                        int(block.global_ids[int(row)]),
                        occurrences[global_row],
                        domain.region_ids[region_index],
                    )
                )
        located_regions = {key[-1] for key in located_cell_keys}
        if located_regions != {expected_region}:
            raise ValueError(
                "Mapped region seed must locate wholly inside its declared source region."
            )


def _mapped_grid_organization(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    construction: MappedGridHexConstruction,
    /,
) -> NativeMeshOrganization:
    domain = source.domain
    mesh = construction.mesh
    _, root_face_sides, root_face_occurrence_paths = _mapped_domain_face_data(source)
    vertex_paths = dict(
        zip(domain.root_vertex_ids, domain.root_vertex_occurrence_paths, strict=True)
    )
    cell_paths = dict(
        zip(domain.root_cell_ids, domain.root_cell_occurrence_paths, strict=True)
    )
    source_roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    associations = []
    for dimension in range(4):
        entity_set = mesh.entity_set(dimension)
        parent_dimensions = np.asarray(
            construction.entity_parent_dimensions[dimension], dtype=np.int8
        )
        parent_ids = np.asarray(construction.entity_parent_ids[dimension], dtype=np.int64)
        source_ids = tuple(
            domain.image_entity_id(int(parent_dimension), int(parent_id))
            for parent_dimension, parent_id in zip(
                parent_dimensions, parent_ids, strict=True
            )
        )
        paths = tuple(
            vertex_paths[int(parent_id)]
            if int(parent_dimension) == 0
            else cell_paths[int(parent_id)]
            if int(parent_dimension) == 3
            else ()
            for parent_dimension, parent_id in zip(
                parent_dimensions, parent_ids, strict=True
            )
        )
        region_sides = tuple(
            root_face_sides.get(int(parent_id), (None, None))
            if int(parent_dimension) == 2
            else (None, None)
            for parent_dimension, parent_id in zip(
                parent_dimensions, parent_ids, strict=True
            )
        )
        associations.append(
            GeometryAssociation(
                GeometryAssociationKind.MAPPED_REFERENCE,
                source.source_id,
                source.source_revision,
                entity_set.entity_set_id,
                np.asarray(entity_set.entity_ids, dtype=np.int64),
                source_ids,
                np.zeros((entity_set.entity_ids.size,), dtype=np.float64),
                exact=True,
                source_dimensions=parent_dimensions,
                source_indices=parent_ids,
                source_roles=tuple(
                    source_roles[int(value)] for value in parent_dimensions.tolist()
                ),
                source_occurrence_paths=paths,
                source_region_sides=region_sides,
                parent_dimensions=parent_dimensions,
                parent_ids=parent_ids,
            )
        )
    target_cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    zone_bindings: dict[int, tuple[str, str, RegionRole]] = {}

    def bind_region(region_name: str, material_id: str, role: RegionRole, /) -> None:
        region_index = domain.region_ids.index(region_name)
        binding = (region_name, material_id, role)
        previous = zone_bindings.get(region_index)
        if previous is not None and previous != binding:
            raise ValueError(
                "Mapped region seeds and controls disagree on material or role."
            )
        zone_bindings[region_index] = binding

    for seed in specification.region_seeds:
        bind_region(seed.region_name, seed.material_id, seed.role)
    for control in specification.region_controls:
        if not control.meshing_enabled:
            raise ValueError("Mapped tensor grids cannot omit a declared root region.")
        bind_region(control.region_name, control.material_id, control.role)
    for region_index, region_name in enumerate(domain.region_ids):
        zone_bindings.setdefault(
            region_index, (region_name, region_name, RegionRole.USER)
        )
    zones = []
    for region_index, (region_name, material_id, role) in sorted(zone_bindings.items()):
        selected = target_cell_ids[construction.cell_regions == region_index]
        if selected.size:
            zones.append(
                MeshZone(
                    region_name,
                    MeshZoneRole.REGION,
                    _mesh_scope(mesh, 3, selected),
                    material_id=material_id,
                    region_role=role,
                )
            )
    zones_by_name = {zone.name: zone for zone in zones}

    patches = []
    complex_ = source.reference.complex
    facet_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(complex_.facet_entity_ids, dtype=np.int64).tolist()
        )
    }
    relation_facet_rows = np.asarray(
        source.root_relation.polygon_source_facet_rows, dtype=np.int64
    )
    relation_root_faces = np.asarray(
        source.root_relation.polygon_root_face_global_ids, dtype=np.int64
    )
    target_face_entity_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    target_face_parent_dimensions = np.asarray(
        construction.entity_parent_dimensions[2], dtype=np.int8
    )
    target_face_parent_ids = np.asarray(construction.entity_parent_ids[2], dtype=np.int64)
    from ._native_volume import _faces_connected

    for control in specification.patch_controls:
        selected_source_rows = {
            facet_rows[int(identifier)]
            for identifier in np.asarray(
                control.scope.entity_ids, dtype=np.int64
            ).tolist()
        }
        selected_root_faces = {
            int(root_face)
            for facet_row, root_face in zip(
                relation_facet_rows.tolist(), relation_root_faces.tolist(), strict=True
            )
            if int(facet_row) in selected_source_rows
        }
        target_mask = (target_face_parent_dimensions == 2) & np.isin(
            target_face_parent_ids, tuple(sorted(selected_root_faces))
        )
        target_rows = np.flatnonzero(target_mask)
        if not target_rows.size:
            if control.required:
                raise ValueError("Required mapped root patch has no target faces.")
            continue
        source_side_occurrences = {
            (
                root_face_sides[int(face_id)],
                root_face_occurrence_paths[int(face_id)],
            )
            for face_id in selected_root_faces
        }
        if len(source_side_occurrences) != 1:
            raise ValueError(
                "One mapped patch control spans differently oriented root-face region occurrences."
            )
        source_pair, source_paths = next(iter(source_side_occurrences))
        adjacent_zone_ids = tuple(
            zones_by_name[name].zone_id
            for name in control.adjacent_region_names
            if name in zones_by_name
        )
        target_faces = _entity_vertex_rows(mesh, 2)[target_rows]
        patches.append(
            MeshPatch(
                control.name,
                _mesh_scope(mesh, 2, target_face_entity_ids[target_rows]),
                connected=_faces_connected(target_faces),
                adjacent_zone_ids=adjacent_zone_ids,
                source_adjacent_region_ids=source_pair,
                source_adjacent_region_occurrence_paths=source_paths,
            )
        )
    return NativeMeshOrganization(
        patches=tuple(patches), zones=tuple(zones), associations=tuple(associations)
    )


def _mapped_grid_compliance(
    specification: VolumeMeshingSpec,
    construction: MappedGridHexConstruction,
    frame: PreparedSourceHexFrame,
    source: NativeMappedHexSource,
    /,
) -> MeshingComplianceReport:
    mesh = construction.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = _entity_vertex_rows(mesh, 1)
    edge_lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    root_face_ids, _, _ = _mapped_domain_face_data(source)
    face_dimensions = np.asarray(construction.entity_parent_dimensions[2], dtype=np.int8)
    face_ids = np.asarray(construction.entity_parent_ids[2], dtype=np.int64)
    faces = _entity_vertex_rows(mesh, 2)
    edge_rows = {
        tuple(sorted(int(value) for value in edge)): row for row, edge in enumerate(edges)
    }
    selected_edge_rows: set[int] = set()
    root_face_set = set(root_face_ids)
    for row, face in enumerate(faces):
        if int(face_dimensions[row]) != 2 or int(face_ids[row]) not in root_face_set:
            continue
        for first in range(face.size):
            for second in range(first + 1, face.size):
                edge_row = edge_rows.get(
                    tuple(sorted((int(face[first]), int(face[second]))))
                )
                if edge_row is not None:
                    selected_edge_rows.add(edge_row)
    selected_lengths = edge_lengths[
        np.asarray(sorted(selected_edge_rows), dtype=np.int64)
    ]
    if not selected_lengths.size:
        raise ValueError("Mapped reference boundary has no realized target edges.")
    requested = []
    achieved = []
    issues = []
    policy = specification.size_compliance
    for control in _uniform_size_controls(specification):
        key = f"size:{control.control_id}"
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        statistics = {
            name: float(np.quantile(selected_lengths, 0.5 if name == "p50" else 0.95))
            for name in policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(selected_lengths))),
                (f"{key}:maximum_edge", float(np.max(selected_lengths))),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if (
                control.minimum_size is not None
                and float(np.min(selected_lengths)) < control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            if (
                control.maximum_size is not None
                and float(np.max(selected_lengths)) > control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
    achieved.append(("mapped_source:fidelity_bound", 0.0))
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.append((f"protected:{feature.feature_id}:source_fidelity_bound", 0.0))
    if specification.maximum_deviation is not None:
        requested.append(("source:maximum_deviation", specification.maximum_deviation))
    achieved.append(
        ("mapped_source:lipschitz_bound", mapped_source_lipschitz_bound(frame))
    )
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _mapped_face_chart_witness(
    source: NativeMappedHexSource,
    construction: MappedGridHexConstruction,
    geometry: CellGeometrySpec,
    query: MappedReferenceBoundarySource,
    /,
) -> SourceBoundaryChartWitness:
    """Partition each target hex face by its exact source PLC triangle charts."""
    mesh = construction.mesh
    relation = source.root_relation
    controls_by_cell = _mapped_reference_cell_charts(source, construction)
    source_keys = tuple(
        zip(
            relation.polygon_source_entity_keys,
            relation.polygon_source_occurrence_paths,
            strict=True,
        )
    )
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Mapped source PLC chart identities must be unique.")
    charts_by_root_face: dict[int, list[tuple[str, int]]] = {}
    selected_face_keys = set(query.face_scope_keys)
    for source_row, chart_id in enumerate(query.face_chart_ids):
        key = source_keys[source_row]
        if key not in selected_face_keys:
            continue
        if query.chart_source_key(2, chart_id) != key:
            raise ValueError("Mapped source query chart lost its authored PLC facet key.")
        root_face_id = int(relation.polygon_root_face_global_ids[source_row])
        charts_by_root_face.setdefault(root_face_id, []).append((chart_id, source_row))

    def cross2(
        first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
    ) -> Fraction:
        return first[0] * second[1] - first[1] * second[0]

    def signed_double_area(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
        return sum(
            (
                cross2(points[row], points[(row + 1) % len(points)])
                for row in range(len(points))
            ),
            Fraction(0),
        )

    def clip_triangle(
        subject: tuple[tuple[Fraction, Fraction], ...],
        triangle: tuple[tuple[Fraction, Fraction], ...],
        /,
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        clip = triangle if signed_double_area(triangle) > 0 else tuple(reversed(triangle))
        polygon = list(subject)
        for edge_row in range(3):
            edge_start = clip[edge_row]
            edge_end = clip[(edge_row + 1) % 3]
            direction = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
            result: list[tuple[Fraction, Fraction]] = []
            if not polygon:
                break
            start = polygon[-1]
            start_gap = cross2(
                direction,
                (start[0] - edge_start[0], start[1] - edge_start[1]),
            )
            for end in polygon:
                end_gap = cross2(
                    direction,
                    (end[0] - edge_start[0], end[1] - edge_start[1]),
                )
                if (start_gap >= 0) != (end_gap >= 0):
                    fraction = start_gap / (start_gap - end_gap)
                    intersection = (
                        start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]),
                    )
                    if not result or result[-1] != intersection:
                        result.append(intersection)
                if end_gap >= 0 and (not result or result[-1] != end):
                    result.append(end)
                start, start_gap = end, end_gap
            if len(result) > 1 and result[0] == result[-1]:
                result.pop()
            polygon = result
        changed = True
        while changed and len(polygon) >= 3:
            changed = False
            for row in range(len(polygon)):
                previous = polygon[row - 1]
                current = polygon[row]
                following = polygon[(row + 1) % len(polygon)]
                if (
                    cross2(
                        (current[0] - previous[0], current[1] - previous[1]),
                        (following[0] - current[0], following[1] - current[1]),
                    )
                    == 0
                ):
                    polygon.pop(row)
                    changed = True
                    break
        return tuple(polygon)

    def local_face_coordinates(
        point: tuple[Fraction, Fraction],
        quad: tuple[tuple[Fraction, Fraction], ...],
        /,
    ) -> tuple[Fraction, Fraction]:
        first = (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
        second = (quad[3][0] - quad[0][0], quad[3][1] - quad[0][1])
        delta = (point[0] - quad[0][0], point[1] - quad[0][1])
        determinant = cross2(first, second)
        if determinant == 0:
            raise ValueError("Mapped target root-face chart is singular.")
        return (
            cross2(delta, second) / determinant,
            cross2(first, delta) / determinant,
        )

    def source_parameters(
        point: tuple[Fraction, Fraction],
        triangle: tuple[tuple[Fraction, Fraction], ...],
        /,
    ) -> tuple[Fraction, Fraction]:
        first = (triangle[1][0] - triangle[0][0], triangle[1][1] - triangle[0][1])
        second = (triangle[2][0] - triangle[0][0], triangle[2][1] - triangle[0][1])
        delta = (point[0] - triangle[0][0], point[1] - triangle[0][1])
        determinant = cross2(first, second)
        if determinant == 0:
            raise ValueError("Mapped source root-face chart is singular.")
        return (
            cross2(delta, second) / determinant,
            cross2(first, delta) / determinant,
        )

    target_face_rows = {
        frozenset(int(value) for value in face): row
        for row, face in enumerate(_entity_vertex_rows(mesh, 2))
    }
    target_face_entity_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    target_parent_dimensions = np.asarray(
        construction.entity_parent_dimensions[2], dtype=np.int8
    )
    target_parent_ids = np.asarray(construction.entity_parent_ids[2], dtype=np.int64)
    target_topology = reference_cell_topology("hexahedron")
    target_local_faces = tuple(
        tuple(int(value) for value in face) for face in target_topology.entities[2]
    )
    target_reference_vertices = tuple(
        tuple(Fraction(value) for value in row) for row in target_topology.vertices
    )
    face_owners: dict[int, list[tuple[int, int]]] = {}
    for block in mesh.blocks:
        if block.cell_kind != "hexahedron":
            raise ValueError("Mapped boundary charts require hexahedral target cells.")
        block_cells = np.asarray(block.vertices, dtype=np.int64)
        block_ids = np.asarray(block.global_ids, dtype=np.int64)
        for cell_row, cell_id in enumerate(block_ids.tolist()):
            cell_vertices = block_cells[cell_row]
            for local_face, local_vertices in enumerate(target_local_faces):
                vertex_rows = tuple(int(cell_vertices[index]) for index in local_vertices)
                face_row = target_face_rows.get(frozenset(vertex_rows))
                if face_row is None or int(target_parent_dimensions[face_row]) != 2:
                    continue
                face_owners.setdefault(face_row, []).append((int(cell_id), local_face))

    root_face_ids = np.asarray(relation.polygon_root_face_global_ids, dtype=np.int64)
    root_cells = np.asarray(relation.polygon_root_cell_global_ids, dtype=np.int64)
    root_local_faces = np.asarray(
        relation.polygon_root_local_face_indices, dtype=np.int32
    )
    rows_by_root_face: dict[int, list[int]] = {}
    for row, root_face_id in enumerate(root_face_ids.tolist()):
        rows_by_root_face.setdefault(int(root_face_id), []).append(row)
    chart_cell_ids: list[int] = []
    chart_face_ids: list[int] = []
    chart_local_faces: list[int] = []
    chart_target_vertices: list[tuple[tuple[float, ...], ...]] = []
    chart_target_rational_vertices: list[tuple[tuple[Fraction, ...], ...]] = []
    chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    source_parameter_rows: list[tuple[tuple[float, float], ...]] = []
    source_parameter_rational_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    orientations: list[int] = []
    for target_face_row, raw_root_face_id in enumerate(target_parent_ids.tolist()):
        if int(target_parent_dimensions[target_face_row]) != 2:
            continue
        root_face_id = int(raw_root_face_id)
        face_chart_entries = charts_by_root_face.get(root_face_id)
        if not face_chart_entries:
            continue
        face_chart_rows = [row for _, row in face_chart_entries]
        root_owner = min(
            int(cell_id)
            for relation_row in face_chart_rows
            for cell_id in root_cells[relation_row].tolist()
            if int(cell_id) >= 0
        )
        target_owner = next(
            (
                (cell_id, local_face)
                for cell_id, local_face in face_owners.get(target_face_row, ())
                if controls_by_cell[cell_id].source_cell_id == root_owner
            ),
            None,
        )
        if target_owner is None:
            raise ValueError(
                "Mapped target root face lacks its declared root-cell owner."
            )
        target_cell_id, target_local_face = target_owner
        target_control = controls_by_cell[target_cell_id]
        target_local_vertices = target_local_faces[target_local_face]
        target_root_corners = tuple(
            _mapped_root_reference_corner(
                target_control,
                index,
                target_reference_vertices[index],
            )
            for index in target_local_vertices
        )
        root_local_face = next(
            int(root_local_faces[face_chart_rows[0], side])
            for side in range(2)
            if int(root_cells[face_chart_rows[0], side]) == root_owner
        )
        target_uv = query.root_face_uv(
            root_face_id,
            root_owner,
            root_local_face,
            target_root_corners,
        )
        target_area = abs(signed_double_area(target_uv))
        covered_area = Fraction(0)
        for chart_id, source_row in face_chart_entries:
            source_key = query.chart_source_key(2, chart_id)
            source_uv = query.chart_face_uv(chart_id)
            clipped = clip_triangle(target_uv, source_uv)
            if len(clipped) < 3:
                continue
            area = abs(signed_double_area(clipped))
            if area == 0:
                continue
            covered_area += area
            relation_side = next(
                side
                for side, cell_id in enumerate(root_cells[source_row].tolist())
                if int(cell_id) == root_owner
            )
            source_local_face = int(root_local_faces[source_row, relation_side])
            if source_local_face < 0:
                raise ValueError("Mapped source chart lost its local root-face index.")
            key, occurrence = source_key
            for triangle_row in range(1, len(clipped) - 1):
                uv_triangle = (
                    clipped[0],
                    clipped[triangle_row],
                    clipped[triangle_row + 1],
                )
                target_local_points = tuple(
                    local_face_coordinates(point, target_uv) for point in uv_triangle
                )
                target_reference_rational_points = tuple(
                    tuple(
                        target_reference_vertices[target_local_vertices[0]][axis]
                        + target_local_points[point_row][0]
                        * (
                            target_reference_vertices[target_local_vertices[1]][axis]
                            - target_reference_vertices[target_local_vertices[0]][axis]
                        )
                        + target_local_points[point_row][1]
                        * (
                            target_reference_vertices[target_local_vertices[3]][axis]
                            - target_reference_vertices[target_local_vertices[0]][axis]
                        )
                        for axis in range(3)
                    )
                    for point_row in range(3)
                )
                target_reference_points = tuple(
                    tuple(float(value) for value in point)
                    for point in target_reference_rational_points
                )
                source_uv_parameters = tuple(
                    source_parameters(point, source_uv) for point in uv_triangle
                )
                target_orientation = cross2(
                    (
                        target_local_points[1][0] - target_local_points[0][0],
                        target_local_points[1][1] - target_local_points[0][1],
                    ),
                    (
                        target_local_points[2][0] - target_local_points[0][0],
                        target_local_points[2][1] - target_local_points[0][1],
                    ),
                )
                source_orientation = cross2(
                    (
                        source_uv_parameters[1][0] - source_uv_parameters[0][0],
                        source_uv_parameters[1][1] - source_uv_parameters[0][1],
                    ),
                    (
                        source_uv_parameters[2][0] - source_uv_parameters[0][0],
                        source_uv_parameters[2][1] - source_uv_parameters[0][1],
                    ),
                )
                if target_orientation == 0 or source_orientation == 0:
                    raise ValueError(
                        "Mapped source/target chart intersection is degenerate."
                    )
                chart_cell_ids.append(target_cell_id)
                chart_face_ids.append(int(target_face_entity_ids[target_face_row]))
                chart_local_faces.append(target_local_face)
                chart_target_vertices.append(target_reference_points)
                chart_target_rational_vertices.append(target_reference_rational_points)
                chart_ids.append(chart_id)
                source_entity_ids.append(key)
                occurrence_paths.append(occurrence)
                source_parameter_rows.append(
                    tuple(
                        (float(value[0]), float(value[1]))
                        for value in source_uv_parameters
                    )
                )
                source_parameter_rational_rows.append(source_uv_parameters)
                orientations.append(
                    1 if source_orientation * target_orientation > 0 else -1
                )
        if covered_area != target_area:
            raise ValueError(
                "Mapped source charts do not partition the target root-face chart."
            )
    if not chart_cell_ids:
        raise ValueError(
            "Mapped source query produced no target boundary chart witnesses."
        )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(chart_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(chart_face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(chart_local_faces, dtype=np.int32),
        target_reference_vertices=np.asarray(chart_target_vertices, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            chart_target_rational_vertices
        ),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_parameter_rows, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _mapped_edge_chart_witness(
    source: NativeMappedHexSource,
    construction: MappedGridHexConstruction,
    geometry: CellGeometrySpec,
    query: MappedReferenceBoundarySource,
    /,
) -> SourceBoundaryChartWitness:
    """Bind mapped target root-edge pieces to their authored PLC segments."""
    mesh = construction.mesh
    relation = source.root_relation
    controls_by_cell = _mapped_reference_cell_charts(source, construction)
    relation_keys = tuple(
        zip(
            relation.segment_source_entity_keys,
            relation.segment_source_occurrence_paths,
            strict=True,
        )
    )
    if len(set(relation_keys)) != len(relation_keys):
        raise ValueError("Mapped source segment identities must be unique.")
    charts_by_root_edge: dict[int, list[tuple[str, int]]] = {}
    selected_edge_keys = set(query.edge_scope_keys)
    for relation_row, chart_id in enumerate(query.edge_chart_ids):
        key = relation_keys[relation_row]
        if key not in selected_edge_keys:
            continue
        if query.chart_source_key(1, chart_id) != key:
            raise ValueError("Mapped edge query chart lost its authored PLC segment key.")
        if int(relation.segment_root_dimensions[relation_row]) != 1:
            raise ValueError("Mapped curve chart is not an original root-edge stratum.")
        root_edge_id = int(relation.segment_root_entity_ids[relation_row])
        charts_by_root_edge.setdefault(root_edge_id, []).append((chart_id, relation_row))

    target_edge_rows = {
        frozenset(int(value) for value in edge): row
        for row, edge in enumerate(_entity_vertex_rows(mesh, 1))
    }
    target_edge_entity_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    target_parent_dimensions = np.asarray(
        construction.entity_parent_dimensions[1], dtype=np.int8
    )
    target_parent_ids = np.asarray(construction.entity_parent_ids[1], dtype=np.int64)
    target_topology = reference_cell_topology("hexahedron")
    target_local_edges = tuple(
        tuple(int(value) for value in edge) for edge in target_topology.entities[1]
    )
    target_reference_vertices = tuple(
        tuple(Fraction(value) for value in point) for point in target_topology.vertices
    )
    edge_owners: dict[int, list[tuple[int, int]]] = {}
    for block in mesh.blocks:
        if block.cell_kind != "hexahedron":
            raise ValueError("Mapped curve charts require hexahedral target cells.")
        block_cells = np.asarray(block.vertices, dtype=np.int64)
        block_ids = np.asarray(block.global_ids, dtype=np.int64)
        for cell_row, cell_id in enumerate(block_ids.tolist()):
            cell_vertices = block_cells[cell_row]
            for local_edge, local_vertices in enumerate(target_local_edges):
                vertex_rows = tuple(int(cell_vertices[index]) for index in local_vertices)
                edge_row = target_edge_rows.get(frozenset(vertex_rows))
                if edge_row is None or int(target_parent_dimensions[edge_row]) != 1:
                    continue
                edge_owners.setdefault(edge_row, []).append((int(cell_id), local_edge))

    chart_cell_ids: list[int] = []
    chart_edge_ids: list[int] = []
    chart_local_edges: list[int] = []
    chart_target_vertices: list[tuple[tuple[float, float, float], ...]] = []
    chart_target_rational_vertices: list[tuple[tuple[Fraction, ...], ...]] = []
    chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    source_parameter_rows: list[tuple[tuple[float], ...]] = []
    source_parameter_rational_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    orientations: list[int] = []
    intervals_by_chart: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for target_edge_row, raw_root_edge_id in enumerate(target_parent_ids.tolist()):
        if int(target_parent_dimensions[target_edge_row]) != 1:
            continue
        root_edge_id = int(raw_root_edge_id)
        chart_entries = charts_by_root_edge.get(root_edge_id)
        if not chart_entries:
            continue
        relation_row = chart_entries[0][1]
        root_cell_id = int(relation.segment_root_cell_global_ids[relation_row])
        target_owner = next(
            (
                (cell_id, local_edge)
                for cell_id, local_edge in edge_owners.get(target_edge_row, ())
                if controls_by_cell[cell_id].source_cell_id == root_cell_id
            ),
            None,
        )
        if target_owner is None:
            raise ValueError(
                "Mapped target root edge lacks its declared root-cell owner."
            )
        target_cell_id, target_local_edge = target_owner
        target_control = controls_by_cell[target_cell_id]
        target_local_vertices = target_local_edges[target_local_edge]
        target_root_points = tuple(
            _mapped_root_reference_corner(
                target_control,
                index,
                target_reference_vertices[index],
            )
            for index in target_local_vertices
        )
        for chart_id, source_row in chart_entries:
            source_local_edge = int(
                relation.segment_root_local_entity_indices[source_row]
            )
            root_edge_orientation = int(relation.segment_orientation_signs[source_row])
            root_topology = reference_cell_topology("hexahedron")
            if not 0 <= source_local_edge < len(root_topology.entities[1]):
                raise ValueError(
                    "Mapped source segment has an invalid root-local edge index."
                )
            root_reference_vertices = tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in root_topology.vertices
            )
            root_edge_vertices = tuple(
                int(value) for value in root_topology.entities[1][source_local_edge]
            )
            source_root_points = tuple(
                root_reference_vertices[index] for index in root_edge_vertices
            )
            if root_edge_orientation == -1:
                source_root_points = tuple(reversed(source_root_points))
            elif root_edge_orientation != 1:
                raise ValueError(
                    "Mapped source segment lacks an exact root-edge orientation."
                )
            declared_root_points = np.asarray(
                relation.segment_root_reference_vertices[source_row], dtype=np.float64
            )
            if not np.array_equal(
                declared_root_points,
                np.asarray(
                    [[float(value) for value in point] for point in source_root_points],
                    dtype=np.float64,
                ),
            ):
                raise ValueError(
                    "Mapped source segment changed its exact root-edge endpoints."
                )
            delta = tuple(
                source_root_points[1][axis] - source_root_points[0][axis]
                for axis in range(3)
            )
            varying = next(axis for axis, value in enumerate(delta) if value != 0)
            parameters = []
            for point in target_root_points:
                parameter = (point[varying] - source_root_points[0][varying]) / delta[
                    varying
                ]
                if any(
                    source_root_points[0][axis] + parameter * delta[axis] != point[axis]
                    for axis in range(3)
                ) or not Fraction(0) <= parameter <= Fraction(1):
                    raise ValueError(
                        "Mapped target edge leaves its authored source segment chart."
                    )
                parameters.append(parameter)
            chart_cell_ids.append(target_cell_id)
            chart_edge_ids.append(int(target_edge_entity_ids[target_edge_row]))
            chart_local_edges.append(target_local_edge)
            target_reference_rational_points = tuple(
                target_reference_vertices[index] for index in target_local_vertices
            )
            chart_target_rational_vertices.append(target_reference_rational_points)
            chart_target_vertices.append(
                tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in target_reference_rational_points
                )
            )
            chart_ids.append(chart_id)
            key, occurrence = query.chart_source_key(1, chart_id)
            source_entity_ids.append(key)
            occurrence_paths.append(occurrence)
            source_parameter_rows.append(
                ((float(parameters[0]),), (float(parameters[1]),))
            )
            source_parameter_rational_rows.append(((parameters[0],), (parameters[1],)))
            orientations.append(1 if parameters[1] > parameters[0] else -1)
            intervals_by_chart.setdefault(chart_id, []).append(
                (min(parameters), max(parameters))
            )
    for chart_id, intervals in intervals_by_chart.items():
        ordered = sorted(intervals)
        if (
            not ordered
            or ordered[0][0] != 0
            or ordered[-1][1] != 1
            or any(
                left[1] != right[0]
                for left, right in zip(ordered, ordered[1:], strict=True)
            )
        ):
            raise ValueError(
                f"Mapped output edges do not exhaust source curve chart {chart_id}."
            )
    if not chart_cell_ids:
        raise ValueError("Mapped source query produced no target curve chart witnesses.")
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=1,
        reference_dimension=3,
        cell_global_ids=np.asarray(chart_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(chart_edge_ids, dtype=np.int64),
        local_entity_indices=np.asarray(chart_local_edges, dtype=np.int32),
        target_reference_vertices=np.asarray(chart_target_vertices, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            chart_target_rational_vertices
        ),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_parameter_rows, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _mapped_point_feature_witness(
    source: NativeMappedHexSource,
    construction: MappedGridHexConstruction,
    geometry: CellGeometrySpec,
    query: MappedReferenceBoundarySource,
    feature: ProtectedFeature,
    /,
) -> SourceBoundaryPointWitness:
    mesh = construction.mesh
    complex_ = source.reference.complex
    source_vertex_ids = np.asarray(complex_.vertex_entity_ids, dtype=np.int64)
    source_rows = {
        int(identifier): row for row, identifier in enumerate(source_vertex_ids.tolist())
    }
    target_entity_ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    target_parent_dimensions = np.asarray(
        construction.entity_parent_dimensions[0], dtype=np.int8
    )
    target_parent_ids = np.asarray(construction.entity_parent_ids[0], dtype=np.int64)
    root_vertex_ids = np.asarray(
        source.root_relation.vertex_root_global_ids, dtype=np.int64
    )
    feature_vertex_ids = np.asarray(feature.scope.entity_ids, dtype=np.int64)
    target_ids: list[int] = []
    source_keys: list[str] = []
    occurrences: list[tuple[str, ...]] = []
    for source_vertex_id in feature_vertex_ids.tolist():
        source_row = source_rows.get(int(source_vertex_id))
        if source_row is None:
            raise ValueError("Mapped corner feature names an absent authored PLC vertex.")
        root_vertex_id = int(root_vertex_ids[source_row])
        target_rows = np.flatnonzero(
            (target_parent_dimensions == 0) & (target_parent_ids == root_vertex_id)
        )
        if target_rows.size != 1:
            raise ValueError("Mapped corner feature lacks one exact target root vertex.")
        source_key = source.source_entity_key_for_scope(0, int(source_vertex_id))
        target_ids.append(int(target_entity_ids[int(target_rows[0])]))
        source_keys.append(source_key[0])
        occurrences.append(source_key[1])
    return SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_query_id,
        vertex_global_ids=np.asarray(target_ids, dtype=np.int64),
        source_entity_ids=tuple(source_keys),
        occurrence_paths=tuple(occurrences),
    )


@final
class _PreparedNativeMappedGridHex(StrictModule, NonTrainableState):
    source: NativeMappedHexSource
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    frame: PreparedSourceHexFrame
    connection: SourceBlockConnection

    def __init__(
        self,
        source: NativeMappedHexSource,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        frame: PreparedSourceHexFrame,
        connection: SourceBlockConnection,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.frame = frame
        self.connection = connection

    def execute(self, /) -> NativeMeshConstruction:
        schedule = self.options.grid_schedule
        if schedule is None:
            raise RuntimeError(
                "Prepared mapped hex route lost its integer grid schedule."
            )
        _validate_mapped_region_seeds(self.source, self.specification)
        controls = _uniform_size_controls(self.specification)
        target_size = min(float(control.target_size) for control in controls)
        minimum_size = max(
            (
                float(control.minimum_size)
                for control in controls
                if control.minimum_size is not None
            ),
            default=0.0,
        )
        maximum_values = tuple(
            float(control.maximum_size)
            for control in controls
            if control.maximum_size is not None
        )
        maximum_size = min(maximum_values) if maximum_values else None
        target_size = max(target_size, minimum_size)
        if maximum_size is not None:
            if target_size > maximum_size:
                raise ValueError(
                    "Mapped uniform size controls have incompatible minimum and maximum sizes."
                )
            target_size = min(target_size, maximum_size)
        grid = generate_integer_grid_hexes(
            self.source.domain,
            self.frame,
            self.connection,
            schedule,
            target_size,
            maximum_size=maximum_size,
        )
        mapped = realize_mapped_grid_hexes(
            grid,
            self.source.domain,
            self.frame,
            schedule,
        )
        if mapped.source_connection_id != self.connection.connection_id:
            raise ValueError(
                "Mapped root-grid realization lost its exact source-block connection."
            )
        organization = _mapped_grid_organization(
            self.source,
            self.specification,
            mapped,
        )
        compliance = _mapped_grid_compliance(
            self.specification,
            mapped,
            self.frame,
            self.source,
        )
        fidelity_source = self.source.boundary_query()
        fidelity_witness = _mapped_face_chart_witness(
            self.source,
            mapped,
            mapped.geometry,
            fidelity_source,
        )
        scoped_fidelity = []
        for feature in self.specification.protected_features:
            if feature.feature_kind in (
                FeatureKind.SURFACE,
                FeatureKind.MATERIAL_INTERFACE,
            ):
                scope_keys = tuple(
                    self.source.source_entity_key_for_scope(2, int(identifier))
                    for identifier in np.asarray(
                        feature.scope.entity_ids, dtype=np.int64
                    ).tolist()
                )
                feature_query = self.source.boundary_query(face_scope_keys=scope_keys)
                feature_witness = _mapped_face_chart_witness(
                    self.source, mapped, mapped.geometry, feature_query
                )
            elif feature.feature_kind is FeatureKind.CURVE:
                scope_keys = tuple(
                    self.source.source_entity_key_for_scope(1, int(identifier))
                    for identifier in np.asarray(
                        feature.scope.entity_ids, dtype=np.int64
                    ).tolist()
                )
                feature_query = self.source.boundary_query(edge_scope_keys=scope_keys)
                feature_witness = _mapped_edge_chart_witness(
                    self.source, mapped, mapped.geometry, feature_query
                )
            elif feature.feature_kind is FeatureKind.CORNER:
                feature_query = fidelity_source
                feature_witness = _mapped_point_feature_witness(
                    self.source, mapped, mapped.geometry, feature_query, feature
                )
            else:
                raise ValueError(
                    f"Mapped tensor-grid feature {feature.feature_id} has no exact chart witness route."
                )
            if feature.hard:
                scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
        # The grid geometry is the exact per-cell restriction of the root source
        # map, so its source deviation is zero. Without a requested bound the
        # certificate proves that exact claim instead of an invented allowance.
        requested = self.specification.maximum_deviation
        fidelity_tolerance = 0.0 if requested is None else requested
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            self.source.source_id,
            self.source.source_revision,
            domain=self.source.domain,
            cell_regions=np.asarray(mapped.cell_regions, dtype=np.int64),
            fidelity_source=fidelity_source,
            fidelity_tolerance=fidelity_tolerance,
            fidelity_witness=fidelity_witness,
            protected_features=self.specification.protected_features,
            scoped_fidelity=tuple(scoped_fidelity),
            feature_source=self.source if self.specification.protected_features else None,
        )
        return NativeMeshConstruction(
            mapped.mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=mapped.geometry,
            organization=organization,
            reference_maps=mapped.reference_maps,
            source_mesh=self.source.domain.reference_mesh,
            source_geometry=self.source.domain.source_geometry,
        )


def _prepare_native_mapped_grid_hex(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeMappedGridHex:
    if source.domain.coordinate_contract.spatial_id != coordinate_contract.spatial_id:
        raise ValueError(
            "The coordinate contract must bind the mapped source's physical frame."
        )
    frame = prepare_source_hex_frame(
        source.domain.reference_mesh,
        source.domain.source_geometry,
    )
    connection = prepare_source_block_connection(frame)
    return _PreparedNativeMappedGridHex(
        source,
        specification,
        options,
        coordinate_contract,
        frame,
        connection,
    )


@checked
def prepare_native_mapped_balanced_grid_hex(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeMappedGridHex:
    issues = mapped_balanced_grid_hex_support_issues(source, specification, options)
    if issues:
        raise ValueError(
            "Unsupported native mapped balanced-grid hex request: " + "; ".join(issues)
        )
    return _prepare_native_mapped_grid_hex(
        source, specification, options, coordinate_contract
    )


@checked
def prepare_native_mapped_frame_grid_hex(
    source: NativeMappedHexSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeMappedGridHex:
    issues = mapped_frame_grid_hex_support_issues(source, specification, options)
    if issues:
        raise ValueError(
            "Unsupported native mapped frame-grid hex request: " + "; ".join(issues)
        )
    return _prepare_native_mapped_grid_hex(
        source, specification, options, coordinate_contract
    )


__all__ = [
    "balanced_hex_support_issues",
    "dual_hex_support_issues",
    "frame_hex_support_issues",
    "hex_dominant_support_issues",
    "mapped_balanced_grid_hex_support_issues",
    "mapped_frame_grid_hex_support_issues",
    "prepare_native_balanced_grid_hex",
    "prepare_native_dual_hex",
    "prepare_native_dual_quad",
    "prepare_native_frame_grid_hex",
    "prepare_native_hex_dominant",
    "prepare_native_mapped_balanced_grid_hex",
    "prepare_native_mapped_frame_grid_hex",
    "quad_support_issues",
]
