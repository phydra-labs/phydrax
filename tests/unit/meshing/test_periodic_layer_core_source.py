from __future__ import annotations

from copy import copy
from typing import Any, TypedDict

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._layer_core_periodic import prepare_periodic_layer_core_source
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_sources import NativeLayerCoreSource


class _LayerArguments(TypedDict):
    vertex_layer_ids: np.ndarray
    cap_polygon_ids: np.ndarray
    layer_regions: np.ndarray
    region_entity_ids: np.ndarray
    region_occurrence_paths: tuple[tuple[str, ...], ...]
    region_ids: tuple[str, ...]
    core_region_map: np.ndarray
    core_facet_source_ids: tuple[str | None, ...]
    core_facet_source_occurrence_paths: tuple[tuple[str, ...] | None, ...]
    core_vertex_representatives: np.ndarray
    core_vertex_shifts: np.ndarray
    core_seam_polygon_pairs: np.ndarray
    layer_control: phx.meshing.BoundaryLayerControl
    layer_policy: phx.meshing.BoundaryLayerPolicy


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="periodic wall layers require meshcore"
)


def _periodic_wall_source() -> tuple[
    CellMesh, MeshingDomain, SurfaceSourceCharts, MeshingDomainBoundarySource
]:
    source_id = "periodic-layer-wall"
    revision = "revision-1"
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
    block = CellBlock(
        "periodic-wall-triangles",
        "triangle",
        triangles,
        global_ids=np.asarray((10, 11), dtype=np.int64),
    )
    vertex_ids = np.asarray((20, 21, 22, 23), dtype=np.int64)
    wall = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)

    patch = PlanePatch(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
    )
    pcurves = (
        LineCurve(np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0))),
        LineCurve(np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0))),
        LineCurve(np.asarray((1.0, 1.0)), np.asarray((-1.0, 0.0))),
        LineCurve(np.asarray((0.0, 1.0)), np.asarray((0.0, -1.0))),
    )
    patch_uses = tuple(
        PatchCurveUse(index, curve, 0.0, 1.0) for index, curve in enumerate(pcurves)
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (patch_uses,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id=source_id,
        source_revision=revision,
        corner_points=coordinates[[0, 1, 3, 2]],
    )
    boundary_source = MeshingDomainBoundarySource(domain, (0,))
    cell_ids = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    face_association = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        source_id,
        revision,
        wall.entity_set(2).entity_set_id,
        cell_ids,
        (domain.entity_id(2, 0),) * cell_ids.size,
        np.zeros((cell_ids.size,), dtype=np.float64),
        exact=True,
        source_dimensions=np.full((cell_ids.size,), 2, dtype=np.int8),
        source_indices=np.zeros((cell_ids.size,), dtype=np.int32),
        source_roles=(GeometrySourceEntityRole.FACET,) * cell_ids.size,
        source_occurrence_paths=(domain.source_occurrences[2][0],) * cell_ids.size,
    )
    source_root_id = int(domain.scope_entity_ids(2)[0])
    charts = SurfaceSourceCharts(
        domain,
        wall,
        CellGeometrySpec.affine(wall),
        phx.SpatialCoordinateContract.si(),
        np.asarray(
            (
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
                ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ),
            dtype=np.float64,
        ),
        np.zeros((cell_ids.size,), dtype=np.int64),
        (face_association,),
        boundary_source,
        source_root_cell_ids=np.full((cell_ids.size,), source_root_id, dtype=np.int64),
        source_piece_ids=((0, int(domain.source_indices[2][0])),) * cell_ids.size,
    )
    return wall, domain, charts, boundary_source


def _bind_periodic_layer_identity(
    layers: phx.meshing.BoundaryLayerMesh,
) -> phx.meshing.BoundaryLayerMesh:
    mesh = layers.mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    roots = np.arange(coordinates.shape[0], dtype=np.int64)
    shifts = np.zeros((coordinates.shape[0], 1), dtype=np.int64)
    root_by_transverse_coordinate: dict[tuple[float, float], int] = {}
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Periodic layer coordinates must be a 3D vertex bank.")
    for row in range(coordinates.shape[0]):
        x, y, z = (float(value) for value in coordinates[row])
        if x == 0.0:
            root_by_transverse_coordinate[(y, z)] = row
    if not root_by_transverse_coordinate:
        raise ValueError("Periodic layer mesh lost its authored x=0 seam.")
    for row in range(coordinates.shape[0]):
        x, y, z = (float(value) for value in coordinates[row])
        if x == 1.0:
            key = (y, z)
            try:
                roots[row] = root_by_transverse_coordinate[key]
            except KeyError as error:
                raise ValueError(
                    "Periodic layer mesh has an unmatched authored x=1 seam vertex."
                ) from error
            shifts[row, 0] = 1
    periodic_cell = PeriodicCell(np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64))
    periodic = PeriodicMeshTopology(mesh, periodic_cell, roots, shifts)
    periodic_mesh = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version=mesh.numeric_version,
        periodic_topology=periodic,
    )
    return phx.meshing.BoundaryLayerMesh(
        periodic_mesh,
        layers.cap,
        layers.wall_vertices,
        layers.cap_vertices,
        layers.layer_index,
        layers.validity,
        layers.evidence,
        source_wall=layers.source_wall,
        wall_association=layers.wall_association,
        obstacles=layers.obstacles,
        column_index=layers.column_index,
        column_scale=layers.column_scale,
        column_layer_count=layers.column_layer_count,
        column_stretch=layers.column_stretch,
        column_merged_top=layers.column_merged_top,
        column_level_coordinates=layers.column_level_coordinates,
        column_vertex_sources=layers.column_vertex_sources,
        column_vertex_ids=layers.column_vertex_ids,
        column_vertex_identity=layers.column_vertex_identity,
        column_source_geometry_entity_ids=layers.column_source_geometry_entity_ids,
        column_source_occurrence_paths=layers.column_source_occurrence_paths,
        column_source_face_offsets=layers.column_source_face_offsets,
        column_source_face_ids=layers.column_source_face_ids,
        column_source_face_orientations=layers.column_source_face_orientations,
        column_source_face_weights=layers.column_source_face_weights,
        column_source_face_roles=layers.column_source_face_roles,
        column_parent_offsets=layers.column_parent_offsets,
        column_parent_ids=layers.column_parent_ids,
        column_parent_weights=layers.column_parent_weights,
        column_construction_kinds=layers.column_construction_kinds,
        front_row_ids=layers.front_row_ids,
        front_row_identity=layers.front_row_identity,
        front_row_source_piece_indices=layers.front_row_source_piece_indices,
        front_row_column_ids=layers.front_row_column_ids,
        front_row_source_entity_ids=layers.front_row_source_entity_ids,
        front_row_source_entity_kinds=layers.front_row_source_entity_kinds,
        front_row_source_global_ids=layers.front_row_source_global_ids,
        front_row_orientations=layers.front_row_orientations,
        front_row_source_occurrence_paths=layers.front_row_source_occurrence_paths,
        vertex_source_vertex_ids=layers.vertex_source_vertex_ids,
        vertex_source_entity_ids=layers.vertex_source_entity_ids,
        vertex_column_ids=layers.vertex_column_ids,
        vertex_source_occurrence_paths=layers.vertex_source_occurrence_paths,
        vertex_column_parent_offsets=layers.vertex_column_parent_offsets,
        vertex_column_parent_ids=layers.vertex_column_parent_ids,
        vertex_column_parent_weights=layers.vertex_column_parent_weights,
        control_id=layers.control_id,
        policy_id=layers.policy_id,
    )


def _periodic_core_complex(
    layers: phx.meshing.BoundaryLayerMesh,
) -> tuple[
    PiecewiseLinearComplex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cap = layers.cap
    if cap is None or layers.mesh.periodic_topology is None:
        raise ValueError(
            "Periodic layer fixture requires an identified volume layer mesh."
        )
    cap_triangles = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks)
    )
    cap_layer_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
    source_vertex_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)[
        cap_layer_vertices
    ]
    cap_row_by_source_id = {
        int(identifier): row for row, identifier in enumerate(source_vertex_ids.tolist())
    }
    if set(cap_row_by_source_id) != {20, 21, 22, 23}:
        raise ValueError("Periodic cap must retain every source wall vertex once.")
    source_rows = {
        identifier: cap_row_by_source_id[identifier] for identifier in range(20, 24)
    }
    c0, c1, c2, c3 = (
        source_rows[20],
        source_rows[21],
        source_rows[22],
        source_rows[23],
    )
    cap_count = cap.coordinates.shape[0]
    top = cap_count + np.arange(cap_count, dtype=np.int64)
    t0, t1, t2, t3 = (int(top[source_rows[key]]) for key in range(20, 24))
    polygons = (
        *tuple(row for row in cap_triangles),
        *tuple(row for row in (cap_triangles[:, ::-1] + cap_count)),
        np.asarray((c0, t2, t0), dtype=np.int64),
        np.asarray((c0, c2, t2), dtype=np.int64),
        np.asarray((c1, t1, t3), dtype=np.int64),
        np.asarray((c1, t3, c3), dtype=np.int64),
        np.asarray((c0, t1, c1), dtype=np.int64),
        np.asarray((c0, t0, t1), dtype=np.int64),
        np.asarray((c2, t3, t2), dtype=np.int64),
        np.asarray((c2, c3, t3), dtype=np.int64),
    )
    vertices = np.concatenate(
        (
            np.asarray(cap.coordinates, dtype=np.float64),
            np.asarray(cap.coordinates, dtype=np.float64)
            + np.asarray((0.0, 0.0, 0.4), dtype=np.float64),
        )
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(polygons), 1)),
        ("core",),
        vertex_ids=tuple(f"core-vertex:{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"core-facet:{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segments=np.empty((0, 2), dtype=np.int64),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    layer_topology = layers.mesh.periodic_topology
    if layer_topology is None:
        raise ValueError("Periodic layer fixture lost its volume orbit identity.")
    roots = np.empty((vertices.shape[0],), dtype=np.int64)
    shifts = np.empty((vertices.shape[0], layer_topology.cell.rank), dtype=np.int64)
    layer_roots = np.asarray(layer_topology.vertex_representatives, dtype=np.int64)[
        cap_layer_vertices
    ]
    layer_root_source_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)[
        layer_roots
    ]
    cap_root_rows = np.asarray(
        [cap_row_by_source_id[int(identifier)] for identifier in layer_root_source_ids],
        dtype=np.int64,
    )
    cap_shifts = np.asarray(layer_topology.vertex_shifts, dtype=np.int64)[
        cap_layer_vertices
    ]
    roots[:cap_count] = cap_root_rows
    roots[cap_count:] = cap_count + cap_root_rows
    shifts[:cap_count] = cap_shifts
    shifts[cap_count:] = cap_shifts
    seam_start = cap_triangles.shape[0] * 2
    seam_pairs = np.asarray(
        ((seam_start, seam_start + 2), (seam_start + 1, seam_start + 3)),
        dtype=np.int64,
    )
    vertex_layer_ids = np.concatenate(
        (
            cap_layer_vertices,
            np.full((cap_count,), -1, dtype=np.int64),
        )
    )
    return (
        complex_,
        vertex_layer_ids,
        np.arange(cap_triangles.shape[0], dtype=np.int64),
        roots,
        shifts,
        seam_pairs,
    )


@pytest.mark.parametrize("declared_group", (False, True))
def test_embedded_wall_propagates_authored_periodicity_to_layer_and_cap(
    declared_group: bool,
) -> None:
    carrier, _, _, _ = _periodic_wall_source()
    translation = np.eye(4)
    translation[0, 3] = 1.0
    cell = (
        PeriodicIsometryGroup(translation[None])
        if declared_group
        else PeriodicCell(np.asarray(((1.0, 0.0, 0.0),)))
    )
    roots = np.asarray((0, 0, 2, 2), dtype=np.int64)
    shifts = np.asarray(((0,), (1,), (0,), (1,)), dtype=np.int64)
    topology = PeriodicMeshTopology(carrier, cell, roots, shifts)
    wall = CellMesh(
        carrier.coordinates,
        carrier.blocks,
        vertex_global_ids=carrier.vertex_global_ids,
        periodic_topology=topology,
    )
    control = phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope.whole(wall, 2),
        phx.meshing.LayerSchedule((0.1,)),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    layers = phx.meshing.prepare_boundary_layers(wall, control)
    from phydrax.meshing._boundary_layer import _reprove_retained_boundary_layers

    assert _reprove_retained_boundary_layers(layers, wall, control) is layers
    assert layers.cap is not None
    for mesh in (layers.mesh, layers.cap):
        quotient = mesh.periodic_topology
        assert quotient is not None
        assert quotient.cell is cell
        assert np.any(np.asarray(quotient.vertex_shifts) != 0)
        quotient.require_coordinates(mesh.coordinates)
    np.testing.assert_array_equal(
        layers.cap.coordinates, layers.mesh.coordinates[np.asarray(layers.cap_vertices)]
    )


def test_original_ridge_periodic_seam_partition_uses_exact_source_stencils() -> None:
    from dataclasses import replace
    from fractions import Fraction

    from phydrax.discretization._periodic_exact import exact_periodic_element
    from phydrax.meshing._layer_core_periodic import partition_periodic_seams
    from phydrax.meshing._layer_core_subdivision import subdivide_nested_prism_columns
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
    )

    source, _, _ = _archived_ridge_source(cap_refinement="conforming")
    partition = subdivide_nested_prism_columns(source.layers.mesh, 2)
    records = partition_periodic_seams(source, partition)
    assert records
    assert source.core_seam_polygon_pairs is not None
    assert source.layers.mesh.periodic_topology is not None
    roots = np.asarray(source.core_vertex_representatives)
    assert sum(len(record.first) for record in records) > len(
        source.core_seam_polygon_pairs
    )
    coordinates = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(source.complex.vertices)
    )
    for record in records:
        assert all(len(rows) == 2 for rows in record.source_polygons)
        action = exact_periodic_element(
            source.layers.mesh.periodic_topology.cell, np.asarray(record.deck_shift)
        )
        for first, second in zip(record.first, record.second, strict=True):
            for source_stencil, target_stencil in zip(
                first, reversed(second), strict=True
            ):
                assert sum(weight for _, weight in source_stencil) == 1
                assert sum(weight for _, weight in target_stencil) == 1
                assert sorted(
                    (int(roots[row]), weight) for row, weight in source_stencil
                ) == sorted((int(roots[row]), weight) for row, weight in target_stencil)
                point = tuple(
                    sum(
                        (
                            weight * coordinates[row][axis]
                            for row, weight in source_stencil
                        ),
                        Fraction(),
                    )
                    for axis in range(3)
                ) + (Fraction(1),)
                image = tuple(
                    sum(
                        (action[axis][column] * point[column] for column in range(4)),
                        Fraction(),
                    )
                    for axis in range(3)
                )
                target = tuple(
                    sum(
                        (
                            weight * coordinates[row][axis]
                            for row, weight in target_stencil
                        ),
                        Fraction(),
                    )
                    for axis in range(3)
                )
                assert image == target
    forged = partition.coordinates.copy()
    forged[-1, 0] += 1.0
    with pytest.raises(ValueError, match="source stencils"):
        partition_periodic_seams(source, replace(partition, coordinates=forged))


def test_periodic_wall_layer_source_retains_orbit_and_bound_charts() -> None:
    wall, domain, charts, boundary_source = _periodic_wall_source()
    control = phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope.whole(wall, 2),
        phx.meshing.LayerSchedule((0.1,)),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    layers = phx.meshing.prepare_boundary_layers(wall, control)
    layers = _bind_periodic_layer_identity(layers)
    if layers.cap is None or layers.mesh.periodic_topology is None:
        raise ValueError("Periodic layer construction lost its cap or quotient topology.")
    complex_, vertex_layer_ids, cap_polygon_ids, roots, shifts, seams = (
        _periodic_core_complex(layers)
    )
    layer_cell_count = sum(block.cell_count for block in layers.mesh.blocks)
    source = NativeLayerCoreSource(
        layers,
        complex_,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=vertex_layer_ids,
        cap_polygon_ids=cap_polygon_ids,
        layer_regions=np.zeros((layer_cell_count,), dtype=np.int64),
        region_entity_ids=np.asarray(complex_.region_entity_ids, dtype=np.int64),
        region_occurrence_paths=complex_.region_occurrence_paths,
        region_ids=complex_.region_ids,
        core_region_map=np.arange(len(complex_.region_ids), dtype=np.int64),
        fidelity_source=boundary_source,
        source_domain=domain,
        source_wall_charts=charts,
        source_boundary_charts=charts,
        core_facet_source_ids=complex_.facet_ids,
        core_facet_source_occurrence_paths=complex_.facet_occurrence_paths,
    )

    periodic_source = prepare_periodic_layer_core_source(
        source,
        roots,
        shifts,
        seams,
    )

    assert periodic_source.periodic_topology_id == (
        layers.mesh.periodic_topology.periodic_topology_id
    )
    assert periodic_source.source_wall_charts is charts
    assert periodic_source.source_boundary_charts is charts
    charts.require_bound(wall, charts.geometry, boundary_source)


def test_original_curved_periodic_reference_retains_column_and_core_orbits() -> None:
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _control,
        _curved_periodic_profile_geometry,
        _curved_periodic_reference_core,
        _curved_periodic_root_geometry,
        _curved_periodic_wall,
    )

    wall, _ = _curved_periodic_wall()
    schedule = phx.meshing.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
    control = _control(
        wall,
        schedule,
        corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
        feature_angle=0.1,
        growth_rate_bounds=(0.9999999999, 1.0000000001),
    )
    layers = phx.meshing.prepare_boundary_layers(wall, control)
    lower, upper, orbits = _curved_periodic_profile_geometry(wall)
    lower_points = np.asarray(lower.coordinates)
    upper_points = np.asarray(upper.coordinates)
    np.testing.assert_array_equal(
        upper_points - lower_points,
        np.broadcast_to(np.asarray((0.0, 0.0, 1.0 / 16.0)), lower_points.shape),
    )
    assert np.max(lower_points[:, 2]) == 3.0 / 32.0
    for orbit in {value for value in orbits if "x-orbit" in value}:
        rows = [row for row, value in enumerate(orbits) if value == orbit]
        assert len(rows) == 2
        rows.sort(key=lambda row: lower_points[row, 0])
        np.testing.assert_array_equal(
            lower_points[rows[1]] - lower_points[rows[0]], (1.0, 0.0, 0.0)
        )
    assert len({value for value in orbits if "x-orbit" in value}) == 3
    root, geometry, _ = _curved_periodic_root_geometry(wall, layers)
    complex_, domain, vertex_map, cap_ids, roots, shifts, seams = (
        _curved_periodic_reference_core(wall, layers, root)
    )
    assert all(block.cell_kind == "prism" for block in root.blocks)
    assert len(geometry.resolve(root)[1]) == len(root.blocks)
    assert domain.region_ids == ("wall-material", "core-material")
    assert layers.cap is not None
    assert len(cap_ids) == sum(block.cell_count for block in layers.cap.blocks)
    assert np.all(vertex_map[: len(layers.cap_vertices)] >= 0)
    assert len(seams) == 2
    assert roots.shape == (complex_.vertices.shape[0],)
    assert shifts.shape == (complex_.vertices.shape[0], 1)


def test_mapped_layer_core_accepts_composite_regions_outside_the_core(
    tmp_path: Any,
) -> None:
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
    from phydrax.meshing._layer_core_controls import (
        mapped_wall_source_scope,
        realized_layer_control,
    )
    from phydrax.meshing._layer_core_source_restore import (
        validate_mapped_layer_core_source,
    )
    from phydrax.meshing._volume_generation import _mapped_plc_region_rows
    from phydrax.meshing.providers._native_layer import _mapped_reference_support_issues
    from phydrax.meshing.providers._native_sources import (
        _mapped_layer_core_facet_relation,
    )
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _control,
        _curved_periodic_full_authority,
        _curved_periodic_reference_core,
        _curved_periodic_root_geometry,
        _curved_periodic_wall,
    )

    wall, _ = _curved_periodic_wall()
    schedule = phx.meshing.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
    wall_control = _control(
        wall,
        schedule,
        corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
        feature_angle=0.1,
        growth_rate_bounds=(0.9999999999, 1.001),
    )
    wall_policy = phx.meshing.BoundaryLayerPolicy()
    layers = phx.meshing.prepare_boundary_layers(wall, wall_control, policy=wall_policy)
    root, geometry, _ = _curved_periodic_root_geometry(wall, layers)
    complex_, domain, vertex_map, cap_ids, roots, shifts, seams = (
        _curved_periodic_reference_core(wall, layers, root)
    )
    source_id, revision = domain.source_id, domain.source_revision
    layer_count = sum(block.cell_count for block in layers.mesh.blocks)
    mapped = MappedReferenceDomain(
        domain,
        root,
        geometry,
        np.concatenate(
            (np.zeros(layer_count, dtype=np.int64), np.ones(len(cap_ids), dtype=np.int64))
        ),
        source_id=source_id,
        source_revision=revision,
        reference_domain_vertex_root_ids=np.asarray(root.vertex_global_ids),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    authority, query, source_ids, source_paths = _curved_periodic_full_authority(
        wall, schedule, layers, complex_, vertex_map, cap_ids, mapped
    )
    arguments: _LayerArguments = {
        "vertex_layer_ids": vertex_map,
        "cap_polygon_ids": cap_ids,
        "layer_regions": np.zeros(layer_count, dtype=np.int64),
        "region_entity_ids": np.arange(2, dtype=np.int64),
        "region_occurrence_paths": ((), ()),
        "region_ids": domain.region_ids,
        "core_region_map": np.asarray((1,), dtype=np.int64),
        "core_facet_source_ids": source_ids,
        "core_facet_source_occurrence_paths": source_paths,
        "core_vertex_representatives": roots,
        "core_vertex_shifts": shifts,
        "core_seam_polygon_pairs": seams,
        "layer_control": wall_control,
        "layer_policy": wall_policy,
    }
    reference = NativeLayerCoreSource(layers, complex_, source_id, revision, **arguments)
    source = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        **arguments,
        reference_source=reference,
        mapped_domain=mapped,
        fidelity_source=query,
    )
    validate_mapped_layer_core_source(source)
    import subprocess
    import sys

    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        layer_core_authoring_recipe,
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        restore_layer_core_authoring,
        write_model_artifact,
    )

    recipe = layer_core_authoring_recipe(source, wall_control, wall_policy)
    entry = CompositionEntry(
        recipe,
        entry_id="source",
        role="topology",
        owner_id=source.source_id,
        structure_id=wall.topology_id,
        revision_id=source.binding_id,
        semantics_id=source.source_revision,
    )
    path = tmp_path / "mapped-layer-inputs.zip"
    write_model_artifact(
        path,
        Composition((entry,), boundary_id="mapped-layer-inputs"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    receipt = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from phydrax.lifecycle import Composition\n"
            "from phydrax.meshing._model_codec import (MODEL_ARCHIVE_LIMITS, "
            "read_model_artifact, restore_model_archive_value, restore_layer_core_authoring)\n"
            "import sys\n"
            "c = read_model_artifact(sys.argv[1], expected_type=Composition, "
            "restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS, "
            "expected_content_id=sys.argv[2])\n"
            "s = restore_layer_core_authoring(c.entries[0].value)\n"
            "assert s.binding_id == sys.argv[3]\n"
            "assert s.reference_source.binding_id == sys.argv[4]\n"
            "assert s.mapped_domain.domain_id == sys.argv[5]\n"
            "assert s.fidelity_source.source_query_id == sys.argv[6]\n",
            str(path),
            receipt,
            source.binding_id,
            reference.binding_id,
            mapped.domain_id,
            query.source_query_id,
        ],
        check=True,
    )
    for changed in (
        recipe._replace(reference_binding_id="wrong-reference"),
        recipe._replace(source_id="wrong-source"),
        recipe._replace(
            core_facet_source_ids=tuple(
                "wrong-face" if item is not None else None
                for item in recipe.core_facet_source_ids
            )
        ),
        recipe._replace(core_region_map=np.zeros_like(recipe.core_region_map)),
        recipe._replace(core_vertex_shifts=np.asarray(recipe.core_vertex_shifts) + 1),
    ):
        with pytest.raises((TypeError, ValueError)):
            restore_layer_core_authoring(changed)
    altered_layer_regions = np.zeros(layer_count, dtype=np.int64)
    altered_layer_regions[0] = 1
    changed_materials = arguments.copy()
    changed_materials["layer_regions"] = altered_layer_regions
    wrong_material = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        **changed_materials,
        reference_source=reference,
        mapped_domain=mapped,
        fidelity_source=query,
    )
    with pytest.raises(ValueError):
        validate_mapped_layer_core_source(wrong_material)
    changed_shifts = shifts.copy()
    changed_shifts[0, 0] += 1
    changed_shift_arguments = arguments.copy()
    changed_shift_arguments["core_vertex_shifts"] = changed_shifts
    wrong_periodic = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        **changed_shift_arguments,
        reference_source=reference,
        mapped_domain=mapped,
        fidelity_source=query,
    )
    with pytest.raises(ValueError):
        validate_mapped_layer_core_source(wrong_periodic)
    wrong_reference = NativeLayerCoreSource(
        layers, complex_, source_id, revision, **changed_materials
    )
    wrong_reference_source = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        **arguments,
        reference_source=wrong_reference,
        mapped_domain=mapped,
        fidelity_source=query,
    )
    with pytest.raises(ValueError):
        validate_mapped_layer_core_source(wrong_reference_source)
    wrong_frame = MappedReferenceDomain(
        domain,
        root,
        geometry,
        mapped.cell_regions,
        source_id=source_id,
        source_revision=revision,
        reference_domain_vertex_root_ids=np.asarray(root.vertex_global_ids),
        coordinate_contract=phx.SpatialCoordinateContract(
            mapped.coordinate_contract.length_unit, reference_frame="wrong-frame"
        ),
    )
    wrong_frame_source = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        **arguments,
        reference_source=reference,
        mapped_domain=wrong_frame,
        fidelity_source=query,
    )
    with pytest.raises(ValueError):
        validate_mapped_layer_core_source(wrong_frame_source)
    assert domain.region_ids == ("wall-material", "core-material")
    assert complex_.region_ids == ("core-material",)
    np.testing.assert_array_equal(_mapped_plc_region_rows(complex_, mapped), (1,))
    wall_scope = mapped_wall_source_scope(source)
    layer_volume = phx.meshing.MeshingScope(
        source_id,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        np.asarray((source.region_entity_ids[0],)),
    )
    source_control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=layer_volume,
        corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
        feature_angle=0.1,
        growth_rate_bounds=wall_control.growth_rate_bounds,
    )
    assert realized_layer_control(source, source_control).control_id == layers.control_id
    from fractions import Fraction

    from phydrax.meshing._layer_core_mapped import (
        _certify_mapped_column_controls,
        _mapped_wall_distance_bounds,
        _mapped_wall_graph,
    )

    volume_scope = phx.meshing.MeshingScope(
        source_id,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"), allow_mixed=True
            ),
        ),
        volume_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                volume_scope,
                2.0,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        layer_controls=(source_control,),
    )
    graph, wall_bounds = _mapped_wall_graph(source, specification)
    assert graph == {(1,): Fraction(3, 8), (2,): Fraction(-3, 8)}
    assert wall_bounds == (Fraction(0), Fraction(1), Fraction(0), Fraction(1))
    points = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(layers.mesh.coordinates)
    )
    source_vertex_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    columns = {}
    for identifier in np.unique(source_vertex_ids):
        rows = np.flatnonzero(source_vertex_ids == identifier)
        rows = sorted(rows, key=lambda row: points[int(row)][2])
        columns[((int(identifier), Fraction(1)),)] = {
            level: int(row) for level, row in enumerate(rows)
        }
    _certify_mapped_column_controls(source, specification, columns, points)
    from phydrax.meshing._layer_core_mapped import _block_order_regions, _geometry_groups

    parent_mesh = reference.layers.mesh
    parent_blocks, parent_geometry, _, _, parent_points = _geometry_groups(
        mapped,
        parent_mesh,
        CellGeometrySpec.affine(parent_mesh),
        _block_order_regions(parent_mesh, reference.layer_regions),
    )
    mapped_parent_mesh = CellMesh(
        parent_points, parent_blocks, vertex_global_ids=parent_mesh.vertex_global_ids
    )
    parent_ids = np.sort(np.asarray(parent_mesh.entity_set(3).entity_ids, dtype=np.int64))
    layer_indices = np.asarray(layers.layer_index, dtype=np.int64)
    cell_columns = np.asarray(layers.column_index, dtype=np.int64)
    column = next(
        int(value)
        for value in np.unique(cell_columns)
        if set(layer_indices[cell_columns == value]) == {0, 1}
    )
    parent_by_level = {
        int(level): int(identifier)
        for identifier, level, value in zip(
            parent_ids, layer_indices, cell_columns, strict=True
        )
        if int(value) == column
    }
    quarter = Fraction(1, 4)
    corners = ((quarter, quarter, Fraction(0)), (quarter, quarter, Fraction(1)))
    first_pair = parent_geometry.evaluate_exact(
        mapped_parent_mesh, np.asarray((parent_by_level[0],), dtype=np.int64), (corners,)
    )[0]
    second_pair = parent_geometry.evaluate_exact(
        mapped_parent_mesh, np.asarray((parent_by_level[1],), dtype=np.int64), (corners,)
    )[0]
    assert tuple(first_pair[1]) == tuple(second_pair[0])
    exact_interior = (tuple(first_pair[0]), tuple(first_pair[1]), tuple(second_pair[1]))
    assert all(isinstance(value, Fraction) for point in exact_interior for value in point)
    interior = tuple(
        tuple(value for value in point if isinstance(value, Fraction))
        for point in exact_interior
    )
    assert Fraction(0) < interior[0][1] < Fraction(1)
    source_cells = {
        int(identifier): np.asarray(vertices, dtype=np.int64)
        for block in parent_mesh.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    parent_vertices = source_cells[parent_by_level[0]]
    vertex_columns = np.asarray(layers.vertex_column_ids, dtype=np.int64)
    weights = (Fraction(1, 2), quarter, quarter)
    key = tuple(
        sorted(
            (int(vertex_columns[parent_vertices[3 + index]]), weight)
            for index, weight in enumerate(weights)
        )
    )
    start = len(points)
    points = (*points, *interior)
    columns[key] = {level: start + level for level in range(3)}
    _certify_mapped_column_controls(source, specification, columns, points)
    first = _mapped_wall_distance_bounds(
        graph, (Fraction(0), Fraction(1)), Fraction(0), points[4][2], Fraction(1, 2**40)
    )
    second = _mapped_wall_distance_bounds(
        graph, (Fraction(0), Fraction(1)), Fraction(0), points[8][2], Fraction(1, 2**40)
    )
    assert second[0] - first[1] > Fraction.from_float(1.0000000001) * first[1]
    assert second[1] - first[0] < Fraction.from_float(1.001) * first[0]
    forged_points = list(points)
    forged_points[8] = forged_points[0]
    with pytest.raises(ValueError, match="inward"):
        _certify_mapped_column_controls(
            source, specification, columns, tuple(forged_points)
        )
    assert _mapped_reference_support_issues(source) == []
    assert authority.root_relation is not None
    full = authority.complex
    original_keys = tuple(
        authority.source_entity_key_for_scope(2, int(full.facet_entity_ids[int(facet)]))
        for facet in full.polygon_facets
    )
    piece_keys = tuple(
        zip(
            authority.root_relation.polygon_source_entity_keys,
            authority.root_relation.polygon_source_occurrence_paths,
            strict=True,
        )
    )
    assert piece_keys != original_keys
    relation = _mapped_layer_core_facet_relation(source, authority)
    assert all(relation[int(row)] >= 0 for row in cap_ids)
    for polygon_row, original_row in enumerate(relation):
        if original_row < 0:
            continue
        facet = int(complex_.polygon_facets[polygon_row])
        assert original_keys[original_row] == (source_ids[facet], source_paths[facet])
    from phydrax.meshing._layer_core_edge_policy import noncap_source_edge_classification

    classification = noncap_source_edge_classification(source)
    assert classification is not None
    removable, authored_segments = classification
    assert removable
    assert not authored_segments
    locked_polygons = set(int(value) for value in cap_ids)
    locked_polygons.update(int(value) for value in np.asarray(seams).reshape(-1))
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    locked_edges = set()
    for polygon in locked_polygons:
        vertices = tuple(
            int(value)
            for value in polygon_vertices[offsets[polygon] : offsets[polygon + 1]]
        )
        locked_edges.update(
            tuple(sorted((first, vertices[(index + 1) % len(vertices)])))
            for index, first in enumerate(vertices)
        )
    assert not (set(removable) & locked_edges)
    from phydrax.meshing._layer_core_periodic import partition_periodic_seams
    from phydrax.meshing._layer_core_subdivision import subdivide_nested_prism_columns

    refined_partition = subdivide_nested_prism_columns(reference.layers.mesh, 2)
    paired_seams = partition_periodic_seams(reference, refined_partition)
    proven_diagonals = set()
    for seam in paired_seams:
        for first_polygon, second_polygon in seam.source_polygons:

            def polygon_edges(row: int) -> set[tuple[int, int]]:
                vertices = tuple(
                    int(value)
                    for value in polygon_vertices[offsets[row] : offsets[row + 1]]
                )
                return {
                    (
                        min(first, vertices[(index + 1) % len(vertices)]),
                        max(first, vertices[(index + 1) % len(vertices)]),
                    )
                    for index, first in enumerate(vertices)
                }

            shared = polygon_edges(first_polygon) & polygon_edges(second_polygon)
            assert len(shared) == 1
            proven_diagonals.update(shared)
    refined_classification = noncap_source_edge_classification(
        source, partition=refined_partition
    )
    assert refined_classification is not None
    refined_removable, _ = refined_classification
    assert set(refined_removable) & proven_diagonals
    assert (set(refined_removable) & locked_edges) <= proven_diagonals
    assert (0, 7) in refined_removable
    from phydrax.meshing._layer_core_facet_authority import MappedCoreFacetAuthority

    roots_by_key = {}
    for row, chart_id in enumerate(query.face_chart_ids):
        key = query.chart_source_key(2, chart_id)
        roots_by_key.setdefault(key, set()).add(
            (
                int(query.polygon_root_face_global_ids[row]),
                int(query.polygon_root_cell_global_ids[row]),
                int(query.polygon_root_local_face_indices[row]),
            )
        )
    for pair in removable | refined_removable:
        edge_row = tuple(
            tuple(sorted(edge)) for edge in complex_.plc_edge_vertices
        ).index(pair)
        facets = tuple(int(facet) for facet in complex_.plc_edge_facets[edge_row])
        facet_keys = tuple((source_ids[facet], source_paths[facet]) for facet in facets)
        if all(key == (None, None) for key in facet_keys):
            relation = MappedCoreFacetAuthority(
                source, tuple(sorted(set(facets)))
            ).root_relation
            roots = {
                facet: {
                    (int(face), int(cell[0]), int(local[0]))
                    for row, face, cell, local in zip(
                        relation.polygon_source_facet_rows,
                        relation.polygon_root_face_global_ids,
                        relation.polygon_root_cell_global_ids,
                        relation.polygon_root_local_face_indices,
                        strict=True,
                    )
                    if int(row) == facet
                }
                for facet in facets
            }
            assert all(len(roots[facet]) == 1 for facet in facets)
            assert all(roots[facet] == roots[facets[0]] for facet in facets)
        else:
            assert facet_keys and all(key in roots_by_key for key in facet_keys)
            assert len(roots_by_key[facet_keys[0]]) == 1
            assert all(
                roots_by_key[key] == roots_by_key[facet_keys[0]] for key in facet_keys
            )
    restored = NativeLayerCoreSource.from_stored_fields(source)
    assert restored.binding_id == source.binding_id
    assert restored.periodic_topology_id == source.periodic_topology_id
    stale = copy(source)
    object.__setattr__(stale, "binding_id", "stale-layer-core-binding")
    with pytest.raises(ValueError, match="identity no longer validates"):
        NativeLayerCoreSource.from_stored_fields(stale)
    bound = [index for index, value in enumerate(source_ids) if value is not None]
    bad_ids = list(source_ids)
    bad_ids[bound[0]], bad_ids[bound[1]] = bad_ids[bound[1]], bad_ids[bound[0]]
    bad_arguments = arguments.copy()
    bad_arguments["core_facet_source_ids"] = tuple(bad_ids)
    bad_reference = NativeLayerCoreSource(
        layers, complex_, source_id, revision, **bad_arguments
    )
    bad_source = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        **bad_arguments,
        reference_source=bad_reference,
        mapped_domain=mapped,
        fidelity_source=query,
    )
    with pytest.raises(ValueError, match="source identity"):
        validate_mapped_layer_core_source(bad_source)
    with pytest.raises(ValueError, match="source identity"):
        _mapped_layer_core_facet_relation(bad_source, authority)
    source_polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    incidence = np.asarray(complex_.facet_regions, dtype=np.int64)
    exterior = next(
        row
        for row, value in enumerate(source_ids)
        if value is not None and row not in set(cap_ids.tolist())
    )
    for mode in ("cap-sides", "exterior-sides", "exterior-orientation"):
        changed_polygons = source_polygons.copy()
        changed_sides = incidence.copy()
        if mode == "cap-sides":
            changed_sides[int(cap_ids[0])] = changed_sides[int(cap_ids[0]), ::-1].copy()
        elif mode == "exterior-sides":
            changed_sides[exterior] = changed_sides[exterior, ::-1].copy()
        else:
            changed_polygons[exterior] = changed_polygons[exterior, ::-1].copy()
        with pytest.raises(ValueError):
            changed = PiecewiseLinearComplex(
                np.asarray(complex_.vertices),
                tuple(row for row in changed_polygons),
                np.asarray(complex_.polygon_facets),
                changed_sides,
                complex_.region_ids,
                vertex_ids=complex_.vertex_ids,
                vertex_occurrence_paths=complex_.vertex_occurrence_paths,
                facet_ids=complex_.facet_ids,
                facet_occurrence_paths=complex_.facet_occurrence_paths,
                segments=np.asarray(complex_.segments),
                segment_ids=complex_.segment_ids,
                segment_occurrence_paths=complex_.segment_occurrence_paths,
                region_occurrence_paths=complex_.region_occurrence_paths,
                boundary="fixed",
            )
            changed_reference = NativeLayerCoreSource(
                layers, changed, source_id, revision, **arguments
            )
            changed_source = NativeLayerCoreSource(
                layers,
                changed,
                source_id,
                revision,
                **arguments,
                reference_source=changed_reference,
                mapped_domain=mapped,
                fidelity_source=query,
            )
            _mapped_layer_core_facet_relation(changed_source, authority)


def test_original_archived_ridge_reconstructs_fixed_periodic_core() -> None:
    from phydrax.meshing._layer_core_periodic import prepare_core_periodic
    from phydrax.meshing._layer_core_topology import prepare_identity
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
    )

    source, control, policy = _archived_ridge_source()
    assert source.layers.control_id == control.control_id
    assert source.layers.policy_id == policy.policy_id
    assert source.region_ids == ("wall-material", "core-material")
    assert source.source_domain is not None
    assert source.fidelity_source is not None
    assert all(
        source.core_facet_source_ids[int(row)] is not None
        and source.core_facet_source_occurrence_paths[int(row)] is not None
        for row in np.asarray(source.cap_polygon_ids)
    )
    np.testing.assert_allclose(
        source.layers.evidence.requested_thicknesses, (0.01, 0.01), rtol=0.0, atol=0.0
    )
    count = source.complex.vertices.shape[0] // 2
    np.testing.assert_allclose(
        np.asarray(source.complex.vertices[count:])
        - np.asarray(source.complex.vertices[:count]),
        np.broadcast_to((0.0, 0.0, 0.03), (count, 3)),
        rtol=0.0,
        atol=2.0e-17,
    )
    mapping, _, polygons, points = prepare_identity(
        source.layers, source.complex, source.vertex_layer_ids, source.cap_polygon_ids
    )
    assert prepare_core_periodic(source, mapping, polygons, points) is not None


def test_original_ridge_layer_core_retains_full_boundary_chart_witness() -> None:
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
        _periodic_layer_core_specification,
    )

    source, _, _ = _archived_ridge_source()
    specification = _periodic_layer_core_specification(source)
    result = (
        phx.meshing.NativeMeshingProvider(phx.meshing.NativeMeshingOptions("layer_core"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    certificate = result.certification
    assert certificate is not None and certificate.passed
    witness = certificate.inputs.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    ids = np.asarray(result.mesh.entity_set(2).entity_ids, dtype=np.int64)
    boundary = np.asarray(result.mesh.boundary_masks[2], dtype=np.bool_)
    assert set(witness.entity_global_ids.tolist()) == set(ids[boundary].tolist())
    assert witness.source_id == source.source_id
    assert witness.source_revision == source.source_revision


def test_original_ridge_authoring_archive_replays_and_rejects_tampering(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        layer_core_authoring_recipe,
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_layer_core_authoring,
        restore_model_archive_value,
        write_model_artifact,
    )
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
    )

    source, control, policy = _archived_ridge_source()
    recipe = layer_core_authoring_recipe(source, control, policy)
    entry = CompositionEntry(
        recipe,
        entry_id="source",
        role="topology",
        owner_id=source.source_id,
        structure_id=source.layers.source_wall.topology_id,
        revision_id=source.binding_id,
        semantics_id=source.source_revision,
    )
    path = tmp_path / "ridge-model.zip"
    write_model_artifact(
        path,
        Composition((entry,), boundary_id="ridge-source"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    content_id = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
        expected_content_id=content_id,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
            expected_content_id="0" * 64,
        )
    import phydrax.meshing._boundary_layer as boundary_layer

    def refuse_growth(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Cold layer restoration must not regrow the source.")

    monkeypatch.setattr(boundary_layer, "_grow_boundary_layers", refuse_growth)
    recipe = restored.entries[0].value
    rebuilt = restore_layer_core_authoring(recipe)
    assert rebuilt.binding_id == source.binding_id
    assert rebuilt.noncap_refinement == source.noncap_refinement
    assert rebuilt.layers.result_id == source.layers.result_id
    np.testing.assert_array_equal(
        rebuilt.layers.mesh.coordinates, source.layers.mesh.coordinates
    )
    np.testing.assert_array_equal(rebuilt.layers.column_index, source.layers.column_index)
    np.testing.assert_array_equal(
        rebuilt.layers.vertex_column_ids, source.layers.vertex_column_ids
    )
    assert rebuilt.layers.mesh.periodic_topology is not None
    assert source.layers.mesh.periodic_topology is not None
    assert rebuilt.layers.mesh.periodic_topology.periodic_topology_id == (
        source.layers.mesh.periodic_topology.periodic_topology_id
    )

    wall = copy(recipe.wall)
    changed_coordinates = np.array(wall.coordinates, copy=True)
    changed_coordinates[0, 2] += 0.001
    object.__setattr__(wall, "coordinates", changed_coordinates)
    changed_control = copy(recipe.control)
    object.__setattr__(changed_control, "feature_angle", control.feature_angle + 0.01)
    changed_policy = copy(recipe.policy)
    object.__setattr__(
        changed_policy, "proximity_fraction", policy.proximity_fraction - 0.01
    )
    changed_core = copy(source.complex)
    changed_sides = np.array(changed_core.facet_regions, copy=True)
    changed_sides[0] = changed_sides[0, ::-1]
    object.__setattr__(changed_core, "facet_regions", changed_sides)
    identities = list(recipe.core_facet_source_ids)
    identities[0] = "changed-source-face"
    occurrences = list(recipe.core_facet_source_occurrence_paths)
    occurrences[0] = ("changed-occurrence",)
    changed_domain = recipe.source_domain._replace(expected_domain_id="changed-domain")
    regions = list(recipe.source_domain.regions)
    region = regions[0]
    boundary = list(region.boundary)
    patch, orientation = boundary[0]
    boundary[0] = (patch, -orientation)
    regions[0] = type(region)(region.name, tuple(boundary))
    changed_side_domain = recipe.source_domain._replace(regions=tuple(regions))
    altered_layers = copy(recipe.layers)
    altered_column_ids = np.array(altered_layers.vertex_column_ids, copy=True)
    altered_column_ids[0] += 1
    object.__setattr__(altered_layers, "vertex_column_ids", altered_column_ids)
    tampered = (
        recipe._replace(wall=wall),
        recipe._replace(layers=altered_layers),
        recipe._replace(control=changed_control),
        recipe._replace(policy=changed_policy),
        recipe._replace(noncap_refinement="conforming"),
        recipe._replace(complex=changed_core),
        recipe._replace(core_facet_source_ids=tuple(identities)),
        recipe._replace(core_facet_source_occurrence_paths=tuple(occurrences)),
        recipe._replace(source_domain=changed_domain),
        recipe._replace(
            source_domain=changed_side_domain,
            fidelity_source=recipe.fidelity_source._replace(domain=changed_side_domain),
        ),
        recipe._replace(
            fidelity_source=recipe.fidelity_source._replace(
                expected_source_scope_id="changed-query-scope"
            )
        ),
    )
    for changed in tampered:
        with pytest.raises((TypeError, ValueError)):
            restore_layer_core_authoring(changed)


def test_ridge_source_cold_retains_actual_layers_without_growth(tmp_path: Any) -> None:
    import json
    import subprocess
    import sys

    writer = """
import json, os, sys
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._model_codec import (MODEL_ARCHIVE_LIMITS, layer_core_authoring_recipe,
    model_archive_content_id, write_model_artifact)
from tools.meshing_qualification.scenarios.native_hybrid_profiles import _archived_ridge_source
s, c, p = _archived_ridge_source()
e = CompositionEntry(layer_core_authoring_recipe(s, c, p), entry_id="source", role="topology",
    owner_id=s.source_id, structure_id=s.layers.source_wall.topology_id,
    revision_id=s.binding_id, semantics_id=s.source_revision)
write_model_artifact(sys.argv[1], Composition((e,), boundary_id="ridge-source"), limits=MODEL_ARCHIVE_LIMITS)
print(json.dumps({"pid": os.getpid(), "binding": s.binding_id, "layers": s.layers.result_id,
    "mesh": s.layers.mesh.mesh_id, "content": model_archive_content_id(sys.argv[1], limits=MODEL_ARCHIVE_LIMITS)}))
"""
    reader = """
import json, os, sys
from phydrax.lifecycle import Composition
from phydrax.meshing._model_codec import (MODEL_ARCHIVE_LIMITS, read_model_artifact,
    restore_model_archive_value, restore_layer_core_authoring)
import phydrax.meshing._boundary_layer as b
def no_growth(*args, **kwargs):
    raise AssertionError("Cold restore attempted to regrow source layers")
b._grow_boundary_layers = no_growth
expected = json.loads(sys.argv[2])
c = read_model_artifact(sys.argv[1], expected_type=Composition, restore=restore_model_archive_value,
    expected_content_id=expected["content"], limits=MODEL_ARCHIVE_LIMITS)
s = restore_layer_core_authoring(c.value("source"))
assert (s.binding_id, s.layers.result_id, s.layers.mesh.mesh_id) == (expected["binding"], expected["layers"], expected["mesh"])
assert os.getpid() != expected["pid"]
print(json.dumps({"pid": os.getpid(), "binding": s.binding_id, "layers": s.layers.result_id}))
"""
    path = tmp_path / "ridge-cold.phx"
    produced = subprocess.run(
        [sys.executable, "-c", writer, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    expected = json.loads(produced.stdout.strip().splitlines()[-1])
    consumed = subprocess.run(
        [sys.executable, "-c", reader, str(path), json.dumps(expected)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert json.loads(consumed.stdout.strip().splitlines()[-1])["pid"] != expected["pid"]


def test_mapped_wall_minimum_handles_endpoint_equality_and_nonconvex_refusal() -> None:
    from fractions import Fraction

    from phydrax.meshing._layer_core_mapped import _mapped_wall_distance_bounds

    half = Fraction(1, 2)
    lower, upper, _, squared = _mapped_wall_distance_bounds(
        {}, (Fraction(0), Fraction(1)), Fraction(0), half, Fraction(1, 1024)
    )
    assert (lower, upper, squared) == (half, half, Fraction(1, 4))
    with pytest.raises(ValueError, match="convexity"):
        _mapped_wall_distance_bounds(
            {(2,): Fraction(1)},
            (Fraction(0), Fraction(1)),
            half,
            Fraction(100),
            Fraction(1, 1024),
        )


def test_original_mapped_cap_scope_rejects_missing_and_extra_interface_faces(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    from copy import copy

    from phydrax.meshing import (
        _layer_core_mapped as mapped_owner,
        _layer_core_mapped_replay as mapped_replay,
        _layer_core_mapped_segment_overlay as mapped_overlay,
    )
    from phydrax.meshing._layer_core_boundary_replay import (
        _require_mapped_cap_chart_cover,
        mapped_cap_edge_owner,
    )
    from phydrax.meshing._layer_core_root_geometry import ComputedLayerCoreRoot
    from phydrax.meshing._volume_generation import _unmet_rows
    from phydrax.meshing.providers import _native_layer
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _run_wall_profile,
    )

    original = _native_layer._require_mapped_cap_interface
    original_chart = _native_layer._mapped_layer_surface_chart_witness
    original_proposal = mapped_owner._propose_mapped_physical_improvement
    original_root_init = ComputedLayerCoreRoot.__init__
    original_native_construction = _native_layer._native_construction
    primary_witnesses = []
    proposal_checked = []
    cap_queries = []
    checked = []
    root_checked = []
    cap_witnesses = []

    class _CapProofComplete(BaseException):
        pass

    class _FullProofRequired(Exception):
        pass

    def inspect_cap(mesh: Any, core_cell_ids: Any, witness: Any) -> None:
        original(mesh, core_cell_ids, witness)
        assert set(witness.source_entity_ids) == {
            "physical-facet:wall:0:level:2:wall:1:level:2:wall:3:level:2",
            "physical-facet:wall:0:level:2:wall:2:level:2:wall:3:level:2",
        }
        assert cap_queries
        query = cap_queries[-1]
        _require_mapped_cap_chart_cover(query, witness)
        missing_piece = copy(witness)
        object.__setattr__(
            missing_piece, "source_chart_ids", witness.source_chart_ids[:-1]
        )
        object.__setattr__(
            missing_piece,
            "source_parameter_rational_tokens",
            witness.source_parameter_rational_tokens[:-1],
        )
        with pytest.raises(ValueError, match="coverage is incomplete"):
            _require_mapped_cap_chart_cover(query, missing_piece)
        repeated_piece = copy(witness)
        object.__setattr__(
            repeated_piece,
            "source_chart_ids",
            witness.source_chart_ids + (witness.source_chart_ids[0],),
        )
        object.__setattr__(
            repeated_piece,
            "source_parameter_rational_tokens",
            witness.source_parameter_rational_tokens
            + (witness.source_parameter_rational_tokens[0],),
        )
        with pytest.raises(
            ValueError, match="coverage is incomplete or repeated|pieces overlap"
        ):
            _require_mapped_cap_chart_cover(query, repeated_piece)
        faces = np.asarray(witness.entity_global_ids, dtype=np.int64)
        assert len(set(faces.tolist())) > 2
        missing = copy(witness)
        object.__setattr__(missing, "entity_global_ids", faces[faces != faces[0]])
        with pytest.raises(ValueError, match="exactly the layer/core interface"):
            original(mesh, core_cell_ids, missing)
        noncap = set(np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)) - set(
            faces
        )
        assert noncap
        extra = copy(witness)
        object.__setattr__(extra, "entity_global_ids", np.append(faces, min(noncap)))
        with pytest.raises(ValueError, match="exactly the layer/core interface"):
            original(mesh, core_cell_ids, extra)
        checked.append(True)
        cap_witnesses.append(witness)

    def inspect_proposal(
        domain: Any, arrays: Any, regions: Any, inputs: Any, schedule: Any
    ) -> Any:
        result = original_proposal(domain, arrays, regions, inputs, schedule)
        assert result is not None
        proposal, represented, _, quality = result
        assert not _unmet_rows(quality)
        old_ids = np.asarray(arrays.vertex_ids, dtype=np.int64)
        old_reference = np.asarray(arrays.points, dtype=np.float64)
        old_mesh = CellMesh(
            old_reference,
            (
                CellBlock(
                    "candidate-test",
                    "tetrahedron",
                    arrays.tetrahedra,
                    global_ids=arrays.cell_ids,
                ),
            ),
            vertex_global_ids=old_ids,
        )
        old_physical = mapped_owner._geometry_groups(
            domain,
            old_mesh,
            CellGeometrySpec.affine(old_mesh),
            np.asarray(regions, dtype=np.int64)[
                np.asarray(arrays.tetrahedron_regions, dtype=np.int64)
            ],
        )[4]
        old_rows = {int(value): row for row, value in enumerate(old_ids)}
        new_rows = {int(value): row for row, value in enumerate(proposal.vertex_ids)}
        boundary = np.unique(
            np.concatenate(
                (
                    np.asarray(arrays.faces).reshape(-1),
                    np.asarray(arrays.segments).reshape(-1),
                )
            )
        )
        fixed = set(np.asarray(inputs, dtype=np.int64)) | set(old_ids[boundary])
        for identifier in fixed:
            assert identifier in new_rows
            assert np.array_equal(
                np.asarray(proposal.points[new_rows[identifier]]).view(np.uint64),
                old_physical[old_rows[identifier]].view(np.uint64),
            )
            assert np.array_equal(
                represented[new_rows[identifier]].view(np.uint64),
                old_reference[old_rows[identifier]].view(np.uint64),
            )
        changed = next(
            row
            for row, identifier in enumerate(proposal.vertex_ids)
            if int(identifier) not in old_rows
            or not np.array_equal(
                np.asarray(proposal.points[row]).view(np.uint64),
                old_physical[old_rows[int(identifier)]].view(np.uint64),
            )
        )
        outside = np.asarray(proposal.points).copy()
        outside[changed] = np.max(np.abs(old_physical)) + 1000
        with pytest.raises(
            ValueError, match="unique actual root cell|inverse-root search"
        ):
            mapped_owner._represent_mapped_proposal_vertices(
                domain,
                old_ids,
                old_reference,
                old_physical,
                proposal._replace(points=outside),
            )
        root_vertices = np.asarray(domain.reference_mesh.coordinates)
        uses = np.zeros(len(root_vertices), dtype=np.int64)
        for block in domain.reference_mesh.blocks:
            np.add.at(uses, np.asarray(block.vertices).reshape(-1), 1)
        shared = next(
            row
            for row in range(len(old_reference))
            if any(
                uses[index] > 1 and np.array_equal(point, old_reference[row])
                for index, point in enumerate(root_vertices)
            )
        )
        ambiguous = np.asarray(proposal.points).copy()
        ambiguous[changed] = old_physical[shared]
        with pytest.raises(
            ValueError,
            match="ambiguous root ancestry|unique actual root cell|inverse-root search",
        ):
            mapped_owner._represent_mapped_proposal_vertices(
                domain,
                old_ids,
                old_reference,
                old_physical,
                proposal._replace(points=ambiguous),
            )
        proposal_checked.append(True)
        return result

    def capture_chart(source: Any, construction: Any, query: Any = None) -> Any:
        witness = original_chart(source, construction, query)
        if query is not None and {key for key, _ in query.face_scope_keys} == {
            "physical-facet:wall:0:level:2:wall:1:level:2:wall:3:level:2",
            "physical-facet:wall:0:level:2:wall:2:level:2:wall:3:level:2",
        }:
            cap_queries.append(query)
        return witness

    def inspect_root(token: Any, *args: Any, **kwargs: Any) -> None:
        original_root_init(token, *args, **kwargs)
        assert checked and proposal_checked and token.mapped_chart_bank_equal
        source, specification, root = token.source, token.specification, token.root
        from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
        from phydrax.meshing._layer_source_association_replay import (
            replay_layer_source_associations,
        )

        cap_token = validate_source_fidelity_witness(
            token.mesh, token.geometry, cap_queries[-1], cap_witnesses[-1]
        )
        layer = replay_layer_source_associations(
            source, token.mesh, token.geometry, root, computed_root=token
        )
        assert (
            mapped_cap_edge_owner(source, token.mesh, token.geometry, layer, (cap_token,))
            is cap_witnesses[-1]
        )
        from fractions import Fraction

        from phydrax.meshing._association import (
            _prepare_layer_image_support,
            source_boundary_edge_intervals,
        )
        from phydrax.meshing._association_routes import incidence_pairs
        from phydrax.meshing._layer_core_association_transfer import _merge_face_tables
        from phydrax.meshing._layer_core_boundary_replay import (
            _validate_partial_edge_association_rows,
            core_adjacent_face_ids,
            core_cap_source_keys,
            replay_core_boundary_faces,
            replay_core_facet_edges,
        )

        primary = primary_witnesses[-1]
        primary_token = validate_source_fidelity_witness(
            token.mesh, token.geometry, source.fidelity_source, primary
        )
        core_ids = set(layer.core_cell_ids)
        primary_faces = frozenset(
            int(face)
            for cell, face in zip(
                primary.cell_global_ids, primary.entity_global_ids, strict=True
            )
            if int(cell) in core_ids
        )
        cap_faces = core_adjacent_face_ids(token.mesh, core_ids) & frozenset(
            int(face) for face in cap_witnesses[-1].entity_global_ids
        )
        face_tables = _merge_face_tables(
            token.mesh,
            (
                *replay_core_boundary_faces(
                    source, token.mesh, token.geometry, primary_token, primary_faces
                ),
                *replay_core_boundary_faces(
                    source, token.mesh, token.geometry, cap_token, cap_faces
                ),
            ),
        )
        preparation = _prepare_layer_image_support(
            source, token.mesh, token.geometry, layer.core_cell_ids
        )
        cap_owner = mapped_cap_edge_owner(
            source, token.mesh, token.geometry, layer, (cap_token,)
        )
        primary_edges = replay_core_facet_edges(
            source,
            token.mesh,
            token.geometry,
            primary,
            layer,
            face_tables,
            preparation=preparation,
            include_cap=False,
            cap_coverage_proved=True,
            cap_owner=cap_owner,
        )
        cap_edges = replay_core_facet_edges(
            source,
            token.mesh,
            token.geometry,
            cap_witnesses[-1],
            layer,
            face_tables,
            preparation=preparation,
            include_cap=False,
            cap_coverage_proved=True,
            cap_owner=cap_owner,
        )
        cap_keys = core_cap_source_keys(source)
        primary_raw = source_boundary_edge_intervals(token.mesh, primary)
        cap_raw = source_boundary_edge_intervals(token.mesh, cap_witnesses[-1])
        assert any(key in cap_raw and key[1:] in cap_keys for key in primary_raw)
        assert not any(
            (identity, path) in cap_keys
            for table in primary_edges
            for identity, path in zip(
                table.source_entity_ids, table.source_occurrence_paths, strict=True
            )
        )
        assert any(
            (identity, path) in cap_keys
            for table in cap_edges
            for identity, path in zip(
                table.source_entity_ids, table.source_occurrence_paths, strict=True
            )
        )
        facet_sides = {
            (identity, path): sides
            for table in face_tables
            for identity, path, sides in zip(
                table.source_entity_ids,
                table.source_occurrence_paths,
                table.source_region_sides,
                strict=True,
            )
        }
        edge_ids = np.asarray(token.mesh.entity_set(1).entity_ids, dtype=np.int64)
        cell_ids = np.asarray(token.mesh.entity_set(3).entity_ids, dtype=np.int64)
        core_edges = {
            int(edge_ids[edge])
            for edge, cell in incidence_pairs(token.mesh, 1, 3)
            if int(cell_ids[cell]) in core_ids
        }
        expected_noncap = {
            (edge, identity, path, interval, facet_sides[(identity, path)])
            for (edge, identity, path), intervals in primary_raw.items()
            if edge in core_edges
            and (identity, path) in facet_sides
            and (identity, path) not in cap_keys
            for interval in intervals
        }
        actual_noncap = {
            (int(edge), identity, path, interval or (Fraction(0), Fraction(1)), sides)
            for table in primary_edges
            for edge, identity, path, interval, sides in zip(
                table.target_global_ids,
                table.source_entity_ids,
                table.source_occurrence_paths,
                table.target_parameter_intervals,
                table.source_region_sides,
                strict=True,
            )
            if (identity, path) not in cap_keys
        }
        assert actual_noncap == expected_noncap
        for chart_token, rows in ((primary_token, primary_edges), (cap_token, cap_edges)):
            _validate_partial_edge_association_rows(
                token.mesh,
                token.geometry,
                chart_token,
                rows,
                source,
                (primary_token, cap_token),
                root,
                computed_root=token,
            )
        with pytest.raises(
            ValueError, match="actual layer/core interface|exactly the actual interface"
        ):
            mapped_cap_edge_owner(
                source,
                token.mesh,
                token.geometry,
                layer._replace(core_cell_ids=()),
                (cap_token,),
            )
        with pytest.raises(ValueError, match="different exact operands"):
            token.require(
                source,
                specification,
                token.mesh,
                CellGeometrySpec.affine(token.mesh),
                root,
            )
        from phydrax.meshing._layer_source_association_replay import (
            _conforming_current_rows,
        )

        # Cold restoration supplies equivalent, independently reconstructed
        # endpoint objects rather than the root's live object identities.
        identity_rows = tuple(
            _conforming_current_rows(root, copy(token.mesh), copy(token.geometry))
        )
        assert len(identity_rows) == sum(block.cell_count for block in token.mesh.blocks)
        assert all(
            cell == chart.target_cell_id == chart.source_cell_id
            and kind == chart.target_cell_kind == chart.source_cell_kind
            and chart.arguments is not None
            for cell, kind, chart in identity_rows
        )
        wrong_geometry = copy(token.geometry)
        wrong_coordinates = np.asarray(token.geometry.coordinates).copy()
        wrong_coordinates[0, 0] += 0.001
        object.__setattr__(wrong_geometry, "coordinates", wrong_coordinates)
        with pytest.raises(ValueError, match="full mesh or coordinate map"):
            tuple(_conforming_current_rows(root, copy(token.mesh), wrong_geometry))

        def full_proof_required(*args: Any, **kwargs: Any) -> None:
            raise _FullProofRequired

        with monkeypatch.context() as local:
            local.setattr(
                mapped_overlay, "_require_exact_reference_group", full_proof_required
            )
            local.setattr(
                mapped_replay, "_require_exact_reference_group", full_proof_required
            )
            charts = mapped_overlay.mapped_local_reference_maps(
                source, token.mesh, token.geometry, root, computed_root=token
            )
            assert charts
            associations = mapped_replay.replay_mapped_root_associations(
                source.mapped_domain,
                token.mesh,
                token.geometry,
                source=source,
                specification=specification,
                root=root,
                computed_root=token,
            )
            assert associations
            fallback = object.__new__(ComputedLayerCoreRoot)
            for name in ComputedLayerCoreRoot.__slots__:
                object.__setattr__(
                    fallback,
                    name,
                    False if name == "mapped_chart_bank_equal" else getattr(token, name),
                )
            with pytest.raises(_FullProofRequired):
                mapped_overlay.mapped_local_reference_maps(
                    source, token.mesh, token.geometry, root, computed_root=fallback
                )
            with pytest.raises(_FullProofRequired):
                mapped_replay.replay_mapped_root_associations(
                    source.mapped_domain,
                    token.mesh,
                    token.geometry,
                    source=source,
                    specification=specification,
                    root=root,
                    computed_root=fallback,
                )
        root_checked.append(True)
        raise _CapProofComplete

    def capture_native(*args: Any, **kwargs: Any) -> Any:
        curving = args[5]
        assert curving is not None and curving.chart_witness is not None
        primary_witnesses.append(curving.chart_witness)
        return original_native_construction(*args, **kwargs)

    monkeypatch.setattr(
        mapped_owner, "_propose_mapped_physical_improvement", inspect_proposal
    )
    monkeypatch.setattr(_native_layer, "_native_construction", capture_native)
    monkeypatch.setattr(ComputedLayerCoreRoot, "__init__", inspect_root)
    monkeypatch.setattr(
        _native_layer, "_mapped_layer_surface_chart_witness", capture_chart
    )

    monkeypatch.setattr(_native_layer, "_require_mapped_cap_interface", inspect_cap)
    with pytest.raises(_CapProofComplete):
        _run_wall_profile("curved-periodic-layer-core", mapped_archive_dir=tmp_path)
    assert checked
    assert proposal_checked
    assert root_checked
