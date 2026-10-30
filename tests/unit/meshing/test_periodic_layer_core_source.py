from __future__ import annotations

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
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
        column_index=layers.column_index,
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
        source_surface_charts=charts,
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
    assert periodic_source.source_surface_charts is charts
    charts.require_bound(wall, charts.geometry, boundary_source)
