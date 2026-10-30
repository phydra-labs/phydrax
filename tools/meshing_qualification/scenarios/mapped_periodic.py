from __future__ import annotations

from itertools import permutations

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._periodic import PeriodicPointOrbits, periodic_delaunay_mesh
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeMappedHexSource, NativePlcSource
from phydrax.meshing._plc_source import PiecewiseLinearComplex

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


def _mapped_cube(coordinate_contract: SpatialCoordinateContract, /) -> tuple[NativeMappedHexSource, MappedReferenceDomain]:
    source_id = "qualification:mapped-reference-cube"
    revision = "revision-1"
    vertices = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    triangle_rows = (
        (0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    segments = np.asarray(
        tuple(tuple(sorted(edge)) for edge in reference_cell_topology("hexahedron").entities[1]),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(face, dtype=np.int64) for face in triangle_rows),
        np.arange(len(triangle_rows), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(triangle_rows), 1)),
        ("solid",),
        segments=segments,
        vertex_ids=tuple(f"cube-vertex-{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"cube-facet-{row}" for row in range(len(triangle_rows))),
        facet_occurrence_paths=((),) * len(triangle_rows),
        segment_ids=tuple(f"cube-edge-{row}" for row in range(segments.shape[0])),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
    )
    reference_domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(triangle_rows, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(triangle_rows), 1)),
        ("solid",),
        source_id=source_id,
    )
    reference_mesh = CellMesh(
        vertices,
        (CellBlock("root", "hexahedron", np.arange(8, dtype=np.int32)[None, :]),),
    )
    base_geometry = CellGeometrySpec.affine(reference_mesh)
    translation = np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
    scale = 2.0
    mapped_coordinates = scale * vertices + translation
    source_geometry = CellGeometrySpec(
        dict(zip(base_geometry.block_names, base_geometry.elements, strict=True)),
        {name: np.asarray(route) for name, route in zip(base_geometry.block_names, base_geometry.geometry_dofs, strict=True)},
        mapped_coordinates,
    )
    domain = MappedReferenceDomain(
        reference_domain, reference_mesh, source_geometry, np.asarray((0,), dtype=np.int64),
        source_id=source_id, source_revision=revision,
        coordinate_contract=coordinate_contract,
    )
    return NativeMappedHexSource(NativePlcSource(complex_, source_id, revision), domain), domain


def native_mapped() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    source, domain = _mapped_cube(contract)
    scope = MeshingScope(
        source.source_id, source.source_revision, MeshingEntityKind.GEOMETRY, 2,
        source.scope_entity_set_id(2), source.scope_entity_ids(2),
    )
    translation = np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
    scale = 2.0
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("hexahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 2.0, strength=SizeControlStrength.SOFT),),
        region_seeds=(
            RegionSeed(translation + np.full((3,), scale * 0.5), "solid", "mapped-material", RegionRole.SOLID),
        ),
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions("mapped_balanced_grid_hex")
    ).plan(source, specification, coordinate_contract=contract)
    result = plan.execute()
    accepted = accepted_result_record(result)
    if any(block.cell_kind != "hexahedron" for block in result.mesh.blocks):
        raise RuntimeError("Mapped route violated the required hexahedron family.")
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    np.testing.assert_allclose(np.min(points, axis=0), translation, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.max(points, axis=0), translation + scale, rtol=0.0, atol=1.0e-12)
    volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    if not np.isclose(volume, scale**3, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(f"Mapped affine source volume is {volume}, expected {scale**3}.")
    if not result.associations or any(
        association.source_id != source.source_id
        or association.source_revision != source.source_revision
        for association in result.associations
    ):
        raise RuntimeError("Mapped route discarded its revision-bound source associations.")
    if not result.geometry.restriction_source:
        raise RuntimeError("Mapped route omitted its retained reference-cell coordinate map.")
    return {
        "status": "passed",
        "route": plan.route,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "mapped_domain_identity": domain.domain_id,
        "mapped_volume": volume,
        "exact_affine_volume": scale**3,
        "source_bounds": [translation.tolist(), (translation + scale).tolist()],
        "accepted": accepted,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }


def native_periodic() -> dict[str, object]:
    meshcore = require_meshcore()
    cell = PeriodicCell(np.asarray(((1.2, 0.1), (0.35, 1.05)), dtype=np.float64))
    fractional = np.asarray(((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)), dtype=np.float64)
    points = np.asarray(cell.cartesian(fractional), dtype=np.float64)
    point_ids = np.asarray((91, 4, 53, 17), dtype=np.int64)
    orbit_source = PeriodicPointOrbits(cell, points, point_ids)
    construction = periodic_delaunay_mesh(orbit_source)
    permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
    reordered = periodic_delaunay_mesh(
        PeriodicPointOrbits(cell, points[permutation], point_ids[permutation])
    )
    periodic = construction.mesh.periodic_topology
    if periodic is None:
        raise RuntimeError("Periodic Delaunay did not construct quotient topology.")
    block = construction.mesh.blocks[0]
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    triangles = coordinates[np.asarray(block.vertices, dtype=np.int64)]
    signed_areas = 0.5 * (
        (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    area = float(np.sum(signed_areas))
    if block.cell_kind != "triangle" or np.any(signed_areas <= 0.0):
        raise RuntimeError("Periodic Delaunay did not publish positively oriented triangles.")
    if not np.isclose(area, cell.cell_measure, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(f"Periodic quotient area {area} differs from lattice area {cell.cell_measure}.")
    if periodic.vertex_orbit_count != point_ids.size:
        raise RuntimeError("Periodic quotient does not preserve each authored site orbit.")
    if construction.mesh.mesh_id != reordered.mesh.mesh_id:
        raise RuntimeError("Periodic Delaunay identity depends on authored site order.")
    return {
        "status": "passed",
        "route": "periodic_delaunay",
        "construction_id": construction.construction_id,
        "mesh_id": construction.mesh.mesh_id,
        "periodic_topology_id": periodic.periodic_topology_id,
        "quotient_vertex_orbits": periodic.vertex_orbit_count,
        "input_point_ids": point_ids.tolist(),
        "signed_area_sum": area,
        "analytic_lattice_area": cell.cell_measure,
        "positive_triangle_count": int(signed_areas.size),
        "permutation_identity_preserved": True,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
