from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellBlock, CellMesh, PolyhedralConnectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._mapped_power_geometry import MappedPowerCellGeometrySource
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.discretization.finite_volume._polyhedral import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    FeatureKind,
    MeshingEntityKind,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePeriodicSource,
    PeriodicPointOrbits,
    ProtectedFeature,
    RegionRole,
    RegionSeed,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._model_codec import (
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_generation import NativePolyhedralSchedule
from phydrax.meshing.providers._native_sources import (
    NativePlcSource,
    NativePolyhedralSource,
)


def _mapped_stretched_polyhedral_source() -> tuple[
    NativePolyhedralSource, VolumeMeshingSpec
]:
    source_id = "mapped-stretched-tetra"
    source_revision = "r1"
    topology = reference_cell_topology("tetrahedron")
    reference = np.asarray(topology.vertices, dtype=np.float64)
    root_vertex_ids = np.asarray((31, 37, 41, 43), dtype=np.int64)
    root_cell_ids = np.asarray((901,), dtype=np.int64)
    faces = tuple(
        np.asarray(face, dtype=np.int64)
        for face in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    )
    reference_domain = PiecewiseLinearDomain(
        reference,
        np.asarray(faces, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(faces), 1)),
        ("bulk",),
        source_id=source_id,
    )
    root_mesh = CellMesh(
        reference,
        (
            CellBlock(
                "root",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=root_cell_ids,
            ),
        ),
        vertex_global_ids=root_vertex_ids,
    )
    element = lagrange_element("tetrahedron", 1)
    physical_coefficients = np.array(reference, copy=True)
    physical_coefficients[:, 0] *= 2.0
    geometry = CellGeometrySpec(
        {"root": element},
        {"root": np.arange(4, dtype=np.int64)[None, :]},
        physical_coefficients,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        geometry,
        np.asarray((0,), dtype=np.int64),
        source_id=source_id,
        source_revision=source_revision,
        reference_domain_vertex_root_ids=root_vertex_ids,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    segments = np.asarray(
        tuple(
            tuple(sorted(int(value) for value in edge)) for edge in topology.entities[1]
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        reference,
        faces,
        np.arange(len(faces), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(faces), 1)),
        ("bulk",),
        segments=segments,
        vertex_ids=tuple(f"mapped-vertex-{row}" for row in range(4)),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=tuple(f"mapped-facet-{row}" for row in range(len(faces))),
        facet_occurrence_paths=((),) * len(faces),
        segment_ids=tuple(f"mapped-edge-{row}" for row in range(segments.shape[0])),
        segment_occurrence_paths=((),) * segments.shape[0],
        region_occurrence_paths=((),),
        boundary="fixed",
        geometry=geometry,
    )
    plc = NativePlcSource(
        complex_,
        source_id,
        source_revision,
        mapped_domain=domain,
        maximum_deviation=0.0,
        plc_vertex_root_global_ids=root_vertex_ids,
    )
    source = NativePolyhedralSource(
        complex_,
        source_id,
        source_revision,
        sites=np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64),
        weights=np.zeros((1,), dtype=np.float64),
        site_ids=("bulk-site",),
        mapped_domain=domain,
        fidelity_source=plc.boundary_query(),
        maximum_deviation=0.0,
    )
    boundary_scope = MeshingScope.whole(source, 2)
    protected = ProtectedFeature(
        MeshingScope(
            source_id,
            source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            np.asarray((source.scope_entity_ids(2)[0],), dtype=np.int64),
        ),
        FeatureKind.SURFACE,
        maximum_deviation=0.0,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        boundary_scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
        protected_features=(protected,),
        region_seeds=(
            RegionSeed(
                np.asarray((0.25, 0.25, 0.25), dtype=np.float64),
                "bulk",
                "bulk-material",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.0,
    )
    return source, specification


def _independent_polyhedral_volumes(mesh: CellMesh, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Independent volume oracle requires polyhedral connectivity.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    result = np.zeros((connectivity.cell_count,), dtype=np.float64)
    for cell in range(connectivity.cell_count):
        first, last = map(int, cell_offsets[cell : cell + 2])
        for face, sign in zip(
            cell_faces[first:last], cell_signs[first:last], strict=True
        ):
            face_first, face_last = map(int, face_offsets[int(face) : int(face) + 2])
            polygon = points[face_vertices[face_first:face_last]]
            for row in range(1, polygon.shape[0] - 1):
                result[cell] += (
                    sign
                    * np.dot(polygon[0], np.cross(polygon[row], polygon[row + 1]))
                    / 6.0
                )
    return np.abs(result)


def test_mapped_stretched_polyhedral_provider_preserves_features_and_fv_measure() -> None:
    source, specification = _mapped_stretched_polyhedral_source()
    result = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "plc_restricted_power",
                polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
            )
        )
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )

    geometry = prepare_polyhedral_finite_volume_geometry(result.mesh)
    independent = _independent_polyhedral_volumes(result.mesh)

    assert result.certification is not None and result.certification.passed
    assert {block.cell_kind for block in result.mesh.blocks} == {"polyhedron"}
    np.testing.assert_allclose(geometry.cell_volumes, independent, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(independent), 1.0 / 3.0, rtol=0.0, atol=1e-12)


def test_mapped_power_geometry_source_round_trips_through_the_model_archive(
    tmp_path: Path,
) -> None:
    source, specification = _mapped_stretched_polyhedral_source()
    result = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "plc_restricted_power",
                polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
            )
        )
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    source_map = result.geometry.source_map
    assert isinstance(source_map, MappedPowerCellGeometrySource)

    path = write_model_artifact(tmp_path / "mapped-power.phx", source_map, limits=None)
    restored = read_model_artifact(
        path,
        expected_type=MappedPowerCellGeometrySource,
        restore=restore_model_archive_value,
        limits=None,
    )

    assert type(restored) is MappedPowerCellGeometrySource
    assert restored is not source_map
    assert restored.source_map_id == source_map.source_map_id
    assert restored.exact_vertices == source_map.exact_vertices
    assert restored.cell_piece_tetrahedra() == source_map.cell_piece_tetrahedra()
    np.testing.assert_array_equal(
        np.asarray(restored.runtime_vertices), np.asarray(source_map.runtime_vertices)
    )

    first = source_map.exact_vertices[0]
    tampered = copy.copy(source_map)
    object.__setattr__(
        tampered,
        "exact_vertices",
        (
            (first[0] + Fraction(1, 7), first[1], first[2]),
            *source_map.exact_vertices[1:],
        ),
    )
    with pytest.raises(ValueError, match="differs from its renewed root-chart images"):
        MappedPowerCellGeometrySource.from_stored_fields(tampered)


def test_periodic_polyhedral_source_reaches_fv_measure_consumer() -> None:
    cell = PeriodicCell(np.eye(3, dtype=np.float64))
    orbits = PeriodicPointOrbits(
        cell,
        np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64),
        np.asarray((701,), dtype=np.int64),
    )
    source = NativePeriodicSource(orbits, "one-periodic-site", "r1")
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
    )
    result = (
        NativeMeshingProvider(NativeMeshingOptions("periodic_power"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )

    geometry = prepare_polyhedral_finite_volume_geometry(result.mesh)
    independent = _independent_polyhedral_volumes(result.mesh)

    assert result.mesh.periodic_topology is not None
    np.testing.assert_allclose(geometry.cell_volumes, independent, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(independent), abs(np.linalg.det(cell.vectors)), rtol=0.0, atol=1e-12
    )
