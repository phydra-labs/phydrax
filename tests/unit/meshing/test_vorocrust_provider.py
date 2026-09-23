import os
import sys

import manifold3d
import numpy as np
import pytest

import phydrax as phx


@pytest.mark.meshing_vorocrust
def test_real_vorocrust_preserves_closed_cube_volume():
    executable = os.environ.get("PHYDRAX_VOROCRUST_EXECUTABLE")
    extractor = os.environ.get("PHYDRAX_VOROCRUST_EXTRACTOR")
    if executable is None or extractor is None:
        pytest.skip(
            "Set VoroCrust mesher and extraction bridge paths for real qualification."
        )
    arrays = manifold3d.Manifold.cube().to_mesh64()
    source = phx.geometry.SurfaceModel.from_triangles(
        arrays.vert_properties[:, :3],
        arrays.tri_verts,
        phx.geometry.SurfaceMetadata(
            source_id="cube",
            source_revision="0",
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            provenance=("qualification",),
        ),
    )
    result = phx.meshing.VoroCrustProvider(executable, extractor).execute(
        source,
        phx.meshing.VoroCrustOptions(1.0),
        limits=phx.meshing.MeshingLimits(maximum_wall_seconds=120.0),
    )
    assert all(
        isinstance(block, phx.discretization.PolyhedralBlock)
        for block in result.mesh.blocks
    )
    assert float(np.sum(np.asarray(result.quality.evaluation.measures))) == pytest.approx(
        1.0, rel=1e-6
    )
    assert result.audit.passed
    assert result.compliance.passed
    assert "native_output_entities_preallocation" in result.runtime.unenforced_limits
    assert "native_output_connectivity_preallocation" in result.runtime.unenforced_limits
    assert result.derivative_mode is phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE
    coordinates = phx.interchange.GeospatialContract.local_cartesian(
        result.coordinate_contract,
        vertical_datum="local-survey-datum",
    )
    qualified = phx.applications.porous_media.qualify_vorocrust_porous_mesh(
        result,
        coordinates,
        surface_faces=np.flatnonzero(np.asarray(result.mesh.connectivity.boundary_faces)),
    )
    assert qualified.surface_trace is not None
    assert not qualified.generator_dual_available
    assert not qualified.tpfa_certified
    with pytest.raises(ValueError, match="does not certify TPFA"):
        qualified.require_tpfa()


def test_vorocrust_porous_qualification_certifies_consumed_geometry_not_tpfa():
    coordinates = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype="float64",
    )
    cells = (
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
    )
    base = phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_polyhedra(coordinates, cells),
        phx.SpatialCoordinateContract.si(),
    )
    provider = phx.meshing.MeshingProviderInfo(
        "vorocrust",
        "qualification-fixture",
        "BSD-3-Clause",
        operations=(phx.meshing.MeshingOperation.MESH_VOLUME,),
        source_kinds=(phx.meshing.MeshingSourceKind.SURFACE,),
        capabilities=(phx.meshing.MeshingCapability.POLYHEDRAL,),
        cell_kinds=("polyhedron",),
        dimensions=(3,),
        execution_modes=(phx.meshing.MeshingExecutionMode.SUBPROCESS,),
    )
    runtime = phx.meshing.MeshingRuntimeInfo(
        provider.provider_id,
        "qualification-fixture",
        phx.meshing.MeshingExecutionMode.SUBPROCESS,
        deterministic=False,
    )
    result = phx.meshing.CellMeshingResult(
        base.mesh,
        base.geometry,
        base.coordinate_contract,
        base.audit,
        base.quality,
        base.compliance,
        base.trace,
        provider,
        runtime,
        phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE,
        base.provenance,
    )
    geospatial = phx.interchange.GeospatialContract.local_cartesian(
        result.coordinate_contract,
        vertical_datum="local-survey-datum",
    )
    boundary = np.flatnonzero(np.asarray(result.mesh.connectivity.boundary_faces))
    qualified = phx.applications.porous_media.qualify_vorocrust_porous_mesh(
        result,
        geospatial,
        surface_faces=boundary,
    )
    assert qualified.discretization.geometry_id
    assert qualified.require_surface_trace().parent_faces.size == 6
    assert not qualified.generator_dual_available
    assert not qualified.tpfa_certified
    with pytest.raises(ValueError, match="does not certify TPFA"):
        qualified.require_tpfa()


def test_vorocrust_version_probe_maps_nonzero_exit_to_meshing_failure(tmp_path):
    extractor = tmp_path / "extractor"
    extractor.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    extractor.chmod(0o755)
    provider = phx.meshing.VoroCrustProvider(sys.executable, extractor)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.info()

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
    )


def test_vorocrust_version_probe_requires_protocol_and_binary_identities(tmp_path):
    extractor = tmp_path / "extractor"
    digest = "a" * 64
    extractor.write_text(
        "#!/bin/sh\n"
        f"echo 'phydrax-vorocrust/1 vorocrust/r1 source-sha256/{digest} "
        f"config-sha256/{digest} library-sha256/{digest}'\n",
        encoding="utf-8",
    )
    extractor.chmod(0o755)

    info = phx.meshing.VoroCrustProvider(sys.executable, extractor).info()

    assert info.version == "r1"


def test_vorocrust_version_probe_bounds_output_before_decoding(tmp_path):
    extractor = tmp_path / "extractor"
    extractor.write_text(
        f"#!{sys.executable}\nprint('x' * 20000)\n",
        encoding="utf-8",
    )
    extractor.chmod(0o755)
    provider = phx.meshing.VoroCrustProvider(sys.executable, extractor)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.info()

    assert failure.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_vorocrust_fails_closed_when_native_preallocation_is_required(tmp_path):
    extractor = tmp_path / "extractor"
    extractor.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    extractor.chmod(0o755)
    arrays = manifold3d.Manifold.cube().to_mesh64()
    source = phx.geometry.SurfaceModel.from_triangles(
        arrays.vert_properties[:, :3],
        arrays.tri_verts,
        phx.geometry.SurfaceMetadata(
            source_id="cube",
            source_revision="0",
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            provenance=("qualification",),
        ),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.VoroCrustProvider(sys.executable, extractor).execute(
            source,
            phx.meshing.VoroCrustOptions(
                1.0,
                require_native_output_preallocation=True,
            ),
        )

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    )
