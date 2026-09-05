import os

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
    assert result.derivative_mode is phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE
