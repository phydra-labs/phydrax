import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.meshing import MeshingDerivativeMode
from phydrax.meshing.providers._poisson import (
    OrientedPointCloud,
    PoissonProvider,
    PoissonReconstructionSpec,
)


def test_poisson_reconstructs_translated_sphere_without_sample_lineage():
    pytest.importorskip("open3d")
    count = 400
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    angle = np.arange(count) * np.pi * (3.0 - np.sqrt(5.0))
    normals = np.column_stack(
        (
            np.sqrt(1.0 - z * z) * np.cos(angle),
            np.sqrt(1.0 - z * z) * np.sin(angle),
            z,
        )
    )
    center = np.asarray((12.0, -4.0, 8.0))
    radius = 2.0
    contract = SpatialCoordinateContract.si()
    source = OrientedPointCloud(
        center + radius * normals,
        normals,
        contract,
        source_id="sphere-samples",
        source_revision="r1",
    )
    result = PoissonProvider().execute(source, PoissonReconstructionSpec(depth=5))

    radii = np.linalg.norm(np.asarray(result.mesh.coordinates) - center, axis=1)
    np.testing.assert_allclose(radii, radius, atol=0.2, rtol=0)
    assert result.audit.passed and result.compliance.passed
    assert result.coordinate_contract.spatial_id == contract.spatial_id
    assert result.derivative_mode is MeshingDerivativeMode.NONDIFFERENTIABLE
    assert result.boundary.metadata.source_id != source.source_id
    assert result.associations == ()
    assert result.labels == ()


def test_poisson_requires_nonzero_supplied_normals():
    with pytest.raises(ValueError, match="nonzero"):
        OrientedPointCloud(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
            np.zeros((4, 3)),
            SpatialCoordinateContract.si(),
            source_id="unoriented",
            source_revision="r1",
        )
