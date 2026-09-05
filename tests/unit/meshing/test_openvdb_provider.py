import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.discretization.spatial import (
    MortonAddressPlan,
    SparseVoxelField,
    SparseVoxelGridPlan,
)
from phydrax.meshing import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers._openvdb import OpenVDBMeshingSpec, OpenVDBProvider


def _ellipsoid_field(*, background_mode="constant"):
    lower = np.asarray((10.0, -6.0, 20.0))
    upper = np.asarray((14.0, 2.0, 32.0))
    center = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    indices = np.stack(np.meshgrid(*([np.arange(16)] * 3), indexing="ij"), axis=-1)
    indices = indices.reshape(-1, 3)
    normalized = (indices + 0.5) / 8.0 - 1.0
    indices = indices[np.linalg.norm(normalized, axis=1) < 0.9]
    grid = SparseVoxelGridPlan(
        MortonAddressPlan(lower, upper, 4),
        brick_size=4,
        brick_capacity=64,
    ).prepare(indices)
    values = (
        np.linalg.norm(
            (np.asarray(grid.voxel_centers()) - center) / half_width,
            axis=-1,
        )
        - 0.55
    )
    # Inactive storage is not source data and must never leak into the VDB tree.
    values[~np.asarray(grid.voxel_active)] = np.nan
    field = SparseVoxelField(
        grid,
        values,
        background_mode=background_mode,
        background_value=1.0,
    )
    return field, center, half_width


def test_openvdb_extracts_sparse_anisotropic_cell_centered_isosurface():
    pytest.importorskip("openvdb")
    field, center, half_width = _ellipsoid_field()
    contract = SpatialCoordinateContract.si()
    result = OpenVDBProvider().execute(
        field,
        contract,
        OpenVDBMeshingSpec(isovalue=0.1),
        source_id="sampled-ellipsoid",
        source_revision="r1",
    )

    radius = np.linalg.norm(
        (np.asarray(result.mesh.coordinates) - center) / half_width,
        axis=1,
    )
    np.testing.assert_allclose(radius, 0.65, atol=0.04, rtol=0)
    assert result.audit.passed and result.compliance.passed
    assert result.coordinate_contract.spatial_id == contract.spatial_id
    assert result.boundary.metadata.source_id != "sampled-ellipsoid"
    assert result.associations == ()
    assert result.labels == ()


def test_openvdb_rejects_unknown_background_instead_of_inventing_exterior():
    field, _, _ = _ellipsoid_field(background_mode="unsupported")
    with pytest.raises(MeshingFailure) as caught:
        OpenVDBProvider().execute(
            field,
            SpatialCoordinateContract.si(),
            OpenVDBMeshingSpec(),
            source_id="unknown-exterior",
            source_revision="r1",
        )
    assert caught.value.category is MeshingFailureCategory.UNSUPPORTED_CAPABILITY


def test_openvdb_rejects_empty_isosurface_without_substituting_geometry():
    pytest.importorskip("openvdb")
    field, _, _ = _ellipsoid_field()
    with pytest.raises(MeshingFailure) as caught:
        OpenVDBProvider().execute(
            field,
            SpatialCoordinateContract.si(),
            OpenVDBMeshingSpec(isovalue=2.0),
            source_id="sampled-ellipsoid",
            source_revision="r1",
        )
    assert caught.value.category is MeshingFailureCategory.CONVERSION_FAILED
