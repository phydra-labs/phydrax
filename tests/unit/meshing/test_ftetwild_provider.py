from importlib.util import find_spec

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing.providers._ftetwild import FTetWildOptions, FTetWildProvider


pytestmark = pytest.mark.meshing_ftetwild


def _source(scale=1.0, *, triangle_soup=False):
    points = scale * np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
    if triangle_soup:
        points = points[faces].reshape((-1, 3))
        faces = np.arange(12).reshape((4, 3))
    return phx.geometry.surface.SurfaceModel.from_triangles(
        points,
        faces,
        phx.geometry.surface.SurfaceMetadata(
            source_id="tetrahedral-surface",
            source_revision=f"scale:{scale}",
            coordinate_contract=phx.SpatialCoordinateContract(phx.units.METER),
            provenance=("qualification",),
        ),
    )


def _specification(provider, source, size, *, hard=False):
    scope = provider.whole_scope(source)
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                strength=phx.meshing.SizeControlStrength.HARD
                if hard
                else phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        deterministic=False,
    )


@pytest.mark.parametrize("scale,triangle_soup", ((1.0, False), (0.001, True)))
def test_native_ftetwild_filters_exterior_and_handles_duplicate_surface_seams(
    scale, triangle_soup
):
    if find_spec("wildmeshing") is None:
        pytest.skip("optional wildmeshing binding is not installed")
    source = _source(scale, triangle_soup=triangle_soup)
    provider = FTetWildProvider(
        FTetWildOptions(
            envelope_distance=0.005 * scale,
            maximum_iterations=20,
        )
    )
    result = provider.plan(
        source, _specification(provider, source, 0.3 * scale)
    ).execute()
    points = np.asarray(result.mesh.coordinates)
    cells = points[np.asarray(result.mesh.blocks[0].vertices)]
    volumes = np.linalg.det(cells[:, 1:] - cells[:, :1]) / 6
    assert np.all(volumes > 0)
    assert volumes.sum() == pytest.approx(scale**3 / 6, rel=0.03)
    centroids = cells.mean(axis=1)
    assert np.all(centroids >= -0.005 * scale)
    assert np.all(centroids.sum(axis=1) <= 1.015 * scale)
    achieved = dict(result.compliance.achieved)
    assert achieved["maximum_sampled_boundary_deviation"] <= 0.005 * scale
    assert result.boundary is not None
    assert not np.any(np.asarray(result.boundary.mesh.connectivity.boundary_edges))
    assert not result.associations[0].exact
    assert not result.associations[0].complete
    assert result.audit.passed and result.compliance.passed
    assert not np.intersect1d(
        result.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    ).size
    assert not np.intersect1d(
        result.mesh.blocks[0].global_ids, source.mesh.blocks[0].global_ids
    ).size
    assert phx.meshing.MeshingCapability.LINEAGE not in result.provider.capabilities
    assert result.derivative_mode is phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE


def test_ftetwild_rejects_hard_sizing_without_silently_weakening_it():
    source = _source()
    provider = FTetWildProvider()
    with pytest.raises(phx.meshing.MeshingFailure) as caught:
        provider.plan(source, _specification(provider, source, 0.3, hard=True))
    assert (
        caught.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )


def test_ftetwild_rejects_a_scope_bound_to_another_surface():
    source, other = _source(), _source(2.0)
    provider = FTetWildProvider()
    with pytest.raises(phx.meshing.MeshingFailure) as caught:
        provider.plan(other, _specification(provider, source, 0.3))
    assert (
        caught.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
