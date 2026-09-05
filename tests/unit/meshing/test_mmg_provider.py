from importlib.util import find_spec

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing.providers._mmg import MmgOptions, MmgProvider


pytestmark = pytest.mark.meshing_mmg


def _metric(provider, source, matrices):
    return phx.meshing.MeshMetricField(
        provider.vertex_scope(source),
        matrices,
        minimum_size=0.1,
        maximum_size=1.0,
        maximum_anisotropy=10.0,
    )


@pytest.mark.parametrize("kind", ("planar", "surface", "volume"))
def test_native_mmg_routes_preserve_domain_measure_without_inventing_ids(kind):
    if find_spec("mmg3d") is None:
        pytest.skip("optional pymmg binary wheel is not installed")
    provider = MmgProvider(MmgOptions(hausdorff_distance=0.005))
    if kind == "planar":
        source = phx.discretization.CellMesh.from_triangles(
            np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            np.array(((0, 1, 2), (0, 2, 3))),
            vertex_global_ids=np.array((90, 7, 52, 11)),
            cell_global_ids=np.array((102, 55)),
        )
        expected = 1.0
    else:
        points = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        if kind == "surface":
            source = phx.discretization.CellMesh.from_triangles(
                points,
                np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))),
            )
            expected = 1.5 + np.sqrt(3.0) / 2
        else:
            source = phx.discretization.CellMesh.from_tetrahedra(
                points, np.array(((0, 1, 2, 3),))
            )
            expected = 1 / 6
    metric = phx.meshing.MeshMetricField(
        provider.vertex_scope(source),
        np.tile(np.eye(source.ambient_dimension) * 16.0, (len(source.coordinates), 1, 1)),
        minimum_size=0.25,
        maximum_size=0.25,
    )
    result = provider.adapt(
        source, metric, phx.SpatialCoordinateContract(phx.units.METER)
    )
    cells = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    if kind == "volume":
        signed = np.linalg.det(cells[:, 1:] - cells[:, :1]) / 6
        assert np.all(signed > 0)
        measure = signed.sum()
    elif kind == "planar":
        measure = np.linalg.det(cells[:, 1:] - cells[:, :1]).sum() / 2
    else:
        measure = (
            np.linalg.norm(
                np.cross(cells[:, 1] - cells[:, 0], cells[:, 2] - cells[:, 0]), axis=1
            ).sum()
            / 2
        )
    assert measure == pytest.approx(expected, rel=0.02)
    assert result.audit.passed and result.compliance.passed
    assert not np.intersect1d(
        result.mesh.vertex_global_ids, source.vertex_global_ids
    ).size
    assert not np.intersect1d(
        result.mesh.blocks[0].global_ids, source.blocks[0].global_ids
    ).size
    assert result.derivative_mode is phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE
    assert phx.meshing.MeshingCapability.LINEAGE not in result.provider.capabilities
    assert result.adapter_reports[0].losses


def test_mmg_metric_conversion_binds_sorted_global_ids_and_rotated_anisotropy(tmp_path):
    if find_spec("mmg2d") is None:
        pytest.skip("optional pymmg binary wheel is not installed")
    provider = MmgProvider()
    source = phx.discretization.CellMesh.from_triangles(
        np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.array(((0, 1, 2), (0, 2, 3))),
        vertex_global_ids=np.array((90, 7, 52, 11)),
    )
    rotation = np.array(((0.8, -0.6), (0.6, 0.8)))
    tensor = rotation @ np.diag((9.0, 64.0)) @ rotation.T
    metric = _metric(provider, source, np.tile(tensor, (4, 1, 1)))
    result = provider.adapt(
        source, metric, phx.SpatialCoordinateContract(phx.units.METER)
    )
    points = np.asarray(result.mesh.coordinates)
    edges = np.asarray(result.mesh.connectivity.edges)
    displacement = points[edges[:, 1]] - points[edges[:, 0]]
    normalized_lengths = np.sqrt(
        np.einsum("ni,ij,nj->n", displacement, tensor, displacement)
    )
    # Dropping the off-diagonal entry or swapping tensor coefficients changes
    # actual physical edge lengths in this rotated metric.
    assert np.quantile(normalized_lengths, 0.95) < 2.0
    assert np.median(normalized_lengths) > 0.5
    varying = _metric(
        provider,
        source,
        np.array([np.eye(2) * value for value in (4.0, 9.0, 16.0, 25.0)]),
    )
    from phydrax.meshing.providers._mmg import _metric_rows, _write_metric

    exported = tmp_path / "metric.sol"
    _write_metric(exported, _metric_rows(source, varying))
    rows = np.loadtxt(exported, skiprows=5, max_rows=4)
    np.testing.assert_array_equal(rows[:, 0], (25.0, 4.0, 16.0, 9.0))
    np.testing.assert_array_equal(rows[:, 2], rows[:, 0])


def test_mmg_refuses_a_metric_from_another_numeric_mesh_revision():
    provider = MmgProvider()
    source = phx.discretization.CellMesh.from_triangles(
        np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.array(((0, 1, 2),)),
        numeric_version="source",
    )
    other = phx.discretization.CellMesh.from_triangles(
        np.asarray(source.coordinates),
        np.asarray(source.blocks[0].vertices),
        numeric_version="other",
    )
    metric = _metric(provider, source, np.tile(np.eye(2) * 16.0, (3, 1, 1)))
    with pytest.raises(phx.meshing.MeshingFailure) as caught:
        provider.plan(other, metric, phx.SpatialCoordinateContract(phx.units.METER))
    assert caught.value.category is phx.meshing.MeshingFailureCategory.INVALID_SOURCE
