import numpy as np
import pytest

import phydrax as phx


def _scope(dimension):
    return phx.meshing.MeshingScope(
        "geometry",
        "r1",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        f"geometry-{dimension}",
        np.asarray([1], dtype=np.int64),
    )


def test_size_controls_validate_physical_bounds():
    scope = _scope(2)
    control = phx.meshing.UniformSizeControl(
        scope,
        0.1,
        minimum_size=0.05,
        maximum_size=0.2,
        maximum_growth_rate=1.25,
    )
    assert control.minimum_size <= control.target_size <= control.maximum_size
    with pytest.raises(ValueError, match="minimum <= target <= maximum"):
        phx.meshing.UniformSizeControl(
            scope,
            0.1,
            minimum_size=0.2,
            maximum_size=0.3,
        )


def test_layer_and_periodic_controls_are_revision_bound():
    surface = _scope(2)
    volume = _scope(3)
    layer = phx.meshing.PrismLayerControl(
        surface,
        volume,
        5,
        1.0e-3,
        growth_rate=1.2,
    )
    transform = np.eye(4)
    transform[0, 3] = 1.0
    periodic = phx.meshing.PeriodicConstraint(surface, surface, transform)

    assert layer.layer_count == 5
    assert periodic.transform.shape == (4, 4)
    singular = np.eye(4)
    singular[0, 0] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        phx.meshing.PeriodicConstraint(surface, surface, singular)


def test_size_resolution_rejects_hard_conflicts_and_enforces_gradation():
    scope = phx.meshing.MeshingScope(
        "mesh",
        "r1",
        phx.meshing.MeshingEntityKind.MESH,
        0,
        "vertices",
        np.asarray((10, 20, 30), dtype=np.int64),
    )
    first = phx.meshing.UniformSizeControl(scope, 0.1, maximum_growth_rate=1.5)
    second = phx.meshing.UniformSizeControl(scope, 0.2)
    points = np.asarray(((0.0,), (1.0,), (2.0,)))
    with pytest.raises(ValueError, match="hard size controls conflict"):
        phx.meshing.resolve_size_controls(
            (first, second),
            points,
            scope.entity_ids,
            phx.meshing.SizeFieldDomain.MESH_GEODESIC,
        )

    field, report = phx.meshing.resolve_size_controls(
        (first,),
        points,
        scope.entity_ids,
        phx.meshing.SizeFieldDomain.MESH_GEODESIC,
        adjacency=np.asarray(((0, 1), (1, 2)), dtype=np.int32),
    )
    assert report.field_id == field.field_id
    assert np.all(np.asarray(field.values) > 0.0)


def test_metric_normalization_clamps_size_and_anisotropy():
    scope = phx.meshing.MeshingScope(
        "mesh",
        "r1",
        phx.meshing.MeshingEntityKind.MESH,
        0,
        "vertices",
        np.asarray((1, 2), dtype=np.int64),
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.asarray((np.diag((1.0, 10_000.0)), np.diag((0.01, 1.0)))),
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_anisotropy=4.0,
    )
    normalized = phx.meshing.normalize_mesh_metric(
        metric,
        adjacency=np.asarray(((0, 1),), dtype=np.int32),
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(normalized.values))
    assert np.all(eigenvalues >= 1.0 / 2.0**2 - 1.0e-12)
    assert np.all(eigenvalues <= 1.0 / 0.1**2 + 1.0e-12)
    assert np.all(np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0]) <= 4.0 + 1.0e-12)
