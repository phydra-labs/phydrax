import numpy as np
import pytest

import phydrax as phx


def _scope(dimension, ids=(1,)):
    return phx.meshing.MeshingScope(
        "geometry",
        "r1",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        f"geometry-{dimension}",
        np.asarray(ids, dtype=np.int64),
    )


def _mesh_scope(ids):
    return phx.meshing.MeshingScope(
        "mesh",
        "r1",
        phx.meshing.MeshingEntityKind.MESH,
        0,
        "vertices",
        np.asarray(ids, dtype=np.int64),
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
    with pytest.raises(ValueError):
        phx.meshing.UniformSizeControl(
            scope,
            0.1,
            minimum_size=0.2,
            maximum_size=0.3,
        )


def test_layer_and_periodic_controls_are_revision_bound():
    surface = _scope(2)
    target_surface = _scope(2, ids=(2,))
    volume = _scope(3)
    schedule = phx.meshing.LayerSchedule.geometric(5, 1.0e-3, growth_rate=1.2)
    layer = phx.meshing.BoundaryLayerControl(
        surface,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=volume,
        cap_scope=target_surface,
    )
    transform = np.eye(4)
    transform[0, 3] = 1.0
    periodic = phx.meshing.PeriodicConstraint(surface, surface, transform)

    assert len(layer.schedule.thicknesses) == 5
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
    with pytest.raises(ValueError):
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
    np.testing.assert_array_equal(field.sample_entity_ids, scope.entity_ids)


def test_hard_growth_limits_bound_sizes_by_edge_length():
    vertices = _mesh_scope((10, 20, 30))
    refined = _mesh_scope((10,))
    coarse = phx.meshing.UniformSizeControl(vertices, 1.0, maximum_growth_rate=1.2)
    fine = phx.meshing.UniformSizeControl(refined, 0.1, priority=1)

    field, report = phx.meshing.resolve_size_controls(
        (coarse, fine),
        np.asarray(((0.0,), (1.0,), (3.0,))),
        vertices.entity_ids,
        phx.meshing.SizeFieldDomain.MESH_GEODESIC,
        adjacency=np.asarray(((1, 2), (0, 1)), dtype=np.int32),
        combination=phx.meshing.SizeCombinationPolicy.EXPLICIT_PRIORITY,
    )

    np.testing.assert_allclose(field.values, (0.1, 0.3, 0.7))
    assert report.graded_count == 2 and report.clamped
    assert report.maximum_gradation_violation <= 1.0e-12


def test_proximity_gaps_are_measured_between_facing_scopes():
    source = _mesh_scope((1,))
    target = _mesh_scope((2,))
    control = phx.meshing.ProximitySizeControl(source, target, 3, maximum_size=1.0)
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.2, 0.3), (0.9, 0.6)))
    entities = np.asarray((1, 1, 2, 2), dtype=np.int64)
    facing = np.asarray(((0.0, 1.0), (0.0, 1.0), (0.0, -1.0), (0.0, -1.0)))

    field, _ = phx.meshing.resolve_size_controls(
        (control,),
        points,
        entities,
        phx.meshing.SizeFieldDomain.SAMPLE_CLOUD,
        normals=facing,
    )
    gaps = (
        np.hypot(0.2, 0.3),
        np.hypot(0.1, 0.6),
        np.hypot(0.2, 0.3),
        np.hypot(0.1, 0.6),
    )
    np.testing.assert_allclose(field.values, np.asarray(gaps) / 3.0)

    averted, _ = phx.meshing.resolve_size_controls(
        (control,),
        points,
        entities,
        phx.meshing.SizeFieldDomain.SAMPLE_CLOUD,
        normals=facing * np.asarray((1.0, 1.0, -1.0, -1.0))[:, None],
    )
    np.testing.assert_allclose(averted.values, 1.0)
    with pytest.raises(ValueError, match="normals"):
        phx.meshing.resolve_size_controls(
            (control,), points, entities, phx.meshing.SizeFieldDomain.SAMPLE_CLOUD
        )


def test_resolved_sizes_compile_into_isotropic_metric_constraints():
    scope = _mesh_scope((4, 5))
    field, _ = phx.meshing.resolve_size_controls(
        (phx.meshing.UniformSizeControl(scope, 0.25),),
        np.asarray(((0.0, 0.0), (1.0, 0.0))),
        scope.entity_ids,
        phx.meshing.SizeFieldDomain.SAMPLE_CLOUD,
    )
    metric = phx.meshing.size_field_metric(field, scope, maximum_gradation=1.2)
    np.testing.assert_allclose(metric.values, np.tile(16.0 * np.eye(2), (2, 1, 1)))
    with pytest.raises(ValueError, match="entity IDs"):
        phx.meshing.size_field_metric(field, _mesh_scope((4, 6)), maximum_gradation=1.2)


def test_metric_normalization_clamps_size_and_anisotropy():
    metric = phx.meshing.MeshMetricField(
        _mesh_scope((1, 2)),
        np.asarray((np.diag((1.0, 10_000.0)), np.diag((0.01, 1.0)))),
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_anisotropy=4.0,
    )
    normalized, evidence = phx.meshing.normalize_mesh_metric(
        metric,
        policy=phx.meshing.MetricNormalizationPolicy(
            minimum_size=0.1,
            maximum_size=2.0,
            maximum_anisotropy=4.0,
            gradation=phx.meshing.MetricGradationPolicy(1.3),
        ),
        adjacency=np.asarray(((0, 1),), dtype=np.int32),
        coordinates=np.asarray(((0.0, 0.0), (1.0, 0.0))),
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(normalized.values))
    assert np.all(eigenvalues >= 1.0 / 2.0**2 - 1.0e-12)
    assert np.all(eigenvalues <= 1.0 / 0.1**2 + 1.0e-12)
    assert np.all(np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0]) <= 4.0 + 1.0e-12)
    assert evidence.minimum_size_clamped_count == 1
    assert evidence.maximum_size_clamped_count == 1
    assert evidence.anisotropy_clamped_count == 1
    assert evidence.passed and evidence.gradation.maximum_violation <= 1.0e-12
