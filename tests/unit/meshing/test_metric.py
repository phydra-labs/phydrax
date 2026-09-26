import numpy as np
import pytest

import phydrax as phx


meshing = phx.meshing


def _scope(count):
    return meshing.MeshingScope(
        "mesh",
        "r1",
        meshing.MeshingEntityKind.MESH,
        0,
        "vertices",
        np.arange(count, dtype=np.int64),
    )


def _grid(count):
    axis = np.linspace(0.0, 1.0, count)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((first.ravel(), second.ravel()), axis=1)
    rows = np.arange(count * count).reshape(count, count)
    edges = np.concatenate(
        (
            np.stack((rows[:-1, :].ravel(), rows[1:, :].ravel()), axis=1),
            np.stack((rows[:, :-1].ravel(), rows[:, 1:].ravel()), axis=1),
            np.stack((rows[:-1, :-1].ravel(), rows[1:, 1:].ravel()), axis=1),
        )
    )
    return points, edges


def _anisotropic_metrics(count, seed):
    generator = np.random.default_rng(seed)
    angles = generator.uniform(0.0, np.pi, count)
    sizes = generator.uniform(0.05, 0.4, count)
    sizes[0] = 0.01
    rotation = np.stack(
        (
            np.stack((np.cos(angles), -np.sin(angles)), axis=-1),
            np.stack((np.sin(angles), np.cos(angles)), axis=-1),
        ),
        axis=-2,
    )
    eigenvalues = np.stack((sizes**-2, (3.0 * sizes) ** -2), axis=-1)
    return (rotation * eigenvalues[:, None, :]) @ np.swapaxes(rotation, -1, -2)


def _determinant_sizes(values):
    return np.linalg.det(values) ** (-1.0 / (2 * values.shape[-1]))


@pytest.mark.parametrize("kind", tuple(meshing.MetricGradationKind))
def test_scalar_gradation_bounds_every_edge_and_ignores_numbering(kind):
    points, edges = _grid(7)
    count = points.shape[0]
    values = _anisotropic_metrics(count, 0)
    field = meshing.MeshMetricField(
        _scope(count), values, minimum_size=0.005, maximum_size=1.0
    )
    policy = meshing.MetricGradationPolicy(1.2, kind=kind)
    graded, evidence = meshing.grade_mesh_metric(
        field, policy=policy, adjacency=edges, coordinates=points
    )

    sizes = _determinant_sizes(np.asarray(graded.values))
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    for first, second in ((0, 1), (1, 0)):
        source = sizes[edges[:, first]]
        if kind is meshing.MetricGradationKind.PHYSICAL:
            bound = source + 0.2 * lengths
        else:
            bound = source * 1.2 ** (lengths / source)
        assert np.all(sizes[edges[:, second]] <= bound * (1.0 + 1.0e-12))
    assert evidence.converged and evidence.maximum_violation <= 1.0e-12
    assert evidence.modified_count > 0
    original = _determinant_sizes(values)
    assert np.all(sizes <= original * (1.0 + 1.0e-12))
    eigenvalues = np.linalg.eigvalsh(np.asarray(graded.values))
    assert np.all(eigenvalues[:, 1] / eigenvalues[:, 0] <= 9.0 * (1.0 + 1.0e-9))

    permutation = np.random.default_rng(3).permutation(count)
    inverse = np.argsort(permutation)
    permuted, _ = meshing.grade_mesh_metric(
        meshing.MeshMetricField(
            _scope(count), values[permutation], minimum_size=0.005, maximum_size=1.0
        ),
        policy=policy,
        adjacency=inverse[edges][::-1, ::-1],
        coordinates=points[permutation],
    )
    np.testing.assert_array_equal(
        np.asarray(permuted.values), np.asarray(graded.values)[permutation]
    )


def test_anisotropic_gradation_converges_and_certifies_every_edge():
    points, edges = _grid(6)
    count = points.shape[0]
    field = meshing.MeshMetricField(
        _scope(count),
        _anisotropic_metrics(count, 1),
        minimum_size=0.005,
        maximum_size=1.0,
    )
    graded, evidence = meshing.grade_mesh_metric(
        field,
        policy=meshing.MetricGradationPolicy(1.3, anisotropic=True),
        adjacency=edges,
        coordinates=points,
    )

    assert evidence.converged and evidence.maximum_violation <= 1.0e-9
    difference = np.asarray(graded.values) - np.asarray(field.values)
    assert np.all(np.linalg.eigvalsh(difference) >= -1.0e-8 * np.abs(difference).max())

    stalled, stalled_evidence = meshing.grade_mesh_metric(
        field,
        policy=meshing.MetricGradationPolicy(1.3, anisotropic=True, maximum_sweeps=1),
        adjacency=edges,
        coordinates=points,
    )
    assert not stalled_evidence.converged
    assert stalled_evidence.maximum_violation > 1.0e-9


def test_metric_combination_dominates_is_idempotent_and_order_independent():
    count = 9
    first = meshing.MeshMetricField(
        _scope(count),
        _anisotropic_metrics(count, 4),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    second = meshing.MeshMetricField(
        _scope(count),
        _anisotropic_metrics(count, 5),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    combined, evidence = meshing.combine_mesh_metrics((first, second))
    reversed_, _ = meshing.combine_mesh_metrics((second, first))
    same, _ = meshing.combine_mesh_metrics((first, first))

    np.testing.assert_array_equal(
        np.asarray(combined.values), np.asarray(reversed_.values)
    )
    for field in (first, second):
        excess = np.asarray(combined.values) - np.asarray(field.values)
        scale = np.max(np.abs(np.asarray(field.values)))
        assert np.all(np.linalg.eigvalsh(excess) >= -1.0e-9 * scale)
    np.testing.assert_allclose(
        np.asarray(same.values), np.asarray(first.values), rtol=1.0e-10, atol=0.0
    )
    assert evidence.passed

    restrictive = meshing.MeshMetricField(
        _scope(count),
        np.tile(np.eye(2), (count, 1, 1)),
        minimum_size=0.5,
        maximum_size=2.0,
    )
    _, conflict = meshing.combine_mesh_metrics((first, restrictive))
    assert not conflict.passed
    assert bool(np.asarray(conflict.minimum_size_conflict)[0])
    disjoint = meshing.MeshMetricField(
        _scope(count),
        np.tile(np.eye(2), (count, 1, 1)),
        minimum_size=3.0,
        maximum_size=4.0,
    )
    with pytest.raises(ValueError, match="empty intersection"):
        meshing.combine_mesh_metrics((first, disjoint))


def test_log_euclidean_interpolation_is_spd_and_interpolates_determinants():
    metrics = _anisotropic_metrics(8, 6).reshape(4, 2, 2, 2)
    weights = np.asarray(((0.5, 0.5), (0.25, 0.75), (1.0, 0.0), (0.1, 0.9)))
    mean = np.asarray(meshing.interpolate_mesh_metric(metrics, weights))

    assert np.all(np.linalg.eigvalsh(mean) > 0.0)
    np.testing.assert_allclose(mean, np.swapaxes(mean, -1, -2), atol=1.0e-9)
    expected = np.sum(weights * np.log(np.linalg.det(metrics)), axis=1)
    np.testing.assert_allclose(np.log(np.linalg.det(mean)), expected, rtol=1.0e-10)
    np.testing.assert_allclose(mean[2], metrics[2, 0], rtol=1.0e-10)


def test_normalization_meets_complexity_target_within_bounds():
    count = 25
    field = meshing.MeshMetricField(
        _scope(count),
        _anisotropic_metrics(count, 7),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    volumes = np.full((count,), 1.0 / count)
    policy = meshing.MetricNormalizationPolicy(
        minimum_size=0.02,
        maximum_size=0.5,
        maximum_anisotropy=2.0,
        target_complexity=300.0,
    )
    normalized, evidence = meshing.normalize_mesh_metric(
        field, policy=policy, vertex_volumes=volumes
    )

    eigenvalues = np.linalg.eigvalsh(np.asarray(normalized.values))
    complexity = np.sum(volumes * np.sqrt(np.prod(eigenvalues, axis=1)))
    assert evidence.passed
    assert evidence.complexity_status is meshing.MetricComplexityStatus.MET
    np.testing.assert_allclose(complexity, 300.0, rtol=1.0e-9)
    assert np.all(eigenvalues >= 0.5**-2 * (1.0 - 1.0e-12))
    assert np.all(eigenvalues <= 0.02**-2 * (1.0 + 1.0e-12))
    assert np.all(eigenvalues[:, 1] <= 4.0 * eigenvalues[:, 0] * (1.0 + 1.0e-12))

    _, unreachable = meshing.normalize_mesh_metric(
        field,
        policy=meshing.MetricNormalizationPolicy(
            minimum_size=0.02, maximum_size=0.5, target_complexity=1.0e7
        ),
        vertex_volumes=volumes,
    )
    assert unreachable.complexity_status is (
        meshing.MetricComplexityStatus.TARGET_ABOVE_MAXIMUM
    )
    assert not unreachable.passed


def test_normalization_repairs_samples_only_when_the_policy_requests_it():
    count = 4
    raw = np.asarray((((-3.0, 4.0), (0.0, 2.0)),) * count)
    samples = meshing.MeshMetricSamples(_scope(count), raw)
    strict = meshing.MetricNormalizationPolicy(minimum_size=0.1, maximum_size=2.0)
    with pytest.raises(ValueError, match="symmetric"):
        meshing.normalize_mesh_metric(samples, policy=strict)

    repaired, evidence = meshing.normalize_mesh_metric(
        samples,
        policy=meshing.MetricNormalizationPolicy(
            minimum_size=0.1, maximum_size=2.0, symmetrize=True, project_indefinite=True
        ),
    )
    assert evidence.symmetrized_count == count
    assert evidence.projected_tensor_count == count
    assert np.all(np.linalg.eigvalsh(np.asarray(repaired.values)) >= 0.25 - 1.0e-12)


def test_lp_hessian_metric_meets_complexity_and_reports_spectral_handling():
    count = 16
    hessian = np.zeros((count, 2, 2))
    hessian[:, 0, 0] = np.linspace(1.0, 4.0, count)
    hessian[:, 1, 1] = -0.25
    hessian[:2] = 0.0
    volumes = np.full((count,), 1.0 / count)
    metric, evidence = meshing.lp_metric_from_hessian(
        hessian,
        p=2.0,
        target_complexity=80.0,
        minimum_size=1.0e-3,
        maximum_size=1.0,
        maximum_anisotropy=50.0,
        vertex_volumes=volumes,
    )

    eigenvalues = np.linalg.eigvalsh(metric)
    np.testing.assert_allclose(
        np.sum(volumes * np.sqrt(np.prod(eigenvalues, axis=1))), 80.0, rtol=1.0e-9
    )
    assert evidence.passed
    assert evidence.indefinite_tensor_count == count - 2
    assert evidence.zero_tensor_count == 2
    np.testing.assert_allclose(eigenvalues[:2], 1.0, rtol=1.0e-12)
    # |H| keeps its eigenvectors: the curvature direction is refined most.
    assert np.all(metric[2:, 0, 0] > metric[2:, 1, 1])


def test_metric_edge_lengths_match_constant_metric_lengths():
    points = np.asarray(((0.0, 0.0), (0.3, 0.4), (1.0, 0.0)))
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int32)
    metric = np.tile(np.diag((4.0, 9.0)), (3, 1, 1))
    lengths = np.asarray(meshing.metric_edge_lengths(metric, points, edges))
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    np.testing.assert_allclose(
        lengths, np.sqrt(4.0 * delta[:, 0] ** 2 + 9.0 * delta[:, 1] ** 2)
    )
