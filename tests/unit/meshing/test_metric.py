import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests._support.metric_meshes import (
    anisotropic_metric_samples,
    METRIC_DIFFERENTIAL_TOLERANCE,
    metric_grid,
    METRIC_INTERPOLATION_TOLERANCE,
    METRIC_ITERATION_TOLERANCE,
    METRIC_PSD_TOLERANCE,
    METRIC_ROUNDOFF_TOLERANCE,
    tensor_metric_scope,
)


meshing = phx.meshing


def _determinant_sizes(values: np.ndarray) -> np.ndarray:
    return np.linalg.det(values) ** (-1.0 / (2 * values.shape[-1]))


@pytest.mark.strict_jax
def test_log_euclidean_metric_interpolation_derivatives_at_isotropy() -> None:
    identity = jnp.eye(2, dtype=jnp.float64)
    samples = jnp.stack((2.0 * identity, 3.0 * identity))
    weights = jnp.asarray((0.25, 0.75), dtype=jnp.float64)
    direction = jnp.asarray(
        (((0.2, 0.1), (0.1, -0.3)), ((-0.1, 0.04), (0.04, 0.25))),
        dtype=jnp.float64,
    )

    def interpolate(value: jax.Array) -> jax.Array:
        return meshing.interpolate_mesh_metric(value, weights)

    value, tangent = jax.jit(
        lambda tensors, perturbation: jax.jvp(interpolate, (tensors,), (perturbation,))
    )(samples, direction)
    scale = 2.0**0.25 * 3.0**0.75
    expected = scale * (0.25 * direction[0] / 2.0 + 0.75 * direction[1] / 3.0)
    np.testing.assert_allclose(
        value,
        scale * identity,
        rtol=METRIC_DIFFERENTIAL_TOLERANCE,
        atol=METRIC_DIFFERENTIAL_TOLERANCE,
    )
    np.testing.assert_allclose(
        tangent,
        expected,
        rtol=METRIC_DIFFERENTIAL_TOLERANCE,
        atol=METRIC_DIFFERENTIAL_TOLERANCE,
    )

    def objective(tensors: jax.Array) -> jax.Array:
        return jnp.sum(interpolate(tensors) * identity)

    reverse = jax.jit(jax.grad(objective))(samples)
    np.testing.assert_allclose(
        reverse,
        jnp.stack((scale * 0.25 / 2.0 * identity, scale * 0.75 / 3.0 * identity)),
        rtol=METRIC_DIFFERENTIAL_TOLERANCE,
        atol=METRIC_DIFFERENTIAL_TOLERANCE,
    )


def test_metric_structural_contract_refuses_transformed_shape_and_dtype() -> None:
    scope = tensor_metric_scope(3)
    field = meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.eye(2), (3, 2, 2)),
        minimum_size=0.5,
        maximum_size=2.0,
    )
    phx.typing.validate(field)

    nonsquare = eqx.tree_at(
        lambda value: value.values,
        field,
        jnp.ones((3, 2, 3), dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="values"):
        phx.typing.validate(nonsquare)

    wrong_dtype = eqx.tree_at(
        lambda value: value.values,
        field,
        field.values.astype(jnp.float32),
    )
    with pytest.raises(TypeError, match="values"):
        phx.typing.validate(wrong_dtype)


def test_scalar_gradation_bounds_every_edge_and_ignores_numbering() -> None:
    for kind in tuple(meshing.MetricGradationKind):
        points, edges = metric_grid(7)
        count = points.shape[0]
        values = anisotropic_metric_samples(count, 0)
        field = meshing.MeshMetricField(
            tensor_metric_scope(count), values, minimum_size=0.005, maximum_size=2.0
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
            assert np.all(
                sizes[edges[:, second]] <= bound * (1.0 + METRIC_ROUNDOFF_TOLERANCE)
            )
        assert (
            evidence.converged and evidence.maximum_violation <= METRIC_ROUNDOFF_TOLERANCE
        )
        assert evidence.modified_count > 0
        original = _determinant_sizes(values)
        assert np.all(sizes <= original * (1.0 + METRIC_ROUNDOFF_TOLERANCE))
        eigenvalues = np.linalg.eigvalsh(np.asarray(graded.values))
        assert np.all(
            eigenvalues[:, 1] / eigenvalues[:, 0]
            <= 9.0 * (1.0 + METRIC_ITERATION_TOLERANCE)
        )

        permutation = np.random.default_rng(3).permutation(count)
        inverse = np.argsort(permutation)
        permuted, _ = meshing.grade_mesh_metric(
            meshing.MeshMetricField(
                tensor_metric_scope(count),
                values[permutation],
                minimum_size=0.005,
                maximum_size=2.0,
            ),
            policy=policy,
            adjacency=inverse[edges][::-1, ::-1],
            coordinates=points[permutation],
        )
        np.testing.assert_array_equal(
            np.asarray(permuted.values), np.asarray(graded.values)[permutation]
        )


def test_anisotropic_gradation_converges_and_certifies_every_edge() -> None:
    points, edges = metric_grid(6)
    count = points.shape[0]
    field = meshing.MeshMetricField(
        tensor_metric_scope(count),
        anisotropic_metric_samples(count, 1),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    graded, evidence = meshing.grade_mesh_metric(
        field,
        policy=meshing.MetricGradationPolicy(1.3, anisotropic=True),
        adjacency=edges,
        coordinates=points,
    )

    assert evidence.converged and evidence.maximum_violation <= METRIC_ITERATION_TOLERANCE
    difference = np.asarray(graded.values) - np.asarray(field.values)
    assert np.all(
        np.linalg.eigvalsh(difference) >= -METRIC_PSD_TOLERANCE * np.abs(difference).max()
    )

    with pytest.raises(meshing.MetricGradationError) as failure:
        meshing.grade_mesh_metric(
            field,
            policy=meshing.MetricGradationPolicy(1.3, anisotropic=True, maximum_sweeps=1),
            adjacency=edges,
            coordinates=points,
        )
    stalled_evidence = failure.value.evidence
    assert stalled_evidence.status is meshing.MetricGradationStatus.SWEEP_LIMIT
    assert stalled_evidence.maximum_violation > METRIC_ITERATION_TOLERANCE


def test_anisotropic_gradation_withholds_growth_beyond_hard_bounds() -> None:
    points, edges = metric_grid(6)
    count = points.shape[0]
    angles = np.random.default_rng(0).uniform(0.0, np.pi, count)
    cosine, sine = np.cos(angles), np.sin(angles)
    rotation = np.stack(
        (np.stack((cosine, -sine), axis=-1), np.stack((sine, cosine), axis=-1)),
        axis=-2,
    )
    # Every metric sits at the minimum size, so each rotated intersection would
    # request sizes below it.
    eigenvalues = np.asarray((0.3**-2, 0.9**-2))
    values = (rotation * eigenvalues) @ np.swapaxes(rotation, -1, -2)
    field = meshing.MeshMetricField(
        tensor_metric_scope(count),
        values,
        minimum_size=0.3,
        maximum_size=2.0,
        maximum_anisotropy=3.0,
    )
    policy = meshing.MetricGradationPolicy(1.05, anisotropic=True)
    with pytest.raises(meshing.MetricGradationError) as failure:
        meshing.grade_mesh_metric(
            field,
            policy=policy,
            adjacency=edges,
            coordinates=points,
        )
    evidence = failure.value.evidence
    assert evidence.status is meshing.MetricGradationStatus.BOUNDS_CONFLICT
    assert evidence.maximum_violation > 1.0
    with pytest.raises(meshing.MetricGradationError) as normalized:
        meshing.normalize_mesh_metric(
            field,
            policy=meshing.MetricNormalizationPolicy(
                minimum_size=0.3,
                maximum_size=2.0,
                maximum_anisotropy=3.0,
                gradation=policy,
            ),
            adjacency=edges,
            coordinates=points,
        )
    assert normalized.value.evidence.status is (
        meshing.MetricGradationStatus.BOUNDS_CONFLICT
    )


def test_metric_field_rejects_tensors_outside_declared_bounds() -> None:
    for eigenvalues, violated in (
        ((400.0, 400.0), "minimum_size"),
        ((0.1, 0.1), "maximum_size"),
        ((1.0, 25.0), "maximum_anisotropy"),
    ):
        with pytest.raises(ValueError, match=violated):
            meshing.MeshMetricField(
                tensor_metric_scope(2),
                np.tile(np.diag(eigenvalues), (2, 1, 1)),
                minimum_size=0.1,
                maximum_size=2.0,
                maximum_anisotropy=2.0,
            )


def test_metric_field_admits_only_eigenvalue_roundoff_at_its_bounds() -> None:
    angle = 0.3
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    # Sizes exactly 1 and 2 at minimum_size 1, maximum_size 2, anisotropy 2.
    tensor = rotation @ np.diag((1.0, 0.25)) @ rotation.T
    bounds = {"minimum_size": 1.0, "maximum_size": 2.0, "maximum_anisotropy": 2.0}
    for scale in (1.0, 1.0 + 4.0 * np.finfo(np.float64).eps):
        meshing.MeshMetricField(
            tensor_metric_scope(2), np.tile(scale * tensor, (2, 1, 1)), **bounds
        )
    with pytest.raises(ValueError, match="minimum_size"):
        meshing.MeshMetricField(
            tensor_metric_scope(2), np.tile((1.0 + 1.0e-9) * tensor, (2, 1, 1)), **bounds
        )


def test_adaptation_routes_accept_only_certified_metric_fields() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3))),
    )
    source = meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    vertices = source.mesh.entity_set(0)
    samples = meshing.MeshMetricSamples(
        meshing.MeshingScope(
            source.mesh.mesh_id,
            source.mesh.numeric_version,
            meshing.MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            vertices.entity_ids,
        ),
        np.tile(np.eye(2), (vertices.entity_ids.shape[0], 1, 1)),
    )
    for request in (meshing.MetricMeshAdaptation, meshing.RelocationMeshAdaptation):
        with pytest.raises(TypeError, match="MeshMetricField"):
            # ty: ignore[invalid-argument-type]
            request(samples)
    with pytest.raises(TypeError, match="MeshMetricField"):
        meshing.BackgroundMetricControl(
            source.mesh,
            # ty: ignore[invalid-argument-type]
            samples,
            phx.SpatialCoordinateContract.si(),
        )
    with pytest.raises(TypeError, match="MeshMetricField"):
        meshing.MmgAdaptationPlan(
            source,
            meshing.MmgOptions(),
            meshing.MeshingLimits(),
            meshing.CellMeshAuditPolicy(),
            # ty: ignore[invalid-argument-type]
            metric=samples,
        )


def test_metric_combination_dominates_is_idempotent_and_order_independent() -> None:
    count = 9
    first = meshing.MeshMetricField(
        tensor_metric_scope(count),
        anisotropic_metric_samples(count, 4),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    second = meshing.MeshMetricField(
        tensor_metric_scope(count),
        anisotropic_metric_samples(count, 5),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    combined = meshing.combine_mesh_metrics((first, second))
    reversed_ = meshing.combine_mesh_metrics((second, first))
    same = meshing.combine_mesh_metrics((first, first))

    assert combined.successful and combined.evidence.passed
    assert combined.field is not None
    values = np.asarray(combined.field.values)
    assert reversed_.field is not None
    np.testing.assert_array_equal(values, np.asarray(reversed_.field.values))
    assert combined.evidence.evidence_id == reversed_.evidence.evidence_id
    assert combined.result_id == reversed_.result_id
    for field in (first, second):
        excess = values - np.asarray(field.values)
        scale = np.max(np.abs(np.asarray(field.values)))
        assert np.all(np.linalg.eigvalsh(excess) >= -METRIC_ITERATION_TOLERANCE * scale)
    assert same.field is not None
    np.testing.assert_allclose(
        # The successful idempotent combination must retain its field.
        np.asarray(same.field.values),
        np.asarray(first.values),
        rtol=METRIC_INTERPOLATION_TOLERANCE,
        atol=0.0,
    )
    spectrum = np.linalg.eigvalsh(values)
    assert np.all(spectrum[:, 1] <= 0.005**-2 * (1.0 + METRIC_ROUNDOFF_TOLERANCE))
    assert np.all(spectrum[:, 0] >= 2.0**-2 * (1.0 - METRIC_ROUNDOFF_TOLERANCE))


def test_conflicting_metric_combination_withholds_the_field() -> None:
    count = 9
    first = meshing.MeshMetricField(
        tensor_metric_scope(count),
        anisotropic_metric_samples(count, 4),
        minimum_size=0.005,
        maximum_size=2.0,
    )
    restrictive = meshing.MeshMetricField(
        tensor_metric_scope(count),
        np.tile(np.eye(2), (count, 1, 1)),
        minimum_size=0.5,
        maximum_size=2.0,
    )
    conflict = meshing.combine_mesh_metrics((first, restrictive))
    assert not conflict.successful and conflict.field is None
    assert not conflict.evidence.passed
    assert conflict.evidence.minimum_size == 0.5
    assert bool(np.asarray(conflict.evidence.minimum_size_conflict)[0])

    isotropic = meshing.MeshMetricField(
        tensor_metric_scope(count),
        np.tile(np.eye(2), (count, 1, 1)),
        minimum_size=0.005,
        maximum_size=2.0,
        maximum_anisotropy=2.0,
    )
    anisotropic = meshing.combine_mesh_metrics((first, isotropic))
    assert anisotropic.field is None
    assert np.all(np.asarray(anisotropic.evidence.anisotropy_conflict))
    assert not np.any(np.asarray(anisotropic.evidence.maximum_size_conflict))

    disjoint = meshing.MeshMetricField(
        tensor_metric_scope(count),
        np.tile(np.eye(2) / 3.0**2, (count, 1, 1)),
        minimum_size=3.0,
        maximum_size=4.0,
    )
    incompatible = meshing.combine_mesh_metrics((first, disjoint))
    assert not incompatible.successful and incompatible.field is None
    assert incompatible.evidence.size_interval_conflict
    assert incompatible.evidence.conflict_count == 1


def test_log_euclidean_interpolation_is_spd_and_interpolates_determinants() -> None:
    metrics = anisotropic_metric_samples(8, 6).reshape(4, 2, 2, 2)
    weights = np.asarray(((0.5, 0.5), (0.25, 0.75), (1.0, 0.0), (0.1, 0.9)))
    mean = np.asarray(meshing.interpolate_mesh_metric(metrics, weights))

    assert np.all(np.linalg.eigvalsh(mean) > 0.0)
    np.testing.assert_allclose(
        mean, np.swapaxes(mean, -1, -2), atol=METRIC_ITERATION_TOLERANCE
    )
    expected = np.sum(weights * np.log(np.linalg.det(metrics)), axis=1)
    np.testing.assert_allclose(
        np.log(np.linalg.det(mean)), expected, rtol=METRIC_INTERPOLATION_TOLERANCE
    )
    np.testing.assert_allclose(
        mean[2], metrics[2, 0], rtol=METRIC_INTERPOLATION_TOLERANCE
    )


def test_normalization_meets_complexity_target_within_bounds() -> None:
    count = 25
    field = meshing.MeshMetricField(
        tensor_metric_scope(count),
        anisotropic_metric_samples(count, 7),
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
    np.testing.assert_allclose(complexity, 300.0, rtol=METRIC_ITERATION_TOLERANCE)
    assert np.all(eigenvalues >= 0.5**-2 * (1.0 - METRIC_ROUNDOFF_TOLERANCE))
    assert np.all(eigenvalues <= 0.02**-2 * (1.0 + METRIC_ROUNDOFF_TOLERANCE))
    assert np.all(
        eigenvalues[:, 1] <= 4.0 * eigenvalues[:, 0] * (1.0 + METRIC_ROUNDOFF_TOLERANCE)
    )

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


def test_normalization_repairs_samples_only_when_the_policy_requests_it() -> None:
    count = 4
    raw = np.asarray((((-3.0, 4.0), (0.0, 2.0)),) * count)
    samples = meshing.MeshMetricSamples(tensor_metric_scope(count), raw)
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
    assert np.all(
        np.linalg.eigvalsh(np.asarray(repaired.values))
        >= 0.25 - METRIC_ROUNDOFF_TOLERANCE
    )


def test_lp_hessian_metric_meets_complexity_and_reports_spectral_handling() -> None:
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
        np.sum(volumes * np.sqrt(np.prod(eigenvalues, axis=1))),
        80.0,
        rtol=METRIC_ITERATION_TOLERANCE,
    )
    assert evidence.passed
    assert evidence.indefinite_tensor_count == count - 2
    assert evidence.zero_tensor_count == 2
    np.testing.assert_allclose(eigenvalues[:2], 1.0, rtol=METRIC_ROUNDOFF_TOLERANCE)
    # |H| keeps its eigenvectors: the curvature direction is refined most.
    assert np.all(metric[2:, 0, 0] > metric[2:, 1, 1])


def test_metric_edge_lengths_match_constant_metric_lengths() -> None:
    points = np.asarray(((0.0, 0.0), (0.3, 0.4), (1.0, 0.0)))
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int32)
    metric = np.tile(np.diag((4.0, 9.0)), (3, 1, 1))
    lengths = np.asarray(meshing.metric_edge_lengths(metric, points, edges))
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    np.testing.assert_allclose(
        lengths, np.sqrt(4.0 * delta[:, 0] ** 2 + 9.0 * delta[:, 1] ** 2)
    )
