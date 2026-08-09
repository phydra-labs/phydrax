#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _target(
    points,
    weights,
    *,
    normalized=True,
    provenance="scalable-test",
    mask=None,
):
    mask_field = (
        None
        if mask is None
        else cx.Field(jnp.asarray(mask, dtype=bool), dims=("atom",))
    )
    return phx.integration.discrete(
        jnp.asarray(points, dtype=float),
        cx.Field(jnp.asarray(weights, dtype=float), dims=("atom",)),
        axes="atom",
        mask=mask_field,
        normalized=normalized,
        provenance=provenance,
    )


def _problem(
    source_points,
    target_points,
    *,
    source_weights=None,
    target_weights=None,
    source_mask=None,
    target_mask=None,
    normalized=True,
):
    source_points = jnp.asarray(source_points, dtype=float)
    target_points = jnp.asarray(target_points, dtype=float)
    if source_weights is None:
        source_weights = jnp.ones((source_points.shape[0],))
    if target_weights is None:
        target_weights = jnp.ones((target_points.shape[0],))
    return phx.transport.discrete_problem(
        _target(
            source_points,
            source_weights,
            normalized=normalized,
            provenance="scalable-source",
            mask=source_mask,
        ),
        _target(
            target_points,
            target_weights,
            normalized=normalized,
            provenance="scalable-target",
            mask=target_mask,
        ),
        cost=phx.transport.SquaredEuclideanCost(),
    )


def _solver(
    *,
    rank=2048,
    key=jax.random.key(17),
    probe_tolerance=jnp.inf,
    max_iterations=300,
    tolerance=1e-7,
):
    features = phx.transport.GaussianPositiveFeatures(
        key,
        rank,
        num_probes=32,
        probe_tolerance=probe_tolerance,
    )
    return phx.transport.PositiveFeatureSinkhorn(
        1.0,
        features,
        max_iterations=max_iterations,
        tolerance=tolerance,
        check_every=5,
        store_history=True,
    )


def test_positive_features_are_nonnegative_replayable_and_carry_probe_evidence():
    problem = _problem(
        [[-0.3, 0.1], [0.0, -0.2], [0.25, 0.2]],
        [[-0.15, -0.1], [0.1, 0.15], [0.35, -0.05]],
    )
    feature_map = phx.transport.GaussianPositiveFeatures(
        jax.random.key(5), 4096, num_probes=24
    )
    left = feature_map(problem, 1.0)
    replay = feature_map(problem, 1.0)
    exact = jnp.exp(-problem.cost_matrix())
    approximate = left.kernel_matrix()

    assert jnp.all(left.source_factors >= 0.0)
    assert jnp.all(left.target_factors >= 0.0)
    assert jnp.array_equal(left.source_factors, replay.source_factors)
    assert jnp.array_equal(left.target_factors, replay.target_factors)
    assert jnp.array_equal(left.diagnostics.key_data, replay.diagnostics.key_data)
    assert left.rank == 4096
    assert left.diagnostics.rank == 4096
    assert left.diagnostics.num_probes == 24
    assert left.diagnostics.probe_source_indices.shape == (24,)
    assert left.diagnostics.probe_target_indices.shape == (24,)
    assert left.diagnostics.successful
    assert jnp.linalg.norm(approximate - exact) / jnp.linalg.norm(exact) < 0.2

    negative = left.source_factors.at[0, 0].set(-1.0)
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="nonnegative"):
        invalid = phx.transport.PositiveKernelFactors(
            negative,
            left.target_factors,
            source_log_scale=left.source_log_scale,
            target_log_scale=left.target_log_scale,
            source_points=left.source_points,
            target_points=left.target_points,
            epsilon=left.epsilon,
            diagnostics=left.diagnostics,
            factorization_id=left.factorization_id,
        )
        jax.block_until_ready(invalid.source_factors)


def test_factorized_sinkhorn_matches_dense_plan_marginals_actions_and_objectives():
    problem = _problem(
        [[-0.35], [0.0], [0.3]],
        [[-0.2], [0.12], [0.4]],
        source_weights=[0.2, 0.5, 0.3],
        target_weights=[0.4, 0.1, 0.5],
    )
    approximate = _solver(rank=4096)(problem, exact_ground_cost=True)
    exact = phx.transport.Sinkhorn(
        1.0,
        max_iterations=1000,
        tolerance=1e-9,
        check_every=5,
    )(problem)
    plan = approximate.dense_plan()
    payload = jnp.asarray([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]])

    assert approximate.converged
    assert approximate.exact_ground_cost_computed
    assert approximate.provenance.execution == "factorized"
    assert approximate.provenance.approximation.startswith(
        "gaussian-positive-features"
    )
    assert jnp.allclose(approximate.source_marginal(), problem.source_weights, atol=2e-6)
    assert jnp.allclose(approximate.target_marginal(), problem.target_weights, atol=2e-6)
    assert jnp.allclose(
        approximate.apply_source_to_target(payload), plan.T @ payload, atol=2e-6
    )
    assert jnp.allclose(
        approximate.apply_target_to_source(payload), plan @ payload, atol=2e-6
    )
    assert jnp.allclose(
        approximate.exact_transport_cost,
        jnp.sum(plan * problem.cost_matrix()),
        rtol=2e-6,
        atol=2e-6,
    )
    probability_plan = plan / problem.mass
    kernel = approximate.factors.kernel_matrix()
    source_probability = problem.source_probabilities
    target_probability = problem.target_probabilities
    log_ratio = jnp.where(
        probability_plan > 0.0,
        jnp.log(probability_plan)
        - jnp.log(source_probability)[:, None]
        - jnp.log(target_probability)[None, :],
        0.0,
    )
    surrogate_cost = -approximate.epsilon * jnp.log(kernel)
    kl = (
        jnp.sum(probability_plan * log_ratio)
        - jnp.sum(probability_plan)
        + 1.0
    )
    dense_surrogate_objective = problem.mass * (
        jnp.sum(probability_plan * surrogate_cost) + approximate.epsilon * kl
    )
    assert jnp.allclose(
        approximate.surrogate_regularized_cost,
        dense_surrogate_objective,
        rtol=2e-5,
        atol=2e-5,
    )
    assert jnp.allclose(plan, exact.dense_plan(), rtol=0.12, atol=0.04)


def test_approximation_failure_zero_rows_and_extreme_translation_are_explicit():
    problem = _problem([[-0.4], [0.0], [0.35]], [[-0.25], [0.2], [0.45]])
    rejected = _solver(rank=2, probe_tolerance=0.0)(problem)
    assert not rejected.converged
    assert rejected.approximation.status == int(
        phx.transport.KernelApproximationStatus.PROBE_TOLERANCE_EXCEEDED
    )
    assert rejected.diagnostics.status == int(
        phx.transport.TransportStatus.APPROXIMATION_FAILED
    )

    solver = _solver(rank=128)
    factors = solver.feature_map(problem, solver.epsilon)
    zero_factors = eqx.tree_at(
        lambda item: item.source_factors,
        factors,
        factors.source_factors.at[1].set(0.0),
    )
    zero_result = solver(problem, factors=zero_factors)
    assert not zero_result.converged
    assert zero_result.diagnostics.status == int(
        phx.transport.TransportStatus.ZERO_KERNEL_ROW
    )

    translated = _problem(
        [[1.0e6 - 0.2], [1.0e6], [1.0e6 + 0.3]],
        [[1.0e6 - 0.1], [1.0e6 + 0.15], [1.0e6 + 0.4]],
    )
    translated_result = _solver(rank=2048)(translated)
    assert translated_result.converged
    assert jnp.all(jnp.isfinite(translated_result.dense_plan()))
    assert translated_result.diagnostics.normalized_marginal_residual < 2e-6


def test_factorized_solver_is_jittable_vmappable_differentiable_and_preserves_mass():
    target_points = jnp.asarray([[-0.3], [0.15], [0.45]])
    source_points = jnp.asarray([[-0.4], [0.0], [0.35]])
    solver = _solver(rank=1024, tolerance=2e-6)

    def objective(points):
        return solver(
            _problem(
                points,
                target_points,
                source_weights=[0.4, 0.0, 0.6],
                target_weights=[0.2, 0.3, 0.5],
            )
        ).regularized_cost

    compiled = jax.jit(objective)(source_points)
    gradient = jax.grad(objective)(source_points)
    batched = jax.vmap(lambda shift: objective(source_points + shift))(
        jnp.asarray([-0.1, 0.0, 0.2])
    )
    physical_problem = _problem(
        source_points,
        target_points,
        source_weights=[0.8, 0.0, 1.2],
        target_weights=[0.4, 0.6, 1.0],
        normalized=False,
    )
    physical = solver(physical_problem)
    masked_problem = _problem(
        [[-0.4], [999.0], [0.35], [-999.0]],
        [[-0.3], [999.0], [0.45], [-999.0]],
        source_weights=[0.8, 50.0, 1.2, 50.0],
        target_weights=[0.4, 50.0, 1.6, 50.0],
        source_mask=[True, False, True, False],
        target_mask=[True, False, True, False],
        normalized=False,
    )
    masked = solver(masked_problem)
    assert masked.converged
    assert jnp.all(masked.factors.source_factors[jnp.asarray([1, 3])] == 0.0)
    assert jnp.all(masked.factors.target_factors[jnp.asarray([1, 3])] == 0.0)
    assert jnp.allclose(
        masked.source_marginal(), masked_problem.source_weights, atol=5e-6
    )

    assert jnp.isfinite(compiled)
    assert jnp.all(jnp.isfinite(gradient))
    assert batched.shape == (3,)
    assert jnp.all(jnp.isfinite(batched))
    assert physical.converged
    assert jnp.allclose(
        jnp.sum(physical.dense_plan()), physical_problem.mass, atol=5e-6
    )
    assert jnp.allclose(
        physical.source_marginal(), physical_problem.source_weights, atol=5e-6
    )


def test_scientific_particle_transform_keeps_approximation_provenance_and_rejects_failure():
    particles = jnp.asarray(
        [
            [[-0.5], [0.0], [0.6]],
            [[0.1], [0.4], [0.8]],
        ]
    )
    weights = jnp.asarray([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])
    solver = _solver(rank=2048, tolerance=2e-6)
    metric = phx.uq.predictive_sinkhorn_divergence(
        particles[0],
        particles[1],
        solver=solver,
    )
    median = phx.transport.soft_quantile(
        particles[0, :, 0],
        0.5,
        solver=solver,
    )
    transformed = phx.uq.optimal_transport_ensemble_transform(
        particles,
        weights,
        particle_axis=1,
        solver=solver,
    )

    assert metric.converged
    assert metric.cross.provenance.approximation.startswith(
        "gaussian-positive-features"
    )
    assert jnp.isfinite(median)
    assert transformed.particles.shape == particles.shape
    assert jnp.all(transformed.transport.converged)
    assert transformed.transport.provenance.approximation.startswith(
        "gaussian-positive-features"
    )
    assert jnp.allclose(transformed.mean_error, 0.0, atol=5e-6)

    failing = _solver(rank=8, max_iterations=1, tolerance=0.0)
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="did not converge"):
        rejected = phx.uq.optimal_transport_ensemble_transform(
            particles[0],
            weights[0],
            solver=failing,
        )
        jax.block_until_ready(rejected.particles)


def test_scalable_balanced_transport_public_catalog_is_complete():
    expected = {
        "AbstractBalancedTransportPlan",
        "AbstractBalancedTransportSolver",
        "GaussianPositiveFeatures",
        "KernelApproximationStatus",
        "PositiveFeatureSinkhorn",
        "PositiveFeatureSinkhornResult",
        "PositiveKernelApproximationDiagnostics",
        "PositiveKernelFactors",
    }
    assert expected <= set(phx.transport.__all__)
    assert all(vars(phx.transport)[name] is not None for name in expected)
