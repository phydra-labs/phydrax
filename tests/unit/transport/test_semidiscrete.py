#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _finite(points, weights, *, normalized=True, mask=None, provenance="atoms"):
    return phx.integration.discrete(
        jnp.asarray(points, dtype=float),
        cx.Field(jnp.asarray(weights, dtype=float), dims=("atom",)),
        axes="atom",
        mask=(
            None
            if mask is None
            else cx.Field(jnp.asarray(mask, dtype=bool), dims=("atom",))
        ),
        normalized=normalized,
        provenance=provenance,
    )


def _uniform_problem(
    support,
    weights,
    *,
    order=32,
    normalized=True,
    lower=0.0,
    upper=1.0,
    mask=None,
    log_density=None,
):
    domain = phx.domain.ScalarInterval(lower, upper, label="x")
    base = phx.integration.over(domain.component())
    log_density = (
        domain.Function("x")(lambda x: jnp.zeros_like(x))
        if log_density is None
        else log_density(domain)
    )
    source = (
        phx.integration.normalized_density(base, log_density)
        if normalized
        else phx.integration.density(base, log_density)
    )
    realization = phx.integration.materialize(
        source,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(order)),
    )
    target = _finite(
        support,
        weights,
        normalized=normalized,
        mask=mask,
        provenance="quantization-support",
    )
    return phx.transport.semidiscrete_problem(
        source,
        realization,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )


def _solver(*, tolerance=1e-9, iterations=200):
    return phx.transport.SemidiscreteSinkhorn(
        0.08,
        tolerance=tolerance,
        max_iterations=iterations,
        check_every=1,
        store_history=True,
    )


def test_uniform_density_to_one_atom_matches_analytic_cost_and_is_approximate():
    problem = _uniform_problem([0.5], [1.0], order=12)
    result = _solver()(problem)

    assert result.converged
    assert jnp.allclose(result.target_marginal, jnp.asarray([1.0]), atol=1e-10)
    assert jnp.allclose(result.transport_cost, 1.0 / 12.0, atol=1e-10)
    assert jnp.allclose(result.regularization, 0.0, atol=1e-10)
    assert result.approximate
    assert result.provenance.approximation == "fixed-integration-realization"
    assert result.provenance.fixed_realization
    assert result.integration_status == int(phx.integration.IntegrationStatus.CONVERGED)


def test_two_atom_uniform_solution_is_symmetric_and_exposes_soft_c_transform():
    result = _solver()(_uniform_problem([0.25, 0.75], [0.5, 0.5], order=48))

    assert result.converged
    assert jnp.allclose(result.target_potential[0], result.target_potential[1], atol=1e-9)
    assert jnp.allclose(result.target_marginal, jnp.asarray([0.5, 0.5]), atol=1e-8)
    transformed = result.soft_c_transform(jnp.asarray([0.25, 0.5, 0.75]))
    assert transformed.shape == (3,)
    assert jnp.allclose(transformed[0], transformed[2], atol=1e-10)


def test_quadrature_refinement_changes_only_declared_integration_approximation():
    solver = _solver(tolerance=1e-10, iterations=300)
    low = solver(_uniform_problem([0.2, 0.8], [0.3, 0.7], order=3))
    medium = solver(_uniform_problem([0.2, 0.8], [0.3, 0.7], order=8))
    reference = solver(_uniform_problem([0.2, 0.8], [0.3, 0.7], order=48))

    assert jnp.abs(medium.regularized_cost - reference.regularized_cost) < jnp.abs(
        low.regularized_cost - reference.regularized_cost
    )
    assert low.provenance.approximation == reference.provenance.approximation
    assert low.integration_diagnostics.objective_num_evaluations < (
        reference.integration_diagnostics.objective_num_evaluations
    )


def test_normalized_and_unnormalized_density_preserve_physical_mass():
    normalized = _solver()(_uniform_problem([0.5], [1.0], normalized=True))
    physical = _solver()(
        _uniform_problem(
            [0.5],
            [2.0],
            normalized=False,
            lower=0.0,
            upper=2.0,
        )
    )

    assert jnp.allclose(normalized.source_mass, 1.0)
    assert jnp.allclose(physical.source_mass, 2.0)
    assert jnp.allclose(physical.target_marginal, jnp.asarray([2.0]))
    assert jnp.allclose(physical.transport_cost, 2.0 * (7.0 / 12.0), atol=1e-9)


def test_fixed_random_batch_has_common_random_number_replay_semantics():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    source = phx.integration.normalized_density(
        phx.integration.over(domain.component()),
        domain.Function("x")(lambda x: jnp.zeros_like(x)),
    )
    realization = phx.integration.materialize(
        source,
        phx.integration.MonteCarloPlan(1024),
        key=jr.key(17),
    )
    problem = phx.transport.semidiscrete_problem(
        source,
        realization,
        _finite([0.25, 0.75], [0.5, 0.5]),
        cost=phx.transport.SquaredEuclideanCost(),
    )
    solver = _solver(tolerance=2e-3)
    first = solver(problem)
    replay = solver(problem)

    assert jnp.array_equal(first.target_potential, replay.target_potential)
    assert jnp.array_equal(first.regularized_cost, replay.regularized_cost)
    assert first.provenance.common_random_numbers
    assert first.provenance.deterministic_replay


def test_failed_density_integration_is_not_reported_as_transport_convergence():
    problem = _uniform_problem(
        [0.5],
        [1.0],
        order=8,
        normalized=False,
        log_density=lambda domain: domain.Function("x")(
            lambda x: jnp.where(x > 0.0, jnp.nan, 0.0)
        ),
    )
    result = _solver()(problem)

    assert result.integration_status == int(
        phx.integration.IntegrationStatus.NONFINITE_INTEGRAND
    )
    assert result.diagnostics.status == int(
        phx.transport.TransportStatus.INTEGRATION_FAILURE
    )
    assert not result.converged


def test_mass_mismatch_is_distinct_from_integration_failure():
    problem = _uniform_problem(
        [0.5],
        [3.0],
        normalized=False,
        lower=0.0,
        upper=2.0,
    )
    result = _solver()(problem)

    assert result.integration_status == int(phx.integration.IntegrationStatus.CONVERGED)
    assert result.diagnostics.status == int(phx.transport.TransportStatus.MASS_MISMATCH)
    assert not result.converged


def test_transport_nonconvergence_does_not_overwrite_successful_integration():
    result = phx.transport.SemidiscreteSinkhorn(
        0.08,
        max_iterations=1,
        tolerance=0.0,
    )(_uniform_problem([0.1, 0.9], [0.2, 0.8], order=24))

    assert result.integration_status == int(phx.integration.IntegrationStatus.CONVERGED)
    assert result.diagnostics.status == int(
        phx.transport.TransportStatus.MAXIMUM_ITERATIONS_REACHED
    )
    assert not result.converged


def test_support_gradient_is_finite_symmetric_and_jittable():
    problem = _uniform_problem([0.2, 0.8], [0.5, 0.5], order=32)
    solver = _solver(iterations=100)

    def objective(support):
        return solver(problem.with_target_support(support)).regularized_cost

    gradient = eqx.filter_jit(jax.grad(objective))(problem.target_support)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient[0], -gradient[1], atol=1e-8)


def test_quantizer_composes_bounded_parameterization_without_clipping():
    problem = _uniform_problem([0.2, 0.8], [0.5, 0.5], order=24)
    quantizer = phx.transport.SemidiscreteQuantizer(
        _solver(tolerance=1e-7, iterations=120),
        optax.adam(2e-2),
        num_steps=3,
        support_transform=jax.nn.sigmoid,
    )
    result = quantizer(problem, initial_parameters=jnp.asarray([-1.0, 1.0]))

    assert result.converged
    assert jnp.all(result.support > 0.0)
    assert jnp.all(result.support < 1.0)
    assert result.diagnostics.constrained
    assert result.diagnostics.objective_history.shape == (3,)
    assert (
        result.transport.problem.provenance.realization == problem.provenance.realization
    )


def test_quantizer_rejects_nonconverged_transport_as_a_training_objective():
    problem = _uniform_problem([0.1, 0.9], [0.2, 0.8], order=16)
    quantizer = phx.transport.SemidiscreteQuantizer(
        phx.transport.SemidiscreteSinkhorn(
            0.08,
            max_iterations=1,
            tolerance=0.0,
        ),
        optax.sgd(1e-2),
        num_steps=1,
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="requires converged"):
        value = quantizer.objective(problem, problem.target_support)
        jax.block_until_ready(value)


def test_masks_and_event_shape_survive_semidiscrete_solve():
    domain = phx.domain.HyperRectangle([0.0, 0.0], [1.0, 1.0], label="x")
    source = phx.integration.normalized_density(
        phx.integration.over(domain.component()),
        domain.Function("x")(lambda x: jnp.zeros(x.shape[:-1])),
    )
    realization = phx.integration.materialize(
        source,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )
    target = _finite(
        [[0.25, 0.5], [0.75, 0.5], [jnp.nan, jnp.nan]],
        [0.5, 0.5, jnp.nan],
        mask=[True, True, False],
    )
    problem = phx.transport.semidiscrete_problem(
        source,
        realization,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )
    result = _solver(tolerance=1e-7)(problem)

    assert result.converged
    assert result.target_event_shape == (2,)
    assert result.target_support.shape == (3, 2)
    assert jnp.array_equal(result.target_mask, jnp.asarray([True, True, False]))
    assert result.target_weights[-1] == 0.0
    assert result.target_marginal[-1] == 0.0


def test_solver_batches_over_support_while_sharing_one_realization():
    problem = _uniform_problem([0.25, 0.75], [0.5, 0.5], order=24)
    supports = jnp.asarray([[0.2, 0.8], [0.3, 0.7]])
    solver = _solver(tolerance=1e-7, iterations=100)
    costs, statuses = jax.vmap(
        lambda support: (
            solver(problem.with_target_support(support)).regularized_cost,
            solver(problem.with_target_support(support)).diagnostics.status,
        )
    )(supports)

    assert costs.shape == (2,)
    assert jnp.all(jnp.isfinite(costs))
    assert jnp.all(statuses == int(phx.transport.TransportStatus.CONVERGED))


def test_public_semidiscrete_catalog_is_intentional_and_complete():
    expected = {
        "SemidiscreteIntegrationDiagnostics",
        "SemidiscreteProblemProvenance",
        "SemidiscreteQuantizationDiagnostics",
        "SemidiscreteQuantizationResult",
        "SemidiscreteQuantizer",
        "SemidiscreteSinkhorn",
        "SemidiscreteTransportDiagnostics",
        "SemidiscreteTransportProblem",
        "SemidiscreteTransportProvenance",
        "SemidiscreteTransportResult",
        "semidiscrete_problem",
    }
    assert expected <= set(phx.transport.__all__)
    assert all(vars(phx.transport)[name] is not None for name in expected)
