#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _measure(points, weights, *, mask=None, provenance="unbalanced-test"):
    weight_field = cx.Field(jnp.asarray(weights, dtype=float), dims=("atom",))
    mask_field = (
        None if mask is None else cx.Field(jnp.asarray(mask, dtype=bool), dims=("atom",))
    )
    return phx.integration.discrete(
        jnp.asarray(points, dtype=float),
        weight_field,
        axes="atom",
        mask=mask_field,
        normalized=False,
        provenance=provenance,
    )


def _problem(
    source,
    target,
    *,
    source_penalty=1.3,
    target_penalty=2.1,
    cost=None,
):
    return phx.transport.unbalanced_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost() if cost is None else cost,
        source_marginal_penalty=source_penalty,
        target_marginal_penalty=target_penalty,
    )


def _solver(
    *, epsilon=0.7, block_size=None, max_iterations=800, tolerance=1e-11, **kwargs
):
    return phx.transport.UnbalancedSinkhorn(
        epsilon,
        block_size=block_size,
        max_iterations=max_iterations,
        tolerance=tolerance,
        check_every=2,
        early_stop=False,
        store_history=True,
        **kwargs,
    )


def test_one_atom_unequal_mass_matches_generalized_kl_analytic_solution():
    source_mass = 2.0
    target_mass = 5.0
    cost = 3.0
    epsilon = 0.7
    source_penalty = 1.2
    target_penalty = 2.1
    problem = _problem(
        _measure([[0.0]], [source_mass]),
        _measure([[1.0]], [target_mass]),
        source_penalty=source_penalty,
        target_penalty=target_penalty,
        cost=phx.transport.PrecomputedCost([[cost]]),
    )
    result = _solver(epsilon=epsilon)(problem)
    expected_log_mass = (
        epsilon * jnp.log(source_mass * target_mass)
        + source_penalty * jnp.log(source_mass)
        + target_penalty * jnp.log(target_mass)
        - cost
    ) / (epsilon + source_penalty + target_penalty)
    expected_mass = jnp.exp(expected_log_mass)

    assert result.converged
    assert jnp.allclose(result.dense_plan()[0, 0], expected_mass, rtol=1e-10)
    assert jnp.allclose(result.source_marginal(), jnp.asarray([expected_mass]))
    assert jnp.allclose(result.target_marginal(), jnp.asarray([expected_mass]))
    assert jnp.allclose(result.transported_mass, expected_mass)
    assert jnp.allclose(
        result.regularized_cost,
        result.transport_cost
        + result.entropy_regularization
        + result.source_marginal_regularization
        + result.target_marginal_regularization,
    )
    assert jnp.allclose(result.regularized_cost, result.dual_cost, atol=1e-9)


def test_asymmetric_penalties_have_oriented_physical_semantics():
    source = _measure([[0.0]], [2.0])
    target = _measure([[1.0]], [5.0])
    source_relaxed = _solver()(
        _problem(source, target, source_penalty=0.2, target_penalty=4.0)
    )
    target_relaxed = _solver()(
        _problem(source, target, source_penalty=4.0, target_penalty=0.2)
    )

    assert source_relaxed.converged & target_relaxed.converged
    assert not jnp.allclose(
        source_relaxed.transported_mass,
        target_relaxed.transported_mass,
    )
    assert source_relaxed.problem.source_marginal_penalty == 0.2
    assert source_relaxed.problem.target_marginal_penalty == 4.0


def test_large_marginal_penalties_recover_balanced_sinkhorn_for_unit_mass():
    source = phx.integration.discrete(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        cx.Field(jnp.asarray([0.2, 0.5, 0.3]), dims=("atom",)),
        axes="atom",
        normalized=True,
    )
    target = phx.integration.discrete(
        jnp.asarray([[0.2], [1.5]]),
        cx.Field(jnp.asarray([0.6, 0.4]), dims=("atom",)),
        axes="atom",
        normalized=True,
    )
    cost = phx.transport.SquaredEuclideanCost()
    balanced_problem = phx.transport.discrete_problem(source, target, cost=cost)
    balanced = phx.transport.Sinkhorn(
        0.5,
        max_iterations=1000,
        tolerance=1e-11,
        check_every=2,
    )(balanced_problem)
    unbalanced = _solver(epsilon=0.5)(
        _problem(
            source,
            target,
            source_penalty=1e8,
            target_penalty=1e8,
            cost=cost,
        )
    )

    assert balanced.converged & unbalanced.converged
    assert jnp.allclose(
        unbalanced.dense_plan(), balanced.dense_plan(), rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_dense_and_blockwise_unbalanced_solutions_and_actions_agree(block_size):
    source = _measure(
        jnp.linspace(-1.0, 1.0, 5)[:, None],
        [1.0, 2.0, 0.5, 3.0, 1.5],
    )
    target = _measure(
        jnp.linspace(-0.8, 1.4, 7)[:, None],
        [1.0, 0.5, 2.0, 1.0, 0.2, 0.8, 1.5],
    )
    problem = _problem(source, target)
    dense = _solver()(problem)
    blockwise = _solver(block_size=block_size)(problem)
    payload = jnp.arange(15.0).reshape((5, 3))
    target_payload = jnp.arange(14.0).reshape((7, 2))

    assert dense.converged & blockwise.converged
    assert blockwise.provenance.execution == "blockwise"
    assert jnp.allclose(blockwise.dense_plan(), dense.dense_plan(), rtol=1e-9, atol=1e-10)
    assert jnp.allclose(
        blockwise.apply_source_to_target(payload),
        dense.apply_source_to_target(payload),
        rtol=1e-9,
        atol=1e-10,
    )
    assert jnp.allclose(
        blockwise.apply_target_to_source(target_payload),
        dense.apply_target_to_source(target_payload),
        rtol=1e-9,
        atol=1e-10,
    )
    assert jnp.allclose(blockwise.regularized_cost, dense.regularized_cost, rtol=1e-9)


def test_masked_atoms_remain_zero_without_changing_static_plan_shape():
    source = _measure(
        [[0.0], [jnp.nan], [2.0]],
        [1.0, jnp.nan, 3.0],
        mask=[True, False, True],
    )
    target = _measure(
        [[0.5], [1.5], [jnp.nan]],
        [2.0, 1.0, jnp.nan],
        mask=[True, True, False],
    )
    result = _solver()(_problem(source, target))
    plan = result.dense_plan()

    assert result.converged
    assert plan.shape == (3, 3)
    assert jnp.array_equal(plan[1], jnp.zeros((3,)))
    assert jnp.array_equal(plan[:, 2], jnp.zeros((3,)))
    assert result.problem.source.event_shape == (1,)


def test_joint_mass_scaling_follows_declared_product_reference_kl_convention():
    epsilon = 0.7
    source_penalty = 1.3
    target_penalty = 2.1
    scale = 4.0
    base = _solver(epsilon=epsilon)(
        _problem(
            _measure([[0.0]], [2.0]),
            _measure([[1.0]], [5.0]),
            source_penalty=source_penalty,
            target_penalty=target_penalty,
        )
    )
    scaled = _solver(epsilon=epsilon)(
        _problem(
            _measure([[0.0]], [scale * 2.0]),
            _measure([[1.0]], [scale * 5.0]),
            source_penalty=source_penalty,
            target_penalty=target_penalty,
        )
    )
    exponent = (2.0 * epsilon + source_penalty + target_penalty) / (
        epsilon + source_penalty + target_penalty
    )

    assert jnp.allclose(
        scaled.transported_mass,
        scale**exponent * base.transported_mass,
        rtol=1e-10,
    )


def test_unbalanced_solver_is_jittable_vmappable_and_differentiable():
    source = _measure([[0.0], [1.0]], [1.0, 2.0])
    target = _measure([[0.2], [1.5]], [0.5, 1.0])
    solver = _solver(max_iterations=100)

    def objective(cost_scale):
        problem = _problem(
            source,
            target,
            cost=phx.transport.PrecomputedCost(
                cost_scale * jnp.asarray([[0.04, 2.25], [0.64, 0.25]])
            ),
        )
        return solver(problem).regularized_cost

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    mapped = jax.vmap(objective)(jnp.asarray([0.5, 1.0, 2.0]))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert mapped.shape == (3,)
    assert jnp.all(jnp.isfinite(mapped))


def test_nonconvergence_and_transport_mass_collapse_have_distinct_statuses():
    problem = _problem(
        _measure([[0.0], [1.0]], [1.0, 2.0]),
        _measure([[10.0], [12.0]], [2.0, 1.0]),
    )
    nonconverged = _solver(max_iterations=1, tolerance=0.0)(problem)
    collapsed_problem = _problem(
        _measure([[0.0]], [1.0]),
        _measure([[1.0]], [1.0]),
        cost=phx.transport.PrecomputedCost([[1e6]]),
    )
    collapsed = _solver(
        epsilon=0.01,
        mass_collapse_tolerance=1e-300,
    )(collapsed_problem)

    assert not nonconverged.converged
    assert nonconverged.diagnostics.status == int(
        phx.transport.TransportStatus.MAXIMUM_ITERATIONS_REACHED
    )
    assert not collapsed.converged
    assert collapsed.mass_collapsed
    assert collapsed.diagnostics.status == int(
        phx.transport.TransportStatus.TRANSPORT_MASS_COLLAPSED
    )


def test_unbalanced_divergence_keeps_three_solves_mass_correction_and_prepared_target():
    source = _measure([[0.0], [1.0]], [1.0, 2.0])
    target = _measure([[0.5], [1.5]], [2.0, 3.0])
    solver = _solver()
    problem = _problem(source, target)
    result = phx.transport.unbalanced_sinkhorn_divergence(problem, solver)
    expected_correction = 0.5 * solver.epsilon * (3.0 - 5.0) ** 2
    reference = phx.transport.prepare_unbalanced_sinkhorn_reference(
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=solver,
        source_marginal_penalty=1.3,
        target_marginal_penalty=2.1,
    )
    prepared = phx.transport.unbalanced_sinkhorn_divergence_against(source, reference)
    identical = phx.transport.unbalanced_sinkhorn_divergence(
        _problem(source, source),
        solver,
    )

    assert result.converged & prepared.converged
    assert jnp.allclose(result.mass_correction, expected_correction)
    assert jnp.allclose(result.value, prepared.value, atol=1e-10)
    assert jnp.allclose(identical.value, 0.0, atol=1e-12)
    assert prepared.target_self is reference.target_self


def test_density_and_materialized_realization_inputs_preserve_physical_mass():
    base = _measure([[0.0], [1.0]], [1.0, 2.0], provenance="base-intensity")
    density = phx.integration.density(base, jnp.log(jnp.asarray([2.0, 3.0])))
    realization = phx.integration.materialize(base)
    target = _measure([[0.5]], [4.0])
    density_problem = _problem(density, target)
    realization_problem = _problem(realization, target)

    assert jnp.allclose(density_problem.source_mass, 8.0)
    assert jnp.allclose(realization_problem.source_mass, 3.0)
    assert density_problem.source.event_shape == realization_problem.source.event_shape


def test_uq_and_training_term_use_unbalanced_transport_only_for_physical_measures():
    source = _measure([[0.0], [1.0]], [1.0, 2.0])
    target = _measure([[0.5], [1.5]], [2.0, 3.0])
    solver = _solver()
    metric = phx.uq.spatial_unbalanced_sinkhorn_divergence(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=solver,
        source_marginal_penalty=1.3,
        target_marginal_penalty=2.1,
    )
    reference = phx.transport.prepare_unbalanced_sinkhorn_reference(
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=solver,
        source_marginal_penalty=1.3,
        target_marginal_penalty=2.1,
    )
    term = phx.terms.SpatialUnbalancedSinkhornDivergenceTerm(
        lambda _: source,
        reference,
        weight=2.0,
    )
    evaluation = term.term_evaluation({})

    assert metric.converged
    assert jnp.allclose(evaluation.value, 2.0 * metric.value)
    assert evaluation.diagnostics.cross.problem.source_mass == 3.0
    assert evaluation.diagnostics.cross.problem.target_mass == 5.0


def test_unbalanced_training_term_rejects_nonconverged_scientific_solve():
    source = _measure([[0.0], [1.0]], [1.0, 2.0])
    target = _measure([[10.0], [12.0]], [2.0, 3.0])
    reference = phx.transport.prepare_unbalanced_sinkhorn_reference(
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=_solver(),
        source_marginal_penalty=1.3,
        target_marginal_penalty=2.1,
    )
    bad_reference = eqx.tree_at(
        lambda item: item.solver,
        reference,
        _solver(max_iterations=1, tolerance=0.0),
    )
    term = phx.terms.SpatialUnbalancedSinkhornDivergenceTerm(
        lambda _: source,
        bad_reference,
    )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        jax.block_until_ready(term.term_evaluation({}).value)


def test_unbalanced_public_catalogs_are_explicit():
    transport_symbols = {
        "PreparedUnbalancedSinkhornReference",
        "UnbalancedSinkhorn",
        "UnbalancedSinkhornDiagnostics",
        "UnbalancedSinkhornDivergenceResult",
        "UnbalancedSinkhornResult",
        "UnbalancedTransportProblem",
        "prepare_unbalanced_sinkhorn_reference",
        "require_unbalanced_converged",
        "unbalanced_problem",
        "unbalanced_sinkhorn_divergence",
        "unbalanced_sinkhorn_divergence_against",
    }
    assert transport_symbols <= set(phx.transport.__all__)
    assert "spatial_unbalanced_sinkhorn_divergence" in phx.uq.__all__
    assert "SpatialUnbalancedSinkhornDivergenceTerm" in phx.terms.__all__
