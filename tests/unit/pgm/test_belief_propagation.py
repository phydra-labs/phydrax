import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _chain_graph(length=4):
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(length,), num_states=2)
    unary = phx.pgm.DenseTableFactorGroup(
        (phx.pgm.VariableSelection.all(variables),),
        jnp.stack([jnp.asarray([0.2 * index, -0.1 * index]) for index in range(length)]),
    )
    edges = jnp.stack([jnp.arange(length - 1), jnp.arange(1, length)], axis=-1)
    pairwise = jnp.stack(
        [jnp.asarray([[0.3, -0.2], [-0.2, 0.4]]) for _ in range(length - 1)]
    )
    interactions = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        pairwise,
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (unary, interactions))


def test_forest_sum_product_matches_exact_marginals_factor_beliefs_and_normalizer():
    graph = _chain_graph()
    exact = phx.pgm.enumerate_factor_graph(graph)
    prepared = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(),
    )
    result = phx.pgm.run_belief_propagation(
        prepared,
        phx.pgm.initialize_belief_propagation(prepared),
    )

    assert prepared.forest
    assert result.successful
    assert result.marginals_exact
    assert result.log_normalizer_exact
    assert result.log_normalizer_kind == "exact"
    assert jnp.allclose(
        jnp.exp(result.variable_log_probabilities.values),
        exact.variable_probabilities.values,
        atol=1e-10,
    )
    assert result.log_normalizer == pytest.approx(float(exact.log_normalizer), abs=1e-10)
    for inferred, expected in zip(
        result.factor_probabilities, exact.factor_probabilities
    ):
        assert jnp.allclose(inferred, expected, atol=1e-10)


def test_forest_max_product_backtracks_consistent_exact_map():
    graph = _chain_graph()
    exact = phx.pgm.enumerate_factor_graph(graph)
    prepared = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.MaxProductBeliefPropagation(),
    )
    result = phx.pgm.run_belief_propagation(
        prepared,
        phx.pgm.initialize_belief_propagation(prepared),
    )

    assert result.map_available
    assert result.optimal
    assert jnp.array_equal(result.map_assignment, exact.map_assignment)
    assert result.map_log_score == pytest.approx(float(exact.map_log_score))


def test_loopy_result_reports_fixed_point_without_claiming_exactness():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(3,), num_states=2)
    edges = jnp.asarray([[0, 1], [1, 2], [2, 0]])
    factor = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        jnp.broadcast_to(jnp.asarray([[0.2, -0.1], [-0.1, 0.2]]), (3, 2, 2)),
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    prepared = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(maximum_steps=50, relaxation=0.7),
    )
    result = phx.pgm.run_belief_propagation(
        prepared,
        phx.pgm.initialize_belief_propagation(prepared),
    )

    assert not prepared.forest
    assert result.successful
    assert not result.marginals_exact
    assert not result.log_normalizer_exact
    assert result.log_normalizer_kind == "bethe"
    assert jnp.all(jnp.isfinite(result.variable_log_probabilities.values))


def test_hard_evidence_remains_exact_and_never_creates_nan_messages():
    graph = _chain_graph(2)
    evidence = phx.pgm.pack_evidence(
        graph,
        jnp.asarray([0.0, -jnp.inf, 0.0, 0.0]),
    )
    prepared = phx.pgm.prepare_belief_propagation(graph)
    result = phx.pgm.run_belief_propagation(
        prepared,
        phx.pgm.initialize_belief_propagation(prepared, evidence=evidence),
    )

    assert result.successful
    assert not jnp.any(jnp.isnan(result.state.messages))
    assert jnp.exp(result.variable_log_probabilities.values[0]) == pytest.approx(1.0)
    assert jnp.isneginf(result.variable_log_probabilities.values[1])


def test_sum_product_log_normalizer_gradient_matches_exact_factor_marginals():
    graph = _chain_graph(2)
    prepared = phx.pgm.prepare_belief_propagation(graph)
    base_table = graph.factor_groups[1].log_potentials

    def inferred_log_normalizer(table):
        updated_graph = eqx.tree_at(
            lambda value: value.factor_groups[1].log_potentials,
            graph,
            table,
        )
        refreshed = phx.pgm.refresh_belief_propagation(prepared, updated_graph)
        result = phx.pgm.run_belief_propagation(
            refreshed,
            phx.pgm.initialize_belief_propagation(refreshed),
        )
        return result.log_normalizer

    gradient = jax.grad(inferred_log_normalizer)(base_table)
    exact = phx.pgm.enumerate_factor_graph(graph)

    assert jnp.allclose(gradient, exact.factor_probabilities[1], atol=1e-9)
