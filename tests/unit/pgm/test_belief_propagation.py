from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _chain_graph(length: Any = 4) -> Any:
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


def test_belief_propagation_scenario_1() -> None:
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
    # ty: ignore[unresolved-attribute]
    assert result.marginals_exact
    # ty: ignore[unresolved-attribute]
    assert result.log_normalizer_exact
    # ty: ignore[unresolved-attribute]
    assert result.log_normalizer_kind == "exact"
    assert jnp.allclose(
        # ty: ignore[unresolved-attribute]
        jnp.exp(result.variable_log_probabilities.values),
        exact.variable_probabilities.values,
        atol=1e-10,
    )
    # ty: ignore[unresolved-attribute]
    assert result.log_normalizer == pytest.approx(float(exact.log_normalizer), abs=1e-10)
    for inferred, expected in zip(
        # ty: ignore[unresolved-attribute]
        result.factor_probabilities,
        exact.factor_probabilities,
    ):
        assert jnp.allclose(inferred, expected, atol=1e-10)
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

    # ty: ignore[unresolved-attribute]
    assert result.map_available
    # ty: ignore[unresolved-attribute]
    assert result.optimal
    # ty: ignore[unresolved-attribute]
    assert jnp.array_equal(result.map_assignment, exact.map_assignment)
    # ty: ignore[unresolved-attribute]
    assert result.map_log_score == pytest.approx(float(exact.map_log_score))
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
    # ty: ignore[unresolved-attribute]
    assert not result.marginals_exact
    # ty: ignore[unresolved-attribute]
    assert not result.log_normalizer_exact
    # ty: ignore[unresolved-attribute]
    assert result.log_normalizer_kind == "bethe"
    # ty: ignore[unresolved-attribute]
    assert jnp.all(jnp.isfinite(result.variable_log_probabilities.values))
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
    # ty: ignore[unresolved-attribute]
    assert jnp.exp(result.variable_log_probabilities.values[0]) == pytest.approx(1.0)
    # ty: ignore[unresolved-attribute]
    assert jnp.isneginf(result.variable_log_probabilities.values[1])


def test_sum_product_log_normalizer_gradient_matches_exact_factor_marginals() -> None:
    graph = _chain_graph(2)
    prepared = phx.pgm.prepare_belief_propagation(graph)
    base_table = graph.factor_groups[1].log_potentials

    def inferred_log_normalizer(table: Any) -> Any:
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
        # ty: ignore[unresolved-attribute]
        return result.log_normalizer

    gradient = jax.grad(inferred_log_normalizer)(base_table)
    exact = phx.pgm.enumerate_factor_graph(graph)

    assert jnp.allclose(gradient, exact.factor_probabilities[1], atol=1e-9)
