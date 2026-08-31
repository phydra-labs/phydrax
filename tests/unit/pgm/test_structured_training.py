import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_ising_and_potts_constructors_preserve_declared_scores_and_cardinality():
    ising = phx.pgm.ising_factor_graph(
        jnp.asarray([0.2, -0.1]),
        jnp.asarray([[0, 1]]),
        jnp.asarray([0.4]),
    )
    states = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    spins = 2 * states - 1
    expected = 0.2 * spins[:, 0] - 0.1 * spins[:, 1] + 0.4 * spins[:, 0] * spins[:, 1]

    assert jnp.allclose(phx.pgm.factor_graph_log_score(ising, states), expected)

    cardinality = 300
    unary = jnp.zeros((1, cardinality)).at[0, 280].set(5.0)
    potts = phx.pgm.potts_factor_graph(
        unary,
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros((0, cardinality, cardinality)),
    )
    exact = phx.pgm.enumerate_factor_graph(potts, max_configurations=cardinality)

    assert exact.map_assignment.dtype == jnp.int32
    assert int(exact.map_assignment[0]) == 280


def test_logical_and_cardinality_factors_match_declared_hard_semantics():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(3,), num_states=2)
    logical = phx.pgm.LogicalFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        phx.pgm.VariableSelection(variables, [2]),
        kind="or",
    )
    cardinality = phx.pgm.BinaryCardinalityFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        jnp.asarray([[-jnp.inf, 0.0, -jnp.inf]]),
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (logical, cardinality))

    assert jnp.isfinite(phx.pgm.factor_graph_log_score(graph, jnp.asarray([1, 0, 1])))
    assert jnp.isneginf(phx.pgm.factor_graph_log_score(graph, jnp.asarray([1, 1, 1])))
    assert jnp.isneginf(phx.pgm.factor_graph_log_score(graph, jnp.asarray([1, 0, 0])))


def test_exact_likelihood_and_contrastive_divergence_have_correct_values_and_gradients():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(1,), num_states=2)
    factor = phx.pgm.DenseTableFactorGroup(
        (phx.pgm.VariableSelection.all(variables),),
        jnp.asarray([[0.2, -0.3]]),
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    exact = phx.pgm.enumerate_factor_graph(graph)
    objective, diagnostics = phx.pgm.exact_factor_graph_negative_log_likelihood(
        graph,
        jnp.asarray([[0], [0], [1]]),
        exact,
    )
    expected = exact.log_normalizer - (0.2 + 0.2 - 0.3) / 3

    assert objective == pytest.approx(float(expected))
    assert diagnostics.valid

    cd, cd_diagnostics = phx.pgm.contrastive_divergence_loss(
        graph,
        jnp.asarray([[0], [0]]),
        jnp.asarray([[1], [1]]),
    )
    assert cd == pytest.approx(-0.5)
    assert cd_diagnostics.valid

    base = factor.log_potentials

    def loss(table):
        updated_graph = eqx.tree_at(
            lambda value: value.factor_groups[0].log_potentials,
            graph,
            table,
        )
        updated_exact = phx.pgm.enumerate_factor_graph(updated_graph)
        return phx.pgm.exact_factor_graph_negative_log_likelihood(
            updated_graph,
            jnp.asarray([[0]]),
            updated_exact,
        )[0]

    gradient = jax.grad(loss)(base)
    expected_gradient = exact.factor_probabilities[0] - jnp.asarray([[1.0, 0.0]])
    assert jnp.allclose(gradient, expected_gradient, atol=1e-10)


def test_factor_graph_moments_return_empirical_configuration_probabilities():
    graph = phx.pgm.ising_factor_graph(
        jnp.zeros((2,)),
        jnp.asarray([[0, 1]]),
        jnp.asarray([0.0]),
    )
    moments = phx.pgm.factor_graph_moments(
        graph,
        jnp.asarray([[0, 0], [0, 1], [0, 1], [1, 1]]),
    )

    assert moments[0].shape == (2, 2)
    assert moments[1].shape == (1, 2, 2)
    assert jnp.allclose(moments[1][0], jnp.asarray([[0.25, 0.5], [0.0, 0.25]]))
