import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _binary_pair(log_potentials=None):
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(2,), num_states=2)
    values = (
        jnp.asarray([[[0.0, -1.0], [-1.0, 0.0]]])
        if log_potentials is None
        else jnp.asarray(log_potentials)
    )
    factor = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        values,
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (factor,))


def test_factor_graph_log_score_pack_and_topology_are_stable():
    graph = _binary_pair()
    assignments = phx.pgm.pack_assignments(
        graph,
        {"x": jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])},
    )

    assert jnp.array_equal(
        phx.pgm.factor_graph_log_score(graph, assignments),
        jnp.asarray([0.0, -1.0, -1.0, 0.0]),
    )
    assert phx.pgm.factor_graph_contains(graph, assignments).all()
    assert graph.topology.graph.num_nodes == 3
    assert graph.topology.graph.num_edges == 4
    assert (
        graph.structure_id
        == _binary_pair(jnp.asarray([[[2.0, 1.0], [1.0, 2.0]]])).structure_id
    )
    assert phx.pgm.unpack_assignments(graph, assignments)["x"].shape == (4, 2)
    evidence = phx.pgm.pack_evidence(
        graph,
        {"x": jnp.zeros((2, 2))},
    )
    assert evidence.values.shape == (4,)
    assert jnp.allclose(
        jax.jit(lambda state: phx.pgm.factor_graph_log_score(graph, state))(assignments),
        phx.pgm.factor_graph_log_score(graph, assignments),
    )


def test_exact_enumeration_returns_normalizer_marginals_and_deterministic_map():
    graph = _binary_pair()
    result = phx.pgm.enumerate_factor_graph(graph)
    expected_log_normalizer = jnp.log(2.0 + 2.0 * jnp.exp(-1.0))

    assert result.successful
    assert result.optimal
    assert result.log_normalizer == pytest.approx(float(expected_log_normalizer))
    assert jnp.allclose(result.variable_probabilities.values, 0.5)
    assert jnp.array_equal(result.map_assignment, jnp.asarray([0, 0]))
    assert result.map_log_score == 0.0
    assert result.feasible_configurations == 4
    assert result.factor_probabilities[0].shape == (1, 2, 2)


def test_exact_mixed_cardinality_and_hard_support_match_independent_reference():
    variables = phx.pgm.DiscreteVariableGroup(
        "x",
        shape=(2,),
        num_states=jnp.asarray([2, 3]),
    )
    configs = jnp.asarray([[0, 0], [0, 2], [1, 1]], dtype=jnp.int32)
    values = jnp.asarray([[0.0, -0.5, 1.0]])
    factor = phx.pgm.EnumeratedFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        configs,
        values,
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    result = phx.pgm.enumerate_factor_graph(graph)
    expected = np.log(np.exp(0.0) + np.exp(-0.5) + np.exp(1.0))

    assert result.log_normalizer == pytest.approx(expected)
    assert result.feasible_configurations == 3
    assert jnp.array_equal(result.map_assignment, jnp.asarray([1, 1]))
    assert result.variable_probabilities.values.shape == (5,)


def test_infeasible_graph_and_resource_cap_fail_closed():
    graph = _binary_pair(jnp.full((1, 2, 2), -jnp.inf))
    result = phx.pgm.enumerate_factor_graph(graph)

    assert not result.valid
    assert result.status == int(phx.pgm.ExactFactorGraphStatus.INFEASIBLE)
    assert jnp.isneginf(result.log_normalizer)
    assert jnp.all(result.variable_probabilities.values == 0)

    with pytest.raises(ValueError, match="exceeding max_configurations"):
        phx.pgm.enumerate_factor_graph(_binary_pair(), max_configurations=3)


def test_factor_graph_rejects_duplicate_scope_and_numerical_contract_violations():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(2,), num_states=2)
    with pytest.raises(ValueError, match="repeat a variable"):
        phx.pgm.DiscreteFactorGraph(
            (variables,),
            (
                phx.pgm.DenseTableFactorGroup(
                    (
                        phx.pgm.VariableSelection(variables, [0]),
                        phx.pgm.VariableSelection(variables, [0]),
                    ),
                    jnp.zeros((1, 2, 2)),
                ),
            ),
        )
    with pytest.raises(ValueError, match="finite values and -inf"):
        phx.pgm.DenseTableFactorGroup(
            (phx.pgm.VariableSelection.all(variables),),
            jnp.asarray([[0.0, jnp.inf], [0.0, 0.0]]),
        )
