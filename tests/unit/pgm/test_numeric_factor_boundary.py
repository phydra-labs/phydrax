import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import pgm


def _prepared():
    variables = pgm.DiscreteVariableGroup("states", num_states=np.asarray([2, 3]))
    factor = pgm.DenseTableFactorGroup(
        (pgm.VariableSelection(variables, [0]), pgm.VariableSelection(variables, [1])),
        np.asarray([[[0.1, -0.2, 0.3], [-0.1, 0.4, -0.3]]]),
    )
    graph = pgm.DiscreteFactorGraph((variables,), (factor,))
    bp = pgm.prepare_belief_propagation(
        graph,
        pgm.SumProductBeliefPropagation(
            absolute_tolerance=1e-12, relative_tolerance=1e-12
        ),
    )
    return graph, bp, pgm.initialize_belief_propagation(bp)


def test_dynamic_tables_have_exact_log_normalizer_derivatives_under_jit():
    graph, bp, initial = _prepared()
    exact = pgm.prepare_exact_factor_graph(graph)

    def exact_logz(table):
        return pgm.run_exact_factor_graph(exact, (table,)).log_normalizer

    def implicit_logz(table):
        updated = pgm.replace_belief_propagation_tables(bp, (table,))
        return pgm.run_implicit_belief_propagation(
            updated, initial
        ).inference.log_normalizer

    table = bp.factor_tables[0]
    exact_value, exact_gradient = jax.jit(jax.value_and_grad(exact_logz))(table)
    bp_value, bp_gradient = jax.jit(jax.value_and_grad(implicit_logz))(table)
    expected = jax.nn.softmax(table.reshape(-1)).reshape(table.shape)
    np.testing.assert_allclose(exact_gradient, expected, atol=1e-12)
    np.testing.assert_allclose(bp_value, exact_value, atol=1e-11)
    np.testing.assert_allclose(bp_gradient, expected, atol=2e-10)
    direction = jnp.linspace(-0.4, 0.6, 6).reshape(table.shape)
    _, tangent = jax.jit(
        lambda value, vector: jax.jvp(implicit_logz, (value,), (vector,))
    )(table, direction)
    np.testing.assert_allclose(tangent, jnp.sum(direction * expected), atol=2e-10)


def test_numeric_nonfinite_inputs_fail_explicitly_and_support_cannot_change():
    graph, bp, initial = _prepared()
    exact = pgm.prepare_exact_factor_graph(graph)
    bad_table = bp.factor_tables[0].at[0, 0, 1].set(jnp.nan)
    exact_result = jax.jit(lambda table: pgm.run_exact_factor_graph(exact, (table,)))(
        bad_table
    )
    bp_result = eqx.filter_jit(
        lambda table: pgm.run_implicit_belief_propagation(
            pgm.replace_belief_propagation_tables(bp, (table,)), initial
        )
    )(bad_table)
    assert not bool(exact_result.successful | bp_result.inference.successful)
    assert int(exact_result.status) == int(pgm.ExactFactorGraphStatus.NONFINITE_INPUT)
    assert int(bp_result.inference.status) == int(
        pgm.BeliefPropagationStatus.NONFINITE_INPUT
    )
    with pytest.raises(ValueError, match="shape"):
        pgm.replace_belief_propagation_tables(bp, (jnp.zeros((1, 2, 2)),))
    with pytest.raises(ValueError, match="shape"):
        pgm.run_exact_factor_graph(exact, (jnp.zeros((1, 2, 2)),))
    forbidden = jnp.full_like(bp.factor_tables[0], -jnp.inf)
    result = pgm.run_exact_factor_graph(exact, (forbidden,))
    assert int(result.status) == int(pgm.ExactFactorGraphStatus.INFEASIBLE)
    np.testing.assert_array_equal(result.variable_probabilities.values, 0.0)


def test_exact_preparation_preserves_existing_parameter_gradient_contract():
    graph, bp, _ = _prepared()

    def objective(table):
        updated = eqx.tree_at(
            lambda item: item.factor_groups[0].log_potentials, graph, table
        )
        return pgm.enumerate_factor_graph(updated).log_normalizer

    value, gradient = jax.value_and_grad(objective)(bp.factor_tables[0])
    expected = jax.nn.softmax(bp.factor_tables[0].reshape(-1)).reshape(
        bp.factor_tables[0].shape
    )
    np.testing.assert_allclose(gradient, expected, atol=1e-12)
    np.testing.assert_allclose(
        value, jax.scipy.special.logsumexp(bp.factor_tables[0]), atol=1e-12
    )


def test_table_replacement_rejects_structured_parameter_shortcuts():
    variable = pgm.DiscreteVariableGroup("spin", shape=(2,), num_states=2)
    factor = pgm.IsingFactorGroup(
        (pgm.VariableSelection(variable, [0]), pgm.VariableSelection(variable, [1])),
        [0.2],
    )
    prepared = pgm.prepare_belief_propagation(
        pgm.DiscreteFactorGraph((variable,), (factor,))
    )
    with pytest.raises(TypeError, match="dense/Potts"):
        pgm.replace_belief_propagation_tables(prepared, prepared.factor_tables)


def test_underflowed_finite_factor_beliefs_keep_log_normalizer_gradient():
    graph, bp, initial = _prepared()
    exact = pgm.prepare_exact_factor_graph(graph)
    table = jnp.asarray([[[-0.2, -1000.0, -2000.0], [-1000.0, -2000.0, -3000.0]]])

    def implicit_logz(value):
        prepared = pgm.replace_belief_propagation_tables(bp, (value,))
        return pgm.run_implicit_belief_propagation(
            prepared, initial
        ).inference.log_normalizer

    value, gradient = jax.jit(jax.value_and_grad(implicit_logz))(table)
    reference = pgm.run_exact_factor_graph(exact, (table,))
    np.testing.assert_allclose(value, reference.log_normalizer, atol=1e-10)
    np.testing.assert_allclose(gradient, reference.factor_probabilities[0], atol=1e-10)


def test_custom_kernel_factor_axis_matches_direct_and_packed_exact_inference():
    variables = pgm.DiscreteVariableGroup("kernel-states", shape=(3,), num_states=2)
    kernel = pgm.CallableFactorKernel(
        lambda parameters, states: parameters * states[..., 0],
        kernel_id="distinct-unary-factor-parameters",
        capabilities=pgm.FactorKernelCapabilities(sum_product=True, factor_beliefs=True),
    )
    parameters = jnp.asarray([-0.8, 0.3, 1.1])
    factor = pgm.KernelFactorGroup(
        (pgm.VariableSelection.all(variables),), kernel, parameters
    )
    graph = pgm.DiscreteFactorGraph((variables,), (factor,))
    shifted = eqx.tree_at(
        lambda value: value.factor_groups[0].parameters, graph, parameters + 0.4
    )
    assignments = pgm.enumerate_assignments(graph.cardinalities)
    direct_scores = pgm.factor_graph_log_score(graph, assignments)
    np.testing.assert_allclose(
        direct_scores, jnp.sum(parameters * assignments, axis=-1), atol=1e-12
    )
    packed_results = pgm.enumerate_packed_factor_graphs(
        pgm.pack_factor_graphs((graph, shifted))
    )
    for current_graph, result in zip((graph, shifted), packed_results):
        scores = pgm.factor_graph_log_score(current_graph, assignments)
        probabilities = jax.nn.softmax(scores)
        expected_marginals = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        jnp.sum(
                            jnp.where(assignments[:, index] == state, probabilities, 0.0)
                        )
                        for state in range(2)
                    )
                )
                for index in range(3)
            )
        )
        assert bool(result.successful)
        np.testing.assert_allclose(
            result.log_normalizer, jax.scipy.special.logsumexp(scores), atol=1e-12
        )
        np.testing.assert_allclose(
            result.factor_probabilities[0], expected_marginals, atol=1e-12
        )

    def log_normalizer(values):
        current_graph = eqx.tree_at(
            lambda value: value.factor_groups[0].parameters, graph, values
        )
        return pgm.enumerate_factor_graph(current_graph).log_normalizer

    np.testing.assert_allclose(
        jax.grad(log_normalizer)(parameters), jax.nn.sigmoid(parameters), atol=1e-12
    )
