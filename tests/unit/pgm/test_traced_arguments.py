from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _binary_pair(log_potentials: Any = None) -> Any:
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


def _chain_graph(length: Any = 3) -> Any:
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(length,), num_states=2)
    unary = phx.pgm.DenseTableFactorGroup(
        (phx.pgm.VariableSelection.all(variables),),
        jnp.stack([jnp.asarray([0.2 * index, -0.1 * index]) for index in range(length)]),
    )
    edges = jnp.stack([jnp.arange(length - 1), jnp.arange(1, length)], axis=-1)
    pairwise = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        jnp.broadcast_to(
            jnp.asarray([[0.3, -0.2], [-0.2, 0.4]]),
            (length - 1, 2, 2),
        ),
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (unary, pairwise))


def _triangle_graph() -> Any:
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(3,), num_states=2)
    edges = jnp.asarray([[0, 1], [1, 2], [2, 0]])
    factor = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        jnp.broadcast_to(jnp.asarray([[0.4, -0.2], [-0.2, 0.4]]), (3, 2, 2)),
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (factor,))


def test_traced_arguments_scenario_1() -> None:
    graph = _binary_pair()
    evidence = jnp.asarray([0.1, -0.2, 0.3, -0.4])
    elimination = phx.pgm.plan_variable_elimination(graph)
    junction = phx.pgm.plan_junction_tree(elimination)

    packed = eqx.filter_jit(lambda value, raw: phx.pgm.pack_evidence(value, raw).values)(
        graph, evidence
    )
    eliminated = eqx.filter_jit(
        lambda plan, raw: (
            phx.pgm.variable_elimination(
                plan,
                evidence=raw,
            ).log_normalizer
        )
    )(elimination, packed)
    calibrated = eqx.filter_jit(
        lambda plan, raw: (
            phx.pgm.junction_tree_calibrate(
                plan,
                evidence=raw,
            ).elimination.log_normalizer
        )
    )(junction, packed)
    expected = phx.pgm.variable_elimination(elimination, evidence=evidence).log_normalizer

    assert jnp.allclose(eliminated, expected)
    assert jnp.allclose(calibrated, expected)
    law = phx.pgm.NormalizedFactorGraphLaw(
        phx.pgm.plan_variable_elimination(_binary_pair())
    )
    key = jax.random.key(3)

    compiled = eqx.filter_jit(lambda value, sample_key: value.sample(sample_key, (5,)))(
        law, key
    )

    assert jnp.array_equal(compiled, law.sample(key, (5,)))
    empty_plan = phx.pgm.plan_variable_elimination(phx.pgm.DiscreteFactorGraph(()))
    empty = eqx.filter_jit(phx.pgm.variable_elimination)(empty_plan)
    assert empty.log_normalizer == 0.0
    assert empty.map_assignment.shape == (0,)

    variables = phx.pgm.DiscreteVariableGroup(
        "x", shape=(2,), num_states=jnp.asarray([2, 3])
    )
    factor = phx.pgm.EnumeratedFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        jnp.asarray([[0, 0], [0, 2], [1, 1]], dtype=jnp.int32),
        jnp.asarray([[0.0, -0.5, 1.0]]),
    )
    plan = phx.pgm.plan_variable_elimination(
        phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    )
    eager = phx.pgm.variable_elimination(plan)
    compiled = eqx.filter_jit(phx.pgm.variable_elimination)(plan)

    assert jnp.allclose(compiled.log_normalizer, eager.log_normalizer)
    assert jnp.array_equal(compiled.map_assignment, eager.map_assignment)


def test_traced_arguments_scenario_2() -> None:
    graph = _chain_graph()
    prepared = phx.pgm.prepare_belief_propagation(
        graph, phx.pgm.SumProductBeliefPropagation()
    )
    state = phx.pgm.initialize_belief_propagation(prepared)

    eager = phx.pgm.run_belief_propagation(prepared, state)
    compiled = eqx.filter_jit(phx.pgm.run_belief_propagation)(prepared, state)

    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(compiled.log_normalizer, eager.log_normalizer)
    assert jnp.allclose(
        # ty: ignore[unresolved-attribute]
        compiled.variable_log_probabilities.values,
        # ty: ignore[unresolved-attribute]
        eager.variable_log_probabilities.values,
    )
    tree = _chain_graph()
    max_prepared = phx.pgm.prepare_belief_propagation(
        tree, phx.pgm.MaxProductBeliefPropagation()
    )
    max_state = phx.pgm.initialize_belief_propagation(max_prepared)
    eager_max = phx.pgm.run_belief_propagation(max_prepared, max_state)
    compiled_max = eqx.filter_jit(phx.pgm.run_belief_propagation)(max_prepared, max_state)

    loopy = _triangle_graph()
    loopy_prepared = phx.pgm.prepare_belief_propagation(
        loopy,
        phx.pgm.SumProductBeliefPropagation(maximum_steps=20, relaxation=0.7),
    )
    loopy_state = phx.pgm.initialize_belief_propagation(loopy_prepared)
    eager_loopy = phx.pgm.run_belief_propagation(loopy_prepared, loopy_state)
    compiled_loopy = eqx.filter_jit(phx.pgm.run_belief_propagation)(
        loopy_prepared, loopy_state
    )

    # ty: ignore[unresolved-attribute]
    assert jnp.array_equal(compiled_max.map_assignment, eager_max.map_assignment)
    assert jnp.allclose(
        # ty: ignore[unresolved-attribute]
        compiled_loopy.variable_log_probabilities.values,
        # ty: ignore[unresolved-attribute]
        eager_loopy.variable_log_probabilities.values,
    )
    graph = _triangle_graph()
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    state = phx.pgm.initialize_gibbs(prepared, jnp.asarray([[0, 0, 0], [1, 1, 1]]))
    key = jax.random.key(4)

    eager_state, eager_info = phx.pgm.gibbs_sweep(prepared, state, key)
    compiled_state, compiled_info = eqx.filter_jit(phx.pgm.gibbs_sweep)(
        prepared, state, key
    )
    block = phx.pgm.JointDiscreteBlock((0, 1), maximum_configurations=4)
    eager_block, eager_block_info = phx.pgm.joint_block_sweep(prepared, state, block, key)
    compiled_block, compiled_block_info = eqx.filter_jit(phx.pgm.joint_block_sweep)(
        prepared, state, block, key
    )

    assert jnp.array_equal(compiled_state.positions, eager_state.positions)
    assert jnp.array_equal(compiled_info.valid, eager_info.valid)
    assert jnp.array_equal(compiled_block.positions, eager_block.positions)
    assert jnp.array_equal(compiled_block_info.valid, eager_block_info.valid)


def test_traced_arguments_scenario_3() -> None:
    graph = _triangle_graph()
    assignments = jnp.asarray([[0, 0, 0], [1, 1, 1], [0, 1, 0]])

    eager = phx.pgm.pseudolikelihood_loss(graph, assignments)
    compiled = eqx.filter_jit(phx.pgm.pseudolikelihood_loss)(graph, assignments)

    assert jnp.allclose(compiled, eager)
    graph = _triangle_graph()
    bp = phx.pgm.prepare_belief_propagation(graph, phx.pgm.MaxProductBeliefPropagation())
    method = phx.pgm.SmoothDualLP(num_steps=3)
    eager_dual = phx.pgm.solve_smooth_dual_lp(bp, method)
    compiled_dual = eqx.filter_jit(phx.pgm.solve_smooth_dual_lp)(bp, method)

    elimination = phx.pgm.plan_variable_elimination(graph)
    key = jax.random.key(8)
    eager_perturb = phx.pgm.perturb_and_map_log_normalizer(
        elimination, key=key, num_samples=4
    )
    compiled_perturb = eqx.filter_jit(
        lambda plan, sample_key: phx.pgm.perturb_and_map_log_normalizer(
            plan, key=sample_key, num_samples=4
        )
    )(elimination, key)

    assert jnp.allclose(compiled_dual.upper_bound, eager_dual.upper_bound)
    assert jnp.array_equal(compiled_dual.assignment, eager_dual.assignment)
    assert jnp.allclose(compiled_perturb.estimates, eager_perturb.estimates)
    graph = _binary_pair()
    valid_score = eqx.filter_jit(phx.pgm.factor_graph_log_score)(
        graph, jnp.asarray([2**40, 0], dtype=jnp.int64)
    )
    assert jnp.isneginf(valid_score)

    invalid_evidence = jnp.asarray([jnp.nan, 0.0, 0.0, 0.0])
    with pytest.raises(eqx.EquinoxRuntimeError):
        phx.pgm.pack_evidence(graph, invalid_evidence)

    check_evidence = eqx.filter_jit(
        lambda value, raw: phx.pgm.pack_evidence(value, raw).values
    )
    with pytest.raises(eqx.EquinoxRuntimeError):
        check_evidence(
            graph,
            jnp.asarray([jnp.nan, 0.0, 0.0, 0.0]),
        ).block_until_ready()


def test_dynamic_graph_gradients_and_same_structure_cache_reuse() -> None:
    graph = _triangle_graph()
    assignments = jnp.asarray([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    objective = lambda value: phx.pgm.pseudolikelihood_loss(value, assignments)
    eager_gradient = eqx.filter_grad(objective)(graph)
    compiled_gradient = eqx.filter_jit(eqx.filter_grad(objective))(graph)

    assert jnp.allclose(
        compiled_gradient.factor_groups[0].log_potentials,
        eager_gradient.factor_groups[0].log_potentials,
    )

    traces = []

    def score(value: Any, states: Any) -> Any:
        traces.append(None)
        return phx.pgm.factor_graph_log_score(value, states)

    compiled_score = eqx.filter_jit(score)
    compiled_score(graph, assignments).block_until_ready()
    first_count = len(traces)
    updated = eqx.tree_at(
        lambda value: value.factor_groups[0].log_potentials,
        graph,
        graph.factor_groups[0].log_potentials + 0.1,
    )
    compiled_score(updated, assignments).block_until_ready()
    assert len(traces) == first_count

    other = _chain_graph(4)
    compiled_score(other, jnp.zeros((1, 4), dtype=jnp.int32)).block_until_ready()
    assert len(traces) > first_count


def test_reverse_kernel_accepts_dynamic_kernel_state_and_observation() -> None:
    graph = _binary_pair()
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    initial = phx.pgm.initialize_gibbs(prepared, jnp.asarray([[0, 0], [1, 1], [0, 1]]))
    kernel = phx.transport.discrete.FactorGraphReverseKernel(
        graph,
        prepared,
        jnp.asarray([0]),
        jnp.asarray([1]),
        phx.pgm.GibbsSchedule(warmup_sweeps=1, num_draws=2),
    )
    noisy = jnp.asarray([[0], [1], [0]])
    key = jax.random.key(5)

    eager = kernel.sample(key, noisy, initial)
    compiled = eqx.filter_jit(
        lambda value, k, observed, state: value.sample(k, observed, state)
    )(kernel, key, noisy, initial)

    assert jnp.array_equal(compiled, eager)

    with pytest.raises(TypeError):
        kernel.sample(key, noisy.astype(jnp.float64), initial)
    with pytest.raises(eqx.EquinoxRuntimeError):
        kernel.sample(
            key,
            jnp.asarray([[0], [2**40], [0]], dtype=jnp.int64),
            initial,
        )

    invalid_noisy = jnp.asarray([[0], [2], [0]])
    with pytest.raises(eqx.EquinoxRuntimeError):
        kernel.sample(key, invalid_noisy, initial)

    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(
            lambda value, k, observed, state: value.sample(k, observed, state)
        )(
            kernel,
            key,
            invalid_noisy,
            initial,
        ).block_until_ready()
