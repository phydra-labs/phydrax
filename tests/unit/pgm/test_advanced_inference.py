import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _chain_graph(length=3):
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(length,), num_states=2)
    edges = jnp.stack(
        [jnp.arange(length - 1), jnp.arange(1, length)],
        axis=-1,
    )
    interactions = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        jnp.broadcast_to(
            jnp.asarray([[0.4, -0.2], [-0.1, 0.3]]),
            (length - 1, 2, 2),
        ),
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (interactions,))


def _triangle_graph():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(3,), num_states=2)
    edges = jnp.asarray([[0, 1], [1, 2], [2, 0]])
    interactions = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        jnp.broadcast_to(jnp.asarray([[0.1, -0.05], [-0.05, 0.1]]), (3, 2, 2)),
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (interactions,))


def _linear_kernel(parameters, states):
    parameter_shape = (parameters.shape[0],) + (1,) * (states.ndim - 2)
    return parameters.reshape(parameter_shape) * jnp.sum(states, axis=-1)


def test_open_factor_capabilities_are_enforced_and_reported():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(2,), num_states=2)
    selections = (
        phx.pgm.VariableSelection(variables, [0]),
        phx.pgm.VariableSelection(variables, [1]),
    )
    capabilities = phx.pgm.FactorKernelCapabilities(
        sum_product=True,
        max_product=True,
        factor_beliefs=True,
        scalar_conditional=True,
    )
    kernel = phx.pgm.CallableFactorKernel(
        _linear_kernel,
        kernel_id="test-linear-score",
        capabilities=capabilities,
    )
    factor = phx.pgm.KernelFactorGroup(selections, kernel, jnp.asarray([0.3]))
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    prepared = phx.pgm.prepare_belief_propagation(graph)
    result = phx.pgm.run_belief_propagation(
        prepared,
        phx.pgm.initialize_belief_propagation(prepared),
    )

    assert result.successful
    assert (
        prepared.factor_evidence[0].capabilities.capability_id
        == capabilities.capability_id
    )
    assert prepared.factor_evidence[0].kernel_id == "test-linear-score"

    unsupported = phx.pgm.KernelFactorGroup(
        selections,
        phx.pgm.CallableFactorKernel(
            _linear_kernel,
            kernel_id="test-unsupported-score",
        ),
        jnp.asarray([0.3]),
    )
    unsupported_graph = phx.pgm.DiscreteFactorGraph((variables,), (unsupported,))
    with pytest.raises(ValueError, match="sum-product"):
        phx.pgm.prepare_belief_propagation(unsupported_graph)


def test_sparse_enumerated_bp_never_materializes_dense_support():
    variables = phx.pgm.DiscreteVariableGroup(
        "x",
        shape=(2,),
        num_states=jnp.asarray([100, 100]),
    )
    configurations = jnp.asarray([[0, 0], [12, 45], [99, 3]], dtype=jnp.int32)
    log_potentials = jnp.asarray([[0.0, 0.5, -0.2]])
    factor = phx.pgm.EnumeratedFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        configurations,
        log_potentials,
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    resources = phx.pgm.FactorGraphResourcePolicy(
        maximum_configurations=3,
        maximum_dense_elements=1,
    )
    prepared = phx.pgm.prepare_belief_propagation(
        graph,
        max_factor_configurations=3,
        resources=resources,
    )
    result = phx.pgm.run_belief_propagation(
        prepared,
        phx.pgm.initialize_belief_propagation(prepared),
    )
    supported_probabilities = jax.nn.softmax(log_potentials[0])

    assert result.successful
    assert prepared.factor_tables[0].shape == (1, 3)
    assert prepared.factor_evidence[0].dense_elements == 0
    assert result.factor_probabilities[0].shape == (1, 3)
    assert jnp.allclose(result.factor_probabilities[0][0], supported_probabilities)


def test_precision_policy_places_factor_messages_and_outputs_by_stage():
    graph = _chain_graph(2)
    precision = phx.pgm.FactorGraphPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    prepared = phx.pgm.prepare_belief_propagation(graph, precision=precision)
    state = phx.pgm.initialize_belief_propagation(prepared)
    result = phx.pgm.run_belief_propagation(prepared, state)

    assert prepared.factor_tables[0].dtype == jnp.float32
    assert state.messages.dtype == jnp.float64
    assert state.evidence.values.dtype == jnp.float32
    assert result.variable_log_probabilities.values.dtype == jnp.float32
    assert result.factor_probabilities[0].dtype == jnp.float32
    assert result.log_normalizer.dtype == jnp.float32
    assert dict(result.provenance.configuration)["accumulation_dtype"] == "float64"


def test_directed_forest_and_native_batches_match_exact_inference():
    graph = _chain_graph(4)
    exact = phx.pgm.enumerate_factor_graph(graph)
    prepared = phx.pgm.prepare_belief_propagation(graph)
    state = phx.pgm.initialize_belief_propagation(prepared)
    result = phx.pgm.run_belief_propagation(prepared, state)

    assert prepared.forest
    assert result.successful
    assert int(result.diagnostics.iterations) == int(
        graph.topology.incidence_edges.shape[0]
    )
    assert jnp.allclose(
        jnp.exp(result.variable_log_probabilities.values),
        exact.variable_probabilities.values,
    )

    evidence = jnp.stack(
        [state.evidence.values, state.evidence.values.at[1].set(-jnp.inf)]
    )
    batched_state = phx.pgm.BatchedBeliefPropagationState(
        jnp.zeros((2, prepared.message_count)),
        evidence,
        structure_id=graph.structure_id,
    )
    batched = phx.pgm.batch_belief_propagation(prepared, batched_state)
    packed = phx.pgm.pack_factor_graphs((graph, _chain_graph(2)))

    assert batched.num_cases == 2
    assert jnp.all(batched.results.successful)
    assert packed.num_graphs == 2
    assert len(phx.pgm.enumerate_packed_factor_graphs(packed)) == 2


def test_loopy_schedule_acceleration_and_qualified_implicit_root():
    graph = _triangle_graph()
    prepared = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(maximum_steps=50),
    )
    state = phx.pgm.initialize_belief_propagation(prepared)
    synchronous = phx.pgm.run_belief_propagation(
        prepared,
        state,
        schedule=phx.pgm.BeliefPropagationSchedulePolicy("synchronous"),
    )
    asynchronous = phx.pgm.run_belief_propagation(
        prepared,
        state,
        schedule=phx.pgm.BeliefPropagationSchedulePolicy("asynchronous"),
    )
    accelerated = phx.pgm.run_accelerated_belief_propagation(prepared, state)
    implicit = phx.pgm.run_implicit_belief_propagation(prepared, state)

    assert synchronous.successful
    assert asynchronous.successful
    assert accelerated.inference.successful
    assert implicit.inference.successful
    assert implicit.implicit_derivative
    assert jnp.allclose(
        synchronous.variable_log_probabilities.values,
        asynchronous.variable_log_probabilities.values,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="acyclic"):
        phx.pgm.run_belief_propagation(
            prepared,
            state,
            schedule=phx.pgm.BeliefPropagationSchedulePolicy("forest"),
        )


def test_elimination_junction_law_and_map_bounds_are_truthful():
    graph = _chain_graph()
    exact = phx.pgm.enumerate_factor_graph(graph)
    plan = phx.pgm.plan_variable_elimination(graph)
    eliminated = phx.pgm.variable_elimination(plan)
    calibrated = phx.pgm.junction_tree_calibrate(phx.pgm.plan_junction_tree(plan))
    law = phx.pgm.NormalizedFactorGraphLaw(plan, eliminated)
    conditioned_evidence = jnp.zeros((graph.num_variable_states,)).at[1].set(-jnp.inf)
    conditioned = phx.pgm.variable_elimination(
        plan,
        evidence=conditioned_evidence,
    )
    conditioned_law = phx.pgm.NormalizedFactorGraphLaw(plan, conditioned)
    assert jnp.array_equal(conditioned_law.evidence, conditioned_evidence)
    with pytest.raises(ValueError, match="same normalized law"):
        phx.pgm.NormalizedFactorGraphLaw(
            plan,
            conditioned,
            evidence=jnp.zeros_like(conditioned_evidence),
        )

    assert eliminated.successful
    assert calibrated.valid
    assert jnp.allclose(eliminated.log_normalizer, exact.log_normalizer)
    assert jnp.allclose(
        eliminated.variable_probabilities.values,
        exact.variable_probabilities.values,
    )
    samples = law.sample(jax.random.key(4), sample_shape=(4,))
    assert samples.shape == (4, graph.num_variables)
    assert jnp.all(law.contains(samples))

    max_prepared = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.MaxProductBeliefPropagation(),
    )
    dual = phx.pgm.solve_smooth_dual_lp(
        max_prepared,
        phx.pgm.SmoothDualLP(num_steps=100),
    )
    estimate = phx.pgm.perturb_and_map_log_normalizer(
        plan,
        key=jax.random.key(8),
        num_samples=16,
    )
    assert dual.valid
    assert dual.upper_bound >= exact.map_log_score - 1e-8
    assert dual.lower_bound <= exact.map_log_score + 1e-8
    assert estimate.upper_bound_in_expectation
    assert jnp.isfinite(estimate.mean)
    assert estimate.standard_error >= 0

    with pytest.raises(ValueError, match="workspace"):
        phx.pgm.plan_variable_elimination(
            graph,
            resources=phx.pgm.FactorGraphResourcePolicy(
                maximum_elimination_elements=1,
            ),
        )
