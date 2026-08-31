import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def _triangle_graph():
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


def test_scan_policies_are_jittable_and_preserve_per_chain_clamps():
    graph = _triangle_graph()
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    state = phx.pgm.initialize_gibbs(
        prepared,
        jnp.asarray([[0, 0, 0], [1, 1, 1]]),
    )
    key = jax.random.key(2)
    masks = jnp.asarray([[True, False, False], [False, True, False]])

    systematic = phx.pgm.GibbsScanPolicy("systematic")
    systematic_state = jax.jit(
        lambda value: phx.pgm.gibbs_sweep_with_policy(
            prepared,
            value,
            key,
            systematic,
            clamped=masks,
        )[0]
    )(state)
    random_scan = phx.pgm.GibbsScanPolicy("random-scan", updates_per_sweep=5)
    random_state = jax.jit(
        lambda value: phx.pgm.gibbs_sweep_with_policy(
            prepared,
            value,
            key,
            random_scan,
        )[0]
    )(state)
    randomized_colors = phx.pgm.GibbsScanPolicy("randomized-colors")
    color_state = jax.jit(
        lambda value: phx.pgm.gibbs_sweep_with_policy(
            prepared,
            value,
            key,
            randomized_colors,
        )[0]
    )(state)

    assert systematic_state.positions[0, 0] == state.positions[0, 0]
    assert systematic_state.positions[1, 1] == state.positions[1, 1]
    assert random_state.positions.shape == state.positions.shape
    assert color_state.positions.shape == state.positions.shape


def test_joint_blocks_parallel_tempering_clusters_and_online_reducers():
    graph = _triangle_graph()
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    state = phx.pgm.initialize_gibbs(
        prepared,
        jnp.asarray([[0, 0, 0], [1, 1, 1]]),
    )
    block_state, block_info = phx.pgm.joint_block_sweep(
        prepared,
        state,
        phx.pgm.JointDiscreteBlock((0, 1), maximum_configurations=4),
        jax.random.key(3),
    )
    reduced = phx.pgm.reduce_gibbs_chain(
        prepared,
        state,
        phx.pgm.MomentReducer(),
        key=jax.random.key(4),
        num_sweeps=4,
        policy=phx.pgm.GibbsScanPolicy("random-scan"),
    )

    assert jnp.all(block_info.valid)
    assert jnp.all(jnp.isfinite(block_state.log_score))
    assert reduced.reduction["mean"].shape == (graph.num_variables,)
    assert reduced.reduction["variance"].shape == (graph.num_variables,)

    method = phx.pgm.ParallelTempering(jnp.asarray([0.4, 0.7, 1.0]))
    tempering = phx.pgm.initialize_parallel_tempering(
        prepared,
        jnp.asarray([[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
        method,
    )
    best = phx.pgm.reduce_gibbs_chain(
        prepared,
        state,
        phx.pgm.BestStateReducer(),
        key=jax.random.key(40),
        num_sweeps=2,
    )
    assert best.reduction.score == phx.pgm.factor_graph_log_score(
        graph,
        best.reduction.position,
    )
    first, first_info = jax.jit(
        lambda value: phx.pgm.parallel_tempering_step(
            prepared,
            value,
            jax.random.key(5),
        )
    )(tempering)
    _, second_info = jax.jit(
        lambda value: phx.pgm.parallel_tempering_step(
            prepared,
            value,
            jax.random.key(6),
        )
    )(first)

    assert jnp.array_equal(first_info.attempted_swaps, jnp.asarray([True, False]))
    assert jnp.array_equal(second_info.attempted_swaps, jnp.asarray([False, True]))

    ising = phx.pgm.ising_factor_graph(
        jnp.zeros((3,)),
        jnp.asarray([[0, 1], [1, 2]]),
        jnp.asarray([0.4, 0.6]),
    )
    ising_prepared = phx.pgm.prepare_chromatic_gibbs(ising)
    clustered = phx.pgm.wolff_cluster_step(
        ising_prepared,
        jnp.asarray([0, 0, 0]),
        jax.random.key(7),
    )
    assert jnp.any(clustered != 0)

    field_graph = phx.pgm.ising_factor_graph(
        jnp.asarray([0.1, 0.0, 0.0]),
        jnp.asarray([[0, 1], [1, 2]]),
        jnp.asarray([0.4, 0.6]),
    )
    with pytest.raises(ValueError, match="zero unary fields"):
        phx.pgm.wolff_cluster_step(
            phx.pgm.prepare_chromatic_gibbs(field_graph),
            jnp.asarray([0, 0, 0]),
            jax.random.key(8),
        )


def test_training_objectives_persistent_chains_and_exact_em():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(1,), num_states=2)
    factor = phx.pgm.DenseTableFactorGroup(
        (phx.pgm.VariableSelection.all(variables),),
        jnp.asarray([[0.2, -0.3]]),
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    assignments = jnp.asarray([[0], [0], [1]])
    exact = phx.pgm.enumerate_factor_graph(graph)
    pseudolikelihood = phx.pgm.pseudolikelihood_loss(graph, assignments)

    assert pseudolikelihood == pytest.approx(
        float(
            exact.log_normalizer
            - jnp.mean(phx.pgm.factor_graph_log_score(graph, assignments))
        )
    )
    gradient = jax.grad(
        lambda table: phx.pgm.pseudolikelihood_loss(
            eqx.tree_at(
                lambda value: value.factor_groups[0].log_potentials,
                graph,
                table,
            ),
            assignments,
        )
    )(factor.log_potentials)
    assert jnp.all(jnp.isfinite(gradient))

    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    chains = phx.pgm.initialize_gibbs(prepared, jnp.asarray([[0], [1]]))
    optimizer = optax.sgd(0.01)
    training = phx.pgm.initialize_persistent_training(graph, optimizer, chains)
    updated = phx.pgm.persistent_contrastive_divergence_step(
        training,
        optimizer,
        prepared,
        assignments,
        jax.random.key(9),
    )

    assert updated.sampler_valid
    assert int(updated.state.step_index) == 1
    assert updated.state.graph.structure_id == graph.structure_id

    plan = phx.pgm.plan_variable_elimination(graph)
    em = phx.pgm.expectation_maximization_step(
        graph,
        plan,
        lambda current, _posterior: current,
        evidence=jnp.zeros((graph.num_variable_states,)),
    )
    assert em.monotone
    assert em.objective_after == pytest.approx(float(em.objective_before))


def test_factor_graph_checkpoint_round_trip(tmp_path):
    graph = _triangle_graph()
    belief_plan = phx.pgm.prepare_belief_propagation(graph)
    belief_state = phx.pgm.initialize_belief_propagation(belief_plan)
    gibbs_plan = phx.pgm.prepare_chromatic_gibbs(graph)
    gibbs_state = phx.pgm.initialize_gibbs(gibbs_plan, jnp.asarray([[0, 0, 0]]))
    path = tmp_path / "graph.phx"

    phx.pgm.write_factor_graph_checkpoint(
        path,
        graph,
        belief_state=belief_state,
        gibbs_state=gibbs_state,
    )
    restored = phx.pgm.read_factor_graph_checkpoint(path)

    assert restored.graph.structure_id == graph.structure_id
    assert jnp.array_equal(restored.belief_state.messages, belief_state.messages)
    assert jnp.array_equal(restored.gibbs_state.positions, gibbs_state.positions)
