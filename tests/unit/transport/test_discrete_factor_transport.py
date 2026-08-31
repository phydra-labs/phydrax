import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _reverse_kernel():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(2,), num_states=2)
    factor = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, [0]),
            phx.pgm.VariableSelection(variables, [1]),
        ),
        jnp.asarray([[[1.0, -1.0], [-1.0, 1.0]]]),
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    schedule = phx.pgm.GibbsSchedule(
        warmup_sweeps=1,
        num_draws=2,
        sweeps_per_draw=1,
    )
    kernel = phx.transport.discrete.FactorGraphReverseKernel(
        graph,
        prepared,
        jnp.asarray([0]),
        jnp.asarray([1]),
        schedule,
    )
    return graph, prepared, kernel


def test_discrete_forward_and_factor_graph_reverse_process_shapes():
    discrete = phx.transport.discrete
    graph, prepared, reverse = _reverse_kernel()
    forward = discrete.DiscreteForwardProcess((discrete.CategoricalNoisingKernel(0.8),))
    process = discrete.DiscreteDenoisingProcess(forward, (reverse,))
    initial = jnp.asarray([[0, 1], [1, 0], [0, 0]])
    path = forward.sample_path(
        jax.random.key(1),
        initial,
        jnp.asarray([2, 2]),
    )
    chains = phx.pgm.initialize_gibbs(
        prepared,
        jnp.asarray([[0, 0], [1, 1], [0, 1]]),
    )
    restored = process.sample_reverse(
        jax.random.key(2),
        jnp.asarray([[0], [1], [0]]),
        (chains,),
    )

    assert path.shape == (2, 3, graph.num_variables)
    assert restored.shape == (3, 1)
    assert jnp.all((restored >= 0) & (restored < 2))

    with pytest.raises(ValueError, match="shape"):
        process.sample_reverse(
            jax.random.key(2),
            jnp.asarray([[0, 1]]),
            (chains,),
        )


def test_recovery_objective_adaptive_control_and_hybrid_embedding():
    discrete = phx.transport.discrete
    graph, _prepared, reverse = _reverse_kernel()
    positive = jnp.asarray([[0, 0], [1, 1]])
    negative = jnp.asarray([[0, 1], [1, 0]])
    objective, diagnostics = discrete.RecoveryLikelihoodObjective()(
        reverse,
        positive,
        negative,
    )

    expected = -jnp.mean(phx.pgm.factor_graph_log_score(graph, positive)) + jnp.mean(
        phx.pgm.factor_graph_log_score(graph, negative)
    )
    assert objective == pytest.approx(float(expected))
    assert diagnostics.valid

    controller = discrete.AdaptiveMixingPenalty()
    state = controller.initialize(2)
    updated = controller.update(state, jnp.asarray([0.01, 0.2]))
    assert updated.penalties.shape == (2,)
    assert updated.epoch == 1

    process = discrete.DiscreteDenoisingProcess(
        discrete.DiscreteForwardProcess((discrete.CategoricalNoisingKernel(0.8),)),
        (reverse,),
    )
    hybrid = discrete.HybridDiscreteEmbedding(
        lambda value: jnp.asarray(value, dtype=jnp.int32),
        lambda value: jnp.asarray(value, dtype=float),
        process,
    )
    encoded = hybrid.encode(jnp.asarray([1.0]))
    decoded = hybrid.decode(encoded)
    assert encoded.dtype == jnp.int32
    assert jnp.issubdtype(decoded.dtype, jnp.floating)
