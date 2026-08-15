import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem
from phydrax.stochastic._feynman_kac import (
    feynman_kac_label_diagnostics,
    FeynmanKacLabelBatch,
    FeynmanKacSamplingPlan,
    query_feynman_kac_labels,
    sample_feynman_kac_paths,
    trajectory_node_feynman_kac_labels,
)


def _constant_paths(*, invalid=False):
    times = jnp.asarray([0.0, 0.25, 1.0])
    states = jnp.asarray(
        [
            [[1.0], [1.0], [1.0]],
            [[2.0], [2.0], [2.0]],
        ]
    )
    valid = jnp.ones((2, 3), dtype=bool)
    if invalid:
        valid = valid.at[1, 1].set(False)
    return BSDEPathBatch(
        times,
        states,
        jnp.zeros((2, 2, 1)),
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="constant",
        process_id="constant",
        valid=valid,
    )


def _constant_problem(paths, generator):
    return BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones((1, 1)),
        generator,
        lambda state, args: jnp.asarray([state[0]]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="constant-source",
        process_id="constant",
    )


def _brownian_problem(dimension=1):
    placeholder = _constant_paths()
    return BSDEProblem(
        lambda key: placeholder,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.eye(dimension),
        lambda time, state, value, control, args: jnp.zeros_like(value),
        lambda state, args: jnp.asarray([jnp.mean(state)]),
        state_shape=(dimension,),
        noise_shape=(dimension,),
        output_shape=(1,),
        problem_id=f"brownian-{dimension}",
        process_id=f"brownian-{dimension}",
    )


def test_trajectory_nodes_accumulate_constant_source_and_preserve_clusters():
    paths = _constant_paths()
    problem = _constant_problem(
        paths,
        lambda time, state, value, control, args: jnp.asarray([2.0]),
    )
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="trajectory_nodes",
        quadrature="left",
    )

    labels = trajectory_node_feynman_kac_labels(problem, paths, plan)

    expected_first = jnp.asarray([[3.0], [2.5], [1.0]])
    expected_second = jnp.asarray([[4.0], [3.5], [2.0]])
    assert jnp.allclose(
        labels.value_targets, jnp.concatenate((expected_first, expected_second))
    )
    assert jnp.array_equal(labels.cluster_ids, jnp.asarray([0, 0, 0, 1, 1, 1]))
    assert jnp.all(labels.valid)
    assert feynman_kac_label_diagnostics(labels).passed


def test_trajectory_trapezoid_handles_nonuniform_time_grid_and_invalid_paths():
    paths = _constant_paths(invalid=True)
    problem = _constant_problem(
        paths,
        lambda time, state, value, control, args: jnp.asarray([time]),
    )
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="trajectory_nodes",
        quadrature="trapezoid",
        time_weighting="trapezoid",
    )

    labels = trajectory_node_feynman_kac_labels(problem, paths, plan)

    assert jnp.allclose(labels.value_targets[:3, 0], jnp.asarray([1.5, 1.46875, 1.0]))
    assert jnp.array_equal(
        labels.valid, jnp.asarray([True, True, True, False, False, True])
    )
    assert jnp.allclose(labels.sample_weights[:3], jnp.asarray([0.125, 0.5, 0.375]))


def test_query_conditioned_brownian_value_control_and_terminal_query():
    problem = _brownian_problem()
    plan = FeynmanKacSamplingPlan(
        initial_time=0.0,
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=4096,
        num_time_steps=8,
        control_target_mode="martingale",
        antithetic=True,
    )
    times = jnp.asarray([0.0, 0.6, 1.0])
    states = jnp.asarray([[0.2], [-0.4], [0.7]])

    result = query_feynman_kac_labels(
        problem,
        plan,
        query_times=times,
        query_states=states,
        key=jr.key(4),
        return_paths=True,
    )
    assert isinstance(result, tuple)
    labels, paths = result

    assert paths.states.shape == (3, 4096, 9, 1)
    assert labels.control_targets is not None
    assert labels.control_valid is not None
    assert jnp.allclose(labels.value_targets[:, 0], states[:, 0], atol=2e-2)
    assert jnp.allclose(labels.control_targets[:2, 0, 0], 1.0, atol=8e-2)
    assert jnp.allclose(labels.value_targets[2, 0], states[2, 0])
    assert not labels.control_valid[2]
    assert labels.source_path_count == 2048


def test_query_sampling_replays_and_rejects_out_of_interval_queries():
    problem = _brownian_problem()
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=16,
        num_time_steps=3,
    )
    times = jnp.asarray([0.1, 0.8])
    states = jnp.zeros((2, 1))

    first = sample_feynman_kac_paths(problem, times, states, plan, key=jr.key(9))
    second = sample_feynman_kac_paths(problem, times, states, plan, key=jr.key(9))
    assert jnp.array_equal(first.states, second.states)
    assert jnp.array_equal(first.wiener_increments, second.wiener_increments)

    with pytest.raises(ValueError, match="inside"):
        sample_feynman_kac_paths(
            problem,
            jnp.asarray([-0.1]),
            jnp.zeros((1, 1)),
            plan,
        )


def test_dimension_100_query_labels_preserve_shapes_without_hessian_contracts():
    dimension = 100
    problem = _brownian_problem(dimension)
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=32,
        num_time_steps=2,
    )
    states = jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,))))

    labels = query_feynman_kac_labels(
        problem,
        plan,
        query_times=jnp.asarray([0.0, 0.5]),
        query_states=states,
        key=jr.key(2),
    )
    assert isinstance(labels, FeynmanKacLabelBatch)

    assert labels.query_states.shape == (2, dimension)
    assert labels.value_targets.shape == (2, 1)
    assert labels.value_standard_errors.shape == (2, 1)
    assert jnp.all(jnp.isfinite(labels.value_targets))
