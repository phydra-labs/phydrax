from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem
from phydrax.stochastic._feynman_kac import (
    feynman_kac_label_diagnostics,
    FeynmanKacLabelBatch,
    FeynmanKacSamplingPlan,
    query_feynman_kac_labels,
    sample_feynman_kac_paths,
    trajectory_node_feynman_kac_labels,
)


def test_sampling_plan_canonicalizes_quadrature_before_identity() -> None:
    common = {
        "initial_time": 0.0,
        "terminal_time": 1.0,
        "sampling_mode": "queries",
        "num_paths_per_query": 2,
        "num_time_steps": 2,
    }
    # ty: ignore[invalid-argument-type]
    canonical = FeynmanKacSamplingPlan(**common, quadrature="left")
    # ty: ignore[invalid-argument-type]
    equivalent = FeynmanKacSamplingPlan(**common, quadrature=np.str_("left"))

    assert type(equivalent.quadrature) is str
    assert equivalent.plan_id == canonical.plan_id

    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        FeynmanKacSamplingPlan(**common, quadrature=1)


def _constant_paths(*, invalid: Any = False) -> Any:
    times = jnp.asarray([0.0, 0.25, 1.0])
    states = jnp.asarray(
        [
            [[1.0], [1.0], [1.0]],
            [[2.0], [2.0], [2.0]],
        ]
    )
    valid = jnp.ones((2, 3), dtype="bool")
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


def _constant_problem(paths: Any, generator: Any) -> Any:
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


def _brownian_problem(dimension: Any = 1) -> Any:
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


def test_trajectory_nodes_accumulate_constant_source_and_preserve_clusters() -> None:
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


def test_trajectory_trapezoid_handles_nonuniform_time_grid_and_invalid_paths() -> None:
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


def test_query_conditioned_brownian_value_control_and_terminal_query() -> None:
    problem = _brownian_problem()
    # The martingale estimator Y_{t1} dW_0 / dt has per-antithetic-pair variance
    # 2 + (N - 1) = 9 for N = 8 steps here, so its standard error is 3 / sqrt(pairs):
    # 32768 pairs give 0.0166, making the 8e-2 control tolerance 4.8 standard errors.
    plan = FeynmanKacSamplingPlan(
        initial_time=0.0,
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=65536,
        num_time_steps=8,
        control_target_mode="martingale",
        antithetic=True,
        path_chunk_size=8192,
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

    assert paths.states.shape == (3, 65536, 9, 1)
    assert labels.control_targets is not None
    assert labels.control_valid is not None
    assert jnp.allclose(labels.value_targets[:, 0], states[:, 0], atol=2e-2)
    assert jnp.allclose(labels.control_targets[:2, 0, 0], 1.0, atol=8e-2)
    assert jnp.allclose(labels.value_targets[2, 0], states[2, 0])
    assert not labels.control_valid[2]
    assert labels.source_path_count == 32768
    assert labels.metadata["path_chunk_size"] == 8192


def test_query_sampling_replays_and_rejects_out_of_interval_queries() -> None:
    problem = _brownian_problem()
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=16,
        num_time_steps=3,
    )
    times = jnp.asarray([0.1, 0.8])
    states = jnp.zeros((2, 1))

    first = sample_feynman_kac_paths(
        problem,
        times,
        states,
        plan,
        key=jr.key(9),
        num_paths=8,
    )
    second = sample_feynman_kac_paths(
        problem,
        times,
        states,
        plan,
        key=jr.key(9),
        num_paths=8,
    )
    extended = sample_feynman_kac_paths(
        problem,
        times,
        states,
        plan,
        key=jr.key(9),
        num_paths=16,
    )
    assert jnp.array_equal(first.states, second.states)
    assert jnp.array_equal(first.wiener_increments, second.wiener_increments)
    assert jnp.array_equal(first.states, extended.states[:, :8])
    assert jnp.array_equal(
        first.wiener_increments,
        extended.wiener_increments[:, :8],
    )

    with pytest.raises(ValueError, match="inside"):
        sample_feynman_kac_paths(
            problem,
            jnp.asarray([-0.1]),
            jnp.zeros((1, 1)),
            plan,
        )


def test_dimension_100_query_labels_preserve_shapes_without_hessian_contracts() -> None:
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


def test_stochastic_source_keys_replay_across_path_chunk_sizes() -> None:
    base = _brownian_problem()
    problem = BSDEProblem(
        base.forward_sampler,
        base.drift,
        base.diffusion,
        lambda _time, _state, value, control, _args: value + control[..., 0],
        base.terminal,
        state_shape=base.state_shape,
        noise_shape=base.noise_shape,
        output_shape=base.output_shape,
        problem_id="chunk-invariant-source",
        process_id=base.process_id,
    )
    domain = phx.domain.Interval1d(-5.0, 5.0) @ phx.domain.TimeInterval(0.0, 1.0)
    keyed = phx.domain.FunctionBinding(pass_key=True)
    source_value = domain.Function("t", "x", binding=keyed)(
        lambda _time, _state, *, key: jr.normal(key, (1,))
    )
    source_control = domain.Function("t", "x", binding=keyed)(
        lambda _time, _state, *, key: jr.normal(key, (1, 1))
    )
    common = {
        "terminal_time": 1.0,
        "sampling_mode": "queries",
        "num_paths_per_query": 8,
        "num_time_steps": 3,
    }
    # ty: ignore[invalid-argument-type]
    unchunked = FeynmanKacSamplingPlan(**common)
    # ty: ignore[invalid-argument-type]
    chunked = FeynmanKacSamplingPlan(**common, path_chunk_size=2)
    kwargs = {
        "query_times": jnp.asarray([0.0, 0.4]),
        "query_states": jnp.asarray([[0.2], [-0.3]]),
        "source_value": source_value,
        "source_control": source_control,
        "key": jr.key(73),
    }

    # ty: ignore[invalid-argument-type]
    first = query_feynman_kac_labels(problem, unchunked, **kwargs)
    # ty: ignore[invalid-argument-type]
    second = query_feynman_kac_labels(problem, chunked, **kwargs)

    # ty: ignore[unresolved-attribute]
    assert jnp.array_equal(first.value_targets, second.value_targets)
    # ty: ignore[unresolved-attribute]
    assert jnp.array_equal(first.value_standard_errors, second.value_standard_errors)


def test_scalar_feynman_kac_labels_preserve_query_path_and_time_axes() -> None:
    placeholder = BSDEPathBatch(
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        sample_shape=(1,),
        state_shape=(),
        noise_shape=(),
        path_id="scalar-placeholder",
        process_id="scalar-brownian",
    )
    problem = BSDEProblem(
        lambda _key: placeholder,
        lambda _time, state, _args: jnp.zeros_like(state),
        lambda _time, _state, _args: jnp.asarray(1.0),
        lambda _time, _state, value, _control, _args: jnp.zeros_like(value),
        lambda state, _args: state,
        state_shape=(),
        noise_shape=(),
        output_shape=(),
        problem_id="scalar-feynman-kac",
        process_id="scalar-brownian",
    )
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=8,
        num_time_steps=2,
        path_chunk_size=4,
    )
    # ty: ignore[not-iterable]
    labels, paths = query_feynman_kac_labels(
        problem,
        plan,
        query_times=jnp.asarray([0.0, 0.5]),
        query_states=jnp.asarray([0.2, -0.4]),
        key=jr.key(18),
        return_paths=True,
    )

    assert paths.states.shape == (2, 8, 3)
    assert paths.wiener_increments.shape == (2, 8, 2)
    assert labels.value_targets.shape == (2,)
    assert labels.query_states.shape == (2,)
