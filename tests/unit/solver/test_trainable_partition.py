#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.domain import DomainFunction, Interval1d, TrajectoryDatasetDomain
from phydrax.enforcement import enforce_ragged_time_series
from phydrax.nn import MLP
from phydrax.terms import TrajectorySignal


def _inexact_leaves(tree):
    return tuple(
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    )


def _array_leaves(tree):
    return tuple(leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf))


def _make_trajectory_problem():
    inputs = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    lengths = jnp.asarray([2, 4, 3])
    domain = TrajectoryDatasetDomain(inputs, lengths, dt=0.5)
    times = domain.start + domain.dt * jnp.arange(domain.max_length)
    values = inputs[:, 0, None] + times[None, :]
    return domain, inputs, values


class _ScaledQueryTransfer(eqx.Module):
    transfer: phx.graph.QueryGraphOperator
    scale: jnp.ndarray

    def __init__(self, transfer: phx.graph.QueryGraphOperator, scale):
        self.transfer = transfer
        self.scale = jnp.asarray(scale, dtype=float)

    def __call__(self, graph):
        out = self.transfer(graph)
        nodes = dict(out.nodes)
        nodes["out"] = nodes["out"] * self.scale
        return out.replace(nodes=nodes, validate=False)


class _ScaledNodeRate(eqx.Module):
    scale: jnp.ndarray

    def __init__(self, scale):
        self.scale = jnp.asarray(scale, dtype=float)

    def __call__(self, graph):
        return graph.replace(
            nodes=jnp.ones_like(graph.nodes) * self.scale,
            validate=False,
        )


def test_trajectory_signal_construction_does_not_make_static_jax_arrays():
    domain, _inputs, values = _make_trajectory_problem()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrajectorySignal(domain, values, interpolation="linear")

    messages = tuple(str(w.message) for w in caught)
    assert not any("JAX array is being set as static" in message for message in messages)


def test_trajectory_signal_values_are_not_trainable_solver_leaves():
    domain, _inputs, values = _make_trajectory_problem()
    signal = TrajectorySignal(domain, values, interpolation="linear")

    params, non_trainable = partition_trainable({"forcing": signal})
    assert _inexact_leaves(params) == ()
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(non_trainable))
    assert values.shape in fixed_shapes


def test_domain_parameter_stays_trainable_but_plain_constant_is_fixed():
    domain = Interval1d(0.0, 1.0)
    param = domain.Parameter(1.0)
    const = DomainFunction(domain=domain, deps=(), func=jnp.asarray(2.0))

    param_params, _param_fixed = partition_trainable({"lambda": param})
    const_params, const_fixed = partition_trainable({"c": const})

    assert len(_inexact_leaves(param_params)) == 1
    assert _inexact_leaves(const_params) == ()
    assert any(bool(jnp.allclose(leaf, 2.0)) for leaf in _inexact_leaves(const_fixed))


def test_trajectory_domain_arrays_are_not_trainable_model_leaves():
    domain, inputs, _values = _make_trajectory_problem()
    model = MLP(
        in_size=3,
        out_size="scalar",
        width_size=5,
        depth=1,
        key=jr.key(0),
    )
    u = domain.Model("data", "t")(model)

    params, non_trainable = partition_trainable({"u": u})
    param_shapes = tuple(leaf.shape for leaf in _inexact_leaves(params))
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(non_trainable))

    assert param_shapes
    assert inputs.shape not in param_shapes
    assert inputs.shape in fixed_shapes
    assert domain.lengths.shape in fixed_shapes


def test_hard_ragged_table_is_fixed_but_free_model_stays_trainable():
    domain, _inputs, values = _make_trajectory_problem()
    model = MLP(
        in_size=3,
        out_size="scalar",
        width_size=5,
        depth=1,
        key=jr.key(1),
    )
    free = domain.Model("data", "t")(model)
    hard = enforce_ragged_time_series(free, domain, values)

    params, non_trainable = partition_trainable({"u": hard})
    param_shapes = tuple(leaf.shape for leaf in _inexact_leaves(params))
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(non_trainable))

    assert param_shapes
    assert values.shape not in param_shapes
    assert values.shape in fixed_shapes


def test_embedded_query_graph_state_is_fixed_but_graph_model_params_trainable():
    source = phx.graph.GraphIR(
        nodes={
            "positions": jnp.array([[0.0], [1.0]]),
            "features": jnp.array([[1.0], [3.0]]),
        },
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    query = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
    )
    transfer = phx.graph.QueryGraphOperator(
        query,
        source_key="features",
        input_key="u",
        output_key="out",
    )
    domain = phx.domain.GraphDomain(source)
    model = _ScaledQueryTransfer(transfer, 2.0)
    u = domain.GraphModel(model, output_key="out")

    params, non_trainable = partition_trainable({"u": u})
    trainable_leaves = _inexact_leaves(params)
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(non_trainable))

    assert len(trainable_leaves) == 1
    assert jnp.allclose(trainable_leaves[0], 2.0)
    assert source.nodes["features"].shape in fixed_shapes
    assert query.graph.edges["kernel_weight"].shape in fixed_shapes


def test_graph_rollout_stepper_dt_is_fixed_but_vector_field_params_trainable():
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    stepper = phx.graph.EulerGraphStepper(_ScaledNodeRate(2.0), dt=0.25)
    rollout = domain.GraphRolloutModel(stepper, steps=1)

    params, non_trainable = partition_trainable({"rollout": rollout})
    trainable_leaves = _inexact_leaves(params)
    fixed_leaves = _array_leaves(non_trainable)

    assert len(trainable_leaves) == 1
    assert jnp.allclose(trainable_leaves[0], 2.0)
    assert not any(bool(jnp.allclose(leaf, 0.25)) for leaf in trainable_leaves)
    assert graph.nodes.shape in tuple(leaf.shape for leaf in fixed_leaves)
