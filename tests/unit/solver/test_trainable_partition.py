#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax.domain import DomainFunction, Interval1d, TrajectoryDatasetDomain
from phydrax.enforcement import enforce_ragged_time_series
from phydrax.nn.models import MLP
from phydrax.nn.parameters import ParameterSubspace
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
    scale: jnp.ndarray = phx.parameter_field()

    def __init__(self, transfer: phx.graph.QueryGraphOperator, scale):
        self.transfer = transfer
        self.scale = jnp.asarray(scale, dtype="float64")

    def __call__(self, graph):
        out = self.transfer(graph)
        nodes = dict(out.nodes)
        nodes["out"] = nodes["out"] * self.scale
        return out.replace(nodes=nodes, validate=False)


class _ScaledNodeRate(eqx.Module):
    scale: jnp.ndarray = phx.parameter_field()

    def __init__(self, scale):
        self.scale = jnp.asarray(scale, dtype="float64")

    def __call__(self, graph):
        return graph.replace(
            nodes=jnp.ones_like(graph.nodes) * self.scale,
            validate=False,
        )


class _RawScale(eqx.Module):
    scale: jax.Array

    def __call__(self, x, **_kwargs):
        return self.scale * x[0]


class _StatefulScale(phx.ParameterOwner, eqx.Module):
    scale: jax.Array
    calls: jax.Array = phx.model_state_field()

    def __call__(self, x, **_kwargs):
        return self.scale * x[0]


def _residual_solver(func):
    domain = Interval1d(0.0, 1.0)
    field = DomainFunction(domain=domain, deps=("x",), func=func)
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda current: current)
    batch = component.points({"x": jnp.asarray([[0.25], [0.5], [0.75]])})
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(phx.integration.mean_over(component), batch)
        ),
    )
    return phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))


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

    params, _model_state, fixed = phx.partition_parameters({"forcing": signal})
    assert _inexact_leaves(params) == ()
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(fixed))
    assert values.shape in fixed_shapes


def test_domain_parameter_stays_trainable_but_plain_constant_is_fixed():
    domain = Interval1d(0.0, 1.0)
    param = domain.Parameter(1.0)
    const = DomainFunction(domain=domain, deps=(), func=jnp.asarray(2.0))

    param_params, _, _param_fixed = phx.partition_parameters({"lambda": param})
    const_params, _, const_fixed = phx.partition_parameters({"c": const})

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

    params, _model_state, fixed = phx.partition_parameters({"u": u})
    param_shapes = tuple(leaf.shape for leaf in _inexact_leaves(params))
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(fixed))

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

    params, _model_state, fixed = phx.partition_parameters({"u": hard})
    param_shapes = tuple(leaf.shape for leaf in _inexact_leaves(params))
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(fixed))

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

    params, _model_state, fixed = phx.partition_parameters({"u": u})
    trainable_leaves = _inexact_leaves(params)
    fixed_shapes = tuple(leaf.shape for leaf in _array_leaves(fixed))

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

    params, _model_state, fixed = phx.partition_parameters({"rollout": rollout})
    trainable_leaves = _inexact_leaves(params)
    fixed_leaves = _array_leaves(fixed)

    assert len(trainable_leaves) == 1
    assert jnp.allclose(trainable_leaves[0], 2.0)
    assert not any(bool(jnp.allclose(leaf, 0.25)) for leaf in trainable_leaves)
    assert graph.nodes.shape in tuple(leaf.shape for leaf in fixed_leaves)


def test_partition_functions_returns_recombinable_role_lanes():
    domain = Interval1d(0.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={
            "lambda": domain.Parameter(1.5),
            "c": DomainFunction(domain=domain, deps=(), func=jnp.asarray(2.0)),
        },
        terms=(),
    )

    parameters, model_state, fixed = solver.partition_functions()

    assert [float(leaf) for leaf in _inexact_leaves(parameters)] == [1.5]
    assert _array_leaves(model_state) == ()
    assert any(bool(jnp.allclose(leaf, 2.0)) for leaf in _inexact_leaves(fixed))
    assert eqx.tree_equal(solver.trainable_functions(), parameters)
    assert eqx.tree_equal(
        phx.combine_parameters(parameters, model_state, fixed), solver.functions
    )


def test_functional_solve_rejects_undeclared_arrays_unless_explicitly_selected():
    solver = _residual_solver(_RawScale(jnp.asarray(2.0)))

    with pytest.raises(ValueError, match=r"(?s)FunctionalSolver\.solve.*\.func\.scale"):
        solver.solve(num_iter=1, optim=optax.sgd(0.1), keep_best=False, jit=False)

    subspace = ParameterSubspace.from_leaf_paths(
        solver.functions, ParameterSubspace.array_leaf_paths(solver.functions)
    )
    trained = solver.solve(
        num_iter=1,
        optim=optax.sgd(0.1),
        parameter_subspace=subspace,
        keep_best=False,
        jit=False,
    )

    assert float(trained.functions["u"].func.scale) < 2.0


def test_gradient_training_carries_model_state_but_linear_trial_space_rejects_it():
    solver = _residual_solver(_StatefulScale(jnp.asarray(2.0), jnp.asarray(7)))

    trained = solver.solve(num_iter=1, optim=optax.sgd(0.1), keep_best=False, jit=False)

    assert float(trained.functions["u"].func.scale) < 2.0
    assert int(trained.functions["u"].func.calls) == 7
    with pytest.raises(ValueError, match="solve_linear_trial_space requires an empty"):
        phx.solver.solve_linear_trial_space(solver)
