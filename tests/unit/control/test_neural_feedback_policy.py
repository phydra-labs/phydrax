#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Learned state feedback in the control-parameterization decision slot."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.control import (
    AbstractControlParameterization,
    CONTROL_SUCCESS,
    ControlProblem,
    NeuralFeedbackPolicy,
)
from phydrax.dynamics import TimeGrid
from phydrax.nn.models import MLP
from tests._control_systems import make_discrete_control_dynamics
from tests._ported_models import full_port, in_order, PortedAffine


def _network(in_size=2, out_size=1, seed=0):
    return MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=8,
        depth=1,
        key=jax.random.key(seed),
    )


def test_policy_is_jittable_vmappable_and_requires_the_state():
    network = _network()
    policy = NeuralFeedbackPolicy(
        network, state_shape=(2,), control_shape=(1,), policy_id="neural"
    )
    states = jnp.asarray([[0.1, -0.2], [0.5, 0.3], [-1.0, 2.0]])

    batched = policy.evaluate(jnp.zeros((3,)), 0.25, case_shape=(3,), state=states)
    mapped = jax.vmap(lambda state: policy.evaluate(jnp.asarray(0.0), 0.25, state=state))(
        states
    )
    jitted = eqx.filter_jit(
        lambda policy, state: policy.evaluate(jnp.asarray(0.0), 0.25, state=state)
    )(policy, states[1])

    assert batched.shape == (3, 1)
    np.testing.assert_allclose(batched, mapped, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(jitted, network(states[1]), rtol=1e-12, atol=1e-14)
    assert policy.parameter_shape == ()
    with pytest.raises(ValueError, match="requires the current state"):
        policy.evaluate(jnp.asarray(0.0), 0.25)
    with pytest.raises(ValueError, match="without states"):
        policy.sample(jnp.asarray(0.0), jnp.asarray([0.0, 0.5]))
    with pytest.raises(ValueError, match="shape"):
        policy.evaluate(jnp.ones((2,)), 0.25, state=states[0])


def test_policy_sizes_and_binding_are_exact():
    with pytest.raises(ValueError, match="in_size must be 3"):
        NeuralFeedbackPolicy(
            _network(),
            state_shape=(2,),
            control_shape=(1,),
            policy_id="time",
            time_input=True,
        )
    with pytest.raises(ValueError, match="out_size"):
        NeuralFeedbackPolicy(
            _network(out_size=2),
            state_shape=(2,),
            control_shape=(1,),
            policy_id="wide",
        )
    policy = NeuralFeedbackPolicy(
        _network(in_size=3),
        state_shape=(2,),
        control_shape=(1,),
        policy_id="time",
        time_input=True,
    )
    contract = policy.component_contract()
    assert contract.authority is phx.ComponentAuthority.DECISION
    assert contract.slot_semantic_id == AbstractControlParameterization.slot_semantic_id


def test_port_declaring_policy_binds_only_to_declared_owner_ports():
    owner = phx.ModelPorts(
        inputs=(full_port("plant.state", (2,)), full_port("plant.time", ())),
        outputs=(full_port("plant.control", (1,)),),
    )
    model = PortedAffine(owner, out_size=1, weight=jnp.asarray([[1.0, -2.0, 0.5]]))
    arguments = dict(
        state_shape=(2,), control_shape=(1,), policy_id="ported", time_input=True
    )
    with pytest.raises(ValueError, match="parameterization'.*owner_ports"):
        NeuralFeedbackPolicy(model, **arguments)
    with pytest.raises(ValueError, match="event shapes"):
        NeuralFeedbackPolicy(
            model,
            **arguments,
            ports=phx.ModelPorts(inputs=owner.inputs[:1], outputs=owner.outputs),
        )

    policy = NeuralFeedbackPolicy(
        model, **arguments, ports=owner, port_mapping=in_order(owner, owner)
    )
    evidence = policy.component_contract().port_binding
    assert evidence.inputs == tuple((port.port_id,) * 2 for port in owner.inputs)
    assert evidence.outputs == ((owner.outputs[0].port_id,) * 2,)
    assert evidence.unverified == ()
    value = policy.evaluate(jnp.asarray(0.0), 0.5, state=jnp.asarray([1.0, 1.0]))
    np.testing.assert_allclose(value, [1.0 - 2.0 + 0.25])


def test_rollout_trains_only_the_policy_network():
    grid = TimeGrid(jnp.linspace(0.0, 1.0, 5), time_id="time:neural-feedback")
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + 0.25 * control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="neural-feedback-integrator",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([1.0]),
        running_cost=lambda time, state, control, args: jnp.sum(control**2),
        terminal_cost=lambda time, state, args: jnp.sum(state**2),
        problem_id="neural-feedback",
    )
    policy = NeuralFeedbackPolicy(
        _network(in_size=1, seed=1),
        state_shape=(1,),
        control_shape=(1,),
        policy_id="neural-feedback-policy",
    )
    token = jnp.asarray(0.0)

    trajectory = problem.rollout(policy, token)
    assert int(trajectory.status) == CONTROL_SUCCESS
    expected = jnp.asarray([1.0])
    for control in trajectory.controls:
        np.testing.assert_allclose(control, policy.model(expected), rtol=1e-12)
        expected = expected + 0.25 * control

    parameters, model_state, fixed = phx.partition_parameters(policy)
    assert {id(leaf) for leaf in jax.tree.leaves(parameters)} == {
        id(leaf) for leaf in jax.tree.leaves(eqx.filter(policy.model, eqx.is_array))
    }

    def loss(parameters):
        bound = phx.combine_parameters(parameters, model_state, fixed)
        return problem.evaluate(bound, token).sampled_loss.total

    gradient = eqx.filter_jit(jax.grad(loss))(parameters)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    assert any(jnp.any(leaf != 0.0) for leaf in jax.tree.leaves(gradient))
