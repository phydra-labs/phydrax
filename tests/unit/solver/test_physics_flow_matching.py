from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx


class _ConstantVelocity(eqx.Module):
    value: jnp.ndarray

    def __call__(self, state, time):
        del time
        return jnp.broadcast_to(self.value, state.shape)


class _LinearVelocity(eqx.Module):
    def __call__(self, state, time):
        del time
        return state


def _velocity_function(model):
    state = phx.domain.HyperRectangle(
        jnp.asarray((-10.0,)),
        jnp.asarray((10.0,)),
        label="x",
    )
    domain = state @ phx.domain.TimeInterval(0.0, 1.0)
    return domain.Function("x", "t")(model)


def _endpoints(count=8):
    source = jnp.zeros((count, 1))
    target = jnp.ones((count, 1))
    return phx.transport.EndpointCouplingSample(
        source=source,
        target=target,
        source_indices=jnp.arange(count),
        target_indices=jnp.arange(count),
        valid=jnp.ones((count,), dtype=bool),
        log_weights=jnp.zeros((count,)),
        context={"desired": target},
        coupling_id="unit-translation",
        provenance="unit-test",
    )


def _functional():
    return phx.terms.CallableFlowEndpointFunctional(
        lambda state, context: jnp.sum(jnp.square(state - context["desired"])),
        event_shape=(1,),
        functional_id="squared-target-distance",
    )


def _term(steps, *, velocity=0.0, sampling_mode="fixed", endpoints=None):
    endpoint_source = _endpoints() if endpoints is None else endpoints
    term = phx.terms.PhysicsFlowMatchingTerm(
        "velocity",
        endpoint_source,
        phx.transport.LinearEndpointInterpolant((1,)),
        _functional(),
        phx.terms.FlowEndpointRolloutPolicy(steps),
        sampling_mode=sampling_mode,
    )
    return term, _velocity_function(_ConstantVelocity(jnp.asarray((velocity,))))


def test_logit_normal_time_sampling_is_bounded_and_reproducible():
    policy = phx.terms.LogitNormalTimeSamplingPolicy(
        0.1,
        0.9,
        location=0.0,
        scale=1.5,
    )
    first = policy.sample(jr.key(4), (128,), dtype=jnp.float32)
    second = policy.sample(jr.key(4), (128,), dtype=jnp.float32)

    assert jnp.array_equal(first, second)
    assert jnp.all(first > 0.1)
    assert jnp.all(first < 0.9)


def test_exact_constant_velocity_satisfies_both_shared_batch_objectives():
    term, velocity = _term(4, velocity=1.0)
    batch = term.sample(key=jr.key(7))
    flow, physics = term.objective_components(
        {"velocity": velocity},
        batch=batch,
        iter_=jnp.asarray(0),
    )
    diagnostics = term.diagnostics(
        {"velocity": velocity},
        batch=batch,
        iter_=jnp.asarray(0),
    )

    assert jnp.allclose(flow, 0.0, atol=1e-7)
    assert jnp.allclose(physics, 0.0, atol=1e-7)
    assert diagnostics.active_steps == 4
    assert diagnostics.terminal_valid_fraction == 1.0
    assert diagnostics.finite


def test_endpoint_refinement_reduces_nonlinear_euler_error():
    endpoints = _endpoints(count=1)
    interpolant = phx.transport.LinearEndpointInterpolant((1,))
    batch = phx.terms.FlowMatchingBatch(
        state=jnp.ones((1, 1)),
        time=jnp.zeros((1,)),
        target_velocity=jnp.zeros((1, 1)),
        valid=jnp.ones((1,), dtype=bool),
        log_weights=jnp.zeros((1,)),
        context={"desired": jnp.full((1, 1), jnp.e)},
        evaluation_key=jr.key(1),
        source_indices=jnp.zeros((1,), dtype=jnp.int32),
        target_indices=jnp.zeros((1,), dtype=jnp.int32),
        interpolant_id=interpolant.interpolant_id,
        coupling_id=endpoints.coupling_id,
        policy_id="manual",
        batch_id="manual-nonlinear",
    )
    velocity = _velocity_function(_LinearVelocity())
    one = phx.terms.PhysicsFlowMatchingTerm(
        "velocity",
        endpoints,
        interpolant,
        _functional(),
        phx.terms.FlowEndpointRolloutPolicy(1),
    )
    four = phx.terms.PhysicsFlowMatchingTerm(
        "velocity",
        endpoints,
        interpolant,
        _functional(),
        phx.terms.FlowEndpointRolloutPolicy(4, rematerialize=True),
    )

    one_physics = one.objective_components({"velocity": velocity}, batch=batch)[1]
    four_physics = four.objective_components({"velocity": velocity}, batch=batch)[1]
    assert four_physics < one_physics


def test_physics_flow_term_trains_with_conflict_free_component_gradients():
    calls = []

    def provider(key):
        calls.append(key)
        return _endpoints()

    term, velocity = _term(
        2,
        velocity=0.5,
        sampling_mode="resample",
        endpoints=provider,
    )
    solver = phx.solver.FunctionalSolver(
        functions={"velocity": velocity},
        terms=(term,),
    )
    trained = solver.solve(
        num_iter=1,
        optim=optax.sgd(0.1),
        jit=False,
        keep_best=False,
        log_every=0,
        training=phx.solver.FunctionalTrainingPlan(
            gradient_composition=phx.optim.ConflictFreeGradientPolicy()
        ),
    )

    assert len(calls) == 1
    assert trained.functions["velocity"].func.function.value[0] > 0.5
