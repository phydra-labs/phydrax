import jax
import jax.numpy as jnp
import numpy as np

from phydrax.solver import (
    hybrid_event_jvp,
    hybrid_event_vjp,
    HybridEventPlan,
    HybridGuardPlan,
    localize_hybrid_event,
    localize_hybrid_event_root,
    localize_numerical_event,
)


def test_numerical_root_differentiates_segment_not_continuous_vector_field():
    # One backward-Euler segment for y'=a*y, not the exact exponential flow.
    # Its numerical time slope is a*y0/(1-a*h)^2, not a*y.
    def localized(parameters):
        initial, rate, threshold, guard_time = parameters
        return localize_numerical_event(
            lambda time: {
                "voltage": initial / (1.0 - rate * (time - 2.0)),
                "memory": initial * time,
            },
            lambda time, state: state["voltage"] - threshold + guard_time * time,
            2.0,
            2.75,
            tolerance=1.0e-6,
        )

    parameters = jnp.asarray([1.0, 1.0, 2.0, 0.0])
    result = localized(parameters)
    assert result.successful & result.derivative_valid
    assert result.crossing == 1
    np.testing.assert_allclose(result.event_time, 2.5)
    np.testing.assert_allclose(result.transversality, 4.0)
    expected = jnp.asarray([-0.5, -0.5, 0.25, -0.625])
    root_function = lambda values: localized(values).event_time
    np.testing.assert_allclose(jax.jacfwd(root_function)(parameters), expected)
    np.testing.assert_allclose(jax.grad(root_function)(parameters), expected)
    np.testing.assert_allclose(
        jax.jacfwd(lambda values: localized(values).state["voltage"])(parameters),
        jnp.asarray([0.0, 0.0, 1.0, -2.5]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        jax.jacfwd(lambda values: localized(values).state["memory"])(parameters),
        expected + jnp.asarray([2.5, 0.0, 0.0, 0.0]),
    )


def test_numerical_root_vmap_and_bracket_endpoints_are_branch_selectors():
    def root(threshold, left, right):
        return localize_numerical_event(
            lambda time: time * time,
            lambda time, state: state - threshold,
            left,
            right,
            tolerance=1.0e-6,
        ).event_time

    thresholds = jnp.asarray([1.0, 4.0, 9.0])
    roots = jax.jit(jax.vmap(lambda threshold: root(threshold, 0.5, 3.5)))(thresholds)
    np.testing.assert_allclose(roots, jnp.sqrt(thresholds), atol=1.0e-6)
    np.testing.assert_allclose(
        jax.vmap(jax.grad(lambda threshold: root(threshold, 0.5, 3.5)))(thresholds),
        0.5 / jnp.sqrt(thresholds),
    )
    # Endpoint roots retain the implicit derivative, not a derivative of clipping.
    np.testing.assert_allclose(jax.grad(lambda value: root(value, 1.0, 2.0))(1.0), 0.5)
    np.testing.assert_allclose(jax.grad(lambda value: root(value, 1.0, 2.0))(4.0), 0.25)
    assert jax.grad(lambda left: root(2.0, left, 2.0))(1.0) == 0.0


def test_grazing_primal_root_and_invalid_brackets_poison_both_derivative_modes():
    def grazing(parameter):
        return localize_numerical_event(
            lambda time: (time - 0.5) ** 3,
            lambda time, state: state - parameter,
            0.0,
            1.0,
        )

    result = grazing(0.0)
    assert result.successful & result.grazing & (~result.derivative_valid)
    assert jnp.isnan(jax.grad(lambda parameter: grazing(parameter).event_time)(0.0))
    assert jnp.isnan(
        jax.jvp(lambda parameter: grazing(parameter).event_time, (0.0,), (1.0,))[1]
    )

    def unbracketed(parameter):
        return localize_numerical_event(
            lambda time: time, lambda time, state: state - parameter, 0.0, 1.0
        )

    assert not unbracketed(2.0).successful
    assert jnp.isnan(jax.grad(lambda parameter: unbracketed(parameter).event_time)(2.0))
    for left, right in ((1.0, 0.0), (0.0, jnp.inf)):
        invalid = localize_numerical_event(
            lambda time: time, lambda time, state: state - 0.5, left, right
        )
        assert not invalid.bracketed
        assert not invalid.derivative_valid


def _parameter_event():
    def guard(time, state, args):
        return state[0] + time * args[0] - args[1]

    return HybridEventPlan(
        HybridGuardPlan(guard, guard_id="parameter-reset"),
        lambda time, state, args: args[2] * state + time * args[3],
        lambda time, state, args: jnp.arange(1, state.size + 1, dtype=state.dtype),
        lambda time, state, args: jnp.arange(4, state.size + 4, dtype=state.dtype),
        event_tolerance=1.0e-6,
        dense_diagnostics=True,
        plan_id="parameter-reset",
    )


def test_physical_matrix_free_actions_include_time_and_reset_parameters():
    plan = _parameter_event()
    time = jnp.asarray(0.25)
    state = jnp.asarray([0.5, 1.0, 2.0])
    args = jnp.asarray([2.0, 1.0, 1.5, 0.3])
    tangent = jnp.asarray([0.2, -0.4, 0.7])
    time_tangent = jnp.asarray(0.3)
    args_tangent = jnp.asarray([-0.2, 0.4, 0.1, -0.5])
    before = jnp.asarray([1.0, 2.0, 3.0])
    after = jnp.asarray([4.0, 5.0, 6.0])
    jump = after - args[3] - args[2] * before
    denominator = before[0] + args[0]
    direct_guard = (
        tangent[0] + args[0] * time_tangent + time * args_tangent[0] - args_tangent[1]
    )
    direct_reset = (
        args[2] * tangent
        + state * args_tangent[2]
        + args[3] * time_tangent
        + time * args_tangent[3]
    )
    result = hybrid_event_jvp(
        plan,
        time,
        state,
        tangent,
        args=args,
        time_tangent=time_tangent,
        args_tangent=args_tangent,
    )
    assert result.successful
    np.testing.assert_allclose(
        result.action, direct_reset + jump * direct_guard / denominator, rtol=1.0e-6
    )
    cotangent = jnp.asarray([-0.7, 0.1, 0.9])
    dt, dy, da, evidence = hybrid_event_vjp(plan, time, state, cotangent, args=args)
    assert evidence.successful
    np.testing.assert_allclose(
        jnp.vdot(cotangent, result.action),
        dt * time_tangent + jnp.vdot(dy, tangent) + jnp.vdot(da, args_tangent),
        rtol=1.0e-6,
    )
    dense = localize_hybrid_event(
        plan, lambda t, a: state + (t - time) * before, 0.0, 0.5, args=args
    )
    assert dense.successful & dense.log_jacobian_valid
    state_only = hybrid_event_jvp(plan, time, state, tangent, args=args)
    np.testing.assert_allclose(state_only.action, dense.saltation_matrix @ tangent)


def test_matrix_free_storage_never_contains_a_state_squared_intermediate():
    # The public matrix-free guarantee is a memory-complexity contract. Inspect
    # traced buffers rather than mocking a particular Jacobian constructor.
    size = 128
    plan = _parameter_event()
    state = jnp.full((size,), 0.5)
    args = jnp.asarray([2.0, 1.0, 1.5, 0.3])
    functions = (
        lambda value: hybrid_event_jvp(plan, 0.25, state, value, args=args).action,
        lambda value: hybrid_event_vjp(plan, 0.25, state, value, args=args)[1],
    )
    for function in functions:
        graph = jax.make_jaxpr(function)(jnp.ones_like(state)).jaxpr
        for equation in graph.eqns:
            for variable in equation.outvars:
                assert np.prod(variable.aval.shape) < size * size


def test_singular_reset_retains_action_but_not_density_and_invalid_actions_are_nan():
    guard = HybridGuardPlan(
        lambda time, state, args: state[0], guard_id="absorbing-reset"
    )
    plan = HybridEventPlan(
        guard,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones_like(state),
        lambda time, state, args: jnp.zeros_like(state),
        dense_diagnostics=True,
        plan_id="absorbing-reset",
    )
    trajectory = lambda time, args: jnp.asarray([time - 0.5])
    root = localize_hybrid_event_root(plan, trajectory, 0.0, 1.0)
    dense = localize_hybrid_event(plan, trajectory, 0.0, 1.0)
    action = hybrid_event_jvp(plan, 0.5, root.state_before, jnp.ones((1,)))
    assert root.successful & dense.successful & action.successful
    assert not dense.log_jacobian_valid
    np.testing.assert_array_equal(action.action, jnp.zeros((1,)))
    invalid = hybrid_event_jvp(plan, 0.5, root.state_before, jnp.asarray([jnp.nan]))
    assert not invalid.successful
    assert jnp.all(jnp.isnan(invalid.action))
    dt, dy, _, reverse = hybrid_event_vjp(
        plan, 0.5, root.state_before, jnp.asarray([jnp.inf])
    )
    assert not reverse.successful
    assert jnp.isnan(dt) & jnp.all(jnp.isnan(dy))

    grazing_guard = HybridGuardPlan(
        lambda time, state, args: state[0] ** 2, guard_id="grazing"
    )
    grazing_plan = HybridEventPlan(
        grazing_guard,
        lambda time, state, args: state,
        lambda time, state, args: jnp.ones_like(state),
        lambda time, state, args: jnp.ones_like(state),
        plan_id="grazing",
    )
    grazing = hybrid_event_jvp(grazing_plan, 0.5, jnp.zeros((1,)), jnp.ones((1,)))
    dt, dy, _, reverse = hybrid_event_vjp(
        grazing_plan, 0.5, jnp.zeros((1,)), jnp.ones((1,))
    )
    assert grazing.grazing & (~grazing.successful) & (~reverse.successful)
    assert jnp.all(jnp.isnan(grazing.action)) & jnp.isnan(dt) & jnp.all(jnp.isnan(dy))
