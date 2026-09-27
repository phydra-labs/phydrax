from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax.solver._deep_picard import (
    PicardSourceContext,
    solve_deep_picard,
    StructuredPicardSource,
)
from phydrax.stochastic._bsde import BSDEPathBatch, BSDEProblem
from phydrax.stochastic._feynman_kac import FeynmanKacSamplingPlan


class _TimeCoefficient(eqx.Module):
    coefficient: jnp.ndarray = phx.parameter_field()

    def __call__(self, time: Any, state: Any, *, key: Any = None) -> Any:
        del state, key
        return jnp.asarray([self.coefficient * (1.0 - time)])


class _QuadraticTimeCoefficient(eqx.Module):
    coefficient: jnp.ndarray = phx.parameter_field()

    def __call__(self, time: Any, state: Any, *, key: Any = None) -> Any:
        del key
        return jnp.asarray([state[0] ** 2 + self.coefficient * (1.0 - time)])


class _ConstantControl(eqx.Module):
    coefficient: jnp.ndarray = phx.parameter_field()

    def __call__(self, time: Any, state: Any, *, key: Any = None) -> Any:
        del time, state, key
        return jnp.asarray([[self.coefficient]])


def _placeholder(process_id: Any = "picard") -> Any:
    return BSDEPathBatch(
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((1, 2, 1)),
        jnp.zeros((1, 1, 1)),
        sample_shape=(1,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="placeholder",
        process_id=process_id,
    )


def _problem(
    *,
    generator: Any,
    terminal: Any,
    problem_id: Any = "picard",
    process_id: Any = "picard",
) -> Any:
    paths = _placeholder(process_id)
    return BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: jnp.ones((1, 1)),
        generator,
        terminal,
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id=problem_id,
        process_id=process_id,
    )


def _domain() -> Any:
    return phx.domain.Interval1d(-4.0, 4.0) @ phx.domain.TimeInterval(0.0, 1.0)


def _function(model: Any) -> Any:
    return _domain().Function("t", "x")(model)


def _coefficient(function: Any) -> Any:
    return float(function.func.function.coefficient)


def test_deep_picard_scenario_1() -> None:
    problem = _problem(
        generator=lambda time, state, value, control, args: jnp.asarray([1.0]),
        terminal=lambda state, args: jnp.asarray([0.0]),
    )
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=32,
        num_time_steps=4,
        refresh_mode="fixed",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"value": _function(_TimeCoefficient(jnp.asarray(0.0)))},
        terms=(),
    )
    times = jnp.linspace(0.0, 1.0, 9)
    states = jnp.zeros((9, 1))

    result = solve_deep_picard(
        solver,
        problem,
        value_name="value",
        sampling_plan=plan,
        num_picard_steps=2,
        inner_num_iter=100,
        optim=optax.adam(0.05),
        query_times=times,
        query_states=states,
        seed=2,
        jit=True,
        keep_best=False,
    )

    assert abs(_coefficient(result.solver["value"]) - 1.0) < 2e-2
    assert result.diagnostics.target_rmse[-1] < 2e-2
    assert result.diagnostics.terminal_rmse[-1] < 2e-2
    assert result.diagnostics.passed
    assert result.solver.terms == solver.terms == ()
    assert _coefficient(solver["value"]) == 0.0
    problem = _problem(
        generator=lambda time, state, value, control, args: jnp.zeros_like(value),
        terminal=lambda state, args: jnp.asarray([state[0]]),
        problem_id="control-picard",
    )
    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=4096,
        num_time_steps=4,
        control_target_mode="martingale",
        antithetic=True,
        refresh_mode="fixed",
    )
    exact_value = _domain().Function("t", "x")(
        lambda time, state: jnp.asarray([state[0]])
    )
    solver = phx.solver.FunctionalSolver(
        functions={
            "value": exact_value,
            "control": _function(_ConstantControl(jnp.asarray(0.0))),
        },
        terms=(),
    )
    times = jnp.asarray([0.0, 0.25, 0.5, 0.75])
    states = jnp.asarray([[-0.4], [0.0], [0.2], [0.7]])

    result = solve_deep_picard(
        solver,
        problem,
        value_name="value",
        control_name="control",
        sampling_plan=plan,
        num_picard_steps=1,
        inner_num_iter=100,
        optim=optax.adam(0.05),
        query_times=times,
        query_states=states,
        value_weight=0.0,
        control_weight=1.0,
        seed=4,
        jit=True,
        keep_best=False,
    )

    assert abs(_coefficient(result.solver["control"]) - 1.0) < 0.08
    assert result.diagnostics.control_target_rmse[-1] < 0.08


def test_structured_source_context_uses_factor_hvps_and_trains_quadratic_case() -> None:
    problem = _problem(
        generator=lambda time, state, value, control, args: jnp.zeros_like(value),
        terminal=lambda state, args: jnp.asarray([state[0] ** 2]),
        problem_id="structured-picard",
    )
    source_model = _function(_QuadraticTimeCoefficient(jnp.asarray(0.0)))
    context = PicardSourceContext(source_model, problem, jr.key(0))
    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        context.directional_hessian(0.2, jnp.asarray([0.3]), jnp.ones((1,))),
        2.0,
    )
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(context.covariance_trace(0.2, jnp.asarray([0.3])), 2.0)

    def source_builder(_context: Any) -> Any:
        return StructuredPicardSource(
            lambda time, state, current, args: (
                0.5 * current.covariance_trace(time, state)
            ),
            source_id="half-covariance-trace",
        )

    plan = FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=4096,
        num_time_steps=4,
        antithetic=True,
        refresh_mode="fixed",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"value": source_model},
        terms=(),
    )
    times = jnp.linspace(0.0, 1.0, 7)
    states = jnp.linspace(-0.5, 0.5, 7)[:, None]

    result = solve_deep_picard(
        solver,
        problem,
        value_name="value",
        sampling_plan=plan,
        num_picard_steps=1,
        inner_num_iter=150,
        optim=optax.adam(0.04),
        query_times=times,
        query_states=states,
        source_builder=source_builder,
        initial_source="current",
        seed=3,
        jit=True,
        keep_best=False,
    )

    assert abs(_coefficient(result.solver["value"]) - 2.0) < 0.12
    assert result.diagnostics.finite[-1]
