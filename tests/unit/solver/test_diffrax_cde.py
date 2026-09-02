import warnings

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


class _ScaledPathValue(eqx.Module):
    amplitude: jax.Array

    def __call__(self, time, side):
        del side
        return jnp.asarray([self.amplitude * time])


class _ScaledPathDerivative(eqx.Module):
    amplitude: jax.Array

    def __call__(self, time, side):
        del time, side
        return jnp.asarray([self.amplitude])


class _LinearField(eqx.Module):
    rate: jax.Array

    def __call__(self, time, state, args):
        del time, args
        return self.rate * state[..., None]


def _declared_path(value, derivative, *, dimension, path_id, breakpoints=()):
    points = jnp.asarray(breakpoints, dtype=float)
    return phx.solver.CallableDrivingPath(
        value,
        derivative,
        support=jnp.asarray([0.0, 1.0]),
        value_shape=(dimension,),
        path_id=path_id,
        breakpoints=points,
        breakpoint_mask=jnp.ones(points.shape, dtype=bool),
    )


def test_smooth_cde_matches_analytic_solution_and_preserves_provenance():
    rate = 0.7
    path = _declared_path(
        lambda time, side: jnp.asarray([time**2]),
        lambda time, side: jnp.asarray([2.0 * time]),
        dimension=1,
        path_id="quadratic-control",
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: rate * state[..., None],
        jnp.asarray([1.3]),
        driver_dimension=1,
        problem_id="analytic-smooth-cde",
    )
    times = jnp.asarray([0.0, 0.2, 0.7, 1.0])

    solution = phx.solver.solve_diffrax_cde(
        problem,
        path,
        save_times=times,
        dense=True,
        rtol=1e-9,
        atol=1e-11,
    )

    expected = 1.3 * jnp.exp(rate * times**2)
    assert isinstance(solution.differential_solution, phx.solver.DifferentialSolution)
    assert jnp.allclose(solution.states[:, 0], expected, rtol=2e-7, atol=2e-9)
    assert jnp.all(solution.valid)
    assert bool(solution.successful)
    assert solution.path is path
    assert solution.path_id == "quadratic-control"
    assert solution.problem_id == "analytic-smooth-cde"
    assert solution.path_interpolation == "CallableDrivingPath"
    assert solution.control_dimension == 1
    assert solution.metadata["lowering"] == "differentiable-control-to-ode"
    assert solution.solver_id == solution.differential_solution.solver_id
    assert jnp.allclose(
        solution.evaluate(jnp.asarray([0.35]))[0, 0],
        1.3 * jnp.exp(rate * 0.35**2),
        rtol=2e-7,
    )


def test_drift_and_multidimensional_control_contract_along_driver_axis():
    matrix = jnp.asarray([[1.0, -2.0], [0.5, 3.0]])
    drift = jnp.asarray([0.3, -0.2])
    initial = jnp.asarray([0.4, -1.0])
    path = _declared_path(
        lambda time, side: jnp.asarray([time**2, -time]),
        lambda time, side: jnp.asarray([2.0 * time, -1.0]),
        dimension=2,
        path_id="two-channel-control",
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: matrix,
        initial,
        driver_dimension=2,
        drift=lambda time, state, args: drift,
    )

    solution = phx.solver.solve_diffrax_cde(
        problem,
        path,
        save_times=jnp.asarray([1.0]),
        rtol=1e-9,
        atol=1e-11,
    )

    expected = initial + drift + matrix @ jnp.asarray([1.0, -1.0])
    assert jnp.allclose(solution.states[0], expected, rtol=2e-8, atol=2e-9)


def test_piecewise_linear_derivative_knot_is_declared_and_landed():
    path = phx.solver.PiecewiseLinearDrivingPath(
        jnp.asarray([0.0, 0.3, 1.0]),
        jnp.asarray([[0.0], [0.3], [3.1]]),
        time_mask=jnp.ones((3,), dtype=bool),
        value_mask=jnp.ones((3, 1), dtype=bool),
        path_id="one-knot-control",
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.ones(state.shape + (1,)),
        jnp.asarray([0.0]),
        driver_dimension=1,
    )

    solution = phx.solver.solve_diffrax_cde(
        problem,
        path,
        save_times=jnp.asarray([0.3, 1.0]),
        dt0=1.0,
        rtol=0.5,
        atol=0.5,
    )

    assert jnp.array_equal(solution.derivative_discontinuities, jnp.asarray([0.3]))
    assert jnp.array_equal(solution.derivative_discontinuity_mask, jnp.asarray([True]))
    assert solution.metadata["derivative_discontinuity_policy"] == "diffrax-jump-landing"
    assert int(solution.stats["num_accepted_steps"]) >= 2
    assert jnp.allclose(solution.states[:, 0], jnp.asarray([0.3, 3.1]), atol=2e-6)


@pytest.mark.parametrize("right_offset", [1.0, jnp.nan])
def test_callable_value_jumps_and_nonfinite_breakpoint_limits_are_rejected(
    right_offset,
):
    def value(time, side):
        offset = jnp.where(time == 0.5, right_offset, 0.0) if side == "right" else 0.0
        return jnp.asarray([time + offset])

    path = _declared_path(
        value,
        lambda time, side: jnp.asarray([1.0]),
        dimension=1,
        path_id="invalid-callable-value-break",
        breakpoints=(0.5,),
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.ones(state.shape + (1,)),
        jnp.asarray([0.0]),
        driver_dimension=1,
    )

    with pytest.raises(Exception, match="finite matching left/right"):
        solution = phx.solver.solve_diffrax_cde(
            problem,
            path,
            save_times=jnp.asarray([1.0]),
        )
        jax.block_until_ready(solution.states)


def test_callable_derivative_jump_works_and_inactive_capacity_is_not_terminal_jump(
    monkeypatch,
):
    captured = {}
    clip_controller = dfx.ClipStepSizeController

    class CaptureSchedule(clip_controller):
        def __init__(self, controller, *args, **kwargs):
            captured["step_ts"] = kwargs["step_ts"]
            captured["jump_ts"] = kwargs["jump_ts"]
            super().__init__(controller, *args, **kwargs)

    monkeypatch.setattr(dfx, "ClipStepSizeController", CaptureSchedule)

    def value(time, side):
        del side
        return jnp.asarray([jnp.where(time <= 0.5, time, 2.0 * time - 0.5)])

    def derivative(time, side):
        before_break = time <= 0.5 if side == "left" else time < 0.5
        return jnp.asarray([jnp.where(before_break, 1.0, 2.0)])

    path = phx.solver.CallableDrivingPath(
        value,
        derivative,
        support=jnp.asarray([0.0, 1.0]),
        value_shape=(1,),
        path_id="callable-derivative-break",
        breakpoints=jnp.asarray([0.5, jnp.nan]),
        breakpoint_mask=jnp.asarray([True, False]),
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.ones(state.shape + (1,)),
        jnp.asarray([0.0]),
        driver_dimension=1,
    )

    solution = phx.solver.solve_diffrax_cde(
        problem,
        path,
        save_times=jnp.asarray([1.0]),
        dt0=1.0,
        rtol=0.5,
        atol=0.5,
    )

    assert bool(solution.successful)
    assert jnp.allclose(solution.states[0, 0], 1.5, atol=2e-6)
    assert jnp.array_equal(captured["step_ts"], captured["jump_ts"])
    assert jnp.allclose(captured["step_ts"][0], 0.5)
    assert jnp.isposinf(captured["step_ts"][1])


def test_gradients_flow_through_vector_field_and_path_coefficients():
    def terminal(rate, amplitude):
        path = _declared_path(
            _ScaledPathValue(amplitude),
            _ScaledPathDerivative(amplitude),
            dimension=1,
            path_id="differentiable-linear-control",
        )
        problem = phx.solver.RoughDifferentialProblem(
            _LinearField(rate),
            jnp.asarray([1.0]),
            driver_dimension=1,
        )
        return phx.solver.solve_diffrax_cde(
            problem,
            path,
            save_times=jnp.asarray([1.0]),
            rtol=1e-8,
            atol=1e-10,
        ).states[0, 0]

    rate = jnp.asarray(0.4)
    amplitude = jnp.asarray(1.7)
    rate_gradient, path_gradient = jax.grad(terminal, argnums=(0, 1))(rate, amplitude)
    expected = jnp.exp(rate * amplitude)

    assert jnp.allclose(rate_gradient, amplitude * expected, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(path_gradient, rate * expected, rtol=2e-5, atol=2e-6)


def test_smooth_cde_agrees_with_refined_geometric_rough_solve():
    rate = 0.45
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: rate * state[..., None],
        jnp.asarray([1.2]),
        driver_dimension=1,
    )
    path = _declared_path(
        lambda time, side: jnp.asarray([time]),
        lambda time, side: jnp.asarray([1.0]),
        dimension=1,
        path_id="identity-smooth-control",
    )
    cde = phx.solver.solve_diffrax_cde(
        problem,
        path,
        save_times=jnp.asarray([1.0]),
        rtol=1e-9,
        atol=1e-11,
    )
    partition = jnp.linspace(0.0, 1.0, 257)
    rough_path = phx.stochastic.GeometricRoughPath.from_values(
        partition, partition[:, None]
    )
    rde = phx.solver.solve_rough_differential(
        problem,
        rough_path,
        save_times=jnp.asarray([1.0]),
        solver=phx.solver.Davie(),
    )

    assert jnp.allclose(cde.states[0], rde.states[0], rtol=2e-6, atol=2e-6)


def test_rough_second_level_control_is_rejected_with_rde_direction():
    rough_path = phx.stochastic.GeometricRoughPath.from_values(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [1.0]]),
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: state[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
    )

    with pytest.raises(TypeError, match="solve_rough_differential"):
        phx.solver.solve_diffrax_cde(
            problem,
            rough_path,
            save_times=jnp.asarray([1.0]),
        )


def test_complex_cde_uses_declared_real_coordinates():
    path = _declared_path(
        lambda time, side: jnp.asarray([time]),
        lambda time, side: jnp.asarray([1.0]),
        dimension=1,
        path_id="complex-identity-control",
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: 1j * state[..., None],
        jnp.asarray([1.0 + 0.0j]),
        driver_dimension=1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Complex dtype support in Diffrax.*",
        )
        solution = phx.solver.solve_diffrax_cde(
            problem,
            path,
            save_times=jnp.asarray([0.5, 1.0]),
            rtol=1e-9,
            atol=1e-11,
        )

    evidence = solution.differential_solution.temporal_evidence
    assert evidence is not None
    assert evidence.state_coordinates is not None
    assert evidence.state_coordinates.domain_kind == "full"
    assert solution.states.dtype == jnp.complex128
    assert jnp.allclose(
        solution.states[:, 0],
        jnp.exp(1j * jnp.asarray([0.5, 1.0])),
        rtol=2e-7,
        atol=2e-9,
    )
