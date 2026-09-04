import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import phydrax as phx


def _smooth_neutral_problem(base, *, amplitude=1.0, t1=1.2):
    delay = 0.4
    exponent = 0.35
    neutral_weight = 0.3
    retarded_weight = exponent - neutral_weight * exponent * jnp.exp(-exponent * delay)
    base_value = jnp.asarray(base)

    def history(time, args):
        del args
        return amplitude * jnp.exp(exponent * time) * base_value

    def history_derivative(time, args):
        return exponent * history(time, args)

    def drift(time, state, memory, args):
        del time, args
        return retarded_weight * state + neutral_weight * memory["velocity"]

    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (
            phx.solver.DerivativeDelay(
                "velocity",
                phx.solver.ConstantDelay("velocity_lag", delay),
            ),
        ),
        t0=0.0,
        t1=t1,
        history_derivative=history_derivative,
    )


def _solve_smooth(problem, **kwargs):
    return phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.linspace(problem.t0, problem.t1, 13),
        max_steps=512,
        max_discontinuities=32,
        **kwargs,
    )


def _transformed_neutral_problem(
    *,
    amplitude=1.0,
    endpoint_weight=0.0,
    t1=0.8,
):
    delay = 0.4
    rate = 0.3
    neutral_weight = 0.2
    coefficient = rate * (1.0 - neutral_weight * jnp.exp(-rate * delay) - endpoint_weight)

    def history(time, args):
        del args
        return jnp.asarray([amplitude * jnp.exp(rate * time)])

    endpoint_neutral = (
        None
        if endpoint_weight == 0.0
        else lambda time, state, memory, args: endpoint_weight * state
    )
    return phx.solver.NeutralDelayProblem(
        lambda time, memory, args: neutral_weight * memory["past"],
        lambda time, state, memory, args: coefficient * state,
        history,
        (phx.solver.ConstantDelay("past", delay),),
        t0=0.0,
        t1=t1,
        endpoint_neutral=endpoint_neutral,
    )


def _solve_transformed_neutral(problem, step, *, dense=False, history_mode="full"):
    return phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.linspace(problem.t0, problem.t1, 9),
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=step,
        dense=dense,
        history_mode=history_mode,
        max_steps=None if history_mode == "rolling" else 512,
    )


def test_neutral_scalar_uses_exact_prehistory_and_native_dense_derivative():
    problem = _smooth_neutral_problem(jnp.asarray([1.0]))
    solution = _solve_smooth(problem, dense=True, rtol=1e-9, atol=1e-11)
    exact = jnp.exp(0.35 * solution.times)

    assert jnp.allclose(solution.states[:, 0], exact, rtol=3e-7, atol=3e-8)
    query = jnp.asarray([0.0, 0.17, 0.4, 0.83, 1.2])
    derivative = solution.interpolation.derivative(query)
    assert jnp.allclose(
        derivative[:, 0],
        0.35 * jnp.exp(0.35 * query),
        rtol=2e-6,
        atol=2e-7,
    )
    assert bool(problem.initial_derivative_compatible)
    assert solution.metadata["delay_mode"] == "declared-neutral"


def test_neutral_matrix_state_preserves_shape_and_manufactured_solution():
    base = jnp.asarray([[1.0, -0.5], [0.25, 2.0]])
    problem = _smooth_neutral_problem(base, t1=0.9)
    solution = _solve_smooth(problem, rtol=2e-9, atol=2e-11)
    expected = jnp.exp(0.35 * solution.times[:, None, None]) * base

    assert solution.states.shape == expected.shape
    assert jnp.allclose(solution.states, expected, rtol=4e-7, atol=4e-8)


def test_neutral_derivative_jump_has_one_sided_knot_semantics_and_provenance():
    factor = 0.6

    def history(time, args):
        del args
        return jnp.asarray([jax.lax.stop_gradient(time)])

    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: factor * memory["velocity"],
        history,
        (
            phx.solver.DerivativeDelay(
                "velocity",
                phx.solver.ConstantDelay("lag", 1.0),
            ),
        ),
        t0=0.0,
        t1=3.6,
        history_derivative=lambda time, args: jnp.ones((1,)),
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.5, 1.5, 2.5, 3.5]),
        dense=True,
        rtol=1e-9,
        atol=1e-11,
        max_steps=512,
        max_discontinuities=16,
    )
    assert solution.interpolation is not None
    intervals = jnp.arange(4)
    expected = jnp.asarray(
        [sum(factor**power for power in range(1, n + 1)) for n in range(4)]
    ) + 0.5 * factor ** (intervals + 1)

    assert jnp.allclose(solution.states[:, 0], expected, rtol=2e-6, atol=2e-7)
    assert jnp.allclose(
        solution.interpolation.derivative(jnp.asarray(1.0), left=True)[0],
        factor,
        rtol=2e-6,
    )
    assert jnp.allclose(
        solution.interpolation.derivative(jnp.asarray(1.0), left=False)[0],
        factor**2,
        rtol=2e-6,
    )
    assert not bool(problem.initial_derivative_compatible)
    assert jnp.allclose(problem.initial_derivative_jump, jnp.asarray([factor - 1.0]))
    assert jnp.allclose(solution.metadata["initial_derivative_source_time"], 0.0)
    assert bool(solution.metadata["initial_derivative_source_active"])
    assert int(solution.stats["num_tracked_discontinuities"]) == 4
    tracked = solution.metadata["tracked_discontinuity_times"]
    assert jnp.allclose(tracked[jnp.isfinite(tracked)], jnp.arange(4.0))
    assert jnp.allclose(solution.stats["neutral_discontinuity_horizon"], 3.6)


def test_neutral_implicit_euler_supports_fixed_causal_steps():
    problem = _smooth_neutral_problem(jnp.asarray([1.0]), t1=0.8)
    solver = dfx.ImplicitEuler(
        root_finder=optx.Newton(rtol=1e-10, atol=1e-10),
        root_find_max_steps=50,
    )
    solution = _solve_smooth(
        problem,
        solver=solver,
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.02,
    )

    assert solution.stats["controller_mode"] == "fixed"
    assert jnp.allclose(
        solution.states[-1, 0],
        jnp.exp(0.35 * problem.t1),
        rtol=8e-3,
        atol=8e-4,
    )


def test_neutral_fixed_step_euler_has_first_order_convergence():
    problem = _smooth_neutral_problem(jnp.asarray([1.0]), t1=0.8)

    def error(step):
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([problem.t1]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=step,
            max_steps=512,
            max_discontinuities=16,
        )
        return jnp.abs(solution.states[0, 0] - jnp.exp(0.35 * problem.t1))

    coarse = error(0.1)
    medium = error(0.05)
    fine = error(0.025)
    assert coarse / medium > 1.6
    assert medium / fine > 1.6


def test_neutral_solution_gradient_flows_through_native_derivative_history():
    terminal_time = 0.8

    def terminal(amplitude):
        problem = _smooth_neutral_problem(
            jnp.asarray([1.0]), amplitude=amplitude, t1=terminal_time
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([terminal_time]),
            rtol=2e-8,
            atol=2e-10,
            max_steps=256,
            max_discontinuities=16,
        ).states[0, 0]

    amplitude = jnp.asarray(1.3)
    expected_gradient = jnp.exp(0.35 * terminal_time)
    assert jnp.allclose(jax.grad(terminal)(amplitude), expected_gradient, rtol=2e-5)


def test_derivative_delay_requires_explicit_history_derivative():
    with pytest.raises(ValueError, match="history_derivative is required"):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: memory["velocity"],
            lambda time, args: jnp.asarray([time]),
            (
                phx.solver.DerivativeDelay(
                    "velocity",
                    phx.solver.ConstantDelay("lag", 0.5),
                ),
            ),
            t0=0.0,
            t1=1.0,
        )


def test_stochastic_neutral_problem_is_rejected_before_execution():
    derivative_term = phx.solver.DerivativeDelay(
        "velocity",
        phx.solver.ConstantDelay("lag", 0.5),
    )
    wiener_term = phx.solver.DelayWienerTerm(
        "noise",
        lambda time, state, memory, args: jnp.ones((1, 1)),
        (1,),
    )
    with pytest.raises(ValueError, match="Stochastic neutral"):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: memory["velocity"],
            lambda time, args: jnp.asarray([time]),
            (derivative_term,),
            t0=0.0,
            t1=1.0,
            history_derivative=lambda time, args: jnp.ones((1,)),
            wiener_terms=(wiener_term,),
        )


def test_nontrivial_manifold_neutral_term_uses_geometry_transport():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["velocity"],
        lambda time, args: jnp.eye(2),
        (
            phx.solver.DerivativeDelay(
                "velocity",
                phx.solver.ConstantDelay("lag", 0.5),
            ),
        ),
        t0=0.0,
        t1=0.5,
        history_derivative=lambda time, args: jnp.zeros((2, 2)),
        state_geometry=geometry,
    )
    assert problem.neutral
    assert jnp.allclose(problem.initial_right_derivative, jnp.zeros((2, 2)))


def test_manifold_neutral_transport_must_return_current_state_tangent():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="transport must return a tangent",
    ):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: jnp.zeros_like(state),
            lambda time, args: jnp.eye(2),
            (
                phx.solver.DerivativeDelay(
                    "velocity",
                    phx.solver.ConstantDelay("lag", 0.25),
                    transport=lambda delayed, current, tangent, args: jnp.eye(2),
                ),
            ),
            t0=0.0,
            t1=0.4,
            history_derivative=lambda time, args: jnp.zeros((2, 2)),
            state_geometry=geometry,
        )


def test_trivial_geometry_does_not_require_neutral_transport():
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["velocity"],
        lambda time, args: jnp.asarray([jnp.exp(time)]),
        (
            phx.solver.DerivativeDelay(
                "velocity",
                phx.solver.ConstantDelay("lag", 0.5),
            ),
        ),
        t0=0.0,
        t1=0.2,
        history_derivative=lambda time, args: jnp.asarray([jnp.exp(time)]),
        state_geometry=phx.metrix.EuclideanStateGeometry(),
    )
    assert problem.neutral


def test_state_dependent_neutral_jump_roots_propagate_to_the_solve_horizon():
    factor = 0.5
    final_time = 2.6
    first_root = 1.0 / 0.9
    second_root = (first_root + 1.0) / 0.9
    point_delay = phx.solver.StateDependentDelay(
        "state_lag",
        lambda time, state, args: 1.0 + 0.1 * time,
        minimum_delay=1.0,
        maximum_delay=1.3,
        monotone_argument=True,
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: factor * memory["velocity"],
        lambda time, args: jnp.asarray([jax.lax.stop_gradient(time)]),
        (phx.solver.DerivativeDelay("velocity", point_delay),),
        t0=0.0,
        t1=final_time,
        history_derivative=lambda time, args: jnp.ones((1,)),
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([final_time]),
        rtol=1e-9,
        atol=1e-11,
        max_steps=512,
        max_discontinuities=16,
    )
    expected = (
        factor * first_root
        + factor**2 * (second_root - first_root)
        + factor**3 * (final_time - second_root)
    )

    assert jnp.allclose(solution.states[0, 0], expected, rtol=3e-5, atol=3e-6)
    assert solution.stats["state_dependent_tracking"] == "high-order-dynamic-roots"
    assert int(solution.stats["num_dynamic_discontinuity_roots"]) == 2
    assert int(solution.stats["num_internal_discontinuity_restarts"]) >= 2


@pytest.mark.parametrize("endpoint_weight", [0.0, 0.15])
def test_transformed_neutral_manufactured_solution_and_recovery_provenance(
    endpoint_weight,
):
    problem = _transformed_neutral_problem(
        endpoint_weight=endpoint_weight,
        t1=0.8,
    )
    solution = _solve_transformed_neutral(problem, 0.01, dense=True)
    expected = jnp.exp(0.3 * solution.times)

    assert jnp.allclose(solution.states[:, 0], expected, rtol=2e-3, atol=2e-4)
    query = jnp.asarray([0.13, 0.41, 0.77])
    assert jnp.allclose(
        solution.evaluate(query)[:, 0],
        jnp.exp(0.3 * query),
        rtol=2e-3,
        atol=2e-4,
    )
    expected_mode = "implicit-root" if endpoint_weight else "explicit"
    assert solution.stats["neutral_recovery_mode"] == expected_mode
    assert solution.metadata["delay_mode"] == "transformed-neutral"
    assert solution.resolved_method == "Euler:transformed-neutral-method-of-steps"


def test_transformed_neutral_euler_has_first_order_convergence():
    problem = _transformed_neutral_problem(t1=0.8)

    def error(step):
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([problem.t1]),
            dt0=step,
            max_steps=128,
        )
        return jnp.abs(solution.states[0, 0] - jnp.exp(0.3 * problem.t1))

    coarse = error(0.1)
    medium = error(0.05)
    fine = error(0.025)
    assert coarse / medium > 1.7
    assert medium / fine > 1.7


def test_transformed_neutral_full_rolling_and_segmented_execution_agree():
    problem = _transformed_neutral_problem(endpoint_weight=0.15, t1=1.2)
    whole = _solve_transformed_neutral(problem, 0.05, dense=True)
    rolling = _solve_transformed_neutral(
        problem,
        0.05,
        dense=True,
        history_mode="rolling",
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=whole.times,
        dt0=0.05,
        dense=True,
        max_steps_per_segment=4,
    )
    query = jnp.asarray([0.85, 1.05, 1.2])

    assert jnp.array_equal(rolling.states, whole.states)
    assert jnp.array_equal(segmented.states, whole.states)
    assert jnp.array_equal(rolling.evaluate(query), whole.evaluate(query))
    assert jnp.array_equal(segmented.evaluate(query), whole.evaluate(query))
    assert segmented.stats["neutral_recovery_mode"] == "implicit-root"
    assert segmented.metadata["delay_mode"] == "segmented-transformed-neutral"
    assert int(segmented.stats["num_segments"]) > 1


def test_transformed_neutral_implicit_recovery_is_jittable_vectorizable_and_differentiable():
    def terminal(amplitude):
        problem = _transformed_neutral_problem(
            amplitude=amplitude,
            endpoint_weight=0.15,
            t1=0.4,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.4]),
            dt0=0.05,
            max_steps=16,
        ).states[0, 0]

    compiled_terminal = eqx.filter_jit(terminal)
    unit = compiled_terminal(jnp.asarray(1.0))
    amplitudes = jnp.asarray([0.7, 1.0, 1.4])

    assert jnp.allclose(jax.vmap(terminal)(amplitudes), amplitudes * unit, rtol=2e-7)
    assert jnp.allclose(jax.grad(terminal)(jnp.asarray(1.0)), unit, rtol=2e-7)


def test_transformed_neutral_nonconvergent_endpoint_recovery_raises():
    problem = phx.solver.NeutralDelayProblem(
        lambda time, memory, args: jnp.zeros((1,)),
        lambda time, state, memory, args: jnp.ones((1,)),
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("past", 0.5),),
        endpoint_neutral=lambda time, state, memory, args: state + 1.0,
        recovery_max_steps=2,
        t0=0.0,
        t1=0.2,
    )

    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="maximum number of steps was reached in the nonlinear solver",
    ):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2]),
            dt0=0.1,
            max_steps=4,
        )


def test_transformed_neutral_rejects_unsupported_solver_and_geometry():
    problem = _transformed_neutral_problem()
    with pytest.raises(ValueError, match="requires diffrax.Euler"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.8]),
            solver=dfx.Heun(),
            dt0=0.05,
        )

    with pytest.raises(ValueError, match="requires Euclidean state geometry"):
        phx.solver.NeutralDelayProblem(
            lambda time, memory, args: 0.1 * memory["past"],
            lambda time, state, memory, args: state,
            lambda time, args: jnp.eye(2),
            (phx.solver.ConstantDelay("past", 0.4),),
            state_geometry=phx.metrix.SpecialOrthogonalStateGeometry(2),
            t0=0.0,
            t1=0.8,
        )
