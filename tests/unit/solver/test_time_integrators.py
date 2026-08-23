import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _decay_problem(rate=1.0, *, t1=1.0):
    return phx.solver.DifferentialProblem(
        lambda time, state, value: -value * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=t1,
        args=jnp.asarray(rate),
        problem_id=f"test-decay-{rate}-{t1}",
    )


def _decay_dae(rate=1.0):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, value: state_rate + value * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id=f"test-decay-dae-{rate}",
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray([1.0]),
        args=jnp.asarray(rate),
        problem_id=f"test-decay-dae-problem-{rate}",
    )


def test_diffrax_preflight_resolves_fixed_solver_and_records_configuration():
    solution = phx.solver.solve_diffrax(
        _decay_problem(),
        save_times=jnp.asarray([0.0, 1.0]),
        solver=dfx.Euler(),
        dt0=0.001,
    )

    assert solution.successful
    assert solution.temporal_evidence is not None
    assert not solution.temporal_evidence.adaptive
    assert solution.problem_id == "test-decay-1.0-1.0"
    assert jnp.allclose(solution.states[-1, 0], jnp.exp(-1.0), rtol=8e-4)

    with pytest.raises(ValueError, match="does not provide an error estimate"):
        phx.solver.solve_diffrax(
            _decay_problem(),
            save_times=jnp.asarray([1.0]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
            dt0=0.01,
        )


def test_split_differential_problem_unlocks_kencarp_and_gradients():
    problem = phx.solver.SplitDifferentialProblem(
        lambda time, state, args: args[0] * state,
        lambda time, state, args: args[1] * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray([1.0, -10.0]),
        problem_id="test-imex-decay",
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 1.0]),
        dt0=0.01,
        rtol=1e-7,
        atol=1e-9,
    )

    assert solution.successful
    assert solution.solver_name == "KenCarp4"
    assert solution.temporal_evidence.equation_form == "additive-ode"
    assert jnp.allclose(solution.states[-1, 0], jnp.exp(-9.0), rtol=2e-4)

    def terminal(explicit_rate):
        candidate = phx.solver.SplitDifferentialProblem(
            lambda time, state, args: args * state,
            lambda time, state, args: -10.0 * state,
            jnp.asarray([1.0]),
            t0=0.0,
            t1=0.2,
            args=explicit_rate,
            problem_id="test-imex-gradient",
        )
        return phx.solver.solve_diffrax(
            candidate,
            save_times=jnp.asarray([0.2]),
            dt0=0.01,
        ).states[-1, 0]

    value, gradient = jax.jit(jax.value_and_grad(terminal))(jnp.asarray(1.0))
    assert jnp.allclose(value, jnp.exp(-1.8), rtol=1e-4)
    assert jnp.allclose(gradient, 0.2 * jnp.exp(-1.8), rtol=2e-3)


def test_ssprk_methods_have_expected_smooth_accuracy_and_shared_fv_kernel():
    def integrate(step, method):
        state = jnp.asarray([1.0])
        time = jnp.asarray(0.0)
        count = int(round(1.0 / step))
        for _ in range(count):
            state = method(
                lambda t, y, args: -y,
                time,
                state,
                jnp.asarray(step),
            )
            time = time + step
        return state[0]

    error33 = jnp.abs(integrate(0.1, phx.solver.ssprk33_step) - jnp.exp(-1.0))
    error54 = jnp.abs(integrate(0.1, phx.solver.ssprk54_step) - jnp.exp(-1.0))
    assert error33 < 2e-5
    assert error54 < error33

    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="temporal-kernel-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "temporal-kernel-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.WENOReconstructionPlan(5),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    constant = jnp.ones((16, 1))
    stepper = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(dynamics)
    result = stepper.advance(jnp.asarray(0.0), constant, 0.01)
    direct = phx.solver.ssprk33_step(
        dynamics, jnp.asarray(0.0), constant, jnp.asarray(0.01)
    )
    assert jnp.array_equal(result.state, constant)
    assert jnp.array_equal(result.state, direct)
    assert result.temporal_method_id == "temporal:ssprk:3:3"


def test_theta_endpoint_and_higher_bdf_follow_expected_decay():
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 21), time_id="theta")
    theta = phx.solver.solve_dae(
        _decay_dae(),
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.ThetaMethod(0.5, endpoint=True)
        ),
    )
    factor = (1.0 - 0.5 * grid.durations[0]) / (1.0 + 0.5 * grid.durations[0])
    assert theta.successful
    assert theta.method_id.startswith("temporal:theta")
    assert jnp.allclose(theta.states[-1, 0], factor**grid.num_steps, rtol=2e-6)

    bdf5 = phx.solver.solve_dae(
        _decay_dae(),
        grid,
        policy=phx.solver.DAESolvePolicy(method=phx.solver.BDFMethod(5)),
    )
    assert bdf5.successful
    assert int(jnp.max(bdf5.step_history.orders)) == 5
    assert jnp.allclose(bdf5.states[-1, 0], jnp.exp(-1.0), rtol=2e-3)


def test_matrix_free_rosenbrock_w_is_stable_and_differentiable():
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 33), time_id="rosenbrock")

    def terminal(rate):
        problem = _decay_problem(rate)
        return phx.solver.solve_rosenbrock(problem, grid, args=rate).states[-1, 0]

    value, gradient = jax.jit(jax.value_and_grad(terminal))(jnp.asarray(10.0))
    assert jnp.isfinite(value)
    assert jnp.allclose(value, jnp.exp(-10.0), rtol=1e-2, atol=2e-7)
    assert jnp.allclose(gradient, -jnp.exp(-10.0), rtol=5e-2, atol=2e-7)


def test_adaptive_rosenbrock_replays_a_frozen_accepted_grid():
    problem = _decay_problem(10.0)
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.5, 1.0]), time_id="rosenbrock-adaptive"
    )
    controller = phx.solver.RosenbrockAdaptivePolicy(
        relative_tolerance=1e-5,
        absolute_tolerance=1e-8,
        initial_step=0.05,
        maximum_accepted_steps=256,
        maximum_attempts=512,
    )

    def terminal(rate):
        return phx.solver.solve_rosenbrock_adaptive(
            problem,
            grid,
            adaptive=controller,
            args=rate,
        ).states[-1, 0]

    solution = phx.solver.solve_rosenbrock_adaptive(problem, grid, adaptive=controller)
    value, gradient = jax.jit(jax.value_and_grad(terminal))(jnp.asarray(10.0))
    assert solution.successful
    assert solution.temporal_evidence.adaptive
    assert int(solution.stats["accepted_steps"]) > grid.num_steps
    assert jnp.allclose(value, jnp.exp(-10.0), rtol=3e-4)
    assert jnp.allclose(gradient, -jnp.exp(-10.0), rtol=3e-4)


def test_generalized_alpha_tracks_undamped_oscillator():
    system = phx.dynamics.SecondOrderDifferentialSystem(
        lambda time, configuration, velocity, acceleration, omega: (
            acceleration + omega**2 * configuration
        ),
        state_shape=(1,),
        system_id="test-second-order-oscillator",
    )
    problem = phx.dynamics.SecondOrderDifferentialProblem(
        system,
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        initial_acceleration=jnp.asarray([-1.0]),
        args=jnp.asarray(1.0),
        problem_id="test-second-order-problem",
    )
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 41), time_id="generalized-alpha")
    solution = phx.solver.solve_generalized_alpha(problem, grid)

    assert solution.successful
    assert jnp.allclose(solution.configurations[-1, 0], jnp.cos(1.0), atol=2e-4)
    assert jnp.allclose(solution.velocities[-1, 0], -jnp.sin(1.0), atol=2e-4)


def test_variable_step_bdf_coefficients_satisfy_order_conditions_through_five():
    from phydrax.solver._bdf_method import bdf_coefficients

    target = jnp.asarray(0.37)
    history = jnp.asarray([0.2, 0.05, -0.1, -0.3, -0.55])
    nodes = jnp.concatenate((target[None], history))
    for order in range(1, 6):
        coefficients = bdf_coefficients(history, target, jnp.asarray(order))
        for degree in range(order + 1):
            observed = jnp.dot(coefficients, nodes**degree)
            expected = (
                jnp.asarray(0.0) if degree == 0 else degree * target ** (degree - 1)
            )
            assert jnp.allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_rosenbrock_and_gauss_collocation_reach_declared_orders():
    problem = _decay_problem()
    exact = jnp.exp(-1.0)
    rosenbrock_errors = []
    for steps in (4, 8):
        grid = phx.dynamics.TimeGrid(
            jnp.linspace(0.0, 1.0, steps + 1), time_id=f"rosenbrock-order-{steps}"
        )
        value = phx.solver.solve_rosenbrock(problem, grid).states[-1, 0]
        rosenbrock_errors.append(jnp.abs(value - exact))
    assert rosenbrock_errors[0] / rosenbrock_errors[1] > 7.0

    gauss_errors = []
    for steps in (2, 4):
        grid = phx.dynamics.TimeGrid(
            jnp.linspace(0.0, 1.0, steps + 1), time_id=f"gauss-order-{steps}"
        )
        value = phx.solver.solve_implicit_runge_kutta(
            problem,
            grid,
            method=phx.solver.GaussLegendreIRK(2),
        ).states[-1, 0]
        gauss_errors.append(jnp.abs(value - exact))
    assert gauss_errors[0] / gauss_errors[1] > 15.0


def test_partitioned_multirate_and_gauss_irk_preserve_declared_contracts():
    partition = phx.solver.StatePartition(
        {
            "slow": jnp.asarray([True, False]),
            "fast": jnp.asarray([False, True]),
        }
    )
    problem = phx.solver.PartitionedDifferentialProblem(
        lambda time, state, args: jnp.asarray([-state[0], 0.0]),
        lambda time, state, args: jnp.asarray([0.0, -5.0 * state[1]]),
        jnp.asarray([1.0, 1.0]),
        t0=0.0,
        t1=1.0,
        partition=partition,
        problem_id="test-partitioned-decay",
    )
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 21), time_id="multirate")
    method = phx.solver.MultiratePartitionedRK(3, refinement_ratio=3)
    multirate = phx.solver.solve_multirate(
        problem,
        grid,
        method=method,
    )
    assert multirate.successful
    assert jnp.allclose(
        multirate.states[-1], jnp.asarray([jnp.exp(-1.0), jnp.exp(-5.0)]), rtol=2e-3
    )
    amr = phx.solver.multirate_amr_subcycling_plan(method)
    assert amr.refinement_ratio == method.refinement_ratio
    assert amr.temporal_method_id == method.method_id

    irk_grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 9), time_id="irk")
    irk = phx.solver.solve_implicit_runge_kutta(
        _decay_problem(),
        irk_grid,
        method=phx.solver.GaussLegendreIRK(2),
        dense=True,
    )
    assert irk.successful
    assert jnp.allclose(irk.states[-1, 0], jnp.exp(-1.0), rtol=3e-6)
    assert jnp.allclose(irk.evaluate(jnp.asarray(0.5))[0], jnp.exp(-0.5), rtol=2e-5)
