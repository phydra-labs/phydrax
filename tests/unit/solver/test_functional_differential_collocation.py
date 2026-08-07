import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _left_boundary(value=0.0):
    def boundary(left, right, trajectory, args):
        del right, trajectory, args
        return left - value

    return boundary


def _linear_guess(time, args):
    del args
    return time


def test_manufactured_retarded_equation_uses_global_collocation():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: 2.0 * values[0] + 1.0 - time,
        argument_times=lambda time, state, args: jnp.asarray([0.5 * time]),
        num_arguments=1,
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.linspace(0.0, 1.0, 4), degree=2)

    solution = phx.solver.solve_functional_differential(problem, plan, _linear_guess)
    query = jnp.linspace(0.0, 1.0, 17)

    assert bool(solution.successful)
    assert solution.resolved_method == "root"
    assert solution.metadata["global_semantics"]
    assert solution.solver_id == "solver:functional-differential-collocation"
    assert jnp.allclose(solution.evaluate(query), query, atol=2e-9)
    assert jnp.allclose(solution.derivative(query), jnp.ones_like(query), atol=2e-9)
    assert float(solution.max_differential_residual) < 2e-9


@pytest.mark.parametrize(
    ("argument_times", "num_arguments", "vector_field"),
    [
        (
            lambda time, state, args: jnp.asarray([0.5 * (time + 1.0)]),
            1,
            lambda time, state, values, args: 2.0 * values[0] - time,
        ),
        (
            lambda time, state, args: jnp.asarray([0.5 * time, 0.5 * (time + 1.0)]),
            2,
            lambda time, state, values, args: values[0] + values[1] + 0.5 - time,
        ),
    ],
)
def test_manufactured_advanced_and_mixed_equations(
    argument_times, num_arguments, vector_field
):
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        vector_field,
        argument_times=argument_times,
        num_arguments=num_arguments,
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(
        jnp.asarray([0.0, 0.25, 0.7, 1.0]), degree=2
    )

    solution = phx.solver.solve_functional_differential(problem, plan, _linear_guess)
    query = jnp.linspace(0.0, 1.0, 21)

    assert bool(solution.successful)
    assert jnp.allclose(solution.evaluate(query), query, atol=3e-9)
    assert not isinstance(problem, phx.solver.DelayDifferentialProblem)


def test_state_dependent_functional_argument_is_evaluated_from_global_iterate():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: 2.0 * values[0] + 1.0 - time,
        argument_times=lambda time, state, args: jnp.asarray([0.5 * state]),
        num_arguments=1,
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.linspace(0.0, 1.0, 5), degree=2)

    solution = phx.solver.solve_functional_differential(problem, plan, _linear_guess)

    assert bool(solution.successful)
    assert jnp.allclose(solution.states, plan.mesh, atol=3e-9)


def test_periodic_orbit_phase_condition_uses_overdetermined_least_squares():
    period = 2.0 * jnp.pi

    def rotation(time, state, values, args):
        del time, values, args
        return jnp.asarray([-state[1], state[0]])

    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        rotation,
        state_shape=(2,),
        periodic=True,
        phase=lambda trajectory, args: trajectory.evaluate(0.0)[0] - 1.0,
    )
    plan = phx.solver.FunctionalCollocationPlan(
        jnp.linspace(0.0, period, 5), degree=7, method="auto"
    )

    def initial(time, args):
        del args
        return jnp.asarray([jnp.cos(time), jnp.sin(time)])

    solution = phx.solver.solve_functional_differential(problem, plan, initial)
    query = jnp.linspace(0.0, period, 33)
    expected = jnp.stack((jnp.cos(query), jnp.sin(query)), axis=-1)

    assert bool(solution.successful)
    assert solution.resolved_method == "least-squares"
    assert solution.residual_size == solution.unknown_size + 1
    assert jnp.allclose(solution.evaluate(query), expected, atol=2e-5)
    assert float(jnp.max(jnp.abs(solution.periodic_residual))) < 2e-7
    assert float(jnp.max(jnp.abs(solution.phase_residual))) < 2e-7


def test_mesh_and_degree_refinement_reduce_advanced_solution_error():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: values[0] ** 2 / jnp.e,
        argument_times=lambda time, state, args: jnp.asarray([0.5 * (time + 1.0)]),
        num_arguments=1,
        boundary=_left_boundary(1.0),
        observation_times=jnp.asarray([1.0]),
        observation_values=jnp.asarray([jnp.e]),
    )

    def initial(time, args):
        del args
        return jnp.exp(time)

    coarse = phx.solver.solve_functional_differential(
        problem,
        phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 1.0]), degree=2),
        initial,
    )
    refined = phx.solver.solve_functional_differential(
        problem,
        phx.solver.FunctionalCollocationPlan(jnp.linspace(0.0, 1.0, 4), degree=5),
        initial,
    )
    query = jnp.linspace(0.0, 1.0, 41)
    exact = jnp.exp(query)
    coarse_error = jnp.max(jnp.abs(coarse.evaluate(query) - exact))
    refined_error = jnp.max(jnp.abs(refined.evaluate(query) - exact))

    assert bool(coarse.successful)
    assert bool(refined.successful)
    assert refined_error < 0.05 * coarse_error


def test_independent_element_polynomials_are_joined_by_continuity_residuals():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: jnp.ones_like(state),
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(
        jnp.asarray([0.0, 0.15, 0.6, 1.0]), degree=3
    )
    solution = phx.solver.solve_functional_differential(problem, plan, jnp.asarray(0.0))

    jumps = solution.collocation_values[:-1, -1] - solution.collocation_values[1:, 0]
    assert solution.continuity_residual.shape == (2,)
    assert jnp.allclose(jumps, solution.continuity_residual)
    assert float(solution.max_continuity_residual) < 2e-9
    assert jnp.allclose(solution.evaluate(plan.mesh), plan.mesh, atol=2e-9)


def test_collocation_preserves_arbitrary_array_state_shape():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: jnp.ones_like(state),
        state_shape=(2, 2),
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 0.4, 1.0]), degree=2)
    solution = phx.solver.solve_functional_differential(problem, plan, jnp.zeros((2, 2)))
    query = jnp.asarray([0.1, 0.6, 0.9])

    assert solution.collocation_values.shape == (2, 3, 2, 2)
    assert solution.evaluate(query).shape == (3, 2, 2)
    assert jnp.allclose(
        solution.evaluate(query),
        jnp.broadcast_to(query[:, None, None], (3, 2, 2)),
        atol=2e-9,
    )


def test_observations_select_least_squares_and_preserve_observable_residual():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: jnp.ones_like(state),
        boundary=_left_boundary(),
        observation_times=jnp.asarray([0.25, 0.75]),
        observation_values=jnp.asarray([0.25, 0.75]),
        observation_weights=jnp.asarray([2.0, 3.0]),
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 0.5, 1.0]), degree=2)

    solution = phx.solver.solve_functional_differential(problem, plan, jnp.asarray(0.0))

    assert bool(solution.successful)
    assert solution.resolved_method == "least-squares"
    assert solution.residual_size == solution.unknown_size + 2
    assert float(jnp.max(jnp.abs(solution.observation_residual))) < 2e-8


def test_explicit_root_rejects_overdetermined_residual_shape():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: jnp.ones_like(state),
        boundary=_left_boundary(),
        observation_times=jnp.asarray([0.5]),
        observation_values=jnp.asarray([0.5]),
    )
    plan = phx.solver.FunctionalCollocationPlan(
        jnp.asarray([0.0, 1.0]), degree=2, method="root"
    )

    with pytest.raises(ValueError, match="requires a square residual"):
        phx.solver.solve_functional_differential(problem, plan, jnp.asarray(0.0))


def test_missing_boundary_constraint_is_reported_as_underdetermined():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: jnp.ones_like(state)
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 1.0]), degree=2)

    with pytest.raises(ValueError, match="underdetermined"):
        phx.solver.solve_functional_differential(problem, plan, jnp.asarray(0.0))


def test_functional_argument_outside_global_interval_fails_explicitly():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: values[0],
        argument_times=lambda time, state, args: jnp.asarray([time + 2.0]),
        num_arguments=1,
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 1.0]), degree=2)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="outside the declared mesh interval",
    ):
        phx.solver.solve_functional_differential(problem, plan, jnp.asarray(0.0))


def test_nonlinear_nonconvergence_is_returned_when_throw_is_disabled():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: state**2 + 1.0,
        boundary=_left_boundary(),
    )
    plan = phx.solver.FunctionalCollocationPlan(
        jnp.asarray([0.0, 0.8]), degree=5, max_steps=1, throw=False
    )

    solution = phx.solver.solve_functional_differential(problem, plan, jnp.asarray(0.0))

    assert not bool(solution.converged)
    assert not bool(solution.successful)
    assert solution.status_message
    assert "num_steps" in solution.stats


def test_functional_collocation_solve_is_jittable():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, rate: rate,
        boundary=lambda left, right, trajectory, rate: left,
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 0.4, 1.0]), degree=2)

    solve_endpoint = jax.jit(
        lambda rate: phx.solver.solve_functional_differential(
            problem, plan, jnp.asarray(0.0), args=rate
        ).evaluate(1.0)
    )

    assert jnp.allclose(solve_endpoint(jnp.asarray(2.5)), 2.5, atol=2e-9)


def test_parameter_gradient_uses_optimistix_implicit_solve():
    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, rate: rate * (1.0 + time),
        boundary=lambda left, right, trajectory, rate: left,
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 0.3, 1.0]), degree=2)

    def endpoint(rate):
        solution = phx.solver.solve_functional_differential(
            problem, plan, jnp.asarray(0.0), args=rate
        )
        return solution.evaluate(1.0)

    derivative = jax.grad(endpoint)(jnp.asarray(0.7))

    assert jnp.allclose(endpoint(jnp.asarray(0.7)), 1.05, atol=2e-9)
    assert jnp.allclose(derivative, 1.5, atol=2e-8)


def test_unknown_parameter_is_inferred_and_differentiable_through_context():
    def vector_field(time, state, values, context):
        del time, state, values
        assert isinstance(context, phx.solver.FunctionalDifferentialContext)
        return context.parameters

    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        vector_field,
        parameter_shape=(),
        boundary=_left_boundary(),
        observation_residual=lambda trajectory, context: (
            trajectory.evaluate(1.0) - context.args
        ),
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 0.35, 1.0]), degree=2)

    def infer(target):
        return phx.solver.solve_functional_differential(
            problem,
            plan,
            jnp.asarray(0.0),
            args=target,
            parameter_guess=jnp.asarray(1.0),
        ).parameters

    inferred = jax.jit(infer)(jnp.asarray(2.5))
    sensitivity = jax.grad(infer)(jnp.asarray(2.5))

    assert jnp.allclose(inferred, 2.5, atol=2e-9)
    assert jnp.allclose(sensitivity, 1.0, atol=2e-8)


def test_unknown_period_rescales_physical_time_and_has_implicit_gradient():
    def oscillator(time, state, values, context):
        del time, values
        frequency = context.args
        rotation = frequency * jnp.asarray([-state[1], state[0]])
        radial = (1.0 - jnp.sum(state**2)) * state
        return rotation + radial

    def phase(trajectory, context):
        assert context.period is not None
        return trajectory.evaluate(0.0)[0] - 1.0

    problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        oscillator,
        state_shape=(2,),
        periodic=True,
        phase=phase,
        unknown_period=True,
    )
    plan = phx.solver.FunctionalCollocationPlan(jnp.linspace(0.0, 1.0, 5), degree=7)
    angle = 2.0 * jnp.pi * plan.collocation_times
    initial = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)

    def infer_period(frequency):
        return phx.solver.solve_functional_differential(
            problem,
            plan,
            initial,
            args=frequency,
            period_guess=jnp.asarray(6.0),
        ).period

    period = jax.jit(infer_period)(jnp.asarray(1.0))
    sensitivity = jax.grad(infer_period)(jnp.asarray(1.0))
    solution = phx.solver.solve_functional_differential(
        problem,
        plan,
        initial,
        args=jnp.asarray(1.0),
        period_guess=jnp.asarray(6.0),
    )

    assert bool(solution.successful)
    assert solution.resolved_method == "root"
    assert solution.metadata["solved_period"]
    assert jnp.allclose(period, 2.0 * jnp.pi, atol=2e-6)
    assert jnp.allclose(sensitivity, -2.0 * jnp.pi, atol=2e-5)
    assert jnp.allclose(solution.times[-1], solution.period, atol=2e-9)
    assert jnp.allclose(
        solution.evaluate(0.25 * solution.period),
        jnp.asarray([0.0, 1.0]),
        atol=2e-6,
    )


def test_unknown_parameter_and_period_contracts_fail_before_nonlinear_solve():
    plan = phx.solver.FunctionalCollocationPlan(jnp.asarray([0.0, 1.0]), degree=2)
    ordinary = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, args: jnp.ones_like(state),
        boundary=_left_boundary(),
    )
    with pytest.raises(ValueError, match="parameter_shape"):
        phx.solver.solve_functional_differential(
            ordinary,
            plan,
            _linear_guess,
            parameter_guess=jnp.asarray(1.0),
        )
    with pytest.raises(ValueError, match="unknown_period"):
        phx.solver.solve_functional_differential(
            ordinary,
            plan,
            _linear_guess,
            period_guess=jnp.asarray(1.0),
        )

    parameter_problem = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, context: context.parameters[0],
        boundary=_left_boundary(),
        parameter_shape=(1,),
    )
    with pytest.raises(ValueError, match="parameter_guess is required"):
        phx.solver.solve_functional_differential(
            parameter_problem,
            plan,
            _linear_guess,
        )
    with pytest.raises(ValueError, match="must match"):
        phx.solver.solve_functional_differential(
            parameter_problem,
            plan,
            _linear_guess,
            parameter_guess=jnp.ones((2,)),
        )

    with pytest.raises(ValueError, match="periodic"):
        phx.solver.FunctionalDifferentialBoundaryProblem(
            lambda time, state, values, context: state,
            unknown_period=True,
            phase=lambda trajectory, context: trajectory.evaluate(0.0),
        )
    with pytest.raises(ValueError, match="phase"):
        phx.solver.FunctionalDifferentialBoundaryProblem(
            lambda time, state, values, context: state,
            periodic=True,
            unknown_period=True,
        )
    periodic = phx.solver.FunctionalDifferentialBoundaryProblem(
        lambda time, state, values, context: state,
        periodic=True,
        unknown_period=True,
        phase=lambda trajectory, context: trajectory.evaluate(0.0),
    )
    with pytest.raises(ValueError, match="period_guess is required"):
        phx.solver.solve_functional_differential(
            periodic,
            plan,
            jnp.ones((1, 3)),
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="finite and positive",
    ):
        phx.solver.solve_functional_differential(
            periodic,
            plan,
            jnp.ones((1, 3)),
            period_guess=jnp.asarray(0.0),
        )
