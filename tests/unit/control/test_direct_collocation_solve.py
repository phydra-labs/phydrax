import jax.numpy as jnp
import pytest

import phydrax as phx


def _mesh(nodes=(0.0, 0.5, 1.0), *, identity="solve-direct-mesh"):
    return phx.discretization.TemporalMesh(
        jnp.asarray(nodes),
        role="collocation",
        mesh_id=identity,
    )


def _method():
    return phx.optim.FilterInteriorPoint(max_dense_dimension=128)


def _termination():
    return phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-7,
        relative_optimality=0.0,
        absolute_step=1.0e-12,
        relative_step=0.0,
        maximum_steps=80,
    )


def _analytic_dae_problem():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, control, args: jnp.asarray(
            (
                state_rate[0] - control[0],
                state[1] - state[0],
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="analytic-control-dae",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state[0],
        lower=1.0,
        upper=1.0,
        constraint_id="analytic-terminal",
    )
    return phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.asarray((0.0, 0.0)),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        trajectory_constraints=(terminal,),
        problem_id="analytic-direct-dae",
    )


def _plan(*, variable_duration=False, identity="analytic-direct-plan"):
    return phx.control.DirectCollocationPlan(
        _mesh(identity=f"{identity}:mesh"),
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        variable_duration=variable_duration,
        derivatives=phx.control.DirectCollocationDerivativePolicy(
            verify=True,
            num_verification_probes=2,
        ),
        audit=phx.control.DirectCollocationAuditPolicy(
            defect_tolerance=1.0e-6,
            constraint_tolerance=1.0e-6,
            off_grid_points=2,
        ),
        plan_id=identity,
    )


def test_native_direct_collocation_solves_analytic_controlled_dae():
    problem = _analytic_dae_problem()
    result = phx.control.solve_direct_collocation(
        problem,
        _plan(),
        jnp.asarray(((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))),
        jnp.ones((2, 1)),
        method=_method(),
        termination=_termination(),
    )
    assert bool(result.successful)
    assert bool(result.optimization_result.successful)
    assert jnp.allclose(result.decision.states[:, 0], jnp.asarray((0.0, 0.5, 1.0)), atol=1e-6)
    assert jnp.allclose(result.decision.states[:, 1], result.decision.states[:, 0], atol=1e-6)
    assert jnp.allclose(result.decision.controls, 1.0, atol=1e-6)
    assert result.diagnostics.maximum_defect <= 1e-6
    assert result.diagnostics.maximum_constraint_violation <= 1e-6
    assert not result.diagnostics.off_grid_certified
    assert result.optimization_result.certificate is not None


def test_variable_duration_recovers_unit_time_integrator_solution():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="variable-time-integrator",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, context: trajectory.final_state,
        lower=1.0,
        upper=1.0,
        constraint_id="variable-time-terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.asarray((0.0,)),
        running_cost=lambda time, state, control, context: control[0] ** 2,
        trajectory_cost=lambda trajectory, context: context.duration,
        trajectory_constraints=(terminal,),
        problem_id="variable-time-direct",
    )
    result = phx.control.solve_direct_collocation(
        problem,
        _plan(variable_duration=True, identity="variable-time-plan"),
        jnp.asarray(((0.0,), (0.5,), (1.0,))),
        jnp.ones((2, 1)),
        duration_guess=1.0,
        bounds=phx.control.DirectCollocationBounds(duration=(0.25, 4.0)),
        method=_method(),
        termination=_termination(),
    )
    assert bool(result.successful)
    assert jnp.allclose(result.duration, 1.0, atol=2e-5)
    assert jnp.allclose(result.decision.controls, 1.0, atol=2e-5)
    assert jnp.allclose(result.objective, 2.0, atol=2e-5)


def test_continuous_control_problem_is_a_lossless_fixed_duration_input():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.5, 1.0)),
        time_id="control-adapter-time",
    )
    dynamics = phx.control.DifferentialControlDynamics(
        phx.dynamics.ContinuousSystem(
            lambda time, state, control, args: control,
            state_layout=phx.dynamics.StateLayout((1,)),
            input_layout=phx.dynamics.InputLayout((1,), roles="control"),
            system_id="control-adapter-system",
        )
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray((0.0,)),
        running_cost=lambda time, state, control, args: control[0] ** 2,
        terminal_constraints=(lambda time, state, args: (state[0] - 1.0) ** 2,),
        problem_id="control-adapter-problem",
    )
    plan = phx.control.DirectCollocationPlan(
        phx.discretization.TemporalMesh(
            grid.times,
            role="collocation",
            mesh_id="control-adapter-mesh",
        ),
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        derivatives=phx.control.DirectCollocationDerivativePolicy(verify=False),
        plan_id="control-adapter-plan",
    )
    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        jnp.asarray(((0.0,), (0.5,), (1.0,))),
        jnp.ones((2, 1)),
    )
    assert compilation.problem.problem_id == problem.problem_id
    assert compilation.problem.initial_state is not None
    assert jnp.allclose(compilation.problem.initial_state, problem.initial_state)
    assert len(compilation.problem.path_constraints) == 0
    assert len(compilation.problem.trajectory_constraints) == 1


def test_off_grid_audit_preserves_matrix_state_and_control_events():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((2, 2)),
        input_layout=phx.dynamics.InputLayout((2, 2), roles="control"),
        system_id="matrix-event-integrator",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state,
        lower=jnp.ones((2, 2)),
        upper=jnp.ones((2, 2)),
        constraint_id="matrix-event-terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros((2, 2)),
        running_cost=lambda time, state, control, args: jnp.sum(control**2),
        trajectory_constraints=(terminal,),
        problem_id="matrix-event-direct",
    )
    times = jnp.asarray((0.0, 0.5, 1.0))
    result = phx.control.solve_direct_collocation(
        problem,
        _plan(identity="matrix-event-plan"),
        times[:, None, None] * jnp.ones((1, 2, 2)),
        jnp.ones((2, 2, 2)),
        method=_method(),
        termination=_termination(),
    )
    assert bool(result.successful)
    assert result.decision.states.shape == (3, 2, 2)
    assert result.diagnostics.maximum_off_grid_defect <= 1e-7



def test_direct_collocation_rejects_out_of_bound_initial_guess():
    problem = _analytic_dae_problem()
    with pytest.raises(ValueError, match="violates its bounds"):
        phx.control.compile_direct_collocation(
            problem,
            _plan(identity="bounded-direct-plan"),
            jnp.asarray(((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))),
            2.0 * jnp.ones((2, 1)),
            bounds=phx.control.DirectCollocationBounds(
                controls=phx.optim.Bounds(-1.0, 1.0)
            ),
        )


def test_optional_ipopt_structured_route_matches_analytic_solution():
    pytest.importorskip("cyipopt")
    problem = _analytic_dae_problem()
    result = phx.control.solve_direct_collocation(
        problem,
        _plan(identity="ipopt-direct-plan"),
        jnp.asarray(((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))),
        jnp.ones((2, 1)),
        method=phx.optim.IpoptMinimize(options={"print_level": 0}),
        termination=_termination(),
    )
    assert bool(result.successful)
    assert result.optimization_result.provenance.backend == "ipopt"
    assert jnp.allclose(result.decision.controls, 1.0, atol=1e-6)
