import jax
import jax.numpy as jnp

import phydrax as phx


def test_direct_collocation_decision_uses_six_pose_coordinates():
    geometry = phx.metrix.QuaternionPoseStateGeometry()
    local_space = phx.linalg.ArraySpace((6,), dtype=jnp.float32)
    state_layout = phx.dynamics.StateLayout(
        (7,),
        geometry=geometry,
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:direct-collocation-pose",
    )
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    anchors = jnp.stack((pose, pose))
    layout = phx.control.DirectCollocationDecisionLayout(
        state_layout=state_layout,
        state_anchors=anchors,
        state_array_shape=(2, 7),
        control_array_shape=(1, 1),
        parameter_space=None,
        state_scale=jnp.ones((2, 6)),
        control_scale=jnp.ones((1, 1)),
        parameter_scale=None,
        duration_scale=jnp.asarray(1.0),
        variable_duration=False,
        layout_id="test:direct-collocation-pose-decision",
    )
    equivalent = anchors.at[:, :4].multiply(-1.0)
    coordinates = layout.pack(
        phx.control.DirectCollocationDecision(
            equivalent,
            jnp.zeros((1, 1)),
            None,
            None,
        )
    )
    decoded = layout.unpack(coordinates)

    assert layout.state_coordinate_shape == (2, 6)
    assert coordinates.shape == (13,)
    assert jnp.allclose(
        jax.vmap(geometry.inverse_retract)(equivalent, decoded.states),
        0.0,
    )


def test_direct_collocation_pose_defect_uses_exact_six_dimensional_tangent():
    geometry = phx.metrix.QuaternionPoseStateGeometry()
    local_space = phx.linalg.ArraySpace((6,), dtype=jnp.float32)
    state_layout = phx.dynamics.StateLayout(
        (7,),
        geometry=geometry,
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:direct-collocation-pose-defect",
    )
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: jnp.zeros_like(state),
        state_layout=state_layout,
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="test:direct-collocation-stationary-pose",
    )
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    equivalent = pose.at[:4].multiply(-1.0)
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=pose,
        problem_id="test:direct-collocation-pose-problem",
    )
    mesh = phx.discretization.TemporalMesh(
        jnp.asarray([0.0, 1.0]),
        role="collocation",
        mesh_id="test:direct-collocation-pose-mesh",
    )
    plan = phx.control.DirectCollocationPlan(
        mesh,
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        derivatives=phx.control.DirectCollocationDerivativePolicy(verify=False),
        plan_id="test:direct-collocation-pose-plan",
    )

    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        jnp.stack((pose, equivalent)),
        jnp.zeros((1, 1)),
    )
    values = compilation.values(compilation.initial_coordinates)

    assert compilation.decision_layout.state_coordinate_shape == (2, 6)
    assert values.stage_states.shape == (1, 7)
    assert values.state_rates.shape == (1, 6)
    assert values.dynamics.shape == (1, 6)
    assert values.initial.shape == (6,)
    assert jnp.allclose(values.state_rates, 0.0)
    assert jnp.allclose(values.dynamics, 0.0)
    assert jnp.allclose(values.initial, 0.0)


def _mesh(nodes=(0.0, 0.25, 1.0), *, identity="direct-mesh"):
    return phx.discretization.TemporalMesh(
        jnp.asarray(nodes),
        role="collocation",
        mesh_id=identity,
    )


def _integrator_problem(*, running_cost=None, trajectory_cost=None):
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="direct-integrator",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state,
        lower=1.0,
        upper=1.0,
        constraint_id="unit-terminal",
    )
    return phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.asarray((0.0,)),
        running_cost=running_cost,
        trajectory_cost=trajectory_cost,
        trajectory_constraints=(terminal,),
        problem_id="direct-integrator-problem",
    )


def _plan(method, *, variable_duration=False, hessian="limited-memory"):
    return phx.control.DirectCollocationPlan(
        _mesh(),
        method=method,
        variable_duration=variable_duration,
        derivatives=phx.control.DirectCollocationDerivativePolicy(
            hessian=hessian,
            verify=True,
            num_verification_probes=2,
        ),
        plan_id=f"direct-plan:{method.method_id}:{variable_duration}:{hessian}",
    )


def test_backward_euler_transcription_uses_nonuniform_physical_widths():
    problem = _integrator_problem()
    plan = _plan(phx.solver.ThetaMethod(1.0, endpoint=True))
    states = jnp.asarray(((0.0,), (0.5,), (2.0,)))
    controls = jnp.asarray(((1.0,), (3.0,)))
    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        states,
        controls,
    )
    values = compilation.values(compilation.initial_coordinates)

    assert jnp.allclose(values.stage_times, jnp.asarray((0.25, 1.0)))
    assert jnp.allclose(values.stage_states, states[1:])
    assert jnp.allclose(values.state_rates[:, 0], jnp.asarray((2.0, 2.0)))
    assert jnp.allclose(values.dynamics[:, 0], jnp.asarray((1.0, -1.0)))
    assert compilation.constraint_layout.dynamics_slice == (0, 2)
    assert compilation.constraint_layout.initial_slice == (2, 3)
    assert bool(compilation.jacobian_verification.passed)

    control_start, control_stop = compilation.decision_layout.control_slice
    jacobian_columns = set(
        map(int, compilation.structured_program.jacobian_plan.pattern.cols.tolist())
    )
    assert set(range(control_start, control_stop)) <= jacobian_columns


def test_midpoint_objective_gradient_matches_its_discretized_scalar():
    running = lambda time, state, control, args: state[0] ** 2
    problem = _integrator_problem(running_cost=running)
    plan = _plan(phx.solver.ThetaMethod(0.5, endpoint=False))
    states = jnp.asarray(((0.2,), (0.6,), (1.4,)))
    controls = jnp.zeros((2, 1))
    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        states,
        controls,
    )
    gradient = jax.grad(
        lambda coordinates: compilation.minimization_problem.value(
            coordinates, problem.args
        )[0]
    )(compilation.initial_coordinates)
    state_start, state_stop = compilation.decision_layout.state_slice
    state_gradient = gradient[state_start:state_stop]
    expected = jnp.asarray(
        (
            0.25 * (states[0, 0] + states[1, 0]) / 2.0,
            0.25 * (states[0, 0] + states[1, 0]) / 2.0
            + 0.75 * (states[1, 0] + states[2, 0]) / 2.0,
            0.75 * (states[1, 0] + states[2, 0]) / 2.0,
        )
    )
    assert jnp.allclose(state_gradient, expected)


def test_variable_duration_rescales_time_and_state_rate():
    problem = _integrator_problem()
    plan = _plan(
        phx.solver.ThetaMethod(0.5, endpoint=False),
        variable_duration=True,
    )
    states = jnp.asarray(((0.0,), (0.5,), (2.0,)))
    controls = jnp.zeros((2, 1))
    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        states,
        controls,
        duration_guess=2.0,
        bounds=phx.control.DirectCollocationBounds(duration=(0.5, 4.0)),
    )
    values = compilation.values(compilation.initial_coordinates)
    assert jnp.allclose(values.times, jnp.asarray((0.0, 0.5, 2.0)))
    assert jnp.allclose(values.stage_times, jnp.asarray((0.25, 1.25)))
    assert jnp.allclose(values.state_rates[:, 0], jnp.asarray((1.0, 1.0)))
    assert jnp.allclose(values.decision.duration, 2.0)


def test_input_aware_dae_and_shared_parameters_compile_as_one_sparse_nlp():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, control, context: jnp.asarray(
            (
                state_rate[0] - context.parameters[0] * control[0],
                state[1] - state[0],
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="parameterized-direct-dae",
    )
    parameter_space = phx.linalg.ArraySpace((1,), dtype=jnp.float64)
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        case_shape=(2,),
        parameter_space=parameter_space,
        problem_id="shared-parameter-direct-dae",
    )
    plan = _plan(phx.solver.ThetaMethod(0.5, endpoint=False))
    states = jnp.zeros((2, 3, 2))
    controls = jnp.ones((2, 2, 1))
    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        states,
        controls,
        parameter_guess=jnp.asarray((2.0,)),
    )
    values = compilation.values(compilation.initial_coordinates)
    assert values.dynamics.shape == (2, 2, 2)
    assert jnp.allclose(values.dynamics[..., 0], -2.0)
    parameter_start, parameter_stop = compilation.decision_layout.parameter_slice
    assert parameter_stop - parameter_start == 1
    parameter_column = parameter_start
    rows = compilation.structured_program.jacobian_plan.pattern.rows
    columns = compilation.structured_program.jacobian_plan.pattern.cols
    affected_rows = rows[columns == parameter_column]
    assert affected_rows.size == 4


def test_sparse_jacobian_action_matches_direct_jvp():
    problem = _integrator_problem(
        trajectory_cost=lambda trajectory, args: 0.1 * jnp.sum(trajectory.states**2)
    )
    plan = _plan(
        phx.solver.ThetaMethod(0.5, endpoint=False),
        hessian="exact-sparse",
    )
    compilation = phx.control.compile_direct_collocation(
        problem,
        plan,
        jnp.asarray(((0.0,), (0.4,), (1.0,))),
        jnp.asarray(((0.5,), (0.8,))),
    )
    program = compilation.structured_program
    direction = jnp.linspace(0.1, 0.9, program.num_variables)
    operator_value = program.jacobian_plan.operator(
        compilation.initial_coordinates,
        problem.args,
    ).mv(direction)
    direct_value = jax.jvp(
        lambda coordinates: program.constraints(coordinates, problem.args),
        (compilation.initial_coordinates,),
        (direction,),
    )[1]
    assert jnp.allclose(operator_value, direct_value)
    assert bool(compilation.hessian_verification.passed)
