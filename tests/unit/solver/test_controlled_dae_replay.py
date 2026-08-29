import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _layout():
    return phx.dynamics.InputLayout(
        (1,),
        roles="control",
        layout_id="controlled-dae-input",
    )


def _system():
    return phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, control, args: jnp.asarray(
            (
                state_rate[0] - control[0],
                state[1] - state[0],
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        input_layout=_layout(),
        system_id="controlled-dae-replay",
    )


def test_held_input_policy_has_explicit_node_convention_and_derivatives():
    times = jnp.asarray((0.0, 1.0, 2.0))
    values = jnp.asarray(((3.0,), (5.0,)))
    left = phx.dynamics.HeldInputPolicy(
        times,
        values,
        input_layout=_layout(),
        node_side="left",
        policy_id="held-left",
    )
    right = phx.dynamics.HeldInputPolicy(
        times,
        values,
        input_layout=_layout(),
        node_side="right",
        policy_id="held-right",
    )
    assert jnp.allclose(left(1.0, jnp.zeros(2)), jnp.asarray((3.0,)))
    assert jnp.allclose(right(1.0, jnp.zeros(2)), jnp.asarray((5.0,)))
    assert jnp.allclose(left(2.0, jnp.zeros(2)), jnp.asarray((5.0,)))
    derivative = jax.jacrev(
        lambda coefficients: phx.dynamics.HeldInputPolicy(
            times,
            coefficients,
            input_layout=_layout(),
            policy_id="held-differentiation",
        )(0.5, jnp.zeros(2))
    )(values)
    assert jnp.allclose(derivative[0, :, 0], jnp.asarray((1.0, 0.0)))
    with pytest.raises(ValueError, match="outside its time grid"):
        left(2.1, jnp.zeros(2))


def test_dae_problem_requires_exact_input_policy_layout():
    system = _system()
    with pytest.raises(ValueError, match="requires input_policy"):
        phx.solver.DifferentialAlgebraicProblem(
            system,
            jnp.zeros(2),
            problem_id="missing-input-policy",
        )
    mismatch = phx.dynamics.HeldInputPolicy(
        jnp.asarray((0.0, 1.0)),
        jnp.ones((1, 2)),
        input_layout=phx.dynamics.InputLayout((2,), roles="control"),
        policy_id="mismatched-policy",
    )
    with pytest.raises(ValueError, match="exactly match"):
        phx.solver.DifferentialAlgebraicProblem(
            system,
            jnp.zeros(2),
            input_policy=mismatch,
            problem_id="mismatched-input-policy",
        )
    autonomous = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state_rate + state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="autonomous-policy-rejection",
    )
    policy = phx.dynamics.HeldInputPolicy(
        jnp.asarray((0.0, 1.0)),
        jnp.ones((1, 1)),
        input_layout=_layout(),
        policy_id="extra-policy",
    )
    with pytest.raises(ValueError, match="does not accept"):
        phx.solver.DifferentialAlgebraicProblem(
            autonomous,
            jnp.zeros(1),
            input_policy=policy,
            problem_id="extra-input-policy",
        )


def test_controlled_dae_initialization_and_stages_use_held_policy():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.5, 1.0)),
        time_id="controlled-dae-grid",
    )
    policy = phx.dynamics.HeldInputPolicy(
        grid.times,
        jnp.ones((2, 1)),
        input_layout=_layout(),
        policy_id="unit-held-control",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        _system(),
        jnp.zeros(2),
        initial_state_rate=jnp.asarray((1.0, 0.0)),
        input_policy=policy,
        problem_id="controlled-dae-problem",
    )
    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.ThetaMethod(1.0, endpoint=True)
        ),
    )
    assert bool(solution.successful)
    assert jnp.allclose(solution.states[:, 0], grid.times, atol=1e-7)
    assert jnp.allclose(solution.states[:, 1], grid.times, atol=1e-7)
    assert solution.input_policy_id == policy.policy_id
    assert solution.continuation.input_policy_id == policy.policy_id


def _direct_result():
    problem = phx.control.TrajectoryOptimizationProblem(
        _system(),
        initial_state=jnp.zeros(2),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        trajectory_constraints=(
            phx.control.BoundedTrajectoryConstraint(
                lambda trajectory, args: trajectory.final_state[0],
                lower=1.0,
                upper=1.0,
                constraint_id="replay-terminal",
            ),
        ),
        problem_id="direct-replay-problem",
    )
    mesh = phx.discretization.TemporalMesh.uniform(
        0.0,
        1.0,
        4,
        role="collocation",
        mesh_id="direct-replay-mesh",
    )
    plan = phx.control.DirectCollocationPlan(
        mesh,
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        derivatives=phx.control.DirectCollocationDerivativePolicy(verify=False),
        plan_id="direct-replay-plan",
    )
    return phx.control.solve_direct_collocation(
        problem,
        plan,
        jnp.stack((mesh.nodes, mesh.nodes), axis=-1),
        jnp.ones((mesh.num_steps, 1)),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="dense-filter", max_dense_dimension=128
        ),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-8,
            relative_optimality=0.0,
            maximum_steps=80,
        ),
    )


def test_direct_collocation_replay_is_independent_typed_evidence():
    result = _direct_result()
    evidence = phx.control.replay_direct_collocation(
        result,
        phx.control.DirectCollocationReplayPolicy(
            dae_policy=phx.solver.DAESolvePolicy(
                method=phx.solver.ThetaMethod(1.0, endpoint=True)
            ),
            node_state_tolerance=1.0e-6,
            terminal_state_tolerance=1.0e-6,
            algebraic_constraint_tolerance=1.0e-7,
        ),
    )
    assert bool(result.successful)
    assert bool(evidence.solution.successful)
    assert bool(evidence.passed)
    assert evidence.maximum_node_discrepancy <= 1e-6
    assert evidence.terminal_discrepancy <= 1e-6
    assert evidence.maximum_algebraic_residual <= 1e-7
    assert evidence.source_result_id == result.result_id
    assert evidence.input_policy.policy_id == evidence.solution.input_policy_id
