"""Solve and replay a two-player affine linear-quadratic feedback Nash game."""

import jax.numpy as jnp

import phydrax as phx


partition = phx.control.games.PlayerControlPartition(
    ("player-1", "player-2"),
    (1, 1),
)
time_grid = phx.dynamics.TimeGrid(
    jnp.asarray([0.0, 1.0]),
    time_id="example-lq-feedback-nash",
)
dynamics_matrices = jnp.asarray([[[1.0]]])
control_matrices = jnp.asarray([[[1.0, 1.0]]])
dynamics_bias = jnp.asarray([[1.0]])
state_costs = jnp.asarray([[[[2.0]]], [[[-1.0]]]])
control_costs = jnp.asarray(
    [
        [[[1.0, -0.5], [-0.5, 2.0]]],
        [[[2.0, -3.0], [-3.0, 1.0]]],
    ]
)
state_control_cross = jnp.asarray(
    [
        [[[1.0, -1.0]]],
        [[[0.5, -0.5]]],
    ]
)
state_linear = jnp.asarray([[[1.0]], [[-2.0]]])
control_linear = jnp.asarray([[[0.5, -1.0]], [[0.25, 1.0]]])
stage_constants = jnp.asarray([[3.0], [-1.0]])
terminal_state_costs = jnp.asarray([[[1.0]], [[2.0]]])
terminal_linear = jnp.asarray([[0.5], [-1.0]])
terminal_constants = jnp.asarray([0.25, 2.0])

solution = phx.control.games.finite_horizon_lq_feedback_nash(
    dynamics_matrices,
    control_matrices,
    state_costs,
    control_costs,
    terminal_state_costs,
    partition,
    dynamics_bias=dynamics_bias,
    state_control_cross=state_control_cross,
    state_linear=state_linear,
    control_linear=control_linear,
    stage_constants=stage_constants,
    terminal_linear=terminal_linear,
    terminal_constants=terminal_constants,
    time_grid=time_grid,
    policy_id="example-lq-feedback-nash",
)
if not bool(solution.valid):
    raise RuntimeError(
        "feedback Nash solve failed: "
        f"status={int(solution.status)}, "
        f"stage={int(solution.diagnostics.first_failed_stage)}"
    )


def transition(context, state, control, args):
    step = context.step_index
    return args["A"][step] @ state + args["B"][step] @ control + args["c"][step]


dynamics = phx.control.DiscreteControlDynamics(
    phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((2,), roles="control"),
        system_id="example-lq-feedback-nash",
    )
)
initial_state = jnp.asarray([0.3])
problem = phx.control.ControlProblem(
    dynamics,
    time_grid,
    initial_state,
    args={
        "A": dynamics_matrices,
        "B": control_matrices,
        "c": dynamics_bias,
    },
    problem_id="example-lq-feedback-nash",
)
evaluation = problem.evaluate(solution.policy, jnp.zeros(()))
if not bool(evaluation.successful):
    raise RuntimeError(f"feedback policy rollout failed: status={int(evaluation.status)}")

trajectory = evaluation.trajectory
player_controls = partition.split_controls(trajectory.controls)
player_costs = []
for player in range(partition.num_players):
    state = trajectory.states[0]
    control = trajectory.controls[0]
    terminal_state = trajectory.states[1]
    stage_cost = (
        0.5 * state @ state_costs[player, 0] @ state
        + state @ state_control_cross[player, 0] @ control
        + 0.5 * control @ control_costs[player, 0] @ control
        + state_linear[player, 0] @ state
        + control_linear[player, 0] @ control
        + stage_constants[player, 0]
    )
    terminal_cost = (
        0.5 * terminal_state @ terminal_state_costs[player] @ terminal_state
        + terminal_linear[player] @ terminal_state
        + terminal_constants[player]
    )
    player_costs.append(stage_cost + terminal_cost)

print(
    {
        "feedback_gain": solution.feedback_gain.tolist(),
        "feedforward": solution.feedforward.tolist(),
        "player_controls": [control.tolist() for control in player_controls],
        "player_costs": [float(cost) for cost in player_costs],
        "initial_values": [
            float(value(time_grid.times[0], initial_state)) for value in solution.values
        ],
        "maximum_stationarity_residual": float(
            solution.diagnostics.maximum_stationarity_residual
        ),
        "maximum_bellman_residual": float(solution.diagnostics.maximum_bellman_residual),
        "maximum_condition_number": float(
            solution.diagnostics.maximum_coupled_condition_number
        ),
        "status": int(solution.status),
    }
)
