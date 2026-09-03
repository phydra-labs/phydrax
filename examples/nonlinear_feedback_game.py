"""Solve a nonlinear game and independently recompute its nominal residual."""

import jax.numpy as jnp

import phydrax as phx


HORIZON = 4
partition = phx.control.games.PlayerControlPartition(("left", "right"), (1, 1))
time_grid = phx.dynamics.TimeGrid(
    jnp.arange(HORIZON + 1, dtype=float),
    time_id="example-nonlinear-feedback-game",
)


def transition(context, state, control, args):
    del context, args
    return state + control


def stage_cost(player):
    def cost(context, state, control, args):
        del context
        error = state[player] - args["target"][player]
        effort = control[player]
        return 0.05 * error**2 + 0.5 * effort**2 + 0.01 * effort**4

    return cost


def terminal_cost(player):
    def cost(time, state, args):
        del time
        error = state[player] - args["target"][player]
        return 0.5 * error**2 + 0.01 * error**4

    return cost


system = phx.dynamics.DiscreteSystem(
    transition,
    state_layout=phx.dynamics.StateLayout((2,)),
    input_layout=phx.dynamics.InputLayout((2,), roles="control"),
    system_id="example-nonlinear-feedback-game",
)
problem = phx.control.games.DeterministicFeedbackGameProblem(
    phx.control.DiscreteControlDynamics(system),
    time_grid,
    jnp.asarray([1.2, -0.8]),
    partition,
    stage_costs=(stage_cost(0), stage_cost(1)),
    terminal_costs=(terminal_cost(0), terminal_cost(1)),
    args={"target": jnp.asarray([0.25, -0.15])},
    problem_id="example-nonlinear-feedback-game",
)
initial_policy = phx.control.AffineFeedbackPolicy(
    jnp.zeros((HORIZON, 2, 2)),
    jnp.zeros((HORIZON, 2)),
    time_grid=time_grid,
    state_size=2,
    policy_id="example-nonlinear-feedback-initial-policy",
)
scaling = phx.control.games.ILQGameScaling(
    jnp.ones(2),
    jnp.ones(2),
    jnp.ones(2),
    scaling_id="example-nonlinear-feedback-scaling",
)

solution = phx.control.games.solve_ilq_feedback_game(
    problem,
    scaling,
    initial_policy,
    maximum_iterations=12,
    maximum_line_search_steps=10,
    residual_tolerance=2.0e-5,
    step_tolerance=2.0e-5,
    dynamics_tolerance=2.0e-6,
)
if not bool(solution.valid):
    raise RuntimeError(f"nonlinear iLQ solve failed: status={int(solution.status)}")

# This recomputation does not reuse the residual stored on the solver result.
evaluation = phx.control.games.evaluate_game_policy(problem, solution.policy)
independent_residual = phx.control.games.nominal_nash_residual(
    problem, evaluation, scaling
)
if not bool(evaluation.successful) or not bool(independent_residual.successful):
    raise RuntimeError("independent policy evaluation or nominal residual failed")
if solution.diagnostics.feedback_nash_claimed:
    raise RuntimeError("the local nominal certificate must not claim feedback Nash")
if solution.diagnostics.global_convergence_claimed:
    raise RuntimeError("the local nominal certificate must not claim global convergence")

print(
    {
        "certificate": solution.certificate_label,
        "claim_boundary": "local nominal first-order stationarity only",
        "controls": solution.trajectory.controls.tolist(),
        "player_costs": solution.player_costs.tolist(),
        "accepted_iterations": int(solution.diagnostics.accepted_iterations),
        "independent_stationarity_rms": float(independent_residual.stationarity_rms_norm),
        "independent_stationarity_infinity": float(
            independent_residual.stationarity_infinity_norm
        ),
        "independent_dynamics_infinity": float(
            independent_residual.dynamics_defect_infinity_norm
        ),
        "status": int(solution.status),
    }
)
