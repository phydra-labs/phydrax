"""Solve an opponent-dependent private nonlinear open-loop KKT system."""

import jax.numpy as jnp

import phydrax as phx


HORIZON = 2
partition = phx.control.games.PlayerControlPartition(("left", "right"), (1, 1))


def left_capacity(time, state, control, args):
    del time, state, args
    return control[0] + 0.1 * control[1] ** 2 - 1.0


capacity = phx.control.games.GameConstraintBlock(
    phx.control.BoundedPathConstraint(
        left_capacity,
        lower=-jnp.inf,
        upper=0.0,
        constraint_id="left-owned-opponent-dependent-capacity",
    ),
    scope=phx.control.games.GameConstraintScope.PLAYER_OWNED_COUPLED,
    participants=("left", "right"),
    owner="left",
    site=phx.control.games.GameConstraintSite.PATH,
    equality=False,
    residual_shape=(),
    time_dependent=False,
    state_dependent=False,
    control_dependencies=("left", "right"),
)
constraints = phx.control.games.OpenLoopGameConstraints(partition, (capacity,))


def transition(context, state, control, args):
    del context, args
    return jnp.asarray([state[0] + control[0] - control[1] + 0.02 * state[0] ** 2])


def left_cost(context, state, control, args):
    del context, state, args
    return 0.5 * (control[0] - 2.0) ** 2 + 0.01 * control[0] ** 4


def right_cost(context, state, control, args):
    del context, state, args
    return 0.5 * control[1] ** 2 + 0.01 * control[1] ** 4


def zero_terminal(time, state, args):
    del time, state, args
    return 0.0


system = phx.dynamics.DiscreteSystem(
    transition,
    state_layout=phx.dynamics.StateLayout((1,)),
    input_layout=phx.dynamics.InputLayout((2,), roles="control"),
    system_id="example-private-nonlinear-open-loop-kkt",
)
problem = phx.control.games.NonlinearOpenLoopGameProblem(
    phx.control.DiscreteControlDynamics(system),
    phx.dynamics.TimeGrid(
        jnp.arange(HORIZON + 1, dtype=float),
        time_id="example-private-nonlinear-open-loop-kkt",
    ),
    jnp.asarray([0.0]),
    partition,
    stage_costs=(left_cost, right_cost),
    terminal_costs=(zero_terminal, zero_terminal),
    constraints=constraints,
    problem_id="example-private-nonlinear-open-loop-kkt",
)
solution = phx.control.games.solve_open_loop_game_kkt(
    problem,
    jnp.zeros((HORIZON, 2)),
)
if not bool(solution.valid):
    raise RuntimeError(
        f"private nonlinear open-loop KKT solve failed: status={int(solution.status)}"
    )
if solution.feedback_claim or solution.global_equilibrium_claim:
    raise RuntimeError("a local open-loop KKT certificate must not make global claims")
if solution.constraint_scope != "opponent-dependent-private-player-feasible-sets":
    raise RuntimeError("the result did not retain the private coupled constraint scope")

print(
    {
        "certificate": solution.certificate_label,
        "claim": solution.certification_claim,
        "claim_boundary": "local nominal open-loop KKT only; not feedback or global GNE",
        "controls": solution.controls.tolist(),
        "left_private_multipliers": solution.private_multipliers[0].tolist(),
        "right_private_multipliers": solution.private_multipliers[1].tolist(),
        "original_stationarity_residual": float(solution.original_stationarity_residual),
        "original_primal_residual": float(solution.original_primal_residual),
        "original_dual_violation": float(solution.original_dual_violation),
        "original_ncp_residual": float(solution.original_ncp_residual),
        "original_complementarity_residual": float(
            solution.original_complementarity_residual
        ),
        "original_kkt_residual": float(solution.original_kkt_residual),
        "status": int(solution.status),
    }
)
