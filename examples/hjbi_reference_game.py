"""Compute lower and upper bounded-grid HJBI references with Isaacs evidence."""

import jax.numpy as jnp

import phydrax as phx


def zero_coefficient(time, state, minimizer, maximizer, args):
    del time, state, minimizer, maximizer, args
    return 0.0


def zero_sum_running_cost(time, state, minimizer, maximizer, args):
    del time, state, args
    return minimizer**2 - maximizer**2


spatial_grid = phx.control.stochastic.BoundedUniformGrid1D(-1.0, 1.0, 9)
time_grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 0.2, 9),
    time_id="example-hjbi-reference",
)
actions = jnp.asarray([-1.0, 0.0, 1.0])
problem = phx.control.games.DiscreteZeroSumHJBIProblem(
    spatial_grid,
    time_grid,
    actions,
    actions,
    jnp.zeros((spatial_grid.num_points,)),
    jnp.zeros((time_grid.num_times, 2)),
    zero_coefficient,
    zero_coefficient,
    zero_sum_running_cost,
    lower_order="max_min",
    upper_order="min_max",
    problem_id="example-hjbi-reference",
)
solution = phx.control.games.solve_discrete_hjbi_reference(
    problem,
    refinement_absolute_tolerance=0.0,
    refinement_relative_tolerance=0.0,
    isaacs_absolute_tolerance=0.0,
    isaacs_relative_tolerance=0.0,
)
if not bool(solution.evidence.finite):
    raise RuntimeError("HJBI reference produced non-finite evidence")
if not bool(solution.saddle):
    raise RuntimeError(
        "the declared discrete saddle gates failed: "
        f"status={solution.status_label}, "
        f"Isaacs gap={float(solution.evidence.maximum_isaacs_gap)}"
    )

print(
    {
        "status": solution.status_label,
        "lower_order": solution.lower_order,
        "upper_order": solution.upper_order,
        "lower_initial_values": solution.lower_values[0].tolist(),
        "upper_initial_values": solution.upper_values[0].tolist(),
        "maximum_isaacs_gap": float(solution.evidence.maximum_isaacs_gap),
        "isaacs_threshold": float(solution.evidence.isaacs_threshold),
        "refinement_difference": float(
            jnp.maximum(
                solution.evidence.maximum_lower_refinement_difference,
                solution.evidence.maximum_upper_refinement_difference,
            )
        ),
        "all_discrete_gates_passed": {
            "boundary": bool(solution.evidence.boundary_passed),
            "terminal": bool(solution.evidence.terminal_passed),
            "operator": bool(solution.evidence.operator_passed),
            "action_orders": bool(solution.evidence.action_orders_passed),
            "refinement": bool(solution.evidence.refinement_passed),
            "isaacs_gap": bool(solution.evidence.isaacs_gap_passed),
        },
        "claim_scope": solution.evidence.scope,
    }
)
