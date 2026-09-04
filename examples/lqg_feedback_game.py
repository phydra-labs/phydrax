"""Solve an exact additive-noise two-player LQG feedback Nash game."""

import jax.numpy as jnp

import phydrax as phx


partition = phx.control.games.PlayerControlPartition(
    ("left", "right"),
    (1, 1),
)
time_grid = phx.dynamics.TimeGrid(
    jnp.asarray([0.0, 1.0]),
    time_id="example-lqg-feedback-nash",
)
dynamics_matrices = jnp.asarray([[[1.0]]])
control_matrices = jnp.asarray([[[1.0, 1.0]]])
state_costs = jnp.zeros((2, 1, 1, 1))
control_costs = jnp.stack((jnp.eye(2), jnp.eye(2)))[:, None, :, :]
terminal_state_costs = jnp.asarray([[[2.0]], [[4.0]]])
process_noise_factors = jnp.asarray([[[2.0]]])
process_noise_covariances = jnp.asarray([[[0.25]]])

solution = phx.control.games.finite_horizon_lqg_feedback_nash(
    dynamics_matrices,
    control_matrices,
    state_costs,
    control_costs,
    terminal_state_costs,
    partition,
    process_noise_factors=process_noise_factors,
    process_noise_covariances=process_noise_covariances,
    initial_mean=jnp.asarray([0.5]),
    initial_covariance=jnp.asarray([[0.5]]),
    time_grid=time_grid,
    policy_id="example-lqg-feedback-nash",
)
if not bool(solution.valid):
    raise RuntimeError(
        f"additive LQG feedback-Nash solve failed: status={int(solution.status)}"
    )

# G Omega G^T = 1.  At the only transition, each player's exact
# correction is 0.5 * trace(P[1] G Omega G^T), hence [1, 2].
expected_process_covariance = jnp.asarray([[[1.0]]])
expected_trace_increments = jnp.asarray([[1.0], [2.0]])
expected_constant_corrections = jnp.asarray([[1.0, 0.0], [2.0, 0.0]])
if not bool(jnp.allclose(solution.process_covariances, expected_process_covariance)):
    raise RuntimeError("the implied additive process covariance is incorrect")
if not bool(jnp.allclose(solution.trace_increments, expected_trace_increments)):
    raise RuntimeError("the player-wise trace identity was not reproduced")
if not bool(
    jnp.allclose(
        solution.value_constant_corrections,
        expected_constant_corrections,
    )
):
    raise RuntimeError("the reverse trace accumulation was not reproduced")
for player, value in enumerate(solution.values):
    expected_constants = (
        solution.deterministic_result.values[player].constants
        + expected_constant_corrections[player]
    )
    if not bool(jnp.allclose(value.constants, expected_constants)):
        raise RuntimeError(f"player {player} has an incorrect corrected value")

print(
    {
        "method": solution.method,
        "feedback_gain": solution.feedback_gain.tolist(),
        "process_covariances": solution.process_covariances.tolist(),
        "player_trace_increments": solution.trace_increments.tolist(),
        "player_value_constant_corrections": (
            solution.value_constant_corrections.tolist()
        ),
        "initial_expected_costs": solution.initial_expected_cost.tolist(),
        "covariance_minimum_eigenvalue": float(
            jnp.min(solution.covariance_minimum_eigenvalues)
        ),
        "claim_scope": (
            "exact finite-horizon full-state feedback Nash and value constants "
            "for the declared zero-mean additive-noise LQG game only"
        ),
    }
)
