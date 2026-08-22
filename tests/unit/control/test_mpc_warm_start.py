#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _problem(initial=1.0, horizon=4):
    return phx.control.LinearQuadraticControlProblem(
        jnp.ones((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        jnp.asarray([initial]),
        jnp.ones((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        jnp.ones((1, 1)),
        control_lower_bounds=-2.0 * jnp.ones((horizon, 1)),
        control_upper_bounds=2.0 * jnp.ones((horizon, 1)),
        problem_id="prepared-control",
    )


def test_prepared_control_refresh_preserves_layout_and_increments_version():
    initial = phx.control.prepare_linear_quadratic_control(_problem())
    refreshed = phx.control.refresh_linear_quadratic_control(
        initial,
        _problem(initial=2.0),
    )
    result = phx.control.solve_prepared_linear_quadratic_control(refreshed)

    assert initial.prepared.numeric_version == 0
    assert refreshed.prepared.numeric_version == 1
    assert initial.prepared.plan.plan_id == refreshed.prepared.plan.plan_id
    np.testing.assert_allclose(result.states[0], [2.0], atol=1e-10)
    assert result.successful


def test_mpc_shifted_warm_starts_preserve_realized_solution():
    problem = _problem()
    cold = phx.control.solve_receding_horizon_mpc(
        problem,
        prediction_horizon=2,
        terminal_policy="none",
    )
    warm = phx.control.solve_receding_horizon_mpc(
        problem,
        prediction_horizon=2,
        terminal_policy="none",
        warm_start_policy=phx.control.MPCWarmStartPolicy(
            terminal_control="hold",
            interior_margin=1e-7,
        ),
    )

    np.testing.assert_allclose(warm.controls, cold.controls, atol=2e-5)
    np.testing.assert_allclose(warm.states, cold.states, atol=2e-5)
    np.testing.assert_allclose(warm.objective, cold.objective, atol=2e-5)
    assert warm.successful
    assert sum(int(result.iterations) for result in warm.qp_results[1:]) <= sum(
        int(result.iterations) for result in cold.qp_results[1:]
    )


def test_external_mpc_warm_start_requires_policy_and_matching_solution():
    problem = _problem()
    seed = phx.control.solve_linear_quadratic_control(problem)
    controller = phx.control.RecedingHorizonMPC(
        problem,
        prediction_horizon=2,
        terminal_policy="none",
    )
    with pytest.raises(ValueError, match="requires an explicit MPCWarmStartPolicy"):
        controller.solve(warm_start=seed)


def test_qpax_rejects_mpc_warm_policy_before_rollout():
    problem = _problem()
    policy = phx.optim.ConvexSolvePolicy(phx.optim.QPaxInteriorPoint())
    with pytest.raises(ValueError, match="does not support MPC warm starts"):
        phx.control.RecedingHorizonMPC(
            problem,
            prediction_horizon=2,
            terminal_policy="none",
            policy=policy,
            warm_start_policy=phx.control.MPCWarmStartPolicy(),
        )
