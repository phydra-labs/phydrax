#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._stochastic_saa import (
    plan_stochastic_policy_game,
    prepare_stochastic_policy_game,
    PreparedStochasticPolicyGame,
    refresh_stochastic_policy_game,
    solve_prepared_stochastic_policy_game,
    solve_stochastic_policy_game,
    StochasticPolicyGameProblem,
    StochasticPolicyGameStatus,
)
from phydrax.control.stochastic import PreparedControlledNoise
from phydrax.nonlinear import NonlinearStatus, NonlinearTermination


def _noise(values, prefix, *, labels=None, coupling_id=None, valid=None):
    values = jnp.asarray(values, dtype=float)
    count = int(values.shape[0])
    if labels is None:
        labels = jnp.arange(count, dtype=jnp.int32)
    if valid is None:
        valid = jnp.ones((count,), dtype=bool)
    return PreparedControlledNoise(
        values[:, None, None],
        valid=valid,
        realization_ids=tuple(f"{prefix}-{index}" for index in range(count)),
        coupling_id=prefix if coupling_id is None else coupling_id,
        independence_labels=labels,
        noise_shape=(1,),
    )


def _quadratic_costs(parameters, noise, args):
    signal = noise.increments[:, 0, 0]
    first_target = signal + args["first_shift"]
    second_target = args["second_scale"] * signal + args["second_shift"]
    first = parameters[..., 0, None]
    second = parameters[..., 1, None]
    # The large cross-player terms make an incorrect summed-objective gradient
    # conspicuously different while leaving the owned pseudo-gradient unchanged.
    player_zero = 0.5 * (first - first_target) ** 2 + 17.0 * second
    player_one = 0.5 * (second - second_target) ** 2 - 23.0 * first
    return jnp.stack((player_zero, player_one), axis=-1)


def _quadratic_problem(*, case_shape=(), args=None, suffix="base"):
    if args is None:
        args = {
            "first_shift": 1.0,
            "second_scale": 2.0,
            "second_shift": -1.0,
        }
    return StochasticPolicyGameProblem(
        _quadratic_costs,
        PlayerControlPartition(("first", "second"), (1, 1)),
        case_shape=case_shape,
        args=args,
        callback_id=f"quadratic-path-costs-{suffix}",
        feasible_set_id="all-finite-policy-parameters",
        problem_id=f"stochastic-quadratic-game-{suffix}",
    )


def test_weighted_stochastic_quadratic_game_solves_owned_saa_root_and_holds_out():
    problem = _quadratic_problem()
    training = _noise([-2.0, 0.5, 3.0], "train")
    holdout = _noise(
        [20.0, 30.0, 40.0, 50.0],
        "holdout",
        labels=jnp.asarray([4, 4, 9, 12]),
    )
    weights = jnp.asarray([0.2, 0.3, 0.5])
    result = solve_stochastic_policy_game(
        problem,
        jnp.asarray([8.0, -7.0]),
        training,
        holdout,
        training_weights=weights,
        holdout_weights=jnp.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    expected_signal = weights @ jnp.asarray([-2.0, 0.5, 3.0])
    expected = jnp.asarray([expected_signal + 1.0, 2.0 * expected_signal - 1.0])

    assert bool(result.successful)
    assert int(result.root_status) == int(NonlinearStatus.SUCCESS)
    np.testing.assert_allclose(result.parameters, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result.original_residual, 0.0, atol=3e-6)
    assert result.training_path_costs.shape == (3, 2)
    assert result.holdout_path_costs.shape == (4, 2)
    assert result.training_complete_path_gradients.shape == (3, 2, 2)
    assert result.training_owned_path_gradients.shape == (3, 2)
    np.testing.assert_array_equal(result.holdout_cluster_counts, [2, 1, 1])
    np.testing.assert_allclose(result.holdout_cluster_weights, [0.3, 0.3, 0.4])
    assert bool(jnp.all(result.holdout_cluster_valid))
    assert result.training_realization_ids == training.realization_ids
    assert result.holdout_realization_ids == holdout.realization_ids
    assert result.training_bundle_id != result.holdout_bundle_id


def test_complete_player_gradients_are_formed_before_owned_rows_are_selected():
    problem = _quadratic_problem()
    training = _noise([-1.0, 2.0], "ownership-train")
    holdout = _noise([4.0, 6.0], "ownership-holdout")
    plan = plan_stochastic_policy_game(problem)
    prepared = prepare_stochastic_policy_game(
        plan,
        problem,
        jnp.asarray([0.25, -0.75]),
        training,
        holdout,
    )
    result = solve_prepared_stochastic_policy_game(prepared)
    complete = result.training_complete_path_gradients

    # Player zero's derivative in the other player's parameter is +17, and
    # player one's derivative in player zero's parameter is -23. Neither may be
    # introduced by summing player objectives before ownership selection.
    np.testing.assert_allclose(complete[:, 0, 1], 17.0)
    np.testing.assert_allclose(complete[:, 1, 0], -23.0)
    np.testing.assert_allclose(
        result.training_owned_path_gradients[:, 0], complete[:, 0, 0]
    )
    np.testing.assert_allclose(
        result.training_owned_path_gradients[:, 1], complete[:, 1, 1]
    )


def test_pathwise_gradient_is_not_the_gradient_of_a_mutated_mean_trajectory():
    def quartic_path_costs(parameters, noise, args):
        del args
        path_value = noise.increments[:, 0, 0]
        difference = parameters[..., 0, None] - path_value
        return (0.25 * difference**4)[..., None]

    problem = StochasticPolicyGameProblem(
        quartic_path_costs,
        PlayerControlPartition(("player",), (1,)),
        callback_id="quartic-complete-path-cost",
        feasible_set_id="real-line",
        problem_id="pathwise-not-mean-trajectory",
    )
    training = _noise([0.0, 0.0, 3.0], "pathwise-train")
    holdout = _noise([5.0, 7.0, 9.0], "pathwise-holdout")
    prepared = prepare_stochastic_policy_game(
        plan_stochastic_policy_game(problem),
        problem,
        jnp.asarray([1.0]),
        training,
        holdout,
    )
    residual_at_path_mean = prepared.root_problem.residual(jnp.asarray([1.0]))

    # E[(theta-Z)^3] = -2 here, whereas mutating the bundle to E[Z] before
    # differentiation would incorrectly report zero at theta = E[Z] = 1.
    np.testing.assert_allclose(residual_at_path_mean, [-2.0], atol=1e-6)


def test_prepared_training_bundle_is_frozen_common_randomness_across_solves():
    problem = _quadratic_problem()
    training = _noise([-3.0, 0.0, 4.0], "frozen-train")
    holdout = _noise([6.0, 8.0, 10.0], "frozen-holdout")
    prepared = prepare_stochastic_policy_game(
        plan_stochastic_policy_game(problem),
        problem,
        jnp.asarray([3.0, -5.0]),
        training,
        holdout,
    )

    first = solve_prepared_stochastic_policy_game(prepared)
    second = solve_prepared_stochastic_policy_game(prepared)

    np.testing.assert_array_equal(first.parameters, second.parameters)
    np.testing.assert_array_equal(first.original_residual, second.original_residual)
    np.testing.assert_array_equal(first.training_path_costs, second.training_path_costs)
    np.testing.assert_array_equal(first.root_status, second.root_status)
    np.testing.assert_array_equal(
        first.root_diagnostics.residual_evaluations,
        second.root_diagnostics.residual_evaluations,
    )
    assert first.training_realization_ids == second.training_realization_ids


def test_training_and_holdout_realization_identity_must_be_disjoint():
    problem = _quadratic_problem()
    training = _noise([0.0, 1.0], "shared")
    holdout = PreparedControlledNoise(
        jnp.asarray([[[2.0]], [[3.0]]]),
        valid=jnp.ones((2,), dtype=bool),
        realization_ids=(training.realization_ids[1], "other"),
        coupling_id="different-coupling",
        independence_labels=jnp.asarray([0, 1]),
        noise_shape=(1,),
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        prepare_stochastic_policy_game(
            plan_stochastic_policy_game(problem),
            problem,
            jnp.zeros((2,)),
            training,
            holdout,
        )


def test_training_and_holdout_coupling_id_must_identify_independence():
    problem = _quadratic_problem()
    training = _noise([0.0, 1.0], "coupled-train", coupling_id="same-coupling")
    holdout = _noise([2.0, 3.0], "coupled-holdout", coupling_id="same-coupling")

    with pytest.raises(ValueError, match="distinct coupling_id"):
        prepare_stochastic_policy_game(
            plan_stochastic_policy_game(problem),
            problem,
            jnp.zeros((2,)),
            training,
            holdout,
        )


def test_refresh_requires_same_topology_and_entirely_new_realization_ids():
    problem = _quadratic_problem()
    initial_training = _noise([0.0, 1.0], "refresh-old-train")
    initial_holdout = _noise([2.0, 3.0], "refresh-old-holdout")
    prepared = prepare_stochastic_policy_game(
        plan_stochastic_policy_game(problem),
        problem,
        jnp.zeros((2,)),
        initial_training,
        initial_holdout,
    )
    new_training = _noise([4.0, 5.0], "refresh-new-train")
    new_holdout = _noise([6.0, 7.0], "refresh-new-holdout")
    refreshed = refresh_stochastic_policy_game(
        prepared,
        problem,
        jnp.ones((2,)),
        training_noise=new_training,
        holdout_noise=new_holdout,
    )

    assert isinstance(refreshed, PreparedStochasticPolicyGame)
    assert int(refreshed.numeric_version) == 1
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.training_realization_ids == new_training.realization_ids
    assert refreshed.holdout_realization_ids == new_holdout.realization_ids
    with pytest.raises(ValueError, match="new realization_ids"):
        refresh_stochastic_policy_game(
            prepared,
            problem,
            training_noise=initial_training,
        )
    with pytest.raises(ValueError, match="topology"):
        refresh_stochastic_policy_game(
            prepared,
            problem,
            training_noise=_noise([1.0, 2.0, 3.0], "wrong-count"),
        )


def test_case_axes_are_separate_from_path_player_and_parameter_axes():
    training = _noise([-1.0, 0.5, 2.0], "cases-train")
    holdout = _noise([3.0, 4.0, 5.0], "cases-holdout")
    targets = jnp.asarray([[-2.0, 1.0, 4.0], [5.0, 0.0, -1.0]])

    def case_costs(parameters, noise, args):
        del noise
        difference = parameters[..., 0, None] - args["targets"]
        return (0.5 * difference**2)[..., None]

    problem = StochasticPolicyGameProblem(
        case_costs,
        PlayerControlPartition(("only",), (1,)),
        case_shape=(2,),
        args={"targets": targets},
        callback_id="case-batched-quadratic-cost",
        feasible_set_id="two-real-lines",
        problem_id="case-batched-stochastic-game",
    )
    result = solve_stochastic_policy_game(
        problem,
        jnp.asarray([[10.0], [-10.0]]),
        training,
        holdout,
    )

    np.testing.assert_allclose(result.parameters[:, 0], jnp.mean(targets, axis=1))
    assert result.parameters.shape == (2, 1)
    assert result.original_residual.shape == (2, 1)
    assert result.training_path_costs.shape == (2, 3, 1)
    assert result.training_complete_path_gradients.shape == (2, 3, 1, 1)
    assert result.holdout_path_costs.shape == (2, 3, 1)
    assert result.status.shape == (2,)


def test_root_failure_and_nonfinite_training_paths_fail_closed():
    training = _noise([0.0, 0.0], "root-failure-train")
    holdout = _noise([1.0, 2.0], "root-failure-holdout")

    def quartic(parameters, noise, args):
        del args
        difference = parameters[..., 0, None] - noise.increments[:, 0, 0]
        return (0.25 * difference**4)[..., None]

    problem = StochasticPolicyGameProblem(
        quartic,
        PlayerControlPartition(("only",), (1,)),
        callback_id="root-failure-quartic",
        feasible_set_id="real-line",
        problem_id="root-failure-game",
    )
    termination = NonlinearTermination(
        absolute_residual=0.0,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=1,
    )
    failed = solve_stochastic_policy_game(
        problem,
        jnp.asarray([2.0]),
        training,
        holdout,
        termination=termination,
    )
    assert int(failed.status) == int(StochasticPolicyGameStatus.ROOT_FAILED)
    assert not bool(failed.successful)

    bad_training = _noise([0.0, jnp.nan], "nonfinite-train")
    nonfinite = solve_stochastic_policy_game(
        problem,
        jnp.asarray([1.0]),
        bad_training,
        holdout,
    )
    assert int(nonfinite.status) == int(
        StochasticPolicyGameStatus.INVALID_TRAINING_BUNDLE
    )
    assert not bool(nonfinite.stationarity_certified)


def test_nonfinite_callback_cost_and_holdout_cost_have_stable_statuses():
    def costs(parameters, noise, args):
        values = 0.5 * (parameters[..., 0, None] - noise.increments[:, 0, 0]) ** 2
        if noise.coupling_id == args["bad_coupling"]:
            values = values.at[..., 0].set(jnp.nan)
        return values[..., None]

    problem = StochasticPolicyGameProblem(
        costs,
        PlayerControlPartition(("only",), (1,)),
        args={"bad_coupling": "bad-training"},
        callback_id="selectively-nonfinite-costs",
        feasible_set_id="real-line",
        problem_id="nonfinite-path-cost-game",
    )
    bad_training = _noise([0.0, 1.0], "train-ids", coupling_id="bad-training")
    good_holdout = _noise([2.0, 3.0], "good-holdout")
    training_result = solve_stochastic_policy_game(
        problem,
        jnp.asarray([0.0]),
        bad_training,
        good_holdout,
    )
    assert int(training_result.status) == int(
        StochasticPolicyGameStatus.NONFINITE_TRAINING_PATH_COSTS
    )

    holdout_problem = StochasticPolicyGameProblem(
        costs,
        problem.partition,
        args={"bad_coupling": "bad-holdout"},
        callback_id="selectively-nonfinite-holdout-costs",
        feasible_set_id="real-line",
        problem_id="nonfinite-holdout-cost-game",
    )
    good_training = _noise([0.0, 1.0], "good-train")
    bad_holdout = _noise([2.0, 3.0], "holdout-ids", coupling_id="bad-holdout")
    holdout_result = solve_stochastic_policy_game(
        holdout_problem,
        jnp.asarray([0.0]),
        good_training,
        bad_holdout,
    )
    assert int(holdout_result.status) == int(
        StochasticPolicyGameStatus.NONFINITE_HOLDOUT_PATH_COSTS
    )
    assert bool(jnp.any(~holdout_result.holdout_cluster_valid))


def test_filtered_jit_preserves_frozen_bundle_solution_and_evidence():
    problem = _quadratic_problem(suffix="jit")
    prepared = prepare_stochastic_policy_game(
        plan_stochastic_policy_game(problem),
        problem,
        jnp.asarray([4.0, -3.0]),
        _noise([-1.0, 1.0, 2.0], "jit-train"),
        _noise([3.0, 5.0, 7.0], "jit-holdout"),
    )
    eager = solve_prepared_stochastic_policy_game(prepared)
    compiled = eqx.filter_jit(solve_prepared_stochastic_policy_game)(prepared)

    np.testing.assert_allclose(compiled.parameters, eager.parameters)
    np.testing.assert_allclose(compiled.original_residual, eager.original_residual)
    np.testing.assert_allclose(compiled.holdout_path_costs, eager.holdout_path_costs)
    np.testing.assert_array_equal(compiled.status, eager.status)


def test_result_never_claims_population_or_feedback_nash():
    problem = _quadratic_problem(suffix="claim")
    result = solve_stochastic_policy_game(
        problem,
        jnp.asarray([0.0, 0.0]),
        _noise([-1.0, 1.0], "claim-train"),
        _noise([2.0, 4.0], "claim-holdout"),
    )

    assert result.certification_claim == "LOCAL_SAA_POLICY_STATIONARITY"
    lowered = result.certification_claim.lower()
    assert "population" not in lowered
    assert "feedback" not in lowered
    assert result.callback_id == problem.callback_id
    assert result.feasible_set_id == problem.feasible_set_id
