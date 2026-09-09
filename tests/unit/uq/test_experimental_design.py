#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _binary_channel(error_probability: float) -> jnp.ndarray:
    error = jnp.asarray(error_probability)
    return jnp.asarray([[[1.0 - error, error], [error, 1.0 - error]]])


def test_exact_finite_information_distinguishes_target_semantics_and_error():
    error = 0.1
    conditional = _binary_channel(error)
    prior = jnp.asarray([0.5, 0.5])
    candidate = phx.uq.ExperimentalDesignCandidate(
        "candidate-a",
        "condition-a",
        1.0,
        "assay",
        "prediction-model-a",
    )
    parameter = phx.uq.exact_finite_expected_utility(
        conditional,
        prior,
        candidates=(candidate,),
        model_ids=("model-a",),
        utility_target="parameter",
    )
    predictive = phx.uq.exact_finite_expected_utility(
        conditional,
        prior,
        candidates=(candidate,),
        model_ids=("model-a",),
        utility_target="predictive",
    )
    expected = (
        jnp.log(2.0) + error * jnp.log(error) + (1.0 - error) * jnp.log(1.0 - error)
    )

    assert bool(parameter.valid[0])
    assert parameter.utility_target == "parameter"
    assert predictive.utility_target == "predictive"
    assert jnp.allclose(parameter.expected_utility[0], expected)
    assert jnp.array_equal(parameter.estimator_standard_error, jnp.zeros((1,)))
    assert jnp.array_equal(parameter.estimator_bias_bound, jnp.zeros((1,)))
    assert parameter.approximation == "exact_finite_enumeration"


def test_nested_monte_carlo_reports_outer_error_and_unknown_finite_inner_bias():
    outer_count = 1024
    inner_count = 256
    theta_key, noise_key, inner_key = jr.split(jr.key(14), 3)
    theta = jr.normal(theta_key, (outer_count,))
    observation = theta + jr.normal(noise_key, (outer_count,))
    inner_theta = jr.normal(inner_key, (outer_count, inner_count))
    log_normalizer = 0.5 * jnp.log(2.0 * jnp.pi)
    conditioned = (-0.5 * (observation - theta) ** 2 - log_normalizer)[None, :, None]
    marginal_samples = (
        -0.5 * (observation[:, None] - inner_theta) ** 2 - log_normalizer
    )[None, :, :]

    candidate = phx.uq.ExperimentalDesignCandidate(
        "normal-observation",
        "standard",
        1.0,
        "assay",
        "normal-model",
    )
    result = phx.uq.nested_monte_carlo_expected_utility(
        conditioned,
        marginal_samples,
        candidates=(candidate,),
        model_ids=("normal-model",),
        utility_target="parameter",
    )

    assert bool(result.valid[0])
    assert result.expected_utility[0] == pytest.approx(0.5 * jnp.log(2.0), abs=0.08)
    assert 0.0 < float(result.estimator_standard_error[0]) < 0.08
    assert bool(jnp.isnan(result.estimator_bias_bound[0]))
    assert result.outer_sample_count == outer_count
    assert result.inner_sample_count == inner_count
    assert "finite-inner" in result.error_basis


def test_parameter_estimator_samples_observations_through_posterior_problem():
    parameter_space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        log_prior=lambda value: -0.5 * value * value,
    )
    problem = phx.uq.PosteriorProblem(
        parameter_space,
        lambda value: jnp.asarray(0.0),
        sample_observation=lambda key, value, candidate: (
            value
            + jr.normal(key) * (1.0 if candidate.condition_id == "standard" else 2.0)
        ),
    )
    candidate = phx.uq.ExperimentalDesignCandidate(
        "observe-standard",
        "standard",
        1.0,
        "assay",
        "calibrated-normal-law",
    )

    result = phx.uq.posterior_parameter_expected_utility(
        problem,
        jr.key(3),
        jr.normal(jr.key(4), (64,)),
        (candidate,),
        lambda observation, position, selected: (
            -0.5
            * (
                (observation - position)
                / (1.0 if selected.condition_id == "standard" else 2.0)
            )
            ** 2
            - jnp.log(1.0 if selected.condition_id == "standard" else 2.0)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        ),
        num_outer_samples=8,
        model_ids=("normal-model",),
        num_inner_samples=16,
    )

    assert bool(result.valid[0])
    assert result.utility_target == "parameter"
    assert result.method_id == "posterior_problem_nested_parameter_information"
    assert result.estimator_standard_error.shape == (1,)


def _constraint_candidates():
    return (
        phx.uq.ExperimentalDesignCandidate(
            "control",
            "buffer-only",
            1.0,
            "plate",
            "prediction-v1",
            setup_id="plate-setup",
            setup_cost=2.0,
            diversity_group="control",
            mandatory_control=True,
        ),
        phx.uq.ExperimentalDesignCandidate(
            "perturb-a",
            "ligand-a",
            2.0,
            "plate",
            "prediction-v1",
            setup_id="plate-setup",
            setup_cost=2.0,
            diversity_group="mechanism-a",
        ),
        phx.uq.ExperimentalDesignCandidate(
            "perturb-b",
            "ligand-b",
            2.0,
            "plate",
            "prediction-v1",
            setup_id="plate-setup",
            setup_cost=2.0,
            diversity_group="mechanism-a",
        ),
        phx.uq.ExperimentalDesignCandidate(
            "orthogonal-c",
            "temperature-c",
            3.0,
            "orthogonal",
            "prediction-v1",
            diversity_group="mechanism-b",
        ),
    )


def _constraint_utility(candidates):
    return phx.uq.ExpectedUtilityResult(
        expected_utility=jnp.asarray([0.0, 3.0, 2.0, 2.0]),
        estimator_standard_error=jnp.asarray([0.01, 0.01, 0.01, 0.01]),
        estimator_bias_bound=jnp.full((4,), jnp.nan),
        valid=jnp.ones((4,), dtype=bool),
        candidates=candidates,
        model_ids=("additive", "interaction"),
        utility_target="model_discrimination",
        method_id="fixture-utility",
        approximation="fixture-finite-inner",
        error_basis="fixture sampling standard error; bias unknown",
        outer_sample_count=100,
        inner_sample_count=100,
    )


def test_batch_selection_is_deterministic_and_respects_all_constraints():
    candidates = _constraint_candidates()
    utility = _constraint_utility(candidates)
    constraints = phx.uq.ExperimentalBatchConstraints(
        8.0,
        3,
        required_candidate_ids=("control",),
        mutually_exclusive_candidate_groups=(("perturb-a", "perturb-b"),),
        minimum_diversity_groups=3,
        maximum_per_diversity_group=1,
    )

    first = phx.uq.select_experimental_batch(
        candidates,
        utility,
        constraints,
        objective_id="model-information-v1",
        model_ids=("additive", "interaction"),
        analysis_id="analysis-source-tree-123",
    )
    second = phx.uq.select_experimental_batch(
        tuple(reversed(candidates)),
        utility,
        constraints,
        objective_id="model-information-v1",
        model_ids=("interaction", "additive"),
        analysis_id="analysis-source-tree-123",
    )

    assert first.selected_candidate_ids == (
        "control",
        "orthogonal-c",
        "perturb-a",
    )
    assert first.planned_total_cost == 8.0
    assert first.plan_id == second.plan_id
    restored = phx.uq.ExperimentalBatchPlan.from_record(first.to_record())
    assert restored.plan_id == first.plan_id


def test_mandatory_control_is_costed_with_explicit_zero_information_utility():
    candidates = _constraint_candidates()
    utility = phx.uq.ExpectedUtilityResult(
        expected_utility=jnp.asarray([jnp.nan, 3.0, 2.0, 2.0]),
        estimator_standard_error=jnp.asarray([jnp.nan, 0.01, 0.01, 0.01]),
        estimator_bias_bound=jnp.full((4,), jnp.nan),
        valid=jnp.asarray([False, True, True, True]),
        candidates=candidates,
        model_ids=("additive", "interaction"),
        utility_target="model_discrimination",
        method_id="control-has-no-information-estimator",
        approximation="fixture-finite-inner",
        error_basis="mandatory control utility is not estimable",
    )
    constraints = phx.uq.ExperimentalBatchConstraints(
        8.0,
        3,
        minimum_diversity_groups=3,
        maximum_per_diversity_group=1,
    )

    plan = phx.uq.select_experimental_batch(
        candidates,
        utility,
        constraints,
        objective_id="model-information-v1",
        model_ids=("additive", "interaction"),
        analysis_id="analysis-source-tree-123",
    )

    assert "control" in plan.selected_candidate_ids
    assert plan.planned_total_cost == 8.0
    assert plan.objective_value == pytest.approx(5.0)


def test_prospective_plan_identity_freezes_registered_inputs():
    candidates = _constraint_candidates()
    utility = _constraint_utility(candidates)
    constraints = phx.uq.ExperimentalBatchConstraints(
        8.0,
        3,
        mutually_exclusive_candidate_groups=(("perturb-a", "perturb-b"),),
        minimum_diversity_groups=3,
        maximum_per_diversity_group=1,
    )

    def select(
        values,
        *,
        objective="model-information-v1",
        models=("additive", "interaction"),
        analysis="analysis-a",
    ):
        return phx.uq.select_experimental_batch(
            values,
            utility,
            constraints,
            objective_id=objective,
            model_ids=models,
            analysis_id=analysis,
        )

    reference = select(candidates)
    changed_analysis = select(candidates, analysis="analysis-b")
    changed_objective = select(candidates, objective="prediction-information-v1")
    with pytest.raises(ValueError, match="model IDs must exactly match"):
        select(candidates, models=("additive", "other-model"))
    changed_candidates = list(candidates)
    changed_candidates[3] = phx.uq.ExperimentalDesignCandidate(
        "orthogonal-c",
        "temperature-c",
        2.5,
        "orthogonal",
        "substituted-prediction-source",
        diversity_group="mechanism-b",
    )
    with pytest.raises(
        ValueError, match="content and prediction sources must exactly match"
    ):
        select(tuple(changed_candidates))

    assert (
        len(
            {
                reference.plan_id,
                changed_analysis.plan_id,
                changed_objective.plan_id,
            }
        )
        == 3
    )


def test_retrospective_replay_reports_and_normalizes_unmatched_realized_budgets():
    candidates = tuple(
        phx.uq.ExperimentalDesignCandidate(
            f"candidate-{index}",
            f"condition-{index}",
            (0.5, 1.0, 1.5, 3.0)[index],
            "assay",
            "prediction-v1",
            diversity_group=f"group-{index}",
        )
        for index in range(4)
    )
    proposed = phx.uq.exact_finite_expected_utility(
        jnp.concatenate(
            tuple(_binary_channel(error) for error in (0.45, 0.3, 0.2, 0.1)),
            axis=0,
        ),
        jnp.asarray([0.5, 0.5]),
        candidates=candidates,
        model_ids=("model-a", "model-b"),
        utility_target="model_discrimination",
    )
    constraints = phx.uq.ExperimentalBatchConstraints(3.0, 2)
    coordinates = jnp.arange(4.0)[:, None]
    distances = jnp.abs(coordinates - coordinates.T)
    realized_gain = {
        candidate.candidate_id: float(index + 1)
        for index, candidate in enumerate(candidates)
    }

    replay = phx.uq.evaluate_retrospective_design(
        candidates,
        constraints,
        proposed,
        random_key=jr.key(8),
        space_filling_distances=distances,
        uncertainty_scores=jnp.asarray([4.0, 3.0, 2.0, 1.0]),
        domain_heuristic_scores=jnp.asarray([1.0, 3.0, 4.0, 2.0]),
        realized_utility=lambda selected: sum(realized_gain[value] for value in selected),
        metric_id="locked-predictive-loss-reduction",
        model_ids=("model-a", "model-b"),
        analysis_id="retrospective-analysis-v1",
        objective_id="historical-replay",
    )

    assert replay.strategy_ids == (
        "random",
        "space_filling",
        "uncertainty_only",
        "domain_heuristic",
        "proposed_design",
    )
    assert replay.evaluation_kind == "retrospective_cost_normalized_replay"
    assert replay.comparison_basis == "realized_utility_per_planned_total_cost"
    assert all(plan.budget == 3.0 for plan in replay.plans)
    assert all(plan.planned_total_cost <= plan.budget for plan in replay.plans)
    assert not replay.matched_planned_total_cost
    assert not replay.matched_batch_size
    assert bool(jnp.all(replay.realized_valid))
    assert bool(jnp.all(replay.cost_normalized_valid))
    assert jnp.allclose(
        replay.cost_normalized_realized_utility,
        replay.realized_utility / replay.planned_total_costs,
    )
    assert tuple(replay.selected_batch_sizes.tolist())[-1] == 1
