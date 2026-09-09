#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _dataset():
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="poisson",
        observable_id="u",
        model_id="low-pinn",
        approximation_id="low",
        observable_contract_id="scalar-field",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="poisson",
        observable_id="u",
        model_id="high-pinn",
        approximation_id="high",
        observable_contract_id="scalar-field",
    )
    hierarchy = phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("low", "high"),),
        target_level_id="high",
    )
    points = jnp.linspace(-1.0, 1.0, 9)
    cases = tuple(
        phx.fidelity.FidelityCaseSpec(
            {"x": point},
            case_id=f"case-{index}",
            split_group_id=f"physical-{index}",
        )
        for index, point in enumerate(points)
    )
    evaluations = []
    for index, point in enumerate(points):
        evaluations.append(
            phx.fidelity.FidelityEvaluation(
                0.8 * point * point + 0.1,
                case_id=f"case-{index}",
                pair_id=f"pair-{index}",
                level_id="low",
                evaluator_id="observations",
                valid=True,
                cost=1.0 + index,
                cost_unit="relative",
            )
        )
        if index in (0, 3, 6, 8):
            valid = index != 8
            evaluations.append(
                phx.fidelity.FidelityEvaluation(
                    point * point if valid else jnp.nan,
                    case_id=f"case-{index}",
                    pair_id=f"pair-{index}",
                    level_id="high",
                    evaluator_id="observations",
                    valid=valid,
                    cost=100.0 + index,
                    cost_unit="relative",
                )
            )
    return phx.fidelity.FidelityDataset(hierarchy, cases, tuple(evaluations))


def test_fidelity_observations_become_exact_fixed_residual_penalty():
    dataset = _dataset()
    geometry = phx.domain.Interval1d(-1.0, 1.0)
    prepared = phx.terms.prepare_fidelity_observation_penalty(
        dataset,
        "high",
        field="u",
        component=geometry.component(),
    )
    field = geometry.Function("x")(lambda x: x * x)

    assert isinstance(prepared.term, phx.terms.ResidualPenalty)
    assert prepared.targets.shape == (3,)
    assert prepared.rejected_evaluation_ids
    assert jnp.allclose(prepared.weights, jnp.ones((3,)))
    assert jnp.allclose(prepared.term.loss({"u": field}), 0.0)

    other = geometry.component().points({"x": jnp.asarray([0.0])})
    with pytest.raises(ValueError, match="metadata"):
        prepared.term.condition.target(other)


def test_fidelity_split_requirements_reserve_target_groups():
    dataset = _dataset()
    split = phx.fidelity.split_fidelity_dataset(
        dataset,
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=3,
        requirements=phx.fidelity.FidelitySplitRequirements(
            train={"high": 1},
            validation={"high": 1},
            test={"high": 1},
        ),
    )

    assert len(split.train.evaluations_at("high", active_only=True)) >= 1
    assert len(split.validation.evaluations_at("high", active_only=True)) >= 1
    assert len(split.test.evaluations_at("high", active_only=True)) >= 1

    with pytest.raises(ValueError, match="Unable to construct"):
        phx.fidelity.split_fidelity_dataset(
            dataset,
            train_fraction=0.5,
            validation_fraction=0.25,
            seed=3,
            requirements=phx.fidelity.FidelitySplitRequirements(
                train={"high": 2},
                validation={"high": 2},
                test={"high": 2},
                max_attempts=8,
            ),
        )
