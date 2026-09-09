#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _hierarchy():
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="manufactured",
        observable_id="response",
        model_id="low-model",
        approximation_id="coarse",
        observable_contract_id="scalar-response",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="manufactured",
        observable_id="response",
        model_id="high-model",
        approximation_id="fine",
        observable_contract_id="scalar-response",
    )
    relation = phx.fidelity.FidelityRelation("low", "high")
    return phx.fidelity.FidelityHierarchy(
        (high, low),
        (relation,),
        target_level_id="high",
    )


def _dataset():
    hierarchy = _hierarchy()
    cases = tuple(
        phx.fidelity.FidelityCaseSpec(
            jnp.asarray([float(index)]),
            case_id=f"case-{index}",
            split_group_id=f"physical-{index}",
        )
        for index in range(6)
    )
    evaluations = tuple(
        phx.fidelity.FidelityEvaluation(
            jnp.asarray(float(index) + offset),
            case_id=f"case-{index}",
            pair_id=f"pair-{index}",
            level_id=level,
            evaluator_id=f"{level}-evaluator",
            valid=True,
            cost=cost,
            cost_unit="seconds",
        )
        for index in range(6)
        for level, offset, cost in (("low", 0.5, 1.0), ("high", 0.0, 10.0))
    )
    return phx.fidelity.FidelityDataset(hierarchy, cases, evaluations)


def test_fidelity_hierarchy_dataset_split_and_archive(tmp_path):
    hierarchy = _hierarchy()
    assert hierarchy.is_linear
    assert hierarchy.linear_path().level_ids == ("low", "high")
    assert hierarchy.hierarchy_id == _hierarchy().hierarchy_id

    dataset = _dataset()
    assert len(dataset.paired("low", "high")) == 6
    split = phx.fidelity.split_fidelity_dataset(
        dataset,
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=4,
    )
    groups = [
        {case.split_group_id for case in partition.cases}
        for partition in (split.train, split.validation, split.test)
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]

    path = tmp_path / "fidelity.phx"
    phx.fidelity.write_fidelity_dataset(path, dataset)
    restored = phx.fidelity.read_fidelity_dataset(path, dataset)
    assert restored.dataset_id == dataset.dataset_id
    assert len(restored.paired("low", "high")) == 6


def test_fidelity_hierarchy_rejects_ambiguous_and_disconnected_models():
    hierarchy = _hierarchy()
    extra = phx.fidelity.FidelityLevelSpec(
        "other",
        problem_id="manufactured",
        observable_id="response",
        model_id="other-model",
        approximation_id="other",
        observable_contract_id="scalar-response",
    )
    with pytest.raises(ValueError, match="disconnected"):
        phx.fidelity.FidelityHierarchy(
            (*hierarchy.levels, extra),
            hierarchy.relations,
            target_level_id="high",
        )


def test_native_fidelity_sampler_runs_finest_level_mlmc():
    path = _hierarchy().linear_path()

    def sample_inputs(indices, key):
        keys = jax.vmap(lambda index: jr.fold_in(key, index))(indices)
        return jax.vmap(jr.normal)(keys)

    def evaluate_level(level, inputs):
        values = inputs * inputs + (0.5 if level.level_id == "low" else 0.0)
        return phx.integration.FidelityBatchEvaluation(
            values,
            level_id=level.level_id,
            evaluator_id="quadratic-levels",
            costs=1.0 if level.level_id == "low" else 10.0,
        )

    target = phx.integration.fidelity_multilevel_target(
        path,
        sample_inputs,
        evaluate_level,
        sampler_id="quadratic-fidelity-sampler",
        input_sampler_id="indexed-normal-inputs",
        evaluator_id="quadratic-levels",
    )
    estimate = phx.integration.integrate(
        lambda values, level: values,
        target,
        phx.integration.MultilevelMonteCarloPlan(
            samples_per_level=(4096, 64),
            batch_size=4096,
            estimand="finest_level",
        ),
        key=jr.key(17),
    )

    assert estimate.successful
    assert estimate.error_kind == "mlmc-finest-level-rmse"
    assert jnp.allclose(estimate.value, 1.0, atol=0.08)
    assert estimate.diagnostics.bias_estimate == 0.0
