#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax._model import AbstractArrayModel
from phydrax._strict import StrictModule
from phydrax.ml import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.model_selection import (
    cross_validate,
    DifferentiableSearchAdapter,
    GridSearch,
    KFoldPlan,
    nested_cross_validate,
    NestedSplitPlan,
    RandomSearch,
    SuccessiveHalvingSearch,
)
from phydrax.optim import DifferentialEvolutionSearch


class _ConstantModel(AbstractArrayModel):
    center: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(self, center):
        self.center = jnp.asarray(center)
        self.in_size = 1
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None):
        del key
        values = jnp.asarray(x)
        center = jnp.reshape(
            self.center,
            self.center.shape + (1,) * (values.ndim - 1 - self.center.ndim),
        )
        return jnp.broadcast_to(center, values.shape[:-1])


class _MeanRecipe(AbstractRecipe):
    offset: jax.Array
    differentiable: bool = eqx.field(static=True)

    def __init__(self, offset=0.0, *, differentiable=True):
        self.offset = jnp.asarray(offset)
        self.differentiable = bool(differentiable)

    def fit_batch(self, batch, /, *, key=None):
        if key is None:
            raise ValueError("test recipe requires an explicit key")
        targets = batch.require_targets()
        if batch.target_shape != ():
            raise ValueError("test recipe requires scalar targets")
        target_valid = (
            jnp.ones_like(targets, dtype=bool)
            if batch.target_mask is None
            else batch.target_mask
        )
        active = batch.sample_mask & target_valid
        weights = jnp.where(active, batch.sample_weight, 0.0)
        mass = jnp.sum(weights, axis=-1)
        center = (
            jnp.sum(jnp.where(active, weights * targets, 0.0), axis=-1)
            / jnp.where(mass > 0.0, mass, 1.0)
            + self.offset
        )
        valid = mass > 0.0
        status = jnp.where(valid, ML_SUCCESS, 1).astype(jnp.int32)
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.zeros_like(center),
            effective_samples=mass,
            method="test_mean",
        )
        contract = (
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_targets="smooth",
                fit_weights="conditional",
                fit_hyperparameters="smooth",
                fit_mode="direct",
                conditions=("Positive training mass.",),
            )
            if self.differentiable
            else GradientContract()
        )
        return FitResult(
            _ConstantModel(center),
            diagnostics,
            valid=valid,
            status=status,
            method="test_mean",
            gradient_contract=contract,
        )


class _MetricResult(StrictModule):
    value: jax.Array
    valid: jax.Array
    status: jax.Array
    effective_weight: jax.Array

    def __init__(self, value, *, valid, status, effective_weight):
        self.value = jnp.asarray(value)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_weight = jnp.asarray(effective_weight)


def _structured_scorer(predictions, targets, *, sample_weight, mask):
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets)
    active = jnp.asarray(mask, dtype=bool)
    weights = jnp.where(active, jnp.asarray(sample_weight), 0.0)
    mass = jnp.sum(weights, axis=-1)
    error = predictions - targets
    mse = jnp.sum(jnp.where(active, weights * error**2, 0.0), axis=-1) / jnp.where(
        mass > 0.0, mass, 1.0
    )
    bias = jnp.sum(jnp.where(active, weights * error, 0.0), axis=-1) / jnp.where(
        mass > 0.0, mass, 1.0
    )
    valid = mass > 0.0
    status = jnp.where(valid, 0, 1)
    return {
        "neg_mse": _MetricResult(-mse, valid=valid, status=status, effective_weight=mass),
        "bias": _MetricResult(bias, valid=valid, status=status, effective_weight=mass),
    }


def _batch(targets):
    targets = jnp.asarray(targets, dtype=float)
    features = jnp.arange(targets.size, dtype=float).reshape(targets.shape + (1,))
    return MLBatch(features, targets)


def test_cross_validation_refits_on_training_only_and_aggregates_structured_scores():
    batch = _batch(jnp.arange(9.0))
    splits = KFoldPlan(3, shuffle=False).split(batch, key=jr.key(1))
    result = cross_validate(
        _MeanRecipe(), batch, splits, _structured_scorer, key=jr.key(2)
    )

    assert set(result.aggregate_score.value) == {"bias", "neg_mse"}
    for evaluation in result.folds:
        expected = jnp.mean(batch.targets[evaluation.fold.train_indices])
        fitted = evaluation.fit_result.as_trainable()
        assert jnp.allclose(fitted.center, expected)
        assert not bool(
            jnp.any(
                jnp.isin(
                    evaluation.fold.train_indices,
                    evaluation.fold.validation_indices,
                )
            )
        )
    values = jnp.stack(
        tuple(fold.scorer_result.value["neg_mse"] for fold in result.folds)
    )
    masses = jnp.stack(
        tuple(fold.scorer_result.effective_weight["neg_mse"] for fold in result.folds)
    )
    expected_aggregate = jnp.sum(values * masses) / jnp.sum(masses)
    assert jnp.allclose(result.aggregate_score.value["neg_mse"], expected_aggregate)
    assert bool(result.valid)


def test_grid_random_and_successive_halving_are_deterministic_and_exact():
    batch = _batch(jnp.zeros(12))
    splits = KFoldPlan(4, shuffle=True).split(batch, key=jr.key(3))
    parameters = {"offset": (-2.0, 0.0, 2.0)}

    grid = GridSearch(parameters, primary_metric="neg_mse").run(
        _MeanRecipe, batch, splits, _structured_scorer, key=jr.key(4)
    )
    random_plan = RandomSearch(parameters, 2, primary_metric="neg_mse")
    random_first = random_plan.run(
        _MeanRecipe, batch, splits, _structured_scorer, key=jr.key(5)
    )
    random_repeated = random_plan.run(
        _MeanRecipe, batch, splits, _structured_scorer, key=jr.key(5)
    )
    halving = SuccessiveHalvingSearch(
        parameters, factor=2, min_folds=1, primary_metric="neg_mse"
    ).run(_MeanRecipe, batch, splits, _structured_scorer, key=jr.key(6))

    assert grid.best_candidate.as_kwargs()["offset"] == 0.0
    assert jnp.allclose(grid.best_fit.as_trainable().center, 0.0)
    assert tuple(
        item.candidate.candidate_id for item in random_first.evaluations
    ) == tuple(item.candidate.candidate_id for item in random_repeated.evaluations)
    assert random_first.best_candidate.candidate_id == (
        random_repeated.best_candidate.candidate_id
    )
    assert halving.best_candidate.as_kwargs()["offset"] == 0.0
    assert halving.rungs[-1].num_folds == len(splits.folds)
    assert grid.gradient_contract.fit_hyperparameters == "none"
    assert halving.gradient_contract.fit_hyperparameters == "none"


def test_nested_search_never_exposes_outer_holdouts_to_inner_fits():
    batch = _batch(jnp.zeros(12))
    nested_plan = NestedSplitPlan(KFoldPlan(3, shuffle=True), KFoldPlan(2, shuffle=True))
    result = nested_cross_validate(
        GridSearch({"offset": (-1.0, 0.0, 1.0)}, primary_metric="neg_mse"),
        _MeanRecipe,
        batch,
        nested_plan,
        _structured_scorer,
        key=jr.key(8),
    )

    assert len(result.folds) == 3
    for evaluation in result.folds:
        outer = evaluation.split.outer_fold
        searched = evaluation.search_result.split_result
        assert jnp.array_equal(
            jnp.sort(searched.sample_indices), jnp.sort(outer.train_indices)
        )
        assert not bool(
            jnp.any(jnp.isin(searched.sample_indices, outer.validation_indices))
        )
        for candidate in evaluation.search_result.evaluations:
            for inner_fold in candidate.cross_validation.folds:
                assert not bool(
                    jnp.any(
                        jnp.isin(
                            inner_fold.fold.train_indices,
                            outer.validation_indices,
                        )
                    )
                )
    assert bool(result.valid)


def test_fixed_fold_objective_is_differentiable_but_choices_are_stopped():
    batch = _batch(jnp.linspace(-1.0, 1.0, 8))
    splits = KFoldPlan(2, shuffle=False).split(batch, key=jr.key(9))

    def objective(offset):
        result = cross_validate(
            _MeanRecipe(offset),
            batch,
            splits,
            _structured_scorer,
            key=jr.key(10),
        )
        return result.aggregate_score.value["neg_mse"]

    derivative = jax.grad(objective)(jnp.asarray(0.25))
    assert jnp.isfinite(derivative)
    assert splits.gradient_contract.fit_hyperparameters == "none"

    adapter = DifferentiableSearchAdapter(
        jnp.asarray([0.0]),
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        DifferentialEvolutionSearch(4, 0),
        scorer_differentiable=True,
        primary_metric="neg_mse",
    )
    with pytest.raises(ValueError, match="rejects hyperparameter differentiation"):
        adapter.run(
            lambda vector: _MeanRecipe(vector[0], differentiable=False),
            batch,
            splits,
            _structured_scorer,
            key=jr.key(11),
        )
