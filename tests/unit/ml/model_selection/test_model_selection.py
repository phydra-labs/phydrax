#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax import (
    DerivativeContract,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from phydrax._model import AbstractArrayModel
from phydrax._strict import StrictModule
from phydrax.linalg.svd import RandomizedSVD
from phydrax.ml import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.compose import FittedPipeline, Pipeline
from phydrax.ml.decomposition import IncrementalPCA, PCA, SubspaceModel
from phydrax.ml.linear import RidgeRecipe
from phydrax.ml.metrics import (
    accuracy_score,
    FunctionScorer,
    log_loss,
    mean_squared_error,
)
from phydrax.ml.model_selection import (
    assemble_out_of_fold_predictions,
    cross_validate,
    DifferentiableSearchAdapter,
    FoldRecord,
    GridSearch,
    KFoldPlan,
    nested_cross_validate,
    NestedSplitPlan,
    RandomSearch,
    SplitPlanResult,
    SuccessiveHalvingSearch,
    TimeSeriesSplitPlan,
)
from phydrax.optim import DifferentialEvolutionSearch


_STOPPED_CONTRACT = DerivativeContract(
    (SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.NONE),),
    route=DerivativeRoute.STOPPED,
)


class _ConstantModel(AbstractArrayModel):
    center: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(self, center: Any) -> None:
        self.center = jnp.asarray(center)
        self.in_size = 1
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
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

    def __init__(self, offset: Any = 0.0, *, differentiable: Any = True) -> None:
        self.offset = jnp.asarray(offset)
        self.differentiable = bool(differentiable)

    def fit_batch(self, batch: Any, /, *, key: Any = None) -> Any:
        if key is None:
            raise ValueError("test recipe requires an explicit key")
        targets = batch.require_targets()
        if batch.target_shape != ():
            raise ValueError("test recipe requires scalar targets")
        target_valid = (
            jnp.ones_like(targets, dtype="bool")
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
            DerivativeContract(
                (
                    SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
                    SurfaceDerivative(
                        DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH
                    ),
                    SurfaceDerivative(
                        DerivativeSurface.FIT_TARGETS, GradientLevel.SMOOTH
                    ),
                    SurfaceDerivative(
                        DerivativeSurface.FIT_WEIGHTS, GradientLevel.CONDITIONAL
                    ),
                    SurfaceDerivative(
                        DerivativeSurface.FIT_HYPERPARAMETERS, GradientLevel.SMOOTH
                    ),
                ),
                route=DerivativeRoute.DIRECT,
                conditions=("Positive training mass.",),
            )
            if self.differentiable
            else _STOPPED_CONTRACT
        )
        return FitResult(
            _ConstantModel(center),
            diagnostics,
            valid=valid,
            status=status,
            method="test_mean",
            derivative_contract=contract,
        )


class _ResponseClassifierModel(AbstractArrayModel):
    in_size: int = 1
    out_size: int = 2

    def decision_function(self, x: Any, /) -> Any:
        return jnp.asarray(x)[..., 0]

    def predict_proba(self, x: Any, /) -> Any:
        positive = jax.nn.sigmoid(self.decision_function(x))
        return jnp.stack((1.0 - positive, positive), axis=-1)

    def predict(self, x: Any, /) -> Any:
        return (self.decision_function(x) >= 0.0).astype(jnp.int32)

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        del key
        return self.decision_function(x)


class _ResponseClassifierRecipe(AbstractRecipe):
    def fit_batch(self, batch: Any, /, *, key: Any = None) -> Any:
        del batch, key
        return FitResult(
            _ResponseClassifierModel(),
            FitDiagnostics(valid=True, status=0, method="response-test"),
            valid=True,
            status=0,
            method="response-test",
            derivative_contract=_STOPPED_CONTRACT,
        )


class _MetricResult(StrictModule):
    value: jax.Array
    valid: jax.Array
    status: jax.Array
    effective_weight: jax.Array

    def __init__(
        self, value: Any, *, valid: Any, status: Any, effective_weight: Any
    ) -> None:
        self.value = jnp.asarray(value)
        self.valid = jnp.asarray(valid, dtype="bool")
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_weight = jnp.asarray(effective_weight)


def _structured_metric(
    targets: Any, predictions: Any, *, sample_weight: Any, mask: Any
) -> Any:
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets)
    active = jnp.asarray(mask, dtype="bool")
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


_structured_scorer = FunctionScorer(
    _structured_metric,
    name="structured",
    greater_is_better=True,
)


def _batch(targets: Any) -> Any:
    targets = jnp.asarray(targets, dtype="float64")
    features = jnp.arange(targets.size, dtype="float64").reshape(targets.shape + (1,))
    return MLBatch(features, targets)


def _split(batch: Any, pairs: Any, *, sample_indices: Any = None) -> Any:
    folds = tuple(
        FoldRecord(
            train,
            validation,
            fold_id=fold_id,
            num_samples=batch.sample_count,
        )
        for fold_id, (train, validation) in enumerate(pairs)
    )
    samples = (
        jnp.arange(batch.sample_count, dtype=jnp.int32)
        if sample_indices is None
        else jnp.asarray(sample_indices, dtype=jnp.int32)
    )
    return SplitPlanResult(
        folds,
        sample_indices=samples,
        key=jr.key(90),
        method="test",
    )


def test_model_selection_scenario_1() -> None:
    batch = _batch(jnp.arange(9.0))
    splits = KFoldPlan(3, shuffle=False).split(batch, key=jr.key(1))
    result = cross_validate(
        _MeanRecipe(), batch, splits, _structured_scorer, key=jr.key(2)
    )
    assembled = assemble_out_of_fold_predictions(result, batch)

    assert set(result.aggregate_score.value) == {"bias", "neg_mse"}
    for evaluation in result.folds:
        expected = jnp.mean(batch.targets[evaluation.fold.train_indices])
        fitted = evaluation.fit_result.as_trainable()
        assert isinstance(fitted, _ConstantModel)
        assert jnp.allclose(fitted.center, expected)
        assert jnp.allclose(
            assembled.predictions[evaluation.fold.validation_indices], expected
        )
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
    assert jnp.array_equal(assembled.sample_mask, batch.sample_mask)
    for evaluation in result.folds:
        assert jnp.all(
            assembled.fold_ids[evaluation.fold.validation_indices]
            == evaluation.fold.fold_id
        )
    assert bool(result.valid)
    batch = MLBatch(
        jnp.asarray([[-2.0], [-1.0], [1.0], [2.0]]),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
    )
    splits = KFoldPlan(2, shuffle=False).split(batch, key=jr.key(30))
    accuracy = cross_validate(
        _ResponseClassifierRecipe(),
        batch,
        splits,
        FunctionScorer(
            accuracy_score,
            greater_is_better=True,
            response_method="predict",
        ),
        key=jr.key(31),
    )
    probability = cross_validate(
        _ResponseClassifierRecipe(),
        batch,
        splits,
        FunctionScorer(
            log_loss,
            greater_is_better=False,
            response_method="predict_proba",
        ),
        key=jr.key(32),
    )
    decision = cross_validate(
        _ResponseClassifierRecipe(),
        batch,
        splits,
        FunctionScorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="decision_function",
        ),
        key=jr.key(33),
    )

    assert jnp.allclose(accuracy.aggregate_score.value, 1.0)
    assert all(fold.predictions.ndim == 2 for fold in probability.folds)
    assert all(fold.predictions.ndim == 1 for fold in decision.folds)
    batch = _batch(jnp.linspace(-1.0, 1.0, 6))
    splits = KFoldPlan(2, shuffle=False).split(batch, key=jr.key(34))
    with pytest.raises(TypeError, match="FunctionScorer"):
        cross_validate(
            _MeanRecipe(),
            batch,
            splits,
            _structured_metric,
            key=jr.key(35),
        )
    with pytest.raises(TypeError, match="does not provide the requested predict"):
        cross_validate(
            _MeanRecipe(),
            batch,
            splits,
            FunctionScorer(
                accuracy_score,
                greater_is_better=True,
                response_method="predict",
            ),
            key=jr.key(36),
        )

    previous = IncrementalPCA(1).fit_batch(batch).as_trainable()
    with pytest.raises(ValueError, match="fresh-fit"):
        cross_validate(
            # ty: ignore[invalid-argument-type]
            IncrementalPCA(1, previous=previous),
            batch,
            splits,
            _structured_scorer,
            key=jr.key(37),
        )
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
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(grid.best_fit.as_trainable().center, 0.0)
    assert tuple(
        item.candidate.candidate_id for item in random_first.evaluations
    ) == tuple(item.candidate.candidate_id for item in random_repeated.evaluations)
    assert random_first.best_candidate.candidate_id == (
        random_repeated.best_candidate.candidate_id
    )
    assert halving.best_candidate.as_kwargs()["offset"] == 0.0
    assert halving.rungs[-1].num_folds == len(splits.folds)
    assert (
        grid.derivative_contract.level(DerivativeSurface.FIT_HYPERPARAMETERS)
        is GradientLevel.NONE
    )
    assert (
        halving.derivative_contract.level(DerivativeSurface.FIT_HYPERPARAMETERS)
        is GradientLevel.NONE
    )
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


def _subspace_batch() -> MLBatch:
    coordinate = jnp.linspace(-2.0, 2.0, 12)
    features = jnp.stack((coordinate, 0.2 * coordinate**2, jnp.sin(coordinate)), axis=-1)
    return MLBatch(features, coordinate[:, None])


@pytest.mark.parametrize("recipe_kind", ("dense", "randomized", "incremental"))
def test_subspace_cross_validation_routes_fit_keys(recipe_kind: str) -> None:
    batch = _subspace_batch()
    recipe: AbstractRecipe
    if recipe_kind == "incremental":
        recipe = IncrementalPCA(1)
    elif recipe_kind == "randomized":
        recipe = PCA(1, method=RandomizedSVD(oversampling=2, power_iterations=1))
        with pytest.raises(ValueError, match="key"):
            recipe.fit_batch(batch)
    else:
        recipe = PCA(1)
        with pytest.raises(ValueError, match="key"):
            recipe.fit_batch(batch, key=jr.key(101))
    splits = KFoldPlan(3, shuffle=False).split(batch, key=jr.key(102))
    root = jr.key(103)
    scorer = FunctionScorer(mean_squared_error, greater_is_better=False)
    result = cross_validate(recipe, batch, splits, scorer, key=root)

    assert bool(result.valid)
    for position, evaluation in enumerate(result.folds):
        addressed_fit_key, prediction_key = jr.split(jr.fold_in(root, position))
        fit_key = addressed_fit_key if recipe_kind == "randomized" else None
        if fit_key is None:
            assert evaluation.fit_key is None
        else:
            assert jnp.array_equal(jr.key_data(evaluation.fit_key), jr.key_data(fit_key))
        assert jnp.array_equal(
            jr.key_data(evaluation.prediction_key), jr.key_data(prediction_key)
        )
        expected = recipe.fit_batch(
            batch.take_samples(evaluation.fold.train_indices), key=fit_key
        ).as_trainable()
        validation = batch.take_samples(evaluation.fold.validation_indices)
        assert jnp.allclose(
            evaluation.predictions,
            expected(validation.dense_features(), key=prediction_key),
        )


@pytest.mark.parametrize("search_kind", ("grid", "random", "halving"))
@pytest.mark.parametrize("randomized", (False, True), ids=("dense", "randomized"))
def test_subspace_search_refit_routes_fit_keys(
    search_kind: str, randomized: bool
) -> None:
    batch = _subspace_batch()
    method = RandomizedSVD(oversampling=2, power_iterations=1) if randomized else None

    def factory(n_components: int) -> PCA:
        return PCA(n_components, method=method)

    parameters = {"n_components": (1,)}
    if search_kind == "random":
        plan = RandomSearch(parameters, 1)
    elif search_kind == "halving":
        plan = SuccessiveHalvingSearch(parameters, min_folds=1)
    else:
        plan = GridSearch(parameters)
    root = jr.key(104)
    streams = jr.split(root, 4 if search_kind == "random" else 3)
    addressed_refit_key = streams[-1]
    result = plan.run(
        factory,
        batch,
        KFoldPlan(3, shuffle=False),
        FunctionScorer(mean_squared_error, greater_is_better=False),
        key=root,
    )
    fit_key = addressed_refit_key if randomized else None
    if fit_key is None:
        assert result.refit_key is None
    else:
        assert jnp.array_equal(
            jr.key_data(result.refit_key), jr.key_data(addressed_refit_key)
        )
    assert bool(result.valid)
    fitted = result.best_fit.as_trainable()
    expected = result.best_recipe.fit_batch(
        batch.take_samples(result.split_result.sample_indices), key=fit_key
    ).as_trainable()
    assert isinstance(fitted, SubspaceModel)
    assert isinstance(expected, SubspaceModel)
    assert jnp.allclose(fitted.components, expected.components)
    assert jnp.allclose(fitted(batch.dense_features()), expected(batch.dense_features()))


@pytest.mark.parametrize("randomized", (False, True), ids=("dense", "randomized"))
def test_subspace_pipeline_cv_and_search_refit(randomized: bool) -> None:
    vector_batch = _subspace_batch()
    batch = MLBatch(vector_batch.features, vector_batch.require_targets()[..., 0])
    method = RandomizedSVD(oversampling=2, power_iterations=1) if randomized else None

    def factory(alpha: float) -> Pipeline:
        return Pipeline(
            (("pca", PCA(1, method=method)), ("regressor", RidgeRecipe(alpha)))
        )

    root = jr.key(105)
    _, evaluation_key, refit_key = jr.split(root, 3)
    result = GridSearch({"alpha": (0.1, 1.0)}).run(
        factory,
        batch,
        KFoldPlan(3, shuffle=False),
        FunctionScorer(mean_squared_error, greater_is_better=False),
        key=root,
    )
    assert bool(result.valid)
    for candidate in result.evaluations:
        candidate_key = jr.fold_in(evaluation_key, candidate.candidate.candidate_id)
        for position, evaluation in enumerate(candidate.cross_validation.folds):
            fit_key, prediction_key = jr.split(jr.fold_in(candidate_key, position))
            expected = candidate.recipe.fit_batch(
                batch.take_samples(evaluation.fold.train_indices), key=fit_key
            ).as_trainable()
            fitted = evaluation.fit_result.as_trainable()
            assert isinstance(fitted, FittedPipeline)
            assert isinstance(expected, FittedPipeline)
            subspace = fitted.steps[0][1]
            expected_subspace = expected.steps[0][1]
            assert isinstance(subspace, SubspaceModel)
            assert isinstance(expected_subspace, SubspaceModel)
            assert jnp.allclose(subspace.components, expected_subspace.components)
            assert jnp.allclose(
                evaluation.predictions,
                expected(
                    batch.take_samples(evaluation.fold.validation_indices).features,
                    key=prediction_key,
                ),
            )
    assert jnp.array_equal(jr.key_data(result.refit_key), jr.key_data(refit_key))
    expected_refit = result.best_recipe.fit_batch(
        batch.take_samples(result.split_result.sample_indices), key=refit_key
    ).as_trainable()
    assert jnp.allclose(
        result.best_fit.as_trainable()(batch.features),
        expected_refit(batch.features),
    )


def test_out_of_fold_assembly_contracts() -> None:
    batch = _batch(jnp.arange(12.0).reshape(2, 6))
    splits = _split(
        batch,
        (
            ((4, 5), (1, 2)),
            ((1, 2), (4, 5)),
        ),
        sample_indices=(1, 2, 4, 5),
    )
    result = cross_validate(
        _MeanRecipe(), batch, splits, _structured_scorer, key=jr.key(12)
    )
    vector_folds = tuple(
        eqx.tree_at(
            lambda item: item.predictions,
            fold,
            jnp.stack((fold.predictions, fold.predictions + 1.0), axis=-1),
        )
        for fold in result.folds
    )
    vector_result = eqx.tree_at(lambda item: item.folds, result, vector_folds)

    assembled = assemble_out_of_fold_predictions(vector_result, batch)

    assert assembled.predictions.shape == (2, 6, 2)
    assert jnp.all(assembled.predictions[:, (0, 3), :] == 0.0)
    assert jnp.array_equal(
        assembled.sample_mask,
        jnp.asarray(
            [
                [False, True, True, False, True, True],
                [False, True, True, False, True, True],
            ]
        ),
    )
    assert jnp.array_equal(assembled.fold_ids, jnp.asarray([-1, 0, 0, -1, 1, 1]))
    for fold in vector_result.folds:
        expected = jnp.stack(
            (fold.predictions[..., 0], fold.predictions[..., 0] + 1.0), axis=-1
        )
        assert jnp.allclose(
            assembled.predictions[:, fold.fold.validation_indices, :], expected
        )
    batch = _batch(jnp.arange(6.0))
    duplicate = _split(
        batch,
        (
            ((3, 4, 5), (0, 1, 2)),
            ((0, 1), (2, 3, 4, 5)),
        ),
    )
    missing = _split(
        batch,
        (
            ((2, 3, 4, 5), (0, 1)),
            ((0, 1, 4, 5), (2, 3)),
        ),
    )
    temporal = TimeSeriesSplitPlan(2, validation_size=2, min_train_size=2).split(
        batch, key=jr.key(13)
    )

    for split in (duplicate, missing, temporal):
        result = cross_validate(
            _MeanRecipe(), batch, split, _structured_scorer, key=jr.key(14)
        )
        with pytest.raises(ValueError, match="cover every selected sample exactly once"):
            assemble_out_of_fold_predictions(result, batch)
    groups = jnp.asarray([0, 0, 1, 1, 2, 2])
    batch = MLBatch(
        jnp.arange(6.0).reshape(6, 1),
        jnp.arange(6.0),
        groups=groups,
    )
    leaking = _split(
        batch,
        (
            ((1, 3, 5), (0, 2, 4)),
            ((0, 2, 4), (1, 3, 5)),
        ),
    )
    leaking_result = cross_validate(
        _MeanRecipe(), batch, leaking, _structured_scorer, key=jr.key(15)
    )
    with pytest.raises(ValueError, match="group"):
        assemble_out_of_fold_predictions(leaking_result, batch)

    cut = _split(
        batch,
        (
            ((4, 5), (0, 2, 3)),
            ((0, 2, 3), (4, 5)),
        ),
        sample_indices=(0, 2, 3, 4, 5),
    )
    cut_result = cross_validate(
        _MeanRecipe(), batch, cut, _structured_scorer, key=jr.key(16)
    )
    with pytest.raises(ValueError, match="cut through a group"):
        assemble_out_of_fold_predictions(cut_result, batch)

    case_batch = MLBatch(
        jnp.broadcast_to(jnp.arange(6.0).reshape(1, 6, 1), (2, 6, 1)),
        jnp.broadcast_to(jnp.arange(6.0), (2, 6)),
        groups=jnp.asarray(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 1, 1, 1, 2, 2],
            ]
        ),
    )
    case_splits = KFoldPlan(2, shuffle=False).split(case_batch, key=jr.key(17))
    case_result = cross_validate(
        _MeanRecipe(), case_batch, case_splits, _structured_scorer, key=jr.key(18)
    )
    with pytest.raises(ValueError, match="Case-dependent groups"):
        assemble_out_of_fold_predictions(case_result, case_batch)
    batch = _batch(jnp.arange(12.0).reshape(2, 6))
    splits = KFoldPlan(2, shuffle=False).split(batch, key=jr.key(19))
    result = cross_validate(
        _MeanRecipe(), batch, splits, _structured_scorer, key=jr.key(20)
    )

    replacements = (
        (result.folds[0].predictions[0], ValueError, "case prefix"),
        (result.folds[0].predictions[:, :-1], ValueError, "sample dimension"),
        (
            jnp.stack(
                (result.folds[0].predictions, result.folds[0].predictions), axis=-1
            ),
            ValueError,
            "trailing shape",
        ),
        (result.folds[0].predictions.astype(jnp.int32), TypeError, "common dtype"),
        ({"prediction": result.folds[0].predictions}, TypeError, "not PyTrees"),
    )
    for replacement, error, message in replacements:
        changed_fold = eqx.tree_at(
            lambda item: item.predictions,
            result.folds[0],
            replacement,
        )
        changed_result = eqx.tree_at(
            lambda item: item.folds,
            result,
            (changed_fold, *result.folds[1:]),
        )
        with pytest.raises(error, match=message):
            assemble_out_of_fold_predictions(changed_result, batch)


def test_fixed_fold_objective_is_differentiable_but_choices_are_stopped() -> None:
    batch = _batch(jnp.linspace(-1.0, 1.0, 8))
    splits = KFoldPlan(2, shuffle=False).split(batch, key=jr.key(9))

    def objective(offset: Any) -> Any:
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
    assert (
        splits.derivative_contract.level(DerivativeSurface.FIT_HYPERPARAMETERS)
        is GradientLevel.NONE
    )

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
