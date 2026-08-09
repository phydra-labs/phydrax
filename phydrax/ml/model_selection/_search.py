#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from math import ceil
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...optim._differential_evolution import _bounded_differential_evolution
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract
from ._cross_validation import (
    _aggregate_scores,
    _call_scorer,
    _cross_validate_materialized,
    _cross_validation_contract,
    _predict,
    CrossValidationResult,
    FoldEvaluation,
    MetricPath,
    select_metric,
)
from ._splits import (
    _require_key,
    _validate_split_result_for_batch,
    AbstractSplitPlan,
    NestedSplitPlan,
    NestedSplitResult,
    SplitPlanResult,
)


SELECTION_SUCCESS = 0
SELECTION_INVALID_CANDIDATE = 1
SELECTION_NO_VALID_CANDIDATE = 2
SELECTION_UNSUPPORTED_GRADIENT = 3


@runtime_checkable
class _DirectionalScorer(Protocol):
    greater_is_better: bool


_EXACT_SEARCH_CONTRACT = GradientContract(
    prediction_inputs="none",
    prediction_parameters="none",
    fit_features="none",
    fit_targets="none",
    fit_weights="none",
    fit_hyperparameters="none",
    fit_mode="stopped",
    nondifferentiable_outputs=(
        "candidate_indices",
        "surviving_candidates",
        "best_candidate",
        "status",
    ),
    conditions=(
        "Candidate generation, ranking, tie-breaking, and split membership are discrete.",
        "No gradient is defined through an exact search choice.",
    ),
)


class CandidateSpec(StrictModule):
    """One canonical immutable point in a finite hyperparameter grid."""

    values: tuple[Any, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    candidate_id: int = eqx.field(static=True)

    def __init__(
        self,
        names: tuple[str, ...],
        values: tuple[Any, ...],
        /,
        *,
        candidate_id: int,
    ):
        if len(names) != len(values):
            raise ValueError(
                "Candidate parameter names and values must have equal length."
            )
        self.names = tuple(str(name) for name in names)
        self.values = tuple(values)
        self.candidate_id = int(candidate_id)

    def as_kwargs(self, /) -> dict[str, Any]:
        return dict(zip(self.names, self.values, strict=True))


class ParameterGrid(StrictModule):
    """Canonical Cartesian product with lexicographically ordered parameter names."""

    values: tuple[tuple[Any, ...], ...]
    names: tuple[str, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)

    def __init__(self, parameters: Mapping[str, Sequence[Any]], /):
        if not isinstance(parameters, Mapping) or not parameters:
            raise TypeError("parameters must be a non-empty mapping of finite sequences.")
        if any(not isinstance(name, str) for name in parameters):
            raise TypeError("Parameter names must be strings.")
        names = tuple(sorted(parameters))
        values: list[tuple[Any, ...]] = []
        size = 1
        for name in names:
            raw = parameters[name]
            if isinstance(raw, (str, bytes, set, frozenset, Mapping)):
                raise TypeError(
                    f"Parameter {name!r} values must be an ordered finite sequence."
                )
            choices = tuple(raw)
            if not choices:
                raise ValueError(f"Parameter {name!r} has no candidate values.")
            values.append(choices)
            size *= len(choices)
        self.names = names
        self.values = tuple(values)
        self.size = int(size)

    def candidates(self, /) -> tuple[CandidateSpec, ...]:
        return tuple(
            CandidateSpec(self.names, tuple(values), candidate_id=index)
            for index, values in enumerate(product(*self.values))
        )


class CandidateEvaluation(StrictModule):
    """A candidate recipe and all of its independently fitted CV folds."""

    candidate: CandidateSpec
    recipe: AbstractRecipe
    cross_validation: CrossValidationResult
    utility: Any
    valid: Any
    status: Any

    def __init__(
        self,
        candidate: CandidateSpec,
        recipe: AbstractRecipe,
        cross_validation: CrossValidationResult,
        /,
        *,
        utility: Any,
        valid: Any,
        status: Any,
    ):
        self.candidate = candidate
        self.recipe = recipe
        self.cross_validation = cross_validation
        self.utility = jnp.asarray(utility)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)


class HalvingRung(StrictModule):
    """One successive-halving comparison at a fixed number of folds."""

    evaluations: tuple[CandidateEvaluation, ...]
    surviving_candidate_ids: tuple[int, ...] = eqx.field(static=True)
    num_folds: int = eqx.field(static=True)

    def __init__(
        self,
        evaluations: tuple[CandidateEvaluation, ...],
        /,
        *,
        surviving_candidate_ids: tuple[int, ...],
        num_folds: int,
    ):
        self.evaluations = tuple(evaluations)
        self.surviving_candidate_ids = tuple(int(i) for i in surviving_candidate_ids)
        self.num_folds = int(num_folds)


class SearchResult(StrictModule):
    """Immutable exact-search outcome, including the final full-universe refit."""

    evaluations: tuple[CandidateEvaluation, ...]
    best_candidate: CandidateSpec
    best_recipe: AbstractRecipe
    best_fit: FitResult
    best_score: Any
    split_result: SplitPlanResult
    rungs: tuple[HalvingRung, ...]
    valid: Any
    status: Any
    key: Any
    refit_key: Any
    gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(
        self,
        evaluations: tuple[CandidateEvaluation, ...],
        best_evaluation: CandidateEvaluation,
        best_fit: FitResult,
        split_result: SplitPlanResult,
        /,
        *,
        rungs: tuple[HalvingRung, ...],
        key: Any,
        refit_key: Any,
        method: str,
    ):
        refit_valid = jnp.all(jnp.asarray(best_fit.valid))
        self.evaluations = tuple(evaluations)
        self.best_candidate = best_evaluation.candidate
        self.best_recipe = best_evaluation.recipe
        self.best_fit = best_fit
        self.best_score = best_evaluation.cross_validation.aggregate_score.value
        self.split_result = split_result
        self.rungs = tuple(rungs)
        self.valid = best_evaluation.valid & refit_valid
        self.status = jnp.maximum(
            best_evaluation.status,
            jnp.max(jnp.asarray(best_fit.status, dtype=jnp.int32)),
        )
        self.key = _require_key(key)
        self.refit_key = _require_key(refit_key)
        self.gradient_contract = _EXACT_SEARCH_CONTRACT
        self.method = str(method)


class AbstractSearchPlan(StrictModule):
    """Immutable selection configuration over recipe factories."""

    @abstractmethod
    def run(
        self,
        recipe_factory: Callable[..., AbstractRecipe],
        batch: MLBatch,
        split_plan: AbstractSplitPlan | SplitPlanResult,
        scorer: Any,
        /,
        *,
        key: Any,
    ) -> Any:
        raise NotImplementedError


def _materialize_splits(
    split_plan: AbstractSplitPlan | SplitPlanResult,
    batch: MLBatch,
    /,
    *,
    key: Any,
) -> SplitPlanResult:
    if isinstance(split_plan, SplitPlanResult):
        result = split_plan
    elif isinstance(split_plan, AbstractSplitPlan):
        result = split_plan.split(batch, key=key)
    else:
        raise TypeError("split_plan must be an AbstractSplitPlan or SplitPlanResult.")
    _validate_split_result_for_batch(result, batch)
    return result


def _resolve_maximize(configured: bool | None, scorer: Any, /) -> bool:
    if configured is not None:
        return bool(configured)
    if isinstance(scorer, _DirectionalScorer):
        return bool(scorer.greater_is_better)
    return True


def _make_recipe(
    recipe_factory: Callable[..., AbstractRecipe], candidate: CandidateSpec, /
) -> AbstractRecipe:
    if not callable(recipe_factory):
        raise TypeError("recipe_factory must be callable.")
    recipe = recipe_factory(**candidate.as_kwargs())
    if not isinstance(recipe, AbstractRecipe):
        raise TypeError("recipe_factory must return an unfitted AbstractRecipe.")
    return recipe


def _candidate_utility(
    result: CrossValidationResult,
    primary_metric: MetricPath,
    maximize: bool,
    /,
) -> tuple[Any, Any, Any]:
    score = jnp.asarray(select_metric(result.aggregate_score.value, primary_metric))
    metric_valid = jnp.asarray(
        select_metric(result.aggregate_score.valid, primary_metric), dtype=bool
    )
    metric_status = jnp.asarray(
        select_metric(result.aggregate_score.status, primary_metric), dtype=jnp.int32
    )
    if jnp.issubdtype(score.dtype, jnp.complexfloating):
        raise TypeError("Search ranking requires a real-valued primary metric.")
    if not jnp.issubdtype(score.dtype, jnp.number):
        raise TypeError("Search ranking requires a numeric primary metric.")
    valid = (
        jnp.asarray(result.valid, dtype=bool)
        & jnp.all(metric_valid)
        & jnp.all(jnp.isfinite(score))
    )
    mean_score = jnp.mean(score)
    utility = mean_score if maximize else -mean_score
    status = jnp.maximum(result.status, jnp.max(metric_status))
    status = jnp.where(valid, status, jnp.maximum(status, SELECTION_INVALID_CANDIDATE))
    return utility, valid, status


def _evaluate_candidate(
    candidate: CandidateSpec,
    recipe_factory: Callable[..., AbstractRecipe],
    batch: MLBatch,
    splits: SplitPlanResult,
    scorer: Any,
    /,
    *,
    key: Any,
    primary_metric: MetricPath,
    maximize: bool,
) -> CandidateEvaluation:
    recipe = _make_recipe(recipe_factory, candidate)
    result = _cross_validate_materialized(recipe, batch, splits, scorer, key=key)
    utility, valid, status = _candidate_utility(result, primary_metric, maximize)
    return CandidateEvaluation(
        candidate,
        recipe,
        result,
        utility=utility,
        valid=valid,
        status=status,
    )


def _rank_valid(
    evaluations: tuple[CandidateEvaluation, ...], /
) -> tuple[CandidateEvaluation, ...]:
    valid = [evaluation for evaluation in evaluations if bool(evaluation.valid)]
    if not valid:
        raise ValueError("Search produced no valid candidate.")
    return tuple(
        sorted(
            valid,
            key=lambda evaluation: (
                -float(evaluation.utility),
                evaluation.candidate.candidate_id,
            ),
        )
    )


def _refit_best(
    best: CandidateEvaluation,
    batch: MLBatch,
    splits: SplitPlanResult,
    /,
    *,
    key: Any,
) -> FitResult:
    universe = batch.take_samples(splits.sample_indices)
    result = best.recipe.fit_batch(universe, key=key)
    if not isinstance(result, FitResult):
        raise TypeError("Recipe.fit_batch must return a FitResult during final refit.")
    return result


class GridSearch(AbstractSearchPlan):
    """Exhaustive exact search over a finite Cartesian parameter grid."""

    parameter_grid: ParameterGrid
    maximize: bool | None = eqx.field(static=True)
    primary_metric: MetricPath = eqx.field(static=True)

    def __init__(
        self,
        parameters: Mapping[str, Sequence[Any]],
        /,
        *,
        maximize: bool | None = None,
        primary_metric: MetricPath = None,
    ):
        self.parameter_grid = ParameterGrid(parameters)
        self.maximize = None if maximize is None else bool(maximize)
        self.primary_metric = primary_metric

    def run(
        self,
        recipe_factory: Callable[..., AbstractRecipe],
        batch: MLBatch,
        split_plan: AbstractSplitPlan | SplitPlanResult,
        scorer: Any,
        /,
        *,
        key: Any,
    ) -> SearchResult:
        key = _require_key(key)
        split_key, evaluation_key, refit_key = jr.split(key, 3)
        splits = _materialize_splits(split_plan, batch, key=split_key)
        maximize = _resolve_maximize(self.maximize, scorer)
        evaluations = tuple(
            _evaluate_candidate(
                candidate,
                recipe_factory,
                batch,
                splits,
                scorer,
                key=jr.fold_in(evaluation_key, candidate.candidate_id),
                primary_metric=self.primary_metric,
                maximize=maximize,
            )
            for candidate in self.parameter_grid.candidates()
        )
        best = _rank_valid(evaluations)[0]
        best_fit = _refit_best(best, batch, splits, key=refit_key)
        return SearchResult(
            evaluations,
            best,
            best_fit,
            splits,
            rungs=(),
            key=key,
            refit_key=refit_key,
            method="grid_search",
        )


class RandomSearch(AbstractSearchPlan):
    """Exact evaluation of a key-selected subset of a finite parameter grid."""

    parameter_grid: ParameterGrid
    num_candidates: int = eqx.field(static=True)
    maximize: bool | None = eqx.field(static=True)
    primary_metric: MetricPath = eqx.field(static=True)

    def __init__(
        self,
        parameters: Mapping[str, Sequence[Any]],
        num_candidates: int,
        /,
        *,
        maximize: bool | None = None,
        primary_metric: MetricPath = None,
    ):
        grid = ParameterGrid(parameters)
        if int(num_candidates) < 1 or int(num_candidates) > grid.size:
            raise ValueError("num_candidates must lie between one and the grid size.")
        self.parameter_grid = grid
        self.num_candidates = int(num_candidates)
        self.maximize = None if maximize is None else bool(maximize)
        self.primary_metric = primary_metric

    def run(
        self,
        recipe_factory: Callable[..., AbstractRecipe],
        batch: MLBatch,
        split_plan: AbstractSplitPlan | SplitPlanResult,
        scorer: Any,
        /,
        *,
        key: Any,
    ) -> SearchResult:
        key = _require_key(key)
        sample_key, split_key, evaluation_key, refit_key = jr.split(key, 4)
        selected_ids = jnp.sort(
            jr.permutation(sample_key, self.parameter_grid.size)[: self.num_candidates]
        )
        all_candidates = self.parameter_grid.candidates()
        candidates = tuple(all_candidates[int(index)] for index in selected_ids)
        splits = _materialize_splits(split_plan, batch, key=split_key)
        maximize = _resolve_maximize(self.maximize, scorer)
        evaluations = tuple(
            _evaluate_candidate(
                candidate,
                recipe_factory,
                batch,
                splits,
                scorer,
                key=jr.fold_in(evaluation_key, candidate.candidate_id),
                primary_metric=self.primary_metric,
                maximize=maximize,
            )
            for candidate in candidates
        )
        best = _rank_valid(evaluations)[0]
        best_fit = _refit_best(best, batch, splits, key=refit_key)
        return SearchResult(
            evaluations,
            best,
            best_fit,
            splits,
            rungs=(),
            key=key,
            refit_key=refit_key,
            method="random_search",
        )


class SuccessiveHalvingSearch(AbstractSearchPlan):
    """Exact fold-budget successive halving with a final all-fold comparison."""

    parameter_grid: ParameterGrid
    factor: int = eqx.field(static=True)
    min_folds: int = eqx.field(static=True)
    maximize: bool | None = eqx.field(static=True)
    primary_metric: MetricPath = eqx.field(static=True)

    def __init__(
        self,
        parameters: Mapping[str, Sequence[Any]],
        /,
        *,
        factor: int = 3,
        min_folds: int = 1,
        maximize: bool | None = None,
        primary_metric: MetricPath = None,
    ):
        if int(factor) < 2:
            raise ValueError("factor must be at least 2.")
        if int(min_folds) < 1:
            raise ValueError("min_folds must be positive.")
        self.parameter_grid = ParameterGrid(parameters)
        self.factor = int(factor)
        self.min_folds = int(min_folds)
        self.maximize = None if maximize is None else bool(maximize)
        self.primary_metric = primary_metric

    def run(
        self,
        recipe_factory: Callable[..., AbstractRecipe],
        batch: MLBatch,
        split_plan: AbstractSplitPlan | SplitPlanResult,
        scorer: Any,
        /,
        *,
        key: Any,
    ) -> SearchResult:
        key = _require_key(key)
        split_key, evaluation_key, refit_key = jr.split(key, 3)
        full_splits = _materialize_splits(split_plan, batch, key=split_key)
        total_folds = len(full_splits.folds)
        if self.min_folds > total_folds:
            raise ValueError("min_folds cannot exceed the available CV folds.")
        maximize = _resolve_maximize(self.maximize, scorer)
        survivors = self.parameter_grid.candidates()
        resource = self.min_folds
        rungs: list[HalvingRung] = []
        final_evaluations: tuple[CandidateEvaluation, ...] | None = None
        rung_id = 0
        while True:
            rung_splits = SplitPlanResult(
                full_splits.folds[:resource],
                sample_indices=full_splits.sample_indices,
                key=full_splits.key,
                method=f"halving_{resource}_folds",
            )
            evaluations = tuple(
                _evaluate_candidate(
                    candidate,
                    recipe_factory,
                    batch,
                    rung_splits,
                    scorer,
                    key=jr.fold_in(
                        jr.fold_in(evaluation_key, rung_id), candidate.candidate_id
                    ),
                    primary_metric=self.primary_metric,
                    maximize=maximize,
                )
                for candidate in survivors
            )
            ranked = _rank_valid(evaluations)
            if resource == total_folds:
                kept = (ranked[0],)
            else:
                kept = ranked[: max(1, ceil(len(ranked) / self.factor))]
            rungs.append(
                HalvingRung(
                    evaluations,
                    surviving_candidate_ids=tuple(
                        evaluation.candidate.candidate_id for evaluation in kept
                    ),
                    num_folds=resource,
                )
            )
            if resource == total_folds:
                final_evaluations = evaluations
                best = ranked[0]
                break
            survivors = tuple(evaluation.candidate for evaluation in kept)
            resource = min(total_folds, max(resource + 1, resource * self.factor))
            rung_id += 1
        if final_evaluations is None:
            raise RuntimeError("Successive halving failed to evaluate a final rung.")
        best_fit = _refit_best(best, batch, full_splits, key=refit_key)
        return SearchResult(
            final_evaluations,
            best,
            best_fit,
            full_splits,
            rungs=tuple(rungs),
            key=key,
            refit_key=refit_key,
            method="successive_halving",
        )


class _DifferentiableCVObjective(StrictModule):
    batch: MLBatch
    splits: SplitPlanResult
    key: Any
    recipe_factory: Any = eqx.field(static=True)
    scorer: Any
    primary_metric: MetricPath = eqx.field(static=True)
    maximize: bool = eqx.field(static=True)

    def __init__(
        self,
        batch: MLBatch,
        splits: SplitPlanResult,
        recipe_factory: Callable[[Any], AbstractRecipe],
        scorer: Any,
        /,
        *,
        key: Any,
        primary_metric: MetricPath,
        maximize: bool,
    ):
        self.batch = batch
        self.splits = splits
        self.recipe_factory = recipe_factory
        self.scorer = scorer
        self.key = key
        self.primary_metric = primary_metric
        self.maximize = bool(maximize)

    def __call__(self, vector: Any, /) -> Any:
        recipe = self.recipe_factory(vector)
        result = _cross_validate_materialized(
            recipe, self.batch, self.splits, self.scorer, key=self.key
        )
        utility, valid, _ = _candidate_utility(result, self.primary_metric, self.maximize)
        return jnp.where(valid, -utility, jnp.inf)


class DifferentiableSearchResult(StrictModule):
    """Continuous objective audit plus a nondifferentiable DE-selected refit."""

    best_vector: Any
    best_objective: Any
    population_vectors: Any
    population_objectives: Any
    objective_history: Any
    best_recipe: AbstractRecipe
    best_fit: FitResult
    cross_validation: CrossValidationResult
    split_result: SplitPlanResult
    optimizer_result: Any
    valid: Any
    status: Any
    key: Any
    refit_key: Any
    gradient_contract: GradientContract
    objective_gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(
        self,
        optimizer_result: Any,
        best_recipe: AbstractRecipe,
        best_fit: FitResult,
        cross_validation: CrossValidationResult,
        split_result: SplitPlanResult,
        /,
        *,
        key: Any,
        refit_key: Any,
        objective_gradient_contract: GradientContract,
    ):
        finite = jnp.isfinite(optimizer_result.raw_objective)
        self.best_vector = optimizer_result.best_vector
        self.best_objective = optimizer_result.raw_objective
        self.population_vectors = optimizer_result.population_vectors
        self.population_objectives = optimizer_result.population_objectives
        self.objective_history = optimizer_result.best_objective_history
        self.best_recipe = best_recipe
        self.best_fit = best_fit
        self.cross_validation = cross_validation
        self.split_result = split_result
        self.optimizer_result = optimizer_result
        self.valid = (
            finite
            & cross_validation.valid
            & jnp.all(jnp.asarray(best_fit.valid, dtype=bool))
        )
        self.status = jnp.where(
            self.valid,
            jnp.maximum(
                cross_validation.status,
                jnp.max(jnp.asarray(best_fit.status, dtype=jnp.int32)),
            ),
            SELECTION_NO_VALID_CANDIDATE,
        ).astype(jnp.int32)
        self.key = _require_key(key)
        self.refit_key = _require_key(refit_key)
        self.gradient_contract = _EXACT_SEARCH_CONTRACT
        self.objective_gradient_contract = objective_gradient_contract
        self.method = "differentiable_objective_differential_evolution"


class DifferentiableSearchAdapter(AbstractSearchPlan):
    """Expose a differentiable CV objective to bounded differential evolution.

    Differential evolution and its final argmin remain explicitly nondifferentiable.
    Only the objective at a fixed vector and fixed folds carries the separately
    reported ``objective_gradient_contract``.
    """

    initial_vector: Any
    lower_bounds: Any
    upper_bounds: Any
    search: DifferentialEvolutionSearch
    scorer_differentiable: bool = eqx.field(static=True)
    maximize: bool | None = eqx.field(static=True)
    primary_metric: MetricPath = eqx.field(static=True)

    def __init__(
        self,
        initial_vector: Any,
        lower_bounds: Any,
        upper_bounds: Any,
        search: DifferentialEvolutionSearch,
        /,
        *,
        scorer_differentiable: bool,
        maximize: bool | None = None,
        primary_metric: MetricPath = None,
    ):
        if not isinstance(search, DifferentialEvolutionSearch):
            raise TypeError("search must be a DifferentialEvolutionSearch.")
        initial = jnp.asarray(initial_vector)
        lower = jnp.asarray(lower_bounds)
        upper = jnp.asarray(upper_bounds)
        if (
            initial.ndim != 1
            or lower.shape != initial.shape
            or upper.shape != initial.shape
        ):
            raise ValueError(
                "Continuous search vectors and bounds must share a 1-D shape."
            )
        if not jnp.issubdtype(initial.dtype, jnp.floating):
            raise TypeError(
                "Continuous hyperparameter vectors must be real floating arrays."
            )
        self.initial_vector = initial
        self.lower_bounds = lower
        self.upper_bounds = upper
        self.search = search
        self.scorer_differentiable = bool(scorer_differentiable)
        self.maximize = None if maximize is None else bool(maximize)
        self.primary_metric = primary_metric

    def run(
        self,
        recipe_factory: Callable[[Any], AbstractRecipe],
        batch: MLBatch,
        split_plan: AbstractSplitPlan | SplitPlanResult,
        scorer: Any,
        /,
        *,
        key: Any,
    ) -> DifferentiableSearchResult:
        key = _require_key(key)
        if not self.scorer_differentiable:
            raise ValueError(
                "Differentiable search requires an explicit scorer_differentiable=True contract."
            )
        split_key, audit_key, objective_key, search_key, refit_key = jr.split(key, 5)
        splits = _materialize_splits(split_plan, batch, key=split_key)
        initial_recipe = recipe_factory(self.initial_vector)
        if not isinstance(initial_recipe, AbstractRecipe):
            raise TypeError("recipe_factory(vector) must return an AbstractRecipe.")
        audit = _cross_validate_materialized(
            initial_recipe, batch, splits, scorer, key=audit_key
        )
        contracts = tuple(fold.fit_result.gradient_contract for fold in audit.folds)
        if any(contract.fit_hyperparameters == "none" for contract in contracts):
            raise ValueError(
                "At least one fold recipe rejects hyperparameter differentiation; "
                "the differentiable adapter fails closed."
            )
        audit_score = jnp.asarray(
            select_metric(audit.aggregate_score.value, self.primary_metric)
        )
        if not jnp.issubdtype(audit_score.dtype, jnp.floating):
            raise TypeError(
                "The differentiable search objective must have a real floating dtype."
            )
        maximize = _resolve_maximize(self.maximize, scorer)
        objective_contract = GradientContract(
            prediction_inputs=audit.gradient_contract.prediction_inputs,
            prediction_parameters=audit.gradient_contract.prediction_parameters,
            fit_features=audit.gradient_contract.fit_features,
            fit_targets=audit.gradient_contract.fit_targets,
            fit_weights=audit.gradient_contract.fit_weights,
            fit_hyperparameters=audit.gradient_contract.fit_hyperparameters,
            fit_mode=audit.gradient_contract.fit_mode,
            nondifferentiable_outputs=("fold_indices", "valid", "status"),
            conditions=audit.gradient_contract.conditions
            + (
                "The recipe factory must be smooth in its vector argument.",
                "This contract applies only to the fixed-vector CV objective, not the DE-selected vector.",
            ),
        )
        objective = _DifferentiableCVObjective(
            batch,
            splits,
            recipe_factory,
            scorer,
            key=objective_key,
            primary_metric=self.primary_metric,
            maximize=maximize,
        )
        optimizer_result = _bounded_differential_evolution(
            objective,
            self.initial_vector,
            self.lower_bounds,
            self.upper_bounds,
            self.search,
            key=search_key,
        )
        best_recipe = recipe_factory(optimizer_result.best_vector)
        if not isinstance(best_recipe, AbstractRecipe):
            raise TypeError("recipe_factory(vector) must return an AbstractRecipe.")
        best_cv = _cross_validate_materialized(
            best_recipe, batch, splits, scorer, key=objective_key
        )
        best_fit = best_recipe.fit_batch(
            batch.take_samples(splits.sample_indices), key=refit_key
        )
        if not isinstance(best_fit, FitResult):
            raise TypeError(
                "Recipe.fit_batch must return a FitResult during final refit."
            )
        return DifferentiableSearchResult(
            optimizer_result,
            best_recipe,
            best_fit,
            best_cv,
            splits,
            key=key,
            refit_key=refit_key,
            objective_gradient_contract=objective_contract,
        )


class NestedFoldEvaluation(StrictModule):
    """Inner search and untouched outer-fold score for one nested fold."""

    split: Any
    search_result: Any
    outer_evaluation: FoldEvaluation

    def __init__(
        self, split: Any, search_result: Any, outer_evaluation: FoldEvaluation, /
    ):
        self.split = split
        self.search_result = search_result
        self.outer_evaluation = outer_evaluation


class NestedCrossValidationResult(StrictModule):
    """Nested selection results with every outer holdout scored exactly once."""

    folds: tuple[NestedFoldEvaluation, ...]
    outer_cross_validation: CrossValidationResult
    split_result: NestedSplitResult
    valid: Any
    status: Any
    key: Any
    gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(
        self,
        folds: tuple[NestedFoldEvaluation, ...],
        outer_cross_validation: CrossValidationResult,
        split_result: NestedSplitResult,
        /,
        *,
        key: Any,
    ):
        self.folds = tuple(folds)
        self.outer_cross_validation = outer_cross_validation
        self.split_result = split_result
        self.valid = outer_cross_validation.valid
        self.status = outer_cross_validation.status
        self.key = _require_key(key)
        self.gradient_contract = _EXACT_SEARCH_CONTRACT
        self.method = "nested_cross_validation"


def nested_cross_validate(
    search_plan: AbstractSearchPlan,
    recipe_factory: Callable[..., AbstractRecipe],
    batch: MLBatch,
    split_plan: NestedSplitPlan | NestedSplitResult,
    scorer: Any,
    /,
    *,
    key: Any,
) -> NestedCrossValidationResult:
    """Select only on inner folds, refit on outer training data, then score holdouts."""
    if not isinstance(search_plan, AbstractSearchPlan):
        raise TypeError("search_plan must be an AbstractSearchPlan.")
    key = _require_key(key)
    if isinstance(split_plan, NestedSplitPlan):
        nested = split_plan.split(batch, key=jr.fold_in(key, 0))
    elif isinstance(split_plan, NestedSplitResult):
        nested = split_plan
    else:
        raise TypeError("split_plan must be a NestedSplitPlan or NestedSplitResult.")
    outer_evaluations: list[FoldEvaluation] = []
    nested_evaluations: list[NestedFoldEvaluation] = []
    for position, nested_fold in enumerate(nested.folds):
        fold_key = jr.fold_in(key, position + 1)
        search_key, prediction_key = jr.split(fold_key)
        search_result = search_plan.run(
            recipe_factory,
            batch,
            nested_fold.inner_split,
            scorer,
            key=search_key,
        )
        validation_batch = batch.take_samples(nested_fold.outer_fold.validation_indices)
        predictions = _predict(
            search_result.best_fit, validation_batch, key=prediction_key
        )
        score = _call_scorer(scorer, predictions, validation_batch)
        outer_evaluation = FoldEvaluation(
            nested_fold.outer_fold,
            search_result.best_fit,
            score,
            predictions,
            fit_key=search_result.refit_key,
            prediction_key=prediction_key,
        )
        outer_evaluations.append(outer_evaluation)
        nested_evaluations.append(
            NestedFoldEvaluation(nested_fold, search_result, outer_evaluation)
        )
    outer_folds = tuple(outer_evaluations)
    outer_split = SplitPlanResult(
        tuple(fold.outer_fold for fold in nested.folds),
        sample_indices=jnp.arange(batch.sample_count, dtype=jnp.int32),
        key=nested.key,
        method="nested_outer",
    )
    outer_cv = CrossValidationResult(
        outer_folds,
        outer_split,
        _aggregate_scores(outer_folds),
        key=key,
        gradient_contract=_cross_validation_contract(outer_folds),
    )
    return NestedCrossValidationResult(
        tuple(nested_evaluations), outer_cv, nested, key=key
    )


__all__ = [
    "AbstractSearchPlan",
    "CandidateEvaluation",
    "CandidateSpec",
    "DifferentiableSearchAdapter",
    "DifferentiableSearchResult",
    "GridSearch",
    "HalvingRung",
    "NestedCrossValidationResult",
    "NestedFoldEvaluation",
    "ParameterGrid",
    "RandomSearch",
    "SELECTION_INVALID_CANDIDATE",
    "SELECTION_NO_VALID_CANDIDATE",
    "SELECTION_SUCCESS",
    "SELECTION_UNSUPPORTED_GRADIENT",
    "SearchResult",
    "SuccessiveHalvingSearch",
    "nested_cross_validate",
]
