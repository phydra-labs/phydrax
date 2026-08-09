#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._numerics import normalize_least_squares_design
from ..._strict import StrictModule
from ...data_utils import train_test_split_indices
from .._evolution import DiscreteEvolution
from .._layout import InputLayout
from .._system import AbstractInputPolicy
from ._sindy import _result_from_regression, SINDyResult
from ._sindy_design import SINDyDesign, SINDyDesignDiagnostics, SINDyProblem
from ._sparse_regression import AbstractSparseRegression, SparseRegressionResult


SelectionCriterion: TypeAlias = Literal[
    "equation", "one_step", "rollout", "combined", "bic"
]


def _variant_design(
    design: SINDyDesign,
    /,
    *,
    valid: Array,
    weight_multiplier: Array | None = None,
    feature_indices: Array | None = None,
    variant_id: str,
) -> SINDyDesign:
    resolved_valid = design.valid & jnp.asarray(valid, dtype=bool)
    multiplier = (
        jnp.ones_like(design.weights)
        if weight_multiplier is None
        else jnp.asarray(weight_multiplier, dtype=design.weights.dtype)
    )
    if multiplier.shape != design.weights.shape:
        raise ValueError("weight_multiplier must have one entry per design row.")
    weights = jnp.where(resolved_valid, design.weights * multiplier, 0.0)
    if feature_indices is None:
        matrix = design.matrix
        names = design.feature_names
    else:
        indices = jnp.asarray(feature_indices, dtype=jnp.int32)
        matrix = jnp.take(design.matrix, indices, axis=1)
        names = tuple(design.feature_names[int(index)] for index in np.asarray(indices))
    normalized = normalize_least_squares_design(
        matrix,
        mask=resolved_valid,
        weights=weights,
        max_features=int(matrix.shape[1]),
    )
    design_variant = {
        "valid": np.asarray(resolved_valid).tolist(),
        "features": names,
    }
    identifier = (
        f"{design.design_id}:{variant_id}:{canonical_fingerprint(design_variant)}"
    )
    return SINDyDesign(
        matrix=matrix,
        target=design.target,
        valid=resolved_valid,
        weights=weights,
        coordinates=design.coordinates,
        case_index=design.case_index,
        window_start=design.window_start,
        window_end=design.window_end,
        diagnostics=SINDyDesignDiagnostics(
            singular_values=normalized.singular_values,
            sample_count=normalized.sample_count,
            rank=normalized.rank,
            condition_number=normalized.condition_number,
            weight_sum=normalized.weight_sum,
        ),
        state_layout=design.state_layout,
        input_layout=design.input_layout,
        feature_names=names,
        output_names=design.output_names,
        formulation=design.formulation,
        source_id=design.source_id,
        coordinate_id=design.coordinate_id,
        library_id=design.library_id,
        formulation_id=design.formulation_id,
        design_id=identifier,
    )


def _split_rows(
    problem: SINDyProblem,
    design: SINDyDesign,
    /,
    *,
    validation_fraction: float,
    key: Array,
    embargo: int | None,
) -> tuple[Array, Array, int]:
    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1).")
    if problem.data.num_cases > 1:
        train_cases, validation_cases = train_test_split_indices(
            problem.data.num_cases,
            test_fraction=fraction,
            key=key,
            shuffle=True,
        )
        train = jnp.any(design.case_index[:, None] == train_cases[None, :], axis=-1)
        validation = jnp.any(
            design.case_index[:, None] == validation_cases[None, :], axis=-1
        )
        return train, validation, 0
    split = int(np.floor((1.0 - fraction) * problem.data.capacity))
    split = min(max(split, 1), problem.data.capacity - 1)
    maximum_span = int(
        np.max(
            np.asarray(design.window_end - design.window_start),
            initial=0,
        )
    )
    resolved_embargo = max(1, maximum_span) if embargo is None else int(embargo)
    if resolved_embargo < maximum_span:
        raise ValueError(
            "embargo must be at least the widest derivative or equation window."
        )
    train = design.window_end < (split - resolved_embargo)
    validation = design.window_start >= split
    if not bool(jnp.any(design.valid & train)) or not bool(
        jnp.any(design.valid & validation)
    ):
        raise ValueError(
            "Blocked split leaves no valid training or validation equations."
        )
    return train, validation, resolved_embargo


class _ObservedInputPolicy(AbstractInputPolicy):
    coordinates: Array
    values: Array
    input_layout: InputLayout
    alignment: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def evaluate(self, coordinate, state, args=None, /) -> Array:
        del state, args
        query = jnp.asarray(coordinate)
        index = jnp.clip(
            jnp.searchsorted(self.coordinates, query, side="right") - 1,
            0,
            self.values.shape[0] - 1,
        )
        if self.alignment == "transitions" or self.values.shape[0] == 1:
            return self.values[index]
        upper = jnp.minimum(index + 1, self.values.shape[0] - 1)
        left_time = self.coordinates[index]
        right_time = self.coordinates[upper]
        denominator = jnp.where(right_time > left_time, right_time - left_time, 1.0)
        fraction = (query - left_time) / denominator
        return (1.0 - fraction) * self.values[index] + fraction * self.values[upper]


def _case_input_policy(problem: SINDyProblem, case: int, /):
    data = problem.data
    if data.inputs is None:
        return None
    coordinates = data.coordinates.reshape((data.num_cases, data.capacity))[case]
    count = data.capacity if data.input_alignment == "samples" else data.capacity - 1
    values = data.inputs.reshape((data.num_cases, count) + data.input_layout.shape)[case]
    policy_coordinates = (
        coordinates if data.input_alignment == "samples" else coordinates[:-1]
    )
    return _ObservedInputPolicy(
        coordinates=policy_coordinates,
        values=values,
        input_layout=data.input_layout,
        alignment=data.input_alignment,
        policy_id=f"observed-inputs:{data.dataset_id}:case={case}",
    )


def _rollout_error(
    result: SINDyResult,
    problem: SINDyProblem,
    row_mask: Array,
    /,
    *,
    horizon: int,
    max_rollouts: int,
) -> Array:
    if not bool(result.valid):
        return jnp.asarray(jnp.inf, dtype=result.coefficients.dtype)
    data = problem.data
    state_size = data.state_layout.size
    flat_states = data.states.reshape((data.num_cases, data.capacity, state_size))
    flat_coordinates = data.coordinates.reshape((data.num_cases, data.capacity))
    flat_samples = data.sample_valid.reshape((data.num_cases, data.capacity))
    flat_transitions = data.transition_valid.reshape((data.num_cases, data.capacity - 1))
    candidate_rows = np.flatnonzero(np.asarray(result.design.valid & row_mask))
    pairs = []
    seen = set()
    starts = np.asarray(result.design.window_start)
    cases = np.asarray(result.design.case_index)
    for row in candidate_rows:
        pair = (int(cases[row]), int(starts[row]))
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
        if len(pairs) >= max_rollouts:
            break
    errors = []
    weights = []
    system = result.to_system()
    for case, start in pairs:
        end = min(start + horizon, data.capacity - 1)
        if end <= start:
            continue
        if not bool(np.all(np.asarray(flat_transitions[case, start:end]))) or not bool(
            flat_samples[case, end]
        ):
            continue
        if data.input_valid is not None:
            flat_input_valid = data.input_valid.reshape(
                (data.num_cases, data.input_valid.shape[-1])
            )
            input_stop = end + 1 if data.input_alignment == "samples" else end
            if not bool(np.all(np.asarray(flat_input_valid[case, start:input_stop]))):
                continue
        policy = _case_input_policy(problem, case)
        current = flat_states[case, start].reshape(data.state_layout.shape)
        if result.formulation == "discrete":
            evolution = DiscreteEvolution(system, input_policy=policy)
            valid = jnp.asarray(True)
            for step in range(start, end):
                advanced = evolution.advance(
                    current,
                    flat_coordinates[case, step],
                    flat_coordinates[case, step + 1],
                    None,
                )
                current = advanced.final_state
                valid = valid & advanced.valid
        else:
            from ...solver import DiffraxEvolution

            evolution = DiffraxEvolution(system, input_policy=policy)
            advanced = evolution.advance(
                current,
                flat_coordinates[case, start],
                flat_coordinates[case, end],
                None,
            )
            current = advanced.final_state
            valid = advanced.valid
        target = flat_states[case, end].reshape(data.state_layout.shape)
        errors.append(jnp.where(valid, jnp.mean(jnp.abs(current - target) ** 2), jnp.inf))
        weights.append(data.weights.reshape((data.num_cases, data.capacity))[case, end])
    if not errors:
        return jnp.asarray(jnp.inf)
    error_values = jnp.stack(tuple(errors))
    weight_values = jnp.stack(tuple(weights))
    return jnp.sum(weight_values * error_values) / jnp.maximum(
        jnp.sum(weight_values), 1.0
    )


def _equation_error(result: SINDyResult, mask: Array, /) -> Array:
    valid = result.design.valid & mask
    residual = result.design.target - result.predict_design()
    squared = jnp.mean(jnp.abs(residual) ** 2, axis=-1)
    weights = jnp.where(valid, result.design.weights, 0.0)
    return jnp.sum(weights * squared) / jnp.maximum(jnp.sum(weights), 1.0)


class SINDySelectionPolicy(StrictModule):
    """Leakage-safe split, scoring, and complexity policy for candidate selection."""

    criterion: SelectionCriterion = eqx.field(static=True)
    validation_fraction: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    embargo: int | None = eqx.field(static=True)
    rollout_horizon: int = eqx.field(static=True)
    max_rollouts: int = eqx.field(static=True)
    complexity_weight: float = eqx.field(static=True)
    combined_weights: tuple[float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        *,
        criterion: SelectionCriterion = "combined",
        validation_fraction: float = 0.2,
        seed: int = 0,
        embargo: int | None = None,
        rollout_horizon: int = 4,
        max_rollouts: int = 32,
        complexity_weight: float = 0.0,
        combined_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        if criterion not in ("equation", "one_step", "rollout", "combined", "bic"):
            raise ValueError("Unsupported selection criterion.")
        if not 0.0 < float(validation_fraction) < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1).")
        if int(rollout_horizon) < 1 or int(max_rollouts) < 1:
            raise ValueError("rollout_horizon and max_rollouts must be positive.")
        if embargo is not None and int(embargo) < 0:
            raise ValueError("embargo must be nonnegative or None.")
        complexity = float(complexity_weight)
        weights = tuple(float(value) for value in combined_weights)
        if not np.isfinite(complexity) or complexity < 0.0:
            raise ValueError("complexity_weight must be finite and nonnegative.")
        if len(weights) != 3 or any(
            not np.isfinite(value) or value < 0.0 for value in weights
        ):
            raise ValueError(
                "combined_weights must contain three finite nonnegative values."
            )
        self.criterion = criterion
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)
        self.embargo = None if embargo is None else int(embargo)
        self.rollout_horizon = int(rollout_horizon)
        self.max_rollouts = int(max_rollouts)
        self.complexity_weight = complexity
        self.combined_weights = weights


class SINDySelectionResult(StrictModule):
    """All candidates, split evidence, metrics, and the declared winning index."""

    candidates: tuple[SINDyResult, ...]
    equation_error: Array
    one_step_error: Array
    rollout_error: Array
    complexity: Array
    score: Array
    candidate_valid: Array
    selected_index: Array
    train_mask: Array
    validation_mask: Array
    policy: SINDySelectionPolicy
    embargo: int = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.selected_index >= 0

    @property
    def selected(self) -> SINDyResult:
        index = int(self.selected_index)
        if index < 0:
            raise ValueError("No valid SINDy candidate was selected.")
        return self.candidates[index]


def select_sindy_model(
    problem: SINDyProblem,
    regressors: Sequence[AbstractSparseRegression],
    /,
    *,
    policy: SINDySelectionPolicy | None = None,
) -> SINDySelectionResult:
    """Fit and score every candidate; invalid candidates remain in the result."""
    if not isinstance(problem, SINDyProblem):
        raise TypeError("problem must be a SINDyProblem.")
    candidates_policies = tuple(regressors)
    if not candidates_policies or any(
        not isinstance(regressor, AbstractSparseRegression)
        for regressor in candidates_policies
    ):
        raise TypeError("regressors must contain AbstractSparseRegression instances.")
    resolved_policy = SINDySelectionPolicy() if policy is None else policy
    if not isinstance(resolved_policy, SINDySelectionPolicy):
        raise TypeError("policy must be a SINDySelectionPolicy or None.")
    design = problem.build_design()
    train_mask, validation_mask, embargo = _split_rows(
        problem,
        design,
        validation_fraction=resolved_policy.validation_fraction,
        key=jax.random.key(resolved_policy.seed),
        embargo=resolved_policy.embargo,
    )
    training = _variant_design(
        design,
        valid=train_mask,
        variant_id="training",
    )
    candidates = []
    equation_errors = []
    one_step_errors = []
    rollout_errors = []
    complexities = []
    scores = []
    candidate_valid = []
    needs_one_step = resolved_policy.criterion in ("one_step", "combined")
    needs_rollout = resolved_policy.criterion in ("rollout", "combined")
    not_computed = jnp.asarray(jnp.nan, dtype=design.matrix.dtype)
    for regressor in candidates_policies:
        regression = regressor.fit(training)
        candidate = _result_from_regression(problem, design, regression)
        equation = _equation_error(candidate, validation_mask)
        one_step = (
            _rollout_error(
                candidate,
                problem,
                validation_mask,
                horizon=1,
                max_rollouts=resolved_policy.max_rollouts,
            )
            if needs_one_step
            else not_computed
        )
        rollout = (
            _rollout_error(
                candidate,
                problem,
                validation_mask,
                horizon=resolved_policy.rollout_horizon,
                max_rollouts=resolved_policy.max_rollouts,
            )
            if needs_rollout
            else not_computed
        )
        complexity = jnp.sum(candidate.support).astype(design.matrix.dtype)
        if resolved_policy.criterion == "equation":
            base_score = equation
        elif resolved_policy.criterion == "one_step":
            base_score = one_step
        elif resolved_policy.criterion == "rollout":
            base_score = rollout
        elif resolved_policy.criterion == "bic":
            count = jnp.maximum(jnp.sum(design.valid & validation_mask), 1)
            base_score = count * jnp.log(
                jnp.maximum(equation, jnp.finfo(equation.dtype).tiny)
            ) + complexity * jnp.log(count)
        else:
            base_score = (
                resolved_policy.combined_weights[0] * equation
                + resolved_policy.combined_weights[1] * one_step
                + resolved_policy.combined_weights[2] * rollout
            )
        valid_candidate = candidate.valid & jnp.isfinite(base_score)
        score = jnp.where(
            valid_candidate,
            base_score + resolved_policy.complexity_weight * complexity,
            jnp.inf,
        )
        candidates.append(candidate)
        equation_errors.append(equation)
        one_step_errors.append(one_step)
        rollout_errors.append(rollout)
        complexities.append(complexity)
        scores.append(score)
        candidate_valid.append(valid_candidate)
    score_values = jnp.stack(tuple(scores))
    valid_values = jnp.stack(tuple(candidate_valid))
    selected = jnp.where(jnp.any(valid_values), jnp.argmin(score_values), -1).astype(
        jnp.int32
    )
    return SINDySelectionResult(
        candidates=tuple(candidates),
        equation_error=jnp.stack(tuple(equation_errors)),
        one_step_error=jnp.stack(tuple(one_step_errors)),
        rollout_error=jnp.stack(tuple(rollout_errors)),
        complexity=jnp.stack(tuple(complexities)),
        score=score_values,
        candidate_valid=valid_values,
        selected_index=selected,
        train_mask=train_mask,
        validation_mask=validation_mask,
        policy=resolved_policy,
        embargo=embargo,
    )


class EnsembleSINDyResult(StrictModule):
    """Bootstrap coefficient samples, validity, and inclusion/selection frequencies."""

    coefficients: Array
    support: Array
    feature_included: Array
    member_valid: Array
    coefficient_mean: Array
    coefficient_lower: Array
    coefficient_upper: Array
    inclusion_frequency: Array
    selection_frequency: Array
    regressions: tuple[SparseRegressionResult, ...]
    quantiles: tuple[float, float] = eqx.field(static=True)
    design_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return jnp.any(self.member_valid)


def fit_ensemble_sindy(
    problem: SINDyProblem,
    regressor: AbstractSparseRegression,
    /,
    *,
    num_members: int = 32,
    sample_fraction: float = 0.8,
    feature_fraction: float = 1.0,
    seed: int = 0,
    quantiles: Sequence[float] = (0.05, 0.95),
) -> EnsembleSINDyResult:
    """Bootstrap complete cases, or blocked intervals for one trajectory."""
    if not isinstance(problem, SINDyProblem):
        raise TypeError("problem must be a SINDyProblem.")
    if not isinstance(regressor, AbstractSparseRegression):
        raise TypeError("regressor must be an AbstractSparseRegression.")
    members = int(num_members)
    sample_rate = float(sample_fraction)
    feature_rate = float(feature_fraction)
    interval = tuple(float(value) for value in quantiles)
    if members < 1:
        raise ValueError("num_members must be positive.")
    if not 0.0 < sample_rate <= 1.0 or not 0.0 < feature_rate <= 1.0:
        raise ValueError("sample_fraction and feature_fraction must lie in (0, 1].")
    if len(interval) != 2 or not 0.0 <= interval[0] < interval[1] <= 1.0:
        raise ValueError("quantiles must contain two ordered probabilities in [0, 1].")
    design = problem.build_design()
    feature_count = max(1, int(np.ceil(feature_rate * design.num_features)))
    root_key = jax.random.key(int(seed))
    coefficient_samples = []
    support_samples = []
    inclusion_samples = []
    valid_samples = []
    regressions = []
    for member in range(members):
        member_key = jax.random.fold_in(root_key, member)
        sample_key, feature_key = jax.random.split(member_key)
        if problem.data.num_cases > 1:
            sample_count = max(
                1,
                int(np.ceil(sample_rate * problem.data.num_cases)),
            )
            sampled_cases = jax.random.randint(
                sample_key,
                (sample_count,),
                0,
                problem.data.num_cases,
            )
            counts = jnp.bincount(sampled_cases, length=problem.data.num_cases)
            row_multiplier = counts[design.case_index]
            row_valid = row_multiplier > 0
        else:
            block_size = max(2, int(np.ceil(sample_rate * problem.data.capacity)))
            block_size = min(block_size, problem.data.capacity)
            max_start = problem.data.capacity - block_size
            start = int(jax.random.randint(sample_key, (), 0, max_start + 1))
            end = start + block_size - 1
            row_valid = (design.window_start >= start) & (design.window_end <= end)
            row_multiplier = jnp.ones_like(design.weights)
        feature_indices = jnp.sort(
            jax.random.permutation(feature_key, design.num_features)[:feature_count]
        )
        member_design = _variant_design(
            design,
            valid=row_valid,
            weight_multiplier=row_multiplier,
            feature_indices=feature_indices,
            variant_id=f"ensemble-member={member}",
        )
        regression = regressor.fit(member_design)
        expanded_coefficients = (
            jnp.zeros(
                (design.output_size, design.num_features),
                dtype=regression.coefficients.dtype,
            )
            .at[:, feature_indices]
            .set(regression.coefficients)
        )
        expanded_support = (
            jnp.zeros((design.output_size, design.num_features), dtype=bool)
            .at[:, feature_indices]
            .set(regression.support)
        )
        included = (
            jnp.zeros((design.num_features,), dtype=bool).at[feature_indices].set(True)
        )
        coefficient_samples.append(expanded_coefficients)
        support_samples.append(expanded_support)
        inclusion_samples.append(included)
        valid_samples.append(regression.successful)
        regressions.append(regression)
    coefficients = jnp.stack(tuple(coefficient_samples))
    support = jnp.stack(tuple(support_samples))
    included = jnp.stack(tuple(inclusion_samples))
    valid = jnp.stack(tuple(valid_samples))
    member_weight = valid.astype(coefficients.dtype)
    denominator = jnp.maximum(jnp.sum(member_weight), 1.0)
    mean = jnp.sum(member_weight[:, None, None] * coefficients, axis=0) / denominator
    masked_coefficients = jnp.where(valid[:, None, None], coefficients, jnp.nan)
    lower = jnp.nanquantile(masked_coefficients, interval[0], axis=0)
    upper = jnp.nanquantile(masked_coefficients, interval[1], axis=0)
    inclusion_frequency = jnp.mean(included.astype(coefficients.dtype), axis=0)
    eligible = included[:, None, :] & valid[:, None, None]
    selection_frequency = jnp.sum(support & eligible, axis=0) / jnp.maximum(
        jnp.sum(eligible, axis=0), 1
    )
    return EnsembleSINDyResult(
        coefficients=coefficients,
        support=support,
        feature_included=included,
        member_valid=valid,
        coefficient_mean=mean,
        coefficient_lower=lower,
        coefficient_upper=upper,
        inclusion_frequency=inclusion_frequency,
        selection_frequency=selection_frequency,
        regressions=tuple(regressions),
        quantiles=interval,
        design_id=design.design_id,
    )


__all__ = [
    "EnsembleSINDyResult",
    "SINDySelectionPolicy",
    "SINDySelectionResult",
    "SelectionCriterion",
    "fit_ensemble_sindy",
    "select_sindy_model",
]
