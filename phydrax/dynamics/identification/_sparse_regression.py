#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._numerics import (
    normalize_least_squares_design,
    solve_normalized_least_squares,
)
from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    RankPolicy,
    solve,
)
from ._sindy_design import SINDyDesign
from ._status import (
    IDENTIFICATION_INSUFFICIENT_SAMPLES,
    IDENTIFICATION_NOT_CONVERGED,
    IDENTIFICATION_RANK_DEFICIENT,
    IDENTIFICATION_SUCCESS,
)


ThresholdSpace: TypeAlias = Literal["normalized", "physical"]


class SparseRegressionHistory(StrictModule):
    """Fixed-length support, coefficient, rank, and residual iteration history."""

    coefficients: Array
    support: Array
    residual_norm: Array
    rank: Array
    condition_number: Array
    active_count: Array


class SparseRegressionResult(StrictModule):
    """Sparse coefficients in physical and normalized feature coordinates."""

    coefficients: Array
    normalized_coefficients: Array
    support: Array
    feature_scale: Array
    target_scale: Array
    residual: Array
    residual_norm: Array
    rank: Array
    condition_number: Array
    iterations: Array
    converged: Array
    valid: Array
    status: Array
    history: SparseRegressionHistory
    solver_diagnostics: Any
    feature_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    design_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)


class AbstractSparseRegression(StrictModule):
    """Sparse regression policy over a prebuilt reusable SINDy design."""

    @abc.abstractmethod
    def fit(self, design: SINDyDesign, /) -> SparseRegressionResult:
        raise NotImplementedError


def _thresholds(value: float | Sequence[float], output_size: int, /) -> tuple[float, ...]:
    if isinstance(value, Sequence):
        resolved = tuple(float(item) for item in value)
    else:
        resolved = (float(value),) * output_size
    if len(resolved) != output_size or any(
        not np.isfinite(item) or item < 0.0 for item in resolved
    ):
        raise ValueError("threshold must be one nonnegative finite scalar per output.")
    return resolved


def _target_scales(design: SINDyDesign, enabled: bool, /) -> Array:
    if not enabled:
        return jnp.ones((design.output_size,), dtype=design.target.dtype)
    denominator = jnp.maximum(jnp.sum(design.weights), 1.0)
    second_moment = (
        jnp.sum(design.weights[:, None] * jnp.abs(design.target) ** 2, axis=0)
        / denominator
    )
    scale = jnp.sqrt(second_moment)
    tolerance = jnp.finfo(scale.dtype).eps
    return jnp.where(scale > tolerance, scale, 1.0)


def _solve_outputs(normalized, target, support, ridge: float, /):
    coefficients = []
    ranks = []
    conditions = []
    valids = []
    for output in range(target.shape[1]):
        result = solve_normalized_least_squares(
            normalized,
            target[:, output],
            mask=normalized.valid_rows,
            ridge=ridge,
            feature_mask=support[output],
            min_samples=1,
        )
        coefficients.append(result.coefficients)
        ranks.append(result.rank)
        conditions.append(result.condition_number)
        valids.append(result.valid)
    return (
        jnp.stack(tuple(coefficients), axis=0),
        jnp.stack(tuple(ranks)),
        jnp.stack(tuple(conditions)),
        jnp.stack(tuple(valids)),
    )


class SequentialThresholdedLeastSquares(AbstractSparseRegression):
    """Monotone-support STLSQ with optional scaling and an explicit final refit."""

    thresholds: float | tuple[float, ...] = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)
    scale_features: bool = eqx.field(static=True)
    scale_targets: bool = eqx.field(static=True)
    threshold_space: ThresholdSpace = eqx.field(static=True)
    unbiased_refit: bool = eqx.field(static=True)
    zero_tolerance: float | None = eqx.field(static=True)

    def __init__(
        self,
        threshold: float | Sequence[float],
        /,
        *,
        ridge: float = 0.0,
        max_iterations: int = 20,
        rcond: float | None = None,
        scale_features: bool = True,
        scale_targets: bool = False,
        threshold_space: ThresholdSpace = "normalized",
        unbiased_refit: bool = True,
        zero_tolerance: float | None = None,
    ):
        ridge_value = float(ridge)
        iterations = int(max_iterations)
        if not np.isfinite(ridge_value) or ridge_value < 0.0:
            raise ValueError("ridge must be finite and nonnegative.")
        if iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
            raise ValueError("rcond must be finite and nonnegative or None.")
        if threshold_space not in ("normalized", "physical"):
            raise ValueError("threshold_space must be 'normalized' or 'physical'.")
        resolved_zero = None if zero_tolerance is None else float(zero_tolerance)
        if resolved_zero is not None and (
            not np.isfinite(resolved_zero) or resolved_zero < 0.0
        ):
            raise ValueError("zero_tolerance must be finite and nonnegative or None.")
        self.thresholds = (
            tuple(float(item) for item in threshold)
            if isinstance(threshold, Sequence)
            else float(threshold)
        )
        self.ridge = ridge_value
        self.max_iterations = iterations
        self.rcond = None if rcond is None else float(rcond)
        self.scale_features = bool(scale_features)
        self.scale_targets = bool(scale_targets)
        self.threshold_space = threshold_space
        self.unbiased_refit = bool(unbiased_refit)
        self.zero_tolerance = resolved_zero

    def fit(self, design: SINDyDesign, /) -> SparseRegressionResult:
        if not isinstance(design, SINDyDesign):
            raise TypeError("design must be a SINDyDesign.")
        thresholds = jnp.asarray(
            _thresholds(self.thresholds, design.output_size),
            dtype=design.matrix.dtype,
        )
        normalized = normalize_least_squares_design(
            design.matrix,
            mask=design.valid,
            weights=design.weights,
            scale=self.scale_features,
            rcond=self.rcond,
            max_features=design.num_features,
        )
        target_scale = _target_scales(design, self.scale_targets)
        normalized_target = design.target / target_scale[None, :]
        support = jnp.ones((design.output_size, design.num_features), dtype=bool)
        coefficients, ranks, conditions, solve_valid = _solve_outputs(
            normalized, normalized_target, support, self.ridge
        )
        converged = jnp.zeros((design.output_size,), dtype=bool)
        iteration_count = jnp.zeros((design.output_size,), dtype=jnp.int32)
        coefficient_history = [coefficients]
        support_history = [support]
        rank_history = [ranks]
        condition_history = [conditions]
        residual_history = []

        def physical(normalized_coefficients):
            return (
                target_scale[:, None]
                * normalized_coefficients
                / normalized.scale[None, :]
            )

        initial_physical = physical(coefficients)
        initial_residual = design.target - design.matrix @ initial_physical.T
        residual_history.append(
            jnp.sqrt(
                jnp.sum(
                    design.weights[:, None] * jnp.abs(initial_residual) ** 2,
                    axis=0,
                )
            )
        )

        for _ in range(self.max_iterations):
            threshold_coefficients = (
                coefficients
                if self.threshold_space == "normalized"
                else physical(coefficients)
            )
            candidate_support = support & (
                jnp.abs(threshold_coefficients) > thresholds[:, None]
            )
            candidate_support = jnp.where(converged[:, None], support, candidate_support)
            unchanged = jnp.all(candidate_support == support, axis=-1)
            iteration_count = iteration_count + (~converged).astype(jnp.int32)
            support = candidate_support
            coefficients, ranks, conditions, solve_valid = _solve_outputs(
                normalized,
                normalized_target,
                support,
                self.ridge,
            )
            converged = converged | unchanged
            physical_coefficients = physical(coefficients)
            residual = design.target - design.matrix @ physical_coefficients.T
            residual_norm = jnp.sqrt(
                jnp.sum(
                    design.weights[:, None] * jnp.abs(residual) ** 2,
                    axis=0,
                )
            )
            coefficient_history.append(coefficients)
            support_history.append(support)
            rank_history.append(ranks)
            condition_history.append(conditions)
            residual_history.append(residual_norm)

        final_ridge = 0.0 if self.unbiased_refit else self.ridge
        coefficients, ranks, conditions, solve_valid = _solve_outputs(
            normalized,
            normalized_target,
            support,
            final_ridge,
        )
        physical_coefficients = physical(coefficients)
        residual = design.target - design.matrix @ physical_coefficients.T
        residual = jnp.where(design.valid[:, None], residual, 0.0)
        residual_norm = jnp.sqrt(
            jnp.sum(
                design.weights[:, None] * jnp.abs(residual) ** 2,
                axis=0,
            )
        )
        active_count = jnp.sum(support, axis=-1).astype(jnp.int32)
        denominator = jnp.maximum(jnp.sum(design.weights), 1.0)
        zero_error = jnp.sqrt(
            jnp.sum(
                design.weights[:, None] * jnp.abs(design.target) ** 2,
                axis=0,
            )
            / denominator
        )
        empty_acceptable = (
            jnp.zeros_like(converged)
            if self.zero_tolerance is None
            else zero_error <= self.zero_tolerance
        )
        empty = active_count == 0
        output_valid = (
            converged
            & jnp.where(empty, empty_acceptable, solve_valid)
            & jnp.all(jnp.isfinite(physical_coefficients), axis=-1)
        )
        insufficient = ~empty & ~solve_valid & (normalized.sample_count < active_count)
        status = jnp.where(
            ~converged,
            IDENTIFICATION_NOT_CONVERGED,
            jnp.where(
                insufficient,
                IDENTIFICATION_INSUFFICIENT_SAMPLES,
                jnp.where(
                    empty & ~empty_acceptable,
                    IDENTIFICATION_RANK_DEFICIENT,
                    jnp.where(
                        output_valid,
                        IDENTIFICATION_SUCCESS,
                        IDENTIFICATION_RANK_DEFICIENT,
                    ),
                ),
            ),
        ).astype(jnp.int32)
        history = SparseRegressionHistory(
            coefficients=jnp.stack(tuple(coefficient_history), axis=0),
            support=jnp.stack(tuple(support_history), axis=0),
            residual_norm=jnp.stack(tuple(residual_history), axis=0),
            rank=jnp.stack(tuple(rank_history), axis=0),
            condition_number=jnp.stack(tuple(condition_history), axis=0),
            active_count=jnp.sum(
                jnp.stack(tuple(support_history), axis=0), axis=-1
            ).astype(jnp.int32),
        )
        return SparseRegressionResult(
            coefficients=physical_coefficients,
            normalized_coefficients=coefficients,
            support=support,
            feature_scale=normalized.scale,
            target_scale=target_scale,
            residual=residual,
            residual_norm=residual_norm,
            rank=ranks,
            condition_number=conditions,
            iterations=iteration_count,
            converged=converged,
            valid=output_valid,
            status=status,
            history=history,
            solver_diagnostics=None,
            feature_names=design.feature_names,
            output_names=design.output_names,
            method_id=(
                f"stlsq:threshold-space={self.threshold_space}:"
                f"ridge={self.ridge:g}:unbiased={self.unbiased_refit}"
            ),
            design_id=design.design_id,
        )


class DenseBlockRidgeRegression(AbstractSparseRegression):
    """Dense block-ridge solve through one augmented native least-squares problem."""

    block_sizes: tuple[int, ...] = eqx.field(static=True)
    regularization: tuple[float, ...] = eqx.field(static=True)
    scale_features: bool = eqx.field(static=True)
    scale_targets: bool = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        block_sizes: Sequence[int],
        regularization: Sequence[float],
        /,
        *,
        scale_features: bool = True,
        scale_targets: bool = False,
        rcond: float | None = None,
    ):
        sizes = tuple(int(size) for size in block_sizes)
        penalties = tuple(float(value) for value in regularization)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("block_sizes must contain positive block widths.")
        if len(penalties) != len(sizes) or any(
            not np.isfinite(value) or value < 0.0 for value in penalties
        ):
            raise ValueError(
                "regularization must contain one finite nonnegative value per block."
            )
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
            raise ValueError("rcond must be finite and nonnegative or None.")
        self.block_sizes = sizes
        self.regularization = penalties
        self.scale_features = bool(scale_features)
        self.scale_targets = bool(scale_targets)
        self.rcond = None if rcond is None else float(rcond)
        self.method_id = canonical_fingerprint(
            {
                "kind": "dense-block-ridge",
                "block_sizes": list(sizes),
                "regularization": list(penalties),
                "scale_features": self.scale_features,
                "scale_targets": self.scale_targets,
                "rcond": self.rcond,
            }
        )

    def fit(self, design: SINDyDesign, /) -> SparseRegressionResult:
        if not isinstance(design, SINDyDesign):
            raise TypeError("design must be a SINDyDesign.")
        if sum(self.block_sizes) != design.num_features:
            raise ValueError("block_sizes must exactly partition the SINDy features.")
        normalized = normalize_least_squares_design(
            design.matrix,
            mask=design.valid,
            weights=design.weights,
            scale=self.scale_features,
            rcond=self.rcond,
            max_features=design.num_features,
        )
        target_scale = _target_scales(design, self.scale_targets)
        target = design.target / target_scale[None, :]
        denominator = jnp.maximum(normalized.weight_sum, 1.0)
        root_weight = jnp.sqrt(normalized.weights / denominator)
        matrix = root_weight[:, None] * jnp.where(
            normalized.valid_rows[:, None],
            normalized.values,
            0.0,
        )
        response = root_weight[:, None] * jnp.where(
            normalized.valid_rows[:, None],
            target,
            0.0,
        )
        penalties = jnp.concatenate(
            tuple(
                jnp.full((size,), value, dtype=matrix.real.dtype)
                for size, value in zip(
                    self.block_sizes,
                    self.regularization,
                    strict=True,
                )
            )
        )
        augmented_matrix = jnp.concatenate(
            (matrix, jnp.diag(jnp.sqrt(penalties)).astype(matrix.dtype)),
            axis=0,
        )
        augmented_response = jnp.concatenate(
            (
                response,
                jnp.zeros(
                    (design.num_features, design.output_size),
                    dtype=response.dtype,
                ),
            ),
            axis=0,
        )
        cutoff = (
            max(augmented_matrix.shape) * jnp.finfo(augmented_matrix.real.dtype).eps
            if self.rcond is None
            else self.rcond
        )
        linear_result = solve(
            LeastSquaresProblem(DenseLinearOperator(augmented_matrix)),
            augmented_response,
            policy=LinearSolvePolicy(
                DenseSVD(),
                rank=RankPolicy(relative_cutoff=float(cutoff)),
            ),
        )
        normalized_coefficients = jnp.swapaxes(linear_result.value, -1, -2)
        physical_coefficients = (
            target_scale[:, None] * normalized_coefficients / normalized.scale[None, :]
        )
        residual = design.target - design.matrix @ physical_coefficients.T
        residual = jnp.where(design.valid[:, None], residual, 0.0)
        residual_norm = jnp.sqrt(
            jnp.sum(
                design.weights[:, None] * jnp.abs(residual) ** 2,
                axis=0,
            )
        )
        support = jnp.ones(
            (design.output_size, design.num_features),
            dtype=bool,
        )
        rank_scalar = jnp.asarray(linear_result.diagnostics.rank).reshape(-1)[0]
        condition_scalar = jnp.asarray(
            linear_result.diagnostics.condition_estimate
        ).reshape(-1)[0]
        ranks = jnp.full((design.output_size,), rank_scalar, dtype=jnp.int32)
        conditions = jnp.full(
            (design.output_size,),
            condition_scalar,
            dtype=residual_norm.dtype,
        )
        finite = jnp.all(jnp.isfinite(physical_coefficients), axis=-1) & jnp.isfinite(
            residual_norm
        )
        enough = normalized.sample_count >= design.num_features
        valid = finite & linear_result.successful & (enough | jnp.any(penalties > 0.0))
        status = jnp.where(
            valid,
            IDENTIFICATION_SUCCESS,
            jnp.where(
                ~enough & ~jnp.any(penalties > 0.0),
                IDENTIFICATION_INSUFFICIENT_SAMPLES,
                IDENTIFICATION_RANK_DEFICIENT,
            ),
        ).astype(jnp.int32)
        history = SparseRegressionHistory(
            coefficients=normalized_coefficients[None, ...],
            support=support[None, ...],
            residual_norm=residual_norm[None, ...],
            rank=ranks[None, ...],
            condition_number=conditions[None, ...],
            active_count=jnp.full(
                (1, design.output_size),
                design.num_features,
                dtype=jnp.int32,
            ),
        )
        return SparseRegressionResult(
            coefficients=physical_coefficients,
            normalized_coefficients=normalized_coefficients,
            support=support,
            feature_scale=normalized.scale,
            target_scale=target_scale,
            residual=residual,
            residual_norm=residual_norm,
            rank=ranks,
            condition_number=conditions,
            iterations=jnp.ones((design.output_size,), dtype=jnp.int32),
            converged=valid,
            valid=valid,
            status=status,
            history=history,
            solver_diagnostics=linear_result.diagnostics,
            feature_names=design.feature_names,
            output_names=design.output_names,
            method_id=self.method_id,
            design_id=design.design_id,
        )


__all__ = [
    "DenseBlockRidgeRegression",
    "AbstractSparseRegression",
    "SequentialThresholdedLeastSquares",
    "SparseRegressionHistory",
    "SparseRegressionResult",
    "ThresholdSpace",
]
