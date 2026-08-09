#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._numerics import normalize_least_squares_design
from ..._strict import StrictModule
from ._sindy_design import SINDyDesign
from ._sparse_regression import (
    _solve_outputs,
    _target_scales,
    AbstractSparseRegression,
    SparseRegressionHistory,
    SparseRegressionResult,
)
from ._status import (
    IDENTIFICATION_NONFINITE,
    IDENTIFICATION_NOT_CONVERGED,
    IDENTIFICATION_RANK_DEFICIENT,
    IDENTIFICATION_SUCCESS,
)


SR3Penalty: TypeAlias = Literal["l0", "l1"]


class SR3Diagnostics(StrictModule):
    """Relaxed objective and coupling-distance history."""

    objective: Array
    relaxation_error: Array
    relaxation_strength: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    penalty: SR3Penalty = eqx.field(static=True)


class SR3Regression(AbstractSparseRegression):
    """Sparse relaxed regularized regression with explicit L0 or L1 proximal policy."""

    regularization: float = eqx.field(static=True)
    relaxation_strength: float = eqx.field(static=True)
    penalty: SR3Penalty = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)
    scale_features: bool = eqx.field(static=True)
    scale_targets: bool = eqx.field(static=True)
    unbiased_refit: bool = eqx.field(static=True)
    zero_tolerance: float | None = eqx.field(static=True)
    max_features: int = eqx.field(static=True)

    def __init__(
        self,
        regularization: float,
        /,
        *,
        relaxation_strength: float = 1.0,
        penalty: SR3Penalty = "l0",
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        rcond: float | None = None,
        scale_features: bool = True,
        scale_targets: bool = False,
        unbiased_refit: bool = False,
        zero_tolerance: float | None = None,
        max_features: int = 4096,
    ):
        regularization_value = float(regularization)
        relaxation = float(relaxation_strength)
        convergence_tolerance = float(tolerance)
        iterations = int(max_iterations)
        feature_limit = int(max_features)
        if not np.isfinite(regularization_value) or regularization_value < 0.0:
            raise ValueError("regularization must be finite and nonnegative.")
        if not np.isfinite(relaxation) or relaxation <= 0.0:
            raise ValueError("relaxation_strength must be finite and positive.")
        if penalty not in ("l0", "l1"):
            raise ValueError("penalty must be 'l0' or 'l1'.")
        if iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if not np.isfinite(convergence_tolerance) or convergence_tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
            raise ValueError("rcond must be finite and nonnegative or None.")
        resolved_zero = None if zero_tolerance is None else float(zero_tolerance)
        if resolved_zero is not None and (
            not np.isfinite(resolved_zero) or resolved_zero < 0.0
        ):
            raise ValueError("zero_tolerance must be finite and nonnegative or None.")
        if feature_limit < 1:
            raise ValueError("max_features must be positive.")
        self.regularization = regularization_value
        self.relaxation_strength = relaxation
        self.penalty = penalty
        self.max_iterations = iterations
        self.tolerance = convergence_tolerance
        self.rcond = None if rcond is None else float(rcond)
        self.scale_features = bool(scale_features)
        self.scale_targets = bool(scale_targets)
        self.unbiased_refit = bool(unbiased_refit)
        self.zero_tolerance = resolved_zero
        self.max_features = feature_limit

    def fit(self, design: SINDyDesign, /) -> SparseRegressionResult:
        if not isinstance(design, SINDyDesign):
            raise TypeError("design must be a SINDyDesign.")
        if design.num_features > self.max_features:
            raise ValueError(
                f"SR3 design has {design.num_features} features; "
                f"max_features={self.max_features}."
            )
        normalized = normalize_least_squares_design(
            design.matrix,
            mask=design.valid,
            weights=design.weights,
            scale=self.scale_features,
            rcond=self.rcond,
            max_features=self.max_features,
        )
        target_scale = _target_scales(design, self.scale_targets)
        target = design.target / target_scale[None, :]
        rows = normalized.valid_rows
        weights = jnp.where(rows, normalized.weights, 0.0)
        denominator = jnp.maximum(jnp.sum(weights), 1.0)
        matrix = jnp.where(rows[:, None], normalized.values, 0.0)
        response = jnp.where(rows[:, None], target, 0.0)
        moment = matrix.T @ (weights[:, None] * response) / denominator
        gram = matrix.T @ (weights[:, None] * matrix) / denominator
        inverse_relaxation = 1.0 / self.relaxation_strength
        system = gram + inverse_relaxation * jnp.eye(
            design.num_features, dtype=gram.dtype
        )
        sparse = jnp.zeros((design.num_features, design.output_size), dtype=moment.dtype)
        relaxed = sparse
        converged = jnp.zeros((design.output_size,), dtype=bool)
        iterations = jnp.zeros((design.output_size,), dtype=jnp.int32)
        coefficient_history = [sparse.T]
        support_history = [(sparse.T != 0.0)]
        residual_history = [
            jnp.sqrt(
                jnp.sum(
                    design.weights[:, None] * jnp.abs(design.target) ** 2,
                    axis=0,
                )
            )
        ]
        objective_history = [
            0.5 * jnp.sum(weights[:, None] * jnp.abs(response) ** 2, axis=0) / denominator
        ]
        relaxation_history = [jnp.zeros((design.output_size,), dtype=moment.real.dtype)]

        for _ in range(self.max_iterations):
            candidate_relaxed = jnp.linalg.solve(
                system, moment + inverse_relaxation * sparse
            )
            if self.penalty == "l0":
                threshold = jnp.sqrt(2.0 * self.regularization * self.relaxation_strength)
                candidate_sparse = jnp.where(
                    jnp.abs(candidate_relaxed) > threshold,
                    candidate_relaxed,
                    0.0,
                )
                penalty_value = self.regularization * jnp.sum(
                    candidate_sparse != 0.0, axis=0
                )
            else:
                threshold = self.regularization * self.relaxation_strength
                candidate_sparse = jnp.sign(candidate_relaxed) * jnp.maximum(
                    jnp.abs(candidate_relaxed) - threshold, 0.0
                )
                penalty_value = self.regularization * jnp.sum(
                    jnp.abs(candidate_sparse), axis=0
                )
            delta = jnp.sqrt(
                jnp.linalg.norm(candidate_sparse - sparse, axis=0) ** 2
                + jnp.linalg.norm(candidate_relaxed - relaxed, axis=0) ** 2
            )
            reference = jnp.maximum(
                1.0,
                jnp.maximum(
                    jnp.linalg.norm(sparse, axis=0),
                    jnp.linalg.norm(relaxed, axis=0),
                ),
            )
            newly_converged = delta <= self.tolerance * reference
            active = ~converged
            sparse = jnp.where(active[None, :], candidate_sparse, sparse)
            relaxed = jnp.where(active[None, :], candidate_relaxed, relaxed)
            iterations = iterations + active.astype(jnp.int32)
            converged = converged | newly_converged
            prediction = matrix @ relaxed
            data_error = (
                0.5
                * jnp.sum(
                    weights[:, None] * jnp.abs(response - prediction) ** 2,
                    axis=0,
                )
                / denominator
            )
            coupling = (
                0.5 * inverse_relaxation * jnp.sum(jnp.abs(relaxed - sparse) ** 2, axis=0)
            )
            physical = target_scale[:, None] * sparse.T / normalized.scale[None, :]
            physical_residual = design.target - design.matrix @ physical.T
            residual_norm = jnp.sqrt(
                jnp.sum(
                    design.weights[:, None] * jnp.abs(physical_residual) ** 2,
                    axis=0,
                )
            )
            coefficient_history.append(sparse.T)
            support_history.append(sparse.T != 0.0)
            residual_history.append(residual_norm)
            objective_history.append(data_error + coupling + penalty_value)
            relaxation_history.append(jnp.linalg.norm(relaxed - sparse, axis=0))

        support = sparse.T != 0.0
        diagnostic_coefficients, ranks, conditions, refit_valid = _solve_outputs(
            normalized, target, support, 0.0
        )
        normalized_coefficients = (
            diagnostic_coefficients if self.unbiased_refit else sparse.T
        )
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
        active_count = jnp.sum(support, axis=-1).astype(jnp.int32)
        denominator = jnp.maximum(jnp.sum(design.weights), 1.0)
        zero_error = jnp.sqrt(
            jnp.sum(
                design.weights[:, None] * jnp.abs(design.target) ** 2,
                axis=0,
            )
            / denominator
        )
        empty = active_count == 0
        empty_acceptable = (
            jnp.zeros_like(converged)
            if self.zero_tolerance is None
            else zero_error <= self.zero_tolerance
        )
        finite = jnp.all(jnp.isfinite(physical_coefficients), axis=-1) & jnp.isfinite(
            residual_norm
        )
        refit_ok = refit_valid if self.unbiased_refit else jnp.ones_like(converged)
        valid = converged & finite & refit_ok & jnp.where(empty, empty_acceptable, True)
        status = jnp.where(
            ~finite,
            IDENTIFICATION_NONFINITE,
            jnp.where(
                ~converged,
                IDENTIFICATION_NOT_CONVERGED,
                jnp.where(
                    valid,
                    IDENTIFICATION_SUCCESS,
                    IDENTIFICATION_RANK_DEFICIENT,
                ),
            ),
        ).astype(jnp.int32)
        history = SparseRegressionHistory(
            coefficients=jnp.stack(tuple(coefficient_history), axis=0),
            support=jnp.stack(tuple(support_history), axis=0),
            residual_norm=jnp.stack(tuple(residual_history), axis=0),
            rank=jnp.broadcast_to(
                ranks,
                (len(coefficient_history), design.output_size),
            ),
            condition_number=jnp.broadcast_to(
                conditions,
                (len(coefficient_history), design.output_size),
            ),
            active_count=jnp.sum(
                jnp.stack(tuple(support_history), axis=0), axis=-1
            ).astype(jnp.int32),
        )
        result = SparseRegressionResult(
            coefficients=physical_coefficients,
            normalized_coefficients=normalized_coefficients,
            support=support,
            feature_scale=normalized.scale,
            target_scale=target_scale,
            residual=residual,
            residual_norm=residual_norm,
            rank=ranks,
            condition_number=conditions,
            iterations=iterations,
            converged=converged,
            valid=valid,
            status=status,
            history=history,
            solver_diagnostics=SR3Diagnostics(
                objective=jnp.stack(tuple(objective_history), axis=0),
                relaxation_error=jnp.stack(tuple(relaxation_history), axis=0),
                relaxation_strength=self.relaxation_strength,
                regularization=self.regularization,
                penalty=self.penalty,
            ),
            feature_names=design.feature_names,
            output_names=design.output_names,
            method_id=(
                f"sr3:penalty={self.penalty}:regularization={self.regularization:g}:"
                f"relaxation={self.relaxation_strength:g}:unbiased={self.unbiased_refit}"
            ),
            design_id=design.design_id,
        )
        return result


__all__ = ["SR3Diagnostics", "SR3Penalty", "SR3Regression"]
