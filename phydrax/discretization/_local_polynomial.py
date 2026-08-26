#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    FailurePolicy,
    LeastSquaresProblem,
    LinearSolvePolicy,
    RankPolicy,
    RHSLayout,
    solve,
)


class WeightedLeastSquaresReport(StrictModule, NonTrainableState):
    maximum_condition_number: float = eqx.field(static=True)
    minimum_singular_value: float = eqx.field(static=True)
    minimum_rank: int = eqx.field(static=True)
    worst_row: int = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PreparedWeightedLeastSquares(StrictModule, NonTrainableState):
    factors: Array
    valid: Array
    report: WeightedLeastSquaresReport
    prepared_id: str = eqx.field(static=True)

    def coefficients(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[:2] != self.valid.shape:
            raise ValueError(
                "Weighted least-squares values must match row/support shape."
            )
        payload_shape = value.shape[2:]
        masked = jnp.where(
            self.valid.reshape(self.valid.shape + (1,) * len(payload_shape)),
            value,
            0,
        )
        return oe.contract("rfk,rk...->rf...", self.factors, masked)


def prepare_weighted_least_squares(
    design: ArrayLike,
    weights: ArrayLike,
    valid: ArrayLike,
    /,
    *,
    rcond: float = 1e-12,
    condition_limit: float = 1e8,
) -> PreparedWeightedLeastSquares:
    design_ = np.asarray(design, dtype=float)
    weights_ = np.asarray(weights, dtype=float)
    valid_host = np.asarray(valid)
    if valid_host.dtype != np.dtype(bool):
        raise TypeError("valid must have boolean dtype.")
    valid_ = np.asarray(valid_host, dtype=bool)
    if design_.ndim != 3:
        raise ValueError("design must have shape (rows, support, features).")
    if weights_.shape != design_.shape[:2] or valid_.shape != design_.shape[:2]:
        raise ValueError("weights/valid must match design rows and support.")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive.")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must exceed one.")
    rows, capacity, features = design_.shape
    factors = np.zeros((rows, features, capacity))
    conditions = np.empty(rows)
    singular_minimum = np.empty(rows)
    ranks = np.empty(rows, dtype=np.int32)
    for row in range(rows):
        active = valid_[row]
        if np.count_nonzero(active) < features:
            raise ValueError(f"Weighted least-squares row {row} is undersampled.")
        matrix = design_[row, active]
        weight = weights_[row, active]
        if (
            np.any(~np.isfinite(matrix))
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
        ):
            raise ValueError(
                f"Weighted least-squares row {row} is nonfinite or nonpositive."
            )
        root = np.sqrt(weight)
        weighted = root[:, None] * matrix
        scale = np.sqrt(np.sum(weighted * weighted, axis=0))
        if np.any(scale <= 0.0) or np.any(~np.isfinite(scale)):
            raise ValueError(
                f"Weighted least-squares row {row} has a zero feature column."
            )
        normalized = weighted / scale[None, :]
        solved = solve(
            LeastSquaresProblem(
                DenseLinearOperator(jnp.asarray(normalized)),
                problem_id=f"weighted-least-squares-row-{row}",
            ),
            jnp.asarray(np.diag(root)),
            policy=LinearSolvePolicy(
                DenseSVD(),
                rank=RankPolicy(
                    relative_cutoff=float(rcond),
                    require_full_rank=True,
                ),
                failure=FailurePolicy("error"),
            ),
            rhs_layout=RHSLayout((active.size,)),
        )
        pseudoinverse = np.asarray(solved.value)
        if solved.diagnostics.singular_values is None:
            raise RuntimeError("Dense SVD did not return singular-value evidence.")
        singular_values = np.asarray(solved.diagnostics.singular_values)
        singular = singular_values.reshape((-1, singular_values.shape[-1]))[0]
        rank = int(np.min(np.asarray(solved.diagnostics.rank)))
        condition = float(np.max(np.asarray(solved.diagnostics.condition_estimate)))
        factors[row][:, active] = pseudoinverse / scale[:, None]
        conditions[row] = condition
        singular_minimum[row] = singular[-1]
        ranks[row] = rank
    worst = int(np.argmax(conditions))
    report_id = canonical_fingerprint(
        {
            "kind": "weighted-least-squares-report",
            "design": array_tree_fingerprint(design_),
            "weights": array_tree_fingerprint(weights_),
            "valid": array_tree_fingerprint(valid_),
            "rcond": float(rcond),
            "condition_limit": float(condition_limit),
        }
    )
    report = WeightedLeastSquaresReport(
        maximum_condition_number=float(conditions[worst]),
        minimum_singular_value=float(np.min(singular_minimum)),
        minimum_rank=int(np.min(ranks)),
        worst_row=worst,
        row_count=rows,
        feature_count=features,
        report_id=report_id,
    )
    return PreparedWeightedLeastSquares(
        factors=jnp.asarray(factors),
        valid=jnp.asarray(valid_),
        report=report,
        prepared_id=canonical_fingerprint(
            {"kind": "prepared-weighted-least-squares", "report": report_id}
        ),
    )


__all__ = [
    "PreparedWeightedLeastSquares",
    "WeightedLeastSquaresReport",
    "prepare_weighted_least_squares",
]
