#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics import normalize_least_squares_design
from ..._strict import StrictModule
from ._sindy_design import SINDyDesign
from ._sparse_regression import (
    AbstractSparseRegression,
    SparseRegressionHistory,
    SparseRegressionResult,
)
from ._status import (
    IDENTIFICATION_INFEASIBLE,
    IDENTIFICATION_INSUFFICIENT_SAMPLES,
    IDENTIFICATION_NONFINITE,
    IDENTIFICATION_NOT_CONVERGED,
    IDENTIFICATION_RANK_DEFICIENT,
    IDENTIFICATION_SUCCESS,
)


CoefficientIndex = tuple[int, int]


class LinearCoefficientConstraint(StrictModule):
    """Physical-coordinate equalities `matrix @ coefficients.ravel() = rhs`."""

    matrix: Array
    rhs: Array
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        rhs: ArrayLike,
        /,
        *,
        constraint_id: str,
    ):
        matrix_values = jnp.asarray(matrix)
        rhs_values = jnp.asarray(rhs)
        if matrix_values.ndim != 2 or rhs_values.shape != (matrix_values.shape[0],):
            raise ValueError(
                "constraint matrix and rhs must have shapes (m, p) and (m,)."
            )
        if matrix_values.shape[0] < 1 or matrix_values.shape[1] < 1:
            raise ValueError("constraint matrix must be non-empty.")
        if not bool(
            jnp.all(jnp.isfinite(matrix_values)) & jnp.all(jnp.isfinite(rhs_values))
        ):
            raise ValueError("constraint matrix and rhs must be finite.")
        if not isinstance(constraint_id, str) or not constraint_id:
            raise ValueError("constraint_id must be a non-empty string.")
        self.matrix = matrix_values
        self.rhs = rhs_values
        self.constraint_id = constraint_id


class CoefficientStructure(StrictModule):
    """Group support, forbidden coefficients, and linear conservation constraints."""

    groups: tuple[tuple[CoefficientIndex, ...], ...] = eqx.field(static=True)
    allowed: Array | None
    constraint: LinearCoefficientConstraint | None
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        groups: Sequence[Sequence[CoefficientIndex]] = (),
        allowed: ArrayLike | None = None,
        constraint: LinearCoefficientConstraint | None = None,
        structure_id: str | None = None,
    ):
        resolved_groups = tuple(
            tuple((int(output), int(feature)) for output, feature in group)
            for group in groups
        )
        if any(not group or len(set(group)) != len(group) for group in resolved_groups):
            raise ValueError("Every coefficient group must be non-empty and unique.")
        flattened_members = tuple(member for group in resolved_groups for member in group)
        if len(set(flattened_members)) != len(flattened_members):
            raise ValueError("Coefficient groups must not overlap.")
        allowed_values = None if allowed is None else jnp.asarray(allowed, dtype=bool)
        if allowed_values is not None and allowed_values.ndim != 2:
            raise ValueError("allowed must be a rank-two output-by-feature mask.")
        if constraint is not None and not isinstance(
            constraint, LinearCoefficientConstraint
        ):
            raise TypeError("constraint must be a LinearCoefficientConstraint or None.")
        identifier = (
            "coefficient-structure:"
            + canonical_fingerprint(
                {
                    "groups": resolved_groups,
                    "allowed": None
                    if allowed_values is None
                    else np.asarray(allowed_values).tolist(),
                    "constraint": None
                    if constraint is None
                    else constraint.constraint_id,
                }
            )
            if structure_id is None
            else str(structure_id)
        )
        if not identifier:
            raise ValueError("structure_id must be non-empty.")
        self.groups = resolved_groups
        self.allowed = allowed_values
        self.constraint = constraint
        self.structure_id = identifier


class StructuredRegressionDiagnostics(StrictModule):
    """Joint objective, constraint defect, group norm, and active-count histories."""

    objective: Array
    constraint_residual: Array
    group_norm: Array
    active_count: Array
    structure_id: str = eqx.field(static=True)


def named_coefficient_constraint(
    design: SINDyDesign,
    equations: Sequence[Mapping[tuple[str, str], float]],
    rhs: Sequence[float],
    /,
    *,
    constraint_id: str,
) -> LinearCoefficientConstraint:
    """Bind named output/feature coefficient equalities to one immutable design."""
    if not isinstance(design, SINDyDesign):
        raise TypeError("design must be a SINDyDesign.")
    rows = tuple(equations)
    rhs_values = tuple(float(value) for value in rhs)
    if not rows or len(rhs_values) != len(rows):
        raise ValueError("rhs must contain one value per named equation.")
    matrix = np.zeros((len(rows), design.output_size * design.num_features))
    for row_index, equation in enumerate(rows):
        if not equation:
            raise ValueError("Named coefficient equations must not be empty.")
        for (output_name, feature_name), coefficient in equation.items():
            if output_name not in design.output_names:
                raise ValueError(f"Unknown output name {output_name!r}.")
            if feature_name not in design.feature_names:
                raise ValueError(f"Unknown feature name {feature_name!r}.")
            output = design.output_names.index(output_name)
            feature = design.feature_names.index(feature_name)
            matrix[row_index, output * design.num_features + feature] = float(coefficient)
    return LinearCoefficientConstraint(
        matrix,
        np.asarray(rhs_values),
        constraint_id=constraint_id,
    )


def shared_feature_groups(
    output_size: int,
    num_features: int,
    /,
) -> tuple[tuple[CoefficientIndex, ...], ...]:
    """Group each feature across all outputs for shared-support thresholding."""
    outputs = int(output_size)
    features = int(num_features)
    if outputs < 1 or features < 1:
        raise ValueError("output_size and num_features must be positive.")
    return tuple(
        tuple((output, feature) for output in range(outputs))
        for feature in range(features)
    )


def _resolve_structure(
    structure: CoefficientStructure,
    design: SINDyDesign,
    /,
) -> tuple[Array, tuple[tuple[int, ...], ...], Array, Array]:
    shape = (design.output_size, design.num_features)
    allowed = (
        jnp.ones(shape, dtype=bool) if structure.allowed is None else structure.allowed
    )
    if allowed.shape != shape:
        raise ValueError(f"structure.allowed must have shape {shape}.")
    groups = []
    assigned = set()
    for group in structure.groups:
        flat_group = []
        for output, feature in group:
            if output < 0 or output >= shape[0] or feature < 0 or feature >= shape[1]:
                raise ValueError("Coefficient group index is out of range.")
            if bool(allowed[output, feature]):
                index = output * shape[1] + feature
                flat_group.append(index)
                assigned.add(index)
        if flat_group:
            groups.append(tuple(flat_group))
    for index, permitted in enumerate(np.asarray(allowed).reshape(-1)):
        if permitted and index not in assigned:
            groups.append((index,))
    coefficient_count = shape[0] * shape[1]
    if structure.constraint is None:
        constraint_matrix = jnp.zeros((0, coefficient_count), dtype=design.matrix.dtype)
        constraint_rhs = jnp.zeros((0,), dtype=design.matrix.dtype)
    else:
        constraint_matrix = jnp.asarray(
            structure.constraint.matrix, dtype=design.matrix.dtype
        )
        constraint_rhs = jnp.asarray(structure.constraint.rhs, dtype=design.matrix.dtype)
        if constraint_matrix.shape[1] != coefficient_count:
            raise ValueError(
                f"constraint matrix must have {coefficient_count} coefficient columns."
            )
    return allowed, tuple(groups), constraint_matrix, constraint_rhs


def _rank_condition(
    matrix: Array,
    weights: Array,
    support: Array,
    /,
) -> tuple[Array, Array]:
    root_weight = jnp.sqrt(weights / jnp.maximum(jnp.sum(weights), 1.0))
    ranks = []
    conditions = []
    for output in range(support.shape[0]):
        weighted = root_weight[:, None] * matrix * support[output][None, :]
        singular = jnp.linalg.svd(weighted, compute_uv=False)
        largest = jnp.max(singular, initial=0.0)
        tolerance = largest * jnp.finfo(singular.dtype).eps * max(matrix.shape)
        retained = singular > tolerance
        rank = jnp.sum(retained).astype(jnp.int32)
        active_count = jnp.sum(support[output]).astype(jnp.int32)
        smallest = jnp.min(jnp.where(retained, singular, jnp.inf), initial=jnp.inf)
        condition = jnp.where(
            (active_count > 0) & (rank == active_count),
            largest / smallest,
            jnp.inf,
        )
        ranks.append(rank)
        conditions.append(condition)
    return jnp.stack(tuple(ranks)), jnp.stack(tuple(conditions))


class StructuredSequentialThresholdedLeastSquares(AbstractSparseRegression):
    """Joint group-sparse least squares with exact physical linear equalities."""

    threshold: float = eqx.field(static=True)
    structure: CoefficientStructure
    max_iterations: int = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    normalize_columns: bool = eqx.field(static=True)
    normalize_targets: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_coefficients: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        threshold: float,
        /,
        *,
        structure: CoefficientStructure | None = None,
        max_iterations: int = 20,
        ridge: float = 0.0,
        normalize_columns: bool = True,
        normalize_targets: bool = False,
        tolerance: float = 1e-8,
        max_coefficients: int = 2048,
    ):
        threshold_value = float(threshold)
        ridge_value = float(ridge)
        tolerance_value = float(tolerance)
        if not np.isfinite(threshold_value) or threshold_value < 0.0:
            raise ValueError("threshold must be finite and nonnegative.")
        if not np.isfinite(ridge_value) or ridge_value < 0.0:
            raise ValueError("ridge must be finite and nonnegative.")
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if int(max_iterations) < 1 or int(max_coefficients) < 1:
            raise ValueError("max_iterations and max_coefficients must be positive.")
        self.threshold = threshold_value
        self.structure = CoefficientStructure() if structure is None else structure
        if not isinstance(self.structure, CoefficientStructure):
            raise TypeError("structure must be a CoefficientStructure or None.")
        self.max_iterations = int(max_iterations)
        self.ridge = ridge_value
        self.normalize_columns = bool(normalize_columns)
        self.normalize_targets = bool(normalize_targets)
        self.tolerance = tolerance_value
        self.max_coefficients = int(max_coefficients)
        self.method_id = (
            f"structured-stlsq:threshold={threshold_value:g}:ridge={ridge_value:g}:"
            f"structure={self.structure.structure_id}"
        )

    def fit(self, design: SINDyDesign, /) -> SparseRegressionResult:
        if not isinstance(design, SINDyDesign):
            raise TypeError("design must be a SINDyDesign.")
        coefficient_count = design.output_size * design.num_features
        if coefficient_count > self.max_coefficients:
            raise ValueError(
                f"Structured regression has {coefficient_count} coefficients; "
                f"max_coefficients={self.max_coefficients}."
            )
        allowed, groups, physical_constraint, constraint_rhs = _resolve_structure(
            self.structure, design
        )
        normalized = normalize_least_squares_design(
            design.matrix,
            mask=design.valid,
            weights=design.weights,
            scale=self.normalize_columns,
            max_features=design.num_features,
        )
        if self.normalize_targets:
            denominator = jnp.maximum(jnp.sum(normalized.weights), 1.0)
            second_moment = (
                jnp.sum(
                    normalized.weights[:, None] * jnp.abs(design.target) ** 2,
                    axis=0,
                )
                / denominator
            )
            target_scale = jnp.where(
                second_moment > jnp.finfo(second_moment.dtype).eps,
                jnp.sqrt(second_moment),
                1.0,
            )
        else:
            target_scale = jnp.ones((design.output_size,), dtype=design.target.dtype)
        normalized_target = design.target / target_scale[None, :]
        valid_rows = normalized.valid_rows & jnp.all(
            jnp.isfinite(normalized_target), axis=-1
        )
        row_weights = jnp.where(valid_rows, normalized.weights, 0.0)
        denominator = jnp.maximum(jnp.sum(row_weights), 1.0)
        root_weight = jnp.sqrt(row_weights / denominator)
        block_matrix = jnp.kron(
            jnp.eye(design.output_size, dtype=design.matrix.dtype),
            normalized.values,
        )
        response = jnp.swapaxes(normalized_target, 0, 1).reshape((-1,))
        repeated_root_weight = jnp.tile(root_weight, design.output_size)
        weighted_matrix = repeated_root_weight[:, None] * block_matrix
        weighted_response = repeated_root_weight * response
        hessian = weighted_matrix.T @ weighted_matrix + self.ridge * jnp.eye(
            coefficient_count, dtype=design.matrix.dtype
        )
        moment = weighted_matrix.T @ weighted_response
        physical_scale = (target_scale[:, None] / normalized.scale[None, :]).reshape(
            (-1,)
        )
        normalized_constraint = physical_constraint * physical_scale[None, :]
        active = allowed.reshape((-1,))
        coefficients = jnp.zeros((coefficient_count,), dtype=design.matrix.dtype)
        coefficient_history = []
        support_history = []
        residual_history = []
        rank_history = []
        condition_history = []
        active_count_history = []
        objective_history = []
        constraint_history = []
        group_history = []
        converged = False

        def solve(active_mask: Array) -> Array:
            active_indices = np.flatnonzero(np.asarray(active_mask))
            if active_indices.size == 0:
                return jnp.zeros((coefficient_count,), dtype=design.matrix.dtype)
            indices = jnp.asarray(active_indices)
            reduced_hessian = hessian[indices[:, None], indices[None, :]]
            reduced_moment = moment[indices]
            reduced_constraint = normalized_constraint[:, indices]
            if reduced_constraint.shape[0] == 0:
                reduced = jnp.linalg.lstsq(reduced_hessian, reduced_moment, rcond=None)[0]
            else:
                zero = jnp.zeros(
                    (
                        reduced_constraint.shape[0],
                        reduced_constraint.shape[0],
                    ),
                    dtype=hessian.dtype,
                )
                kkt = jnp.block(
                    [
                        [
                            reduced_hessian,
                            jnp.swapaxes(reduced_constraint, 0, 1),
                        ],
                        [reduced_constraint, zero],
                    ]
                )
                right = jnp.concatenate((reduced_moment, constraint_rhs))
                reduced = jnp.linalg.lstsq(kkt, right, rcond=None)[0][
                    : active_indices.size
                ]
            return (
                jnp.zeros((coefficient_count,), dtype=reduced.dtype)
                .at[indices]
                .set(reduced)
            )

        for _ in range(self.max_iterations):
            coefficients = solve(active)
            group_norms = jnp.stack(
                tuple(
                    jnp.linalg.norm(coefficients[jnp.asarray(group)]) for group in groups
                )
            )
            next_active = jnp.zeros_like(active)
            for group, norm in zip(groups, group_norms, strict=True):
                next_active = next_active.at[jnp.asarray(group)].set(
                    norm >= self.threshold
                )
            next_active = next_active & allowed.reshape((-1,))
            physical = (physical_scale * coefficients).reshape(
                (design.output_size, design.num_features)
            )
            residual = design.target - design.matrix @ physical.T
            residual_norm = jnp.sqrt(
                jnp.sum(design.weights[:, None] * jnp.abs(residual) ** 2, axis=0)
            )
            support = active.reshape((design.output_size, design.num_features))
            ranks, conditions = _rank_condition(normalized.values, row_weights, support)
            prediction_error = jnp.sum(
                repeated_root_weight**2
                * jnp.abs(response - block_matrix @ coefficients) ** 2
            )
            constraint_defect = (
                physical_constraint @ physical.reshape((-1,)) - constraint_rhs
            )
            coefficient_history.append(physical)
            support_history.append(support)
            residual_history.append(residual_norm)
            rank_history.append(ranks)
            condition_history.append(conditions)
            active_count_history.append(jnp.sum(support, axis=-1).astype(jnp.int32))
            objective_history.append(
                prediction_error + self.ridge * jnp.sum(jnp.abs(coefficients) ** 2)
            )
            constraint_history.append(jnp.linalg.norm(constraint_defect))
            group_history.append(group_norms)
            if bool(jnp.array_equal(next_active, active)):
                converged = True
                break
            active = next_active
        if not converged:
            coefficients = solve(active)
        physical = (physical_scale * coefficients).reshape(
            (design.output_size, design.num_features)
        )
        support = active.reshape((design.output_size, design.num_features))
        physical = jnp.where(support, physical, 0.0)
        residual = design.target - design.matrix @ physical.T
        residual = jnp.where(design.valid[:, None], residual, 0.0)
        residual_norm = jnp.sqrt(
            jnp.sum(design.weights[:, None] * jnp.abs(residual) ** 2, axis=0)
        )
        ranks, conditions = _rank_condition(normalized.values, row_weights, support)
        constraint_defect = physical_constraint @ physical.reshape((-1,)) - constraint_rhs
        constraint_valid = jnp.linalg.norm(
            constraint_defect
        ) <= self.tolerance * jnp.maximum(1.0, jnp.linalg.norm(constraint_rhs))
        finite = (
            jnp.all(jnp.isfinite(physical))
            & jnp.all(jnp.isfinite(residual_norm))
            & jnp.all(jnp.isfinite(conditions) | jnp.isinf(conditions))
        )
        sample_count = jnp.sum(valid_rows).astype(jnp.int32)
        active_count = jnp.sum(support, axis=-1).astype(jnp.int32)
        sufficient = sample_count >= jnp.maximum(active_count, 1)
        nonempty = active_count > 0
        output_valid = (
            jnp.asarray(converged) & constraint_valid & finite & sufficient & nonempty
        )
        status = jnp.where(
            ~finite,
            IDENTIFICATION_NONFINITE,
            jnp.where(
                ~jnp.asarray(converged),
                IDENTIFICATION_NOT_CONVERGED,
                jnp.where(
                    ~constraint_valid,
                    IDENTIFICATION_INFEASIBLE,
                    jnp.where(
                        ~sufficient,
                        IDENTIFICATION_INSUFFICIENT_SAMPLES,
                        jnp.where(
                            ~nonempty,
                            IDENTIFICATION_RANK_DEFICIENT,
                            IDENTIFICATION_SUCCESS,
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        if not coefficient_history:
            raise RuntimeError("Structured regression produced no iterations.")
        history = SparseRegressionHistory(
            coefficients=jnp.stack(tuple(coefficient_history)),
            support=jnp.stack(tuple(support_history)),
            residual_norm=jnp.stack(tuple(residual_history)),
            rank=jnp.stack(tuple(rank_history)),
            condition_number=jnp.stack(tuple(condition_history)),
            active_count=jnp.stack(tuple(active_count_history)),
        )
        diagnostics = StructuredRegressionDiagnostics(
            objective=jnp.stack(tuple(objective_history)),
            constraint_residual=jnp.stack(tuple(constraint_history)),
            group_norm=jnp.stack(tuple(group_history)),
            active_count=jnp.stack(tuple(active_count_history)),
            structure_id=self.structure.structure_id,
        )
        return SparseRegressionResult(
            coefficients=physical,
            normalized_coefficients=coefficients.reshape(
                (design.output_size, design.num_features)
            ),
            support=support,
            feature_scale=normalized.scale,
            target_scale=target_scale,
            residual=residual,
            residual_norm=residual_norm,
            rank=ranks,
            condition_number=conditions,
            iterations=jnp.full(
                (design.output_size,), len(coefficient_history), dtype=jnp.int32
            ),
            converged=jnp.full((design.output_size,), converged, dtype=bool),
            valid=output_valid,
            status=jnp.broadcast_to(status, (design.output_size,)),
            history=history,
            solver_diagnostics=diagnostics,
            feature_names=design.feature_names,
            output_names=design.output_names,
            method_id=self.method_id,
            design_id=design.design_id,
        )


__all__ = [
    "CoefficientIndex",
    "CoefficientStructure",
    "LinearCoefficientConstraint",
    "StructuredRegressionDiagnostics",
    "StructuredSequentialThresholdedLeastSquares",
    "named_coefficient_constraint",
    "shared_feature_groups",
]
