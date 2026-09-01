#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._numerics import (
    normalize_least_squares_design,
    solve_weighted_least_squares,
)
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


DESIGN_SUCCESS = 0
DESIGN_RANK_DEFICIENT = 1
DESIGN_INSUFFICIENT_ROWS = 2
DESIGN_NONFINITE = 3

TERM_INTERCEPT = 0
TERM_CONDITION = 1
TERM_DONOR = 2
TERM_BATCH = 3
TERM_INTERACTION = 4
TERM_NESTED_BATCH = 5
TERM_COVARIATE = 6


def _design_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "experimental-design-encoding",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement="Rank and condition are diagnosed on valid design rows.",
        truncation_statement="Categorical capacities are explicit and no level is truncated.",
        capacity_semantics="Matrix width is fixed by declared factor levels and covariates.",
        assumptions=("Integer factor codes enumerate declared finite factor spaces.",),
        nondifferentiable_outputs=("rank", "status", "estimable"),
    )


def _row_array(
    name: str,
    value: ArrayLike | None,
    samples: int,
    /,
    *,
    default: int = -1,
) -> Array:
    if value is None:
        return jnp.full((samples,), default, dtype=jnp.int32)
    result = jnp.asarray(value)
    if result.shape != (samples,):
        raise ValueError(f"{name} must have shape ({samples},); got {result.shape}.")
    if not jnp.issubdtype(result.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    return result.astype(jnp.int32)


def _factor(
    name: str,
    value: ArrayLike,
    samples: int,
    levels: int,
    valid_rows: Array,
    /,
) -> Array:
    factor = _row_array(name, value, samples)
    if levels < 1:
        raise ValueError(f"num_{name}s must be positive.")
    if bool(jnp.any(valid_rows & ((factor < 0) | (factor >= levels)))):
        raise ValueError(f"A valid {name} code lies outside [0, {levels}).")
    return factor


class ExperimentalDesign(StrictModule):
    """Audited fixed-width experimental design and biological row metadata."""

    matrix: Array
    valid_rows: Array
    sample_indices: Array
    condition_indices: Array
    donor_indices: Array
    batch_indices: Array
    coefficient_terms: Array
    coefficient_levels: Array
    coefficient_projection: Array
    singular_values: Array
    rank: Array
    condition_number: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    num_samples: int = eqx.field(static=True)
    num_coefficients: int = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        valid_rows: ArrayLike | None = None,
        sample_indices: ArrayLike | None = None,
        condition_indices: ArrayLike | None = None,
        donor_indices: ArrayLike | None = None,
        batch_indices: ArrayLike | None = None,
        coefficient_terms: ArrayLike | None = None,
        coefficient_levels: ArrayLike | None = None,
        rcond: float | None = None,
    ):
        values = jnp.asarray(matrix)
        if values.ndim != 2:
            raise ValueError("matrix must have shape (samples, coefficients).")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        samples, coefficients = (int(size) for size in values.shape)
        if samples < 1 or coefficients < 1:
            raise ValueError("An experimental design must be non-empty.")
        rows = (
            jnp.ones((samples,), dtype=bool)
            if valid_rows is None
            else jnp.asarray(valid_rows, dtype=bool)
        )
        if rows.shape != (samples,):
            raise ValueError(
                f"valid_rows must have shape ({samples},); got {rows.shape}."
            )
        finite_rows = jnp.all(jnp.isfinite(values), axis=1)
        diagnosed = normalize_least_squares_design(
            values,
            mask=rows,
            rcond=rcond,
            max_features=coefficients,
        )
        projection_fit = solve_weighted_least_squares(
            values,
            values,
            mask=rows,
            rcond=rcond,
            min_samples=1,
            max_features=coefficients,
        )
        terms = (
            jnp.full((coefficients,), -1, dtype=jnp.int32)
            if coefficient_terms is None
            else jnp.asarray(coefficient_terms, dtype=jnp.int32)
        )
        if terms.shape != (coefficients,):
            raise ValueError(
                "coefficient_terms must have one entry per design coefficient."
            )
        levels = (
            jnp.full((coefficients, 2), -1, dtype=jnp.int32)
            if coefficient_levels is None
            else jnp.asarray(coefficient_levels, dtype=jnp.int32)
        )
        if levels.shape != (coefficients, 2):
            raise ValueError("coefficient_levels must have shape (coefficients, 2).")
        sample_count = jnp.sum(rows & finite_rows).astype(jnp.int32)
        finite = jnp.all(finite_rows | ~rows)
        full_rank = diagnosed.rank == coefficients
        enough = sample_count >= coefficients
        valid = finite & full_rank & enough
        status = jnp.where(
            ~finite,
            DESIGN_NONFINITE,
            jnp.where(
                ~enough,
                DESIGN_INSUFFICIENT_ROWS,
                jnp.where(~full_rank, DESIGN_RANK_DEFICIENT, DESIGN_SUCCESS),
            ),
        ).astype(jnp.int32)
        self.matrix = values
        self.valid_rows = rows & finite_rows
        self.sample_indices = (
            jnp.arange(samples, dtype=jnp.int32)
            if sample_indices is None
            else _row_array("sample_indices", sample_indices, samples)
        )
        self.condition_indices = _row_array(
            "condition_indices", condition_indices, samples
        )
        self.donor_indices = _row_array("donor_indices", donor_indices, samples)
        self.batch_indices = _row_array("batch_indices", batch_indices, samples)
        self.coefficient_terms = terms
        self.coefficient_levels = levels
        self.coefficient_projection = projection_fit.raw_coefficients
        self.singular_values = diagnosed.singular_values
        self.rank = diagnosed.rank
        self.condition_number = diagnosed.condition_number
        self.valid = valid
        self.status = status
        self.evidence = jnp.stack(
            (
                sample_count.astype(values.dtype),
                jnp.asarray(coefficients, dtype=values.dtype),
                diagnosed.rank.astype(values.dtype),
                diagnosed.condition_number.astype(values.dtype),
            )
        )
        self.method_contract = _design_contract()
        self.num_samples = samples
        self.num_coefficients = coefficients


class DesignContrast(StrictModule):
    """One linear contrast in the coefficient coordinates of a design."""

    weights: Array
    estimable: Array
    estimability_error: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    num_coefficients: int = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        /,
        *,
        design: ExperimentalDesign | None = None,
        tolerance: float = 1.0e-7,
    ):
        values = jnp.asarray(weights)
        if values.ndim != 1:
            raise ValueError("contrast weights must be rank one.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        coefficients = int(values.shape[0])
        if coefficients < 1:
            raise ValueError("A contrast must contain at least one coefficient.")
        threshold = float(tolerance)
        if threshold < 0.0:
            raise ValueError("tolerance must be nonnegative.")
        finite = jnp.all(jnp.isfinite(values))
        if design is None:
            residual = jnp.asarray(jnp.nan, dtype=values.dtype)
            estimable = finite
        else:
            if not isinstance(design, ExperimentalDesign):
                raise TypeError("design must be an ExperimentalDesign or None.")
            if design.num_coefficients != coefficients:
                raise ValueError("contrast width does not match the design.")
            projected = design.coefficient_projection @ values
            residual = jnp.max(jnp.abs(values - projected), initial=0.0)
            scale = jnp.maximum(jnp.max(jnp.abs(values), initial=0.0), 1.0)
            estimable = finite & (residual <= threshold * scale)
        self.weights = values
        self.estimable = estimable
        self.estimability_error = residual
        self.valid = estimable
        self.status = jnp.where(estimable, DESIGN_SUCCESS, DESIGN_RANK_DEFICIENT).astype(
            jnp.int32
        )
        self.evidence = jnp.stack((residual, jnp.asarray(threshold, dtype=values.dtype)))
        self.method_contract = _design_contract()
        self.num_coefficients = coefficients


def build_experimental_design(
    condition_indices: ArrayLike,
    /,
    *,
    num_conditions: int,
    donor_indices: ArrayLike | None = None,
    num_donors: int | None = None,
    batch_indices: ArrayLike | None = None,
    num_batches: int | None = None,
    covariates: ArrayLike | None = None,
    valid_rows: ArrayLike | None = None,
    include_intercept: bool = True,
    paired: bool = False,
    include_batch: bool = False,
    condition_batch_interaction: bool = False,
    batch_nested_in_donor: bool = False,
    rcond: float | None = None,
) -> ExperimentalDesign:
    """Encode treatment, pairing, batch, nesting, and interaction effects.

    Treatment coding uses level zero as reference. A paired design includes donor
    fixed effects. Nested-batch columns compare each non-reference batch within
    each donor, so technical cells do not create biological replication.
    """

    conditions = jnp.asarray(condition_indices)
    if conditions.ndim != 1:
        raise ValueError("condition_indices must be rank one.")
    samples = int(conditions.shape[0])
    if samples < 1:
        raise ValueError("At least one sample is required.")
    rows = (
        jnp.ones((samples,), dtype=bool)
        if valid_rows is None
        else jnp.asarray(valid_rows, dtype=bool)
    )
    if rows.shape != (samples,):
        raise ValueError(f"valid_rows must have shape ({samples},).")
    condition_count = int(num_conditions)
    condition = _factor("condition", conditions, samples, condition_count, rows)

    needs_donor = bool(paired or batch_nested_in_donor)
    if needs_donor:
        if donor_indices is None or num_donors is None:
            raise ValueError("Paired or nested designs require donor codes and capacity.")
        donor_count = int(num_donors)
        donors = _factor("donor", donor_indices, samples, donor_count, rows)
    else:
        donor_count = 0 if num_donors is None else int(num_donors)
        donors = _row_array("donor_indices", donor_indices, samples)

    needs_batch = bool(
        include_batch or condition_batch_interaction or batch_nested_in_donor
    )
    if needs_batch:
        if batch_indices is None or num_batches is None:
            raise ValueError("Batch terms require batch codes and capacity.")
        batch_count = int(num_batches)
        batches = _factor("batch", batch_indices, samples, batch_count, rows)
    else:
        batch_count = 0 if num_batches is None else int(num_batches)
        batches = _row_array("batch_indices", batch_indices, samples)

    columns: list[Array] = []
    terms: list[int] = []
    levels: list[tuple[int, int]] = []

    if include_intercept:
        columns.append(jnp.ones((samples,), dtype=float))
        terms.append(TERM_INTERCEPT)
        levels.append((0, -1))
    for level in range(1, condition_count):
        columns.append((condition == level).astype(float))
        terms.append(TERM_CONDITION)
        levels.append((level, -1))
    if paired or batch_nested_in_donor:
        for level in range(1, donor_count):
            columns.append((donors == level).astype(float))
            terms.append(TERM_DONOR)
            levels.append((level, -1))
    if include_batch or condition_batch_interaction:
        for level in range(1, batch_count):
            columns.append((batches == level).astype(float))
            terms.append(TERM_BATCH)
            levels.append((level, -1))
    if condition_batch_interaction:
        for condition_level in range(1, condition_count):
            for batch_level in range(1, batch_count):
                columns.append(
                    ((condition == condition_level) & (batches == batch_level)).astype(
                        float
                    )
                )
                terms.append(TERM_INTERACTION)
                levels.append((condition_level, batch_level))
    if batch_nested_in_donor:
        for donor_level in range(donor_count):
            for batch_level in range(1, batch_count):
                columns.append(
                    ((donors == donor_level) & (batches == batch_level)).astype(float)
                )
                terms.append(TERM_NESTED_BATCH)
                levels.append((donor_level, batch_level))

    if covariates is not None:
        continuous = jnp.asarray(covariates)
        if continuous.ndim == 1:
            continuous = continuous[:, None]
        if continuous.ndim != 2 or int(continuous.shape[0]) != samples:
            raise ValueError("covariates must have shape (samples, covariates).")
        if not jnp.issubdtype(continuous.dtype, jnp.inexact):
            continuous = continuous.astype(float)
        for column in range(int(continuous.shape[1])):
            columns.append(continuous[:, column])
            terms.append(TERM_COVARIATE)
            levels.append((column, -1))
    if not columns:
        raise ValueError("The requested design contains no coefficient columns.")

    matrix = jnp.stack(columns, axis=1)
    return ExperimentalDesign(
        matrix,
        valid_rows=rows,
        condition_indices=condition,
        donor_indices=donors,
        batch_indices=batches,
        coefficient_terms=jnp.asarray(terms, dtype=jnp.int32),
        coefficient_levels=jnp.asarray(levels, dtype=jnp.int32),
        rcond=rcond,
    )


def pairwise_condition_contrast(
    design: ExperimentalDesign,
    positive_condition: int,
    negative_condition: int = 0,
    /,
    *,
    batch_level: int = 0,
    tolerance: float = 1.0e-7,
) -> DesignContrast:
    """Construct a condition contrast at one batch level."""

    if not isinstance(design, ExperimentalDesign):
        raise TypeError("design must be an ExperimentalDesign.")
    positive = int(positive_condition)
    negative = int(negative_condition)
    batch = int(batch_level)
    term = design.coefficient_terms
    level0 = design.coefficient_levels[:, 0]
    level1 = design.coefficient_levels[:, 1]
    main = (term == TERM_CONDITION).astype(design.matrix.dtype) * (
        (level0 == positive).astype(design.matrix.dtype)
        - (level0 == negative).astype(design.matrix.dtype)
    )
    interaction = ((term == TERM_INTERACTION) & (level1 == batch)).astype(
        design.matrix.dtype
    ) * (
        (level0 == positive).astype(design.matrix.dtype)
        - (level0 == negative).astype(design.matrix.dtype)
    )
    return DesignContrast(main + interaction, design=design, tolerance=tolerance)


__all__ = [
    "DESIGN_INSUFFICIENT_ROWS",
    "DESIGN_NONFINITE",
    "DESIGN_RANK_DEFICIENT",
    "DESIGN_SUCCESS",
    "DesignContrast",
    "ExperimentalDesign",
    "TERM_BATCH",
    "TERM_CONDITION",
    "TERM_COVARIATE",
    "TERM_DONOR",
    "TERM_INTERACTION",
    "TERM_INTERCEPT",
    "TERM_NESTED_BATCH",
    "build_experimental_design",
    "pairwise_condition_contrast",
]
