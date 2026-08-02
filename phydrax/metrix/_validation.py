#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._metric import RiemannianMetric


class MetricValidationReport(StrictModule):
    """Aggregate diagnostics from explicit metric validation points."""

    valid: Array
    finite: Array
    maximum_asymmetry: Array
    minimum_eigenvalue: Array
    maximum_condition_number: Array

    def __init__(
        self,
        *,
        valid: Array,
        finite: Array,
        maximum_asymmetry: Array,
        minimum_eigenvalue: Array,
        maximum_condition_number: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.maximum_asymmetry = jnp.asarray(maximum_asymmetry)
        self.minimum_eigenvalue = jnp.asarray(minimum_eigenvalue)
        self.maximum_condition_number = jnp.asarray(maximum_condition_number)


def validate_metric(
    metric: RiemannianMetric,
    points: ArrayLike,
    /,
    *,
    symmetry_tolerance: float = 1e-8,
    eigenvalue_floor: float = 0.0,
    maximum_condition_number: float | None = None,
    raise_on_error: bool = True,
) -> MetricValidationReport:
    """Validate, without modifying, a Riemannian metric at representative points."""

    if symmetry_tolerance < 0.0:
        raise ValueError("symmetry_tolerance must be non-negative.")
    if maximum_condition_number is not None and maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must be greater than one.")
    matrix = metric(points)
    finite = jnp.all(jnp.isfinite(matrix))
    asymmetry = matrix - jnp.swapaxes(matrix, -1, -2)
    maximum_asymmetry_ = jnp.max(jnp.abs(asymmetry))
    symmetric_matrix = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    eigenvalues = jnp.linalg.eigvalsh(symmetric_matrix)
    minimum_eigenvalue = jnp.min(eigenvalues)
    condition_numbers = jnp.max(eigenvalues, axis=-1) / jnp.min(eigenvalues, axis=-1)
    maximum_condition_number_ = jnp.max(condition_numbers)
    valid = (
        finite
        & (maximum_asymmetry_ <= symmetry_tolerance)
        & (minimum_eigenvalue > eigenvalue_floor)
    )
    if maximum_condition_number is not None:
        valid = valid & (maximum_condition_number_ <= maximum_condition_number)
    report = MetricValidationReport(
        valid=valid,
        finite=finite,
        maximum_asymmetry=maximum_asymmetry_,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_condition_number=maximum_condition_number_,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Metric validation failed: "
            f"finite={bool(jax.device_get(finite))}, "
            f"maximum_asymmetry={float(jax.device_get(maximum_asymmetry_))}, "
            f"minimum_eigenvalue={float(jax.device_get(minimum_eigenvalue))}, "
            "maximum_condition_number="
            f"{float(jax.device_get(maximum_condition_number_))}."
        )
    return report
