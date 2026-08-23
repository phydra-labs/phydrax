#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._metric import RiemannianMetric


class MetricValidationReport(StrictModule):
    """Aggregate diagnostics from explicit metric validation points."""

    valid: Array
    finite: Array
    maximum_asymmetry: Array
    minimum_eigenvalue: Array
    maximum_condition_number: Array
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        *,
        valid: Array,
        finite: Array,
        maximum_asymmetry: Array,
        minimum_eigenvalue: Array,
        maximum_condition_number: Array,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        evidence = (
            GeometryPrecisionPolicy().evidence_for(minimum_eigenvalue)
            if precision_evidence is None
            else precision_evidence
        )
        if not isinstance(evidence, PrecisionEvidenceEnvelope):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.maximum_asymmetry = jnp.asarray(maximum_asymmetry)
        self.minimum_eigenvalue = jnp.asarray(minimum_eigenvalue)
        self.maximum_condition_number = jnp.asarray(maximum_condition_number)
        self.precision_evidence = evidence


def validate_metric(
    metric: RiemannianMetric,
    points: ArrayLike,
    /,
    *,
    symmetry_tolerance: float = 1e-8,
    eigenvalue_floor: float = 0.0,
    maximum_condition_number: float | None = None,
    raise_on_error: bool = True,
    precision: GeometryPrecisionPolicy | None = None,
) -> MetricValidationReport:
    """Validate, without modifying, a Riemannian metric at representative points."""

    if symmetry_tolerance < 0.0:
        raise ValueError("symmetry_tolerance must be non-negative.")
    if maximum_condition_number is not None and maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must be greater than one.")
    precision_ = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    point_array = jnp.asarray(points)
    precision_.validate_coordinates(point_array)
    matrix = precision_.compute(metric(precision_.compute(point_array)))
    finite = jnp.all(jnp.isfinite(matrix))
    asymmetry = matrix - jnp.swapaxes(matrix, -1, -2)
    maximum_asymmetry_ = precision_.decision(jnp.max(jnp.abs(asymmetry)))
    symmetric_matrix = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    eigenvalues = jnp.linalg.eigvalsh(symmetric_matrix)
    minimum_eigenvalue = precision_.decision(jnp.min(eigenvalues))
    condition_numbers = jnp.max(eigenvalues, axis=-1) / jnp.min(
        eigenvalues,
        axis=-1,
    )
    maximum_condition_number_ = precision_.decision(jnp.max(condition_numbers))
    valid = (
        finite
        & (
            maximum_asymmetry_
            <= jnp.asarray(symmetry_tolerance, dtype=maximum_asymmetry_.dtype)
        )
        & (
            minimum_eigenvalue
            > jnp.asarray(eigenvalue_floor, dtype=minimum_eigenvalue.dtype)
        )
    )
    if maximum_condition_number is not None:
        valid = valid & (
            maximum_condition_number_
            <= jnp.asarray(
                maximum_condition_number,
                dtype=maximum_condition_number_.dtype,
            )
        )
    report = MetricValidationReport(
        valid=valid,
        finite=finite,
        maximum_asymmetry=maximum_asymmetry_,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_condition_number=maximum_condition_number_,
        precision_evidence=precision_.evidence_for(point_array),
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
