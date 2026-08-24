#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._precision import FDExecutionPrecisionPolicy


def fornberg_weights(
    nodes: ArrayLike,
    evaluation_point: float,
    derivative_order: int,
    /,
) -> np.ndarray:
    """Stable Fornberg recursion for arbitrary distinct one-dimensional nodes."""
    coordinates = np.asarray(nodes, dtype=float).reshape((-1,))
    point = float(evaluation_point)
    derivative = int(derivative_order)
    if coordinates.size == 0 or derivative < 0 or derivative >= coordinates.size:
        raise ValueError("Fornberg nodes must outnumber the derivative order.")
    if np.any(~np.isfinite(coordinates)) or not np.isfinite(point):
        raise ValueError("Fornberg nodes and evaluation point must be finite.")
    if np.unique(coordinates).size != coordinates.size:
        raise ValueError("Fornberg nodes must be distinct.")
    count = int(coordinates.size)
    coefficients = np.zeros((count, derivative + 1), dtype=float)
    coefficients[0, 0] = 1.0
    c1 = 1.0
    c4 = coordinates[0] - point
    for i in range(1, count):
        maximum = min(i, derivative)
        c2 = 1.0
        c5 = c4
        c4 = coordinates[i] - point
        for j in range(i):
            c3 = coordinates[i] - coordinates[j]
            c2 *= c3
            if j == i - 1:
                for order in range(maximum, 0, -1):
                    coefficients[i, order] = (
                        c1
                        * (
                            order * coefficients[i - 1, order - 1]
                            - c5 * coefficients[i - 1, order]
                        )
                        / c2
                    )
                coefficients[i, 0] = -c1 * c5 * coefficients[i - 1, 0] / c2
            for order in range(maximum, 0, -1):
                coefficients[j, order] = (
                    c4 * coefficients[j, order] - order * coefficients[j, order - 1]
                ) / c3
            coefficients[j, 0] = c4 * coefficients[j, 0] / c3
        c1 = c2
    return coefficients[:, derivative]


class StencilCoefficientPlan(StrictModule, NonTrainableState):
    """Prepared derivative weights and auditable polynomial moments."""

    nodes: Array
    evaluation_point: float = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    accuracy_order: int = eqx.field(static=True)
    weights: Array
    moment_residuals: Array
    condition_estimate: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        evaluation_point: float,
        derivative_order: int,
        accuracy_order: int,
        /,
        *,
        weights: ArrayLike | None = None,
        residual_tolerance: float = 1e-9,
        plan_id: str | None = None,
        precision: FDExecutionPrecisionPolicy | None = None,
    ):
        precision_ = FDExecutionPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, FDExecutionPrecisionPolicy):
            raise TypeError("precision must be an FDExecutionPrecisionPolicy.")
        certification_dtype = np.dtype(precision_.certification_dtype)
        coefficient_dtype = np.dtype(precision_.coefficient_dtype)
        coordinates = np.asarray(nodes, dtype=certification_dtype).reshape((-1,))
        point = certification_dtype.type(evaluation_point)
        derivative = int(derivative_order)
        accuracy = int(accuracy_order)
        if derivative <= 0 or accuracy <= 0:
            raise ValueError("Derivative and accuracy orders must be positive.")
        if coordinates.size <= derivative:
            raise ValueError("Stencil node count must exceed derivative_order.")
        resolved_weights = (
            fornberg_weights(coordinates, point, derivative)
            if weights is None
            else np.asarray(weights, dtype=certification_dtype).reshape((-1,))
        )
        if resolved_weights.shape != coordinates.shape or np.any(
            ~np.isfinite(resolved_weights)
        ):
            raise ValueError("Stencil weights must be finite and align with nodes.")
        offsets = coordinates - point
        required_degree = derivative + accuracy - 1
        maximum_degree = max(coordinates.size - 1, required_degree)
        residuals = np.asarray(
            [
                np.sum(
                    resolved_weights * offsets**degree,
                    dtype=certification_dtype,
                )
                - (
                    certification_dtype.type(factorial(derivative))
                    if degree == derivative
                    else certification_dtype.type(0.0)
                )
                for degree in range(maximum_degree + 1)
            ],
            dtype=certification_dtype,
        )
        tolerance = float(residual_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        scale = max(1.0, float(factorial(derivative)))
        if np.max(np.abs(residuals[: required_degree + 1])) > tolerance * scale:
            raise ValueError("Stencil weights violate their declared moment conditions.")
        vandermonde = np.vander(offsets, N=coordinates.size, increasing=True).T
        condition = float(np.linalg.cond(vandermonde))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "stencil-coefficients",
                    "nodes": array_tree_fingerprint(coordinates),
                    "evaluation_point": point,
                    "derivative_order": derivative,
                    "accuracy_order": accuracy,
                    "weights": array_tree_fingerprint(resolved_weights),
                    "coefficient_dtype": precision_.coefficient_dtype,
                    "certification_dtype": precision_.certification_dtype,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.nodes = jnp.asarray(coordinates)
        self.evaluation_point = float(point)
        self.derivative_order = derivative
        self.accuracy_order = accuracy
        self.weights = jnp.asarray(resolved_weights, dtype=coefficient_dtype)
        self.moment_residuals = jnp.asarray(residuals)
        self.condition_estimate = condition
        self.plan_id = identifier


__all__ = ["StencilCoefficientPlan", "fornberg_weights"]
