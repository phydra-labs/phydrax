#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


class CoordinateLayout(StrictModule, NonTrainableState):
    labels: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, labels: tuple[str, ...], /):
        labels_ = tuple(str(label).strip() for label in labels)
        if (
            not labels_
            or len(set(labels_)) != len(labels_)
            or any(not label for label in labels_)
        ):
            raise ValueError("Coordinate labels must be non-empty and unique.")
        self.labels = labels_
        self.layout_id = canonical_fingerprint(
            {"kind": "coordinate-layout", "labels": list(labels_)}
        )

    @property
    def size(self) -> int:
        return len(self.labels)


class TheoryVector(StrictModule):
    values: Array
    layout: CoordinateLayout
    product_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, layout: CoordinateLayout, product_id: str, /):
        value = jnp.asarray(values)
        if value.shape != (layout.size,):
            raise ValueError("Observation product must match its coordinate layout.")
        product_id_ = str(product_id).strip()
        if not product_id_:
            raise ValueError("Observation product ID must be non-empty.")
        self.values = value
        self.layout = layout
        self.product_id = product_id_


ObservationProduct = TheoryVector


class LinearObservationPlan(StrictModule, NonTrainableState):
    matrix: Array
    source: CoordinateLayout
    target: CoordinateLayout
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        source: CoordinateLayout,
        target: CoordinateLayout,
        /,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(matrix))
        if values.shape != (target.size, source.size):
            raise ValueError("Observation matrix shape must match layouts.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Observation matrix must be finite.",
        )
        self.matrix = values
        self.source = source
        self.target = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-observation-plan",
                "source": source.layout_id,
                "target": target.layout_id,
                "matrix": array_tree_fingerprint(values),
            }
        )

    def apply(self, theory: TheoryVector, /) -> TheoryVector:
        if theory.layout.layout_id != self.source.layout_id:
            raise ValueError("Observation product layout does not match response source.")
        values = contract("oi,i->o", self.matrix, theory.values)
        return TheoryVector(
            values,
            self.target,
            canonical_fingerprint(
                {
                    "kind": "observed-product",
                    "parent": theory.product_id,
                    "plan": self.plan_id,
                }
            ),
        )


class PrecisionCovarianceAction(StrictModule, NonTrainableState):
    precision: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        precision: ArrayLike,
        logdet_covariance: ArrayLike,
        layout: CoordinateLayout,
        /,
    ):
        matrix = jax.lax.stop_gradient(jnp.asarray(precision))
        logdet = jax.lax.stop_gradient(jnp.asarray(logdet_covariance, dtype=matrix.dtype))
        if matrix.shape != (layout.size, layout.size) or logdet.shape != ():
            raise ValueError("Precision/covariance determinant shapes are invalid.")
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix))
            | ~jnp.isfinite(logdet)
            | jnp.any(jnp.abs(matrix - matrix.T) > 1.0e-10)
            | jnp.any(jnp.diag(matrix) <= 0.0),
            "Precision action must be finite, symmetric, and positive on the diagonal.",
        )
        self.precision = matrix
        self.logdet_covariance = logdet
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "precision-covariance-action",
                "layout": layout.layout_id,
                "precision": array_tree_fingerprint(matrix),
                "logdet_covariance": array_tree_fingerprint(logdet),
            }
        )

    def quadratic(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual, dtype=self.precision.dtype)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        return contract("i,ij,j->", value, self.precision, value)


class CholeskyCovarianceAction(StrictModule, NonTrainableState):
    lower_cholesky: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(self, lower_cholesky: ArrayLike, layout: CoordinateLayout, /):
        cholesky = jax.lax.stop_gradient(jnp.asarray(lower_cholesky))
        if cholesky.shape != (layout.size, layout.size):
            raise ValueError("Covariance Cholesky shape must match its layout.")
        cholesky = eqx.error_if(
            cholesky,
            jnp.any(~jnp.isfinite(cholesky))
            | jnp.any(jnp.triu(cholesky, 1) != 0.0)
            | jnp.any(jnp.diag(cholesky) <= 0.0),
            "Covariance Cholesky must be finite, lower triangular, and positive.",
        )
        self.lower_cholesky = cholesky
        self.logdet_covariance = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "cholesky-covariance-action",
                "layout": layout.layout_id,
                "cholesky": array_tree_fingerprint(cholesky),
            }
        )

    def whiten(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual, dtype=self.lower_cholesky.dtype)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        return jsp.linalg.solve_triangular(self.lower_cholesky, value, lower=True)

    def quadratic(self, residual: ArrayLike, /) -> Array:
        whitened = self.whiten(residual)
        return jnp.sum(whitened * whitened)


CovarianceAction = PrecisionCovarianceAction | CholeskyCovarianceAction


class CorrelatedGaussianResult(StrictModule):
    residual: Array
    quadratic: Array
    log_probability: Array
    finite: Array
    successful: Array


class CorrelatedGaussianPlan(StrictModule, NonTrainableState):
    data: Array
    observation: LinearObservationPlan
    covariance: CovarianceAction
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        data: ArrayLike,
        observation: LinearObservationPlan,
        covariance: CovarianceAction,
        /,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(data))
        if values.shape != (observation.target.size,):
            raise ValueError("Observed data must match response target layout.")
        if covariance.layout.layout_id != observation.target.layout_id:
            raise ValueError("Covariance and response target layouts disagree.")
        values = eqx.error_if(
            values, jnp.any(~jnp.isfinite(values)), "Observed data must be finite."
        )
        self.data = values
        self.observation = observation
        self.covariance = covariance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "correlated-gaussian-plan",
                "observation": observation.plan_id,
                "covariance": covariance.action_id,
                "data": array_tree_fingerprint(values),
            }
        )

    def evaluate(self, theory: TheoryVector, /) -> CorrelatedGaussianResult:
        observed = self.observation.apply(theory)
        residual = self.data - observed.values
        quadratic = self.covariance.quadratic(residual)
        size = jnp.asarray(self.data.size, dtype=residual.dtype)
        log_probability = -0.5 * (
            quadratic
            + self.covariance.logdet_covariance
            + size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=residual.dtype))
        )
        finite = jnp.all(jnp.isfinite(residual)) & jnp.isfinite(log_probability)
        return CorrelatedGaussianResult(
            residual, quadratic, log_probability, finite, finite
        )


__all__ = [
    "CholeskyCovarianceAction",
    "CoordinateLayout",
    "CorrelatedGaussianPlan",
    "CorrelatedGaussianResult",
    "CovarianceAction",
    "LinearObservationPlan",
    "ObservationProduct",
    "PrecisionCovarianceAction",
    "TheoryVector",
]
