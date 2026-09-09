#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._observation_covariance import (
    CirculantCovarianceAction,
    DiagonalCovarianceAction,
    KroneckerCholeskyCovarianceAction,
    LowRankDiagonalCovarianceAction,
    PrecisionOperatorCovarianceAction,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .linalg import LinearSystem, solve, TriangularLinearOperator
from .observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    PrecisionCovarianceAction,
)


class _CovarianceProtocol(Protocol):
    layout: CoordinateLayout
    action_id: str

    def quadratic(self, residual: ArrayLike, /) -> Array: ...


def _triangular_solve(
    operator: TriangularLinearOperator, right_hand_side: ArrayLike, /
) -> Array:
    values = jnp.asarray(right_hand_side)
    if values.ndim == 0 or values.shape[0] != operator.target.size:
        raise ValueError("Triangular nuisance right-hand side has wrong leading size.")
    result_dtype = jnp.result_type(values.dtype, operator.matrix.dtype)
    if result_dtype != operator.matrix.dtype:
        raise TypeError(
            "Triangular nuisance solve would change operator coordinate dtype."
        )
    values = values.astype(operator.matrix.dtype)
    flat = values.reshape((operator.target.size, -1))

    def solve_column(column: Array) -> Array:
        result = solve(LinearSystem(operator), column)
        return eqx.error_if(
            result.value,
            ~result.successful,
            "Native triangular nuisance solve failed.",
        )

    solved = jax.vmap(solve_column, in_axes=1, out_axes=1)(flat)
    return solved.reshape(values.shape)


class NuisanceProjectionResult(StrictModule):
    parameters: Array
    nuisance_prediction: Array
    residual: Array
    quadratic: Array
    normal_residual: Array
    successful: Array


def _whiten(covariance, value):
    if isinstance(
        covariance,
        (
            CholeskyCovarianceAction,
            DiagonalCovarianceAction,
            KroneckerCholeskyCovarianceAction,
            CirculantCovarianceAction,
        ),
    ):
        if value.ndim == 1:
            return covariance.whiten(value)
        return jnp.stack(
            [covariance.whiten(value[:, column]) for column in range(value.shape[1])],
            axis=1,
        )
    return None


def _precision_apply(covariance, value):
    whitened = _whiten(covariance, value)
    if whitened is not None:
        if value.ndim == 1:
            # WᴴW action is obtained without materializing W.
            _, pullback = jax.vjp(covariance.whiten, jnp.zeros_like(value))
            return pullback(whitened)[0]
        return jnp.stack(
            [
                _precision_apply(covariance, value[:, column])
                for column in range(value.shape[1])
            ],
            axis=1,
        )
    if isinstance(covariance, PrecisionCovarianceAction):
        return covariance.precision @ value
    if isinstance(covariance, PrecisionOperatorCovarianceAction):
        if value.ndim == 1:
            source = covariance.precision.source.unflatten(value)
            return covariance.precision.target.flatten(covariance.precision.mv(source))
        return jnp.stack(
            [
                _precision_apply(covariance, value[:, column])
                for column in range(value.shape[1])
            ],
            axis=1,
        )
    if isinstance(covariance, LowRankDiagonalCovarianceAction):
        return covariance.solve(value)
    raise TypeError(
        "Covariance action does not expose whitening or precision application."
    )


class LinearNuisancePlan(StrictModule, NonTrainableState):
    """Profile linear calibration/trend parameters from a correlated objective."""

    design: Array
    precision_design: Array
    prior_precision: Array
    prior_mean: Array
    normal_lower: TriangularLinearOperator
    normal_upper: TriangularLinearOperator
    covariance: _CovarianceProtocol
    layout: CoordinateLayout
    nuisance_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        design: ArrayLike,
        covariance: _CovarianceProtocol,
        layout: CoordinateLayout,
        nuisance_names: tuple[str, ...],
        /,
        *,
        prior_precision: ArrayLike | None = None,
        prior_mean: ArrayLike | None = None,
    ):
        matrix = jnp.asarray(design)
        if matrix.ndim != 2 or matrix.shape[0] != layout.size or matrix.shape[1] == 0:
            raise ValueError(
                "Nuisance design must have shape (observations, parameters)."
            )
        if bool(jnp.any(~jnp.isfinite(matrix))):
            raise ValueError("Nuisance design must be finite.")
        names = tuple(str(name).strip() for name in nuisance_names)
        if (
            len(names) != matrix.shape[1]
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Nuisance parameter names must be unique and match design columns."
            )
        if covariance.layout.layout_id != layout.layout_id:
            raise ValueError("Nuisance covariance and observation layout disagree.")
        count = matrix.shape[1]
        ridge = jnp.zeros((count, count), dtype=matrix.dtype)
        if prior_precision is not None:
            ridge = jnp.asarray(prior_precision, dtype=matrix.dtype)
            if ridge.shape != (count, count) or bool(jnp.any(~jnp.isfinite(ridge))):
                raise ValueError("Nuisance prior precision must be finite and square.")
            if not np.allclose(np.asarray(ridge), np.asarray(ridge).conj().T):
                raise ValueError("Nuisance prior precision must be Hermitian.")
            ridge_eigenvalues = np.linalg.eigvalsh(np.asarray(ridge))
            ridge_tolerance = np.finfo(ridge_eigenvalues.dtype).eps * max(
                1.0, float(np.max(np.abs(ridge_eigenvalues)))
            )
            if np.min(ridge_eigenvalues) < -ridge_tolerance:
                raise ValueError(
                    "Nuisance prior precision must be positive semidefinite."
                )
        mean = (
            jnp.zeros((count,), dtype=matrix.dtype)
            if prior_mean is None
            else jnp.asarray(prior_mean, dtype=matrix.dtype)
        )
        if mean.shape != (count,) or bool(jnp.any(~jnp.isfinite(mean))):
            raise ValueError(
                "Nuisance prior mean must match design columns and be finite."
            )
        precision_design = _precision_apply(covariance, matrix)
        normal = matrix.conj().T @ precision_design + ridge
        host_normal = np.asarray(normal)
        eigenvalues = np.linalg.eigvalsh(host_normal)
        threshold = np.finfo(eigenvalues.dtype).eps * max(
            1.0, float(np.max(np.abs(eigenvalues)))
        )
        if np.min(eigenvalues) <= threshold:
            raise ValueError(
                "Nuisance design/prior is rank deficient under the observation precision."
            )
        cholesky = np.linalg.cholesky(host_normal)
        self.design = matrix
        self.precision_design = precision_design
        self.prior_precision = ridge
        self.prior_mean = mean
        lower = jnp.asarray(cholesky)
        upper = jnp.asarray(cholesky.conj().T)
        self.normal_lower = TriangularLinearOperator(
            lower,
            lower=True,
            operator_id=canonical_fingerprint(
                {"kind": "nuisance-normal-lower", "factor": cholesky}
            ),
        )
        self.normal_upper = TriangularLinearOperator(
            upper,
            lower=False,
            operator_id=canonical_fingerprint(
                {"kind": "nuisance-normal-upper", "factor": cholesky.conj().T}
            ),
        )
        self.covariance = covariance
        self.layout = layout
        self.nuisance_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-nuisance-plan",
                "layout": layout.layout_id,
                "covariance": covariance.action_id,
                "design": array_tree_fingerprint(matrix),
                "prior_precision": array_tree_fingerprint(ridge),
                "prior_mean": array_tree_fingerprint(mean),
                "names": names,
            }
        )

    def evaluate(
        self, data: ArrayLike, nonlinear_prediction: ArrayLike, /
    ) -> NuisanceProjectionResult:
        data_ = jnp.asarray(data)
        prediction = jnp.asarray(nonlinear_prediction, dtype=data_.dtype)
        if data_.shape != (self.layout.size,) or prediction.shape != data_.shape:
            raise ValueError("Data and nonlinear prediction must match nuisance layout.")
        right = data_ - prediction
        rhs = (
            self.precision_design.conj().T @ right
            + self.prior_precision @ self.prior_mean
        )
        lower = _triangular_solve(self.normal_lower, rhs)
        parameters = _triangular_solve(self.normal_upper, lower)
        nuisance = self.design @ parameters
        residual = right - nuisance
        normal_residual = (
            self.precision_design.conj().T @ residual
            - self.prior_precision @ (parameters - self.prior_mean)
        )
        quadratic = self.covariance.quadratic(residual)
        centered = parameters - self.prior_mean
        quadratic = quadratic + jnp.real(
            jnp.vdot(centered, self.prior_precision @ centered)
        )
        finite = (
            jnp.all(jnp.isfinite(parameters))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(quadratic)
        )
        tolerance = (
            100
            * jnp.finfo(jnp.real(quadratic).dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(rhs)))
        )
        successful = finite & (jnp.max(jnp.abs(normal_residual)) <= tolerance)
        return NuisanceProjectionResult(
            parameters,
            nuisance,
            residual,
            quadratic,
            normal_residual,
            successful,
        )


__all__ = ["LinearNuisancePlan", "NuisanceProjectionResult"]
