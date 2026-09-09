#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .linalg import (
    AbstractLinearOperator,
    LinearSystem,
    solve,
    TriangularLinearOperator,
)


if TYPE_CHECKING:
    from .observation import CoordinateLayout
    from .uq import AbstractCovariance


def _triangular_solve(
    operator: TriangularLinearOperator, right_hand_side: ArrayLike, /
) -> Array:
    values = jnp.asarray(right_hand_side)
    if values.ndim == 0 or values.shape[0] != operator.target.size:
        raise ValueError("Triangular covariance right-hand side has wrong leading size.")
    result_dtype = jnp.result_type(values.dtype, operator.matrix.dtype)
    if result_dtype != operator.matrix.dtype:
        raise TypeError(
            "Triangular covariance solve would change operator coordinate dtype."
        )
    values = values.astype(operator.matrix.dtype)
    flat = values.reshape((operator.target.size, -1))

    def solve_column(column: Array) -> Array:
        result = solve(LinearSystem(operator), column)
        return eqx.error_if(
            result.value,
            ~result.successful,
            "Native triangular covariance solve failed.",
        )

    solved = jax.vmap(solve_column, in_axes=1, out_axes=1)(flat)
    return solved.reshape(values.shape)


class DiagonalCovarianceAction(StrictModule, NonTrainableState):
    variance: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(self, variance: ArrayLike, layout: CoordinateLayout, /):
        values = jax.lax.stop_gradient(jnp.asarray(variance))
        if values.shape != (layout.size,) or jnp.issubdtype(
            values.dtype, jnp.complexfloating
        ):
            raise ValueError(
                "Diagonal observation variance must be a real layout-sized vector."
            )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values <= 0),
            "Observation variances must be finite and strictly positive.",
        )
        self.variance = values
        self.logdet_covariance = jnp.sum(jnp.log(values))
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "diagonal-observation-covariance",
                "layout": layout.layout_id,
                "variance": array_tree_fingerprint(values),
            }
        )

    def whiten(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        if value.shape != self.variance.shape:
            raise ValueError("Residual must match covariance layout.")
        return value / jnp.sqrt(self.variance)

    def quadratic(self, residual: ArrayLike, /) -> Array:
        whitened = self.whiten(residual)
        return jnp.real(jnp.vdot(whitened, whitened))

    def sample(self, key: PRNGKeyArray, /) -> Array:
        normal = jax.random.normal(key, self.variance.shape, dtype=self.variance.dtype)
        return jnp.sqrt(self.variance) * normal


class LowRankDiagonalCovarianceAction(StrictModule, NonTrainableState):
    """Positive diagonal plus low-rank covariance without dense n-by-n storage."""

    variance: Array
    factors: Array
    woodbury_lower: TriangularLinearOperator
    woodbury_upper: TriangularLinearOperator
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        variance: ArrayLike,
        factors: ArrayLike,
        layout: CoordinateLayout,
        /,
    ):
        diagonal = np.asarray(variance)
        low_rank = np.asarray(factors)
        if (
            diagonal.shape != (layout.size,)
            or low_rank.ndim != 2
            or low_rank.shape[0] != layout.size
        ):
            raise ValueError(
                "Low-rank covariance requires variance (n,) and factors (n, r)."
            )
        if (
            np.iscomplexobj(diagonal)
            or np.iscomplexobj(low_rank)
            or np.any(~np.isfinite(diagonal))
            or np.any(diagonal <= 0)
            or np.any(~np.isfinite(low_rank))
        ):
            raise ValueError(
                "Low-rank covariance diagonal/factors must be finite and diagonal positive."
            )
        inverse = 1.0 / diagonal
        gram = (
            np.eye(low_rank.shape[1], dtype=low_rank.dtype)
            + (low_rank.conj().T * inverse) @ low_rank
        )
        cholesky = np.linalg.cholesky(gram)
        logdet = np.sum(np.log(diagonal)) + 2.0 * np.sum(
            np.log(np.real(np.diag(cholesky)))
        )
        self.variance = jnp.asarray(diagonal)
        self.factors = jnp.asarray(low_rank)
        lower = jnp.asarray(cholesky)
        upper = jnp.asarray(cholesky.conj().T)
        self.woodbury_lower = TriangularLinearOperator(
            lower,
            lower=True,
            operator_id=canonical_fingerprint(
                {"kind": "woodbury-lower-factor", "factor": cholesky}
            ),
        )
        self.woodbury_upper = TriangularLinearOperator(
            upper,
            lower=False,
            operator_id=canonical_fingerprint(
                {"kind": "woodbury-upper-factor", "factor": cholesky.conj().T}
            ),
        )
        self.logdet_covariance = jnp.asarray(logdet)
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "low-rank-diagonal-observation-covariance",
                "layout": layout.layout_id,
                "variance": diagonal,
                "factors": low_rank,
            }
        )

    def solve(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        if value.ndim == 0 or value.shape[0] != self.variance.size:
            raise ValueError("Residual must have covariance layout leading size.")
        if jnp.iscomplexobj(value):
            raise TypeError("Low-rank diagonal covariance requires real residuals.")
        inverse = value / self.variance.reshape((-1,) + (1,) * (value.ndim - 1))
        projected = self.factors.conj().T @ inverse
        lower = _triangular_solve(self.woodbury_lower, projected)
        correction = _triangular_solve(self.woodbury_upper, lower)
        return inverse - (self.factors / self.variance[:, None]) @ correction

    def quadratic(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        if value.shape != self.variance.shape:
            raise ValueError("Residual must match covariance layout.")
        return jnp.real(jnp.vdot(value, self.solve(value)))

    def sample(self, key: PRNGKeyArray, /) -> Array:
        diagonal_key, factor_key = jax.random.split(key)
        real_dtype = self.variance.dtype
        diagonal = jax.random.normal(diagonal_key, self.variance.shape, dtype=real_dtype)
        rank = self.factors.shape[1]
        factor = jax.random.normal(factor_key, (rank,), dtype=real_dtype)
        return jnp.sqrt(self.variance) * diagonal + self.factors @ factor


class PrecisionOperatorCovarianceAction(StrictModule, NonTrainableState):
    """Prepared precision operator with an independently supplied exact logdet."""

    precision: AbstractLinearOperator
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        precision: AbstractLinearOperator,
        logdet_covariance: ArrayLike,
        layout: CoordinateLayout,
        /,
    ):
        if not isinstance(precision, AbstractLinearOperator):
            raise TypeError("Precision action must be a native AbstractLinearOperator.")
        if (
            not precision.source.compatible(precision.target)
            or not precision.properties.certifies("self_adjoint")
            or not precision.properties.certifies("positive_definite")
        ):
            raise ValueError(
                "Precision covariance requires a certified self-adjoint positive-definite endomorphism."
            )
        if precision.source.size != layout.size or precision.target.size != layout.size:
            raise ValueError(
                "Precision operator spaces must match observation layout size."
            )
        logdet = jax.lax.stop_gradient(jnp.asarray(logdet_covariance))
        if logdet.shape != ():
            raise ValueError("Precision covariance log determinant must be scalar.")
        logdet = eqx.error_if(
            logdet, ~jnp.isfinite(logdet), "Log determinant must be finite."
        )
        self.precision = precision
        self.logdet_covariance = logdet
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "precision-operator-observation-covariance",
                "layout": layout.layout_id,
                "operator": precision.operator_id,
                "logdet": array_tree_fingerprint(logdet),
            }
        )

    def quadratic(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        structured = self.precision.source.unflatten(value)
        applied = self.precision.mv(structured)
        return jnp.real(self.precision.source.inner(structured, applied))


class KroneckerCholeskyCovarianceAction(StrictModule, NonTrainableState):
    """Exact separable covariance from small axis Cholesky factors."""

    factors: tuple[Array, ...]
    factor_operators: tuple[TriangularLinearOperator, ...]
    shape: tuple[int, ...] = eqx.field(static=True)
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        layout: CoordinateLayout,
        /,
    ):
        arrays = tuple(np.asarray(value) for value in factors)
        if not arrays or any(
            value.ndim != 2
            or value.shape[0] != value.shape[1]
            or value.shape[0] == 0
            or np.any(~np.isfinite(value))
            or np.any(np.triu(value, 1) != 0)
            or np.any(np.imag(np.diag(value)) != 0)
            or np.any(np.real(np.diag(value)) <= 0)
            for value in arrays
        ):
            raise ValueError("Kronecker factors must be finite lower Cholesky matrices.")
        common_dtype = np.result_type(*(value.dtype for value in arrays))
        arrays = tuple(value.astype(common_dtype, copy=False) for value in arrays)
        shape = tuple(value.shape[0] for value in arrays)
        if int(np.prod(shape)) != layout.size:
            raise ValueError("Kronecker factor dimensions must multiply to layout size.")
        total = int(np.prod(shape))
        logdet = sum(
            (total // value.shape[0]) * 2.0 * np.sum(np.log(np.real(np.diag(value))))
            for value in arrays
        )
        self.factors = tuple(jnp.asarray(value) for value in arrays)
        self.factor_operators = tuple(
            TriangularLinearOperator(
                factor,
                lower=True,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "kronecker-lower-factor",
                        "axis": axis,
                        "factor": factor,
                    }
                ),
            )
            for axis, factor in enumerate(arrays)
        )
        self.shape = shape
        self.logdet_covariance = jnp.asarray(logdet)
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "kronecker-cholesky-observation-covariance",
                "layout": layout.layout_id,
                "factors": arrays,
            }
        )

    def whiten(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        tensor = value.reshape(self.shape)
        for axis, operator in enumerate(self.factor_operators):
            moved = jnp.moveaxis(tensor, axis, 0)
            flat = moved.reshape((operator.source.size, -1))
            solved = _triangular_solve(operator, flat)
            tensor = jnp.moveaxis(solved.reshape(moved.shape), 0, axis)
        return tensor.reshape(-1)

    def quadratic(self, residual: ArrayLike, /) -> Array:
        whitened = self.whiten(residual)
        return jnp.real(jnp.vdot(whitened, whitened))


class CirculantCovarianceAction(StrictModule, NonTrainableState):
    """Exact periodic stationary covariance in a real Fourier basis."""

    spectrum: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(self, spectrum: ArrayLike, layout: CoordinateLayout, /):
        values = np.asarray(spectrum, dtype=float)
        expected = layout.size // 2 + 1
        if (
            values.shape != (expected,)
            or np.any(~np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "Circulant rFFT spectrum must be finite, positive, and layout-sized."
            )
        multiplicity = np.full(expected, 2.0)
        multiplicity[0] = 1.0
        if layout.size % 2 == 0:
            multiplicity[-1] = 1.0
        self.spectrum = jnp.asarray(values)
        self.logdet_covariance = jnp.asarray(np.sum(multiplicity * np.log(values)))
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "circulant-observation-covariance",
                "layout": layout.layout_id,
                "spectrum": values,
            }
        )

    def whiten(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        transformed = jnp.fft.rfft(value, norm="ortho") / jnp.sqrt(self.spectrum)
        return jnp.fft.irfft(transformed, n=self.layout.size, norm="ortho")

    def quadratic(self, residual: ArrayLike, /) -> Array:
        whitened = self.whiten(residual)
        return jnp.real(jnp.vdot(whitened, whitened))


def prepare_observation_covariance(
    covariance: AbstractCovariance,
    layout: CoordinateLayout,
    /,
    *,
    diagonal_nugget: ArrayLike | None = None,
):
    """Lower a native UQ covariance into one likelihood-capable observation action."""
    from .observation import CholeskyCovarianceAction
    from .uq import (
        CovarianceOperator,
        DenseCovariance,
        DiagonalCovariance,
        FactorCovariance,
    )

    if isinstance(covariance, DiagonalCovariance):
        leaves = jax.tree_util.tree_leaves(covariance.variance)
        if len(leaves) != 1:
            raise ValueError(
                "Observation covariance lowering requires one flat array leaf."
            )
        return DiagonalCovarianceAction(jnp.ravel(leaves[0]), layout)
    if isinstance(covariance, DenseCovariance):
        matrix = np.asarray(covariance.matrix)
        if matrix.shape != (layout.size, layout.size):
            raise ValueError("Dense UQ covariance does not match observation layout.")
        return CholeskyCovarianceAction(np.linalg.cholesky(matrix), layout)
    if isinstance(covariance, FactorCovariance):
        if diagonal_nugget is None:
            raise ValueError(
                "Factor observation covariance requires a positive diagonal nugget."
            )
        leaves = jax.tree_util.tree_leaves(covariance.factors)
        if len(leaves) != 1:
            raise ValueError("Observation factor lowering requires one flat array leaf.")
        factors = jnp.asarray(leaves[0]).reshape((covariance.rank, layout.size)).T
        return LowRankDiagonalCovarianceAction(diagonal_nugget, factors, layout)
    if isinstance(covariance, CovarianceOperator):
        raise ValueError(
            "A covariance matvec alone cannot define likelihood precision or normalization."
        )
    raise TypeError("covariance must implement native UQ AbstractCovariance.")


ObservationCovarianceAction = (
    DiagonalCovarianceAction
    | LowRankDiagonalCovarianceAction
    | PrecisionOperatorCovarianceAction
    | KroneckerCholeskyCovarianceAction
    | CirculantCovarianceAction
)


__all__ = [
    "CirculantCovarianceAction",
    "DiagonalCovarianceAction",
    "KroneckerCholeskyCovarianceAction",
    "LowRankDiagonalCovarianceAction",
    "ObservationCovarianceAction",
    "PrecisionOperatorCovarianceAction",
    "prepare_observation_covariance",
]
