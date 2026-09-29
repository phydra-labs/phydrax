#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax import Array
from jax.typing import ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .linalg import (
    AbstractLinearOperator,
    LinearSystem,
    solve,
    TriangularLinearOperator,
)
from .typing import PRNGKey


if TYPE_CHECKING:
    from .observation import (
        CholeskyCovarianceAction,
        CoordinateLayout,
        PrecisionCovarianceAction,
    )
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

    def __init__(self, variance: ArrayLike, layout: CoordinateLayout, /) -> None:
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

    def sample(self, key: PRNGKey, /) -> Array:
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
    ) -> None:
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

    def sample(self, key: PRNGKey, /) -> Array:
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
    ) -> None:
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
    ) -> None:
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

    def __init__(self, spectrum: ArrayLike, layout: CoordinateLayout, /) -> None:
        values = np.asarray(spectrum, dtype=np.float64)
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
) -> (
    DiagonalCovarianceAction | CholeskyCovarianceAction | LowRankDiagonalCovarianceAction
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


def _active_coordinates(active: ArrayLike, layout: CoordinateLayout, /) -> np.ndarray:
    mask = np.asarray(active)
    if mask.dtype != np.bool_ or mask.shape != (layout.size,):
        raise ValueError("active must be a boolean mask over the covariance layout.")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            "Covariance restriction requires at least one active coordinate."
        )
    return indices


def _restrict_cholesky(
    covariance: CholeskyCovarianceAction,
    indices: np.ndarray,
    layout: CoordinateLayout,
    /,
) -> CholeskyCovarianceAction:
    from .observation import CholeskyCovarianceAction

    # Sigma_AA = L_A L_A^T = R^T R for the QR factorization L_A^T = Q R, which
    # factors the marginal without squaring the condition number of L_A.
    rows = np.asarray(covariance.lower_cholesky)[indices, :]
    upper = np.linalg.qr(rows.T, mode="r")
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0).astype(upper.dtype)
    return CholeskyCovarianceAction((signs[:, None] * upper).T, layout)


def _restrict_precision(
    covariance: PrecisionCovarianceAction,
    indices: np.ndarray,
    layout: CoordinateLayout,
    /,
) -> PrecisionCovarianceAction:
    from .observation import PrecisionCovarianceAction

    # Subsetting a precision conditions on the inactive coordinates. The
    # marginal precision is the Schur complement P_AA - P_AB P_BB^-1 P_BA, and
    # log det Sigma_AA = log det Sigma + log det P_BB. With P_BB = L L^T and
    # W = L^-1 P_BA the complement is P_AA - W^T W; the product is symmetrized
    # because a rounded Gram product is symmetric only up to its entries' ulp.
    precision = np.asarray(covariance.precision)
    inactive = np.setdiff1d(np.arange(precision.shape[0]), indices)
    active_block = precision[np.ix_(indices, indices)]
    coupling = precision[np.ix_(indices, inactive)]
    inactive_block = precision[np.ix_(inactive, inactive)]
    try:
        factor = np.linalg.cholesky(inactive_block)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Inactive precision block must be positive definite for exact marginalization."
        ) from error
    whitened = scipy.linalg.solve_triangular(factor, coupling.T, lower=True)
    marginal = active_block - whitened.T @ whitened
    marginal = 0.5 * (marginal + marginal.T)
    logdet_inactive = 2.0 * np.sum(np.log(np.diag(factor)))
    logdet = np.asarray(covariance.logdet_covariance) + logdet_inactive
    return PrecisionCovarianceAction(marginal, logdet, layout)


def restrict_observation_covariance(
    covariance: ObservationCovarianceAction
    | CholeskyCovarianceAction
    | PrecisionCovarianceAction,
    active: ArrayLike,
    /,
) -> ObservationCovarianceAction | CholeskyCovarianceAction | PrecisionCovarianceAction:
    """Return the exact Gaussian marginal covariance on active layout coordinates.

    `active` is a host boolean mask over `covariance.layout`; the restricted
    layout keeps the active labels in their original order. Diagonal and
    diagonal-plus-low-rank marginals keep their structure, a dense Cholesky
    marginal is refactored, and a dense precision marginal uses the Schur
    complement. Separable Kronecker, stationary circulant, and matrix-free
    precision operators have no structure-preserving marginal on an arbitrary
    subset and are refused unless every coordinate is active; declare the
    active-set covariance explicitly instead.
    """
    from .observation import (
        CholeskyCovarianceAction,
        CoordinateLayout,
        PrecisionCovarianceAction,
    )

    if not isinstance(
        covariance,
        (
            DiagonalCovarianceAction,
            LowRankDiagonalCovarianceAction,
            CholeskyCovarianceAction,
            PrecisionCovarianceAction,
            KroneckerCholeskyCovarianceAction,
            CirculantCovarianceAction,
            PrecisionOperatorCovarianceAction,
        ),
    ):
        raise TypeError("covariance must be a supported observation covariance.")
    indices = _active_coordinates(active, covariance.layout)
    if indices.size == covariance.layout.size:
        return covariance
    labels = covariance.layout.labels
    layout = CoordinateLayout(tuple(labels[index] for index in indices))
    match covariance:
        case DiagonalCovarianceAction():
            return DiagonalCovarianceAction(
                np.asarray(covariance.variance)[indices], layout
            )
        case LowRankDiagonalCovarianceAction():
            return LowRankDiagonalCovarianceAction(
                np.asarray(covariance.variance)[indices],
                np.asarray(covariance.factors)[indices, :],
                layout,
            )
        case CholeskyCovarianceAction():
            return _restrict_cholesky(covariance, indices, layout)
        case PrecisionCovarianceAction():
            return _restrict_precision(covariance, indices, layout)
        case KroneckerCholeskyCovarianceAction() | CirculantCovarianceAction():
            raise ValueError(
                "A partial restriction destroys separable or stationary covariance "
                "structure and would require dense materialization; declare the "
                "active-set covariance explicitly."
            )
        case PrecisionOperatorCovarianceAction():
            raise ValueError(
                "A matrix-free precision operator has no prepared Schur complement "
                "for a partial restriction; declare the active-set covariance "
                "explicitly."
            )
        case _:
            assert_never(covariance)


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
    "restrict_observation_covariance",
]
