#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.axes as cx
from phydrax.kernels import AbstractPositiveDefiniteKernel, kernel_features
from phydrax.linalg import (
    LinearSolveStatus,
    LinearSystem,
    solve,
    TriangularLinearOperator,
)

from .._strict import StrictModule
from ..typing import parse
from ._gp_backend import _as_kernel_inputs
from ._gp_scalar import (
    _as_vector,
    ExactGaussianProcessFactor,
    FiniteFeatureGaussianProcessFactor,
    SparseGaussianProcessFactor,
)


type _MeanRepresentation = Literal["exact", "finite-feature", "fitc"]


@final
class GaussianProcessMeanPredictor(StrictModule):
    """Result-only posterior mean for fixed covariance and residual data.

    Prepare a new artifact when covariance parameters or residuals change. Numerical
    leaves, including the kernel, remain dynamic under JAX transformations. The
    artifact retains neither an observation factor nor posterior covariance.

    ``max_workspace_bytes`` bounds explicit query kernel/feature blocks, query
    rows, and block means. Caller-owned inputs, the returned mean, and intrinsic
    kernel-evaluation scratch are not included. Query blocks are recomputed in
    reverse mode rather than retaining a query-by-training cross covariance.
    """

    kernel: AbstractPositiveDefiniteKernel
    reference_points: Array | None
    coefficients: Array
    solve_status: Array
    representation: _MeanRepresentation = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        kernel: AbstractPositiveDefiniteKernel,
        reference_points: Array | None,
        coefficients: Array,
        solve_status: Array,
        representation: _MeanRepresentation,
        max_workspace_bytes: int,
    ) -> None:
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel.")
        representation = parse(representation, _MeanRepresentation, "representation")
        if not isinstance(max_workspace_bytes, int) or isinstance(
            max_workspace_bytes, bool
        ):
            raise TypeError("max_workspace_bytes must be an integer.")
        if max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive.")
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("Predictor coefficients must be a nonempty vector.")
        match representation:
            case "finite-feature":
                if reference_points is not None:
                    raise ValueError(
                        "Finite-feature means must not retain training points."
                    )
            case "exact" | "fitc":
                if (
                    reference_points is None
                    or reference_points.ndim != kernel.input_ndim + 1
                    or reference_points.shape[0] != coefficients.size
                ):
                    raise ValueError(
                        "Predictor reference points must align with coefficients."
                    )
            case invalid:
                assert_never(invalid)
        bytes_per_row = (coefficients.size + 1) * coefficients.dtype.itemsize
        if max_workspace_bytes < bytes_per_row:
            raise ValueError(
                "Mean-prediction workspace cannot hold one query row and its kernel/features."
            )
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(solve_status != int(LinearSolveStatus.SUCCESS))
            | jnp.any(~jnp.isfinite(coefficients)),
            "GP mean preparation requires a successful finite covariance solve.",
        )
        self.kernel = kernel
        self.reference_points = reference_points
        self.coefficients = coefficients
        self.solve_status = solve_status
        self.representation = representation
        self.max_workspace_bytes = max_workspace_bytes

    @property
    def retained_storage_elements(self) -> int:
        """All retained numerical elements, including kernel parameters and status."""
        return sum(leaf.size for leaf in jax.tree.leaves(self) if eqx.is_array(leaf))

    @property
    def retained_storage_bytes(self) -> int:
        """All retained numerical bytes, including kernel parameters and status."""
        return sum(
            leaf.size * leaf.dtype.itemsize
            for leaf in jax.tree.leaves(self)
            if eqx.is_array(leaf)
        )

    def __call__(self, query_points: ArrayLike | cx.AxisArray, /) -> Array:
        """Evaluate scalar means with bounded, checkpointed query-row batches."""
        query = _as_kernel_inputs(
            query_points.data if isinstance(query_points, cx.AxisArray) else query_points,
            kernel=self.kernel,
            name="query_points",
        )
        if query.shape[0] == 0:
            return jnp.empty((0,), dtype=self.coefficients.dtype)
        bytes_per_row = (
            self.coefficients.size + prod(query.shape[1:]) + 1
        ) * self.coefficients.dtype.itemsize
        batch_size = min(query.shape[0], self.max_workspace_bytes // bytes_per_row)
        if batch_size < 1:
            raise ValueError(
                "Mean-prediction workspace cannot hold one query row and its kernel/features."
            )

        def one(point: Array) -> Array:
            point = eqx.error_if(
                point, jnp.any(~jnp.isfinite(point)), "GP query points must be finite."
            )
            if self.reference_points is None:
                row = kernel_features(self.kernel, point[None, ...])[0]
            else:
                row = self.kernel.matrix(point[None, ...], self.reference_points)[0]
            row = eqx.error_if(
                row,
                jnp.any(~jnp.isfinite(row)),
                "GP prediction requires finite kernel/features.",
            )
            return row @ self.coefficients

        return jax.lax.map(jax.checkpoint(one), query, batch_size=batch_size)


def prepare_gaussian_process_mean(
    factor: ExactGaussianProcessFactor
    | FiniteFeatureGaussianProcessFactor
    | SparseGaussianProcessFactor,
    residual: ArrayLike,
    /,
    *,
    max_workspace_bytes: int = 16 * 1024 * 1024,
) -> GaussianProcessMeanPredictor:
    """Solve once, retaining only the data needed for changing-query means.

    Exact means retain training points and alpha; finite-feature means retain
    weight coefficients only; FITC means retain inducing points and coefficients
    already transformed through the inducing Cholesky factor. Preparation itself
    uses the supplied factor's solve resources, not the query workspace budget.
    """
    if not isinstance(
        factor,
        (
            ExactGaussianProcessFactor,
            FiniteFeatureGaussianProcessFactor,
            SparseGaussianProcessFactor,
        ),
    ):
        raise TypeError("factor must be an exact, finite-feature, or FITC GP factor.")
    values = _as_vector(residual, name="GP residual")
    if values.shape[0] != factor.observation_points.shape[0]:
        raise ValueError("GP residual must align with factor observations.")
    values = eqx.error_if(
        values, jnp.any(~jnp.isfinite(values)), "GP residual must be finite."
    )
    if isinstance(factor, ExactGaussianProcessFactor):
        result = factor.factorization.solve(values)
        coefficients = factor.factorization.operator.source.flatten(result.value)
        status = result.status
        reference_points = factor.observation_points
        representation: _MeanRepresentation = "exact"
    else:
        diagonal = eqx.error_if(
            factor.diagonal,
            jnp.any(~jnp.isfinite(factor.diagonal))
            | jnp.any(factor.diagonal <= 0)
            | jnp.any(~jnp.isfinite(factor.features)),
            "GP mean preparation requires finite features and positive observation diagonal.",
        )
        right = factor.features.T @ (values / diagonal)
        coefficients, status = _solve_cholesky(factor.correction_cholesky, right)
        if isinstance(factor, SparseGaussianProcessFactor):
            coefficients, inducing_status = _solve_triangular(
                factor.inducing_cholesky, coefficients, transpose=True
            )
            status = jnp.maximum(status, inducing_status)
            reference_points = factor.inducing_points
            representation = "fitc"
        else:
            reference_points = None
            representation = "finite-feature"
    return GaussianProcessMeanPredictor(
        kernel=factor.state.kernel,
        reference_points=reference_points,
        coefficients=coefficients,
        solve_status=status,
        representation=representation,
        max_workspace_bytes=max_workspace_bytes,
    )


def _solve_cholesky(lower: Array, right: Array, /) -> tuple[Array, Array]:
    intermediate, forward_status = _solve_triangular(lower, right, transpose=False)
    value, backward_status = _solve_triangular(lower, intermediate, transpose=True)
    return value, jnp.maximum(forward_status, backward_status)


def _solve_triangular(
    lower: Array, right: Array, /, *, transpose: bool
) -> tuple[Array, Array]:
    lower = eqx.error_if(
        lower,
        jnp.any(~jnp.isfinite(lower)) | jnp.any(jnp.diag(lower) <= 0),
        "GP mean preparation requires a valid positive Cholesky factor.",
    )
    operator = TriangularLinearOperator(
        lower.T if transpose else lower, lower=not transpose
    )
    result = solve(LinearSystem(operator), right)
    value = operator.source.flatten(result.value)
    return value, result.status


__all__ = ["GaussianProcessMeanPredictor", "prepare_gaussian_process_mean"]
