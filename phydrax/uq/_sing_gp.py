#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..kernels import AbstractPositiveDefiniteKernel


class SINGSparseGPDrift(StrictModule):
    """Fixed-inducing, whitened independent-output sparse GP drift.

    Inducing selection is outside this object and therefore frozen.  The represented
    FITC diagonal residual is available as evidence; it is never silently added to a
    transition covariance by this callable.
    """

    inducing_points: Array
    kernels: tuple[AbstractPositiveDefiniteKernel, ...]
    whitened_mean: Array
    whitened_factor: Array
    output_mixing: Array | None
    cholesky_factors: Array
    valid: Array
    drift_id: str = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)

    def __init__(
        self,
        inducing_points: ArrayLike,
        kernels: tuple[AbstractPositiveDefiniteKernel, ...],
        whitened_mean: ArrayLike,
        whitened_factor: ArrayLike,
        /,
        *,
        output_mixing: ArrayLike | None = None,
        drift_id: str | None = None,
    ):
        points = jnp.asarray(inducing_points)
        selected = tuple(kernels)
        mean = jnp.asarray(whitened_mean)
        factor = jnp.asarray(whitened_factor)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("inducing_points must have shape (count, dimension).")
        if not selected or any(
            not isinstance(item, AbstractPositiveDefiniteKernel) for item in selected
        ):
            raise TypeError("kernels must contain positive-definite kernel objects.")
        output_count = len(selected)
        inducing_count = int(points.shape[0])
        if mean.shape != (output_count, inducing_count):
            raise ValueError("whitened_mean must have shape (outputs, inducing_count).")
        if (
            factor.ndim != 3
            or factor.shape[:2] != mean.shape
            or factor.shape[-1] < inducing_count
        ):
            raise ValueError(
                "whitened_factor must have shape (outputs, inducing_count, rank) "
                "with rank at least inducing_count."
            )
        grams = jnp.stack(tuple(kernel.matrix(points, points) for kernel in selected))
        symmetric = 0.5 * (grams + jnp.swapaxes(grams, -1, -2))
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        valid = (
            jnp.all(jnp.isfinite(points))
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(factor))
            & jnp.all(eigenvalues > 0.0)
        )
        safe = jnp.where(
            valid,
            symmetric,
            jnp.broadcast_to(
                jnp.eye(inducing_count, dtype=points.dtype), symmetric.shape
            ),
        )
        cholesky = jnp.linalg.cholesky(safe)
        mixing = None if output_mixing is None else jnp.asarray(output_mixing)
        if mixing is not None and (mixing.ndim != 2 or mixing.shape[1] != output_count):
            raise ValueError("output_mixing must have shape (state_size, outputs).")
        resolved_id = drift_id or canonical_fingerprint(
            {
                "kind": "sing-sparse-gp-drift-v1",
                "inducing_count": inducing_count,
                "input_dimension": int(points.shape[1]),
                "outputs": output_count,
                "mixed_outputs": None if mixing is None else int(mixing.shape[0]),
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("drift_id must be non-empty.")
        self.inducing_points = points
        self.kernels = selected
        self.whitened_mean = mean
        self.whitened_factor = factor
        self.output_mixing = mixing
        self.cholesky_factors = cholesky
        self.valid = valid
        self.drift_id = resolved_id
        self.approximation_kind = "fixed-inducing-fitc"

    def _features(self, value: Array, /) -> Array:
        cross = jnp.stack(
            tuple(
                kernel.matrix(value[None, :], self.inducing_points)[0]
                for kernel in self.kernels
            )
        )
        return jax.vmap(lambda factor, row: jnp.linalg.solve(factor, row))(
            self.cholesky_factors, cross
        )

    def __call__(self, state: ArrayLike, /, *, key=None) -> Array:
        del key
        value = jnp.asarray(state).reshape((-1,))
        if value.shape != (self.inducing_points.shape[1],):
            raise ValueError("state dimension must match inducing_points.")
        features = self._features(value)
        outputs = oe.contract("om,om->o", features, self.whitened_mean)
        checked = eqx.error_if(
            outputs, ~self.valid, "Sparse GP drift kernel factorization is invalid."
        )
        if self.output_mixing is None:
            return checked
        return oe.contract("so,o->s", self.output_mixing, checked)

    def fitc_variance(self, state: ArrayLike, /) -> Array:
        """Return the declared independent-output FITC predictive variance."""
        value = jnp.asarray(state).reshape((-1,))
        features = self._features(value)
        prior = jnp.stack(
            tuple(kernel.diagonal(value[None, :])[0] for kernel in self.kernels)
        )
        conditional = prior - oe.contract("om,om->o", features, features)
        posterior_projection = oe.contract(
            "om,omr,onr,on->o",
            features,
            self.whitened_factor,
            self.whitened_factor,
            features,
        )
        variance = conditional + posterior_projection
        return eqx.error_if(
            variance,
            (~self.valid) | jnp.any(variance < 0.0) | jnp.any(~jnp.isfinite(variance)),
            "Sparse GP FITC variance is invalid; no clipping or repair is applied.",
        )

    def kl_divergence(self) -> Array:
        """KL of the represented full-rank whitened Gaussian to unit normal."""
        covariances = oe.contract(
            "omr,onr->omn", self.whitened_factor, self.whitened_factor
        )
        signs, log_determinants = jnp.linalg.slogdet(covariances)
        traces = jnp.trace(covariances, axis1=-2, axis2=-1)
        squares = jnp.sum(self.whitened_mean**2, axis=-1)
        dimension = self.whitened_mean.shape[-1]
        values = 0.5 * (traces + squares - dimension - log_determinants)
        valid = self.valid & jnp.all(signs > 0.0) & jnp.all(jnp.isfinite(values))
        return eqx.error_if(
            jnp.sum(values), ~valid, "Sparse GP KL factor is rank deficient."
        )


__all__ = ["SINGSparseGPDrift"]
