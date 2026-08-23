#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


def canonical_commutation_matrix(mode_count: int, /, *, dtype=float) -> Array:
    modes = int(mode_count)
    if modes < 1:
        raise ValueError("mode_count must be positive.")
    block = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=dtype)
    return jnp.kron(jnp.eye(modes, dtype=dtype), block)


class BosonicGaussianState(StrictModule):
    mean: Array
    covariance: Array
    symplectic_form: Array
    uncertainty_margin: Array
    symmetry_residual: Array
    symplectic_eigenvalues: Array
    purity: Array
    valid: Array
    mode_count: int = eqx.field(static=True)
    hbar: float = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        hbar: float = 1.0,
        tolerance: float = 1e-9,
    ):
        mean_ = jnp.asarray(mean)
        covariance_ = jnp.asarray(covariance, dtype=mean_.dtype)
        if mean_.ndim != 1 or mean_.shape[0] % 2:
            raise ValueError("Gaussian mean must have even vector dimension.")
        if covariance_.shape != (mean_.shape[0], mean_.shape[0]):
            raise ValueError("Gaussian covariance shape must match the mean dimension.")
        modes = mean_.shape[0] // 2
        omega = canonical_commutation_matrix(modes, dtype=mean_.dtype)
        symmetric = 0.5 * (covariance_ + covariance_.T)
        uncertainty = symmetric.astype(complex) + 0.5j * float(hbar) * omega
        margin = jnp.min(jnp.linalg.eigvalsh(uncertainty))
        eigenvalues = jnp.linalg.eigvals(1j * omega @ symmetric)
        symplectic = jnp.sort(jnp.abs(eigenvalues))[::2]
        determinant = jnp.linalg.det(2.0 * symmetric / float(hbar))
        purity = 1.0 / jnp.sqrt(jnp.maximum(determinant, 1.0))
        residual = jnp.max(jnp.abs(covariance_ - covariance_.T))
        self.mean = mean_
        self.covariance = symmetric
        self.symplectic_form = omega
        self.uncertainty_margin = margin
        self.symmetry_residual = residual
        self.symplectic_eigenvalues = symplectic
        self.purity = purity
        self.valid = (
            jnp.all(jnp.isfinite(mean_))
            & jnp.all(jnp.isfinite(covariance_))
            & (residual <= tolerance)
            & (margin >= -tolerance)
        )
        self.mode_count = modes
        self.hbar = float(hbar)


class BosonicGaussianChannel(StrictModule):
    x: Array
    y: Array
    displacement: Array
    cp_margin: Array
    valid: Array
    mode_count: int = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        displacement: ArrayLike,
        /,
        *,
        channel_id: str,
        hbar: float = 1.0,
        tolerance: float = 1e-9,
    ):
        x_ = jnp.asarray(x)
        y_ = jnp.asarray(y, dtype=x_.dtype)
        displacement_ = jnp.asarray(displacement, dtype=x_.dtype)
        if x_.ndim != 2 or x_.shape[0] != x_.shape[1] or x_.shape[0] % 2:
            raise ValueError("Gaussian channel X must be an even square matrix.")
        if y_.shape != x_.shape or displacement_.shape != (x_.shape[0],):
            raise ValueError("Gaussian channel Y/displacement shapes do not match X.")
        modes = x_.shape[0] // 2
        omega = canonical_commutation_matrix(modes, dtype=x_.dtype)
        cp_matrix = 0.5 * (y_ + y_.T).astype(complex) + 0.5j * float(hbar) * (
            omega - x_ @ omega @ x_.T
        )
        margin = jnp.min(jnp.linalg.eigvalsh(cp_matrix))
        self.x = x_
        self.y = 0.5 * (y_ + y_.T)
        self.displacement = displacement_
        self.cp_margin = margin
        self.valid = jnp.all(jnp.isfinite(cp_matrix)) & (margin >= -tolerance)
        self.mode_count = modes
        self.channel_id = str(channel_id)

    def apply(self, state: BosonicGaussianState, /) -> BosonicGaussianState:
        if state.mode_count != self.mode_count:
            raise ValueError("Gaussian state and channel mode counts differ.")
        return BosonicGaussianState(
            self.x @ state.mean + self.displacement,
            self.x @ state.covariance @ self.x.T + self.y,
            hbar=state.hbar,
        )

    def compose(self, after: BosonicGaussianChannel, /) -> BosonicGaussianChannel:
        if after.mode_count != self.mode_count:
            raise ValueError("Gaussian channel mode counts differ.")
        return BosonicGaussianChannel(
            after.x @ self.x,
            after.x @ self.y @ after.x.T + after.y,
            after.x @ self.displacement + after.displacement,
            channel_id=f"{after.channel_id}∘{self.channel_id}",
        )


__all__ = [
    "BosonicGaussianChannel",
    "BosonicGaussianState",
    "canonical_commutation_matrix",
]
