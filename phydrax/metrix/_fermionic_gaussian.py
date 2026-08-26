#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import HermitianSpectrum


class FermionicGaussianState(StrictModule):
    covariance: Array
    antisymmetry_residual: Array
    mode_spectrum: Array
    physicality_margin: Array
    purity_residual: Array
    valid: Array
    mode_count: int

    def __init__(
        self,
        covariance: ArrayLike,
        /,
        *,
        tolerance: float = 1e-9,
    ):
        raw = jnp.asarray(covariance)
        if jnp.iscomplexobj(raw):
            raise TypeError("Majorana covariance must be real-valued.")
        value = raw.astype(jnp.result_type(raw, 0.0))
        if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] % 2:
            raise ValueError("Majorana covariance must be an even square matrix.")
        antisymmetric = 0.5 * (value - value.T)
        spectrum = HermitianSpectrum(1j * antisymmetric)
        mode_spectrum = jnp.sort(jnp.abs(spectrum.eigenvalues))[::2]
        margin = 1.0 - jnp.max(mode_spectrum)
        purity = jnp.max(jnp.abs(mode_spectrum - 1.0))
        residual = jnp.max(jnp.abs(value + value.T))
        self.covariance = antisymmetric
        self.antisymmetry_residual = residual
        self.mode_spectrum = mode_spectrum
        self.physicality_margin = margin
        self.purity_residual = purity
        self.valid = (
            jnp.all(jnp.isfinite(value))
            & (residual <= tolerance)
            & (margin >= -tolerance)
        )
        self.mode_count = value.shape[0] // 2

    def occupation(self, mode: int, /) -> Array:
        index = 2 * int(mode)
        if not 0 <= index < self.covariance.shape[0] - 1:
            raise ValueError("Fermionic mode index is out of range.")
        return 0.5 * (1.0 + self.covariance[index, index + 1])


__all__ = ["FermionicGaussianState"]
