#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._laguerre import RadialLaguerrePlan
from ._wigner import WignerTransformPlan


class WignerLaguerrePlan(StrictModule, NonTrainableState):
    """Separable Laguerre-Wigner transform on the radial-SO(3) product."""

    radial: RadialLaguerrePlan
    wigner: WignerTransformPlan
    sample_shape: tuple[int, int, int, int]
    coefficient_shape: tuple[int, int, int, int]
    fingerprint: str
    layout_id: str

    def __init__(
        self,
        radial: RadialLaguerrePlan,
        wigner: WignerTransformPlan,
        /,
    ):
        if not isinstance(radial, RadialLaguerrePlan):
            raise TypeError("radial must be a RadialLaguerrePlan.")
        if not isinstance(wigner, WignerTransformPlan):
            raise TypeError("wigner must be a WignerTransformPlan.")
        self.radial = radial
        self.wigner = wigner
        self.sample_shape = (radial.radial_bandlimit, *wigner.sample_shape)
        self.coefficient_shape = (
            radial.radial_bandlimit,
            *wigner.coefficient_shape,
        )
        self.layout_id = canonical_fingerprint(
            {
                "kind": "wigner-laguerre-layout-v1",
                "radial": radial.layout_id,
                "wigner": wigner.layout_id,
                "axes": ("p", "n", "ell", "m"),
            }
        )
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "wigner-laguerre-plan-v1",
                "radial": radial.transform_id,
                "wigner": wigner.transform_id,
                "sample_axes": ("p", "gamma", "beta", "alpha"),
                "coefficient_axes": ("p", "n", "ell", "m"),
            }
        )

    @property
    def transform_id(self) -> str:
        """Identity of the product sampling theorem."""
        return self.fingerprint

    @property
    def execution_id(self) -> str:
        """Identity of the concrete radial and Wigner executions."""
        return canonical_fingerprint(
            {
                "kind": "wigner-laguerre-execution-v1",
                "transform": self.transform_id,
                "radial": self.radial.execution_id,
                "wigner": self.wigner.execution_id,
            }
        )

    @property
    def precompute_bytes(self) -> int:
        """Bytes retained by the two component transforms."""
        return self.radial.precompute_bytes + self.wigner.precompute_bytes

    def analysis(self, values: ArrayLike, /) -> Array:
        """Transform radial-SO(3) samples to padded ``(p, n, ell, m)`` modes."""
        array = jnp.asarray(values)
        scalar = (
            array.ndim >= 4
            and tuple(int(size) for size in array.shape[-4:]) == self.sample_shape
        )
        channel = (
            array.ndim >= 5
            and tuple(int(size) for size in array.shape[-5:-1]) == self.sample_shape
        )
        if not scalar and not channel:
            raise ValueError(
                "Wigner-Laguerre analysis expects (..., p, n_gamma, n_beta, "
                "n_alpha) or (..., p, n_gamma, n_beta, n_alpha, channels)."
            )
        wigner_coefficients = self.wigner.analysis(array)
        if scalar:
            radial_last = jnp.moveaxis(wigner_coefficients, -4, -1)
            radial_coefficients = self.radial.analysis(radial_last)
            return jnp.moveaxis(radial_coefficients, -1, -4)
        radial_last = jnp.moveaxis(wigner_coefficients, -5, -2)
        radial_coefficients = self.radial.analysis(radial_last)
        return jnp.moveaxis(radial_coefficients, -2, -5)

    def synthesis(self, coefficients: ArrayLike, /) -> Array:
        """Transform padded ``(p, n, ell, m)`` modes to radial-SO(3) samples."""
        array = jnp.asarray(coefficients)
        scalar = (
            array.ndim >= 4
            and tuple(int(size) for size in array.shape[-4:]) == self.coefficient_shape
        )
        channel = (
            array.ndim >= 5
            and tuple(int(size) for size in array.shape[-5:-1]) == self.coefficient_shape
        )
        if not scalar and not channel:
            raise ValueError(
                "Wigner-Laguerre synthesis expects (..., p, 2*N-1, L, "
                "2*L-1) or (..., p, 2*N-1, L, 2*L-1, channels)."
            )
        if scalar:
            radial_last = jnp.moveaxis(array, -4, -1)
            radial_values = self.radial.synthesis(radial_last)
            wigner_coefficients = jnp.moveaxis(radial_values, -1, -4)
        else:
            radial_last = jnp.moveaxis(array, -5, -2)
            radial_values = self.radial.synthesis(radial_last)
            wigner_coefficients = jnp.moveaxis(radial_values, -2, -5)
        return self.wigner.synthesis(wigner_coefficients)


__all__ = ["WignerLaguerrePlan"]
