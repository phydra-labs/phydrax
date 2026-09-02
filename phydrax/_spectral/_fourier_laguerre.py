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
from ._spherical import SphericalHarmonicPlan


class FourierLaguerrePlan(StrictModule, NonTrainableState):
    """Separable Fourier-Laguerre transform on the radial-spherical product."""

    radial: RadialLaguerrePlan
    angular: SphericalHarmonicPlan
    sample_shape: tuple[int, int, int]
    coefficient_shape: tuple[int, int, int]
    fingerprint: str
    layout_id: str

    def __init__(
        self,
        radial: RadialLaguerrePlan,
        angular: SphericalHarmonicPlan,
        /,
    ):
        if not isinstance(radial, RadialLaguerrePlan):
            raise TypeError("radial must be a RadialLaguerrePlan.")
        if not isinstance(angular, SphericalHarmonicPlan):
            raise TypeError("angular must be a SphericalHarmonicPlan.")
        self.radial = radial
        self.angular = angular
        self.sample_shape = (radial.radial_bandlimit, *angular.sample_shape)
        self.coefficient_shape = (
            radial.radial_bandlimit,
            *angular.coefficient_shape,
        )
        self.layout_id = canonical_fingerprint(
            {
                "kind": "fourier-laguerre-layout-v1",
                "radial": radial.layout_id,
                "angular": angular.layout_id,
                "axes": ("p", "ell", "m"),
            }
        )
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "fourier-laguerre-plan-v1",
                "radial": radial.transform_id,
                "angular": angular.transform_id,
                "sample_axes": ("p", "theta", "phi"),
                "coefficient_axes": ("p", "ell", "m"),
            }
        )

    @property
    def transform_id(self) -> str:
        """Identity of the composed sampling theorem."""
        return self.fingerprint

    @property
    def execution_id(self) -> str:
        """Identity of the concrete radial and angular executions."""
        return canonical_fingerprint(
            {
                "kind": "fourier-laguerre-execution-v1",
                "transform": self.transform_id,
                "radial": self.radial.execution_id,
                "angular": self.angular.execution_id,
            }
        )

    @property
    def precompute_bytes(self) -> int:
        """Bytes retained by the two component transforms."""
        return self.radial.precompute_bytes + self.angular.precompute_bytes

    def analysis(self, values: ArrayLike, /) -> Array:
        """Transform radial-spherical samples to padded ``(p, ell, m)`` modes."""
        array = jnp.asarray(values)
        scalar = (
            array.ndim >= 3
            and tuple(int(size) for size in array.shape[-3:]) == self.sample_shape
        )
        channel = (
            array.ndim >= 4
            and tuple(int(size) for size in array.shape[-4:-1]) == self.sample_shape
        )
        if not scalar and not channel:
            raise ValueError(
                "Fourier-Laguerre analysis expects (..., p, n_theta, n_phi) or "
                "(..., p, n_theta, n_phi, channels)."
            )
        angular_coefficients = self.angular.analysis(array)
        if scalar:
            radial_last = jnp.moveaxis(angular_coefficients, -3, -1)
            radial_coefficients = self.radial.analysis(radial_last)
            return jnp.moveaxis(radial_coefficients, -1, -3)
        radial_last = jnp.moveaxis(angular_coefficients, -4, -2)
        radial_coefficients = self.radial.analysis(radial_last)
        return jnp.moveaxis(radial_coefficients, -2, -4)

    def synthesis(self, coefficients: ArrayLike, /) -> Array:
        """Transform padded ``(p, ell, m)`` modes to radial-spherical samples."""
        array = jnp.asarray(coefficients)
        scalar = (
            array.ndim >= 3
            and tuple(int(size) for size in array.shape[-3:]) == self.coefficient_shape
        )
        channel = (
            array.ndim >= 4
            and tuple(int(size) for size in array.shape[-4:-1]) == self.coefficient_shape
        )
        if not scalar and not channel:
            raise ValueError(
                "Fourier-Laguerre synthesis expects (..., p, bandlimit, "
                "2*bandlimit-1) or (..., p, bandlimit, 2*bandlimit-1, channels)."
            )
        if scalar:
            radial_last = jnp.moveaxis(array, -3, -1)
            radial_values = self.radial.synthesis(radial_last)
            angular_coefficients = jnp.moveaxis(radial_values, -1, -3)
        else:
            radial_last = jnp.moveaxis(array, -4, -2)
            radial_values = self.radial.synthesis(radial_last)
            angular_coefficients = jnp.moveaxis(radial_values, -2, -4)
        return self.angular.synthesis(angular_coefficients)


__all__ = ["FourierLaguerrePlan"]
