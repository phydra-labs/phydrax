#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_INV_FOUR_PI = 1.0 / (4.0 * math.pi)
_SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)
_GAUSSIAN_NORMALIZATION_3D = (2.0 * math.pi) ** -1.5


class GaussianErfKernelEvaluation3D(StrictModule):
    """One Gaussian-blob contribution at broadcast-compatible displacements."""

    velocity: Array
    velocity_gradient: Array
    vorticity: Array
    finite: Array
    coincident: Array


class GaussianErfVortexKernel3D(StrictModule, NonTrainableState):
    """Stable free-space Biot--Savart kernel for isotropic Gaussian blobs.

    ``core_radius`` is the Gaussian standard deviation and ``strength`` is the
    integrated vector vorticity.  For ``r = target - source`` this evaluates

    ``u = (Gamma x r) [erf(q/sqrt(2)) - sqrt(2/pi) q exp(-q^2/2)] /(4 pi r^3)``.

    The near-origin branch evaluates the analytic power series of the radial
    coefficient and its derivative, so coincident points never form ``0/0``.
    """

    series_threshold: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    core_model: str = eqx.field(static=True)
    core_radius_convention: str = eqx.field(static=True)
    strength_semantics: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, *, series_threshold: float = 0.25):
        threshold = float(series_threshold)
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("series_threshold must be finite and lie in (0, 1].")
        self.series_threshold = threshold
        self.dimension = 3
        self.core_model = "gaussian-erf"
        self.core_radius_convention = "isotropic-standard-deviation"
        self.strength_semantics = "integrated-vorticity-vector"
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "gaussian-erf-vortex-kernel-3d-v1",
                "series_threshold": threshold,
                "core_radius_convention": self.core_radius_convention,
                "strength_semantics": self.strength_semantics,
            }
        )

    def evaluate(
        self,
        displacement: ArrayLike,
        strength: ArrayLike,
        core_radius: ArrayLike,
        /,
    ) -> GaussianErfKernelEvaluation3D:
        """Evaluate velocity, analytic target gradient, and Gaussian vorticity."""

        offset = jnp.asarray(displacement)
        circulation = jnp.asarray(strength)
        radius = jnp.asarray(core_radius)
        if offset.ndim < 1 or offset.shape[-1] != 3:
            raise ValueError("displacement must have trailing shape (3,).")
        if circulation.ndim < 1 or circulation.shape[-1] != 3:
            raise ValueError("strength must have trailing shape (3,).")
        leading_shape = jnp.broadcast_shapes(
            offset.shape[:-1], circulation.shape[:-1], radius.shape
        )

        dtype = jnp.result_type(offset.dtype, circulation.dtype, radius.dtype, float)
        offset = jnp.broadcast_to(offset.astype(dtype), leading_shape + (3,))
        circulation = jnp.broadcast_to(circulation.astype(dtype), leading_shape + (3,))
        radius = jnp.broadcast_to(radius.astype(dtype), leading_shape)
        radius = eqx.error_if(
            radius,
            jnp.any(~jnp.isfinite(radius) | (radius <= 0.0)),
            "Gaussian core radii must be finite and strictly positive.",
        )

        scaled = offset / radius[..., None]
        squared_scaled_distance = jnp.sum(scaled * scaled, axis=-1)
        z = 0.5 * squared_scaled_distance
        use_series = z <= self.series_threshold

        # P(z) is defined by F(r, sigma) = sqrt(2/pi) P(z) / sigma^3,
        # where z = r^2/(2 sigma^2).  Its entire series is
        # P(z) = sum_m (-z)^m / (m! (2m + 3)).
        p_series = 1.0 / 10800.0
        p_series = -1.0 / 1560.0 + z * p_series
        p_series = 1.0 / 264.0 + z * p_series
        p_series = -1.0 / 54.0 + z * p_series
        p_series = 1.0 / 14.0 + z * p_series
        p_series = -1.0 / 5.0 + z * p_series
        p_series = 1.0 / 3.0 + z * p_series

        dp_series = 1.0 / 1800.0
        dp_series = -1.0 / 312.0 + z * dp_series
        dp_series = 1.0 / 66.0 + z * dp_series
        dp_series = -1.0 / 18.0 + z * dp_series
        dp_series = 1.0 / 7.0 + z * dp_series
        dp_series = -1.0 / 5.0 + z * dp_series

        # Keep the unselected exact branch finite at z=0.  This is important
        # for reverse-mode differentiation through jnp.where.
        safe_z = jnp.where(use_series, jnp.asarray(self.series_threshold, dtype), z)
        root_z = jnp.sqrt(safe_z)
        gaussian_tail = jnp.exp(-safe_z)
        g = jsp.erf(root_z) - (2.0 / math.sqrt(math.pi)) * root_z * gaussian_tail
        p_exact = math.sqrt(math.pi) * g / (4.0 * safe_z * root_z)
        dp_exact = 0.5 * gaussian_tail / safe_z - 3.0 * math.sqrt(math.pi) * g / (
            8.0 * safe_z * safe_z * root_z
        )
        p = jnp.where(use_series, p_series, p_exact)
        dp = jnp.where(use_series, dp_series, dp_exact)

        inverse_radius = 1.0 / radius
        inverse_radius_cubed = inverse_radius**3
        radial = _SQRT_TWO_OVER_PI * inverse_radius_cubed * p
        radial_derivative_over_distance = (
            _SQRT_TWO_OVER_PI * inverse_radius_cubed * inverse_radius**2 * dp
        )

        cross = jnp.cross(circulation, offset)
        velocity = _INV_FOUR_PI * radial[..., None] * cross

        gx, gy, gz = (
            circulation[..., 0],
            circulation[..., 1],
            circulation[..., 2],
        )
        zeros = jnp.zeros_like(gx)
        cross_matrix = jnp.stack(
            (
                jnp.stack((zeros, -gz, gy), axis=-1),
                jnp.stack((gz, zeros, -gx), axis=-1),
                jnp.stack((-gy, gx, zeros), axis=-1),
            ),
            axis=-2,
        )
        velocity_gradient = _INV_FOUR_PI * (
            radial[..., None, None] * cross_matrix
            + radial_derivative_over_distance[..., None, None]
            * cross[..., :, None]
            * offset[..., None, :]
        )

        density = _GAUSSIAN_NORMALIZATION_3D * inverse_radius_cubed * jnp.exp(-z)
        vorticity = density[..., None] * circulation
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(velocity_gradient))
            & jnp.all(jnp.isfinite(vorticity))
        )
        coincident = jnp.all(offset == 0.0, axis=-1)
        return GaussianErfKernelEvaluation3D(
            velocity, velocity_gradient, vorticity, finite, coincident
        )

    def __call__(
        self,
        displacement: ArrayLike,
        strength: ArrayLike,
        core_radius: ArrayLike,
        /,
    ) -> GaussianErfKernelEvaluation3D:
        return self.evaluate(displacement, strength, core_radius)


__all__ = ["GaussianErfKernelEvaluation3D", "GaussianErfVortexKernel3D"]
