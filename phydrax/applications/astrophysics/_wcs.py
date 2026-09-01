#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._observation_status import AstrophysicsObservationStatus


def _polynomial_2d(coefficients: Array, x: Array, y: Array, /) -> Array:
    value = jnp.asarray(0.0, dtype=x.dtype)
    for i in range(int(coefficients.shape[0])):
        for j in range(int(coefficients.shape[1])):
            value = value + coefficients[i, j] * x**i * y**j
    return value


class WcsResult(StrictModule):
    coordinates: Array
    residual: Array
    iterations: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class TangentSipWcsPlan(StrictModule, NonTrainableState):
    reference_sky: Array
    reference_pixel: Array
    cd_matrix: Array
    sip_a: Array
    sip_b: Array
    inverse_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_sky,
        reference_pixel,
        cd_matrix,
        sip_a,
        sip_b,
        /,
        *,
        inverse_iterations=12,
        tolerance=1.0e-11,
        wcs_id="tan-sip",
    ):
        sky = np.asarray(reference_sky, dtype=float)
        pixel = np.asarray(reference_pixel, dtype=float)
        cd = np.asarray(cd_matrix, dtype=float)
        a = np.asarray(sip_a, dtype=float)
        b = np.asarray(sip_b, dtype=float)
        if (
            sky.shape != (2,)
            or pixel.shape != (2,)
            or cd.shape != (2, 2)
            or a.ndim != 2
            or b.shape != a.shape
            or np.any(~np.isfinite(cd))
        ):
            raise ValueError("TAN-SIP arrays are invalid.")
        determinant = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
        if determinant == 0.0:
            raise ValueError("WCS CD matrix is singular.")
        self.reference_sky = jnp.asarray(sky)
        self.reference_pixel = jnp.asarray(pixel)
        self.cd_matrix = jnp.asarray(cd)
        self.sip_a = jnp.asarray(a)
        self.sip_b = jnp.asarray(b)
        self.inverse_iterations = int(inverse_iterations)
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {"kind": "tan-sip-wcs", "wcs_id": str(wcs_id), "order": int(a.shape[0] - 1)}
        )

    def _sky_to_tangent(self, sky: Array, /) -> Array:
        ra, dec = sky
        ra0, dec0 = self.reference_sky
        delta = ra - ra0
        denominator = jnp.sin(dec) * jnp.sin(dec0) + jnp.cos(dec) * jnp.cos(
            dec0
        ) * jnp.cos(delta)
        xi = jnp.cos(dec) * jnp.sin(delta) / denominator
        eta = (
            jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(delta)
        ) / denominator
        return jnp.asarray((xi, eta))

    def _tangent_to_sky(self, tangent: Array, /) -> Array:
        xi, eta = tangent
        ra0, dec0 = self.reference_sky
        rho = jnp.sqrt(xi * xi + eta * eta)
        angle = jnp.arctan(rho)
        sine, cosine = jnp.sin(angle), jnp.cos(angle)
        safe_rho = jnp.where(rho > 0.0, rho, 1.0)
        dec = jnp.arcsin(
            jnp.clip(
                cosine * jnp.sin(dec0) + eta * sine * jnp.cos(dec0) / safe_rho, -1.0, 1.0
            )
        )
        ra = ra0 + jnp.arctan2(
            xi * sine, rho * jnp.cos(dec0) * cosine - eta * jnp.sin(dec0) * sine
        )
        return jnp.where(
            rho > 0.0, jnp.asarray((jnp.mod(ra, 2.0 * jnp.pi), dec)), self.reference_sky
        )

    def world_to_pixel(self, sky: ArrayLike, /) -> WcsResult:
        value = jnp.asarray(sky)
        tangent = self._sky_to_tangent(value)
        determinant = (
            self.cd_matrix[0, 0] * self.cd_matrix[1, 1]
            - self.cd_matrix[0, 1] * self.cd_matrix[1, 0]
        )
        inverse = (
            jnp.asarray(
                (
                    (self.cd_matrix[1, 1], -self.cd_matrix[0, 1]),
                    (-self.cd_matrix[1, 0], self.cd_matrix[0, 0]),
                )
            )
            / determinant
        )
        undistorted = inverse @ tangent
        distorted = undistorted + jnp.asarray(
            (
                _polynomial_2d(self.sip_a, undistorted[0], undistorted[1]),
                _polynomial_2d(self.sip_b, undistorted[0], undistorted[1]),
            )
        )
        pixel = self.reference_pixel + distorted
        valid = jnp.all(jnp.isfinite(pixel))
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.INVALID_GEOMETRY),
        ).astype(jnp.int32)
        return WcsResult(
            pixel,
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            valid,
            status,
            self.plan_id,
        )

    def pixel_to_world(self, pixel: ArrayLike, /) -> WcsResult:
        target = jnp.asarray(pixel) - self.reference_pixel

        def residual(undistorted):
            distortion = jnp.asarray(
                (
                    _polynomial_2d(self.sip_a, undistorted[0], undistorted[1]),
                    _polynomial_2d(self.sip_b, undistorted[0], undistorted[1]),
                )
            )
            return undistorted + distortion - target

        def step(_, value):
            jacobian = jax.jacfwd(residual)(value)
            determinant = (
                jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
            )
            inverse = jnp.asarray(
                ((jacobian[1, 1], -jacobian[0, 1]), (-jacobian[1, 0], jacobian[0, 0]))
            ) / jnp.where(jnp.abs(determinant) > 0.0, determinant, 1.0)
            return value - inverse @ residual(value)

        undistorted = jax.lax.fori_loop(0, self.inverse_iterations, step, target)
        tangent = self.cd_matrix @ undistorted
        sky = self._tangent_to_sky(tangent)
        residual_norm = jnp.sqrt(jnp.sum(residual(undistorted) ** 2))
        valid = jnp.all(jnp.isfinite(sky)) & (residual_norm <= self.tolerance)
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.INVALID_GEOMETRY),
        ).astype(jnp.int32)
        return WcsResult(
            sky,
            residual_norm,
            jnp.asarray(self.inverse_iterations, dtype=jnp.int32),
            valid,
            status,
            self.plan_id,
        )


__all__ = ["TangentSipWcsPlan", "WcsResult"]
