#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity star-shaped spherical-spectral two-surfaces."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein
import phydrax.linalg as la

from ..._fingerprint import canonical_fingerprint
from ..._spectral._spherical import (
    SphericalExecution,
    SphericalHarmonicPlan,
    SphericalSampling,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_METRIC_SOLVE = la.SmallLinearSolvePlan(3)


class SphericalSpectralSurface(StrictModule):
    """One star-shaped surface with a fixed spherical-harmonic capacity.

    ``coefficients`` use the full complex ``(ell, m)`` layout even though the
    represented radius is real.  Retaining the full layout makes spin-raising
    derivatives exact and avoids shape changes in compiled MOTS solves.
    """

    coefficients: Array
    center: Array
    plan_id: str = eqx.field(static=True)


class SurfaceGeometryEvidence(StrictModule):
    """Sampled intrinsic/extrinsic embedding data and metric validity evidence."""

    radius: Array
    points: Array
    theta_tangent: Array
    phi_tangent: Array
    induced_metric: Array
    inverse_spatial_metric: Array
    outward_normal_covector: Array
    outward_normal: Array
    area_density_per_solid_angle: Array
    area_weights: Array
    area: Array
    areal_radius: Array
    metric_inversion_valid: Array
    finite: Array
    physically_valid: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class SphericalSurfacePlan(StrictModule, NonTrainableState):
    """Prepared scalar and spin-one transforms for a fixed-capacity surface.

    The represented embedding is ``x(theta, phi) = center + r e_r``.  The
    spin-raising identity ``eth r = -(d_theta + i csc(theta)d_phi) r`` supplies
    spectral first derivatives without finite-difference pole stencils.
    """

    scalar_transform: SphericalHarmonicPlan
    gradient_transform: SphericalHarmonicPlan
    unit_radial: Array
    unit_theta: Array
    unit_phi: Array
    solid_angle_weights: Array
    bandlimit: int = eqx.field(static=True)
    sampling: SphericalSampling = eqx.field(static=True)
    sample_shape: tuple[int, int] = eqx.field(static=True)
    coefficient_shape: tuple[int, int] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bandlimit: int,
        /,
        *,
        sampling: SphericalSampling = "gl",
        execution: SphericalExecution = "recursive",
        max_precompute_bytes: int = 512 * 1024**2,
    ):
        bandlimit_ = int(bandlimit)
        if bandlimit_ < 2:
            raise ValueError("Surface bandlimit must be at least two.")
        scalar = SphericalHarmonicPlan(
            bandlimit_,
            sampling=sampling,
            spin=0,
            reality=False,
            execution=execution,
            max_precompute_bytes=max_precompute_bytes,
        )
        gradient = SphericalHarmonicPlan(
            bandlimit_,
            sampling=sampling,
            spin=1,
            reality=False,
            execution=execution,
            max_precompute_bytes=max_precompute_bytes,
        )
        theta = np.asarray(scalar.theta)[:, None]
        phi = np.asarray(scalar.phi)[None, :]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sample_shape = scalar.sample_shape
        unit_radial = np.stack(
            np.broadcast_arrays(
                sin_theta * cos_phi,
                sin_theta * sin_phi,
                cos_theta * np.ones_like(phi),
            ),
            axis=-1,
        )
        unit_theta = np.stack(
            np.broadcast_arrays(
                cos_theta * cos_phi,
                cos_theta * sin_phi,
                -sin_theta * np.ones_like(phi),
            ),
            axis=-1,
        )
        unit_phi = np.stack(
            np.broadcast_arrays(
                -np.ones_like(theta) * sin_phi,
                np.ones_like(theta) * cos_phi,
                np.zeros(sample_shape),
            ),
            axis=-1,
        )
        solid_angle_weights = np.asarray(scalar.theta_quadrature_weights)[:, None] * np.asarray(
            scalar.phi_quadrature_weights
        )[None, :]
        self.scalar_transform = scalar
        self.gradient_transform = gradient
        self.unit_radial = jnp.asarray(unit_radial)
        self.unit_theta = jnp.asarray(unit_theta)
        self.unit_phi = jnp.asarray(unit_phi)
        self.solid_angle_weights = jnp.asarray(solid_angle_weights)
        self.bandlimit = bandlimit_
        self.sampling = scalar.sampling
        self.sample_shape = sample_shape
        self.coefficient_shape = scalar.coefficient_shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spherical-spectral-surface-plan",
                "scalar_transform": scalar.transform_id,
                "gradient_transform": gradient.transform_id,
            }
        )

    def from_samples(
        self,
        radius: ArrayLike,
        /,
        *,
        center: ArrayLike = (0.0, 0.0, 0.0),
    ) -> SphericalSpectralSurface:
        values = jnp.asarray(radius)
        if values.shape != self.sample_shape:
            raise ValueError("Surface radius samples do not match the fixed capacity.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        center_ = jnp.asarray(center, dtype=values.real.dtype)
        if center_.shape != (3,):
            raise ValueError("Surface center must have shape (3,).")
        coefficients = self.scalar_transform.analysis(values)
        return SphericalSpectralSurface(coefficients, center_, self.plan_id)

    def constant(
        self,
        radius: ArrayLike,
        /,
        *,
        center: ArrayLike = (0.0, 0.0, 0.0),
    ) -> SphericalSpectralSurface:
        value = jnp.asarray(radius).reshape(())
        return self.from_samples(
            jnp.full(self.sample_shape, value, dtype=value.dtype), center=center
        )

    def radius(self, surface: SphericalSpectralSurface, /) -> Array:
        self._validate_surface(surface)
        return jnp.real(self.scalar_transform.synthesis(surface.coefficients))

    def angular_derivatives(
        self, surface: SphericalSpectralSurface, /
    ) -> tuple[Array, Array]:
        """Return ``d_theta r`` and ``csc(theta) d_phi r`` on the sample grid."""
        self._validate_surface(surface)
        degree = jnp.arange(self.bandlimit, dtype=self.unit_radial.dtype)[:, None]
        eth_multiplier = jnp.sqrt(degree * (degree + 1.0))
        eth = self.gradient_transform.synthesis(
            surface.coefficients * eth_multiplier
        )
        return -jnp.real(eth), -jnp.imag(eth)

    def mean_radius(self, surface: SphericalSpectralSurface, /) -> Array:
        radius = self.radius(surface)
        return jnp.sum(self.solid_angle_weights * radius) / (4.0 * jnp.pi)

    def geometry(
        self,
        surface: SphericalSpectralSurface,
        spatial_metric: ArrayLike | None = None,
        /,
    ) -> SurfaceGeometryEvidence:
        """Evaluate the induced geometry in a sampled positive-definite 3-metric."""
        radius = self.radius(surface)
        radius_theta, radius_phi_over_sin = self.angular_derivatives(surface)
        theta_tangent = (
            radius_theta[..., None] * self.unit_radial
            + radius[..., None] * self.unit_theta
        )
        sin_theta = jnp.sin(self.scalar_transform.theta)[:, None]
        phi_tangent = sin_theta[..., None] * (
            radius_phi_over_sin[..., None] * self.unit_radial
            + radius[..., None] * self.unit_phi
        )
        points = surface.center + radius[..., None] * self.unit_radial
        if spatial_metric is None:
            metric = jnp.broadcast_to(
                jnp.eye(3, dtype=radius.dtype), self.sample_shape + (3, 3)
            )
        else:
            metric = jnp.asarray(spatial_metric, dtype=radius.dtype)
            if metric.shape != self.sample_shape + (3, 3):
                raise ValueError("Sampled spatial metric shape does not match the surface.")
        inverse_result = la.inverse_small_linear(_METRIC_SOLVE, metric)
        inverse_metric = inverse_result.value
        q_theta_theta = ein.contract(
            "...i,...ij,...j->...", theta_tangent, metric, theta_tangent
        )
        q_theta_phi = ein.contract(
            "...i,...ij,...j->...", theta_tangent, metric, phi_tangent
        )
        q_phi_phi = ein.contract(
            "...i,...ij,...j->...", phi_tangent, metric, phi_tangent
        )
        induced = jnp.stack(
            (
                jnp.stack((q_theta_theta, q_theta_phi), axis=-1),
                jnp.stack((q_theta_phi, q_phi_phi), axis=-1),
            ),
            axis=-2,
        )
        determinant = q_theta_theta * q_phi_phi - q_theta_phi**2
        safe_sine = jnp.where(sin_theta != 0.0, jnp.abs(sin_theta), 1.0)
        area_density = jnp.sqrt(jnp.maximum(determinant, 0.0)) / safe_sine
        area_weights = self.solid_angle_weights * area_density
        area = jnp.sum(area_weights)
        areal_radius = jnp.sqrt(jnp.maximum(area, 0.0) / (4.0 * jnp.pi))

        normal_density_covector = jnp.cross(theta_tangent, phi_tangent) / safe_sine[..., None]
        normal_squared = ein.contract(
            "...i,...ij,...j->...",
            normal_density_covector,
            inverse_metric,
            normal_density_covector,
        )
        normal_norm = jnp.sqrt(jnp.maximum(normal_squared, 0.0))
        safe_normal_norm = jnp.where(normal_norm > 0.0, normal_norm, 1.0)
        normal_covector = normal_density_covector / safe_normal_norm[..., None]
        normal = ein.contract("...ij,...j->...i", inverse_metric, normal_covector)
        finite = (
            jnp.all(jnp.isfinite(radius))
            & jnp.all(jnp.isfinite(metric))
            & jnp.all(jnp.isfinite(induced))
            & jnp.all(jnp.isfinite(normal))
            & jnp.isfinite(area)
        )
        positive_induced = (q_theta_theta > 0.0) & (determinant > 0.0)
        physically_valid = (
            finite
            & jnp.all(inverse_result.successful)
            & jnp.all(radius > 0.0)
            & jnp.all(positive_induced)
            & (area > 0.0)
        )
        return SurfaceGeometryEvidence(
            radius,
            points,
            theta_tangent,
            phi_tangent,
            induced,
            inverse_metric,
            normal_covector,
            normal,
            area_density,
            area_weights,
            area,
            areal_radius,
            jnp.all(inverse_result.successful),
            finite,
            physically_valid,
            finite & jnp.all(inverse_result.successful),
            self.plan_id,
        )

    def _validate_surface(self, surface: SphericalSpectralSurface, /) -> None:
        if not isinstance(surface, SphericalSpectralSurface):
            raise TypeError("surface must be a SphericalSpectralSurface.")
        if surface.plan_id != self.plan_id:
            raise ValueError("Surface and spherical plan identities differ.")
        if surface.coefficients.shape != self.coefficient_shape or surface.center.shape != (3,):
            raise ValueError("Surface arrays do not match the fixed-capacity plan.")


def schwarzschild_isotropic_spatial_metric(
    points: ArrayLike,
    mass: ArrayLike,
    /,
) -> Array:
    """Time-symmetric Schwarzschild spatial metric in isotropic Cartesian coordinates."""
    coordinates = jnp.asarray(points)
    mass_ = jnp.asarray(mass, dtype=coordinates.dtype).reshape(())
    radius = jnp.sqrt(ein.contract("...i,...i->...", coordinates, coordinates))
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    conformal_factor = 1.0 + mass_ / (2.0 * safe_radius)
    return conformal_factor[..., None, None] ** 4 * jnp.eye(3, dtype=coordinates.dtype)


__all__ = [
    "SphericalSpectralSurface",
    "SphericalSurfacePlan",
    "SurfaceGeometryEvidence",
    "schwarzschild_isotropic_spatial_metric",
]
