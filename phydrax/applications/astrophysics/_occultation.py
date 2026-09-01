#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration import GaussLegendreRule
from ._observation_status import AstrophysicsObservationStatus


class PolynomialLimbDarkenedDisk(StrictModule, NonTrainableState):
    """Radial stellar intensity ``1 - sum u_n (1 - mu)^(n+1)``."""

    coefficients: Array
    normalization: Array
    model_id: str = eqx.field(static=True)

    def __init__(self, coefficients: ArrayLike, /):
        host = np.asarray(coefficients, dtype=float)
        if host.ndim != 1 or np.any(~np.isfinite(host)):
            raise ValueError(
                "Limb-darkening coefficients must be a finite rank-one array."
            )
        polynomial = np.concatenate((np.ones((1,)), -host))
        derivative = np.polynomial.polynomial.polyder(polynomial)
        roots = (
            np.polynomial.polynomial.polyroots(derivative)
            if derivative.size
            else np.empty((0,))
        )
        candidates = [0.0, 1.0]
        candidates.extend(
            float(root.real)
            for root in roots
            if abs(root.imag) <= 1.0e-12 and 0.0 < root.real < 1.0
        )
        values = np.polynomial.polynomial.polyval(np.asarray(candidates), polynomial)
        if np.any(values < -1.0e-12):
            raise ValueError("Limb-darkening law is negative on the stellar disk.")
        orders = np.arange(1, host.size + 1, dtype=float)
        radial_integral = 0.5 - np.sum(host / ((orders + 1.0) * (orders + 2.0)))
        normalization = 2.0 * np.pi * radial_integral
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("Limb-darkening law has non-positive total flux.")
        self.coefficients = jnp.asarray(host)
        self.normalization = jnp.asarray(normalization)
        self.model_id = canonical_fingerprint(
            {
                "kind": "polynomial-limb-darkened-disk",
                "coefficients": host.tolist(),
            }
        )

    def intensity(self, mu: ArrayLike, /) -> Array:
        mu_ = jnp.asarray(mu)
        power = 1.0 - mu_
        if int(self.coefficients.size) == 0:
            return jnp.ones_like(mu_)
        exponents = jnp.arange(1, int(self.coefficients.size) + 1)
        return 1.0 - jnp.sum(
            self.coefficients * power[..., None] ** exponents,
            axis=-1,
        )


class CircularOccultationResult(StrictModule):
    relative_flux: Array
    occulted_flux: Array
    branch: Array
    contact: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class CircularOccultationPlan(StrictModule, NonTrainableState):
    disk: PolynomialLimbDarkenedDisk
    radial_nodes: Array
    radial_weights: Array
    quadrature_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        disk: PolynomialLimbDarkenedDisk,
        /,
        *,
        quadrature_order: int = 192,
    ):
        if not isinstance(disk, PolynomialLimbDarkenedDisk):
            raise TypeError("disk must be a PolynomialLimbDarkenedDisk.")
        order = int(quadrature_order)
        if order < 16:
            raise ValueError("quadrature_order must be at least 16.")
        data = GaussLegendreRule(order).data()
        self.disk = disk
        self.radial_nodes = 0.5 * (data.nodes + 1.0)
        self.radial_weights = 0.5 * data.weights
        self.quadrature_order = order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "circular-occultation-plan",
                "disk": disk.model_id,
                "quadrature_order": order,
            }
        )

    def evaluate(
        self,
        projected_separation: ArrayLike,
        radius_ratio: ArrayLike,
        /,
        *,
        foreground: ArrayLike = True,
    ) -> CircularOccultationResult:
        separation, ratio, foreground_ = jnp.broadcast_arrays(
            jnp.asarray(projected_separation),
            jnp.asarray(radius_ratio),
            jnp.asarray(foreground, dtype=bool),
        )
        finite = jnp.isfinite(separation) & jnp.isfinite(ratio)
        domain = finite & (separation >= 0.0) & (ratio >= 0.0)
        b = jnp.where(domain, separation, 0.0)
        p = jnp.where(domain, ratio, 0.0)
        reference_nodes = 2.0 * self.radial_nodes - 1.0
        reference_weights = 2.0 * self.radial_weights
        full_upper = jnp.clip(p - b, 0.0, 1.0)
        full_radius = 0.5 * full_upper[..., None] * (reference_nodes + 1.0)
        full_weights = 0.5 * full_upper[..., None] * reference_weights
        full_mu = jnp.sqrt(jnp.maximum(1.0 - full_radius * full_radius, 0.0))
        full_flux = (
            2.0
            * jnp.pi
            * jnp.sum(
                full_weights * self.disk.intensity(full_mu) * full_radius,
                axis=-1,
            )
        )

        partial_lower = jnp.clip(jnp.abs(b - p), 0.0, 1.0)
        partial_upper = jnp.clip(b + p, 0.0, 1.0)
        partial_width = jnp.maximum(partial_upper - partial_lower, 0.0)
        partial_active = partial_width[..., None] > 0.0
        mapped_partial_radius = partial_lower[..., None] + 0.5 * partial_width[
            ..., None
        ] * (reference_nodes + 1.0)
        partial_radius = jnp.where(partial_active, mapped_partial_radius, 0.5)
        partial_weights = 0.5 * partial_width[..., None] * reference_weights
        partial_mu = jnp.sqrt(jnp.maximum(1.0 - partial_radius * partial_radius, 0.0))
        b_expanded = b[..., None]
        p_expanded = p[..., None]
        denominator = 2.0 * partial_radius * b_expanded
        cosine = (
            partial_radius * partial_radius
            + b_expanded * b_expanded
            - p_expanded * p_expanded
        ) / jnp.where(partial_active & (denominator > 0.0), denominator, 1.0)
        epsilon = 16.0 * jnp.finfo(cosine.dtype).eps
        safe_cosine = jnp.where(
            partial_active,
            jnp.clip(cosine, -1.0 + epsilon, 1.0 - epsilon),
            0.0,
        )
        partial_angle = jnp.where(partial_active, 2.0 * jnp.arccos(safe_cosine), 0.0)
        partial_flux = jnp.sum(
            partial_weights
            * self.disk.intensity(partial_mu)
            * partial_radius
            * partial_angle,
            axis=-1,
        )
        occulted = full_flux + partial_flux
        disjoint = b >= 1.0 + p
        complete = p >= 1.0 + b
        relative = 1.0 - occulted / self.disk.normalization.astype(occulted.dtype)
        relative = jnp.where(
            disjoint | ~foreground_, 1.0, jnp.where(complete, 0.0, relative)
        )
        relative = jnp.clip(relative, 0.0, 1.0)
        contact_tolerance = 64.0 * jnp.finfo(relative.dtype).eps
        contact = (
            (jnp.abs(b - (1.0 + p)) <= contact_tolerance)
            | (jnp.abs(b - jnp.abs(1.0 - p)) <= contact_tolerance)
            | (p == 0.0)
        )
        branch = jnp.where(
            ~foreground_,
            0,
            jnp.where(disjoint, 1, jnp.where(complete, 3, 2)),
        ).astype(jnp.int32)
        status = jnp.where(
            ~finite,
            int(AstrophysicsObservationStatus.NONFINITE_INPUT),
            jnp.where(
                domain,
                int(AstrophysicsObservationStatus.SUCCESS),
                int(AstrophysicsObservationStatus.INVALID_GEOMETRY),
            ),
        ).astype(jnp.int32)
        return CircularOccultationResult(
            jnp.where(domain, relative, 1.0),
            jnp.where(domain, occulted, 0.0),
            branch,
            contact,
            domain,
            status,
            self.plan_id,
        )


__all__ = [
    "CircularOccultationPlan",
    "CircularOccultationResult",
    "PolynomialLimbDarkenedDisk",
]
