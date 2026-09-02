#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground
from ._products import MatterField, MatterPowerTable


class SphericalOverdensityMassDefinition(StrictModule, NonTrainableState):
    """Physical spherical-overdensity mass relative to mean or critical density."""

    overdensity: float = eqx.field(static=True)
    reference_density: Literal["mean_matter", "critical"] = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        overdensity: float,
        reference_density: Literal["mean_matter", "critical"],
        /,
    ):
        value = float(overdensity)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Spherical overdensity must be finite and positive.")
        if reference_density not in ("mean_matter", "critical"):
            raise ValueError("reference_density must be mean_matter or critical.")
        self.overdensity = value
        self.reference_density = reference_density
        self.definition_id = canonical_fingerprint(
            {
                "kind": "spherical-overdensity-mass-definition",
                "overdensity": value,
                "reference_density": reference_density,
            }
        )

    def reference_density_value(
        self,
        background: FLRWBackground,
        scale_factor: ArrayLike,
        gravitational_constant: ArrayLike,
        /,
    ) -> Array:
        scale = jnp.asarray(scale_factor, dtype=background.hubble_constant.dtype)
        gravity = jnp.asarray(gravitational_constant, dtype=scale.dtype)
        if gravity.shape != ():
            raise ValueError("gravitational_constant must be scalar.")
        gravity = eqx.error_if(
            gravity,
            ~jnp.isfinite(gravity) | (gravity <= 0.0),
            "gravitational_constant must be finite and positive.",
        )
        critical = 3.0 * background.hubble(scale) ** 2 / (8.0 * jnp.pi * gravity)
        return (
            critical * background.matter_fraction(scale)
            if self.reference_density == "mean_matter"
            else critical
        )

    def radius(
        self,
        background: FLRWBackground,
        mass: ArrayLike,
        scale_factor: ArrayLike,
        gravitational_constant: ArrayLike,
        /,
    ) -> Array:
        mass_ = jnp.asarray(mass, dtype=background.hubble_constant.dtype)
        density = self.reference_density_value(
            background, scale_factor, gravitational_constant
        )
        mass_ = eqx.error_if(
            mass_,
            jnp.any(~jnp.isfinite(mass_)) | jnp.any(mass_ <= 0.0),
            "Halo mass must be finite and positive.",
        )
        return (3.0 * mass_ / (4.0 * jnp.pi * self.overdensity * density)) ** (1.0 / 3.0)

    def mass(
        self,
        background: FLRWBackground,
        radius: ArrayLike,
        scale_factor: ArrayLike,
        gravitational_constant: ArrayLike,
        /,
    ) -> Array:
        radius_ = jnp.asarray(radius, dtype=background.hubble_constant.dtype)
        density = self.reference_density_value(
            background, scale_factor, gravitational_constant
        )
        radius_ = eqx.error_if(
            radius_,
            jnp.any(~jnp.isfinite(radius_)) | jnp.any(radius_ <= 0.0),
            "Halo radius must be finite and positive.",
        )
        return 4.0 * jnp.pi * self.overdensity * density * radius_**3 / 3.0


class SphericalCollapseEdS(StrictModule, NonTrainableState):
    """Exact Einstein-de Sitter spherical-collapse constants."""

    linear_threshold: float = eqx.field(static=True)
    virial_overdensity: float = eqx.field(static=True)

    def __init__(self):
        self.linear_threshold = float((3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0))
        self.virial_overdensity = float(18.0 * np.pi**2)


class LinearVariancePlan(StrictModule, NonTrainableState):
    """Top-hat linear variance sigma(M,a) on a fixed input power grid."""

    gravitational_constant: float = eqx.field(static=True)
    required_field: MatterField = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        /,
        *,
        required_field: MatterField = "total_matter",
    ):
        gravity = float(gravitational_constant)
        if not np.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("gravitational_constant must be finite and positive.")
        if required_field not in (
            "cold_baryon",
            "total_matter",
            "massive_neutrino_total",
        ):
            raise ValueError("Unknown linear-variance matter field.")
        self.gravitational_constant = gravity
        self.required_field = required_field
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-top-hat-variance",
                "gravitational_constant": gravity,
                "required_field": required_field,
            }
        )

    @staticmethod
    def _top_hat(argument: Array, /) -> Array:
        squared = argument**2
        series = 1.0 - squared / 10.0 + squared**2 / 280.0
        safe = jnp.where(argument == 0.0, 1.0, argument)
        direct = 3.0 * (jnp.sin(safe) - safe * jnp.cos(safe)) / safe**3
        return jnp.where(jnp.abs(argument) < 1.0e-3, series, direct)

    def sigma(
        self,
        background: FLRWBackground,
        power: MatterPowerTable,
        masses: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> Array:
        if not isinstance(power, MatterPowerTable):
            raise TypeError("power must be MatterPowerTable.")
        if (
            power.descriptor.stage != "linear"
            or power.descriptor.left_field != self.required_field
            or power.descriptor.right_field != self.required_field
            or power.descriptor.spatial_dimension != 3
        ):
            raise ValueError(
                "Linear variance requires the declared 3-D linear auto-power."
            )
        token = background.realization.require_compatible(
            power.realization, jnp.asarray(scale_factor)
        )
        masses_ = jnp.asarray(masses, dtype=power.power_values.dtype)
        masses_ = eqx.error_if(
            masses_,
            jnp.any(~jnp.isfinite(masses_)) | jnp.any(masses_ <= 0.0),
            "Variance masses must be finite and positive.",
        )
        critical_today = (
            3.0
            * background.hubble_constant**2
            / (8.0 * jnp.pi * self.gravitational_constant)
        )
        mean_today = background.matter_density * critical_today
        radius = (3.0 * masses_ / (4.0 * jnp.pi * mean_today)) ** (1.0 / 3.0)
        k = power.wavenumbers
        values = power.evaluate(k, token)
        log_k = jnp.log(k)
        differences = jnp.diff(log_k)
        weights = jnp.concatenate(
            (
                differences[:1] / 2.0,
                (differences[:-1] + differences[1:]) / 2.0,
                differences[-1:] / 2.0,
            )
        )
        flat_radius = radius.reshape((-1,))
        window = self._top_hat(flat_radius[:, None] * k[None, :])
        integrand = k[None, :] ** 3 * values[None, :] * window**2 / (2.0 * jnp.pi**2)
        variance = contract("k,mk->m", weights, integrand)
        return jnp.sqrt(jnp.maximum(variance, 0.0)).reshape(radius.shape)


class NFWProfile(StrictModule, NonTrainableState):
    """Truncated NFW profile under one spherical-overdensity mass definition."""

    mass_definition: SphericalOverdensityMassDefinition
    quadrature_nodes: Array
    quadrature_weights: Array
    order: int = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass_definition: SphericalOverdensityMassDefinition,
        /,
        *,
        quadrature_order: int = 64,
    ):
        if not isinstance(mass_definition, SphericalOverdensityMassDefinition):
            raise TypeError("mass_definition must be SphericalOverdensityMassDefinition.")
        order = int(quadrature_order)
        if order < 8:
            raise ValueError("NFW quadrature order must be at least eight.")
        rule = gauss_legendre_data(order)
        self.mass_definition = mass_definition
        self.quadrature_nodes = jnp.asarray(rule.nodes)
        self.quadrature_weights = jnp.asarray(rule.weights)
        self.order = order
        self.profile_id = canonical_fingerprint(
            {
                "kind": "truncated-nfw-profile",
                "mass_definition": mass_definition.definition_id,
                "quadrature_order": order,
            }
        )

    @staticmethod
    def normalization(concentration: ArrayLike, /) -> Array:
        concentration_ = jnp.asarray(concentration)
        concentration_ = eqx.error_if(
            concentration_,
            jnp.any(~jnp.isfinite(concentration_)) | jnp.any(concentration_ <= 0.0),
            "NFW concentration must be finite and positive.",
        )
        return jnp.log1p(concentration_) - concentration_ / (1.0 + concentration_)

    def density(
        self,
        radius: ArrayLike,
        mass: ArrayLike,
        halo_radius: ArrayLike,
        concentration: ArrayLike,
        /,
    ) -> Array:
        radius_ = jnp.asarray(radius)
        mass_ = jnp.asarray(mass, dtype=radius_.dtype)
        halo_radius_ = jnp.asarray(halo_radius, dtype=radius_.dtype)
        concentration_ = jnp.asarray(concentration, dtype=radius_.dtype)
        scale_radius = halo_radius_ / concentration_
        x = radius_ / scale_radius
        safe_x = jnp.where(x > 0.0, x, 1.0)
        density = (
            mass_
            / (4.0 * jnp.pi * scale_radius**3 * self.normalization(concentration_))
            / (safe_x * (1.0 + safe_x) ** 2)
        )
        return jnp.where((x > 0.0) & (x <= concentration_), density, 0.0)

    def enclosed_mass_fraction(
        self, radius_fraction: ArrayLike, concentration: ArrayLike, /
    ) -> Array:
        fraction = jnp.asarray(radius_fraction)
        concentration_ = jnp.asarray(concentration, dtype=fraction.dtype)
        x = jnp.clip(fraction, 0.0, 1.0) * concentration_
        enclosed = jnp.log1p(x) - x / (1.0 + x)
        return enclosed / self.normalization(concentration_)

    def fourier(
        self,
        wavenumber: ArrayLike,
        halo_radius: ArrayLike,
        concentration: ArrayLike,
        /,
    ) -> Array:
        k = jnp.asarray(wavenumber)
        radius = jnp.asarray(halo_radius, dtype=k.dtype)
        concentration_ = jnp.asarray(concentration, dtype=k.dtype)
        if radius.shape != () or concentration_.shape != ():
            raise ValueError("NFW halo_radius and concentration must be scalar.")
        x = 0.5 * concentration_ * (self.quadrature_nodes + 1.0)
        weights = 0.5 * concentration_ * self.quadrature_weights
        scale_radius = radius / concentration_
        argument = k.reshape((-1, 1)) * scale_radius * x[None, :]
        sinc = jnp.sinc(argument / jnp.pi)
        integrand = x[None, :] / (1.0 + x[None, :]) ** 2 * sinc
        values = contract("q,kq->k", weights, integrand) / self.normalization(
            concentration_
        )
        return values.reshape(k.shape)


__all__ = [
    "LinearVariancePlan",
    "NFWProfile",
    "SphericalCollapseEdS",
    "SphericalOverdensityMassDefinition",
]
