#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground
from ._closure import ScientificArtifactEnvelope
from ._halos import LinearVariancePlan, NFWProfile, SphericalOverdensityMassDefinition
from ._products import LagrangianGrowthHistory, MatterPowerTable


class SmoothSphericalCollapseResult(StrictModule):
    scale_factor: Array
    linear_threshold: Array
    virial_overdensity_mean: Array
    initial_overdensity: Array
    collapsed: Array
    approximate_virial: Array
    successful: Array


class SmoothComponentSphericalCollapsePlan(StrictModule, NonTrainableState):
    """Fixed-step top-hat collapse with homogeneous non-matter components."""

    initial_scale_factor: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    collapse_radius: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_scale_factor: float = 1.0e-2,
        steps: int = 1024,
        bisection_iterations: int = 64,
        collapse_radius: float = 1.0e-3,
    ):
        initial = float(initial_scale_factor)
        steps_ = int(steps)
        iterations = int(bisection_iterations)
        radius = float(collapse_radius)
        if (
            not np.isfinite(initial)
            or initial <= 0.0
            or initial >= 1.0
            or steps_ < 64
            or iterations < 16
            or not np.isfinite(radius)
            or radius <= 0.0
            or radius >= 1.0
        ):
            raise ValueError("Spherical-collapse numerical policy is invalid.")
        self.initial_scale_factor = initial
        self.steps = steps_
        self.bisection_iterations = iterations
        self.collapse_radius = radius
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-component-spherical-collapse",
                "initial_scale_factor": initial,
                "steps": steps_,
                "bisection_iterations": iterations,
                "collapse_radius": radius,
            }
        )

    def _terminal_radius(
        self, background: FLRWBackground, target: Array, overdensity: Array
    ) -> Array:
        log_start = jnp.log(jnp.asarray(self.initial_scale_factor, dtype=target.dtype))
        log_end = jnp.log(target)
        step = (log_end - log_start) / self.steps
        initial = jnp.stack((1.0 - overdensity / 3.0, -overdensity / 3.0))

        def rate(log_scale, state):
            scale = jnp.exp(log_scale)
            radius = jnp.maximum(state[0], self.collapse_radius * 0.1)
            velocity = state[1]
            source = 0.5 * background.matter_fraction(scale) * (radius**-3 - 1.0) * radius
            return jnp.stack(
                (
                    velocity,
                    -(2.0 + background.dlog_hubble_dlog_scale(scale)) * velocity - source,
                )
            )

        def advance(index, state):
            time = log_start + index * step
            k1 = rate(time, state)
            k2 = rate(time + 0.5 * step, state + 0.5 * step * k1)
            k3 = rate(time + 0.5 * step, state + 0.5 * step * k2)
            k4 = rate(time + step, state + step * k3)
            candidate = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            return candidate.at[0].set(
                jnp.maximum(candidate[0], self.collapse_radius * 0.1)
            )

        return jax.lax.fori_loop(0, self.steps, advance, initial)[0]

    def solve(
        self,
        background: FLRWBackground,
        growth: LagrangianGrowthHistory,
        scale_factor: ArrayLike,
        /,
    ) -> SmoothSphericalCollapseResult:
        target = background.require_flat(jnp.asarray(scale_factor))
        target = background.realization.require_compatible(growth.realization, target)
        target = eqx.error_if(
            target,
            (target <= self.initial_scale_factor) | (target > 1.0),
            "Spherical-collapse target must lie after the initial epoch and at or before a=1.",
        )

        def bisect(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            terminal = self._terminal_radius(background, target, midpoint)
            collapsed = terminal <= self.collapse_radius
            return (
                jnp.where(collapsed, lower, midpoint),
                jnp.where(collapsed, midpoint, upper),
            )

        lower, upper = jax.lax.fori_loop(
            0,
            self.bisection_iterations,
            bisect,
            (
                jnp.asarray(1.0e-6, dtype=target.dtype),
                jnp.asarray(0.9, dtype=target.dtype),
            ),
        )
        initial_overdensity = upper
        first_start = growth.first_order_growth[0]
        first_target = growth.evaluate(target)[0]
        linear_threshold = initial_overdensity * first_target / first_start
        matter_fraction = background.matter_fraction(target)
        x = matter_fraction - 1.0
        virial_critical = 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2
        virial_mean = virial_critical / matter_fraction
        terminal = self._terminal_radius(background, target, initial_overdensity)
        collapsed = terminal <= self.collapse_radius * 1.01
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack((linear_threshold, virial_mean, initial_overdensity, terminal))
            )
        )
        return SmoothSphericalCollapseResult(
            target,
            linear_threshold,
            virial_mean,
            initial_overdensity,
            collapsed,
            jnp.asarray(True),
            collapsed & finite,
        )


class HaloTripletResult(StrictModule):
    masses: Array
    mass_function_dndlnm: Array
    linear_bias: Array
    concentration: Array
    sigma: Array
    dlog_sigma_dlog_mass: Array
    within_domain: Array
    successful: Array


class TinkerDuffy200mPlan(StrictModule, NonTrainableState):
    """Locked Tinker08/Tinker10/Duffy08 200m calibration triplet."""

    variance: LinearVariancePlan
    minimum_mass: float = eqx.field(static=True)
    maximum_mass: float = eqx.field(static=True)
    maximum_redshift: float = eqx.field(static=True)
    pivot_mass: float = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)

    def __init__(
        self,
        variance: LinearVariancePlan,
        /,
        *,
        mass_domain: tuple[float, float],
        maximum_redshift: float = 2.0,
        pivot_mass: float = 2.0e12,
    ):
        if not isinstance(variance, LinearVariancePlan):
            raise TypeError("variance must be LinearVariancePlan.")
        minimum, maximum = (float(value) for value in mass_domain)
        redshift = float(maximum_redshift)
        pivot = float(pivot_mass)
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or minimum <= 0.0
            or maximum <= minimum
            or not np.isfinite(redshift)
            or redshift < 0.0
            or not np.isfinite(pivot)
            or pivot <= 0.0
        ):
            raise ValueError("Halo calibration domain is invalid.")
        self.variance = variance
        self.minimum_mass = minimum
        self.maximum_mass = maximum
        self.maximum_redshift = redshift
        self.pivot_mass = pivot
        self.calibration_id = canonical_fingerprint(
            {
                "kind": "tinker08-tinker10-duffy08-200m",
                "mass_domain": [minimum, maximum],
                "maximum_redshift": redshift,
                "pivot_mass": pivot,
                "mass_definition": "200m",
                "matter_field": "total_matter",
                "delta_c": "eds-fixed",
            }
        )

    def evaluate(
        self,
        background: FLRWBackground,
        linear_power: MatterPowerTable,
        masses: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> HaloTripletResult:
        if (
            linear_power.descriptor.stage != "linear"
            or linear_power.descriptor.left_field != "total_matter"
            or linear_power.descriptor.right_field != "total_matter"
        ):
            raise ValueError(
                "The calibrated 200m triplet requires total linear matter power."
            )
        mass = jnp.asarray(masses, dtype=linear_power.power_values.dtype)
        scale = jnp.asarray(scale_factor, dtype=mass.dtype)
        redshift = 1.0 / scale - 1.0
        within = (
            jnp.all(mass >= self.minimum_mass)
            & jnp.all(mass <= self.maximum_mass)
            & (redshift >= 0.0)
            & (redshift <= self.maximum_redshift)
        )
        log_mass = jnp.log(mass)

        def log_sigma(log_value):
            value = jnp.exp(log_value)
            sigma = self.variance.sigma(background, linear_power, value, scale)
            return jnp.log(sigma)

        sigma = jax.vmap(
            lambda value: self.variance.sigma(background, linear_power, value, scale)
        )(mass.reshape((-1,))).reshape(mass.shape)
        derivative = jax.vmap(jax.grad(log_sigma))(log_mass.reshape((-1,))).reshape(
            mass.shape
        )
        z_factor = 1.0 + redshift
        alpha = 10.0 ** (-((0.75 / jnp.log10(200.0 / 75.0)) ** 1.2))
        amplitude = 0.186 * z_factor**-0.14
        exponent = 1.47 * z_factor**-0.06
        scale_sigma = 2.57 * z_factor**-alpha
        f_sigma = (
            amplitude
            * ((sigma / scale_sigma) ** (-exponent) + 1.0)
            * jnp.exp(-1.19 / sigma**2)
        )
        critical_today = (
            3.0
            * background.hubble_constant**2
            / (8.0 * jnp.pi * self.variance.gravitational_constant)
        )
        mean_today = background.matter_density * critical_today
        mass_function = mean_today / mass * f_sigma * jnp.abs(derivative)
        nu = 1.68647019984 / sigma
        y = jnp.log10(jnp.asarray(200.0, dtype=sigma.dtype))
        a_bias = 1.0 + 0.24 * y * jnp.exp(-((4.0 / y) ** 4))
        exponent_bias = 0.44 * y - 0.88
        c_bias = 0.019 + 0.107 * y + 0.19 * jnp.exp(-((4.0 / y) ** 4))
        bias = (
            1.0
            - a_bias
            * nu**exponent_bias
            / (nu**exponent_bias + 1.68647019984**exponent_bias)
            + 0.183 * nu**1.5
            + c_bias * nu**2.4
        )
        concentration = 10.14 * (mass / self.pivot_mass) ** -0.081 * z_factor**-1.01
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack((mass_function, bias, concentration, sigma, derivative))
            )
        )
        successful = (
            within & finite & jnp.all(mass_function >= 0.0) & jnp.all(concentration > 0.0)
        )
        return HaloTripletResult(
            mass,
            mass_function,
            bias,
            concentration,
            sigma,
            derivative,
            within,
            successful,
        )


class MatterHaloModelResult(StrictModule):
    wavenumbers: Array
    one_halo: Array
    two_halo: Array
    total: Array
    mass_completeness: Array
    bias_normalization: Array
    successful: Array


class MatterHaloModel200mPlan(StrictModule, NonTrainableState):
    triplet: TinkerDuffy200mPlan
    profile: NFWProfile
    plan_id: str = eqx.field(static=True)

    def __init__(self, triplet: TinkerDuffy200mPlan, profile: NFWProfile, /):
        if not isinstance(triplet, TinkerDuffy200mPlan) or not isinstance(
            profile, NFWProfile
        ):
            raise TypeError("Halo model requires calibrated triplet and NFW profile.")
        if (
            profile.mass_definition.reference_density != "mean_matter"
            or profile.mass_definition.overdensity != 200.0
        ):
            raise ValueError(
                "First halo model requires the exact 200m profile definition."
            )
        self.triplet = triplet
        self.profile = profile
        self.plan_id = canonical_fingerprint(
            {
                "kind": "matter-halo-model-200m",
                "triplet": triplet.calibration_id,
                "profile": profile.profile_id,
            }
        )

    def evaluate(
        self,
        background: FLRWBackground,
        linear_power: MatterPowerTable,
        masses: ArrayLike,
        wavenumbers: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> MatterHaloModelResult:
        mass = jnp.asarray(masses)
        k = jnp.asarray(wavenumbers, dtype=mass.dtype)
        triplet = self.triplet.evaluate(background, linear_power, mass, scale_factor)
        log_mass = jnp.log(mass)
        differences = jnp.diff(log_mass)
        weights = jnp.concatenate(
            (
                differences[:1] / 2.0,
                (differences[:-1] + differences[1:]) / 2.0,
                differences[-1:] / 2.0,
            )
        )
        definition = self.profile.mass_definition
        radius = definition.radius(
            background,
            mass,
            scale_factor,
            self.triplet.variance.gravitational_constant,
        )
        profiles = jax.vmap(
            lambda radius_, concentration_: self.profile.fourier(
                k, radius_, concentration_
            )
        )(radius, triplet.concentration)
        critical_today = (
            3.0
            * background.hubble_constant**2
            / (8.0 * jnp.pi * self.triplet.variance.gravitational_constant)
        )
        mean_today = background.matter_density * critical_today
        mass_weight = triplet.mass_function_dndlnm * mass / mean_today
        one_halo = contract(
            "m,m,mk->k",
            weights,
            triplet.mass_function_dndlnm * (mass / mean_today) ** 2,
            profiles**2,
        )
        bias_kernel = contract(
            "m,m,m,mk->k",
            weights,
            mass_weight,
            triplet.linear_bias,
            profiles,
        )
        linear = linear_power.evaluate(k, scale_factor)
        two_halo = bias_kernel**2 * linear
        total = one_halo + two_halo
        completeness = contract("m,m->", weights, mass_weight)
        bias_normalization = contract(
            "m,m,m->", weights, mass_weight, triplet.linear_bias
        )
        successful = (
            triplet.successful & jnp.all(jnp.isfinite(total)) & jnp.all(total >= 0.0)
        )
        return MatterHaloModelResult(
            k,
            one_halo,
            two_halo,
            total,
            completeness,
            bias_normalization,
            successful,
        )


class HaloCatalog(StrictModule):
    halo_ids: Array
    positions: Array
    velocities: Array
    masses: Array
    active_mask: Array
    mass_definition: SphericalOverdensityMassDefinition
    scale_factor: Array
    box_size: tuple[float, ...] = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    catalog_id: str = eqx.field(static=True)

    def __init__(
        self,
        halo_ids: ArrayLike,
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike,
        active_mask: ArrayLike,
        mass_definition: SphericalOverdensityMassDefinition,
        scale_factor: ArrayLike,
        box_size: tuple[float, ...],
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        ids = jax.lax.stop_gradient(jnp.asarray(halo_ids))
        position = jax.lax.stop_gradient(jnp.asarray(positions))
        velocity = jax.lax.stop_gradient(jnp.asarray(velocities, dtype=position.dtype))
        mass = jax.lax.stop_gradient(jnp.asarray(masses, dtype=position.dtype))
        active = jax.lax.stop_gradient(jnp.asarray(active_mask, dtype=bool))
        scale = jax.lax.stop_gradient(jnp.asarray(scale_factor, dtype=position.dtype))
        if (
            ids.ndim != 1
            or position.shape != velocity.shape
            or position.shape[0] != ids.size
            or mass.shape != ids.shape
            or active.shape != ids.shape
            or position.shape[1] != len(box_size)
            or scale.shape != ()
        ):
            raise ValueError("Halo catalog array shapes are inconsistent.")
        valid = active & (mass > 0.0)
        position = eqx.error_if(
            position,
            jnp.any(~jnp.isfinite(position[active]))
            | jnp.any(~jnp.isfinite(velocity[active]))
            | jnp.any(~jnp.isfinite(mass[active]))
            | jnp.any(~valid[active]),
            "Active halo catalog entries must be finite with positive mass.",
        )
        self.halo_ids = ids
        self.positions = position
        self.velocities = velocity
        self.masses = mass
        self.active_mask = active
        self.mass_definition = mass_definition
        self.scale_factor = scale
        self.box_size = tuple(float(value) for value in box_size)
        self.artifact = artifact
        self.catalog_id = canonical_fingerprint(
            {
                "kind": "halo-catalog",
                "artifact": artifact.artifact_id,
                "mass_definition": mass_definition.definition_id,
                "arrays": array_tree_fingerprint((ids, position, velocity, mass, active)),
            }
        )

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        /,
        *,
        id_dataset: str,
        position_dataset: str,
        velocity_dataset: str,
        mass_dataset: str,
        mass_definition: SphericalOverdensityMassDefinition,
        scale_factor: float,
        box_size: tuple[float, ...],
        artifact: ScientificArtifactEnvelope,
    ) -> HaloCatalog:
        with h5py.File(Path(path), "r") as handle:
            ids = np.asarray(handle[id_dataset])
            positions = np.asarray(handle[position_dataset])
            velocities = np.asarray(handle[velocity_dataset])
            masses = np.asarray(handle[mass_dataset])
        active = np.ones(ids.shape, dtype=bool)
        return cls(
            ids,
            positions,
            velocities,
            masses,
            active,
            mass_definition,
            scale_factor,
            box_size,
            artifact,
        )


class Zheng07OccupationResult(StrictModule):
    central_probability: Array
    satellite_mean: Array
    total_mean: Array
    finite: Array
    successful: Array


class Zheng07OccupationExpectation200m(StrictModule):
    log10_minimum_mass: Array
    log10_scatter: Array
    cutoff_mass: Array
    satellite_mass: Array
    satellite_slope: Array

    def __init__(
        self,
        log10_minimum_mass: ArrayLike,
        log10_scatter: ArrayLike,
        cutoff_mass: ArrayLike,
        satellite_mass: ArrayLike,
        satellite_slope: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                log10_minimum_mass,
                log10_scatter,
                cutoff_mass,
                satellite_mass,
                satellite_slope,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Zheng07 parameters must be scalar.")
        values = tuple(
            eqx.error_if(
                value,
                ~jnp.isfinite(value),
                "Zheng07 parameters must be finite.",
            )
            for value in values
        )
        if values[1] <= 0.0 or values[2] < 0.0 or values[3] <= 0.0 or values[4] <= 0.0:
            raise ValueError("Zheng07 scatter/mass/slope constraints failed.")
        (
            self.log10_minimum_mass,
            self.log10_scatter,
            self.cutoff_mass,
            self.satellite_mass,
            self.satellite_slope,
        ) = values

    def evaluate(self, masses: ArrayLike, /) -> Zheng07OccupationResult:
        mass = jnp.asarray(masses, dtype=self.log10_minimum_mass.dtype)
        mass = eqx.error_if(
            mass,
            jnp.any(~jnp.isfinite(mass)) | jnp.any(mass <= 0.0),
            "HOD masses must be finite and positive.",
        )
        central = 0.5 * (
            1.0
            + jax.scipy.special.erf(
                (jnp.log10(mass) - self.log10_minimum_mass) / self.log10_scatter
            )
        )
        satellite_base = jnp.maximum((mass - self.cutoff_mass) / self.satellite_mass, 0.0)
        satellite = central * satellite_base**self.satellite_slope
        total = central + satellite
        finite = jnp.all(jnp.isfinite(total))
        return Zheng07OccupationResult(central, satellite, total, finite, finite)


__all__ = [
    "HaloCatalog",
    "HaloTripletResult",
    "MatterHaloModel200mPlan",
    "MatterHaloModelResult",
    "SmoothComponentSphericalCollapsePlan",
    "SmoothSphericalCollapseResult",
    "TinkerDuffy200mPlan",
    "Zheng07OccupationExpectation200m",
    "Zheng07OccupationResult",
]
