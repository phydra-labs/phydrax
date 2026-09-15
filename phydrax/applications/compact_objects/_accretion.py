#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._relativistic_eos import GammaLawEOS
from ...metrix._adm_exchange import ADMGridGeometry
from ...metrix._spacetime_conventions import RelativityConvention


class AccretionInitialDataStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    OUTSIDE_METRIC_DOMAIN = 2
    ROOT_NOT_BRACKETED = 3
    ROOT_NOT_CONVERGED = 4
    EOS_INVALID = 5
    SUPERLUMINAL = 6


class MichelBondiInitialData(StrictModule):
    radius: Array
    rest_mass_density: Array
    pressure: Array
    radial_four_velocity: Array
    specific_enthalpy: Array
    mass_accretion_rate: Array
    continuity_residual: Array
    bernoulli_residual: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class MichelBondiValenciaData(StrictModule):
    primitive: Array
    solution: MichelBondiInitialData
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class MichelBondiAccretionPlan(StrictModule, NonTrainableState):
    """Transonic relativistic Michel--Bondi flow onto Schwarzschild.

    The plan solves both conserved integrals rather than attaching a Newtonian
    Bondi profile to a relativistic state.  ``radial_four_velocity`` is negative
    for inflow and the reported accretion rate is positive.
    """

    eos: GammaLawEOS
    convention: RelativityConvention
    mass_parameter: float = eqx.field(static=True)
    infinity_density: float = eqx.field(static=True)
    infinity_sound_speed_squared: float = eqx.field(static=True)
    polytropic_constant: float = eqx.field(static=True)
    infinity_specific_enthalpy: float = eqx.field(static=True)
    critical_sound_speed_squared: float = eqx.field(static=True)
    critical_radius: float = eqx.field(static=True)
    critical_radial_speed: float = eqx.field(static=True)
    mass_accretion_rate: float = eqx.field(static=True)
    root_iterations: int = eqx.field(static=True)
    bracket_samples: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        eos: GammaLawEOS,
        mass_parameter: float,
        infinity_density: float,
        infinity_sound_speed_squared: float,
        /,
        *,
        convention: RelativityConvention | None = None,
        root_iterations: int = 64,
        bracket_samples: int = 96,
        absolute_tolerance: float = 1.0e-11,
        relative_tolerance: float = 1.0e-9,
    ):
        if not isinstance(eos, GammaLawEOS):
            raise TypeError("Michel--Bondi initial data require GammaLawEOS.")
        if float(eos.scale.speed_of_light) != 1.0:
            raise ValueError("Michel--Bondi data require a geometric c=1 EOS scale.")
        convention_ = (
            RelativityConvention.canonical() if convention is None else convention
        )
        if not isinstance(convention_, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if (
            convention_.metric_signature != "mostly_plus"
            or convention_.future_time_orientation != 1
        ):
            raise ValueError(
                "Michel--Bondi data require mostly-plus, future-oriented conventions."
            )
        mass = float(mass_parameter)
        density = float(infinity_density)
        sound_squared = float(infinity_sound_speed_squared)
        iterations = int(root_iterations)
        samples = int(bracket_samples)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        gamma = eos.adiabatic_index
        if (
            any(
                not np.isfinite(value)
                for value in (mass, density, sound_squared, absolute, relative)
            )
            or mass <= 0.0
            or density <= 0.0
            or not 0.0 < sound_squared < gamma - 1.0
            or iterations <= 0
            or samples < 8
            or absolute <= 0.0
            or relative <= 0.0
        ):
            raise ValueError("Michel--Bondi controls are invalid.")
        infinity_enthalpy = (gamma - 1.0) / (gamma - 1.0 - sound_squared)
        infinity_internal = sound_squared / (gamma * (gamma - 1.0 - sound_squared))
        infinity_pressure = (gamma - 1.0) * density * infinity_internal
        polytropic = infinity_pressure / density**gamma

        def critical_residual(sound):
            enthalpy = (gamma - 1.0) / (gamma - 1.0 - sound)
            return enthalpy / np.sqrt(1.0 + 3.0 * sound) - infinity_enthalpy

        lower = sound_squared
        upper = (gamma - 1.0) * (1.0 - 1.0e-13)
        lower_residual = critical_residual(lower)
        upper_residual = critical_residual(upper)
        if lower_residual > 0.0 or upper_residual < 0.0:
            raise ValueError("Michel critical point is not bracketed by the EOS domain.")
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            if critical_residual(middle) <= 0.0:
                lower = middle
            else:
                upper = middle
        critical_sound = 0.5 * (lower + upper)
        critical_radius = mass * (1.0 + 3.0 * critical_sound) / (2.0 * critical_sound)
        critical_speed = float(np.sqrt(critical_sound / (1.0 + 3.0 * critical_sound)))
        critical_enthalpy = (gamma - 1.0) / (gamma - 1.0 - critical_sound)
        critical_pressure_over_density = (critical_enthalpy - 1.0) * (gamma - 1.0) / gamma
        critical_density = (critical_pressure_over_density / polytropic) ** (
            1.0 / (gamma - 1.0)
        )
        accretion_rate = float(
            4.0 * np.pi * critical_radius**2 * critical_density * critical_speed
        )
        self.eos = eos
        self.convention = convention_
        self.mass_parameter = mass
        self.infinity_density = density
        self.infinity_sound_speed_squared = sound_squared
        self.polytropic_constant = polytropic
        self.infinity_specific_enthalpy = infinity_enthalpy
        self.critical_sound_speed_squared = critical_sound
        self.critical_radius = critical_radius
        self.critical_radial_speed = critical_speed
        self.mass_accretion_rate = accretion_rate
        self.root_iterations = iterations
        self.bracket_samples = samples
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-michel-bondi-accretion",
                "eos": eos.eos_id,
                "convention": convention_.convention_id,
                "mass_parameter": mass,
                "infinity_density": density,
                "infinity_sound_speed_squared": sound_squared,
                "root_iterations": iterations,
                "bracket_samples": samples,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def _integrals(self, radius: Array, speed: Array, /):
        gamma = jnp.asarray(self.eos.adiabatic_index, dtype=radius.dtype)
        mass = jnp.asarray(self.mass_parameter, dtype=radius.dtype)
        rate = jnp.asarray(self.mass_accretion_rate, dtype=radius.dtype)
        density = rate / (
            4.0 * jnp.asarray(jnp.pi, dtype=radius.dtype) * radius**2 * speed
        )
        pressure = (
            jnp.asarray(self.polytropic_constant, dtype=radius.dtype) * density**gamma
        )
        eos_state = self.eos.evaluate_pressure(density, pressure)
        schwarzschild_factor = 1.0 - 2.0 * mass / radius
        bernoulli = eos_state.specific_enthalpy * jnp.sqrt(
            jnp.maximum(
                schwarzschild_factor + speed**2,
                jnp.finfo(radius.dtype).tiny,
            )
        )
        residual = bernoulli - jnp.asarray(
            self.infinity_specific_enthalpy, dtype=radius.dtype
        )
        return residual, density, pressure, eos_state, bernoulli

    def evaluate(self, radius: ArrayLike, /) -> MichelBondiInitialData:
        radius_ = jnp.asarray(radius)
        if not jnp.issubdtype(radius_.dtype, jnp.floating):
            radius_ = radius_.astype(jnp.float32)
        mass = jnp.asarray(self.mass_parameter, dtype=radius_.dtype)
        critical_radius = jnp.asarray(self.critical_radius, dtype=radius_.dtype)
        critical_speed = jnp.asarray(self.critical_radial_speed, dtype=radius_.dtype)
        continuity_speed = jnp.asarray(self.mass_accretion_rate, dtype=radius_.dtype) / (
            4.0
            * jnp.asarray(jnp.pi, dtype=radius_.dtype)
            * radius_**2
            * jnp.asarray(self.infinity_density, dtype=radius_.dtype)
        )
        lower_speed = jnp.maximum(
            1.0e-4 * continuity_speed,
            jnp.sqrt(jnp.finfo(radius_.dtype).tiny),
        )
        maximum_speed = jnp.maximum(
            jnp.maximum(8.0 * critical_speed, 100.0 * continuity_speed),
            4.0 * jnp.sqrt(jnp.maximum(2.0 * mass / radius_, 0.0)) + 1.0,
        )
        fractions = jnp.linspace(
            0.0,
            1.0,
            self.bracket_samples,
            dtype=radius_.dtype,
        )
        log_lower = jnp.log(lower_speed)
        samples = jnp.exp(
            log_lower[..., None]
            + fractions * (jnp.log(maximum_speed)[..., None] - log_lower[..., None])
        )
        sample_residual = self._integrals(radius_[..., None], samples)[0]
        crossings = sample_residual[..., :-1] * sample_residual[..., 1:] <= 0.0
        indices = jnp.arange(self.bracket_samples - 1, dtype=jnp.int32)
        outer_index = jnp.argmax(crossings.astype(jnp.int32), axis=-1)
        inner_index = jnp.max(
            jnp.where(crossings, indices, jnp.asarray(-1, dtype=jnp.int32)),
            axis=-1,
        )
        outer = radius_ >= critical_radius
        selected_index = jnp.where(outer, outer_index, inner_index)
        bracketed = jnp.any(crossings, axis=-1)
        selected_index = jnp.clip(selected_index, 0, self.bracket_samples - 2)
        low_log = jnp.log(
            jnp.take_along_axis(samples, selected_index[..., None], axis=-1)[..., 0]
        )
        high_log = jnp.log(
            jnp.take_along_axis(samples, (selected_index + 1)[..., None], axis=-1)[..., 0]
        )
        low_residual = self._integrals(radius_, jnp.exp(low_log))[0]

        def body(_, values):
            lower, upper, lower_value = values
            middle = 0.5 * (lower + upper)
            middle_value = self._integrals(radius_, jnp.exp(middle))[0]
            lower_half = lower_value * middle_value <= 0.0
            return (
                jnp.where(lower_half, lower, middle),
                jnp.where(lower_half, middle, upper),
                jnp.where(lower_half, lower_value, middle_value),
            )

        low_log, high_log, _ = jax.lax.fori_loop(
            0,
            self.root_iterations,
            body,
            (low_log, high_log, low_residual),
        )
        log_speed = 0.5 * (low_log + high_log)

        def refine(_, current_log_speed):
            current_speed = jnp.exp(current_log_speed)
            current = self._integrals(radius_, current_speed)
            schwarzschild_factor = 1.0 - 2.0 * mass / radius_
            redshift = jnp.sqrt(
                jnp.maximum(
                    schwarzschild_factor + current_speed**2,
                    jnp.finfo(radius_.dtype).tiny,
                )
            )
            enthalpy_derivative = -(self.eos.adiabatic_index - 1.0) * (
                current[3].specific_enthalpy - 1.0
            )
            residual_derivative = (
                enthalpy_derivative * redshift
                + current[3].specific_enthalpy * current_speed**2 / redshift
            )
            safe_derivative = jnp.where(
                jnp.abs(residual_derivative) > jnp.sqrt(jnp.finfo(radius_.dtype).eps),
                residual_derivative,
                1.0,
            )
            candidate = current_log_speed - current[0] / safe_derivative
            usable = (
                jnp.isfinite(candidate)
                & jnp.isfinite(residual_derivative)
                & (jnp.abs(residual_derivative) > jnp.sqrt(jnp.finfo(radius_.dtype).eps))
            )
            return jnp.where(usable, candidate, current_log_speed)

        log_speed = jax.lax.fori_loop(0, 3, refine, log_speed)
        speed = jnp.exp(log_speed)
        at_critical = jnp.abs(radius_ - critical_radius) <= (
            self.absolute_tolerance
            + self.relative_tolerance * jnp.maximum(critical_radius, 1.0)
        )
        speed = jnp.where(at_critical, critical_speed, speed)
        residual, density, pressure, eos_state, bernoulli = self._integrals(
            radius_, speed
        )
        radial_four_velocity = -speed
        continuity = -4.0 * jnp.asarray(
            jnp.pi, dtype=radius_.dtype
        ) * radius_**2 * density * radial_four_velocity - jnp.asarray(
            self.mass_accretion_rate, dtype=radius_.dtype
        )
        horizon_outside = radius_ > 2.0 * mass
        input_finite = jnp.isfinite(radius_)
        finite = (
            input_finite
            & jnp.isfinite(density)
            & jnp.isfinite(pressure)
            & jnp.isfinite(radial_four_velocity)
            & jnp.isfinite(residual)
            & eos_state.finite
        )
        residual_scale = jnp.maximum(jnp.abs(bernoulli), 1.0)
        residual_tolerance = jnp.maximum(
            self.absolute_tolerance + self.relative_tolerance * residual_scale,
            128.0 * jnp.finfo(radius_.dtype).eps * residual_scale,
        )
        continuity_scale = jnp.maximum(
            jnp.asarray(self.mass_accretion_rate, dtype=radius_.dtype), 1.0
        )
        continuity_tolerance = jnp.maximum(
            self.absolute_tolerance + self.relative_tolerance * self.mass_accretion_rate,
            128.0 * jnp.finfo(radius_.dtype).eps * continuity_scale,
        )
        converged = (
            (bracketed | at_critical)
            & (jnp.abs(residual) <= residual_tolerance)
            & (jnp.abs(continuity) <= continuity_tolerance)
        )
        physically_valid = (
            finite
            & horizon_outside
            & (density > 0.0)
            & (pressure > 0.0)
            & eos_state.physically_valid
        )
        qualified = physically_valid & converged & eos_state.qualified
        derivative_valid = qualified & eos_state.derivative_valid & ~at_critical
        status = jnp.where(
            ~input_finite,
            int(AccretionInitialDataStatus.NONFINITE_INPUT),
            jnp.where(
                ~horizon_outside,
                int(AccretionInitialDataStatus.OUTSIDE_METRIC_DOMAIN),
                jnp.where(
                    ~bracketed & ~at_critical,
                    int(AccretionInitialDataStatus.ROOT_NOT_BRACKETED),
                    jnp.where(
                        ~converged,
                        int(AccretionInitialDataStatus.ROOT_NOT_CONVERGED),
                        jnp.where(
                            ~eos_state.physically_valid,
                            int(AccretionInitialDataStatus.EOS_INVALID),
                            int(AccretionInitialDataStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return MichelBondiInitialData(
            radius_,
            density,
            pressure,
            radial_four_velocity,
            eos_state.specific_enthalpy,
            jnp.full_like(radius_, self.mass_accretion_rate),
            continuity,
            residual,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )

    def valencia_primitive(
        self,
        radius: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> MichelBondiValenciaData:
        solution = self.evaluate(radius)
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if geometry.leading_shape != solution.radius.shape:
            raise ValueError("ADM geometry shape must match Michel radii.")
        if geometry.scale_id != self.eos.scale.scale_id:
            raise ValueError("Michel EOS and ADM geometry scale identities differ.")
        shift_squared = ein.contract(
            "...i,...ij,...j->...",
            geometry.beta_contravariant,
            geometry.spatial_metric,
            geometry.beta_contravariant,
        )
        spatial_four_velocity = eqx.error_if(
            jnp.zeros(solution.radius.shape + (3,), dtype=solution.radius.dtype)
            .at[..., 0]
            .set(solution.radial_four_velocity),
            jnp.any(shift_squared > 128.0 * jnp.finfo(solution.radius.dtype).eps),
            "Michel Valencia conversion requires the zero-shift Schwarzschild slicing.",
        )
        spatial_norm_squared = ein.contract(
            "...i,...ij,...j->...",
            spatial_four_velocity,
            geometry.spatial_metric,
            spatial_four_velocity,
        )
        lorentz_factor = jnp.sqrt(1.0 + spatial_norm_squared)
        velocity = spatial_four_velocity / lorentz_factor[..., None]
        primitive = jnp.concatenate(
            (
                solution.rest_mass_density[..., None],
                velocity,
                solution.pressure[..., None],
                jnp.zeros_like(velocity),
            ),
            axis=-1,
        )
        velocity_squared = ein.contract(
            "...i,...ij,...j->...", velocity, geometry.spatial_metric, velocity
        )
        finite = solution.finite & jnp.all(jnp.isfinite(primitive), axis=-1)
        physically_valid = (
            solution.physically_valid
            & geometry.physically_valid
            & (velocity_squared < 1.0)
        )
        return MichelBondiValenciaData(
            primitive,
            solution,
            finite,
            solution.converged,
            physically_valid,
            solution.qualified & physically_valid,
            solution.derivative_valid & physically_valid,
        )


class FishboneMoncriefInitialData(StrictModule):
    coordinates: Array
    primitive: Array
    vector_potential_covector: Array
    angular_velocity: Array
    specific_angular_momentum: Array
    minus_covariant_time_velocity: Array
    specific_enthalpy: Array
    effective_potential: Array
    equilibrium_residual: Array
    inside_torus: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class FishboneMoncriefTorusPlan(StrictModule, NonTrainableState):
    """Constant-angular-momentum Fishbone--Moncrief torus in Kerr BL data."""

    eos: GammaLawEOS
    convention: RelativityConvention
    mass_parameter: float = eqx.field(static=True)
    spin_parameter: float = eqx.field(static=True)
    inner_radius: float = eqx.field(static=True)
    pressure_maximum_radius: float = eqx.field(static=True)
    outer_radius: float = eqx.field(static=True)
    specific_angular_momentum: float = eqx.field(static=True)
    inner_potential: float = eqx.field(static=True)
    polytropic_constant: float = eqx.field(static=True)
    atmosphere_density: float = eqx.field(static=True)
    atmosphere_pressure: float = eqx.field(static=True)
    magnetic_seed_amplitude: float = eqx.field(static=True)
    magnetic_seed_cutoff: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        eos: GammaLawEOS,
        mass_parameter: float,
        spin_parameter: float,
        inner_radius: float,
        pressure_maximum_radius: float,
        polytropic_constant: float,
        /,
        *,
        convention: RelativityConvention | None = None,
        atmosphere_density: float = 1.0e-10,
        atmosphere_pressure: float = 1.0e-12,
        magnetic_seed_amplitude: float = 0.0,
        magnetic_seed_cutoff: float = 0.2,
        residual_tolerance: float = 1.0e-9,
    ):
        if not isinstance(eos, GammaLawEOS):
            raise TypeError("Fishbone--Moncrief data require GammaLawEOS.")
        if float(eos.scale.speed_of_light) != 1.0:
            raise ValueError("Fishbone--Moncrief data require a geometric c=1 EOS scale.")
        convention_ = (
            RelativityConvention.canonical() if convention is None else convention
        )
        if not isinstance(convention_, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if (
            convention_.metric_signature != "mostly_plus"
            or convention_.future_time_orientation != 1
            or convention_.azimuthal_orientation != 1
        ):
            raise ValueError(
                "Fishbone--Moncrief data require canonical time and azimuth orientation."
            )
        values = tuple(
            float(value)
            for value in (
                mass_parameter,
                spin_parameter,
                inner_radius,
                pressure_maximum_radius,
                polytropic_constant,
                atmosphere_density,
                atmosphere_pressure,
                magnetic_seed_amplitude,
                magnetic_seed_cutoff,
                residual_tolerance,
            )
        )
        (
            mass,
            spin,
            inner,
            pressure_maximum,
            polytropic,
            atmosphere_density_,
            atmosphere_pressure_,
            seed,
            cutoff,
            tolerance,
        ) = values
        horizon = mass + np.sqrt(mass**2 - spin**2) if abs(spin) <= mass else np.nan
        if (
            any(not np.isfinite(value) for value in values)
            or mass <= 0.0
            or abs(spin) >= mass
            or not horizon < inner < pressure_maximum
            or polytropic <= 0.0
            or atmosphere_density_ <= 0.0
            or atmosphere_pressure_ <= 0.0
            or seed < 0.0
            or not 0.0 <= cutoff < 1.0
            or tolerance <= 0.0
        ):
            raise ValueError("Fishbone--Moncrief controls are invalid.")
        x = pressure_maximum / mass
        a = spin / mass
        root_x = np.sqrt(x)
        numerator = x**2 - 2.0 * a * root_x + a**2
        denominator = x**1.5 - 2.0 * root_x + a
        angular_momentum = float(mass * numerator / denominator)
        inner_g_tt, inner_g_t_phi, inner_g_phi_phi = self._metric_components_host(
            mass, spin, inner, np.pi / 2.0
        )
        inner_minus_ut = self._minus_ut_host(
            inner_g_tt,
            inner_g_t_phi,
            inner_g_phi_phi,
            angular_momentum,
        )
        if not np.isfinite(inner_minus_ut) or inner_minus_ut <= 0.0:
            raise ValueError(
                "Torus inner edge does not admit the selected circular flow."
            )
        center_g_tt, center_g_t_phi, center_g_phi_phi = self._metric_components_host(
            mass, spin, pressure_maximum, np.pi / 2.0
        )
        center_minus_ut = self._minus_ut_host(
            center_g_tt,
            center_g_t_phi,
            center_g_phi_phi,
            angular_momentum,
        )
        if (
            not np.isfinite(center_minus_ut)
            or center_minus_ut <= 0.0
            or center_minus_ut >= inner_minus_ut
        ):
            raise ValueError(
                "Torus pressure maximum does not lie inside the closed potential."
            )
        inner_potential = float(np.log(inner_minus_ut))

        def outer_surface_residual(radius):
            g_tt, g_t_phi, g_phi_phi = self._metric_components_host(
                mass, spin, radius, np.pi / 2.0
            )
            minus_ut = self._minus_ut_host(
                g_tt,
                g_t_phi,
                g_phi_phi,
                angular_momentum,
            )
            return np.log(minus_ut) - inner_potential

        outer_lower = pressure_maximum
        outer_upper = 2.0 * pressure_maximum
        for _ in range(128):
            if outer_surface_residual(outer_upper) > 0.0:
                break
            outer_upper *= 2.0
        upper_residual = outer_surface_residual(outer_upper)
        if not np.isfinite(upper_residual) or upper_residual <= 0.0:
            raise ValueError(
                "The outer Fishbone--Moncrief equipotential could not be bracketed."
            )
        for _ in range(128):
            outer_middle = 0.5 * (outer_lower + outer_upper)
            if outer_surface_residual(outer_middle) <= 0.0:
                outer_lower = outer_middle
            else:
                outer_upper = outer_middle
        outer_radius = float(0.5 * (outer_lower + outer_upper))
        self.eos = eos
        self.convention = convention_
        self.mass_parameter = mass
        self.spin_parameter = spin
        self.inner_radius = inner
        self.pressure_maximum_radius = pressure_maximum
        self.outer_radius = outer_radius
        self.specific_angular_momentum = angular_momentum
        self.inner_potential = inner_potential
        self.polytropic_constant = polytropic
        self.atmosphere_density = atmosphere_density_
        self.atmosphere_pressure = atmosphere_pressure_
        self.magnetic_seed_amplitude = seed
        self.magnetic_seed_cutoff = cutoff
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fishbone-moncrief-kerr-torus",
                "eos": eos.eos_id,
                "convention": convention_.convention_id,
                "mass_parameter": mass,
                "spin_parameter": spin,
                "inner_radius": inner,
                "pressure_maximum_radius": pressure_maximum,
                "outer_radius": outer_radius,
                "specific_angular_momentum": angular_momentum,
                "polytropic_constant": polytropic,
                "atmosphere_density": atmosphere_density_,
                "atmosphere_pressure": atmosphere_pressure_,
                "magnetic_seed_amplitude": seed,
                "magnetic_seed_cutoff": cutoff,
                "residual_tolerance": tolerance,
            }
        )

    @staticmethod
    def _metric_components_host(mass, spin, radius, polar):
        sigma = radius**2 + spin**2 * np.cos(polar) ** 2
        sine_squared = np.sin(polar) ** 2
        g_tt = -(1.0 - 2.0 * mass * radius / sigma)
        g_t_phi = -2.0 * mass * spin * radius * sine_squared / sigma
        g_phi_phi = sine_squared * (
            radius**2 + spin**2 + 2.0 * mass * spin**2 * radius * sine_squared / sigma
        )
        return g_tt, g_t_phi, g_phi_phi

    @staticmethod
    def _minus_ut_host(g_tt, g_t_phi, g_phi_phi, angular_momentum):
        numerator = g_t_phi**2 - g_tt * g_phi_phi
        denominator = (
            g_phi_phi + 2.0 * angular_momentum * g_t_phi + angular_momentum**2 * g_tt
        )
        return np.sqrt(numerator / denominator)

    def _circular_fields(self, radius: Array, polar: Array, /):
        mass = jnp.asarray(self.mass_parameter, dtype=radius.dtype)
        spin = jnp.asarray(self.spin_parameter, dtype=radius.dtype)
        angular_momentum = jnp.asarray(self.specific_angular_momentum, dtype=radius.dtype)
        sigma = radius**2 + spin**2 * jnp.cos(polar) ** 2
        sine_squared = jnp.sin(polar) ** 2
        delta = radius**2 - 2.0 * mass * radius + spin**2
        a_function = (radius**2 + spin**2) ** 2 - spin**2 * delta * sine_squared
        g_tt = -(1.0 - 2.0 * mass * radius / sigma)
        g_t_phi = -2.0 * mass * spin * radius * sine_squared / sigma
        g_phi_phi = sine_squared * (
            radius**2 + spin**2 + 2.0 * mass * spin**2 * radius * sine_squared / sigma
        )
        denominator = (
            g_phi_phi + 2.0 * angular_momentum * g_t_phi + angular_momentum**2 * g_tt
        )
        minus_ut = jnp.sqrt(
            jnp.maximum(
                (g_t_phi**2 - g_tt * g_phi_phi) / denominator,
                jnp.finfo(radius.dtype).tiny,
            )
        )
        omega = -(g_t_phi + angular_momentum * g_tt) / (
            g_phi_phi + angular_momentum * g_t_phi
        )
        alpha = jnp.sqrt(
            jnp.maximum(
                sigma * delta / a_function,
                jnp.finfo(radius.dtype).tiny,
            )
        )
        beta_phi = g_t_phi / g_phi_phi
        gamma_phi_phi = g_phi_phi
        return (
            minus_ut,
            omega,
            alpha,
            beta_phi,
            gamma_phi_phi,
            denominator,
            delta,
        )

    def evaluate(self, coordinates: ArrayLike, /) -> FishboneMoncriefInitialData:
        points = jnp.asarray(coordinates)
        if points.shape[-1:] != (3,):
            raise ValueError("Torus coordinates must have trailing shape (3,).")
        if not jnp.issubdtype(points.dtype, jnp.floating):
            points = points.astype(jnp.float32)
        radius = points[..., 0]
        polar = points[..., 1]
        (
            minus_ut,
            omega,
            alpha,
            beta_phi,
            gamma_phi_phi,
            circular_denominator,
            delta,
        ) = self._circular_fields(radius, polar)
        potential = jnp.log(minus_ut)
        enthalpy = jnp.exp(
            jnp.asarray(self.inner_potential, dtype=points.dtype) - potential
        )
        gamma = jnp.asarray(self.eos.adiabatic_index, dtype=points.dtype)
        thermal_enthalpy = jnp.maximum(enthalpy - 1.0, 0.0)
        density_inside = (
            thermal_enthalpy
            * (gamma - 1.0)
            / (gamma * jnp.asarray(self.polytropic_constant, dtype=points.dtype))
        ) ** (1.0 / (gamma - 1.0))
        pressure_inside = (
            jnp.asarray(self.polytropic_constant, dtype=points.dtype)
            * density_inside**gamma
        )
        inside = (
            (radius >= self.inner_radius)
            & (radius <= self.outer_radius)
            & (potential <= self.inner_potential)
            & (circular_denominator > 0.0)
            & (delta > 0.0)
            & (thermal_enthalpy > 0.0)
        )
        density = jnp.where(inside, density_inside, self.atmosphere_density)
        pressure = jnp.where(inside, pressure_inside, self.atmosphere_pressure)
        eos_state = self.eos.evaluate_pressure(density, pressure)
        velocity_phi = (omega + beta_phi) / alpha
        velocity_phi = jnp.where(inside, velocity_phi, 0.0)
        velocity = (
            jnp.zeros(points.shape, dtype=points.dtype).at[..., 2].set(velocity_phi)
        )
        magnetic = jnp.zeros_like(velocity)
        primitive = jnp.concatenate(
            (density[..., None], velocity, pressure[..., None], magnetic),
            axis=-1,
        )
        center_minus_ut = self._circular_fields(
            jnp.asarray(self.pressure_maximum_radius, dtype=points.dtype),
            jnp.asarray(jnp.pi / 2.0, dtype=points.dtype),
        )[0]
        center_enthalpy = jnp.exp(self.inner_potential - jnp.log(center_minus_ut))
        center_density = (
            (center_enthalpy - 1.0) * (gamma - 1.0) / (gamma * self.polytropic_constant)
        ) ** (1.0 / (gamma - 1.0))
        seed_argument = density / center_density - self.magnetic_seed_cutoff
        vector_potential_phi = self.magnetic_seed_amplitude * jnp.where(
            inside,
            jnp.maximum(seed_argument, 0.0) ** 2,
            0.0,
        )
        vector_potential = jnp.zeros_like(velocity).at[..., 2].set(vector_potential_phi)
        equilibrium = jnp.where(
            inside,
            jnp.log(eos_state.specific_enthalpy) + potential - self.inner_potential,
            0.0,
        )
        velocity_squared = gamma_phi_phi * velocity_phi**2
        input_finite = jnp.all(jnp.isfinite(points), axis=-1)
        finite = (
            input_finite
            & jnp.all(jnp.isfinite(primitive), axis=-1)
            & jnp.all(jnp.isfinite(vector_potential), axis=-1)
            & jnp.isfinite(equilibrium)
            & eos_state.finite
        )
        metric_domain = (delta > 0.0) & (circular_denominator > 0.0)
        equilibrium_scale = jnp.maximum(
            jnp.abs(jnp.log(eos_state.specific_enthalpy))
            + jnp.abs(potential)
            + jnp.abs(self.inner_potential),
            1.0,
        )
        equilibrium_tolerance = jnp.maximum(
            self.residual_tolerance,
            128.0 * jnp.finfo(points.dtype).eps * equilibrium_scale,
        )
        converged = (~inside) | (jnp.abs(equilibrium) <= equilibrium_tolerance)
        physically_valid = (
            finite
            & metric_domain
            & (density > 0.0)
            & (pressure > 0.0)
            & (velocity_squared < 1.0)
            & eos_state.physically_valid
        )
        qualified = physically_valid & converged & eos_state.qualified
        derivative_valid = (
            qualified & eos_state.derivative_valid & (~inside | (density_inside > 0.0))
        )
        status = jnp.where(
            ~input_finite,
            int(AccretionInitialDataStatus.NONFINITE_INPUT),
            jnp.where(
                ~metric_domain,
                int(AccretionInitialDataStatus.OUTSIDE_METRIC_DOMAIN),
                jnp.where(
                    velocity_squared >= 1.0,
                    int(AccretionInitialDataStatus.SUPERLUMINAL),
                    jnp.where(
                        ~eos_state.physically_valid,
                        int(AccretionInitialDataStatus.EOS_INVALID),
                        jnp.where(
                            ~converged,
                            int(AccretionInitialDataStatus.ROOT_NOT_CONVERGED),
                            int(AccretionInitialDataStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return FishboneMoncriefInitialData(
            points,
            primitive,
            vector_potential,
            omega,
            jnp.full_like(radius, self.specific_angular_momentum),
            minus_ut,
            eos_state.specific_enthalpy,
            potential,
            equilibrium,
            inside,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )


__all__ = [
    "AccretionInitialDataStatus",
    "FishboneMoncriefInitialData",
    "FishboneMoncriefTorusPlan",
    "MichelBondiAccretionPlan",
    "MichelBondiInitialData",
    "MichelBondiValenciaData",
]
