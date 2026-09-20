#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import apply_gather_stencil, rectilinear_stencil
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import UnitDefinition


RelativisticEOSStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
RELATIVISTIC_EOS_SUCCESS: RelativisticEOSStatus = 0
RELATIVISTIC_EOS_NONFINITE: RelativisticEOSStatus = 1
RELATIVISTIC_EOS_DENSITY_BELOW_DOMAIN: RelativisticEOSStatus = 2
RELATIVISTIC_EOS_DENSITY_ABOVE_DOMAIN: RelativisticEOSStatus = 3
RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN: RelativisticEOSStatus = 4
RELATIVISTIC_EOS_THERMAL_ABOVE_DOMAIN: RelativisticEOSStatus = 5
RELATIVISTIC_EOS_COMPOSITION_BELOW_DOMAIN: RelativisticEOSStatus = 6
RELATIVISTIC_EOS_COMPOSITION_ABOVE_DOMAIN: RelativisticEOSStatus = 7
RELATIVISTIC_EOS_COLD_CONSTRAINT_MISMATCH: RelativisticEOSStatus = 8
RELATIVISTIC_EOS_UNSTABLE: RelativisticEOSStatus = 9
RELATIVISTIC_EOS_ACAUSAL: RelativisticEOSStatus = 10
RELATIVISTIC_EOS_NONCONVERGED: RelativisticEOSStatus = 11


def relativistic_eos_status_name(status: int, /) -> str:
    """Return the stable name of one relativistic-EOS status code."""
    names = (
        "success",
        "nonfinite",
        "density_below_domain",
        "density_above_domain",
        "thermal_below_domain",
        "thermal_above_domain",
        "composition_below_domain",
        "composition_above_domain",
        "cold_constraint_mismatch",
        "unstable",
        "acausal",
        "nonconverged",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown relativistic-EOS status {code}.")
    return names[code]


class RelativisticEOSDomainEvidence(StrictModule):
    """Pointwise numerical, domain, physical, and derivative evidence."""

    finite: Array
    converged: Array
    density_below_domain: Array
    density_above_domain: Array
    thermal_below_domain: Array
    thermal_above_domain: Array
    composition_below_domain: Array
    composition_above_domain: Array
    cold_constraint_satisfied: Array
    mechanically_stable: Array
    causal: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    temperature_defined: Array
    status: Array
    eos_id: str = eqx.field(static=True)


class RelativisticEOSState(StrictModule):
    """One broadcast EOS evaluation in the scale carried by its model."""

    rest_mass_density: Array
    specific_internal_energy: Array
    pressure: Array
    total_energy_density: Array
    specific_enthalpy: Array
    sound_speed_squared: Array
    temperature: Array
    composition: Array
    pressure_density_derivative: Array
    pressure_specific_internal_energy_derivative: Array
    evidence: RelativisticEOSDomainEvidence
    eos_id: str = eqx.field(static=True)

    @property
    def finite(self) -> Array:
        return self.evidence.finite

    @property
    def converged(self) -> Array:
        return self.evidence.converged

    @property
    def physically_valid(self) -> Array:
        return self.evidence.physically_valid

    @property
    def qualified(self) -> Array:
        return self.evidence.qualified

    @property
    def derivative_valid(self) -> Array:
        return self.evidence.derivative_valid

    @property
    def status(self) -> Array:
        return self.evidence.status


class RelativisticEOSTableEvidence(StrictModule, NonTrainableState):
    """Host-established facts for one bounded finite-temperature table."""

    axes_monotone: bool = eqx.field(static=True)
    values_finite: bool = eqx.field(static=True)
    pressure_monotone_density: bool = eqx.field(static=True)
    pressure_monotone_temperature: bool = eqx.field(static=True)
    positive_heat_capacity: bool = eqx.field(static=True)
    mechanically_stable: bool = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    minimum_heat_capacity: float = eqx.field(static=True)
    minimum_adiabatic_pressure_derivative: float = eqx.field(static=True)
    minimum_causality_margin: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def qualified(self) -> bool:
        return (
            self.axes_monotone
            and self.values_finite
            and self.pressure_monotone_density
            and self.pressure_monotone_temperature
            and self.positive_heat_capacity
            and self.mechanically_stable
            and self.causal
        )


class AbstractRelativisticEOS(StrictModule, NonTrainableState):
    """Immutable relativistic EOS evaluated in one explicit physical scale."""

    scale: eqx.AbstractVar[RelativityScaleContract]
    eos_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_pressure(
        self,
        rest_mass_density: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        raise NotImplementedError

    def pressure(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.evaluate(
            rest_mass_density, specific_internal_energy, composition
        ).pressure

    def energy_density(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.evaluate(
            rest_mass_density, specific_internal_energy, composition
        ).total_energy_density

    def specific_enthalpy(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.evaluate(
            rest_mass_density, specific_internal_energy, composition
        ).specific_enthalpy

    def sound_speed_squared(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.evaluate(
            rest_mass_density, specific_internal_energy, composition
        ).sound_speed_squared

    def sound_speed(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return jnp.sqrt(
            self.sound_speed_squared(
                rest_mass_density, specific_internal_energy, composition
            )
        )

    def specific_internal_energy_from_pressure(
        self,
        rest_mass_density: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.evaluate_pressure(
            rest_mass_density, pressure, composition
        ).specific_internal_energy

    @property
    def rest_mass_density_unit(self) -> UnitDefinition:
        return self.scale.mass_density_unit

    @property
    def pressure_unit(self) -> UnitDefinition:
        return self.scale.energy_density_unit

    @property
    def energy_density_unit(self) -> UnitDefinition:
        return self.scale.energy_density_unit

    @property
    def specific_energy_unit(self) -> UnitDefinition:
        return self.scale.specific_energy_unit

    @property
    def specific_enthalpy_unit(self) -> UnitDefinition:
        return self.scale.specific_energy_unit

    @property
    def sound_speed_unit(self) -> UnitDefinition:
        return self.scale.dimensional_scale.velocity_unit

    @property
    def sound_speed_squared_unit(self) -> UnitDefinition:
        return self.scale.specific_energy_unit

    @property
    def temperature_unit(self) -> UnitDefinition:
        return self.scale.temperature_unit


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.float32)
    return array


def _broadcast_state_inputs(
    rest_mass_density: ArrayLike,
    thermal: ArrayLike,
    composition: ArrayLike | None,
    /,
) -> tuple[Array, Array, Array, bool]:
    density = _real_array(rest_mass_density, "rest_mass_density")
    thermal_value = _real_array(thermal, "thermal coordinate")
    supplied_composition = composition is not None
    if supplied_composition:
        composition_value = _real_array(composition, "composition")
        density, thermal_value, composition_value = jnp.broadcast_arrays(
            density, thermal_value, composition_value
        )
    else:
        density, thermal_value = jnp.broadcast_arrays(density, thermal_value)
        composition_value = jnp.full_like(density, jnp.nan)
    return density, thermal_value, composition_value, supplied_composition


def _upper_violation(value: Array, upper: float | None, /) -> Array:
    if upper is None:
        return jnp.zeros_like(value, dtype=jnp.bool_)
    return value > jnp.asarray(upper, dtype=value.dtype)


def _density_violations(
    density: Array,
    minimum_density: float,
    maximum_density: float | None,
    /,
) -> tuple[Array, Array]:
    lower = jnp.asarray(minimum_density, dtype=density.dtype)
    below = (density <= 0.0) | (density < lower)
    return below, _upper_violation(density, maximum_density)


def _domain_evidence(
    *,
    input_finite: Array,
    output_finite: Array,
    converged: Array,
    density_below: Array,
    density_above: Array,
    thermal_below: Array,
    thermal_above: Array,
    composition_below: Array,
    composition_above: Array,
    cold_constraint_satisfied: Array,
    mechanically_stable: Array,
    causal: Array,
    branch_derivative_valid: Array,
    temperature_defined: Array,
    model_qualified: bool,
    eos_id: str,
) -> RelativisticEOSDomainEvidence:
    finite = input_finite & output_finite
    within_domain = ~(
        density_below
        | density_above
        | thermal_below
        | thermal_above
        | composition_below
        | composition_above
    )
    physically_valid = (
        finite
        & converged
        & within_domain
        & cold_constraint_satisfied
        & mechanically_stable
        & causal
    )
    qualified = physically_valid & jnp.asarray(model_qualified)
    derivative_valid = qualified & branch_derivative_valid
    status = jnp.select(
        (
            ~input_finite,
            density_below,
            density_above,
            thermal_below,
            thermal_above,
            composition_below,
            composition_above,
            ~cold_constraint_satisfied,
            ~output_finite,
            ~converged,
            ~mechanically_stable,
            ~causal,
        ),
        (
            RELATIVISTIC_EOS_NONFINITE,
            RELATIVISTIC_EOS_DENSITY_BELOW_DOMAIN,
            RELATIVISTIC_EOS_DENSITY_ABOVE_DOMAIN,
            RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN,
            RELATIVISTIC_EOS_THERMAL_ABOVE_DOMAIN,
            RELATIVISTIC_EOS_COMPOSITION_BELOW_DOMAIN,
            RELATIVISTIC_EOS_COMPOSITION_ABOVE_DOMAIN,
            RELATIVISTIC_EOS_COLD_CONSTRAINT_MISMATCH,
            RELATIVISTIC_EOS_NONFINITE,
            RELATIVISTIC_EOS_NONCONVERGED,
            RELATIVISTIC_EOS_UNSTABLE,
            RELATIVISTIC_EOS_ACAUSAL,
        ),
        default=RELATIVISTIC_EOS_SUCCESS,
    ).astype(jnp.int32)
    return RelativisticEOSDomainEvidence(
        finite,
        converged,
        density_below,
        density_above,
        thermal_below,
        thermal_above,
        composition_below,
        composition_above,
        cold_constraint_satisfied,
        mechanically_stable,
        causal,
        physically_valid,
        qualified,
        derivative_valid,
        temperature_defined,
        status,
        eos_id,
    )


def _finite_outputs(*values: Array) -> Array:
    result = jnp.ones_like(values[0], dtype=jnp.bool_)
    for value in values:
        result = result & jnp.isfinite(value)
    return result


def _positive_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _optional_upper_bound(
    value: float | None, lower: float, name: str, /
) -> float | None:
    if value is None:
        return None
    upper = float(value)
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"{name} must be finite and greater than its lower bound.")
    return upper


def _declared_text(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string.")
    return value


class GammaLawEOS(AbstractRelativisticEOS):
    """Causal gamma-law EOS with independent specific internal energy."""

    scale: RelativityScaleContract = eqx.field(static=True)
    adiabatic_index: float = eqx.field(static=True)
    minimum_density: float = eqx.field(static=True)
    maximum_density: float | None = eqx.field(static=True)
    minimum_specific_internal_energy: float = eqx.field(static=True)
    maximum_specific_internal_energy: float | None = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        adiabatic_index: float,
        /,
        *,
        minimum_density: float = 0.0,
        maximum_density: float | None = None,
        minimum_specific_internal_energy: float = 0.0,
        maximum_specific_internal_energy: float | None = None,
        provenance: str = "analytic gamma-law",
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        gamma = float(adiabatic_index)
        if not np.isfinite(gamma) or not 1.0 < gamma <= 2.0:
            raise ValueError("adiabatic_index must be finite and lie in (1, 2].")
        density_lower = float(minimum_density)
        energy_lower = float(minimum_specific_internal_energy)
        if (
            not np.isfinite(density_lower)
            or density_lower < 0.0
            or not np.isfinite(energy_lower)
            or energy_lower < 0.0
        ):
            raise ValueError("EOS lower bounds must be finite and non-negative.")
        density_upper = _optional_upper_bound(
            maximum_density, density_lower, "maximum_density"
        )
        energy_upper = _optional_upper_bound(
            maximum_specific_internal_energy,
            energy_lower,
            "maximum_specific_internal_energy",
        )
        source = _declared_text(provenance, "provenance")
        self.scale = scale
        self.adiabatic_index = gamma
        self.minimum_density = density_lower
        self.maximum_density = density_upper
        self.minimum_specific_internal_energy = energy_lower
        self.maximum_specific_internal_energy = energy_upper
        self.provenance = source
        self.eos_id = canonical_fingerprint(
            {
                "kind": "relativistic-gamma-law-eos",
                "scale": scale.scale_id,
                "adiabatic_index": gamma,
                "density_domain": [density_lower, density_upper],
                "specific_internal_energy_domain": [energy_lower, energy_upper],
                "provenance": source,
            }
        )

    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        if specific_internal_energy is None:
            raise TypeError("Gamma-law EOS requires specific_internal_energy.")
        density, energy, composition_value, composition_supplied = (
            _broadcast_state_inputs(
                rest_mass_density, specific_internal_energy, composition
            )
        )
        gamma = jnp.asarray(self.adiabatic_index, dtype=density.dtype)
        light_speed_squared = jnp.asarray(
            float(self.scale.speed_of_light**2), dtype=density.dtype
        )
        pressure = (gamma - 1.0) * density * energy
        total_energy_density = density * (light_speed_squared + energy)
        enthalpy = light_speed_squared + energy + pressure / density
        pressure_density_derivative = (gamma - 1.0) * energy
        pressure_energy_derivative = (gamma - 1.0) * density
        sound_speed_squared = (
            light_speed_squared * gamma * pressure / (density * enthalpy)
        )
        density_below, density_above = _density_violations(
            density, self.minimum_density, self.maximum_density
        )
        energy_lower = jnp.asarray(
            self.minimum_specific_internal_energy, dtype=energy.dtype
        )
        thermal_below = energy < energy_lower
        thermal_above = _upper_violation(energy, self.maximum_specific_internal_energy)
        input_finite = jnp.isfinite(density) & jnp.isfinite(energy)
        if composition_supplied:
            input_finite = input_finite & jnp.isfinite(composition_value)
        output_finite = _finite_outputs(
            pressure,
            total_energy_density,
            enthalpy,
            sound_speed_squared,
            pressure_density_derivative,
            pressure_energy_derivative,
        )
        mechanically_stable = (
            (pressure >= 0.0) & (enthalpy > 0.0) & (sound_speed_squared >= 0.0)
        )
        causal = sound_speed_squared <= light_speed_squared
        false = jnp.zeros_like(density, dtype=jnp.bool_)
        true = jnp.ones_like(density, dtype=jnp.bool_)
        evidence = _domain_evidence(
            input_finite=input_finite,
            output_finite=output_finite,
            converged=true,
            density_below=density_below,
            density_above=density_above,
            thermal_below=thermal_below,
            thermal_above=thermal_above,
            composition_below=false,
            composition_above=false,
            cold_constraint_satisfied=true,
            mechanically_stable=mechanically_stable,
            causal=causal,
            branch_derivative_valid=true,
            temperature_defined=false,
            model_qualified=True,
            eos_id=self.eos_id,
        )
        return RelativisticEOSState(
            density,
            energy,
            pressure,
            total_energy_density,
            enthalpy,
            sound_speed_squared,
            jnp.full_like(density, jnp.nan),
            composition_value,
            pressure_density_derivative,
            pressure_energy_derivative,
            evidence,
            self.eos_id,
        )

    def evaluate_pressure(
        self,
        rest_mass_density: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        density = _real_array(rest_mass_density, "rest_mass_density")
        pressure_value = _real_array(pressure, "pressure")
        density, pressure_value = jnp.broadcast_arrays(density, pressure_value)
        gamma = jnp.asarray(self.adiabatic_index, dtype=density.dtype)
        energy = pressure_value / ((gamma - 1.0) * density)
        return self.evaluate(density, energy, composition)


class PiecewisePolytropicEOS(AbstractRelativisticEOS):
    """Continuous cold piecewise polytrope with derived energy constants."""

    scale: RelativityScaleContract = eqx.field(static=True)
    density_breaks: Array
    adiabatic_indices: Array
    polytropic_constants: Array
    specific_energy_offsets: Array
    minimum_density: float = eqx.field(static=True)
    maximum_density: float | None = eqx.field(static=True)
    cold_constraint_tolerance: float = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        density_breaks: ArrayLike,
        adiabatic_indices: ArrayLike,
        initial_polytropic_constant: float,
        /,
        *,
        initial_specific_energy_offset: float = 0.0,
        minimum_density: float = 0.0,
        maximum_density: float | None = None,
        cold_constraint_tolerance: float = 1.0e-7,
        provenance: str = "analytic continuous piecewise polytrope",
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        breaks = np.asarray(density_breaks, dtype=np.float64)
        gammas = np.asarray(adiabatic_indices, dtype=np.float64)
        if breaks.ndim != 1 or gammas.shape != (breaks.size + 1,):
            raise ValueError(
                "adiabatic_indices must contain exactly one more entry than the rank-one density_breaks array."
            )
        if (
            np.any(~np.isfinite(breaks))
            or np.any(breaks <= 0.0)
            or np.any(np.diff(breaks) <= 0.0)
        ):
            raise ValueError("density_breaks must be finite, positive, and increasing.")
        if np.any(~np.isfinite(gammas)) or np.any(gammas <= 1.0) or np.any(gammas > 2.0):
            raise ValueError("adiabatic_indices must be finite and lie in (1, 2].")
        first_constant = _positive_finite(
            initial_polytropic_constant, "initial_polytropic_constant"
        )
        first_offset = float(initial_specific_energy_offset)
        density_lower = float(minimum_density)
        tolerance = _positive_finite(
            cold_constraint_tolerance, "cold_constraint_tolerance"
        )
        if (
            not np.isfinite(first_offset)
            or first_offset < 0.0
            or not np.isfinite(density_lower)
            or density_lower < 0.0
        ):
            raise ValueError(
                "Initial energy offset and minimum_density must be finite and non-negative."
            )
        density_upper = _optional_upper_bound(
            maximum_density, density_lower, "maximum_density"
        )
        if breaks.size and breaks[0] <= density_lower:
            raise ValueError("All density breaks must exceed minimum_density.")
        if density_upper is not None and breaks.size and breaks[-1] >= density_upper:
            raise ValueError("All density breaks must lie below maximum_density.")
        constants = np.empty_like(gammas)
        offsets = np.empty_like(gammas)
        constants[0] = first_constant
        offsets[0] = first_offset
        for index, transition_density in enumerate(breaks, start=1):
            previous_gamma = gammas[index - 1]
            gamma = gammas[index]
            constants[index] = constants[index - 1] * transition_density ** (
                previous_gamma - gamma
            )
            previous_energy = offsets[index - 1] + (
                constants[index - 1]
                * transition_density ** (previous_gamma - 1.0)
                / (previous_gamma - 1.0)
            )
            offsets[index] = previous_energy - (
                constants[index] * transition_density ** (gamma - 1.0) / (gamma - 1.0)
            )
        light_speed_squared = float(scale.speed_of_light**2)
        for index, gamma in enumerate(gammas):
            sample_densities: list[float] = []
            if index:
                sample_densities.append(float(breaks[index - 1]))
            elif density_lower > 0.0:
                sample_densities.append(density_lower)
            if index < breaks.size:
                sample_densities.append(float(breaks[index]))
            elif density_upper is not None:
                sample_densities.append(density_upper)
            for density in sample_densities:
                pressure = constants[index] * density**gamma
                energy = offsets[index] + (
                    constants[index] * density ** (gamma - 1.0) / (gamma - 1.0)
                )
                enthalpy = light_speed_squared + energy + pressure / density
                sound_speed_squared = (
                    light_speed_squared * gamma * pressure / (density * enthalpy)
                )
                if (
                    not np.isfinite(sound_speed_squared)
                    or sound_speed_squared < 0.0
                    or sound_speed_squared > light_speed_squared
                ):
                    raise ValueError(
                        "Piecewise-polytropic parameters are unstable or acausal on the declared density domain."
                    )
        source = _declared_text(provenance, "provenance")
        self.scale = scale
        self.density_breaks = jnp.asarray(breaks)
        self.adiabatic_indices = jnp.asarray(gammas)
        self.polytropic_constants = jnp.asarray(constants)
        self.specific_energy_offsets = jnp.asarray(offsets)
        self.minimum_density = density_lower
        self.maximum_density = density_upper
        self.cold_constraint_tolerance = tolerance
        self.provenance = source
        self.eos_id = canonical_fingerprint(
            {
                "kind": "relativistic-piecewise-polytropic-eos",
                "scale": scale.scale_id,
                "density_breaks": array_tree_fingerprint(breaks),
                "adiabatic_indices": array_tree_fingerprint(gammas),
                "polytropic_constants": array_tree_fingerprint(constants),
                "specific_energy_offsets": array_tree_fingerprint(offsets),
                "density_domain": [density_lower, density_upper],
                "cold_constraint_tolerance": tolerance,
                "provenance": source,
            }
        )

    def _cold_values(
        self, rest_mass_density: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        density = _real_array(rest_mass_density, "rest_mass_density")
        segment = jnp.searchsorted(self.density_breaks, density, side="right")
        gamma = self.adiabatic_indices.astype(density.dtype)[segment]
        constant = self.polytropic_constants.astype(density.dtype)[segment]
        offset = self.specific_energy_offsets.astype(density.dtype)[segment]
        pressure = constant * density**gamma
        energy = offset + constant * density ** (gamma - 1.0) / (gamma - 1.0)
        light_speed_squared = jnp.asarray(
            float(self.scale.speed_of_light**2), dtype=density.dtype
        )
        enthalpy = light_speed_squared + energy + pressure / density
        pressure_density_derivative = gamma * pressure / density
        sound_speed_squared = light_speed_squared * pressure_density_derivative / enthalpy
        on_break = jnp.any(
            density[..., None] == self.density_breaks.astype(density.dtype), axis=-1
        )
        return (
            density,
            pressure,
            energy,
            enthalpy,
            sound_speed_squared,
            pressure_density_derivative,
            ~on_break,
        )

    def _state(
        self,
        rest_mass_density: ArrayLike,
        supplied_value: ArrayLike | None,
        composition: ArrayLike | None,
        /,
        *,
        supplied_kind: Literal["energy", "pressure"],
    ) -> RelativisticEOSState:
        density = _real_array(rest_mass_density, "rest_mass_density")
        composition_supplied = composition is not None
        supplied: Array | None = None
        if supplied_value is not None:
            supplied = _real_array(supplied_value, supplied_kind)
        if composition_supplied:
            composition_value = _real_array(composition, "composition")
            if supplied is None:
                density, composition_value = jnp.broadcast_arrays(
                    density, composition_value
                )
            else:
                density, supplied, composition_value = jnp.broadcast_arrays(
                    density, supplied, composition_value
                )
        elif supplied is None:
            composition_value = jnp.full_like(density, jnp.nan)
        else:
            density, supplied = jnp.broadcast_arrays(density, supplied)
            composition_value = jnp.full_like(density, jnp.nan)
        (
            density,
            pressure,
            energy,
            enthalpy,
            sound_speed_squared,
            pressure_density_derivative,
            branch_derivative_valid,
        ) = self._cold_values(density)
        input_finite = jnp.isfinite(density)
        cold_constraint = jnp.ones_like(density, dtype=jnp.bool_)
        if supplied is not None:
            expected = energy if supplied_kind == "energy" else pressure
            scale = jnp.maximum(1.0, jnp.abs(expected))
            cold_constraint = (
                jnp.abs(supplied - expected)
                <= jnp.asarray(self.cold_constraint_tolerance, dtype=density.dtype)
                * scale
            )
            input_finite = input_finite & jnp.isfinite(supplied)
        if composition_supplied:
            input_finite = input_finite & jnp.isfinite(composition_value)
        light_speed_squared = jnp.asarray(
            float(self.scale.speed_of_light**2), dtype=density.dtype
        )
        total_energy_density = density * (light_speed_squared + energy)
        density_below, density_above = _density_violations(
            density, self.minimum_density, self.maximum_density
        )
        pressure_energy_derivative = jnp.zeros_like(density)
        output_finite = _finite_outputs(
            pressure,
            energy,
            enthalpy,
            total_energy_density,
            sound_speed_squared,
            pressure_density_derivative,
        )
        mechanically_stable = (
            (pressure >= 0.0)
            & (enthalpy > 0.0)
            & (pressure_density_derivative >= 0.0)
            & (sound_speed_squared >= 0.0)
        )
        causal = sound_speed_squared <= light_speed_squared
        false = jnp.zeros_like(density, dtype=jnp.bool_)
        true = jnp.ones_like(density, dtype=jnp.bool_)
        evidence = _domain_evidence(
            input_finite=input_finite,
            output_finite=output_finite,
            converged=true,
            density_below=density_below,
            density_above=density_above,
            thermal_below=false,
            thermal_above=false,
            composition_below=false,
            composition_above=false,
            cold_constraint_satisfied=cold_constraint,
            mechanically_stable=mechanically_stable,
            causal=causal,
            branch_derivative_valid=branch_derivative_valid,
            temperature_defined=false,
            model_qualified=True,
            eos_id=self.eos_id,
        )
        return RelativisticEOSState(
            density,
            energy,
            pressure,
            total_energy_density,
            enthalpy,
            sound_speed_squared,
            jnp.full_like(density, jnp.nan),
            composition_value,
            pressure_density_derivative,
            pressure_energy_derivative,
            evidence,
            self.eos_id,
        )

    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        return self._state(
            rest_mass_density,
            specific_internal_energy,
            composition,
            supplied_kind="energy",
        )

    def evaluate_pressure(
        self,
        rest_mass_density: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        return self._state(
            rest_mass_density, pressure, composition, supplied_kind="pressure"
        )


class HybridColdThermalEOS(AbstractRelativisticEOS):
    """Continuous cold piecewise polytrope plus a gamma-law thermal excess."""

    cold_eos: PiecewisePolytropicEOS
    scale: RelativityScaleContract = eqx.field(static=True)
    thermal_adiabatic_index: float = eqx.field(static=True)
    minimum_thermal_specific_energy: float = eqx.field(static=True)
    maximum_thermal_specific_energy: float | None = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        cold_eos: PiecewisePolytropicEOS,
        thermal_adiabatic_index: float,
        /,
        *,
        minimum_thermal_specific_energy: float = 0.0,
        maximum_thermal_specific_energy: float | None = None,
        provenance: str = "analytic hybrid cold-plus-thermal EOS",
    ):
        if not isinstance(cold_eos, PiecewisePolytropicEOS):
            raise TypeError("cold_eos must be a PiecewisePolytropicEOS.")
        gamma = float(thermal_adiabatic_index)
        if not np.isfinite(gamma) or not 1.0 < gamma <= 2.0:
            raise ValueError("thermal_adiabatic_index must be finite and lie in (1, 2].")
        lower = float(minimum_thermal_specific_energy)
        if not np.isfinite(lower) or lower < 0.0:
            raise ValueError(
                "minimum_thermal_specific_energy must be finite and non-negative."
            )
        upper = _optional_upper_bound(
            maximum_thermal_specific_energy,
            lower,
            "maximum_thermal_specific_energy",
        )
        source = _declared_text(provenance, "provenance")
        self.cold_eos = cold_eos
        self.scale = cold_eos.scale
        self.thermal_adiabatic_index = gamma
        self.minimum_thermal_specific_energy = lower
        self.maximum_thermal_specific_energy = upper
        self.provenance = source
        self.eos_id = canonical_fingerprint(
            {
                "kind": "relativistic-hybrid-cold-thermal-eos",
                "cold_eos": cold_eos.eos_id,
                "thermal_adiabatic_index": gamma,
                "thermal_specific_energy_domain": [lower, upper],
                "provenance": source,
            }
        )

    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        if specific_internal_energy is None:
            raise TypeError("Hybrid EOS requires specific_internal_energy.")
        density, energy, composition_value, composition_supplied = (
            _broadcast_state_inputs(
                rest_mass_density, specific_internal_energy, composition
            )
        )
        (
            density,
            cold_pressure,
            cold_energy,
            _cold_enthalpy,
            _cold_sound_speed_squared,
            cold_pressure_derivative,
            branch_derivative_valid,
        ) = self.cold_eos._cold_values(density)
        thermal_energy = energy - cold_energy
        gamma = jnp.asarray(self.thermal_adiabatic_index, dtype=density.dtype)
        pressure = cold_pressure + (gamma - 1.0) * density * thermal_energy
        pressure_energy_derivative = (gamma - 1.0) * density
        pressure_density_derivative = cold_pressure_derivative + (gamma - 1.0) * (
            thermal_energy - cold_pressure / density
        )
        light_speed_squared = jnp.asarray(
            float(self.scale.speed_of_light**2), dtype=density.dtype
        )
        total_energy_density = density * (light_speed_squared + energy)
        enthalpy = light_speed_squared + energy + pressure / density
        adiabatic_pressure_derivative = pressure_density_derivative + (
            pressure * pressure_energy_derivative / density**2
        )
        sound_speed_squared = (
            light_speed_squared * adiabatic_pressure_derivative / enthalpy
        )
        density_below, density_above = _density_violations(
            density,
            self.cold_eos.minimum_density,
            self.cold_eos.maximum_density,
        )
        thermal_below = thermal_energy < jnp.asarray(
            self.minimum_thermal_specific_energy, dtype=density.dtype
        )
        thermal_above = _upper_violation(
            thermal_energy, self.maximum_thermal_specific_energy
        )
        input_finite = jnp.isfinite(density) & jnp.isfinite(energy)
        if composition_supplied:
            input_finite = input_finite & jnp.isfinite(composition_value)
        output_finite = _finite_outputs(
            pressure,
            total_energy_density,
            enthalpy,
            sound_speed_squared,
            pressure_density_derivative,
            pressure_energy_derivative,
        )
        mechanically_stable = (
            (pressure >= 0.0)
            & (enthalpy > 0.0)
            & (adiabatic_pressure_derivative >= 0.0)
            & (sound_speed_squared >= 0.0)
        )
        causal = sound_speed_squared <= light_speed_squared
        false = jnp.zeros_like(density, dtype=jnp.bool_)
        true = jnp.ones_like(density, dtype=jnp.bool_)
        evidence = _domain_evidence(
            input_finite=input_finite,
            output_finite=output_finite,
            converged=true,
            density_below=density_below,
            density_above=density_above,
            thermal_below=thermal_below,
            thermal_above=thermal_above,
            composition_below=false,
            composition_above=false,
            cold_constraint_satisfied=true,
            mechanically_stable=mechanically_stable,
            causal=causal,
            branch_derivative_valid=branch_derivative_valid,
            temperature_defined=false,
            model_qualified=True,
            eos_id=self.eos_id,
        )
        return RelativisticEOSState(
            density,
            energy,
            pressure,
            total_energy_density,
            enthalpy,
            sound_speed_squared,
            jnp.full_like(density, jnp.nan),
            composition_value,
            pressure_density_derivative,
            pressure_energy_derivative,
            evidence,
            self.eos_id,
        )

    def evaluate_pressure(
        self,
        rest_mass_density: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        density = _real_array(rest_mass_density, "rest_mass_density")
        pressure_value = _real_array(pressure, "pressure")
        density, pressure_value = jnp.broadcast_arrays(density, pressure_value)
        (
            density,
            cold_pressure,
            cold_energy,
            _cold_enthalpy,
            _cold_sound_speed_squared,
            _cold_pressure_derivative,
            _branch_derivative_valid,
        ) = self.cold_eos._cold_values(density)
        gamma = jnp.asarray(self.thermal_adiabatic_index, dtype=density.dtype)
        energy = cold_energy + (pressure_value - cold_pressure) / (
            (gamma - 1.0) * density
        )
        return self.evaluate(density, energy, composition)


class TabulatedFiniteTemperatureEOS(AbstractRelativisticEOS):
    """Bounded trilinear EOS over density, temperature, and composition.

    Pressure and specific-energy tables are host validated. The tabulated sound
    speed is derived from thermodynamic partial derivatives, so enthalpy and wave
    speeds share the same pressure-energy convention. Every inverse follows a
    monotone, fixed-capacity table row and returns NaNs with exact status outside
    support; boundary values are never reused as extrapolated states.
    """

    scale: RelativityScaleContract = eqx.field(static=True)
    density_nodes: Array
    temperature_nodes: Array
    composition_nodes: Array
    fields: Array
    table_evidence: RelativisticEOSTableEvidence = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    source_checksum: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        density_nodes: ArrayLike,
        temperature_nodes: ArrayLike,
        composition_nodes: ArrayLike,
        pressure: ArrayLike,
        specific_internal_energy: ArrayLike,
        /,
        *,
        provenance: str,
        source_checksum: str,
        license_id: str,
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        density = np.asarray(density_nodes, dtype=np.float64)
        temperature = np.asarray(temperature_nodes, dtype=np.float64)
        composition = np.asarray(composition_nodes, dtype=np.float64)
        axes = (density, temperature, composition)
        if any(axis.ndim != 1 or axis.size < 2 for axis in axes):
            raise ValueError(
                "Tabulated EOS axes must be rank-one arrays with at least two nodes."
            )
        if any(
            np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0) for axis in axes
        ):
            raise ValueError("Tabulated EOS axes must be finite and strictly increasing.")
        if np.any(density <= 0.0) or np.any(temperature <= 0.0):
            raise ValueError("Density and temperature nodes must be strictly positive.")
        if composition[0] < 0.0 or composition[-1] > 1.0:
            raise ValueError("Composition nodes must lie in the closed interval [0, 1].")
        pressure_values = np.asarray(pressure, dtype=np.float64)
        energy_values = np.asarray(specific_internal_energy, dtype=np.float64)
        expected_shape = (density.size, temperature.size, composition.size)
        if (
            pressure_values.shape != expected_shape
            or energy_values.shape != expected_shape
        ):
            raise ValueError(
                "Pressure and specific-energy tables must match the three axis sizes."
            )
        if np.any(~np.isfinite(pressure_values)) or np.any(~np.isfinite(energy_values)):
            raise ValueError("Tabulated EOS values must be finite.")
        if np.any(pressure_values < 0.0):
            raise ValueError("Tabulated pressure must be non-negative.")
        pressure_density_monotone = bool(np.all(np.diff(pressure_values, axis=0) >= 0.0))
        pressure_temperature_monotone = bool(
            np.all(np.diff(pressure_values, axis=1) > 0.0)
        )
        if not pressure_density_monotone:
            raise ValueError("Tabulated pressure must be nondecreasing with density.")
        if not pressure_temperature_monotone:
            raise ValueError(
                "Tabulated pressure must increase strictly with temperature."
            )
        energy_temperature_differences = np.diff(energy_values, axis=1)
        if np.any(energy_temperature_differences <= 0.0):
            raise ValueError(
                "Tabulated specific energy must increase strictly with temperature."
            )
        light_speed_squared = float(scale.speed_of_light**2)
        minimum_heat_capacity = np.inf
        minimum_adiabatic = np.inf
        minimum_causality_margin = np.inf
        for density_index in range(density.size - 1):
            density_lower = density[density_index]
            density_upper = density[density_index + 1]
            density_width = density_upper - density_lower
            for temperature_index in range(temperature.size - 1):
                temperature_width = (
                    temperature[temperature_index + 1] - temperature[temperature_index]
                )
                for composition_index in range(composition.size - 1):
                    density_slice = slice(density_index, density_index + 2)
                    temperature_slice = slice(temperature_index, temperature_index + 2)
                    composition_slice = slice(composition_index, composition_index + 2)
                    pressure_cell = pressure_values[
                        density_slice, temperature_slice, composition_slice
                    ]
                    energy_cell = energy_values[
                        density_slice, temperature_slice, composition_slice
                    ]
                    pressure_density_slopes = (
                        pressure_values[
                            density_index + 1,
                            temperature_slice,
                            composition_slice,
                        ]
                        - pressure_values[
                            density_index,
                            temperature_slice,
                            composition_slice,
                        ]
                    ) / density_width
                    pressure_temperature_slopes = (
                        pressure_values[
                            density_slice,
                            temperature_index + 1,
                            composition_slice,
                        ]
                        - pressure_values[
                            density_slice,
                            temperature_index,
                            composition_slice,
                        ]
                    ) / temperature_width
                    energy_density_slopes = (
                        energy_values[
                            density_index + 1,
                            temperature_slice,
                            composition_slice,
                        ]
                        - energy_values[
                            density_index,
                            temperature_slice,
                            composition_slice,
                        ]
                    ) / density_width
                    heat_capacity_slopes = (
                        energy_values[
                            density_slice,
                            temperature_index + 1,
                            composition_slice,
                        ]
                        - energy_values[
                            density_slice,
                            temperature_index,
                            composition_slice,
                        ]
                    ) / temperature_width
                    heat_capacity_lower = float(np.min(heat_capacity_slopes))
                    heat_capacity_upper = float(np.max(heat_capacity_slopes))
                    pressure_temperature_lower = float(
                        np.min(pressure_temperature_slopes)
                    )
                    pressure_temperature_upper = float(
                        np.max(pressure_temperature_slopes)
                    )
                    kappa_lower = pressure_temperature_lower / heat_capacity_upper
                    kappa_upper = pressure_temperature_upper / heat_capacity_lower
                    energy_density_lower = float(np.min(energy_density_slopes))
                    energy_density_upper = float(np.max(energy_density_slopes))
                    product_candidates = (
                        kappa_lower * energy_density_lower,
                        kappa_lower * energy_density_upper,
                        kappa_upper * energy_density_lower,
                        kappa_upper * energy_density_upper,
                    )
                    chi_lower = float(np.min(pressure_density_slopes)) - max(
                        product_candidates
                    )
                    chi_upper = float(np.max(pressure_density_slopes)) - min(
                        product_candidates
                    )
                    pressure_lower = float(np.min(pressure_cell))
                    pressure_upper = float(np.max(pressure_cell))
                    thermal_term_lower = pressure_lower * kappa_lower / density_upper**2
                    thermal_term_upper = pressure_upper * kappa_upper / density_lower**2
                    adiabatic_lower = chi_lower + thermal_term_lower
                    adiabatic_upper = chi_upper + thermal_term_upper
                    enthalpy_lower = (
                        light_speed_squared
                        + float(np.min(energy_cell))
                        + pressure_lower / density_upper
                    )
                    sound_speed_upper = (
                        light_speed_squared * adiabatic_upper / enthalpy_lower
                    )
                    if not np.isfinite(heat_capacity_lower) or heat_capacity_lower <= 0.0:
                        raise ValueError(
                            "Tabulated specific energy must have finite positive heat capacity."
                        )
                    if (
                        not np.isfinite(adiabatic_lower)
                        or adiabatic_lower < 0.0
                        or not np.isfinite(enthalpy_lower)
                        or enthalpy_lower <= 0.0
                        or not np.isfinite(sound_speed_upper)
                        or sound_speed_upper > light_speed_squared
                    ):
                        raise ValueError(
                            "Tabulated EOS must be mechanically stable and causal."
                        )
                    minimum_heat_capacity = min(
                        minimum_heat_capacity, heat_capacity_lower
                    )
                    minimum_adiabatic = min(minimum_adiabatic, adiabatic_lower)
                    minimum_causality_margin = min(
                        minimum_causality_margin,
                        light_speed_squared - sound_speed_upper,
                    )
        minimum_heat_capacity = float(minimum_heat_capacity)
        minimum_adiabatic = float(minimum_adiabatic)
        minimum_causality_margin = float(minimum_causality_margin)
        source = _declared_text(provenance, "provenance")
        checksum = _declared_text(source_checksum, "source_checksum")
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValueError("source_checksum must be a lowercase SHA-256 hex digest.")
        license_name = _declared_text(license_id, "license_id")
        fields_host = np.stack((pressure_values, energy_values), axis=-1)
        table_id = canonical_fingerprint(
            {
                "kind": "bounded-finite-temperature-composition-eos-table",
                "scale": scale.scale_id,
                "density_nodes": array_tree_fingerprint(density),
                "temperature_nodes": array_tree_fingerprint(temperature),
                "composition_nodes": array_tree_fingerprint(composition),
                "pressure": array_tree_fingerprint(pressure_values),
                "specific_internal_energy": array_tree_fingerprint(energy_values),
                "provenance": source,
                "source_checksum": checksum,
                "license_id": license_name,
            }
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "relativistic-eos-table-evidence",
                "table": table_id,
                "axes_monotone": True,
                "values_finite": True,
                "pressure_monotone_density": pressure_density_monotone,
                "pressure_monotone_temperature": pressure_temperature_monotone,
                "positive_heat_capacity": True,
                "mechanically_stable": True,
                "causal": True,
                "minimum_heat_capacity": minimum_heat_capacity,
                "minimum_adiabatic_pressure_derivative": minimum_adiabatic,
                "minimum_causality_margin": minimum_causality_margin,
            }
        )
        table_evidence = RelativisticEOSTableEvidence(
            True,
            True,
            pressure_density_monotone,
            pressure_temperature_monotone,
            True,
            True,
            True,
            minimum_heat_capacity,
            minimum_adiabatic,
            minimum_causality_margin,
            evidence_id,
        )
        self.scale = scale
        self.density_nodes = jnp.asarray(density)
        self.temperature_nodes = jnp.asarray(temperature)
        self.composition_nodes = jnp.asarray(composition)
        self.fields = jnp.asarray(fields_host)
        self.table_evidence = table_evidence
        self.provenance = source
        self.source_checksum = checksum
        self.license_id = license_name
        self.table_id = table_id
        self.eos_id = canonical_fingerprint(
            {
                "kind": "relativistic-tabulated-finite-temperature-eos",
                "table": table_id,
                "interpolation": "bounded-trilinear",
                "thermodynamic_derivatives": "jax-differentiated-trilinear",
            }
        )

    def _interpolate_fields(
        self,
        rest_mass_density: Array,
        temperature: Array,
        composition: Array,
        /,
    ) -> tuple[Array, Array]:
        density, temperature_value, composition_value = jnp.broadcast_arrays(
            rest_mass_density, temperature, composition
        )
        finite = (
            jnp.isfinite(density)
            & jnp.isfinite(temperature_value)
            & jnp.isfinite(composition_value)
        )
        coordinates = jnp.stack(
            (
                jnp.where(finite, density, self.density_nodes[0]),
                jnp.where(finite, temperature_value, self.temperature_nodes[0]),
                jnp.where(finite, composition_value, self.composition_nodes[0]),
            ),
            axis=-1,
        )
        stencil = rectilinear_stencil(
            (self.density_nodes, self.temperature_nodes, self.composition_nodes),
            coordinates,
            boundary=("constant", "constant", "constant"),
        )
        interpolated = apply_gather_stencil(
            self.fields.reshape((-1, self.fields.shape[-1])), stencil
        )
        support = interpolated.support & finite
        values = jnp.where(
            support[..., None],
            interpolated.values,
            jnp.asarray(jnp.nan, dtype=interpolated.values.dtype),
        )
        return values, support

    def _branch_derivative_valid(
        self, density: Array, temperature: Array, composition: Array, /
    ) -> Array:
        density_knot = jnp.any(
            density[..., None] == self.density_nodes.astype(density.dtype),
            axis=-1,
        )
        temperature_knot = jnp.any(
            temperature[..., None] == self.temperature_nodes.astype(temperature.dtype),
            axis=-1,
        )
        composition_knot = jnp.any(
            composition[..., None] == self.composition_nodes.astype(composition.dtype),
            axis=-1,
        )
        return ~(density_knot | temperature_knot | composition_knot)

    def _state_from_temperature(
        self,
        density: Array,
        temperature: Array,
        composition: Array,
        /,
        *,
        input_finite: Array,
        density_below: Array,
        density_above: Array,
        thermal_below: Array,
        thermal_above: Array,
        composition_below: Array,
        composition_above: Array,
        converged: Array,
        branch_derivative_valid: Array,
        expose_temperature: Array,
    ) -> RelativisticEOSState:
        values, support = self._interpolate_fields(density, temperature, composition)
        _values, density_derivatives = jax.jvp(
            lambda value: self._interpolate_fields(value, temperature, composition)[0],
            (density,),
            (jnp.ones_like(density),),
        )
        _values, temperature_derivatives = jax.jvp(
            lambda value: self._interpolate_fields(density, value, composition)[0],
            (temperature,),
            (jnp.ones_like(temperature),),
        )
        pressure = values[..., 0]
        energy = values[..., 1]
        pressure_temperature_derivative = temperature_derivatives[..., 0]
        energy_density_derivative = density_derivatives[..., 1]
        heat_capacity = temperature_derivatives[..., 1]
        pressure_energy_derivative = pressure_temperature_derivative / heat_capacity
        pressure_density_derivative = density_derivatives[..., 0] - (
            pressure_energy_derivative * energy_density_derivative
        )
        light_speed_squared = jnp.asarray(
            float(self.scale.speed_of_light**2), dtype=density.dtype
        )
        total_energy_density = density * (light_speed_squared + energy)
        enthalpy = light_speed_squared + energy + pressure / density
        adiabatic_pressure_derivative = pressure_density_derivative + (
            pressure * pressure_energy_derivative / density**2
        )
        sound_speed_squared = (
            light_speed_squared * adiabatic_pressure_derivative / enthalpy
        )
        mechanically_stable = (
            support
            & (pressure >= 0.0)
            & (enthalpy > 0.0)
            & (adiabatic_pressure_derivative >= 0.0)
            & (heat_capacity > 0.0)
            & (sound_speed_squared >= 0.0)
        )
        causal = support & (sound_speed_squared <= light_speed_squared)
        output_finite = support & _finite_outputs(
            pressure,
            energy,
            sound_speed_squared,
            pressure_density_derivative,
            pressure_energy_derivative,
            total_energy_density,
            enthalpy,
        )
        true = jnp.ones_like(density, dtype=jnp.bool_)
        evidence = _domain_evidence(
            input_finite=input_finite,
            output_finite=output_finite,
            converged=converged,
            density_below=density_below,
            density_above=density_above,
            thermal_below=thermal_below,
            thermal_above=thermal_above,
            composition_below=composition_below,
            composition_above=composition_above,
            cold_constraint_satisfied=true,
            mechanically_stable=mechanically_stable,
            causal=causal,
            branch_derivative_valid=branch_derivative_valid,
            temperature_defined=expose_temperature,
            model_qualified=self.table_evidence.qualified,
            eos_id=self.eos_id,
        )
        return RelativisticEOSState(
            density,
            energy,
            pressure,
            total_energy_density,
            enthalpy,
            sound_speed_squared,
            jnp.where(expose_temperature, temperature, jnp.nan),
            composition,
            pressure_density_derivative,
            pressure_energy_derivative,
            evidence,
            self.eos_id,
        )

    def evaluate_temperature(
        self,
        rest_mass_density: ArrayLike,
        temperature: ArrayLike,
        composition: ArrayLike,
        /,
    ) -> RelativisticEOSState:
        density, temperature_value, composition_value, _ = _broadcast_state_inputs(
            rest_mass_density, temperature, composition
        )
        input_finite = (
            jnp.isfinite(density)
            & jnp.isfinite(temperature_value)
            & jnp.isfinite(composition_value)
        )
        density_below = density < self.density_nodes[0]
        density_above = density > self.density_nodes[-1]
        thermal_below = temperature_value < self.temperature_nodes[0]
        thermal_above = temperature_value > self.temperature_nodes[-1]
        composition_below = composition_value < self.composition_nodes[0]
        composition_above = composition_value > self.composition_nodes[-1]
        within_domain = ~(
            density_below
            | density_above
            | thermal_below
            | thermal_above
            | composition_below
            | composition_above
        )
        branch_valid = self._branch_derivative_valid(
            density, temperature_value, composition_value
        )
        return self._state_from_temperature(
            density,
            temperature_value,
            composition_value,
            input_finite=input_finite,
            density_below=density_below,
            density_above=density_above,
            thermal_below=thermal_below,
            thermal_above=thermal_above,
            composition_below=composition_below,
            composition_above=composition_above,
            converged=jnp.ones_like(density, dtype=jnp.bool_),
            branch_derivative_valid=branch_valid,
            expose_temperature=input_finite & within_domain,
        )

    def _thermal_profile(
        self,
        density: Array,
        composition: Array,
        field_index: int,
        /,
    ) -> Array:
        shape = density.shape + (self.temperature_nodes.size,)
        density_grid = jnp.broadcast_to(density[..., None], shape)
        temperature_grid = jnp.broadcast_to(self.temperature_nodes, shape)
        composition_grid = jnp.broadcast_to(composition[..., None], shape)
        values, _support = self._interpolate_fields(
            density_grid, temperature_grid, composition_grid
        )
        return values[..., field_index]

    def _evaluate_inverse(
        self,
        rest_mass_density: ArrayLike,
        target: ArrayLike,
        composition: ArrayLike | None,
        /,
        *,
        field_index: Literal[0, 1],
    ) -> RelativisticEOSState:
        if composition is None:
            raise TypeError("Tabulated finite-temperature EOS requires composition.")
        density, target_value, composition_value, _ = _broadcast_state_inputs(
            rest_mass_density, target, composition
        )
        input_finite = (
            jnp.isfinite(density)
            & jnp.isfinite(target_value)
            & jnp.isfinite(composition_value)
        )
        density_below = density < self.density_nodes[0]
        density_above = density > self.density_nodes[-1]
        composition_below = composition_value < self.composition_nodes[0]
        composition_above = composition_value > self.composition_nodes[-1]
        profile = self._thermal_profile(density, composition_value, field_index)
        thermal_below = target_value < profile[..., 0]
        thermal_above = target_value > profile[..., -1]
        flat_profile = profile.reshape((-1, profile.shape[-1]))
        flat_target = target_value.reshape((-1,))
        upper_flat = jax.vmap(
            lambda row, value: jnp.searchsorted(row, value, side="right")
        )(flat_profile, flat_target)
        upper = jnp.clip(upper_flat, 1, profile.shape[-1] - 1).reshape(target_value.shape)
        lower = upper - 1
        lower_value = jnp.take_along_axis(profile, lower[..., None], axis=-1)[..., 0]
        upper_value = jnp.take_along_axis(profile, upper[..., None], axis=-1)[..., 0]
        lower_temperature = self.temperature_nodes[lower]
        upper_temperature = self.temperature_nodes[upper]
        fraction = (target_value - lower_value) / (upper_value - lower_value)
        solved_temperature = lower_temperature + fraction * (
            upper_temperature - lower_temperature
        )
        values, support = self._interpolate_fields(
            density, solved_temperature, composition_value
        )
        reconstructed = values[..., field_index]
        tolerance = (
            16.0
            * jnp.finfo(reconstructed.dtype).eps
            * jnp.maximum(1.0, jnp.abs(target_value))
        )
        converged = support & (jnp.abs(reconstructed - target_value) <= tolerance)
        on_thermal_knot = jnp.any(target_value[..., None] == profile[..., 1:-1], axis=-1)
        branch_valid = (
            self._branch_derivative_valid(density, solved_temperature, composition_value)
            & ~on_thermal_knot
        )
        within_domain = ~(
            density_below
            | density_above
            | thermal_below
            | thermal_above
            | composition_below
            | composition_above
        )
        expose_temperature = input_finite & within_domain & converged
        state = self._state_from_temperature(
            density,
            solved_temperature,
            composition_value,
            input_finite=input_finite,
            density_below=density_below,
            density_above=density_above,
            thermal_below=thermal_below,
            thermal_above=thermal_above,
            composition_below=composition_below,
            composition_above=composition_above,
            converged=converged,
            branch_derivative_valid=branch_valid,
            expose_temperature=expose_temperature,
        )
        valid = state.qualified
        nan = jnp.asarray(jnp.nan, dtype=state.pressure.dtype)
        return RelativisticEOSState(
            state.rest_mass_density,
            jnp.where(valid, state.specific_internal_energy, nan),
            jnp.where(valid, state.pressure, nan),
            jnp.where(valid, state.total_energy_density, nan),
            jnp.where(valid, state.specific_enthalpy, nan),
            jnp.where(valid, state.sound_speed_squared, nan),
            jnp.where(valid, state.temperature, nan),
            state.composition,
            jnp.where(valid, state.pressure_density_derivative, nan),
            jnp.where(valid, state.pressure_specific_internal_energy_derivative, nan),
            state.evidence,
            self.eos_id,
        )

    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        specific_internal_energy: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        if specific_internal_energy is None:
            raise TypeError(
                "Tabulated finite-temperature EOS requires specific_internal_energy."
            )
        return self._evaluate_inverse(
            rest_mass_density,
            specific_internal_energy,
            composition,
            field_index=1,
        )

    def evaluate_pressure(
        self,
        rest_mass_density: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> RelativisticEOSState:
        return self._evaluate_inverse(
            rest_mass_density, pressure, composition, field_index=0
        )


__all__ = [
    "AbstractRelativisticEOS",
    "GammaLawEOS",
    "HybridColdThermalEOS",
    "PiecewisePolytropicEOS",
    "RELATIVISTIC_EOS_ACAUSAL",
    "RELATIVISTIC_EOS_COLD_CONSTRAINT_MISMATCH",
    "RELATIVISTIC_EOS_COMPOSITION_ABOVE_DOMAIN",
    "RELATIVISTIC_EOS_COMPOSITION_BELOW_DOMAIN",
    "RELATIVISTIC_EOS_DENSITY_ABOVE_DOMAIN",
    "RELATIVISTIC_EOS_DENSITY_BELOW_DOMAIN",
    "RELATIVISTIC_EOS_NONCONVERGED",
    "RELATIVISTIC_EOS_NONFINITE",
    "RELATIVISTIC_EOS_SUCCESS",
    "RELATIVISTIC_EOS_THERMAL_ABOVE_DOMAIN",
    "RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN",
    "RELATIVISTIC_EOS_UNSTABLE",
    "RelativisticEOSDomainEvidence",
    "RelativisticEOSState",
    "RelativisticEOSStatus",
    "RelativisticEOSTableEvidence",
    "TabulatedFiniteTemperatureEOS",
    "relativistic_eos_status_name",
]
