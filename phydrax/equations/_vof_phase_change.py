#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._multiphase import TwoMaterialVOFSystem


class ConservativePhaseTransferEvaluation(StrictModule):
    raw_mass_rate: Array
    limited_mass_rate: Array
    phase_mass_rates: Array
    volume_rate: Array
    latent_power: Array
    transfer_factor: Array
    explicit_step_restriction: Array
    donor_margin: Array
    equilibrium_margin: Array
    mass_defect: Array
    energy_defect: Array
    finite: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class VOFPhaseChangeStepResult(StrictModule):
    previous_state: Array
    state: Array
    transfer: ConservativePhaseTransferEvaluation
    alpha_rate: Array
    pressure_before: Array
    pressure_after: Array
    admissible: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class AbstractVOFMassTransferPlan(StrictModule, NonTrainableState):
    latent_heat: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def raw_rate(
        self,
        pressure: Array,
        temperature: Array,
        alpha0: Array,
        density0: Array,
        density1: Array,
        interface_area_density: Array,
        heat_flux0: Array,
        heat_flux1: Array,
        /,
    ) -> tuple[Array, Array]:
        raise NotImplementedError


def _positive_material_parameters(values, owner):
    result = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{owner} parameters must be positive and finite.")
    return result


class MerkleCavitationPlan(AbstractVOFMassTransferPlan):
    saturation_pressure: float = eqx.field(static=True)
    vaporization_coefficient: float = eqx.field(static=True)
    condensation_coefficient: float = eqx.field(static=True)

    def __init__(
        self,
        saturation_pressure,
        vaporization_coefficient,
        condensation_coefficient,
        latent_heat,
        /,
    ):
        p_sat, vaporization, condensation, latent = _positive_material_parameters(
            (
                saturation_pressure,
                vaporization_coefficient,
                condensation_coefficient,
                latent_heat,
            ),
            "Merkle cavitation",
        )
        self.saturation_pressure = p_sat
        self.vaporization_coefficient = vaporization
        self.condensation_coefficient = condensation
        self.latent_heat = latent
        self.model_id = canonical_fingerprint(
            {
                "kind": "merkle-cavitation",
                "saturation_pressure": p_sat,
                "vaporization_coefficient": vaporization,
                "condensation_coefficient": condensation,
                "latent_heat": latent,
            }
        )

    def raw_rate(
        self,
        pressure,
        temperature,
        alpha0,
        density0,
        density1,
        interface_area_density,
        heat_flux0,
        heat_flux1,
        /,
    ):
        del temperature, interface_area_density, heat_flux0, heat_flux1
        deficit = jnp.maximum(self.saturation_pressure - pressure, 0.0)
        excess = jnp.maximum(pressure - self.saturation_pressure, 0.0)
        evaporation = self.vaporization_coefficient * alpha0 * density0 * deficit
        condensation = self.condensation_coefficient * (1.0 - alpha0) * density1 * excess
        return evaporation - condensation, pressure - self.saturation_pressure


class KunzCavitationPlan(AbstractVOFMassTransferPlan):
    saturation_pressure: float = eqx.field(static=True)
    vaporization_time: float = eqx.field(static=True)
    condensation_time: float = eqx.field(static=True)
    reference_dynamic_pressure: float = eqx.field(static=True)

    def __init__(
        self,
        saturation_pressure,
        vaporization_time,
        condensation_time,
        reference_dynamic_pressure,
        latent_heat,
        /,
    ):
        p_sat, vaporization, condensation, dynamic, latent = (
            _positive_material_parameters(
                (
                    saturation_pressure,
                    vaporization_time,
                    condensation_time,
                    reference_dynamic_pressure,
                    latent_heat,
                ),
                "Kunz cavitation",
            )
        )
        self.saturation_pressure = p_sat
        self.vaporization_time = vaporization
        self.condensation_time = condensation
        self.reference_dynamic_pressure = dynamic
        self.latent_heat = latent
        self.model_id = canonical_fingerprint(
            {
                "kind": "kunz-cavitation",
                "saturation_pressure": p_sat,
                "vaporization_time": vaporization,
                "condensation_time": condensation,
                "reference_dynamic_pressure": dynamic,
                "latent_heat": latent,
            }
        )

    def raw_rate(
        self,
        pressure,
        temperature,
        alpha0,
        density0,
        density1,
        interface_area_density,
        heat_flux0,
        heat_flux1,
        /,
    ):
        del temperature, interface_area_density, heat_flux0, heat_flux1
        deficit = jnp.maximum(self.saturation_pressure - pressure, 0.0)
        excess = jnp.maximum(pressure - self.saturation_pressure, 0.0)
        evaporation = (
            alpha0
            * density0
            * deficit
            / (self.reference_dynamic_pressure * self.vaporization_time)
        )
        condensation = (
            (1.0 - alpha0) ** 2
            * density1
            * excess
            / (self.reference_dynamic_pressure * self.condensation_time)
        )
        return evaporation - condensation, pressure - self.saturation_pressure


class SchnerrSauerCavitationPlan(AbstractVOFMassTransferPlan):
    saturation_pressure: float = eqx.field(static=True)
    nuclei_number_density: float = eqx.field(static=True)
    nuclei_diameter: float = eqx.field(static=True)
    vaporization_coefficient: float = eqx.field(static=True)
    condensation_coefficient: float = eqx.field(static=True)
    radius_floor: float = eqx.field(static=True)

    def __init__(
        self,
        saturation_pressure,
        nuclei_number_density,
        nuclei_diameter,
        vaporization_coefficient,
        condensation_coefficient,
        latent_heat,
        /,
        *,
        radius_floor=1.0e-12,
    ):
        values = _positive_material_parameters(
            (
                saturation_pressure,
                nuclei_number_density,
                nuclei_diameter,
                vaporization_coefficient,
                condensation_coefficient,
                latent_heat,
                radius_floor,
            ),
            "Schnerr-Sauer cavitation",
        )
        (
            p_sat,
            nuclei,
            diameter,
            vaporization,
            condensation,
            latent,
            floor,
        ) = values
        self.saturation_pressure = p_sat
        self.nuclei_number_density = nuclei
        self.nuclei_diameter = diameter
        self.vaporization_coefficient = vaporization
        self.condensation_coefficient = condensation
        self.latent_heat = latent
        self.radius_floor = floor
        self.model_id = canonical_fingerprint(
            {
                "kind": "schnerr-sauer-cavitation",
                "values": values,
            }
        )

    def raw_rate(
        self,
        pressure,
        temperature,
        alpha0,
        density0,
        density1,
        interface_area_density,
        heat_flux0,
        heat_flux1,
        /,
    ):
        del temperature, interface_area_density, heat_flux0, heat_flux1
        vapor_fraction = 1.0 - alpha0
        nuclei_volume = self.nuclei_number_density * np.pi * self.nuclei_diameter**3 / 6.0
        nuclei_fraction = nuclei_volume / (1.0 + nuclei_volume)
        effective_fraction = jnp.maximum(vapor_fraction, nuclei_fraction)
        radius = (
            3.0
            * effective_fraction
            / (
                4.0
                * np.pi
                * self.nuclei_number_density
                * jnp.maximum(1.0 - effective_fraction, self.radius_floor)
            )
        ) ** (1.0 / 3.0)
        radius = jnp.maximum(radius, self.radius_floor)
        pressure_difference = pressure - self.saturation_pressure
        speed = jnp.sqrt(2.0 * jnp.abs(pressure_difference) / (3.0 * density0))
        mixture_density = alpha0 * density0 + vapor_fraction * density1
        coefficient = (
            3.0
            * density0
            * density1
            / (jnp.maximum(mixture_density, self.radius_floor) * radius)
            * speed
        )
        evaporation = (
            self.vaporization_coefficient
            * coefficient
            * alpha0
            * jnp.maximum(-pressure_difference, 0.0)
            / jnp.maximum(jnp.abs(pressure_difference), 1.0)
        )
        condensation = (
            self.condensation_coefficient
            * coefficient
            * vapor_fraction
            * jnp.maximum(pressure_difference, 0.0)
            / jnp.maximum(jnp.abs(pressure_difference), 1.0)
        )
        return evaporation - condensation, pressure_difference


class InterfaceHeatResistancePhaseChangePlan(AbstractVOFMassTransferPlan):
    saturation_temperature: float = eqx.field(static=True)
    interface_heat_transfer_coefficient: float = eqx.field(static=True)

    def __init__(
        self,
        saturation_temperature,
        interface_heat_transfer_coefficient,
        latent_heat,
        /,
    ):
        saturation, coefficient, latent = _positive_material_parameters(
            (
                saturation_temperature,
                interface_heat_transfer_coefficient,
                latent_heat,
            ),
            "Interface heat-resistance phase change",
        )
        self.saturation_temperature = saturation
        self.interface_heat_transfer_coefficient = coefficient
        self.latent_heat = latent
        self.model_id = canonical_fingerprint(
            {
                "kind": "interface-heat-resistance-phase-change",
                "saturation_temperature": saturation,
                "interface_heat_transfer_coefficient": coefficient,
                "latent_heat": latent,
            }
        )

    def raw_rate(
        self,
        pressure,
        temperature,
        alpha0,
        density0,
        density1,
        interface_area_density,
        heat_flux0,
        heat_flux1,
        /,
    ):
        del pressure, alpha0, density0, density1, heat_flux0, heat_flux1
        difference = temperature - self.saturation_temperature
        return (
            interface_area_density
            * self.interface_heat_transfer_coefficient
            * difference
            / self.latent_heat,
            difference,
        )


class TemperatureRelaxationPhaseChangePlan(AbstractVOFMassTransferPlan):
    saturation_temperature: float = eqx.field(static=True)
    vaporization_time: float = eqx.field(static=True)
    condensation_time: float = eqx.field(static=True)

    def __init__(
        self,
        saturation_temperature,
        vaporization_time,
        condensation_time,
        latent_heat,
        /,
    ):
        saturation, vaporization, condensation, latent = _positive_material_parameters(
            (
                saturation_temperature,
                vaporization_time,
                condensation_time,
                latent_heat,
            ),
            "Temperature-relaxation phase change",
        )
        self.saturation_temperature = saturation
        self.vaporization_time = vaporization
        self.condensation_time = condensation
        self.latent_heat = latent
        self.model_id = canonical_fingerprint(
            {
                "kind": "temperature-relaxation-phase-change",
                "saturation_temperature": saturation,
                "vaporization_time": vaporization,
                "condensation_time": condensation,
                "latent_heat": latent,
            }
        )

    def raw_rate(
        self,
        pressure,
        temperature,
        alpha0,
        density0,
        density1,
        interface_area_density,
        heat_flux0,
        heat_flux1,
        /,
    ):
        del pressure, interface_area_density, heat_flux0, heat_flux1
        difference = temperature - self.saturation_temperature
        evaporation = (
            alpha0
            * density0
            * jnp.maximum(difference, 0.0)
            / (self.saturation_temperature * self.vaporization_time)
        )
        condensation = (
            (1.0 - alpha0)
            * density1
            * jnp.maximum(-difference, 0.0)
            / (self.saturation_temperature * self.condensation_time)
        )
        return evaporation - condensation, difference


class StefanHeatFluxPhaseChangePlan(AbstractVOFMassTransferPlan):
    def __init__(self, latent_heat, /):
        (latent,) = _positive_material_parameters(
            (latent_heat,), "Stefan heat-flux phase change"
        )
        self.latent_heat = latent
        self.model_id = canonical_fingerprint(
            {"kind": "stefan-heat-flux-phase-change", "latent_heat": latent}
        )

    def raw_rate(
        self,
        pressure,
        temperature,
        alpha0,
        density0,
        density1,
        interface_area_density,
        heat_flux0,
        heat_flux1,
        /,
    ):
        del pressure, temperature, alpha0, density0, density1
        heat_jump = heat_flux0 - heat_flux1
        return interface_area_density * heat_jump / self.latent_heat, heat_jump


class VOFPhaseChangeDifferentialSource(StrictModule):
    state_rate: Array
    transfer: ConservativePhaseTransferEvaluation
    alpha_rate: Array
    explicit_step_restriction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class TwoMaterialVOFPhaseChangePlan(StrictModule, NonTrainableState):
    system: TwoMaterialVOFSystem
    rate_law: AbstractVOFMassTransferPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: TwoMaterialVOFSystem,
        rate_law: AbstractVOFMassTransferPlan,
        /,
    ):
        if not isinstance(system, TwoMaterialVOFSystem):
            raise TypeError("system must be TwoMaterialVOFSystem.")
        if not isinstance(rate_law, AbstractVOFMassTransferPlan):
            raise TypeError("rate_law must implement AbstractVOFMassTransferPlan.")
        self.system = system
        self.rate_law = rate_law
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-material-vof-phase-change",
                "system": system.system_id,
                "rate_law": rate_law.model_id,
            }
        )

    def differential_source(
        self,
        state: ArrayLike,
        /,
        *,
        interface_area_density: ArrayLike = 0.0,
        heat_flux0: ArrayLike = 0.0,
        heat_flux1: ArrayLike = 0.0,
    ) -> VOFPhaseChangeDifferentialSource:
        value = self.system._state(state)
        pressure = self.system.pressure(value)
        temperature = self.system.eos.temperature(value)
        density0, density1 = self.system.phase_densities(value)
        alpha0 = value[..., self.system.alpha_index]
        area = jnp.broadcast_to(
            jnp.asarray(interface_area_density, dtype=value.dtype), alpha0.shape
        )
        flux0 = jnp.broadcast_to(jnp.asarray(heat_flux0, dtype=value.dtype), alpha0.shape)
        flux1 = jnp.broadcast_to(jnp.asarray(heat_flux1, dtype=value.dtype), alpha0.shape)
        raw, equilibrium_margin = self.rate_law.raw_rate(
            pressure,
            temperature,
            alpha0,
            density0,
            density1,
            area,
            flux0,
            flux1,
        )
        mixed = (
            (alpha0 > 0.0)
            & (alpha0 < 1.0)
            & (alpha0 >= self.system.eos.alpha_floor)
            & (alpha0 <= 1.0 - self.system.eos.alpha_floor)
        )
        raw = jnp.where(mixed, raw, 0.0)
        safe_density0 = jnp.where(mixed, density0, 1.0)
        safe_density1 = jnp.where(mixed, density1, 1.0)
        inverse_density0 = jnp.where(mixed, 1.0 / safe_density0, 0.0)
        inverse_density1 = jnp.where(mixed, 1.0 / safe_density1, 0.0)
        donor = jnp.where(raw >= 0.0, value[..., 0], value[..., 1])
        phase_rates = jnp.stack((-raw, raw), axis=-1)
        volume_rate = raw * (inverse_density1 - inverse_density0)
        alpha_rate = -raw * inverse_density0 - alpha0 * volume_rate
        donor_step = jnp.where(jnp.abs(raw) > 0.0, donor / jnp.abs(raw), jnp.inf)
        alpha_step = jnp.where(
            alpha_rate > 0.0,
            (1.0 - alpha0) / alpha_rate,
            jnp.where(alpha_rate < 0.0, alpha0 / -alpha_rate, jnp.inf),
        )
        restriction = jnp.minimum(donor_step, alpha_step)
        state_rate = jnp.zeros_like(value)
        state_rate = state_rate.at[..., 0].set(-raw)
        state_rate = state_rate.at[..., 1].set(raw)
        state_rate = state_rate.at[..., self.system.alpha_index].set(alpha_rate)
        mass_defect = jnp.sum(phase_rates, axis=-1)
        finite = (
            jnp.all(jnp.isfinite(state_rate), axis=-1)
            & jnp.isfinite(volume_rate)
            & jnp.isfinite(equilibrium_margin)
            & ~jnp.isnan(restriction)
        )
        successful = finite & self.system.admissible(value)
        transfer = ConservativePhaseTransferEvaluation(
            raw,
            raw,
            phase_rates,
            volume_rate,
            raw * self.rate_law.latent_heat,
            jnp.ones_like(raw),
            donor_step,
            donor,
            equilibrium_margin,
            mass_defect,
            jnp.zeros_like(raw),
            finite,
            successful,
            self.rate_law.model_id,
        )
        return VOFPhaseChangeDifferentialSource(
            state_rate,
            transfer,
            alpha_rate,
            restriction,
            finite,
            successful,
            self.plan_id,
        )

    def evaluate(
        self,
        state: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        interface_area_density: ArrayLike = 0.0,
        heat_flux0: ArrayLike = 0.0,
        heat_flux1: ArrayLike = 0.0,
    ) -> ConservativePhaseTransferEvaluation:
        value = self.system._state(state)
        step = jnp.asarray(step_size, dtype=value.dtype)
        pressure = self.system.pressure(value)
        temperature = self.system.eos.temperature(value)
        density0, density1 = self.system.phase_densities(value)
        alpha0 = value[..., self.system.alpha_index]
        area = jnp.broadcast_to(
            jnp.asarray(interface_area_density, dtype=value.dtype), alpha0.shape
        )
        flux0 = jnp.broadcast_to(jnp.asarray(heat_flux0, dtype=value.dtype), alpha0.shape)
        flux1 = jnp.broadcast_to(jnp.asarray(heat_flux1, dtype=value.dtype), alpha0.shape)
        raw, equilibrium_margin = self.rate_law.raw_rate(
            pressure,
            temperature,
            alpha0,
            density0,
            density1,
            area,
            flux0,
            flux1,
        )
        mixed = (
            (alpha0 > 0.0)
            & (alpha0 < 1.0)
            & (alpha0 >= self.system.eos.alpha_floor)
            & (alpha0 <= 1.0 - self.system.eos.alpha_floor)
        )
        raw = jnp.where(mixed, raw, 0.0)
        safe_density0 = jnp.where(mixed, density0, 1.0)
        safe_density1 = jnp.where(mixed, density1, 1.0)
        inverse_density0 = jnp.where(mixed, 1.0 / safe_density0, 0.0)
        inverse_density1 = jnp.where(mixed, 1.0 / safe_density1, 0.0)
        donor = jnp.where(raw >= 0.0, value[..., 0], value[..., 1])
        requested = jnp.abs(raw) * step
        factor = jnp.minimum(
            1.0,
            donor / jnp.where(requested > 0.0, requested, 1.0),
        )
        factor = jnp.where((step > 0.0) & (donor >= 0.0), factor, 0.0)
        limited = factor * raw
        phase_rates = jnp.stack((-limited, limited), axis=-1)
        volume_rate = limited * (inverse_density1 - inverse_density0)
        latent_power = limited * self.rate_law.latent_heat
        restriction = jnp.where(jnp.abs(raw) > 0.0, donor / jnp.abs(raw), jnp.inf)
        donor_margin = donor - jnp.abs(limited) * step
        mass_defect = jnp.sum(phase_rates, axis=-1)
        energy_defect = jnp.zeros_like(limited)
        finite = (
            jnp.isfinite(raw)
            & jnp.isfinite(limited)
            & jnp.isfinite(volume_rate)
            & jnp.isfinite(latent_power)
            & jnp.isfinite(factor)
            & ~jnp.isnan(restriction)
            & jnp.isfinite(donor_margin)
            & jnp.isfinite(equilibrium_margin)
        )
        successful = (
            finite
            & (step > 0.0)
            & self.system.admissible(value)
            & (
                donor_margin
                >= -64.0 * jnp.finfo(value.dtype).eps * jnp.maximum(donor, 1.0)
            )
        )
        return ConservativePhaseTransferEvaluation(
            raw,
            limited,
            phase_rates,
            volume_rate,
            latent_power,
            factor,
            restriction,
            donor_margin,
            equilibrium_margin,
            mass_defect,
            energy_defect,
            finite,
            successful,
            self.rate_law.model_id,
        )

    def step(
        self,
        state: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        interface_area_density: ArrayLike = 0.0,
        heat_flux0: ArrayLike = 0.0,
        heat_flux1: ArrayLike = 0.0,
    ) -> VOFPhaseChangeStepResult:
        value = self.system._state(state)
        step = jnp.asarray(step_size, dtype=value.dtype)
        transfer = self.evaluate(
            value,
            step,
            interface_area_density=interface_area_density,
            heat_flux0=heat_flux0,
            heat_flux1=heat_flux1,
        )
        density0, density1 = self.system.phase_densities(value)
        delta = step * transfer.limited_mass_rate
        mass0 = value[..., 0] - delta
        mass1 = value[..., 1] + delta
        safe_density0 = jnp.where(density0 > 0.0, density0, 1.0)
        safe_density1 = jnp.where(density1 > 0.0, density1, 1.0)
        volume0 = jnp.where(mass0 > 0.0, mass0 / safe_density0, 0.0)
        volume1 = jnp.where(mass1 > 0.0, mass1 / safe_density1, 0.0)
        total_volume = volume0 + volume1
        alpha0 = volume0 / jnp.where(total_volume > 0.0, total_volume, 1.0)
        candidate = value.at[..., 0].set(mass0)
        candidate = candidate.at[..., 1].set(mass1)
        candidate = candidate.at[..., self.system.alpha_index].set(alpha0)
        pressure_before = self.system.pressure(value)
        pressure_after = self.system.pressure(candidate)
        admissible = self.system.admissible(candidate)
        finite = (
            transfer.finite
            & jnp.all(jnp.isfinite(candidate), axis=-1)
            & jnp.isfinite(pressure_after)
            & (total_volume > 0.0)
        )
        accepted = finite & transfer.successful & admissible
        accepted_state = jnp.where(accepted[..., None], candidate, value)
        previous_alpha = value[..., self.system.alpha_index]
        alpha_rate = (alpha0 - previous_alpha) / jnp.where(step > 0.0, step, 1.0)
        return VOFPhaseChangeStepResult(
            value,
            accepted_state,
            transfer,
            alpha_rate,
            pressure_before,
            pressure_after,
            admissible,
            finite,
            accepted,
            self.plan_id,
        )


__all__ = [
    "AbstractVOFMassTransferPlan",
    "ConservativePhaseTransferEvaluation",
    "InterfaceHeatResistancePhaseChangePlan",
    "KunzCavitationPlan",
    "MerkleCavitationPlan",
    "SchnerrSauerCavitationPlan",
    "StefanHeatFluxPhaseChangePlan",
    "TemperatureRelaxationPhaseChangePlan",
    "TwoMaterialVOFPhaseChangePlan",
    "VOFPhaseChangeDifferentialSource",
    "VOFPhaseChangeStepResult",
]
