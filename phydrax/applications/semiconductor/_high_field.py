#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Local, isotropic high-field laws with explicit SI and validity domains.

These closures do not establish velocity overshoot, momentum transport, or an
empirical material qualification. All coefficients and their provenance are
required. Runtime ``successful`` masks are constitutive-domain evidence, not
calibration evidence; inadmissible evaluations contain NaNs, not extrapolations.
"""

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...units import derived_unit, JOULE, KELVIN, METER, SECOND, VOLT
from ._quantities import _positive_scalar, _si, _text, ELEMENTARY_CHARGE_SI


VELOCITY_UNIT = derived_unit("m/s", ((METER, 1), (SECOND, -1)))
FIELD_UNIT = derived_unit("V/m", ((VOLT, 1), (METER, -1)))
PER_METER = derived_unit("1/m", ((METER, -1),))


def _finite_scalar(value, name):
    host = np.asarray(value)
    if host.shape != () or not np.isfinite(host):
        raise ValueError(f"{name} must be one finite scalar.")
    return jnp.asarray(value)


def _temperature_bounds(bounds, unit=KELVIN):
    values = _si(bounds, unit, KELVIN)
    host = np.asarray(values)
    if host.shape != (2,) or not np.all(np.isfinite(host)) or not 0 < host[0] < host[1]:
        raise ValueError(
            "temperature_range must contain two increasing positive Kelvin values."
        )
    return values


def _temperature_valid(temperature, bounds):
    return (
        jnp.isfinite(temperature)
        & (temperature >= bounds[0])
        & (temperature <= bounds[1])
    )


def _admitted(value, valid):
    return jnp.where(valid, value, jnp.nan)


class HighFieldDrivingForce(StrEnum):
    """Scalar, oriented force divided by positive elementary charge, in V/m."""

    ELECTRIC_FIELD = "electric_field"
    QUASI_FERMI_GRADIENT = "quasi_fermi_gradient"

    def from_fields(self, electric_field, quasi_fermi_energy_gradient):
        """Select E (V/m) or grad(EF)/q (input J/m), without changing signs.

        The scalar is the component along the face/path, not an inferred norm
        of a multidimensional field. A built-in electric field and a carrier
        electrochemical gradient are intentionally different driving choices.
        """
        if self is HighFieldDrivingForce.ELECTRIC_FIELD:
            return jnp.asarray(electric_field)
        return jnp.asarray(quasi_fermi_energy_gradient) / ELEMENTARY_CHARGE_SI


class SaturationEvaluation(StrictModule):
    mobility: Array
    reduction: Array
    drift_speed: Array
    saturation_velocity: Array
    successful: Array


class LocalVelocitySaturation(StrictModule):
    """Caughey–Thomas local saturation, mu=mu0/[1+(mu0|F|/vs)^b]^(1/b).

    ``vs(T)=vs_ref*(T/T_ref)**temperature_exponent``. Orientation is a declared
    material/crystal calibration direction; this is not an anisotropic tensor
    law. ``force`` passed to evaluate is already the selected scalar in V/m.
    Applying the positive mobility reduction to ONE conservative face-number
    flux preserves its equilibrium zero. ``drift_speed`` is nonnegative and
    does not infer the carrier charge sign.
    """

    reference_velocity: Array
    exponent: Array
    reference_temperature: Array
    temperature_exponent: Array
    maximum_force: Array
    temperature_range: Array
    driving_force: HighFieldDrivingForce = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        reference_velocity,
        exponent,
        /,
        *,
        reference_temperature,
        temperature_exponent,
        maximum_force,
        temperature_range,
        driving_force,
        orientation,
        provenance,
        velocity_unit=VELOCITY_UNIT,
        field_unit=FIELD_UNIT,
        temperature_unit=KELVIN,
    ):
        self.reference_velocity = _positive_scalar(
            reference_velocity, velocity_unit, VELOCITY_UNIT, "saturation velocity"
        )
        self.exponent = _finite_scalar(exponent, "saturation exponent")
        if float(self.exponent) < 1:
            raise ValueError("saturation exponent must be at least one.")
        self.reference_temperature = _positive_scalar(
            reference_temperature, temperature_unit, KELVIN, "reference temperature"
        )
        self.temperature_exponent = _finite_scalar(
            temperature_exponent, "temperature exponent"
        )
        self.maximum_force = _positive_scalar(
            maximum_force, field_unit, FIELD_UNIT, "maximum force"
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        if not bool(
            _temperature_valid(self.reference_temperature, self.temperature_range)
        ):
            raise ValueError("reference temperature must lie in temperature_range.")
        self.driving_force = HighFieldDrivingForce(driving_force)
        self.orientation = _text(orientation, "calibration orientation")
        self.provenance = _text(provenance, "saturation provenance")

    def evaluate(self, low_field_mobility, force, temperature):
        """Return SI mobility m²/(V s), speeds m/s and an elementwise mask."""
        mu, force, temperature = jnp.broadcast_arrays(
            *map(jnp.asarray, (low_field_mobility, force, temperature))
        )
        speed = (
            self.reference_velocity
            * (temperature / self.reference_temperature) ** self.temperature_exponent
        )
        ratio = mu * jnp.abs(force) / speed
        # logaddexp avoids overflow in the high-field power law.
        positive_ratio = jnp.where(ratio > 0, ratio, 1.0)
        log_denominator = (
            jnp.logaddexp(0.0, self.exponent * jnp.log(positive_ratio)) / self.exponent
        )
        reduction = jnp.where(ratio == 0, 1.0, jnp.exp(-log_denominator))
        mobility = mu * reduction
        valid = (
            _temperature_valid(temperature, self.temperature_range)
            & jnp.isfinite(mu)
            & (mu > 0)
            & jnp.isfinite(force)
            & (jnp.abs(force) <= self.maximum_force)
            & jnp.isfinite(speed)
            & (speed > 0)
            & jnp.isfinite(ratio)
        )
        return SaturationEvaluation(
            _admitted(mobility, valid),
            _admitted(reduction, valid),
            _admitted(mobility * jnp.abs(force), valid),
            _admitted(speed, valid),
            valid,
        )


class ImpactIonizationEvaluation(StrictModule):
    """Volume rates: number m^-3 s^-1, charge A/m³, energy W/m³.

    Carrier energy sources are KINETIC; ``band_energy_source`` is additional
    electronic band storage. Their sum with lattice heat is exactly zero.
    Initiating carriers pay the pair's band gap plus explicit birth kinetic
    energies. Do not add that ionization loss a second time in lattice heat.
    """

    electron_source: Array
    hole_source: Array
    electron_charge_source: Array
    hole_charge_source: Array
    electron_energy_source: Array
    hole_energy_source: Array
    band_energy_source: Array
    lattice_energy_source: Array
    electron_initiated_rate: Array
    hole_initiated_rate: Array
    successful: Array


class LocalImpactIonization(StrictModule):
    """Local Chynoweth pair creation driven by physical electric field.

    alpha_s=A_s*(T/Tref)^a_s*exp[-(B_s/|E|)^b_s], in m^-1.
    Pair rate is alpha_n*|Gamma_n|+alpha_p*|Gamma_p| for number-flux
    DENSITIES in m^-2 s^-1 along the admitted direction. This is not a
    nonlocal avalanche or hot-carrier distribution closure. Birth energies
    (J above each band edge) and initiator energy withdrawal are explicit.
    """

    prefactors: Array
    critical_fields: Array
    exponents: Array
    temperature_exponents: Array
    birth_energies: Array
    reference_temperature: Array
    temperature_range: Array
    maximum_field: Array
    orientation: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        electron_prefactor,
        hole_prefactor,
        electron_critical_field,
        hole_critical_field,
        /,
        *,
        electron_exponent,
        hole_exponent,
        electron_temperature_exponent,
        hole_temperature_exponent,
        electron_birth_energy,
        hole_birth_energy,
        reference_temperature,
        temperature_range,
        maximum_field,
        orientation,
        provenance,
        inverse_length_unit=PER_METER,
        field_unit=FIELD_UNIT,
        energy_unit=JOULE,
        temperature_unit=KELVIN,
    ):
        self.prefactors = jnp.stack(
            tuple(
                _positive_scalar(
                    v,
                    inverse_length_unit,
                    PER_METER,
                    "ionization prefactor",
                    nonnegative=True,
                )
                for v in (electron_prefactor, hole_prefactor)
            )
        )
        self.critical_fields = jnp.stack(
            tuple(
                _positive_scalar(v, field_unit, FIELD_UNIT, "critical field")
                for v in (electron_critical_field, hole_critical_field)
            )
        )
        self.exponents = jnp.stack(
            tuple(
                _finite_scalar(v, "ionization exponent")
                for v in (electron_exponent, hole_exponent)
            )
        )
        if np.any(np.asarray(self.exponents) <= 0):
            raise ValueError("ionization exponents must be positive.")
        self.temperature_exponents = jnp.stack(
            tuple(
                _finite_scalar(v, "temperature exponent")
                for v in (electron_temperature_exponent, hole_temperature_exponent)
            )
        )
        self.birth_energies = jnp.stack(
            tuple(
                _positive_scalar(
                    v, energy_unit, JOULE, "birth kinetic energy", nonnegative=True
                )
                for v in (electron_birth_energy, hole_birth_energy)
            )
        )
        self.reference_temperature = _positive_scalar(
            reference_temperature, temperature_unit, KELVIN, "reference temperature"
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        if not bool(
            _temperature_valid(self.reference_temperature, self.temperature_range)
        ):
            raise ValueError("reference temperature must lie in temperature_range.")
        self.maximum_field = _positive_scalar(
            maximum_field, field_unit, FIELD_UNIT, "maximum ionization field"
        )
        self.orientation = _text(orientation, "ionization orientation")
        self.provenance = _text(provenance, "ionization provenance")

    def evaluate(
        self,
        electric_field,
        electron_number_flux_density,
        hole_number_flux_density,
        temperature,
        band_gap,
    ):
        """Evaluate scalar/array SI fields, oriented flux densities, T(K), gap(J)."""
        field, gn, gp, temperature, gap = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (
                    electric_field,
                    electron_number_flux_density,
                    hole_number_flux_density,
                    temperature,
                    band_gap,
                ),
            )
        )
        magnitude = jnp.abs(field)[..., None]
        safe_field = jnp.where(magnitude > 0, magnitude, 1.0)
        alpha = (
            self.prefactors
            * (temperature[..., None] / self.reference_temperature)
            ** self.temperature_exponents
            * jnp.exp(-((self.critical_fields / safe_field) ** self.exponents))
        )
        alpha = jnp.where(magnitude == 0, 0.0, alpha)
        rn, rp = alpha[..., 0] * jnp.abs(gn), alpha[..., 1] * jnp.abs(gp)
        generation = rn + rp
        threshold = gap + jnp.sum(self.birth_energies)
        electron_energy = self.birth_energies[0] * generation - threshold * rn
        hole_energy = self.birth_energies[1] * generation - threshold * rp
        valid = (
            _temperature_valid(temperature, self.temperature_range)
            & jnp.isfinite(field)
            & (jnp.abs(field) <= self.maximum_field)
            & jnp.isfinite(gn)
            & jnp.isfinite(gp)
            & jnp.isfinite(gap)
            & (gap > 0)
            & jnp.isfinite(generation)
            & jnp.isfinite(electron_energy)
            & jnp.isfinite(hole_energy)
            & jnp.isfinite(gap * generation)
        )
        return ImpactIonizationEvaluation(
            _admitted(generation, valid),
            _admitted(generation, valid),
            _admitted(-ELEMENTARY_CHARGE_SI * generation, valid),
            _admitted(ELEMENTARY_CHARGE_SI * generation, valid),
            _admitted(electron_energy, valid),
            _admitted(hole_energy, valid),
            _admitted(gap * generation, valid),
            _admitted(jnp.zeros_like(generation), valid),
            _admitted(rn, valid),
            _admitted(rp, valid),
            valid,
        )


__all__ = [
    "HighFieldDrivingForce",
    "ImpactIonizationEvaluation",
    "LocalImpactIonization",
    "LocalVelocitySaturation",
    "SaturationEvaluation",
]
