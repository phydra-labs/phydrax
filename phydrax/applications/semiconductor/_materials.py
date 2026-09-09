#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit constant-band, nondegenerate materials and charge-paired SRH."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...units import ELECTRONVOLT, JOULE, KELVIN, SECOND, UnitDefinition
from ._high_field import LocalVelocitySaturation
from ._quantities import (
    _positive_scalar,
    _si,
    _text,
    BOLTZMANN_CONSTANT_SI,
    MOBILITY_UNIT,
    PER_CUBIC_METER,
    PERMITTIVITY_UNIT,
    VACUUM_PERMITTIVITY_SI,
)
from ._thermal import (
    CarrierEnergyRelaxation,
    CarrierEnergyTransport,
    ConstantLatticeHeatCapacity,
    THERMAL_CONDUCTIVITY_UNIT,
)
from ._thermodynamics import BandThermodynamics, IncompleteIonization


class ConstantMobility(StrictModule):
    """Low-field mobility independent of doping; SI m2/(V s)."""

    value: Array

    def __init__(self, value: ArrayLike, /, *, unit: UnitDefinition = MOBILITY_UNIT):
        self.value = _positive_scalar(value, unit, MOBILITY_UNIT, "mobility")

    def __call__(self, total_ionized_density: ArrayLike, /) -> Array:
        return jnp.broadcast_to(self.value, jnp.shape(total_ionized_density))


class DopingDependentMobility(StrictModule):
    """Caughey–Thomas low-field mobility in total ionized dopant density.

    mu = minimum + (maximum-minimum)/(1+(N/reference_density)**exponent).
    Parameters are explicit and user-calibrated; no high-field, degeneracy,
    temperature, surface-scattering, or alloy correction is silently applied.
    """

    minimum: Array
    maximum: Array
    reference_density: Array
    exponent: Array
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        minimum,
        maximum,
        reference_density,
        exponent,
        /,
        *,
        mobility_unit: UnitDefinition = MOBILITY_UNIT,
        density_unit: UnitDefinition = PER_CUBIC_METER,
        provenance: str,
    ):
        self.minimum = _positive_scalar(
            minimum, mobility_unit, MOBILITY_UNIT, "minimum mobility"
        )
        self.maximum = _positive_scalar(
            maximum, mobility_unit, MOBILITY_UNIT, "maximum mobility"
        )
        self.reference_density = _positive_scalar(
            reference_density, density_unit, PER_CUBIC_METER, "reference density"
        )
        exponent_ = np.asarray(exponent)
        if exponent_.shape != () or not np.isfinite(exponent_) or exponent_ <= 0:
            raise ValueError("Mobility exponent must be a finite positive scalar.")
        if float(self.minimum) > float(self.maximum):
            raise ValueError("Minimum mobility must not exceed maximum mobility.")
        self.exponent = jnp.asarray(exponent)
        self.provenance = _text(provenance, "mobility provenance")

    def __call__(self, total_ionized_density: ArrayLike, /) -> Array:
        ratio = jnp.asarray(total_ionized_density) / self.reference_density
        return self.minimum + (self.maximum - self.minimum) / (1 + ratio**self.exponent)


class SemiconductorMaterial(StrictModule):
    """Low-field transport and recombination with an explicit thermodynamic owner.

    Supply ``thermodynamics`` for aligned DOS-based MB/FD bands. The alternative
    ``intrinsic_density`` chart is restricted to homogeneous nondegenerate bands;
    it does not infer either density of states or an absolute Fermi reference.
    """

    permittivity: Array
    intrinsic_density: Array
    electron_mobility: ConstantMobility | DopingDependentMobility
    hole_mobility: ConstantMobility | DopingDependentMobility
    electron_lifetime: Array
    hole_lifetime: Array
    band_gap: Array | None
    electron_affinity: Array | None
    reference_temperature: Array
    thermodynamics: BandThermodynamics | None
    incomplete_ionization: IncompleteIonization | None
    electron_saturation: LocalVelocitySaturation | None
    hole_saturation: LocalVelocitySaturation | None
    lattice_heat_capacity: ConstantLatticeHeatCapacity | None
    lattice_thermal_conductivity: Array
    electron_energy_transport: CarrierEnergyTransport | None
    hole_energy_transport: CarrierEnergyTransport | None
    electron_energy_relaxation: CarrierEnergyRelaxation | None
    hole_energy_relaxation: CarrierEnergyRelaxation | None
    name: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    temperature_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        permittivity,
        intrinsic_density=None,
        electron_mobility,
        hole_mobility,
        electron_lifetime=1e-6,
        hole_lifetime=1e-6,
        band_gap=None,
        electron_affinity=None,
        reference_temperature=300.0,
        temperature_range=(300.0, 300.0),
        provenance: str,
        thermodynamics: BandThermodynamics | None = None,
        incomplete_ionization: IncompleteIonization | None = None,
        electron_saturation: LocalVelocitySaturation | None = None,
        hole_saturation: LocalVelocitySaturation | None = None,
        lattice_heat_capacity: ConstantLatticeHeatCapacity | None = None,
        lattice_thermal_conductivity=0.0,
        electron_energy_transport: CarrierEnergyTransport | None = None,
        hole_energy_transport: CarrierEnergyTransport | None = None,
        electron_energy_relaxation: CarrierEnergyRelaxation | None = None,
        hole_energy_relaxation: CarrierEnergyRelaxation | None = None,
        permittivity_unit: UnitDefinition = PERMITTIVITY_UNIT,
        density_unit: UnitDefinition = PER_CUBIC_METER,
        mobility_unit: UnitDefinition = MOBILITY_UNIT,
        lifetime_unit: UnitDefinition = SECOND,
        energy_unit: UnitDefinition = ELECTRONVOLT,
        temperature_unit: UnitDefinition = KELVIN,
        thermal_conductivity_unit: UnitDefinition = THERMAL_CONDUCTIVITY_UNIT,
    ):
        self.name = _text(name, "material name")
        self.provenance = _text(provenance, "material provenance")
        self.permittivity = _positive_scalar(
            permittivity, permittivity_unit, PERMITTIVITY_UNIT, "permittivity"
        )
        if (intrinsic_density is None) == (thermodynamics is None):
            raise ValueError(
                "Specify exactly one of intrinsic_density and thermodynamics."
            )
        if thermodynamics is not None and not isinstance(
            thermodynamics, BandThermodynamics
        ):
            raise TypeError("thermodynamics must be BandThermodynamics.")
        self.thermodynamics = thermodynamics
        self.incomplete_ionization = incomplete_ionization
        if incomplete_ionization is not None:
            incomplete_ionization.admit(thermodynamics)
        for law, label in (
            (electron_saturation, "electron_saturation"),
            (hole_saturation, "hole_saturation"),
        ):
            if law is not None and not isinstance(law, LocalVelocitySaturation):
                raise TypeError(f"{label} must be LocalVelocitySaturation or None.")
        if lattice_heat_capacity is not None and not isinstance(
            lattice_heat_capacity, ConstantLatticeHeatCapacity
        ):
            raise TypeError(
                "lattice_heat_capacity must be ConstantLatticeHeatCapacity or None."
            )
        for law, kind, label in (
            (
                electron_energy_transport,
                CarrierEnergyTransport,
                "electron_energy_transport",
            ),
            (hole_energy_transport, CarrierEnergyTransport, "hole_energy_transport"),
            (
                electron_energy_relaxation,
                CarrierEnergyRelaxation,
                "electron_energy_relaxation",
            ),
            (hole_energy_relaxation, CarrierEnergyRelaxation, "hole_energy_relaxation"),
        ):
            if law is not None and not isinstance(law, kind):
                raise TypeError(f"{label} must be {kind.__name__} or None.")
        for carrier, transport, relaxation in (
            ("electron", electron_energy_transport, electron_energy_relaxation),
            ("hole", hole_energy_transport, hole_energy_relaxation),
        ):
            for law in (transport, relaxation):
                if law is None:
                    continue
                if (
                    thermodynamics is None
                    or law.thermodynamics is not thermodynamics
                    or law.carrier != carrier
                ):
                    raise ValueError(
                        f"{carrier} energy laws must share this material's "
                        "thermodynamics and carrier identity."
                    )
        self.electron_saturation, self.hole_saturation = (
            electron_saturation,
            hole_saturation,
        )
        self.lattice_heat_capacity = lattice_heat_capacity
        self.lattice_thermal_conductivity = _si(
            lattice_thermal_conductivity,
            thermal_conductivity_unit,
            THERMAL_CONDUCTIVITY_UNIT,
        )
        if (
            self.lattice_thermal_conductivity.shape != ()
            or not np.isfinite(float(self.lattice_thermal_conductivity))
            or float(self.lattice_thermal_conductivity) < 0
        ):
            raise ValueError("Lattice conductivity must be finite and nonnegative.")
        self.electron_energy_transport = electron_energy_transport
        self.hole_energy_transport = hole_energy_transport
        self.electron_energy_relaxation = electron_energy_relaxation
        self.hole_energy_relaxation = hole_energy_relaxation
        if thermodynamics is None:
            self.intrinsic_density = _positive_scalar(
                intrinsic_density, density_unit, PER_CUBIC_METER, "intrinsic density"
            )
            self.band_gap = _positive_scalar(
                1.12 if band_gap is None else band_gap,
                energy_unit,
                JOULE,
                "band gap",
            )
            self.electron_affinity = _positive_scalar(
                4.05 if electron_affinity is None else electron_affinity,
                energy_unit,
                JOULE,
                "electron affinity",
                nonnegative=True,
            )
        else:
            if band_gap is not None or electron_affinity is not None:
                raise ValueError(
                    "Explicit BandThermodynamics owns its band gap and alignment; "
                    "do not also supply legacy band_gap/electron_affinity."
                )
            self.band_gap = None
            self.electron_affinity = None
            reference_temperature = thermodynamics.reference_temperature
            temperature_unit = KELVIN
            temperature_range = thermodynamics.temperature_range
            neutral = thermodynamics.equilibrium_fermi_energy(0.0, reference_temperature)
            self.intrinsic_density = thermodynamics.electron_density(
                0.0, neutral, reference_temperature
            )
        self.electron_mobility = (
            electron_mobility
            if isinstance(electron_mobility, (ConstantMobility, DopingDependentMobility))
            else ConstantMobility(electron_mobility, unit=mobility_unit)
        )
        self.hole_mobility = (
            hole_mobility
            if isinstance(hole_mobility, (ConstantMobility, DopingDependentMobility))
            else ConstantMobility(hole_mobility, unit=mobility_unit)
        )
        self.electron_lifetime = _positive_scalar(
            electron_lifetime, lifetime_unit, SECOND, "electron lifetime"
        )
        self.hole_lifetime = _positive_scalar(
            hole_lifetime, lifetime_unit, SECOND, "hole lifetime"
        )
        self.reference_temperature = _positive_scalar(
            reference_temperature, temperature_unit, KELVIN, "reference temperature"
        )
        bounds = np.asarray(_si(temperature_range, temperature_unit, KELVIN))
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] <= 0
            or bounds[1] < bounds[0]
            or not bounds[0] <= float(self.reference_temperature) <= bounds[1]
        ):
            raise ValueError(
                "Temperature validity bounds must include the material reference temperature."
            )
        self.temperature_range = (float(bounds[0]), float(bounds[1]))

    def intrinsic_density_at(self, temperature: ArrayLike, /) -> Array:
        if self.thermodynamics is not None:
            ef = self.thermodynamics.equilibrium_fermi_energy(0.0, temperature)
            return self.thermodynamics.electron_density(0.0, ef, temperature)
        temperature_ = jnp.asarray(temperature)
        exponent = (
            -self.band_gap
            / (2 * BOLTZMANN_CONSTANT_SI)
            * (1 / temperature_ - 1 / self.reference_temperature)
        )
        return (
            self.intrinsic_density
            * (temperature_ / self.reference_temperature) ** 1.5
            * jnp.exp(exponent)
        )

    @classmethod
    def silicon(cls) -> SemiconductorMaterial:
        """Illustrative 300 K silicon homojunction constants, not a process PDK.

        ni=1e16 m^-3, epsilon_r=11.7, low-field mobilities 0.135/0.048
        m2/(V s), symmetric 1 us midgap lifetimes, Eg=1.12 eV, chi=4.05 eV.
        Lifetimes are explicit demonstration assumptions, not silicon constants.
        """
        return cls(
            "silicon-300K",
            permittivity=11.7 * VACUUM_PERMITTIVITY_SI,
            intrinsic_density=1e16,
            electron_mobility=0.135,
            hole_mobility=0.048,
            provenance=(
                "Illustrative constant-band silicon at 300 K; 1 us lifetimes are "
                "demonstration assumptions, not a process calibration."
            ),
        )


class DielectricMaterial(StrictModule):
    """Linear isotropic dielectric with optional declared lattice heat storage."""

    permittivity: Array
    lattice_heat_capacity: ConstantLatticeHeatCapacity | None
    lattice_thermal_conductivity: Array
    name: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        permittivity,
        provenance: str,
        lattice_heat_capacity: ConstantLatticeHeatCapacity | None = None,
        lattice_thermal_conductivity=0.0,
        permittivity_unit: UnitDefinition = PERMITTIVITY_UNIT,
        thermal_conductivity_unit: UnitDefinition = THERMAL_CONDUCTIVITY_UNIT,
    ):
        self.name = _text(name, "material name")
        self.provenance = _text(provenance, "material provenance")
        self.permittivity = _positive_scalar(
            permittivity, permittivity_unit, PERMITTIVITY_UNIT, "permittivity"
        )
        if lattice_heat_capacity is not None and not isinstance(
            lattice_heat_capacity, ConstantLatticeHeatCapacity
        ):
            raise TypeError(
                "lattice_heat_capacity must be ConstantLatticeHeatCapacity or None."
            )
        self.lattice_heat_capacity = lattice_heat_capacity
        self.lattice_thermal_conductivity = _si(
            lattice_thermal_conductivity,
            thermal_conductivity_unit,
            THERMAL_CONDUCTIVITY_UNIT,
        )
        if (
            self.lattice_thermal_conductivity.shape != ()
            or not np.isfinite(float(self.lattice_thermal_conductivity))
            or float(self.lattice_thermal_conductivity) < 0
        ):
            raise ValueError("Lattice conductivity must be finite and nonnegative.")


def srh_recombination(
    electron_density,
    hole_density,
    intrinsic_density,
    electron_lifetime,
    hole_lifetime,
    *,
    log_mass_action=None,
):
    """Midgap SRH pair rate in m^-3 s^-1, positive for recombination.

    Insert this *same* rate in both number continuity equations. Multiplication
    by -q/+q therefore cancels exactly in the charge continuity equation.
    If supplied, ``log_mass_action`` is log(n*p/ni**2), equivalently u1-u2.
    It evaluates the numerator with expm1, preserving exact common-Fermi
    equilibrium and accuracy for small departures from mass action. The density
    interface also admits n=0 or p=0 without evaluating their logarithms.
    """
    n, p, ni = (
        jnp.asarray(electron_density),
        jnp.asarray(hole_density),
        jnp.asarray(intrinsic_density),
    )
    numerator = (
        n * p - ni * ni
        if log_mass_action is None
        else ni * ni * jnp.expm1(log_mass_action)
    )
    return numerator / (hole_lifetime * (n + ni) + electron_lifetime * (p + ni))


__all__ = [
    "ConstantMobility",
    "DopingDependentMobility",
    "SemiconductorMaterial",
    "DielectricMaterial",
    "srh_recombination",
]
