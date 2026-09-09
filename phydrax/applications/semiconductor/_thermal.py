#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""SI lattice and reduced carrier-energy constitutive laws, not a thermal solver.

Extensive thermal ports reuse native thermofluids components. Carrier kinetic
energy uses the shared parabolic-band thermodynamics. Face energy bookkeeping
uses the SAME number flux as carrier continuity and includes band-force work;
adding another copy of J·E to these kinetic sources double counts that work.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...units import derived_unit, JOULE, KELVIN, METER, SECOND
from ..thermofluids import (
    temperature_boundary_component,
    thermal_capacitance_component,
    thermal_conductor_component,
)
from ._high_field import _admitted, _temperature_bounds, _temperature_valid
from ._quantities import _positive_scalar, _text
from ._thermodynamics import BandThermodynamics


VOLUMETRIC_HEAT_CAPACITY_UNIT = derived_unit(
    "J/(m3*K)", ((JOULE, 1), (METER, -3), (KELVIN, -1))
)
THERMAL_CONDUCTANCE_UNIT = derived_unit("W/K", ((JOULE, 1), (SECOND, -1), (KELVIN, -1)))
THERMAL_CONDUCTIVITY_UNIT = derived_unit(
    "W/(m*K)", ((JOULE, 1), (SECOND, -1), (METER, -1), (KELVIN, -1))
)
CUBIC_METER = derived_unit("m3", ((METER, 3),))


def _carrier_name(carrier):
    if carrier not in ("electron", "hole"):
        raise ValueError("carrier must be 'electron' or 'hole'.")
    return carrier


def _kinetic_energy(thermodynamics, carrier, density, temperature):
    if carrier == "electron":
        return thermodynamics.electron_energy_density(density, temperature)
    return thermodynamics.hole_energy_density(density, temperature)


class LatticeEnergyEvaluation(StrictModule):
    internal_energy_density: Array
    heat_capacity: Array
    successful: Array


class ConstantLatticeHeatCapacity(StrictModule):
    """Rigid-lattice sensible u=cv*(T-Tref), J/m³, with constant positive cv.

    The zero at Tref is an explicit energy datum, not a positivity constraint
    on internal energy. No phase change, thermal expansion, or electronic
    heat capacity is included. The specified temperature interval bounds the
    constant-property approximation. Integrators must store volume*u, not T.
    """

    volumetric_heat_capacity: Array
    reference_temperature: Array
    temperature_range: Array
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        volumetric_heat_capacity,
        /,
        *,
        reference_temperature,
        temperature_range,
        provenance,
        capacity_unit=VOLUMETRIC_HEAT_CAPACITY_UNIT,
        temperature_unit=KELVIN,
    ):
        self.volumetric_heat_capacity = _positive_scalar(
            volumetric_heat_capacity,
            capacity_unit,
            VOLUMETRIC_HEAT_CAPACITY_UNIT,
            "lattice heat capacity",
        )
        self.reference_temperature = _positive_scalar(
            reference_temperature,
            temperature_unit,
            KELVIN,
            "energy reference temperature",
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        if not bool(
            _temperature_valid(self.reference_temperature, self.temperature_range)
        ):
            raise ValueError("reference temperature must lie in temperature_range.")
        self.provenance = _text(provenance, "lattice heat capacity provenance")

    def evaluate(self, temperature):
        temperature = jnp.asarray(temperature)
        valid = _temperature_valid(temperature, self.temperature_range)
        capacity = jnp.broadcast_to(self.volumetric_heat_capacity, temperature.shape)
        energy = capacity * (temperature - self.reference_temperature)
        valid = valid & jnp.isfinite(energy)
        return LatticeEnergyEvaluation(
            _admitted(energy, valid), _admitted(capacity, valid), valid
        )

    def internal_energy(self, temperature):
        return self.evaluate(temperature).internal_energy_density

    def heat_capacity(self, temperature):
        return self.evaluate(temperature).heat_capacity

    def temperature(self, internal_energy_density):
        temperature = (
            self.reference_temperature
            + jnp.asarray(internal_energy_density) / self.volumetric_heat_capacity
        )
        return _admitted(
            temperature, _temperature_valid(temperature, self.temperature_range)
        )

    def component(self, name, volume, /, *, port_count=1, volume_unit=CUBIC_METER):
        """Prepare the native constant-capacity lumped component (C=volume*cv).

        This is a preparation-time adapter; its generic DAE must separately
        retain this law's operating-domain evidence when used in a campaign.
        """
        volume = _positive_scalar(volume, volume_unit, CUBIC_METER, "thermal volume")
        return thermal_capacitance_component(
            name,
            heat_capacity=float(volume * self.volumetric_heat_capacity),
            port_count=port_count,
        )


class ThermalExchangeEvaluation(StrictModule):
    """Paired inward heat powers W and nonnegative entropy production W/K."""

    left_energy_source: Array
    right_energy_source: Array
    entropy_production: Array
    successful: Array


class ThermalConductance(StrictModule):
    """Passive two-terminal conductance G in W/K, including interface/package G.

    A uniform bulk link has G=k*A/L; geometry preparation owns that positive
    metric. A thermal interface uses its measured area-scaled conductance,
    not a semiconductor mobility or band-offset average.
    """

    conductance: Array
    temperature_range: Array
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        conductance,
        /,
        *,
        temperature_range,
        provenance,
        conductance_unit=THERMAL_CONDUCTANCE_UNIT,
        temperature_unit=KELVIN,
    ):
        self.conductance = _positive_scalar(
            conductance, conductance_unit, THERMAL_CONDUCTANCE_UNIT, "thermal conductance"
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        self.provenance = _text(provenance, "thermal conductance provenance")

    def evaluate(self, temperature_left, temperature_right):
        left, right = jnp.broadcast_arrays(
            *map(jnp.asarray, (temperature_left, temperature_right))
        )
        power = self.conductance * (left - right)
        valid = _temperature_valid(left, self.temperature_range) & _temperature_valid(
            right, self.temperature_range
        )
        entropy = power * (left - right) / (left * right)
        valid = valid & jnp.isfinite(power) & jnp.isfinite(entropy)
        return ThermalExchangeEvaluation(
            _admitted(-power, valid),
            _admitted(power, valid),
            _admitted(entropy, valid),
            valid,
        )

    def component(self, name, /):
        """Native conductor; its inward terminal flows are opposite body gains."""
        return thermal_conductor_component(name, conductance=float(self.conductance))


class ThermalBoundaryExchange(StrictModule):
    """Finite device/reservoir conductance, with positive power into each body."""

    conductor: ThermalConductance
    reservoir_temperature: Array

    def __init__(self, conductor, reservoir_temperature, /, *, temperature_unit=KELVIN):
        if not isinstance(conductor, ThermalConductance):
            raise TypeError("conductor must be a ThermalConductance.")
        self.conductor = conductor
        self.reservoir_temperature = _positive_scalar(
            reservoir_temperature, temperature_unit, KELVIN, "reservoir temperature"
        )
        if not bool(
            _temperature_valid(self.reservoir_temperature, conductor.temperature_range)
        ):
            raise ValueError("reservoir temperature must lie in the conductor domain.")

    def evaluate(self, device_temperature):
        """Left is device, right is the external thermal reservoir."""
        return self.conductor.evaluate(device_temperature, self.reservoir_temperature)

    def reservoir_component(self, name, /):
        return temperature_boundary_component(
            name, temperature=float(self.reservoir_temperature)
        )


class CarrierRelaxationEvaluation(StrictModule):
    """Kinetic energy densities J/m³ and paired carrier/lattice powers W/m³."""

    carrier_internal_energy: Array
    equilibrium_internal_energy: Array
    carrier_energy_source: Array
    lattice_energy_source: Array
    successful: Array


class CarrierEnergyRelaxation(StrictModule):
    """Reduced energy relaxation Qc=-(u(n,Tc)-u(n,Tl))/tau at fixed density.

    The shared FD/MB thermodynamics owns statistics and kinetic energy; bands
    and electrostatic energies are excluded. This is an energy-balance moment
    law, not a momentum/hydrodynamic model. The equal/opposite lattice source
    is the ONLY relaxation heat contribution. Carrier creation/removal has
    separate, explicitly energy-resolved source ledgers.
    Material Helmholtz-derived internal energy and its temperature-derivative
    exchange are additional thermodynamic owners, not inferred from this
    kinetic relaxation law or from n*Ec(T).
    """

    thermodynamics: BandThermodynamics
    relaxation_time: Array
    temperature_range: Array
    carrier: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics,
        carrier,
        relaxation_time,
        /,
        *,
        temperature_range,
        provenance,
        time_unit=SECOND,
        temperature_unit=KELVIN,
    ):
        self.thermodynamics = thermodynamics
        self.carrier = _carrier_name(carrier)
        self.relaxation_time = _positive_scalar(
            relaxation_time, time_unit, SECOND, "carrier relaxation time"
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        self.thermodynamics.admit_temperature(self.temperature_range)
        self.provenance = _text(provenance, "carrier relaxation provenance")

    def evaluate(self, density, carrier_temperature, lattice_temperature):
        density, tc, tl = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (density, carrier_temperature, lattice_temperature),
            )
        )
        energy = _kinetic_energy(self.thermodynamics, self.carrier, density, tc)
        equilibrium = _kinetic_energy(self.thermodynamics, self.carrier, density, tl)
        source = -(energy - equilibrium) / self.relaxation_time
        valid = (
            jnp.isfinite(density)
            & (density >= 0)
            & jnp.isfinite(source)
            & _temperature_valid(tc, self.temperature_range)
            & _temperature_valid(tl, self.temperature_range)
            & jnp.isfinite(energy)
            & (energy >= 0)
            & jnp.isfinite(equilibrium)
            & (equilibrium >= 0)
        )
        return CarrierRelaxationEvaluation(
            _admitted(energy, valid),
            _admitted(equilibrium, valid),
            _admitted(source, valid),
            _admitted(-source, valid),
            valid,
        )


class CarrierEnergyFluxEvaluation(StrictModule):
    """One oriented face's powers W; sources are inward into the named cells.

    ``left/right_kinetic_source`` already include band-force work. Adding
    their corresponding band-storage sources yields ``-/+total_power``.
    This is also valid for a gauge shift: energy flux shifts by the datum
    times number flux while kinetic work is invariant.

    ``total_power`` denotes this ELECTRONIC-REFERENCE transfer, not physical
    device total energy including its Poisson field. The band sources are
    only b*dN/dt: they are not complete thermodynamic storage derivatives.
    When the supplied Ec/Ev include -q*psi, storing n*Ec-p*Ev in addition to
    Poisson field energy would double count electrostatic energy. Physical
    storage must retain the field once and use material carrier internal
    energy derived from its Helmholtz free energy, including T derivatives.
    """

    kinetic_power: Array
    band_power: Array
    total_power: Array
    left_kinetic_source: Array
    right_kinetic_source: Array
    left_band_storage_source: Array
    right_band_storage_source: Array
    successful: Array


class CarrierEnergyTransport(StrictModule):
    """3D parabolic energy moment: upwind (5u/3n)Gamma plus Fourier conduction.

    The number flux Gamma is supplied in s^-1 by its sole transport owner.
    Conductivity is in W/(m K); transmissibility A/L is in m. Band edges are
    electron energies in J sharing one reference; holes transport -Ev.
    Arithmetic face band energy allocates band-force work equally to the
    two adjacent cells. Electronic-reference face transfer is conservative;
    physical total-energy assembly must follow the evaluation's ownership
    convention rather than treating this transfer as extra field storage.

    This first-order bulk energy closure does not supply thermopower in the
    number equation, resolve ballistic overshoot, or model abrupt-interface
    transmission. A nonisothermal device additionally needs a compatible
    number/thermoelectric flux, supplied by the assembly rather than inferred
    here. All constant-property coefficients have a bounded temperature range.
    """

    thermodynamics: BandThermodynamics
    thermal_conductivity: Array
    temperature_range: Array
    carrier: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics,
        carrier,
        thermal_conductivity,
        /,
        *,
        temperature_range,
        provenance,
        conductivity_unit=THERMAL_CONDUCTIVITY_UNIT,
        temperature_unit=KELVIN,
    ):
        self.thermodynamics = thermodynamics
        self.carrier = _carrier_name(carrier)
        self.thermal_conductivity = _positive_scalar(
            thermal_conductivity,
            conductivity_unit,
            THERMAL_CONDUCTIVITY_UNIT,
            "carrier thermal conductivity",
            nonnegative=True,
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        self.thermodynamics.admit_temperature(self.temperature_range)
        self.provenance = _text(provenance, "carrier energy flux provenance")

    def evaluate(
        self,
        number_flux,
        density_left,
        density_right,
        temperature_left,
        temperature_right,
        band_edge_left,
        band_edge_right,
        transmissibility,
    ):
        flux, nl, nr, tl, tr, bl, br, metric = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (
                    number_flux,
                    density_left,
                    density_right,
                    temperature_left,
                    temperature_right,
                    band_edge_left,
                    band_edge_right,
                    transmissibility,
                ),
            )
        )
        ul = _kinetic_energy(self.thermodynamics, self.carrier, nl, tl)
        ur = _kinetic_energy(self.thermodynamics, self.carrier, nr, tr)
        enthalpy = jnp.where(flux >= 0, 5 * ul / (3 * nl), 5 * ur / (3 * nr))
        kinetic = flux * enthalpy + self.thermal_conductivity * metric * (tl - tr)
        if self.carrier == "hole":
            bl, br = -bl, -br
        band_power = 0.5 * (bl + br) * flux
        work = 0.5 * (bl - br) * flux
        valid = (
            jnp.isfinite(flux)
            & jnp.isfinite(nl)
            & (nl > 0)
            & jnp.isfinite(nr)
            & (nr > 0)
            & _temperature_valid(tl, self.temperature_range)
            & _temperature_valid(tr, self.temperature_range)
            & jnp.isfinite(bl)
            & jnp.isfinite(br)
            & jnp.isfinite(metric)
            & (metric > 0)
            & jnp.isfinite(ul)
            & (ul >= 0)
            & jnp.isfinite(ur)
            & (ur >= 0)
            & jnp.isfinite(kinetic)
            & jnp.isfinite(band_power)
        )
        return CarrierEnergyFluxEvaluation(
            _admitted(kinetic, valid),
            _admitted(band_power, valid),
            _admitted(kinetic + band_power, valid),
            _admitted(-kinetic + work, valid),
            _admitted(kinetic + work, valid),
            _admitted(-bl * flux, valid),
            _admitted(br * flux, valid),
            valid,
        )


__all__ = [
    "CarrierEnergyFluxEvaluation",
    "CarrierEnergyRelaxation",
    "CarrierEnergyTransport",
    "CarrierRelaxationEvaluation",
    "ConstantLatticeHeatCapacity",
    "LatticeEnergyEvaluation",
    "ThermalBoundaryExchange",
    "ThermalConductance",
    "ThermalExchangeEvaluation",
]
