#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._types import AtomisticScaleContract


class AtomisticUnitSystem(StrictModule, NonTrainableState):
    """Complete unit identity for atomistic dynamics.

    ``AtomisticScaleContract`` remains the authoritative length/energy identity.
    This contract supplies the additional dimensions and physical constants needed
    by temporal dynamics.  Numeric states stay in the declared units; conversion
    factors are resolved once and become part of the immutable identity.
    """

    scale: AtomisticScaleContract
    mass_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    charge_unit: str = eqx.field(static=True)
    temperature_unit: str = eqx.field(static=True)
    mass_to_reference: float = eqx.field(static=True)
    time_to_reference: float = eqx.field(static=True)
    charge_to_reference: float = eqx.field(static=True)
    temperature_to_reference: float = eqx.field(static=True)
    kinetic_to_energy: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    coulomb_constant: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        /,
        *,
        mass_unit: str,
        time_unit: str,
        charge_unit: str,
        temperature_unit: str,
        mass_to_reference: float = 1.0,
        time_to_reference: float = 1.0,
        charge_to_reference: float = 1.0,
        temperature_to_reference: float = 1.0,
        kinetic_to_energy: float,
        boltzmann_constant: float,
        coulomb_constant: float,
        reduced_planck_constant: float,
    ):
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        names = tuple(
            str(value).strip()
            for value in (mass_unit, time_unit, charge_unit, temperature_unit)
        )
        if any(not value for value in names):
            raise ValueError("Dynamics unit names must be non-empty.")
        factors = tuple(
            float(value)
            for value in (
                mass_to_reference,
                time_to_reference,
                charge_to_reference,
                temperature_to_reference,
                kinetic_to_energy,
                boltzmann_constant,
                coulomb_constant,
                reduced_planck_constant,
            )
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError(
                "Dynamics conversion factors and constants must be finite and positive."
            )
        (
            mass_factor,
            time_factor,
            charge_factor,
            temperature_factor,
            kinetic_factor,
            boltzmann,
            coulomb,
            hbar,
        ) = factors
        self.scale = scale
        self.mass_unit, self.time_unit, self.charge_unit, self.temperature_unit = names
        self.mass_to_reference = mass_factor
        self.time_to_reference = time_factor
        self.charge_to_reference = charge_factor
        self.temperature_to_reference = temperature_factor
        self.kinetic_to_energy = kinetic_factor
        self.boltzmann_constant = boltzmann
        self.coulomb_constant = coulomb
        self.reduced_planck_constant = hbar
        self.unit_system_id = canonical_fingerprint(
            {
                "kind": "atomistic-unit-system",
                "scale": scale.scale_id,
                "mass_unit": names[0],
                "time_unit": names[1],
                "charge_unit": names[2],
                "temperature_unit": names[3],
                "mass_to_reference": mass_factor,
                "time_to_reference": time_factor,
                "charge_to_reference": charge_factor,
                "temperature_to_reference": temperature_factor,
                "kinetic_to_energy": kinetic_factor,
                "boltzmann_constant": boltzmann,
                "coulomb_constant": coulomb,
                "reduced_planck_constant": hbar,
            }
        )

    @classmethod
    def electronvolt_angstrom_dalton_femtosecond(
        cls, scale: AtomisticScaleContract | None = None, /
    ) -> "AtomisticUnitSystem":
        """Return the independently specified eV–Å–Da–fs–K unit system."""

        scale_ = (
            AtomisticScaleContract("angstrom", "electronvolt") if scale is None else scale
        )
        if (
            scale_.length_unit.strip().lower() != "angstrom"
            or scale_.energy_unit.strip().lower() != "electronvolt"
            or scale_.length_to_reference != 1.0
            or scale_.energy_to_reference != 1.0
        ):
            raise ValueError(
                "The eV–Å–Da–fs constructor requires an exact angstrom/electronvolt scale."
            )
        return cls(
            scale_,
            mass_unit="dalton",
            time_unit="femtosecond",
            charge_unit="elementary_charge",
            temperature_unit="kelvin",
            kinetic_to_energy=103.64269652680505,
            boltzmann_constant=8.617333262145e-5,
            coulomb_constant=14.399645478425668,
            reduced_planck_constant=0.6582119569509067,
        )

    @classmethod
    def reduced(
        cls, scale: AtomisticScaleContract | None = None, /
    ) -> "AtomisticUnitSystem":
        """Return an explicit dimensionless reduced unit system."""

        scale_ = (
            AtomisticScaleContract("reduced_length", "reduced_energy")
            if scale is None
            else scale
        )
        return cls(
            scale_,
            mass_unit="reduced_mass",
            time_unit="reduced_time",
            charge_unit="reduced_charge",
            temperature_unit="reduced_temperature",
            kinetic_to_energy=1.0,
            boltzmann_constant=1.0,
            coulomb_constant=1.0,
            reduced_planck_constant=1.0,
        )

    @property
    def force_to_momentum_rate(self) -> float:
        """Convert an energy/length force to mass·length/time²."""

        return 1.0 / self.kinetic_to_energy


__all__ = ["AtomisticUnitSystem"]
