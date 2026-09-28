#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import frexp, isfinite, pi
from numbers import Integral, Real
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .typing import parse
from .units import (
    CHARGE,
    COULOMB,
    derived_unit,
    DimensionSignature,
    KILOGRAM,
    LENGTH,
    MASS,
    METER,
    SECOND,
    TEMPERATURE,
    TIME,
    UnitDefinition,
)


type LengthCoordinateKind = Literal["physical", "comoving", "code"]


class SpatialCoordinateContract(StrictModule, NonTrainableState):
    """Exact length-unit, coordinate-kind, coordinate-system, and frame identity."""

    length_unit: UnitDefinition = eqx.field(static=True)
    length_coordinate_kind: LengthCoordinateKind = eqx.field(static=True)
    coordinate_system: str = eqx.field(static=True)
    reference_frame: str = eqx.field(static=True)
    spatial_id: str = eqx.field(static=True)

    def __init__(
        self,
        length_unit: UnitDefinition,
        /,
        *,
        length_coordinate_kind: LengthCoordinateKind = "physical",
        coordinate_system: str = "cartesian",
        reference_frame: str = "world",
    ) -> None:
        if not isinstance(length_unit, UnitDefinition):
            raise TypeError("length_unit must be a UnitDefinition.")
        if length_unit.dimension != LENGTH:
            raise ValueError("Spatial coordinate length_unit must have length dimension.")
        kind = str(length_coordinate_kind).strip()
        kind = parse(kind, LengthCoordinateKind, "length_coordinate_kind")
        system = str(coordinate_system).strip()
        frame = str(reference_frame).strip()
        if not system or not frame:
            raise ValueError("Coordinate system and reference frame must be non-empty.")
        self.length_unit = length_unit
        self.length_coordinate_kind = kind
        self.coordinate_system = system
        self.reference_frame = frame
        self.spatial_id = canonical_fingerprint(
            {
                "kind": "spatial-coordinate-contract",
                "length_unit": length_unit.unit_id,
                "length_coordinate_kind": kind,
                "coordinate_system": system,
                "reference_frame": frame,
            }
        )

    @classmethod
    def si(cls) -> SpatialCoordinateContract:
        return cls(METER)

    def to_dict(self) -> dict[str, object]:
        return {
            "length_unit": self.length_unit.to_dict(),
            "length_coordinate_kind": self.length_coordinate_kind,
            "coordinate_system": self.coordinate_system,
            "reference_frame": self.reference_frame,
            "spatial_id": self.spatial_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpatialCoordinateContract:
        if not isinstance(payload, Mapping):
            raise TypeError("Spatial coordinate payload must be a mapping.")
        expected = {
            "length_unit",
            "length_coordinate_kind",
            "coordinate_system",
            "reference_frame",
            "spatial_id",
        }
        if set(payload) != expected:
            raise ValueError("Spatial coordinate payload must use the canonical fields.")
        unit_payload = payload["length_unit"]
        if not isinstance(unit_payload, Mapping):
            raise TypeError("Spatial coordinate length unit must be a mapping.")
        coordinate_kind = payload["length_coordinate_kind"]
        coordinate_system = payload["coordinate_system"]
        reference_frame = payload["reference_frame"]
        if not all(
            isinstance(value, str)
            for value in (coordinate_kind, coordinate_system, reference_frame)
        ):
            raise TypeError("Spatial coordinate labels must be strings.")
        contract = cls(
            UnitDefinition.from_dict(unit_payload),
            length_coordinate_kind=coordinate_kind,
            coordinate_system=coordinate_system,
            reference_frame=reference_frame,
        )
        claimed_id = payload["spatial_id"]
        if not isinstance(claimed_id, str):
            raise TypeError("Spatial coordinate spatial_id must be a string.")
        if claimed_id != contract.spatial_id:
            raise ValueError(
                "Spatial coordinate payload fingerprint does not match its content."
            )
        return contract


class DimensionalScaleContract(StrictModule, NonTrainableState):
    """Shared exact length, mass, time, and coordinate-kind identity."""

    length_unit: UnitDefinition = eqx.field(static=True)
    mass_unit: UnitDefinition = eqx.field(static=True)
    time_unit: UnitDefinition = eqx.field(static=True)
    length_coordinate_kind: LengthCoordinateKind = eqx.field(static=True)
    velocity_unit: UnitDefinition = eqx.field(static=True)
    acceleration_unit: UnitDefinition = eqx.field(static=True)
    gravitational_parameter_unit: UnitDefinition = eqx.field(static=True)
    gravitational_constant_unit: UnitDefinition = eqx.field(static=True)
    hubble_unit: UnitDefinition = eqx.field(static=True)
    wavenumber_unit: UnitDefinition = eqx.field(static=True)
    power_spectrum_unit: UnitDefinition = eqx.field(static=True)
    potential_unit: UnitDefinition = eqx.field(static=True)
    canonical_momentum_unit: UnitDefinition = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        length_unit: UnitDefinition,
        mass_unit: UnitDefinition,
        time_unit: UnitDefinition,
        /,
        *,
        length_coordinate_kind: LengthCoordinateKind = "physical",
    ) -> None:
        if not all(
            isinstance(unit, UnitDefinition)
            for unit in (length_unit, mass_unit, time_unit)
        ):
            raise TypeError("Dimensional scale units must be UnitDefinition values.")
        if (
            length_unit.dimension != LENGTH
            or mass_unit.dimension != MASS
            or time_unit.dimension != TIME
        ):
            raise ValueError(
                "Dimensional scale units must have length, mass, and time dimensions."
            )
        if (
            len(
                {
                    length_unit.reference_system_id,
                    mass_unit.reference_system_id,
                    time_unit.reference_system_id,
                }
            )
            != 1
        ):
            raise ValueError(
                "Dimensional scale units must share one explicit reference system."
            )
        kind = str(length_coordinate_kind).strip()
        kind = parse(kind, LengthCoordinateKind, "length_coordinate_kind")

        length_symbol = length_unit.symbol
        mass_symbol = mass_unit.symbol
        time_symbol = time_unit.symbol
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.time_unit = time_unit
        self.length_coordinate_kind = kind
        self.velocity_unit = derived_unit(
            f"{length_symbol}/{time_symbol}",
            ((length_unit, 1), (time_unit, -1)),
        )
        self.acceleration_unit = derived_unit(
            f"{length_symbol}/{time_symbol}^2",
            ((length_unit, 1), (time_unit, -2)),
        )
        self.gravitational_parameter_unit = derived_unit(
            f"{length_symbol}^3/{time_symbol}^2",
            ((length_unit, 3), (time_unit, -2)),
        )
        self.gravitational_constant_unit = derived_unit(
            f"{length_symbol}^3/({mass_symbol}*{time_symbol}^2)",
            ((length_unit, 3), (mass_unit, -1), (time_unit, -2)),
        )
        self.hubble_unit = derived_unit(f"1/{time_symbol}", ((time_unit, -1),))
        self.wavenumber_unit = derived_unit(f"1/{length_symbol}", ((length_unit, -1),))
        self.power_spectrum_unit = derived_unit(f"{length_symbol}^3", ((length_unit, 3),))
        self.potential_unit = derived_unit(
            f"{length_symbol}^2/{time_symbol}^2",
            ((length_unit, 2), (time_unit, -2)),
        )
        self.canonical_momentum_unit = derived_unit(
            f"{mass_symbol}*{length_symbol}/{time_symbol}",
            ((mass_unit, 1), (length_unit, 1), (time_unit, -1)),
        )
        self.scale_id = canonical_fingerprint(
            {
                "kind": "dimensional-scale-contract",
                "length_unit": length_unit.unit_id,
                "mass_unit": mass_unit.unit_id,
                "time_unit": time_unit.unit_id,
                "length_coordinate_kind": kind,
            }
        )

    @classmethod
    def si(cls) -> DimensionalScaleContract:
        return cls(METER, KILOGRAM, SECOND, length_coordinate_kind="physical")

    def to_dict(self) -> dict[str, object]:
        return {
            "length_unit": self.length_unit.to_dict(),
            "mass_unit": self.mass_unit.to_dict(),
            "time_unit": self.time_unit.to_dict(),
            "length_coordinate_kind": self.length_coordinate_kind,
            "scale_id": self.scale_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DimensionalScaleContract:
        if not isinstance(payload, Mapping):
            raise TypeError("Dimensional scale payload must be a mapping.")
        expected = {
            "length_unit",
            "mass_unit",
            "time_unit",
            "length_coordinate_kind",
            "scale_id",
        }
        if set(payload) != expected:
            raise ValueError("Dimensional scale payload must use the canonical fields.")
        length_payload = payload.get("length_unit")
        mass_payload = payload.get("mass_unit")
        time_payload = payload.get("time_unit")
        if (
            not isinstance(length_payload, Mapping)
            or not isinstance(mass_payload, Mapping)
            or not isinstance(time_payload, Mapping)
        ):
            raise TypeError("Dimensional scale unit payloads must be mappings.")
        coordinate_kind = payload.get("length_coordinate_kind")
        if not isinstance(coordinate_kind, str):
            raise TypeError("Dimensional scale coordinate kind must be a string.")
        scale = cls(
            UnitDefinition.from_dict(length_payload),
            UnitDefinition.from_dict(mass_payload),
            UnitDefinition.from_dict(time_payload),
            length_coordinate_kind=coordinate_kind,
        )
        claimed_id = payload.get("scale_id")
        if not isinstance(claimed_id, str):
            raise TypeError("Dimensional scale payload scale_id must be a string.")
        if claimed_id != scale.scale_id:
            raise ValueError(
                "Dimensional scale payload fingerprint does not match its content."
            )
        return scale


type PhysicalConstant = int | float | str | Fraction


def _positive_constant(value: PhysicalConstant, name: str, /) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    if isinstance(value, (Integral, Fraction, str)):
        result = Fraction(value)
    elif isinstance(value, Real):
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite and positive.")
        result = Fraction(str(value))
    else:
        raise TypeError(
            f"{name} must be an exact rational, decimal string, or real scalar."
        )
    if result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _fraction_payload(value: Fraction, /) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_payload(payload: object, name: str, /) -> Fraction:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} payload must be a mapping.")
    if set(payload) != {"numerator", "denominator"}:
        raise ValueError(f"{name} payload must use numerator and denominator.")
    numerator = payload["numerator"]
    denominator = payload["denominator"]
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, Integral)
        or not isinstance(denominator, Integral)
    ):
        raise TypeError(f"{name} numerator and denominator must be integers.")
    return _positive_constant(
        Fraction(int(numerator), int(denominator)),
        name,
    )


def _apply_exact_factor(value: ArrayLike, factor: Fraction, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError("Relativity scale conversions require real-valued inputs.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        dtype = jnp.result_type(array, jnp.asarray(1.0))
        array = array.astype(dtype)
    mantissa, exponent = frexp(float(factor))
    scalar = jnp.asarray(mantissa, dtype=array.dtype)
    return jnp.ldexp(array * scalar, exponent)


class RelativityScaleContract(StrictModule, NonTrainableState):
    """Exact base-unit realization of relativistic and quantum constants.

    Constant values are stored as exact rational numbers in units derived from
    ``dimensional_scale``. Numerical conversions cast their final exact factor
    once to the input dtype and never clip or repair the input.
    """

    dimensional_scale: DimensionalScaleContract = eqx.field(static=True)
    gravitational_constant: Fraction = eqx.field(static=True)
    speed_of_light: Fraction = eqx.field(static=True)
    reduced_planck_constant: Fraction = eqx.field(static=True)
    boltzmann_constant: Fraction = eqx.field(static=True)
    quantum_constants_explicit: bool = eqx.field(static=True)
    temperature_unit: UnitDefinition = eqx.field(static=True)
    energy_unit: UnitDefinition = eqx.field(static=True)
    mass_density_unit: UnitDefinition = eqx.field(static=True)
    energy_density_unit: UnitDefinition = eqx.field(static=True)
    specific_energy_unit: UnitDefinition = eqx.field(static=True)
    reduced_planck_constant_unit: UnitDefinition = eqx.field(static=True)
    boltzmann_constant_unit: UnitDefinition = eqx.field(static=True)
    entropy_unit: UnitDefinition = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimensional_scale: DimensionalScaleContract,
        gravitational_constant: PhysicalConstant,
        speed_of_light: PhysicalConstant,
        reduced_planck_constant: PhysicalConstant,
        boltzmann_constant: PhysicalConstant,
        quantum_constants_explicit: bool = True,
    ) -> None:
        if not isinstance(dimensional_scale, DimensionalScaleContract):
            raise TypeError("dimensional_scale must be a DimensionalScaleContract.")
        if not isinstance(quantum_constants_explicit, bool):
            raise TypeError("quantum_constants_explicit must be a bool.")
        gravitational = _positive_constant(
            gravitational_constant,
            "gravitational_constant",
        )
        light_speed = _positive_constant(speed_of_light, "speed_of_light")
        planck = _positive_constant(
            reduced_planck_constant,
            "reduced_planck_constant",
        )
        boltzmann = _positive_constant(boltzmann_constant, "boltzmann_constant")

        length = dimensional_scale.length_unit
        mass = dimensional_scale.mass_unit
        time = dimensional_scale.time_unit
        reference_system_id = length.reference_system_id
        temperature = UnitDefinition(
            "K",
            TEMPERATURE,
            reference_system_id,
        )
        energy = derived_unit(
            f"{mass.symbol}*{length.symbol}^2/{time.symbol}^2",
            ((mass, 1), (length, 2), (time, -2)),
        )
        mass_density = derived_unit(
            f"{mass.symbol}/{length.symbol}^3",
            ((mass, 1), (length, -3)),
        )
        energy_density = derived_unit(
            f"{energy.symbol}/{length.symbol}^3",
            ((energy, 1), (length, -3)),
        )
        specific_energy = derived_unit(
            f"{length.symbol}^2/{time.symbol}^2",
            ((length, 2), (time, -2)),
        )
        planck_unit = derived_unit(
            f"{mass.symbol}*{length.symbol}^2/{time.symbol}",
            ((mass, 1), (length, 2), (time, -1)),
        )
        boltzmann_unit = derived_unit(
            f"{energy.symbol}/{temperature.symbol}",
            ((energy, 1), (temperature, -1)),
        )

        self.dimensional_scale = dimensional_scale
        self.gravitational_constant = gravitational
        self.speed_of_light = light_speed
        self.reduced_planck_constant = planck
        self.boltzmann_constant = boltzmann
        self.quantum_constants_explicit = quantum_constants_explicit
        self.temperature_unit = temperature
        self.energy_unit = energy
        self.mass_density_unit = mass_density
        self.energy_density_unit = energy_density
        self.specific_energy_unit = specific_energy
        self.reduced_planck_constant_unit = planck_unit
        self.boltzmann_constant_unit = boltzmann_unit
        self.entropy_unit = boltzmann_unit
        self.scale_id = canonical_fingerprint(
            {
                "kind": "relativity-scale-contract",
                "dimensional_scale": dimensional_scale.scale_id,
                "gravitational_constant": [
                    gravitational.numerator,
                    gravitational.denominator,
                ],
                "speed_of_light": [
                    light_speed.numerator,
                    light_speed.denominator,
                ],
                "reduced_planck_constant": [
                    planck.numerator,
                    planck.denominator,
                ],
                "boltzmann_constant": [
                    boltzmann.numerator,
                    boltzmann.denominator,
                ],
                "quantum_constants_explicit": quantum_constants_explicit,
            }
        )

    @classmethod
    def si(cls) -> RelativityScaleContract:
        """Return the declared SI realization of ``G``, ``c``, ``hbar``, and ``k_B``."""
        return cls(
            DimensionalScaleContract.si(),
            "6.67430e-11",
            299_792_458,
            # SI fixes h = 6.62607015e-34 J s exactly; hbar = h / (2 pi) is
            # irrational, so store it as the exact decimal rational carrying 40
            # significant digits (relative error < 1e-39), far below float64
            # resolution, so float(hbar) is correctly rounded.
            "1.054571817646156391262428003302280744723e-34",
            "1.380649e-23",
        )

    @classmethod
    def from_si(
        cls,
        dimensional_scale: DimensionalScaleContract,
        /,
    ) -> RelativityScaleContract:
        """Express the declared SI constants exactly in another SI-referenced scale."""
        if not isinstance(dimensional_scale, DimensionalScaleContract):
            raise TypeError("dimensional_scale must be a DimensionalScaleContract.")
        if dimensional_scale.length_unit.reference_system_id != "si":
            raise ValueError("from_si requires units referenced to the SI system.")
        length_scale = dimensional_scale.length_unit.scale_to_reference
        mass_scale = dimensional_scale.mass_unit.scale_to_reference
        time_scale = dimensional_scale.time_unit.scale_to_reference
        speed_unit_scale = length_scale / time_scale
        gravitational_unit_scale = length_scale**3 / (mass_scale * time_scale**2)
        planck_unit_scale = mass_scale * length_scale**2 / time_scale
        energy_unit_scale = mass_scale * length_scale**2 / time_scale**2
        return cls(
            dimensional_scale,
            Fraction(66_743, 10**15) / gravitational_unit_scale,
            Fraction(299_792_458) / speed_unit_scale,
            Fraction(1_054_571_817, 10**43) / planck_unit_scale,
            Fraction(1_380_649, 10**29) / energy_unit_scale,
        )

    @classmethod
    def geometric(
        cls,
        mass_unit: UnitDefinition,
        /,
    ) -> RelativityScaleContract:
        """Return ``G=c=1`` units anchored by one SI-referenced mass unit."""
        if not isinstance(mass_unit, UnitDefinition):
            raise TypeError("mass_unit must be a UnitDefinition.")
        if mass_unit.dimension != MASS:
            raise ValueError("Geometric relativity mass_unit must have mass dimension.")
        if mass_unit.reference_system_id != "si":
            raise ValueError("Geometric relativity units require an SI-referenced mass.")
        gravitational_si = Fraction(66_743, 10**15)
        light_speed_si = Fraction(299_792_458)
        mass_scale = mass_unit.scale_to_reference
        length_scale = gravitational_si * mass_scale / light_speed_si**2
        time_scale = gravitational_si * mass_scale / light_speed_si**3
        length_unit = UnitDefinition(
            f"G*{mass_unit.symbol}/c^2",
            LENGTH,
            "si",
            length_scale,
        )
        time_unit = UnitDefinition(
            f"G*{mass_unit.symbol}/c^3",
            TIME,
            "si",
            time_scale,
        )
        return cls.from_si(
            DimensionalScaleContract(
                length_unit,
                mass_unit,
                time_unit,
                length_coordinate_kind="physical",
            )
        )

    def mass_to_geometric_length(self, mass: ArrayLike, /) -> Array:
        """Convert mass to ``G M / c^2`` in the contract length unit."""
        return _apply_exact_factor(
            mass,
            self.gravitational_constant / self.speed_of_light**2,
        )

    def geometric_length_to_mass(self, length: ArrayLike, /) -> Array:
        """Invert :meth:`mass_to_geometric_length` without a repair branch."""
        return _apply_exact_factor(
            length,
            self.speed_of_light**2 / self.gravitational_constant,
        )

    def mass_to_geometric_time(self, mass: ArrayLike, /) -> Array:
        """Convert mass to ``G M / c^3`` in the contract time unit."""
        return _apply_exact_factor(
            mass,
            self.gravitational_constant / self.speed_of_light**3,
        )

    def geometric_time_to_mass(self, time: ArrayLike, /) -> Array:
        """Invert :meth:`mass_to_geometric_time` without a repair branch."""
        return _apply_exact_factor(
            time,
            self.speed_of_light**3 / self.gravitational_constant,
        )

    def _require_quantum_constants(self) -> None:
        if not self.quantum_constants_explicit:
            raise ValueError(
                "Quantum conversion requires explicitly declared hbar and k_B."
            )

    def surface_gravity_to_temperature(self, surface_gravity: ArrayLike, /) -> Array:
        """Convert physical surface gravity to Hawking temperature."""
        self._require_quantum_constants()
        factor = self.reduced_planck_constant / (
            2 * Fraction.from_float(pi) * self.boltzmann_constant * self.speed_of_light
        )
        return _apply_exact_factor(surface_gravity, factor)

    def temperature_to_surface_gravity(self, temperature: ArrayLike, /) -> Array:
        """Invert :meth:`surface_gravity_to_temperature`."""
        self._require_quantum_constants()
        factor = (
            2
            * Fraction.from_float(pi)
            * self.boltzmann_constant
            * self.speed_of_light
            / self.reduced_planck_constant
        )
        return _apply_exact_factor(temperature, factor)

    def area_to_entropy(self, area: ArrayLike, /) -> Array:
        """Convert horizon area to Bekenstein--Hawking entropy."""
        self._require_quantum_constants()
        factor = (
            self.boltzmann_constant
            * self.speed_of_light**3
            / (4 * self.reduced_planck_constant * self.gravitational_constant)
        )
        return _apply_exact_factor(area, factor)

    def entropy_to_area(self, entropy: ArrayLike, /) -> Array:
        """Invert :meth:`area_to_entropy`."""
        self._require_quantum_constants()
        factor = (
            4
            * self.reduced_planck_constant
            * self.gravitational_constant
            / (self.boltzmann_constant * self.speed_of_light**3)
        )
        return _apply_exact_factor(entropy, factor)

    def to_dict(self) -> dict[str, object]:
        return {
            "dimensional_scale": self.dimensional_scale.to_dict(),
            "gravitational_constant": _fraction_payload(self.gravitational_constant),
            "speed_of_light": _fraction_payload(self.speed_of_light),
            "reduced_planck_constant": _fraction_payload(self.reduced_planck_constant),
            "boltzmann_constant": _fraction_payload(self.boltzmann_constant),
            "quantum_constants_explicit": self.quantum_constants_explicit,
            "scale_id": self.scale_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RelativityScaleContract:
        if not isinstance(payload, Mapping):
            raise TypeError("Relativity scale payload must be a mapping.")
        expected = {
            "dimensional_scale",
            "gravitational_constant",
            "speed_of_light",
            "reduced_planck_constant",
            "boltzmann_constant",
            "quantum_constants_explicit",
            "scale_id",
        }
        if set(payload) != expected:
            raise ValueError("Relativity scale payload must use the canonical fields.")
        dimensional_payload = payload["dimensional_scale"]
        if not isinstance(dimensional_payload, Mapping):
            raise TypeError("Relativity dimensional scale must be a mapping.")
        explicit = payload["quantum_constants_explicit"]
        if not isinstance(explicit, bool):
            raise TypeError("quantum_constants_explicit payload must be a bool.")
        contract = cls(
            DimensionalScaleContract.from_dict(dimensional_payload),
            _fraction_from_payload(
                payload["gravitational_constant"],
                "gravitational_constant",
            ),
            _fraction_from_payload(payload["speed_of_light"], "speed_of_light"),
            _fraction_from_payload(
                payload["reduced_planck_constant"],
                "reduced_planck_constant",
            ),
            _fraction_from_payload(
                payload["boltzmann_constant"],
                "boltzmann_constant",
            ),
            explicit,
        )
        claimed_id = payload["scale_id"]
        if not isinstance(claimed_id, str):
            raise TypeError("Relativity scale payload scale_id must be a string.")
        if claimed_id != contract.scale_id:
            raise ValueError(
                "Relativity scale payload fingerprint does not match its content."
            )
        return contract


_CODATA_2022_ELEMENTARY_CHARGE = Fraction(1_602_176_634, 10**28)
_CODATA_2022_ELECTRON_MASS = Fraction(91_093_837_139, 10**41)
_CODATA_2022_VACUUM_PERMITTIVITY = Fraction(88_541_878_188, 10**22)

# openPMD unitDimension axis order: L, M, T, I, theta, N, J.
_OPENPMD_AXES = ("length", "mass", "time", "current", "temperature", "amount", "luminous")


def _openpmd_unit_dimension(dimension: DimensionSignature, /) -> tuple[float, ...]:
    exponents = {axis: Fraction(0) for axis in _OPENPMD_AXES}
    for axis, numerator, denominator in dimension.terms:
        exponent = Fraction(numerator, denominator)
        match axis:
            case "charge":
                exponents["current"] += exponent
                exponents["time"] += exponent
            case "length" | "mass" | "time" | "temperature" | "amount":
                exponents[axis] += exponent
            case _:
                raise ValueError(
                    f"Dimension axis {axis!r} has no openPMD representation."
                )
    return tuple(float(exponents[axis]) for axis in _OPENPMD_AXES)


class ElectromagneticScaleContract(StrictModule, NonTrainableState):
    """Exact realization of ``e``, ``m_e``, and ``epsilon_0`` over a relativity scale.

    The speed of light and reduced Planck constant are owned by ``relativity``.
    Constants are exact rationals in the units of ``relativity.dimensional_scale``
    and ``charge_unit``; derived constants are exact up to the declared ``pi``.
    """

    relativity: RelativityScaleContract = eqx.field(static=True)
    charge_unit: UnitDefinition = eqx.field(static=True)
    elementary_charge: Fraction = eqx.field(static=True)
    electron_mass: Fraction = eqx.field(static=True)
    vacuum_permittivity: Fraction = eqx.field(static=True)
    constant_set_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        relativity: RelativityScaleContract,
        charge_unit: UnitDefinition,
        elementary_charge: PhysicalConstant,
        electron_mass: PhysicalConstant,
        vacuum_permittivity: PhysicalConstant,
        /,
        *,
        constant_set_id: str,
    ) -> None:
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be a RelativityScaleContract.")
        if not relativity.quantum_constants_explicit:
            raise ValueError(
                "Electromagnetic scales require explicitly declared hbar and k_B."
            )
        if not isinstance(charge_unit, UnitDefinition):
            raise TypeError("charge_unit must be a UnitDefinition.")
        if charge_unit.dimension != CHARGE:
            raise ValueError("charge_unit must have charge dimension.")
        reference_system_id = relativity.dimensional_scale.length_unit.reference_system_id
        if charge_unit.reference_system_id != reference_system_id:
            raise ValueError(
                "charge_unit must share the dimensional scale reference system."
            )
        if not isinstance(constant_set_id, str) or not constant_set_id.strip():
            raise ValueError("constant_set_id must be a non-empty string.")
        charge = _positive_constant(elementary_charge, "elementary_charge")
        mass = _positive_constant(electron_mass, "electron_mass")
        permittivity = _positive_constant(vacuum_permittivity, "vacuum_permittivity")
        constant_set = constant_set_id.strip()
        self.relativity = relativity
        self.charge_unit = charge_unit
        self.elementary_charge = charge
        self.electron_mass = mass
        self.vacuum_permittivity = permittivity
        self.constant_set_id = constant_set
        self.scale_id = canonical_fingerprint(
            {
                "kind": "electromagnetic-scale-contract",
                "relativity": relativity.scale_id,
                "charge_unit": charge_unit.unit_id,
                "elementary_charge": [charge.numerator, charge.denominator],
                "electron_mass": [mass.numerator, mass.denominator],
                "vacuum_permittivity": [
                    permittivity.numerator,
                    permittivity.denominator,
                ],
                "constant_set_id": constant_set,
            }
        )

    @classmethod
    def si(cls) -> ElectromagneticScaleContract:
        """Return CODATA 2022 SI values over :meth:`RelativityScaleContract.si`."""
        return cls(
            RelativityScaleContract.si(),
            COULOMB,
            _CODATA_2022_ELEMENTARY_CHARGE,
            _CODATA_2022_ELECTRON_MASS,
            _CODATA_2022_VACUUM_PERMITTIVITY,
            constant_set_id="codata-2022",
        )

    @classmethod
    def code_units(
        cls,
        dimensional_scale: DimensionalScaleContract,
        charge_unit: UnitDefinition,
        /,
        *,
        gravitational_constant: PhysicalConstant,
        speed_of_light: PhysicalConstant,
        reduced_planck_constant: PhysicalConstant,
        boltzmann_constant: PhysicalConstant,
        elementary_charge: PhysicalConstant,
        electron_mass: PhysicalConstant,
        vacuum_permittivity: PhysicalConstant,
        constant_set_id: str,
    ) -> ElectromagneticScaleContract:
        """Build a code-unit scale with every constant declared explicitly."""
        return cls(
            RelativityScaleContract(
                dimensional_scale,
                gravitational_constant,
                speed_of_light,
                reduced_planck_constant,
                boltzmann_constant,
            ),
            charge_unit,
            elementary_charge,
            electron_mass,
            vacuum_permittivity,
            constant_set_id=constant_set_id,
        )

    @property
    def speed_of_light(self) -> Fraction:
        return self.relativity.speed_of_light

    @property
    def reduced_planck_constant(self) -> Fraction:
        return self.relativity.reduced_planck_constant

    @property
    def vacuum_permeability(self) -> Fraction:
        """``mu_0 = 1 / (epsilon_0 c^2)``."""
        return 1 / (self.vacuum_permittivity * self.speed_of_light**2)

    @property
    def vacuum_impedance(self) -> Fraction:
        """``Z_0 = 1 / (epsilon_0 c)``."""
        return 1 / (self.vacuum_permittivity * self.speed_of_light)

    @property
    def fine_structure(self) -> Fraction:
        """``alpha = e^2 / (4 pi epsilon_0 hbar c)``."""
        return self.elementary_charge**2 / (
            4
            * Fraction.from_float(pi)
            * self.vacuum_permittivity
            * self.reduced_planck_constant
            * self.speed_of_light
        )

    @property
    def classical_electron_radius(self) -> Fraction:
        """``r_e = e^2 / (4 pi epsilon_0 m_e c^2)``."""
        return self.elementary_charge**2 / (
            4
            * Fraction.from_float(pi)
            * self.vacuum_permittivity
            * self.electron_mass
            * self.speed_of_light**2
        )

    @property
    def schwinger_field(self) -> Fraction:
        """Critical field ``E_S = m_e^2 c^3 / (e hbar)``."""
        return (
            self.electron_mass**2
            * self.speed_of_light**3
            / (self.elementary_charge * self.reduced_planck_constant)
        )

    def unit_si_map(self) -> dict[str, tuple[float, tuple[float, ...]]]:
        """Return openPMD ``(unitSI, unitDimension)`` for each field quantity.

        ``unitDimension`` holds seven powers in openPMD order ``(L, M, T, I,
        theta, N, J)``; charge contributes ``I*T``.
        """
        if self.charge_unit.reference_system_id != "si":
            raise ValueError("unit_si_map requires units referenced to the SI system.")
        dimensional = self.relativity.dimensional_scale
        length = dimensional.length_unit
        mass = dimensional.mass_unit
        time = dimensional.time_unit
        charge = self.charge_unit
        units = {
            "length": length,
            "mass": mass,
            "time": time,
            "charge": charge,
            "velocity": dimensional.velocity_unit,
            "momentum": dimensional.canonical_momentum_unit,
            "energy": self.relativity.energy_unit,
            "current": derived_unit("Q/T", ((charge, 1), (time, -1))),
            "charge_density": derived_unit("Q/L^3", ((charge, 1), (length, -3))),
            "current_density": derived_unit(
                "Q/(T*L^2)", ((charge, 1), (time, -1), (length, -2))
            ),
            "electric_field": derived_unit(
                "M*L/(Q*T^2)", ((mass, 1), (length, 1), (charge, -1), (time, -2))
            ),
            "magnetic_field": derived_unit(
                "M/(Q*T)", ((mass, 1), (charge, -1), (time, -1))
            ),
        }
        return {
            name: (
                float(unit.scale_to_reference),
                _openpmd_unit_dimension(unit.dimension),
            )
            for name, unit in units.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "relativity": self.relativity.to_dict(),
            "charge_unit": self.charge_unit.to_dict(),
            "elementary_charge": _fraction_payload(self.elementary_charge),
            "electron_mass": _fraction_payload(self.electron_mass),
            "vacuum_permittivity": _fraction_payload(self.vacuum_permittivity),
            "constant_set_id": self.constant_set_id,
            "scale_id": self.scale_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ElectromagneticScaleContract:
        if not isinstance(payload, Mapping):
            raise TypeError("Electromagnetic scale payload must be a mapping.")
        expected = {
            "relativity",
            "charge_unit",
            "elementary_charge",
            "electron_mass",
            "vacuum_permittivity",
            "constant_set_id",
            "scale_id",
        }
        if set(payload) != expected:
            raise ValueError(
                "Electromagnetic scale payload must use the canonical fields."
            )
        relativity_payload = payload["relativity"]
        charge_payload = payload["charge_unit"]
        if not isinstance(relativity_payload, Mapping) or not isinstance(
            charge_payload, Mapping
        ):
            raise TypeError("Electromagnetic scale nested payloads must be mappings.")
        constant_set_id = payload["constant_set_id"]
        if not isinstance(constant_set_id, str):
            raise TypeError("constant_set_id payload must be a string.")
        contract = cls(
            RelativityScaleContract.from_dict(relativity_payload),
            UnitDefinition.from_dict(charge_payload),
            _fraction_from_payload(payload["elementary_charge"], "elementary_charge"),
            _fraction_from_payload(payload["electron_mass"], "electron_mass"),
            _fraction_from_payload(
                payload["vacuum_permittivity"],
                "vacuum_permittivity",
            ),
            constant_set_id=constant_set_id,
        )
        claimed_id = payload["scale_id"]
        if not isinstance(claimed_id, str):
            raise TypeError("Electromagnetic scale payload scale_id must be a string.")
        if claimed_id != contract.scale_id:
            raise ValueError(
                "Electromagnetic scale payload fingerprint does not match its content."
            )
        return contract


__all__ = [
    "DimensionalScaleContract",
    "ElectromagneticScaleContract",
    "LengthCoordinateKind",
    "SpatialCoordinateContract",
]
