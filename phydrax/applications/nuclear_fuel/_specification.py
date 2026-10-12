#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""As-fabricated fuel-rod design data (SI units, validated once at construction)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...typing import parse
from ...units import (
    conversion_factor,
    CUBIC_METER,
    derived_unit,
    KELVIN,
    KILOGRAM_PER_CUBIC_METER,
    METER,
    PASCAL,
    UnitDefinition,
)
from ._identity import (
    CladdingAlloy,
    CladdingHeatTreatment,
    FUEL_GAS_SPECIES,
    FuelForm,
)


# Admitted fabrication envelope of the UO2 / Zircaloy profile.
_MINIMUM_DENSITY_FRACTION = 0.8
_MAXIMUM_URANIUM_235_WEIGHT_FRACTION = 0.05
_MINIMUM_OXYGEN_TO_METAL_RATIO = 1.98
_MAXIMUM_OXYGEN_TO_METAL_RATIO = 2.02
# Absolute tolerance on the unit sum of the fill-gas mole fractions.
_MOLE_FRACTION_SUM_TOLERANCE = 1.0e-12


def _si_inputs(
    entries: Mapping[str, tuple[float, UnitDefinition]],
    units: Mapping[str, UnitDefinition] | None,
    /,
) -> dict[str, float]:
    """Convert declared fabrication inputs once, before storing static SI data."""

    supplied = {} if units is None else units
    if not isinstance(supplied, Mapping):
        raise TypeError("units must be a mapping of input names to UnitDefinition.")
    unknown = set(supplied) - set(entries)
    if unknown:
        raise ValueError(f"Units supplied for unknown inputs: {sorted(unknown)}.")
    return {
        name: float(value) * float(conversion_factor(supplied.get(name, target), target))
        for name, (value, target) in entries.items()
    }


def _nonnegative(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number


def _fraction(value: float, name: str, /, *, closed_upper: bool = False) -> float:
    number = float(value)
    upper_ok = number <= 1.0 if closed_upper else number < 1.0
    if not math.isfinite(number) or number < 0.0 or not upper_ok:
        interval = "[0, 1]" if closed_upper else "[0, 1)"
        raise ValueError(f"{name} must be a fraction in {interval}.")
    return number


def _roughness(value: float, /) -> float:
    """Require the strictly positive roughness the gap conductance law needs."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            "roughness must be finite and positive: gap conductance is undefined "
            "for perfectly smooth surfaces."
        )
    return number


@final
class PelletSpecification(StrictModule, NonTrainableState):
    """One as-fabricated fuel pellet design.

    ``density_fraction`` is the sintered density as a fraction of theoretical
    density; ``open_porosity_fraction`` is the connected (open) porosity volume
    fraction of the pellet. ``resintering_density_change`` is the measured
    resintering density increase [kg/m^3] consumed by the densification law.
    ``roughness`` [m] is strictly positive because the gap conductance law takes
    its logarithm.
    """

    form: FuelForm = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    height: float = eqx.field(static=True)
    dish_volume: float = eqx.field(static=True)
    chamfer_volume: float = eqx.field(static=True)
    density_fraction: float = eqx.field(static=True)
    open_porosity_fraction: float = eqx.field(static=True)
    grain_radius: float = eqx.field(static=True)
    roughness: float = eqx.field(static=True)
    uranium_235_weight_fraction: float = eqx.field(static=True)
    oxygen_to_metal_ratio: float = eqx.field(static=True)
    resintering_density_change: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        radius: float,
        height: float,
        density_fraction: float,
        uranium_235_weight_fraction: float,
        grain_radius: float,
        resintering_density_change: float,
        roughness: float,
        dish_volume: float = 0.0,
        chamfer_volume: float = 0.0,
        open_porosity_fraction: float = 0.0,
        oxygen_to_metal_ratio: float = 2.0,
        form: FuelForm = "uo2-pellet",
        units: Mapping[str, UnitDefinition] | None = None,
    ) -> None:
        form_ = parse(form, FuelForm, "form")
        physical = _si_inputs(
            {
                "radius": (radius, METER),
                "height": (height, METER),
                "grain_radius": (grain_radius, METER),
                "roughness": (roughness, METER),
                "dish_volume": (dish_volume, CUBIC_METER),
                "chamfer_volume": (chamfer_volume, CUBIC_METER),
                "resintering_density_change": (
                    resintering_density_change,
                    KILOGRAM_PER_CUBIC_METER,
                ),
            },
            units,
        )
        radius_ = positive_finite_float(physical["radius"], "radius")
        height_ = positive_finite_float(physical["height"], "height")
        dish = _nonnegative(physical["dish_volume"], "dish_volume")
        chamfer = _nonnegative(physical["chamfer_volume"], "chamfer_volume")
        density = float(density_fraction)
        if not math.isfinite(density) or not _MINIMUM_DENSITY_FRACTION <= density <= 1.0:
            raise ValueError(
                f"density_fraction must lie in [{_MINIMUM_DENSITY_FRACTION}, 1.0]."
            )
        open_porosity = _fraction(open_porosity_fraction, "open_porosity_fraction")
        if open_porosity > 1.0 - density:
            raise ValueError("Open porosity cannot exceed total porosity.")
        grain = positive_finite_float(physical["grain_radius"], "grain_radius")
        roughness_ = _roughness(physical["roughness"])
        enrichment = float(uranium_235_weight_fraction)
        if (
            not math.isfinite(enrichment)
            or not 0.0 < enrichment <= _MAXIMUM_URANIUM_235_WEIGHT_FRACTION
        ):
            raise ValueError(
                "uranium_235_weight_fraction must lie in "
                f"(0, {_MAXIMUM_URANIUM_235_WEIGHT_FRACTION}] for the admitted profile."
            )
        ratio = float(oxygen_to_metal_ratio)
        if not (
            math.isfinite(ratio)
            and _MINIMUM_OXYGEN_TO_METAL_RATIO <= ratio <= _MAXIMUM_OXYGEN_TO_METAL_RATIO
        ):
            raise ValueError(
                "oxygen_to_metal_ratio must lie in "
                f"[{_MINIMUM_OXYGEN_TO_METAL_RATIO}, {_MAXIMUM_OXYGEN_TO_METAL_RATIO}]."
            )
        resintering = _nonnegative(
            physical["resintering_density_change"], "resintering_density_change"
        )
        if dish + chamfer >= math.pi * radius_**2 * height_:
            raise ValueError("Dish and chamfer volumes must be smaller than the pellet.")
        self.form = form_
        self.radius = radius_
        self.height = height_
        self.dish_volume = dish
        self.chamfer_volume = chamfer
        self.density_fraction = density
        self.open_porosity_fraction = open_porosity
        self.grain_radius = grain
        self.roughness = roughness_
        self.uranium_235_weight_fraction = enrichment
        self.oxygen_to_metal_ratio = ratio
        self.resintering_density_change = resintering

    @property
    def removed_volume_fraction(self) -> float:
        """Dish plus chamfer volume as a fraction of the right-cylinder volume."""

        return (self.dish_volume + self.chamfer_volume) / (
            math.pi * self.radius**2 * self.height
        )

    def to_record(self, /) -> dict[str, object]:
        return {
            "form": self.form,
            "radius": self.radius,
            "height": self.height,
            "dish_volume": self.dish_volume,
            "chamfer_volume": self.chamfer_volume,
            "density_fraction": self.density_fraction,
            "open_porosity_fraction": self.open_porosity_fraction,
            "grain_radius": self.grain_radius,
            "roughness": self.roughness,
            "uranium_235_weight_fraction": self.uranium_235_weight_fraction,
            "oxygen_to_metal_ratio": self.oxygen_to_metal_ratio,
            "resintering_density_change": self.resintering_density_change,
        }


@final
class CladdingSpecification(StrictModule, NonTrainableState):
    """One as-fabricated cladding tube design.

    ``cold_work`` is a fraction in [0, 1]; ``roughness`` [m] of the inner surface
    is strictly positive, as for the pellet.
    """

    alloy: CladdingAlloy = eqx.field(static=True)
    heat_treatment: CladdingHeatTreatment = eqx.field(static=True)
    inner_radius: float = eqx.field(static=True)
    outer_radius: float = eqx.field(static=True)
    cold_work: float = eqx.field(static=True)
    roughness: float = eqx.field(static=True)
    initial_oxide_thickness: float = eqx.field(static=True)
    initial_hydrogen_ppm: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        inner_radius: float,
        outer_radius: float,
        roughness: float,
        cold_work: float = 0.5,
        initial_oxide_thickness: float = 0.0,
        initial_hydrogen_ppm: float = 0.0,
        alloy: CladdingAlloy = "zircaloy-4",
        heat_treatment: CladdingHeatTreatment = "stress-relief-annealed",
        units: Mapping[str, UnitDefinition] | None = None,
    ) -> None:
        alloy_ = parse(alloy, CladdingAlloy, "alloy")
        treatment = parse(heat_treatment, CladdingHeatTreatment, "heat_treatment")
        physical = _si_inputs(
            {
                "inner_radius": (inner_radius, METER),
                "outer_radius": (outer_radius, METER),
                "roughness": (roughness, METER),
                "initial_oxide_thickness": (initial_oxide_thickness, METER),
            },
            units,
        )
        inner = positive_finite_float(physical["inner_radius"], "inner_radius")
        outer = positive_finite_float(physical["outer_radius"], "outer_radius")
        if outer <= inner:
            raise ValueError("outer_radius must exceed inner_radius.")
        work = _fraction(cold_work, "cold_work", closed_upper=True)
        roughness_ = _roughness(physical["roughness"])
        oxide = _nonnegative(
            physical["initial_oxide_thickness"], "initial_oxide_thickness"
        )
        if oxide >= outer - inner:
            raise ValueError("Initial oxide cannot consume the cladding wall.")
        hydrogen = _nonnegative(initial_hydrogen_ppm, "initial_hydrogen_ppm")
        self.alloy = alloy_
        self.heat_treatment = treatment
        self.inner_radius = inner
        self.outer_radius = outer
        self.cold_work = work
        self.roughness = roughness_
        self.initial_oxide_thickness = oxide
        self.initial_hydrogen_ppm = hydrogen

    def to_record(self, /) -> dict[str, object]:
        return {
            "alloy": self.alloy,
            "heat_treatment": self.heat_treatment,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "cold_work": self.cold_work,
            "roughness": self.roughness,
            "initial_oxide_thickness": self.initial_oxide_thickness,
            "initial_hydrogen_ppm": self.initial_hydrogen_ppm,
        }


@final
class RodSpecification(StrictModule, NonTrainableState):
    """Stack, plenum, spring, and fill-gas design of one rod.

    ``fill_gas_mole_fractions`` follow ``FUEL_GAS_SPECIES`` order. The fill
    inventory is fixed by ``fill_pressure`` at ``fill_temperature`` in the
    as-fabricated free volume.
    """

    stack_length: float = eqx.field(static=True)
    plenum_length: float = eqx.field(static=True)
    spring_volume: float = eqx.field(static=True)
    spring_preload: float = eqx.field(static=True)
    spring_stiffness: float = eqx.field(static=True)
    fill_pressure: float = eqx.field(static=True)
    fill_temperature: float = eqx.field(static=True)
    fill_gas_mole_fractions: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        stack_length: float,
        plenum_length: float,
        fill_pressure: float,
        fill_temperature: float = 293.15,
        fill_gas_mole_fractions: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 0.0),
        spring_volume: float = 0.0,
        spring_preload: float = 0.0,
        spring_stiffness: float = 0.0,
        units: Mapping[str, UnitDefinition] | None = None,
    ) -> None:
        force_unit = derived_unit("N", ((PASCAL, 1), (METER, 2)))
        stiffness_unit = derived_unit("N/m", ((PASCAL, 1), (METER, 1)))
        physical = _si_inputs(
            {
                "stack_length": (stack_length, METER),
                "plenum_length": (plenum_length, METER),
                "spring_volume": (spring_volume, CUBIC_METER),
                "spring_preload": (spring_preload, force_unit),
                "spring_stiffness": (spring_stiffness, stiffness_unit),
                "fill_pressure": (fill_pressure, PASCAL),
                "fill_temperature": (fill_temperature, KELVIN),
            },
            units,
        )
        stack = positive_finite_float(physical["stack_length"], "stack_length")
        plenum = positive_finite_float(physical["plenum_length"], "plenum_length")
        spring = _nonnegative(physical["spring_volume"], "spring_volume")
        preload = _nonnegative(physical["spring_preload"], "spring_preload")
        stiffness = _nonnegative(physical["spring_stiffness"], "spring_stiffness")
        pressure = positive_finite_float(physical["fill_pressure"], "fill_pressure")
        temperature = positive_finite_float(
            physical["fill_temperature"], "fill_temperature"
        )
        fractions = np.asarray(fill_gas_mole_fractions, dtype=np.float64)
        if (
            fractions.shape != (len(FUEL_GAS_SPECIES),)
            or not np.all(np.isfinite(fractions))
            or np.any(fractions < 0.0)
            or not math.isclose(
                float(np.sum(fractions)), 1.0, abs_tol=_MOLE_FRACTION_SUM_TOLERANCE
            )
        ):
            raise ValueError(
                "fill_gas_mole_fractions must be nonnegative, sum to one, and follow "
                f"{FUEL_GAS_SPECIES}."
            )
        self.fill_gas_mole_fractions = tuple(float(value) for value in fractions)
        self.stack_length = stack
        self.plenum_length = plenum
        self.spring_volume = spring
        self.spring_preload = preload
        self.spring_stiffness = stiffness
        self.fill_pressure = pressure
        self.fill_temperature = temperature

    def to_record(self, /) -> dict[str, object]:
        return {
            "stack_length": self.stack_length,
            "plenum_length": self.plenum_length,
            "spring_volume": self.spring_volume,
            "spring_preload": self.spring_preload,
            "spring_stiffness": self.spring_stiffness,
            "fill_pressure": self.fill_pressure,
            "fill_temperature": self.fill_temperature,
            "fill_gas_mole_fractions": list(self.fill_gas_mole_fractions),
        }


class FuelRodSpecification(StrictModule, NonTrainableState):
    """Complete as-fabricated design of one fuel rod."""

    pellet: PelletSpecification
    cladding: CladdingSpecification
    rod: RodSpecification
    specification_id: str = eqx.field(static=True)

    def __init__(
        self,
        pellet: PelletSpecification,
        cladding: CladdingSpecification,
        rod: RodSpecification,
        /,
    ) -> None:
        if not isinstance(pellet, PelletSpecification):
            raise TypeError("pellet must be PelletSpecification.")
        if not isinstance(cladding, CladdingSpecification):
            raise TypeError("cladding must be CladdingSpecification.")
        if not isinstance(rod, RodSpecification):
            raise TypeError("rod must be RodSpecification.")
        if cladding.inner_radius <= pellet.radius:
            raise ValueError("The as-fabricated pellet-cladding gap must be open.")
        self.pellet = pellet
        self.cladding = cladding
        self.rod = rod
        self.specification_id = canonical_fingerprint(
            {
                "kind": "fuel-rod-specification",
                "pellet": pellet.to_record(),
                "cladding": cladding.to_record(),
                "rod": rod.to_record(),
            }
        )

    @property
    def cold_gap(self) -> float:
        """As-fabricated radial gap width [m]."""

        return self.cladding.inner_radius - self.pellet.radius


__all__ = [
    "CladdingSpecification",
    "FuelRodSpecification",
    "PelletSpecification",
    "RodSpecification",
]
