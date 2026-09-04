#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from ...units import (
    AMPERE,
    conversion_factor as _conversion_factor,
    CUBIC_METER,
    derived_unit,
    FARAD,
    JOULE,
    KILOGRAM,
    KILOPASCAL,
    METER,
    MICROAMPERE,
    MICROFARAD,
    MILLIGRAM,
    MILLIMETER,
    MILLIMOLAR,
    MILLISECOND,
    MILLISIEMENS,
    MILLIVOLT,
    MOLE,
    MOLE_PER_CUBIC_METER,
    ONE,
    PASCAL,
    SECOND,
    SI_REFERENCE_SYSTEM_ID,
    SIEMENS,
    UnitDefinition,
    VOLT,
)
from .._quantity_contract import canonical_quantity_text, resolve_application_quantity


_MILLIMOLE = UnitDefinition("mmol", MOLE.dimension, SI_REFERENCE_SYSTEM_ID, "1e-3")
_SQUARE_METER = derived_unit("m2", ((METER, 2),))
_SQUARE_MILLIMETER = derived_unit("mm2", ((MILLIMETER, 2),))
_CUBIC_MILLIMETER = derived_unit("mm3", ((MILLIMETER, 3),))
_VOLT_PER_METER = derived_unit("V/m", ((VOLT, 1), (METER, -1)))
_MILLIVOLT_PER_MILLIMETER = derived_unit("mV/mm", ((MILLIVOLT, 1), (MILLIMETER, -1)))
_AMPERE_PER_SQUARE_METER = derived_unit("A/m2", ((AMPERE, 1), (METER, -2)))
_MICROAMPERE_PER_SQUARE_MILLIMETER = derived_unit(
    "uA/mm2", ((MICROAMPERE, 1), (MILLIMETER, -2))
)
_FARAD_PER_SQUARE_METER = derived_unit("F/m2", ((FARAD, 1), (METER, -2)))
_MICROFARAD_PER_SQUARE_MILLIMETER = derived_unit(
    "uF/mm2", ((MICROFARAD, 1), (MILLIMETER, -2))
)
_SIEMENS_PER_METER = derived_unit("S/m", ((SIEMENS, 1), (METER, -1)))
_MILLISIEMENS_PER_MILLIMETER = derived_unit(
    "mS/mm", ((MILLISIEMENS, 1), (MILLIMETER, -1))
)
_SQUARE_METER_PER_SECOND = derived_unit("m2/s", ((METER, 2), (SECOND, -1)))
_SQUARE_MILLIMETER_PER_MILLISECOND = derived_unit(
    "mm2/ms", ((MILLIMETER, 2), (MILLISECOND, -1))
)
_CONCENTRATION_PER_SECOND = derived_unit(
    "mol/(m3*s)", ((MOLE_PER_CUBIC_METER, 1), (SECOND, -1))
)
_MILLIMOLAR_PER_MILLISECOND = derived_unit("mM/ms", ((MILLIMOLAR, 1), (MILLISECOND, -1)))
_MOLE_PER_SQUARE_METER_SECOND = derived_unit(
    "mol/(m2*s)", ((MOLE, 1), (METER, -2), (SECOND, -1))
)
_MILLIMOLE_PER_SQUARE_MILLIMETER_MILLISECOND = derived_unit(
    "mmol/(mm2*ms)",
    ((_MILLIMOLE, 1), (MILLIMETER, -2), (MILLISECOND, -1)),
)
_METER_PER_SECOND = derived_unit("m/s", ((METER, 1), (SECOND, -1)))
_MILLIMETER_PER_MILLISECOND = derived_unit("mm/ms", ((MILLIMETER, 1), (MILLISECOND, -1)))
_METER_PER_SECOND_SQUARED = derived_unit("m/s2", ((METER, 1), (SECOND, -2)))
_MILLIMETER_PER_MILLISECOND_SQUARED = derived_unit(
    "mm/ms2", ((MILLIMETER, 1), (MILLISECOND, -2))
)
_KILOGRAM_PER_CUBIC_METER = derived_unit("kg/m3", ((KILOGRAM, 1), (METER, -3)))
_MILLIGRAM_PER_CUBIC_MILLIMETER = derived_unit(
    "mg/mm3", ((MILLIGRAM, 1), (MILLIMETER, -3))
)
_NEWTON = derived_unit("N", ((KILOGRAM, 1), (METER, 1), (SECOND, -2)))
_MILLIGRAM_MILLIMETER_PER_MILLISECOND_SQUARED = derived_unit(
    "mg*mm/ms2", ((MILLIGRAM, 1), (MILLIMETER, 1), (MILLISECOND, -2))
)
_PER_SECOND = derived_unit("1/s", ((ONE, 1), (SECOND, -1)))
_PER_MILLISECOND = derived_unit("1/ms", ((ONE, 1), (MILLISECOND, -1)))
_MILLIGRAM_MILLIMETER_SQUARED_PER_MILLISECOND_SQUARED = derived_unit(
    "mg*mm2/ms2", ((MILLIGRAM, 1), (MILLIMETER, 2), (MILLISECOND, -2))
)
_WATT = derived_unit("W", ((JOULE, 1), (SECOND, -1)))
_MILLIGRAM_MILLIMETER_SQUARED_PER_MILLISECOND_CUBED = derived_unit(
    "mg*mm2/ms3", ((MILLIGRAM, 1), (MILLIMETER, 2), (MILLISECOND, -3))
)
_PASCAL_SECOND = derived_unit("Pa*s", ((PASCAL, 1), (SECOND, 1)))
_KILOPASCAL_MILLISECOND = derived_unit("kPa*ms", ((KILOPASCAL, 1), (MILLISECOND, 1)))
_CUBIC_METER_PER_SECOND = derived_unit("m3/s", ((CUBIC_METER, 1), (SECOND, -1)))
_CUBIC_MILLIMETER_PER_MILLISECOND = derived_unit(
    "mm3/ms", ((MILLIMETER, 3), (MILLISECOND, -1))
)
_PASCAL_SECOND_PER_CUBIC_METER = derived_unit(
    "Pa*s/m3", ((PASCAL, 1), (SECOND, 1), (METER, -3))
)
_KILOPASCAL_MILLISECOND_PER_CUBIC_MILLIMETER = derived_unit(
    "kPa*ms/mm3", ((KILOPASCAL, 1), (MILLISECOND, 1), (MILLIMETER, -3))
)
_PASCAL_SECOND_SQUARED_PER_CUBIC_METER = derived_unit(
    "Pa*s2/m3", ((PASCAL, 1), (SECOND, 2), (METER, -3))
)
_KILOPASCAL_MILLISECOND_SQUARED_PER_CUBIC_MILLIMETER = derived_unit(
    "kPa*ms2/mm3",
    ((KILOPASCAL, 1), (MILLISECOND, 2), (MILLIMETER, -3)),
)
_CUBIC_METER_PER_PASCAL = derived_unit("m3/Pa", ((METER, 3), (PASCAL, -1)))
_CUBIC_MILLIMETER_PER_KILOPASCAL = derived_unit(
    "mm3/kPa", ((MILLIMETER, 3), (KILOPASCAL, -1))
)
_PASCAL_PER_CUBIC_METER = derived_unit("Pa/m3", ((PASCAL, 1), (METER, -3)))
_KILOPASCAL_PER_CUBIC_MILLIMETER = derived_unit(
    "kPa/mm3", ((KILOPASCAL, 1), (MILLIMETER, -3))
)

_REFERENCE_UNIT_BY_KIND = MappingProxyType(
    {
        "time": SECOND,
        "length": METER,
        "area": _SQUARE_METER,
        "volume": CUBIC_METER,
        "mass": KILOGRAM,
        "electric_potential": VOLT,
        "electric_field": _VOLT_PER_METER,
        "electric_current": AMPERE,
        "surface_current_density": _AMPERE_PER_SQUARE_METER,
        "capacitance": FARAD,
        "surface_capacitance_density": _FARAD_PER_SQUARE_METER,
        "electrical_conductivity": _SIEMENS_PER_METER,
        "amount_of_substance": MOLE,
        "amount_concentration": MOLE_PER_CUBIC_METER,
        "chemical_diffusivity": _SQUARE_METER_PER_SECOND,
        "concentration_rate": _CONCENTRATION_PER_SECOND,
        "molar_surface_flux": _MOLE_PER_SQUARE_METER_SECOND,
        "pressure": PASCAL,
        "velocity": _METER_PER_SECOND,
        "acceleration": _METER_PER_SECOND_SQUARED,
        "mass_density": _KILOGRAM_PER_CUBIC_METER,
        "force": _NEWTON,
        "strain": ONE,
        "strain_rate": _PER_SECOND,
        "energy": JOULE,
        "power": _WATT,
        "dynamic_viscosity": _PASCAL_SECOND,
        "volumetric_flow_rate": _CUBIC_METER_PER_SECOND,
        "hydraulic_resistance": _PASCAL_SECOND_PER_CUBIC_METER,
        "hydraulic_inertance": _PASCAL_SECOND_SQUARED_PER_CUBIC_METER,
        "hydraulic_compliance": _CUBIC_METER_PER_PASCAL,
        "hydraulic_elastance": _PASCAL_PER_CUBIC_METER,
    }
)


@dataclass(frozen=True, slots=True, init=False)
class CardiovascularQuantitySpec:
    """Immutable physical meaning and exact unit for one kernel quantity."""

    name: str
    quantity_kind: str
    unit: UnitDefinition
    axes: tuple[str, ...]
    sign_convention: str
    support_association: str
    reference_configuration: str
    quantity_id: str = field(init=False)

    def __init__(
        self,
        name: str,
        quantity_kind: str,
        unit: UnitDefinition,
        axes: tuple[str, ...] = (),
        sign_convention: str = "",
        support_association: str = "",
        reference_configuration: str = "",
    ):
        resolved = resolve_application_quantity(
            domain="cardiovascular",
            reference_units=_REFERENCE_UNIT_BY_KIND,
            name=name,
            quantity_kind=quantity_kind,
            unit=unit,
            axes=axes,
            sign_convention=sign_convention,
            support_association=support_association,
            reference_configuration=reference_configuration,
        )
        object.__setattr__(self, "name", resolved.name)
        object.__setattr__(self, "quantity_kind", resolved.quantity_kind)
        object.__setattr__(self, "unit", resolved.unit)
        object.__setattr__(self, "axes", resolved.axes)
        object.__setattr__(self, "sign_convention", resolved.sign_convention)
        object.__setattr__(self, "support_association", resolved.support_association)
        object.__setattr__(
            self, "reference_configuration", resolved.reference_configuration
        )
        object.__setattr__(self, "quantity_id", resolved.quantity_id)

    @property
    def spec_id(self) -> str:
        """Alias used by runtime manifests that bind quantity specification IDs."""
        return self.quantity_id

    @property
    def kernel_unit(self) -> str:
        """Display symbol for the stored kernel unit definition."""
        return self.unit.symbol

    @property
    def reference_unit(self) -> UnitDefinition:
        """Canonical SI reference unit for this quantity kind."""
        return _REFERENCE_UNIT_BY_KIND[self.quantity_kind]

    @property
    def si_unit(self) -> str:
        """Display symbol for the canonical SI reference unit."""
        return self.reference_unit.symbol

    @property
    def si_factor(self) -> Fraction:
        """Exact multiplier from the kernel unit to the SI reference unit."""
        return _conversion_factor(self.unit, self.reference_unit)

    def to_si(self, value: Any, /) -> Any:
        """Convert a scalar or array from the declared kernel unit to SI."""
        factor = self.si_factor
        return value * factor.numerator / factor.denominator

    def from_si(self, value: Any, /) -> Any:
        """Convert a scalar or array from SI to the declared kernel unit."""
        factor = self.si_factor
        return value * factor.denominator / factor.numerator


def _quantity(
    name: str,
    quantity_kind: str,
    unit: UnitDefinition,
    *,
    axes: tuple[str, ...] = (),
    sign: str,
    support: str,
    reference: str,
) -> CardiovascularQuantitySpec:
    return CardiovascularQuantitySpec(
        name,
        quantity_kind,
        unit,
        axes=axes,
        sign_convention=sign,
        support_association=support,
        reference_configuration=reference,
    )


_SCALAR_SIGN = "positive magnitude unless the owning model declares an oriented flux"
_CURRENT_SIGN = "positive outward across the cell membrane"
_SPATIAL_REFERENCE = (
    "right-handed anatomy spatial frame in the manifest reference configuration"
)
_MATERIAL_REFERENCE = "stress and strain measure named by the owning constitutive model"
_LUMPED_REFERENCE = (
    "positive pressure and flow follow the owning terminal-port orientation"
)


_SPECS = (
    _quantity(
        "time",
        "time",
        MILLISECOND,
        sign="increases forward",
        support="global",
        reference="protocol time origin",
    ),
    _quantity(
        "length",
        "length",
        MILLIMETER,
        axes=("x", "y", "z"),
        sign=_SCALAR_SIGN,
        support="geometry coordinates",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "area",
        "area",
        _SQUARE_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="surface or cross-section",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "volume",
        "volume",
        _CUBIC_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="cell, chamber, or control volume",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "mass",
        "mass",
        MILLIGRAM,
        sign=_SCALAR_SIGN,
        support="material or control volume",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "transmembrane_potential",
        "electric_potential",
        MILLIVOLT,
        sign="intracellular minus extracellular potential",
        support="cardiac cell or nodal scalar",
        reference="extracellular electric potential",
    ),
    _quantity(
        "electric_field",
        "electric_field",
        _MILLIVOLT_PER_MILLIMETER,
        axes=("x", "y", "z"),
        sign="negative spatial gradient of electric potential",
        support="cell or quadrature vector",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "membrane_current",
        "electric_current",
        MICROAMPERE,
        sign=_CURRENT_SIGN,
        support="cardiac cell",
        reference="cell membrane orientation",
    ),
    _quantity(
        "membrane_current_density",
        "surface_current_density",
        _MICROAMPERE_PER_SQUARE_MILLIMETER,
        sign=_CURRENT_SIGN,
        support="membrane surface or homogenized tissue",
        reference="cell membrane orientation",
    ),
    _quantity(
        "membrane_capacitance",
        "capacitance",
        MICROFARAD,
        sign=_SCALAR_SIGN,
        support="cardiac cell",
        reference="cell membrane",
    ),
    _quantity(
        "membrane_capacitance_density",
        "surface_capacitance_density",
        _MICROFARAD_PER_SQUARE_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="membrane surface or homogenized tissue",
        reference="cell membrane",
    ),
    _quantity(
        "electrical_conductivity",
        "electrical_conductivity",
        _MILLISIEMENS_PER_MILLIMETER,
        axes=("component_i", "component_j"),
        sign="positive-semidefinite constitutive tensor",
        support="cell or quadrature tensor",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "chemical_amount",
        "amount_of_substance",
        _MILLIMOLE,
        sign=_SCALAR_SIGN,
        support="cardiac cell or control volume",
        reference="named chemical species",
    ),
    _quantity(
        "species_concentration",
        "amount_concentration",
        MILLIMOLAR,
        sign=_SCALAR_SIGN,
        support="cardiac cell, node, or control volume",
        reference="named compartment and chemical species",
    ),
    _quantity(
        "chemical_diffusivity",
        "chemical_diffusivity",
        _SQUARE_MILLIMETER_PER_MILLISECOND,
        axes=("component_i", "component_j"),
        sign="positive-semidefinite constitutive tensor",
        support="cell or quadrature tensor",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "concentration_rate",
        "concentration_rate",
        _MILLIMOLAR_PER_MILLISECOND,
        sign="positive produces the named species",
        support="cardiac cell or control volume",
        reference="named compartment and chemical species",
    ),
    _quantity(
        "molar_surface_flux",
        "molar_surface_flux",
        _MILLIMOLE_PER_SQUARE_MILLIMETER_MILLISECOND,
        sign="positive along the owning surface normal",
        support="oriented surface",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "pressure",
        "pressure",
        KILOPASCAL,
        sign="positive in compression",
        support="chamber, vessel node, cell, or quadrature scalar",
        reference="absolute or gauge reference named by the owning model",
    ),
    _quantity(
        "velocity",
        "velocity",
        _MILLIMETER_PER_MILLISECOND,
        axes=("x", "y", "z"),
        sign="positive along the corresponding spatial axis",
        support="node, cell, or quadrature vector",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "acceleration",
        "acceleration",
        _MILLIMETER_PER_MILLISECOND_SQUARED,
        axes=("x", "y", "z"),
        sign="positive along the corresponding spatial axis",
        support="node, cell, or quadrature vector",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "mass_density",
        "mass_density",
        _MILLIGRAM_PER_CUBIC_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="cell or quadrature scalar",
        reference="current or reference volume named by the owning model",
    ),
    _quantity(
        "force",
        "force",
        _MILLIGRAM_MILLIMETER_PER_MILLISECOND_SQUARED,
        axes=("x", "y", "z"),
        sign="positive along the corresponding spatial axis",
        support="node, surface, or body resultant",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "stress",
        "pressure",
        KILOPASCAL,
        axes=("component_i", "component_j"),
        sign="tension-positive tensor components; pressure remains compression-positive",
        support="cell or quadrature tensor",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "strain",
        "strain",
        ONE,
        axes=("component_i", "component_j"),
        sign="extension-positive tensor components",
        support="cell or quadrature tensor",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "strain_rate",
        "strain_rate",
        _PER_MILLISECOND,
        axes=("component_i", "component_j"),
        sign="extension-rate-positive tensor components",
        support="cell or quadrature tensor",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "energy",
        "energy",
        _MILLIGRAM_MILLIMETER_SQUARED_PER_MILLISECOND_SQUARED,
        sign=_SCALAR_SIGN,
        support="global, chamber, cell, or quadrature scalar",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "power",
        "power",
        _MILLIGRAM_MILLIMETER_SQUARED_PER_MILLISECOND_CUBED,
        sign="positive power is supplied to the modeled system",
        support="global, chamber, or cell scalar",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "dynamic_viscosity",
        "dynamic_viscosity",
        _KILOPASCAL_MILLISECOND,
        sign=_SCALAR_SIGN,
        support="fluid cell or quadrature scalar",
        reference="current fluid configuration",
    ),
    _quantity(
        "volumetric_flow_rate",
        "volumetric_flow_rate",
        _CUBIC_MILLIMETER_PER_MILLISECOND,
        sign="positive along the owning terminal-port orientation",
        support="oriented terminal port or vessel segment",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_resistance",
        "hydraulic_resistance",
        _KILOPASCAL_MILLISECOND_PER_CUBIC_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="lumped vessel or terminal relation",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_inertance",
        "hydraulic_inertance",
        _KILOPASCAL_MILLISECOND_SQUARED_PER_CUBIC_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="lumped vessel or terminal relation",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_compliance",
        "hydraulic_compliance",
        _CUBIC_MILLIMETER_PER_KILOPASCAL,
        sign=_SCALAR_SIGN,
        support="lumped chamber or vessel compartment",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_elastance",
        "hydraulic_elastance",
        _KILOPASCAL_PER_CUBIC_MILLIMETER,
        sign=_SCALAR_SIGN,
        support="lumped chamber or vessel compartment",
        reference=_LUMPED_REFERENCE,
    ),
)

if len({spec.name for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("Canonical cardiovascular quantity names must be unique.")
if len({spec.quantity_id for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("Canonical cardiovascular quantity content IDs must be unique.")

CARDIOVASCULAR_QUANTITIES = MappingProxyType({spec.name: spec for spec in _SPECS})


def cardiovascular_quantity(name: str, /) -> CardiovascularQuantitySpec:
    """Return one canonical application quantity by stable name."""
    name_ = canonical_quantity_text(name, "name")
    if name_ not in CARDIOVASCULAR_QUANTITIES:
        raise KeyError(f"Unknown cardiovascular quantity {name_!r}.")
    return CARDIOVASCULAR_QUANTITIES[name_]


__all__ = [
    "CARDIOVASCULAR_QUANTITIES",
    "CardiovascularQuantitySpec",
    "cardiovascular_quantity",
]
