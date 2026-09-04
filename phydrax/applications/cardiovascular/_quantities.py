#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from .._quantity_contract import (
    canonical_quantity_text,
    resolve_application_quantity,
)


# These are the application kernel's supported conversions, not a general-purpose
# unit registry. Fractions retain the exact decimal scale used in content identity.
_SUPPORTED_UNIT_CONVERSIONS = MappingProxyType(
    {
        ("time", "ms", "s"): Fraction(1, 1_000),
        ("length", "mm", "m"): Fraction(1, 1_000),
        ("area", "mm2", "m2"): Fraction(1, 1_000_000),
        ("volume", "mm3", "m3"): Fraction(1, 1_000_000_000),
        ("mass", "mg", "kg"): Fraction(1, 1_000_000),
        ("electric_potential", "mV", "V"): Fraction(1, 1_000),
        ("electric_field", "mV/mm", "V/m"): Fraction(1, 1),
        ("electric_current", "uA", "A"): Fraction(1, 1_000_000),
        ("surface_current_density", "uA/mm2", "A/m2"): Fraction(1, 1),
        ("capacitance", "uF", "F"): Fraction(1, 1_000_000),
        ("surface_capacitance_density", "uF/mm2", "F/m2"): Fraction(1, 1),
        ("electrical_conductivity", "mS/mm", "S/m"): Fraction(1, 1),
        ("amount_of_substance", "mmol", "mol"): Fraction(1, 1_000),
        ("amount_concentration", "mM", "mol/m3"): Fraction(1, 1),
        ("chemical_diffusivity", "mm2/ms", "m2/s"): Fraction(1, 1_000),
        ("concentration_rate", "mM/ms", "mol/(m3*s)"): Fraction(1_000, 1),
        (
            "molar_surface_flux",
            "mmol/(mm2*ms)",
            "mol/(m2*s)",
        ): Fraction(1_000_000, 1),
        ("pressure", "kPa", "Pa"): Fraction(1_000, 1),
        ("velocity", "mm/ms", "m/s"): Fraction(1, 1),
        ("acceleration", "mm/ms2", "m/s2"): Fraction(1_000, 1),
        ("mass_density", "mg/mm3", "kg/m3"): Fraction(1_000, 1),
        ("force", "mg*mm/ms2", "N"): Fraction(1, 1_000),
        ("strain", "1", "1"): Fraction(1, 1),
        ("strain_rate", "1/ms", "1/s"): Fraction(1_000, 1),
        ("energy", "mg*mm2/ms2", "J"): Fraction(1, 1_000_000),
        ("power", "mg*mm2/ms3", "W"): Fraction(1, 1_000),
        ("dynamic_viscosity", "kPa*ms", "Pa*s"): Fraction(1, 1),
        ("volumetric_flow_rate", "mm3/ms", "m3/s"): Fraction(1, 1_000_000),
        (
            "hydraulic_resistance",
            "kPa*ms/mm3",
            "Pa*s/m3",
        ): Fraction(1_000_000_000, 1),
        (
            "hydraulic_inertance",
            "kPa*ms2/mm3",
            "Pa*s2/m3",
        ): Fraction(1_000_000, 1),
        ("hydraulic_compliance", "mm3/kPa", "m3/Pa"): Fraction(1, 1_000_000_000_000),
        ("hydraulic_elastance", "kPa/mm3", "Pa/m3"): Fraction(1_000_000_000_000, 1),
    }
)




@dataclass(frozen=True, slots=True, init=False)
class CardiovascularQuantitySpec:
    """Immutable physical meaning and exact SI scale for one kernel quantity."""

    name: str
    physical_dimension: str
    kernel_unit: str
    si_unit: str
    si_factor: Fraction
    axes: tuple[str, ...]
    sign_convention: str
    support_association: str
    reference_configuration: str
    quantity_id: str = field(init=False)

    def __init__(
        self,
        name: str,
        physical_dimension: str,
        kernel_unit: str,
        si_unit: str,
        si_factor: Any,
        axes: tuple[str, ...] = (),
        sign_convention: str = "",
        support_association: str = "",
        reference_configuration: str = "",
    ):
        resolved = resolve_application_quantity(
            domain="cardiovascular",
            supported_conversions=_SUPPORTED_UNIT_CONVERSIONS,
            name=name,
            physical_dimension=physical_dimension,
            kernel_unit=kernel_unit,
            si_unit=si_unit,
            si_factor=si_factor,
            axes=axes,
            sign_convention=sign_convention,
            support_association=support_association,
            reference_configuration=reference_configuration,
        )
        object.__setattr__(self, "name", resolved.name)
        object.__setattr__(self, "physical_dimension", resolved.physical_dimension)
        object.__setattr__(self, "kernel_unit", resolved.kernel_unit)
        object.__setattr__(self, "si_unit", resolved.si_unit)
        object.__setattr__(self, "si_factor", resolved.si_factor)
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

    def to_si(self, value: Any, /) -> Any:
        """Convert a scalar or array from the declared kernel unit to SI."""
        return value * self.si_factor.numerator / self.si_factor.denominator

    def from_si(self, value: Any, /) -> Any:
        """Convert a scalar or array from SI to the declared kernel unit."""
        return value * self.si_factor.denominator / self.si_factor.numerator


def _quantity(
    name: str,
    dimension: str,
    kernel_unit: str,
    si_unit: str,
    *,
    axes: tuple[str, ...] = (),
    sign: str,
    support: str,
    reference: str,
) -> CardiovascularQuantitySpec:
    return CardiovascularQuantitySpec(
        name,
        dimension,
        kernel_unit,
        si_unit,
        _SUPPORTED_UNIT_CONVERSIONS[(dimension, kernel_unit, si_unit)],
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
        "ms",
        "s",
        sign="increases forward",
        support="global",
        reference="protocol time origin",
    ),
    _quantity(
        "length",
        "length",
        "mm",
        "m",
        axes=("x", "y", "z"),
        sign=_SCALAR_SIGN,
        support="geometry coordinates",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "area",
        "area",
        "mm2",
        "m2",
        sign=_SCALAR_SIGN,
        support="surface or cross-section",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "volume",
        "volume",
        "mm3",
        "m3",
        sign=_SCALAR_SIGN,
        support="cell, chamber, or control volume",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "mass",
        "mass",
        "mg",
        "kg",
        sign=_SCALAR_SIGN,
        support="material or control volume",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "transmembrane_potential",
        "electric_potential",
        "mV",
        "V",
        sign="intracellular minus extracellular potential",
        support="cardiac cell or nodal scalar",
        reference="extracellular electric potential",
    ),
    _quantity(
        "electric_field",
        "electric_field",
        "mV/mm",
        "V/m",
        axes=("x", "y", "z"),
        sign="negative spatial gradient of electric potential",
        support="cell or quadrature vector",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "membrane_current",
        "electric_current",
        "uA",
        "A",
        sign=_CURRENT_SIGN,
        support="cardiac cell",
        reference="cell membrane orientation",
    ),
    _quantity(
        "membrane_current_density",
        "surface_current_density",
        "uA/mm2",
        "A/m2",
        sign=_CURRENT_SIGN,
        support="membrane surface or homogenized tissue",
        reference="cell membrane orientation",
    ),
    _quantity(
        "membrane_capacitance",
        "capacitance",
        "uF",
        "F",
        sign=_SCALAR_SIGN,
        support="cardiac cell",
        reference="cell membrane",
    ),
    _quantity(
        "membrane_capacitance_density",
        "surface_capacitance_density",
        "uF/mm2",
        "F/m2",
        sign=_SCALAR_SIGN,
        support="membrane surface or homogenized tissue",
        reference="cell membrane",
    ),
    _quantity(
        "electrical_conductivity",
        "electrical_conductivity",
        "mS/mm",
        "S/m",
        axes=("component_i", "component_j"),
        sign="positive-semidefinite constitutive tensor",
        support="cell or quadrature tensor",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "chemical_amount",
        "amount_of_substance",
        "mmol",
        "mol",
        sign=_SCALAR_SIGN,
        support="cardiac cell or control volume",
        reference="named chemical species",
    ),
    _quantity(
        "species_concentration",
        "amount_concentration",
        "mM",
        "mol/m3",
        sign=_SCALAR_SIGN,
        support="cardiac cell, node, or control volume",
        reference="named compartment and chemical species",
    ),
    _quantity(
        "chemical_diffusivity",
        "chemical_diffusivity",
        "mm2/ms",
        "m2/s",
        axes=("component_i", "component_j"),
        sign="positive-semidefinite constitutive tensor",
        support="cell or quadrature tensor",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "concentration_rate",
        "concentration_rate",
        "mM/ms",
        "mol/(m3*s)",
        sign="positive produces the named species",
        support="cardiac cell or control volume",
        reference="named compartment and chemical species",
    ),
    _quantity(
        "molar_surface_flux",
        "molar_surface_flux",
        "mmol/(mm2*ms)",
        "mol/(m2*s)",
        sign="positive along the owning surface normal",
        support="oriented surface",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "pressure",
        "pressure",
        "kPa",
        "Pa",
        sign="positive in compression",
        support="chamber, vessel node, cell, or quadrature scalar",
        reference="absolute or gauge reference named by the owning model",
    ),
    _quantity(
        "velocity",
        "velocity",
        "mm/ms",
        "m/s",
        axes=("x", "y", "z"),
        sign="positive along the corresponding spatial axis",
        support="node, cell, or quadrature vector",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "acceleration",
        "acceleration",
        "mm/ms2",
        "m/s2",
        axes=("x", "y", "z"),
        sign="positive along the corresponding spatial axis",
        support="node, cell, or quadrature vector",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "mass_density",
        "mass_density",
        "mg/mm3",
        "kg/m3",
        sign=_SCALAR_SIGN,
        support="cell or quadrature scalar",
        reference="current or reference volume named by the owning model",
    ),
    _quantity(
        "force",
        "force",
        "mg*mm/ms2",
        "N",
        axes=("x", "y", "z"),
        sign="positive along the corresponding spatial axis",
        support="node, surface, or body resultant",
        reference=_SPATIAL_REFERENCE,
    ),
    _quantity(
        "stress",
        "pressure",
        "kPa",
        "Pa",
        axes=("component_i", "component_j"),
        sign="tension-positive tensor components; pressure remains compression-positive",
        support="cell or quadrature tensor",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "strain",
        "strain",
        "1",
        "1",
        axes=("component_i", "component_j"),
        sign="extension-positive tensor components",
        support="cell or quadrature tensor",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "strain_rate",
        "strain_rate",
        "1/ms",
        "1/s",
        axes=("component_i", "component_j"),
        sign="extension-rate-positive tensor components",
        support="cell or quadrature tensor",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "energy",
        "energy",
        "mg*mm2/ms2",
        "J",
        sign=_SCALAR_SIGN,
        support="global, chamber, cell, or quadrature scalar",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "power",
        "power",
        "mg*mm2/ms3",
        "W",
        sign="positive power is supplied to the modeled system",
        support="global, chamber, or cell scalar",
        reference=_MATERIAL_REFERENCE,
    ),
    _quantity(
        "dynamic_viscosity",
        "dynamic_viscosity",
        "kPa*ms",
        "Pa*s",
        sign=_SCALAR_SIGN,
        support="fluid cell or quadrature scalar",
        reference="current fluid configuration",
    ),
    _quantity(
        "volumetric_flow_rate",
        "volumetric_flow_rate",
        "mm3/ms",
        "m3/s",
        sign="positive along the owning terminal-port orientation",
        support="oriented terminal port or vessel segment",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_resistance",
        "hydraulic_resistance",
        "kPa*ms/mm3",
        "Pa*s/m3",
        sign=_SCALAR_SIGN,
        support="lumped vessel or terminal relation",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_inertance",
        "hydraulic_inertance",
        "kPa*ms2/mm3",
        "Pa*s2/m3",
        sign=_SCALAR_SIGN,
        support="lumped vessel or terminal relation",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_compliance",
        "hydraulic_compliance",
        "mm3/kPa",
        "m3/Pa",
        sign=_SCALAR_SIGN,
        support="lumped chamber or vessel compartment",
        reference=_LUMPED_REFERENCE,
    ),
    _quantity(
        "hydraulic_elastance",
        "hydraulic_elastance",
        "kPa/mm3",
        "Pa/m3",
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
