"""Solver-unknown scales and source-support slot order of the rod problem."""

from typing import Final, get_args, Literal, TypeAlias

from ...typing import parse


# Solver unknowns are physical fields divided by these scales, which bring each
# unknown block to order one; displacements use the body's outer radius [m].
TEMPERATURE_SCALE_K: Final[float] = 1.0e3
AXIAL_STRAIN_SCALE: Final[float] = 1.0
CONTACT_PRESSURE_SCALE_PA: Final[float] = 1.0e8
FRICTION_FORCE_SCALE_N: Final[float] = 1.0e3
GAS_PRESSURE_SCALE_PA: Final[float] = 1.0e7
COOLANT_HEAT_FLUX_SCALE_W_PER_M2: Final[float] = 1.0e6
FREE_GAS_SCALE_MOL: Final[float] = 1.0e-3
CHANNEL_ENTHALPY_SCALE_J_PER_KG: Final[float] = 1.0e6
# Joint endpoint coordinates append the fuel-site hydrostatic stresses divided
# by this stress to the field unknowns.
HYDROSTATIC_STRESS_SCALE_PA: Final[float] = 1.0e9

# Source-support slots in evidence order. The fuel-site and cladding-site groups
# are the columns of the constitutive responses' ``source_support``; the rod
# group is assembled by the rod forms.
FuelSiteSourceSlot: TypeAlias = Literal[
    "fuel_heat",
    "fuel_expansion",
    "fuel_conductivity",
    "fuel_elasticity",
    "fuel_emissivity",
    "densification",
    "solid_swelling",
    "relocation",
    "fuel_creep",
    "radial_power",
    "fission_gas",
    "helium",
    # History-dependent translation of the relocation correlation; the fuel
    # response always reports it outside its source support.
    "relocation_history",
]
CladdingSiteSourceSlot: TypeAlias = Literal[
    "cladding_heat",
    "cladding_conductivity",
    "cladding_expansion",
    "cladding_elasticity",
    "cladding_growth",
    "cladding_creep",
    "cladding_plasticity",
]
RodSourceSlot: TypeAlias = Literal[
    "gap",
    "corrosion",
    "hydrogen",
    "oxide_conductivity",
    "bore_emissivity",
    "plenum_temperature",
    "coolant",
]
FuelSourceSlot: TypeAlias = Literal[
    FuelSiteSourceSlot, CladdingSiteSourceSlot, RodSourceSlot
]

FUEL_SITE_SOURCE_SLOTS: Final[tuple[FuelSiteSourceSlot, ...]] = tuple(
    parse(slot, FuelSiteSourceSlot, "fuel_site_source_slot")
    for slot in get_args(FuelSiteSourceSlot)
)
CLADDING_SITE_SOURCE_SLOTS: Final[tuple[CladdingSiteSourceSlot, ...]] = tuple(
    parse(slot, CladdingSiteSourceSlot, "cladding_site_source_slot")
    for slot in get_args(CladdingSiteSourceSlot)
)
ROD_SOURCE_SLOTS: Final[tuple[RodSourceSlot, ...]] = tuple(
    parse(slot, RodSourceSlot, "rod_source_slot") for slot in get_args(RodSourceSlot)
)
SOURCE_SLOTS: Final[tuple[FuelSourceSlot, ...]] = tuple(
    parse(slot, FuelSourceSlot, "source_slot") for slot in get_args(FuelSourceSlot)
)
# Slots whose closures have no source support at any state: the fuel and rod
# forms report them unsupported, and a strict-source-support bundle refuses
# construction while any remains.
STATICALLY_UNSOURCED_SLOTS: Final[tuple[FuelSourceSlot, ...]] = (
    "relocation_history",
    "plenum_temperature",
)
