#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Closed selectors and nominal array extents of the nuclear-fuel application."""

from __future__ import annotations

from typing import get_args, Literal, TypeAlias

from ...typing import Dim, parse


FuelForm: TypeAlias = Literal["uo2-pellet"]
CladdingAlloy: TypeAlias = Literal["zircaloy-4"]
CladdingHeatTreatment: TypeAlias = Literal["stress-relief-annealed"]
FuelGeometryReduction: TypeAlias = Literal["layered-radial", "axisymmetric-rz"]
AxialInteraction: TypeAlias = Literal["frictionless", "coulomb"]
CoolantBoundaryKind: TypeAlias = Literal[
    "prescribed-wall-temperature", "single-channel", "coupled-channel"
]
FuelGasSpecies: TypeAlias = Literal["He", "Ar", "Kr", "Xe", "N2"]

# Canonical order of every gas-composition axis in this application.
FUEL_GAS_SPECIES: tuple[FuelGasSpecies, ...] = tuple(
    parse(name, FuelGasSpecies, "gas_species") for name in get_args(FuelGasSpecies)
)


class FuelSiteDim(Dim):
    """Fuel material integration sites."""


class CladdingSiteDim(Dim):
    """Cladding material integration sites."""


class AxialLayerDim(Dim):
    """Axial layers of a layered-radial rod."""


class RadialNodeDim(Dim):
    """Radial finite-element nodes of one body in one layer."""


class IntragranularModeDim(Dim):
    """Spectral modes of intragranular gas diffusion."""


class GasSpeciesDim(Dim):
    """Gas species in ``FUEL_GAS_SPECIES`` order."""


__all__ = [
    "AxialInteraction",
    "AxialLayerDim",
    "CladdingAlloy",
    "CladdingHeatTreatment",
    "CladdingSiteDim",
    "CoolantBoundaryKind",
    "FUEL_GAS_SPECIES",
    "FuelForm",
    "FuelGasSpecies",
    "FuelGeometryReduction",
    "FuelSiteDim",
    "GasSpeciesDim",
    "IntragranularModeDim",
    "RadialNodeDim",
]
