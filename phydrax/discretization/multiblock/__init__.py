#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Oriented conforming and nonconforming structured multiblock coupling."""

from ._core import (
    BlockInterface,
    BlockSide,
    InterfaceOrientation,
    MultiblockGridPlan,
    MultiblockInterfaceReport,
    PreparedBlock,
    PreparedMultiblockGrid,
)
from ._interpolation import MortarSide, NormCompatibleInterpolationPlan
from ._sat import MultiblockNumericalFlux, MultiblockSATCoupling


__all__ = [
    "BlockInterface",
    "BlockSide",
    "InterfaceOrientation",
    "MortarSide",
    "MultiblockGridPlan",
    "MultiblockInterfaceReport",
    "MultiblockNumericalFlux",
    "MultiblockSATCoupling",
    "NormCompatibleInterpolationPlan",
    "PreparedBlock",
    "PreparedMultiblockGrid",
]
