#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nominal coordinate-source admission shared by finite-volume preparation."""

from typing import assert_never

from ...typing import checked
from .._cell_geometry import CellGeometrySpec
from .._exact_algebraic_geometry import ExactAlgebraicCellGeometrySource
from .._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from .._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from .._exact_rational_geometry import ExactRationalCellGeometrySource


@checked
def has_power_source(geometry: CellGeometrySpec | None, /) -> bool:
    """Identify a retained power coordinate source without inferring identity from cells."""
    if geometry is None:
        return False
    match geometry.exact_source:
        case (
            ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            return True
        case (
            None
            | ExactPlcCellGeometrySource()
            | ExactPlcCellGeometryConvexSource()
            | ExactRationalCellGeometrySource()
            | ExactAlgebraicCellGeometrySource()
        ):
            return False
        case invalid:
            assert_never(invalid)
