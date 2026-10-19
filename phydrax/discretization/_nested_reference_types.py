#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Integration-piece records shared by nested geometry and field transfer."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._coordinate_enclosure import Polynomial


class NestedReferencePair(NamedTuple):
    """Map the finer reference cell affinely into its paired coarser cell."""

    source_cell: int
    target_cell: int
    fine_is_source: bool
    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]


class PolynomialReferencePair(NamedTuple):
    """Map the finer reference cell polynomially with enclosed Jacobian bounds."""

    source_cell: int
    target_cell: int
    fine_is_source: bool
    arguments: tuple[Polynomial, ...]
    source_kind: str
    target_kind: str
    jacobian_bounds: tuple[Fraction, Fraction]
