#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Read-only host source arrays retained by one coordinate preparation budget."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .._strict import StrictModule


class BarycentricSourceSnapshot(StrictModule):
    """Retain the host coefficient action while execution keeps its dynamic leaf.

    ``exact_weights`` retains the source-owned rational rows of a rational
    homogeneous action, whose binary64 execution view is ``barycentric_weights``;
    ``None`` declares the binary64 rows themselves exactly representable.
    """

    barycentric_weights: NDArray[np.float64]
    exact_weights: tuple[tuple[Fraction, ...], ...] | None


class SplineSourceSnapshot(StrictModule):
    """Retain one knot/weight bank without changing the numerical source fields."""

    u_knots: NDArray[np.float64]
    v_knots: NDArray[np.float64]
    weights: NDArray[np.float64]
