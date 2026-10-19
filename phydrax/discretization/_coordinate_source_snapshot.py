#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Read-only host source arrays retained by one coordinate preparation budget."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._strict import StrictModule


class BarycentricSourceSnapshot(StrictModule):
    """Retain the host coefficient action while execution keeps its dynamic leaf."""

    barycentric_weights: NDArray[np.float64]


class SplineSourceSnapshot(StrictModule):
    """Retain one knot/weight bank without changing the numerical source fields."""

    u_knots: NDArray[np.float64]
    v_knots: NDArray[np.float64]
    weights: NDArray[np.float64]
