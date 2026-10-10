#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Weighted reductions of site values onto axial zones."""

from __future__ import annotations

import jax
from jax import Array


def zone_average(values: Array, weights: Array, zones: Array, count: int, /) -> Array:
    """Weighted mean of ``values`` over each of ``count`` zones.

    Every zone index in ``0..count-1`` must carry a positive total weight.
    """

    totals = jax.ops.segment_sum(values * weights, zones, count)
    return totals / jax.ops.segment_sum(weights, zones, count)


__all__ = ["zone_average"]
