#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Deterministic host point realizations from batched interval programs."""

from collections.abc import Callable

import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._interval_enclosure import (
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)


def prepare_host_program(
    function: Callable[[Array], Array], dimension: int, /
) -> PreparedIntervalFunction | None:
    """Prepare an interval point map, or decline an unsupported native lowering."""
    try:
        return prepare_interval_function(
            function, dimension, batch_capacity=1, execution="batched"
        )
    except IntervalExtensionError:
        return None


def evaluate_host_program(
    program: PreparedIntervalFunction, points: NDArray[np.float64], /
) -> NDArray[np.float64]:
    """Use lower endpoints consistently as batch-independent point representatives."""
    lower, _ = program.evaluate(points, points)
    return lower
