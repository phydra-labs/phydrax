#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._frequency import FrequencyAxis


def evaluate_frequency_sweep(
    axis: FrequencyAxis, evaluator: Callable[[Array], ArrayLike], /
) -> Array:
    return jnp.stack(tuple(evaluator(value) for value in axis.frequency_hz))


__all__ = ["evaluate_frequency_sweep"]
