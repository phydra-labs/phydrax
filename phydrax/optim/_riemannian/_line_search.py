#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jaxtyping import Array, PyTree

from .._iterative._globalization import (
    armijo_backtracking as _armijo_backtracking,
    ArmijoLineSearch,
    ArmijoResult,
)
from ._parameter_geometry import ParameterGeometry


def armijo_backtracking(
    value_function: Callable[[PyTree[Any]], Array],
    parameter_geometry: ParameterGeometry,
    parameters: PyTree[Any],
    value: Array,
    gradient: PyTree[Any],
    direction: PyTree[Any],
    /,
    *,
    policy: ArmijoLineSearch,
) -> ArmijoResult:
    """Search one retraction ray while reusing the caller's frozen objective closure."""

    if not isinstance(parameter_geometry, ParameterGeometry):
        raise TypeError("parameter_geometry must be a ParameterGeometry.")
    directional = parameter_geometry.inner(parameters, gradient, direction)

    def step(base, tangent, rate):
        scaled = jax.tree.map(lambda leaf: rate * leaf, tangent)
        return parameter_geometry.retract(base, scaled)

    return _armijo_backtracking(
        value_function,
        parameters,
        value,
        direction,
        directional,
        step=step,
        contains=parameter_geometry.contains,
        policy=policy,
    )


__all__ = ["ArmijoLineSearch", "ArmijoResult", "armijo_backtracking"]
