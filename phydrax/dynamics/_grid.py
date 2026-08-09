#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class TimeGrid(StrictModule):
    """A finite, strictly increasing physical-time grid with stable identity."""

    times: Array
    time_id: str = eqx.field(static=True)

    def __init__(self, times: ArrayLike, /, *, time_id: str):
        values = jnp.asarray(times)
        if values.ndim != 1 or int(values.shape[0]) < 2:
            raise ValueError("TimeGrid times must be rank one with at least two entries.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("TimeGrid times must be real-valued.")
        host = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(host)):
            raise ValueError("TimeGrid times must be finite.")
        if np.any(np.diff(host) <= 0.0):
            raise ValueError("TimeGrid times must be strictly increasing.")
        self.times = values.astype(jnp.result_type(values, float))
        self.time_id = _identifier(time_id, "TimeGrid time_id")

    @property
    def coordinates(self) -> Array:
        return self.times

    @property
    def grid_id(self) -> str:
        return self.time_id

    @property
    def coordinate_kind(self) -> str:
        return "time"

    @property
    def num_points(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_times(self) -> int:
        return self.num_points

    @property
    def num_steps(self) -> int:
        return self.num_points - 1

    @property
    def durations(self) -> Array:
        return jnp.diff(self.times)

    @property
    def t0(self) -> Array:
        return self.times[0]

    @property
    def t1(self) -> Array:
        return self.times[-1]


class IterationGrid(StrictModule):
    """Consecutive integer iterates for repeated application of one discrete map."""

    iterations: Array
    iteration_id: str = eqx.field(static=True)

    def __init__(self, iterations: ArrayLike, /, *, iteration_id: str):
        values = jnp.asarray(iterations)
        if values.ndim != 1 or int(values.shape[0]) < 2:
            raise ValueError(
                "IterationGrid iterations must be rank one with at least two entries."
            )
        if not jnp.issubdtype(values.dtype, jnp.integer):
            raise TypeError("IterationGrid iterations must have an integer dtype.")
        host = np.asarray(values, dtype=np.int64)
        if np.any(np.diff(host) != 1):
            raise ValueError(
                "IterationGrid iterations must be consecutive and increasing."
            )
        self.iterations = values.astype(jnp.int32)
        self.iteration_id = _identifier(iteration_id, "IterationGrid iteration_id")

    @classmethod
    def from_steps(
        cls,
        num_steps: int,
        /,
        *,
        start: int = 0,
        iteration_id: str,
    ) -> "IterationGrid":
        steps = int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be positive.")
        begin = int(start)
        return cls(
            jnp.arange(begin, begin + steps + 1, dtype=jnp.int32),
            iteration_id=iteration_id,
        )

    @property
    def coordinates(self) -> Array:
        return self.iterations

    @property
    def grid_id(self) -> str:
        return self.iteration_id

    @property
    def coordinate_kind(self) -> str:
        return "iteration"

    @property
    def num_points(self) -> int:
        return int(self.iterations.shape[0])

    @property
    def num_steps(self) -> int:
        return self.num_points - 1


EvolutionGrid: TypeAlias = TimeGrid | IterationGrid


__all__ = ["EvolutionGrid", "IterationGrid", "TimeGrid"]
