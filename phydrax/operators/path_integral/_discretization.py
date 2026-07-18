#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PathDiscretization(StrictModule, NonTrainableState):
    r"""A fixed uniform time slicing for finite-dimensional paths.

    `num_steps` is the number of transitions, so a materialized path contains
    `num_steps + 1` nodes including both endpoints.
    """

    t0: Array
    t1: Array
    num_steps: int
    times: Array
    midpoints: Array
    dt: Array

    def __init__(
        self,
        t0: ArrayLike,
        t1: ArrayLike,
        /,
        *,
        num_steps: int,
    ):
        t0_arr = jnp.asarray(t0, dtype=float)
        t1_arr = jnp.asarray(t1, dtype=float)
        if t0_arr.shape != () or t1_arr.shape != ():
            raise ValueError("PathDiscretization endpoints must be scalar.")
        if not bool(jnp.isfinite(t0_arr)) or not bool(jnp.isfinite(t1_arr)):
            raise ValueError("PathDiscretization endpoints must be finite.")
        if bool(t1_arr <= t0_arr):
            raise ValueError("PathDiscretization requires t1 > t0.")

        if isinstance(num_steps, bool):
            raise TypeError("num_steps must be an integer.")
        try:
            steps = operator.index(num_steps)
        except TypeError as error:
            raise TypeError("num_steps must be an integer.") from error
        if steps < 1:
            raise ValueError("PathDiscretization requires num_steps >= 1.")

        times = jnp.linspace(t0_arr, t1_arr, steps + 1)
        self.t0 = t0_arr
        self.t1 = t1_arr
        self.num_steps = steps
        self.times = times
        self.midpoints = 0.5 * (times[:-1] + times[1:])
        self.dt = (t1_arr - t0_arr) / float(steps)

    @property
    def num_nodes(self) -> int:
        """Number of path nodes, including both endpoints."""
        return self.num_steps + 1

    @property
    def duration(self) -> Array:
        """Total path duration."""
        return self.t1 - self.t0


__all__ = ["PathDiscretization"]
