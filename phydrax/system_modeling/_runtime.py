#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class SystemEvent:
    event_id: str
    indicator_index: int
    direction: int = 0


@dataclass(frozen=True, slots=True)
class SystemRuntimeState:
    continuous: Array
    discrete: Array
    time_s: Array

    @classmethod
    def create(cls, continuous: ArrayLike, discrete: ArrayLike, time_s: ArrayLike):
        return cls(jnp.asarray(continuous), jnp.asarray(discrete), jnp.asarray(time_s))


def detect_zero_crossings(previous: ArrayLike, current: ArrayLike, /):
    a = jnp.asarray(previous)
    b = jnp.asarray(current)
    return (a * b <= 0) & (a != b)


__all__ = ["SystemEvent", "SystemRuntimeState", "detect_zero_crossings"]
