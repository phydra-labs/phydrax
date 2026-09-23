#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class SystemEvent:
    event_id: str
    indicator_index: int
    direction: int = 0

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("System event identifier must be nonempty.")
        if (
            isinstance(self.indicator_index, bool)
            or not isinstance(self.indicator_index, int)
            or self.indicator_index < 0
        ):
            raise ValueError("System event indicator index must be nonnegative.")
        if self.direction not in (-1, 0, 1):
            raise ValueError("System event direction must be -1, 0, or 1.")


@dataclass(frozen=True, slots=True)
class SystemRuntimeState:
    continuous: Array
    discrete: Array
    time_s: Array

    @classmethod
    def create(cls, continuous: ArrayLike, discrete: ArrayLike, time_s: ArrayLike):
        continuous_ = jnp.asarray(continuous)
        discrete_ = jnp.asarray(discrete)
        time_ = jnp.asarray(time_s)
        if continuous_.ndim != 1 or discrete_.ndim != 1 or time_.shape != ():
            raise ValueError(
                "System runtime continuous/discrete states must be vectors and time scalar."
            )
        continuous_ = eqx.error_if(
            continuous_,
            jnp.any(~jnp.isfinite(continuous_))
            | jnp.any(~jnp.isfinite(discrete_))
            | ~jnp.isfinite(time_),
            "System runtime state and time must be finite.",
        )
        return cls(continuous_, discrete_, time_)


def detect_zero_crossings(previous: ArrayLike, current: ArrayLike, /):
    a = jnp.asarray(previous)
    b = jnp.asarray(current)
    if a.shape != b.shape:
        raise ValueError("Zero-crossing arrays must have identical shapes.")
    if a.ndim == 0:
        raise ValueError("Zero-crossing inputs must have an indicator axis.")
    a = eqx.error_if(
        a,
        jnp.any(~jnp.isfinite(a)) | jnp.any(~jnp.isfinite(b)),
        "Zero-crossing indicators must be finite.",
    )
    return (a * b <= 0) & (a != b)


__all__ = ["SystemEvent", "SystemRuntimeState", "detect_zero_crossings"]
