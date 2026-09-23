#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


class MaterialActivationState(StrictModule, NonTrainableState):
    active: Array
    activation_time_s: Array

    def __init__(self, active: ArrayLike, activation_time_s: ArrayLike, /):
        a = jnp.asarray(active, dtype=jnp.bool_)
        t = jnp.asarray(activation_time_s)
        if a.shape != t.shape:
            raise ValueError("Activation arrays must align.")
        t = eqx.error_if(
            t,
            jnp.any(a & ~jnp.isfinite(t)),
            "Active material cells require finite activation times.",
        )
        self.active = a
        self.activation_time_s = t

    def activate(self, selection: ArrayLike, time_s: ArrayLike, /):
        s = jnp.asarray(selection, dtype=jnp.bool_)
        if s.shape != self.active.shape:
            raise ValueError("Activation selection must match the material topology.")
        time = jnp.asarray(time_s)
        if time.shape != ():
            raise ValueError("Activation time must be scalar.")
        time = eqx.error_if(
            time,
            ~jnp.isfinite(time),
            "Activation time must be finite.",
        )
        newly = s & ~self.active
        return MaterialActivationState(
            self.active | s, jnp.where(newly, time, self.activation_time_s)
        )

    def remove(self, selection: ArrayLike, /):
        s = jnp.asarray(selection, dtype=jnp.bool_)
        if s.shape != self.active.shape:
            raise ValueError("Removal selection must match the material topology.")
        return MaterialActivationState(self.active & ~s, self.activation_time_s)


__all__ = ["MaterialActivationState"]
