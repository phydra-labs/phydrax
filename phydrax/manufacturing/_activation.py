#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
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
        self.active = a
        self.activation_time_s = t

    def activate(self, selection: ArrayLike, time_s: ArrayLike, /):
        s = jnp.asarray(selection, dtype=jnp.bool_)
        newly = s & ~self.active
        return MaterialActivationState(
            self.active | s, jnp.where(newly, jnp.asarray(time_s), self.activation_time_s)
        )

    def remove(self, selection: ArrayLike, /):
        s = jnp.asarray(selection, dtype=jnp.bool_)
        return MaterialActivationState(self.active & ~s, self.activation_time_s)


__all__ = ["MaterialActivationState"]
