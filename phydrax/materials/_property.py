#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._interpolation import linear_interpolate
from .._strict import StrictModule
from .._trainable import NonTrainableState


class TabulatedProperty(StrictModule, NonTrainableState):
    argument: Array
    values: Array
    unit: str = eqx.field(static=True)

    def __init__(self, argument: ArrayLike, values: ArrayLike, unit: str):
        x = np.asarray(argument, float)
        y = np.asarray(values)
        if (
            x.ndim != 1
            or x.size < 2
            or y.shape[0] != x.size
            or np.any(np.diff(x) <= 0)
            or not unit
        ):
            raise ValueError("Tabulated property arguments must be ordered and aligned.")
        self.argument = jnp.asarray(x)
        self.values = jnp.asarray(y)
        self.unit = unit

    def evaluate(self, argument: ArrayLike):
        q = jnp.asarray(argument)
        if self.values.ndim == 1:
            return linear_interpolate(
                self.argument, self.values, q, bounds="error"
            ).values
        flat = self.values.reshape((self.values.shape[0], -1))
        out = jnp.stack(
            tuple(
                linear_interpolate(self.argument, flat[:, i], q, bounds="error").values
                for i in range(flat.shape[1])
            ),
            axis=-1,
        )
        return out.reshape((*q.shape, *self.values.shape[1:]))


__all__ = ["TabulatedProperty"]
