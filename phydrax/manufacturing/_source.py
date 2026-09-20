#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class GaussianMovingSource(StrictModule, NonTrainableState):
    power_w: Array
    absorptivity: Array
    radius_m: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, power_w, absorptivity, radius_m, /):
        if power_w < 0 or not 0 <= absorptivity <= 1 or radius_m <= 0:
            raise ValueError("Gaussian source parameters invalid.")
        self.power_w = jnp.asarray(power_w)
        self.absorptivity = jnp.asarray(absorptivity)
        self.radius_m = jnp.asarray(radius_m)
        self.source_id = canonical_fingerprint(
            {
                "kind": "gaussian-source",
                "power": power_w,
                "absorptivity": absorptivity,
                "radius": radius_m,
            }
        )

    def evaluate(self, coordinates: ArrayLike, center: ArrayLike, /):
        delta = jnp.asarray(coordinates) - jnp.asarray(center)
        r2 = jnp.sum(delta * delta, axis=-1)
        return (
            2
            * self.absorptivity
            * self.power_w
            / (jnp.pi * self.radius_m**2)
            * jnp.exp(-2 * r2 / self.radius_m**2)
        )


class GoldakDoubleEllipsoidSource(StrictModule, NonTrainableState):
    power_w: Array
    efficiency: Array
    front_m: Array
    rear_m: Array
    width_m: Array
    depth_m: Array

    def __init__(self, power_w, efficiency, front_m, rear_m, width_m, depth_m, /):
        if min(power_w, efficiency, front_m, rear_m, width_m, depth_m) <= 0:
            raise ValueError("Goldak parameters must be positive.")
        self.power_w = jnp.asarray(power_w)
        self.efficiency = jnp.asarray(efficiency)
        self.front_m = jnp.asarray(front_m)
        self.rear_m = jnp.asarray(rear_m)
        self.width_m = jnp.asarray(width_m)
        self.depth_m = jnp.asarray(depth_m)

    def evaluate(self, coordinates: ArrayLike, center: ArrayLike, /):
        p = jnp.asarray(coordinates) - jnp.asarray(center)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        length = jnp.where(x >= 0, self.front_m, self.rear_m)
        fraction = jnp.where(
            x >= 0,
            2 * self.front_m / (self.front_m + self.rear_m),
            2 * self.rear_m / (self.front_m + self.rear_m),
        )
        norm = (
            6
            * jnp.sqrt(3.0)
            * fraction
            * self.efficiency
            * self.power_w
            / (jnp.pi * jnp.sqrt(jnp.pi) * length * self.width_m * self.depth_m)
        )
        return norm * jnp.exp(
            -3
            * (
                x * x / (length * length)
                + y * y / (self.width_m**2)
                + z * z / (self.depth_m**2)
            )
        )


__all__ = ["GaussianMovingSource", "GoldakDoubleEllipsoidSource"]
