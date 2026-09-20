#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def hamrock_dowson_central_film(
    radius_m: float,
    speed_parameter: ArrayLike,
    material_parameter: ArrayLike,
    load_parameter: ArrayLike,
    /,
):
    return (
        2.69
        * float(radius_m)
        * jnp.asarray(speed_parameter) ** 0.67
        * jnp.asarray(material_parameter) ** 0.53
        * jnp.asarray(load_parameter) ** -0.067
    )


__all__ = ["hamrock_dowson_central_film"]
