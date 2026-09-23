#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def hamrock_dowson_central_film(
    radius_m: float,
    speed_parameter: ArrayLike,
    material_parameter: ArrayLike,
    load_parameter: ArrayLike,
    /,
):
    if not isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Hamrock-Dowson radius must be finite and positive.")
    speed = jnp.asarray(speed_parameter)
    material = jnp.asarray(material_parameter)
    load = jnp.asarray(load_parameter)
    speed = eqx.error_if(
        speed,
        jnp.any(
            ~jnp.isfinite(speed)
            | ~jnp.isfinite(material)
            | ~jnp.isfinite(load)
            | (speed <= 0)
            | (material <= 0)
            | (load <= 0)
        ),
        "Hamrock-Dowson parameters must be finite and positive.",
    )
    return 2.69 * float(radius_m) * speed**0.67 * material**0.53 * load**-0.067


__all__ = ["hamrock_dowson_central_film"]
