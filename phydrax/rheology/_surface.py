#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def boussinesq_scriven_stress(
    surface_rate: ArrayLike,
    surface_divergence: ArrayLike,
    shear_viscosity_n_s_m: float,
    dilatational_viscosity_n_s_m: float,
    projector: ArrayLike,
    /,
):
    return 2 * float(shear_viscosity_n_s_m) * jnp.asarray(surface_rate) + (
        float(dilatational_viscosity_n_s_m) - float(shear_viscosity_n_s_m)
    ) * jnp.asarray(surface_divergence)[..., None, None] * jnp.asarray(projector)


__all__ = ["boussinesq_scriven_stress"]
