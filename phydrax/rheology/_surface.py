#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def boussinesq_scriven_stress(
    surface_rate: ArrayLike,
    surface_divergence: ArrayLike,
    shear_viscosity_n_s_m: ArrayLike,
    dilatational_viscosity_n_s_m: ArrayLike,
    projector: ArrayLike,
    /,
) -> Array:
    """Return ``2 mu_s D_s + (kappa_s - mu_s) (div_s u) P`` for one interface.

    Viscosities may be dynamic (traced or differentiated) scalars.
    """
    shear = jnp.asarray(shear_viscosity_n_s_m)
    dilatational = jnp.asarray(dilatational_viscosity_n_s_m)
    return 2.0 * shear * jnp.asarray(surface_rate) + (dilatational - shear) * jnp.asarray(
        surface_divergence
    )[..., None, None] * jnp.asarray(projector)


__all__ = ["boussinesq_scriven_stress"]
