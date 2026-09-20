#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def surface_species_rate(
    surface_concentration: ArrayLike,
    tangential_flux_divergence: ArrayLike,
    surface_divergence_s_inv: ArrayLike,
    bulk_exchange_mol_m2_s: ArrayLike = 0.0,
    /,
):
    gamma = jnp.asarray(surface_concentration)
    return (
        -jnp.asarray(tangential_flux_divergence)
        - gamma * jnp.asarray(surface_divergence_s_inv)
        + jnp.asarray(bulk_exchange_mol_m2_s)
    )


__all__ = ["surface_species_rate"]
