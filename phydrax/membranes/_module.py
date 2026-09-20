#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def concentration_polarization_bulk_to_wall(
    bulk_concentration: ArrayLike,
    solvent_flux_m_s: ArrayLike,
    mass_transfer_m_s: ArrayLike,
    /,
):
    return jnp.asarray(bulk_concentration) * jnp.exp(
        jnp.asarray(solvent_flux_m_s) / jnp.asarray(mass_transfer_m_s)
    )


def membrane_module_recovery(permeate_flow: ArrayLike, feed_flow: ArrayLike, /):
    return jnp.asarray(permeate_flow) / jnp.asarray(feed_flow)


__all__ = ["concentration_polarization_bulk_to_wall", "membrane_module_recovery"]
