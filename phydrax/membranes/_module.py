#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def concentration_polarization_bulk_to_wall(
    bulk_concentration: ArrayLike,
    solvent_flux_m_s: ArrayLike,
    mass_transfer_m_s: ArrayLike,
    /,
):
    bulk = jnp.asarray(bulk_concentration)
    flux = jnp.asarray(solvent_flux_m_s)
    transfer = jnp.asarray(mass_transfer_m_s)
    bulk, flux, transfer = jnp.broadcast_arrays(bulk, flux, transfer)
    bulk = eqx.error_if(
        bulk,
        jnp.any(
            ~jnp.isfinite(bulk)
            | ~jnp.isfinite(flux)
            | ~jnp.isfinite(transfer)
            | (bulk < 0)
            | (transfer <= 0)
        ),
        "Concentration polarization inputs must be finite with nonnegative concentration and positive transfer.",
    )
    return bulk * jnp.exp(flux / transfer)


def membrane_module_recovery(permeate_flow: ArrayLike, feed_flow: ArrayLike, /):
    permeate = jnp.asarray(permeate_flow)
    feed = jnp.asarray(feed_flow)
    permeate, feed = jnp.broadcast_arrays(permeate, feed)
    permeate = eqx.error_if(
        permeate,
        jnp.any(
            ~jnp.isfinite(permeate)
            | ~jnp.isfinite(feed)
            | (permeate < 0)
            | (feed <= 0)
            | (permeate > feed)
        ),
        "Membrane recovery requires finite flows with 0 <= permeate <= feed and feed > 0.",
    )
    return permeate / feed


__all__ = ["concentration_polarization_bulk_to_wall", "membrane_module_recovery"]
