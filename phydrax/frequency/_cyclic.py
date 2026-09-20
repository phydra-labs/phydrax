#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp


def cyclic_phase_shift(harmonic_index: int, sector_count: int):
    if sector_count <= 0:
        raise ValueError("Sector count must be positive.")
    return jnp.exp(2j * jnp.pi * int(harmonic_index) / int(sector_count))


__all__ = ["cyclic_phase_shift"]
