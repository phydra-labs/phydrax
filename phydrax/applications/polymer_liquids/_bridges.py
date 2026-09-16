#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from phydrax.ein import contract

from ...atomistic import PolymerChainLayoutPlan
from ._mixture import TabulatedFormFactorPlan


def trajectory_intramolecular_form_factor(
    wave_numbers: ArrayLike,
    positions: ArrayLike,
    chain_layout: PolymerChainLayoutPlan,
    site_type_ids: ArrayLike,
    site_type_count: int,
    /,
    *,
    source_id: str,
) -> TabulatedFormFactorPlan:
    """Measure per-chain isotropic Ω_ab(k) for direct PRISM admission."""

    wave = jnp.asarray(wave_numbers)
    values = jnp.asarray(positions)
    if values.ndim == 2:
        values = values[None, ...]
    site_types = jnp.asarray(site_type_ids, dtype=jnp.int32)
    type_count = int(site_type_count)
    if not isinstance(chain_layout, PolymerChainLayoutPlan):
        raise TypeError("chain_layout must be PolymerChainLayoutPlan.")
    if (
        wave.ndim != 1
        or wave.size == 0
        or jnp.any(wave <= 0.0)
        or values.ndim != 3
        or values.shape[-1] != 3
        or site_types.shape != (values.shape[1],)
        or type_count <= 0
        or jnp.any(site_types < 0)
        or jnp.any(site_types >= type_count)
        or int(np.max(np.asarray(chain_layout.particle_indices))) >= values.shape[1]
    ):
        raise ValueError("Trajectory form-factor inputs are invalid.")
    outputs = []
    channels = jnp.arange(type_count)
    for chain_indices, chain_mask in zip(
        np.asarray(chain_layout.particle_indices),
        np.asarray(chain_layout.chain_mask),
        strict=True,
    ):
        indices = jnp.asarray(chain_indices[chain_mask], dtype=jnp.int32)
        chain_positions = values[:, indices, :]
        chain_types = site_types[indices]
        membership = chain_types[:, None] == channels[None, :]
        counts = jnp.sum(membership, axis=0)
        if bool(jnp.any(counts == 0)):
            raise ValueError(
                "Every chain must contain each declared PRISM site type for this bridge."
            )
        displacement = chain_positions[:, :, None, :] - chain_positions[:, None, :, :]
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        kernel = jnp.sinc(wave[None, :, None, None] * distance[:, None, :, :] / jnp.pi)
        matrix = contract("ia,jb,fqij->fqab", membership, membership, kernel)
        normalization = jnp.sqrt(counts[:, None] * counts[None, :])
        outputs.append(jnp.mean(matrix, axis=0) / normalization[None, :, :])
    omega = jnp.mean(jnp.stack(outputs), axis=0)
    return TabulatedFormFactorPlan(wave, omega, source_id=str(source_id))


__all__ = ["trajectory_intramolecular_form_factor"]
