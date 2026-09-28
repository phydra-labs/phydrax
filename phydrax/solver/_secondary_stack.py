#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity per-history secondary-particle stacks for radiation transport.

A transport plan that has a `SecondaryStackSpec` attached records every
secondary it creates (photoelectrons, Compton electrons, and pair daughters
for photon histories; bremsstrahlung photons for charged histories) into one
`SecondaryParticleStack` with a fixed number of slots per history. Each slot
keeps the creation position, direction, kinetic energy, material, particle
kind, and creating event or step index; the parent identity is the history
that owns the row. Secondaries below `minimum_energy` are deposited locally by
the transport instead of being recorded. A history whose stack is full when a
further secondary is created keeps transporting, tallies that secondary's
energy as truncated, counts it in `overflow_count`, and reports a
capacity-exhausted status, so a consumer can refuse the history atomically.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..typing import Bool, Dim, Float64, Identifier, Int32, Size


class _StackHistoryDim(Dim, minimum=1):
    """Histories owning one secondary stack row each."""


class _StackSlotDim(Dim, minimum=1):
    """Fixed secondary slots per history."""


class SecondaryStackSpec(StrictModule):
    """Per-history secondary capacity and production threshold in eV."""

    __strict_contract__ = True

    capacity: Size[_StackSlotDim] = eqx.field(static=True)
    minimum_energy: float = eqx.field(static=True)
    spec_id: Identifier = eqx.field(static=True)

    def __init__(self, capacity: int, /, *, minimum_energy: float) -> None:
        slots = int(capacity)
        threshold = float(minimum_energy)
        if slots < 1 or not isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "Secondary stack capacity must be positive and minimum_energy finite "
                "and positive."
            )
        self.capacity = slots
        self.minimum_energy = threshold
        self.spec_id = canonical_fingerprint(
            {
                "kind": "secondary-particle-stack",
                "capacity": slots,
                "minimum_energy": threshold,
            }
        )


class SecondaryParticleStack(StrictModule):
    """Recorded secondaries per history; inactive slots are padding."""

    __strict_contract__ = True

    positions: Float64[_StackHistoryDim, _StackSlotDim, Literal[3]]
    directions: Float64[_StackHistoryDim, _StackSlotDim, Literal[3]]
    energies: Float64[_StackHistoryDim, _StackSlotDim]
    material_index: Int32[_StackHistoryDim, _StackSlotDim]
    creation_index: Int32[_StackHistoryDim, _StackSlotDim]
    particle_kind: Int32[_StackHistoryDim, _StackSlotDim]
    active: Bool[_StackHistoryDim, _StackSlotDim]
    count: Int32[_StackHistoryDim]
    energy: Float64[_StackHistoryDim]
    overflow_count: Int32[_StackHistoryDim]
    spec_id: Identifier = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return self.active.shape[1]

    @property
    def overflowed(self) -> Array:
        return self.overflow_count > 0


def empty_secondary_stack(
    capacity: int, dtype: jnp.dtype, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Per-history scratch buffers for one stack row inside a transport loop."""
    return (
        jnp.zeros((capacity, 3), dtype=dtype),
        jnp.zeros((capacity, 3), dtype=dtype),
        jnp.zeros((capacity,), dtype=dtype),
        -jnp.ones((capacity,), dtype=jnp.int32),
        -jnp.ones((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.bool_),
    )


def push_secondary(
    buffers: tuple[Array, Array, Array, Array, Array, Array, Array],
    count: Array,
    push: Array,
    position: Array,
    direction: Array,
    energy: Array,
    material: Array,
    creation: Array,
    particle_kind: Array,
    /,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], Array, Array]:
    """Append one secondary when `push` holds and a slot is free.

    Returns the updated buffers, the updated count, and a flag that is true
    when `push` held but the row was already full.
    """
    positions, directions, energies, materials, creations, kinds, active = buffers
    capacity = active.shape[0]
    fits = push & (count < capacity)
    slot = jnp.clip(count, 0, capacity - 1)
    updated = (
        positions.at[slot].set(jnp.where(fits, position, positions[slot])),
        directions.at[slot].set(jnp.where(fits, direction, directions[slot])),
        energies.at[slot].set(jnp.where(fits, energy, energies[slot])),
        materials.at[slot].set(jnp.where(fits, material, materials[slot])),
        creations.at[slot].set(jnp.where(fits, creation, creations[slot])),
        kinds.at[slot].set(jnp.where(fits, particle_kind, kinds[slot])),
        active.at[slot].set(active[slot] | fits),
    )
    return updated, count + fits.astype(jnp.int32), push & ~fits


__all__ = [
    "SecondaryParticleStack",
    "SecondaryStackSpec",
    "empty_secondary_stack",
    "push_secondary",
]
