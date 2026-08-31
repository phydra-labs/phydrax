#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice import LatticeBoltzmannVelocitySet


class AALatticeBoltzmannAddressing(StrictModule, NonTrainableState):
    """Logical-to-storage direction and pull-source offsets for one AA parity."""

    parity: Array
    storage_direction_indices: Array
    source_offsets: Array
    addressing_id: str = eqx.field(static=True)


class AALatticeBoltzmannParityState(StrictModule):
    """Raw AA storage with parity represented as checkpointable array state."""

    storage: Array
    parity: Array
    addressing_id: str = eqx.field(static=True)


class AALatticeBoltzmannCheckpoint(StrictModule, NonTrainableState):
    """Exact raw AA restart state and its caller-assigned checkpoint identity."""

    storage: Array
    parity: Array
    addressing_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    identity: str = eqx.field(static=True)


class AALatticeBoltzmannPlan(StrictModule, NonTrainableState):
    """Explicit, round-trippable even/odd AA direction addressing."""

    velocity_set: LatticeBoltzmannVelocitySet
    addressing_id: str = eqx.field(static=True)

    def __init__(self, velocity_set: LatticeBoltzmannVelocitySet, /):
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be a LatticeBoltzmannVelocitySet.")
        self.velocity_set = velocity_set
        self.addressing_id = canonical_fingerprint(
            {
                "kind": "aa-lattice-boltzmann-addressing",
                "lattice": velocity_set.lattice_id,
                "layout": "trailing-q",
                "parity": "explicit-array",
            }
        )

    def _populations(self, value: Array, /) -> Array:
        if not eqx.is_array(value):
            raise TypeError("AA storage accepts JAX arrays only.")
        if (
            value.ndim != self.velocity_set.dimension + 1
            or value.shape[-1] != self.velocity_set.population_count
        ):
            raise ValueError("AA storage must use the velocity set's trailing-Q shape.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("AA storage requires an inexact population dtype.")
        return value

    @staticmethod
    def _parity(value: Any, /) -> Array:
        parity = jnp.asarray(value, dtype=jnp.int32)
        if parity.shape != ():
            raise ValueError("AA parity must be one scalar.")
        return eqx.error_if(
            parity,
            (parity != 0) & (parity != 1),
            "AA parity must be zero (even) or one (odd).",
        )

    def _state(self, state: AALatticeBoltzmannParityState, /) -> None:
        if not isinstance(state, AALatticeBoltzmannParityState):
            raise TypeError("state must be AALatticeBoltzmannParityState.")
        if state.addressing_id != self.addressing_id:
            raise ValueError("AA state was created by a different addressing plan.")
        self._populations(state.storage)
        self._parity(state.parity)

    def addressing(
        self, state: AALatticeBoltzmannParityState, /
    ) -> AALatticeBoltzmannAddressing:
        """Return the concrete logical reads and pull offsets for ``state``."""

        self._state(state)
        direct = jnp.arange(self.velocity_set.population_count, dtype=jnp.int32)
        indices = jnp.where(state.parity == 0, direct, self.velocity_set.opposite)
        source_offsets = -jnp.take(self.velocity_set.velocities, indices, axis=0)
        return AALatticeBoltzmannAddressing(
            state.parity,
            indices,
            source_offsets,
            self.addressing_id,
        )

    def encode(
        self,
        canonical_populations: Array,
        /,
        *,
        parity: Any = 0,
    ) -> AALatticeBoltzmannParityState:
        """Encode one canonical trailing-Q field at explicit even or odd parity."""

        populations = self._populations(canonical_populations)
        parity_ = self._parity(parity)
        opposite = jnp.take(populations, self.velocity_set.opposite, axis=-1)
        storage = jax.lax.cond(
            parity_ == 0,
            lambda _: populations,
            lambda _: opposite,
            operand=None,
        )
        return AALatticeBoltzmannParityState(storage, parity_, self.addressing_id)

    def canonical(self, state: AALatticeBoltzmannParityState, /) -> Array:
        """Decode raw parity storage into the unique canonical trailing-Q field."""

        self._state(state)
        opposite = jnp.take(state.storage, self.velocity_set.opposite, axis=-1)
        return jax.lax.cond(
            state.parity == 0,
            lambda _: state.storage,
            lambda _: opposite,
            operand=None,
        )

    def advance(
        self,
        state: AALatticeBoltzmannParityState,
        canonical_next: Array,
        /,
    ) -> AALatticeBoltzmannParityState:
        """Store a canonical next state in the opposite AA parity."""

        self._state(state)
        next_parity = jnp.asarray(1, dtype=jnp.int32) - state.parity
        return self.encode(canonical_next, parity=next_parity)

    def checkpoint(
        self,
        state: AALatticeBoltzmannParityState,
        checkpoint_id: str,
        /,
    ) -> AALatticeBoltzmannCheckpoint:
        """Capture raw storage and parity without canonicalizing or copying semantics."""

        self._state(state)
        identifier = str(checkpoint_id)
        if not identifier:
            raise ValueError("checkpoint_id must be non-empty.")
        identity = canonical_fingerprint(
            {
                "kind": "aa-lattice-boltzmann-checkpoint",
                "checkpoint": identifier,
                "addressing": self.addressing_id,
                "state": array_tree_fingerprint(
                    {"storage": state.storage, "parity": state.parity}
                ),
            }
        )
        return AALatticeBoltzmannCheckpoint(
            state.storage,
            state.parity,
            self.addressing_id,
            identifier,
            identity,
        )

    def restore(
        self, checkpoint: AALatticeBoltzmannCheckpoint, /
    ) -> AALatticeBoltzmannParityState:
        """Restore the exact raw representation, including its next-step parity."""

        if not isinstance(checkpoint, AALatticeBoltzmannCheckpoint):
            raise TypeError("checkpoint must be AALatticeBoltzmannCheckpoint.")
        if checkpoint.addressing_id != self.addressing_id:
            raise ValueError("AA checkpoint belongs to a different addressing plan.")
        state = AALatticeBoltzmannParityState(
            checkpoint.storage,
            checkpoint.parity,
            checkpoint.addressing_id,
        )
        self._state(state)
        expected = canonical_fingerprint(
            {
                "kind": "aa-lattice-boltzmann-checkpoint",
                "checkpoint": checkpoint.checkpoint_id,
                "addressing": self.addressing_id,
                "state": array_tree_fingerprint(
                    {"storage": checkpoint.storage, "parity": checkpoint.parity}
                ),
            }
        )
        if checkpoint.identity != expected:
            raise ValueError("AA checkpoint identity is inconsistent with its metadata.")
        return state


__all__ = [
    "AALatticeBoltzmannAddressing",
    "AALatticeBoltzmannCheckpoint",
    "AALatticeBoltzmannParityState",
    "AALatticeBoltzmannPlan",
]
