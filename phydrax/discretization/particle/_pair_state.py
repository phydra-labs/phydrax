#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._pairwise import ParticlePairRelation


class ParticlePairKeys(StrictModule, NonTrainableState):
    """Collision-free stable keys for one realized same-set pair relation."""

    keys: Array
    valid: Array
    successful: Array
    key_space_id: str = eqx.field(static=True)


class ParticlePairKeySpace(StrictModule, NonTrainableState):
    """Stable triangular ordinals for unordered pairs on one particle support."""

    sorted_particle_ids: Array
    particle_discretization_id: str = eqx.field(static=True)
    particle_support_id: str = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    pair_count: int = eqx.field(static=True)
    key_space_id: str = eqx.field(static=True)

    def __init__(self, particles: ParticleDiscretization, /):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        ids = np.asarray(particles.particle_ids, dtype=np.int64)
        if np.unique(ids).size != ids.size:
            raise ValueError("Particle pair key space requires unique stable IDs.")
        capacity = int(ids.size)
        pair_count = capacity * (capacity - 1) // 2
        if pair_count > np.iinfo(np.int64).max:
            raise OverflowError("Particle pair key space exceeds int64 capacity.")
        self.sorted_particle_ids = jnp.asarray(np.sort(ids), dtype=jnp.int64)
        self.particle_discretization_id = particles.prepared_id
        self.particle_support_id = particles.support.support_id
        self.particle_capacity = capacity
        self.pair_count = pair_count
        self.key_space_id = canonical_fingerprint(
            {
                "kind": "particle-pair-key-space",
                "particles": particles.prepared_id,
                "capacity": capacity,
                "pair_count": pair_count,
                "sorted_particle_ids": ids[np.argsort(ids)].tolist(),
            }
        )

    def keys(self, pairs: ParticlePairRelation, /) -> ParticlePairKeys:
        if not isinstance(pairs, ParticlePairRelation):
            raise TypeError("pairs must be a ParticlePairRelation.")
        if (
            pairs.source_support_id != self.particle_support_id
            or pairs.target_support_id != self.particle_support_id
            or not pairs.same_set
            or not pairs.unordered
        ):
            raise ValueError(
                "Pair relation must be unordered on the key space particle support."
            )
        sorted_ids = self.sorted_particle_ids
        last = self.particle_capacity - 1
        left_rank_raw = jnp.searchsorted(sorted_ids, pairs.left_particle_ids)
        right_rank_raw = jnp.searchsorted(sorted_ids, pairs.right_particle_ids)
        left_rank = jnp.clip(left_rank_raw, 0, last)
        right_rank = jnp.clip(right_rank_raw, 0, last)
        left_match = (left_rank_raw < self.particle_capacity) & (
            sorted_ids[left_rank] == pairs.left_particle_ids
        )
        right_match = (right_rank_raw < self.particle_capacity) & (
            sorted_ids[right_rank] == pairs.right_particle_ids
        )
        valid = pairs.valid & left_match & right_match & (left_rank < right_rank)
        keys = left_rank.astype(jnp.int64) * (
            2 * self.particle_capacity - left_rank.astype(jnp.int64) - 1
        ) // 2 + (right_rank.astype(jnp.int64) - left_rank.astype(jnp.int64) - 1)
        keys = jnp.where(valid, keys, jnp.asarray(-1, dtype=jnp.int64))
        successful = jnp.all(~pairs.valid | valid)
        return ParticlePairKeys(keys, valid, successful, self.key_space_id)


class ParticlePairRemap(StrictModule, NonTrainableState):
    """Deterministic route map from old pair slots to new pair slots."""

    source_indices: Array
    continued: Array
    born: Array
    ended_count: Array
    successful: Array
    old_capacity: int = eqx.field(static=True)
    new_capacity: int = eqx.field(static=True)


def _validated_keys(
    name: str, keys: ArrayLike, valid: ArrayLike, maximum_key: int, /
) -> tuple[Array, Array, Array]:
    keys_ = jnp.asarray(keys)
    valid_ = jnp.asarray(valid, dtype=bool)
    if keys_.ndim != 1 or valid_.shape != keys_.shape:
        raise ValueError(f"{name} keys and validity must have shape (pairs,).")
    if not jnp.issubdtype(keys_.dtype, jnp.integer):
        raise TypeError(f"{name} keys must be integers.")
    in_range = (keys_ >= 0) & (keys_ <= maximum_key)
    return keys_.astype(jnp.int64), valid_, jnp.all(~valid_ | in_range)


def match_particle_pair_keys(
    old_keys: ArrayLike,
    old_valid: ArrayLike,
    new_keys: ArrayLike,
    new_valid: ArrayLike,
    /,
    *,
    maximum_key: int,
) -> ParticlePairRemap:
    """Match two fixed-capacity key realizations without differentiating routes."""

    maximum = int(maximum_key)
    if maximum < 0:
        raise ValueError("maximum_key must be nonnegative.")
    old, old_mask, old_in_range = _validated_keys(
        "Old pair", old_keys, old_valid, maximum
    )
    new, new_mask, new_in_range = _validated_keys(
        "New pair", new_keys, new_valid, maximum
    )
    old_capacity = old.shape[0]
    new_capacity = new.shape[0]
    if old_capacity == 0 or new_capacity == 0:
        if old_capacity != new_capacity:
            raise ValueError("Zero-capacity pair remapping requires equal capacities.")
        return ParticlePairRemap(
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=bool),
            jnp.zeros((0,), dtype=bool),
            jnp.zeros((), dtype=jnp.int32),
            old_in_range & new_in_range,
            0,
            0,
        )
    sentinel = jnp.asarray(maximum + 1, dtype=jnp.int64)
    old_sort_keys = jnp.where(old_mask, old, sentinel)
    old_order = jnp.argsort(old_sort_keys, stable=True)
    old_sorted = old_sort_keys[old_order]
    new_sort_keys = jnp.where(new_mask, new, sentinel)
    new_sorted = jnp.sort(new_sort_keys)
    duplicate_old = jnp.any(
        (old_sorted[1:] == old_sorted[:-1]) & (old_sorted[1:] != sentinel)
    )
    duplicate_new = jnp.any(
        (new_sorted[1:] == new_sorted[:-1]) & (new_sorted[1:] != sentinel)
    )
    positions = jnp.searchsorted(old_sorted, new, side="left")
    safe_positions = jnp.clip(positions, 0, old_capacity - 1)
    continued = (
        new_mask & (positions < old_capacity) & (old_sorted[safe_positions] == new)
    )
    source_indices = jnp.where(
        continued, old_order[safe_positions], jnp.asarray(0, dtype=old_order.dtype)
    ).astype(jnp.int32)
    born = new_mask & ~continued
    ended = jnp.maximum(
        jnp.sum(old_mask, dtype=jnp.int32) - jnp.sum(continued, dtype=jnp.int32),
        0,
    )
    successful = old_in_range & new_in_range & ~duplicate_old & ~duplicate_new
    return ParticlePairRemap(
        jax.lax.stop_gradient(source_indices),
        jax.lax.stop_gradient(continued),
        jax.lax.stop_gradient(born),
        jax.lax.stop_gradient(ended),
        jax.lax.stop_gradient(successful),
        old_capacity,
        new_capacity,
    )


def remap_particle_pair_values(
    remap: ParticlePairRemap,
    values: PyTree[Array],
    /,
    *,
    fill_value: Any = 0,
) -> PyTree[Array]:
    """Gather edge-local values for continued routes and fill contact births."""

    if not isinstance(remap, ParticlePairRemap):
        raise TypeError("remap must be a ParticlePairRemap.")
    leaves = jax.tree.leaves(values)
    if not leaves:
        raise ValueError("Pair values must contain at least one array leaf.")
    for leaf in leaves:
        if not eqx.is_array(leaf) or leaf.ndim == 0:
            raise TypeError("Every pair-value leaf must be a nonscalar array.")
        if leaf.shape[0] != remap.old_capacity:
            raise ValueError("Every pair-value leaf must use the old pair capacity.")

    def gather(leaf):
        selected = leaf[remap.source_indices]
        fill = jnp.asarray(fill_value, dtype=leaf.dtype)
        mask = remap.continued.reshape((remap.new_capacity,) + (1,) * (leaf.ndim - 1))
        return jnp.where(mask, selected, fill)

    return jax.tree.map(gather, values)


__all__ = [
    "ParticlePairKeys",
    "ParticlePairKeySpace",
    "ParticlePairRemap",
    "match_particle_pair_keys",
    "remap_particle_pair_values",
]
