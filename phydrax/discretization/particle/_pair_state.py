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


INTERACTION_KEY_WIDTH = 5
PARTICLE_PAIR_INTERACTION = 0
PARTICLE_WALL_INTERACTION = 1
CLUMP_COMPONENT_INTERACTION = 2
IMPLICIT_BARRIER_INTERACTION = 3


class ParticlePairKeys(StrictModule, NonTrainableState):
    """Capacity-independent structured identities for one pair relation."""

    keys: Array
    valid: Array
    successful: Array
    key_space_id: str = eqx.field(static=True)


class ParticlePairKeySpace(StrictModule, NonTrainableState):
    """Stable unordered-pair identities built from physical particle IDs."""

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
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError("Particle pair identities require unique nonnegative IDs.")
        capacity = int(ids.size)
        pair_count = capacity * (capacity - 1) // 2
        self.sorted_particle_ids = jnp.asarray(np.sort(ids), dtype=jnp.int64)
        self.particle_discretization_id = particles.prepared_id
        self.particle_support_id = particles.support.support_id
        self.particle_capacity = capacity
        self.pair_count = pair_count
        self.key_space_id = canonical_fingerprint(
            {
                "kind": "particle-pair-identity-space",
                "particles": particles.prepared_id,
                "sorted_particle_ids": np.sort(ids).tolist(),
                "identity_width": INTERACTION_KEY_WIDTH,
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
                "Pair relation must be unordered on the identity-space support."
            )
        sorted_ids = self.sorted_particle_ids
        last = self.particle_capacity - 1
        left_id = jnp.minimum(pairs.left_particle_ids, pairs.right_particle_ids)
        right_id = jnp.maximum(pairs.left_particle_ids, pairs.right_particle_ids)
        left_rank_raw = jnp.searchsorted(sorted_ids, left_id)
        right_rank_raw = jnp.searchsorted(sorted_ids, right_id)
        left_rank = jnp.clip(left_rank_raw, 0, last)
        right_rank = jnp.clip(right_rank_raw, 0, last)
        left_match = (left_rank_raw < self.particle_capacity) & (
            sorted_ids[left_rank] == left_id
        )
        right_match = (right_rank_raw < self.particle_capacity) & (
            sorted_ids[right_rank] == right_id
        )
        valid = pairs.valid & left_match & right_match & (left_id < right_id)
        zeros = jnp.zeros_like(left_id, dtype=jnp.int64)
        identity = jnp.stack(
            (
                jnp.full_like(left_id, PARTICLE_PAIR_INTERACTION, dtype=jnp.int64),
                left_id.astype(jnp.int64),
                right_id.astype(jnp.int64),
                zeros,
                zeros,
            ),
            axis=-1,
        )
        identity = jnp.where(valid[:, None], identity, -jnp.ones_like(identity))
        successful = jnp.all(~pairs.valid | valid)
        return ParticlePairKeys(identity, valid, successful, self.key_space_id)


def particle_wall_interaction_keys(
    particle_ids: ArrayLike,
    object_ids: ArrayLike,
    feature_kinds: ArrayLike,
    feature_ids: ArrayLike,
    valid: ArrayLike,
    /,
    *,
    interaction_kind: int = PARTICLE_WALL_INTERACTION,
) -> Array:
    """Build stable particle/object/feature identities without packed hashes."""

    particle = jnp.asarray(particle_ids, dtype=jnp.int64)
    object_ = jnp.asarray(object_ids, dtype=jnp.int64)
    feature_kind = jnp.asarray(feature_kinds, dtype=jnp.int64)
    feature = jnp.asarray(feature_ids, dtype=jnp.int64)
    mask = jnp.asarray(valid, dtype=bool)
    if (
        particle.ndim != 1
        or object_.shape != particle.shape
        or feature_kind.shape != particle.shape
        or feature.shape != particle.shape
        or mask.shape != particle.shape
    ):
        raise ValueError("Interaction identity components must share one route shape.")
    kind = int(interaction_kind)
    if kind < 0:
        raise ValueError("interaction_kind must be nonnegative.")
    identity = jnp.stack(
        (
            jnp.full_like(particle, kind),
            particle,
            object_,
            feature_kind,
            feature,
        ),
        axis=-1,
    )
    component_valid = jnp.all(identity >= 0, axis=-1)
    return jnp.where(
        (mask & component_valid)[:, None], identity, -jnp.ones_like(identity)
    )


class ParticlePairRemap(StrictModule, NonTrainableState):
    """Deterministic route map from old interaction slots to new slots."""

    source_indices: Array
    continued: Array
    born: Array
    ended_count: Array
    successful: Array
    old_capacity: int = eqx.field(static=True)
    new_capacity: int = eqx.field(static=True)


def _validated_keys(
    name: str, keys: ArrayLike, valid: ArrayLike, /
) -> tuple[Array, Array, Array]:
    keys_ = jnp.asarray(keys)
    valid_ = jnp.asarray(valid, dtype=bool)
    if (
        keys_.ndim != 2
        or keys_.shape[1] != INTERACTION_KEY_WIDTH
        or valid_.shape != keys_.shape[:1]
    ):
        raise ValueError(f"{name} keys must have shape (routes,{INTERACTION_KEY_WIDTH}).")
    if not jnp.issubdtype(keys_.dtype, jnp.integer):
        raise TypeError(f"{name} keys must be integers.")
    keys_ = keys_.astype(jnp.int64)
    in_range = jnp.all(keys_ >= 0, axis=-1)
    return keys_, valid_, jnp.all(~valid_ | in_range)


def _lexicographic_order(keys: Array, valid: Array, /):
    sentinel = jnp.asarray(np.iinfo(np.int64).max, dtype=jnp.int64)
    sortable = jnp.where(valid[:, None], keys, sentinel)
    order = jnp.lexsort(
        tuple(sortable[:, index] for index in range(INTERACTION_KEY_WIDTH - 1, -1, -1))
    )
    return order, sortable[order], sentinel


def _lexicographic_less(left: Array, right: Array, /) -> Array:
    different = left != right
    first = jnp.argmax(different.astype(jnp.int32))
    return jnp.any(different) & (left[first] < right[first])


def _lexicographic_search(sorted_keys: Array, queries: Array, /) -> Array:
    capacity = int(sorted_keys.shape[0])
    if capacity == 0:
        return jnp.zeros((queries.shape[0],), dtype=jnp.int32)
    iterations = max(1, capacity.bit_length())

    def search(query):
        def iteration(_, bounds):
            lower, upper = bounds
            middle = (lower + upper) // 2
            safe_middle = jnp.minimum(middle, capacity - 1)
            less = _lexicographic_less(sorted_keys[safe_middle], query)
            active = lower < upper
            next_lower = jnp.where(active & less, middle + 1, lower)
            next_upper = jnp.where(active & less, upper, jnp.where(active, middle, upper))
            return next_lower, next_upper

        lower, _ = jax.lax.fori_loop(
            0,
            iterations,
            iteration,
            (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(capacity, dtype=jnp.int32)),
        )
        return lower

    return jax.vmap(search)(queries)


def match_particle_pair_keys(
    old_keys: ArrayLike,
    old_valid: ArrayLike,
    new_keys: ArrayLike,
    new_valid: ArrayLike,
    /,
) -> ParticlePairRemap:
    """Match structured identities without differentiating route decisions."""

    old, old_mask, old_in_range = _validated_keys("Old interaction", old_keys, old_valid)
    new, new_mask, new_in_range = _validated_keys("New interaction", new_keys, new_valid)
    old_capacity = int(old.shape[0])
    new_capacity = int(new.shape[0])
    if new_capacity == 0:
        return ParticlePairRemap(
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=bool),
            jnp.zeros((0,), dtype=bool),
            jnp.sum(old_mask, dtype=jnp.int32),
            old_in_range & new_in_range,
            old_capacity,
            0,
        )
    if old_capacity == 0:
        return ParticlePairRemap(
            jnp.zeros((new_capacity,), dtype=jnp.int32),
            jnp.zeros((new_capacity,), dtype=bool),
            new_mask,
            jnp.zeros((), dtype=jnp.int32),
            old_in_range & new_in_range,
            0,
            new_capacity,
        )
    old_order, old_sorted, sentinel = _lexicographic_order(old, old_mask)
    _, new_sorted, _ = _lexicographic_order(new, new_mask)
    duplicate_old = jnp.any(
        jnp.all(old_sorted[1:] == old_sorted[:-1], axis=-1)
        & jnp.any(old_sorted[1:] != sentinel, axis=-1)
    )
    duplicate_new = jnp.any(
        jnp.all(new_sorted[1:] == new_sorted[:-1], axis=-1)
        & jnp.any(new_sorted[1:] != sentinel, axis=-1)
    )
    positions = _lexicographic_search(old_sorted, new)
    safe_positions = jnp.clip(positions, 0, old_capacity - 1)
    continued = (
        new_mask
        & (positions < old_capacity)
        & jnp.all(old_sorted[safe_positions] == new, axis=-1)
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
            raise ValueError("Every pair-value leaf must use the old route capacity.")

    def gather(leaf):
        fill = jnp.asarray(fill_value, dtype=leaf.dtype)
        if remap.old_capacity == 0:
            return jnp.full(
                (remap.new_capacity,) + leaf.shape[1:],
                fill,
                dtype=leaf.dtype,
            )
        selected = leaf[remap.source_indices]
        mask = remap.continued.reshape((remap.new_capacity,) + (1,) * (leaf.ndim - 1))
        return jnp.where(mask, selected, fill)

    return jax.tree.map(gather, values)


__all__ = [
    "CLUMP_COMPONENT_INTERACTION",
    "IMPLICIT_BARRIER_INTERACTION",
    "INTERACTION_KEY_WIDTH",
    "PARTICLE_PAIR_INTERACTION",
    "PARTICLE_WALL_INTERACTION",
    "ParticlePairKeys",
    "ParticlePairKeySpace",
    "ParticlePairRemap",
    "match_particle_pair_keys",
    "particle_wall_interaction_keys",
    "remap_particle_pair_values",
]
