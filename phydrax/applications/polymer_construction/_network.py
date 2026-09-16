#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ._reactions import PolymerReactionKind, PolymerReactionState


class PolymerNetworkResult(StrictModule):
    conversion: Array
    available_ports: Array
    component_count: Array
    largest_component_size: Array
    largest_component_fraction: Array
    cycle_rank: Array
    periodic_spanning: Array
    accepted_events: Array
    refused_events: Array
    successful: Array
    state_id: str = eqx.field(static=True)


def polymer_network_observables(state: PolymerReactionState, /) -> PolymerNetworkResult:
    if not isinstance(state, PolymerReactionState):
        raise TypeError("state must be PolymerReactionState.")
    system = state.system
    active = np.asarray(system.active_mask, dtype=bool)
    ids = np.asarray(system.particle_ids, dtype=np.int64)[active]
    index = {int(value): position for position, value in enumerate(ids)}
    parent = np.arange(ids.size, dtype=np.int32)
    size = np.ones((ids.size,), dtype=np.int32)
    rank = int(state.image_counts.shape[1])
    offset = np.zeros((ids.size, rank), dtype=np.int64)
    spanning = False

    def find(node: int) -> tuple[int, np.ndarray]:
        if parent[node] == node:
            return node, np.zeros((rank,), dtype=np.int64)
        root, upstream = find(int(parent[node]))
        total = offset[node] + upstream
        parent[node] = root
        offset[node] = total
        return root, total

    winding_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for event in state.ledger:
        if not event.accepted:
            continue
        pair = tuple(sorted((event.left_particle_id, event.right_particle_id)))
        if event.reaction_kind is PolymerReactionKind.CURE:
            winding_by_pair[pair] = np.asarray(event.image_shift, dtype=np.int64)
        else:
            winding_by_pair.pop(pair, None)
    bonds = np.asarray(system.topology.bonds, dtype=np.int64)
    for left_id, right_id in bonds:
        left = index[int(left_id)]
        right = index[int(right_id)]
        shift = winding_by_pair.get(
            tuple(sorted((int(left_id), int(right_id)))),
            np.zeros((rank,), dtype=np.int64),
        )
        left_root, left_offset = find(left)
        right_root, right_offset = find(right)
        if left_root == right_root:
            if rank and np.any((right_offset - left_offset) != shift):
                spanning = True
            continue
        if size[left_root] < size[right_root]:
            parent[left_root] = right_root
            offset[left_root] = right_offset - left_offset - shift
            size[right_root] += size[left_root]
        else:
            parent[right_root] = left_root
            offset[right_root] = shift + left_offset - right_offset
            size[left_root] += size[right_root]
    roots = np.asarray([find(value)[0] for value in range(ids.size)], dtype=np.int32)
    unique, counts = np.unique(roots, return_counts=True)
    components = int(unique.size)
    largest = int(np.max(counts)) if counts.size else 0
    cycle_rank = int(bonds.shape[0] - ids.size + components)
    available = sum(port.maximum_uses - port.uses for port in state.ports)
    used = state.initial_port_capacity - available
    accepted = sum(event.accepted for event in state.ledger)
    refused = len(state.ledger) - accepted
    successful = (
        0 <= used <= state.initial_port_capacity and cycle_rank >= 0 and components >= 1
    )
    return PolymerNetworkResult(
        jnp.asarray(used / state.initial_port_capacity),
        jnp.asarray(available, dtype=jnp.int32),
        jnp.asarray(components, dtype=jnp.int32),
        jnp.asarray(largest, dtype=jnp.int32),
        jnp.asarray(largest / ids.size),
        jnp.asarray(cycle_rank, dtype=jnp.int32),
        jnp.asarray(spanning),
        jnp.asarray(accepted, dtype=jnp.int32),
        jnp.asarray(refused, dtype=jnp.int32),
        jnp.asarray(successful),
        state.state_id,
    )


__all__ = ["PolymerNetworkResult", "polymer_network_observables"]
