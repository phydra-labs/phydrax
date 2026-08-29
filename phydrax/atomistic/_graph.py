#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..graph import GraphIR
from ._types import AtomisticBatch


class AtomisticGraph(StrictModule, NonTrainableState):
    """Dense-candidate, cutoff-masked, case-isolated molecular graph realization."""

    graph: GraphIR
    neighbor_counts: Array
    maximum_neighbor_count: Array
    overflow: Array
    cutoff: float = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    maximum_dense_atoms: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return ~self.overflow

    def require_success(self, value: ArrayLike, /) -> Array:
        return eqx.error_if(
            jnp.asarray(value),
            jnp.any(self.overflow),
            "Atomistic neighborhood capacity overflow; prediction is not valid.",
        )


def realize_atomistic_graph(
    batch: AtomisticBatch,
    /,
    *,
    cutoff: float,
    maximum_neighbors: int,
    maximum_dense_atoms: int,
    positions: ArrayLike | None = None,
) -> AtomisticGraph:
    """Realize all directed candidates without truncation, then mask by cutoff.

    Candidate topology is fixed by the batch capacity. ``maximum_neighbors`` is a
    fail-closed runtime capacity contract: overflow is reported and never repaired
    by dropping edges. The explicit dense-atom guard prevents accidental quadratic
    allocation outside the declared resource envelope.
    """

    if not isinstance(batch, AtomisticBatch):
        raise TypeError("batch must be an AtomisticBatch.")
    cutoff_value = float(cutoff)
    neighbor_limit = int(maximum_neighbors)
    dense_limit = int(maximum_dense_atoms)
    if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if neighbor_limit < 0:
        raise ValueError("maximum_neighbors must be non-negative.")
    if dense_limit <= 0:
        raise ValueError("maximum_dense_atoms must be positive.")
    if batch.atom_capacity > dense_limit:
        raise ValueError(
            f"Dense atomistic graph capacity {batch.atom_capacity} exceeds the explicit "
            f"maximum_dense_atoms={dense_limit} resource guard."
        )
    coordinate = batch.positions if positions is None else jnp.asarray(positions)
    if coordinate.shape != batch.positions.shape:
        raise ValueError("positions must have the batch position shape.")
    flat_position = coordinate.reshape((-1, 3))
    flat_numbers = batch.atomic_numbers.reshape((-1,))
    flat_masses = batch.masses.reshape((-1,))
    flat_mask = batch.atom_mask.reshape((-1,))
    displacement = flat_position[batch.receivers] - flat_position[batch.senders]
    squared_distance = jnp.sum(displacement * displacement, axis=-1)
    tiny = jnp.asarray(jnp.finfo(coordinate.dtype).tiny, dtype=coordinate.dtype)
    positive_distance = jnp.sqrt(jnp.maximum(squared_distance, tiny))
    distance = jnp.where(squared_distance > 0.0, positive_distance, 0.0)
    safe_distance = jnp.where(distance > 0.0, distance, 1.0)
    direction = displacement / safe_distance[:, None]
    endpoint_mask = flat_mask[batch.senders] & flat_mask[batch.receivers]
    edge_mask = endpoint_mask & (distance < jnp.asarray(cutoff_value, coordinate.dtype))
    neighbor_counts = jnp.zeros(
        (batch.case_count * batch.atom_capacity,), dtype=jnp.int32
    ).at[batch.receivers].add(edge_mask.astype(jnp.int32))
    neighbor_counts = neighbor_counts.reshape(
        (batch.case_count, batch.atom_capacity)
    )
    maximum_neighbor_count = jnp.max(neighbor_counts, axis=1)
    overflow = maximum_neighbor_count > neighbor_limit
    active_edges = jnp.zeros((batch.case_count,), dtype=jnp.int32).at[
        batch.edge_cases
    ].add(edge_mask.astype(jnp.int32))
    edge_capacity = batch.atom_capacity * (batch.atom_capacity - 1)
    graph = GraphIR(
        nodes={
            "atomic_numbers": flat_numbers,
            "masses": flat_masses,
            "case_index": batch.atom_cases,
        },
        edges={
            "displacement": displacement,
            "distance": distance[:, None],
            "direction": direction,
            "case_index": batch.edge_cases,
        },
        senders=batch.senders,
        receivers=batch.receivers,
        globals={
            "active_edge_count": active_edges,
            "maximum_neighbor_count": maximum_neighbor_count,
            "overflow": overflow,
        },
        n_node=jnp.full((batch.case_count,), batch.atom_capacity, dtype=jnp.int32),
        n_edge=jnp.full((batch.case_count,), edge_capacity, dtype=jnp.int32),
        node_mask=flat_mask,
        edge_mask=edge_mask,
        graph_mask=jnp.ones((batch.case_count,), dtype=bool),
    )
    graph_id = canonical_fingerprint(
        {
            "kind": "atomistic-graph-realization",
            "candidate_topology": batch.candidate_topology_id,
            "cutoff": cutoff_value,
            "maximum_neighbors": neighbor_limit,
            "maximum_dense_atoms": dense_limit,
        }
    )
    return AtomisticGraph(
        graph=graph,
        neighbor_counts=neighbor_counts,
        maximum_neighbor_count=maximum_neighbor_count,
        overflow=overflow,
        cutoff=cutoff_value,
        maximum_neighbors=neighbor_limit,
        maximum_dense_atoms=dense_limit,
        topology_id=batch.candidate_topology_id,
        graph_id=graph_id,
    )


__all__ = ["AtomisticGraph", "realize_atomistic_graph"]
