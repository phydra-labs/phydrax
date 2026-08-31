#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState
from ..discretization.particle._periodic_cell import ParticleCell
from ..graph import GraphIR
from ._system import PreparedAtomisticSystem
from ._types import AtomisticBatch


AtomisticGraphBackend: TypeAlias = Literal["dense", "particle"]


class AtomisticGraphExecutionPlan(StrictModule, NonTrainableState):
    """Resource and realization policy independent of learned model parameters."""

    maximum_neighbors: int = eqx.field(static=True)
    maximum_dense_atoms: int | None = eqx.field(static=True)
    backend: AtomisticGraphBackend = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_neighbors: int,
        /,
        *,
        backend: AtomisticGraphBackend = "dense",
        maximum_dense_atoms: int | None = None,
        plan_id: str | None = None,
    ):
        neighbors = int(maximum_neighbors)
        dense = None if maximum_dense_atoms is None else int(maximum_dense_atoms)
        if neighbors < 0:
            raise ValueError("maximum_neighbors must be non-negative.")
        if backend not in ("dense", "particle"):
            raise ValueError("backend must be 'dense' or 'particle'.")
        if backend == "dense" and (dense is None or dense <= 0):
            raise ValueError(
                "Dense graph execution requires positive maximum_dense_atoms."
            )
        if backend == "particle" and dense is not None:
            raise ValueError(
                "maximum_dense_atoms is valid only for dense graph execution."
            )
        generated = canonical_fingerprint(
            {
                "kind": "atomistic-graph-execution-plan",
                "maximum_neighbors": neighbors,
                "maximum_dense_atoms": dense,
                "backend": backend,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.maximum_neighbors = neighbors
        self.maximum_dense_atoms = dense
        self.backend = backend
        self.plan_id = identifier


class AtomisticGraph(StrictModule, NonTrainableState):
    """Fixed-capacity directed atomistic graph and complete overflow evidence."""

    graph: GraphIR
    neighbor_counts: Array
    maximum_neighbor_count: Array
    overflow: Array
    cutoff: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
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


def _edge_geometry(
    positions: Array,
    senders: Array,
    receivers: Array,
    endpoint_mask: Array,
    cell: ParticleCell | None,
    /,
) -> tuple[Array, Array, Array]:
    raw = positions[receivers] - positions[senders]
    if cell is not None:
        raw = cell.minimum_image(raw)
    displacement = jnp.where(endpoint_mask[:, None], raw, 0.0)
    squared = jnp.sum(displacement * displacement, axis=-1)
    tiny = jnp.asarray(jnp.finfo(positions.dtype).tiny, dtype=positions.dtype)
    positive = jnp.sqrt(jnp.maximum(squared, tiny))
    distance = jnp.where(squared > 0.0, positive, 0.0)
    safe = jnp.where(distance > 0.0, distance, 1.0)
    return displacement, distance, displacement / safe[:, None]


def _assemble_graph(
    *,
    atomic_numbers: Array,
    masses: Array,
    atom_mask: Array,
    atom_cases: Array,
    case_count: int,
    atom_capacity: int,
    senders: Array,
    receivers: Array,
    edge_cases: Array,
    candidate_valid: Array,
    positions: Array,
    cutoff: float,
    topology_id: str,
    execution: AtomisticGraphExecutionPlan,
    cell: ParticleCell | None,
) -> AtomisticGraph:
    displacement, distance, direction = _edge_geometry(
        positions, senders, receivers, candidate_valid, cell
    )
    edge_mask = candidate_valid & (distance < jnp.asarray(cutoff, positions.dtype))
    neighbor_counts_flat = (
        jnp.zeros((case_count * atom_capacity,), dtype=jnp.int32)
        .at[receivers]
        .add(edge_mask.astype(jnp.int32))
    )
    neighbor_counts = neighbor_counts_flat.reshape((case_count, atom_capacity))
    maximum_neighbor_count = jnp.max(neighbor_counts, axis=1)
    overflow = maximum_neighbor_count > execution.maximum_neighbors
    active_edges = (
        jnp.zeros((case_count,), dtype=jnp.int32)
        .at[edge_cases]
        .add(edge_mask.astype(jnp.int32))
    )
    edge_capacity = int(senders.shape[0] // case_count) if case_count else 0
    graph = GraphIR(
        nodes={
            "atomic_numbers": atomic_numbers,
            "masses": masses,
            "case_index": atom_cases,
        },
        edges={
            "displacement": displacement,
            "distance": distance[:, None],
            "direction": direction,
            "case_index": edge_cases,
        },
        senders=senders,
        receivers=receivers,
        globals={
            "active_edge_count": active_edges,
            "maximum_neighbor_count": maximum_neighbor_count,
            "overflow": overflow,
        },
        n_node=jnp.full((case_count,), atom_capacity, dtype=jnp.int32),
        n_edge=jnp.full((case_count,), edge_capacity, dtype=jnp.int32),
        node_mask=atom_mask,
        edge_mask=edge_mask,
        graph_mask=jnp.ones((case_count,), dtype=bool),
    )
    graph_id = canonical_fingerprint(
        {
            "kind": "atomistic-graph-realization",
            "topology": topology_id,
            "execution": execution.plan_id,
            "cutoff": float(cutoff),
        }
    )
    return AtomisticGraph(
        graph=graph,
        neighbor_counts=neighbor_counts,
        maximum_neighbor_count=maximum_neighbor_count,
        overflow=overflow,
        cutoff=float(cutoff),
        topology_id=topology_id,
        execution_id=execution.plan_id,
        graph_id=graph_id,
    )


def realize_atomistic_graph(
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    cutoff: float,
    positions: ArrayLike | None = None,
) -> AtomisticGraph:
    """Realize dense directed candidates under one explicit execution plan."""

    if not isinstance(batch, AtomisticBatch):
        raise TypeError("batch must be an AtomisticBatch.")
    if (
        not isinstance(execution, AtomisticGraphExecutionPlan)
        or execution.backend != "dense"
    ):
        raise TypeError("Dense realization requires a dense AtomisticGraphExecutionPlan.")
    cutoff_value = float(cutoff)
    if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    dense_limit = execution.maximum_dense_atoms
    if dense_limit is None:
        raise RuntimeError("Validated dense graph plan unexpectedly lacks a dense limit.")
    if batch.atom_capacity > dense_limit:
        raise ValueError(
            f"Dense atomistic graph capacity {batch.atom_capacity} exceeds the explicit "
            f"maximum_dense_atoms={dense_limit} resource guard."
        )
    atom_capacity = batch.atom_capacity
    case_count = batch.case_count
    edge_capacity = atom_capacity * (atom_capacity - 1)
    local_senders = np.repeat(np.arange(atom_capacity, dtype=np.int32), atom_capacity - 1)
    local_offsets = np.tile(np.arange(atom_capacity - 1, dtype=np.int32), atom_capacity)
    local_receivers = local_offsets + (local_offsets >= local_senders)
    case_offsets = np.repeat(
        np.arange(case_count, dtype=np.int32) * atom_capacity, edge_capacity
    )
    senders = jnp.asarray(np.tile(local_senders, case_count) + case_offsets)
    receivers = jnp.asarray(np.tile(local_receivers, case_count) + case_offsets)
    edge_cases = jnp.repeat(jnp.arange(case_count, dtype=jnp.int32), edge_capacity)
    coordinate = batch.positions if positions is None else jnp.asarray(positions)
    if coordinate.shape != batch.positions.shape:
        raise ValueError("positions must have the batch position shape.")
    flat_mask = batch.atom_mask.reshape((-1,))
    candidate_valid = flat_mask[senders] & flat_mask[receivers]
    return _assemble_graph(
        atomic_numbers=batch.atomic_numbers.reshape((-1,)),
        masses=batch.masses.reshape((-1,)),
        atom_mask=flat_mask,
        atom_cases=batch.atom_cases,
        case_count=case_count,
        atom_capacity=atom_capacity,
        senders=senders,
        receivers=receivers,
        edge_cases=edge_cases,
        candidate_valid=candidate_valid,
        positions=coordinate.reshape((-1, 3)),
        cutoff=cutoff_value,
        topology_id=batch.atom_topology_id,
        execution=execution,
        cell=None,
    )


def realize_particle_atomistic_graph(
    system: PreparedAtomisticSystem,
    neighborhood: ParticleNeighborhoodState,
    execution: AtomisticGraphExecutionPlan,
    positions: ArrayLike,
    /,
    *,
    cutoff: float,
    cell: ParticleCell | None = None,
) -> AtomisticGraph:
    """Expand one pair-once particle relation into a directed atomistic graph."""

    if not isinstance(system, PreparedAtomisticSystem):
        raise TypeError("system must be a PreparedAtomisticSystem.")
    if not isinstance(neighborhood, ParticleNeighborhoodState):
        raise TypeError("neighborhood must be a ParticleNeighborhoodState.")
    if (
        not isinstance(execution, AtomisticGraphExecutionPlan)
        or execution.backend != "particle"
    ):
        raise TypeError("Particle realization requires a particle graph execution plan.")
    coordinate = jnp.asarray(positions)
    expected = (system.capacity, 3)
    if coordinate.shape != expected:
        raise ValueError(f"positions must have shape {expected}.")
    pairs = neighborhood.pair_relation
    left = pairs.left_indices
    right = pairs.right_indices
    senders = jnp.concatenate((left, right))
    receivers = jnp.concatenate((right, left))
    candidate_valid = jnp.concatenate((pairs.valid, pairs.valid))
    edge_cases = jnp.zeros((senders.shape[0],), dtype=jnp.int32)
    atom_cases = jnp.zeros((system.capacity,), dtype=jnp.int32)
    return _assemble_graph(
        atomic_numbers=system.plan.atomic_numbers,
        masses=system.plan.masses,
        atom_mask=system.active_mask,
        atom_cases=atom_cases,
        case_count=1,
        atom_capacity=system.capacity,
        senders=senders,
        receivers=receivers,
        edge_cases=edge_cases,
        candidate_valid=candidate_valid,
        positions=coordinate,
        cutoff=float(cutoff),
        topology_id=pairs.relation_schema_id,
        execution=execution,
        cell=cell,
    )


__all__ = [
    "AtomisticGraph",
    "AtomisticGraphExecutionPlan",
    "realize_atomistic_graph",
    "realize_particle_atomistic_graph",
]
