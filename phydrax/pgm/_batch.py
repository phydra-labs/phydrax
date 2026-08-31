#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..graph import batch_graphs, GraphIR
from ._belief_propagation import (
    BeliefPropagationResult,
    BeliefPropagationState,
    PreparedBeliefPropagation,
    run_belief_propagation,
)
from ._exact import enumerate_factor_graph
from ._gibbs import (
    GibbsSampleResult,
    GibbsSchedule,
    GibbsState,
    PreparedChromaticGibbs,
    sample_gibbs,
)
from ._model import DiscreteFactorGraph, VariableStateValues
from ._types import ExactFactorGraphResult


class BatchedBeliefPropagationState(StrictModule):
    """Same-topology case batch of messages and unary evidence."""

    messages: Array
    evidence: Array
    step_index: Array
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        messages: ArrayLike,
        evidence: ArrayLike,
        /,
        *,
        structure_id: str,
        step_index: ArrayLike | int = 0,
    ):
        message_values = jnp.asarray(messages)
        evidence_values = jnp.asarray(evidence)
        if message_values.ndim != 2 or evidence_values.ndim != 2:
            raise ValueError("Batched BP messages/evidence require leading case axes.")
        if int(message_values.shape[0]) != int(evidence_values.shape[0]):
            raise ValueError("Batched BP messages and evidence must share case count.")
        index = jnp.asarray(step_index, dtype=jnp.int32)
        if index.shape not in ((), (int(message_values.shape[0]),)):
            raise ValueError("step_index must be scalar or one value per case.")
        self.messages = message_values
        self.evidence = evidence_values
        self.step_index = index
        self.structure_id = structure_id

    @property
    def num_cases(self) -> int:
        return int(self.messages.shape[0])


class BatchedBeliefPropagationResult(StrictModule):
    """Case-preserving PyTree stack of independent same-topology BP results."""

    results: BeliefPropagationResult
    num_cases: int = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


class PackedFactorGraphBatch(StrictModule):
    """Heterogeneous graph collection with block-diagonal topology and ownership offsets."""

    graphs: tuple[DiscreteFactorGraph, ...]
    topology: GraphIR
    variable_offsets: Array
    factor_offsets: Array
    incidence_offsets: Array
    batch_id: str = eqx.field(static=True)

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)


class FactorGraphShardingPolicy(StrictModule):
    """Explicit supported sharding axis for case, chain, or independent graph batches."""

    axis: Literal["case", "chain", "graph", "replicated"] = eqx.field(static=True)
    device_count: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: Literal["case", "chain", "graph", "replicated"] = "replicated",
        /,
        *,
        device_count: int = 1,
    ):
        if axis not in ("case", "chain", "graph", "replicated"):
            raise ValueError("Unknown factor-graph sharding axis.")
        count = int(device_count)
        if count < 1:
            raise ValueError("device_count must be positive.")
        self.axis = axis
        self.device_count = count
        self.policy_id = canonical_fingerprint(
            {"kind": "factor-graph-sharding", "axis": axis, "devices": count}
        )


def batch_belief_propagation(
    prepared: PreparedBeliefPropagation,
    state: BatchedBeliefPropagationState,
    /,
) -> BatchedBeliefPropagationResult:
    """Execute one prepared topology over native leading case axes."""
    if not isinstance(prepared, PreparedBeliefPropagation):
        raise TypeError("prepared must be PreparedBeliefPropagation.")
    if not isinstance(state, BatchedBeliefPropagationState):
        raise TypeError("state must be BatchedBeliefPropagationState.")
    if state.structure_id != prepared.graph.structure_id:
        raise ValueError("Batched state structure does not match the prepared graph.")
    if state.messages.shape[1:] != (prepared.message_count,):
        raise ValueError("Batched message width does not match the prepared plan.")
    if state.evidence.shape[1:] != (int(prepared.state_variable_indices.shape[0]),):
        raise ValueError("Batched evidence width does not match the prepared graph.")

    def one(messages, evidence, step):
        return run_belief_propagation(
            prepared,
            BeliefPropagationState(
                messages,
                VariableStateValues(evidence, structure_id=prepared.graph.structure_id),
                step_index=step,
            ),
        )

    steps = (
        jnp.broadcast_to(state.step_index, (state.num_cases,))
        if state.step_index.shape == ()
        else state.step_index
    )
    results = jax.vmap(one)(state.messages, state.evidence, steps)
    return BatchedBeliefPropagationResult(
        results=results,
        num_cases=state.num_cases,
        structure_id=prepared.graph.structure_id,
    )


def pack_factor_graphs(
    graphs: Sequence[DiscreteFactorGraph], /
) -> PackedFactorGraphBatch:
    """Pack heterogeneous topology block-diagonally while retaining each semantic graph."""
    values = tuple(graphs)
    if not values or any(not isinstance(graph, DiscreteFactorGraph) for graph in values):
        raise ValueError(
            "graphs must be a nonempty sequence of DiscreteFactorGraph values."
        )
    topology = batch_graphs([graph.topology.graph for graph in values])
    variable_offsets = [0]
    factor_offsets = [0]
    incidence_offsets = [0]
    for graph in values:
        variable_offsets.append(variable_offsets[-1] + graph.num_variables)
        factor_offsets.append(factor_offsets[-1] + graph.num_factors)
        incidence_offsets.append(
            incidence_offsets[-1] + int(graph.topology.incidence_edges.shape[0])
        )
    batch_id = canonical_fingerprint(
        {
            "kind": "packed-factor-graph-batch",
            "graphs": [graph.structure_id for graph in values],
        }
    )
    return PackedFactorGraphBatch(
        graphs=values,
        topology=topology,
        variable_offsets=jnp.asarray(variable_offsets, dtype=jnp.int32),
        factor_offsets=jnp.asarray(factor_offsets, dtype=jnp.int32),
        incidence_offsets=jnp.asarray(incidence_offsets, dtype=jnp.int32),
        batch_id=batch_id,
    )


def enumerate_packed_factor_graphs(
    batch: PackedFactorGraphBatch,
    /,
    *,
    max_configurations: int = 65_536,
) -> tuple[ExactFactorGraphResult, ...]:
    """Return per-graph exact results without allowing cross-graph state coupling."""
    if not isinstance(batch, PackedFactorGraphBatch):
        raise TypeError("batch must be PackedFactorGraphBatch.")
    return tuple(
        enumerate_factor_graph(graph, max_configurations=max_configurations)
        for graph in batch.graphs
    )


def sample_gibbs_per_chain_clamps(
    prepared: PreparedChromaticGibbs,
    state: GibbsState,
    /,
    *,
    key: Key[Array, ""],
    schedule: GibbsSchedule,
    clamped: ArrayLike,
) -> tuple[GibbsSampleResult, ...]:
    """Run independent persistent chains with distinct clamped-site masks."""
    masks = jnp.asarray(clamped, dtype=bool)
    expected = (state.num_chains, prepared.graph.num_variables)
    if masks.shape != expected:
        raise ValueError(f"clamped must have shape {expected}; got {masks.shape}.")
    keys = jax.random.split(key, state.num_chains)
    outputs = []
    for chain in range(state.num_chains):
        chain_state = GibbsState(
            state.positions[chain : chain + 1],
            state.log_score[chain : chain + 1],
            valid=state.valid[chain : chain + 1],
            sweep_index=state.sweep_index,
        )
        outputs.append(
            sample_gibbs(
                prepared,
                chain_state,
                key=keys[chain],
                schedule=schedule,
                clamped=masks[chain],
            )
        )
    return tuple(outputs)


__all__ = [
    "BatchedBeliefPropagationResult",
    "BatchedBeliefPropagationState",
    "FactorGraphShardingPolicy",
    "PackedFactorGraphBatch",
    "batch_belief_propagation",
    "enumerate_packed_factor_graphs",
    "pack_factor_graphs",
    "sample_gibbs_per_chain_clamps",
]
