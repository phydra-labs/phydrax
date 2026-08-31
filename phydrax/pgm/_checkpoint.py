#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp

from .._array_archive import read_array_archive, write_array_archive
from .._strict import StrictModule
from ._belief_propagation import BeliefPropagationState
from ._gibbs import GibbsState
from ._model import (
    BinaryCardinalityFactorGroup,
    DenseTableFactorGroup,
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    EnumeratedFactorGroup,
    IsingFactorGroup,
    KernelFactorGroup,
    LogicalFactorGroup,
    PottsFactorGroup,
    VariableSelection,
    VariableStateValues,
)


class FactorGraphCheckpoint(StrictModule):
    graph: DiscreteFactorGraph
    belief_state: BeliefPropagationState | None
    gibbs_state: GibbsState | None
    path: str


def _selection_manifest(group_index, selections, arrays):
    output = []
    for position, selection in enumerate(selections):
        name = f"factor_{group_index}_selection_{position}"
        arrays[name] = selection.indices
        output.append({"group": selection.group_name, "indices": name})
    return output


def write_factor_graph_checkpoint(
    path: str | Path,
    graph: DiscreteFactorGraph,
    /,
    *,
    belief_state: BeliefPropagationState | None = None,
    gibbs_state: GibbsState | None = None,
) -> Path:
    """Write a pickle-free graph and optional persistent inference states."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be DiscreteFactorGraph.")
    if belief_state is not None and (
        not isinstance(belief_state, BeliefPropagationState)
        or belief_state.evidence.structure_id != graph.structure_id
    ):
        raise ValueError("belief_state must match the checkpoint graph.")
    if gibbs_state is not None and (
        not isinstance(gibbs_state, GibbsState)
        or gibbs_state.positions.shape[1:] != (graph.num_variables,)
    ):
        raise ValueError("gibbs_state must match the checkpoint graph.")
    arrays: dict[str, Any] = {}
    variables = []
    for index, group in enumerate(graph.variable_groups):
        name = f"variable_{index}_cardinalities"
        arrays[name] = group.cardinalities
        variables.append(
            {
                "name": group.name,
                "shape": list(group.shape),
                "cardinalities": name,
            }
        )
    factors = []
    for index, group in enumerate(graph.factor_groups):
        if isinstance(group, KernelFactorGroup):
            raise TypeError(
                "Callable custom kernels cannot be restored from a neutral archive."
            )
        record = {
            "selections": _selection_manifest(index, group.selections, arrays),
        }
        if isinstance(group, DenseTableFactorGroup):
            record["kind"] = "dense"
            record["parameters"] = f"factor_{index}_log_potentials"
            arrays[record["parameters"]] = group.log_potentials
        elif isinstance(group, EnumeratedFactorGroup):
            record["kind"] = "enumerated"
            record["configurations"] = f"factor_{index}_configurations"
            record["parameters"] = f"factor_{index}_log_potentials"
            arrays[record["configurations"]] = group.configurations
            arrays[record["parameters"]] = group.log_potentials
        elif isinstance(group, IsingFactorGroup):
            record["kind"] = "ising"
            record["parameters"] = f"factor_{index}_weights"
            arrays[record["parameters"]] = group.weights
        elif isinstance(group, PottsFactorGroup):
            record["kind"] = "potts"
            record["parameters"] = f"factor_{index}_log_potentials"
            arrays[record["parameters"]] = group.log_potentials
        elif isinstance(group, LogicalFactorGroup):
            record["kind"] = "logical"
            record["logical_kind"] = group.kind
        elif isinstance(group, BinaryCardinalityFactorGroup):
            record["kind"] = "cardinality"
            record["parameters"] = f"factor_{index}_count_potentials"
            arrays[record["parameters"]] = group.log_count_potentials
        else:
            raise TypeError("Unsupported factor group in checkpoint.")
        factors.append(record)
    states: dict[str, Any] = {}
    if belief_state is not None:
        arrays["belief_messages"] = belief_state.messages
        arrays["belief_evidence"] = belief_state.evidence.values
        arrays["belief_step"] = belief_state.step_index
        states["belief"] = True
    if gibbs_state is not None:
        arrays["gibbs_positions"] = gibbs_state.positions
        arrays["gibbs_log_score"] = gibbs_state.log_score
        arrays["gibbs_valid"] = gibbs_state.valid
        arrays["gibbs_step"] = gibbs_state.sweep_index
        states["gibbs"] = True
    return write_array_archive(
        path,
        manifest={
            "kind": "factor-graph-checkpoint",
            "structure_id": graph.structure_id,
            "variables": variables,
            "factors": factors,
            "states": states,
        },
        arrays=arrays,
    )


def read_factor_graph_checkpoint(path: str | Path, /) -> FactorGraphCheckpoint:
    """Restore and checksum-validate a neutral factor-graph checkpoint."""
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "factor-graph-checkpoint":
        raise ValueError("Archive is not a factor-graph checkpoint.")
    variables = tuple(
        DiscreteVariableGroup(
            record["name"],
            shape=tuple(record["shape"]),
            num_states=arrays[record["cardinalities"]],
        )
        for record in manifest["variables"]
    )
    lookup = {group.name: group for group in variables}
    factors = []
    for record in manifest["factors"]:
        selections = tuple(
            VariableSelection(
                lookup[item["group"]],
                arrays[item["indices"]],
            )
            for item in record["selections"]
        )
        kind = record["kind"]
        if kind == "dense":
            group = DenseTableFactorGroup(selections, arrays[record["parameters"]])
        elif kind == "enumerated":
            group = EnumeratedFactorGroup(
                selections,
                arrays[record["configurations"]],
                arrays[record["parameters"]],
            )
        elif kind == "ising":
            group = IsingFactorGroup(selections, arrays[record["parameters"]])
        elif kind == "potts":
            group = PottsFactorGroup(selections, arrays[record["parameters"]])
        elif kind == "logical":
            group = LogicalFactorGroup(
                selections[:-1],
                selections[-1],
                kind=record["logical_kind"],
            )
        elif kind == "cardinality":
            group = BinaryCardinalityFactorGroup(
                selections,
                arrays[record["parameters"]],
            )
        else:
            raise ValueError(f"Unknown archived factor kind {kind!r}.")
        factors.append(group)
    graph = DiscreteFactorGraph(variables, tuple(factors))
    if graph.structure_id != manifest["structure_id"]:
        raise ValueError("Restored factor-graph structure identity mismatch.")
    states = manifest["states"]
    belief_state = None
    if states.get("belief"):
        belief_state = BeliefPropagationState(
            jnp.asarray(arrays["belief_messages"]),
            VariableStateValues(
                arrays["belief_evidence"],
                structure_id=graph.structure_id,
            ),
            step_index=arrays["belief_step"],
        )
    gibbs_state = None
    if states.get("gibbs"):
        gibbs_state = GibbsState(
            arrays["gibbs_positions"],
            arrays["gibbs_log_score"],
            valid=arrays["gibbs_valid"],
            sweep_index=arrays["gibbs_step"],
        )
    return FactorGraphCheckpoint(
        graph=graph,
        belief_state=belief_state,
        gibbs_state=gibbs_state,
        path=str(path),
    )


__all__ = [
    "FactorGraphCheckpoint",
    "read_factor_graph_checkpoint",
    "write_factor_graph_checkpoint",
]
