#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._contraction import ContractionResourcePolicy, plan_contraction
from ._precision import TensorNetworkPrecisionPolicy
from ._topology import ContractionStructure


class TreeTensorNetwork(StrictModule):
    structure: ContractionStructure
    tensors: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure: ContractionStructure,
        tensors: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        if not isinstance(structure, ContractionStructure):
            raise TypeError("structure must be ContractionStructure.")
        arrays = tuple(jnp.asarray(value) for value in tensors)
        if len(arrays) != len(structure.operands):
            raise ValueError("Tree tensor count must equal topology node count.")
        for array, operand in zip(arrays, structure.operands, strict=True):
            if array.shape != tuple(leg.dimension for leg in operand.legs):
                raise ValueError("A tree tensor shape differs from its node incidences.")
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        precision_.validate_storage(arrays)
        _validate_tree(structure)
        self.structure = structure
        self.tensors = arrays
        self.precision = precision_
        self.network_id = canonical_fingerprint(
            {
                "kind": "tree-tensor-network",
                "structure": structure.structure_id,
                "dtype": str(arrays[0].dtype),
                "precision": precision_.policy_id,
            }
        )


class TreeMessageEvidence(StrictModule):
    network_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    message_step_ids: tuple[str, ...] = eqx.field(static=True)
    message_norms: Array
    finite: Array
    accepted: Array
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    peak_live_bytes: int = eqx.field(static=True)


class TreeContractionResult(StrictModule):
    value: Array
    evidence: TreeMessageEvidence


def _validate_tree(structure: ContractionStructure, /) -> None:
    incidences: dict[str, list[int]] = {}
    for index, operand in enumerate(structure.operands):
        for leg in operand.legs:
            incidences.setdefault(leg.label, []).append(index)
    edges = []
    for label, sites in incidences.items():
        if label in structure.outputs or len(sites) == 1:
            continue
        if len(sites) != 2 or sites[0] == sites[1]:
            raise ValueError(
                "Tree networks require every internal edge to join two distinct nodes."
            )
        edges.append((sites[0], sites[1]))
    if len(edges) != len(structure.operands) - 1:
        raise ValueError("Tree topology must have node_count - 1 internal edges.")
    reached = {0}
    changed = True
    while changed:
        changed = False
        for left, right in edges:
            if left in reached and right not in reached:
                reached.add(right)
                changed = True
            elif right in reached and left not in reached:
                reached.add(left)
                changed = True
    if len(reached) != len(structure.operands):
        raise ValueError("Tree topology must be connected.")


def contract_tree_messages(
    network: TreeTensorNetwork,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> TreeContractionResult:
    """Exactly eliminate a tree through its deterministic SSA message schedule."""

    if not isinstance(network, TreeTensorNetwork):
        raise TypeError("network must be TreeTensorNetwork.")
    plan = plan_contraction(
        network.structure,
        precision=network.precision,
        resources=resources,
        optimizer="greedy",
        dtype=str(network.tensors[0].dtype),
    )
    values = list(plan.precision.contraction(network.tensors))
    value_ids = list(plan.schedule.initial_value_ids)
    message_norms = []
    for step in plan.schedule.steps:
        positions = tuple(value_ids.index(value_id) for value_id in step.input_value_ids)
        result = ein.contract(
            step.equation,
            *(values[position] for position in positions),
            optimize=False,
        )
        message_norms.append(jnp.linalg.norm(result))
        for position in sorted(positions, reverse=True):
            values.pop(position)
            value_ids.pop(position)
        values.append(result)
        value_ids.append(step.output_value_id)
    if value_ids != [plan.schedule.output_value_id]:
        raise RuntimeError("Tree message replay did not reach its declared root value.")
    value = plan.precision.output(values[0])
    norms = (
        jnp.stack(tuple(message_norms))
        if message_norms
        else jnp.zeros((0,), dtype=jnp.real(value).dtype)
    )
    finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(norms))
    replay_id = canonical_fingerprint(
        {
            "kind": "exact-tree-message-replay",
            "network": network.network_id,
            "schedule": plan.schedule.schedule_id,
        }
    )
    evidence = TreeMessageEvidence(
        network.network_id,
        plan.plan_id,
        replay_id,
        tuple(step.step_id for step in plan.schedule.steps),
        norms,
        finite,
        finite,
        True,
        "exact finite-tree sum-product message elimination",
        plan.schedule.peak_live_bytes,
    )
    return TreeContractionResult(value, evidence)


__all__ = [
    "TreeContractionResult",
    "TreeMessageEvidence",
    "TreeTensorNetwork",
    "contract_tree_messages",
]
