#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...pgm import (
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    EnumeratedFactorGroup,
    VariableSelection,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class RegulatoryStatus(IntEnum):
    """Portable status for exact finite-state regulatory operations."""

    SUCCESS = 0
    INVALID_STATE = 1
    RESOURCE_LIMIT = 2
    INCOMPLETE_RULES = 3


class RegulatoryCapacityError(ValueError):
    """Raised before truth-table materialization when declared capacity is exceeded."""


class RegulatoryRule(StrictModule):
    """One synchronous deterministic finite-state update represented by a truth table."""

    target_index: Array
    parent_indices: Array
    truth_table: Array
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_index: int | ArrayLike,
        parent_indices: ArrayLike,
        truth_table: ArrayLike,
        /,
        *,
        rule_id: str,
    ):
        target = jnp.asarray(target_index, dtype=jnp.int32)
        parents = jnp.asarray(parent_indices, dtype=jnp.int32).reshape((-1,))
        table = jnp.asarray(truth_table)
        if target.shape != ():
            raise ValueError("target_index must be scalar.")
        if table.ndim != 1 or table.shape[0] != 2 ** parents.shape[0]:
            raise ValueError("truth_table must contain exactly 2**num_parents entries.")
        if len(set(np.asarray(parents).tolist())) != parents.shape[0]:
            raise ValueError("parent_indices must be unique.")
        table = table.astype(jnp.int32)
        table = eqx.error_if(
            table,
            jnp.any((table != 0) & (table != 1)),
            "Regulatory truth-table entries must be binary.",
        )
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError("rule_id must be a non-empty string.")
        self.target_index = target
        self.parent_indices = parents
        self.truth_table = table
        self.rule_id = rule_id.strip()

    def evaluate(self, state: ArrayLike, /) -> Array:
        state_ = jnp.asarray(state, dtype=jnp.int32)
        parent_state = state_[self.parent_indices]
        powers = 2 ** jnp.arange(
            self.parent_indices.shape[0] - 1, -1, -1, dtype=jnp.int32
        )
        table_index = jnp.sum(parent_state * powers, dtype=jnp.int32)
        return self.truth_table[table_index]


class RegulatoryTransitionEvidence(StrictModule):
    """Exact synchronous update and cycle-preserving semantic evidence."""

    input_state: Array
    output_state: Array
    applied_rule_mask: Array
    has_regulatory_cycles: Array
    synchronous: bool = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class RegulatoryTransitionResult(StrictModule):
    """One exact synchronous regulatory transition."""

    valid: Array
    status: Array
    state: Array
    evidence: RegulatoryTransitionEvidence
    method_contract: BioinformaticsMethodContract
    network_id: str = eqx.field(static=True)


class RegulatoryPGMEvidence(StrictModule):
    """Exact relation-factor lowering evidence, including capacity accounting."""

    supported_configurations: Array
    rule_configuration_counts: Array
    has_regulatory_cycles: Array
    synchronous_two_slice: bool = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    pgm_substrate: str = eqx.field(static=True)


class RegulatoryPGMResult(StrictModule):
    """Exact two-slice transition relation lowered to a native discrete factor graph."""

    valid: Array
    status: Array
    factor_graph: DiscreteFactorGraph
    evidence: RegulatoryPGMEvidence
    method_contract: BioinformaticsMethodContract
    network_id: str = eqx.field(static=True)


class DiscreteRegulatoryNetwork(StrictModule):
    """Named binary variables and simultaneous update rules, including directed cycles."""

    rules: tuple[RegulatoryRule, ...]
    regulated_mask: Array
    node_ids: tuple[str, ...] = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    has_cycles: bool = eqx.field(static=True)

    def __init__(
        self,
        node_ids: tuple[str, ...] | list[str],
        rules: tuple[RegulatoryRule, ...] | list[RegulatoryRule],
        /,
    ):
        identifiers = tuple(str(value).strip() for value in node_ids)
        if not identifiers or any(not value for value in identifiers):
            raise ValueError("node_ids must contain non-empty strings.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("node_ids must be unique.")
        rules_ = tuple(rules)
        if any(not isinstance(rule, RegulatoryRule) for rule in rules_):
            raise TypeError("rules must contain RegulatoryRule values.")
        targets = [int(np.asarray(rule.target_index)) for rule in rules_]
        if len(set(targets)) != len(targets):
            raise ValueError("At most one update rule may target each node.")
        size = len(identifiers)
        for rule in rules_:
            target = int(np.asarray(rule.target_index))
            parents = np.asarray(rule.parent_indices)
            if target < 0 or target >= size:
                raise ValueError("A regulatory target index is outside node capacity.")
            if parents.size and (parents.min() < 0 or parents.max() >= size):
                raise ValueError("A regulatory parent index is outside node capacity.")
        adjacency = np.zeros((size, size), dtype=bool)
        for rule in rules_:
            adjacency[
                np.asarray(rule.parent_indices), int(np.asarray(rule.target_index))
            ] = True
        reachability = adjacency.copy()
        for intermediate in range(size):
            reachability |= (
                reachability[:, intermediate, None] & reachability[None, intermediate, :]
            )
        has_cycles = bool(np.any(np.diag(reachability)))
        regulated = np.zeros((size,), dtype=bool)
        regulated[targets] = True
        self.rules = rules_
        self.regulated_mask = jnp.asarray(regulated)
        self.node_ids = identifiers
        self.has_cycles = has_cycles
        self.network_id = canonical_fingerprint(
            {
                "kind": "discrete-regulatory-network",
                "nodes": list(identifiers),
                "rules": [
                    {
                        "id": rule.rule_id,
                        "target": int(np.asarray(rule.target_index)),
                        "parents": np.asarray(rule.parent_indices).tolist(),
                        "truth_table": np.asarray(rule.truth_table).tolist(),
                    }
                    for rule in rules_
                ],
                "update": "synchronous",
            }
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def step(self, state: ArrayLike, /) -> RegulatoryTransitionResult:
        """Apply all rules simultaneously; unregulated nodes retain their state."""

        state_ = jnp.asarray(state, dtype=jnp.int32)
        if state_.shape != (self.num_nodes,):
            raise ValueError("state must contain one binary value per node.")
        binary = jnp.all((state_ == 0) | (state_ == 1))
        output = state_
        for rule in self.rules:
            output = output.at[rule.target_index].set(rule.evaluate(state_))
        status = jnp.where(
            binary, int(RegulatoryStatus.SUCCESS), int(RegulatoryStatus.INVALID_STATE)
        ).astype(jnp.int32)
        output = jnp.where(binary, output, state_)
        evidence = RegulatoryTransitionEvidence(
            input_state=state_,
            output_state=output,
            applied_rule_mask=self.regulated_mask,
            has_regulatory_cycles=jnp.asarray(self.has_cycles),
            synchronous=True,
            exact=True,
        )
        return RegulatoryTransitionResult(
            valid=binary,
            status=status,
            state=output,
            evidence=evidence,
            method_contract=_transition_contract(),
            network_id=self.network_id,
        )

    def to_factor_graph(
        self, /, *, max_supported_configurations: int = 1_000_000
    ) -> RegulatoryPGMResult:
        """Lower the complete two-slice transition relation to native PGM factors."""

        capacity = int(max_supported_configurations)
        if capacity < 1:
            raise ValueError("max_supported_configurations must be positive.")
        counts = [2 ** rule.parent_indices.shape[0] for rule in self.rules]
        counts.extend([2] * int(np.count_nonzero(~np.asarray(self.regulated_mask))))
        required = int(sum(counts))
        if required > capacity:
            raise RegulatoryCapacityError(
                "Regulatory PGM lowering requires "
                f"{required} supported configurations; capacity is {capacity}."
            )
        current = DiscreteVariableGroup("current", num_states=2, shape=(self.num_nodes,))
        following = DiscreteVariableGroup("next", num_states=2, shape=(self.num_nodes,))
        factors = []
        for rule in self.rules:
            parent_count = rule.parent_indices.shape[0]
            configuration_count = 2**parent_count
            parent_configurations = (
                np.arange(configuration_count, dtype=np.int32)[:, None]
                >> np.arange(parent_count - 1, -1, -1, dtype=np.int32)[None, :]
            ) & 1
            target_values = np.asarray(rule.truth_table, dtype=np.int32)[:, None]
            configurations = np.concatenate(
                (parent_configurations, target_values), axis=1
            )
            selections = [
                VariableSelection(current, jnp.asarray([int(index)], dtype=jnp.int32))
                for index in np.asarray(rule.parent_indices)
            ]
            selections.append(
                VariableSelection(
                    following,
                    jnp.asarray([int(np.asarray(rule.target_index))], dtype=jnp.int32),
                )
            )
            factors.append(
                EnumeratedFactorGroup(
                    selections,
                    jnp.asarray(configurations, dtype=jnp.int32),
                    jnp.zeros((1, configuration_count)),
                )
            )
        for index in np.flatnonzero(~np.asarray(self.regulated_mask)):
            factors.append(
                EnumeratedFactorGroup(
                    (
                        VariableSelection(current, jnp.asarray([index], dtype=jnp.int32)),
                        VariableSelection(
                            following, jnp.asarray([index], dtype=jnp.int32)
                        ),
                    ),
                    jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32),
                    jnp.zeros((1, 2)),
                )
            )
        graph = DiscreteFactorGraph((current, following), tuple(factors))
        evidence = RegulatoryPGMEvidence(
            supported_configurations=jnp.asarray(required, dtype=jnp.int32),
            rule_configuration_counts=jnp.asarray(counts, dtype=jnp.int32),
            has_regulatory_cycles=jnp.asarray(self.has_cycles),
            synchronous_two_slice=True,
            complete=True,
            exact=True,
            pgm_substrate="phydrax.pgm.DiscreteFactorGraph",
        )
        return RegulatoryPGMResult(
            valid=jnp.asarray(True),
            status=jnp.asarray(int(RegulatoryStatus.SUCCESS), dtype=jnp.int32),
            factor_graph=graph,
            evidence=evidence,
            method_contract=_pgm_contract(),
            network_id=self.network_id,
        )


def _transition_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "synchronous-discrete-regulatory-transition",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.DISCRETE,
        conditioning_statement="Finite binary truth-table evaluation has no numerical conditioning.",
        truncation_statement="Every declared rule is evaluated from the same source state.",
        capacity_semantics="State storage is exactly one binary value per declared node.",
        assumptions=("Rules use synchronous update semantics.",),
        nondifferentiable_outputs=("state", "status", "valid"),
    )


def _pgm_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "regulatory-transition-pgm-lowering",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.GRAPH,
        conditioning_statement="Hard transition factors have exact finite support.",
        truncation_statement="All truth-table and identity configurations are represented.",
        capacity_semantics=(
            "The complete supported-configuration count is checked before any factor is built."
        ),
        assumptions=(
            "Current and next states are separate variable slices.",
            "Directed regulatory cycles are represented without a DAG assumption.",
        ),
        nondifferentiable_outputs=("factor support", "status", "valid"),
    )


__all__ = [
    "DiscreteRegulatoryNetwork",
    "RegulatoryCapacityError",
    "RegulatoryPGMEvidence",
    "RegulatoryPGMResult",
    "RegulatoryRule",
    "RegulatoryStatus",
    "RegulatoryTransitionEvidence",
    "RegulatoryTransitionResult",
]
