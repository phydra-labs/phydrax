#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lower a partitioned participant declaration onto the canonical temporal runtime.

Lowering produces an existing `CouplingProblem`; execution remains
`solve_coupling`, `advance_coupling_window`, or `rollout_adaptive_coupling`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition
from .._partitioned_coupling_graph import (
    _admit_route_participants,
    CouplingGraph,
    CouplingResourcePolicy,
    prepare_coupling,
    PreparedCoupling,
)
from .._partitioned_coupling_runtime import _apply_exchange
from .._partitioned_coupling_solve import CouplingProblem
from .._partitioned_coupling_types import (
    AbstractCouplingPolicy,
    AbstractCouplingSubsystem,
    CouplingDifferentiationPolicy,
    CouplingExchange,
    CouplingPort,
    CouplingWindow,
)
from .._partitioned_coupling_waveform import CouplingWaveform
from ._method_participants import AbstractMethodCouplingParticipant


def _zero_signal(port: CouplingPort, /) -> Any:
    zeros = jax.tree.map(
        lambda spec: jnp.zeros(spec.shape, spec.dtype), port.space.structure()
    )
    plan = port.waveform_plan
    if plan is None:
        return zeros
    return CouplingWaveform.constant(plan.initial_grid(), zeros, port.space)


class PartitionedCouplingDeclaration(StrictModule, NonTrainableState):
    """Participants, physical exchanges, and one temporal policy declared once.

    The declaration owns no window loop. `lower_partitioned_coupling` binds initial
    checkpoints and lowers it to the canonical `CouplingProblem`. `time_unit` is
    the unit of the one coupling clock, required by waveform integrations and
    host participants.
    """

    graph: CouplingGraph
    policy: AbstractCouplingPolicy
    differentiation: CouplingDifferentiationPolicy
    resources: CouplingResourcePolicy
    declaration_id: str = eqx.field(static=True)

    def __init__(
        self,
        participants: tuple[AbstractCouplingSubsystem, ...],
        exchanges: tuple[CouplingExchange, ...],
        policy: AbstractCouplingPolicy,
        /,
        *,
        differentiation: CouplingDifferentiationPolicy | None = None,
        resources: CouplingResourcePolicy | None = None,
        time_unit: UnitDefinition | None = None,
    ) -> None:
        graph = CouplingGraph(tuple(participants), tuple(exchanges), time_unit=time_unit)
        if not isinstance(policy, AbstractCouplingPolicy):
            raise TypeError("policy must be an AbstractCouplingPolicy.")
        differentiation_ = (
            CouplingDifferentiationPolicy()
            if differentiation is None
            else differentiation
        )
        if not isinstance(differentiation_, CouplingDifferentiationPolicy):
            raise TypeError(
                "differentiation must be CouplingDifferentiationPolicy or None."
            )
        resources_ = CouplingResourcePolicy() if resources is None else resources
        if not isinstance(resources_, CouplingResourcePolicy):
            raise TypeError("resources must be CouplingResourcePolicy or None.")
        self.graph = graph
        self.policy = policy
        self.differentiation = differentiation_
        self.resources = resources_
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "partitioned-coupling-declaration",
                "graph": graph.graph_id,
                "policy": policy.policy_id,
                "differentiation": differentiation_.policy_id,
                "resources": [
                    resources_.maximum_interface_size,
                    resources_.maximum_state_bytes,
                    resources_.maximum_history_bytes,
                ],
            }
        )


def _ordered_states(
    graph: CouplingGraph, participant_states: Mapping[str, Any], /
) -> tuple[Any, ...]:
    declared = tuple(subsystem.subsystem_id for subsystem in graph.subsystems)
    if set(participant_states) != set(declared):
        raise ValueError(
            "Lowering requires exactly one initial checkpoint per declared participant."
        )
    return tuple(participant_states[identifier] for identifier in declared)


def _explicit_values(
    graph: CouplingGraph, exchange_values: Mapping[str, Any], /
) -> dict[str, Any]:
    """Explicit initial values only where no source can observe its checkpoint."""
    ports = {
        port.port_id: subsystem
        for subsystem in graph.subsystems
        for port in subsystem.output_ports
    }
    unknown = sorted(set(exchange_values) - {e.exchange_id for e in graph.exchanges})
    if unknown:
        raise ValueError("Unknown initial exchange values: " + ", ".join(unknown))
    values: dict[str, Any] = {}
    missing: list[str] = []
    for exchange in graph.exchanges:
        observable = isinstance(
            ports[exchange.source_port_id], AbstractMethodCouplingParticipant
        )
        supplied = exchange.exchange_id in exchange_values
        if observable and supplied:
            raise ValueError(
                f"Exchange {exchange.exchange_id!r} is derived from its source "
                "checkpoint and cannot be overridden."
            )
        if not observable and not supplied:
            missing.append(exchange.exchange_id)
        if supplied:
            values[exchange.exchange_id] = exchange_values[exchange.exchange_id]
    if missing:
        raise ValueError(
            "Exchanges from participants without checkpoint observations need "
            "explicit initial values: " + ", ".join(sorted(missing))
        )
    return values


def _derived_values(
    prepared: PreparedCoupling,
    states: tuple[Any, ...],
    window: CouplingWindow,
    args: Any,
    /,
) -> dict[str, Any]:
    """Map every observable source checkpoint through its declared exchange."""
    values: dict[str, Any] = {}
    for index, exchange in enumerate(prepared.exchanges):
        source_index = prepared.exchange_source_subsystems[index]
        source = prepared.subsystems[source_index]
        if not isinstance(source, AbstractMethodCouplingParticipant):
            continue
        outputs = source.initial_outputs(states[source_index], args)
        output = outputs[prepared.exchange_source_output_indices[index]]
        values[exchange.exchange_id] = _apply_exchange(prepared, index, output, window)
    return values


def lower_partitioned_coupling(
    declaration: PartitionedCouplingDeclaration,
    participant_states: Mapping[str, Any],
    /,
    *,
    t0: float,
    t1: float,
    window_size: float,
    args: Any = None,
    exchange_values: Mapping[str, Any] | None = None,
    problem_id: str | None = None,
) -> CouplingProblem:
    """Lower a declaration and initial checkpoints to a fixed-window `CouplingProblem`.

    Every exchange whose source is a native method participant starts from that
    source's checkpoint observation mapped through the declared spatial transfer,
    unit conversion, and temporal conversion; whole-window amounts start at zero.
    Other sources require an explicit initial value. The returned problem
    prepares and executes on the canonical partitioned runtime.
    """

    if not isinstance(declaration, PartitionedCouplingDeclaration):
        raise TypeError("declaration must be PartitionedCouplingDeclaration.")
    graph = declaration.graph
    # Host participants never lower onto the native runtime; refuse them before
    # initial values are requested from sources that have no native checkpoint.
    _admit_route_participants(graph.subsystems, host_execution=False)
    states = _ordered_states(graph, participant_states)
    explicit = _explicit_values(graph, {} if exchange_values is None else exchange_values)
    targets = {
        port.port_id: port
        for subsystem in graph.subsystems
        for port in subsystem.input_ports
    }
    provisional = tuple(
        explicit[exchange.exchange_id]
        if exchange.exchange_id in explicit
        else _zero_signal(targets[exchange.target_port_id])
        for exchange in graph.exchanges
    )
    # Preparation proves the exchange routes before any initial value is mapped.
    prepared = prepare_coupling(
        graph,
        states,
        provisional,
        policy=declaration.policy,
        differentiation=declaration.differentiation,
        time=t0,
        args=args,
        resources=declaration.resources,
    )
    window = CouplingWindow(0, t0, t0 + window_size)
    derived = _derived_values(
        prepared, prepared.reference_state.participant_states, window, args
    )
    values = tuple(
        explicit[exchange.exchange_id]
        if exchange.exchange_id in explicit
        else derived[exchange.exchange_id]
        for exchange in graph.exchanges
    )
    return CouplingProblem(
        graph,
        states,
        values,
        declaration.policy,
        t0=t0,
        t1=t1,
        window_size=window_size,
        differentiation=declaration.differentiation,
        args=args,
        resources=declaration.resources,
        problem_id=problem_id,
    )


__all__ = ["PartitionedCouplingDeclaration", "lower_partitioned_coupling"]
