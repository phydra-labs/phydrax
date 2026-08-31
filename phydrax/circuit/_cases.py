#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import RecyclingState, refresh_recycling, solve_recycled
from ._mna import (
    MNASolvePolicy,
    NodalCircuit,
    prepare_mna,
    PreparedMNA,
    solve_mna,
)
from ._network import ScatteringNetworkResult, WaveExcitation
from ._network_action import (
    _ordered_excitation,
    prepare_scattering_action,
    PreparedScatteringAction,
    ScatteringActionPolicy,
    solve_scattering_action,
)
from ._topology import ScatteringNetwork


class PreparedScatteringActionCaseBatch(StrictModule):
    cases: tuple[PreparedScatteringAction, ...]
    case_shape: tuple[int, ...] = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)


class ScatteringActionCaseBatchResult(StrictModule):
    external_outgoing: Array
    status: Array
    relative_residual: Array
    results: tuple[ScatteringNetworkResult, ...]
    case_shape: tuple[int, ...] = eqx.field(static=True)


class PreparedMNACaseBatch(StrictModule):
    cases: tuple[PreparedMNA, ...]
    case_shape: tuple[int, ...] = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)


class MNACaseBatchResult(StrictModule):
    outgoing: Array
    status: Array
    relative_residual: Array
    results: tuple
    case_shape: tuple[int, ...] = eqx.field(static=True)


class RecycledScatteringSweepResult(StrictModule):
    external_outgoing: tuple[Array, ...]
    linear_status: tuple[Array, ...]
    recycling: RecyclingState
    sweep_id: str = eqx.field(static=True)


def _case_shape(shape: Sequence[int], count: int, /) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in shape)
    if not resolved or any(value <= 0 for value in resolved) or prod(resolved) != count:
        raise ValueError("case_shape must be positive and match the case count.")
    return resolved


def prepare_scattering_action_case_batch(
    networks: Sequence[ScatteringNetwork],
    angular_frequencies: Sequence[ArrayLike],
    case_shape: Sequence[int],
    /,
    *,
    policy: ScatteringActionPolicy | None = None,
) -> PreparedScatteringActionCaseBatch:
    network_tuple, frequency_tuple = tuple(networks), tuple(angular_frequencies)
    if not network_tuple or len(network_tuple) != len(frequency_tuple):
        raise ValueError("Networks and frequencies must be nonempty and aligned.")
    shape = _case_shape(case_shape, len(network_tuple))
    cases = tuple(
        prepare_scattering_action(network, frequency, policy)
        for network, frequency in zip(network_tuple, frequency_tuple, strict=True)
    )
    external_ids = cases[0].plan.external_channel_ids
    if any(case.plan.external_channel_ids != external_ids for case in cases[1:]):
        raise ValueError("Scattering action cases must share external channel layout.")
    batch_id = canonical_fingerprint(
        {
            "kind": "prepared-scattering-action-case-batch",
            "cases": [case.prepared_id for case in cases],
            "shape": shape,
        }
    )
    return PreparedScatteringActionCaseBatch(cases, shape, batch_id)


def solve_scattering_action_case_batch(
    prepared: PreparedScatteringActionCaseBatch,
    excitations: WaveExcitation | ArrayLike | Sequence[WaveExcitation | ArrayLike],
    /,
) -> ScatteringActionCaseBatchResult:
    if not isinstance(prepared, PreparedScatteringActionCaseBatch):
        raise TypeError("prepared must be PreparedScatteringActionCaseBatch.")
    if isinstance(excitations, (WaveExcitation, jax.Array)):
        values = (excitations,) * len(prepared.cases)
    else:
        values = tuple(excitations)
    if len(values) != len(prepared.cases):
        raise ValueError("Excitations must be shared or contain one value per case.")
    results = tuple(
        solve_scattering_action(case, excitation)
        for case, excitation in zip(prepared.cases, values, strict=True)
    )
    shape = prepared.case_shape
    outgoing_shape = results[0].external_outgoing.shape
    if any(result.external_outgoing.shape != outgoing_shape for result in results[1:]):
        raise ValueError("Scattering case result shapes changed after preparation.")
    outgoing = jnp.stack(tuple(result.external_outgoing for result in results)).reshape(
        shape + outgoing_shape
    )
    status = jnp.stack(tuple(result.diagnostics.status for result in results)).reshape(
        shape
    )
    residual = jnp.stack(
        tuple(result.diagnostics.relative_residual for result in results)
    ).reshape(shape)
    return ScatteringActionCaseBatchResult(outgoing, status, residual, results, shape)


def prepare_mna_case_batch(
    circuits: Sequence[NodalCircuit],
    angular_frequencies: Sequence[ArrayLike],
    case_shape: Sequence[int],
    /,
    *,
    policy: MNASolvePolicy | None = None,
) -> PreparedMNACaseBatch:
    circuit_tuple, frequency_tuple = tuple(circuits), tuple(angular_frequencies)
    if not circuit_tuple or len(circuit_tuple) != len(frequency_tuple):
        raise ValueError("Circuits and frequencies must be nonempty and aligned.")
    shape = _case_shape(case_shape, len(circuit_tuple))
    cases = tuple(
        prepare_mna(circuit, frequency, policy)
        for circuit, frequency in zip(circuit_tuple, frequency_tuple, strict=True)
    )
    port_ids = tuple(port.port_id for port in cases[0].circuit.ports)
    if any(
        tuple(port.port_id for port in case.circuit.ports) != port_ids
        for case in cases[1:]
    ):
        raise ValueError("MNA cases must share external port layout.")
    batch_id = canonical_fingerprint(
        {
            "kind": "prepared-mna-case-batch",
            "cases": [case.prepared_id for case in cases],
            "shape": shape,
        }
    )
    return PreparedMNACaseBatch(cases, shape, batch_id)


def solve_mna_case_batch(
    prepared: PreparedMNACaseBatch,
    incident: ArrayLike | Sequence[ArrayLike],
    /,
) -> MNACaseBatchResult:
    if not isinstance(prepared, PreparedMNACaseBatch):
        raise TypeError("prepared must be PreparedMNACaseBatch.")
    if isinstance(incident, jax.Array):
        values = (incident,) * len(prepared.cases)
    else:
        values = tuple(incident)
    if len(values) != len(prepared.cases):
        raise ValueError("Incident values must be shared or contain one value per case.")
    results = tuple(
        solve_mna(case, value) for case, value in zip(prepared.cases, values, strict=True)
    )
    shape = prepared.case_shape
    outgoing_shape = results[0].outgoing.shape
    if any(result.outgoing.shape != outgoing_shape for result in results[1:]):
        raise ValueError("MNA case result shapes changed after preparation.")
    outgoing = jnp.stack(tuple(result.outgoing for result in results)).reshape(
        shape + outgoing_shape
    )
    status = jnp.stack(tuple(result.diagnostics.status for result in results)).reshape(
        shape
    )
    residual = jnp.stack(
        tuple(result.diagnostics.relative_residual for result in results)
    ).reshape(shape)
    return MNACaseBatchResult(outgoing, status, residual, results, shape)


def solve_scattering_action_recycled_sweep(
    prepared_cases: Sequence[PreparedScatteringAction],
    excitation: WaveExcitation | ArrayLike,
    /,
) -> RecycledScatteringSweepResult:
    cases = tuple(prepared_cases)
    if not cases:
        raise ValueError("prepared_cases must be nonempty.")
    topology = cases[0].plan.channel_paths
    external = cases[0].plan.external_channel_ids
    if any(
        case.plan.channel_paths != topology or case.plan.external_channel_ids != external
        for case in cases[1:]
    ):
        raise ValueError("Recycled sweep cases must share topology and channel layout.")
    excitation_ = (
        excitation
        if isinstance(excitation, WaveExcitation)
        else WaveExcitation(excitation)
    )
    recycling = None
    outputs: list[Array] = []
    statuses: list[Array] = []
    for case in cases:
        external_incident = _ordered_excitation(case, excitation_)
        rhs = (
            jnp.zeros(
                (case.plan.cost.channels, external_incident.shape[-1]),
                dtype=jnp.complex128,
            )
            .at[jnp.asarray(case.plan.external_channels), :]
            .set(external_incident)
        )
        if recycling is not None:
            recycling = refresh_recycling(recycling, case.linear, refresh="rebuild")
        recycled = solve_recycled(case.linear, rhs, recycling=recycling)
        recycling = recycled.recycling
        incident = jnp.asarray(recycled.result.value)
        outgoing = case.scattering.mv(incident)
        outputs.append(outgoing[jnp.asarray(case.plan.external_channels), :])
        statuses.append(jnp.asarray(recycled.result.status, dtype=jnp.int32))
    sweep_id = canonical_fingerprint(
        {
            "kind": "recycled-scattering-action-sweep",
            "plans": [case.plan.plan_id for case in cases],
        }
    )
    return RecycledScatteringSweepResult(
        tuple(outputs), tuple(statuses), recycling, sweep_id
    )


__all__ = [
    "MNACaseBatchResult",
    "PreparedMNACaseBatch",
    "PreparedScatteringActionCaseBatch",
    "RecycledScatteringSweepResult",
    "ScatteringActionCaseBatchResult",
    "prepare_mna_case_batch",
    "prepare_scattering_action_case_batch",
    "solve_mna_case_batch",
    "solve_scattering_action_case_batch",
    "solve_scattering_action_recycled_sweep",
]
