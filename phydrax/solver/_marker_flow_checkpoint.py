#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..uq._checkpoint import (
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)


_MARKER_FLOW_CHECKPOINT_KIND = "marker-flow-accepted-state"


class MarkerFlowCheckpointPayload(StrictModule):
    """Complete accepted marker-flow state; inactive subsystems use empty arrays."""

    time: Array
    accepted_steps: Array
    fluid_state: object
    pressure: Array
    marker_position: Array
    marker_velocity: Array
    marker_multiplier: Array
    rigid_state: object
    deformable_state: object
    contact_state: object
    route_state: object
    topology_state: object
    amr_state: object
    stochastic_state: object
    solver_history: object


class MarkerFlowCheckpointPlan(StrictModule, NonTrainableState):
    method_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    decomposition_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        method_id: str,
        operator_id: str,
        boundary_id: str,
        transfer_id: str,
        topology_id: str = "fixed-marker-topology",
        decomposition_id: str = "serial",
    ):
        identities = tuple(
            str(value)
            for value in (
                method_id,
                operator_id,
                boundary_id,
                transfer_id,
                topology_id,
                decomposition_id,
            )
        )
        if any(not value for value in identities):
            raise ValueError("Marker-flow checkpoint identities must be nonempty.")
        (
            self.method_id,
            self.operator_id,
            self.boundary_id,
            self.transfer_id,
            self.topology_id,
            self.decomposition_id,
        ) = identities
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": _MARKER_FLOW_CHECKPOINT_KIND,
                "method": self.method_id,
                "operator": self.operator_id,
                "boundary": self.boundary_id,
                "transfer": self.transfer_id,
                "topology": self.topology_id,
                "decomposition": self.decomposition_id,
            }
        )

    def compatibility(self, /) -> dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "method_id": self.method_id,
            "operator_id": self.operator_id,
            "boundary_id": self.boundary_id,
            "transfer_id": self.transfer_id,
            "topology_id": self.topology_id,
            "decomposition_id": self.decomposition_id,
        }


class MarkerFlowReplayRecord(StrictModule):
    accepted_time: Array
    attempted_step: Array
    accepted: Array
    status: Array
    route_epoch: Array
    topology_epoch: Array
    stochastic_counter: Array
    event_parameter: Array
    replay_id: str = eqx.field(static=True)


class MarkerFlowReplayDerivativeReport(StrictModule):
    fixed_routes: Array
    fixed_topology: Array
    event_map_certified: Array
    pathwise_noise: Array
    differentiable: Array


class MarkerFlowReplayResult(StrictModule):
    state: object
    time: Array
    accepted_steps: Array
    status_match: Array
    time_match: Array
    finite: Array
    derivative: MarkerFlowReplayDerivativeReport
    successful: Array
    replay_id: str = eqx.field(static=True)


def write_marker_flow_checkpoint(
    path: str | os.PathLike[str],
    plan: MarkerFlowCheckpointPlan,
    payload: MarkerFlowCheckpointPayload,
    /,
) -> Path:
    if not isinstance(plan, MarkerFlowCheckpointPlan):
        raise TypeError("plan must be MarkerFlowCheckpointPlan.")
    if not isinstance(payload, MarkerFlowCheckpointPayload):
        raise TypeError("payload must be MarkerFlowCheckpointPayload.")
    arrays: dict[str, Any] = {}
    specification = pack_array_tree("payload", payload, arrays)
    return write_checkpoint_archive(
        path,
        kind=_MARKER_FLOW_CHECKPOINT_KIND,
        compatibility=plan.compatibility(),
        state={"payload": specification},
        arrays=arrays,
    )


def read_marker_flow_checkpoint(
    path: str | os.PathLike[str],
    plan: MarkerFlowCheckpointPlan,
    template: MarkerFlowCheckpointPayload,
    /,
) -> MarkerFlowCheckpointPayload:
    if not isinstance(plan, MarkerFlowCheckpointPlan):
        raise TypeError("plan must be MarkerFlowCheckpointPlan.")
    if not isinstance(template, MarkerFlowCheckpointPayload):
        raise TypeError("template must be MarkerFlowCheckpointPayload.")
    state, arrays = read_checkpoint_archive(
        path,
        kind=_MARKER_FLOW_CHECKPOINT_KIND,
        compatibility=plan.compatibility(),
    )
    specification = state.get("payload")
    if not isinstance(specification, Mapping):
        raise TypeError("Marker-flow checkpoint payload specification is missing.")
    restored = unpack_array_tree(specification, arrays, template)
    if not isinstance(restored, MarkerFlowCheckpointPayload):
        raise TypeError("Restored marker-flow payload has an invalid tree type.")
    return restored


class MarkerFlowReplayPlan(StrictModule, NonTrainableState):
    """Deterministic accepted-step replay with explicit event and derivative boundaries."""

    replay_id: str = eqx.field(static=True)
    fixed_routes: bool = eqx.field(static=True)
    fixed_topology: bool = eqx.field(static=True)
    event_map_certified: bool = eqx.field(static=True)
    pathwise_noise: bool = eqx.field(static=True)

    def __init__(
        self,
        replay_id: str,
        /,
        *,
        fixed_routes: bool = True,
        fixed_topology: bool = True,
        event_map_certified: bool = False,
        pathwise_noise: bool = True,
    ):
        identifier = str(replay_id)
        if not identifier:
            raise ValueError("replay_id must be nonempty.")
        self.replay_id = identifier
        self.fixed_routes = bool(fixed_routes)
        self.fixed_topology = bool(fixed_topology)
        self.event_map_certified = bool(event_map_certified)
        self.pathwise_noise = bool(pathwise_noise)

    def replay(
        self,
        initial_state,
        record: MarkerFlowReplayRecord,
        step: Callable[[object, Array, Array, Array, Array], tuple[object, Array, Array]],
        /,
        *,
        initial_time: ArrayLike = 0.0,
    ) -> MarkerFlowReplayResult:
        if not isinstance(record, MarkerFlowReplayRecord):
            raise TypeError("record must be MarkerFlowReplayRecord.")
        if record.replay_id != self.replay_id:
            raise ValueError("Replay record belongs to another plan.")
        if not callable(step):
            raise TypeError("step must be callable.")
        count = int(record.accepted_time.size)
        arrays = (
            record.attempted_step,
            record.accepted,
            record.status,
            record.route_epoch,
            record.topology_epoch,
            record.stochastic_counter,
        )
        if any(value.shape != (count,) for value in arrays):
            raise ValueError("Replay record arrays must share one step shape.")
        if record.event_parameter.shape[0] != count:
            raise ValueError("Replay event parameters must have one row per step.")
        state = initial_state
        time = jnp.asarray(initial_time)
        time_match = jnp.asarray(True)
        status_match = jnp.asarray(True)
        finite = jnp.asarray(True)
        accepted_steps = jnp.asarray(0, dtype=jnp.int32)
        for index in range(count):
            state, accepted, status = step(
                state,
                record.attempted_step[index],
                record.event_parameter[index],
                record.stochastic_counter[index],
                record.route_epoch[index],
            )
            status_match = (
                status_match
                & (accepted == record.accepted[index])
                & (status == record.status[index])
            )
            time = jnp.where(accepted, time + record.attempted_step[index], time)
            time_scale = jnp.maximum(1.0, jnp.abs(record.accepted_time[index]))
            time_match = time_match & (
                jnp.abs(time - record.accepted_time[index])
                <= 32.0 * jnp.finfo(time.dtype).eps * time_scale
            )
            finite = finite & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(state)
                    )
                )
            )
            accepted_steps = accepted_steps + accepted.astype(jnp.int32)
        derivative = MarkerFlowReplayDerivativeReport(
            jnp.asarray(self.fixed_routes),
            jnp.asarray(self.fixed_topology),
            jnp.asarray(self.event_map_certified),
            jnp.asarray(self.pathwise_noise),
            jnp.asarray(
                self.fixed_routes
                and (self.fixed_topology or self.event_map_certified)
                and self.pathwise_noise
            ),
        )
        successful = status_match & time_match & finite
        return MarkerFlowReplayResult(
            state,
            time,
            accepted_steps,
            status_match,
            time_match,
            finite,
            derivative,
            successful,
            self.replay_id,
        )


__all__ = [
    "MarkerFlowCheckpointPayload",
    "MarkerFlowCheckpointPlan",
    "MarkerFlowReplayDerivativeReport",
    "MarkerFlowReplayPlan",
    "MarkerFlowReplayRecord",
    "MarkerFlowReplayResult",
    "read_marker_flow_checkpoint",
    "write_marker_flow_checkpoint",
]
