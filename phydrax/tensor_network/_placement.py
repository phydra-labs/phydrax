#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._contraction import PreparedContraction
from ._slicing import (
    execute_slice_assignment,
    mixed_radix_assignments,
    SlicedContractionPlan,
)


class PlacementResourcePolicy(StrictModule):
    maximum_devices: int = eqx.field(static=True)
    maximum_local_bytes: int = eqx.field(static=True)
    maximum_transfer_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_devices: int = 1024,
        maximum_local_bytes: int = 2**31,
        maximum_transfer_bytes: int = 2**34,
    ):
        values = (
            int(maximum_devices),
            int(maximum_local_bytes),
            int(maximum_transfer_bytes),
        )
        if any(value < 1 for value in values):
            raise ValueError("Placement resource limits must be positive.")
        self.maximum_devices, self.maximum_local_bytes, self.maximum_transfer_bytes = (
            values
        )
        self.policy_id = canonical_fingerprint(
            {"kind": "placement-resource-policy", "limits": values}
        )


class TensorNetworkMesh(NonTrainableState):
    """Caller-owned JAX mesh used only for independent exact slices."""

    def __init__(self, devices: Sequence[object], axis_name: str = "slices", /):
        devices_ = tuple(devices)
        axis = str(axis_name)
        if not devices_ or not axis:
            raise ValueError("A tensor-network mesh needs devices and an axis name.")
        self.devices = devices_
        self.axis_name = axis
        self.mesh = Mesh(np.asarray(devices_, dtype=object), (axis,))
        self.device_ids = tuple(str(device) for device in devices_)
        self.mesh_id = canonical_fingerprint(
            {"kind": "tensor-network-mesh", "axis": axis, "devices": self.device_ids}
        )

    @property
    def device_count(self) -> int:
        return len(self.devices)


class DistributedReplayStep(StrictModule):
    device_ordinal: int = eqx.field(static=True)
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)
    assignment_transfer_bytes: int = eqx.field(static=True)
    result_transfer_bytes: int = eqx.field(static=True)
    step_id: str = eqx.field(static=True)


class DistributedSliceReplaySchedule(StrictModule):
    slice_plan_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    padded_slice_count: int = eqx.field(static=True)
    steps: tuple[DistributedReplayStep, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)


class SlicePlacementPlan(NonTrainableState):
    def __init__(
        self,
        slice_plan: SlicedContractionPlan,
        mesh: TensorNetworkMesh,
        resources: PlacementResourcePolicy,
        replay: DistributedSliceReplaySchedule,
        local_peak_bytes: int,
        transfer_bytes: int,
        plan_id: str,
        /,
    ):
        self.slice_plan = slice_plan
        self.mesh = mesh
        self.resources = resources
        self.replay = replay
        self.local_peak_bytes = int(local_peak_bytes)
        self.transfer_bytes = int(transfer_bytes)
        self.plan_id = str(plan_id)
        self.exact = True
        self.claim = (
            "exact independent-slice placement with deterministic ordinal reduction"
        )


class TransferEvidence(StrictModule):
    planned_bytes: int = eqx.field(static=True)
    completed_bytes: int = eqx.field(static=True)
    assignment_sharding: str = eqx.field(static=True)
    result_sharding: str = eqx.field(static=True)
    device_ids: tuple[str, ...] = eqx.field(static=True)


class DistributedExecutionEvidence(StrictModule):
    placement_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    device_count: int = eqx.field(static=True)
    deterministic_reduction: bool = eqx.field(static=True)
    finite: Array
    completed_all_slices: Array
    accepted: Array
    exact: Array
    failure: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    transfer: TransferEvidence


class DistributedContractionResult(StrictModule):
    aggregate: Array | None
    evidence: DistributedExecutionEvidence


def create_tensor_network_mesh(
    devices: Sequence[object] | None = None,
    /,
    *,
    axis_name: str = "slices",
    maximum_devices: int | None = None,
) -> TensorNetworkMesh:
    devices_ = tuple(jax.devices() if devices is None else devices)
    if maximum_devices is not None:
        maximum = int(maximum_devices)
        if maximum < 1:
            raise ValueError("maximum_devices must be positive.")
        devices_ = devices_[:maximum]
    return TensorNetworkMesh(devices_, axis_name)


def plan_slice_placement(
    slice_plan: SlicedContractionPlan,
    mesh: TensorNetworkMesh,
    /,
    *,
    resources: PlacementResourcePolicy | None = None,
) -> SlicePlacementPlan:
    if not isinstance(slice_plan, SlicedContractionPlan):
        raise TypeError("slice_plan must be SlicedContractionPlan.")
    if not isinstance(mesh, TensorNetworkMesh):
        raise TypeError("mesh must be TensorNetworkMesh.")
    resources_ = PlacementResourcePolicy() if resources is None else resources
    if not isinstance(resources_, PlacementResourcePolicy):
        raise TypeError("resources must be PlacementResourcePolicy or None.")
    if mesh.device_count > resources_.maximum_devices:
        raise MemoryError("Placement exceeds maximum_devices.")
    per_device = ceil(slice_plan.slice_count / mesh.device_count)
    padded = per_device * mesh.device_count
    itemsize = precision_itemsize(slice_plan.original.dtype)
    output_elements = slice_plan.original.structure.output_elements
    local_peak = (
        slice_plan.residual.schedule.peak_live_bytes
        + per_device * output_elements * itemsize
    )
    assignment_bytes = padded * len(slice_plan.labels) * 4
    result_bytes = padded * output_elements * itemsize
    transfer_bytes = assignment_bytes + result_bytes
    if local_peak > resources_.maximum_local_bytes:
        raise MemoryError("Placement exceeds maximum_local_bytes.")
    if transfer_bytes > resources_.maximum_transfer_bytes:
        raise MemoryError("Placement exceeds maximum_transfer_bytes.")

    steps = []
    for ordinal in range(mesh.device_count):
        start = ordinal * per_device
        stop = min(start + per_device, slice_plan.slice_count)
        assignment_transfer = max(0, stop - start) * len(slice_plan.labels) * 4
        result_transfer = max(0, stop - start) * output_elements * itemsize
        step_id = canonical_fingerprint(
            {
                "kind": "distributed-slice-replay-step",
                "slice_plan": slice_plan.plan_id,
                "mesh": mesh.mesh_id,
                "device_ordinal": ordinal,
                "range": (start, stop),
            }
        )
        steps.append(
            DistributedReplayStep(
                ordinal,
                start,
                stop,
                assignment_transfer,
                result_transfer,
                step_id,
            )
        )
    schedule_id = canonical_fingerprint(
        {
            "kind": "distributed-slice-replay-schedule",
            "slice_plan": slice_plan.plan_id,
            "mesh": mesh.mesh_id,
            "padded_slice_count": padded,
            "steps": tuple(step.step_id for step in steps),
        }
    )
    replay = DistributedSliceReplaySchedule(
        slice_plan.plan_id, mesh.mesh_id, padded, tuple(steps), schedule_id
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "slice-placement-plan",
            "replay": schedule_id,
            "resources": resources_.policy_id,
        }
    )
    return SlicePlacementPlan(
        slice_plan,
        mesh,
        resources_,
        replay,
        local_peak,
        transfer_bytes,
        plan_id,
    )


def _failure_result(
    placement: SlicePlacementPlan,
    failure: str,
    completed_bytes: int,
    /,
) -> DistributedContractionResult:
    transfer = TransferEvidence(
        placement.transfer_bytes,
        int(completed_bytes),
        f"NamedSharding(PartitionSpec({placement.mesh.axis_name!r}, None))",
        "host-device-ordinal",
        placement.mesh.device_ids,
    )
    evidence = DistributedExecutionEvidence(
        placement.plan_id,
        canonical_fingerprint(
            {
                "kind": "failed-distributed-replay",
                "placement": placement.plan_id,
                "failure": failure,
            }
        ),
        placement.replay.schedule_id,
        placement.mesh.device_count,
        True,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        failure,
        "no aggregate is accepted after any distributed execution failure",
        transfer,
    )
    return DistributedContractionResult(None, evidence)


def execute_distributed_slices(
    prepared: PreparedContraction,
    placement: SlicePlacementPlan,
    /,
) -> DistributedContractionResult:
    """Shard independent slices and reduce results in global mixed-radix order."""

    if not isinstance(prepared, PreparedContraction) or not isinstance(
        placement, SlicePlacementPlan
    ):
        raise TypeError("prepared and placement have invalid types.")
    plan = placement.slice_plan
    if prepared.plan.plan_id != plan.original.plan_id:
        raise ValueError("Prepared contraction belongs to another placement.")
    assignments = mixed_radix_assignments(
        plan.dimensions,
        maximum_assignments=plan.resources.maximum_slices,
    )
    padding = placement.replay.padded_slice_count - plan.slice_count
    if padding:
        assignments = jnp.concatenate(
            (
                assignments,
                jnp.zeros((padding, len(plan.labels)), dtype=assignments.dtype),
            ),
            axis=0,
        )
    sharding = NamedSharding(
        placement.mesh.mesh,
        PartitionSpec(placement.mesh.axis_name, None),
    )
    completed_bytes = 0
    try:
        sharded_assignments = jax.device_put(assignments, sharding)
        completed_bytes += assignments.size * assignments.dtype.itemsize
        sharded_results = jax.vmap(
            lambda assignment: execute_slice_assignment(prepared, plan, assignment)
        )(sharded_assignments)
        host_results = np.asarray(jax.device_get(sharded_results))
        completed_bytes += host_results.nbytes
    except (RuntimeError, ValueError) as error:
        return _failure_result(
            placement, f"distributed-io:{type(error).__name__}", completed_bytes
        )

    host_results = host_results[: plan.slice_count]
    if not np.all(np.isfinite(host_results)):
        return _failure_result(placement, "non-finite-slice-result", completed_bytes)
    aggregate = jnp.zeros(host_results.shape[1:], dtype=sharded_results.dtype)
    for ordinal in range(plan.slice_count):
        aggregate = aggregate + jnp.asarray(host_results[ordinal])
    finite = jnp.all(jnp.isfinite(aggregate))
    if not bool(np.asarray(finite)):
        return _failure_result(
            placement, "non-finite-deterministic-reduction", completed_bytes
        )
    transfer = TransferEvidence(
        placement.transfer_bytes,
        completed_bytes,
        f"NamedSharding(PartitionSpec({placement.mesh.axis_name!r}, None))",
        "host-device-ordinal",
        placement.mesh.device_ids,
    )
    replay_id = canonical_fingerprint(
        {
            "kind": "distributed-slice-replay",
            "placement": placement.plan_id,
            "schedule": placement.replay.schedule_id,
        }
    )
    evidence = DistributedExecutionEvidence(
        placement.plan_id,
        replay_id,
        placement.replay.schedule_id,
        placement.mesh.device_count,
        True,
        finite,
        jnp.asarray(True),
        finite,
        finite,
        "",
        placement.claim,
        transfer,
    )
    return DistributedContractionResult(aggregate, evidence)


__all__ = [
    "DistributedContractionResult",
    "DistributedExecutionEvidence",
    "DistributedReplayStep",
    "DistributedSliceReplaySchedule",
    "PlacementResourcePolicy",
    "SlicePlacementPlan",
    "TensorNetworkMesh",
    "TransferEvidence",
    "create_tensor_network_mesh",
    "execute_distributed_slices",
    "plan_slice_placement",
]
