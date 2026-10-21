#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.sharding import Mesh

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._contraction import PreparedContraction
from ._slicing import (
    execute_slice_assignment,
    mixed_radix_assignments,
    prepare_sliced_contraction,
    PreparedSlicedContraction,
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
    ) -> None:
        raw_values = (
            maximum_devices,
            maximum_local_bytes,
            maximum_transfer_bytes,
        )
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in raw_values
        ):
            raise TypeError("Placement resource limits must be integers.")
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

    def __init__(self, devices: Sequence[object], axis_name: str = "slices", /) -> None:
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
    ) -> None:
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
        if not isinstance(maximum_devices, Integral) or isinstance(maximum_devices, bool):
            raise TypeError("maximum_devices must be an integer.")
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
    storage_itemsize = precision_itemsize(slice_plan.original.dtype)
    _, accumulation_dtype = slice_plan.original.precision.resolve_contraction(
        slice_plan.original.dtype
    )
    output_itemsize = precision_itemsize(accumulation_dtype)
    output_elements = slice_plan.original.structure.output_elements
    operand_bytes = slice_plan.original.cost.operand_elements * storage_itemsize
    assignment_local_bytes = per_device * len(slice_plan.labels) * 4
    local_peak = (
        operand_bytes
        + per_device * slice_plan.residual.cost.peak_live_bytes
        + per_device * output_elements * output_itemsize
        + assignment_local_bytes
    )
    assignment_bytes = padded * len(slice_plan.labels) * 4
    result_bytes = padded * output_elements * output_itemsize
    transfer_bytes = (
        assignment_bytes + result_bytes + mesh.device_count * (operand_bytes + 4)
    )
    if local_peak > resources_.maximum_local_bytes:
        raise MemoryError("Placement exceeds maximum_local_bytes.")
    if transfer_bytes > resources_.maximum_transfer_bytes:
        raise MemoryError("Placement exceeds maximum_transfer_bytes.")

    steps = []
    for ordinal in range(mesh.device_count):
        start = ordinal * per_device
        stop = min(start + per_device, slice_plan.slice_count)
        assignment_transfer = max(0, stop - start) * len(slice_plan.labels) * 4
        result_transfer = max(0, stop - start) * output_elements * output_itemsize
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
    prepared: PreparedContraction,
    placement: SlicePlacementPlan,
    failure: str,
    completed_bytes: int,
    /,
) -> DistributedContractionResult:
    transfer = TransferEvidence(
        placement.transfer_bytes,
        int(completed_bytes),
        "device-local residual executors with explicit operand replication",
        "host-device-ordinal",
        placement.mesh.device_ids,
    )
    evidence = DistributedExecutionEvidence(
        placement.plan_id,
        canonical_fingerprint(
            {
                "kind": "failed-distributed-replay",
                "placement": placement.plan_id,
                "prepared": prepared.prepared_id,
                "operands": prepared.operand_id,
                "numeric_version": int(prepared.numeric_version),
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


@eqx.filter_jit
def _execute_local_slices(
    binding: PreparedSlicedContraction, assignments: Array, /
) -> Array:
    return jax.vmap(lambda assignment: execute_slice_assignment(binding, assignment))(
        assignments
    )


def execute_distributed_slices(
    prepared: PreparedContraction,
    placement: SlicePlacementPlan,
    /,
) -> DistributedContractionResult:
    """Host-only placement/gather with device-local kernels and global ordinal reduction."""

    if not isinstance(prepared, PreparedContraction) or not isinstance(
        placement, SlicePlacementPlan
    ):
        raise TypeError("prepared and placement have invalid types.")
    plan = placement.slice_plan
    if prepared.plan.plan_id != plan.original.plan_id:
        raise ValueError("Prepared contraction belongs to another placement.")
    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(prepared)):
        raise TypeError(
            "Distributed slice placement/gather is host-only and cannot be transformed."
        )
    binding = prepare_sliced_contraction(prepared, plan)
    external = (
        binding.executor.external_cache_bound_bytes + binding.executor.normalization_bytes
    )
    if placement.local_peak_bytes + external > placement.resources.maximum_local_bytes:
        raise MemoryError(
            "Residual executor external memory exceeds maximum_local_bytes."
        )
    if binding.executor.policy.provider != "native":
        from .._execution_device import device_platform

        if any(device_platform(device) != "cuda" for device in placement.mesh.devices):
            raise ValueError("CUDA residual executors require an all-CUDA placement.")
    assignments = mixed_radix_assignments(
        plan.dimensions,
        maximum_assignments=plan.resources.maximum_slices,
    )
    completed_bytes = 0
    pending: list[Array] = []
    try:
        # Do not execute padded lanes or idle devices. All operands are explicitly
        # replicated to each active device; independent kernels enqueue before
        # host gathering. This is not vendor distributed execution or a collective.
        for device, step in zip(
            placement.mesh.devices, placement.replay.steps, strict=True
        ):
            if step.stop <= step.start:
                continue
            local = jax.device_put(binding, device)
            completed_bytes += sum(value.nbytes for value in prepared.operands) + 4
            for start in range(step.start, step.stop, plan.batch_size):
                stop = min(start + plan.batch_size, step.stop)
                local_assignments = jax.device_put(assignments[start:stop], device)
                completed_bytes += local_assignments.nbytes
                pending.append(_execute_local_slices(local, local_assignments))
        host_blocks = []
        for result in pending:
            block = np.asarray(jax.device_get(result))
            completed_bytes += block.nbytes
            host_blocks.append(block)
        host_results = np.concatenate(host_blocks, axis=0)
    except (RuntimeError, ValueError, MemoryError) as error:
        failures = [f"{type(error).__name__}:{error}"]
        # Drain every already-submitted local result, including on another
        # device, before publishing a failed host aggregate.
        for result in pending:
            try:
                result.block_until_ready()
            except RuntimeError as drain_error:
                failures.append(f"{type(drain_error).__name__}:{drain_error}")
        return _failure_result(
            prepared,
            placement,
            "distributed-io:" + "; ".join(failures),
            completed_bytes,
        )

    # Blocks were submitted and gathered in monotonically increasing ordinal order.
    if not np.all(np.isfinite(host_results)):
        return _failure_result(
            prepared,
            placement,
            "non-finite-slice-result",
            completed_bytes,
        )
    _, accumulation_dtype = plan.original.precision.resolve_contraction(
        plan.original.dtype
    )
    aggregate = jnp.zeros(host_results.shape[1:], dtype=accumulation_dtype)
    for ordinal in range(plan.slice_count):
        aggregate = aggregate + jnp.asarray(
            host_results[ordinal], dtype=accumulation_dtype
        )
    aggregate = plan.original.precision.output(aggregate)
    finite = jnp.all(jnp.isfinite(aggregate))
    if not bool(np.asarray(finite)):
        return _failure_result(
            prepared,
            placement,
            "non-finite-deterministic-reduction",
            completed_bytes,
        )
    transfer = TransferEvidence(
        placement.transfer_bytes,
        completed_bytes,
        "device-local residual executors with explicit operand replication",
        "host-device-ordinal",
        placement.mesh.device_ids,
    )
    replay_id = canonical_fingerprint(
        {
            "kind": "distributed-slice-replay",
            "prepared": prepared.prepared_id,
            "operands": prepared.operand_id,
            "numeric_version": int(prepared.numeric_version),
            "placement": placement.plan_id,
            "schedule": placement.replay.schedule_id,
            "execution": binding.executor.lowering_id,
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
