#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain import LocalFieldFamily, SubdomainCover
from ._schwarz import SchwarzTraceState, TraceExchangeState


class FunctionalDecompositionShardingPlan(StrictModule, NonTrainableState):
    """Explicit patch ownership over a fixed JAX device collection."""

    devices: tuple[jax.Device, ...] = eqx.field(static=True)
    assignments: tuple[tuple[str, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        cover: SubdomainCover,
        /,
        *,
        devices: Sequence[jax.Device] | None = None,
        assignments: Mapping[str, int] | None = None,
    ):
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be a SubdomainCover.")
        devices_ = tuple(jax.devices() if devices is None else devices)
        if not devices_:
            raise ValueError("At least one JAX device is required.")
        if assignments is None:
            assignments_ = tuple(
                (patch_id, index % len(devices_))
                for index, patch_id in enumerate(cover.patch_ids)
            )
        else:
            mapping = {str(name): int(index) for name, index in assignments.items()}
            if set(mapping) != set(cover.patch_ids):
                raise ValueError(
                    "assignments must define every cover patch exactly once."
                )
            assignments_ = tuple(
                (patch_id, mapping[patch_id]) for patch_id in cover.patch_ids
            )
        if any(not 0 <= index < len(devices_) for _, index in assignments_):
            raise ValueError("Patch assignment references an unavailable device.")
        self.devices = devices_
        self.assignments = assignments_

    def device_index(self, patch_id: str, /) -> int:
        for name, index in self.assignments:
            if name == patch_id:
                return index
        raise KeyError(f"Unknown sharded patch {patch_id!r}.")

    def device(self, patch_id: str, /) -> jax.Device:
        return self.devices[self.device_index(patch_id)]


class DecompositionShardingEvidence(StrictModule, NonTrainableState):
    patch_counts: tuple[int, ...] = eqx.field(static=True)
    cross_device_pairings: int = eqx.field(static=True)
    communicated_bytes: int = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        patch_counts: tuple[int, ...],
        cross_device_pairings: int,
        communicated_bytes: int,
        verified: bool,
    ):
        self.patch_counts = tuple(int(value) for value in patch_counts)
        self.cross_device_pairings = int(cross_device_pairings)
        self.communicated_bytes = int(communicated_bytes)
        self.verified = bool(verified)


class ShardedLocalFieldFamily(StrictModule):
    family: LocalFieldFamily
    plan: FunctionalDecompositionShardingPlan
    evidence: DecompositionShardingEvidence

    def __init__(
        self,
        family: LocalFieldFamily,
        plan: FunctionalDecompositionShardingPlan,
        evidence: DecompositionShardingEvidence,
        /,
    ):
        self.family = family
        self.plan = plan
        self.evidence = evidence


def place_local_field_family(
    family: LocalFieldFamily,
    plan: FunctionalDecompositionShardingPlan,
    /,
) -> ShardedLocalFieldFamily:
    """Place each local function array tree on its declared owning device."""
    if not isinstance(family, LocalFieldFamily):
        raise TypeError("family must be a LocalFieldFamily.")
    if not isinstance(plan, FunctionalDecompositionShardingPlan):
        raise TypeError("plan must be a FunctionalDecompositionShardingPlan.")
    if set(family.cover.patch_ids) != {name for name, _ in plan.assignments}:
        raise ValueError("Sharding plan and local family covers do not match.")
    fields = {
        patch.patch_id: eqx.filter_shard(field, plan.device(patch.patch_id))
        for patch, field in zip(family.cover.patches, family.fields, strict=True)
    }
    placed = LocalFieldFamily(family.field_id, family.cover, fields)
    counts = tuple(
        sum(index == device_index for _, index in plan.assignments)
        for device_index in range(len(plan.devices))
    )
    cross = sum(
        plan.device_index(pairing.left_patch_id)
        != plan.device_index(pairing.right_patch_id)
        for pairing in family.cover.pairings
    )
    evidence = DecompositionShardingEvidence(
        patch_counts=counts,
        cross_device_pairings=cross,
        communicated_bytes=0,
        verified=sum(counts) == len(family.cover.patches),
    )
    return ShardedLocalFieldFamily(placed, plan, evidence)


def place_schwarz_trace_state(
    state: SchwarzTraceState,
    cover: SubdomainCover,
    plan: FunctionalDecompositionShardingPlan,
    /,
) -> tuple[SchwarzTraceState, DecompositionShardingEvidence]:
    """Move incoming trace arrays to target patch devices and account for traffic."""
    if not isinstance(state, SchwarzTraceState):
        raise TypeError("state must be a SchwarzTraceState.")
    exchanges = []
    communicated = 0
    cross = 0
    for exchange in state.exchanges:
        pairing = cover.pairing(exchange.pairing_id)
        left_device = plan.device(pairing.left_patch_id)
        right_device = plan.device(pairing.right_patch_id)
        different = left_device != right_device
        cross += int(different)
        if different:
            communicated += int(
                exchange.left_target.nbytes + exchange.right_target.nbytes
            )
        exchanges.append(
            TraceExchangeState(
                pairing_id=exchange.pairing_id,
                points=exchange.points,
                left_values=jax.device_put(exchange.left_values, left_device),
                right_values=jax.device_put(exchange.right_values, right_device),
                left_target=jax.device_put(exchange.left_target, left_device),
                right_target=jax.device_put(exchange.right_target, right_device),
                quantity_id=exchange.quantity_id,
            )
        )
    counts = tuple(
        sum(index == device_index for _, index in plan.assignments)
        for device_index in range(len(plan.devices))
    )
    evidence = DecompositionShardingEvidence(
        patch_counts=counts,
        cross_device_pairings=cross,
        communicated_bytes=communicated,
        verified=sum(counts) == len(cover.patches),
    )
    return SchwarzTraceState(tuple(exchanges), sweep=state.sweep), evidence


class DistributedCollectiveEvidence(StrictModule, NonTrainableState):
    mode: str = eqx.field(static=True)
    device_count: int = eqx.field(static=True)
    communicated_bytes: int = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: str,
        device_count: int,
        communicated_bytes: int,
        verified: bool,
    ):
        self.mode = str(mode)
        self.device_count = int(device_count)
        self.communicated_bytes = int(communicated_bytes)
        self.verified = bool(verified)


class DistributedCollectiveResult(StrictModule):
    value: jax.Array
    evidence: DistributedCollectiveEvidence

    def __init__(
        self,
        value: jax.Array,
        evidence: DistributedCollectiveEvidence,
        /,
    ):
        self.value = value
        self.evidence = evidence


def distributed_pou_collective(
    local_values,
    local_weights,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> DistributedCollectiveResult:
    """Assemble POU numerator and denominator with an actual device-axis psum."""
    devices_ = tuple(jax.devices() if devices is None else devices)
    values = jax.numpy.asarray(local_values)
    weights = jax.numpy.asarray(local_weights)
    if values.shape[0] != len(devices_) or weights.shape[0] != len(devices_):
        raise ValueError("POU collective inputs need one leading shard per device.")

    def assemble(value, weight):
        expanded = weight
        while expanded.ndim < value.ndim:
            expanded = expanded[..., None]
        numerator = jax.lax.psum(expanded * value, "patch")
        denominator = jax.lax.psum(expanded, "patch")
        return jax.numpy.where(denominator > 0.0, numerator / denominator, jax.numpy.nan)

    assembled = jax.pmap(
        assemble,
        axis_name="patch",
        devices=devices_,
    )(values, weights)
    communicated = int((values.nbytes + weights.nbytes) * max(len(devices_) - 1, 0))
    evidence = DistributedCollectiveEvidence(
        mode="partition-of-unity-psum",
        device_count=len(devices_),
        communicated_bytes=communicated,
        verified=bool(jax.numpy.all(jax.numpy.isfinite(assembled))),
    )
    return DistributedCollectiveResult(assembled, evidence)


def distributed_schwarz_exchange(
    outgoing,
    source_indices,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> DistributedCollectiveResult:
    """Exchange one fixed-shape outgoing trace per device through all-gather routing."""
    devices_ = tuple(jax.devices() if devices is None else devices)
    values = jax.numpy.asarray(outgoing)
    sources = jax.numpy.asarray(source_indices, dtype=jax.numpy.int32)
    if values.shape[0] != len(devices_) or sources.shape != (len(devices_),):
        raise ValueError("Schwarz collective inputs need one route per device.")
    if bool(jax.numpy.any((sources < 0) | (sources >= len(devices_)))):
        raise ValueError("Schwarz source index is outside the device axis.")

    def exchange(value, source):
        gathered = jax.lax.all_gather(value, "patch", tiled=False)
        return gathered[source]

    received = jax.pmap(
        exchange,
        axis_name="patch",
        devices=devices_,
    )(values, sources)
    communicated = int(values.nbytes * max(len(devices_) - 1, 0))
    evidence = DistributedCollectiveEvidence(
        mode="schwarz-all-gather",
        device_count=len(devices_),
        communicated_bytes=communicated,
        verified=bool(jax.numpy.all(jax.numpy.isfinite(received))),
    )
    return DistributedCollectiveResult(received, evidence)


__all__ = [
    "DecompositionShardingEvidence",
    "DistributedCollectiveEvidence",
    "DistributedCollectiveResult",
    "FunctionalDecompositionShardingPlan",
    "ShardedLocalFieldFamily",
    "place_local_field_family",
    "distributed_pou_collective",
    "distributed_schwarz_exchange",
    "place_schwarz_trace_state",
]
