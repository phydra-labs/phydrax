#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Auditable execution records for external HEP matching and event evolution."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..applications.relativistic_scattering._matrix_element_revision import (
    MatrixElementRevision,
)
from ._capabilities import HEPProviderBinding
from ._events import ParticleEventBatch
from ._host_events import HostEventRecord
from ._operations import ProcessNormalization
from ._weights import EventWeightSet, WeightVariationKind


class ProviderExecutionStage(StrEnum):
    MATCHING = "matching"
    MERGING = "merging"
    SHOWER = "shower"
    HADRONIZATION = "hadronization"


class ProviderExecutionStatus(IntEnum):
    SUCCESS = 0
    CAPABILITY_MISMATCH = 1
    INPUT_PROFILE_MISMATCH = 2
    OUTPUT_PROFILE_MISMATCH = 3
    PROCESS_MISMATCH = 4
    NONFINITE_WEIGHT = 5
    MATRIX_REVISION_MISMATCH = 6
    UNIT_FRAME_MISMATCH = 7


_STAGE_ORDER = {
    ProviderExecutionStage.MATCHING: 0,
    ProviderExecutionStage.MERGING: 1,
    ProviderExecutionStage.SHOWER: 2,
    ProviderExecutionStage.HADRONIZATION: 3,
}


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class NamedWeightSnapshot(StrictModule, NonTrainableState):
    """Signed named event weights at one provider boundary."""

    values: Array
    event_active: Array
    finite: Array
    names: tuple[str, ...] = eqx.field(static=True)
    variation_kinds: tuple[WeightVariationKind, ...] = eqx.field(static=True)
    correlation_groups: tuple[str, ...] = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        event_active: ArrayLike,
        names: Sequence[str],
        variation_kinds: Sequence[WeightVariationKind | str],
        correlation_groups: Sequence[str],
    ):
        values_ = jnp.asarray(values)
        active = jnp.asarray(event_active, dtype=jnp.bool_)
        names_ = tuple(str(value).strip() for value in names)
        kinds = tuple(WeightVariationKind(value) for value in variation_kinds)
        groups = tuple(str(value).strip() for value in correlation_groups)
        if values_.ndim != 2 or values_.shape[0] < 1 or values_.shape[1] < 1:
            raise ValueError("values must have shape (event_count, weight_count).")
        if active.shape != (values_.shape[0],):
            raise ValueError("event_active must align with values.")
        if (
            len(names_) != values_.shape[1]
            or len(kinds) != len(names_)
            or len(groups) != len(names_)
            or any(not value for value in names_ + groups)
            or len(set(names_)) != len(names_)
        ):
            raise ValueError("Weight metadata must be complete, distinct, and aligned.")
        self.values = values_
        self.event_active = active
        self.finite = jnp.all(jnp.isfinite(values_), axis=1)
        self.names = names_
        self.variation_kinds = kinds
        self.correlation_groups = groups
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "provider-named-weight-snapshot",
                "names": list(names_),
                "variation_kinds": [value.value for value in kinds],
                "correlation_groups": list(groups),
                "content": array_tree_fingerprint(
                    {"values": np.asarray(values_), "event_active": np.asarray(active)}
                ),
            }
        )

    @classmethod
    def from_event_weights(cls, weights: EventWeightSet, /) -> NamedWeightSnapshot:
        if not isinstance(weights, EventWeightSet):
            raise TypeError("weights must be EventWeightSet.")
        return cls(
            weights.values,
            event_active=weights.event_active,
            names=weights.names,
            variation_kinds=weights.variation_kinds,
            correlation_groups=weights.correlation_groups,
        )

    @classmethod
    def from_host_event(cls, event: HostEventRecord, /) -> NamedWeightSnapshot:
        if not isinstance(event, HostEventRecord):
            raise TypeError("event must be HostEventRecord.")
        return cls(
            [[value.value for value in event.weights]],
            event_active=(True,),
            names=tuple(value.name for value in event.weights),
            variation_kinds=tuple(value.variation_kind for value in event.weights),
            correlation_groups=tuple(value.correlation_group for value in event.weights),
        )


class ExclusiveMatchingAssignment(StrictModule, NonTrainableState):
    multiplicity: Array
    accepted: Array
    resolved_count: Array
    multiplicities: tuple[int, ...] = eqx.field(static=True)
    merging_scale: float = eqx.field(static=True)


def assign_exclusive_matching_bins(
    resolution_scales: ArrayLike,
    scale_active: ArrayLike,
    /,
    *,
    multiplicities: Sequence[int],
    merging_scale: float,
) -> ExclusiveMatchingAssignment:
    """Assign disjoint jet bins; the largest declared bin is inclusive above its edge."""

    scales = jnp.asarray(resolution_scales)
    active = jnp.asarray(scale_active, dtype=jnp.bool_)
    bins = tuple(multiplicities)
    threshold = float(merging_scale)
    if scales.ndim != 2 or active.shape != scales.shape:
        raise ValueError(
            "resolution_scales and scale_active must be aligned rank-two arrays."
        )
    if tuple(sorted(set(bins))) != bins or not bins or bins[0] < 0:
        raise ValueError("multiplicities must be distinct, increasing, and nonnegative.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("merging_scale must be finite and positive.")
    finite = jnp.all(
        jnp.where(active, jnp.isfinite(scales) & (scales >= 0.0), True), axis=1
    )
    resolved = jnp.sum(active & (scales > threshold), axis=1, dtype=jnp.int32)
    declared = jnp.asarray(bins, dtype=jnp.int32)
    exact = resolved[:, None] == declared[None, :]
    below_last = jnp.any(exact, axis=1)
    inclusive_last = resolved >= declared[-1]
    accepted = finite & (below_last | inclusive_last)
    index = jnp.argmax(exact, axis=1)
    assigned = jnp.where(inclusive_last, declared[-1], declared[index])
    return ExclusiveMatchingAssignment(assigned, accepted, resolved, bins, threshold)


def _event_identity(event: ParticleEventBatch | HostEventRecord, /) -> str:
    if isinstance(event, HostEventRecord):
        return canonical_fingerprint(
            {
                "kind": "host-provider-event-boundary",
                "event_id": event.event_id,
                "subevent_id": event.subevent_id,
                "source_id": event.source_id,
                "particles": [value.particle_id for value in event.particles],
                "vertices": [value.vertex_id for value in event.vertices],
            }
        )
    if isinstance(event, ParticleEventBatch):
        return canonical_fingerprint(
            {
                "kind": "device-provider-event-boundary",
                "plan": event.plan_id,
                "source": event.source_id,
                "identity": array_tree_fingerprint(
                    {
                        "event_ids": np.asarray(event.event_ids),
                        "subevent_ids": np.asarray(event.subevent_ids),
                        "event_active": np.asarray(event.event_active),
                    }
                ),
            }
        )
    raise TypeError("event must be ParticleEventBatch or HostEventRecord.")


def _event_weights(event: ParticleEventBatch | HostEventRecord, /) -> NamedWeightSnapshot:
    if isinstance(event, HostEventRecord):
        return NamedWeightSnapshot.from_host_event(event)
    if isinstance(event, ParticleEventBatch):
        return NamedWeightSnapshot.from_event_weights(event.weights)
    raise TypeError("event must be ParticleEventBatch or HostEventRecord.")


class ProviderExecutionRecord(StrictModule, NonTrainableState):
    """Immutable provenance for one pinned provider invocation."""

    binding: HEPProviderBinding
    normalization: ProcessNormalization
    matrix_element_revision: MatrixElementRevision
    input_weights: NamedWeightSnapshot
    output_weights: NamedWeightSnapshot
    status: ProviderExecutionStatus = eqx.field(static=True)
    stage: ProviderExecutionStage = eqx.field(static=True)
    required_capability: str = eqx.field(static=True)
    input_profile_id: str = eqx.field(static=True)
    output_profile_id: str = eqx.field(static=True)
    input_event_id: str = eqx.field(static=True)
    output_event_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is ProviderExecutionStatus.SUCCESS


class ProviderExecutionChain(StrictModule, NonTrainableState):
    records: tuple[ProviderExecutionRecord, ...]
    chain_id: str = eqx.field(static=True)

    def __init__(self, records: Sequence[ProviderExecutionRecord], /):
        values = tuple(records)
        if not values or any(
            not isinstance(value, ProviderExecutionRecord) for value in values
        ):
            raise TypeError("records must contain ProviderExecutionRecord values.")
        if any(not value.successful for value in values):
            raise ValueError("Only successful provider executions may enter a chain.")
        orders = tuple(_STAGE_ORDER[value.stage] for value in values)
        if any(right < left for left, right in zip(orders, orders[1:], strict=False)):
            raise ValueError("Provider execution stages must be monotonically ordered.")
        if any(
            left.output_event_id != right.input_event_id
            for left, right in zip(values, values[1:], strict=False)
        ):
            raise ValueError("Adjacent provider execution event boundaries do not match.")
        if len({value.execution_id for value in values}) != len(values):
            raise ValueError("Provider executions in a chain must be unique.")
        self.records = values
        self.chain_id = canonical_fingerprint(
            {
                "kind": "provider-execution-chain",
                "records": [value.execution_id for value in values],
            }
        )


def record_provider_execution(
    binding: HEPProviderBinding,
    input_event: ParticleEventBatch | HostEventRecord,
    output_event: ParticleEventBatch | HostEventRecord,
    normalization: ProcessNormalization,
    matrix_element_revision: MatrixElementRevision,
    /,
    *,
    stage: ProviderExecutionStage | str,
    required_capability: str,
    input_profile_id: str,
    output_profile_id: str,
    unit_contract_id: str,
    frame_id: str,
    frame_realization_id: str,
    process_id: str,
) -> ProviderExecutionRecord:
    """Capture actual signed weights and reject incompatible pinned providers."""

    if not isinstance(binding, HEPProviderBinding):
        raise TypeError("binding must be HEPProviderBinding.")
    if not isinstance(normalization, ProcessNormalization):
        raise TypeError("normalization must be ProcessNormalization.")
    if not isinstance(matrix_element_revision, MatrixElementRevision):
        raise TypeError("matrix_element_revision must be MatrixElementRevision.")
    stage_ = ProviderExecutionStage(stage)
    capability = _identifier(required_capability, "Required capability")
    input_profile = _identifier(input_profile_id, "Input profile ID")
    output_profile = _identifier(output_profile_id, "Output profile ID")
    process = _identifier(process_id, "Process ID")
    units = _identifier(unit_contract_id, "Unit contract ID")
    frame = _identifier(frame_id, "Frame ID")
    frame_realization = _identifier(frame_realization_id, "Frame realization ID")
    if len(frame_realization) != 64 or any(
        value not in "0123456789abcdef" for value in frame_realization
    ):
        raise ValueError("frame_realization_id must be a lowercase SHA-256 digest.")
    input_weights = _event_weights(input_event)
    output_weights = _event_weights(output_event)
    if not binding.supports(capability):
        status = ProviderExecutionStatus.CAPABILITY_MISMATCH
    elif input_profile not in binding.input_profile_ids:
        status = ProviderExecutionStatus.INPUT_PROFILE_MISMATCH
    elif output_profile not in binding.output_profile_ids:
        status = ProviderExecutionStatus.OUTPUT_PROFILE_MISMATCH
    elif units not in binding.unit_ids or frame not in binding.frame_ids:
        status = ProviderExecutionStatus.UNIT_FRAME_MISMATCH
    elif (
        normalization.process_id != process
        or normalization.provider_id != binding.provider_id
    ):
        status = ProviderExecutionStatus.PROCESS_MISMATCH
    elif (
        matrix_element_revision.process_id != process
        or matrix_element_revision.provider_id != binding.provider_id
        or matrix_element_revision.normalization_id != normalization.normalization_id
        or matrix_element_revision.differentiation_id
        != binding.differentiation.evidence_id
    ):
        status = ProviderExecutionStatus.MATRIX_REVISION_MISMATCH
    elif not bool(
        jnp.all(input_weights.finite | ~input_weights.event_active)
    ) or not bool(jnp.all(output_weights.finite | ~output_weights.event_active)):
        status = ProviderExecutionStatus.NONFINITE_WEIGHT
    else:
        status = ProviderExecutionStatus.SUCCESS
    input_event_id = _event_identity(input_event)
    output_event_id = _event_identity(output_event)
    execution_id = canonical_fingerprint(
        {
            "kind": "hep-provider-execution",
            "binding": binding.binding_id,
            "stage": stage_.value,
            "capability": capability,
            "profiles": [input_profile, output_profile],
            "units": units,
            "frame": frame,
            "frame_realization": frame_realization,
            "events": [input_event_id, output_event_id],
            "weights": [input_weights.snapshot_id, output_weights.snapshot_id],
            "normalization": normalization.normalization_id,
            "matrix_element_revision": matrix_element_revision.revision_id,
            "process": process,
            "status": int(status),
        }
    )
    return ProviderExecutionRecord(
        binding,
        normalization,
        matrix_element_revision,
        input_weights,
        output_weights,
        status,
        stage_,
        capability,
        input_profile,
        output_profile,
        input_event_id,
        output_event_id,
        process,
        units,
        frame,
        frame_realization,
        execution_id,
    )


__all__ = [
    "ExclusiveMatchingAssignment",
    "NamedWeightSnapshot",
    "ProviderExecutionChain",
    "ProviderExecutionRecord",
    "ProviderExecutionStage",
    "ProviderExecutionStatus",
    "assign_exclusive_matching_bins",
    "record_provider_execution",
]
