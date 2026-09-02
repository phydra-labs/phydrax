#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._moving_conservation import ConservativeRemapPlan


TopologyAdjointPolicy = Literal["frozen_event", "smooth_surrogate", "unsupported"]


class ReverseCheckpointSchedule(StrictModule, NonTrainableState):
    step_count: int = eqx.field(static=True)
    checkpoint_budget: int = eqx.field(static=True)
    checkpoint_indices: tuple[int, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, step_count: int, checkpoint_budget: int, /):
        steps = int(step_count)
        budget = int(checkpoint_budget)
        if steps <= 0 or budget <= 0:
            raise ValueError("Reverse checkpoint schedule controls must be positive.")
        count = min(steps + 1, budget + 1)
        indices = tuple(
            sorted(set(int(round(value)) for value in np.linspace(0, steps, count)))
        )
        if indices[0] != 0 or indices[-1] != steps:
            raise RuntimeError("Reverse checkpoint schedule endpoints are missing.")
        self.step_count = steps
        self.checkpoint_budget = budget
        self.checkpoint_indices = indices
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "reverse-checkpoint-schedule",
                "step_count": steps,
                "checkpoint_budget": budget,
                "indices": indices,
            }
        )

    def should_checkpoint(self, step_index: int, /) -> bool:
        return int(step_index) in self.checkpoint_indices


class AcceptedStepAdjointRecord(StrictModule, NonTrainableState):
    pullback: Callable = eqx.field(static=True)
    step_index: int = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        pullback: Callable,
        step_index: int,
        decision_id: str,
        /,
        *,
        valid: bool = True,
    ):
        if not callable(pullback) or int(step_index) < 0 or not str(decision_id):
            raise ValueError("Accepted-step adjoint record is invalid.")
        self.pullback = pullback
        self.step_index = int(step_index)
        self.decision_id = str(decision_id)
        self.valid = bool(valid)
        self.record_id = canonical_fingerprint(
            {
                "kind": "accepted-step-adjoint-record",
                "step": self.step_index,
                "decision": self.decision_id,
                "valid": self.valid,
            }
        )


class TopologyAdjointEvent(StrictModule, NonTrainableState):
    pullback: Callable = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    policy: TopologyAdjointPolicy = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        pullback: Callable,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        /,
        *,
        policy: TopologyAdjointPolicy,
        event_id: str,
    ):
        source = tuple(int(value) for value in source_shape)
        target = tuple(int(value) for value in target_shape)
        if (
            not callable(pullback)
            or any(value <= 0 for value in (*source, *target))
            or policy not in ("frozen_event", "smooth_surrogate", "unsupported")
            or not str(event_id)
        ):
            raise ValueError("Topology adjoint event is invalid.")
        self.pullback = pullback
        self.source_shape = source
        self.target_shape = target
        self.policy = policy
        self.event_id = canonical_fingerprint(
            {
                "kind": "topology-adjoint-event",
                "source_shape": source,
                "target_shape": target,
                "policy": policy,
                "event": str(event_id),
            }
        )

    def apply(self, target_cotangent: ArrayLike, /) -> Array:
        value = jnp.asarray(target_cotangent)
        if value.shape != self.target_shape:
            raise ValueError("Topology event cotangent has wrong target shape.")
        if self.policy == "unsupported":
            return jnp.zeros(self.source_shape, dtype=value.dtype)
        source = jnp.asarray(self.pullback(value))
        if source.shape != self.source_shape:
            raise ValueError("Topology event pullback changed source shape.")
        return source


class TopologyAdjointResult(StrictModule, NonTrainableState):
    initial_cotangent: Array
    valid: Array
    traversed_record_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class ReverseTimeTopologyTape(StrictModule, NonTrainableState):
    records: tuple[AcceptedStepAdjointRecord | TopologyAdjointEvent, ...]
    tape_id: str = eqx.field(static=True)

    def __init__(
        self,
        records: Sequence[AcceptedStepAdjointRecord | TopologyAdjointEvent] = (),
        /,
    ):
        values = tuple(records)
        if any(
            not isinstance(value, (AcceptedStepAdjointRecord, TopologyAdjointEvent))
            for value in values
        ):
            raise TypeError("Reverse topology tape records are invalid.")
        self.records = values
        self.tape_id = canonical_fingerprint(
            {
                "kind": "reverse-time-topology-tape",
                "records": tuple(
                    value.record_id
                    if isinstance(value, AcceptedStepAdjointRecord)
                    else value.event_id
                    for value in values
                ),
            }
        )

    def append_step(
        self, record: AcceptedStepAdjointRecord, /
    ) -> "ReverseTimeTopologyTape":
        return ReverseTimeTopologyTape((*self.records, record))

    def append_event(self, event: TopologyAdjointEvent, /) -> "ReverseTimeTopologyTape":
        return ReverseTimeTopologyTape((*self.records, event))

    def reverse(self, final_cotangent: ArrayLike, /) -> TopologyAdjointResult:
        cotangent = jnp.asarray(final_cotangent)
        valid = jnp.asarray(True)
        traversed = []
        for record in reversed(self.records):
            if isinstance(record, AcceptedStepAdjointRecord):
                pulled = record.pullback(cotangent)
                cotangent = jnp.asarray(
                    pulled[0] if isinstance(pulled, tuple) else pulled
                )
                valid = valid & record.valid
                traversed.append(record.record_id)
            else:
                cotangent = record.apply(cotangent)
                valid = valid & (record.policy != "unsupported")
                traversed.append(record.event_id)
        result_id = canonical_fingerprint(
            {
                "kind": "topology-adjoint-result",
                "tape": self.tape_id,
                "traversed": tuple(traversed),
            }
        )
        return TopologyAdjointResult(
            cotangent,
            valid,
            tuple(traversed),
            result_id,
        )


def conservative_remap_adjoint_event(
    remap: ConservativeRemapPlan,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    /,
    *,
    event_id: str,
) -> TopologyAdjointEvent:
    if not isinstance(remap, ConservativeRemapPlan):
        raise TypeError("remap must be ConservativeRemapPlan.")
    return TopologyAdjointEvent(
        remap.transpose_apply,
        source_shape,
        target_shape,
        policy="frozen_event",
        event_id=event_id,
    )


__all__ = [
    "AcceptedStepAdjointRecord",
    "ReverseCheckpointSchedule",
    "ReverseTimeTopologyTape",
    "TopologyAdjointEvent",
    "TopologyAdjointPolicy",
    "TopologyAdjointResult",
    "conservative_remap_adjoint_event",
]
