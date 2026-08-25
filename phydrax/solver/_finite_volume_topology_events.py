#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Mapping, Sequence, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_EnumT = TypeVar("_EnumT", bound=IntEnum)


class TopologyEventKind(IntEnum):
    REMESH = 0
    AMR_REGRID = 1
    OVERSET_DONOR_REBUILD = 2


class TopologyEventState(IntEnum):
    REQUESTED = 0
    COMMITTED = 1
    FAILED = 2


class TopologyEventStatus(IntEnum):
    PENDING = 0
    SUCCESS = 1
    FAILED = 2
    FAILED_MISSING_ARTIFACT = 3
    FAILED_COVERAGE = 4
    FAILED_POSITIVITY = 5
    FAILED_STALE_EPOCH = 6
    FAILED_RESOURCE_LIMIT = 7

    # Short names are retained as explicit aliases for callers that classify
    # failure reasons without the ``FAILED_`` prefix.
    MISSING_ARTIFACT = 3
    FAILED_ARTIFACT = 3
    COVERAGE = 4
    COVERAGE_FAILURE = 4
    POSITIVITY = 5
    POSITIVITY_FAILURE = 5
    STALE_EPOCH = 6
    RESOURCE_LIMIT = 7


class FiniteVolumeTopologyArtifactEvidence(StrictModule, NonTrainableState):
    """Minimal typed success/failure evidence for one topology transaction."""

    passed: Array
    status: Array
    coverage_error: Array
    conservation_defect: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        passed: ArrayLike,
        status: ArrayLike,
        coverage_error: ArrayLike,
        conservation_defect: ArrayLike,
        evidence_id: str,
    ):
        identifier = _required_identifier(evidence_id, "evidence_id")
        passed_ = jnp.asarray(passed)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if passed_.shape != () or passed_.dtype.kind != "b":
            raise ValueError("passed must be a scalar boolean.")
        if status_.shape != ():
            raise ValueError("status must be a scalar.")
        self.passed = passed_
        self.status = status_
        self.coverage_error = jnp.asarray(coverage_error)
        self.conservation_defect = jnp.asarray(conservation_defect)
        self.evidence_id = identifier


def _require_identifier(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _required_identifier(value: str | None, name: str, /) -> str:
    identifier = _require_identifier(value, name)
    if identifier is None:
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return identifier


def _enum_member(value: Any, enum: type[_EnumT], name: str, /) -> _EnumT:
    if isinstance(value, IntEnum):
        if isinstance(value, enum):
            return value
        raise ValueError(f"{name} is not a member of {enum.__name__}.")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be {enum.__name__}.")
    integer = int(value)
    for member in enum:
        if member.value == integer:
            return member
    raise ValueError(f"{name} is not a valid {enum.__name__}.")


def _is_failed_status(value: TopologyEventStatus, /) -> bool:
    return value.value >= int(TopologyEventStatus.FAILED)


def _host_scalar(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return array


def _host_nonnegative_integer(value: ArrayLike, name: str, /) -> int:
    array = _host_scalar(value, name)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise TypeError(f"{name} must be an integer scalar.")
    integer = int(array)
    if integer < 0 or integer > np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be a nonnegative int32 value.")
    return integer


def _host_finite_time(value: ArrayLike, name: str, /) -> tuple[float, np.ndarray]:
    array = _host_scalar(value, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar.")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar, array


def _strict_archive_record(
    value: Any,
    fields: frozenset[str],
    name: str,
    /,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} archive fields changed.")
    return value


def _archive_schema(value: Any, name: str, /) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"Unsupported {name} archive schema.")


class FiniteVolumeTopologyEpoch(StrictModule, NonTrainableState):
    """Immutable identities for one fully prepared finite-volume topology."""

    parent_epoch_id: str | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_artifact_id: str | None = eqx.field(static=True)
    metrics_artifact_id: str | None = eqx.field(static=True)
    operators_artifact_id: str | None = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_id: str,
        topology_id: str,
        geometry_id: str,
        /,
        *,
        parent_epoch_id: str | None = None,
        topology_artifact_id: str | None = None,
        metrics_artifact_id: str | None = None,
        operators_artifact_id: str | None = None,
    ):
        parent = _require_identifier(parent_epoch_id, "parent_epoch_id")
        prepared = _required_identifier(prepared_id, "prepared_id")
        topology = _required_identifier(topology_id, "topology_id")
        geometry = _required_identifier(geometry_id, "geometry_id")
        topology_artifact = _require_identifier(
            topology_artifact_id, "topology_artifact_id"
        )
        metrics_artifact = _require_identifier(metrics_artifact_id, "metrics_artifact_id")
        operators_artifact = _require_identifier(
            operators_artifact_id, "operators_artifact_id"
        )
        self.parent_epoch_id = parent
        self.prepared_id = prepared
        self.topology_id = topology
        self.geometry_id = geometry
        self.topology_artifact_id = topology_artifact
        self.metrics_artifact_id = metrics_artifact
        self.operators_artifact_id = operators_artifact
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "finite-volume-topology-epoch",
                "schema_version": 1,
                "parent_epoch_id": parent,
                "prepared_id": prepared,
                "topology_id": topology,
                "geometry_id": geometry,
                "topology_artifact_id": topology_artifact,
                "metrics_artifact_id": metrics_artifact,
                "operators_artifact_id": operators_artifact,
            }
        )

    def to_archive_record(self) -> dict[str, Any]:
        """Return the complete finite JSON record used by restart archives."""

        return {
            "schema_version": 1,
            "parent_epoch_id": self.parent_epoch_id,
            "prepared_id": self.prepared_id,
            "topology_id": self.topology_id,
            "geometry_id": self.geometry_id,
            "topology_artifact_id": self.topology_artifact_id,
            "metrics_artifact_id": self.metrics_artifact_id,
            "operators_artifact_id": self.operators_artifact_id,
            "epoch_id": self.epoch_id,
        }

    @classmethod
    def from_archive_record(cls, record: dict[str, Any], /) -> FiniteVolumeTopologyEpoch:
        """Strictly reconstruct an epoch and verify its content identity."""

        payload = _strict_archive_record(
            record,
            frozenset(
                (
                    "schema_version",
                    "parent_epoch_id",
                    "prepared_id",
                    "topology_id",
                    "geometry_id",
                    "topology_artifact_id",
                    "metrics_artifact_id",
                    "operators_artifact_id",
                    "epoch_id",
                )
            ),
            "Topology epoch",
        )
        _archive_schema(payload["schema_version"], "topology epoch")
        expected_epoch_id = _required_identifier(payload["epoch_id"], "epoch_id")
        epoch = cls(
            payload["prepared_id"],
            payload["topology_id"],
            payload["geometry_id"],
            parent_epoch_id=payload["parent_epoch_id"],
            topology_artifact_id=payload["topology_artifact_id"],
            metrics_artifact_id=payload["metrics_artifact_id"],
            operators_artifact_id=payload["operators_artifact_id"],
        )
        if epoch.epoch_id != expected_epoch_id:
            raise ValueError("Topology epoch archive identity changed.")
        return epoch


class FiniteVolumeTopologyEventRequest(StrictModule, NonTrainableState):
    """Solver-owned request for a post-accept topology transaction.

    This schema deliberately contains identities only. Geometry intersection and
    topology construction remain host responsibilities of a later transaction layer.
    """

    kind: TopologyEventKind = eqx.field(static=True)
    input_epoch_id: str = eqx.field(static=True)
    requested_spec_id: str = eqx.field(static=True)
    payload_id: str | None = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: TopologyEventKind,
        input_epoch_id: str,
        requested_spec_id: str,
        /,
        *,
        payload_id: str | None = None,
        reason: str = "",
    ):
        event_kind = _enum_member(kind, TopologyEventKind, "kind")
        input_epoch = _required_identifier(input_epoch_id, "input_epoch_id")
        requested_spec = _required_identifier(requested_spec_id, "requested_spec_id")
        payload = _require_identifier(payload_id, "payload_id")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string.")
        self.kind = event_kind
        self.input_epoch_id = input_epoch
        self.requested_spec_id = requested_spec
        self.payload_id = payload
        self.reason = reason
        self.request_id = canonical_fingerprint(
            {
                "kind": "finite-volume-topology-event-request",
                "schema_version": 1,
                "event_kind": int(event_kind),
                "input_epoch_id": input_epoch,
                "requested_spec_id": requested_spec,
                "payload_id": payload,
                "reason": reason,
            }
        )


class FiniteVolumeTopologyEvent(StrictModule, NonTrainableState):
    """Immutable materialized view of one journal slot."""

    sequence: int = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    kind: TopologyEventKind = eqx.field(static=True)
    state: TopologyEventState = eqx.field(static=True)
    status: TopologyEventStatus = eqx.field(static=True)
    requested_id: str = eqx.field(static=True)
    input_epoch_id: str = eqx.field(static=True)
    result_id: str | None = eqx.field(static=True)
    payload_id: str | None = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        sequence: int,
        accepted_step: int,
        time: float,
        kind: TopologyEventKind,
        state: TopologyEventState,
        status: TopologyEventStatus,
        requested_id: str,
        input_epoch_id: str,
        result_id: str | None,
        payload_id: str | None,
        /,
    ):
        sequence_ = _host_nonnegative_integer(sequence, "sequence")
        accepted_step_ = _host_nonnegative_integer(accepted_step, "accepted_step")
        time_, _ = _host_finite_time(time, "time")
        event_kind = _enum_member(kind, TopologyEventKind, "kind")
        event_state = _enum_member(state, TopologyEventState, "state")
        event_status = _enum_member(status, TopologyEventStatus, "status")
        requested = _required_identifier(requested_id, "requested_id")
        input_epoch = _required_identifier(input_epoch_id, "input_epoch_id")
        result = _require_identifier(result_id, "result_id")
        payload = _require_identifier(payload_id, "payload_id")
        if event_state is TopologyEventState.REQUESTED:
            if event_status is not TopologyEventStatus.PENDING or result is not None:
                raise ValueError("A requested event must be pending without a result.")
        elif event_state is TopologyEventState.COMMITTED:
            if event_status is not TopologyEventStatus.SUCCESS or result is None:
                raise ValueError("A committed event must have a successful result.")
        elif event_state is TopologyEventState.FAILED:
            if not _is_failed_status(event_status):
                raise ValueError("A failed event must have failed status.")
        self.sequence = sequence_
        self.accepted_step = accepted_step_
        self.time = time_
        self.kind = event_kind
        self.state = event_state
        self.status = event_status
        self.requested_id = requested
        self.input_epoch_id = input_epoch
        self.result_id = result
        self.payload_id = payload
        self.event_id = canonical_fingerprint(
            {
                "kind": "finite-volume-topology-event",
                "schema_version": 1,
                "sequence": sequence_,
                "accepted_step": accepted_step_,
                "time": float(time_).hex(),
                "event_kind": int(event_kind),
                "state": int(event_state),
                "status": int(event_status),
                "requested_id": requested,
                "input_epoch_id": input_epoch,
                "result_id": result,
                "payload_id": payload,
            }
        )

    def to_archive_record(self) -> dict[str, Any]:
        """Return the complete finite JSON record used by restart archives."""

        return {
            "schema_version": 1,
            "sequence": self.sequence,
            "accepted_step": self.accepted_step,
            "time_hex": float(self.time).hex(),
            "kind": int(self.kind),
            "state": int(self.state),
            "status": int(self.status),
            "requested_id": self.requested_id,
            "input_epoch_id": self.input_epoch_id,
            "result_id": self.result_id,
            "payload_id": self.payload_id,
            "event_id": self.event_id,
        }

    @classmethod
    def from_archive_record(cls, record: dict[str, Any], /) -> FiniteVolumeTopologyEvent:
        """Strictly reconstruct an event and verify its content identity."""

        payload = _strict_archive_record(
            record,
            frozenset(
                (
                    "schema_version",
                    "sequence",
                    "accepted_step",
                    "time_hex",
                    "kind",
                    "state",
                    "status",
                    "requested_id",
                    "input_epoch_id",
                    "result_id",
                    "payload_id",
                    "event_id",
                )
            ),
            "Topology event",
        )
        _archive_schema(payload["schema_version"], "topology event")
        time_hex = payload["time_hex"]
        if not isinstance(time_hex, str):
            raise TypeError("Topology event archive time_hex must be a string.")
        try:
            time = float.fromhex(time_hex)
        except ValueError as error:
            raise ValueError("Topology event archive time_hex is invalid.") from error
        expected_event_id = _required_identifier(payload["event_id"], "event_id")
        event = cls(
            payload["sequence"],
            payload["accepted_step"],
            time,
            payload["kind"],
            payload["state"],
            payload["status"],
            payload["requested_id"],
            payload["input_epoch_id"],
            payload["result_id"],
            payload["payload_id"],
        )
        if event.event_id != expected_event_id:
            raise ValueError("Topology event archive identity changed.")
        return event


class FiniteVolumeTopologyEventJournal(StrictModule, NonTrainableState):
    """Fixed-capacity accepted-step topology journal.

    Lifecycle operations are host-side validation boundaries. The bounded numeric
    storage remains a JAX pytree so compiled runtime code can inspect journal state
    without carrying Python event objects through a trace.
    """

    kinds: Array
    states: Array
    statuses: Array
    accepted_steps: Array
    times: Array
    next_sequence: Array
    count: Array
    overflowed: Array
    current_epoch_id: str = eqx.field(static=True)
    epoch_table: tuple[FiniteVolumeTopologyEpoch, ...] = eqx.field(static=True)
    input_epoch_ids: tuple[str | None, ...] = eqx.field(static=True)
    requested_ids: tuple[str | None, ...] = eqx.field(static=True)
    result_ids: tuple[str | None, ...] = eqx.field(static=True)
    payload_ids: tuple[str | None, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    journal_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_epoch: FiniteVolumeTopologyEpoch,
        /,
        *,
        capacity: int,
        time: ArrayLike = 0.0,
        _storage: dict[str, Any] | None = None,
    ):
        if not isinstance(initial_epoch, FiniteVolumeTopologyEpoch):
            raise TypeError("initial_epoch must be FiniteVolumeTopologyEpoch.")
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
            or capacity > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Topology event journal capacity must be a positive int32 value."
            )
        _, initial_time = _host_finite_time(time, "time")
        if not np.issubdtype(initial_time.dtype, np.floating):
            initial_time = initial_time.astype(np.float64)
        if _storage is None:
            storage: dict[str, Any] = {
                "kinds": jnp.full((capacity,), -1, dtype=jnp.int32),
                "states": jnp.full((capacity,), -1, dtype=jnp.int32),
                "statuses": jnp.full((capacity,), -1, dtype=jnp.int32),
                "accepted_steps": jnp.full((capacity,), -1, dtype=jnp.int32),
                "times": jnp.full(
                    (capacity,), jnp.nan, dtype=jnp.asarray(initial_time).dtype
                ),
                "next_sequence": jnp.asarray(0, dtype=jnp.int32),
                "count": jnp.asarray(0, dtype=jnp.int32),
                "overflowed": jnp.asarray(False),
                "current_epoch_id": initial_epoch.epoch_id,
                "epoch_table": (initial_epoch,),
                "input_epoch_ids": (None,) * capacity,
                "requested_ids": (None,) * capacity,
                "result_ids": (None,) * capacity,
                "payload_ids": (None,) * capacity,
            }
        else:
            storage = _storage
        self.kinds = jnp.asarray(storage["kinds"], dtype=jnp.int32)
        self.states = jnp.asarray(storage["states"], dtype=jnp.int32)
        self.statuses = jnp.asarray(storage["statuses"], dtype=jnp.int32)
        self.accepted_steps = jnp.asarray(storage["accepted_steps"], dtype=jnp.int32)
        self.times = jnp.asarray(storage["times"])
        self.next_sequence = jnp.asarray(
            storage["next_sequence"], dtype=jnp.int32
        ).reshape(())
        self.count = jnp.asarray(storage["count"], dtype=jnp.int32).reshape(())
        self.overflowed = jnp.asarray(storage["overflowed"], dtype=jnp.bool_).reshape(())
        self.current_epoch_id = storage["current_epoch_id"]
        self.epoch_table = tuple(storage["epoch_table"])
        self.input_epoch_ids = tuple(storage["input_epoch_ids"])
        self.requested_ids = tuple(storage["requested_ids"])
        self.result_ids = tuple(storage["result_ids"])
        self.payload_ids = tuple(storage["payload_ids"])
        self.capacity = capacity
        self._validate()
        self.journal_id = self._content_id()

    @classmethod
    def allocate(
        cls,
        initial_epoch: FiniteVolumeTopologyEpoch,
        /,
        *,
        capacity: int,
        time: ArrayLike = 0.0,
    ) -> FiniteVolumeTopologyEventJournal:
        return cls(initial_epoch, capacity=capacity, time=time)

    @classmethod
    def from_events(
        cls,
        initial_epoch: FiniteVolumeTopologyEpoch,
        events: tuple[FiniteVolumeTopologyEvent, ...],
        /,
        *,
        result_epochs: tuple[FiniteVolumeTopologyEpoch, ...] = (),
        capacity: int,
        overflowed: bool = False,
        time: ArrayLike = 0.0,
    ) -> FiniteVolumeTopologyEventJournal:
        """Reconstruct and verify an immutable journal from persisted records."""
        if not isinstance(initial_epoch, FiniteVolumeTopologyEpoch):
            raise TypeError("initial_epoch must be FiniteVolumeTopologyEpoch.")
        if not isinstance(events, tuple) or any(
            not isinstance(event, FiniteVolumeTopologyEvent) for event in events
        ):
            raise TypeError("events must be a tuple of FiniteVolumeTopologyEvent.")
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
            or capacity > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Topology event journal capacity must be a positive int32 value."
            )
        if not isinstance(result_epochs, tuple) or any(
            not isinstance(epoch, FiniteVolumeTopologyEpoch) for epoch in result_epochs
        ):
            raise TypeError("result_epochs must be a tuple of FiniteVolumeTopologyEpoch.")
        if not isinstance(overflowed, bool):
            raise TypeError("overflowed must be boolean.")
        if len(events) > capacity:
            raise ValueError("Persisted topology events exceed journal capacity.")
        for sequence, event in enumerate(events):
            if event.sequence != sequence:
                raise ValueError(
                    "Persisted topology event sequences must be contiguous from zero."
                )

        _, time_storage = _host_finite_time(time, "time")
        if not np.issubdtype(time_storage.dtype, np.floating):
            time_storage = time_storage.astype(np.float64)
        count = len(events)
        kinds = np.full((capacity,), -1, dtype=np.int32)
        states = np.full((capacity,), -1, dtype=np.int32)
        statuses = np.full((capacity,), -1, dtype=np.int32)
        accepted_steps = np.full((capacity,), -1, dtype=np.int32)
        times = np.full((capacity,), np.nan, dtype=time_storage.dtype)
        input_epoch_ids: list[str | None] = [None] * capacity
        requested_ids: list[str | None] = [None] * capacity
        result_ids: list[str | None] = [None] * capacity
        payload_ids: list[str | None] = [None] * capacity
        for sequence, event in enumerate(events):
            kinds[sequence] = int(event.kind)
            states[sequence] = int(event.state)
            statuses[sequence] = int(event.status)
            accepted_steps[sequence] = event.accepted_step
            times[sequence] = event.time
            input_epoch_ids[sequence] = event.input_epoch_id
            requested_ids[sequence] = event.requested_id
            result_ids[sequence] = event.result_id
            payload_ids[sequence] = event.payload_id

        epoch_table = (initial_epoch, *result_epochs)
        journal = cls(
            initial_epoch,
            capacity=capacity,
            _storage={
                "kinds": kinds,
                "states": states,
                "statuses": statuses,
                "accepted_steps": accepted_steps,
                "times": times,
                "next_sequence": np.asarray(count, dtype=np.int32),
                "count": np.asarray(count, dtype=np.int32),
                "overflowed": np.asarray(overflowed),
                "current_epoch_id": epoch_table[-1].epoch_id,
                "epoch_table": epoch_table,
                "input_epoch_ids": tuple(input_epoch_ids),
                "requested_ids": tuple(requested_ids),
                "result_ids": tuple(result_ids),
                "payload_ids": tuple(payload_ids),
            },
        )
        if any(
            journal.event(sequence).event_id != event.event_id
            for sequence, event in enumerate(events)
        ):
            raise ValueError(
                "Persisted topology event fields changed during reconstruction."
            )
        return journal

    def archive_arrays(self) -> dict[str, np.ndarray]:
        """Return every fixed-capacity numeric journal leaf for persistence."""

        return {
            "kinds": np.asarray(self.kinds),
            "states": np.asarray(self.states),
            "statuses": np.asarray(self.statuses),
            "accepted_steps": np.asarray(self.accepted_steps),
            "times": np.asarray(self.times),
            "next_sequence": np.asarray(self.next_sequence),
            "count": np.asarray(self.count),
            "overflowed": np.asarray(self.overflowed),
        }

    def to_archive_record(self) -> dict[str, Any]:
        """Return all static journal records paired with :meth:`archive_arrays`."""

        count = int(np.asarray(self.count))
        return {
            "schema_version": 1,
            "capacity": self.capacity,
            "journal_id": self.journal_id,
            "current_epoch_id": self.current_epoch_id,
            "times_dtype": np.dtype(self.times.dtype).name,
            "epoch_table": [epoch.to_archive_record() for epoch in self.epoch_table],
            "input_epoch_ids": list(self.input_epoch_ids),
            "requested_ids": list(self.requested_ids),
            "result_ids": list(self.result_ids),
            "payload_ids": list(self.payload_ids),
            "events": [
                self.event(sequence).to_archive_record() for sequence in range(count)
            ],
        }

    @classmethod
    def from_archive_record(
        cls,
        record: dict[str, Any],
        arrays: dict[str, Any],
        /,
    ) -> FiniteVolumeTopologyEventJournal:
        """Strictly reconstruct every numeric and static journal leaf."""

        payload = _strict_archive_record(
            record,
            frozenset(
                (
                    "schema_version",
                    "capacity",
                    "journal_id",
                    "current_epoch_id",
                    "times_dtype",
                    "epoch_table",
                    "input_epoch_ids",
                    "requested_ids",
                    "result_ids",
                    "payload_ids",
                    "events",
                )
            ),
            "Topology event journal",
        )
        _archive_schema(payload["schema_version"], "topology event journal")
        capacity = payload["capacity"]
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
            or capacity > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Topology event journal archive capacity must be a positive int32 value."
            )
        array_names = frozenset(
            (
                "kinds",
                "states",
                "statuses",
                "accepted_steps",
                "times",
                "next_sequence",
                "count",
                "overflowed",
            )
        )
        if not isinstance(arrays, dict) or set(arrays) != array_names:
            raise ValueError("Topology event journal archive array inventory changed.")
        archived_arrays = {name: np.asarray(value) for name, value in arrays.items()}
        for name in ("kinds", "states", "statuses", "accepted_steps"):
            value = archived_arrays[name]
            if value.shape != (capacity,) or value.dtype != np.dtype(np.int32):
                raise ValueError(
                    f"Topology event journal archive array {name!r} changed."
                )
        times = archived_arrays["times"]
        if (
            times.shape != (capacity,)
            or times.dtype.kind != "f"
            or not isinstance(payload["times_dtype"], str)
            or np.dtype(times.dtype).name != payload["times_dtype"]
        ):
            raise ValueError("Topology event journal archive times changed.")
        for name in ("next_sequence", "count"):
            value = archived_arrays[name]
            if value.shape != () or value.dtype != np.dtype(np.int32):
                raise ValueError(
                    f"Topology event journal archive scalar {name!r} changed."
                )
        overflowed = archived_arrays["overflowed"]
        if overflowed.shape != () or overflowed.dtype != np.dtype(np.bool_):
            raise ValueError("Topology event journal archive overflow flag changed.")

        epoch_records = payload["epoch_table"]
        event_records = payload["events"]
        if not isinstance(epoch_records, list) or not epoch_records:
            raise ValueError("Topology event journal archive epoch table is invalid.")
        if not isinstance(event_records, list):
            raise ValueError("Topology event journal archive events are invalid.")
        epochs = tuple(
            FiniteVolumeTopologyEpoch.from_archive_record(epoch)
            for epoch in epoch_records
        )
        events = tuple(
            FiniteVolumeTopologyEvent.from_archive_record(event)
            for event in event_records
        )
        static_table_names = (
            "input_epoch_ids",
            "requested_ids",
            "result_ids",
            "payload_ids",
        )
        for name in static_table_names:
            table = payload[name]
            if not isinstance(table, list) or len(table) != capacity:
                raise ValueError(
                    f"Topology event journal archive table {name!r} changed."
                )
            for value in table:
                _require_identifier(value, name)
        journal = cls.from_events(
            epochs[0],
            events,
            result_epochs=epochs[1:],
            capacity=capacity,
            overflowed=bool(overflowed),
            time=np.asarray(0.0, dtype=times.dtype),
        )
        reconstructed_arrays = journal.archive_arrays()
        for name in array_names:
            if not np.array_equal(
                archived_arrays[name],
                reconstructed_arrays[name],
                equal_nan=True,
            ):
                raise ValueError(
                    f"Topology event journal archive array {name!r} is inconsistent."
                )
        for name in static_table_names:
            if tuple(payload[name]) != getattr(journal, name):
                raise ValueError(
                    f"Topology event journal archive table {name!r} is inconsistent."
                )
        expected_current_epoch = _required_identifier(
            payload["current_epoch_id"], "current_epoch_id"
        )
        expected_journal_id = _required_identifier(payload["journal_id"], "journal_id")
        if (
            journal.current_epoch_id != expected_current_epoch
            or journal.journal_id != expected_journal_id
        ):
            raise ValueError("Topology event journal archive identity changed.")
        return journal

    def _validate(self) -> None:
        arrays = (
            self.kinds,
            self.states,
            self.statuses,
            self.accepted_steps,
            self.times,
        )
        if any(array.shape != (self.capacity,) for array in arrays):
            raise ValueError("Topology event journal array capacity changed.")
        if self.times.dtype.kind != "f":
            raise TypeError("Topology event journal times must have floating dtype.")
        static_tables = (
            self.input_epoch_ids,
            self.requested_ids,
            self.result_ids,
            self.payload_ids,
        )
        if any(len(table) != self.capacity for table in static_tables):
            raise ValueError("Topology event journal static capacity changed.")
        if not self.epoch_table or any(
            not isinstance(epoch, FiniteVolumeTopologyEpoch) for epoch in self.epoch_table
        ):
            raise TypeError("Topology event journal epoch table is invalid.")
        epoch_ids = tuple(epoch.epoch_id for epoch in self.epoch_table)
        if len(set(epoch_ids)) != len(epoch_ids):
            raise ValueError("Topology event journal epoch identities must be unique.")
        if self.epoch_table[0].parent_epoch_id is not None:
            raise ValueError("The initial topology epoch cannot have a parent.")
        for previous, current in zip(
            self.epoch_table[:-1], self.epoch_table[1:], strict=True
        ):
            if current.parent_epoch_id != previous.epoch_id:
                raise ValueError("Topology epoch table must form one exact parent chain.")
        current_epoch = _require_identifier(self.current_epoch_id, "current_epoch_id")
        if current_epoch != epoch_ids[-1]:
            raise ValueError("Current topology epoch must be the epoch-table tip.")
        count = _host_nonnegative_integer(self.count, "count")
        next_sequence = _host_nonnegative_integer(self.next_sequence, "next_sequence")
        overflowed = bool(_host_scalar(self.overflowed, "overflowed"))
        if count > self.capacity or next_sequence != count:
            raise ValueError("Topology event journal sequence/count invariant changed.")
        if overflowed and count != self.capacity:
            raise ValueError("Topology event journal overflow requires full capacity.")
        kinds = np.asarray(self.kinds)
        states = np.asarray(self.states)
        statuses = np.asarray(self.statuses)
        accepted_steps = np.asarray(self.accepted_steps)
        pending_input_epochs: list[str] = []
        times = np.asarray(self.times)
        historical_tip = epoch_ids[0]
        committed_count = 0
        previous_state: TopologyEventState | None = None
        previous_result: str | None = None
        previous_input: str | None = None
        previous_step = -1
        previous_time = np.nan
        for index in range(count):
            _enum_member(int(kinds[index]), TopologyEventKind, "kind")
            state = _enum_member(int(states[index]), TopologyEventState, "state")
            status = _enum_member(int(statuses[index]), TopologyEventStatus, "status")
            input_epoch = _required_identifier(
                self.input_epoch_ids[index], "input_epoch_id"
            )
            _required_identifier(self.requested_ids[index], "requested_id")
            result = _require_identifier(self.result_ids[index], "result_id")
            _require_identifier(self.payload_ids[index], "payload_id")
            if accepted_steps[index] < 0 or not np.isfinite(times[index]):
                raise ValueError("Topology event journal timing is invalid.")
            if index > 0 and (
                accepted_steps[index] < previous_step or times[index] < previous_time
            ):
                raise ValueError("Topology event journal timing must be monotone.")
            same_batch = (
                index > 0
                and accepted_steps[index] == previous_step
                and times[index] == previous_time
            )
            if state is TopologyEventState.REQUESTED:
                if input_epoch != historical_tip:
                    raise ValueError(
                        "Topology event input epoch does not match its historical tip."
                    )
                pending_input_epochs.append(input_epoch)
                if status is not TopologyEventStatus.PENDING or result is not None:
                    raise ValueError("Requested topology event slot is inconsistent.")
                if index != count - 1:
                    raise ValueError(
                        "A pending topology event must be the final journal record."
                    )
                if pending_input_epochs and (
                    input_epoch != self.current_epoch_id
                    or (len(pending_input_epochs) > 1 and not same_batch)
                ):
                    raise ValueError("Pending topology event batch is inconsistent.")
            elif state is TopologyEventState.COMMITTED:
                if (
                    status is not TopologyEventStatus.SUCCESS
                    or result is None
                    or result not in epoch_ids
                ):
                    raise ValueError("Committed topology event slot is inconsistent.")
                duplicate = (
                    result == historical_tip
                    and previous_state is TopologyEventState.COMMITTED
                    and previous_result == result
                    and previous_input == input_epoch
                    and same_batch
                )
                if duplicate:
                    pass
                elif (
                    input_epoch == historical_tip
                    and committed_count + 1 < len(epoch_ids)
                    and result == epoch_ids[committed_count + 1]
                ):
                    committed_count += 1
                    historical_tip = result
                else:
                    raise ValueError(
                        "Committed topology events must reproduce the epoch chain."
                    )
            elif state is TopologyEventState.FAILED:
                if not _is_failed_status(status):
                    raise ValueError("Failed topology event slot is inconsistent.")
                if input_epoch != historical_tip:
                    raise ValueError(
                        "Failed topology event input epoch does not match the historical tip."
                    )
            previous_state = state
            previous_result = result
            previous_input = input_epoch
            previous_step = int(accepted_steps[index])
            previous_time = float(times[index])
        if pending_input_epochs and pending_input_epochs[0] != self.current_epoch_id:
            raise ValueError("Pending topology event input epoch is stale.")
        if (
            committed_count != len(epoch_ids) - 1
            or historical_tip != self.current_epoch_id
        ):
            raise ValueError(
                "Every noninitial topology epoch must have one committed event."
            )
        for index in range(count, self.capacity):
            if (
                kinds[index] != -1
                or states[index] != -1
                or statuses[index] != -1
                or accepted_steps[index] != -1
                or not np.isnan(times[index])
                or any(table[index] is not None for table in static_tables)
            ):
                raise ValueError("Unused topology event journal slots must be empty.")

    def _content_id(self) -> str:
        count = int(np.asarray(self.count))
        return canonical_fingerprint(
            {
                "kind": "finite-volume-topology-event-journal",
                "schema_version": 1,
                "capacity": self.capacity,
                "current_epoch_id": self.current_epoch_id,
                "epoch_ids": [epoch.epoch_id for epoch in self.epoch_table],
                "next_sequence": int(np.asarray(self.next_sequence)),
                "count": count,
                "overflowed": bool(np.asarray(self.overflowed)),
                "times_dtype": np.dtype(self.times.dtype).name,
                "kinds": np.asarray(self.kinds[:count]).tolist(),
                "states": np.asarray(self.states[:count]).tolist(),
                "statuses": np.asarray(self.statuses[:count]).tolist(),
                "accepted_steps": np.asarray(self.accepted_steps[:count]).tolist(),
                "times": [float(value).hex() for value in np.asarray(self.times[:count])],
                "input_epoch_ids": list(self.input_epoch_ids[:count]),
                "requested_ids": list(self.requested_ids[:count]),
                "result_ids": list(self.result_ids[:count]),
                "payload_ids": list(self.payload_ids[:count]),
            }
        )

    def _new(self, **updates: Any) -> FiniteVolumeTopologyEventJournal:
        storage: dict[str, Any] = {
            "kinds": self.kinds,
            "states": self.states,
            "statuses": self.statuses,
            "accepted_steps": self.accepted_steps,
            "times": self.times,
            "next_sequence": self.next_sequence,
            "count": self.count,
            "overflowed": self.overflowed,
            "current_epoch_id": self.current_epoch_id,
            "epoch_table": self.epoch_table,
            "input_epoch_ids": self.input_epoch_ids,
            "requested_ids": self.requested_ids,
            "result_ids": self.result_ids,
            "payload_ids": self.payload_ids,
        }
        storage.update(updates)
        return type(self)(
            self.epoch_table[0],
            capacity=self.capacity,
            time=jnp.asarray(0.0, dtype=self.times.dtype),
            _storage=storage,
        )

    def append_requested(
        self,
        request: FiniteVolumeTopologyEventRequest,
        accepted_step: ArrayLike,
        time: ArrayLike,
        /,
    ) -> FiniteVolumeTopologyEventJournal:
        if not isinstance(request, FiniteVolumeTopologyEventRequest):
            raise TypeError("request must be FiniteVolumeTopologyEventRequest.")
        if request.input_epoch_id != self.current_epoch_id:
            raise ValueError("Topology event request input epoch is stale.")
        accepted_step_ = _host_nonnegative_integer(accepted_step, "accepted_step")
        time_, _ = _host_finite_time(time, "time")
        count = int(np.asarray(self.count))
        if any(
            state == int(TopologyEventState.REQUESTED)
            for state in np.asarray(self.states[:count])
        ):
            raise ValueError("Topology event journal already has a pending request.")
        if count:
            if accepted_step_ < int(np.asarray(self.accepted_steps[count - 1])):
                raise ValueError("Topology event accepted steps must be monotone.")
            if time_ < float(np.asarray(self.times[count - 1])):
                raise ValueError("Topology event times must be monotone.")
        if count == self.capacity:
            return self._new(overflowed=jnp.asarray(True))
        input_epoch_ids = list(self.input_epoch_ids)
        requested_ids = list(self.requested_ids)
        payload_ids = list(self.payload_ids)
        input_epoch_ids[count] = request.input_epoch_id
        requested_ids[count] = request.request_id
        payload_ids[count] = request.payload_id
        next_count = count + 1
        return self._new(
            kinds=self.kinds.at[count].set(int(request.kind)),
            states=self.states.at[count].set(int(TopologyEventState.REQUESTED)),
            statuses=self.statuses.at[count].set(int(TopologyEventStatus.PENDING)),
            accepted_steps=self.accepted_steps.at[count].set(accepted_step_),
            times=self.times.at[count].set(time_),
            next_sequence=jnp.asarray(next_count, dtype=jnp.int32),
            count=jnp.asarray(next_count, dtype=jnp.int32),
            input_epoch_ids=tuple(input_epoch_ids),
            requested_ids=tuple(requested_ids),
            payload_ids=tuple(payload_ids),
        )

    def append_requested_batch(
        self,
        requests: Sequence[FiniteVolumeTopologyEventRequest],
        accepted_step: ArrayLike,
        time: ArrayLike,
        /,
    ) -> FiniteVolumeTopologyEventJournal:
        """Append one simultaneous request batch without exposing partial state."""

        if not isinstance(requests, (tuple, list)) or not requests:
            raise ValueError("Topology event request batch must be nonempty.")
        if any(
            not isinstance(request, FiniteVolumeTopologyEventRequest)
            for request in requests
        ):
            raise TypeError("Topology event batch contains an invalid request.")
        accepted_step_ = _host_nonnegative_integer(accepted_step, "accepted_step")
        time_, _ = _host_finite_time(time, "time")
        if any(request.input_epoch_id != self.current_epoch_id for request in requests):
            raise ValueError("Topology event request input epoch is stale.")
        count = int(np.asarray(self.count))
        if any(
            state == int(TopologyEventState.REQUESTED)
            for state in np.asarray(self.states[:count])
        ):
            raise ValueError("Topology event journal already has a pending request.")
        if count:
            if accepted_step_ < int(np.asarray(self.accepted_steps[count - 1])):
                raise ValueError("Topology event accepted steps must be monotone.")
            if time_ < float(np.asarray(self.times[count - 1])):
                raise ValueError("Topology event times must be monotone.")
        if count + len(requests) > self.capacity:
            raise OverflowError("Topology event journal capacity is exhausted.")
        input_epoch_ids = list(self.input_epoch_ids)
        requested_ids = list(self.requested_ids)
        payload_ids = list(self.payload_ids)
        kinds = self.kinds
        states = self.states
        statuses = self.statuses
        accepted_steps = self.accepted_steps
        times = self.times
        for offset, request in enumerate(requests):
            index = count + offset
            kinds = kinds.at[index].set(int(request.kind))
            states = states.at[index].set(int(TopologyEventState.REQUESTED))
            statuses = statuses.at[index].set(int(TopologyEventStatus.PENDING))
            accepted_steps = accepted_steps.at[index].set(accepted_step_)
            times = times.at[index].set(time_)
            input_epoch_ids[index] = request.input_epoch_id
            requested_ids[index] = request.request_id
            payload_ids[index] = request.payload_id
        next_count = count + len(requests)
        return self._new(
            kinds=kinds,
            states=states,
            statuses=statuses,
            accepted_steps=accepted_steps,
            times=times,
            next_sequence=jnp.asarray(next_count, dtype=jnp.int32),
            count=jnp.asarray(next_count, dtype=jnp.int32),
            input_epoch_ids=tuple(input_epoch_ids),
            requested_ids=tuple(requested_ids),
            payload_ids=tuple(payload_ids),
        )

    def _requested_slot(self, sequence: int, /) -> int:
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise TypeError("Topology event sequence must be an integer.")
        count = int(np.asarray(self.count))
        if sequence < 0 or sequence >= count:
            raise IndexError("Topology event sequence is unrequested.")
        state = _enum_member(
            int(np.asarray(self.states[sequence])), TopologyEventState, "state"
        )
        if state is not TopologyEventState.REQUESTED:
            raise ValueError("Topology event is no longer requested.")
        if self.input_epoch_ids[sequence] != self.current_epoch_id:
            raise ValueError("Topology event input epoch is stale.")
        return sequence

    def commit_batch(
        self,
        sequences: Sequence[int],
        result_epoch: FiniteVolumeTopologyEpoch,
        /,
        *,
        result_id: str | None = None,
        payload_ids: Sequence[str | None] | None = None,
    ) -> FiniteVolumeTopologyEventJournal:
        """Commit simultaneous requests to one successor epoch atomically."""

        if not isinstance(sequences, (tuple, list)) or not sequences:
            raise ValueError("Topology event commit batch must be nonempty.")
        indexes = tuple(self._requested_slot(sequence) for sequence in sequences)
        if len(set(indexes)) != len(indexes):
            raise ValueError("Topology event commit sequences must be unique.")
        if not isinstance(result_epoch, FiniteVolumeTopologyEpoch):
            raise TypeError("result_epoch must be FiniteVolumeTopologyEpoch.")
        if result_epoch.parent_epoch_id != self.current_epoch_id:
            raise ValueError("Committed topology epoch has the wrong input epoch.")
        supplied_result = _require_identifier(result_id, "result_id")
        if supplied_result is not None and supplied_result != result_epoch.epoch_id:
            raise ValueError(
                "Committed result identity is not the result epoch identity."
            )
        if any(epoch.epoch_id == result_epoch.epoch_id for epoch in self.epoch_table):
            raise ValueError("Committed topology epoch identity already exists.")
        first = indexes[0]
        for index in indexes[1:]:
            if (
                self.input_epoch_ids[index] != self.input_epoch_ids[first]
                or int(np.asarray(self.accepted_steps[index]))
                != int(np.asarray(self.accepted_steps[first]))
                or float(np.asarray(self.times[index]))
                != float(np.asarray(self.times[first]))
            ):
                raise ValueError("Topology commit requests are not simultaneous.")
        if payload_ids is not None:
            if not isinstance(payload_ids, (tuple, list)) or len(payload_ids) != len(
                indexes
            ):
                raise ValueError("Topology commit payload IDs must match sequences.")
            payload_values = tuple(
                _require_identifier(value, "payload_id") for value in payload_ids
            )
        else:
            payload_values = tuple(self.payload_ids[index] for index in indexes)
        result_table = list(self.result_ids)
        payload_table = list(self.payload_ids)
        states = self.states
        statuses = self.statuses
        for index, payload in zip(indexes, payload_values, strict=True):
            result_table[index] = result_epoch.epoch_id
            payload_table[index] = payload
            states = states.at[index].set(int(TopologyEventState.COMMITTED))
            statuses = statuses.at[index].set(int(TopologyEventStatus.SUCCESS))
        return self._new(
            states=states,
            statuses=statuses,
            current_epoch_id=result_epoch.epoch_id,
            epoch_table=(*self.epoch_table, result_epoch),
            result_ids=tuple(result_table),
            payload_ids=tuple(payload_table),
        )

    def commit(
        self,
        sequence: int,
        result_epoch: FiniteVolumeTopologyEpoch,
        /,
        *,
        result_id: str | None = None,
        payload_id: str | None = None,
    ) -> FiniteVolumeTopologyEventJournal:
        return self.commit_batch(
            (sequence,),
            result_epoch,
            result_id=result_id,
            payload_ids=(payload_id,) if payload_id is not None else None,
        )

    def fail_batch(
        self,
        sequences: Sequence[int],
        /,
        *,
        status: TopologyEventStatus = TopologyEventStatus.FAILED,
        result_ids: Sequence[str | None] | None = None,
        payload_ids: Sequence[str | None] | None = None,
    ) -> FiniteVolumeTopologyEventJournal:
        """Fail simultaneous requests atomically while retaining their records."""

        if not isinstance(sequences, (tuple, list)) or not sequences:
            raise ValueError("Topology event failure batch must be nonempty.")
        failure_status = _enum_member(status, TopologyEventStatus, "status")
        if not _is_failed_status(failure_status):
            raise ValueError("Topology failure status must identify a failure reason.")
        indexes = tuple(self._requested_slot(sequence) for sequence in sequences)
        if len(set(indexes)) != len(indexes):
            raise ValueError("Topology event failure sequences must be unique.")
        if result_ids is not None:
            if not isinstance(result_ids, (tuple, list)) or len(result_ids) != len(
                indexes
            ):
                raise ValueError("Topology failure result IDs must match sequences.")
            result_values = tuple(
                _require_identifier(value, "result_id") for value in result_ids
            )
        else:
            result_values = (None,) * len(indexes)
        if payload_ids is not None:
            if not isinstance(payload_ids, (tuple, list)) or len(payload_ids) != len(
                indexes
            ):
                raise ValueError("Topology failure payload IDs must match sequences.")
            payload_values = tuple(
                _require_identifier(value, "payload_id") for value in payload_ids
            )
        else:
            payload_values = tuple(self.payload_ids[index] for index in indexes)
        states = self.states
        statuses = self.statuses
        result_table = list(self.result_ids)
        payload_table = list(self.payload_ids)
        for index, result, payload in zip(
            indexes, result_values, payload_values, strict=True
        ):
            states = states.at[index].set(int(TopologyEventState.FAILED))
            statuses = statuses.at[index].set(int(failure_status))
            result_table[index] = result
            payload_table[index] = payload
        return self._new(
            states=states,
            statuses=statuses,
            result_ids=tuple(result_table),
            payload_ids=tuple(payload_table),
        )

    def fail(
        self,
        sequence: int,
        /,
        *,
        status: TopologyEventStatus = TopologyEventStatus.FAILED,
        result_id: str | None = None,
        payload_id: str | None = None,
    ) -> FiniteVolumeTopologyEventJournal:
        return self.fail_batch(
            (sequence,),
            status=status,
            result_ids=(result_id,),
            payload_ids=(payload_id,) if payload_id is not None else None,
        )

    def event(self, sequence: int, /) -> FiniteVolumeTopologyEvent:
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise TypeError("Topology event sequence must be an integer.")
        count = int(np.asarray(self.count))
        if sequence < 0 or sequence >= count:
            raise IndexError("Topology event sequence is unrequested.")
        requested = _required_identifier(self.requested_ids[sequence], "requested_id")
        input_epoch = _required_identifier(
            self.input_epoch_ids[sequence], "input_epoch_id"
        )
        return FiniteVolumeTopologyEvent(
            sequence,
            int(np.asarray(self.accepted_steps[sequence])),
            float(np.asarray(self.times[sequence])),
            _enum_member(
                int(np.asarray(self.kinds[sequence])), TopologyEventKind, "kind"
            ),
            _enum_member(
                int(np.asarray(self.states[sequence])), TopologyEventState, "state"
            ),
            _enum_member(
                int(np.asarray(self.statuses[sequence])),
                TopologyEventStatus,
                "status",
            ),
            requested,
            input_epoch,
            self.result_ids[sequence],
            self.payload_ids[sequence],
        )


@dataclass(frozen=True)
class FiniteVolumeTopologyEventTransactionResult:
    """Host-side result of one atomic topology-event transaction."""

    journal: FiniteVolumeTopologyEventJournal
    content_state: Any
    result_epoch: FiniteVolumeTopologyEpoch | None
    events: tuple[FiniteVolumeTopologyEvent, ...]
    statuses: tuple[TopologyEventStatus, ...]
    committed: bool
    failure: TopologyEventStatus | None = None

    @property
    def state(self) -> Any:
        return self.content_state

    @property
    def event_status(self) -> TopologyEventStatus:
        return self.statuses[0]

    @property
    def failed(self) -> bool:
        return not self.committed


@dataclass(frozen=True)
class _ScheduledTopologyRequest:
    request: FiniteVolumeTopologyEventRequest
    accepted_step: int | None
    time: float | None


_MISSING = object()


def _host_field(value: Any, name: str, default: Any = _MISSING, /) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    if value is None:
        return default
    return getattr(value, name, default)


def _host_success(value: Any, /) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    passed = _host_field(value, "passed")
    if passed is not _MISSING:
        return bool(np.asarray(passed))
    status = _host_field(value, "status")
    if status is not _MISSING:
        if isinstance(status, str):
            return status.upper() in {"SUCCESS", "PASSED", "OK"}
        if isinstance(status, IntEnum):
            return (
                "FAIL" not in status.name.upper() and "ERROR" not in status.name.upper()
            )
        status_array = np.asarray(status)
        if status_array.shape == () and status_array.dtype.kind in "biuf":
            return bool(float(status_array) == 0.0)
        return False
    return True


def _host_report_value(report: Any, name: str, /) -> np.ndarray | None:
    value = _host_field(report, name)
    if value is _MISSING or value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return np.asarray(0.0)
    return array


def _coverage_passed(remap: Any, tolerance: float, /) -> bool:
    report = _host_field(remap, "report")
    if report is _MISSING:
        report = remap
    coverage_error = _host_field(report, "coverage_error")
    coverage_passed = _host_field(report, "coverage_passed")
    if coverage_error is not _MISSING:
        value = np.asarray(coverage_error)
        return bool(
            np.all(np.isfinite(value))
            and np.all(np.abs(value) <= tolerance)
            and bool(np.asarray(coverage_passed))
            if coverage_passed is not _MISSING
            else np.all(np.isfinite(value)) and np.all(np.abs(value) <= tolerance)
        )
    required = (
        "maximum_target_coverage_defect",
        "maximum_source_coverage_defect",
        "uncovered_target_measure",
        "uncovered_source_measure",
        "donor_excess_measure",
    )
    values = tuple(_host_report_value(report, name) for name in required)
    checked_values: list[np.ndarray] = []
    for value in values:
        if value is None:
            return False
        checked_values.append(np.asarray(value, dtype=np.float64))
    return all(
        np.all(np.isfinite(value)) and np.all(np.abs(value) <= tolerance)
        for value in checked_values
    )


def _required_artifact_success(value: Any, name: str, /) -> bool:
    if isinstance(value, TopologyEventStatus):
        return value is TopologyEventStatus.SUCCESS
    passed = _host_field(value, "passed")
    status = _host_field(value, "status")
    if passed is not _MISSING:
        return bool(np.asarray(passed))
    if status is not _MISSING:
        return _host_success(status)
    return False


def _failure_reason(value: Any, default: TopologyEventStatus, /) -> TopologyEventStatus:
    if isinstance(value, TopologyEventStatus) and _is_failed_status(value):
        return value
    return default


def _conservation_passed(
    remap: Any,
    source_content: Any,
    target_content: Any,
    tolerance: float,
    /,
) -> bool:
    report = _host_field(remap, "report")
    if report is _MISSING:
        report = remap
    if source_content is None and target_content is None:
        return _coverage_passed(remap, tolerance)
    method = _host_field(remap, "conservation_defect")
    defect = _host_field(report, "conservation_defect")
    if method is _MISSING:
        method = defect
    if callable(method) and source_content is not None and target_content is not None:
        source_average = _host_field(source_content, "cell_average")
        target_average = _host_field(target_content, "cell_average")
        if callable(source_average):
            source_average = source_average()
        if callable(target_average):
            target_average = target_average()
        if source_average is _MISSING and isinstance(
            source_content, (np.ndarray, list, tuple)
        ):
            source_average = source_content
        if target_average is _MISSING and isinstance(
            target_content, (np.ndarray, list, tuple)
        ):
            target_average = target_content
        if source_average is not _MISSING and target_average is not _MISSING:
            defect = method(source_average, target_average)
    if defect is _MISSING:
        return False
    value: np.ndarray = np.asarray(defect, dtype=np.float64)
    return bool(np.all(np.isfinite(value)) and np.all(np.abs(value) <= tolerance))


def _active_content_valid(content: Any, expected_mask: Any = _MISSING, /) -> bool:
    if content is None or content is _MISSING:
        return True
    values = _host_field(content, "conservative_content")
    volumes = _host_field(content, "effective_cell_volumes")
    active = _host_field(content, "active_cell_mask")
    if values is _MISSING and isinstance(content, (np.ndarray, list, tuple)):
        values = content
    if values is not _MISSING:
        values_array = np.asarray(values)
        if np.any(~np.isfinite(values_array)):
            return False
    if volumes is not _MISSING:
        volume_array = np.asarray(volumes)
        if np.any(~np.isfinite(volume_array)):
            return False
        if active is not _MISSING:
            active_array = np.asarray(active, dtype=bool)
            if active_array.shape != volume_array.shape:
                return False
            if np.any(active_array & (volume_array <= 0.0)):
                return False
            if np.any((~active_array) & (volume_array != 0.0)):
                return False
            if values is not _MISSING:
                content_array = np.asarray(values)
                mask_shape = active_array.shape + (1,) * (
                    content_array.ndim - active_array.ndim
                )
                if content_array.ndim >= active_array.ndim and np.any(
                    (~active_array.reshape(mask_shape)) & (content_array != 0.0)
                ):
                    return False
    if expected_mask is not _MISSING and active is not _MISSING:
        if not np.array_equal(
            np.asarray(active, dtype=bool), np.asarray(expected_mask, dtype=bool)
        ):
            return False
    return True


def _call_transfer(callback: Callable[..., Any], source: Any, remap: Any, /) -> Any:
    parameters = tuple(inspect.signature(callback).parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    variadic = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    if variadic or len(positional) >= 2:
        return callback(source, remap)
    return callback(source)


class FiniteVolumeTopologyEventTransaction:
    """Host-only atomic preparation and commit of coalesced topology requests."""

    def __init__(
        self,
        journal: FiniteVolumeTopologyEventJournal,
        requests: Sequence[FiniteVolumeTopologyEventRequest],
        accepted_step: ArrayLike,
        time: ArrayLike,
        /,
        *,
        accepted: bool = True,
        prepare: Callable[..., Any] | None = None,
        transfer: Callable[..., Any] | None = None,
        coverage_tolerance: float = 0.0,
        maximum_requests: int | None = None,
        active_cell_mask: Any = _MISSING,
        admissibility: Callable[[Any], Any] | bool | None = None,
        artifact: Any = None,
        candidate_epoch: FiniteVolumeTopologyEpoch | None = None,
        remap: Any = None,
        metrics: Any = None,
        evidence: Any = None,
        status: Any = None,
        source_geometry: Any = None,
        target_geometry: Any = None,
        remap_tolerance: float = 1e-10,
        remap_limits: Any = None,
        remap_provenance: str = "topology-event",
    ):
        if not isinstance(journal, FiniteVolumeTopologyEventJournal):
            raise TypeError("journal must be FiniteVolumeTopologyEventJournal.")
        if not isinstance(requests, (tuple, list)) or not requests:
            raise ValueError("Topology transaction requests must be nonempty.")
        if any(
            not isinstance(request, FiniteVolumeTopologyEventRequest)
            for request in requests
        ):
            raise TypeError("Topology transaction requests are invalid.")
        if not isinstance(accepted, (bool, np.bool_)):
            raise TypeError("accepted must be boolean.")
        tolerance = float(coverage_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("coverage_tolerance must be finite and nonnegative.")
        self.journal = journal
        self.requests = tuple(requests)
        self.accepted_step = _host_nonnegative_integer(accepted_step, "accepted_step")
        self.time, _ = _host_finite_time(time, "time")
        self.accepted = bool(accepted)
        self.prepare = prepare
        self.transfer = transfer
        self.coverage_tolerance = tolerance
        self.maximum_requests = maximum_requests
        self.active_cell_mask = active_cell_mask
        self.admissibility = admissibility
        self._artifact = artifact
        self._candidate_epoch = candidate_epoch
        self._remap = remap
        self._metrics = metrics
        self._evidence = evidence
        self._status = status
        self.source_geometry = source_geometry
        self.target_geometry = target_geometry
        self.remap_tolerance = float(remap_tolerance)
        self.remap_limits = remap_limits
        self.remap_provenance = str(remap_provenance)

    def _outcome(
        self,
        content_state: Any,
        status: TopologyEventStatus,
        /,
        *,
        journal: FiniteVolumeTopologyEventJournal | None = None,
        result_epoch: FiniteVolumeTopologyEpoch | None = None,
        committed: bool = False,
    ) -> FiniteVolumeTopologyEventTransactionResult:
        current_journal = self.journal if journal is None else journal
        count = int(np.asarray(current_journal.count))
        events = tuple(
            current_journal.event(index)
            for index in range(count)
            if current_journal.requested_ids[index]
            in {request.request_id for request in self.requests}
        )
        return FiniteVolumeTopologyEventTransactionResult(
            current_journal,
            content_state,
            result_epoch,
            events,
            tuple(status for _ in self.requests),
            committed,
            None if committed else status,
        )

    def _failure(
        self,
        content_state: Any,
        status: TopologyEventStatus,
        /,
    ) -> FiniteVolumeTopologyEventTransactionResult:
        if status is TopologyEventStatus.FAILED_STALE_EPOCH:
            return self._outcome(content_state, status)
        try:
            requested = self.journal.append_requested_batch(
                self.requests, self.accepted_step, self.time
            )
        except (OverflowError, ValueError):
            return self._outcome(content_state, TopologyEventStatus.FAILED_RESOURCE_LIMIT)
        sequences = tuple(
            int(np.asarray(requested.count)) - len(self.requests) + index
            for index in range(len(self.requests))
        )
        failed = requested.fail_batch(sequences, status=status)
        return self._outcome(content_state, status, journal=failed)

    def execute(
        self,
        source_content: Any = None,
        /,
        *,
        artifact: Any = None,
        candidate_epoch: FiniteVolumeTopologyEpoch | None = None,
        remap: Any = None,
        metrics: Any = None,
        evidence: Any = None,
        status: Any = None,
        transfer: Callable[..., Any] | None = None,
        active_cell_mask: Any = _MISSING,
        admissibility: Callable[[Any], Any] | bool | None = None,
        coverage_ok: bool | None = None,
        positivity_ok: bool | None = None,
        resource_ok: bool | None = None,
        result_id: str | None = None,
        payload_ids: Sequence[str | None] | None = None,
    ) -> FiniteVolumeTopologyEventTransactionResult:
        """Prepare, validate, transfer once, and atomically update the journal."""
        if admissibility is None:
            admissibility = self.admissibility
        if artifact is None:
            artifact = self._artifact
        if candidate_epoch is None:
            candidate_epoch = self._candidate_epoch
        if remap is None:
            remap = self._remap
        if metrics is None:
            metrics = self._metrics
        if evidence is None:
            evidence = self._evidence
        if status is None:
            status = self._status

        if not self.accepted:
            raise ValueError("Topology events may be processed only at an accepted step.")
        if any(
            request.input_epoch_id != self.journal.current_epoch_id
            for request in self.requests
        ):
            return self._failure(source_content, TopologyEventStatus.FAILED_STALE_EPOCH)
        if self.maximum_requests is not None and (
            len(self.requests) > self.maximum_requests
        ):
            return self._failure(
                source_content, TopologyEventStatus.FAILED_RESOURCE_LIMIT
            )
        if self.journal.capacity - int(np.asarray(self.journal.count)) < len(
            self.requests
        ):
            return self._failure(
                source_content, TopologyEventStatus.FAILED_RESOURCE_LIMIT
            )

        prepared = artifact
        if prepared is None and self.prepare is not None:
            prepared = _call_transfer(
                self.prepare, self.requests, self.journal.current_epoch_id
            )
        if prepared is not None:
            if candidate_epoch is None:
                candidate_epoch = _host_field(prepared, "epoch")
                if candidate_epoch is _MISSING:
                    candidate_epoch = _host_field(prepared, "candidate_epoch")
                if candidate_epoch is _MISSING:
                    candidate_epoch = _host_field(prepared, "result_epoch")
                if candidate_epoch is _MISSING:
                    candidate_epoch = None
            remap = remap if remap is not None else _host_field(prepared, "remap")
            if result_id is None:
                result_id = _host_field(prepared, "result_id")
                if result_id is _MISSING:
                    result_id = None
            if payload_ids is None:
                payload_ids = _host_field(prepared, "payload_ids")
                if payload_ids is _MISSING:
                    payload_ids = None
            status = status if status is not None else _host_field(prepared, "status")
            if active_cell_mask is _MISSING:
                active_cell_mask = _host_field(prepared, "active_cell_mask")
            if admissibility is None:
                admissibility = _host_field(prepared, "admissibility")
            if transfer is None:
                candidate_transfer = _host_field(prepared, "transfer")
                transfer = None if candidate_transfer is _MISSING else candidate_transfer
            if positivity_ok is None:
                positivity_ok = _host_field(prepared, "positivity_ok")
            if coverage_ok is None:
                coverage_ok = _host_field(prepared, "coverage_ok")
            if source_content is None:
                source_content = _host_field(prepared, "source_content")
        if (
            self.source_geometry is not None or self.target_geometry is not None
        ) and source_content is None:
            return self._failure(
                source_content,
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
            )
        if resource_ok is False:
            return self._failure(
                source_content, TopologyEventStatus.FAILED_RESOURCE_LIMIT
            )
        if (
            remap is None
            and self.source_geometry is not None
            and self.target_geometry is not None
        ):
            from ..discretization.finite_volume._automatic_remap import (
                build_unstructured_conservative_remap,
            )

            build = build_unstructured_conservative_remap(
                self.source_geometry,
                self.target_geometry,
                tolerance=self.remap_tolerance,
                limits=self.remap_limits,
                provenance=self.remap_provenance,
            )
            if not build.passed or build.plan is None:
                return self._failure(
                    source_content,
                    TopologyEventStatus.FAILED_COVERAGE,
                )
            remap = build.plan
            metrics = build.evidence
            evidence = build.evidence
            status = TopologyEventStatus.SUCCESS
        if (
            remap is None
            or metrics is None
            or evidence is None
            or status is None
            or candidate_epoch is None
        ):
            return self._failure(
                source_content, TopologyEventStatus.FAILED_MISSING_ARTIFACT
            )
        if not _required_artifact_success(status, "status"):
            return self._failure(
                source_content,
                _failure_reason(status, TopologyEventStatus.FAILED_MISSING_ARTIFACT),
            )
        if coverage_ok is False or not _coverage_passed(remap, self.coverage_tolerance):
            return self._failure(source_content, TopologyEventStatus.FAILED_COVERAGE)
        if not _required_artifact_success(
            metrics, "metrics"
        ) or not _required_artifact_success(evidence, "evidence"):
            return self._failure(
                source_content, TopologyEventStatus.FAILED_MISSING_ARTIFACT
            )
        if not isinstance(candidate_epoch, FiniteVolumeTopologyEpoch):
            return self._failure(
                source_content, TopologyEventStatus.FAILED_MISSING_ARTIFACT
            )
        if candidate_epoch.parent_epoch_id != self.journal.current_epoch_id:
            return self._failure(source_content, TopologyEventStatus.FAILED_STALE_EPOCH)
        if self.source_geometry is not None:
            current_epoch = self.journal.epoch_table[-1]
            if (
                current_epoch.prepared_id != self.source_geometry.prepared_id
                or current_epoch.topology_id != self.source_geometry.topology_id
                or current_epoch.geometry_id != self.source_geometry.geometry_id
            ):
                return self._failure(
                    source_content, TopologyEventStatus.FAILED_STALE_EPOCH
                )
        if self.target_geometry is not None and (
            candidate_epoch.prepared_id != self.target_geometry.prepared_id
            or candidate_epoch.topology_id != self.target_geometry.topology_id
            or candidate_epoch.geometry_id != self.target_geometry.geometry_id
        ):
            return self._failure(source_content, TopologyEventStatus.FAILED_STALE_EPOCH)

        candidate_content = _host_field(prepared, "content_state")
        if candidate_content is _MISSING:
            candidate_content = None
        transfer_callback = self.transfer if transfer is None else transfer
        if transfer_callback is not None:
            candidate_content = _call_transfer(transfer_callback, source_content, remap)
        elif source_content is not None:
            apply = _host_field(remap, "apply")
            if callable(apply):
                source_average = _host_field(source_content, "cell_average")
                if callable(source_average):
                    source_average = source_average()
                if source_average is not _MISSING:
                    candidate_content = apply(source_average)
        source_epoch = _host_field(source_content, "topology_epoch_id")
        if source_epoch is not _MISSING:
            required_content_fields = (
                "topology_epoch_id",
                "effective_cell_volumes",
                "active_cell_mask",
                "conservative_content",
            )
            if candidate_content is None or any(
                _host_field(candidate_content, field) is _MISSING
                for field in required_content_fields
            ):
                return self._failure(
                    source_content, TopologyEventStatus.FAILED_MISSING_ARTIFACT
                )
            candidate_content_epoch = _host_field(candidate_content, "topology_epoch_id")
            if candidate_content_epoch != candidate_epoch.epoch_id:
                return self._failure(
                    source_content, TopologyEventStatus.FAILED_STALE_EPOCH
                )
            if self.target_geometry is not None:
                target_volumes = np.asarray(self.target_geometry.cell_volumes)
                candidate_volumes = np.asarray(
                    _host_field(candidate_content, "effective_cell_volumes")
                )
                if candidate_volumes.shape != target_volumes.shape or not np.allclose(
                    candidate_volumes,
                    target_volumes,
                    rtol=0.0,
                    atol=0.0,
                ):
                    return self._failure(
                        source_content, TopologyEventStatus.FAILED_STALE_EPOCH
                    )
        if not _conservation_passed(
            remap,
            source_content,
            candidate_content,
            self.coverage_tolerance,
        ):
            return self._failure(source_content, TopologyEventStatus.FAILED_COVERAGE)
        if positivity_ok is False or not _active_content_valid(
            candidate_content,
            self.active_cell_mask if active_cell_mask is _MISSING else active_cell_mask,
        ):
            return self._failure(source_content, TopologyEventStatus.FAILED_POSITIVITY)
        if admissibility is not None:
            admissible = (
                admissibility(candidate_content)
                if callable(admissibility)
                else bool(admissibility)
            )
            if not bool(np.asarray(admissible)):
                return self._failure(
                    source_content, TopologyEventStatus.FAILED_POSITIVITY
                )
        try:
            requested = self.journal.append_requested_batch(
                self.requests, self.accepted_step, self.time
            )
            start = int(np.asarray(requested.count)) - len(self.requests)
            sequences = tuple(start + index for index in range(len(self.requests)))
            committed = requested.commit_batch(
                sequences,
                candidate_epoch,
                result_id=result_id,
                payload_ids=payload_ids,
            )
        except (OverflowError, ValueError, TypeError):
            return self._failure(
                source_content, TopologyEventStatus.FAILED_RESOURCE_LIMIT
            )
        return self._outcome(
            candidate_content if candidate_content is not None else source_content,
            TopologyEventStatus.SUCCESS,
            journal=committed,
            result_epoch=candidate_epoch,
            committed=True,
        )

    run = execute
    commit = execute


class FiniteVolumeTopologyEventScheduler:
    """Host scheduler that coalesces simultaneous requests before preparation."""

    def __init__(
        self,
        journal: FiniteVolumeTopologyEventJournal,
        /,
        *,
        maximum_requests: int | None = None,
    ):
        if not isinstance(journal, FiniteVolumeTopologyEventJournal):
            raise TypeError("journal must be FiniteVolumeTopologyEventJournal.")
        if maximum_requests is not None and (
            isinstance(maximum_requests, bool)
            or not isinstance(maximum_requests, int)
            or maximum_requests <= 0
        ):
            raise ValueError("maximum_requests must be a positive integer.")
        self.journal = journal
        self.maximum_requests = maximum_requests
        self._pending: list[_ScheduledTopologyRequest] = []

    @property
    def pending_requests(self) -> tuple[FiniteVolumeTopologyEventRequest, ...]:
        return tuple(item.request for item in self._pending)

    def submit(
        self,
        request: FiniteVolumeTopologyEventRequest,
        accepted_step: ArrayLike | None = None,
        time: ArrayLike | None = None,
        /,
        *,
        accepted: bool = True,
    ) -> FiniteVolumeTopologyEventRequest:
        if not isinstance(request, FiniteVolumeTopologyEventRequest):
            raise TypeError("request must be FiniteVolumeTopologyEventRequest.")
        if not accepted:
            raise ValueError("Topology requests are accepted-step events.")
        if request.input_epoch_id != self.journal.current_epoch_id:
            raise ValueError("Topology event request input epoch is stale.")
        step = (
            None
            if accepted_step is None
            else _host_nonnegative_integer(accepted_step, "accepted_step")
        )
        time_value = None if time is None else _host_finite_time(time, "time")[0]
        if (
            self.maximum_requests is not None
            and len(self._pending) >= self.maximum_requests
        ):
            raise OverflowError("Topology event scheduler resource limit reached.")
        self._pending.append(_ScheduledTopologyRequest(request, step, time_value))
        return request

    enqueue = submit
    request = submit

    def coalesce(self) -> tuple[FiniteVolumeTopologyEventRequest, ...]:
        if not self._pending:
            return ()
        first = self._pending[0]
        group: list[FiniteVolumeTopologyEventRequest] = []
        for item in self._pending:
            if (
                (
                    first.accepted_step is not None
                    and item.accepted_step is not None
                    and item.accepted_step != first.accepted_step
                )
                or (
                    first.time is not None
                    and item.time is not None
                    and item.time != first.time
                )
                or item.request.input_epoch_id != first.request.input_epoch_id
            ):
                break
            group.append(item.request)
        return tuple(group)

    def transact(
        self,
        /,
        accepted: bool = True,
        accepted_step: ArrayLike | None = None,
        time: ArrayLike | None = None,
        source_content: Any = None,
        artifact: Any = None,
        prepare: Callable[..., Any] | None = None,
        transfer: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> FiniteVolumeTopologyEventTransactionResult:
        if not accepted:
            raise ValueError("Topology transactions require an accepted step.")
        requests = self.coalesce()
        if not requests:
            raise ValueError("Topology scheduler has no pending requests.")
        first = self._pending[0]
        step = (
            first.accepted_step
            if first.accepted_step is not None
            else (
                None
                if accepted_step is None
                else _host_nonnegative_integer(accepted_step, "accepted_step")
            )
        )
        time_value = (
            first.time
            if first.time is not None
            else (None if time is None else _host_finite_time(time, "time")[0])
        )
        if step is None or time_value is None:
            raise ValueError("Accepted-step transactions require step and time.")
        constructor_kwargs = dict(kwargs)
        source_geometry = constructor_kwargs.pop("source_geometry", None)
        target_geometry = constructor_kwargs.pop("target_geometry", None)
        remap_tolerance = constructor_kwargs.pop("remap_tolerance", 1e-10)
        remap_limits = constructor_kwargs.pop("remap_limits", None)
        remap_provenance = constructor_kwargs.pop("remap_provenance", "topology-event")
        coverage_tolerance = constructor_kwargs.pop("coverage_tolerance", 0.0)
        active_cell_mask = constructor_kwargs.pop("active_cell_mask", _MISSING)
        admissibility = constructor_kwargs.pop("admissibility", None)
        maximum_requests = constructor_kwargs.pop(
            "maximum_requests", self.maximum_requests
        )
        transaction = FiniteVolumeTopologyEventTransaction(
            self.journal,
            requests,
            step,
            time_value,
            accepted=True,
            prepare=prepare,
            transfer=transfer,
            coverage_tolerance=coverage_tolerance,
            maximum_requests=maximum_requests,
            active_cell_mask=active_cell_mask,
            admissibility=admissibility,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            remap_tolerance=remap_tolerance,
            remap_limits=remap_limits,
            remap_provenance=remap_provenance,
        )
        result = transaction.execute(
            source_content,
            artifact=artifact,
            transfer=transfer,
            **constructor_kwargs,
        )
        self._pending = self._pending[len(requests) :]
        if result.journal is not self.journal:
            self.journal = result.journal
        return result

    process = transact
    run = transact

    commit = transact
    schedule = submit
    add_request = submit


__all__ = [
    "FiniteVolumeTopologyEpoch",
    "FiniteVolumeTopologyEvent",
    "FiniteVolumeTopologyEventJournal",
    "FiniteVolumeTopologyEventRequest",
    "FiniteVolumeTopologyEventScheduler",
    "FiniteVolumeTopologyEventTransaction",
    "FiniteVolumeTopologyEventTransactionResult",
    "TopologyEventKind",
    "TopologyEventState",
    "TopologyEventStatus",
]
