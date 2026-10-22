#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Execution-only meshing measurements, deliberately outside scientific identity."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from time import perf_counter
from typing import cast, final, Literal, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    meshcore_runtime_identity,
    MeshcoreStatus,
    NativeExecutionEvidence,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    NativeExecutionWorkCounters,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer
from ..typing import checked, Identifier, parse


type NativeMeshingPhase = Literal[
    "source_preparation",
    "native_validation",
    "native_preparation",
    "boundary_recovery",
    "region_classification",
    "native_publication",
    "refinement",
    "improvement",
    "exudation",
    "construction",
    "topology_construction",
    "frame_solve",
    "source_frame_prepare",
    "frame_graph_decomposition",
    "block_count_compatibility",
    "source_block_integer_extraction",
    "geometry_projection",
    "geometry_evaluation",
    "curving",
    "geometry_association",
    "organization",
    "audit",
    "certification",
    "publication",
    "compliance",
    "image_interpretation",
    "level_set_construction",
    "periodic_construction",
    "periodic_embedding",
    "envelope_construction",
    "power_adjacency",
    "power_clipping",
    "power_components",
    "power_publication",
    "regular_triangulation",
    "site_refinement",
    "feature_certification",
    "topology_adaptation",
    "geometry_transition",
    "common_refinement",
    "metric_preparation",
    "metric_split",
    "metric_collapse",
    "metric_reconnection",
    "metric_relocation",
    "metric_global_preparation",
    "metric_global_solve",
    "metric_global_admission",
    "lineage_construction",
    "donor_location",
    "interpolation_preparation",
]


@final
class NativeExecutionDeviceWorkCounters(StrictModule, NonTrainableState):
    """Dynamic scalar uint64 execution work, with the host owner's vocabulary."""

    construction_work: Array
    source_queries: Array
    peak_cavity_cells: Array
    work_refusals: Array
    query_refusals: Array
    cavity_refusals: Array

    @classmethod
    def from_host(
        cls, counters: NativeExecutionWorkCounters, /
    ) -> NativeExecutionDeviceWorkCounters:
        values = jax.device_put(
            tuple(np.asarray(value, dtype=np.uint64) for _, value in counters.items())
        )
        return cls(*values)

    def items(self) -> tuple[tuple[str, Array], ...]:
        return (
            ("construction_work", self.construction_work),
            ("source_queries", self.source_queries),
            ("peak_cavity_cells", self.peak_cavity_cells),
            ("work_refusals", self.work_refusals),
            ("query_refusals", self.query_refusals),
            ("cavity_refusals", self.cavity_refusals),
        )

    def __post_init__(self) -> None:
        for name, value in self.items():
            if value.shape != () or value.dtype != jnp.uint64:
                raise ValueError(
                    f"NativeExecutionDeviceWorkCounters.{name} must be a scalar uint64 leaf; "
                    f"got {value.dtype} {value.shape}."
                )


@final
class NativeExecutionDeviceMemoryCounters(StrictModule, NonTrainableState):
    """Dynamic scalar uint64 allocations, separate from host storage upper bounds."""

    limit_bytes: Array
    live_bytes: Array
    peak_bytes: Array
    requested_bytes: Array
    allocations: Array
    refusals: Array

    @classmethod
    def from_host(
        cls, counters: NativeExecutionMemoryCounters, /
    ) -> NativeExecutionDeviceMemoryCounters:
        values = jax.device_put(
            tuple(np.asarray(value, dtype=np.uint64) for _, value in counters.items())
        )
        return cls(*values)

    def items(self) -> tuple[tuple[str, Array], ...]:
        return (
            ("limit_bytes", self.limit_bytes),
            ("live_bytes", self.live_bytes),
            ("peak_bytes", self.peak_bytes),
            ("requested_bytes", self.requested_bytes),
            ("allocations", self.allocations),
            ("refusals", self.refusals),
        )

    def __post_init__(self) -> None:
        for name, value in self.items():
            if value.shape != () or value.dtype != jnp.uint64:
                raise ValueError(
                    f"NativeExecutionDeviceMemoryCounters.{name} must be a scalar uint64 leaf; "
                    f"got {value.dtype} {value.shape}."
                )


class NativeExecutionWorkRecord(TypedDict):
    """Named JSON execution work measurements."""

    construction_work: int
    source_queries: int
    peak_cavity_cells: int
    work_refusals: int
    query_refusals: int
    cavity_refusals: int


class NativeExecutionMemoryRecord(TypedDict):
    """Named JSON native allocator measurements."""

    limit_bytes: int
    live_bytes: int
    peak_bytes: int
    requested_bytes: int
    allocations: int
    refusals: int


class NativeExecutionPhaseRecord(TypedDict):
    """JSON host values of one ended phase; edges are local graph indexes.

    ``preparation_evidence`` is this request's charged predecessor chain,
    ``consumer_evidence`` a nested consumer and ``history_evidence`` completed
    earlier requests retained as provenance, never as charges.
    """

    owner_id: str | None
    receipt_id: str
    imported_preparation_ids: list[str]
    work_counters: NativeExecutionWorkRecord
    memory_counters: NativeExecutionMemoryRecord
    status: int
    status_name: str
    elapsed_seconds: float
    prior_elapsed_seconds: float
    externally_charged_work: int
    externally_charged_geometry_queries: int
    native_primitive_queries: int
    host_storage_live_bytes_upper: int
    host_storage_peak_bytes_upper: int
    managed_admission_peak_bytes_upper: int | None
    source_preparation_work_units: int
    source_preparation_geometry_queries: int
    preparation_seconds: float
    preparation_evidence: int | None
    consumer_evidence: int | None
    history_evidence: list[int]
    total_work_units: int
    total_geometry_queries: int
    total_elapsed_seconds: float


class NativeExecutionInterchange(TypedDict):
    """Phase graph preserving numerical evidence and local edge indexes."""

    root: int
    phases: list[NativeExecutionPhaseRecord]


def _phase_record(
    evidence: NativeExecutionEvidence,
    /,
    *,
    owner_id: str | None,
    source_work: int,
    source_queries: int,
    preparation_seconds: float,
    preparation_evidence: int | None,
    consumer_evidence: int | None,
    history_evidence: list[int],
    receipt_id: str,
    imported_preparation_ids: tuple[str, ...],
) -> NativeExecutionPhaseRecord:
    """Lower phase evidence to the host interchange representation."""
    return {
        "owner_id": owner_id,
        "receipt_id": receipt_id,
        "imported_preparation_ids": list(imported_preparation_ids),
        "work_counters": {
            "construction_work": evidence.work_counters.construction_work,
            "source_queries": evidence.work_counters.source_queries,
            "peak_cavity_cells": evidence.work_counters.peak_cavity_cells,
            "work_refusals": evidence.work_counters.work_refusals,
            "query_refusals": evidence.work_counters.query_refusals,
            "cavity_refusals": evidence.work_counters.cavity_refusals,
        },
        "memory_counters": {
            "limit_bytes": evidence.memory_counters.limit_bytes,
            "live_bytes": evidence.memory_counters.live_bytes,
            "peak_bytes": evidence.memory_counters.peak_bytes,
            "requested_bytes": evidence.memory_counters.requested_bytes,
            "allocations": evidence.memory_counters.allocations,
            "refusals": evidence.memory_counters.refusals,
        },
        "status": int(evidence.status),
        "status_name": evidence.status.name,
        "elapsed_seconds": evidence.elapsed_seconds,
        "prior_elapsed_seconds": evidence.prior_elapsed_seconds,
        "externally_charged_work": evidence.externally_charged_work,
        "externally_charged_geometry_queries": evidence.externally_charged_geometry_queries,
        "native_primitive_queries": evidence.native_primitive_queries,
        "host_storage_live_bytes_upper": evidence.host_storage_live_bytes_upper,
        "host_storage_peak_bytes_upper": evidence.host_storage_peak_bytes_upper,
        "managed_admission_peak_bytes_upper": evidence.managed_admission_peak_bytes_upper,
        "source_preparation_work_units": source_work,
        "source_preparation_geometry_queries": source_queries,
        "preparation_seconds": preparation_seconds,
        "preparation_evidence": preparation_evidence,
        "consumer_evidence": consumer_evidence,
        "history_evidence": history_evidence,
        "total_work_units": source_work + evidence.work_counters.construction_work,
        "total_geometry_queries": source_queries + evidence.work_counters.source_queries,
        "total_elapsed_seconds": (
            evidence.elapsed_seconds
            + evidence.prior_elapsed_seconds
            + preparation_seconds
        ),
    }


# One operation can import at most 64 predecessor phases. Fixed digest storage
# keeps numerical record kernels in one shape bucket even for zero imports.
NATIVE_EXECUTION_IMPORT_CAPACITY = 64

# Maximum owned numerical leaves, including the optional managed observation;
# retained predecessor arrays are accounted separately.
_NATIVE_EXECUTION_RECORD_ARRAY_BYTES = (
    (
        len(fields(NativeExecutionDeviceWorkCounters))
        + len(fields(NativeExecutionDeviceMemoryCounters))
        + 8
    )
    * np.dtype(np.uint64).itemsize
    + 3 * np.dtype(np.float64).itemsize
    + np.dtype(np.int32).itemsize
    + np.dtype(np.uint32).itemsize
    + (NATIVE_EXECUTION_IMPORT_CAPACITY + 1) * 32 * np.dtype(np.uint8).itemsize
)


@final
class NativeExecutionImports:
    """A live-ledger snapshot bound to the scope that ends its consumer.

    ``preparation`` names the record's chained preparation. Its matched ledger
    overlap is removed from the ended scope evidence, so raw counters retain
    only this phase's own work beside the separately recorded chain.
    """

    __slots__ = ("_scope", "_receipt_ids", "_preparation", "_overlap", "_evidence")

    def __init__(
        self,
        scope: NativeExecutionScope,
        /,
        *,
        preparation: NativeExecutionRecord | None = None,
    ) -> None:
        from ._native_preparation import native_preparation_ledger

        ledger = native_preparation_ledger(scope)
        self._scope = scope
        self._receipt_ids = ledger.imported_receipt_ids(scope)
        self._preparation = preparation
        self._overlap = ledger.overlap(scope, preparation)
        self._evidence: NativeExecutionEvidence | None = None

    @property
    def receipt_ids(self) -> tuple[str, ...]:
        return self._receipt_ids

    @property
    def preparation(self) -> NativeExecutionRecord | None:
        return self._preparation

    def evidence(self) -> NativeExecutionEvidence:
        """Return the ended scope evidence without its matched preparation overlap."""
        if self._evidence is None:
            self._evidence = self._overlap.normalize(self._scope.completed_evidence())
        return self._evidence

    def require_matches(
        self,
        evidence: NativeExecutionEvidence,
        preparation: NativeExecutionRecord | None,
        /,
    ) -> None:
        if self._scope.evidence is None or evidence is not self.evidence():
            raise ValueError(
                "NativeExecutionImports.evidence must be the ended importing scope's evidence; "
                "got an unrelated or unended execution scope."
            )
        if preparation is not self._preparation:
            raise ValueError(
                "NativeExecutionImports.preparation must be the record's chained preparation; "
                "got a different preparation chain."
            )


def _receipt_fingerprint(
    evidence: NativeExecutionEvidence,
    owner_id: str | None,
    source_work: int,
    source_queries: int,
    preparation_seconds: float,
    preparation: NativeExecutionRecord | None,
    consumer: NativeExecutionRecord | None,
    history: tuple[NativeExecutionRecord, ...],
    imported_ids: tuple[str, ...],
) -> str:
    return _receipt_phase_fingerprint(
        _phase_record(
            evidence,
            owner_id=owner_id,
            source_work=source_work,
            source_queries=source_queries,
            preparation_seconds=preparation_seconds,
            preparation_evidence=None,
            consumer_evidence=None,
            history_evidence=[],
            receipt_id="",
            imported_preparation_ids=imported_ids,
        ),
        None if preparation is None else preparation.receipt_hex(),
        None if consumer is None else consumer.receipt_hex(),
        [record.receipt_hex() for record in history],
    )


def _receipt_phase_fingerprint(
    phase: NativeExecutionPhaseRecord,
    preparation: str | None,
    consumer: str | None,
    history: list[str],
) -> str:
    """One digest schema for numerical records and their JSON consumers."""
    return canonical_fingerprint(
        {
            "kind": "native-execution-receipt",
            "phase": {
                **phase,
                "preparation_evidence": None,
                "consumer_evidence": None,
                "history_evidence": [],
                "receipt_id": "",
            },
            "preparation": preparation,
            "consumer": consumer,
            "history": history,
        }
    )


NATIVE_RUNTIME_MEASUREMENT_REQUIREMENTS = (
    "compiler_temporary_bytes",
    "compiler_output_bytes",
    "compiler_generated_code_bytes",
    "device_allocation_peak_bytes",
    "process_resident_peak_bytes",
)


class NativeManagedResourceClaim(TypedDict):
    """Exact declarations within the request scope, never normalized coverage."""

    scope: Literal["native-managed-request"]
    enforced: list[str]
    unenforced: list[str]


class _NativePublicationResourceBindingBody(TypedDict):
    kind: Literal["native-publication-resource-binding"]
    result_id: str
    request_id: str | None
    runtime_id: str
    native_build: dict[str, str]
    root_receipt_id: str
    native_managed_claim: NativeManagedResourceClaim
    outside_admission_measurement_requirements: list[str]


class NativePublicationResourceBinding(_NativePublicationResourceBindingBody):
    """Publication-owned association, not a signature or an admission verdict."""

    binding_id: str


def native_publication_resource_binding(
    result_id: str,
    runtime_id: str,
    receipt: NativeExecutionRecord | None,
    *,
    enforced_limits: tuple[str, ...],
    unenforced_limits: tuple[str, ...],
) -> NativePublicationResourceBinding | None:
    """Bind actual canonical publication/runtime owners to their ended root."""
    if receipt is None:
        return None
    receipt.require_valid()
    body: _NativePublicationResourceBindingBody = {
        "kind": "native-publication-resource-binding",
        "result_id": result_id,
        # This is the executing receipt's request/plan owner, never a guessed
        # route-specific provenance or trace key.
        "request_id": None if receipt.owner_id is None else str(receipt.owner_id),
        "runtime_id": runtime_id,
        "native_build": meshcore_runtime_identity(),
        "root_receipt_id": receipt.receipt_hex(),
        "native_managed_claim": {
            "scope": "native-managed-request",
            "enforced": list(enforced_limits),
            "unenforced": list(unenforced_limits),
        },
        # Obligations only: observations belong to the bound workflow/epoch.
        # These are never admitted by the native counters or filled with zero.
        "outside_admission_measurement_requirements": list(
            NATIVE_RUNTIME_MEASUREMENT_REQUIREMENTS
        ),
    }
    return {**body, "binding_id": canonical_fingerprint(body)}


class NativeExecutionInterchangeAssessment(TypedDict):
    """Integrity and observation completeness, without claiming external admission."""

    root_receipt_id: str
    owner_id: str | None
    current_phase_indexes: list[int]
    preparation_complete: bool
    managed_admission_peak_bytes_upper: int | None


def validate_native_execution_interchange(
    value: object, /
) -> NativeExecutionInterchangeAssessment:
    """Validate a serialized ended graph using the numerical owner's digest schema.

    This checks integrity, not issuance. The qualification consumer must also bind
    the root to the actual publication/request and the measured sample/build.
    Historical phases are authenticated but cannot fill current-request gaps.
    No JAX arrays, native scopes or fabricated reservations are created here.
    """
    if not isinstance(value, dict) or set(value) != {"root", "phases"}:
        raise ValueError("NativeExecutionInterchange must contain root and phases.")
    raw_phases = value["phases"]
    if value["root"] != 0 or type(value["root"]) is not int or not isinstance(
        raw_phases, list
    ) or not raw_phases:
        raise ValueError("NativeExecutionInterchange must have a nonempty root-first graph.")
    phases: list[NativeExecutionPhaseRecord] = []

    def uint(item: object) -> int:
        if type(item) is not int or not 0 <= item <= np.iinfo(np.uint64).max:
            raise ValueError("NativeExecutionInterchange counters must be uint64 integers.")
        return item

    def seconds(item: object) -> float:
        if type(item) not in (int, float):
            raise ValueError("NativeExecutionInterchange clocks must be finite nonnegative numbers.")
        number = cast(int | float, item)
        if not math.isfinite(number) or number < 0:
            raise ValueError("NativeExecutionInterchange clocks must be finite nonnegative numbers.")
        return float(number)

    def digest(item: object) -> str:
        if not isinstance(item, str) or len(item) != 64 or any(
            character not in "0123456789abcdef" for character in item
        ):
            raise ValueError("NativeExecutionInterchange IDs must be canonical SHA256 hex.")
        return item

    def edge(item: object) -> int | None:
        if item is None:
            return None
        if type(item) is not int or not 0 <= item < len(raw_phases):
            raise ValueError("NativeExecutionInterchange edges must name a phase in this graph.")
        return item

    for raw in raw_phases:
        if not isinstance(raw, dict) or set(raw) != set(
            NativeExecutionPhaseRecord.__annotations__
        ):
            raise ValueError("NativeExecutionInterchange phases must retain the complete field schema.")
        phase = cast(NativeExecutionPhaseRecord, raw)
        work = phase["work_counters"]
        memory = phase["memory_counters"]
        if not isinstance(work, dict) or set(work) != set(
            NativeExecutionWorkRecord.__annotations__
        ) or not isinstance(memory, dict) or set(memory) != set(
            NativeExecutionMemoryRecord.__annotations__
        ):
            raise ValueError("NativeExecutionInterchange must retain every named counter.")
        owner = phase["owner_id"]
        if owner is not None:
            if not isinstance(owner, str):
                raise ValueError("NativeExecutionInterchange owners must be strings or None.")
            parse(owner, Identifier, "NativeExecutionInterchange.owner_id")
        imported = phase["imported_preparation_ids"]
        if not isinstance(imported, list) or len(imported) > NATIVE_EXECUTION_IMPORT_CAPACITY:
            raise ValueError("NativeExecutionInterchange imports must fit the receipt capacity.")
        imported_ids = [digest(item) for item in imported]
        if imported_ids != sorted(set(imported_ids)):
            raise ValueError("NativeExecutionInterchange imports must be sorted and unique.")
        history = phase["history_evidence"]
        if not isinstance(history, list) or any(item is None for item in history):
            raise ValueError("NativeExecutionInterchange history must contain phase indexes.")
        history_edges = [edge(item) for item in history]
        preparation = edge(phase["preparation_evidence"])
        consumer = edge(phase["consumer_evidence"])
        managed = phase["managed_admission_peak_bytes_upper"]
        evidence = NativeExecutionEvidence(
            work_counters=NativeExecutionWorkCounters(
                **{name: uint(item) for name, item in work.items()}
            ),
            memory_counters=NativeExecutionMemoryCounters(
                **{name: uint(item) for name, item in memory.items()}
            ),
            elapsed_seconds=seconds(phase["elapsed_seconds"]),
            status=MeshcoreStatus(uint(phase["status"])),
            externally_charged_work=uint(phase["externally_charged_work"]),
            externally_charged_geometry_queries=uint(phase["externally_charged_geometry_queries"]),
            native_primitive_queries=uint(phase["native_primitive_queries"]),
            host_storage_live_bytes_upper=uint(phase["host_storage_live_bytes_upper"]),
            host_storage_peak_bytes_upper=uint(phase["host_storage_peak_bytes_upper"]),
            managed_admission_peak_bytes_upper=None if managed is None else uint(managed),
            prior_elapsed_seconds=seconds(phase["prior_elapsed_seconds"]),
        )
        canonical = _phase_record(
            evidence,
            owner_id=owner,
            source_work=uint(phase["source_preparation_work_units"]),
            source_queries=uint(phase["source_preparation_geometry_queries"]),
            preparation_seconds=seconds(phase["preparation_seconds"]),
            preparation_evidence=preparation,
            consumer_evidence=consumer,
            history_evidence=cast(list[int], history_edges),
            receipt_id=digest(phase["receipt_id"]),
            imported_preparation_ids=tuple(imported_ids),
        )
        uint(phase["total_work_units"])
        uint(phase["total_geometry_queries"])
        seconds(phase["total_elapsed_seconds"])
        if canonical != phase:
            raise ValueError("NativeExecutionInterchange aggregates/status must match measured phase values.")
        phases.append(canonical)

    def receipt(index: int | None) -> str | None:
        return None if index is None else phases[index]["receipt_id"]

    visiting: set[int] = set()
    complete: set[int] = set()
    pending = [(0, False)]
    while pending:
        index, leaving = pending.pop()
        phase = phases[index]
        preparation = phase["preparation_evidence"]
        consumer = phase["consumer_evidence"]
        history = phase["history_evidence"]
        if leaving:
            visiting.remove(index)
            complete.add(index)
            expected = _receipt_phase_fingerprint(
                phase, receipt(preparation), receipt(consumer),
                [phases[item]["receipt_id"] for item in history],
            )
            if expected != phase["receipt_id"]:
                raise ValueError("NativeExecutionInterchange receipt ID must bind the complete ended graph.")
            continue
        if index in visiting:
            raise ValueError("NativeExecutionInterchange phase graph must be acyclic.")
        if index in complete:
            continue
        visiting.add(index)
        if preparation is not None:
            previous = phases[preparation]
            if (
                phase["source_preparation_work_units"] != previous["total_work_units"]
                or phase["source_preparation_geometry_queries"] != previous["total_geometry_queries"]
                or phase["preparation_seconds"] != previous["total_elapsed_seconds"]
            ):
                raise ValueError("NativeExecutionInterchange preparation aggregates must match the paid predecessor.")
        for predecessor in (preparation, consumer):
            if predecessor is None:
                continue
            previous = phases[predecessor]
            if phase["owner_id"] is not None and previous["owner_id"] is not None and (
                phase["owner_id"] != previous["owner_id"]
            ):
                imported = (
                    previous["receipt_id"] in phase["imported_preparation_ids"]
                    if predecessor == preparation else
                    preparation is not None and phases[preparation]["receipt_id"]
                    in previous["imported_preparation_ids"]
                )
                if not imported:
                    raise ValueError("NativeExecutionInterchange foreign owners require the actual preparation import.")
        history_ids = [phases[item]["receipt_id"] for item in history]
        if history_ids != sorted(set(history_ids)):
            raise ValueError("NativeExecutionInterchange history must be sorted and unique.")
        charged = preparation
        charged_seen: set[int] = set()
        while charged is not None:
            if charged in charged_seen:
                raise ValueError("NativeExecutionInterchange preparation chain must be acyclic.")
            charged_seen.add(charged)
            if phases[charged]["receipt_id"] in history_ids:
                raise ValueError("NativeExecutionInterchange history cannot replace charged preparation.")
            charged = phases[charged]["preparation_evidence"]
        pending.append((index, True))
        pending.extend(
            (item, False)
            for item in (preparation, consumer, *history)
            if item is not None
        )
    if len(complete) != len(phases):
        raise ValueError("NativeExecutionInterchange cannot contain disconnected receipt phases.")

    current: set[int] = set()
    preparation_complete = True
    observed: list[int] = []
    missing_observation = False
    current_pending = [0]
    while current_pending:
        index = current_pending.pop()
        if index in current:
            continue
        current.add(index)
        phase = phases[index]
        peak = phase["managed_admission_peak_bytes_upper"]
        if peak is None:
            missing_observation = True
        else:
            observed.append(peak)
        if phase["preparation_evidence"] is None and any(
            phase[name] != 0
            for name in (
                "source_preparation_work_units",
                "source_preparation_geometry_queries",
                "preparation_seconds",
            )
        ):
            preparation_complete = False
        current_pending.extend(
            item for item in (phase["preparation_evidence"], phase["consumer_evidence"])
            if item is not None
        )
    current_ids = {phases[index]["receipt_id"] for index in current}
    if any(
        not set(phases[index]["imported_preparation_ids"]) <= current_ids
        for index in current
    ):
        preparation_complete = False
    return {
        "root_receipt_id": phases[0]["receipt_id"],
        "owner_id": phases[0]["owner_id"],
        "current_phase_indexes": sorted(current),
        "preparation_complete": preparation_complete,
        "managed_admission_peak_bytes_upper": (
            None if missing_observation or not preparation_complete else max(observed)
        ),
    }


def _history_identities(
    history: tuple[NativeExecutionRecord, ...],
    preparation: NativeExecutionRecord | None,
    /,
) -> None:
    """Require ordered unique completed requests disjoint from the charged chain."""
    if type(history) is not tuple or any(
        type(record) is not NativeExecutionRecord for record in history
    ):
        raise ValueError(
            "NativeExecutionRecord.history_evidence must be a tuple of NativeExecutionRecord; "
            f"got {type(history).__name__}."
        )
    identities = tuple(record.receipt_hex() for record in history)
    if identities != tuple(sorted(set(identities))):
        raise ValueError(
            "NativeExecutionRecord.history_evidence must be sorted and unique by receipt ID; "
            f"got {identities!r}."
        )
    charged: set[str] = set()
    current = preparation
    while current is not None:
        charged.add(current.receipt_hex())
        current = current.preparation_evidence
    if charged.intersection(identities):
        raise ValueError(
            "NativeExecutionRecord.history_evidence must exclude this request's charged "
            "preparation chain; got a receipt both charged and historical."
        )


@final
class NativeExecutionRecord(StrictModule, NonTrainableState):
    """Store dynamic numerical evidence of an ended native scope.

    Raw counters are this phase's own measured work. ``preparation_evidence``
    chains this request's charged earlier phases and contributes work totals;
    ``consumer_evidence`` retains a nested consumer outside the work totals.
    ``history_evidence`` retains completed earlier requests as provenance:
    their work and geometry queries stay charged to their own allowances.

    ``managed_admission_peak_bytes_upper`` is a simultaneous managed-admission
    envelope, not a sum of native and host peaks. ``None`` means observation
    coverage was incomplete; an observed zero remains a scalar uint64 leaf.
    Its total covers preparation and consumer phases but excludes history.
    """

    __strict_contract__ = True

    work_counters: NativeExecutionDeviceWorkCounters
    memory_counters: NativeExecutionDeviceMemoryCounters
    elapsed_seconds: Array
    prior_elapsed_seconds: Array
    status: Array
    externally_charged_work: Array
    externally_charged_geometry_queries: Array
    native_primitive_queries: Array
    host_storage_live_bytes_upper: Array
    host_storage_peak_bytes_upper: Array
    managed_admission_peak_bytes_upper: Array | None
    source_preparation_work_units: Array
    source_preparation_geometry_queries: Array
    preparation_seconds: Array
    preparation_evidence: NativeExecutionRecord | None
    consumer_evidence: NativeExecutionRecord | None
    history_evidence: tuple[NativeExecutionRecord, ...]
    owner_id: Identifier | None = eqx.field(static=True)
    receipt_id: Array
    imported_preparation_ids: Array
    imported_preparation_count: Array

    @checked
    def __init__(
        self,
        evidence: NativeExecutionEvidence,
        /,
        *,
        source_preparation_work_units: int | None = None,
        source_preparation_geometry_queries: int | None = None,
        preparation_seconds: float | None = None,
        preparation_evidence: NativeExecutionRecord | None = None,
        consumer_evidence: NativeExecutionRecord | None = None,
        history_evidence: tuple[NativeExecutionRecord, ...] = (),
        owner_id: str | None = None,
        imports: NativeExecutionImports | None = None,
    ) -> None:

        owner = None if owner_id is None else parse(owner_id, Identifier, "owner_id")

        if imports is not None:
            imports.require_matches(evidence, preparation_evidence)
        if type(history_evidence) is not tuple or any(
            type(record) is not NativeExecutionRecord for record in history_evidence
        ):
            raise TypeError(
                "NativeExecutionRecord.history_evidence must be a tuple of NativeExecutionRecord; "
                f"got {type(history_evidence).__name__}"
            )
        unique = {record.receipt_hex(): record for record in history_evidence}
        history = tuple(unique[identity] for identity in sorted(unique))
        _history_identities(history, preparation_evidence)
        imported_ids = () if imports is None else imports.receipt_ids
        count = len(imported_ids)
        if count > NATIVE_EXECUTION_IMPORT_CAPACITY:
            raise ValueError(
                "NativeExecutionRecord.imported_preparation_count must fit capacity "
                f"{NATIVE_EXECUTION_IMPORT_CAPACITY}; got {count}."
            )
        if preparation_evidence is not None:
            if type(preparation_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    f"NativeExecutionRecord.preparation_evidence must be NativeExecutionRecord; got {type(preparation_evidence).__name__}"
                )
            preparation_evidence.require_valid()
            totals = jax.device_get(
                (
                    preparation_evidence.total_work_units,
                    preparation_evidence.total_geometry_queries,
                    preparation_evidence.total_elapsed_seconds,
                )
            )
            work, queries, elapsed = (
                int(np.asarray(totals[0])),
                int(np.asarray(totals[1])),
                float(np.asarray(totals[2])),
            )
            if (
                source_preparation_work_units is not None
                and source_preparation_work_units != work
                or source_preparation_geometry_queries is not None
                and source_preparation_geometry_queries != queries
                or preparation_seconds is not None
                and preparation_seconds != elapsed
            ):
                raise ValueError(
                    f"NativeExecutionRecord.preparation_aggregates must match the ended phase chain; got work/queries/seconds {(source_preparation_work_units, source_preparation_geometry_queries, preparation_seconds)!r}"
                )
            source_preparation_work_units = work
            source_preparation_geometry_queries = queries
            preparation_seconds = elapsed
        source_work = nonnegative_integer(
            0 if source_preparation_work_units is None else source_preparation_work_units,
            "source_preparation_work_units",
        )
        source_queries = nonnegative_integer(
            0
            if source_preparation_geometry_queries is None
            else source_preparation_geometry_queries,
            "source_preparation_geometry_queries",
        )
        seconds = 0.0 if preparation_seconds is None else float(preparation_seconds)
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError(
                f"NativeExecutionRecord.preparation_seconds must be a finite nonnegative clock measurement; got {seconds!r}"
            )
        self.work_counters = NativeExecutionDeviceWorkCounters.from_host(
            evidence.work_counters
        )
        self.memory_counters = NativeExecutionDeviceMemoryCounters.from_host(
            evidence.memory_counters
        )
        self.elapsed_seconds = jax.device_put(
            np.asarray(evidence.elapsed_seconds, dtype=np.float64)
        )
        self.prior_elapsed_seconds = jax.device_put(
            np.asarray(evidence.prior_elapsed_seconds, dtype=np.float64)
        )
        self.status = jax.device_put(np.asarray(evidence.status, dtype=np.int32))
        self.externally_charged_work = jax.device_put(
            np.asarray(evidence.externally_charged_work, dtype=np.uint64)
        )
        self.externally_charged_geometry_queries = jax.device_put(
            np.asarray(evidence.externally_charged_geometry_queries, dtype=np.uint64)
        )
        self.native_primitive_queries = jax.device_put(
            np.asarray(evidence.native_primitive_queries, dtype=np.uint64)
        )
        self.host_storage_live_bytes_upper = jax.device_put(
            np.asarray(evidence.host_storage_live_bytes_upper, dtype=np.uint64)
        )
        self.host_storage_peak_bytes_upper = jax.device_put(
            np.asarray(evidence.host_storage_peak_bytes_upper, dtype=np.uint64)
        )
        self.managed_admission_peak_bytes_upper = (
            None
            if evidence.managed_admission_peak_bytes_upper is None
            else jax.device_put(
                np.asarray(evidence.managed_admission_peak_bytes_upper, dtype=np.uint64)
            )
        )
        self.source_preparation_work_units = jax.device_put(
            np.asarray(source_work, dtype=np.uint64)
        )
        self.source_preparation_geometry_queries = jax.device_put(
            np.asarray(source_queries, dtype=np.uint64)
        )
        self.preparation_seconds = jax.device_put(np.asarray(seconds, dtype=np.float64))
        self.preparation_evidence = preparation_evidence
        self.consumer_evidence = consumer_evidence
        self.history_evidence = history
        self.owner_id = owner
        imported_digests = np.zeros(
            (NATIVE_EXECUTION_IMPORT_CAPACITY, 32), dtype=np.uint8
        )
        for row, identity in enumerate(imported_ids):
            imported_digests[row] = np.frombuffer(bytes.fromhex(identity), dtype=np.uint8)
        self.imported_preparation_ids = jax.device_put(imported_digests)
        self.imported_preparation_count = jax.device_put(
            np.asarray(count, dtype=np.uint32)
        )
        self.receipt_id = jax.device_put(
            np.frombuffer(
                bytes.fromhex(
                    _receipt_fingerprint(
                        evidence,
                        owner_id,
                        source_work,
                        source_queries,
                        seconds,
                        preparation_evidence,
                        consumer_evidence,
                        history,
                        imported_ids,
                    )
                ),
                dtype=np.uint8,
            ).copy()
        )
        self.require_valid()

    def require_valid(self) -> None:
        """Validate the ended host evidence graph outside numerical iteration."""
        self._require_graph_valid(frozenset())

    def _require_graph_valid(self, authenticated: frozenset[int], /) -> None:
        """Validate new phases after the caller authenticates exact private owners."""
        pending: list[tuple[NativeExecutionRecord, bool]] = [(self, False)]
        visiting: set[int] = set()
        complete: set[int] = set(authenticated)
        while pending:
            current, leaving = pending.pop()
            if type(current) is not NativeExecutionRecord:
                raise ValueError(
                    f"NativeExecutionRecord.phase_graph must contain NativeExecutionRecord nodes; got {type(current).__name__}"
                )
            identity = id(current)
            if leaving:
                visiting.remove(identity)
                complete.add(identity)
                current._require_receipt_identity()
                continue
            if identity in visiting:
                raise ValueError(
                    "NativeExecutionRecord.phase_graph must be acyclic; got a repeated active node"
                )
            if identity in complete:
                continue
            visiting.add(identity)
            current._require_phase_valid()
            previous = current.preparation_evidence
            if previous is not None:
                if type(previous) is not NativeExecutionRecord:
                    raise ValueError(
                        f"NativeExecutionRecord.preparation_evidence must be NativeExecutionRecord; got {type(previous).__name__}"
                    )
                if (
                    int(np.asarray(current.source_preparation_work_units))
                    != int(np.asarray(previous.total_work_units))
                    or int(np.asarray(current.source_preparation_geometry_queries))
                    != int(np.asarray(previous.total_geometry_queries))
                    or float(np.asarray(current.preparation_seconds))
                    != float(np.asarray(previous.total_elapsed_seconds))
                ):
                    raise ValueError(
                        "NativeExecutionRecord.preparation_aggregates must match the predecessor phase chain; got differing aggregates"
                    )
            _history_identities(current.history_evidence, previous)
            pending.append((current, True))
            for predecessor in (previous, current.consumer_evidence):
                if predecessor is None:
                    continue
                if type(predecessor) is not NativeExecutionRecord:
                    raise ValueError(
                        f"NativeExecutionRecord.phase_graph must contain NativeExecutionRecord predecessors; got {type(predecessor).__name__}"
                    )
                if (
                    current.owner_id is not None
                    and predecessor.owner_id is not None
                    and current.owner_id != predecessor.owner_id
                ):
                    imported = (
                        predecessor.receipt_hex() in current.imported_preparation_hex()
                        if predecessor is previous
                        else previous is not None
                        and previous.receipt_hex()
                        in predecessor.imported_preparation_hex()
                    )
                    if not imported:
                        raise ValueError(
                            "NativeExecutionRecord.owner_id must match predecessor operations "
                            "or include the predecessor preparation ID in its import record; "
                            f"got {current.owner_id!r}/{predecessor.owner_id!r}."
                        )
                pending.append((predecessor, False))
            # Completed earlier requests keep their own owners and allowances.
            pending.extend((record, False) for record in current.history_evidence)

    def _require_receipt_identity(self) -> None:
        expected = _receipt_fingerprint(
            self.native_evidence(),
            None if self.owner_id is None else str(self.owner_id),
            int(np.asarray(self.source_preparation_work_units)),
            int(np.asarray(self.source_preparation_geometry_queries)),
            float(np.asarray(self.preparation_seconds)),
            self.preparation_evidence,
            self.consumer_evidence,
            self.history_evidence,
            self.imported_preparation_hex(),
        )
        if self.receipt_hex() != expected:
            raise ValueError(
                "NativeExecutionRecord.receipt_id must bind its measured phase and imported IDs; "
                f"got {self.receipt_hex()!r}, expected {expected!r}."
            )

    def receipt_hex(self) -> str:
        """Read the numerical phase digest at an ended host evidence boundary."""
        return np.asarray(self.receipt_id, dtype=np.uint8).tobytes().hex()

    def imported_preparation_hex(self) -> tuple[str, ...]:
        """Read imported phase IDs outside numerical iteration."""
        count = int(np.asarray(self.imported_preparation_count))
        return tuple(
            row.tobytes().hex()
            for row in np.asarray(self.imported_preparation_ids)[:count]
        )

    def native_evidence(self) -> NativeExecutionEvidence:
        """Return this phase's host evidence with one explicit device transfer."""
        (
            work,
            memory,
            elapsed,
            status,
            charged_work,
            charged_queries,
            primitive_queries,
            host_live,
            host_peak,
            managed_peak,
            prior,
        ) = jax.device_get(
            (
                self.work_counters,
                self.memory_counters,
                self.elapsed_seconds,
                self.status,
                self.externally_charged_work,
                self.externally_charged_geometry_queries,
                self.native_primitive_queries,
                self.host_storage_live_bytes_upper,
                self.host_storage_peak_bytes_upper,
                self.managed_admission_peak_bytes_upper,
                self.prior_elapsed_seconds,
            )
        )
        return NativeExecutionEvidence(
            work_counters=NativeExecutionWorkCounters(
                **{name: int(np.asarray(value)) for name, value in work.items()}
            ),
            memory_counters=NativeExecutionMemoryCounters(
                **{name: int(np.asarray(value)) for name, value in memory.items()}
            ),
            elapsed_seconds=float(np.asarray(elapsed)),
            status=MeshcoreStatus(int(np.asarray(status))),
            externally_charged_work=int(np.asarray(charged_work)),
            externally_charged_geometry_queries=int(np.asarray(charged_queries)),
            native_primitive_queries=int(np.asarray(primitive_queries)),
            host_storage_live_bytes_upper=int(np.asarray(host_live)),
            host_storage_peak_bytes_upper=int(np.asarray(host_peak)),
            managed_admission_peak_bytes_upper=(
                None if managed_peak is None else int(np.asarray(managed_peak))
            ),
            prior_elapsed_seconds=float(np.asarray(prior)),
        )

    def to_record(self) -> NativeExecutionInterchange:
        """Convert ended phase evidence at a JSON host boundary.

        Root-first traversal visits preparation, consumer, then history edges.
        Shared historical objects retain one local graph index; indexes are not
        scientific identities. Consumer and history evidence never contribute to
        totals.
        This method runs outside numerical iteration. It preserves numerical
        leaves while serializing the ended phase graph.
        """
        self.require_valid()
        receipts = [self]
        positions = {id(self): 0}
        phases: list[NativeExecutionPhaseRecord] = []
        cursor = 0
        while cursor < len(receipts):
            receipt = receipts[cursor]
            for predecessor in (
                receipt.preparation_evidence,
                receipt.consumer_evidence,
                *receipt.history_evidence,
            ):
                if predecessor is not None and id(predecessor) not in positions:
                    positions[id(predecessor)] = len(receipts)
                    receipts.append(predecessor)
            phases.append(
                _phase_record(
                    receipt.native_evidence(),
                    owner_id=None if receipt.owner_id is None else str(receipt.owner_id),
                    source_work=int(np.asarray(receipt.source_preparation_work_units)),
                    source_queries=int(
                        np.asarray(receipt.source_preparation_geometry_queries)
                    ),
                    preparation_seconds=float(np.asarray(receipt.preparation_seconds)),
                    preparation_evidence=(
                        None
                        if receipt.preparation_evidence is None
                        else positions[id(receipt.preparation_evidence)]
                    ),
                    consumer_evidence=(
                        None
                        if receipt.consumer_evidence is None
                        else positions[id(receipt.consumer_evidence)]
                    ),
                    history_evidence=[
                        positions[id(record)] for record in receipt.history_evidence
                    ],
                    receipt_id=receipt.receipt_hex(),
                    imported_preparation_ids=receipt.imported_preparation_hex(),
                )
            )
            cursor += 1
        return {"root": 0, "phases": phases}

    def _require_imports_valid(self) -> None:
        if self.receipt_id.shape != (32,) or self.receipt_id.dtype != jnp.uint8:
            raise ValueError(
                "NativeExecutionRecord.receipt_id must contain 32 uint8 digest bytes; "
                f"got {self.receipt_id.dtype} {self.receipt_id.shape}."
            )
        if (
            self.imported_preparation_ids.shape != (NATIVE_EXECUTION_IMPORT_CAPACITY, 32)
            or self.imported_preparation_ids.dtype != jnp.uint8
            or self.imported_preparation_count.shape != ()
            or self.imported_preparation_count.dtype != jnp.uint32
        ):
            raise ValueError(
                "NativeExecutionRecord.imported_preparation_ids must use fixed uint8 capacity "
                "and a scalar uint32 count; "
                f"got {self.imported_preparation_ids.shape}/{self.imported_preparation_count.shape}."
            )
        count = int(np.asarray(self.imported_preparation_count))
        if count > NATIVE_EXECUTION_IMPORT_CAPACITY:
            raise ValueError(
                "NativeExecutionRecord.imported_preparation_count must fit capacity "
                f"{NATIVE_EXECUTION_IMPORT_CAPACITY}; got {count}."
            )
        if np.any(np.asarray(self.imported_preparation_ids)[count:]):
            raise ValueError(
                "NativeExecutionRecord.imported_preparation_ids must keep unused rows zero; "
                f"got nonzero padding after count {count}."
            )
        identities = self.imported_preparation_hex()
        if identities != tuple(sorted(set(identities))):
            raise ValueError(
                "NativeExecutionRecord.imported_preparation_ids must be sorted and unique; "
                f"got {identities!r}."
            )

    def _require_phase_valid(self) -> None:
        self._require_imports_valid()
        if self.owner_id is not None:
            parse(self.owner_id, Identifier, "owner_id")
        if type(self.work_counters) is not NativeExecutionDeviceWorkCounters:
            raise ValueError(
                "NativeExecutionRecord.work_counters must be NativeExecutionDeviceWorkCounters; "
                f"got {type(self.work_counters).__name__}."
            )
        if type(self.memory_counters) is not NativeExecutionDeviceMemoryCounters:
            raise ValueError(
                "NativeExecutionRecord.memory_counters must be NativeExecutionDeviceMemoryCounters; "
                f"got {type(self.memory_counters).__name__}."
            )
        self.work_counters.__post_init__()
        self.memory_counters.__post_init__()
        for values in (
            self.externally_charged_work,
            self.externally_charged_geometry_queries,
            self.native_primitive_queries,
            self.source_preparation_work_units,
            self.source_preparation_geometry_queries,
            self.host_storage_live_bytes_upper,
            self.host_storage_peak_bytes_upper,
        ):
            if values.shape != () or values.dtype != jnp.uint64:
                raise ValueError(
                    f"NativeExecutionRecord.action_counts must be scalar uint64 leaves; got {values.dtype} {values.shape}"
                )
        managed_peak = self.managed_admission_peak_bytes_upper
        if managed_peak is not None and (
            managed_peak.shape != () or managed_peak.dtype != jnp.uint64
        ):
            raise ValueError(
                "NativeExecutionRecord.managed_admission_peak_bytes_upper must be "
                "a scalar uint64 leaf or None; "
                f"got {managed_peak.dtype} {managed_peak.shape}."
            )
        if (
            self.status.shape != ()
            or self.status.dtype != jnp.int32
            or self.elapsed_seconds.shape != ()
            or self.elapsed_seconds.dtype != jnp.float64
        ):
            raise ValueError(
                f"NativeExecutionRecord.status_and_clock must be scalar int32 status and float64 elapsed time; got {self.status.dtype} {self.status.shape} and {self.elapsed_seconds.dtype} {self.elapsed_seconds.shape}"
            )
        MeshcoreStatus(int(np.asarray(self.status)))
        if int(np.asarray(self.host_storage_live_bytes_upper)) > int(
            np.asarray(self.host_storage_peak_bytes_upper)
        ):
            raise ValueError(
                "NativeExecutionRecord.host_storage_bounds must be ordered live <= peak; got live greater than peak"
            )
        seconds = float(np.asarray(self.elapsed_seconds))
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError(
                f"NativeExecutionRecord.elapsed_seconds must be finite and nonnegative; got {seconds!r}"
            )
        if (
            self.preparation_seconds.shape != ()
            or self.preparation_seconds.dtype != jnp.float64
        ):
            raise ValueError(
                f"NativeExecutionRecord.preparation_seconds must be a scalar float64 leaf; got {self.preparation_seconds.dtype} {self.preparation_seconds.shape}"
            )
        preparation = float(np.asarray(self.preparation_seconds))
        if not np.isfinite(preparation) or preparation < 0.0:
            raise ValueError(
                f"NativeExecutionRecord.preparation_seconds must be finite and nonnegative; got {preparation!r}"
            )
        if (
            self.prior_elapsed_seconds.shape != ()
            or self.prior_elapsed_seconds.dtype != jnp.float64
        ):
            raise ValueError(
                f"NativeExecutionRecord.prior_elapsed_seconds must be a scalar float64 leaf; got {self.prior_elapsed_seconds.dtype} {self.prior_elapsed_seconds.shape}"
            )
        prior = float(np.asarray(self.prior_elapsed_seconds))
        if not np.isfinite(prior) or prior < 0.0:
            raise ValueError(
                f"NativeExecutionRecord.prior_elapsed_seconds must be finite and nonnegative; got {prior!r}"
            )

    @property
    def total_elapsed_seconds(self) -> Array:
        """Return scope duration plus admitted predecessor durations."""
        return (
            self.elapsed_seconds + self.prior_elapsed_seconds + self.preparation_seconds
        )

    @property
    def total_work_units(self) -> Array:
        return self.source_preparation_work_units + self.work_counters.construction_work

    @property
    def total_geometry_queries(self) -> Array:
        return (
            self.source_preparation_geometry_queries + self.work_counters.source_queries
        )

    @property
    def total_host_storage_peak_bytes_upper(self) -> Array:
        """Maximum phase upper bound; callers retain live predecessors first."""
        result = self.host_storage_peak_bytes_upper
        previous = self.preparation_evidence
        while previous is not None:
            result = jnp.maximum(result, previous.host_storage_peak_bytes_upper)
            previous = previous.preparation_evidence
        return result

    @property
    def total_managed_admission_peak_bytes_upper(self) -> Array | None:
        """Maximum complete same-request envelope, excluding historical requests.

        Preparation and nested consumer phases belong to this request. A missing
        observation in any such phase makes its admission envelope incomplete;
        completed earlier requests are provenance only, never extra charges.

        Evaluate this completeness boundary on the host: nonzero preparation
        aggregates without a linked phase do not observe that prior phase's
        memory envelope.
        """
        result = self.managed_admission_peak_bytes_upper
        if result is None:
            return None
        pending: list[NativeExecutionRecord] = [self]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            peak = current.managed_admission_peak_bytes_upper
            if peak is None:
                return None
            if current.preparation_evidence is None:
                aggregates = jax.device_get(
                    (
                        current.source_preparation_work_units,
                        current.source_preparation_geometry_queries,
                        current.preparation_seconds,
                    )
                )
                if any(np.asarray(value) != 0 for value in aggregates):
                    return None
            result = jnp.maximum(result, peak)
            for predecessor in (
                current.preparation_evidence,
                current.consumer_evidence,
            ):
                if predecessor is not None:
                    pending.append(predecessor)
        return result


@dataclass(frozen=True, slots=True)
class NativeMeshingPhaseMeasurement:
    """Store a measured interval and its optional work counter.

    Accumulated native intervals supply their measured invocation count. Missing
    stages produce no record; an observed clock-resolution zero is not missing.
    These host records are never included in a plan, trace, result or fingerprint.
    """

    phase: NativeMeshingPhase
    elapsed_seconds: float
    work_units: int | None = None
    invocations: int = 1

    def __post_init__(self) -> None:
        parse(self.phase, NativeMeshingPhase, "phase")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError(
                f"NativeMeshingPhaseMeasurement.elapsed_seconds must be finite and nonnegative; got {self.elapsed_seconds!r}"
            )
        if self.work_units is not None and (
            isinstance(self.work_units, bool)
            or not isinstance(self.work_units, int)
            or self.work_units < 0
        ):
            raise ValueError(
                f"NativeMeshingPhaseMeasurement.work_units must be a nonnegative integer counter or None; got {self.work_units!r}"
            )
        if (
            isinstance(self.invocations, bool)
            or not isinstance(self.invocations, int)
            or self.invocations < 1
        ):
            raise ValueError(
                f"NativeMeshingPhaseMeasurement.invocations must be a positive measured-call count; got {self.invocations!r}"
            )


type NativeMeshingPhaseRecorder = Callable[[NativeMeshingPhaseMeasurement], None]


def phase_started(record_phase: NativeMeshingPhaseRecorder | None, /) -> float | None:
    """Read the monotonic clock only when the execution channel is enabled."""
    return None if record_phase is None else perf_counter()


def record_elapsed(
    record_phase: NativeMeshingPhaseRecorder | None,
    phase: NativeMeshingPhase,
    started: float | None,
    /,
    *,
    work_units: int | None = None,
) -> None:
    if record_phase is None:
        return
    if started is None:
        raise RuntimeError(
            "NativeMeshingPhaseMeasurement.started must contain a start clock when enabled; got None"
        )
    record_phase(
        NativeMeshingPhaseMeasurement(phase, perf_counter() - started, work_units, 1)
    )


@contextmanager
def measure_phase(
    record_phase: NativeMeshingPhaseRecorder | None,
    phase: NativeMeshingPhase,
    /,
    *,
    work_units: int | None = None,
) -> Iterator[None]:
    """Measure a host stage, including its failed execution interval.

    Boundaries follow host/device completion, rather than timing an asynchronous
    launch as numerical execution. Recorder exceptions
    are consumer errors and propagate; they are never converted to mesh evidence.
    """
    if record_phase is None:
        yield
        return
    started = phase_started(record_phase)
    try:
        yield
    finally:
        record_elapsed(record_phase, phase, started, work_units=work_units)


__all__ = [
    "NativeMeshingPhase",
    "NativeMeshingPhaseMeasurement",
    "NativeMeshingPhaseRecorder",
]
