#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Execution-only meshing measurements, deliberately outside scientific identity."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import final, Literal, TypeAlias, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._meshcore import MeshcoreStatus, NativeExecutionEvidence
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, Identifier, parse


NativeMeshingPhase: TypeAlias = Literal[
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


class NativeExecutionPhaseRecord(TypedDict):
    """JSON host values of one actual phase; edges are local graph indexes."""

    owner_id: str | None
    work: list[int]
    memory: list[int]
    status: int
    status_name: str
    elapsed_seconds: float
    prior_elapsed_seconds: float
    externally_charged_work: int
    externally_charged_geometry_queries: int
    native_primitive_queries: int
    host_storage_live_bytes_upper: int
    host_storage_peak_bytes_upper: int
    source_preparation_work_units: int
    source_preparation_geometry_queries: int
    preparation_seconds: float
    preparation_evidence: int | None
    consumer_evidence: int | None
    total_work_units: int
    total_geometry_queries: int
    total_elapsed_seconds: float


class NativeExecutionInterchange(TypedDict):
    """An identity-preserving receipt graph, not new scientific receipt IDs."""

    root: int
    phases: list[NativeExecutionPhaseRecord]


@final
class NativeExecutionRecord(StrictModule, NonTrainableState):
    """Dynamic scientific-boundary record of an actually ended native scope."""

    __strict_contract__ = True

    work: Array
    memory: Array
    elapsed_seconds: Array
    prior_elapsed_seconds: Array
    status: Array
    externally_charged_work: Array
    externally_charged_geometry_queries: Array
    native_primitive_queries: Array
    host_storage_live_bytes_upper: Array
    host_storage_peak_bytes_upper: Array
    source_preparation_work_units: Array
    source_preparation_geometry_queries: Array
    preparation_seconds: Array
    preparation_evidence: NativeExecutionRecord | None
    consumer_evidence: NativeExecutionRecord | None
    owner_id: Identifier | None = eqx.field(static=True)

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
        owner_id: str | None = None,
    ) -> None:
        from .._validation import nonnegative_integer

        owner = None if owner_id is None else parse(owner_id, Identifier, "owner_id")

        if preparation_evidence is not None:
            if type(preparation_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    "Preparation evidence must be its exact owning native record."
                )
            preparation_evidence.require_valid()
            work = int(np.asarray(preparation_evidence.total_work_units))
            queries = int(np.asarray(preparation_evidence.total_geometry_queries))
            elapsed = float(np.asarray(preparation_evidence.total_elapsed_seconds))
            if (
                source_preparation_work_units is not None
                and source_preparation_work_units != work
                or source_preparation_geometry_queries is not None
                and source_preparation_geometry_queries != queries
                or preparation_seconds is not None
                and preparation_seconds != elapsed
            ):
                raise ValueError(
                    "Preparation aggregates differ from their actual ended phase chain."
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
                "preparation_seconds must be a finite nonnegative clock measurement."
            )
        self.work = jnp.asarray(evidence.work_evidence, dtype=jnp.uint64)
        self.memory = jnp.asarray(evidence.memory_evidence, dtype=jnp.uint64)
        self.elapsed_seconds = jnp.asarray(evidence.elapsed_seconds, dtype=jnp.float64)
        self.prior_elapsed_seconds = jnp.asarray(
            evidence.prior_elapsed_seconds, dtype=jnp.float64
        )
        self.status = jnp.asarray(evidence.status, dtype=jnp.int32)
        self.externally_charged_work = jnp.asarray(
            evidence.externally_charged_work, dtype=jnp.uint64
        )
        self.externally_charged_geometry_queries = jnp.asarray(
            evidence.externally_charged_geometry_queries, dtype=jnp.uint64
        )
        self.native_primitive_queries = jnp.asarray(
            evidence.native_primitive_queries, dtype=jnp.uint64
        )
        self.host_storage_live_bytes_upper = jnp.asarray(
            evidence.host_storage_live_bytes_upper, dtype=jnp.uint64
        )
        self.host_storage_peak_bytes_upper = jnp.asarray(
            evidence.host_storage_peak_bytes_upper, dtype=jnp.uint64
        )
        self.source_preparation_work_units = jnp.asarray(source_work, dtype=jnp.uint64)
        self.source_preparation_geometry_queries = jnp.asarray(
            source_queries, dtype=jnp.uint64
        )
        self.preparation_seconds = jnp.asarray(seconds, dtype=jnp.float64)
        self.preparation_evidence = preparation_evidence
        self.consumer_evidence = consumer_evidence
        self.owner_id = owner
        self.require_valid()

    def require_valid(self) -> None:
        pending: list[tuple[NativeExecutionRecord, bool]] = [(self, False)]
        visiting: set[int] = set()
        complete: set[int] = set()
        while pending:
            current, leaving = pending.pop()
            if type(current) is not NativeExecutionRecord:
                raise ValueError("Native evidence requires exact owning phase records.")
            identity = id(current)
            if leaving:
                visiting.remove(identity)
                complete.add(identity)
                continue
            if identity in visiting:
                raise ValueError(
                    "Native preparation and consumer evidence require an acyclic owning graph."
                )
            if identity in complete:
                continue
            visiting.add(identity)
            current._require_phase_valid()
            previous = current.preparation_evidence
            if previous is not None:
                if type(previous) is not NativeExecutionRecord:
                    raise ValueError(
                        "Native preparation requires its exact ended phase record."
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
                        "Native preparation aggregates lost their actual ended phase chain."
                    )
            pending.append((current, True))
            for predecessor in (previous, current.consumer_evidence):
                if predecessor is None:
                    continue
                if type(predecessor) is not NativeExecutionRecord:
                    raise ValueError(
                        "Native subordinate evidence requires its exact ended phase record."
                    )
                if (
                    current.owner_id is not None
                    and predecessor.owner_id is not None
                    and current.owner_id != predecessor.owner_id
                ):
                    raise ValueError(
                        "Native evidence phases belong to different declared operations."
                    )
                pending.append((predecessor, False))

    def to_record(self) -> NativeExecutionInterchange:
        """Convert validated ended receipts at an explicit JSON host boundary.

        Root-first traversal visits preparation before consumer edges. Shared
        historical objects retain one local graph index; indexes are not
        scientific identities. Consumer evidence never contributes to totals.
        This method is outside numerical iteration and does not snapshot live
        scopes, reconstruct receipts, or change the original numerical leaves.
        """
        self.require_valid()
        receipts = [self]
        positions = {id(self): 0}
        phases: list[NativeExecutionPhaseRecord] = []
        cursor = 0
        while cursor < len(receipts):
            receipt = receipts[cursor]
            for predecessor in (receipt.preparation_evidence, receipt.consumer_evidence):
                if predecessor is not None and id(predecessor) not in positions:
                    positions[id(predecessor)] = len(receipts)
                    receipts.append(predecessor)
            work = [int(value) for value in np.asarray(receipt.work)]
            memory = [int(value) for value in np.asarray(receipt.memory)]
            status = int(np.asarray(receipt.status))
            elapsed = float(np.asarray(receipt.elapsed_seconds))
            prior = float(np.asarray(receipt.prior_elapsed_seconds))
            preparation = float(np.asarray(receipt.preparation_seconds))
            source_work = int(np.asarray(receipt.source_preparation_work_units))
            source_queries = int(np.asarray(receipt.source_preparation_geometry_queries))
            phases.append(
                {
                    "owner_id": None
                    if receipt.owner_id is None
                    else str(receipt.owner_id),
                    "work": work,
                    "memory": memory,
                    "status": status,
                    "status_name": MeshcoreStatus(status).name,
                    "elapsed_seconds": elapsed,
                    "prior_elapsed_seconds": prior,
                    "externally_charged_work": int(
                        np.asarray(receipt.externally_charged_work)
                    ),
                    "externally_charged_geometry_queries": int(
                        np.asarray(receipt.externally_charged_geometry_queries)
                    ),
                    "native_primitive_queries": int(
                        np.asarray(receipt.native_primitive_queries)
                    ),
                    "host_storage_live_bytes_upper": int(
                        np.asarray(receipt.host_storage_live_bytes_upper)
                    ),
                    "host_storage_peak_bytes_upper": int(
                        np.asarray(receipt.host_storage_peak_bytes_upper)
                    ),
                    "source_preparation_work_units": source_work,
                    "source_preparation_geometry_queries": source_queries,
                    "preparation_seconds": preparation,
                    "preparation_evidence": (
                        None
                        if receipt.preparation_evidence is None
                        else positions[id(receipt.preparation_evidence)]
                    ),
                    "consumer_evidence": (
                        None
                        if receipt.consumer_evidence is None
                        else positions[id(receipt.consumer_evidence)]
                    ),
                    "total_work_units": source_work + work[0],
                    "total_geometry_queries": source_queries + work[1],
                    "total_elapsed_seconds": elapsed + prior + preparation,
                }
            )
            cursor += 1
        return {"root": 0, "phases": phases}

    def _require_phase_valid(self) -> None:
        if self.owner_id is not None:
            parse(self.owner_id, Identifier, "owner_id")
        for values in (self.work, self.memory):
            if values.shape != (6,) or values.dtype != jnp.uint64:
                raise ValueError(
                    "Native execution counters require six uint64 numerical leaves."
                )
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
                    "Native execution action counts require scalar uint64 leaves."
                )
        if (
            self.status.shape != ()
            or self.status.dtype != jnp.int32
            or self.elapsed_seconds.shape != ()
            or self.elapsed_seconds.dtype != jnp.float64
        ):
            raise ValueError(
                "Native execution status and clock have invalid scalar representations."
            )
        MeshcoreStatus(int(np.asarray(self.status)))
        if int(np.asarray(self.host_storage_live_bytes_upper)) > int(
            np.asarray(self.host_storage_peak_bytes_upper)
        ):
            raise ValueError(
                "Conservative host-storage bounds must be ordered live <= peak."
            )
        seconds = float(np.asarray(self.elapsed_seconds))
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError(
                "Native execution elapsed time must be finite and nonnegative."
            )
        if (
            self.preparation_seconds.shape != ()
            or self.preparation_seconds.dtype != jnp.float64
        ):
            raise ValueError(
                "Native source preparation clock requires a scalar float64 leaf."
            )
        preparation = float(np.asarray(self.preparation_seconds))
        if not np.isfinite(preparation) or preparation < 0.0:
            raise ValueError(
                "Native source preparation time must be finite and nonnegative."
            )
        if (
            self.prior_elapsed_seconds.shape != ()
            or self.prior_elapsed_seconds.dtype != jnp.float64
        ):
            raise ValueError(
                "Imported native clock debit requires a scalar float64 leaf."
            )
        prior = float(np.asarray(self.prior_elapsed_seconds))
        if not np.isfinite(prior) or prior < 0.0:
            raise ValueError(
                "Imported native clock debit must be finite and nonnegative."
            )

    @property
    def total_elapsed_seconds(self) -> Array:
        """Actual scope duration plus separately admitted predecessor durations."""
        return (
            self.elapsed_seconds + self.prior_elapsed_seconds + self.preparation_seconds
        )

    @property
    def total_work_units(self) -> Array:
        return self.source_preparation_work_units + self.work[0]

    @property
    def total_geometry_queries(self) -> Array:
        return self.source_preparation_geometry_queries + self.work[1]

    @property
    def total_host_storage_peak_bytes_upper(self) -> Array:
        """Maximum phase upper bound; callers retain live predecessors first."""
        result = self.host_storage_peak_bytes_upper
        previous = self.preparation_evidence
        while previous is not None:
            result = jnp.maximum(result, previous.host_storage_peak_bytes_upper)
            previous = previous.preparation_evidence
        return result


@dataclass(frozen=True, slots=True)
class NativeMeshingPhaseMeasurement:
    """One actual measured interval and optional owning work counter.

    Native accumulated intervals supply their actual invocation count. Missing
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
            raise ValueError("elapsed_seconds must be a finite nonnegative measurement.")
        if self.work_units is not None and (
            isinstance(self.work_units, bool)
            or not isinstance(self.work_units, int)
            or self.work_units < 0
        ):
            raise ValueError(
                "work_units must be an actual nonnegative integer counter or None."
            )
        if (
            isinstance(self.invocations, bool)
            or not isinstance(self.invocations, int)
            or self.invocations < 1
        ):
            raise ValueError("invocations must count actual measured calls.")


NativeMeshingPhaseRecorder: TypeAlias = Callable[[NativeMeshingPhaseMeasurement], None]


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
        raise RuntimeError("An enabled measurement needs its actual start clock.")
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
    """Measure an actual host stage, including its failed execution interval.

    Owners must place boundaries after their actual host/device completion, not
    measure an asynchronous launch as numerical execution. Recorder exceptions
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
