#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scope-owned preparation identities, request charges and completed-request history."""

from __future__ import annotations

import ctypes
import os
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from functools import partial
from inspect import getattr_static
from time import monotonic
from typing import final, Protocol, TYPE_CHECKING

import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._host_storage import (
    declared_object_storage_bytes,
    immutable_host_array_layout,
    retained_host_storage_fanout_upper,
    retained_host_storage_profile,
)
from .._meshcore import (
    current_native_execution_scope,
    is_native_host_buffer,
    NativeBatchTicket,
    NativeExecutionEvidence,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    NativeScopeTicket,
)
from ._measurements import (
    _phase_record,
    NATIVE_EXECUTION_IMPORT_CAPACITY,
    native_receipt_validation_bytes_upper,
    NativeExecutionRecord,
)


if TYPE_CHECKING:
    from .._physical import SpatialCoordinateContract
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._coordinate_budget import CoordinateEnclosureBudget
    from ..discretization._coordinate_operands import ContainerOwners, OwnerFields
    from ._measurements import NativeExecutionImports, NativeExecutionPhaseRecord
    from ._result import CellMeshingResult


# The transport has five uint64 counters and two binary64 clock values.
_OVERLAP_LAYOUT = (0, 0, 0, 0, 0, 0, 0)


class NativePreparedOwner(Protocol):
    """Prepared owners expose their scientific identity without a provider import."""

    @property
    def prepared_id(self) -> str: ...


def debit_request_preparation(
    limits: NativeExecutionLimits, preparation: NativeExecutionRecord, /
) -> NativeExecutionLimits:
    """Return the request allowance left after its own ended preparation.

    Callers first admit the preparation against the original limits. The
    ended phase's work, queries and elapsed time are debited exactly once; a
    live caller or workflow deadline still intersects when the scope opens.
    """
    return replace(
        limits,
        maximum_work_units=limits.maximum_work_units
        - int(np.asarray(preparation.total_work_units)),
        maximum_geometry_queries=limits.maximum_geometry_queries
        - int(np.asarray(preparation.total_geometry_queries)),
        maximum_seconds=max(
            0.0,
            limits.maximum_seconds - float(np.asarray(preparation.total_elapsed_seconds)),
        ),
    )


def _ancestors(scope: NativeExecutionScope, /) -> Iterator[NativeExecutionScope]:
    current: NativeExecutionScope | None = scope
    while current is not None:
        yield current
        current = current.parent


def _phase_chain(receipt: NativeExecutionRecord, /) -> Iterator[NativeExecutionRecord]:
    current: NativeExecutionRecord | None = receipt
    while current is not None:
        yield current
        current = current.preparation_evidence


def _receipt_graph(
    receipt: NativeExecutionRecord, /, *, scope: NativeExecutionScope
) -> Iterator[NativeExecutionRecord]:
    """Visit every charged, consumed and historical phase once."""
    pending = [receipt]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        scope.charge(work=35 + len(current.history_evidence))
        seen.add(id(current))
        yield current
        pending.extend(
            child
            for child in (
                current.preparation_evidence,
                current.consumer_evidence,
                *current.history_evidence,
            )
            if child is not None
        )


@final
@dataclass(frozen=True, slots=True)
class NativePreparationOverlap:
    """Matched predecessor contributions represented in one raw native scope."""

    work: int = 0
    geometry_queries: int = 0
    external_work: int = 0
    external_geometry_queries: int = 0
    native_primitive_queries: int = 0
    live_elapsed_seconds: float = 0.0
    imported_elapsed_seconds: float = 0.0

    def normalize(self, evidence: NativeExecutionEvidence, /) -> NativeExecutionEvidence:
        """Remove matched additive contributions while retaining measured capacity peaks."""
        if not (
            self.work
            or self.geometry_queries
            or self.external_work
            or self.external_geometry_queries
            or self.native_primitive_queries
            or self.live_elapsed_seconds
            or self.imported_elapsed_seconds
        ):
            return evidence
        work = int(evidence.work_counters.construction_work) - self.work
        queries = int(evidence.work_counters.source_queries) - self.geometry_queries
        external_work = evidence.externally_charged_work - self.external_work
        external_queries = (
            evidence.externally_charged_geometry_queries - self.external_geometry_queries
        )
        primitive_queries = (
            evidence.native_primitive_queries - self.native_primitive_queries
        )
        elapsed = evidence.elapsed_seconds - self.live_elapsed_seconds
        prior = evidence.prior_elapsed_seconds - self.imported_elapsed_seconds
        if min(work, queries, external_work, external_queries, primitive_queries) < 0:
            raise RuntimeError(
                "NativePreparationOverlap.counters must fit the matched raw evidence; "
                "got a predecessor overlap exceeding its scope."
            )
        if elapsed < 0.0 or prior < 0.0:
            raise RuntimeError(
                "NativePreparationOverlap.elapsed must fit the matched raw interval; "
                "got a predecessor interval exceeding its scope."
            )
        counters = replace(
            evidence.work_counters, construction_work=work, source_queries=queries
        )
        return replace(
            evidence,
            work_counters=counters,
            elapsed_seconds=elapsed,
            prior_elapsed_seconds=prior,
            externally_charged_work=external_work,
            externally_charged_geometry_queries=external_queries,
            native_primitive_queries=primitive_queries,
        )


@final
class NativePreparationLedger:
    """Retain one request's preparations, charges and completed-request history.

    Imported and remembered receipts belong to the live request's single
    allowance. History names completed earlier requests as provenance only:
    their work and geometry queries remain charged to their own requests.
    """

    def __init__(self) -> None:
        self._live: dict[str, dict[int, NativePreparedOwner]] = {}
        self._receipts: dict[NativeExecutionScope, dict[int, NativeExecutionRecord]] = {}
        self._imports: dict[
            NativeExecutionScope, dict[int, NativeExecutionRecord | None]
        ] = {}
        self._sources: dict[NativeExecutionScope, set[str]] = {}
        self._overlaps: dict[
            NativeExecutionScope, dict[int, NativePreparationOverlap]
        ] = {}
        self._storage: list[NativeHostStorageWorkspace] = []
        self._retained_owners: dict[int, object] = {}
        self._retention_edges = 0
        self._history: dict[NativeExecutionScope, dict[str, NativeExecutionRecord]] = {}
        self._publications: dict[
            int, tuple[weakref.ReferenceType[CellMeshingResult], _NativeSourceSeal]
        ] = {}
        self._history_sources: dict[
            int, tuple[weakref.ReferenceType[CellMeshingResult], _NativeSourceSeal]
        ] = {}
        self._receipt_graphs: dict[int, _NativeReceiptGraph] = {}

    def close(self) -> None:
        """Release preparation identities when the root scope ends."""
        self._live.clear()
        self._receipts.clear()
        self._imports.clear()
        self._sources.clear()
        self._overlaps.clear()
        self._history.clear()
        self._publications.clear()
        self._history_sources.clear()
        self._receipt_graphs.clear()
        self._retained_owners.clear()
        self._retention_edges = 0
        for storage in self._storage:
            storage.close()
        self._storage.clear()

    def live_contains(self, prepared: NativePreparedOwner, /) -> bool:
        """Authorize only a registered live object, not an equal prepared ID."""
        owners = self._live.get(prepared.prepared_id)
        return owners is not None and owners.get(id(prepared)) is prepared

    def remember_live(
        self,
        scope: NativeExecutionScope,
        prepared: NativePreparedOwner,
        /,
    ) -> None:
        """Retain a completed preparation without replaying its measured work."""
        prepared_id = prepared.prepared_id
        if not isinstance(prepared_id, str) or not prepared_id:
            raise ValueError(
                "NativePreparedOwner.prepared_id must be a nonempty string; got an invalid identity."
            )
        if self.live_contains(prepared):
            return
        self._retain_storage(scope, (prepared,), 256)
        with scope.batch(1) as ticket:
            ticket.consume(1)
            self._live.setdefault(prepared_id, {})[id(prepared)] = prepared

    def receipt_is_accounted(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord,
        /,
    ) -> bool:
        """Resolve the retained object through the scope's live ancestry."""
        for ancestor in _ancestors(scope):
            scope.charge(work=1)
            if self._receipts.get(ancestor, {}).get(id(receipt)) is receipt:
                return True
        return False

    def imported_receipt_ids(self, scope: NativeExecutionScope, /) -> tuple[str, ...]:
        """Snapshot admitted predecessor IDs visible to this live consumer."""
        if scope.evidence is not None:
            raise ValueError(
                "NativePreparationLedger.scope must remain live when recording imports; "
                "got an ended execution scope."
            )
        with NativeHostStorageWorkspace(scope) as scratch:
            scratch.set_bound(
                1024 + 512 * (len(self._retained_owners) + self._retention_edges)
            )
            receipts: set[str] = set()
            for ancestor in _ancestors(scope):
                scope.charge(work=1)
                entries = self._imports.get(ancestor, {})
                scope.charge(work=len(entries))
                for receipt in entries.values():
                    if receipt is not None:
                        scope.charge(work=32)
                        receipts.add(receipt.receipt_hex())
            scope.charge(work=len(receipts) * (1 + len(receipts).bit_length()))
            identities = tuple(sorted(receipts))
            if identities:
                self._retain_storage(scope, (), 256 + 144 * len(identities))
            return identities

    def authenticate_receipt(
        self, scope: NativeExecutionScope, receipt: NativeExecutionRecord, /
    ) -> None:
        """Authenticate exact issued owners or re-earn cold receipt validation."""
        if native_preparation_ledger(scope) is not self:
            raise ValueError("Receipt authentication must not use a foreign root attachment.")
        graph = self._receipt_graphs.get(id(receipt))
        if graph is not None and graph.phases[0].receipt is receipt:
            with scope.batch(graph.authentication_work, bytes_upper=1024) as ticket:
                graph.require_current()
                ticket.consume(graph.authentication_work)
            return
        # A receipt issued with an accepted result in this process keeps its
        # once-validated immutable graph: later roots bind the exact owner,
        # receipt ID and live leaves instead of revalidating the chain.
        issued = _completed_native_receipts.get(id(receipt))
        prepared = None if issued is None else issued._receipt_graph
        if prepared is not None:
            work = 64 + prepared.authentication_work
            with scope.batch(work, bytes_upper=1024) as ticket:
                graph = _issued_native_receipt_graph(receipt)
                ticket.consume(work)
            if graph is not None:
                self._retain_storage(scope, (receipt,), 0)
                self._receipt_graphs[id(receipt)] = graph
                return
        receipt.require_valid_in_scope(scope)
        graph, _ = _prepare_native_receipt_graph_in_scope(
            receipt, scope,
            retain=lambda amount: self._retain_storage(scope, (receipt,), amount),
        )
        if graph is not None:
            self._receipt_graphs[id(receipt)] = graph
        else:
            self._retain_storage(scope, (receipt,), 0)

    def prepay_receipt_clock(
        self, scope: NativeExecutionScope, receipt: NativeExecutionRecord, /,
        *, expires_at: float | None,
    ) -> None:
        """Intersect a root clock before cold authentication, without renewal."""
        if scope.parent is not None or native_preparation_ledger(scope) is not self:
            raise ValueError("Preparation clock admission must use its original root.")
        if self.receipt_is_accounted(scope, receipt) or id(receipt) in self._overlaps.get(scope, {}):
            return
        # Only three scalar clocks are read here. Their graph relations remain
        # subject to full admitted validation before any scientific operation.
        with scope.batch(12, bytes_upper=2048) as ticket:
            clocks = (receipt.elapsed_seconds, receipt.prior_elapsed_seconds, receipt.preparation_seconds)
            seconds = 0.0
            for clock in clocks:
                if clock.shape != () or clock.dtype != np.dtype(np.float64):
                    raise ValueError("Preparation clocks must be scalar float64 leaves.")
                value = float(np.asarray(clock))
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Preparation clocks must be finite and nonnegative.")
                seconds += value
            ticket.consume(12)
        gap = (
            0.0 if expires_at is None else max(
                0.0, scope.limits.maximum_seconds
                - (expires_at - scope.operation_started) - seconds,
            )
        )
        self._retain_storage(scope, (receipt,), 512)
        scope.import_preparation(work=0, geometry_queries=0, elapsed_seconds=seconds + gap)
        # Reuse the receiving ledger's exact-identity overlap entry. It is
        # replaced by the complete work/query overlap at import, not a registry.
        self._overlaps.setdefault(scope, {})[id(receipt)] = NativePreparationOverlap(
            imported_elapsed_seconds=seconds
        )

    def import_receipt(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord,
        /,
        *,
        import_work: bool = True,
    ) -> None:
        """Import each ended predecessor phase once, without replaying chain totals.
    
        The receipt is this request's own preparation. A completed earlier
        request recorded as history is refused rather than charged again.
        """
        with NativeHostStorageWorkspace(scope) as scratch:
            scratch.set_bound(1024)
            self.authenticate_receipt(scope, receipt)
            scratch.set_bound(
                1024 + 512 * (len(self._retained_owners) + self._retention_edges)
            )
            if native_preparation_ledger(scope) is not self:
                raise ValueError(
                    "NativePreparationLedger.import must use this live request's ledger; "
                    "got a foreign root attachment."
                )
            self._refuse_history(scope, receipt)
            accounted = []
            pending = []
            for current in _phase_chain(receipt):
                scope.charge(work=1)
                if self.receipt_is_accounted(scope, current):
                    accounted.append(current)
                else:
                    pending.append(current)
            self._retain_receipts(scope, pending, import_work=import_work)
            for current in accounted:
                # Remembering a live phase pays for its identity slot once.
                # Explicit import authorizes that same retained object without
                # replaying its work, storage, queries, or elapsed time.
                for ancestor in _ancestors(scope):
                    scope.charge(work=1)
                    if self._receipts.get(ancestor, {}).get(id(current)) is current:
                        self._imports[ancestor][id(current)] = current

    def remember_receipt(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord,
        /,
        *,
        import_work: bool = False,
    ) -> None:
        """Retain one phase identity and optionally debit its work and duration."""
        if self.receipt_is_accounted(scope, receipt):
            return
        self._refuse_history(scope, receipt)
        self._retain_receipts(scope, [receipt], import_work=import_work)

    def record_history(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord,
        /,
        *,
        completed_source: CellMeshingResult | None = None,
    ) -> None:
        """Retain a completed earlier request as provenance, never as a charge.
    
        Its work, geometry queries and elapsed time stay charged to that
        request's own allowance. The absolute workflow clock is a launch
        deadline intersected with this request, not a replay of history.
        """
        with NativeHostStorageWorkspace(scope) as scratch:
            scratch.set_bound(1024)
            self.authenticate_receipt(scope, receipt)
            scratch.set_bound(
                1024 + 512 * (len(self._retained_owners) + self._retention_edges)
            )
            _require_native_source_scope(scope)
            seal = (
                None
                if completed_source is None
                else _require_native_completed_source(completed_source, receipt)
            )
            scope.charge(work=32)
            identity = receipt.receipt_hex()
            ancestors = tuple(_ancestors(scope))
            scope.charge(work=2 * len(ancestors))
            already_recorded = any(
                identity in self._history.get(ancestor, {}) for ancestor in ancestors
            )
            scope.charge(
                work=32 * sum(len(self._receipts.get(ancestor, {})) for ancestor in ancestors)
            )
            charged = {
                charged_receipt.receipt_hex()
                for ancestor in ancestors
                for charged_receipt in self._receipts.get(ancestor, {}).values()
            }
            if any(
                phase.receipt_hex() in charged
                for phase in _receipt_graph(receipt, scope=scope)
            ):
                raise ValueError(
                    "NativePreparationLedger.history must name a completed earlier request; "
                    "got a receipt already charged to this request."
                )
            if completed_source is not None and seal is not None:
                entry = self._history_sources.get(id(completed_source))
                if entry is None or entry[0]() is not completed_source:
                    self._retain_storage(scope, (), seal.metadata_bytes + 512)
                    self._history_sources[id(completed_source)] = (
                        weakref.ref(completed_source),
                        seal,
                    )
            if already_recorded:
                return
            # One 256-byte entry, a 64-byte slot plus two 16-byte references per
            # ancestry index, and a 256-byte header for each new scope mapping.
            new_maps = sum(ancestor not in self._history for ancestor in ancestors)
            self._retain_storage(
                scope, (receipt,), 256 + 96 * len(ancestors) + 256 * new_maps
            )
            # One unit retains the identity; no historical work, query or time is charged.
            scope.import_preparation(work=1, geometry_queries=0, elapsed_seconds=0.0)
            for ancestor in ancestors:
                self._history.setdefault(ancestor, {})[identity] = receipt

    def retain_receipt_snapshot(
        self, scope: NativeExecutionScope, snapshot: _NativeReceiptSnapshot, /,
        *, preparation_outside_scope: bool = False,
    ) -> None:
        """Admit successful or abandoned snapshots at a full-validation boundary."""
        receipt, graph = snapshot.receipt, snapshot.graph
        if graph is not None and self._receipt_graphs.get(id(receipt)) is graph:
            return
        # Only pre-root construction time is imported. History snapshots already
        # execute inside this root's elapsed clock. Neither repeats validation.
        scope.import_preparation(
            work=snapshot.work,
            geometry_queries=0,
            elapsed_seconds=snapshot.seconds if preparation_outside_scope else 0.0,
        )
        if graph is None:
            # Mutable/cold fallbacks still pay the abandoned metadata traversal.
            with scope.batch(0, bytes_upper=snapshot.bytes_upper):
                pass
            return
        self._retain_storage(scope, (receipt,), snapshot.bytes_upper)
        graph.require_current()
        self._receipt_graphs[id(receipt)] = graph

    def completed_coordinate_source_is_authenticated(
        self,
        scope: NativeExecutionScope,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract | None,
        paid_receipt: NativeExecutionRecord,
        earning_receipt: NativeExecutionRecord,
        budget: CoordinateEnclosureBudget,
        /,
    ) -> bool:
        """Authenticate historical numerical input, never import its work or proof."""
        _require_native_source_scope(scope)
        ticket = budget.ticket
        if (
            not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket))
            or ticket.scope is not scope
        ):
            raise ValueError("Coordinate history must use its actual receiving budget.")
        historical = False
        for ancestor in _ancestors(scope):
            for receipt in self._history.get(ancestor, {}).values():
                budget.reserve(1)
                if receipt is paid_receipt:
                    historical = True
                    break
            if historical:
                break
        if not historical:
            return False
        for receipt in _phase_chain(paid_receipt):
            budget.reserve(1)
            if receipt is earning_receipt:
                break
        else:
            return False
        for source_ref, _ in self._history_sources.values():
            budget.reserve(1)
            source = source_ref()
            if (
                source is not None
                and source.mesh is mesh
                and source.geometry is geometry
                and source.coordinate_contract is coordinate_contract
                and source.execution_evidence is paid_receipt
            ):
                _require_native_completed_source(source, paid_receipt, budget=budget)
                return True
        return False

    def _refuse_history(
        self, scope: NativeExecutionScope, receipt: NativeExecutionRecord, /
    ) -> None:
        with NativeHostStorageWorkspace(scope) as scratch:
            base = 1024 + 512 * (len(self._retained_owners) + self._retention_edges)
            scratch.set_bound(base)
            history = []
            for ancestor in _ancestors(scope):
                scope.charge(work=2)
                scratch.set_bound(base + 512 * (len(history) + 1))
                history.append(self._history.get(ancestor))
            if not any(history):
                return
            identities: set[str] = set()
            for phase in _phase_chain(receipt):
                scope.charge(work=33)
                scratch.set_bound(base + 512 * (len(history) + len(identities) + 1))
                identities.add(phase.receipt_hex())
            historical: set[str] = set()
            for entries in history:
                if entries:
                    scope.charge(work=len(entries))
                    for previous in entries.values():
                        for phase in _receipt_graph(previous, scope=scope):
                            historical.add(phase.receipt_hex())
            scope.charge(work=min(len(identities), len(historical)))
            if identities.intersection(historical):
                raise ValueError(
                    "NativePreparationLedger.import must charge this request's own preparation; "
                    "got a completed earlier request recorded as history."
                )

    def _retain_receipts(
        self,
        scope: NativeExecutionScope,
        receipts: list[NativeExecutionRecord],
        /,
        *,
        import_work: bool,
    ) -> None:
        if not receipts:
            return
        with NativeHostStorageWorkspace(scope) as scratch:
            scratch.set_bound(1024 + 512 * len(receipts))
            scope.charge(work=24 * len(receipts))
            overlaps: list[NativePreparationOverlap] = []
            work = queries = 0
            seconds = 0.0
            prepaid_seconds = 0.0
            for receipt in receipts:
                prepaid = self._overlaps.get(scope, {}).get(id(receipt))
                if prepaid is not None:
                    prepaid_seconds += prepaid.imported_elapsed_seconds
                phase_work = int(np.asarray(receipt.work_counters.construction_work))
                phase_queries = int(np.asarray(receipt.work_counters.source_queries))
                phase_seconds = float(np.asarray(receipt.elapsed_seconds))
                if import_work:
                    phase_seconds += float(np.asarray(receipt.prior_elapsed_seconds))
                    if receipt.preparation_evidence is None:
                        phase_work += int(np.asarray(receipt.source_preparation_work_units))
                        phase_queries += int(
                            np.asarray(receipt.source_preparation_geometry_queries)
                        )
                        phase_seconds += float(np.asarray(receipt.preparation_seconds))
                    work += phase_work
                    queries += phase_queries
                    seconds += phase_seconds
                    overlaps.append(
                        NativePreparationOverlap(
                            work=phase_work,
                            geometry_queries=phase_queries,
                            external_work=phase_work,
                            external_geometry_queries=phase_queries,
                            imported_elapsed_seconds=phase_seconds,
                        )
                    )
                else:
                    overlaps.append(
                        NativePreparationOverlap(
                            work=phase_work,
                            geometry_queries=phase_queries,
                            external_work=int(np.asarray(receipt.externally_charged_work)),
                            external_geometry_queries=int(
                                np.asarray(receipt.externally_charged_geometry_queries)
                            ),
                            native_primitive_queries=int(
                                np.asarray(receipt.native_primitive_queries)
                            ),
                            live_elapsed_seconds=phase_seconds,
                        )
                    )
            ancestors = tuple(_ancestors(scope))
            # Reuse the host profile's 256-byte container and 16-byte reference
            # slots, plus seven 8-byte scalar payloads, rounded to 256-byte blocks.
            overlap_bytes = declared_object_storage_bytes(_OVERLAP_LAYOUT) + 8 * len(
                _OVERLAP_LAYOUT
            )
            overlap_upper = 256 * ((overlap_bytes + 255) // 256)
            # Each ancestry index has a 64-byte mapping slot and two 16-byte
            # references. Each newly created scope mapping has a 256-byte header.
            new_maps = sum(
                (ancestor not in self._receipts)
                + (ancestor not in self._overlaps)
                + (ancestor not in self._imports)
                for ancestor in ancestors
            )
            metadata = (
                len(receipts)
                * (256 + overlap_upper + 192 * len(ancestors))
                + 256 * new_maps
            )
            self._retain_storage(scope, tuple(receipts), metadata)
            # One native admission through the active target or scope allowance:
            # an oversize import is a typed resource refusal with refusal evidence,
            # never a child-ticket estimate defect. One unit per retained identity.
            scope.import_preparation(
                work=work + len(receipts), geometry_queries=queries,
                elapsed_seconds=max(0.0, seconds - prepaid_seconds),
            )
            for receipt, overlap in zip(receipts, overlaps, strict=True):
                for ancestor in ancestors:
                    self._receipts.setdefault(ancestor, {})[id(receipt)] = receipt
                    self._overlaps.setdefault(ancestor, {})[id(receipt)] = overlap
                    self._imports.setdefault(ancestor, {})[id(receipt)] = (
                        receipt if import_work else None
                    )

    def _retain_storage(
        self,
        scope: NativeExecutionScope,
        owners: tuple[NativePreparedOwner | NativeExecutionRecord | str, ...],
        metadata_bytes: int,
        /,
    ) -> None:
        """Charge each retained graph increment to its active phase until root closure."""
        storage = scope.host_workspace().open()
        declared_upper = owner_indices = pending_slots = largest_fanout = 0
        pending_row = declared_object_storage_bytes((None, False, False, False)) + 16

        def admit_owner(owner: object) -> None:
            nonlocal declared_upper, owner_indices, pending_slots, largest_fanout
            scope.charge(work=2)
            children = retained_host_storage_fanout_upper(owner)
            declared_upper += declared_object_storage_bytes(owner)
            if isinstance(owner, Array):
                declared_upper += owner.nbytes
            elif isinstance(owner, np.ndarray) and owner.base is None:
                declared_upper += owner.nbytes
            elif isinstance(owner, ctypes.Array) and not is_native_host_buffer(owner):
                declared_upper += ctypes.sizeof(owner)
            owner_indices += 128 + declared_object_storage_bytes(id(owner))
            pending_slots += children + int(children != 0)
            largest_fanout = max(largest_fanout, children)
            storage.set_bound(
                metadata_bytes + declared_upper + 1024 + owner_indices
                + pending_row * pending_slots + 512 * largest_fanout
            )

        try:
            storage.set_bound(metadata_bytes + 1024)
            retained, declared, _, _ = retained_host_storage_profile(
                owners, self._retained_owners, is_native_host_buffer,
                admit_owner=admit_owner,
            )
            storage.set_bound(declared + metadata_bytes + 128 * len(retained))
            self._retained_owners.update(retained)
            self._retention_edges += pending_slots
            self._storage.append(storage)
        except BaseException:
            storage.close()
            raise

    def overlap(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord | None,
        /,
    ) -> NativePreparationOverlap:
        """Snapshot only the matched source ancestry before the request attachment closes."""
        # A root snapshot may be taken while a nested phase of the same request
        # is active; that phase is the only scope that can admit host storage.
        active = current_native_execution_scope()
        payer = scope if active is None else active
        if payer is not scope and scope not in _ancestors(payer):
            raise ValueError("Preparation overlap must be paid by its own live request.")
        self._retain_storage(payer, (), 512)
        payer.charge(work=16)
        work = queries = external_work = external_queries = primitive_queries = 0
        live_seconds = imported_seconds = 0.0
        current = receipt
        represented = self._overlaps.get(scope, {})
        while current is not None:
            payer.charge(work=8)
            overlap = represented.get(id(current))
            if overlap is not None:
                work += overlap.work
                queries += overlap.geometry_queries
                external_work += overlap.external_work
                external_queries += overlap.external_geometry_queries
                primitive_queries += overlap.native_primitive_queries
                live_seconds += overlap.live_elapsed_seconds
                imported_seconds += overlap.imported_elapsed_seconds
            current = current.preparation_evidence
        return NativePreparationOverlap(
            work=work,
            geometry_queries=queries,
            external_work=external_work,
            external_geometry_queries=external_queries,
            native_primitive_queries=primitive_queries,
            live_elapsed_seconds=live_seconds,
            imported_elapsed_seconds=imported_seconds,
        )

    def source_is_active(self, scope: NativeExecutionScope, source_id: str, /) -> bool:
        """Resolve a source registration or retained object through live ancestry."""
        for ancestor in _ancestors(scope):
            if source_id in self._sources.get(ancestor, ()):
                return True
            for receipt in self._receipts.get(ancestor, {}).values():
                scope.charge(work=1)
                if receipt.owner_id == source_id:
                    return True
        return False

    def remember_source(self, scope: NativeExecutionScope, source_id: str, /) -> None:
        """Register a source prepared under this request's allowance."""
        if self.source_is_active(scope, source_id):
            return
        self._retain_storage(scope, (source_id,), 256)
        self._sources.setdefault(scope, set()).add(source_id)


def native_preparation_ledger(scope: NativeExecutionScope, /) -> NativePreparationLedger:
    """Return the root scope's single preparation attachment."""
    attachment = scope.preparation_ledger(NativePreparationLedger)
    match attachment:
        case NativePreparationLedger():
            return attachment
        case _:
            raise TypeError(
                "NativeExecutionScope.preparation_ledger must contain NativePreparationLedger; "
                f"got {type(attachment).__name__}."
            )


def _native_source_root(scope: NativeExecutionScope, /) -> NativeExecutionScope:
    root = scope
    while root.parent is not None:
        root = root.parent
    return root


def _require_native_source_scope(scope: NativeExecutionScope, /) -> NativeExecutionScope:
    actual = current_native_execution_scope()
    if (
        actual is None
        or scope.evidence is not None
        or _native_source_root(actual) is not _native_source_root(scope)
    ):
        raise ValueError(
            "Native source publication requires the actual live request root."
        )
    scope._active_native_library("native_source_publication")
    return _native_source_root(scope)


@contextmanager
def _native_source_budget(
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> Iterator[CoordinateEnclosureBudget]:
    from ..discretization._coordinate_budget import CoordinateEnclosureBudget

    root = _require_native_source_scope(scope)
    if budget is not None:
        ticket = budget.ticket
        if (
            not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket))
            or ticket.settled
            or _native_source_root(ticket.scope) is not root
        ):
            raise ValueError(
                "Native source authentication must use this request's live allowance."
            )
        yield budget
        return
    # This is a view of the actual remaining allowance, not a renewed quota.
    with scope.ticket_view() as ticket:
        yield CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=ticket.remaining_work_units,
            maximum_memory_bytes=ticket.remaining_managed_bytes,
        )


_SOURCE_CONSTRUCTOR_NAMES = (
    "__init__",
    "__new__",
    "from_stored_fields",
    "__dataclass_fields__",
)
_SOURCE_AUTH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class _NativeSourceSeal:
    """Exact immutable owners and complete mutable bytes; not scientific proof."""

    source_type: type
    names: tuple[str, ...]
    values: tuple[object, ...]
    owners: OwnerFields
    containers: ContainerOwners
    immutable_arrays: tuple[Array, ...]
    immutable_host_arrays: tuple[tuple[np.ndarray, tuple[object, ...]], ...]
    mutable_arrays: tuple[np.ndarray, ...]
    mutable_layouts: tuple[tuple[np.dtype, tuple[int, ...]], ...]
    constructors: tuple[tuple[type, tuple[object, ...]], ...]
    content: str
    metadata_bytes: int


def _native_source_array_scan_cost(arrays: tuple[np.ndarray, ...], /) -> tuple[int, int]:
    """Share the live and prepaid complete mutable-byte scan admission."""
    return (
        sum(1 + array.size for array in arrays),
        2
        * min(
            _SOURCE_AUTH_CHUNK_BYTES,
            max((array.nbytes for array in arrays), default=0),
        )
        + sum(1024 * (2 + array.nbytes // _SOURCE_AUTH_CHUNK_BYTES) for array in arrays),
    )


def _require_native_source_array_owners(seal: _NativeSourceSeal, /) -> None:
    # The recorded top fields, owner fields and container edges bind these exact
    # objects. JAX values cannot mutate, but their buffers can be deleted.
    if any(
        not isinstance(array, Array) or array.is_deleted()
        for array in seal.immutable_arrays
    ):
        raise ValueError("Native source immutable array owners must remain live.")
    if any(
        immutable_host_array_layout(array) != layout
        for array, layout in seal.immutable_host_arrays
    ):
        raise ValueError("Native source immutable host array dtype, shape or backing changed.")
    # Refuse layout changes before a prepaid scan can exceed its admitted work
    # or scratch. Readonly flags are deliberately not an immutability claim.
    if any(
        type(array) is not np.ndarray or array.dtype != dtype or array.shape != shape
        for array, (dtype, shape) in zip(
            seal.mutable_arrays, seal.mutable_layouts, strict=True
        )
    ):
        raise ValueError("Native source mutable array dtype or shape changed.")


def _native_source_array_content(
    arrays: tuple[np.ndarray, ...], budget: CoordinateEnclosureBudget, /
) -> str:
    from .._fingerprint import logical_array_value_collection_digest

    with budget.temporary_scope():
        # Hash every logical host byte, including readonly NumPy views, with
        # the same bounded chunk storage admitted for the final scan.
        budget.reserve(*_native_source_array_scan_cost(arrays))
        return logical_array_value_collection_digest(
            {str(index): array for index, array in enumerate(arrays)},
            maximum_chunk_bytes=_SOURCE_AUTH_CHUNK_BYTES,
        )


def _native_source_seal(
    source: CellMeshingResult, budget: CoordinateEnclosureBudget, /
) -> _NativeSourceSeal:
    from ..discretization._coordinate_operands import coordinate_input_owner_fields
    from ._result import CellMeshingResult

    if type(source) is not CellMeshingResult:
        raise TypeError(
            "Native source publication requires the actual typed CellMeshingResult."
        )
    budget.reserve(1, 4096 + 256 * len(fields(source)))
    names = tuple(
        member.name for member in fields(source) if member.name != "execution_evidence"
    )
    values = tuple(getattr(source, name) for name in names)
    arrays: list[Array | np.ndarray] = []
    containers: list[tuple[list[object], tuple[object, ...]]] = []
    owners = coordinate_input_owner_fields(
        values, budget, arrays=arrays, containers=containers
    )
    if owners is None or any(owner is source for owner, _, _, _ in owners):
        raise ValueError(
            "Native source fields must have closed, inspectable, non-self-retaining owners."
        )
    metadata = (
        8192
        + 256 * len(names)
        + 1024 * len(arrays)
        + sum(1024 + 128 * len(member_names) for _, _, member_names, _ in owners)
        + sum(512 + 32 * len(children) for _, children in containers)
    )
    budget.reserve(len(owners) + len(arrays), metadata)
    owner_types = {type(source), *(owner_type for _, owner_type, _, _ in owners)}
    constructors = tuple(
        (
            owner_type,
            tuple(
                getattr_static(owner_type, name, None)
                for name in _SOURCE_CONSTRUCTOR_NAMES
            ),
        )
        for owner_type in owner_types
    )
    immutable_arrays = tuple(array for array in arrays if isinstance(array, Array))
    immutable_host_arrays = []
    mutable_arrays_list = []
    for array in arrays:
        if type(array) is np.ndarray:
            layout = immutable_host_array_layout(array)
            if layout is None:
                mutable_arrays_list.append(array)
            else:
                immutable_host_arrays.append((array, layout))
    mutable_arrays = tuple(mutable_arrays_list)
    seal = _NativeSourceSeal(
        type(source),
        names,
        values,
        owners,
        tuple(containers),
        immutable_arrays,
        tuple(immutable_host_arrays),
        mutable_arrays,
        tuple((array.dtype, array.shape) for array in mutable_arrays),
        constructors,
        _native_source_array_content(mutable_arrays, budget),
        metadata,
    )
    budget.reserve(
        len(immutable_arrays) + len(immutable_host_arrays) + len(mutable_arrays)
    )
    _require_native_source_array_owners(seal)
    return seal


def _require_native_source_top_fields(
    source: CellMeshingResult, seal: _NativeSourceSeal, /
) -> None:
    if type(source) is not seal.source_type or any(
        getattr(source, name, None) is not value
        for name, value in zip(seal.names, seal.values, strict=True)
    ):
        raise ValueError(
            "Native completed source must retain its actual scientific field owners."
        )
    if any(
        getattr_static(owner_type, name, None) is not original
        for owner_type, originals in seal.constructors
        for name, original in zip(_SOURCE_CONSTRUCTOR_NAMES, originals, strict=True)
    ):
        raise ValueError("Native source constructors changed after publication.")


def _require_native_source_seal(
    source: CellMeshingResult,
    seal: _NativeSourceSeal,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    from ..discretization._coordinate_operands import (
        require_coordinate_input_containers,
        require_coordinate_input_owners,
    )

    budget.reserve(1 + len(seal.names) + 4 * len(seal.constructors))
    _require_native_source_top_fields(source, seal)
    budget.reserve(
        len(seal.immutable_arrays)
        + len(seal.immutable_host_arrays)
        + len(seal.mutable_arrays)
    )
    _require_native_source_array_owners(seal)
    if (
        not require_coordinate_input_owners(seal.owners, budget)
        or not require_coordinate_input_containers(seal.containers, budget)
        or _native_source_array_content(seal.mutable_arrays, budget) != seal.content
    ):
        raise ValueError(
            "Native source fields or complete mutable bytes changed after publication."
        )


def stage_native_source_publication(
    scope: NativeExecutionScope,
    source: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    """Stage actual owners after the caller's complete scientific validation.

    This seal authenticates publication ownership, not a mathematical verdict.
    Decoders must independently validate before calling this boundary.
    """
    root = _require_native_source_scope(scope)
    with _native_source_budget(scope, budget) as allowance:
        seal = _native_source_seal(source, allowance)
    ledger = native_preparation_ledger(root)
    ledger._retain_storage(scope, (), seal.metadata_bytes + 512)
    ledger._publications[id(source)] = (weakref.ref(source), seal)


_completed_native_sources: dict[int, NativeCompletedSourceIssue] = {}
# The same completed issues indexed by their exact issued receipt object.
_completed_native_receipts: dict[int, NativeCompletedSourceIssue] = {}


def _require_prepaid_source_seal(
    source: CellMeshingResult, seal: _NativeSourceSeal, /
) -> None:
    """Perform the bounded final scan whose work/storage were admitted live."""
    from .._fingerprint import logical_array_value_collection_digest
    from ..discretization._coordinate_operands import (
        _coordinate_input_container_matches,
        _coordinate_input_owner_matches,
    )

    _require_native_source_top_fields(source, seal)
    if any(not _coordinate_input_owner_matches(*owner) for owner in seal.owners):
        raise ValueError("Native completed source field owners changed.")
    if any(
        not _coordinate_input_container_matches(*container)
        for container in seal.containers
    ):
        raise ValueError("Native completed source containers changed.")
    _require_native_source_array_owners(seal)
    content = logical_array_value_collection_digest(
        {str(index): array for index, array in enumerate(seal.mutable_arrays)},
        maximum_chunk_bytes=_SOURCE_AUTH_CHUNK_BYTES,
    )
    if content != seal.content:
        raise ValueError("Native completed source complete bytes changed.")


@dataclass(frozen=True, slots=True)
class _NativeReceiptPhase:
    """Authenticated host numerics beside the exact dynamic receipt owners."""

    receipt: NativeExecutionRecord
    owners: OwnerFields
    arrays: tuple[Array, ...]
    evidence: NativeExecutionEvidence
    owner_id: str | None
    receipt_id: str
    imported_ids: tuple[str, ...]
    source_work: int
    source_queries: int
    preparation_seconds: float
    preparation: NativeExecutionRecord | None
    consumer: NativeExecutionRecord | None
    history: tuple[NativeExecutionRecord, ...]

    @property
    def authentication_work(self) -> int:
        return sum(1 + len(names) for _, _, names, _ in self.owners) + len(self.arrays)

    def record(self, positions: dict[int, int], /) -> NativeExecutionPhaseRecord:
        return _phase_record(
            self.evidence,
            owner_id=self.owner_id,
            source_work=self.source_work,
            source_queries=self.source_queries,
            preparation_seconds=self.preparation_seconds,
            preparation_evidence=(
                None if self.preparation is None else positions[id(self.preparation)]
            ),
            consumer_evidence=(
                None if self.consumer is None else positions[id(self.consumer)]
            ),
            history_evidence=[positions[id(item)] for item in self.history],
            receipt_id=self.receipt_id,
            imported_preparation_ids=self.imported_ids,
        )


@dataclass(frozen=True, slots=True)
class _NativeReceiptGraph:
    """Private, process-local preparation, never archive or numerical authority.

    Phases retain the canonical root-first ordering and previously validated host
    numerics. Reuse still authenticates every exact owner edge and live immutable
    numerical leaf; a frozen Python wrapper or a readonly NumPy flag is not enough.
    """

    phases: tuple[_NativeReceiptPhase, ...]

    @property
    def graph_id(self) -> str:
        """Canonical ended-graph digest, derived only when explicitly read."""
        positions = {id(phase.receipt): index for index, phase in enumerate(self.phases)}
        return canonical_fingerprint(
            {"root": 0, "phases": [phase.record(positions) for phase in self.phases]}
        )

    @property
    def authentication_work(self) -> int:
        return sum(phase.authentication_work for phase in self.phases)

    def require_current(self) -> None:
        from ..discretization._coordinate_operands import _coordinate_input_owner_matches

        for phase in self.phases:
            if any(
                not _coordinate_input_owner_matches(*owner) for owner in phase.owners
            ) or any(array.is_deleted() for array in phase.arrays):
                raise ValueError("Native completed source receipt graph owners changed.")


@dataclass(frozen=True, slots=True)
class _NativeReceiptSnapshot:
    """Measured provisional metadata; it never supplies numerical authority."""

    receipt: NativeExecutionRecord
    graph: _NativeReceiptGraph | None
    work: int
    bytes_upper: int
    seconds: float


def _native_receipt_phase(receipt: NativeExecutionRecord, /) -> _NativeReceiptPhase | None:
    owners = []
    arrays: list[Array] = []
    for owner in (receipt, receipt.work_counters, receipt.memory_counters):
        names = tuple(member.name for member in fields(owner))
        values = tuple(getattr(owner, name) for name in names)
        owners.append((owner, type(owner), names, values))
        for value in values:
            if isinstance(value, Array):
                if value.is_deleted():
                    raise ValueError("Native receipt immutable array owners must remain live.")
                arrays.append(value)
            elif isinstance(value, np.ndarray):
                # Mutable/decoded numerical receipts keep the full validation path.
                return None
    return _NativeReceiptPhase(
        receipt, tuple(owners), tuple(arrays), receipt.native_evidence(),
        None if receipt.owner_id is None else str(receipt.owner_id),
        receipt.receipt_hex(), receipt.imported_preparation_hex(),
        int(np.asarray(receipt.source_preparation_work_units)),
        int(np.asarray(receipt.source_preparation_geometry_queries)),
        float(np.asarray(receipt.preparation_seconds)),
        receipt.preparation_evidence, receipt.consumer_evidence, receipt.history_evidence,
    )


def _issued_native_receipt_graph(
    receipt: NativeExecutionRecord, /
) -> _NativeReceiptGraph | None:
    """Bind an issued receipt to its already authenticated immutable graph.

    Only the exact current-process completed publication owner of this exact
    receipt object, with its unchanged receipt ID and live immutable leaves,
    supplies authority; a cold, decoded or unissued receipt returns ``None``.
    """
    issue = _completed_native_receipts.get(id(receipt))
    if (
        issue is None
        or issue._process_id != os.getpid()
        or issue._receipt is not receipt
        or issue._receipt_graph is None
        or issue._receipt_graph.phases[0].receipt is not receipt
    ):
        return None
    graph = issue._receipt_graph
    graph.require_current()
    if receipt.receipt_hex() != issue._receipt_id:
        raise ValueError("Native completed source receipt bytes changed after publication.")
    return graph


def _prepare_native_receipt_graph(
    receipt: NativeExecutionRecord, /,
    *, snapshots: list[_NativeReceiptSnapshot] | None = None,
) -> _NativeReceiptGraph | None:
    """Retain the already fully checked graph without changing any public leaf."""
    started = monotonic()
    work = attempts = 0
    receipts = [receipt]
    seen = {id(receipt)}
    phases: list[_NativeReceiptPhase] = []
    for current in receipts:
        attempts += 1
        work += 128
        phase = _native_receipt_phase(current)
        if phase is None:
            if snapshots is not None:
                snapshots.append(_NativeReceiptSnapshot(
                    receipt, None, work, 8192 * attempts, monotonic() - started
                ))
            return None
        phases.append(phase)
        work += 2 * phase.authentication_work + 32 * len(phase.imported_ids)
        for child in (phase.preparation, phase.consumer, *phase.history):
            if child is not None and id(child) not in seen:
                seen.add(id(child))
                receipts.append(child)
    graph = _NativeReceiptGraph(tuple(phases))
    if snapshots is not None:
        snapshots.append(_NativeReceiptSnapshot(
            receipt, graph, work, 8192 * attempts, monotonic() - started
        ))
    return graph


def _prepare_native_receipt_graph_in_scope(
    receipt: NativeExecutionRecord, scope: NativeExecutionScope, /,
    *, retain: Callable[[int], None],
) -> tuple[_NativeReceiptGraph | None, int]:
    """Build immutable receipt metadata with admission before each owner visit."""
    with NativeHostStorageWorkspace(scope) as storage:
        storage.set_bound(1024)
        scope.charge(work=1)
        receipts = [receipt]
        seen = {id(receipt)}
        phases: list[_NativeReceiptPhase] = []
        # Measured retained graph metadata (tracemalloc) is <= 17 KiB for the
        # graph header and first phase, <= 3.7 KiB per further phase, plus
        # one <= 121-byte hex string per imported ID. Admit the import
        # capacity before reading the phase's numerical leaves.
        metadata = 16384
        for current in receipts:
            field_count = (
                len(type(current).__dataclass_fields__)
                + len(type(current.work_counters).__dataclass_fields__)
                + len(type(current.memory_counters).__dataclass_fields__)
            )
            history_count = len(current.history_evidence)
            phase_metadata = 8192 + 256 * NATIVE_EXECUTION_IMPORT_CAPACITY
            storage.set_bound(metadata + phase_metadata + 128 * (history_count + 3))
            # The schema tariff is admitted and consumed before scalar reads or
            # metadata construction; it bounds the owner and imported-ID visit.
            upper = 128 + 2 * (3 + 2 * field_count) + 32 * NATIVE_EXECUTION_IMPORT_CAPACITY
            with scope.batch(upper) as ticket:
                phase = _native_receipt_phase(current)
                ticket.consume(upper)
            if phase is None:
                return None, 0
            # The actual imported-ID count is now host metadata.
            metadata += 8192 + 256 * len(phase.imported_ids) + 128 * (history_count + 3)
            storage.set_bound(metadata)
            scope.charge(work=3 + history_count)
            phases.append(phase)
            for child in (phase.preparation, phase.consumer, *phase.history):
                if child is not None and id(child) not in seen:
                    seen.add(id(child))
                    receipts.append(child)
        scope.charge(work=len(phases))
        graph = _NativeReceiptGraph(tuple(phases))
        retain(metadata)
        return graph, metadata


def _extend_native_receipt_graph(
    receipt: NativeExecutionRecord, graph: _NativeReceiptGraph, /
) -> _NativeReceiptGraph | None:
    """Validate only the appended phase; authenticate, never revalidate, history."""
    graph.require_current()
    previous = graph.phases[0]
    if (
        type(receipt) is not NativeExecutionRecord
        or receipt.preparation_evidence is not previous.receipt
        or receipt.consumer_evidence is not None
        and receipt.consumer_evidence is not previous.receipt
        or receipt.history_evidence != ()
        or receipt.owner_id != previous.owner_id
    ):
        raise ValueError("Native continuation receipt must extend its exact prepared graph.")
    receipt._require_phase_valid()
    if (
        int(np.asarray(receipt.source_preparation_work_units))
        != previous.source_work + previous.evidence.work_counters.construction_work
        or int(np.asarray(receipt.source_preparation_geometry_queries))
        != previous.source_queries + previous.evidence.work_counters.source_queries
        or float(np.asarray(receipt.preparation_seconds))
        != previous.evidence.elapsed_seconds
        + previous.evidence.prior_elapsed_seconds
        + previous.preparation_seconds
    ):
        raise ValueError("Native continuation receipt preparation aggregates changed.")
    receipt._require_receipt_identity()
    phase = _native_receipt_phase(receipt)
    if phase is None:
        return None
    # Only the paid root is referenced by the appended phase, so prepending it
    # preserves the exact public BFS order, including shared historical nodes.
    return _NativeReceiptGraph((phase, *graph.phases))


@dataclass(frozen=True, slots=True)
class _NativeReceiptPrefix:
    """Exact admitted predecessor DAG for a newly published scientific target."""

    phases: tuple[_NativeReceiptPhase, ...]
    preparation: NativeExecutionRecord
    history: tuple[NativeExecutionRecord, ...]

    @property
    def authentication_work(self) -> int:
        return sum(phase.authentication_work for phase in self.phases)

    def require_current(self) -> None:
        _NativeReceiptGraph(self.phases).require_current()

    def authenticate_bindings(
        self, preparation: NativeExecutionRecord | None,
        consumer: NativeExecutionRecord | None,
        history: tuple[NativeExecutionRecord, ...], /
    ) -> frozenset[int]:
        self.require_current()
        if (
            preparation is not self.preparation or consumer is not None
            or len(history) != len(self.history)
            or any(actual is not expected for actual, expected in zip(history, self.history, strict=True))
        ):
            raise ValueError("Receipt construction must bind its exact admitted predecessors.")
        return frozenset(id(phase.receipt) for phase in self.phases)

    def extend(self, receipt: NativeExecutionRecord, /) -> _NativeReceiptGraph | None:
        self.require_current()
        if (
            receipt.preparation_evidence is not self.preparation
            or receipt.consumer_evidence is not None
            or len(receipt.history_evidence) != len(self.history)
            or any(
                actual is not expected
                for actual, expected in zip(
                    receipt.history_evidence, self.history, strict=True
                )
            )
        ):
            raise ValueError("Native publication must extend its exact prepared prefix.")
        receipt._require_graph_valid(
            frozenset(id(phase.receipt) for phase in self.phases)
        )
        phase = _native_receipt_phase(receipt)
        if phase is None:
            return None
        return _NativeReceiptGraph((phase, *self.phases))


def _native_receipt_prefix(
    ledger: NativePreparationLedger,
    imports: NativeExecutionImports | None,
    preparations: tuple[NativeExecutionRecord, ...],
    histories: tuple[NativeExecutionRecord, ...],
    budget: CoordinateEnclosureBudget,
) -> _NativeReceiptPrefix | None:
    if imports is None or imports.preparation is None:
        return None
    roots = (imports.preparation, *histories)
    phases: dict[int, _NativeReceiptPhase] = {}
    for root in roots:
        budget.reserve(1)
        graph = ledger._receipt_graphs.get(id(root))
        if graph is None or graph.phases[0].receipt is not root:
            return None
        budget.reserve(
            graph.authentication_work + len(graph.phases),
            128 * len(graph.phases),
        )
        graph.require_current()
        for phase in graph.phases:
            phases[id(phase.receipt)] = phase
    budget.reserve(len(preparations))
    if any(id(item) not in phases for item in preparations):
        return None
    budget.reserve(len(histories) * (1 + len(histories).bit_length()))
    histories = tuple(sorted(histories, key=lambda item: phases[id(item)].receipt_id))
    pending = [imports.preparation, *histories]
    seen: set[int] = set()
    ordered: list[_NativeReceiptPhase] = []
    for receipt in pending:
        budget.reserve(3 + len(receipt.history_evidence), 128)
        if id(receipt) in seen:
            continue
        phase = phases.get(id(receipt))
        if phase is None or phase.receipt is not receipt:
            return None
        seen.add(id(receipt))
        ordered.append(phase)
        pending.extend(
            child
            for child in (phase.preparation, phase.consumer, *phase.history)
            if child is not None
        )
    if seen != phases.keys():
        return None
    return _NativeReceiptPrefix(tuple(ordered), imports.preparation, histories)


@final
class NativeCompletedSourceIssue:
    """One preadmitted publication handover; never scientific proof authority."""

    def __init__(
        self,
        root: NativeExecutionScope,
        seal: _NativeSourceSeal,
        storage: NativeHostStorageWorkspace,
        owner_id: str,
        imports: NativeExecutionImports | None,
        overlap: NativePreparationOverlap | None,
        history: tuple[NativeExecutionRecord, ...],
        preparations: tuple[NativeExecutionRecord, ...],
        continuation: tuple[
            weakref.ReferenceType[CellMeshingResult], NativeExecutionRecord, str,
            _NativeReceiptGraph | None,
        ] | None = None,
        prefix: _NativeReceiptPrefix | None = None,
        *,
        staged: tuple[weakref.ReferenceType[NativePreparationLedger], int] | None = None,
    ) -> None:
        self._root = weakref.ref(root)
        self._seal: _NativeSourceSeal | None = seal
        self._storage: NativeHostStorageWorkspace | None = storage
        # A staged registry entry is covered by this token, never the ledger.
        self._staged = staged
        self._owner_id = owner_id
        self._imports = imports
        self._overlap = overlap
        self._history = tuple(item.receipt_hex() for item in history)
        self._history_evidence = history
        self._preparations = preparations
        self._continuation = continuation
        self._prefix = prefix
        self._source: weakref.ReferenceType[CellMeshingResult] | None = None
        self._source_identity: int | None = None
        # Keep the actual registry alive through module-global teardown. Closing
        # breaks this ownership edge as well as the weak source callback.
        self._registry: dict[int, NativeCompletedSourceIssue] | None = None
        self._receipt_registry: dict[int, NativeCompletedSourceIssue] | None = None
        self._release_metadata: Callable[[], object] | None = None
        self._receipt: NativeExecutionRecord | None = None
        self._receipt_id: str | None = None
        self._receipt_graph: _NativeReceiptGraph | None = None
        self._process_id = os.getpid()
        self._record_constructor_admitted = False

    @property
    def history_evidence(self) -> tuple[NativeExecutionRecord, ...]:
        """Return the actual history admitted before this publication's root ended."""
        if self._seal is None or self._source is not None:
            raise ValueError("Publication history is available only before completion.")
        return self._history_evidence

    def admit_record_constructor(self, scope: NativeExecutionScope, /) -> _NativeReceiptPrefix | None:
        """Prepay the actual ended record on this existing publication token."""
        if self._root() is not scope or scope.evidence is not None or self._storage is None:
            raise ValueError("Receipt construction admission requires its live issuing root.")
        if self._record_constructor_admitted:
            raise ValueError("Receipt construction must be admitted exactly once.")
        prefix = self._prefix
        if prefix is not None:
            scope.charge(work=len(prefix.phases))
            count = len(prefix.phases)
            work = 4096 + prefix.authentication_work + count
            chain = prefix.preparation
            while chain is not None:
                # Inspect once now; constructor history checks traverse this
                # actual preparation chain twice after scope end.
                scope.charge(work=1)
                work += 66
                chain = chain.preparation_evidence
            history_count = len(prefix.history)
            work += 2 * history_count * (33 + history_count.bit_length())
            # Only the new root phase is validated; its prefix is authenticated.
            validated, edges, widest = 1, 3 + history_count, history_count
        else:
            # Mutable receipt leaves cannot reuse immutable numerical metadata.
            # Their full constructor validation remains prepaid, never skipped.
            ledger = native_preparation_ledger(scope)
            root_count = len(self._preparations) + len(ledger._history.get(scope, {}))
            scope.charge(work=root_count)
            self._storage.set_bound(self._storage.bound + 512 * (root_count + 1))
            roots = (*self._preparations, *ledger._history.get(scope, {}).values())
            pending = list(roots)
            seen: set[int] = set()
            count = edges = widest = 0
            initial = self._storage.bound
            while pending:
                scope.charge(work=1)
                receipt = pending.pop()
                if id(receipt) in seen:
                    continue
                history_count = len(receipt.history_evidence)
                edges += history_count + 3
                widest = max(widest, history_count)
                self._storage.set_bound(initial + 512 * (len(seen) + edges + 1))
                scope.charge(work=history_count + 3)
                seen.add(id(receipt))
                count += 1
                pending.extend(
                    child for child in (
                        receipt.preparation_evidence, receipt.consumer_evidence,
                        *receipt.history_evidence,
                    ) if child is not None
                )
            work = 4096 * (2 * count + 1) + 66 * (count + 1) ** 2
            validated = count + 1
            # The new root's history names at most this root's recorded or
            # issued completed requests, all traversed above.
            root_history = max(len(self._history), len(ledger._history.get(scope, {})))
            edges += 3 + root_history
            widest = max(widest, root_history)
        # Constructor validation and canonical serialization coexist with the
        # retained source seal on this publication token.
        self._storage.set_bound(
            self._storage.bound
            + native_receipt_validation_bytes_upper(
                validated, edges, widest, len(self._owner_id)
            )
        )
        scope.charge(work=work)
        self._record_constructor_admitted = True
        return prefix

    def complete(self, source: CellMeshingResult, /) -> None:
        """Bind exactly one actual final wrapper to its successful ended root."""
        root = self._root()
        if (
            root is None
            or root.evidence is None
            or self._process_id != os.getpid()
            or current_native_execution_scope() is not None
        ):
            raise ValueError(
                "Native completed publication requires its actual ended root."
            )
        if self._seal is None or self._source is not None or self._storage is None:
            raise ValueError(
                "Native completed publication issue is closed or already completed."
            )
        try:
            prepared_graph = None
            if self._continuation is not None:
                previous_ref, paid_receipt, receipt_id, prepared_graph = self._continuation
                previous = previous_ref()
                if (
                    previous is None
                    or source.execution_evidence is None
                    or source.execution_evidence.preparation_evidence is not paid_receipt
                ):
                    raise ValueError(
                        "Native continuation must retain its actual paid source receipt."
                    )
                _, current_id, current_graph = _native_completed_source_entry(
                    previous, paid_receipt
                )
                if current_id != receipt_id or (
                    prepared_graph is not None and current_graph is not prepared_graph
                ):
                    raise ValueError("Native completed source receipt preparation changed.")
                if prepared_graph is None and paid_receipt.receipt_hex() != receipt_id:
                    raise ValueError(
                        "Native completed source receipt bytes changed after publication."
                    )
            receipt = source.execution_evidence
            if receipt is None or receipt.owner_id != self._owner_id:
                raise ValueError(
                    "Native completed publication receipt must bind its request owner."
                )
            if receipt.preparation_evidence is not None and not any(
                receipt.preparation_evidence is item for item in self._preparations
            ):
                raise ValueError(
                    "Native completed publication preparation must belong to its actual root."
                )
            if self._prefix is not None:
                receipt_graph = self._prefix.extend(receipt)
            elif prepared_graph is None:
                receipt.require_valid()
                receipt_graph = _prepare_native_receipt_graph(receipt)
            else:
                receipt_graph = _extend_native_receipt_graph(receipt, prepared_graph)
            expected = root.completed_evidence()
            if self._imports is not None:
                expected = self._imports.evidence()
                self._imports.require_matches(expected, receipt.preparation_evidence)
            elif self._overlap is not None:
                expected = self._overlap.normalize(expected)
            if receipt.native_evidence() != expected:
                raise ValueError(
                    "Native completed publication counters must match its exact ended root evidence."
                )
            if set(self._history) != {
                item.receipt_hex() for item in receipt.history_evidence
            }:
                raise ValueError(
                    "Native completed publication must preserve its completed request history."
                )
            _require_prepaid_source_seal(source, self._seal)
            self._registry = _completed_native_sources
            self._source_identity = id(source)
            self._source = weakref.ref(source, lambda _: self.close())
            self._receipt = receipt
            self._receipt_id = receipt.receipt_hex()
            self._receipt_graph = receipt_graph
            self._preparations = ()
            self._imports = None
            self._history = ()
            self._history_evidence = ()
            self._continuation = None
            self._prefix = None
            self._registry[id(source)] = self
            self._receipt_registry = _completed_native_receipts
            self._receipt_registry[id(receipt)] = self
            finalizer = self._storage.retain_source_metadata(source)
            invocation = finalizer.peek()
            assert invocation is not None
            _, callback, args, kwargs = invocation
            self._release_metadata = partial(callback, *args, **kwargs)
            # The issue's weak callback owns release now, including after
            # weakref.finalize has disabled callbacks at interpreter shutdown.
            finalizer.detach()
            self._storage = None
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release failed/unclaimed metadata and remove weak publication authority."""
        identity = self._source_identity
        registry, self._registry = self._registry, None
        if identity is not None and registry is not None and registry.get(identity) is self:
            del registry[identity]
        receipts, self._receipt_registry = self._receipt_registry, None
        if (
            receipts is not None
            and self._receipt is not None
            and receipts.get(id(self._receipt)) is self
        ):
            del receipts[id(self._receipt)]
        staged, self._staged = self._staged, None
        if staged is not None:
            # Releasing this token also ends the staged entry it covered.
            ledger = staged[0]()
            entry = None if ledger is None else ledger._publications.get(staged[1])
            if ledger is not None and entry is not None and entry[1] is self._seal:
                del ledger._publications[staged[1]]
        self._source_identity = None
        self._source = None
        self._receipt = None
        self._receipt_id = None
        self._receipt_graph = None
        self._prefix = None
        self._preparations = ()
        self._seal = None
        self._imports = None
        self._history = ()
        self._history_evidence = ()
        self._continuation = None
        release, self._release_metadata = self._release_metadata, None
        if release is not None:
            release()
        if self._storage is not None:
            self._storage.close()
            self._storage = None


def prepare_native_completed_source_issue(
    scope: NativeExecutionScope,
    source: CellMeshingResult,
    /,
    *,
    owner_id: str | None = None,
    imports: NativeExecutionImports | None = None,
    overlap: NativePreparationOverlap | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> NativeCompletedSourceIssue:
    """Prepare all authentication and metadata on the actual live root."""
    return _prepare_native_completed_source_issue(
        scope,
        source,
        stage=False,
        owner_id=owner_id,
        imports=imports,
        overlap=overlap,
        budget=budget,
    )


def stage_native_completed_source_issue(
    scope: NativeExecutionScope,
    source: CellMeshingResult,
    /,
    *,
    owner_id: str | None = None,
    imports: NativeExecutionImports | None = None,
    overlap: NativePreparationOverlap | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> NativeCompletedSourceIssue:
    """Atomically stage and prepare an already scientifically validated source.

    There is no caller mutation window between these two operations. Authenticate
    current mutable bytes once here, then always rescan them at final completion.
    Separated staging/prepare callers must use their independently checked path.
    """
    return _prepare_native_completed_source_issue(
        scope,
        source,
        stage=True,
        owner_id=owner_id,
        imports=imports,
        overlap=overlap,
        budget=budget,
    )


def continue_native_completed_source_issue(
    scope: NativeExecutionScope,
    source: CellMeshingResult,
    paid_receipt: NativeExecutionRecord,
    /,
    *,
    overlap: NativePreparationOverlap,
    budget: CoordinateEnclosureBudget | None = None,
) -> NativeCompletedSourceIssue:
    """Prepay an unchanged issued source's same-request final authentication.

    Only the actual process-local weak source and already-accounted exact receipt
    can reuse the existing seal. Mutable bytes are not authenticated here: the
    complete owner/container/byte and receipt checks still run after root end.
    """
    return _prepare_native_completed_source_issue(
        scope,
        source,
        stage=False,
        owner_id=paid_receipt.owner_id,
        imports=None,
        overlap=overlap,
        budget=budget,
        paid_receipt=paid_receipt,
    )


def _prepare_native_completed_source_issue(
    scope: NativeExecutionScope,
    source: CellMeshingResult,
    /,
    *,
    stage: bool,
    owner_id: str | None,
    imports: NativeExecutionImports | None,
    overlap: NativePreparationOverlap | None,
    budget: CoordinateEnclosureBudget | None,
    paid_receipt: NativeExecutionRecord | None = None,
) -> NativeCompletedSourceIssue:
    root = _require_native_source_scope(scope)
    if scope is not root:
        raise ValueError(
            "Native completed publication must be prepared on its actual root."
        )
    if imports is not None and overlap is not None:
        raise ValueError(
            "Native publication imports and overlap normalization are exclusive."
        )
    if imports is not None and imports._scope is not root:
        raise ValueError("Native publication imports must belong to its actual root.")
    ledger = native_preparation_ledger(root)
    continuation = None
    prepared_graph = None
    if paid_receipt is not None:
        if not ledger.receipt_is_accounted(root, paid_receipt):
            raise ValueError(
                "Native continuation receipt must already be paid by this request."
            )
        if overlap != ledger.overlap(root, paid_receipt):
            raise ValueError(
                "Native continuation overlap must match its exact paid receipt."
            )
        seal, receipt_id, prepared_graph = _native_completed_source_entry(
            source, paid_receipt
        )
        with _native_source_budget(scope, budget) as allowance:
            allowance.reserve(1 + len(seal.names) + 4 * len(seal.constructors))
            _require_native_source_top_fields(source, seal)
            allowance.reserve(
                len(seal.immutable_arrays)
                + len(seal.immutable_host_arrays)
                + len(seal.mutable_arrays)
            )
            _require_native_source_array_owners(seal)
            if prepared_graph is not None:
                allowance.reserve(prepared_graph.authentication_work)
                prepared_graph.require_current()
    elif not stage:
        entry = ledger._publications.get(id(source)) or ledger._history_sources.get(
            id(source)
        )
        if entry is None or entry[0]() is not source:
            raise ValueError(
                "Native completed source must have a staged publication or authenticated history."
            )
        seal = entry[1]
        with _native_source_budget(scope, budget) as allowance:
            _require_native_source_seal(source, seal, allowance)
    preparations = tuple(ledger._receipts.get(root, {}).values())
    if (
        imports is not None
        and imports.preparation is not None
        and not any(imports.preparation is item for item in preparations)
    ):
        preparations += (imports.preparation,)
    histories = tuple(ledger._history.get(root, {}).values())
    if prepared_graph is not None:
        prepared_ids = {id(phase.receipt) for phase in prepared_graph.phases}
        if histories or any(id(item) not in prepared_ids for item in preparations):
            # A newly introduced ancestry is not an unchanged same-result graph.
            prepared_graph = None
    if paid_receipt is not None:
        continuation = (weakref.ref(source), paid_receipt, receipt_id, prepared_graph)
    # Bound validation of the fixed receipt fields and admitted ancestry before
    # ending the request; archive receipts never recreate this live authority.
    with _native_source_budget(scope, budget) as allowance:
        allowance.reserve(1, 512 + 128 * (len(preparations) + len(histories) + 1))
        prefix = (
            _native_receipt_prefix(ledger, imports, preparations, histories, allowance)
            if paid_receipt is None else None
        )
        imported = ledger._imports.get(root, {})
        allowance.reserve(len(imported) + 1)
        import_count = (
            len(imports.receipt_ids) if imports is not None
            else sum(item is not None for item in imported.values())
        )
        graph_characters = 256 * (
            len(NativeExecutionRecord.__dataclass_fields__) + 17
            + import_count + len(histories)
        )
        graph_identities = import_count + len(histories)
        if prefix is not None:
            receipt_count = len(prefix.phases)
            allowance.reserve(receipt_count)
            # Completion binds the admitted prefix owners and validates only the
            # appended root; its constructor is admitted separately.
            graph_work = 4096 + prefix.authentication_work + receipt_count
            graph_identities += sum(
                len(phase.imported_ids) + len(phase.history) for phase in prefix.phases
            )
        elif prepared_graph is not None:
            allowance.reserve(len(preparations))
            receipt_count = len(prepared_graph.phases)
            # The issued graph is bound by the record constructor and again at
            # completion; only the appended root is validated (one node tariff
            # each) and digested once by its constructor.
            graph_work = 2 * (4096 + prepared_graph.authentication_work) + receipt_count
            allowance.reserve(receipt_count)
            graph_identities += sum(
                len(phase.imported_ids) + len(phase.history)
                for phase in prepared_graph.phases
            )
        else:
            receipt_nodes = {id(item): item for item in (*preparations, *histories)}
            if source.execution_evidence is not None:
                receipt_nodes[id(source.execution_evidence)] = source.execution_evidence
            pending = list(receipt_nodes.values())
            while pending:
                item = pending.pop()
                allowance.reserve(3 + len(item.history_evidence))
                for child in (
                    item.preparation_evidence,
                    item.consumer_evidence,
                    *item.history_evidence,
                ):
                    if child is not None and id(child) not in receipt_nodes:
                        allowance.reserve(1, 128)
                        receipt_nodes[id(child)] = child
                        pending.append(child)
            receipt_count = len(receipt_nodes)
            graph_work = 4096 * (1 + receipt_count)
            for item in receipt_nodes.values():
                allowance.reserve(4)
                item_identities = int(np.asarray(item.imported_preparation_count)) + len(
                    item.history_evidence
                )
                graph_characters += 256 * (
                    len(NativeExecutionRecord.__dataclass_fields__) + 17 + item_identities
                )
                graph_identities += item_identities
        graph_work += graph_characters
    if stage:
        # Atomic staging: the issue's own token covers this seal and its
        # registry entry for the root's remaining lifetime and afterwards. A
        # separate ledger retention of the same seal would admit it twice.
        with _native_source_budget(scope, budget) as allowance:
            seal = _native_source_seal(source, allowance)
    # Always check owners/liveness and every mutable byte at completion; pay it
    # now, while the root can refuse work/storage. No post-end reservation.
    array_work, array_scratch = _native_source_array_scan_cost(seal.mutable_arrays)
    work = (
        graph_work
        + len(seal.names)
        + 4 * len(seal.constructors)
        + sum(1 + len(names) for _, _, names, _ in seal.owners)
        + sum(1 + len(children) for _, children in seal.containers)
        + len(seal.immutable_arrays)
        + len(seal.immutable_host_arrays)
        + len(seal.mutable_arrays)
        + array_work
    )
    scratch = array_scratch + sum(
        128 + 16 * len(children) for _, children in seal.containers
    )
    # Retained graph metadata plus canonical graph serialization at completion:
    # measured <= 4.3 KiB per phase and each identity's digest text.
    owner_length = len(source.result_id if owner_id is None else owner_id)
    storage = root.admit_source_metadata(
        seal.metadata_bytes + scratch + 8192 * (1 + receipt_count)
        + (8192 + 16 * owner_length) * (1 + receipt_count) + 512 * graph_identities
        + (512 if stage else 0)
        + 64 + 8 * len(histories),
        work,
    )
    if stage:
        ledger._publications[id(source)] = (weakref.ref(source), seal)
    return NativeCompletedSourceIssue(
        root,
        seal,
        storage,
        source.result_id if owner_id is None else owner_id,
        imports,
        overlap,
        histories,
        preparations,
        continuation,
        prefix,
        staged=(weakref.ref(ledger), id(source)) if stage else None,
    )


def _native_completed_source_entry(
    source: CellMeshingResult, receipt: NativeExecutionRecord, /
) -> tuple[_NativeSourceSeal, str, _NativeReceiptGraph | None]:
    """Resolve only the actual current-process weak publication and paid receipt."""
    issue = _completed_native_sources.get(id(source))
    if (
        issue is None
        or issue._process_id != os.getpid()
        or issue._source is None
        or issue._source() is not source
        or issue._receipt is not receipt
        or source.execution_evidence is not receipt
        or issue._seal is None
        or issue._receipt_id is None
    ):
        raise ValueError("Native completed source must be the actual issued weak owner.")
    return issue._seal, issue._receipt_id, issue._receipt_graph


def _require_native_completed_source(
    source: CellMeshingResult,
    receipt: NativeExecutionRecord,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _NativeSourceSeal:
    """Authenticate numeric/publication ownership under the fresh live request."""
    scope = current_native_execution_scope()
    if scope is None:
        raise ValueError(
            "Native completed source authentication requires a live request."
        )
    seal, receipt_id, _ = _native_completed_source_entry(source, receipt)
    with _native_source_budget(scope, budget) as allowance:
        _require_native_source_seal(source, seal, allowance)
        native_preparation_ledger(scope).authenticate_receipt(scope, receipt)
        allowance.reserve(32)
        if receipt.receipt_hex() != receipt_id:
            raise ValueError(
                "Native completed source receipt bytes changed after publication."
            )
    return seal
