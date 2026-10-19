#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scope-owned preparation identities and ended predecessor accounting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import final, Protocol

import numpy as np

from .._host_storage import declared_object_storage_bytes, retained_host_storage_profile
from .._meshcore import (
    is_native_host_buffer,
    NativeExecutionEvidence,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from ._measurements import NativeExecutionRecord


# The transport has five uint64 counters and two binary64 clock values.
_OVERLAP_LAYOUT = (0, 0, 0, 0, 0, 0, 0)


class NativePreparedOwner(Protocol):
    """Prepared owners expose their scientific identity without a provider import."""

    @property
    def prepared_id(self) -> str: ...


def _ancestors(scope: NativeExecutionScope, /) -> Iterator[NativeExecutionScope]:
    current: NativeExecutionScope | None = scope
    while current is not None:
        yield current
        current = current.parent


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
    """Retain preparations in one root attachment with ancestry-local receipts."""

    def __init__(self) -> None:
        self._live: dict[str, dict[int, NativePreparedOwner]] = {}
        self._receipts: dict[NativeExecutionScope, dict[int, NativeExecutionRecord]] = {}
        self._imports: dict[NativeExecutionScope, dict[int, NativeExecutionRecord]] = {}
        self._sources: dict[NativeExecutionScope, set[str]] = {}
        self._overlaps: dict[
            NativeExecutionScope, dict[int, NativePreparationOverlap]
        ] = {}
        self._storage: list[NativeHostStorageWorkspace] = []
        self._retained_owners: dict[int, object] = {}

    def close(self) -> None:
        """Release preparation identities when the root scope ends."""
        self._live.clear()
        self._receipts.clear()
        self._imports.clear()
        self._sources.clear()
        self._overlaps.clear()
        self._retained_owners.clear()
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
        with scope.batch(1) as ticket:
            self._retain_storage(scope, (prepared,), 256)
            ticket.consume(1)
            self._live.setdefault(prepared_id, {})[id(prepared)] = prepared

    def receipt_is_accounted(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord,
        /,
    ) -> bool:
        """Resolve the retained object through the scope's live ancestry."""
        return any(
            self._receipts.get(ancestor, {}).get(id(receipt)) is receipt
            for ancestor in _ancestors(scope)
        )

    def imported_receipt_ids(self, scope: NativeExecutionScope, /) -> tuple[str, ...]:
        """Snapshot admitted predecessor IDs visible to this live consumer."""
        if scope.evidence is not None:
            raise ValueError(
                "NativePreparationLedger.scope must remain live when recording imports; "
                "got an ended execution scope."
            )
        receipts = {
            receipt.receipt_hex()
            for ancestor in _ancestors(scope)
            for receipt in self._imports.get(ancestor, {}).values()
        }
        identities = tuple(sorted(receipts))
        if identities:
            self._retain_storage(scope, (), 256 + 144 * len(identities))
        return identities

    def import_receipt(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord,
        /,
        *,
        import_work: bool = True,
    ) -> None:
        """Import each ended predecessor phase once, without replaying chain totals."""
        receipt.require_valid()
        pending: list[NativeExecutionRecord] = []
        current: NativeExecutionRecord | None = receipt
        while current is not None:
            if not self.receipt_is_accounted(scope, current):
                pending.append(current)
            current = current.preparation_evidence
        self._retain_receipts(scope, pending, import_work=import_work)

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
        self._retain_receipts(scope, [receipt], import_work=import_work)

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
        overlaps: list[NativePreparationOverlap] = []
        work = queries = 0
        seconds = 0.0
        for receipt in receipts:
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
        with scope.batch(work + len(receipts), queries_upper=queries) as ticket:
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
                + (ancestor not in self._imports if import_work else 0)
                for ancestor in ancestors
            )
            metadata = (
                len(receipts)
                * (256 + overlap_upper + (192 if import_work else 96) * len(ancestors))
                + 256 * new_maps
            )
            self._retain_storage(scope, tuple(receipts), metadata)
            if import_work:
                scope.import_preparation(
                    work=0, geometry_queries=0, elapsed_seconds=seconds
                )
            ticket.consume(work + len(receipts), geometry_queries=queries)
            for receipt, overlap in zip(receipts, overlaps, strict=True):
                for ancestor in ancestors:
                    self._receipts.setdefault(ancestor, {})[id(receipt)] = receipt
                    self._overlaps.setdefault(ancestor, {})[id(receipt)] = overlap
                    if import_work:
                        self._imports.setdefault(ancestor, {})[id(receipt)] = receipt

    def _retain_storage(
        self,
        scope: NativeExecutionScope,
        owners: tuple[NativePreparedOwner | NativeExecutionRecord | str, ...],
        metadata_bytes: int,
        /,
    ) -> None:
        """Charge each retained graph increment to its active phase until root closure."""
        retained, declared, _, _ = retained_host_storage_profile(
            owners, self._retained_owners, is_native_host_buffer
        )
        # External-host reservations borrow the active target and keep its
        # storage ancestry alive after settlement. Separate increments avoid
        # growing a reservation while a later phase owns a different target.
        storage = scope.host_workspace().open()
        self._storage.append(storage)
        storage.set_bound(declared + metadata_bytes)
        self._retained_owners.update(retained)

    def overlap(
        self,
        scope: NativeExecutionScope,
        receipt: NativeExecutionRecord | None,
        /,
    ) -> NativePreparationOverlap:
        """Snapshot only the matched source ancestry before the request attachment closes."""
        work = queries = external_work = external_queries = primitive_queries = 0
        live_seconds = imported_seconds = 0.0
        current = receipt
        represented = self._overlaps.get(scope, {})
        while current is not None:
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
            if any(
                receipt.owner_id == source_id
                for receipt in self._receipts.get(ancestor, {}).values()
            ):
                return True
        return False

    def remember_source(self, scope: NativeExecutionScope, source_id: str, /) -> None:
        """Register a source prepared under this scope's cumulative allowance."""
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
