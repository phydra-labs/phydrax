#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Documented HBT-HERONS HDF5 lineage admission."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.cosmology._halo_lineage import (
    HaloLifecycleState,
    HaloLineageEventKind,
    HaloLineageEventLedger,
    HaloLineageProduct,
    HaloTracerEvidence,
    HaloTrackSnapshot,
)
from ...qualification import ReferenceArtifactManifest
from .._report import AdapterLoss, AdapterReport, AdapterStatus
from ._snapshots import _admit_path


_HBT_FIELDS = (
    "TrackId",
    "SinkTrackId",
    "DescendantTrackId",
    "NestedParentTrackId",
    "HostHaloId",
    "Rank",
    "Depth",
    "Nbound",
    "Mbound",
    "NboundType",
    "MboundType",
    "VmaxPhysical",
    "BoundM200Crit",
    "RmaxComoving",
    "RHalfComoving",
    "REncloseComoving",
    "BoundR200CritComoving",
    "ComovingAveragePosition",
    "PhysicalAverageVelocity",
    "ComovingMostBoundPosition",
    "PhysicalMostBoundVelocity",
    "InertialTensor",
    "InertialTensorWeighted",
    "SpecificSelfPotentialEnergy",
    "SpecificSelfKineticEnergy",
    "SpecificAngularMomentum",
    "TracerIndex",
    "MostBoundParticleId",
    "SnapshotOfBirth",
    "SnapshotOfSink",
    "SnapshotOfDeath",
    "SnapshotOfLastIsolation",
    "SnapshotOfLastMaxVmax",
    "SnapshotOfLastMaxMass",
    "LastMaxMass",
    "LastMaxVmaxPhysical",
)


class HbtHeronsSidecar(StrictModule, NonTrainableState):
    """Lossless numeric HBT columns and separately identified memberships."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    field_values: tuple[Array, ...]
    row_mask: Array
    bound_particle_ids: Array
    bound_particle_mask: Array
    source_particle_ids: Array
    source_particle_mask: Array
    snapshot_index: int = eqx.field(static=True)
    sidecar_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: Sequence[str],
        field_values: Sequence[Array],
        row_mask: Array,
        bound_particle_ids: Array,
        bound_particle_mask: Array,
        source_particle_ids: Array,
        source_particle_mask: Array,
        snapshot_index: int,
        /,
    ):
        names = tuple(str(name).strip() for name in field_names)
        values = tuple(
            jax.lax.stop_gradient(jnp.asarray(value)) for value in field_values
        )
        mask = jax.lax.stop_gradient(jnp.asarray(row_mask, dtype=jnp.bool_))
        bound = jax.lax.stop_gradient(jnp.asarray(bound_particle_ids, dtype=jnp.int64))
        bound_mask = jax.lax.stop_gradient(
            jnp.asarray(bound_particle_mask, dtype=jnp.bool_)
        )
        source = jax.lax.stop_gradient(jnp.asarray(source_particle_ids, dtype=jnp.int64))
        source_mask = jax.lax.stop_gradient(
            jnp.asarray(source_particle_mask, dtype=jnp.bool_)
        )
        if (
            not names
            or len(names) != len(values)
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or mask.ndim != 1
            or any(value.shape[0] != mask.size for value in values)
            or bound.shape != bound_mask.shape
            or source.shape != source_mask.shape
            or bound.ndim != 2
            or source.ndim != 2
            or bound.shape[0] != mask.size
            or source.shape[0] != mask.size
        ):
            raise ValueError("HBT-HERONS sidecar fields do not share fixed capacities.")
        self.field_names = names
        self.field_values = values
        self.row_mask = mask
        self.bound_particle_ids = bound
        self.bound_particle_mask = bound_mask
        self.source_particle_ids = source
        self.source_particle_mask = source_mask
        self.snapshot_index = int(snapshot_index)
        self.sidecar_id = canonical_fingerprint(
            {
                "kind": "hbt-herons-sidecar",
                "field_names": list(names),
                "snapshot_index": self.snapshot_index,
                "arrays": array_tree_fingerprint(
                    (values, mask, bound, bound_mask, source, source_mask)
                ),
            }
        )

    def field(self, name: str, /) -> Array:
        """Return one retained producer column without semantic relabeling."""
        key = str(name).strip()
        if key not in self.field_names:
            raise KeyError(key)
        return self.field_values[self.field_names.index(key)]


class HbtHeronsCatalogImport(StrictModule, NonTrainableState):
    lineage: HaloLineageProduct
    sidecar: HbtHeronsSidecar
    source: ReferenceArtifactManifest
    report: AdapterReport

    def __init__(
        self,
        lineage: HaloLineageProduct,
        sidecar: HbtHeronsSidecar,
        source: ReferenceArtifactManifest,
        report: AdapterReport,
        /,
    ):
        if not isinstance(lineage, HaloLineageProduct):
            raise TypeError("lineage must be HaloLineageProduct.")
        if not isinstance(sidecar, HbtHeronsSidecar):
            raise TypeError("sidecar must be HbtHeronsSidecar.")
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("source must be ReferenceArtifactManifest.")
        if not isinstance(report, AdapterReport):
            raise TypeError("report must be AdapterReport.")
        self.lineage = lineage
        self.sidecar = sidecar
        self.source = source
        self.report = report


def _pad_field(value: np.ndarray, count: int, capacity: int, /) -> np.ndarray:
    if value.shape[0] != count or value.dtype.kind not in "biufc":
        raise ValueError("HBT-HERONS Subhalos fields must be fixed numeric columns.")
    result = np.zeros((capacity, *value.shape[1:]), dtype=value.dtype)
    result[:count] = value
    return result


def _memberships(
    dataset: h5py.Dataset | None,
    count: int,
    halo_capacity: int,
    particle_capacity: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((halo_capacity, particle_capacity), -1, dtype=np.int64)
    mask = np.zeros((halo_capacity, particle_capacity), dtype=np.bool_)
    if dataset is None:
        return ids, mask
    if dataset.shape != (count,):
        raise ValueError("HBT-HERONS membership rows must align with Subhalos.")
    for row in range(count):
        particles = np.asarray(dataset[row], dtype=np.int64).reshape((-1,))
        if particles.size > particle_capacity:
            raise MemoryError("HBT-HERONS membership exceeds maximum_particles_per_halo.")
        if np.any(particles < 0) or len(set(particles.tolist())) != particles.size:
            raise ValueError(
                "HBT-HERONS membership particle IDs must be unique and non-negative."
            )
        ids[row, : particles.size] = particles
        mask[row, : particles.size] = True
    return ids, mask


def _loss(path: str, rationale: str, /) -> AdapterLoss:
    return AdapterLoss(
        path,
        "import",
        "dropped",
        rationale,
        changes_interpretation=False,
    )


def read_hbt_herons_catalog(
    path: str | Path,
    source: ReferenceArtifactManifest,
    /,
    *,
    maximum_halos: int,
    maximum_particles_per_halo: int,
    snapshot_index: int | None = None,
    maximum_source_bytes: int = 2_000_000_000,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> HbtHeronsCatalogImport:
    """Read one HBT-HERONS ``SubSnap`` catalog and preserve producer fields."""

    if isinstance(maximum_halos, bool) or int(maximum_halos) <= 0:
        raise ValueError("maximum_halos must be a positive integer.")
    if (
        isinstance(maximum_particles_per_halo, bool)
        or int(maximum_particles_per_halo) <= 0
    ):
        raise ValueError("maximum_particles_per_halo must be a positive integer.")
    halo_capacity = int(maximum_halos)
    particle_capacity = int(maximum_particles_per_halo)
    resource = _admit_path(
        path,
        source,
        maximum_source_bytes=maximum_source_bytes,
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    losses: list[AdapterLoss] = []
    with h5py.File(BytesIO(resource.data), "r") as handle:
        if "Subhalos" not in handle:
            raise ValueError("HBT-HERONS SubSnap omits the Subhalos dataset.")
        records = np.asarray(handle["Subhalos"])
        if records.ndim != 1 or records.dtype.names is None:
            raise ValueError(
                "HBT-HERONS Subhalos must be a one-dimensional compound dataset."
            )
        count = records.shape[0]
        if count < 1 or count > halo_capacity:
            raise MemoryError("HBT-HERONS subhalo count violates maximum_halos.")
        names = tuple(records.dtype.names)
        if "TrackId" not in names:
            raise ValueError(
                "HBT-HERONS TrackId is required for stable lineage identity."
            )
        fields = tuple(
            _pad_field(np.asarray(records[name]), count, halo_capacity) for name in names
        )
        bound_dataset = (
            handle["SubhaloParticles"] if "SubhaloParticles" in handle else None
        )
        source_dataset = (
            handle["SourceSubhaloParticles"]
            if "SourceSubhaloParticles" in handle
            else None
        )
        bound_membership_present = bound_dataset is not None
        source_membership_present = source_dataset is not None
        bound_ids, bound_mask = _memberships(
            bound_dataset, count, halo_capacity, particle_capacity
        )
        source_ids, source_mask = _memberships(
            source_dataset, count, halo_capacity, particle_capacity
        )
        if bound_dataset is None:
            losses.append(
                _loss(
                    "SubhaloParticles",
                    "Bound membership was not written by the source catalog.",
                )
            )
        if source_dataset is None:
            losses.append(
                _loss(
                    "SourceSubhaloParticles",
                    "Source membership is absent; it is not equated with bound membership.",
                )
            )
        if snapshot_index is None:
            if "SnapshotId" not in handle.attrs:
                raise ValueError("HBT-HERONS snapshot index is ambiguous.")
            snapshot = int(handle.attrs["SnapshotId"])
        else:
            snapshot = int(snapshot_index)
        for field in _HBT_FIELDS:
            if field not in names:
                losses.append(
                    _loss(
                        f"Subhalos.{field}",
                        f"Documented HBT-HERONS field {field} was omitted by the producer.",
                    )
                )

    track_values = np.asarray(records["TrackId"], dtype=np.int64)
    if np.any(track_values < 0) or len(set(track_values.tolist())) != count:
        raise ValueError("HBT-HERONS TrackId values must be unique and non-negative.")
    row_mask = np.zeros(halo_capacity, dtype=np.bool_)
    row_mask[:count] = True
    tracks = np.full(halo_capacity, -1, dtype=np.int64)
    tracks[:count] = track_values
    source_halo_ids = tracks.copy()
    source_rows = np.full(halo_capacity, -1, dtype=np.int64)
    source_rows[:count] = np.arange(count)
    nbound = np.asarray(records["Nbound"], dtype=np.int64) if "Nbound" in names else None
    death = (
        np.asarray(records["SnapshotOfDeath"], dtype=np.int64)
        if "SnapshotOfDeath" in names
        else np.full(count, -1, dtype=np.int64)
    )
    states = np.zeros(halo_capacity, dtype=np.int8)
    if nbound is None:
        states[:count] = int(HaloLifecycleState.UNKNOWN)
    else:
        states[:count] = np.where(
            nbound > 0,
            int(HaloLifecycleState.RESOLVED),
            np.where(
                death < 0,
                int(HaloLifecycleState.ORPHAN),
                int(HaloLifecycleState.DISRUPTED),
            ),
        )
    hosts = np.full(halo_capacity, -1, dtype=np.int64)
    if "NestedParentTrackId" in names:
        declared_hosts = np.asarray(records["NestedParentTrackId"], dtype=np.int64)
        present_tracks = set(track_values.tolist())
        hosts[:count] = np.asarray(
            [
                value if value < 0 or value in present_tracks else -1
                for value in declared_hosts
            ],
            dtype=np.int64,
        )
        if np.any((declared_hosts >= 0) & (hosts[:count] < 0)):
            losses.append(
                _loss(
                    "Subhalos.NestedParentTrackId",
                    "Parent tracks absent from this bounded catalog were retained only in the producer sidecar.",
                )
            )

    source_membership_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "hbt-source-membership",
                "track_id": int(tracks[row]),
                "particles": array_tree_fingerprint(source_ids[row, source_mask[row]]),
            }
        )
        if row < count and source_membership_present
        else ""
        for row in range(halo_capacity)
    )
    bound_membership_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "hbt-bound-membership",
                "track_id": int(tracks[row]),
                "particles": array_tree_fingerprint(bound_ids[row, bound_mask[row]]),
            }
        )
        if row < count and bound_membership_present
        else ""
        for row in range(halo_capacity)
    )
    track_snapshot = HaloTrackSnapshot(
        snapshot,
        source_rows,
        source_halo_ids,
        tracks,
        hosts,
        states,
        row_mask,
        source_membership_ids=source_membership_ids,
        bound_membership_ids=bound_membership_ids,
    )

    descendants = np.full((1, halo_capacity), -1, dtype=np.int64)
    descendant_mask = np.zeros((1, halo_capacity), dtype=np.bool_)
    if "DescendantTrackId" in names:
        descendants[0, :count] = np.asarray(records["DescendantTrackId"], dtype=np.int64)
        descendant_mask[0, :count] = descendants[0, :count] >= 0
    sinks = np.full((1, halo_capacity), -1, dtype=np.int64)
    sink_mask = np.zeros((1, halo_capacity), dtype=np.bool_)
    if "SinkTrackId" in names:
        sinks[0, :count] = np.asarray(records["SinkTrackId"], dtype=np.int64)
        sink_mask[0, :count] = sinks[0, :count] >= 0

    tracer_ids = np.full((1, halo_capacity, 1), -1, dtype=np.int64)
    tracer_ranks = np.full((1, halo_capacity, 1), -1, dtype=np.int32)
    tracer_present = np.zeros((1, halo_capacity, 1), dtype=np.bool_)
    if "MostBoundParticleId" in names:
        tracer_ids[0, :count, 0] = np.asarray(
            records["MostBoundParticleId"], dtype=np.int64
        )
        tracer_present[0, :count, 0] = tracer_ids[0, :count, 0] >= 0
    if "TracerIndex" in names:
        tracer_ranks[0, :count, 0] = np.asarray(records["TracerIndex"], dtype=np.int32)
    empty_rows = np.empty((0, halo_capacity), dtype=np.int32)
    evidence = HaloTracerEvidence(
        tracer_ids,
        tracer_ranks,
        tracer_present,
        empty_rows,
        empty_rows,
        empty_rows,
        np.empty((0, halo_capacity), dtype=np.float64),
        np.empty((0, halo_capacity), dtype=np.bool_),
        np.empty((0, halo_capacity), dtype=np.bool_),
        np.empty((0,), dtype=np.bool_),
    )

    event_capacity = max(1, 5 * halo_capacity)
    event_kind = np.zeros(event_capacity, dtype=np.int8)
    event_snapshot = np.full(event_capacity, -1, dtype=np.int32)
    event_track = np.full(event_capacity, -1, dtype=np.int64)
    event_related = np.full(event_capacity, -1, dtype=np.int64)
    event_active = np.zeros(event_capacity, dtype=np.bool_)
    event_success = np.zeros(event_capacity, dtype=np.bool_)
    events: list[tuple[int, int, int, int]] = []
    births = (
        np.asarray(records["SnapshotOfBirth"], dtype=np.int32)
        if "SnapshotOfBirth" in names
        else None
    )
    for row in range(count):
        if births is not None:
            events.append(
                (int(HaloLineageEventKind.BIRTH), int(births[row]), int(tracks[row]), -1)
            )
        if states[row] == int(HaloLifecycleState.ORPHAN):
            events.append(
                (int(HaloLineageEventKind.ORPHANED), snapshot, int(tracks[row]), -1)
            )
        if sink_mask[0, row]:
            sink_snapshot = (
                int(records["SnapshotOfSink"][row])
                if "SnapshotOfSink" in names
                else snapshot
            )
            events.append(
                (
                    int(HaloLineageEventKind.SINK),
                    sink_snapshot,
                    int(tracks[row]),
                    int(sinks[0, row]),
                )
            )
        if death[row] >= 0 and not sink_mask[0, row]:
            events.append(
                (
                    int(HaloLineageEventKind.DISRUPTION),
                    int(death[row]),
                    int(tracks[row]),
                    -1,
                )
            )
    for index, (kind, event_at, track, related) in enumerate(events):
        event_kind[index] = kind
        event_snapshot[index] = event_at
        event_track[index] = track
        event_related[index] = related
        event_active[index] = True
        event_success[index] = True
    ledger = HaloLineageEventLedger(
        event_kind,
        event_snapshot,
        event_track,
        event_related,
        event_active,
        event_success,
    )
    lineage = HaloLineageProduct(
        (track_snapshot,),
        ledger,
        evidence,
        descendants,
        descendant_mask,
        sinks,
        sink_mask,
        True,
    )
    sidecar = HbtHeronsSidecar(
        names,
        fields,
        row_mask,
        bound_ids,
        bound_mask,
        source_ids,
        source_mask,
        snapshot,
    )
    status = AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS
    report = AdapterReport(
        status,
        "HBT-HERONS SubSnap HDF5",
        "HaloLineageProduct with HBT-HERONS sidecar",
        source_id=source.manifest_id,
        target_id=lineage.lineage_id,
        coordinate_mapping=("Subhalos[row] -> fixed-capacity halo track row",),
        preserved_fields=tuple(names)
        + (("SubhaloParticles",) if bound_membership_present else ())
        + (("SourceSubhaloParticles",) if source_membership_present else ())
        + (
            "source_rights",
            "sink_relation",
            "descendant_relation",
        ),
        assumptions=(
            "SinkTrackId remains a physical sink relation and is never substituted for DescendantTrackId.",
            "HBT bound and source particle memberships retain distinct content identities.",
        ),
        losses=tuple(losses),
    )
    return HbtHeronsCatalogImport(lineage, sidecar, source, report)


__all__ = [
    "HbtHeronsCatalogImport",
    "HbtHeronsSidecar",
    "read_hbt_herons_catalog",
]
