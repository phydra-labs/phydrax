#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._report import (
    AdapterError,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    require_lossless,
)


class OpenPTVTargetRecords(StrictModule, NonTrainableState):
    """One camera/frame of legacy target records in native row/column order."""

    target_ids: jnp.ndarray
    positions_rc: jnp.ndarray
    pixel_count: jnp.ndarray
    extent_rc: jnp.ndarray
    summed_intensity: jnp.ndarray
    correspondence_ids: jnp.ndarray
    valid: jnp.ndarray
    camera_id: str = eqx.field(static=True)
    frame_index: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_ids,
        positions_rc,
        pixel_count,
        extent_rc,
        summed_intensity,
        correspondence_ids,
        valid,
        /,
        *,
        camera_id: str,
        frame_index: int,
        source_id: str,
    ):
        ids = np.asarray(target_ids, dtype=np.int64).reshape((-1,))
        positions = np.asarray(positions_rc, dtype=float)
        count = np.asarray(pixel_count, dtype=np.int64).reshape((-1,))
        extent = np.asarray(extent_rc, dtype=np.int64)
        intensity = np.asarray(summed_intensity, dtype=np.int64).reshape((-1,))
        correspondence = np.asarray(correspondence_ids, dtype=np.int64).reshape((-1,))
        valid_ = np.asarray(valid, dtype=bool).reshape((-1,))
        size = ids.size
        if (
            positions.shape != (size, 2)
            or extent.shape != (size, 2)
            or any(
                value.size != size for value in (count, intensity, correspondence, valid_)
            )
        ):
            raise ValueError("OpenPTV target record arrays have inconsistent shapes.")
        if (
            np.any(ids < 0)
            or np.unique(ids).size != size
            or np.any(valid_ & ~np.all(np.isfinite(positions), axis=-1))
            or np.any(
                valid_ & ((count <= 0) | np.any(extent <= 0, axis=-1) | (intensity < 0))
            )
        ):
            raise ValueError("OpenPTV target records contain invalid active values.")
        camera = str(camera_id).strip()
        source = str(source_id).strip()
        if not camera or not source:
            raise ValueError("camera_id and source_id must be non-empty.")
        self.target_ids = jnp.asarray(ids)
        self.positions_rc = jnp.asarray(np.where(valid_[:, None], positions, 0.0))
        self.pixel_count = jnp.asarray(np.where(valid_, count, 0))
        self.extent_rc = jnp.asarray(np.where(valid_[:, None], extent, 0))
        self.summed_intensity = jnp.asarray(np.where(valid_, intensity, 0))
        self.correspondence_ids = jnp.asarray(np.where(valid_, correspondence, -1))
        self.valid = jnp.asarray(valid_)
        self.camera_id = camera
        self.frame_index = int(frame_index)
        self.source_id = source


class OpenPTVReconstructionRecords(StrictModule, NonTrainableState):
    """One legacy rt_is frame with right-handed physical 3-D coordinates."""

    record_ids: jnp.ndarray
    positions_xyz: jnp.ndarray
    target_indices: jnp.ndarray
    valid: jnp.ndarray
    frame_index: int = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        record_ids,
        positions_xyz,
        target_indices,
        valid,
        /,
        *,
        frame_index: int,
        frame_id: str,
        source_id: str,
    ):
        ids = np.asarray(record_ids, dtype=np.int64).reshape((-1,))
        positions = np.asarray(positions_xyz, dtype=float)
        targets = np.asarray(target_indices, dtype=np.int64)
        valid_ = np.asarray(valid, dtype=bool).reshape((-1,))
        size = ids.size
        if (
            positions.shape != (size, 3)
            or targets.shape != (size, 4)
            or valid_.size != size
        ):
            raise ValueError("OpenPTV reconstruction arrays have inconsistent shapes.")
        if (
            np.unique(ids).size != size
            or np.any(ids <= 0)
            or np.any(targets < -1)
            or np.any(valid_ & ~np.all(np.isfinite(positions), axis=-1))
        ):
            raise ValueError(
                "OpenPTV reconstruction records contain invalid active values."
            )
        frame = str(frame_id).strip()
        source = str(source_id).strip()
        if not frame or not source:
            raise ValueError("frame_id and source_id must be non-empty.")
        self.record_ids = jnp.asarray(ids)
        self.positions_xyz = jnp.asarray(np.where(valid_[:, None], positions, 0.0))
        self.target_indices = jnp.asarray(np.where(valid_[:, None], targets, -1))
        self.valid = jnp.asarray(valid_)
        self.frame_index = int(frame_index)
        self.frame_id = frame
        self.source_id = source


class OpenPTVTrackRecords(StrictModule, NonTrainableState):
    """Identity-bearing flat track samples reconstructed from rt_is/ptv_is links."""

    track_ids: jnp.ndarray
    times: jnp.ndarray
    positions_xyz: jnp.ndarray
    valid: jnp.ndarray
    reset_before: jnp.ndarray
    frame_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        track_ids,
        times,
        positions_xyz,
        valid,
        reset_before,
        /,
        *,
        frame_id: str,
        source_id: str,
    ):
        tracks = np.asarray(track_ids, dtype=np.int64).reshape((-1,))
        times_ = np.asarray(times, dtype=float).reshape((-1,))
        positions = np.asarray(positions_xyz, dtype=float)
        valid_ = np.asarray(valid, dtype=bool).reshape((-1,))
        resets = np.asarray(reset_before, dtype=bool).reshape((-1,))
        size = tracks.size
        if positions.shape != (size, 3) or any(
            value.size != size for value in (times_, valid_, resets)
        ):
            raise ValueError("OpenPTV track record arrays have inconsistent shapes.")
        if np.any(tracks < 0) or np.any(
            valid_ & (~np.isfinite(times_) | ~np.all(np.isfinite(positions), axis=-1))
        ):
            raise ValueError("OpenPTV track records contain invalid active values.")
        active_pairs = np.column_stack((tracks[valid_], times_[valid_]))
        if (
            active_pairs.size
            and np.unique(active_pairs, axis=0).shape[0] != active_pairs.shape[0]
        ):
            raise ValueError("OpenPTV tracks contain duplicate (track_id, time) samples.")
        for track in np.unique(tracks[valid_]):
            track_times = times_[valid_ & (tracks == track)]
            if np.any(np.diff(np.sort(track_times)) <= 0.0):
                raise ValueError(
                    "OpenPTV track times must be strictly increasing per identity."
                )
        frame = str(frame_id).strip()
        source = str(source_id).strip()
        if not frame or not source:
            raise ValueError("frame_id and source_id must be non-empty.")
        self.track_ids = jnp.asarray(tracks)
        self.times = jnp.asarray(np.where(valid_, times_, 0.0))
        self.positions_xyz = jnp.asarray(np.where(valid_[:, None], positions, 0.0))
        self.valid = jnp.asarray(valid_)
        self.reset_before = jnp.asarray(resets & valid_)
        self.frame_id = frame
        self.source_id = source


def read_openptv_targets(
    path: str | Path,
    /,
    *,
    camera_id: str,
    frame_index: int,
) -> tuple[OpenPTVTargetRecords, AdapterReport]:
    """Read `pnr x y n nx ny sumg tnr`, mapping x/y to column/row."""
    source = Path(path)
    count, table = _counted_table(source, columns=8)
    ids = _integers(table[:, 0], "pnr")
    pixel_count = _integers(table[:, 3], "n")
    extent_x = _integers(table[:, 4], "nx")
    extent_y = _integers(table[:, 5], "ny")
    intensity = _integers(table[:, 6], "sumg")
    correspondence = _integers(table[:, 7], "tnr")
    source_id = _file_id(source, "openptv-targets")
    records = OpenPTVTargetRecords(
        ids,
        np.column_stack((table[:, 2], table[:, 1])),
        pixel_count,
        np.column_stack((extent_y, extent_x)),
        intensity,
        correspondence,
        np.ones(count, dtype=bool),
        camera_id=camera_id,
        frame_index=frame_index,
        source_id=source_id,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenPTV-targets",
        "OpenPTVTargetRecords",
        source_id=source_id,
        target_id=source_id,
        coordinate_mapping=(
            "x -> column_right",
            "y -> row_down",
            "nx -> extent_column",
            "ny -> extent_row",
        ),
        preserved_fields=("pnr", "x", "y", "n", "nx", "ny", "sumg", "tnr"),
    )
    return records, report


def write_openptv_targets(
    path: str | Path,
    records: OpenPTVTargetRecords,
    /,
) -> AdapterReport:
    """Write the exact counted OpenPTV target record layout."""
    valid = np.asarray(records.valid)
    table = np.column_stack(
        (
            np.asarray(records.target_ids)[valid],
            np.asarray(records.positions_rc)[valid, 1],
            np.asarray(records.positions_rc)[valid, 0],
            np.asarray(records.pixel_count)[valid],
            np.asarray(records.extent_rc)[valid, 1],
            np.asarray(records.extent_rc)[valid, 0],
            np.asarray(records.summed_intensity)[valid],
            np.asarray(records.correspondence_ids)[valid],
        )
    )
    destination = Path(path)
    _write_counted_table(
        destination, table, fmt=("%d", "%.9g", "%.9g", "%d", "%d", "%d", "%d", "%d")
    )
    target_id = _file_id(destination, "openptv-targets")
    return AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenPTVTargetRecords",
        "OpenPTV-targets",
        source_id=records.source_id,
        target_id=target_id,
        coordinate_mapping=(
            "column_right -> x",
            "row_down -> y",
            "extent_column -> nx",
            "extent_row -> ny",
        ),
        preserved_fields=(
            "target_ids",
            "positions_rc",
            "pixel_count",
            "extent_rc",
            "summed_intensity",
            "correspondence_ids",
        ),
    )


def read_openptv_reconstruction(
    path: str | Path,
    /,
    *,
    frame_index: int,
    frame_id: str,
    target_index_base: int = 0,
) -> tuple[OpenPTVReconstructionRecords, AdapterReport]:
    """Read one counted rt_is record file with explicit target-index base."""
    if target_index_base not in (0, 1):
        raise ValueError("target_index_base must be 0 or 1.")
    source = Path(path)
    count, table = _counted_table(source, columns=8)
    ids = _integers(table[:, 0], "record number")
    targets = _integers(table[:, 4:8], "target indices")
    if target_index_base == 1:
        targets = np.where(targets >= 0, targets - 1, targets)
    source_id = _file_id(source, "openptv-rt-is")
    records = OpenPTVReconstructionRecords(
        ids,
        table[:, 1:4],
        targets,
        np.ones(count, dtype=bool),
        frame_index=frame_index,
        frame_id=frame_id,
        source_id=source_id,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenPTV-rt_is",
        "OpenPTVReconstructionRecords",
        source_id=source_id,
        target_id=source_id,
        coordinate_mapping=("x -> x", "y -> y", "z -> z"),
        preserved_fields=("record_ids", "positions_xyz", "target_indices"),
        assumptions=(f"target_index_base={target_index_base}", f"frame_id={frame_id}"),
    )
    return records, report


def write_openptv_reconstruction(
    path: str | Path,
    records: OpenPTVReconstructionRecords,
    /,
    *,
    target_index_base: int = 0,
) -> AdapterReport:
    """Write one exact counted rt_is file."""
    if target_index_base not in (0, 1):
        raise ValueError("target_index_base must be 0 or 1.")
    valid = np.asarray(records.valid)
    targets = np.asarray(records.target_indices)[valid]
    if target_index_base == 1:
        targets = np.where(targets >= 0, targets + 1, targets)
    table = np.column_stack(
        (
            np.asarray(records.record_ids)[valid],
            np.asarray(records.positions_xyz)[valid],
            targets,
        )
    )
    destination = Path(path)
    _write_counted_table(
        destination, table, fmt=("%d", "%.9g", "%.9g", "%.9g", "%d", "%d", "%d", "%d")
    )
    target_id = _file_id(destination, "openptv-rt-is")
    return AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenPTVReconstructionRecords",
        "OpenPTV-rt_is",
        source_id=records.source_id,
        target_id=target_id,
        coordinate_mapping=("x -> x", "y -> y", "z -> z"),
        preserved_fields=("record_ids", "positions_xyz", "target_indices"),
        assumptions=(f"target_index_base={target_index_base}",),
    )


def read_openptv_tracks(
    reconstruction_paths: Sequence[str | Path],
    linkage_paths: Sequence[str | Path],
    /,
    *,
    times: Sequence[float],
    frame_id: str,
) -> tuple[OpenPTVTrackRecords, AdapterReport]:
    """Reconstruct stable track identities from reciprocal rt_is/ptv_is links."""
    rt_paths = tuple(Path(path) for path in reconstruction_paths)
    ptv_paths = tuple(Path(path) for path in linkage_paths)
    times_ = np.asarray(tuple(times), dtype=float)
    if (
        not rt_paths
        or len(rt_paths) != len(ptv_paths)
        or times_.shape != (len(rt_paths),)
    ):
        raise ValueError(
            "Track import requires equal non-empty reconstruction/linkage/time sequences."
        )
    if not np.all(np.isfinite(times_)) or np.any(np.diff(times_) <= 0.0):
        raise ValueError("Track import times must be finite and strictly increasing.")
    positions: list[np.ndarray] = []
    previous: list[np.ndarray] = []
    following: list[np.ndarray] = []
    for rt_path, ptv_path in zip(rt_paths, ptv_paths, strict=True):
        count, rt = _counted_table(rt_path, columns=8)
        link_count, links = _counted_table(ptv_path, columns=5)
        if count != link_count or not np.allclose(
            rt[:, 1:4], links[:, 2:5], rtol=0.0, atol=1e-9
        ):
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "OpenPTV reconstruction and linkage records disagree.",
            )
        positions.append(rt[:, 1:4])
        previous.append(_integers(links[:, 0], "previous links"))
        following.append(_integers(links[:, 1], "next links"))
    assigned: list[np.ndarray] = []
    next_track_id = 0
    reset_frames: list[np.ndarray] = []
    for frame, prev in enumerate(previous):
        ids = np.full(prev.shape, -1, dtype=np.int64)
        resets = np.zeros(prev.shape, dtype=bool)
        for index, predecessor in enumerate(prev):
            if predecessor >= 0 and frame > 0:
                if (
                    predecessor >= assigned[frame - 1].size
                    or following[frame - 1][predecessor] != index
                ):
                    raise AdapterError(
                        AdapterStatus.INCONSISTENT_SOURCE,
                        "OpenPTV linkage is out of range or not reciprocal.",
                    )
                ids[index] = assigned[frame - 1][predecessor]
            else:
                ids[index] = next_track_id
                next_track_id += 1
                resets[index] = predecessor >= 0
        assigned.append(ids)
        reset_frames.append(resets)
    track_ids = np.concatenate(assigned)
    sample_times = np.concatenate(
        [
            np.full(values.shape[0], times_[index])
            for index, values in enumerate(positions)
        ]
    )
    position_values = np.concatenate(positions, axis=0)
    resets = np.concatenate(reset_frames)
    source_id = canonical_fingerprint(
        {
            "kind": "openptv-tracks",
            "frame_id": frame_id,
            "times": times_.tolist(),
            "files": [_file_id(path, "openptv-record") for path in rt_paths + ptv_paths],
            "content": array_tree_fingerprint(
                {
                    "track_ids": track_ids,
                    "times": sample_times,
                    "positions": position_values,
                }
            ),
        }
    )
    records = OpenPTVTrackRecords(
        track_ids,
        sample_times,
        position_values,
        np.ones(track_ids.shape, dtype=bool),
        resets,
        frame_id=frame_id,
        source_id=source_id,
    )
    losses = (
        AdapterLoss(
            "link_candidate_evidence",
            "import",
            "unsupported",
            "Legacy ptv_is stores selected predecessor/successor links but no alternative association evidence.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "track_identity",
            "import",
            "synthesized",
            "Stable track IDs were deterministically reconstructed from reciprocal frame-local links.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "OpenPTV-rt_is+ptv_is",
        "OpenPTVTrackRecords",
        source_id=source_id,
        target_id=source_id,
        coordinate_mapping=("x -> x", "y -> y", "z -> z"),
        preserved_fields=("times", "positions_xyz", "predecessor", "successor"),
        assumptions=(f"frame_id={frame_id}",),
        losses=losses,
    )
    return records, report


def write_openptv_tracks(
    reconstruction_base: str | Path,
    linkage_base: str | Path,
    records: OpenPTVTrackRecords,
    /,
    *,
    frame_numbers: Sequence[int],
    lossless: bool = False,
) -> tuple[tuple[Path, ...], tuple[Path, ...], AdapterReport]:
    """Write link-derived rt_is/ptv_is sequences, explicitly dropping camera tuples."""
    valid = np.asarray(records.valid)
    tracks = np.asarray(records.track_ids)[valid]
    times = np.asarray(records.times)[valid]
    positions = np.asarray(records.positions_xyz)[valid]
    resets = np.asarray(records.reset_before)[valid]
    unique_times = np.unique(times)
    frame_numbers_ = tuple(int(value) for value in frame_numbers)
    if len(frame_numbers_) != unique_times.size:
        raise ValueError("frame_numbers must contain one value per unique track time.")
    indices_by_frame = [np.flatnonzero(times == time) for time in unique_times]
    local_index = [
        {int(sample): local for local, sample in enumerate(indices)}
        for indices in indices_by_frame
    ]
    rt_paths: list[Path] = []
    ptv_paths: list[Path] = []
    for frame, (number, indices) in enumerate(
        zip(frame_numbers_, indices_by_frame, strict=True)
    ):
        prev = np.full(indices.size, -1, dtype=int)
        next_ = np.full(indices.size, -2, dtype=int)
        for local, sample in enumerate(indices):
            same_track = tracks == tracks[sample]
            if frame > 0 and not resets[sample]:
                candidate = np.flatnonzero(
                    same_track & (times == unique_times[frame - 1])
                )
                if candidate.size == 1:
                    prev[local] = local_index[frame - 1][int(candidate[0])]
            if frame + 1 < unique_times.size:
                candidate = np.flatnonzero(
                    same_track & (times == unique_times[frame + 1]) & ~resets
                )
                if candidate.size == 1:
                    next_[local] = local_index[frame + 1][int(candidate[0])]
        rt_path = Path(f"{reconstruction_base}.{number}")
        ptv_path = Path(f"{linkage_base}.{number}")
        rt_table = np.column_stack(
            (
                np.arange(1, indices.size + 1),
                positions[indices],
                np.full((indices.size, 4), -1, dtype=int),
            )
        )
        link_table = np.column_stack((prev, next_, positions[indices]))
        _write_counted_table(
            rt_path, rt_table, fmt=("%d", "%.9g", "%.9g", "%.9g", "%d", "%d", "%d", "%d")
        )
        _write_counted_table(
            ptv_path, link_table, fmt=("%d", "%d", "%.9g", "%.9g", "%.9g")
        )
        rt_paths.append(rt_path)
        ptv_paths.append(ptv_path)
    target_id = canonical_fingerprint(
        {
            "rt": [_file_id(path, "rt_is") for path in rt_paths],
            "ptv": [_file_id(path, "ptv_is") for path in ptv_paths],
        }
    )
    losses = (
        AdapterLoss(
            "target_indices",
            "export",
            "synthesized",
            "OpenPTV rt_is requires four camera target indices; unavailable indices were written as -1.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "track_ids",
            "export",
            "transformed",
            "Stable identities were represented indirectly by frame-local predecessor/successor links.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "uncertainty_and_association_evidence",
            "export",
            "unsupported",
            "Legacy records cannot encode native covariance or association alternatives.",
            changes_interpretation=True,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "OpenPTVTrackRecords",
        "OpenPTV-rt_is+ptv_is",
        source_id=records.source_id,
        target_id=target_id,
        coordinate_mapping=("x -> x", "y -> y", "z -> z"),
        preserved_fields=("times", "positions_xyz", "identity links", "reset boundaries"),
        losses=losses,
    )
    if lossless:
        require_lossless(report)
    return tuple(rt_paths), tuple(ptv_paths), report


def _counted_table(path: Path, /, *, columns: int) -> tuple[int, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines[0].split()) != 1:
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE, f"{path} lacks an exact count header."
        )
    count_value = float(lines[0])
    if (
        not np.isfinite(count_value)
        or count_value < 0
        or count_value != np.floor(count_value)
    ):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE, f"{path} has an invalid count header."
        )
    count = int(count_value)
    nonempty = [line for line in lines[1:] if line.strip()]
    if len(nonempty) != count:
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE, f"{path} count does not match its records."
        )
    if count == 0:
        return 0, np.empty((0, columns), dtype=float)
    table = np.loadtxt(nonempty, dtype=float, ndmin=2)
    if table.shape != (count, columns) or not np.all(np.isfinite(table)):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE, f"{path} has malformed or nonfinite records."
        )
    return count, table


def _write_counted_table(
    path: Path, table: np.ndarray, /, *, fmt: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = str(table.shape[0])
    if table.shape[0] == 0:
        path.write_text(header + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as stream:
        stream.write(header + "\n")
        np.savetxt(stream, table, fmt=fmt, delimiter=" ")


def _integers(values: np.ndarray, owner: str, /) -> np.ndarray:
    values_ = np.asarray(values)
    if np.any(values_ != np.floor(values_)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE, f"OpenPTV {owner} must be integers."
        )
    return values_.astype(np.int64)


def _file_id(path: Path, format_name: str, /) -> str:
    return f"{format_name}:sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


__all__ = [
    "OpenPTVReconstructionRecords",
    "OpenPTVTargetRecords",
    "OpenPTVTrackRecords",
    "read_openptv_reconstruction",
    "read_openptv_targets",
    "read_openptv_tracks",
    "write_openptv_reconstruction",
    "write_openptv_targets",
    "write_openptv_tracks",
]
