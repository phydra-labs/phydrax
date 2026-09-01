#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ...discretization import AxisDiscretization, AxisDomain, PreparedTensorGrid
from ...dynamics import StateLayout, TrajectoryData
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ...stochastic import ObservationSequence
from ..piv import PhysicalPIVResult2D
from ..tracking import to_trajectory_data as native_tracks_to_trajectory_data, TrackResult
from ._openptv import OpenPTVTrackRecords


PhysicalPIVValue = Literal["displacement", "velocity"]


def piv_to_tensor_grid(
    field: PhysicalPIVResult2D,
    /,
    *,
    value: PhysicalPIVValue = "velocity",
) -> tuple[PreparedTensorGrid, object, np.ndarray, np.ndarray, AdapterReport]:
    """Map a calibrated rectilinear PIV field to native physical grid support."""
    y, x, values, valid = _physical_field_arrays(field, value=value)
    y_axis = _uniform_axis(y)
    x_axis = _uniform_axis(x)
    grid = PreparedTensorGrid((y_axis, x_axis), axis_names=("y", "x"))
    space = grid.field_space(
        "piv-" + value,
        component_shape=(2,),
        dtype=values.dtype,
        representation="point_value",
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "PhysicalPIVResult2D",
        "PreparedTensorGrid+DiscreteFieldSpace",
        source_id=field.source_field_id,
        target_id=grid.prepared_id,
        coordinate_mapping=(
            "physical y -> tensor axis y",
            "physical x -> tensor axis x",
            f"({value}_x,{value}_y) -> component axis",
        ),
        preserved_fields=("positions_xy", f"{value}_xy", "valid", "transform_id"),
        assumptions=(
            f"spatial_unit={field.spatial_unit}",
            f"time_unit={field.time_unit}",
        ),
    )
    return grid, space, values, valid, report


def piv_to_observation_sequence(
    fields: PhysicalPIVResult2D | Sequence[PhysicalPIVResult2D],
    times,
    /,
    *,
    value: PhysicalPIVValue = "velocity",
    sequence_id: str = "piv-observations",
) -> tuple[ObservationSequence, PreparedTensorGrid, AdapterReport]:
    """Create masked Eulerian observations without converting invalid vectors to zeros."""
    fields_ = (fields,) if isinstance(fields, PhysicalPIVResult2D) else tuple(fields)
    if not fields_ or not all(
        isinstance(field, PhysicalPIVResult2D) for field in fields_
    ):
        raise TypeError("fields must contain one or more PhysicalPIVResult2D values.")
    times_ = np.asarray(times, dtype=float).reshape((-1,))
    if times_.shape != (len(fields_),) or not np.all(np.isfinite(times_)):
        raise ValueError("times must contain one finite value per PIV field.")
    if len(fields_) > 1 and np.any(np.diff(times_) <= 0.0):
        raise ValueError("PIV observation times must be strictly increasing.")
    grid, _, first_values, first_valid, grid_report = piv_to_tensor_grid(
        fields_[0], value=value
    )
    arrays = [(first_values, first_valid)]
    first_positions = np.asarray(fields_[0].positions_xy)
    for field in fields_[1:]:
        if not np.array_equal(np.asarray(field.positions_xy), first_positions):
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "PIV observation sequences require one unchanged physical grid; implicit resampling is forbidden.",
            )
        _, _, values, valid = _physical_field_arrays(field, value=value)
        arrays.append((values, valid))
    values = np.stack([np.where(valid[..., None], item, 0.0) for item, valid in arrays])
    mask = np.stack(
        [np.broadcast_to(valid[..., None], item.shape) for item, valid in arrays]
    )
    source_id = hashlib.sha256(
        "|".join(field.source_field_id for field in fields_).encode("utf-8")
    ).hexdigest()
    sequence = ObservationSequence(
        times_,
        values,
        observation_axes=("y", "x", "component"),
        step_valid=np.ones(times_.shape, dtype=bool),
        observation_mask=mask,
        sequence_id=sequence_id,
        sensor_id=source_id,
        discretization_id=grid.prepared_id,
        approximation_id=fields_[0].transform_id,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "PhysicalPIVResult2D-sequence",
        "ObservationSequence",
        source_id=source_id,
        target_id=sequence.sequence_id,
        coordinate_mapping=grid_report.coordinate_mapping,
        preserved_fields=(f"{value}_xy", "valid", "times", "transform_id"),
        assumptions=(
            f"spatial_unit={fields_[0].spatial_unit}",
            f"time_unit={fields_[0].time_unit}",
        ),
    )
    return sequence, grid, report


def tracks_to_trajectory_data(
    tracks: OpenPTVTrackRecords | TrackResult,
    /,
    *,
    source_id: str | None = None,
    dataset_id: str | None = None,
) -> tuple[TrajectoryData, AdapterReport]:
    """Pack identity-bearing physical tracks while preserving every reset boundary."""
    if isinstance(tracks, TrackResult):
        if source_id is not None or dataset_id is not None:
            raise ValueError(
                "Native TrackResult conversion owns its canonical source and dataset identities."
            )
        trajectory = native_tracks_to_trajectory_data(tracks)
        report = AdapterReport(
            AdapterStatus.LOSSLESS,
            "TrackResult",
            "TrajectoryData",
            source_id=tracks.result_id,
            target_id=trajectory.dataset_id,
            coordinate_mapping=(
                "track slot -> case axis track",
                "time -> coordinates",
                "(x,y,z,vx,vy,vz) -> state component",
            ),
            preserved_fields=(
                "times",
                "states",
                "observed",
                "track_ids",
                "resets",
            ),
        )
        return trajectory, report
    if not isinstance(tracks, OpenPTVTrackRecords):
        raise TypeError("tracks must be OpenPTVTrackRecords or TrackResult.")
    coordinates, states, sample_valid, transition_valid, reset_mask, track_ids = (
        _pack_tracks(tracks)
    )
    source = tracks.source_id if source_id is None else str(source_id)
    dataset = (
        "tracks:"
        + hashlib.sha256(
            (source + ":" + ",".join(map(str, track_ids))).encode("utf-8")
        ).hexdigest()
        if dataset_id is None
        else str(dataset_id)
    )
    layout = StateLayout(
        (3,),
        axes=("component",),
        component_names=("x", "y", "z"),
    )
    trajectory = TrajectoryData(
        coordinates,
        states,
        state_layout=layout,
        sample_valid=sample_valid,
        transition_valid=transition_valid,
        reset_mask=reset_mask,
        weights=sample_valid.astype(float),
        case_axes=("track",),
        case_axis_roles=("case",),
        coordinate_id="time",
        source_id=source,
        dataset_id=dataset,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenPTVTrackRecords",
        "TrajectoryData",
        source_id=tracks.source_id,
        target_id=trajectory.dataset_id,
        coordinate_mapping=(
            "track_id -> case axis track",
            "time -> coordinates",
            "(x,y,z) -> state component",
        ),
        preserved_fields=("track_ids", "times", "positions_xyz", "valid", "reset_before"),
        assumptions=(f"frame_id={tracks.frame_id}",),
    )
    return trajectory, report


def tracks_to_observation_sequence(
    tracks: OpenPTVTrackRecords | TrackResult,
    /,
    *,
    sequence_id: str = "track-observations",
    allow_reset_loss: bool = False,
) -> tuple[ObservationSequence, AdapterReport]:
    """Pack track positions as masked observations, requiring opt-in if resets are lost."""
    if isinstance(tracks, TrackResult):
        return _native_tracks_to_observations(
            tracks,
            sequence_id=sequence_id,
            allow_reset_loss=allow_reset_loss,
        )
    if not isinstance(tracks, OpenPTVTrackRecords):
        raise TypeError("tracks must be OpenPTVTrackRecords or TrackResult.")
    coordinates, states, sample_valid, _, reset_mask, track_ids = _pack_tracks(tracks)
    has_resets = bool(np.any(reset_mask))
    if has_resets and not allow_reset_loss:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "ObservationSequence cannot encode track reset boundaries; pass "
            "allow_reset_loss=True only when the observation model does not need them.",
        )
    mask = np.broadcast_to(sample_valid[..., None], states.shape)
    sequence = ObservationSequence(
        coordinates,
        states,
        case_axes=("track",),
        case_shape=(len(track_ids),),
        observation_axes=("component",),
        step_valid=sample_valid,
        observation_mask=mask,
        case_ids=tuple(f"track:{track_id}" for track_id in track_ids),
        sequence_id=sequence_id,
        sensor_id=tracks.source_id,
        approximation_id=tracks.frame_id,
    )
    losses = (
        (
            AdapterLoss(
                "reset_before",
                "export",
                "dropped",
                "ObservationSequence has no reset boundary field; the caller explicitly allowed this loss.",
                changes_interpretation=True,
            ),
        )
        if has_resets
        else ()
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if has_resets else AdapterStatus.LOSSLESS,
        "OpenPTVTrackRecords",
        "ObservationSequence",
        source_id=tracks.source_id,
        target_id=sequence.sequence_id,
        coordinate_mapping=(
            "track_id -> case axis track",
            "time -> step time",
            "(x,y,z) -> observation component",
        ),
        preserved_fields=("track_ids", "times", "positions_xyz", "valid"),
        losses=losses,
    )
    return sequence, report


def _native_tracks_to_observations(
    tracks: TrackResult,
    /,
    *,
    sequence_id: str,
    allow_reset_loss: bool,
) -> tuple[ObservationSequence, AdapterReport]:
    times = np.asarray(tracks.times, dtype=float)
    track_ids = np.asarray(tracks.track_ids, dtype=np.int64)
    observed = np.asarray(tracks.observed, dtype=bool)
    observations = np.asarray(tracks.observations)
    if (
        times.ndim != 1
        or track_ids.shape != observed.shape
        or observations.shape != observed.shape + (3,)
        or track_ids.shape[1] != times.size
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "TrackResult arrays are inconsistent with its time capacity.",
        )
    identities = tuple(
        int(value) for value in np.unique(track_ids[observed & (track_ids >= 0)])
    )
    if not identities:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "TrackResult has no observed identity-bearing samples.",
        )
    values = np.zeros((len(identities), times.size, 3), dtype=observations.dtype)
    mask = np.zeros(values.shape, dtype=bool)
    for case, identity in enumerate(identities):
        for step in range(times.size):
            slots = np.flatnonzero(observed[:, step] & (track_ids[:, step] == identity))
            if slots.size > 1:
                raise AdapterError(
                    AdapterStatus.INCONSISTENT_SOURCE,
                    "TrackResult assigns one stable identity to multiple slots at one time.",
                )
            if slots.size == 1:
                values[case, step] = observations[slots[0], step]
                mask[case, step] = True
    has_resets = bool(np.any(np.asarray(tracks.resets, dtype=bool)))
    if has_resets and not allow_reset_loss:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "ObservationSequence cannot encode TrackResult reset boundaries; pass "
            "allow_reset_loss=True only when the observation model does not need them.",
        )
    sequence = ObservationSequence(
        times,
        values,
        case_axes=("track",),
        case_shape=(len(identities),),
        observation_axes=("component",),
        step_valid=np.ones((len(identities), times.size), dtype=bool),
        observation_mask=mask,
        case_ids=tuple(f"track:{identity}" for identity in identities),
        sequence_id=sequence_id,
        sensor_id=tracks.result_id,
        approximation_id=tracks.result_id,
    )
    losses = (
        (
            AdapterLoss(
                "resets",
                "export",
                "dropped",
                "ObservationSequence has no reset boundary field; the caller explicitly allowed this loss.",
                changes_interpretation=True,
            ),
        )
        if has_resets
        else ()
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if has_resets else AdapterStatus.LOSSLESS,
        "TrackResult",
        "ObservationSequence",
        source_id=tracks.result_id,
        target_id=sequence.sequence_id,
        coordinate_mapping=(
            "stable track_id -> case axis track",
            "time -> step time",
            "observed (x,y,z) -> observation component",
        ),
        preserved_fields=("times", "observations", "observed", "track_ids"),
        losses=losses,
    )
    return sequence, report


def _physical_field_arrays(
    field: PhysicalPIVResult2D,
    /,
    *,
    value: PhysicalPIVValue,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(field, PhysicalPIVResult2D):
        raise TypeError("A right-handed calibrated PhysicalPIVResult2D is required.")
    if value not in ("displacement", "velocity"):
        raise ValueError("value must be 'displacement' or 'velocity'.")
    positions = np.asarray(field.positions_xy)
    values = np.asarray(
        field.displacement_xy if value == "displacement" else field.velocity_xy
    )
    valid = np.asarray(field.valid, dtype=bool)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 2
        or values.shape != positions.shape
        or valid.shape != positions.shape[:-1]
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Physical PIV arrays have inconsistent grid shapes.",
        )
    x = positions[..., 0]
    y = positions[..., 1]
    x_axis = x[0, :]
    y_axis = y[:, 0]
    if not (
        np.allclose(x, np.broadcast_to(x_axis[None, :], valid.shape), rtol=0.0, atol=0.0)
        and np.allclose(
            y, np.broadcast_to(y_axis[:, None], valid.shape), rtol=0.0, atol=0.0
        )
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Physical PIV positions are not rectilinear; implicit interpolation is forbidden.",
        )
    if x_axis.size < 2 or y_axis.size < 2:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Tensor-grid adaptation requires at least two samples per physical axis.",
        )
    if np.all(np.diff(y_axis) < 0.0):
        y_axis = y_axis[::-1]
        values = values[::-1]
        valid = valid[::-1]
    elif not np.all(np.diff(y_axis) > 0.0):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Physical y coordinates must be strictly monotone.",
        )
    if np.all(np.diff(x_axis) < 0.0):
        x_axis = x_axis[::-1]
        values = values[:, ::-1]
        valid = valid[:, ::-1]
    elif not np.all(np.diff(x_axis) > 0.0):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Physical x coordinates must be strictly monotone.",
        )
    if np.any(valid & ~np.all(np.isfinite(values), axis=-1)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every valid physical PIV vector must be finite.",
        )
    values = np.where(valid[..., None], values, 0.0)
    return y_axis, x_axis, values, valid


def _uniform_axis(nodes: np.ndarray, /) -> AxisDiscretization:
    differences = np.diff(nodes)
    if not np.allclose(differences, differences[0], rtol=1e-8, atol=1e-12):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "PreparedTensorGrid adaptation currently requires uniform PIV axes; implicit resampling is forbidden.",
        )
    weights = np.full(nodes.shape, differences[0], dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return AxisDiscretization(
        nodes=nodes,
        quad_weights=weights,
        basis="uniform",
        domain=AxisDomain.interval(nodes[0], nodes[-1]),
        lower_endpoint_included=True,
        upper_endpoint_included=True,
    )


def _pack_tracks(
    tracks: OpenPTVTrackRecords,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    active = np.asarray(tracks.valid, dtype=bool)
    ids = np.asarray(tracks.track_ids)[active]
    times = np.asarray(tracks.times)[active]
    positions = np.asarray(tracks.positions_xyz)[active]
    resets = np.asarray(tracks.reset_before)[active]
    track_ids = tuple(int(value) for value in np.unique(ids))
    if not track_ids:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Track adaptation requires at least one valid sample.",
        )
    counts = [int(np.count_nonzero(ids == track_id)) for track_id in track_ids]
    capacity = max(2, max(counts))
    coordinates = np.zeros((len(track_ids), capacity), dtype=float)
    states = np.zeros((len(track_ids), capacity, 3), dtype=float)
    sample_valid = np.zeros((len(track_ids), capacity), dtype=bool)
    reset_mask = np.zeros((len(track_ids), capacity - 1), dtype=bool)
    for case, track_id in enumerate(track_ids):
        selected = np.flatnonzero(ids == track_id)
        order = np.argsort(times[selected], kind="stable")
        selected = selected[order]
        count = selected.size
        coordinates[case, :count] = times[selected]
        states[case, :count] = positions[selected]
        sample_valid[case, :count] = True
        if count > 1:
            reset_mask[case, : count - 1] = resets[selected[1:]]
            step = times[selected[-1]] - times[selected[-2]]
        else:
            step = 1.0
        for index in range(count, capacity):
            coordinates[case, index] = coordinates[case, index - 1] + step
    transition_valid = sample_valid[:, :-1] & sample_valid[:, 1:] & ~reset_mask
    return coordinates, states, sample_valid, transition_valid, reset_mask, track_ids


__all__ = [
    "PhysicalPIVValue",
    "piv_to_observation_sequence",
    "piv_to_tensor_grid",
    "tracks_to_observation_sequence",
    "tracks_to_trajectory_data",
]
