#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, host-only broad phase for conservative geometry coupling.

Coupling requires the complete set of AABB candidates: dropping a pair because
it is not among a nearest-item beam is not a safe failure mode.  Candidates come
from the exact host simultaneous traversal of two float64 packed BVHs, which
enumerates every overlapping pair in bounded blocks before the complete result
is materialized.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from enum import Enum
from typing import Any, Iterable

import numpy as np

from .._bvh import bvh_overlap_pair_blocks, PackedBVH, prepare_bvh


class OverlapSearchStatus(str, Enum):
    """Terminal state of an exhaustive AABB search."""

    SUCCESS = "success"
    INVALID_BOUNDS = "invalid_bounds"


@dataclasses.dataclass(frozen=True, slots=True)
class OverlapTolerance:
    """Absolute/relative tolerance used by host AABB predicates."""

    absolute: float = 0.0
    relative: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class AabbOverlapCandidate:
    """One source/target broad-phase candidate."""

    source_global_id: Any
    target_global_id: Any
    source_local_id: Any
    target_local_id: Any


@dataclasses.dataclass(frozen=True, slots=True)
class AabbOverlapQueryResult:
    """Result of :func:`query_host_aabb_overlaps`.

    The four ID arrays are aligned row-wise and sorted by source global ID,
    target global ID, source local ID, and target local ID.  They are empty for
    every non-success status.
    """

    status: OverlapSearchStatus
    source_global_ids: np.ndarray
    target_global_ids: np.ndarray
    source_local_ids: np.ndarray
    target_local_ids: np.ndarray
    content_identity: str
    candidate_count: int = 0
    message: str = ""

    @property
    def candidates(self) -> tuple[AabbOverlapCandidate, ...]:
        return tuple(
            AabbOverlapCandidate(sg, tg, sl, tl)
            for sg, tg, sl, tl in zip(
                self.source_global_ids,
                self.target_global_ids,
                self.source_local_ids,
                self.target_local_ids,
                strict=True,
            )
        )

    @property
    def pairs(self) -> np.ndarray:
        """Local source/target pair IDs as an ``(n, 2)`` array."""
        if self.candidate_count == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack((self.source_local_ids, self.target_local_ids))

    @property
    def valid(self) -> bool:
        return self.status is OverlapSearchStatus.SUCCESS


def _freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _id_sort_key(value: Any) -> tuple[str, str]:
    """Total ordering for ordinary scalar stable IDs, including strings."""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return (
            "int",
            str(int(value)).zfill(40) if int(value) >= 0 else f"-{abs(int(value)):039d}",
        )
    if isinstance(value, (np.floating, float)) and math.isfinite(float(value)):
        return ("float", repr(float(value)))
    if isinstance(value, (str, bytes)):
        return (
            type(value).__name__,
            value.decode() if isinstance(value, bytes) else value,
        )
    return (type(value).__name__, repr(value))


def _ids_for_hash(values: np.ndarray) -> list[str]:
    return [f"{type(v).__name__}:{v!r}" for v in values.tolist()]


def _fingerprint(
    kind: str,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    global_ids: np.ndarray,
    local_ids: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    include_zero_measure: bool = False,
    extra: Iterable[str] = (),
) -> str:
    payload = {
        "kind": kind,
        "bounds_min": np.asarray(bounds_min, dtype=np.float64).tolist(),
        "bounds_max": np.asarray(bounds_max, dtype=np.float64).tolist(),
        "global_ids": _ids_for_hash(global_ids),
        "local_ids": _ids_for_hash(local_ids),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "include_zero_measure": bool(include_zero_measure),
        "extra": tuple(extra),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_tolerance(
    tolerance: Any,
    absolute_tolerance: Any,
    relative_tolerance: Any,
) -> tuple[float, float]:
    if tolerance is not None:
        if isinstance(tolerance, OverlapTolerance):
            absolute_tolerance, relative_tolerance = (
                tolerance.absolute,
                tolerance.relative,
            )
        elif isinstance(tolerance, dict):
            absolute_tolerance = tolerance.get(
                "absolute", tolerance.get("absolute_tolerance", 0.0)
            )
            relative_tolerance = tolerance.get(
                "relative", tolerance.get("relative_tolerance", 0.0)
            )
        elif isinstance(tolerance, (tuple, list)) and len(tolerance) == 2:
            absolute_tolerance, relative_tolerance = tolerance
        elif hasattr(tolerance, "absolute") or hasattr(tolerance, "absolute_tolerance"):
            absolute_tolerance = getattr(
                tolerance, "absolute", getattr(tolerance, "absolute_tolerance", 0.0)
            )
            relative_tolerance = getattr(
                tolerance, "relative", getattr(tolerance, "relative_tolerance", 0.0)
            )
        else:
            absolute_tolerance, relative_tolerance = tolerance, 0.0
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if not math.isfinite(atol) or atol < 0.0 or not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError("tolerances must be finite and non-negative")
    return atol, rtol


def _coerce_ids(values: Any, count: int, name: str) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=np.int64)
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.shape[0] != count:
        raise ValueError(f"{name} must have shape ({count},).")
    if ids.dtype.kind in "fc" and (not np.all(np.isfinite(ids))):
        raise ValueError(f"{name} must contain finite values.")
    if ids.dtype.kind == "f" and not np.all(ids == np.floor(ids)):
        raise ValueError(f"{name} must contain integral values when numeric.")
    if ids.dtype.kind not in "biufcOSU":
        raise ValueError(f"{name} must contain scalar stable IDs.")
    ids = np.array(ids, copy=True)
    # Duplicate IDs would make deterministic pair ordering and identity
    # ambiguous.  Equality is intentionally checked without requiring IDs to
    # be mutually comparable (mixed scalar object arrays are supported).
    if len({_id_sort_key(item) for item in ids.tolist()}) != count:
        raise ValueError(f"{name} must be unique.")
    return ids


def _coerce_bounds(bounds_min: Any, bounds_max: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    if lower.ndim != 2 or upper.ndim != 2:
        raise ValueError("bounds must be rank-2 arrays with shape (n, dimension).")
    if lower.shape != upper.shape:
        raise ValueError("bounds_min and bounds_max must have matching shapes.")
    if lower.shape[1] == 0:
        raise ValueError("bounds must have a positive dimension.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must contain only finite values.")
    if np.any(lower > upper):
        raise ValueError("every bounds_min component must be <= bounds_max.")
    return lower, upper


def _invalid_bvh(
    status: OverlapSearchStatus,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    message: str,
) -> "HostAabbOverlapBvh":
    empty = np.empty((0, 0), dtype=np.float64)
    ids = np.empty((0,), dtype=np.int64)
    return HostAabbOverlapBvh(
        bbox_min=_freeze_array(empty),
        bbox_max=_freeze_array(empty),
        target_global_ids=_freeze_array(ids),
        target_local_ids=_freeze_array(ids),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        content_identity=hashlib.sha256(
            f"invalid:{status.value}:{message}".encode()
        ).hexdigest(),
        status=status,
        message=message,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HostAabbOverlapBvh:
    """Immutable host-side AABB index used for exhaustive overlap queries."""

    bbox_min: np.ndarray
    bbox_max: np.ndarray
    target_global_ids: np.ndarray
    target_local_ids: np.ndarray
    absolute_tolerance: float
    relative_tolerance: float
    content_identity: str
    status: OverlapSearchStatus = OverlapSearchStatus.SUCCESS
    message: str = ""
    packed_bvh: PackedBVH | None = None

    @property
    def item_bbox_min(self) -> np.ndarray:
        return self.bbox_min

    @property
    def item_bbox_max(self) -> np.ndarray:
        return self.bbox_max

    @property
    def dimension(self) -> int:
        return self.bbox_min.shape[1] if self.bbox_min.ndim == 2 else 0

    @property
    def size(self) -> int:
        return self.bbox_min.shape[0] if self.bbox_min.ndim == 2 else 0


def build_host_aabb_overlap_bvh(
    bounds_min: Any = None,
    bounds_max: Any = None,
    target_global_ids: Any = None,
    target_local_ids: Any = None,
    *,
    tolerance: Any = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> HostAabbOverlapBvh:
    """Build an immutable host AABB index.

    The third and fourth positional arguments are target global and local IDs.
    Malformed bounds or IDs produce an invalid index rather than
    an index that could accidentally emit candidates.
    """
    try:
        atol, rtol = _resolve_tolerance(tolerance, absolute_tolerance, relative_tolerance)
        lower, upper = _coerce_bounds(bounds_min, bounds_max)
        gids = _coerce_ids(target_global_ids, lower.shape[0], "target_global_ids")
        lids = _coerce_ids(target_local_ids, lower.shape[0], "target_local_ids")
    except (TypeError, ValueError, OverflowError) as exc:
        try:
            atol = float(absolute_tolerance)
        except (TypeError, ValueError):
            atol = 0.0
        try:
            rtol = float(relative_tolerance)
        except (TypeError, ValueError):
            rtol = 0.0
        status = OverlapSearchStatus.INVALID_BOUNDS
        return _invalid_bvh(
            status, absolute_tolerance=atol, relative_tolerance=rtol, message=str(exc)
        )

    # Canonical storage makes repeated construction under a permutation have
    # identical content identity while preserving each item's supplied IDs.
    order = sorted(
        range(lower.shape[0]),
        key=lambda i: (_id_sort_key(gids[i]), _id_sort_key(lids[i])),
    )
    order_np = np.asarray(order, dtype=np.int64)
    lower = _freeze_array(lower[order_np])
    upper = _freeze_array(upper[order_np])
    gids = _freeze_array(gids[order_np])
    lids = _freeze_array(lids[order_np])
    identity = _fingerprint(
        "host-aabb-overlap-bvh",
        lower,
        upper,
        gids,
        lids,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
    )
    return HostAabbOverlapBvh(
        lower,
        upper,
        gids,
        lids,
        atol,
        rtol,
        identity,
        OverlapSearchStatus.SUCCESS,
        "",
        prepare_bvh(lower, upper, dtype=np.float64) if lower.shape[0] > 0 else None,
    )


def _empty_result(
    status: OverlapSearchStatus, identity: str, message: str = ""
) -> AabbOverlapQueryResult:
    empty = np.empty((0,), dtype=np.int64)
    return AabbOverlapQueryResult(
        status, empty, empty, empty, empty, identity, 0, message
    )


def query_host_aabb_overlaps(
    bvh: HostAabbOverlapBvh,
    bounds_min: Any,
    bounds_max: Any,
    source_global_ids: Any = None,
    source_local_ids: Any = None,
    *,
    tolerance: Any = None,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    include_zero_measure: bool = False,
) -> AabbOverlapQueryResult:
    """Return every positive-measure source/target AABB candidate pair.

    Search order and input permutation do not affect output order.  Every pair
    is returned.  ``include_zero_measure=True`` includes touching boxes (and
    records them as ordinary candidates) for algorithms whose conservative
    policy handles measure-zero intersections explicitly.
    """
    if not isinstance(bvh, HostAabbOverlapBvh):
        return _empty_result(
            OverlapSearchStatus.INVALID_BOUNDS, "", "bvh must be a HostAabbOverlapBvh"
        )
    if bvh.status is not OverlapSearchStatus.SUCCESS:
        return _empty_result(bvh.status, bvh.content_identity, bvh.message)
    try:
        default_atol = (
            bvh.absolute_tolerance if absolute_tolerance is None else absolute_tolerance
        )
        default_rtol = (
            bvh.relative_tolerance if relative_tolerance is None else relative_tolerance
        )
        atol, rtol = _resolve_tolerance(tolerance, default_atol, default_rtol)
        lower, upper = _coerce_bounds(bounds_min, bounds_max)
        if lower.shape[1] != bvh.dimension:
            raise ValueError("source bounds dimension must match the BVH dimension")
        sgids = _coerce_ids(source_global_ids, lower.shape[0], "source_global_ids")
        slids = _coerce_ids(source_local_ids, lower.shape[0], "source_local_ids")
    except (TypeError, ValueError, OverflowError) as exc:
        return _empty_result(
            OverlapSearchStatus.INVALID_BOUNDS,
            bvh.content_identity,
            str(exc),
        )

    # A query-specific identity includes all policy and source content.
    source_order = sorted(
        range(lower.shape[0]),
        key=lambda i: (_id_sort_key(sgids[i]), _id_sort_key(slids[i])),
    )
    source_order_np = np.asarray(source_order, dtype=np.int64)
    source_lower = lower[source_order_np]
    source_upper = upper[source_order_np]
    sgids = sgids[source_order_np]
    slids = slids[source_order_np]
    query_identity = _fingerprint(
        "host-aabb-overlap-query",
        source_lower,
        source_upper,
        sgids,
        slids,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        include_zero_measure=include_zero_measure,
        extra=(bvh.content_identity,),
    )

    source_items: list[np.ndarray] = []
    target_items: list[np.ndarray] = []
    count = 0
    if source_lower.shape[0] > 0 and bvh.packed_bvh is not None:
        # Tolerances only widen zero-measure inclusion; positive-measure overlap
        # is strict on every axis.
        blocks = bvh_overlap_pair_blocks(
            prepare_bvh(source_lower, source_upper, dtype=np.float64),
            bvh.packed_bvh,
            include_touching=include_zero_measure,
            absolute_tolerance=atol if include_zero_measure else 0.0,
            relative_tolerance=rtol if include_zero_measure else 0.0,
        )
        for block_source, block_target in blocks:
            count += block_source.shape[0]
            source_items.append(block_source)
            target_items.append(block_target)

    # Both sides are stored in canonical unique-ID order, so ordering rows by
    # (source index, target index) is the canonical record order.
    source_index = (
        np.concatenate(source_items) if source_items else np.empty((0,), dtype=np.int64)
    )
    target_index = (
        np.concatenate(target_items) if target_items else np.empty((0,), dtype=np.int64)
    )
    order = np.lexsort((target_index, source_index))
    source_index = source_index[order]
    target_index = target_index[order]
    source_global = _freeze_array(sgids[source_index])
    target_global = _freeze_array(bvh.target_global_ids[target_index])
    source_local = _freeze_array(slids[source_index])
    target_local = _freeze_array(bvh.target_local_ids[target_index])
    return AabbOverlapQueryResult(
        status=OverlapSearchStatus.SUCCESS,
        source_global_ids=source_global,
        target_global_ids=target_global,
        source_local_ids=source_local,
        target_local_ids=target_local,
        content_identity=query_identity,
        candidate_count=count,
    )


__all__ = [
    "AabbOverlapCandidate",
    "AabbOverlapQueryResult",
    "HostAabbOverlapBvh",
    "OverlapSearchStatus",
    "OverlapTolerance",
    "build_host_aabb_overlap_bvh",
    "query_host_aabb_overlaps",
]
