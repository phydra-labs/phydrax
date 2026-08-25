#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, host-only broad phase for conservative geometry coupling.

This module deliberately does not use the JAX nearest-neighbour BVH.  Coupling
requires an exhaustive set of AABB candidates: dropping a pair because it is
not among a nearest-item beam is not a safe failure mode.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from enum import Enum
from typing import Any, Iterable

import numpy as np


class OverlapSearchStatus(str, Enum):
    """Terminal state of an exhaustive AABB search.

    A limit or malformed input always returns an empty result.  In particular,
    a result with a non-success status must never be interpreted as a partial
    candidate list.
    """

    SUCCESS = "success"
    INVALID_BOUNDS = "invalid_bounds"
    INVALID_LIMIT = "invalid_limit"
    CANDIDATE_LIMIT = "candidate_limit"
    MEMORY_LIMIT = "memory_limit"
    TIME_LIMIT = "time_limit"

    # Descriptive spellings retained as enum aliases for callers that prefer
    # explicit ``*_EXCEEDED`` names.
    CANDIDATE_LIMIT_EXCEEDED = "candidate_limit"
    MEMORY_LIMIT_EXCEEDED = "memory_limit"
    TIME_LIMIT_EXCEEDED = "time_limit"
    INVALID_INPUT = "invalid_bounds"


@dataclasses.dataclass(frozen=True, slots=True)
class OverlapTolerance:
    """Absolute/relative tolerance used by host AABB predicates."""

    absolute: float = 0.0
    relative: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class OverlapSearchLimits:
    """Fail-closed resource limits for one host overlap search."""

    max_candidates: int | None = None
    max_memory_bytes: int | None = None
    max_time_seconds: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class AabbOverlapCandidate:
    """One source/target broad-phase candidate."""

    source_global_id: Any
    target_global_id: Any
    source_local_id: Any
    target_local_id: Any

    @property
    def source_id(self) -> Any:
        return self.source_global_id

    @property
    def target_id(self) -> Any:
        return self.target_global_id


@dataclasses.dataclass(frozen=True, slots=True)
class AabbOverlapQueryResult:
    """Result of :func:`query_host_aabb_overlaps`.

    The four ID arrays are aligned row-wise and sorted by source global ID,
    target global ID, source local ID, and target local ID.  They are empty for
    every non-success status, including a search that hit a resource limit.
    """

    status: OverlapSearchStatus
    source_global_ids: np.ndarray
    target_global_ids: np.ndarray
    source_local_ids: np.ndarray
    target_local_ids: np.ndarray
    content_identity: str
    candidate_count: int = 0
    estimated_memory_bytes: int = 0
    elapsed_seconds: float = 0.0
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

    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def source_ids(self) -> np.ndarray:
        return self.source_global_ids

    @property
    def target_ids(self) -> np.ndarray:
        return self.target_global_ids


# A shorter spelling is convenient in downstream type annotations.
OverlapQueryResult = AabbOverlapQueryResult


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
    limits: tuple[int | None, int | None, float | None] = (None, None, None),
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
        "limits": limits,
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


def _resolve_limits(
    limits: Any,
    max_candidates: int | None,
    max_memory_bytes: int | None,
    max_time_seconds: float | None,
) -> tuple[int | None, int | None, float | None]:
    if limits is not None:
        if isinstance(limits, OverlapSearchLimits):
            max_candidates, max_memory_bytes, max_time_seconds = (
                limits.max_candidates,
                limits.max_memory_bytes,
                limits.max_time_seconds,
            )
        elif isinstance(limits, dict):
            max_candidates = limits.get("max_candidates", limits.get("candidate_limit"))
            max_memory_bytes = limits.get("max_memory_bytes", limits.get("memory_limit"))
            max_time_seconds = limits.get("max_time_seconds", limits.get("time_limit"))
        else:
            max_candidates = getattr(
                limits, "max_candidates", getattr(limits, "candidate_limit", None)
            )
            max_memory_bytes = getattr(
                limits, "max_memory_bytes", getattr(limits, "memory_limit", None)
            )
            max_time_seconds = getattr(
                limits, "max_time_seconds", getattr(limits, "time_limit", None)
            )
    status, values = _coerce_limits(max_candidates, max_memory_bytes, max_time_seconds)
    if status is not None:
        raise ValueError("limits must be finite and non-negative integers/seconds")
    return values


def _coerce_limits(
    max_candidates: int | None,
    max_memory_bytes: int | None,
    max_time_seconds: float | None,
) -> tuple[OverlapSearchStatus | None, tuple[int | None, int | None, float | None]]:
    if max_candidates is not None:
        if (
            isinstance(max_candidates, bool)
            or int(max_candidates) != max_candidates
            or int(max_candidates) < 0
        ):
            return OverlapSearchStatus.INVALID_LIMIT, (None, None, None)
        max_candidates = int(max_candidates)
    if max_memory_bytes is not None:
        if (
            isinstance(max_memory_bytes, bool)
            or int(max_memory_bytes) != max_memory_bytes
            or int(max_memory_bytes) < 0
        ):
            return OverlapSearchStatus.INVALID_LIMIT, (None, None, None)
        max_memory_bytes = int(max_memory_bytes)
    if max_time_seconds is not None:
        try:
            max_time_seconds = float(max_time_seconds)
        except (TypeError, ValueError):
            return OverlapSearchStatus.INVALID_LIMIT, (None, None, None)
        if not math.isfinite(max_time_seconds) or max_time_seconds < 0.0:
            return OverlapSearchStatus.INVALID_LIMIT, (None, None, None)
    return None, (max_candidates, max_memory_bytes, max_time_seconds)


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
        global_ids=_freeze_array(ids),
        local_ids=_freeze_array(ids),
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
    global_ids: np.ndarray
    local_ids: np.ndarray
    absolute_tolerance: float
    relative_tolerance: float
    content_identity: str
    status: OverlapSearchStatus = OverlapSearchStatus.SUCCESS
    message: str = ""
    max_candidates: int | None = None
    max_memory_bytes: int | None = None
    max_time_seconds: float | None = None

    @property
    def item_bbox_min(self) -> np.ndarray:
        return self.bbox_min

    @property
    def item_bbox_max(self) -> np.ndarray:
        return self.bbox_max

    @property
    def item_global_ids(self) -> np.ndarray:
        return self.global_ids

    @property
    def item_local_ids(self) -> np.ndarray:
        return self.local_ids

    @property
    def dimension(self) -> int:
        return int(self.bbox_min.shape[1]) if self.bbox_min.ndim == 2 else 0

    @property
    def size(self) -> int:
        return int(self.bbox_min.shape[0]) if self.bbox_min.ndim == 2 else 0


def build_host_aabb_overlap_bvh(
    bounds_min: Any = None,
    bounds_max: Any = None,
    global_ids: Any = None,
    local_ids: Any = None,
    stable_global_ids: Any = None,
    item_global_ids: Any = None,
    item_local_ids: Any = None,
    *,
    tolerance: Any = None,
    limits: Any = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    max_candidates: int | None = None,
    max_memory_bytes: int | None = None,
    max_time_seconds: float | None = None,
) -> HostAabbOverlapBvh:
    """Build an immutable host AABB index.

    The third and fourth positional arguments are global and local IDs.  The
    keyword spellings ``stable_global_ids``, ``item_global_ids`` and
    ``item_local_ids`` are accepted to make the source/target role explicit at
    call sites.  Malformed bounds or limits produce an invalid index rather
    than an index that could accidentally emit candidates.
    """
    if stable_global_ids is not None:
        if global_ids is not None:
            return _invalid_bvh(
                OverlapSearchStatus.INVALID_BOUNDS,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                message="global IDs supplied more than once",
            )
        global_ids = stable_global_ids
    if item_global_ids is not None:
        if global_ids is not None:
            return _invalid_bvh(
                OverlapSearchStatus.INVALID_BOUNDS,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                message="global IDs supplied more than once",
            )
        global_ids = item_global_ids
    if item_local_ids is not None:
        if local_ids is not None:
            return _invalid_bvh(
                OverlapSearchStatus.INVALID_BOUNDS,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                message="local IDs supplied more than once",
            )
        local_ids = item_local_ids
    try:
        atol, rtol = _resolve_tolerance(tolerance, absolute_tolerance, relative_tolerance)
        if limits is not None:
            resolved_limits = _resolve_limits(limits, None, None, None)
        else:
            limit_status, resolved_limits = _coerce_limits(
                max_candidates, max_memory_bytes, max_time_seconds
            )
            if limit_status is not None:
                return _invalid_bvh(
                    limit_status,
                    absolute_tolerance=atol,
                    relative_tolerance=rtol,
                    message="limits must be finite and non-negative integers/seconds",
                )
        lower, upper = _coerce_bounds(bounds_min, bounds_max)
        gids = _coerce_ids(global_ids, lower.shape[0], "global_ids")
        lids = _coerce_ids(local_ids, lower.shape[0], "local_ids")
    except (TypeError, ValueError, OverflowError) as exc:
        try:
            atol = float(absolute_tolerance)
        except (TypeError, ValueError):
            atol = 0.0
        try:
            rtol = float(relative_tolerance)
        except (TypeError, ValueError):
            rtol = 0.0
        status = (
            OverlapSearchStatus.INVALID_LIMIT
            if "limits must" in str(exc)
            else OverlapSearchStatus.INVALID_BOUNDS
        )
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
        limits=resolved_limits,
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
        *resolved_limits,
    )


def _empty_result(
    status: OverlapSearchStatus, identity: str, message: str = "", *, elapsed: float = 0.0
) -> AabbOverlapQueryResult:
    empty = np.empty((0,), dtype=np.int64)
    return AabbOverlapQueryResult(
        status, empty, empty, empty, empty, identity, 0, 0, elapsed, message
    )


def query_host_aabb_overlaps(
    bvh: HostAabbOverlapBvh,
    bounds_min: Any,
    bounds_max: Any,
    source_global_ids: Any = None,
    source_local_ids: Any = None,
    stable_global_ids: Any = None,
    global_ids: Any = None,
    *,
    tolerance: Any = None,
    limits: Any = None,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    include_zero_measure: bool = False,
    max_candidates: int | None = None,
    max_memory_bytes: int | None = None,
    max_time_seconds: float | None = None,
) -> AabbOverlapQueryResult:
    """Return every positive-measure source/target AABB candidate pair.

    Search order and input permutation do not affect output order.  Limits are
    checked before returning any rows; a limit hit is therefore fail-closed.
    ``include_zero_measure=True`` includes touching boxes (and records them as
    ordinary candidates) for algorithms whose conservative policy handles
    measure-zero intersections explicitly.
    """
    started = time.perf_counter()
    if not isinstance(bvh, HostAabbOverlapBvh):
        return _empty_result(
            OverlapSearchStatus.INVALID_BOUNDS, "", "bvh must be a HostAabbOverlapBvh"
        )
    if bvh.status is not OverlapSearchStatus.SUCCESS:
        return _empty_result(bvh.status, bvh.content_identity, bvh.message)
    if stable_global_ids is not None:
        if source_global_ids is not None or global_ids is not None:
            return _empty_result(
                OverlapSearchStatus.INVALID_BOUNDS,
                bvh.content_identity,
                "source global IDs supplied more than once",
            )
        source_global_ids = stable_global_ids
    if global_ids is not None:
        if source_global_ids is not None:
            return _empty_result(
                OverlapSearchStatus.INVALID_BOUNDS,
                bvh.content_identity,
                "source global IDs supplied more than once",
            )
        source_global_ids = global_ids
    limit_status: OverlapSearchStatus | None = None
    try:
        default_atol = (
            bvh.absolute_tolerance if absolute_tolerance is None else absolute_tolerance
        )
        default_rtol = (
            bvh.relative_tolerance if relative_tolerance is None else relative_tolerance
        )
        atol, rtol = _resolve_tolerance(tolerance, default_atol, default_rtol)
        if limits is not None:
            resolved_limits = _resolve_limits(limits, None, None, None)
        else:
            limit_status, resolved_limits = _coerce_limits(
                max_candidates, max_memory_bytes, max_time_seconds
            )
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
            elapsed=time.perf_counter() - started,
        )
    if limit_status is not None:
        return _empty_result(
            limit_status,
            bvh.content_identity,
            "limits must be finite and non-negative integers/seconds",
            elapsed=time.perf_counter() - started,
        )

    # A query-specific identity includes all policy and source content.  It is
    # deliberately independent of elapsed wall-clock time.
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
        limits=resolved_limits,
        extra=(bvh.content_identity,),
    )

    inherited_limits = (bvh.max_candidates, bvh.max_memory_bytes, bvh.max_time_seconds)
    effective_candidates = (
        resolved_limits[0] if resolved_limits[0] is not None else bvh.max_candidates
    )
    effective_memory = (
        resolved_limits[1] if resolved_limits[1] is not None else bvh.max_memory_bytes
    )
    effective_time = (
        resolved_limits[2] if resolved_limits[2] is not None else bvh.max_time_seconds
    )
    if inherited_limits != (None, None, None):
        query_identity = _fingerprint(
            "host-aabb-overlap-query",
            source_lower,
            source_upper,
            sgids,
            slids,
            absolute_tolerance=atol,
            relative_tolerance=rtol,
            include_zero_measure=include_zero_measure,
            limits=(effective_candidates, effective_memory, effective_time),
            extra=(bvh.content_identity,),
        )

    records: list[tuple[Any, Any, Any, Any]] = []
    checks = 0
    # This is intentionally an exhaustive broad-phase loop.  No nearest-item
    # beam or fixed-width truncation is used.
    for si in range(source_lower.shape[0]):
        smin = source_lower[si]
        smax = source_upper[si]
        for ti in range(bvh.size):
            checks += 1
            if effective_time is not None and (checks == 1 or checks % 256 == 0):
                if time.perf_counter() - started >= effective_time:
                    return _empty_result(
                        OverlapSearchStatus.TIME_LIMIT,
                        query_identity,
                        "time limit exceeded",
                        elapsed=time.perf_counter() - started,
                    )
            tmin = bvh.bbox_min[ti]
            tmax = bvh.bbox_max[ti]
            scale = np.maximum(
                1.0,
                np.maximum(
                    np.abs(smin),
                    np.maximum(np.abs(smax), np.maximum(np.abs(tmin), np.abs(tmax))),
                ),
            )
            tol = atol + rtol * scale
            extent = np.minimum(smax, tmax) - np.maximum(smin, tmin)
            if include_zero_measure:
                hit = bool(np.all(extent >= -tol))
            else:
                # Tolerance protects positive-measure boundaries from roundoff,
                # but touching/measure-zero boxes stay out of ordinary remaps.
                hit = bool(np.all(extent > 0.0) and np.all(extent >= -tol))
            if not hit:
                continue
            next_count = len(records) + 1
            if effective_candidates is not None and next_count > effective_candidates:
                return _empty_result(
                    OverlapSearchStatus.CANDIDATE_LIMIT,
                    query_identity,
                    "candidate limit exceeded",
                    elapsed=time.perf_counter() - started,
                )
            # Four ID arrays are materialized in the result.  8 bytes per scalar
            # is a lower bound, so this estimate fails closed for object IDs too.
            estimated_memory = next_count * 32
            if effective_memory is not None and estimated_memory > effective_memory:
                return _empty_result(
                    OverlapSearchStatus.MEMORY_LIMIT,
                    query_identity,
                    "memory limit exceeded",
                    elapsed=time.perf_counter() - started,
                )
            records.append((sgids[si], bvh.global_ids[ti], slids[si], bvh.local_ids[ti]))

    records.sort(key=lambda row: tuple(_id_sort_key(value) for value in row))
    if records:
        source_global = _freeze_array(
            np.asarray([row[0] for row in records], dtype=sgids.dtype)
        )
        target_global = _freeze_array(
            np.asarray([row[1] for row in records], dtype=bvh.global_ids.dtype)
        )
        source_local = _freeze_array(
            np.asarray([row[2] for row in records], dtype=slids.dtype)
        )
        target_local = _freeze_array(
            np.asarray([row[3] for row in records], dtype=bvh.local_ids.dtype)
        )
    else:
        source_global = _freeze_array(np.empty((0,), dtype=sgids.dtype))
        target_global = _freeze_array(np.empty((0,), dtype=bvh.global_ids.dtype))
        source_local = _freeze_array(np.empty((0,), dtype=slids.dtype))
        target_local = _freeze_array(np.empty((0,), dtype=bvh.local_ids.dtype))
    elapsed = time.perf_counter() - started
    return AabbOverlapQueryResult(
        OverlapSearchStatus.SUCCESS,
        source_global,
        target_global,
        source_local,
        target_local,
        query_identity,
        len(records),
        len(records) * 32,
        elapsed,
        "",
    )


__all__ = [
    "AabbOverlapCandidate",
    "AabbOverlapQueryResult",
    "HostAabbOverlapBvh",
    "OverlapQueryResult",
    "OverlapSearchLimits",
    "OverlapSearchStatus",
    "OverlapTolerance",
    "build_host_aabb_overlap_bvh",
    "query_host_aabb_overlaps",
]
