#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Aligned ratio-of-means analysis with synchronous joint batch covariance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntFlag
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..typing import (
    as_array,
    Bool,
    Complex128,
    Dim,
    Float64,
    HostBool,
    HostFloat64,
    HostInt32,
    HostInteger,
    Identifier,
    Identifiers,
    Int32,
    parse,
    Scalar,
    Scope,
)
from ._correlation_selection import correlation_inefficiency, synchronous_block_indices


class RatioStreamDim(Dim):
    """Explicit scientific streams, not inferred independent replicas."""


class RatioDrawDim(Dim):
    """Aligned chronological draw positions."""


class RatioChannelDim(Dim):
    """Smallest real representation of the numerator and denominator."""


class RatioComponentDim(Dim):
    """Real output components of the ratio."""


class RatioGroupDim(Dim):
    """Declared independent dependence groups."""


class RatioDiagnosticDim(Dim):
    """Channels followed by ratio-influence components."""


class RatioBlockDim(Dim):
    """Complete synchronous blocks in canonical group order."""


DenominatorSign: TypeAlias = Literal["any", "positive", "negative"]
FiellerSet: TypeAlias = Literal[
    "bounded",
    "unbounded",
    "disconnected",
    "all-real",
    "empty",
    "singleton",
    "not-applicable",
]
RatioConfidenceKind: TypeAlias = Literal["fieller", "complex-origin-ball"]


class CorrelatedRatioStatus(IntFlag):
    """Composable fail-closed statistical qualification evidence."""

    SUCCESS = 0
    NONFINITE = 1
    INSUFFICIENT_DRAWS = 2
    CORRELATION_UNRESOLVED = 4
    INSUFFICIENT_BLOCKS = 8
    ZERO_VARIATION_UNRESOLVED = 16
    DENOMINATOR_UNSAFE = 32
    NONFINITE_COVARIANCE = 64


class CorrelatedRatioPolicy(StrictModule):
    """Immutable IPS/common-block and denominator confidence policy."""

    __strict_contract__ = True

    max_lag: int | None = eqx.field(static=True)
    minimum_draws: int = eqx.field(static=True)
    minimum_blocks: int = eqx.field(static=True)
    block_length: int | None = eqx.field(static=True)
    confidence_multiplier: float = eqx.field(static=True)
    minimum_denominator_magnitude: float = eqx.field(static=True)
    denominator_sign: DenominatorSign = eqx.field(static=True)
    policy_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_lag: int | None = None,
        minimum_draws: int = 8,
        minimum_blocks: int = 4,
        block_length: int | None = None,
        confidence_multiplier: float = 1.959963984540054,
        minimum_denominator_magnitude: float = 0.0,
        denominator_sign: DenominatorSign = "any",
    ) -> None:
        for name, value, minimum in (
            ("minimum_draws", minimum_draws, 2),
            ("minimum_blocks", minimum_blocks, 2),
            ("max_lag", max_lag, 1),
            ("block_length", block_length, 1),
        ):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer at least {minimum}.")
        z = float(confidence_multiplier)
        floor = float(minimum_denominator_magnitude)
        if not math.isfinite(z) or z <= 0.0:
            raise ValueError("confidence_multiplier must be finite and positive.")
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError(
                "minimum_denominator_magnitude must be finite and nonnegative."
            )
        sign = parse(denominator_sign, DenominatorSign, "denominator_sign")
        self.max_lag = max_lag
        self.minimum_draws = minimum_draws
        self.minimum_blocks = minimum_blocks
        self.block_length = block_length
        self.confidence_multiplier = z
        self.minimum_denominator_magnitude = floor
        self.denominator_sign = sign
        self.policy_id = canonical_fingerprint(
            {
                "kind": "correlated-ratio-policy",
                "max_lag": max_lag,
                "minimum_draws": minimum_draws,
                "minimum_blocks": minimum_blocks,
                "block_length": block_length,
                "confidence_multiplier": z,
                "minimum_denominator_magnitude": floor,
                "denominator_sign": sign,
                "correlation": "initial-positive-sequence-closed-window",
                "covariance": "synchronous-complete-batch-means",
            }
        )


class CorrelatedRatioResult(StrictModule):
    """Exploratory point, qualified ratio, full covariance and refusal evidence.

    Real channels are ordered Re X, optional Im X, Re Y, optional Im Y.
    Independent groups contribute squared sample weights to mean covariance.
    Confidence evidence uses an asymptotic Gaussian approximation, not a
    finite-history theorem. Unsafe qualified values/covariances are NaN.
    """

    __strict_contract__ = True

    numerator_mean: Float64[Scalar] | Complex128[Scalar]
    denominator_mean: Float64[Scalar] | Complex128[Scalar]
    exploratory_value: Float64[Scalar] | Complex128[Scalar]
    value: Float64[Scalar] | Complex128[Scalar]
    mean_covariance: Float64[RatioChannelDim, RatioChannelDim]
    ratio_covariance: Float64[RatioComponentDim, RatioComponentDim]
    standard_error: Float64[Scalar]
    statistically_valid: Bool[Scalar]
    status: Int32[Scalar]
    denominator_confidence_radius: Float64[Scalar]
    denominator_magnitude_lower_bound: Float64[Scalar]
    confidence_bounds: Float64[Literal[2]]
    retained: Bool[RatioStreamDim, RatioDrawDim]
    block_index: Int32[RatioStreamDim, RatioDrawDim]
    block_counts: Int32[RatioGroupDim]
    group_weights: Float64[RatioGroupDim]
    diagnostic_inefficiency: Float64[RatioGroupDim, RatioDiagnosticDim]
    diagnostic_window_closed: Bool[RatioGroupDim, RatioDiagnosticDim]
    diagnostic_varying: Bool[RatioGroupDim, RatioDiagnosticDim]
    discarded_records: Int32[Scalar]
    block_length: int = eqx.field(static=True)
    fieller_set: FiellerSet = eqx.field(static=True)
    confidence_kind: RatioConfidenceKind = eqx.field(static=True)
    stream_ids: Identifiers[RatioStreamDim] = eqx.field(static=True)
    dependence_ids: tuple[str, ...] = eqx.field(static=True)
    group_ids: Identifiers[RatioGroupDim] = eqx.field(static=True)
    sampling_origin_id: Identifier = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    deterministic_ratio: float | complex | None = eqx.field(static=True)
    policy: CorrelatedRatioPolicy


class _AlignedRatioRecords(StrictModule):
    """Validated metadata and canonical real channels at the host boundary."""

    __strict_contract__ = True

    channels: Float64[RatioStreamDim, RatioDrawDim, RatioChannelDim]
    host_channels: HostFloat64[RatioStreamDim, RatioDrawDim, RatioChannelDim]
    host_active: HostBool[RatioStreamDim, RatioDrawDim]
    numerator_complex: bool
    denominator_complex: bool
    finite: bool
    deterministic_relation: bool
    stream_ids: Identifiers[RatioStreamDim]
    dependence_ids: tuple[str, ...]
    group_ids: Identifiers[RatioGroupDim]
    groups: HostInt32[RatioStreamDim]
    sampling_origin_id: Identifier


class _RatioSelection(StrictModule):
    """Immutable host analysis boundary; no synchronization in solver steps."""

    __strict_contract__ = True

    retained: HostBool[RatioStreamDim, RatioDrawDim]
    block_index: HostInt32[RatioStreamDim, RatioDrawDim]
    block_counts: HostInt32[RatioGroupDim]
    block_groups: HostInt32[RatioBlockDim]
    group_weights: HostFloat64[RatioGroupDim]
    inefficiency: HostFloat64[RatioGroupDim, RatioDiagnosticDim]
    closed: HostBool[RatioGroupDim, RatioDiagnosticDim]
    varying: HostBool[RatioGroupDim, RatioDiagnosticDim]
    block_length: int
    status: int


def _fieller_set(
    x: float, y: float, covariance: HostFloat64[Literal[2], Literal[2]], z: float, /
) -> tuple[FiellerSet, tuple[float, float]]:
    """Solve the real Fieller quadratic, including every degenerate boundary."""
    if not all(math.isfinite(value) for value in (x, y, z)) or not np.all(
        np.isfinite(covariance)
    ):
        return "empty", (math.nan, math.nan)
    if np.all(covariance == 0.0):
        if y != 0.0:
            point = x / y
            return "singleton", (point, point)
        return (
            ("all-real", (-math.inf, math.inf))
            if x == 0.0
            else ("empty", (math.nan, math.nan))
        )
    a = y * y - z * z * float(covariance[1, 1])
    b = -2.0 * (x * y - z * z * float(covariance[0, 1]))
    c = x * x - z * z * float(covariance[0, 0])
    scale = max(abs(a), abs(b), abs(c))
    if not math.isfinite(scale):
        return "empty", (math.nan, math.nan)
    if scale == 0.0:
        return "all-real", (-math.inf, math.inf)
    a, b, c = a / scale, b / scale, c / scale
    if a == 0.0:
        if b == 0.0:
            return (
                ("all-real", (-math.inf, math.inf))
                if c <= 0.0
                else ("empty", (math.nan, math.nan))
            )
        boundary = -c / b
        return "unbounded", (-math.inf, boundary) if b > 0.0 else (boundary, math.inf)
    discriminant = b * b - 4.0 * a * c
    rounding_bound = 8.0 * np.finfo(np.float64).eps * (b * b + abs(4.0 * a * c))
    if abs(discriminant) <= rounding_bound:
        discriminant = 0.0
    if discriminant < 0.0:
        return (
            ("all-real", (-math.inf, math.inf))
            if a < 0.0
            else ("empty", (math.nan, math.nan))
        )
    if discriminant == 0.0:
        root = -b / (2.0 * a)
        return (
            ("all-real", (-math.inf, math.inf))
            if a < 0.0
            else ("singleton", (root, root))
        )
    q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
    roots = sorted((q / a, c / q))
    return ("bounded" if a > 0.0 else "disconnected"), (roots[0], roots[1])


def _division_jacobian(
    means: Float64[RatioChannelDim], numerator_complex: bool, denominator_complex: bool, /
) -> Float64[RatioComponentDim, RatioChannelDim]:
    offset = 2 if numerator_complex else 1
    x = means[0].astype(jnp.complex128)
    if numerator_complex:
        x = x + jnp.asarray(1j, dtype=jnp.complex128) * means[1].astype(jnp.complex128)
    y = means[offset].astype(jnp.complex128)
    if denominator_complex:
        y = y + jnp.asarray(1j, dtype=jnp.complex128) * means[offset + 1].astype(
            jnp.complex128
        )
    safe_y = jnp.where(
        jnp.abs(y) > jnp.asarray(0.0, dtype=jnp.float64),
        y,
        jnp.asarray(1.0, dtype=jnp.complex128),
    )
    dx = jnp.asarray(1.0, dtype=jnp.complex128) / safe_y
    dy = -x / (safe_y * safe_y)
    derivatives = [dx]
    if numerator_complex:
        derivatives.append(jnp.asarray(1j, dtype=jnp.complex128) * dx)
    derivatives.append(dy)
    if denominator_complex:
        derivatives.append(jnp.asarray(1j, dtype=jnp.complex128) * dy)
    row = jnp.stack(derivatives)
    if numerator_complex or denominator_complex:
        return jnp.stack((jnp.real(row), jnp.imag(row)))
    return jnp.real(row)[None, :]


def _aligned_runs(
    active: HostBool[RatioStreamDim, RatioDrawDim],
    group_indices: HostInteger[RatioStreamDim],
    /,
) -> tuple[
    HostBool[RatioStreamDim, RatioDrawDim], HostInt32[RatioStreamDim, RatioDrawDim]
]:
    """Retain only synchronous positions and never bridge holes in chronology."""
    retained = np.zeros(active.shape, dtype=np.bool_)
    runs = np.full(active.shape, -1, dtype=np.int32)
    for group in sorted(set(group_indices.tolist())):
        streams = np.nonzero(group_indices == group)[0]
        synchronous = np.all(active[streams], axis=0)
        retained[streams] = synchronous[None, :]
        run = -1
        previous = False
        for draw, selected in enumerate(synchronous):
            if selected and not previous:
                run += 1
            if selected:
                runs[streams, draw] = run
            previous = bool(selected)
    return retained, runs


def _temporal_diagnostics(
    diagnostics: HostFloat64[RatioStreamDim, RatioDrawDim, RatioDiagnosticDim],
    retained: HostBool[RatioStreamDim, RatioDrawDim],
    runs: HostInt32[RatioStreamDim, RatioDrawDim],
    groups: HostInteger[RatioStreamDim],
    group_count: int,
    max_lag: int | None,
    /,
) -> tuple[
    HostFloat64[RatioGroupDim, RatioDiagnosticDim],
    HostBool[RatioGroupDim, RatioDiagnosticDim],
    HostBool[RatioGroupDim, RatioDiagnosticDim],
]:
    stream_count, draw_count, channel_count = diagnostics.shape
    shape = (group_count, channel_count)
    inefficiency = np.ones(shape, dtype=np.float64)
    closed = np.ones(shape, dtype=np.bool_)
    varying = np.zeros(shape, dtype=np.bool_)
    flat_group = np.broadcast_to(groups[:, None], retained.shape).reshape((-1,))
    flat_stream = np.broadcast_to(
        np.arange(stream_count, dtype=np.int32)[:, None], retained.shape
    ).reshape((-1,))
    chain = (
        np.arange(stream_count, dtype=np.int32)[:, None] * (draw_count + 1) + runs
    ).reshape((-1,))
    draw = np.broadcast_to(
        np.arange(draw_count, dtype=np.int32)[None, :], retained.shape
    ).reshape((-1,))
    repeat = np.zeros((retained.size,), dtype=np.int32)
    for channel in range(channel_count):
        values = diagnostics[:, :, channel].reshape((-1,))
        g, resolved, terminated = correlation_inefficiency(
            values,
            retained.reshape((-1,)),
            flat_stream,
            chain,
            draw,
            repeat,
            flat_group,
            stream_count,
            max_lag,
        )
        for stream in range(stream_count):
            group = int(groups[stream])
            inefficiency[group, channel] = max(inefficiency[group, channel], g[stream])
            selected = diagnostics[stream, retained[stream], channel]
            stream_varying = bool(selected.size and np.any(selected != selected[0]))
            varying[group, channel] |= stream_varying
            if stream_varying:
                closed[group, channel] &= resolved[stream] and terminated[stream]
    # Covariance uses synchronous group means, whose slow modes can cancel
    # the fast modes that close every individual stream's lag window.
    for group in range(group_count):
        streams = np.nonzero(groups == group)[0]
        if streams.size < 2:
            continue
        first = streams[0]
        group_retained = retained[first]
        group_diagnostics: HostFloat64[RatioDrawDim, RatioDiagnosticDim] = np.mean(
            diagnostics[streams], axis=0, dtype=np.float64
        )
        group_strata = np.zeros((draw_count,), dtype=np.int32)
        group_draw = np.arange(draw_count, dtype=np.int32)
        for channel in range(channel_count):
            g, resolved, terminated = correlation_inefficiency(
                group_diagnostics[:, channel],
                group_retained,
                group_strata,
                runs[first],
                group_draw,
                group_strata,
                group_strata,
                1,
                max_lag,
            )
            inefficiency[group, channel] = max(inefficiency[group, channel], g[0])
            selected = group_diagnostics[group_retained, channel]
            group_varying = bool(selected.size and np.any(selected != selected[0]))
            varying[group, channel] |= group_varying
            if group_varying:
                closed[group, channel] &= resolved[0] and terminated[0]
    return inefficiency, closed, varying


def _select_complete_blocks(
    diagnostics: HostFloat64[RatioStreamDim, RatioDrawDim, RatioDiagnosticDim],
    active: HostBool[RatioStreamDim, RatioDrawDim],
    groups: HostInteger[RatioStreamDim],
    group_count: int,
    policy: CorrelatedRatioPolicy,
    deterministic: bool,
    deterministic_relation: bool,
    influence_components: int,
    /,
) -> _RatioSelection:
    retained, runs = _aligned_runs(active, groups)
    inefficiency, closed, varying = _temporal_diagnostics(
        diagnostics, retained, runs, groups, group_count, policy.max_lag
    )
    status = CorrelatedRatioStatus.SUCCESS
    exact_constant = deterministic_relation and not np.any(varying)
    if not deterministic and not exact_constant:
        if np.any(np.sum(retained, axis=1) < policy.minimum_draws):
            status |= CorrelatedRatioStatus.INSUFFICIENT_DRAWS
        if np.any(varying & ~closed):
            status |= CorrelatedRatioStatus.CORRELATION_UNRESOLVED
        if not np.any(varying[:, -influence_components:]) and not deterministic_relation:
            status |= CorrelatedRatioStatus.ZERO_VARIATION_UNRESOLVED
    required_length = max(1, math.ceil(float(np.max(inefficiency, initial=1.0))))
    length = required_length if policy.block_length is None else policy.block_length
    if not deterministic and length < required_length:
        status |= CorrelatedRatioStatus.CORRELATION_UNRESOLVED
    if deterministic or (deterministic_relation and not np.any(varying)):
        length = 1
    for group in range(group_count):
        streams = np.nonzero(groups == group)[0]
        first = streams[0]
        for run in sorted(set(runs[first, retained[first]].tolist())):
            positions = np.nonzero(retained[first] & (runs[first] == run))[0]
            complete_count = positions.size // length * length
            retained[np.ix_(streams, positions[complete_count:])] = False
    flat_group = np.broadcast_to(groups[:, None], active.shape).reshape((-1,))
    draw = np.broadcast_to(
        np.arange(active.shape[1], dtype=np.int32)[None, :], active.shape
    ).reshape((-1,))
    blocks, _group_index, block_count = synchronous_block_indices(
        retained.reshape((-1,)), runs.reshape((-1,)), flat_group, draw, length
    )
    counts = np.zeros((group_count,), dtype=np.int32)
    block_groups = np.empty((block_count,), dtype=np.int32)
    for block in range(block_count):
        selected = blocks == block
        group = int(flat_group[np.nonzero(selected)[0][0]])
        block_groups[block] = group
        counts[group] += 1
    if not deterministic and not (deterministic_relation and not np.any(varying)):
        if np.any(counts < policy.minimum_blocks):
            status |= CorrelatedRatioStatus.INSUFFICIENT_BLOCKS
    sizes = np.asarray(
        [np.sum(retained[groups == group]) for group in range(group_count)],
        dtype=np.float64,
    )
    total = float(np.sum(sizes))
    weights = sizes / total if total > 0.0 else np.zeros(sizes.shape, dtype=np.float64)
    if total == 0.0:
        status |= CorrelatedRatioStatus.INSUFFICIENT_DRAWS
    return _RatioSelection(
        retained,
        blocks.reshape(active.shape),
        counts,
        block_groups,
        weights,
        inefficiency,
        closed,
        varying,
        length,
        int(status),
    )


def _batch_mean_covariance(
    channels: Float64[RatioStreamDim, RatioDrawDim, RatioChannelDim],
    selection: _RatioSelection,
    deterministic: bool,
    /,
) -> tuple[Float64[RatioChannelDim], Float64[RatioChannelDim, RatioChannelDim]]:
    """Joint block vectors retain all cross-channel and shared-stream terms."""
    mask = jnp.asarray(selection.retained)
    safe = jnp.where(mask[:, :, None], channels, jnp.zeros((), dtype=jnp.float64))
    count = jnp.sum(mask, dtype=jnp.float64)
    means = jnp.where(
        count > 0.0, jnp.sum(safe, axis=(0, 1)) / jnp.maximum(count, 1.0), jnp.nan
    )
    dimension = channels.shape[2]
    covariance = jnp.zeros((dimension, dimension), dtype=jnp.float64)
    if deterministic:
        return means, covariance
    block_count = selection.block_groups.size
    if block_count == 0:
        return means, jnp.full(covariance.shape, jnp.nan, dtype=jnp.float64)
    block_index = jnp.asarray(selection.block_index).reshape((-1,))
    block_sums = segment_sum(safe.reshape((-1, dimension)), block_index, block_count)
    block_sizes = segment_sum(
        mask.reshape((-1,)).astype(jnp.float64), block_index, block_count
    )
    all_batches = block_sums / block_sizes[:, None]
    for group in range(selection.block_counts.size):
        indices = np.nonzero(selection.block_groups == group)[0]
        if indices.size < 2:
            return means, jnp.full(covariance.shape, jnp.nan, dtype=jnp.float64)
        batches = all_batches[jnp.asarray(indices)]
        centered = batches - jnp.mean(batches, axis=0, keepdims=True)
        group_covariance = centered.T @ centered / (indices.size * (indices.size - 1))
        covariance = covariance + selection.group_weights[group] ** 2 * group_covariance
    return means, covariance


def _denominator_evidence(
    means: HostFloat64[RatioChannelDim],
    covariance: HostFloat64[RatioChannelDim, RatioChannelDim],
    numerator_complex: bool,
    denominator_complex: bool,
    policy: CorrelatedRatioPolicy,
    /,
) -> tuple[float, float, bool, FiellerSet, tuple[float, float], RatioConfidenceKind]:
    offset = 2 if numerator_complex else 1
    y_real = float(means[offset])
    if denominator_complex:
        y = complex(y_real, float(means[offset + 1]))
        trace = float(covariance[offset, offset] + covariance[offset + 1, offset + 1])
        radius = (
            policy.confidence_multiplier * math.sqrt(trace) if trace >= 0.0 else math.nan
        )
        lower = abs(y) - radius
        safe = math.isfinite(lower) and lower > policy.minimum_denominator_magnitude
        return (
            radius,
            lower,
            safe,
            "not-applicable",
            (math.nan, math.nan),
            "complex-origin-ball",
        )
    variance = float(covariance[offset, offset])
    radius = (
        policy.confidence_multiplier * math.sqrt(variance)
        if variance >= 0.0
        else math.nan
    )
    lower = abs(y_real) - radius
    safe = math.isfinite(lower) and lower > policy.minimum_denominator_magnitude
    match policy.denominator_sign:
        case "any":
            pass
        case "positive":
            safe &= y_real - radius > policy.minimum_denominator_magnitude
        case "negative":
            safe &= y_real + radius < -policy.minimum_denominator_magnitude
        case _ as unreachable:
            assert_never(unreachable)
    if numerator_complex:
        return (
            radius,
            lower,
            safe,
            "not-applicable",
            (math.nan, math.nan),
            "complex-origin-ball",
        )
    joint = covariance[np.ix_((0, offset), (0, offset))]
    kind, bounds = _fieller_set(
        float(means[0]), y_real, joint, policy.confidence_multiplier
    )
    safe &= kind == "bounded" or kind == "singleton"
    return radius, lower, safe, kind, bounds, "fieller"


def _ratio_records(
    value: ArrayLike, name: str, scope: Scope, /
) -> Float64[RatioStreamDim, RatioDrawDim] | Complex128[RatioStreamDim, RatioDrawDim]:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        return parse(
            array.astype(jnp.complex128),
            Complex128[RatioStreamDim, RatioDrawDim],
            name,
            scope=scope,
        )
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must contain real or complex floating records.")
    return parse(
        array.astype(jnp.float64),
        Float64[RatioStreamDim, RatioDrawDim],
        name,
        scope=scope,
    )


def _canonical_ratio_records(
    numerator: ArrayLike,
    denominator: ArrayLike,
    policy: CorrelatedRatioPolicy,
    stream_ids: Sequence[str],
    dependence_ids: Sequence[str],
    sampling_origin_id: str,
    valid: ArrayLike | None,
    deterministic: bool,
    deterministic_ratio: float | complex | None,
    /,
) -> _AlignedRatioRecords:
    """Validate source declarations before selecting any chronological records."""
    if not isinstance(policy, CorrelatedRatioPolicy):
        raise TypeError("policy must be CorrelatedRatioPolicy.")
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be bool source metadata.")
    scope = Scope()
    x = _ratio_records(numerator, "numerator", scope)
    y = _ratio_records(denominator, "denominator", scope)
    if x.shape[0] < 1:
        raise ValueError("Ratio records require at least one stream.")
    streams = parse(
        tuple(stream_ids), Identifiers[RatioStreamDim], "stream_ids", scope=scope
    )
    dependencies = tuple(
        parse(value, Identifier, "dependence_id") for value in dependence_ids
    )
    if len(dependencies) != x.shape[0]:
        raise ValueError("dependence_ids must match the stream axis.")
    origin = parse(sampling_origin_id, Identifier, "sampling_origin_id")
    active = (
        jnp.ones(x.shape, dtype=jnp.bool_)
        if valid is None
        else as_array(valid, Bool[RatioStreamDim, RatioDrawDim], "valid", scope=scope)
    )
    numerator_complex = jnp.issubdtype(x.dtype, jnp.complexfloating)
    denominator_complex = jnp.issubdtype(y.dtype, jnp.complexfloating)
    if denominator_complex and policy.denominator_sign != "any":
        raise ValueError("A complex denominator has no positive/negative sign contract.")
    components = [jnp.real(x)]
    if numerator_complex:
        components.append(jnp.imag(x))
    components.append(jnp.real(y))
    if denominator_complex:
        components.append(jnp.imag(y))
    channels = jnp.stack(components, axis=-1)
    host_active = np.asarray(active)
    host_channels = np.asarray(channels)
    finite = bool(np.all(np.isfinite(host_channels[host_active])))
    relation = deterministic_ratio is not None
    if deterministic_ratio is not None:
        declared = complex(deterministic_ratio)
        if not math.isfinite(declared.real) or not math.isfinite(declared.imag):
            raise ValueError("deterministic_ratio must be finite.")
        host_x, host_y = np.asarray(x), np.asarray(y)
        tolerance = 64.0 * np.finfo(np.float64).eps
        residual = host_x[host_active] - declared * host_y[host_active]
        scale = np.maximum(
            np.maximum(
                np.abs(host_x[host_active]), np.abs(declared * host_y[host_active])
            ),
            1.0,
        )
        if np.any(np.abs(residual) > tolerance * scale):
            raise ValueError("Records violate the declared deterministic ratio relation.")
    group_ids = tuple(sorted(set(dependencies)))
    groups = np.asarray(
        [group_ids.index(value) for value in dependencies], dtype=np.int32
    )
    return _AlignedRatioRecords(
        channels,
        host_channels,
        host_active,
        numerator_complex,
        denominator_complex,
        finite,
        relation,
        streams,
        dependencies,
        group_ids,
        groups,
        origin,
    )


def _selected_ratio_covariance(
    records: _AlignedRatioRecords,
    policy: CorrelatedRatioPolicy,
    deterministic: bool,
    /,
) -> tuple[
    _RatioSelection,
    Float64[RatioChannelDim],
    Float64[RatioChannelDim, RatioChannelDim],
]:
    """Select synchronous complete blocks using channels and ratio influence."""
    safe_host = np.where(
        np.isfinite(records.host_channels),
        records.host_channels,
        np.zeros((), dtype=np.float64),
    )
    synchronous, _runs = _aligned_runs(records.host_active, records.groups)
    preliminary_count = np.sum(synchronous)
    preliminary = np.sum(
        np.where(synchronous[:, :, None], safe_host, np.zeros((), dtype=np.float64)),
        axis=(0, 1),
    ) / max(preliminary_count, 1)
    ratio_components = (
        2 if records.numerator_complex or records.denominator_complex else 1
    )
    if deterministic or records.deterministic_relation:
        influence = np.zeros(
            (*records.host_active.shape, ratio_components), dtype=np.float64
        )
    else:
        jacobian = _division_jacobian(
            jnp.asarray(preliminary),
            records.numerator_complex,
            records.denominator_complex,
        )
        influence = np.asarray(
            (jnp.asarray(safe_host) - jnp.asarray(preliminary)[None, None, :])
            @ jacobian.T
        )
    diagnostics = np.concatenate((safe_host, influence), axis=2)
    selection = _select_complete_blocks(
        diagnostics,
        records.host_active,
        records.groups,
        len(records.group_ids),
        policy,
        deterministic,
        records.deterministic_relation,
        ratio_components,
    )
    means, covariance = _batch_mean_covariance(
        records.channels,
        selection,
        deterministic
        or (records.deterministic_relation and not np.any(selection.varying)),
    )
    if not np.any(selection.retained) and preliminary_count > 0:
        exploratory_means = (
            np.sum(
                np.where(
                    synchronous[:, :, None],
                    records.host_channels,
                    np.zeros((), dtype=np.float64),
                ),
                axis=(0, 1),
            )
            / preliminary_count
        )
        means = jnp.asarray(exploratory_means, dtype=jnp.float64)
    return selection, means, covariance


def _qualified_ratio_result(
    records: _AlignedRatioRecords,
    selection: _RatioSelection,
    means: Float64[RatioChannelDim],
    covariance: Float64[RatioChannelDim, RatioChannelDim],
    policy: CorrelatedRatioPolicy,
    deterministic: bool,
    deterministic_ratio: float | complex | None,
    /,
) -> CorrelatedRatioResult:
    """Apply denominator confidence and fail-closed qualification to the result."""
    numerator_complex = records.numerator_complex
    denominator_complex = records.denominator_complex
    relation = records.deterministic_relation
    ratio_components = 2 if numerator_complex or denominator_complex else 1
    offset = 2 if numerator_complex else 1
    x_mean = (
        means[0].astype(jnp.complex128)
        + jnp.asarray(1j, dtype=jnp.complex128) * means[1].astype(jnp.complex128)
        if numerator_complex
        else means[0]
    )
    y_mean = (
        means[offset].astype(jnp.complex128)
        + jnp.asarray(1j, dtype=jnp.complex128) * means[offset + 1].astype(jnp.complex128)
        if denominator_complex
        else means[offset]
    )
    ratio_dtype = (
        jnp.complex128 if numerator_complex or denominator_complex else jnp.float64
    )
    nonzero_denominator = jnp.abs(y_mean) > jnp.asarray(0.0, dtype=jnp.float64)
    safe_denominator = jnp.where(
        nonzero_denominator, y_mean, jnp.asarray(1.0, dtype=y_mean.dtype)
    )
    ratio_nan = jnp.asarray(jnp.nan, dtype=ratio_dtype)
    exploratory = jnp.where(
        nonzero_denominator,
        x_mean.astype(ratio_dtype) / safe_denominator.astype(ratio_dtype),
        ratio_nan,
    )
    status = CorrelatedRatioStatus(selection.status)
    if not records.finite:
        status |= CorrelatedRatioStatus.NONFINITE
    if not bool(jnp.all(jnp.isfinite(covariance))):
        status |= CorrelatedRatioStatus.NONFINITE_COVARIANCE
    radius, lower, denominator_safe, fieller, bounds, confidence_kind = (
        _denominator_evidence(
            np.asarray(means),
            np.asarray(covariance),
            numerator_complex,
            denominator_complex,
            policy,
        )
    )
    if not denominator_safe:
        status |= CorrelatedRatioStatus.DENOMINATOR_UNSAFE
    if deterministic or relation:
        ratio_covariance = jnp.zeros(
            (ratio_components, ratio_components), dtype=jnp.float64
        )
    else:
        jacobian = _division_jacobian(means, numerator_complex, denominator_complex)
        ratio_covariance = jacobian @ covariance @ jacobian.T
    if not bool(jnp.isfinite(exploratory)):
        status |= CorrelatedRatioStatus.NONFINITE
    if not bool(jnp.all(jnp.isfinite(ratio_covariance))):
        status |= CorrelatedRatioStatus.NONFINITE_COVARIANCE
    if not deterministic and not relation and bool(jnp.trace(ratio_covariance) == 0.0):
        status |= CorrelatedRatioStatus.ZERO_VARIATION_UNRESOLVED
    qualified = status == CorrelatedRatioStatus.SUCCESS
    ratio_covariance = jnp.where(
        qualified,
        ratio_covariance,
        jnp.full(ratio_covariance.shape, jnp.nan, dtype=jnp.float64),
    )
    return CorrelatedRatioResult(
        x_mean,
        y_mean,
        exploratory,
        jnp.where(qualified, exploratory, ratio_nan),
        covariance,
        ratio_covariance,
        jnp.sqrt(jnp.trace(ratio_covariance)),
        jnp.asarray(qualified, dtype=jnp.bool_),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(radius, dtype=jnp.float64),
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(bounds, dtype=jnp.float64),
        jnp.asarray(selection.retained),
        jnp.asarray(selection.block_index),
        jnp.asarray(selection.block_counts),
        jnp.asarray(selection.group_weights),
        jnp.asarray(selection.inefficiency),
        jnp.asarray(selection.closed),
        jnp.asarray(selection.varying),
        jnp.asarray(
            np.sum(records.host_active) - np.sum(selection.retained), dtype=jnp.int32
        ),
        selection.block_length,
        fieller,
        confidence_kind,
        records.stream_ids,
        records.dependence_ids,
        records.group_ids,
        records.sampling_origin_id,
        deterministic,
        deterministic_ratio,
        policy,
    )


def correlated_ratio_of_means(
    numerator: ArrayLike,
    denominator: ArrayLike,
    *,
    policy: CorrelatedRatioPolicy,
    stream_ids: Sequence[str],
    dependence_ids: Sequence[str],
    sampling_origin_id: str,
    valid: ArrayLike | None = None,
    deterministic: bool = False,
    deterministic_ratio: float | complex | None = None,
) -> CorrelatedRatioResult:
    """Analyze aligned ``(stream, draw)`` records at an explicit host boundary.

    Stream IDs must be unique; equal dependence IDs declare synchronous
    dependence. Each contiguous complete run is blocked independently. All
    active records must be finite, even those excluded by synchronization or
    incomplete tails. ``deterministic`` is source evidence, never inferred from
    observed constants. A declared ``deterministic_ratio`` is checked against
    every active record; it is not fitted from the data.
    """
    records = _canonical_ratio_records(
        numerator,
        denominator,
        policy,
        stream_ids,
        dependence_ids,
        sampling_origin_id,
        valid,
        deterministic,
        deterministic_ratio,
    )
    selection, means, covariance = _selected_ratio_covariance(
        records, policy, deterministic
    )
    return _qualified_ratio_result(
        records,
        selection,
        means,
        covariance,
        policy,
        deterministic,
        deterministic_ratio,
    )


__all__ = [
    "CorrelatedRatioPolicy",
    "CorrelatedRatioResult",
    "CorrelatedRatioStatus",
    "correlated_ratio_of_means",
]
