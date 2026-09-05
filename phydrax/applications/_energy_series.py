#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Quantity-aware interval operations over the native ordered-series substrate."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .._strict import StrictModule
from ..series import SampledSeries, SeriesSupport
from ..units import conversion_factor, derived_unit, SECOND, UnitDefinition


class EnergySeries(StrictModule):
    """Samples with explicit quantity, interval, clock, and source semantics.

    Civil-time conversion belongs to the ingress adapter. Numerical coordinates
    use ``time_unit`` relative to the declared origin. No missing-value filling
    or counter-reset inference is performed by this envelope.
    """

    samples: SampledSeries
    unit: UnitDefinition
    time_unit: UnitDefinition
    quantity: str = eqx.field(static=True)
    meaning: str = eqx.field(static=True)
    time_basis: str = eqx.field(static=True)
    origin: str | None = eqx.field(static=True)
    timezone: str | None = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)
    reference_id: str | None = eqx.field(static=True)
    sign_convention: str | None = eqx.field(static=True)

    def __init__(
        self,
        samples: SampledSeries,
        *,
        quantity: str,
        unit: UnitDefinition,
        meaning: str,
        time_unit: UnitDefinition = SECOND,
        time_basis: str = "relative",
        origin: str | None = None,
        timezone: str | None = None,
        asset_id: str = "",
        provenance: tuple[str, ...] = (),
        reference_id: str | None = None,
        sign_convention: str | None = None,
    ):
        if not isinstance(samples, SampledSeries):
            raise TypeError("samples must be a native SampledSeries")
        if not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be a native UnitDefinition")
        conversion_factor(time_unit, SECOND)
        if not isinstance(quantity, str) or not quantity.strip():
            raise ValueError("quantity must be a nonempty semantic identifier")
        if meaning not in (
            "instantaneous",
            "interval_average",
            "interval_integral",
            "cumulative",
            "schedule",
        ):
            raise ValueError("unsupported energy-series meaning")
        expected = (
            "edge" if meaning in ("interval_average", "interval_integral") else "node"
        )
        if samples.alignment != expected:
            raise ValueError(f"{meaning} requires {expected}-aligned samples")
        if time_basis not in ("relative", "absolute", "cyclic"):
            raise ValueError("time_basis must be relative, absolute, or cyclic")
        if time_basis == "absolute" and not origin:
            raise ValueError("absolute time requires an explicit origin")
        if samples.support.coordinate_kind != "continuous":
            raise ValueError(
                "energy-series clocks require continuous physical coordinates"
            )
        if any(not isinstance(item, str) or not item for item in provenance):
            raise ValueError("provenance entries must be nonempty identifiers")
        self.samples = samples
        self.quantity = quantity
        self.unit = unit
        self.meaning = meaning
        self.time_unit = time_unit
        self.time_basis = time_basis
        self.origin = origin
        self.timezone = timezone
        self.asset_id = asset_id
        self.provenance = tuple(provenance)
        self.reference_id = reference_id
        self.sign_convention = sign_convention


def _with_samples(
    series: EnergySeries, samples: SampledSeries, meaning: str, operation: str
):
    return EnergySeries(
        samples,
        quantity=series.quantity,
        unit=series.unit,
        meaning=meaning,
        time_unit=series.time_unit,
        time_basis=series.time_basis,
        origin=series.origin,
        timezone=series.timezone,
        asset_id=series.asset_id,
        provenance=(*series.provenance, operation),
        reference_id=series.reference_id,
        sign_convention=series.sign_convention,
    )


def _monotonic_coordinates(support: SeriesSupport):
    coordinates = support.broadcast_coordinates()
    return eqx.error_if(
        coordinates,
        jnp.any(~jnp.isfinite(coordinates)) | jnp.any(jnp.diff(coordinates, axis=-1) < 0),
        "interval operations require a globally ordered finite clock; split restarted episodes first",
    )


def _leaf_masks(samples: SampledSeries):
    if samples.value_valid is None:
        return jax.tree_util.tree_map(
            lambda value: jnp.ones(value.shape, dtype=bool), samples.values
        )
    return samples.value_valid


def rebin_energy_series(series: EnergySeries, support: SeriesSupport) -> EnergySeries:
    """Conservatively align intervals in O(N + M log N) work per channel.

    The destination uses the same physical clock/unit/origin. Integrals assume
    uniform within-source-interval density; averages assume a held value.
    Uncovered destination components are explicitly invalid, not imputed.
    Restarted/overlapping episodes must be selected independently before use.
    """
    if series.meaning not in ("interval_average", "interval_integral"):
        raise ValueError("conservative rebinning requires interval averages or integrals")
    source = series.samples.support
    if not isinstance(support, SeriesSupport):
        raise TypeError("support must be a SeriesSupport")
    if (
        source.series_shape,
        source.series_axes,
        source.coordinate_id,
        source.coordinate_kind,
    ) != (
        support.series_shape,
        support.series_axes,
        support.coordinate_id,
        support.coordinate_kind,
    ):
        raise ValueError(
            "source and destination must describe the same series axes and physical clock"
        )
    if source.capacity < 2 or support.capacity < 2:
        raise ValueError("interval operations require at least two clock nodes")
    x = _monotonic_coordinates(source).reshape((source.num_series, source.capacity))
    target = _monotonic_coordinates(support).reshape(
        (support.num_series, support.capacity)
    )
    active = source.edge_valid.reshape((source.num_series, source.capacity - 1))
    target_active = support.edge_valid.reshape((support.num_series, support.capacity - 1))
    prefix_rank = len(source.series_shape) + 1

    def align_leaf(values, mask):
        event_shape = values.shape[prefix_rank:]
        values = values.reshape((source.num_series, source.capacity - 1) + event_shape)
        mask = mask.reshape(values.shape)

        def align_row(nodes, out_nodes, payload, valid, edges, out_edges):
            width = jnp.diff(nodes)
            expansion = (1,) * len(event_shape)
            valid = valid & edges.reshape(edges.shape + expansion)
            safe_width = jnp.where(width > 0, width, 1).reshape(width.shape + expansion)
            payload = jnp.where(valid, payload, 0.0)
            rate = (
                payload / safe_width if series.meaning == "interval_integral" else payload
            )
            increments = rate * width.reshape(width.shape + expansion)
            invalid = (~valid) & (width > 0).reshape(width.shape + expansion)
            invalid_prefix = jnp.concatenate(
                (
                    jnp.zeros((1,) + event_shape, dtype=jnp.int32),
                    jnp.cumsum(invalid, axis=0, dtype=jnp.int32),
                ),
                axis=0,
            )
            first = jnp.clip(
                jnp.searchsorted(nodes, out_nodes[:-1], side="right") - 1,
                0,
                nodes.size - 1,
            )
            last = jnp.clip(
                jnp.searchsorted(nodes, out_nodes[1:], side="left"), 0, nodes.size - 1
            )
            index = jnp.clip(
                jnp.searchsorted(nodes, out_nodes, side="right") - 1, 0, nodes.size - 2
            )
            partial = jnp.clip(out_nodes - nodes[index], 0, width[index])

            def primitive(delta, slope):
                zero = jnp.zeros((1,) + event_shape, dtype=delta.dtype)
                cumulative = jnp.concatenate((zero, jnp.cumsum(delta, axis=0)), axis=0)
                return cumulative[index] + slope[index] * partial.reshape(
                    partial.shape + expansion
                )

            integrated = jnp.diff(primitive(increments, rate), axis=0)
            out_width = jnp.diff(out_nodes).reshape((out_nodes.size - 1,) + expansion)
            within = (out_nodes[:-1] >= nodes[0]) & (out_nodes[1:] <= nodes[-1])
            complete = (invalid_prefix[last] == invalid_prefix[first]) & (out_width > 0)
            complete = complete & (out_edges & within).reshape(
                out_edges.shape + expansion
            )
            result = (
                integrated / jnp.where(out_width > 0, out_width, 1)
                if series.meaning == "interval_average"
                else integrated
            )
            return jnp.where(complete, result, 0), complete

        result, valid = jax.vmap(align_row)(
            x, target, values, mask, active, target_active
        )
        shape = support.series_shape + (support.capacity - 1,) + event_shape
        return result.reshape(shape), valid.reshape(shape)

    leaves, tree = jax.tree_util.tree_flatten(series.samples.values)
    masks = jax.tree_util.tree_leaves(_leaf_masks(series.samples))
    aligned = [align_leaf(value, mask) for value, mask in zip(leaves, masks, strict=True)]
    values = jax.tree_util.tree_unflatten(tree, [pair[0] for pair in aligned])
    valid = jax.tree_util.tree_unflatten(tree, [pair[1] for pair in aligned])
    samples = SampledSeries(
        support,
        values,
        alignment="edge",
        value_valid=valid,
        series_id=f"{series.samples.series_id}:rebin",
    )
    return _with_samples(series, samples, series.meaning, "conservative-interval-rebin")


def integrate_energy_series(series: EnergySeries) -> tuple[Any, UnitDefinition]:
    """Integrate all declared active intervals, refusing incomplete values.

    Average-valued quantities gain one SI-second factor. Integral-valued
    quantities already carry their integrated unit and are simply summed.
    Disconnected intervals are not claimed to cover gaps outside their support.
    """
    if series.meaning not in ("interval_average", "interval_integral"):
        raise ValueError("integration requires an explicit interval meaning")
    samples = series.samples
    axis = len(samples.support.series_shape)
    width = jnp.diff(samples.support.broadcast_coordinates(), axis=-1)
    width = width * float(conversion_factor(series.time_unit, SECOND))

    def total(value, mask):
        expansion = (1,) * (value.ndim - axis - 1)
        active = samples.support.edge_valid.reshape(
            samples.support.edge_valid.shape + expansion
        )
        value = eqx.error_if(
            value, jnp.any(active & ~mask), "cannot integrate incomplete active intervals"
        )
        weighted = (
            value * width.reshape(width.shape + expansion)
            if series.meaning == "interval_average"
            else value
        )
        return jnp.sum(jnp.where(active, weighted, 0), axis=axis)

    result = jax.tree_util.tree_map(total, samples.values, _leaf_masks(samples))
    unit = (
        derived_unit(f"{series.unit.symbol}·s", ((series.unit, 1), (SECOND, 1)))
        if series.meaning == "interval_average"
        else series.unit
    )
    return result, unit


def counter_to_intervals(
    series: EnergySeries,
    *,
    rollover: float | None = None,
    reset_increments: Any | None = None,
) -> EnergySeries:
    """Difference a monotone counter; decreases need explicit reset evidence.

    ``rollover`` declares a single-wrap modulus. ``reset_increments`` supplies
    authoritative consumed amounts for decreasing intervals (same value PyTree
    with edge-aligned leaves); it is not a guessed reset-to-zero rule.
    """
    if series.meaning != "cumulative":
        raise ValueError("counter conversion requires cumulative samples")
    if rollover is not None and reset_increments is not None:
        raise ValueError("choose rollover or explicit reset increments, not both")
    if rollover is not None and (
        not isinstance(rollover, (int, float)) or not 0 < rollover < float("inf")
    ):
        raise ValueError("rollover modulus must be finite and positive")
    samples = series.samples
    axis = len(samples.support.series_shape)
    if reset_increments is not None and jax.tree_util.tree_structure(
        reset_increments
    ) != jax.tree_util.tree_structure(samples.values):
        raise ValueError("reset increments must match the counter value PyTree")

    def differences(value, mask, reset):
        left = jnp.take(value, jnp.arange(value.shape[axis] - 1), axis=axis)
        right = jnp.take(value, jnp.arange(1, value.shape[axis]), axis=axis)
        valid = jnp.take(mask, jnp.arange(value.shape[axis] - 1), axis=axis) & jnp.take(
            mask, jnp.arange(1, value.shape[axis]), axis=axis
        )
        delta = right - left
        if rollover is not None:
            value = eqx.error_if(
                value,
                jnp.any(mask & ((value < 0) | (value >= rollover))),
                "counter readings must lie within the declared modulus",
            )
            # Keep validation connected to the returned numerical expression.
            delta = jnp.diff(value, axis=axis)
            delta = jnp.where(delta < 0, delta + rollover, delta)
        elif reset is not None:
            reset = jnp.asarray(reset)
            if reset.shape != delta.shape:
                raise ValueError(
                    "reset increments must have the edge-aligned counter shape"
                )
            delta = jnp.where(delta < 0, reset, delta)
        expansion = (1,) * (value.ndim - axis - 1)
        active = (
            samples.support.edge_valid.reshape(
                samples.support.edge_valid.shape + expansion
            )
            & valid
        )
        delta = eqx.error_if(
            delta,
            jnp.any(active & ((delta < 0) | ~jnp.isfinite(delta))),
            "counter decreased without valid explicit reset or rollover evidence",
        )
        return delta, valid

    values, tree = jax.tree_util.tree_flatten(samples.values)
    masks = jax.tree_util.tree_leaves(_leaf_masks(samples))
    resets = (
        [None] * len(values)
        if reset_increments is None
        else jax.tree_util.tree_leaves(reset_increments)
    )
    pairs = [
        differences(value, mask, reset)
        for value, mask, reset in zip(values, masks, resets, strict=True)
    ]
    output = SampledSeries(
        samples.support,
        jax.tree_util.tree_unflatten(tree, [pair[0] for pair in pairs]),
        alignment="edge",
        value_valid=jax.tree_util.tree_unflatten(tree, [pair[1] for pair in pairs]),
        series_id=f"{samples.series_id}:increments",
    )
    return _with_samples(
        series, output, "interval_integral", "explicit-counter-differencing"
    )


__all__ = [
    "EnergySeries",
    "rebin_energy_series",
    "integrate_energy_series",
    "counter_to_intervals",
]
