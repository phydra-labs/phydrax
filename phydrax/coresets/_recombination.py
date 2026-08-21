#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._measure_weights import log_weights_from_normalized, normalized_weights
from .._strict import StrictModule
from ._types import CoresetSelection, MomentRecombinationDiagnostics


class MomentRecombination(StrictModule):
    """Hierarchical positive reduction preserving supplied feature moments."""

    rcond: float | None = eqx.field(static=True)
    tree_reduction_factor: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        rcond: float | None = None,
        tree_reduction_factor: int = 2,
    ):
        if rcond is not None:
            condition = float(rcond)
            if not math.isfinite(condition) or condition <= 0.0:
                raise ValueError("rcond must be finite and strictly positive.")
        factor = int(tree_reduction_factor)
        if factor < 2:
            raise ValueError("tree_reduction_factor must be at least two.")
        self.rcond = None if rcond is None else float(rcond)
        self.tree_reduction_factor = factor


def _rank_and_tolerance(
    augmented: Array,
    weights: Array,
    rcond: float | None,
    /,
) -> tuple[Array, Array]:
    weighted = jnp.sqrt(jnp.maximum(weights, 0.0))[:, None] * augmented
    singular_values = jnp.linalg.svd(weighted, full_matrices=False, compute_uv=False)
    dtype = augmented.dtype
    relative = (
        jnp.asarray(rcond, dtype=dtype)
        if rcond is not None
        else jnp.asarray(
            jnp.finfo(dtype).eps * max(augmented.shape),
            dtype=dtype,
        )
    )
    largest = jnp.max(singular_values, initial=jnp.asarray(0.0, dtype=dtype))
    threshold = relative * largest
    rank = jnp.sum(singular_values > threshold, dtype=jnp.int32)
    weight_tolerance = jnp.asarray(jnp.finfo(dtype).eps * augmented.shape[0] * 16.0)
    return rank, weight_tolerance


def _eliminate_to_rank(
    augmented: Array,
    weights: Array,
    /,
    *,
    rcond: float | None,
) -> tuple[Array, Array]:
    count, feature_count = augmented.shape
    rank, weight_tolerance = _rank_and_tolerance(augmented, weights, rcond)
    slots = feature_count + 1
    slot_ids = jnp.arange(slots)

    def eliminate_once(current: Array) -> Array:
        active = current > weight_tolerance
        active_count = jnp.sum(active, dtype=jnp.int32)
        selected = jnp.nonzero(active, size=slots, fill_value=0)[0]
        valid_slots = slot_ids < jnp.minimum(active_count, slots)
        selected_features = augmented[selected].T
        fixed_zero = jnp.diag((~valid_slots).astype(augmented.dtype))
        system = jnp.concatenate((selected_features, fixed_zero), axis=0)
        _, _, vectors = jnp.linalg.svd(system, full_matrices=False)
        null_vector = jnp.where(valid_slots, vectors[-1], 0.0)
        null_scale = jnp.max(jnp.abs(null_vector), initial=0.0)
        null_vector = jnp.where(
            null_scale > 0.0,
            null_vector / null_scale,
            null_vector,
        )
        selected_weights = current[selected]
        direction_tolerance = jnp.asarray(jnp.finfo(augmented.dtype).eps * 32.0)
        positive = null_vector > direction_tolerance
        negative = null_vector < -direction_tolerance
        positive_step = jnp.min(
            jnp.where(positive & valid_slots, selected_weights / null_vector, jnp.inf),
            initial=jnp.inf,
        )
        negative_step = jnp.min(
            jnp.where(
                negative & valid_slots,
                selected_weights / -null_vector,
                jnp.inf,
            ),
            initial=jnp.inf,
        )
        use_positive = positive_step <= negative_step
        step = jnp.where(use_positive, positive_step, negative_step)
        direction = jnp.where(use_positive, null_vector, -null_vector)
        proposed = selected_weights - step * direction
        proposed = jnp.where(
            valid_slots,
            jnp.where(proposed <= weight_tolerance, 0.0, jnp.maximum(proposed, 0.0)),
            selected_weights,
        )
        delta = jnp.where(valid_slots, proposed - selected_weights, 0.0)
        updated = current.at[selected].add(delta)
        can_reduce = (
            (active_count > rank)
            & jnp.isfinite(step)
            & (null_scale > direction_tolerance)
        )
        return jnp.where(can_reduce, updated, current)

    def body(_, current):
        return eliminate_once(current)

    reduced = jax.lax.fori_loop(0, count, body, weights)
    reduced = jnp.where(reduced > weight_tolerance, reduced, 0.0)
    return reduced, rank


def _standardized_augmented(features: Array, weights: Array, /) -> Array:
    feature_count = int(features.shape[1])
    if feature_count == 0:
        return jnp.ones((features.shape[0], 1), dtype=features.dtype)
    mean = weights @ features
    centered = features - mean
    variance = weights @ (centered * centered)
    scale = jnp.sqrt(jnp.maximum(variance, 0.0))
    floor = jnp.asarray(jnp.finfo(features.dtype).eps * 64.0)
    safe_scale = jnp.where(scale > floor, scale, 1.0)
    standardized = centered / safe_scale
    return jnp.concatenate(
        (jnp.ones((features.shape[0], 1), dtype=features.dtype), standardized),
        axis=1,
    )


def moment_recombine(
    features: Array,
    method: MomentRecombination | None = None,
    /,
    *,
    log_weights: Array | None = None,
    mask: Array | None = None,
) -> CoresetSelection:
    """Reduce one positive empirical measure while preserving feature moments."""
    config = MomentRecombination() if method is None else method
    if not isinstance(config, MomentRecombination):
        raise TypeError("method must be a MomentRecombination.")
    feature_values = jnp.asarray(features, dtype=float)
    if feature_values.ndim != 2:
        raise ValueError("features must have shape (source_points, feature_count).")
    source_points, supplied_feature_count = feature_values.shape
    if source_points < 1:
        raise ValueError("Moment recombination requires at least one source point.")
    rows_valid = jnp.all(jnp.isfinite(feature_values), axis=1)
    weights, _, input_valid, log_source_mass = normalized_weights(
        source_points,
        log_weights=log_weights,
        mask=mask,
        rows_valid=rows_valid,
    )
    safe_features = jnp.nan_to_num(feature_values)
    augmented = _standardized_augmented(safe_features, weights)
    capacity = supplied_feature_count + 1
    factor = config.tree_reduction_factor
    depth = (
        0
        if source_points <= capacity
        else math.ceil(math.log(source_points / capacity, factor))
    )
    padded_count = capacity * factor**depth
    padding = padded_count - source_points
    current_features = jnp.pad(augmented, ((0, padding), (0, 0)))
    current_weights = jnp.pad(weights, (0, padding))
    current_indices = jnp.pad(jnp.arange(source_points, dtype=jnp.int32), (0, padding))
    current_count = padded_count

    for _ in range(depth):
        cluster_count = factor * capacity
        cluster_size = current_count // cluster_count
        order = jnp.argsort(current_weights)
        groups = order.reshape((cluster_size, cluster_count)).T
        grouped_weights = current_weights[groups]
        cluster_mass = jnp.sum(grouped_weights, axis=1)
        weighted_features = jnp.sum(
            grouped_weights[..., None] * current_features[groups],
            axis=1,
        )
        safe_mass = jnp.where(cluster_mass > 0.0, cluster_mass, 1.0)
        centroids = weighted_features / safe_mass[:, None]
        reduced_mass, _ = _eliminate_to_rank(
            centroids,
            cluster_mass,
            rcond=config.rcond,
        )
        selected_clusters = jnp.nonzero(
            reduced_mass > 0.0,
            size=capacity,
            fill_value=0,
        )[0]
        selected_mass = reduced_mass[selected_clusters]
        original_mass = cluster_mass[selected_clusters]
        multiplier = jnp.where(original_mass > 0.0, selected_mass / original_mass, 0.0)
        selected_groups = groups[selected_clusters]
        current_features = current_features[selected_groups].reshape(
            (capacity * cluster_size, capacity)
        )
        current_indices = current_indices[selected_groups].reshape(
            (capacity * cluster_size,)
        )
        current_weights = (
            grouped_weights[selected_clusters] * multiplier[:, None]
        ).reshape((capacity * cluster_size,))
        current_count = capacity * cluster_size

    final_weights, numerical_rank = _eliminate_to_rank(
        current_features,
        current_weights,
        rcond=config.rcond,
    )
    order = jnp.argsort(final_weights)[::-1]
    selected_indices = current_indices[order]
    selected_weights = final_weights[order]
    dtype = selected_weights.dtype
    active_tolerance = jnp.asarray(jnp.finfo(dtype).eps * capacity * 16.0)
    selected_mask = selected_weights > active_tolerance
    selected_weights = jnp.where(selected_mask, selected_weights, 0.0)
    selected_total = jnp.sum(selected_weights)
    selected_weights = jnp.where(
        selected_total > 0.0,
        selected_weights / selected_total,
        selected_weights,
    )
    source_moments = weights @ safe_features
    selected_moments = selected_weights @ safe_features[selected_indices]
    mass_error = jnp.abs(jnp.sum(selected_weights) - jnp.sum(weights))
    max_moment_error = (
        jnp.max(jnp.abs(selected_moments - source_moments))
        if supplied_feature_count
        else jnp.asarray(0.0, dtype=dtype)
    )
    active_points = jnp.sum(selected_mask, dtype=jnp.int32)
    minimum_weight = jnp.min(
        jnp.where(selected_mask, selected_weights, jnp.inf),
        initial=jnp.inf,
    )
    output_valid = (
        input_valid & (active_points > 0) & jnp.all(jnp.isfinite(selected_weights))
    )
    selected_mask = selected_mask & output_valid
    selected_weights = jnp.where(selected_mask, selected_weights, 0.0)
    diagnostics = MomentRecombinationDiagnostics(
        valid=output_valid,
        active_points=active_points,
        numerical_rank=numerical_rank,
        mass_error=mass_error,
        max_moment_error=max_moment_error,
        minimum_weight=minimum_weight,
        log_source_mass=log_source_mass,
        source_points=source_points,
        capacity=capacity,
        feature_count=supplied_feature_count,
        tree_depth=depth,
    )
    return CoresetSelection(
        selected_indices,
        log_weights_from_normalized(selected_weights, selected_mask),
        selected_mask,
        diagnostics,
        method="moment-recombination",
    )


__all__ = ["MomentRecombination", "moment_recombine"]
