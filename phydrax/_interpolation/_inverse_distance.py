#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._stencil import GatherStencil
from ._types import InterpolationCapabilities


SnapPolicy: TypeAlias = Literal["average", "first"]


INVERSE_DISTANCE_CAPABILITIES = InterpolationCapabilities(
    partition_of_unity=True,
    nonnegative_value_weights=True,
    local_support=False,
    mask_renormalizable=True,
    tensor_product_composable=False,
    maximum_explicit_derivative_order=0,
)


def _candidate_parameter(
    name: str,
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
    *,
    positive: bool,
) -> Array:
    raw = jnp.asarray(value)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    parameter = jnp.broadcast_to(raw.astype(float), shape)
    invalid = ~jnp.isfinite(parameter)
    invalid = invalid | (parameter <= 0.0 if positive else parameter < 0.0)
    return eqx.error_if(
        parameter,
        jnp.any(invalid),
        f"{name} must be finite and {'positive' if positive else 'non-negative'}.",
    )


def inverse_distance_stencil(
    indices: ArrayLike,
    squared_distances: ArrayLike,
    /,
    *,
    source_size: int,
    valid: ArrayLike | None = None,
    power: ArrayLike = 2.0,
    regularization: ArrayLike = 0.0,
    snap_tolerance_squared: ArrayLike = 0.0,
    snap_policy: SnapPolicy = "first",
    snap_inclusive: bool = False,
    support: ArrayLike | None = None,
) -> GatherStencil:
    """Build normalized inverse-distance weights over supplied candidates.

    Distances are squared physical distances. ``power=p`` produces weights
    proportional to ``(distance² + regularization)^(-p/2)``. Exact candidates
    always snap, including when the configured snap tolerance is zero.
    """
    if snap_policy not in ("average", "first"):
        raise ValueError("snap_policy must be 'average' or 'first'.")

    indices_ = jnp.asarray(indices)
    distance_input = jnp.asarray(squared_distances)
    if jnp.issubdtype(distance_input.dtype, jnp.complexfloating):
        raise TypeError("Squared distances must be real-valued.")
    distances = distance_input.astype(float)
    if indices_.shape != distances.shape or indices_.ndim < 1:
        raise ValueError(
            "Inverse-distance indices and squared_distances must have matching "
            "shapes ending in a candidate axis."
        )
    if int(indices_.shape[-1]) <= 0:
        raise ValueError("Inverse-distance stencils require at least one candidate.")
    if not jnp.issubdtype(indices_.dtype, jnp.integer):
        raise TypeError("Inverse-distance indices must have an integer dtype.")

    valid_ = (
        jnp.ones(indices_.shape, dtype=bool)
        if valid is None
        else jnp.asarray(valid, dtype=bool)
    )
    if valid_.shape != indices_.shape:
        raise ValueError("valid must match the inverse-distance candidate shape.")
    distances = eqx.error_if(
        distances,
        jnp.any(valid_ & ((distances < 0.0) | ~jnp.isfinite(distances))),
        "Valid squared distances must be finite and non-negative.",
    )

    candidate_shape = tuple(int(size) for size in indices_.shape)
    exponent = _candidate_parameter(
        "power",
        power,
        candidate_shape,
        positive=True,
    )
    shift = _candidate_parameter(
        "regularization",
        regularization,
        candidate_shape,
        positive=False,
    )
    tolerance = _candidate_parameter(
        "snap_tolerance_squared",
        snap_tolerance_squared,
        candidate_shape,
        positive=False,
    )

    masked_distances = jnp.where(valid_, distances, jnp.inf)
    nearest = jnp.argmin(masked_distances, axis=-1)
    nearest_distance = jnp.take_along_axis(
        masked_distances,
        nearest[..., None],
        axis=-1,
    )[..., 0]
    nearest_tolerance = jnp.take_along_axis(
        tolerance,
        nearest[..., None],
        axis=-1,
    )[..., 0]
    within_tolerance = (
        nearest_distance <= nearest_tolerance
        if bool(snap_inclusive)
        else nearest_distance < nearest_tolerance
    )
    should_snap = (nearest_distance == 0.0) | within_tolerance

    if snap_policy == "first":
        snap_weights = jax.nn.one_hot(
            nearest,
            int(indices_.shape[-1]),
            dtype=distances.dtype,
        )
        snap_weights = snap_weights * valid_.astype(distances.dtype)
    else:
        snap_candidates = valid_ & (
            (distances <= tolerance) if bool(snap_inclusive) else (distances < tolerance)
        )
        snap_candidates = snap_candidates | (valid_ & (distances == 0.0))
        snap_count = jnp.sum(snap_candidates, axis=-1, keepdims=True)
        snap_weights = snap_candidates.astype(distances.dtype) / jnp.maximum(
            snap_count,
            1,
        )
        should_snap = should_snap & (snap_count[..., 0] > 0)

    safe_denominator = jnp.where(
        valid_,
        jnp.where(should_snap[..., None], 1.0, distances + shift),
        1.0,
    )
    raw_weights = jnp.where(
        valid_,
        safe_denominator ** (-0.5 * exponent),
        0.0,
    )
    raw_sum = jnp.sum(raw_weights, axis=-1, keepdims=True)
    normalized = raw_weights / jnp.where(raw_sum > 0.0, raw_sum, 1.0)
    weights = jnp.where(should_snap[..., None], snap_weights, normalized)

    query_shape = indices_.shape[:-1]
    base_support = (
        jnp.ones(query_shape, dtype=bool)
        if support is None
        else jnp.asarray(support, dtype=bool)
    )
    if base_support.shape != query_shape:
        raise ValueError("support must match the inverse-distance query shape.")
    has_candidate = jnp.any(valid_, axis=-1)
    finite_weight = jnp.all(jnp.isfinite(weights), axis=-1)
    map_support = base_support & has_candidate & finite_weight

    return GatherStencil(
        indices=indices_,
        weights=weights,
        source_size=source_size,
        valid=valid_,
        support=map_support,
    )


__all__ = [
    "INVERSE_DISTANCE_CAPABILITIES",
    "SnapPolicy",
    "inverse_distance_stencil",
]
