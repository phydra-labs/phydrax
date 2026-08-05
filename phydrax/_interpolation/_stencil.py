#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._types import InterpolationResult, MaskMode


class GatherStencil(StrictModule):
    """A fixed-capacity sparse linear map from source sites to query sites."""

    indices: Array
    weights: Array
    valid: Array
    support: Array
    source_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        indices: ArrayLike,
        weights: ArrayLike,
        source_size: int,
        valid: ArrayLike | None = None,
        support: ArrayLike | None = None,
    ):
        indices_ = jnp.asarray(indices)
        if not jnp.issubdtype(indices_.dtype, jnp.integer):
            raise TypeError("GatherStencil indices must have an integer dtype.")
        if indices_.ndim < 1 or int(indices_.shape[-1]) <= 0:
            raise ValueError(
                "GatherStencil indices must end in a non-empty stencil axis."
            )

        weights_ = jnp.asarray(weights)
        if not jnp.issubdtype(weights_.dtype, jnp.inexact):
            weights_ = weights_.astype(float)
        if weights_.shape != indices_.shape:
            raise ValueError("GatherStencil weights must match indices shape.")

        source_size_ = int(source_size)
        if source_size_ <= 0:
            raise ValueError("GatherStencil source_size must be positive.")

        valid_ = (
            jnp.ones(indices_.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != indices_.shape:
            raise ValueError("GatherStencil valid must match indices shape.")

        query_shape = indices_.shape[:-1]
        support_ = (
            jnp.any(valid_, axis=-1)
            if support is None
            else jnp.asarray(support, dtype=bool)
        )
        if support_.shape != query_shape:
            raise ValueError("GatherStencil support must match the indices query shape.")

        self.indices = indices_
        self.weights = weights_
        self.valid = valid_
        self.support = support_
        self.source_size = source_size_


def gather_patches(
    values: ArrayLike,
    stencil: GatherStencil,
    /,
) -> tuple[Array, Array]:
    """Gather source payloads with invalid indices made numerically inert."""
    array = jnp.asarray(values)
    if array.ndim < 1 or int(array.shape[0]) != stencil.source_size:
        raise ValueError(
            "Gather source values must have one leading entry per source site."
        )
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)

    in_bounds = (stencil.indices >= 0) & (stencil.indices < stencil.source_size)
    array = eqx.error_if(
        array,
        jnp.any(stencil.valid & ~in_bounds),
        "A valid gather stencil index is outside the source array.",
    )
    valid = stencil.valid & in_bounds
    safe_indices = jnp.where(valid, stencil.indices, 0)
    return array[safe_indices], valid


def apply_gather_stencil(
    values: ArrayLike,
    stencil: GatherStencil,
    /,
    *,
    source_mask: ArrayLike | None = None,
    mask_mode: MaskMode = "strict",
) -> InterpolationResult:
    """Apply a weighted gather while preserving explicit query support."""
    if mask_mode not in ("reject", "renormalize", "strict"):
        raise ValueError("mask_mode must be 'reject', 'renormalize', or 'strict'.")

    patches, valid = gather_patches(values, stencil)
    if source_mask is not None:
        mask = jnp.asarray(source_mask, dtype=bool)
        if mask.shape != (stencil.source_size,):
            raise ValueError(
                f"source_mask must have shape {(stencil.source_size,)}, got {mask.shape}."
            )
        if mask_mode == "reject":
            patches = eqx.error_if(
                patches,
                jnp.logical_not(jnp.all(mask)),
                "Gather interpolation in reject mode does not permit source holes.",
            )
        else:
            safe_indices = jnp.where(valid, stencil.indices, 0)
            valid = valid & mask[safe_indices]

    weights = jnp.where(valid, stencil.weights, 0)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(stencil.weights)))
    tolerance = jnp.finfo(stencil.weights.real.dtype).eps * scale

    if source_mask is not None and mask_mode == "renormalize":
        weight_sum = jnp.sum(weights, axis=-1)
        support = stencil.support & (jnp.abs(weight_sum) > tolerance)
        weights = weights / jnp.where(support, weight_sum, 1.0)[..., None]
    else:
        material = jnp.abs(stencil.weights) > tolerance
        support = stencil.support & jnp.all(valid | ~material, axis=-1)

    payload_ndim = patches.ndim - weights.ndim
    expanded_valid = valid.reshape(valid.shape + (1,) * payload_ndim)
    safe_patches = jnp.where(
        expanded_valid,
        patches,
        jnp.zeros((), dtype=patches.dtype),
    )
    expanded_weights = weights.reshape(weights.shape + (1,) * payload_ndim)
    output = jnp.sum(
        safe_patches * expanded_weights.astype(patches.dtype),
        axis=-1 - payload_ndim,
    )
    support_expanded = support.reshape(support.shape + (1,) * payload_ndim)
    output = jnp.where(support_expanded, output, jnp.zeros((), dtype=output.dtype))
    return InterpolationResult(output, support)


__all__ = ["GatherStencil", "apply_gather_stencil", "gather_patches"]
