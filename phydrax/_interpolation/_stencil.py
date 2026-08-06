#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..sparse import gather_routes, route_reduce, RowRelation
from ._types import InterpolationResult, MaskMode


class GatherStencil(StrictModule):
    """A fixed-capacity sparse linear map from source sites to query sites."""

    relation: RowRelation
    weights: Array
    support: Array

    def __init__(
        self,
        *,
        indices: ArrayLike,
        weights: ArrayLike,
        source_size: int,
        valid: ArrayLike | None = None,
        support: ArrayLike | None = None,
        case_shape: tuple[int, ...] = (),
    ):
        relation = RowRelation(
            indices,
            source_size=source_size,
            valid=valid,
            case_shape=case_shape,
        )
        weights_ = jnp.asarray(weights)
        if not jnp.issubdtype(weights_.dtype, jnp.inexact):
            weights_ = weights_.astype(float)
        if weights_.shape != relation.route_shape:
            raise ValueError("GatherStencil weights must match indices shape.")

        support_ = (
            jnp.any(relation.valid, axis=-1)
            if support is None
            else jnp.asarray(support, dtype=bool)
        )
        if support_.shape != relation.output_shape:
            raise ValueError(
                "GatherStencil support must match the relation output shape."
            )

        self.relation = relation
        self.weights = weights_
        self.support = support_

    @property
    def indices(self) -> Array:
        return self.relation.source_indices

    @property
    def valid(self) -> Array:
        return self.relation.valid

    @property
    def source_size(self) -> int:
        return self.relation.source_size

    @property
    def case_shape(self) -> tuple[int, ...]:
        return self.relation.case_shape


def gather_patches(
    values: ArrayLike,
    stencil: GatherStencil,
    /,
) -> tuple[Array, Array]:
    """Gather source payloads with invalid indices made numerically inert."""
    array = jnp.asarray(values)
    expected = stencil.relation.input_shape
    if (
        array.ndim < len(expected)
        or tuple(int(size) for size in array.shape[: len(expected)]) != expected
    ):
        raise ValueError(
            f"Gather source values must begin with source shape {expected}; "
            f"got {array.shape}."
        )
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return gather_routes(stencil.relation, array), stencil.valid


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
        if mask.shape != stencil.relation.input_shape:
            raise ValueError(
                "source_mask must have shape "
                f"{stencil.relation.input_shape}, got {mask.shape}."
            )
        if mask_mode == "reject":
            patches = eqx.error_if(
                patches,
                jnp.logical_not(jnp.all(mask)),
                "Gather interpolation in reject mode does not permit source holes.",
            )
        else:
            valid = valid & gather_routes(stencil.relation, mask)
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

    payload_ndim = patches.ndim - len(stencil.relation.route_shape)
    expanded_valid = valid.reshape(valid.shape + (1,) * payload_ndim)
    safe_patches = jnp.where(
        expanded_valid,
        patches,
        jnp.zeros((), dtype=patches.dtype),
    )
    expanded_weights = weights.reshape(weights.shape + (1,) * payload_ndim)
    messages = safe_patches * expanded_weights.astype(patches.dtype)
    output = route_reduce(stencil.relation, messages)
    support_expanded = support.reshape(support.shape + (1,) * payload_ndim)
    output = jnp.where(support_expanded, output, jnp.zeros((), dtype=output.dtype))
    return InterpolationResult(output, support)


__all__ = ["GatherStencil", "apply_gather_stencil", "gather_patches"]
