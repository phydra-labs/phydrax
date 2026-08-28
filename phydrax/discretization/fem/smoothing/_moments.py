#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._common import SmoothingPatchGeometry, SmoothingPatchLayout


def boundary_moment(
    layout: SmoothingPatchLayout,
    geometry: SmoothingPatchGeometry,
    /,
) -> Array:
    """Compute area-normalized boundary moments Gbar[patch, dof, dim]."""

    if not isinstance(layout, SmoothingPatchLayout) or not isinstance(
        geometry, SmoothingPatchGeometry
    ):
        raise TypeError("Expected smoothing layout and geometry.")
    weighted_boundary = (
        geometry.boundary_lengths[..., None] * layout.rule_weights[None, None, :]
    )
    numerator = oe.contract(
        "peq,peql,ped->pld",
        weighted_boundary,
        layout.boundary_shape_values,
        geometry.boundary_normals,
    )
    moment = numerator / geometry.area[:, None, None]
    return jnp.where(layout.dof_valid[..., None], moment, 0.0)


def shape_average(shape_values: ArrayLike, valid: ArrayLike | None = None, /) -> Array:
    """Average nodal shape values over explicit smoothing sample sites."""

    values = jnp.asarray(shape_values)
    if values.ndim < 2:
        raise ValueError("shape_values must contain sample and local-DOF axes.")
    if valid is None:
        return jnp.mean(values, axis=-2)
    valid_ = jnp.asarray(valid, dtype=bool)
    if valid_.shape != values.shape[:-1]:
        raise ValueError("Shape-average validity must match all non-DOF axes.")
    count = jnp.sum(valid_, axis=-1, keepdims=True)
    return jnp.sum(jnp.where(valid_[..., None], values, 0.0), axis=-2) / count


def primitive_volume_moment(
    layout: SmoothingPatchLayout,
    geometry: SmoothingPatchGeometry,
    primitive_values: ArrayLike,
    radial_axis: int = 0,
    /,
) -> Array:
    """Convert an analytic shape primitive into an area-normalized volume moment."""

    primitive = jnp.asarray(primitive_values)
    if primitive.shape != layout.boundary_shape_values.shape:
        raise ValueError("Primitive values must match boundary shape-value layout.")
    axis = int(radial_axis)
    if axis < 0 or axis >= geometry.boundary_normals.shape[-1]:
        raise ValueError("radial_axis is outside the smoothing geometry dimension.")
    weighted = (
        geometry.boundary_lengths[..., None]
        * layout.rule_weights[None, None, :]
        * geometry.boundary_normals[..., axis, None]
    )
    numerator = jnp.sum(weighted[..., None] * primitive, axis=(1, 2))
    return jnp.where(
        layout.dof_valid,
        numerator / geometry.area[:, None],
        0.0,
    )


def smoothed_symmetric_gradient_matrix(moment: ArrayLike, /) -> Array:
    """Build 2-D engineering-strain Bbar from Gbar moments."""

    gradient = jnp.asarray(moment)
    if gradient.ndim != 3 or gradient.shape[-1] != 2:
        raise ValueError("2-D smoothing gradient must have shape (patch, dof, 2).")
    gx = gradient[..., 0]
    gy = gradient[..., 1]
    zeros = jnp.zeros_like(gx)
    first = jnp.stack((gx, zeros), axis=-1)
    second = jnp.stack((zeros, gy), axis=-1)
    shear = jnp.stack((gy, gx), axis=-1)
    return jnp.stack((first, second, shear), axis=-2)


__all__ = [
    "boundary_moment",
    "primitive_volume_moment",
    "shape_average",
    "smoothed_symmetric_gradient_matrix",
]
