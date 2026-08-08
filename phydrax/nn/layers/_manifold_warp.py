#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey
from ._linear import Linear


class ManifoldWarpDiagnostics(StrictModule):
    """Tangent displacement, retracted routes, and interpolation weights."""

    tangent_displacement: Array
    transported_points: Array
    interpolation_weights: Array

    def __init__(
        self,
        *,
        tangent_displacement: Array,
        transported_points: Array,
        interpolation_weights: Array,
    ):
        self.tangent_displacement = jnp.asarray(tangent_displacement)
        self.transported_points = jnp.asarray(transported_points)
        self.interpolation_weights = jnp.asarray(interpolation_weights)


def sphere_tangent_projection(points: Array, vectors: Array, /) -> Array:
    """Project ambient vectors onto the tangent spaces of a unit sphere."""

    points_ = jnp.asarray(points)
    vectors_ = jnp.asarray(vectors)
    if points_.shape != vectors_.shape:
        raise ValueError("Sphere points and vectors must have matching shapes.")
    normal = points_ / jnp.maximum(
        jnp.linalg.norm(points_, axis=-1, keepdims=True),
        jnp.finfo(points_.dtype).eps,
    )
    return vectors_ - jnp.sum(vectors_ * normal, axis=-1, keepdims=True) * normal


def sphere_retraction(points: Array, tangent: Array, /) -> Array:
    """Retract a tangent update to the unit sphere by radial normalization."""

    moved = jnp.asarray(points) + jnp.asarray(tangent)
    return moved / jnp.maximum(
        jnp.linalg.norm(moved, axis=-1, keepdims=True),
        jnp.finfo(moved.dtype).eps,
    )


class ManifoldMultiheadWarp(StrictModule):
    """Learned tangent-space multihead warp on aligned manifold samples.

    The caller supplies the manifold's tangent projector and retraction. Sampling
    uses a differentiable masked Gaussian kernel in the embedding coordinates.
    """

    value_projection: Linear
    displacement_hidden: Linear
    displacement_condition: Linear | None
    displacement_output: Linear
    tangent_projection: Callable[[Array, Array], Array] = eqx.field(static=True)
    retraction: Callable[[Array, Array], Array] = eqx.field(static=True)
    ambient_dim: int
    in_channels: int
    out_channels: int
    num_heads: int
    conditioning_size: int
    kernel_scale: float

    def __init__(
        self,
        *,
        ambient_dim: int,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        tangent_projection: Callable[[Array, Array], Array],
        retraction: Callable[[Array, Array], Array],
        conditioning_size: int = 0,
        displacement_width: int | None = None,
        kernel_scale: float = 0.2,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.ambient_dim = int(ambient_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.conditioning_size = int(conditioning_size)
        self.kernel_scale = float(kernel_scale)
        self.tangent_projection = tangent_projection
        self.retraction = retraction
        if (
            min(
                self.ambient_dim,
                self.in_channels,
                self.out_channels,
                self.num_heads,
            )
            <= 0
        ):
            raise ValueError("Manifold warp dimensions and channels must be positive.")
        if self.conditioning_size < 0:
            raise ValueError("conditioning_size must be non-negative.")
        if self.out_channels % self.num_heads:
            raise ValueError("out_channels must be divisible by num_heads.")
        if self.kernel_scale <= 0.0:
            raise ValueError("kernel_scale must be positive.")
        hidden = (
            self.out_channels if displacement_width is None else int(displacement_width)
        )
        if hidden <= 0:
            raise ValueError("displacement_width must be positive.")
        value_key, hidden_key, condition_key, output_key = jr.split(key, 4)
        self.value_projection = Linear(
            in_size=self.in_channels,
            out_size=self.out_channels,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=value_key,
        )
        self.displacement_hidden = Linear(
            in_size=self.in_channels,
            out_size=hidden,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=hidden_key,
        )
        self.displacement_condition = (
            None
            if self.conditioning_size == 0
            else Linear(
                in_size=self.conditioning_size,
                out_size=hidden,
                activation=None,
                rwf=False,
                use_bias=False,
                key=condition_key,
            )
        )
        self.displacement_output = Linear(
            in_size=hidden,
            out_size=self.num_heads * self.ambient_dim,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=output_key,
        )

    def _prepare(
        self,
        values: Array,
        points: Array,
        source_mask: Array | None,
        /,
    ) -> tuple[Array, Array, Array, tuple[int, ...], int]:
        field = jnp.asarray(values)
        if field.ndim < 2 or int(field.shape[-1]) != self.in_channels:
            raise ValueError(
                "Manifold warp values must end in points and configured channels."
            )
        case_shape = tuple(int(size) for size in field.shape[:-2])
        count = int(field.shape[-2])
        geometry = jnp.asarray(points, dtype=jnp.result_type(field.dtype, float))
        if geometry.shape == (count, self.ambient_dim):
            geometry = jnp.broadcast_to(
                geometry,
                case_shape + (count, self.ambient_dim),
            )
        elif geometry.shape != case_shape + (count, self.ambient_dim):
            raise ValueError(
                "Manifold points must be shared or match the value case shape; "
                f"got {geometry.shape}."
            )
        if source_mask is None:
            mask = jnp.ones(case_shape + (count,), dtype=bool)
        else:
            mask = jnp.asarray(source_mask, dtype=bool)
            if mask.shape == (count,):
                mask = jnp.broadcast_to(mask, case_shape + (count,))
            elif mask.shape != case_shape + (count,):
                raise ValueError("Manifold source_mask shape does not match samples.")
            field = eqx.error_if(
                field,
                jnp.logical_not(jnp.all(jnp.any(mask, axis=-1))),
                "Every manifold case must contain at least one valid source point.",
            )
        field = jnp.where(mask[..., None], field, 0)
        geometry = jnp.where(mask[..., None], geometry, 0)
        return field, geometry, mask, case_shape, count

    def displacement(
        self,
        values: Array,
        points: Array,
        /,
        *,
        condition: Array | None = None,
        source_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        field, geometry, _, case_shape, count = self._prepare(
            values,
            points,
            source_mask,
        )
        hidden = self.displacement_hidden(field)
        if self.displacement_condition is None:
            if condition is not None:
                raise ValueError(
                    "condition must be None for an unconditioned manifold warp."
                )
        else:
            if condition is None:
                raise ValueError("Conditioned manifold warp requires condition values.")
            condition_ = jnp.asarray(condition)
            expected = case_shape + (self.conditioning_size,)
            if condition_.shape != expected:
                raise ValueError(
                    f"Manifold condition must have shape {expected}; got {condition_.shape}."
                )
            hidden = hidden + self.displacement_condition(condition_).reshape(
                case_shape + (1, int(hidden.shape[-1]))
            )
        ambient = self.displacement_output(jax.nn.gelu(hidden)).reshape(
            case_shape + (count, self.num_heads, self.ambient_dim)
        )
        repeated_points = jnp.broadcast_to(
            geometry[..., :, None, :],
            ambient.shape,
        )
        tangent = self.tangent_projection(repeated_points, ambient)
        transported = self.retraction(repeated_points, tangent)
        if tangent.shape != ambient.shape or transported.shape != ambient.shape:
            raise ValueError(
                "Manifold tangent projection and retraction must preserve route shape."
            )
        return tangent, transported

    def _evaluate(
        self,
        values: Array,
        points: Array,
        /,
        *,
        condition: Array | None,
        source_mask: Array | None,
    ) -> tuple[Array, ManifoldWarpDiagnostics]:
        field, geometry, mask, case_shape, count = self._prepare(
            values,
            points,
            source_mask,
        )
        tangent, transported = self.displacement(
            field,
            geometry,
            condition=condition,
            source_mask=mask,
        )
        head_channels = self.out_channels // self.num_heads
        projected = self.value_projection(field).reshape(
            case_shape + (count, self.num_heads, head_channels)
        )
        head_axis = len(case_shape) + 1
        projected = jnp.moveaxis(projected, head_axis, len(case_shape))
        routes = jnp.moveaxis(transported, head_axis, len(case_shape))
        sources = jnp.broadcast_to(
            geometry[..., None, :, :],
            case_shape + (self.num_heads, count, self.ambient_dim),
        )
        squared_distance = jnp.sum(
            (routes[..., :, None, :] - sources[..., None, :, :]) ** 2,
            axis=-1,
        )
        head_mask = jnp.broadcast_to(
            mask[..., None, None, :],
            case_shape + (self.num_heads, count, count),
        )
        logits = -0.5 * squared_distance / (self.kernel_scale**2)
        logits = jnp.where(head_mask, logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        sampled = oe.contract("...hqs,...hsc->...hqc", weights, projected)
        sampled = jnp.moveaxis(sampled, len(case_shape), head_axis)
        output = sampled.reshape(case_shape + (count, self.out_channels))
        output = output * mask[..., None].astype(output.dtype)
        diagnostics = ManifoldWarpDiagnostics(
            tangent_displacement=tangent,
            transported_points=transported,
            interpolation_weights=jnp.moveaxis(
                weights,
                len(case_shape),
                len(case_shape) + 2,
            ),
        )
        return output, diagnostics

    def diagnostics(
        self,
        values: Array,
        points: Array,
        /,
        *,
        condition: Array | None = None,
        source_mask: Array | None = None,
    ) -> ManifoldWarpDiagnostics:
        return self._evaluate(
            values,
            points,
            condition=condition,
            source_mask=source_mask,
        )[1]

    def __call__(
        self,
        values: Array,
        points: Array,
        /,
        *,
        condition: Array | None = None,
        source_mask: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        del key
        return self._evaluate(
            values,
            points,
            condition=condition,
            source_mask=source_mask,
        )[0]


__all__ = [
    "ManifoldMultiheadWarp",
    "ManifoldWarpDiagnostics",
    "sphere_retraction",
    "sphere_tangent_projection",
]
