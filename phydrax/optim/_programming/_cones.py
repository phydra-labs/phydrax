#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import sqrt
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


def _scaled_norm(value: Array, /) -> Array:
    scale = jnp.max(jnp.abs(value), axis=-1, initial=0.0)
    safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    residual = scale * jnp.linalg.norm(value / safe_scale[..., None], axis=-1)
    return jnp.where(jnp.isinf(scale), jnp.inf, residual)


class AbstractConvexCone(StrictModule):
    """Closed convex cone over one trailing canonical-coordinate axis."""

    dimension: int = eqx.field(static=True)
    cone_id: str = eqx.field(static=True)

    def _validate(self, value: Any, /) -> Array:
        array = jnp.asarray(value)
        if array.ndim < 1 or int(array.shape[-1]) != self.dimension:
            raise ValueError(
                f"Cone value must end in shape ({self.dimension},); got {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Cone values must be real floating-point arrays.")
        return array

    @abc.abstractmethod
    def project(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def project_dual(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def interior_margin(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        """Distance to the nearest nonsmooth stratum of the dual projection."""
        raise NotImplementedError

    def residual(self, value: Any, /) -> Array:
        array = self._validate(value)
        return _scaled_norm(array - self.project(array))

    def dual_residual(self, value: Any, /) -> Array:
        array = self._validate(value)
        return _scaled_norm(array - self.project_dual(array))

    def contains(self, value: Any, /, *, tolerance: float = 0.0) -> Array:
        return self.residual(value) <= float(tolerance)

    def contains_dual(self, value: Any, /, *, tolerance: float = 0.0) -> Array:
        return self.dual_residual(value) <= float(tolerance)

    def complementarity(self, primal: Any, dual: Any, /) -> Array:
        primal_ = self._validate(primal)
        dual_ = self._validate(dual)
        return jnp.sum(primal_ * dual_, axis=-1)


class ZeroCone(AbstractConvexCone):
    """The singleton cone containing only the origin."""

    def __init__(self, dimension: int, /):
        size = int(dimension)
        if size < 0:
            raise ValueError("ZeroCone dimension must be non-negative.")
        self.dimension = size
        self.cone_id = canonical_fingerprint({"kind": "zero-cone", "dimension": size})

    def project(self, value: Any, /) -> Array:
        return jnp.zeros_like(self._validate(value))

    def project_dual(self, value: Any, /) -> Array:
        return self._validate(value)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        return -jnp.linalg.norm(array, axis=-1)

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        return jnp.full(array.shape[:-1], jnp.inf, dtype=array.dtype)


class NonnegativeCone(AbstractConvexCone):
    """Elementwise nonnegative orthant."""

    def __init__(self, dimension: int, /):
        size = int(dimension)
        if size < 0:
            raise ValueError("NonnegativeCone dimension must be non-negative.")
        self.dimension = size
        self.cone_id = canonical_fingerprint(
            {"kind": "nonnegative-cone", "dimension": size}
        )

    def project(self, value: Any, /) -> Array:
        return jnp.maximum(self._validate(value), 0.0)

    def project_dual(self, value: Any, /) -> Array:
        return self.project(value)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        if self.dimension == 0:
            return jnp.full(array.shape[:-1], jnp.inf, dtype=array.dtype)
        return jnp.min(array, axis=-1)

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        if self.dimension == 0:
            return jnp.full(array.shape[:-1], jnp.inf, dtype=array.dtype)
        return jnp.min(jnp.abs(array), axis=-1)


class SecondOrderCone(AbstractConvexCone):
    """Lorentz cone ``(t, x)`` satisfying ``||x||₂ <= t``."""

    def __init__(self, dimension: int, /):
        size = int(dimension)
        if size < 2:
            raise ValueError("SecondOrderCone dimension must be at least two.")
        self.dimension = size
        self.cone_id = canonical_fingerprint(
            {"kind": "second-order-cone", "dimension": size}
        )

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        scalar = array[..., :1]
        vector = array[..., 1:]
        norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
        safe_norm = jnp.maximum(norm, jnp.finfo(array.dtype).tiny)
        middle_scalar = 0.5 * (norm + scalar)
        middle_vector = 0.5 * (1.0 + scalar / safe_norm) * vector
        middle = jnp.concatenate((middle_scalar, middle_vector), axis=-1)
        inside = norm <= scalar
        polar = norm <= -scalar
        return jnp.where(inside, array, jnp.where(polar, jnp.zeros_like(array), middle))

    def project_dual(self, value: Any, /) -> Array:
        return self.project(value)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        return array[..., 0] - jnp.linalg.norm(array[..., 1:], axis=-1)

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        scalar = array[..., 0]
        norm = jnp.linalg.norm(array[..., 1:], axis=-1)
        return jnp.minimum(jnp.abs(norm - scalar), jnp.abs(norm + scalar))


class RotatedSecondOrderCone(AbstractConvexCone):
    """Rotated cone ``(x, y, z)`` satisfying ``2xy >= ||z||₂²`` and ``x,y >= 0``."""

    _soc: SecondOrderCone

    def __init__(self, dimension: int, /):
        size = int(dimension)
        if size < 3:
            raise ValueError("RotatedSecondOrderCone dimension must be at least three.")
        self.dimension = size
        self._soc = SecondOrderCone(size)
        self.cone_id = canonical_fingerprint(
            {"kind": "rotated-second-order-cone", "dimension": size}
        )

    @staticmethod
    def _to_soc(value: Array, /) -> Array:
        scale = jnp.asarray(1.0 / sqrt(2.0), dtype=value.dtype)
        return jnp.concatenate(
            (
                (value[..., :1] + value[..., 1:2]) * scale,
                (value[..., :1] - value[..., 1:2]) * scale,
                value[..., 2:],
            ),
            axis=-1,
        )

    @staticmethod
    def _from_soc(value: Array, /) -> Array:
        scale = jnp.asarray(1.0 / sqrt(2.0), dtype=value.dtype)
        return jnp.concatenate(
            (
                (value[..., :1] + value[..., 1:2]) * scale,
                (value[..., :1] - value[..., 1:2]) * scale,
                value[..., 2:],
            ),
            axis=-1,
        )

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        return self._from_soc(self._soc.project(self._to_soc(array)))

    def project_dual(self, value: Any, /) -> Array:
        return self.project(value)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        return self._soc.interior_margin(self._to_soc(array))

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        return self._soc.dual_projection_smoothness_margin(self._to_soc(array))


class ProductCone(AbstractConvexCone):
    """Ordered product of cone blocks over one flat trailing axis."""

    cones: tuple[AbstractConvexCone, ...]
    slices: tuple[slice, ...] = eqx.field(static=True)

    def __init__(self, cones: Sequence[AbstractConvexCone] = (), /):
        cones_ = tuple(cones)
        if any(not isinstance(cone, AbstractConvexCone) for cone in cones_):
            raise TypeError("ProductCone blocks must be AbstractConvexCone values.")
        cursor = 0
        slices: list[slice] = []
        for cone in cones_:
            slices.append(slice(cursor, cursor + cone.dimension))
            cursor += cone.dimension
        self.cones = cones_
        self.slices = tuple(slices)
        self.dimension = cursor
        self.cone_id = canonical_fingerprint(
            {"kind": "product-cone", "blocks": [cone.cone_id for cone in cones_]}
        )

    def split(self, value: Any, /) -> tuple[Array, ...]:
        array = self._validate(value)
        return tuple(array[..., block] for block in self.slices)

    def _join(self, blocks: tuple[Array, ...], reference: Array, /) -> Array:
        if not blocks:
            return jnp.empty(reference.shape[:-1] + (0,), dtype=reference.dtype)
        return jnp.concatenate(blocks, axis=-1)

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        blocks = tuple(
            cone.project(block)
            for cone, block in zip(self.cones, self.split(array), strict=True)
        )
        return self._join(blocks, array)

    def project_dual(self, value: Any, /) -> Array:
        array = self._validate(value)
        blocks = tuple(
            cone.project_dual(block)
            for cone, block in zip(self.cones, self.split(array), strict=True)
        )
        return self._join(blocks, array)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        if not self.cones:
            return jnp.full(array.shape[:-1], jnp.inf, dtype=array.dtype)
        margins = tuple(
            cone.interior_margin(block)
            for cone, block in zip(self.cones, self.split(array), strict=True)
        )
        return jnp.min(jnp.stack(margins, axis=-1), axis=-1)

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        if not self.cones:
            return jnp.full(array.shape[:-1], jnp.inf, dtype=array.dtype)
        margins = tuple(
            cone.dual_projection_smoothness_margin(block)
            for cone, block in zip(self.cones, self.split(array), strict=True)
        )
        return jnp.min(jnp.stack(margins, axis=-1), axis=-1)

    def block_complementarity(self, primal: Any, dual: Any, /) -> Array:
        primal_ = self.split(primal)
        dual_ = self.split(dual)
        if not self.cones:
            reference = self._validate(primal)
            return jnp.empty(reference.shape[:-1] + (0,), dtype=reference.dtype)
        values = tuple(
            cone.complementarity(p, d)
            for cone, p, d in zip(self.cones, primal_, dual_, strict=True)
        )
        return jnp.stack(values, axis=-1)


__all__ = [
    "AbstractConvexCone",
    "NonnegativeCone",
    "ProductCone",
    "RotatedSecondOrderCone",
    "SecondOrderCone",
    "ZeroCone",
]
