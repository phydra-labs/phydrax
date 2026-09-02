#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ._cones import (
    AbstractConvexCone,
    NonnegativeCone,
    ProductCone,
    RotatedSecondOrderCone,
    SecondOrderCone,
    ZeroCone,
)
from ._exponential_cone import ExponentialCone
from ._power_cone import PowerCone
from ._psd_cone import PositiveSemidefiniteCone


class ConeBarrierOracle(StrictModule):
    """Primal logarithmically homogeneous barrier and differential actions."""

    cone: AbstractConvexCone
    parameter: float = eqx.field(static=True)

    def value(self, point: Array, /) -> Array:
        return _barrier_value(self.cone, point)

    def gradient(self, point: Array, /) -> Array:
        point_ = self.cone._validate(point)
        gradient = jax.grad(lambda value: _barrier_value(self.cone, value))
        return _map_leading_axes(gradient, point_)

    def hessian_action(self, point: Array, vector: Array, /) -> Array:
        point_ = self.cone._validate(point)
        vector_ = self.cone._validate(vector)
        gradient = jax.grad(lambda value: _barrier_value(self.cone, value))

        def action(value, direction):
            return jax.jvp(gradient, (value,), (direction,))[1]

        return _map_leading_axes(action, point_, vector_)

    def hessian(self, point: Array, /) -> Array:
        point_ = self.cone._validate(point)
        hessian = jax.hessian(lambda value: _barrier_value(self.cone, value))
        return _map_leading_axes(hessian, point_)

    def centrality_residual(self, slack: Array, dual: Array, mu: Array, /) -> Array:
        return dual + mu * self.gradient(slack)

    def interior_reference(self, dtype, /) -> Array:
        return _interior_reference(self.cone, dtype)

    def maximum_interior_step(
        self,
        point: Array,
        direction: Array,
        /,
        *,
        fraction: float = 0.995,
        bisection_steps: int = 64,
    ) -> Array:
        fraction_ = float(fraction)
        if not 0.0 < fraction_ < 1.0:
            raise ValueError("fraction must lie in (0, 1).")
        initial = _barrier_margin(self.cone, point)
        point = eqx.error_if(
            point,
            jnp.any(initial <= 0.0),
            "Barrier step requires a strictly interior point.",
        )
        lower = jnp.zeros(initial.shape, dtype=point.dtype)
        upper = jnp.ones(initial.shape, dtype=point.dtype)

        def expand(_, state):
            lo, hi = state
            accepted = _barrier_margin(self.cone, point + hi[..., None] * direction) > 0.0
            return jnp.where(accepted, hi, lo), jnp.where(accepted, 2.0 * hi, hi)

        lower, upper = jax.lax.fori_loop(0, 32, expand, (lower, upper))

        def bisect(_, state):
            lo, hi = state
            middle = 0.5 * (lo + hi)
            accepted = (
                _barrier_margin(self.cone, point + middle[..., None] * direction) > 0.0
            )
            return jnp.where(accepted, middle, lo), jnp.where(accepted, hi, middle)

        lower, _ = jax.lax.fori_loop(0, int(bisection_steps), bisect, (lower, upper))
        return fraction_ * lower


def _map_leading_axes(function, *values):
    mapped = function
    for _ in range(values[0].ndim - 1):
        mapped = jax.vmap(mapped)
    return mapped(*values)


def _barrier_margin(cone, point):
    if isinstance(cone, ZeroCone):
        return jnp.full(point.shape[:-1], jnp.inf, dtype=point.dtype)
    if isinstance(cone, ProductCone):
        margins = tuple(
            _barrier_margin(block, part)
            for block, part in zip(cone.cones, cone.split(point), strict=True)
        )
        return jnp.min(jnp.stack(margins, axis=-1), axis=-1)
    return cone.interior_margin(point)


def _barrier_value(cone, point):
    value = cone._validate(point)
    if isinstance(cone, ZeroCone):
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    if isinstance(cone, NonnegativeCone):
        return -jnp.sum(jnp.log(value), axis=-1)
    if isinstance(cone, SecondOrderCone):
        determinant = value[..., 0] ** 2 - jnp.sum(value[..., 1:] ** 2, axis=-1)
        return -jnp.log(determinant)
    if isinstance(cone, RotatedSecondOrderCone):
        return _barrier_value(cone._soc, cone._to_soc(value))
    if isinstance(cone, PositiveSemidefiniteCone):
        sign, logdet = jnp.linalg.slogdet(cone.unpack(value))
        return jnp.where(sign > 0.0, -logdet, jnp.inf)
    if isinstance(cone, ExponentialCone):
        x, y, z = value[..., 0], value[..., 1], value[..., 2]
        gap = y * jnp.log(z / y) - x
        return -jnp.log(y) - jnp.log(z) - jnp.log(gap)
    if isinstance(cone, PowerCone):
        x, y, z = value[..., 0], value[..., 1], value[..., 2]
        determinant = (
            x ** (2.0 * cone.exponent) * y ** (2.0 * (1.0 - cone.exponent)) - z**2
        )
        return -jnp.log(x) - jnp.log(y) - jnp.log(determinant)
    if isinstance(cone, ProductCone):
        values = tuple(
            _barrier_value(block, part)
            for block, part in zip(cone.cones, cone.split(value), strict=True)
        )
        return sum(values, jnp.zeros(value.shape[:-1], dtype=value.dtype))
    raise TypeError(f"No native barrier oracle for {type(cone).__name__}.")


def _interior_reference(cone, dtype):
    if isinstance(cone, ZeroCone):
        return jnp.zeros((cone.dimension,), dtype=dtype)
    if isinstance(cone, NonnegativeCone):
        return jnp.ones((cone.dimension,), dtype=dtype)
    if isinstance(cone, SecondOrderCone):
        return jnp.concatenate(
            (
                jnp.asarray([2.0], dtype=dtype),
                jnp.zeros((cone.dimension - 1,), dtype=dtype),
            )
        )
    if isinstance(cone, RotatedSecondOrderCone):
        return jnp.concatenate(
            (
                jnp.asarray([1.0, 1.0], dtype=dtype),
                jnp.zeros((cone.dimension - 2,), dtype=dtype),
            )
        )
    if isinstance(cone, PositiveSemidefiniteCone):
        return cone.pack(jnp.eye(cone.matrix_size, dtype=dtype))
    if isinstance(cone, ExponentialCone):
        return jnp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    if isinstance(cone, PowerCone):
        return jnp.asarray([1.0, 1.0, 0.0], dtype=dtype)
    if isinstance(cone, ProductCone):
        return jnp.concatenate(
            tuple(_interior_reference(block, dtype) for block in cone.cones)
        )
    raise TypeError(f"No interior reference for {type(cone).__name__}.")


def cone_barrier_oracle(cone: AbstractConvexCone, /) -> ConeBarrierOracle:
    """Prepare the exact built-in cone barrier or reject a custom cone."""
    if not isinstance(
        cone,
        (
            ZeroCone,
            NonnegativeCone,
            SecondOrderCone,
            RotatedSecondOrderCone,
            PositiveSemidefiniteCone,
            ExponentialCone,
            PowerCone,
            ProductCone,
        ),
    ):
        raise TypeError(
            f"Native conic execution has no barrier for {type(cone).__name__}."
        )
    blocks = cone.cones if isinstance(cone, ProductCone) else (cone,)
    parameter = sum(
        0.0
        if isinstance(block, ZeroCone)
        else float(block.dimension)
        if isinstance(block, NonnegativeCone)
        else float(block.matrix_size)
        if isinstance(block, PositiveSemidefiniteCone)
        else 2.0
        if isinstance(block, (SecondOrderCone, RotatedSecondOrderCone))
        else 3.0
        if isinstance(block, ExponentialCone)
        else 4.0
        for block in blocks
    )
    return ConeBarrierOracle(cone, parameter)


__all__ = ["ConeBarrierOracle", "cone_barrier_oracle"]
