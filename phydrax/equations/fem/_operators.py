#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class FieldJet(StrictModule):
    value: Array
    gradient: Array | None
    divergence: Array | None
    curl: Array | None

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        gradient: ArrayLike | None = None,
        divergence: ArrayLike | None = None,
        curl: ArrayLike | None = None,
    ):
        self.value = jnp.asarray(value)
        self.gradient = None if gradient is None else jnp.asarray(gradient)
        self.divergence = None if divergence is None else jnp.asarray(divergence)
        self.curl = None if curl is None else jnp.asarray(curl)


class FacetJet(StrictModule):
    """Two scalar traces expressed with one plus-oriented facet normal."""

    plus_value: Array
    minus_value: Array
    plus_gradient: Array
    minus_gradient: Array
    plus_normal_derivative: Array
    minus_normal_derivative: Array
    jump: Array
    average: Array
    normal: Array
    measure: Array

    def __init__(
        self,
        plus_value: ArrayLike,
        minus_value: ArrayLike,
        plus_gradient: ArrayLike,
        minus_gradient: ArrayLike,
        normal: ArrayLike,
        measure: ArrayLike,
        /,
    ):
        plus = jnp.asarray(plus_value)
        minus = jnp.asarray(minus_value)
        plus_gradient_ = jnp.asarray(plus_gradient)
        minus_gradient_ = jnp.asarray(minus_gradient)
        normal_ = jnp.asarray(normal)
        measure_ = jnp.asarray(measure)
        if plus.shape != minus.shape:
            raise ValueError("Facet plus/minus values must have identical shapes.")
        if plus_gradient_.shape != minus_gradient_.shape:
            raise ValueError("Facet plus/minus gradients must have identical shapes.")
        if plus_gradient_.shape[:-1] != plus.shape:
            raise ValueError("Facet scalar gradients must extend the value shape.")
        if normal_.shape != plus_gradient_.shape:
            raise ValueError("Facet normals must match scalar-gradient shapes.")
        if measure_.shape != plus.shape:
            raise ValueError("Facet measure must match scalar trace values.")
        self.plus_value = plus
        self.minus_value = minus
        self.plus_gradient = plus_gradient_
        self.minus_gradient = minus_gradient_
        self.plus_normal_derivative = jnp.sum(plus_gradient_ * normal_, axis=-1)
        self.minus_normal_derivative = jnp.sum(minus_gradient_ * normal_, axis=-1)
        self.jump = plus - minus
        self.average = 0.5 * (plus + minus)
        self.normal = normal_
        self.measure = measure_


class CellDerivativeBatch(StrictModule):
    """Cell-local coefficients and evaluated values/gradients for one action."""

    local_coefficients: Array
    value: Array
    gradient: Array

    def __init__(
        self,
        local_coefficients: ArrayLike,
        basis_values: ArrayLike,
        physical_gradients: ArrayLike,
        /,
    ):
        coefficients = jnp.asarray(local_coefficients)
        basis = jnp.asarray(basis_values)
        gradients = jnp.asarray(physical_gradients)
        if coefficients.ndim != 2 or gradients.ndim != 4:
            raise ValueError("Scalar cell derivative staging requires rank-2/4 data.")
        if (
            gradients.shape[0] != coefficients.shape[0]
            or gradients.shape[2] != coefficients.shape[1]
        ):
            raise ValueError("Cell derivative staging local layouts are incompatible.")
        if basis.ndim == 2:
            value = oe.contract("qi,ei->eq", basis, coefficients)
        elif basis.ndim == 3 and basis.shape[:1] == coefficients.shape[:1]:
            value = oe.contract("eqi,ei->eq", basis, coefficients)
        else:
            raise ValueError("Cell derivative basis values have an invalid layout.")
        self.local_coefficients = coefficients
        self.value = value
        self.gradient = oe.contract("eqid,ei->eqd", gradients, coefficients)


class DGTraceBatch(StrictModule):
    """Packed plus/minus DG traces computed once for a facet action group."""

    jet: FacetJet
    plus_local_coefficients: Array
    minus_local_coefficients: Array

    def __init__(
        self,
        plus_local_coefficients: ArrayLike,
        minus_local_coefficients: ArrayLike,
        plus_basis_values: ArrayLike,
        minus_basis_values: ArrayLike,
        plus_physical_gradients: ArrayLike,
        minus_physical_gradients: ArrayLike,
        normal: ArrayLike,
        measure: ArrayLike,
        /,
    ):
        plus = CellDerivativeBatch(
            plus_local_coefficients,
            plus_basis_values,
            plus_physical_gradients,
        )
        minus = CellDerivativeBatch(
            minus_local_coefficients,
            minus_basis_values,
            minus_physical_gradients,
        )
        self.jet = FacetJet(
            plus.value,
            minus.value,
            plus.gradient,
            minus.gradient,
            normal,
            measure,
        )
        self.plus_local_coefficients = plus.local_coefficients
        self.minus_local_coefficients = minus.local_coefficients


def symmetric_gradient(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-1] != gradient_.shape[-2]:
        raise ValueError("Symmetric gradient requires square value/coordinate axes.")
    return 0.5 * (gradient_ + jnp.swapaxes(gradient_, -1, -2))


def divergence(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-1] != gradient_.shape[-2]:
        raise ValueError("Divergence requires matching value/coordinate dimensions.")
    return jnp.trace(gradient_, axis1=-2, axis2=-1)


def curl(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-2:] == (2, 2):
        return gradient_[..., 1, 0] - gradient_[..., 0, 1]
    if gradient_.shape[-2:] == (3, 3):
        return jnp.stack(
            (
                gradient_[..., 2, 1] - gradient_[..., 1, 2],
                gradient_[..., 0, 2] - gradient_[..., 2, 0],
                gradient_[..., 1, 0] - gradient_[..., 0, 1],
            ),
            axis=-1,
        )
    raise ValueError("Curl requires a two- or three-dimensional vector gradient.")


def normal_trace(value: ArrayLike, normal: ArrayLike, /) -> Array:
    value_ = jnp.asarray(value)
    normal_ = jnp.asarray(normal)
    if value_.shape[-1] != normal_.shape[-1]:
        raise ValueError("Normal trace value and normal dimensions must match.")
    return jnp.sum(value_ * normal_, axis=-1)


def tangential_trace(value: ArrayLike, normal: ArrayLike, /) -> Array:
    value_ = jnp.asarray(value)
    normal_ = jnp.asarray(normal)
    if value_.shape[-1] == 2:
        tangent = jnp.stack((-normal_[..., 1], normal_[..., 0]), axis=-1)
        return jnp.sum(value_ * tangent, axis=-1)
    if value_.shape[-1] == 3:
        return value_ - jnp.sum(value_ * normal_, axis=-1, keepdims=True) * normal_
    raise ValueError("Tangential trace requires a two- or three-dimensional value.")


def jump(plus: ArrayLike, minus: ArrayLike, /) -> Array:
    plus_ = jnp.asarray(plus)
    minus_ = jnp.asarray(minus)
    if plus_.shape != minus_.shape:
        raise ValueError("Jump operands must have identical shape.")
    return plus_ - minus_


def average(plus: ArrayLike, minus: ArrayLike, /) -> Array:
    plus_ = jnp.asarray(plus)
    minus_ = jnp.asarray(minus)
    if plus_.shape != minus_.shape:
        raise ValueError("Average operands must have identical shape.")
    return 0.5 * (plus_ + minus_)


__all__ = [
    "CellDerivativeBatch",
    "DGTraceBatch",
    "FacetJet",
    "FieldJet",
    "average",
    "curl",
    "divergence",
    "jump",
    "normal_trace",
    "symmetric_gradient",
    "tangential_trace",
]
