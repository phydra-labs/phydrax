#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import ChartTransition
from ._metric import AbstractSemiRiemannianMetric


TensorVariance: TypeAlias = Literal["contravariant", "covariant"]


class TensorType(StrictModule):
    """Transformation law for the trailing component axes of a tensor field."""

    variance: tuple[TensorVariance, ...]
    density_weight: float

    def __init__(
        self,
        variance: Sequence[TensorVariance] = (),
        /,
        *,
        density_weight: float = 0.0,
    ):
        variance_ = tuple(variance)
        invalid = tuple(
            value for value in variance_ if value not in ("contravariant", "covariant")
        )
        if invalid:
            raise ValueError(
                f"Tensor variance must be 'contravariant' or 'covariant'; got {invalid}."
            )
        density_weight_ = float(density_weight)
        if not isfinite(density_weight_):
            raise ValueError("Tensor density weight must be finite.")
        self.variance = variance_
        self.density_weight = density_weight_

    @property
    def rank(self) -> int:
        return len(self.variance)


SCALAR_TENSOR = TensorType()
VECTOR_TENSOR = TensorType(("contravariant",))
COVECTOR_TENSOR = TensorType(("covariant",))
DENSITY_TENSOR = TensorType(density_weight=1.0)


_COMPONENT_LETTERS = tuple(
    letter for letter in "abcdefghijklmnopqrstuvwxyz" if letter not in ("i", "j")
)


def _tensor_array(
    tensor: ArrayLike,
    tensor_type: TensorType,
    coordinates: ArrayLike,
    dimension: int,
    /,
) -> tuple[Array, Array, int]:
    array = jnp.asarray(tensor)
    points = jnp.asarray(coordinates)
    if points.ndim < 1 or points.shape[-1] != dimension:
        raise ValueError(
            f"Coordinates must have trailing dimension {dimension}; got {points.shape}."
        )
    leading_shape = points.shape[:-1]
    expected = leading_shape + (dimension,) * tensor_type.rank
    if array.shape != expected:
        raise ValueError(
            f"Tensor type of rank {tensor_type.rank} requires shape {expected}; "
            f"got {array.shape}."
        )
    return array, points, len(leading_shape)


def _normalize_component_axis(axis: int, rank: int, /) -> int:
    axis_ = int(axis)
    if axis_ < 0:
        axis_ += rank
    if axis_ < 0 or axis_ >= rank:
        raise ValueError(f"Tensor component axis {axis} is invalid for rank {rank}.")
    return axis_


def _apply_linear_axis(
    tensor: Array,
    matrix: Array,
    axis: int,
    rank: int,
    /,
) -> Array:
    if rank > len(_COMPONENT_LETTERS):
        raise ValueError(
            f"Tensor rank {rank} exceeds the supported rank {len(_COMPONENT_LETTERS)}."
        )
    axis_ = _normalize_component_axis(axis, rank)
    output_letters = list(_COMPONENT_LETTERS[:rank])
    input_letters = list(output_letters)
    output_letters[axis_] = "i"
    input_letters[axis_] = "j"
    return oe.contract(
        f"...ij,...{''.join(input_letters)}->...{''.join(output_letters)}",
        matrix,
        tensor,
    )


def raise_index(
    tensor: ArrayLike,
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
    *,
    axis: int = -1,
    tensor_type: TensorType,
) -> Array:
    """Raise one covariant component axis using the inverse metric."""

    axis_ = _normalize_component_axis(axis, tensor_type.rank)
    if tensor_type.variance[axis_] != "covariant":
        raise ValueError("raise_index requires a covariant source axis.")
    array, points, _ = _tensor_array(
        tensor, tensor_type, coordinates, metric.chart.dimension
    )
    return _apply_linear_axis(
        array,
        metric.inverse(points),
        axis_,
        tensor_type.rank,
    )


def lower_index(
    tensor: ArrayLike,
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
    *,
    axis: int = -1,
    tensor_type: TensorType,
) -> Array:
    """Lower one contravariant component axis using the metric."""

    axis_ = _normalize_component_axis(axis, tensor_type.rank)
    if tensor_type.variance[axis_] != "contravariant":
        raise ValueError("lower_index requires a contravariant source axis.")
    array, points, _ = _tensor_array(
        tensor, tensor_type, coordinates, metric.chart.dimension
    )
    return _apply_linear_axis(
        array,
        metric(points),
        axis_,
        tensor_type.rank,
    )


def inner_product(
    left: ArrayLike,
    right: ArrayLike,
    metric: AbstractSemiRiemannianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Metric bilinear pairing of two contravariant vectors."""

    return metric.bilinear(left, right, coordinates)


def tensor_norm_squared(
    tensor: ArrayLike,
    metric: AbstractSemiRiemannianMetric,
    tensor_type: TensorType,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Pointwise metric self-contraction with declared tensor variance."""

    array, points, leading_ndim = _tensor_array(
        tensor, tensor_type, coordinates, metric.chart.dimension
    )
    dual = array
    matrix = metric(points)
    inverse = metric.inverse(points)
    for axis, variance in enumerate(tensor_type.variance):
        dual = _apply_linear_axis(
            dual,
            matrix if variance == "contravariant" else inverse,
            axis,
            tensor_type.rank,
        )
    product = jnp.conj(array) * dual
    component_axes = tuple(range(leading_ndim, product.ndim))
    return jnp.sum(product, axis=component_axes)


def contract_indices(
    tensor: ArrayLike,
    tensor_type: TensorType,
    first: int,
    second: int,
    coordinates: ArrayLike,
    dimension: int,
    /,
) -> Array:
    """Contract one contravariant and one covariant component axis."""

    first_ = _normalize_component_axis(first, tensor_type.rank)
    second_ = _normalize_component_axis(second, tensor_type.rank)
    if first_ == second_:
        raise ValueError("Tensor contraction requires two distinct axes.")
    if tensor_type.variance[first_] == tensor_type.variance[second_]:
        raise ValueError(
            "Tensor contraction requires one contravariant and one covariant axis."
        )
    array, _, leading_ndim = _tensor_array(tensor, tensor_type, coordinates, dimension)
    return jnp.trace(
        array,
        axis1=leading_ndim + first_,
        axis2=leading_ndim + second_,
    )


def pushforward_vector(
    transition: ChartTransition,
    vector: ArrayLike,
    source_coordinates: ArrayLike,
    /,
) -> Array:
    """Push a source-chart tangent vector into target-chart components."""

    values = jnp.asarray(vector)
    points = jnp.asarray(source_coordinates)
    expected = points.shape[:-1] + (transition.source.dimension,)
    if values.shape != expected:
        raise ValueError(f"Source vector must have shape {expected}; got {values.shape}.")
    return oe.contract("...ai,...i->...a", transition.jacobian(points), values)


def pullback_covector(
    transition: ChartTransition,
    covector: ArrayLike,
    source_coordinates: ArrayLike,
    /,
) -> Array:
    """Pull target-chart covector components back to the source chart."""

    values = jnp.asarray(covector)
    points = jnp.asarray(source_coordinates)
    expected = points.shape[:-1] + (transition.target.dimension,)
    if values.shape != expected:
        raise ValueError(
            f"Target covector must have shape {expected}; got {values.shape}."
        )
    return oe.contract("...ai,...a->...i", transition.jacobian(points), values)


def reexpress_tensor(
    transition: ChartTransition,
    tensor: ArrayLike,
    tensor_type: TensorType,
    source_coordinates: ArrayLike,
    /,
) -> Array:
    """Re-express source components in the target chart at matching points."""

    if transition.source.dimension != transition.target.dimension:
        raise ValueError("Tensor re-expression requires charts of equal dimension.")
    dimension = transition.source.dimension
    array, points, _ = _tensor_array(
        tensor,
        tensor_type,
        source_coordinates,
        dimension,
    )
    jacobian = transition.jacobian(points)
    inverse_transpose = jnp.swapaxes(jnp.linalg.inv(jacobian), -1, -2)
    result = array
    for axis, variance in enumerate(tensor_type.variance):
        result = _apply_linear_axis(
            result,
            jacobian if variance == "contravariant" else inverse_transpose,
            axis,
            tensor_type.rank,
        )
    if tensor_type.density_weight != 0.0:
        density_factor = jnp.abs(jnp.linalg.det(jacobian)) ** (
            -tensor_type.density_weight
        )
        result = result * density_factor[(...,) + (None,) * tensor_type.rank]
    return result
