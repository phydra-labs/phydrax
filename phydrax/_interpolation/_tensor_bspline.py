#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from numbers import Integral
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._bspline import BSplineJetStencil


MultiIndex = tuple[int, ...]


def _complete_multi_indices(dimension: int, maximum_order: int) -> tuple[MultiIndex, ...]:
    def fixed_total(total: int, remaining: int) -> tuple[MultiIndex, ...]:
        if remaining == 1:
            return ((total,),)
        return tuple(
            (first, *rest)
            for first in range(total, -1, -1)
            for rest in fixed_total(total - first, remaining - 1)
        )

    return tuple(
        multi_index
        for total in range(maximum_order + 1)
        for multi_index in fixed_total(total, dimension)
    )


def _masked_axis_jet(stencil: BSplineJetStencil, order: int) -> Array:
    weights = stencil.jets[..., order, :]
    return jnp.where(stencil.support[..., None], weights, jnp.zeros_like(weights))


class TensorBSplineJetPlan(StrictModule):
    """Factorized tensor product of span-local univariate B-spline jets."""

    axis_stencils: tuple[BSplineJetStencil, ...]
    multi_indices: tuple[MultiIndex, ...] = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    query_shape: tuple[int, ...] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        axis_stencils: Sequence[BSplineJetStencil],
        /,
        *,
        maximum_order: int = 2,
        multi_indices: Sequence[Sequence[int]] | None = None,
    ):
        stencils = tuple(axis_stencils)
        if not stencils:
            raise ValueError("A tensor B-spline plan requires at least one axis stencil.")
        if not all(isinstance(stencil, BSplineJetStencil) for stencil in stencils):
            raise TypeError("Tensor B-spline axes must be BSplineJetStencil instances.")
        dimension = len(stencils)
        if multi_indices is None:
            if (
                isinstance(maximum_order, bool)
                or not isinstance(maximum_order, Integral)
                or maximum_order < 0
            ):
                raise ValueError("maximum_order must be a nonnegative integer.")
            indices = _complete_multi_indices(dimension, int(maximum_order))
        else:
            indices = tuple(
                tuple(int(order) for order in value) for value in multi_indices
            )
            if not indices:
                raise ValueError("Tensor B-spline multi_indices must be non-empty.")
            if any(
                len(value) != dimension or any(order < 0 for order in value)
                for value in indices
            ):
                raise ValueError(
                    "Each tensor B-spline multi-index must be nonnegative and match "
                    "the parameter dimension."
                )
            if len(set(indices)) != len(indices):
                raise ValueError("Tensor B-spline multi_indices must be unique.")
            order_lookup = set(indices)
            for value in indices:
                for axis, order in enumerate(value):
                    if order > 0:
                        predecessor = value[:axis] + (order - 1,) + value[axis + 1 :]
                        if predecessor not in order_lookup:
                            raise ValueError(
                                "Tensor B-spline multi_indices must be downward closed."
                            )
        if any(
            multi_index[axis] > stencils[axis].maximum_order
            for multi_index in indices
            for axis in range(dimension)
        ):
            raise ValueError(
                "Axis B-spline stencils do not contain every requested derivative."
            )

        self.axis_stencils = stencils
        self.multi_indices = indices
        self.source_shape = tuple(stencil.source_size for stencil in stencils)
        self.query_shape = tuple(
            size for stencil in stencils for size in stencil.query_shape
        )
        self.local_shape = tuple(stencil.local_support for stencil in stencils)

    @property
    def dimension(self) -> int:
        return len(self.axis_stencils)

    @property
    def local_size(self) -> int:
        return prod(self.local_shape)

    @property
    def jet_axis(self) -> int:
        return len(self.query_shape)

    @property
    def tensor_indices(self) -> Array:
        """Return row-major flattened control indices for every local route."""
        shape = self.query_shape + self.local_shape
        result = jnp.zeros(shape, dtype=jnp.int32)
        query_offset = 0
        for axis, stencil in enumerate(self.axis_stencils):
            query_rank = len(stencil.query_shape)
            route_shape = (
                (1,) * query_offset
                + stencil.query_shape
                + (1,) * (len(self.query_shape) - query_offset - query_rank)
                + (1,) * axis
                + (stencil.local_support,)
                + (1,) * (self.dimension - axis - 1)
            )
            stride = prod(self.source_shape[axis + 1 :])
            result = result + stride * stencil.indices.reshape(route_shape)
            query_offset += query_rank
        return result.reshape(self.query_shape + (self.local_size,))

    def basis(self, multi_index: Sequence[int], /) -> Array:
        """Materialize one local tensor basis derivative in row-major local order."""
        derivative = tuple(int(order) for order in multi_index)
        if len(derivative) != self.dimension or any(order < 0 for order in derivative):
            raise ValueError("Tensor B-spline derivative multi-index is invalid.")
        if any(
            derivative[axis] > stencil.maximum_order
            for axis, stencil in enumerate(self.axis_stencils)
        ):
            raise ValueError("Requested tensor derivative is absent from an axis jet.")

        first = self.axis_stencils[0]
        result = _masked_axis_jet(first, derivative[0])
        query_shape = first.query_shape
        local_shape = (first.local_support,)
        for axis in range(1, self.dimension):
            stencil = self.axis_stencils[axis]
            axis_weights = _masked_axis_jet(stencil, derivative[axis])
            left_shape = (
                query_shape + (1,) * len(stencil.query_shape) + local_shape + (1,)
            )
            right_shape = (
                (1,) * len(query_shape)
                + stencil.query_shape
                + (1,) * len(local_shape)
                + (stencil.local_support,)
            )
            result = result.reshape(left_shape) * axis_weights.reshape(right_shape)
            query_shape = query_shape + stencil.query_shape
            local_shape = local_shape + (stencil.local_support,)
        return result.reshape(self.query_shape + (self.local_size,))

    def gather(self, coefficients: ArrayLike, /) -> Array:
        """Gather tensor-control payloads onto every local tensor route."""
        coefficients_ = jnp.asarray(coefficients)
        if (
            coefficients_.ndim < self.dimension
            or tuple(int(size) for size in coefficients_.shape[: self.dimension])
            != self.source_shape
        ):
            raise ValueError(
                f"Tensor B-spline coefficients must begin with {self.source_shape}."
            )
        payload_shape = coefficients_.shape[self.dimension :]
        flattened = coefficients_.reshape((prod(self.source_shape), *payload_shape))
        return flattened[self.tensor_indices]

    def scatter(self, local_values: ArrayLike, /) -> Array:
        """Apply the exact transpose of :meth:`gather`."""
        values = jnp.asarray(local_values)
        route_shape = self.query_shape + (self.local_size,)
        if (
            values.ndim < len(route_shape)
            or tuple(int(size) for size in values.shape[: len(route_shape)])
            != route_shape
        ):
            raise ValueError(
                "Local tensor values must begin with query_shape + (local_size,)."
            )
        payload_shape = values.shape[len(route_shape) :]
        flattened = values.reshape((-1, *payload_shape))
        routes = self.tensor_indices.reshape((-1,))
        output = jnp.zeros(
            (prod(self.source_shape), *payload_shape),
            dtype=values.dtype,
        )
        output = output.at[routes].add(flattened)
        return output.reshape(self.source_shape + payload_shape)

    def apply(self, coefficients: ArrayLike, multi_index: Sequence[int], /) -> Array:
        """Apply one tensor derivative without materializing a global basis table."""
        coefficients_ = jnp.asarray(coefficients)
        if (
            coefficients_.ndim < self.dimension
            or tuple(int(size) for size in coefficients_.shape[: self.dimension])
            != self.source_shape
        ):
            raise ValueError(
                f"Tensor B-spline coefficients must begin with {self.source_shape}."
            )
        derivative = tuple(int(order) for order in multi_index)
        if len(derivative) != self.dimension:
            raise ValueError("Tensor B-spline derivative multi-index is invalid.")

        result = coefficients_
        for axis in range(self.dimension - 1, -1, -1):
            stencil = self.axis_stencils[axis]
            order = derivative[axis]
            if not 0 <= order <= stencil.maximum_order:
                raise ValueError(
                    "Requested tensor derivative is absent from an axis jet."
                )
            weights = _masked_axis_jet(stencil, order)
            taken = jnp.take(result, stencil.indices, axis=axis)
            weight_shape = (1,) * axis + weights.shape + (1,) * (result.ndim - axis - 1)
            support_axis = axis + len(stencil.query_shape)
            result = jnp.sum(taken * weights.reshape(weight_shape), axis=support_axis)
        return result

    def transpose(
        self,
        messages: ArrayLike,
        multi_index: Sequence[int],
        /,
    ) -> Array:
        """Apply the exact coefficient transpose of one tensor derivative."""
        derivative = tuple(int(order) for order in multi_index)
        if len(derivative) != self.dimension:
            raise ValueError("Tensor B-spline derivative multi-index is invalid.")
        result = jnp.asarray(messages)
        if (
            result.ndim < len(self.query_shape)
            or tuple(int(size) for size in result.shape[: len(self.query_shape)])
            != self.query_shape
        ):
            raise ValueError(
                f"Tensor B-spline messages must begin with {self.query_shape}."
            )

        for axis, stencil in enumerate(self.axis_stencils):
            order = derivative[axis]
            if not 0 <= order <= stencil.maximum_order:
                raise ValueError(
                    "Requested tensor derivative is absent from an axis jet."
                )
            query_rank = len(stencil.query_shape)
            query_axes = tuple(range(axis, axis + query_rank))
            remaining_axes = tuple(
                index for index in range(result.ndim) if index not in query_axes
            )
            permutation = query_axes + remaining_axes
            moved = jnp.transpose(result, permutation) if permutation else result
            query_size = prod(stencil.query_shape)
            remaining_shape = tuple(result.shape[index] for index in remaining_axes)
            flattened = moved.reshape((query_size, -1))
            weights = _masked_axis_jet(stencil, order).reshape(
                (query_size, stencil.local_support)
            )
            contributions = (
                weights[..., None].astype(result.dtype) * flattened[:, None, :]
            )
            scattered = jnp.zeros(
                (stencil.source_size, flattened.shape[1]),
                dtype=result.dtype,
            )
            scattered = scattered.at[stencil.indices.reshape((-1,))].add(
                contributions.reshape((-1, flattened.shape[1]))
            )
            expanded = scattered.reshape((stencil.source_size, *remaining_shape))
            result = jnp.moveaxis(expanded, 0, axis)
        return result

    def jet_apply(self, coefficients: ArrayLike, /) -> Array:
        """Apply every configured multi-index, with the jet axis after query axes."""
        values = tuple(self.apply(coefficients, value) for value in self.multi_indices)
        return jnp.stack(values, axis=self.jet_axis)

    def jet_transpose(self, messages: ArrayLike, /) -> Array:
        """Apply the summed exact transpose of every configured jet component."""
        messages_ = jnp.asarray(messages)
        expected = self.query_shape + (len(self.multi_indices),)
        if (
            messages_.ndim < len(expected)
            or tuple(int(size) for size in messages_.shape[: len(expected)]) != expected
        ):
            raise ValueError("Tensor jet messages have incompatible query and jet axes.")
        result = None
        for index, multi_index in enumerate(self.multi_indices):
            component = self.transpose(
                jnp.take(messages_, index, axis=self.jet_axis),
                multi_index,
            )
            result = component if result is None else result + component
        if result is None:
            raise RuntimeError("Tensor B-spline plan contains no jet components.")
        return result

    def value(self, coefficients: ArrayLike, /) -> Array:
        return self.apply(coefficients, (0,) * self.dimension)

    def gradient(self, coefficients: ArrayLike, /) -> Array:
        components = tuple(
            self.apply(
                coefficients,
                tuple(1 if axis == component else 0 for axis in range(self.dimension)),
            )
            for component in range(self.dimension)
        )
        return jnp.stack(components, axis=-1)

    def hessian(self, coefficients: ArrayLike, /) -> Array:
        rows = tuple(
            jnp.stack(
                tuple(
                    self.apply(
                        coefficients,
                        tuple(
                            int(axis == first) + int(axis == second)
                            for axis in range(self.dimension)
                        ),
                    )
                    for second in range(self.dimension)
                ),
                axis=-1,
            )
            for first in range(self.dimension)
        )
        return jnp.stack(rows, axis=-2)

    def value_transpose(self, messages: ArrayLike, /) -> Array:
        return self.transpose(messages, (0,) * self.dimension)

    def gradient_transpose(self, messages: ArrayLike, /) -> Array:
        messages_ = jnp.asarray(messages)
        if messages_.shape[-1] != self.dimension:
            raise ValueError("Gradient messages must end with the parameter dimension.")
        result = None
        for component in range(self.dimension):
            multi_index = tuple(
                1 if axis == component else 0 for axis in range(self.dimension)
            )
            value = self.transpose(messages_[..., component], multi_index)
            result = value if result is None else result + value
        if result is None:
            raise RuntimeError("Tensor B-spline plan has no parameter axes.")
        return result

    def hessian_transpose(self, messages: ArrayLike, /) -> Array:
        messages_ = jnp.asarray(messages)
        if messages_.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("Hessian messages must end with two parameter axes.")
        result = None
        for first in range(self.dimension):
            for second in range(self.dimension):
                multi_index = tuple(
                    int(axis == first) + int(axis == second)
                    for axis in range(self.dimension)
                )
                value = self.transpose(messages_[..., first, second], multi_index)
                result = value if result is None else result + value
        if result is None:
            raise RuntimeError("Tensor B-spline plan has no parameter axes.")
        return result


__all__ = ["TensorBSplineJetPlan"]
