#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import comb
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._strict import StrictModule
from ._tensor_bspline import MultiIndex, TensorBSplineJetPlan


def _multi_binomial(alpha: MultiIndex, beta: MultiIndex) -> int:
    result = 1
    for alpha_axis, beta_axis in zip(alpha, beta, strict=True):
        result *= comb(alpha_axis, beta_axis)
    return result


def _rational_quotient_jets(
    numerator_jets: ArrayLike,
    denominator_jets: ArrayLike,
    multi_indices: Sequence[Sequence[int]],
    /,
    *,
    jet_axis: int,
) -> Array:
    """Apply the multivariate quotient recurrence on a downward-closed jet."""
    numerator = jnp.asarray(numerator_jets)
    denominator = jnp.asarray(denominator_jets)
    indices = tuple(tuple(int(order) for order in value) for value in multi_indices)
    if not indices:
        raise ValueError("Rational spline jets require at least the value multi-index.")
    axis = int(jet_axis)
    if axis < 0:
        axis += denominator.ndim
    if not 0 <= axis < denominator.ndim:
        raise ValueError("Rational spline jet_axis is out of range.")
    if denominator.shape[axis] != len(indices) or numerator.shape[axis] != len(indices):
        raise ValueError("Rational numerator and denominator jet axes are inconsistent.")

    numerator_front = jnp.moveaxis(numerator, axis, 0)
    denominator_front = jnp.moveaxis(denominator, axis, 0)
    denominator_shape = denominator_front.shape[1:]
    payload_ndim = numerator.ndim - denominator.ndim
    if (
        payload_ndim < 0
        or numerator_front.shape[1 : 1 + len(denominator_shape)] != denominator_shape
    ):
        raise ValueError(
            "Rational numerator jets must append payload axes to denominator jets."
        )
    lookup = {multi_index: index for index, multi_index in enumerate(indices)}
    zero = (0,) * len(indices[0])
    if zero not in lookup or any(len(value) != len(zero) for value in indices):
        raise ValueError("Rational jet multi-indices have incompatible dimensions.")
    for alpha in indices:
        for parameter_axis, order in enumerate(alpha):
            if order > 0:
                predecessor = (
                    alpha[:parameter_axis] + (order - 1,) + alpha[parameter_axis + 1 :]
                )
                if predecessor not in lookup:
                    raise ValueError(
                        "Rational jet multi-indices must be downward closed."
                    )

    denominator_value = denominator_front[lookup[zero]]
    real_dtype = denominator_value.real.dtype
    tolerance = jnp.finfo(real_dtype).eps * jnp.maximum(
        jnp.ones((), dtype=real_dtype),
        jnp.abs(denominator_value),
    )
    denominator_value = eqx.error_if(
        denominator_value,
        jnp.any(~jnp.isfinite(denominator_value))
        | jnp.any(jnp.abs(denominator_value) <= tolerance),
        "Rational spline weights produced a zero or non-finite denominator.",
    )
    payload_expansion = (1,) * payload_ndim
    denominator_value_expanded = denominator_value.reshape(
        denominator_shape + payload_expansion
    )

    results: dict[MultiIndex, Array] = {}
    for alpha in indices:
        value = numerator_front[lookup[alpha]]
        for beta in indices:
            if beta == zero or any(
                beta_axis > alpha_axis
                for alpha_axis, beta_axis in zip(alpha, beta, strict=True)
            ):
                continue
            gamma = tuple(
                alpha_axis - beta_axis
                for alpha_axis, beta_axis in zip(alpha, beta, strict=True)
            )
            denominator_derivative = denominator_front[lookup[beta]].reshape(
                denominator_shape + payload_expansion
            )
            value = value - (
                _multi_binomial(alpha, beta) * denominator_derivative * results[gamma]
            )
        results[alpha] = value / denominator_value_expanded

    stacked = jnp.stack(tuple(results[value] for value in indices), axis=0)
    return jnp.moveaxis(stacked, 0, axis)


class RationalSplineJet(StrictModule):
    """Local rational tensor basis jet with exact payload actions and transposes."""

    plan: TensorBSplineJetPlan
    normalized_weights: Array
    jets: Array
    denominator_jets: Array

    def __init__(self, plan: TensorBSplineJetPlan, weights: ArrayLike, /):
        if not isinstance(plan, TensorBSplineJetPlan):
            raise TypeError("RationalSplineJet requires a TensorBSplineJetPlan.")
        weights_ = jnp.asarray(weights)
        if weights_.shape != plan.source_shape:
            raise ValueError(
                f"Rational spline weights must have shape {plan.source_shape}."
            )
        if jnp.issubdtype(weights_.dtype, jnp.complexfloating):
            raise TypeError("Rational spline weights must be real-valued.")
        if not jnp.issubdtype(weights_.dtype, jnp.inexact):
            weights_ = weights_.astype(float)
        scale = jnp.max(jnp.abs(weights_))
        scale = eqx.error_if(
            scale,
            ~jnp.isfinite(scale) | (scale == 0.0),
            "Rational spline weights must contain a finite nonzero value.",
        )
        normalized_weights = weights_ / scale
        local_weights = plan.gather(normalized_weights)
        polynomial_jets = jnp.stack(
            tuple(plan.basis(value) for value in plan.multi_indices),
            axis=plan.jet_axis,
        )
        weighted_jets = polynomial_jets * jnp.expand_dims(
            local_weights,
            axis=plan.jet_axis,
        )
        denominator_jets = jnp.sum(weighted_jets, axis=-1)
        jets = _rational_quotient_jets(
            weighted_jets,
            denominator_jets,
            plan.multi_indices,
            jet_axis=plan.jet_axis,
        )
        self.plan = plan
        self.normalized_weights = normalized_weights
        self.jets = jets
        self.denominator_jets = denominator_jets

    @property
    def indices(self) -> Array:
        return self.plan.tensor_indices

    def derivative(self, multi_index: Sequence[int], /) -> Array:
        derivative = tuple(int(order) for order in multi_index)
        if derivative not in self.plan.multi_indices:
            raise ValueError("Requested rational derivative is absent from this jet.")
        index = self.plan.multi_indices.index(derivative)
        return jnp.take(self.jets, index, axis=self.plan.jet_axis)

    @property
    def values(self) -> Array:
        return self.derivative((0,) * self.plan.dimension)

    @property
    def gradients(self) -> Array:
        components = tuple(
            self.derivative(
                tuple(
                    1 if axis == component else 0 for axis in range(self.plan.dimension)
                )
            )
            for component in range(self.plan.dimension)
        )
        return jnp.stack(components, axis=-1)

    @property
    def hessians(self) -> Array:
        rows = tuple(
            jnp.stack(
                tuple(
                    self.derivative(
                        tuple(
                            int(axis == first) + int(axis == second)
                            for axis in range(self.plan.dimension)
                        )
                    )
                    for second in range(self.plan.dimension)
                ),
                axis=-1,
            )
            for first in range(self.plan.dimension)
        )
        return jnp.stack(rows, axis=-2)

    def apply(
        self,
        coefficients: ArrayLike,
        multi_index: Sequence[int] | None = None,
        /,
    ) -> Array:
        """Apply one rational basis derivative to tensor-control payloads."""
        derivative = (
            (0,) * self.plan.dimension
            if multi_index is None
            else tuple(int(order) for order in multi_index)
        )
        basis = self.derivative(derivative)
        local = self.plan.gather(coefficients)
        query_rank = len(self.plan.query_shape)
        payload_rank = local.ndim - query_rank - 1
        query_labels = list(range(query_rank))
        local_label = query_rank
        payload_labels = list(range(query_rank + 1, query_rank + 1 + payload_rank))
        return contract(
            basis,
            query_labels + [local_label],
            local,
            query_labels + [local_label] + payload_labels,
            query_labels + payload_labels,
        )

    def transpose(
        self,
        messages: ArrayLike,
        multi_index: Sequence[int] | None = None,
        /,
    ) -> Array:
        """Apply the exact coefficient transpose of one rational derivative."""
        derivative = (
            (0,) * self.plan.dimension
            if multi_index is None
            else tuple(int(order) for order in multi_index)
        )
        basis = self.derivative(derivative)
        messages_ = jnp.asarray(messages)
        query_rank = len(self.plan.query_shape)
        if (
            messages_.ndim < query_rank
            or tuple(int(size) for size in messages_.shape[:query_rank])
            != self.plan.query_shape
        ):
            raise ValueError(
                f"Rational spline messages must begin with {self.plan.query_shape}."
            )
        payload_rank = messages_.ndim - query_rank
        query_labels = list(range(query_rank))
        local_label = query_rank
        payload_labels = list(range(query_rank + 1, query_rank + 1 + payload_rank))
        local_messages = contract(
            basis,
            query_labels + [local_label],
            messages_,
            query_labels + payload_labels,
            query_labels + [local_label] + payload_labels,
        )
        return self.plan.scatter(local_messages)

    def jet_apply(self, coefficients: ArrayLike, /) -> Array:
        """Apply all configured rational jet components to payload coefficients."""
        local = self.plan.gather(coefficients)
        query_rank = len(self.plan.query_shape)
        payload_rank = local.ndim - query_rank - 1
        query_labels = list(range(query_rank))
        jet_label = query_rank
        local_label = query_rank + 1
        payload_labels = list(range(query_rank + 2, query_rank + 2 + payload_rank))
        return contract(
            self.jets,
            query_labels + [jet_label, local_label],
            local,
            query_labels + [local_label] + payload_labels,
            query_labels + [jet_label] + payload_labels,
        )

    def jet_transpose(self, messages: ArrayLike, /) -> Array:
        """Apply the summed exact coefficient transpose of the rational jet."""
        messages_ = jnp.asarray(messages)
        expected = self.plan.query_shape + (len(self.plan.multi_indices),)
        if (
            messages_.ndim < len(expected)
            or tuple(int(size) for size in messages_.shape[: len(expected)]) != expected
        ):
            raise ValueError(
                "Rational jet messages have incompatible query and jet axes."
            )
        query_rank = len(self.plan.query_shape)
        payload_rank = messages_.ndim - query_rank - 1
        query_labels = list(range(query_rank))
        jet_label = query_rank
        local_label = query_rank + 1
        payload_labels = list(range(query_rank + 2, query_rank + 2 + payload_rank))
        local_messages = contract(
            self.jets,
            query_labels + [jet_label, local_label],
            messages_,
            query_labels + [jet_label] + payload_labels,
            query_labels + [local_label] + payload_labels,
        )
        return self.plan.scatter(local_messages)

    def gradient_apply(self, coefficients: ArrayLike, /) -> Array:
        local = self.plan.gather(coefficients)
        query_rank = len(self.plan.query_shape)
        payload_rank = local.ndim - query_rank - 1
        query_labels = list(range(query_rank))
        local_label = query_rank
        parameter_label = query_rank + 1
        payload_labels = list(range(query_rank + 2, query_rank + 2 + payload_rank))
        return contract(
            self.gradients,
            query_labels + [local_label, parameter_label],
            local,
            query_labels + [local_label] + payload_labels,
            query_labels + payload_labels + [parameter_label],
        )

    def hessian_apply(self, coefficients: ArrayLike, /) -> Array:
        local = self.plan.gather(coefficients)
        query_rank = len(self.plan.query_shape)
        payload_rank = local.ndim - query_rank - 1
        query_labels = list(range(query_rank))
        local_label = query_rank
        first_parameter = query_rank + 1
        second_parameter = query_rank + 2
        payload_labels = list(range(query_rank + 3, query_rank + 3 + payload_rank))
        return contract(
            self.hessians,
            query_labels + [local_label, first_parameter, second_parameter],
            local,
            query_labels + [local_label] + payload_labels,
            query_labels + payload_labels + [first_parameter, second_parameter],
        )


__all__ = ["RationalSplineJet"]
