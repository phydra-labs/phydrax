#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._axis_factorization import (
    _apply_gathers,
    AxisFactor,
    AxisFactorizedField,
    AxisProductTerm,
)
from .._strict import StrictModule
from ._api import IntegrationRealization
from ._batches import SeparableIntegrationBatch


class FactorizedBilinearTerm(StrictModule):
    """One coefficient-weighted pairing of two sum-of-products fields."""

    left: AxisFactorizedField
    right: AxisFactorizedField
    coefficient: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left: AxisFactorizedField,
        right: AxisFactorizedField,
        /,
        *,
        coefficient: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(left, AxisFactorizedField) or not isinstance(
            right, AxisFactorizedField
        ):
            raise TypeError("Factorized bilinear fields must be AxisFactorizedField.")
        coefficient_ = jnp.asarray(coefficient)
        if coefficient_.shape != ():
            raise ValueError("Factorized bilinear coefficients must be scalar.")
        self.left = left
        self.right = right
        self.coefficient = coefficient_
        self.label = None if label is None else str(label)


class FactorizedBilinearEvaluation(StrictModule):
    """A form matrix plus evidence that no full tensor support was materialized."""

    value: Array
    valid: Array
    axes: tuple[str, ...] = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    full_point_count: int = eqx.field(static=True)
    maximum_local_point_count: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    @property
    def avoided_full_materialization(self) -> bool:
        return self.maximum_local_point_count < self.full_point_count


def _batch(
    realization_or_batch: IntegrationRealization | SeparableIntegrationBatch,
    /,
) -> tuple[SeparableIntegrationBatch, IntegrationRealization | None]:
    if isinstance(realization_or_batch, IntegrationRealization):
        if not isinstance(realization_or_batch.batch, SeparableIntegrationBatch):
            raise TypeError(
                "Factorized assembly requires a SeparableIntegrationBatch realization."
            )
        return realization_or_batch.batch, realization_or_batch
    if isinstance(realization_or_batch, SeparableIntegrationBatch):
        return realization_or_batch, None
    raise TypeError(
        "Expected an IntegrationRealization or SeparableIntegrationBatch."
    )


def _prepared_term_factors(
    field: AxisFactorizedField,
    term: AxisProductTerm,
    /,
) -> tuple[AxisFactor, ...]:
    if jnp.asarray(term.coefficient).shape != ():
        raise ValueError("Axis product coefficients must be scalar for integration.")
    return tuple(_apply_gathers(field.factor(name)) for name in term.factor_names)


def _factor_shape(factors: tuple[AxisFactor, ...], /) -> tuple[int, int]:
    if not factors:
        raise ValueError("A factorized product term requires factors.")
    shape = tuple(int(size) for size in factors[0].tensor.shape[-2:])
    if any(tuple(int(size) for size in factor.tensor.shape[-2:]) != shape for factor in factors):
        raise ValueError(
            "Every factor in one product term must share latent and output sizes."
        )
    return shape


def _axis_components(
    factors: tuple[AxisFactor, ...],
    axes: tuple[str, ...],
    /,
) -> tuple[tuple[str, ...], ...]:
    allowed = set(axes)
    extra = tuple(
        axis for factor in factors for axis in factor.axes if axis not in allowed
    )
    if extra:
        raise ValueError(f"Factorized fields contain non-integration axes {extra!r}.")
    components: list[set[str]] = [{axis} for axis in axes]
    for factor in factors:
        support = set(factor.axes)
        if not support:
            continue
        matching = [index for index, component in enumerate(components) if component & support]
        merged = set(support)
        for index in reversed(matching):
            merged.update(components.pop(index))
        components.append(merged)
    order = {axis: index for index, axis in enumerate(axes)}
    return tuple(
        tuple(sorted(component, key=order.__getitem__))
        for component in sorted(
            components,
            key=lambda component: min(order[axis] for axis in component),
        )
    )


def _aligned_factor(factor: AxisFactor, axes: tuple[str, ...], /) -> Array:
    present = tuple(axis for axis in axes if axis in factor.axes)
    permutation = tuple(factor.axes.index(axis) for axis in present) + (
        len(factor.axes),
        len(factor.axes) + 1,
    )
    value = factor.tensor
    if permutation != tuple(range(value.ndim)):
        value = jnp.transpose(value, permutation)
    sizes = dict(zip(present, value.shape[: len(present)], strict=True))
    shape = tuple(int(sizes[axis]) if axis in sizes else 1 for axis in axes) + tuple(
        int(size) for size in value.shape[-2:]
    )
    return value.reshape(shape)


def _component_pairing(
    left_factors: tuple[AxisFactor, ...],
    right_factors: tuple[AxisFactor, ...],
    component: tuple[str, ...],
    batch: SeparableIntegrationBatch,
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
    dtype: Any,
    /,
) -> Array:
    axis_sizes = tuple(int(batch.weights_by_axis[axis].data.shape[0]) for axis in component)
    left = jnp.ones(axis_sizes + left_shape, dtype=dtype)
    right = jnp.ones(axis_sizes + right_shape, dtype=dtype)
    for factor in left_factors:
        left = left * _aligned_factor(factor, component)
    for factor in right_factors:
        right = right * _aligned_factor(factor, component)
    axis_labels = tuple(range(len(component)))
    left_latent = len(component)
    left_output = left_latent + 1
    right_latent = left_output + 1
    right_output = right_latent + 1
    operands: list[Any] = [
        jnp.conj(left),
        axis_labels + (left_latent, left_output),
        right,
        axis_labels + (right_latent, right_output),
    ]
    for axis_index, axis in enumerate(component):
        operands.extend((batch.weights_by_axis[axis].data, (axis_index,)))
    operands.append((left_latent, right_latent, left_output, right_output))
    return oe.contract(*operands)


def _term_pairing(
    left: AxisFactorizedField,
    left_term: AxisProductTerm,
    right: AxisFactorizedField,
    right_term: AxisProductTerm,
    batch: SeparableIntegrationBatch,
    /,
) -> tuple[Array, int]:
    left_factors = _prepared_term_factors(left, left_term)
    right_factors = _prepared_term_factors(right, right_term)
    left_shape = _factor_shape(left_factors)
    right_shape = _factor_shape(right_factors)
    all_factors = left_factors + right_factors
    dtype = jnp.result_type(*(factor.tensor for factor in all_factors))
    components = _axis_components(all_factors, batch.axes)
    combined = jnp.ones(
        (left_shape[0], right_shape[0], left_shape[1], right_shape[1]),
        dtype=dtype,
    )
    maximum_points = 1
    for component in components:
        local_left = tuple(
            factor for factor in left_factors if set(factor.axes) & set(component)
        )
        local_right = tuple(
            factor for factor in right_factors if set(factor.axes) & set(component)
        )
        combined = combined * _component_pairing(
            local_left,
            local_right,
            component,
            batch,
            left_shape,
            right_shape,
            dtype,
        )
        maximum_points = max(
            maximum_points,
            math.prod(int(batch.weights_by_axis[axis].data.shape[0]) for axis in component),
        )
    constant_left = jnp.ones(left_shape, dtype=combined.dtype)
    constant_right = jnp.ones(right_shape, dtype=combined.dtype)
    for factor in left_factors:
        if not factor.axes:
            constant_left = constant_left * factor.tensor
    for factor in right_factors:
        if not factor.axes:
            constant_right = constant_right * factor.tensor
    combined = (
        combined
        * jnp.conj(constant_left)[:, None, :, None]
        * constant_right[None, :, None, :]
    )
    value = jnp.sum(combined, axis=(0, 1))
    coefficient = jnp.conj(left_term.coefficient) * right_term.coefficient
    return coefficient * value, maximum_points


def factorized_inner_product(
    left: AxisFactorizedField,
    right: AxisFactorizedField,
    realization_or_batch: IntegrationRealization | SeparableIntegrationBatch,
    /,
) -> FactorizedBilinearEvaluation:
    """Integrate all output-channel pairings without a global Cartesian tensor."""
    return factorized_bilinear_form(
        (FactorizedBilinearTerm(left, right),),
        realization_or_batch,
    )


def factorized_bilinear_form(
    terms: Sequence[FactorizedBilinearTerm],
    realization_or_batch: IntegrationRealization | SeparableIntegrationBatch,
    /,
) -> FactorizedBilinearEvaluation:
    """Assemble a sum of factorized sesquilinear form matrices."""
    terms_ = tuple(terms)
    if not terms_:
        raise ValueError("factorized_bilinear_form requires at least one term.")
    if any(not isinstance(term, FactorizedBilinearTerm) for term in terms_):
        raise TypeError("terms must contain FactorizedBilinearTerm values.")
    batch, realization = _batch(realization_or_batch)
    if batch.coupled_weight is not None or batch.mask is not None:
        raise ValueError(
            "Factorized assembly requires separable weights without a coupled mask."
        )
    value: Array | None = None
    maximum_points = 1
    for term in terms_:
        term_value: Array | None = None
        for left_term in term.left.plan.terms:
            for right_term in term.right.plan.terms:
                paired, local_points = _term_pairing(
                    term.left,
                    left_term,
                    term.right,
                    right_term,
                    batch,
                )
                term_value = paired if term_value is None else term_value + paired
                maximum_points = max(maximum_points, local_points)
        if term_value is None:
            raise RuntimeError("Factorized field plans unexpectedly contained no terms.")
        weighted = term.coefficient * term_value
        value = weighted if value is None else value + weighted
    if value is None:
        raise RuntimeError("Factorized form unexpectedly produced no value.")
    if realization is not None:
        value = realization.precision.output(value)
    full_points = math.prod(
        int(batch.weights_by_axis[axis].data.shape[0]) for axis in batch.axes
    )
    return FactorizedBilinearEvaluation(
        value=value,
        valid=jnp.all(jnp.isfinite(value)),
        axes=batch.axes,
        term_count=len(terms_),
        full_point_count=full_points,
        maximum_local_point_count=maximum_points,
        provenance=batch.provenance,
    )


__all__ = [
    "FactorizedBilinearEvaluation",
    "FactorizedBilinearTerm",
    "factorized_bilinear_form",
    "factorized_inner_product",
]
