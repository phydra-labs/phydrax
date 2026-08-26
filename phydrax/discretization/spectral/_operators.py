#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    DiagonalLinearOperator,
    OperatorProperties,
)
from .._spaces import DiscreteFieldSpace
from ._space import TensorSpectralDiscretization


class PreparedSpectralOperator(StrictModule, NonTrainableState):
    """One exact modal operator with scientific source and target identities."""

    operator: AbstractLinearOperator
    source_space: DiscreteFieldSpace
    target_space: DiscreteFieldSpace
    axes: tuple[int, ...] = eqx.field(static=True)
    derivative_orders: tuple[int, ...] = eqx.field(static=True)
    classification: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        source_space: DiscreteFieldSpace,
        target_space: DiscreteFieldSpace,
        /,
        *,
        axes: Sequence[int],
        derivative_orders: Sequence[int],
        classification: str,
        exact: bool = True,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not isinstance(source_space, DiscreteFieldSpace) or not isinstance(
            target_space, DiscreteFieldSpace
        ):
            raise TypeError(
                "source_space and target_space must be DiscreteFieldSpace values."
            )
        axes_ = tuple(int(value) for value in axes)
        orders = tuple(int(value) for value in derivative_orders)
        if len(axes_) != len(orders) or any(value < 0 for value in orders):
            raise ValueError("axes and non-negative derivative_orders must align.")
        kind = str(classification)
        if not kind:
            raise ValueError("classification must be non-empty.")
        self.operator = operator
        self.source_space = source_space
        self.target_space = target_space
        self.axes = axes_
        self.derivative_orders = orders
        self.classification = kind
        self.exact = bool(exact)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-operator",
                "operator": operator.operator_id,
                "source": source_space.field_space_id,
                "target": target_space.field_space_id,
                "axes": list(axes_),
                "orders": list(orders),
                "classification": kind,
                "exact": bool(exact),
            }
        )

    def __call__(self, coefficients: Array) -> Array:
        return self.operator(coefficients)


def _axis_multiplier(
    discretization: TensorSpectralDiscretization,
    axis: int,
    order: int,
    /,
) -> Array:
    prepared = discretization.axes[axis]
    multiplier = prepared.derivative_multiplier(order)
    shape = [1] * len(discretization.modal_shape)
    shape[axis] = multiplier.size
    return jnp.broadcast_to(multiplier.reshape(tuple(shape)), discretization.modal_shape)


def spectral_derivative_operator(
    discretization: TensorSpectralDiscretization,
    axis: int,
    order: int = 1,
    /,
) -> PreparedSpectralOperator:
    """Return an exact modal derivative endomorphism when the basis is closed."""
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    axis_ = int(axis)
    order_ = int(order)
    if axis_ < 0 or axis_ >= len(discretization.axes):
        raise ValueError(f"axis must lie in [0, {len(discretization.axes)}).")
    if order_ < 0:
        raise ValueError("order must be non-negative.")
    diagonal = _axis_multiplier(discretization, axis_, order_).reshape((-1,))
    operator = DiagonalLinearOperator(
        diagonal,
        space=discretization.modal_space.vector_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "spectral-derivative-operator",
                "discretization": discretization.prepared_id,
                "axis": axis_,
                "order": order_,
            }
        ),
    )
    return PreparedSpectralOperator(
        operator,
        discretization.modal_space,
        discretization.modal_space,
        axes=(axis_,),
        derivative_orders=(order_,),
        classification="pseudospectral",
    )


def spectral_laplacian_operator(
    discretization: TensorSpectralDiscretization,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> PreparedSpectralOperator:
    """Return the exact negative-semidefinite modal Laplacian."""
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    selected = (
        tuple(range(len(discretization.axes)))
        if axes is None
        else (int(axes),)
        if isinstance(axes, int)
        else tuple(int(axis) for axis in axes)
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(axis < 0 or axis >= len(discretization.axes) for axis in selected)
    ):
        raise ValueError("Laplacian axes must be unique valid spectral axes.")
    values = jnp.zeros(
        discretization.modal_shape,
        dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
    )
    for axis in selected:
        values = values + _axis_multiplier(discretization, axis, 2)
    diagonal = values.reshape((-1,))
    properties = OperatorProperties(
        diagonal=True,
        self_adjoint=True,
        evidence={
            "diagonal": "construction",
            "self_adjoint": "construction",
        },
    )
    operator = DiagonalLinearOperator(
        diagonal,
        space=discretization.modal_space.vector_space,
        properties=properties,
        operator_id=canonical_fingerprint(
            {
                "kind": "spectral-laplacian-operator",
                "discretization": discretization.prepared_id,
                "axes": list(selected),
            }
        ),
    )
    return PreparedSpectralOperator(
        operator,
        discretization.modal_space,
        discretization.modal_space,
        axes=selected,
        derivative_orders=(2,) * len(selected),
        classification="pseudospectral",
    )


__all__ = [
    "PreparedSpectralOperator",
    "spectral_derivative_operator",
    "spectral_laplacian_operator",
]
