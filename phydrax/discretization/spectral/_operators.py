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
    FunctionLinearOperator,
    OperatorProperties,
)
from .._spaces import DiscreteFieldSpace
from ._space import TensorSpectralDiscretization


class PreparedSpectralOperator(StrictModule, NonTrainableState):
    """One modal operator with scientific source, target, and exactness evidence."""

    operator: AbstractLinearOperator
    source_space: DiscreteFieldSpace
    target_space: DiscreteFieldSpace
    axes: tuple[int, ...] = eqx.field(static=True)
    axis_actions: tuple[str, ...] = eqx.field(static=True)
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
        axis_actions: Sequence[str],
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
        actions = tuple(str(value) for value in axis_actions)
        if (
            len(axes_) != len(actions)
            or len(set(axes_)) != len(axes_)
            or any(not action for action in actions)
        ):
            raise ValueError("Unique axes and nonempty axis_actions must align.")
        kind = str(classification)
        if not kind:
            raise ValueError("classification must be non-empty.")
        self.operator = operator
        self.source_space = source_space
        self.target_space = target_space
        self.axes = axes_
        self.axis_actions = actions
        self.classification = kind
        self.exact = bool(exact)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-operator",
                "operator": operator.operator_id,
                "source": source_space.field_space_id,
                "target": target_space.field_space_id,
                "axes": list(axes_),
                "axis_actions": list(actions),
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
    prepared = discretization.axes[axis_]
    exact = order_ == 0 or prepared.derivative_exact
    if prepared.derivative_matrix is None:
        diagonal = _axis_multiplier(discretization, axis_, order_).reshape((-1,))
        operator: AbstractLinearOperator = DiagonalLinearOperator(
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
    else:
        operator = FunctionLinearOperator(
            lambda coefficients: discretization.modal_derivative(
                coefficients,
                axis=axis_,
                order=order_,
            ),
            source=discretization.modal_space.vector_space,
            target=discretization.modal_space.vector_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "spectral-derivative-operator",
                    "discretization": discretization.prepared_id,
                    "axis": axis_,
                    "order": order_,
                    "path": "axis-action",
                }
            ),
        )
    return PreparedSpectralOperator(
        operator,
        discretization.modal_space,
        discretization.modal_space,
        axes=(axis_,),
        axis_actions=(f"derivative:{order_}",),
        classification="spectral-derivative",
        exact=exact,
    )


def spectral_hilbert_operator(
    discretization: TensorSpectralDiscretization,
    axis: int,
    /,
) -> PreparedSpectralOperator:
    """Return the exact discrete periodic Hilbert transform on one Fourier axis."""
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    axis_ = int(axis)
    if axis_ < 0 or axis_ >= len(discretization.axes):
        raise ValueError(f"axis must lie in [0, {len(discretization.axes)}).")
    prepared = discretization.axes[axis_]
    if prepared.family != "fourier":
        raise ValueError("The spectral Hilbert transform requires a Fourier axis.")
    numbers = prepared.modes.mode_numbers
    active = ~(prepared.modes.zero_mask | prepared.modes.nyquist_mask)
    multiplier = (
        -1j
        * jnp.sign(numbers).astype(
            jnp.dtype(discretization.plan.precision.coefficient_dtype)
        )
        * active
    )
    shape = [1] * len(discretization.modal_shape)
    shape[axis_] = multiplier.size
    diagonal = jnp.broadcast_to(
        multiplier.reshape(tuple(shape)),
        discretization.modal_shape,
    ).reshape((-1,))
    operator = DiagonalLinearOperator(
        diagonal,
        space=discretization.modal_space.vector_space,
        properties=OperatorProperties(
            diagonal=True,
            evidence={"diagonal": "construction"},
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "spectral-hilbert-operator",
                "discretization": discretization.prepared_id,
                "axis": axis_,
                "zero_mode": "zero",
                "nyquist_mode": "zero",
            }
        ),
    )
    return PreparedSpectralOperator(
        operator,
        discretization.modal_space,
        discretization.modal_space,
        axes=(axis_,),
        axis_actions=("hilbert",),
        classification="periodic-hilbert",
        exact=True,
    )


def spectral_laplacian_operator(
    discretization: TensorSpectralDiscretization,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> PreparedSpectralOperator:
    """Return a prepared modal Laplacian when every axis action is closed."""
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
    diagonal_families = ("fourier", "sine", "cosine")
    closed_actions = all(
        discretization.axes[axis].derivative_matrix is not None
        or discretization.axes[axis].family in diagonal_families
        for axis in selected
    )
    if not closed_actions:
        raise ValueError(
            "The selected basis does not define a closed modal derivative action."
        )
    diagonal_path = all(
        discretization.axes[axis].derivative_matrix is None for axis in selected
    )
    exact = all(discretization.axes[axis].derivative_exact for axis in selected)
    if diagonal_path:
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
        operator: AbstractLinearOperator = DiagonalLinearOperator(
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
    else:
        operator = FunctionLinearOperator(
            lambda coefficients: discretization.modal_laplacian(
                coefficients,
                axes=selected,
            ),
            source=discretization.modal_space.vector_space,
            target=discretization.modal_space.vector_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "spectral-laplacian-operator",
                    "discretization": discretization.prepared_id,
                    "axes": list(selected),
                    "path": "axis-actions",
                }
            ),
        )
    return PreparedSpectralOperator(
        operator,
        discretization.modal_space,
        discretization.modal_space,
        axes=selected,
        axis_actions=("derivative:2",) * len(selected),
        classification="spectral-laplacian",
        exact=exact,
    )


__all__ = [
    "PreparedSpectralOperator",
    "spectral_derivative_operator",
    "spectral_hilbert_operator",
    "spectral_laplacian_operator",
]
