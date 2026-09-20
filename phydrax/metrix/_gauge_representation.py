#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._complex_matrix_manifold import SpecialUnitaryGroup, UnitaryGroup
from ._lie_group import AbstractLieGroup


def _color_axis(value: int, /) -> int:
    axis = int(value)
    if axis >= 0:
        raise ValueError("color_axis must be negative and payload-relative.")
    return axis


def _element(group: AbstractLieGroup, value: ArrayLike, /) -> Array:
    element = jnp.asarray(value)
    if element.shape[-2:] != group.point_shape:
        raise ValueError(
            f"Group elements must have trailing shape {group.point_shape}; got {element.shape}."
        )
    return element


def _apply_matrix(
    matrix: ArrayLike,
    vector: ArrayLike,
    dimension: int,
    color_axis: int,
    /,
) -> Array:
    operator = jnp.asarray(matrix)
    value = jnp.asarray(vector)
    axis = value.ndim + color_axis
    if axis < 0 or axis >= value.ndim:
        raise ValueError("color_axis lies outside the vector payload rank.")
    if value.shape[axis] != dimension:
        raise ValueError(
            f"The declared color axis must have extent {dimension}; got {value.shape[axis]}."
        )
    moved = jnp.moveaxis(value, axis, -1)
    vector_leading_rank = moved.ndim - 1
    matrix_leading_rank = operator.ndim - 2
    if matrix_leading_rank > vector_leading_rank:
        raise ValueError("Group-element batches cannot exceed vector batch rank.")
    expanded = operator.reshape(
        operator.shape[:-2]
        + (1,) * (vector_leading_rank - matrix_leading_rank)
        + operator.shape[-2:]
    )
    applied = (expanded @ moved[..., None])[..., 0]
    return jnp.moveaxis(applied, -1, axis)


class AbstractGaugeRepresentation(StrictModule, NonTrainableState):
    """Finite matrix representation acting on one declared payload color axis."""

    group: eqx.AbstractVar[AbstractLieGroup]
    dimension: eqx.AbstractVar[int]
    color_axis: eqx.AbstractVar[int]
    representation_id: eqx.AbstractVar[str]

    @abstractmethod
    def matrix(self, element: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def apply(self, element: ArrayLike, vector: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def generators(self, /) -> Array:
        """Return one anti-Hermitian representation generator per group coordinate."""
        raise NotImplementedError


class U1ChargeRepresentation(AbstractGaugeRepresentation):
    """Integer-charge unitary representation ``z -> z**charge`` of U(1)."""

    group: UnitaryGroup
    charge: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    color_axis: int = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: UnitaryGroup,
        charge: int,
        /,
        *,
        color_axis: int = -1,
    ):
        if not isinstance(group, UnitaryGroup) or group.dimension != 1:
            raise TypeError("U1ChargeRepresentation requires UnitaryGroup(1).")
        charge_ = int(charge)
        if charge_ != charge:
            raise ValueError("A compact U(1) representation requires integer charge.")
        axis = _color_axis(color_axis)
        self.group = group
        self.charge = charge_
        self.dimension = 1
        self.color_axis = axis
        self.representation_id = canonical_fingerprint(
            {
                "kind": "u1-charge-representation",
                "group": group.group_id,
                "charge": charge_,
                "color_axis": axis,
            }
        )

    def matrix(self, element: ArrayLike, /) -> Array:
        value = _element(self.group, element)
        scalar = value[..., 0, 0] ** self.charge
        return scalar[..., None, None]

    def apply(self, element: ArrayLike, vector: ArrayLike, /) -> Array:
        return _apply_matrix(
            self.matrix(element), vector, self.dimension, self.color_axis
        )

    def generators(self, /) -> Array:
        return jnp.asarray([[[1j * self.charge]]])


class FundamentalGaugeRepresentation(AbstractGaugeRepresentation):
    """Defining representation of U(N) or SU(N)."""

    group: UnitaryGroup | SpecialUnitaryGroup
    dimension: int = eqx.field(static=True)
    color_axis: int = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: UnitaryGroup | SpecialUnitaryGroup,
        /,
        *,
        color_axis: int = -1,
    ):
        if not isinstance(group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("Fundamental representation requires U(N) or SU(N).")
        axis = _color_axis(color_axis)
        self.group = group
        self.dimension = group.dimension
        self.color_axis = axis
        self.representation_id = canonical_fingerprint(
            {
                "kind": "fundamental-gauge-representation",
                "group": group.group_id,
                "dimension": group.dimension,
                "color_axis": axis,
            }
        )

    def matrix(self, element: ArrayLike, /) -> Array:
        return _element(self.group, element)

    def apply(self, element: ArrayLike, vector: ArrayLike, /) -> Array:
        return _apply_matrix(
            self.matrix(element), vector, self.dimension, self.color_axis
        )

    def generators(self, /) -> Array:
        coordinates = jnp.eye(self.group.algebra_shape[0])
        return self.group.hat(coordinates)


class AdjointGaugeRepresentation(AbstractGaugeRepresentation):
    """Real adjoint representation of SU(N) in its canonical algebra coordinates."""

    group: SpecialUnitaryGroup
    dimension: int = eqx.field(static=True)
    color_axis: int = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: SpecialUnitaryGroup,
        /,
        *,
        color_axis: int = -1,
    ):
        if not isinstance(group, SpecialUnitaryGroup):
            raise TypeError("AdjointGaugeRepresentation requires SU(N).")
        axis = _color_axis(color_axis)
        self.group = group
        self.dimension = group.algebra_shape[0]
        self.color_axis = axis
        self.representation_id = canonical_fingerprint(
            {
                "kind": "adjoint-gauge-representation",
                "group": group.group_id,
                "dimension": self.dimension,
                "basis": "canonical-group-hat-vee",
                "color_axis": axis,
            }
        )

    def matrix(self, element: ArrayLike, /) -> Array:
        value = _element(self.group, element)
        basis = self.group.hat(jnp.eye(self.dimension))
        inverse = self.group.inverse(value)
        leading = value.shape[:-2]
        basis_ = basis.reshape((1,) * len(leading) + basis.shape)
        conjugated = value[..., None, :, :] @ basis_ @ inverse[..., None, :, :]
        columns = self.group.vee(conjugated)
        return jnp.swapaxes(columns, -1, -2)

    def apply(self, element: ArrayLike, vector: ArrayLike, /) -> Array:
        return _apply_matrix(
            self.matrix(element), vector, self.dimension, self.color_axis
        )

    def generators(self, /) -> Array:
        basis = self.group.hat(jnp.eye(self.dimension))
        left = basis[:, None, :, :]
        right = basis[None, :, :, :]
        brackets = left @ right - right @ left
        columns = self.group.vee(brackets)
        return jnp.swapaxes(columns, -1, -2)


__all__ = [
    "AbstractGaugeRepresentation",
    "AdjointGaugeRepresentation",
    "FundamentalGaugeRepresentation",
    "U1ChargeRepresentation",
]
