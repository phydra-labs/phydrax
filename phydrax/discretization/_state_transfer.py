#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import hashlib
from collections.abc import Sequence
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._strict import StrictModule
from ..ein import contract
from ..linalg import ArraySpace, FunctionLinearOperator, transpose
from ..typing import checked, parse
from ._spaces import DiscreteFieldSpace
from ._transfer import FieldTransfer, TransferProperties


TensorGridBoundary: TypeAlias = Literal["endpoint", "periodic"]
TensorGridRestriction: TypeAlias = Literal["injection", "weighted"]


def _shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    resolved = tuple(value)
    if not resolved or any(size <= 0 for size in resolved):
        raise ValueError(
            f"StateTransfer.{name} must contain positive dimensions; got {resolved}"
        )
    return resolved


def _transfer_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(f"phydrax-{prefix}\0".encode())
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _apply_axis_matrix(values: Array, matrix: Array, axis: int, /) -> Array:
    transformed = contract("ij,j...->i...", matrix, jnp.moveaxis(values, axis, 0))
    return jnp.moveaxis(transformed, 0, axis)


def _endpoint_prolongation(coarse: int, fine: int, /) -> np.ndarray:
    if coarse == 1:
        return np.ones((fine, 1), dtype=np.float64)
    coarse_points = np.linspace(0.0, 1.0, coarse, dtype=np.float64)
    fine_points = np.linspace(0.0, 1.0, fine, dtype=np.float64)
    matrix = np.zeros((fine, coarse), dtype=np.float64)
    for row, point in enumerate(fine_points):
        upper = int(np.searchsorted(coarse_points, point, side="right"))
        if upper == 0:
            matrix[row, 0] = 1.0
        elif upper >= coarse:
            matrix[row, -1] = 1.0
        else:
            lower = upper - 1
            width = coarse_points[upper] - coarse_points[lower]
            fraction = (point - coarse_points[lower]) / width
            matrix[row, lower] = 1.0 - fraction
            matrix[row, upper] = fraction
    return matrix


def _periodic_prolongation(coarse: int, fine: int, /) -> np.ndarray:
    if coarse == 1:
        return np.ones((fine, 1), dtype=np.float64)
    matrix = np.zeros((fine, coarse), dtype=np.float64)
    for row in range(fine):
        position = float(row) * float(coarse) / float(fine)
        lower_unwrapped = int(np.floor(position))
        fraction = position - float(lower_unwrapped)
        lower = lower_unwrapped % coarse
        upper = (lower + 1) % coarse
        matrix[row, lower] += 1.0 - fraction
        matrix[row, upper] += fraction
    return matrix


def _injection_restriction(
    coarse: int,
    fine: int,
    boundary: TensorGridBoundary,
    /,
) -> np.ndarray:
    if coarse == 1:
        index = 0
        matrix = np.zeros((1, fine), dtype=np.float64)
        matrix[0, index] = 1.0
        return matrix
    if boundary == "periodic":
        if fine % coarse != 0:
            raise ValueError(
                f"TensorGridStateTransfer.fine_shape must have periodic injection axes divisible by coarse size {coarse}; got {fine}"
            )
        indices = np.arange(coarse, dtype=np.int64) * (fine // coarse)
    else:
        if (fine - 1) % (coarse - 1) != 0:
            raise ValueError(
                f"TensorGridStateTransfer.fine_shape must have nested endpoint-inclusive injection axes; got {fine=} and {coarse=}"
            )
        indices = np.arange(coarse, dtype=np.int64) * ((fine - 1) // (coarse - 1))
    matrix = np.zeros((coarse, fine), dtype=np.float64)
    matrix[np.arange(coarse), indices] = 1.0
    return matrix


class AbstractRefinementTransfer(StrictModule):
    """Restriction/prolongation contract between one fine and coarse state layout."""

    fine_shape: tuple[int, ...] = eqx.field(static=True)
    coarse_shape: tuple[int, ...] = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def restrict(self, fine_state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        raise NotImplementedError

    def restriction_operator(
        self,
        /,
        *,
        dtype: DTypeLike = np.float64,
    ) -> FunctionLinearOperator:
        """Return restriction as a rectangular linear operator."""
        source = ArraySpace(self.fine_shape, dtype=dtype)
        target = ArraySpace(self.coarse_shape, dtype=dtype)
        return FunctionLinearOperator(
            lambda value: self.restrict(value).astype(dtype),
            source=source,
            target=target,
            operator_id=f"{self.transfer_id}:restriction",
        )

    def prolongation_operator(
        self,
        /,
        *,
        dtype: DTypeLike = np.float64,
    ) -> FunctionLinearOperator:
        """Return prolongation as a rectangular linear operator."""
        source = ArraySpace(self.coarse_shape, dtype=dtype)
        target = ArraySpace(self.fine_shape, dtype=dtype)
        return FunctionLinearOperator(
            lambda value: self.prolong(value).astype(dtype),
            source=source,
            target=target,
            operator_id=f"{self.transfer_id}:prolongation",
        )

    @checked
    def field_transfer(
        self,
        fine_space: DiscreteFieldSpace,
        coarse_space: DiscreteFieldSpace,
        /,
        *,
        properties: TransferProperties | None = None,
    ) -> FieldTransfer:
        """Bind the refinement restriction to source and target field spaces.

        The primal map is the restriction, its dual pullback the algebraic
        transpose. Prolongation is a separate interpolation, not the Hilbert
        adjoint of the restriction, so no adjoint is claimed.
        """
        if fine_space.vector_space.size != int(
            np.prod(self.fine_shape)
        ) or coarse_space.vector_space.size != int(np.prod(self.coarse_shape)):
            raise ValueError(
                "StateTransfer.field_spaces must match refinement coordinate counts; got incompatible vector-space sizes"
            )
        restriction = FunctionLinearOperator(
            self.restrict,
            source=fine_space.vector_space,
            target=coarse_space.vector_space,
            operator_id=f"{self.transfer_id}:field-restriction",
        )
        return FieldTransfer(
            fine_space,
            coarse_space,
            restriction,
            dual_pullback_operator=transpose(restriction),
            properties=(
                TransferProperties(nested=True, semantics="restriction")
                if properties is None
                else properties
            ),
            transfer_id=self.transfer_id,
        )


class IdentityStateTransfer(AbstractRefinementTransfer):
    """Identity transfer for levels sharing one state layout."""

    def __init__(
        self, state_shape: Sequence[int], /, *, transfer_id: str | None = None
    ) -> None:
        shape = _shape(state_shape, "state_shape")
        self.fine_shape = shape
        self.coarse_shape = shape
        self.transfer_id = transfer_id or _transfer_id("identity-transfer", shape)

    def restrict(self, fine_state: ArrayLike, /) -> Array:
        values = jnp.asarray(fine_state)
        if values.shape[: len(self.fine_shape)] != self.fine_shape:
            raise ValueError(
                f"IdentityStateTransfer.fine_state must begin with declared state shape; got {values.shape}"
            )
        return values

    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse_state)
        if values.shape[: len(self.coarse_shape)] != self.coarse_shape:
            raise ValueError(
                f"IdentityStateTransfer.coarse_state must begin with declared state shape; got {values.shape}"
            )
        return values


class TensorGridStateTransfer(AbstractRefinementTransfer):
    """Nested tensor-grid restriction and multilinear prolongation."""

    prolongation_matrices: tuple[Array, ...]
    restriction_matrices: tuple[Array, ...]
    boundary: TensorGridBoundary = eqx.field(static=True)
    restriction: TensorGridRestriction = eqx.field(static=True)

    def __init__(
        self,
        fine_shape: Sequence[int],
        coarse_shape: Sequence[int],
        /,
        *,
        boundary: TensorGridBoundary = "endpoint",
        restriction: TensorGridRestriction = "injection",
        transfer_id: str | None = None,
    ) -> None:
        fine = _shape(fine_shape, "fine_shape")
        coarse = _shape(coarse_shape, "coarse_shape")
        if len(fine) != len(coarse):
            raise ValueError(
                f"TensorGridStateTransfer.fine_shape must have coarse_shape rank; got {fine=} and {coarse=}"
            )
        if any(fine_size < coarse_size for fine_size, coarse_size in zip(fine, coarse)):
            raise ValueError(
                f"TensorGridStateTransfer.fine_shape must have axes at least as large as coarse_shape; got {fine=} and {coarse=}"
            )
        boundary = parse(boundary, TensorGridBoundary, "boundary")
        restriction = parse(restriction, TensorGridRestriction, "restriction")
        prolongations: list[Array] = []
        restrictions: list[Array] = []
        for fine_size, coarse_size in zip(fine, coarse, strict=True):
            match boundary:
                case "periodic":
                    prolongation = _periodic_prolongation(coarse_size, fine_size)
                case "endpoint":
                    prolongation = _endpoint_prolongation(coarse_size, fine_size)
                case invalid:
                    assert_never(invalid)
            match restriction:
                case "injection":
                    restriction_matrix = _injection_restriction(
                        coarse_size, fine_size, boundary
                    )
                case "weighted":
                    restriction_matrix = prolongation.T
                    row_mass = np.sum(restriction_matrix, axis=1, keepdims=True)
                    restriction_matrix = restriction_matrix / row_mass
                case invalid:
                    assert_never(invalid)
            prolongations.append(jnp.asarray(prolongation))
            restrictions.append(jnp.asarray(restriction_matrix))
        self.fine_shape = fine
        self.coarse_shape = coarse
        self.prolongation_matrices = tuple(prolongations)
        self.restriction_matrices = tuple(restrictions)
        self.boundary = boundary
        self.restriction = restriction
        self.transfer_id = transfer_id or _transfer_id(
            "tensor-grid-transfer", fine, coarse, boundary, restriction
        )

    def restrict(self, fine_state: ArrayLike, /) -> Array:
        values = jnp.asarray(fine_state)
        if values.shape[: len(self.fine_shape)] != self.fine_shape:
            raise ValueError(
                f"TensorGridStateTransfer.fine_state must begin with fine_shape; got {values.shape}"
            )
        result = values
        for axis, matrix in enumerate(self.restriction_matrices):
            result = _apply_axis_matrix(result, matrix, axis)
        return result

    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse_state)
        if values.shape[: len(self.coarse_shape)] != self.coarse_shape:
            raise ValueError(
                f"TensorGridStateTransfer.coarse_state must begin with coarse_shape; got {values.shape}"
            )
        result = values
        for axis, matrix in enumerate(self.prolongation_matrices):
            result = _apply_axis_matrix(result, matrix, axis)
        return result


class SpectralCoefficientStateTransfer(AbstractRefinementTransfer):
    """Truncate or zero-pad ordered spectral coefficients on one state axis."""

    axis: int = eqx.field(static=True)

    def __init__(
        self,
        fine_shape: Sequence[int],
        coarse_shape: Sequence[int],
        /,
        *,
        axis: int = 0,
        transfer_id: str | None = None,
    ) -> None:
        fine = _shape(fine_shape, "fine_shape")
        coarse = _shape(coarse_shape, "coarse_shape")
        if len(fine) != len(coarse):
            raise ValueError(
                "SpectralCoefficientStateTransfer.fine_shape must have the same rank "
                f"as coarse_shape; got {fine!r} and {coarse!r}"
            )
        position = int(axis)
        if position < 0:
            position += len(fine)
        if position < 0 or position >= len(fine):
            raise ValueError(
                "SpectralCoefficientStateTransfer.axis must index the declared state rank; "
                f"got {axis!r} for rank {len(fine)}"
            )
        if any(
            fine_size != coarse_size
            for index, (fine_size, coarse_size) in enumerate(zip(fine, coarse))
            if index != position
        ):
            raise ValueError(
                "SpectralCoefficientStateTransfer.fine_shape must match coarse_shape "
                f"outside axis {position}; got {fine!r} and {coarse!r}"
            )
        if fine[position] < coarse[position]:
            raise ValueError(
                "SpectralCoefficientStateTransfer.fine_shape must be at least as large "
                f"as coarse_shape on axis {position}; got {fine[position]} and {coarse[position]}"
            )
        self.fine_shape = fine
        self.coarse_shape = coarse
        self.axis = position
        self.transfer_id = transfer_id or _transfer_id(
            "spectral-coefficient-transfer", fine, coarse, position
        )

    def restrict(self, fine_state: ArrayLike, /) -> Array:
        values = jnp.asarray(fine_state)
        if values.shape[: len(self.fine_shape)] != self.fine_shape:
            raise ValueError(
                "SpectralCoefficientStateTransfer.fine_state must begin with "
                f"fine_shape {self.fine_shape!r}; got {values.shape!r}"
            )
        indices = jnp.arange(self.coarse_shape[self.axis])
        return jnp.take(values, indices, axis=self.axis)

    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse_state)
        if values.shape[: len(self.coarse_shape)] != self.coarse_shape:
            raise ValueError(
                "SpectralCoefficientStateTransfer.coarse_state must begin with "
                f"coarse_shape {self.coarse_shape!r}; got {values.shape!r}"
            )
        padding = [(0, 0)] * values.ndim
        padding[self.axis] = (
            0,
            self.fine_shape[self.axis] - self.coarse_shape[self.axis],
        )
        return jnp.pad(values, tuple(padding))


__all__ = [
    "AbstractRefinementTransfer",
    "IdentityStateTransfer",
    "SpectralCoefficientStateTransfer",
    "TensorGridBoundary",
    "TensorGridRestriction",
    "TensorGridStateTransfer",
]
