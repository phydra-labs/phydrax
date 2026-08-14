#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import hashlib
from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import ArraySpace, FunctionLinearOperator


TensorGridBoundary: TypeAlias = Literal["endpoint", "periodic"]
TensorGridRestriction: TypeAlias = Literal["injection", "weighted"]


def _shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if not resolved or any(size <= 0 for size in resolved):
        raise ValueError(f"{name} must contain positive dimensions.")
    return resolved


def _transfer_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(f"phydrax-{prefix}\0".encode())
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _apply_axis_matrix(values: Array, matrix: Array, axis: int, /) -> Array:
    transformed = jnp.tensordot(matrix, values, axes=((1,), (axis,)))
    return jnp.moveaxis(transformed, 0, axis)


def _endpoint_prolongation(coarse: int, fine: int, /) -> np.ndarray:
    if coarse == 1:
        return np.ones((fine, 1), dtype=float)
    coarse_points = np.linspace(0.0, 1.0, coarse)
    fine_points = np.linspace(0.0, 1.0, fine)
    matrix = np.zeros((fine, coarse), dtype=float)
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
        return np.ones((fine, 1), dtype=float)
    matrix = np.zeros((fine, coarse), dtype=float)
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
        matrix = np.zeros((1, fine), dtype=float)
        matrix[0, index] = 1.0
        return matrix
    if boundary == "periodic":
        if fine % coarse != 0:
            raise ValueError(
                "Periodic injection requires each fine axis size to be an integer "
                "multiple of its coarse size."
            )
        indices = np.arange(coarse, dtype=int) * (fine // coarse)
    else:
        if (fine - 1) % (coarse - 1) != 0:
            raise ValueError(
                "Endpoint injection requires nested endpoint-inclusive axis sizes."
            )
        indices = np.arange(coarse, dtype=int) * ((fine - 1) // (coarse - 1))
    matrix = np.zeros((coarse, fine), dtype=float)
    matrix[np.arange(coarse), indices] = 1.0
    return matrix


class AbstractStateTransfer(StrictModule):
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
        dtype=np.float64,
    ) -> FunctionLinearOperator:
        """Return restriction as a rectangular canonical linear operator."""
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
        dtype=np.float64,
    ) -> FunctionLinearOperator:
        """Return prolongation as a rectangular canonical linear operator."""
        source = ArraySpace(self.coarse_shape, dtype=dtype)
        target = ArraySpace(self.fine_shape, dtype=dtype)
        return FunctionLinearOperator(
            lambda value: self.prolong(value).astype(dtype),
            source=source,
            target=target,
            operator_id=f"{self.transfer_id}:prolongation",
        )


class IdentityStateTransfer(AbstractStateTransfer):
    """Identity transfer for levels sharing one state layout."""

    def __init__(self, state_shape: Sequence[int], /, *, transfer_id: str | None = None):
        shape = _shape(state_shape, "state_shape")
        self.fine_shape = shape
        self.coarse_shape = shape
        self.transfer_id = transfer_id or _transfer_id("identity-transfer", shape)

    def restrict(self, fine_state: ArrayLike, /) -> Array:
        values = jnp.asarray(fine_state)
        if values.shape[: len(self.fine_shape)] != self.fine_shape:
            raise ValueError("fine_state does not begin with the declared state shape.")
        return values

    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse_state)
        if values.shape[: len(self.coarse_shape)] != self.coarse_shape:
            raise ValueError("coarse_state does not begin with the declared state shape.")
        return values


class TensorGridStateTransfer(AbstractStateTransfer):
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
    ):
        fine = _shape(fine_shape, "fine_shape")
        coarse = _shape(coarse_shape, "coarse_shape")
        if len(fine) != len(coarse):
            raise ValueError("fine_shape and coarse_shape must have equal rank.")
        if any(fine_size < coarse_size for fine_size, coarse_size in zip(fine, coarse)):
            raise ValueError("Every fine grid axis must be at least as large as coarse.")
        if boundary not in ("endpoint", "periodic"):
            raise ValueError("boundary must be 'endpoint' or 'periodic'.")
        if restriction not in ("injection", "weighted"):
            raise ValueError("restriction must be 'injection' or 'weighted'.")
        prolongations: list[Array] = []
        restrictions: list[Array] = []
        for fine_size, coarse_size in zip(fine, coarse, strict=True):
            prolongation = (
                _periodic_prolongation(coarse_size, fine_size)
                if boundary == "periodic"
                else _endpoint_prolongation(coarse_size, fine_size)
            )
            if restriction == "injection":
                restriction_matrix = _injection_restriction(
                    coarse_size, fine_size, boundary
                )
            else:
                restriction_matrix = prolongation.T
                row_mass = np.sum(restriction_matrix, axis=1, keepdims=True)
                restriction_matrix = restriction_matrix / row_mass
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
            raise ValueError("fine_state does not begin with fine_shape.")
        result = values
        for axis, matrix in enumerate(self.restriction_matrices):
            result = _apply_axis_matrix(result, matrix, axis)
        return result

    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse_state)
        if values.shape[: len(self.coarse_shape)] != self.coarse_shape:
            raise ValueError("coarse_state does not begin with coarse_shape.")
        result = values
        for axis, matrix in enumerate(self.prolongation_matrices):
            result = _apply_axis_matrix(result, matrix, axis)
        return result


class SpectralCoefficientStateTransfer(AbstractStateTransfer):
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
    ):
        fine = _shape(fine_shape, "fine_shape")
        coarse = _shape(coarse_shape, "coarse_shape")
        if len(fine) != len(coarse):
            raise ValueError("fine_shape and coarse_shape must have equal rank.")
        position = int(axis)
        if position < 0:
            position += len(fine)
        if position < 0 or position >= len(fine):
            raise ValueError("axis is outside the declared state rank.")
        if any(
            fine_size != coarse_size
            for index, (fine_size, coarse_size) in enumerate(zip(fine, coarse))
            if index != position
        ):
            raise ValueError("Spectral transfer may change only its declared axis.")
        if fine[position] < coarse[position]:
            raise ValueError("The fine spectral axis cannot be smaller than coarse.")
        self.fine_shape = fine
        self.coarse_shape = coarse
        self.axis = position
        self.transfer_id = transfer_id or _transfer_id(
            "spectral-coefficient-transfer", fine, coarse, position
        )

    def restrict(self, fine_state: ArrayLike, /) -> Array:
        values = jnp.asarray(fine_state)
        if values.shape[: len(self.fine_shape)] != self.fine_shape:
            raise ValueError("fine_state does not begin with fine_shape.")
        indices = jnp.arange(self.coarse_shape[self.axis])
        return jnp.take(values, indices, axis=self.axis)

    def prolong(self, coarse_state: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse_state)
        if values.shape[: len(self.coarse_shape)] != self.coarse_shape:
            raise ValueError("coarse_state does not begin with coarse_shape.")
        padding = [(0, 0)] * values.ndim
        padding[self.axis] = (
            0,
            self.fine_shape[self.axis] - self.coarse_shape[self.axis],
        )
        return jnp.pad(values, tuple(padding))


__all__ = [
    "AbstractStateTransfer",
    "IdentityStateTransfer",
    "SpectralCoefficientStateTransfer",
    "TensorGridBoundary",
    "TensorGridRestriction",
    "TensorGridStateTransfer",
]
