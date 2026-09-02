#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._lie_group import AbstractLieGroup
from ._manifold import (
    _array_with_trailing_shape,
    _same_shape,
    AbstractGeodesicManifold,
    AbstractRiemannianManifold,
)


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _hermitian(value: Array, /) -> Array:
    return 0.5 * (value + _adjoint(value))


def _skew_hermitian(value: Array, /) -> Array:
    return 0.5 * (value - _adjoint(value))


def _matrix_map(function, value: Array, /) -> Array:
    if value.ndim == 2:
        return function(value)
    leading = value.shape[:-2]
    flat = value.reshape((-1,) + value.shape[-2:])
    mapped = jax.vmap(function)(flat)
    return mapped.reshape(leading + mapped.shape[1:])


def _matrix_exponential(value: Array, /) -> Array:
    return _matrix_map(jsp.linalg.expm, value)


def _unitary_logarithm(value: Array, /, *, traceless: bool) -> Array:
    def logarithm(matrix: Array) -> Array:
        eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
        angles = jnp.angle(eigenvalues)
        if traceless:
            angles = angles.at[-1].add(-jnp.sum(angles))
        return _skew_hermitian(
            (eigenvectors * (1j * angles)[None, :]) @ _adjoint(eigenvectors)
        )

    return _matrix_map(logarithm, value)


def _hermitian_spectral_map(value: Array, function, /) -> Array:
    def apply(matrix: Array) -> Array:
        eigenvalues, eigenvectors = jnp.linalg.eigh(_hermitian(matrix))
        return _hermitian(
            (eigenvectors * function(eigenvalues)[None, :]) @ _adjoint(eigenvectors)
        )

    return _matrix_map(apply, value)


def _hermitian_square_root(value: Array, /) -> Array:
    return _hermitian_spectral_map(value, jnp.sqrt)


def _hermitian_inverse_square_root(value: Array, /) -> Array:
    return _hermitian_spectral_map(value, lambda eigenvalues: 1.0 / jnp.sqrt(eigenvalues))


def _hermitian_logarithm(value: Array, /) -> Array:
    return _hermitian_spectral_map(value, jnp.log)


class UnitaryGroup(AbstractLieGroup):
    """Dense unitary group U(n) with skew-Hermitian algebra matrices."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-7):
        dimension_ = int(dimension)
        if dimension_ < 1:
            raise ValueError("Unitary dimension must be positive.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Unitary tolerance must be finite and positive.")
        self.dimension = dimension_
        self.tolerance = float(tolerance)
        self.group_id = f"lie-group:u:{dimension_}"
        self.point_shape = (dimension_, dimension_)

    def _matrix(self, value: ArrayLike, name: str, /) -> Array:
        matrix = _array_with_trailing_shape(value, self.point_shape, name)
        if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
            raise TypeError(f"{name} must use complex floating-point coordinates.")
        return matrix

    def identity(self, *, dtype: Any = jnp.complex128) -> Array:
        return jnp.eye(self.dimension, dtype=dtype)

    def contains(self, point: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "Unitary point")
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        residual = jnp.max(jnp.abs(_adjoint(matrix) @ matrix - identity), axis=(-2, -1))
        return jnp.all(jnp.isfinite(matrix)) & jnp.all(residual <= self.tolerance)

    def compose(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self._matrix(left, "Unitary point") @ self._matrix(right, "Unitary point")

    def inverse(self, point: ArrayLike, /) -> Array:
        return _adjoint(self._matrix(point, "Unitary point"))

    def project_algebra(self, value: ArrayLike, /) -> Array:
        return _skew_hermitian(self._matrix(value, "Unitary algebra element"))

    def exp(self, algebra: ArrayLike, /) -> Array:
        return _matrix_exponential(self.project_algebra(algebra))

    def log(self, point: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "Unitary point")
        matrix = eqx.error_if(
            matrix,
            ~self.contains(matrix),
            "Unitary logarithm requires a unitary matrix.",
        )
        return _unitary_logarithm(matrix, traceless=False)


class SpecialUnitaryGroup(AbstractLieGroup):
    """Dense special-unitary group SU(n)."""

    unitary: UnitaryGroup
    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-7):
        if int(dimension) < 2:
            raise ValueError("Special-unitary dimension must be at least two.")
        self.unitary = UnitaryGroup(dimension, tolerance=tolerance)
        self.dimension = self.unitary.dimension
        self.tolerance = self.unitary.tolerance
        self.group_id = f"lie-group:su:{self.dimension}"
        self.point_shape = self.unitary.point_shape

    def identity(self, *, dtype: Any = jnp.complex128) -> Array:
        return self.unitary.identity(dtype=dtype)

    def contains(self, point: ArrayLike, /) -> Array:
        matrix = self.unitary._matrix(point, "Special-unitary point")
        determinant = jnp.linalg.det(matrix)
        return self.unitary.contains(matrix) & jnp.all(
            jnp.abs(determinant - 1.0) <= self.tolerance
        )

    def compose(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.unitary.compose(left, right)

    def inverse(self, point: ArrayLike, /) -> Array:
        return self.unitary.inverse(point)

    def project_algebra(self, value: ArrayLike, /) -> Array:
        matrix = self.unitary.project_algebra(value)
        trace = jnp.trace(matrix, axis1=-2, axis2=-1) / float(self.dimension)
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        return _skew_hermitian(matrix - trace[..., None, None] * identity)

    def exp(self, algebra: ArrayLike, /) -> Array:
        return _matrix_exponential(self.project_algebra(algebra))

    def log(self, point: ArrayLike, /) -> Array:
        matrix = self.unitary._matrix(point, "Special-unitary point")
        matrix = eqx.error_if(
            matrix,
            ~self.contains(matrix),
            "Special-unitary logarithm requires an SU(n) matrix.",
        )
        return self.project_algebra(_unitary_logarithm(matrix, traceless=True))


class UnitaryManifold(AbstractGeodesicManifold):
    """U(n) with its bi-invariant Frobenius metric."""

    group: UnitaryGroup
    dimension: int = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-7):
        self.group = UnitaryGroup(dimension, tolerance=tolerance)
        self.dimension = self.group.dimension
        self.manifold_id = f"manifold:unitary:{dimension}"
        self.point_shape = self.group.point_shape
        self.retraction_method = "group-exponential"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    @property
    def scalar_field(self) -> str:
        return "complex"

    def contains(self, point: ArrayLike, /) -> Array:
        return self.group.contains(point)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        matrix = self.group._matrix(point, "Unitary point")
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        residual = jnp.max(jnp.abs(_adjoint(matrix) @ matrix - identity), initial=0.0)
        return jnp.where(jnp.all(jnp.isfinite(matrix)), residual, jnp.inf)

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        matrix = self.group._matrix(point, "Unitary point")
        vector = self.group._matrix(ambient_vector, "Unitary tangent")
        _same_shape(vector, matrix, "Unitary tangent")
        return matrix @ _skew_hermitian(_adjoint(matrix) @ vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, jnp.conj(ambient_cotangent))

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return jnp.real(jnp.vdot(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        return self.exp(point, tangent_step)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = self.group._matrix(point, "Unitary point")
        self.project_tangent(matrix, tangent_step)
        target = self.group._matrix(destination, "Unitary destination")
        vector = self.group._matrix(tangent, "Unitary transported tangent")
        _same_shape(target, matrix, "Unitary destination")
        _same_shape(vector, matrix, "Unitary transported tangent")
        return self.project_tangent(target, vector)

    def exp(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        matrix = self.group._matrix(point, "Unitary point")
        step = self.project_tangent(matrix, tangent)
        return matrix @ self.group.exp(_adjoint(matrix) @ step)

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        matrix = self.group._matrix(point, "Unitary point")
        target = self.group._matrix(destination, "Unitary destination")
        _same_shape(target, matrix, "Unitary destination")
        return matrix @ self.group.log(_adjoint(matrix) @ target)

    def squared_distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        logarithm = self.log(left, right)
        return jnp.real(jnp.vdot(logarithm, logarithm))


class SpecialUnitaryManifold(AbstractGeodesicManifold):
    """SU(n) with normalized homogeneous matrix representatives."""

    group: SpecialUnitaryGroup
    dimension: int = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-7):
        self.group = SpecialUnitaryGroup(dimension, tolerance=tolerance)
        self.dimension = self.group.dimension
        self.manifold_id = f"manifold:special-unitary:{dimension}"
        self.point_shape = self.group.point_shape
        self.retraction_method = "group-exponential"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    @property
    def scalar_field(self) -> str:
        return "complex"

    def contains(self, point: ArrayLike, /) -> Array:
        return self.group.contains(point)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        matrix = self.group.unitary._matrix(point, "Special-unitary point")
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        unitary = jnp.max(jnp.abs(_adjoint(matrix) @ matrix - identity), initial=0.0)
        determinant = jnp.abs(jnp.linalg.det(matrix) - 1.0)
        return jnp.where(
            jnp.all(jnp.isfinite(matrix)), jnp.maximum(unitary, determinant), jnp.inf
        )

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        matrix = self.group.unitary._matrix(point, "Special-unitary point")
        vector = self.group.unitary._matrix(ambient_vector, "Special-unitary tangent")
        _same_shape(vector, matrix, "Special-unitary tangent")
        return matrix @ self.group.project_algebra(_adjoint(matrix) @ vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, jnp.conj(ambient_cotangent))

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return jnp.real(jnp.vdot(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        return self.exp(point, tangent_step)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        self.project_tangent(point, tangent_step)
        return self.project_tangent(destination, tangent)

    def exp(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        matrix = self.group.unitary._matrix(point, "Special-unitary point")
        step = self.project_tangent(matrix, tangent)
        return matrix @ self.group.exp(_adjoint(matrix) @ step)

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        matrix = self.group.unitary._matrix(point, "Special-unitary point")
        target = self.group.unitary._matrix(destination, "Special-unitary destination")
        _same_shape(target, matrix, "Special-unitary destination")
        return matrix @ self.group.log(_adjoint(matrix) @ target)

    def squared_distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        logarithm = self.log(left, right)
        return jnp.real(jnp.vdot(logarithm, logarithm))


class AffineInvariantHPDManifold(AbstractGeodesicManifold):
    """Hermitian positive-definite matrices with affine-invariant geometry."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-8):
        dimension_ = int(dimension)
        if dimension_ < 1:
            raise ValueError("HPD dimension must be positive.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("HPD tolerance must be finite and positive.")
        self.dimension = dimension_
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:hpd:{dimension_}:affine-invariant"
        self.point_shape = (dimension_, dimension_)
        self.retraction_method = "affine-exponential"
        self.transport_method = "affine-geodesic"
        self.transport_is_isometric = True
        self.transport_is_parallel = True

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _matrix(self, value: ArrayLike, name: str, /) -> Array:
        matrix = _array_with_trailing_shape(value, self.point_shape, name)
        if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
            raise TypeError(f"{name} must use complex floating-point coordinates.")
        return matrix

    def contains(self, point: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "HPD point")
        asymmetry = jnp.max(jnp.abs(matrix - _adjoint(matrix)), axis=(-2, -1))
        eigenvalues = jnp.linalg.eigvalsh(_hermitian(matrix))
        return (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(asymmetry <= self.tolerance)
            & jnp.all(eigenvalues > self.tolerance)
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "HPD point")
        asymmetry = jnp.max(jnp.abs(matrix - _adjoint(matrix)), initial=0.0)
        minimum = jnp.min(jnp.linalg.eigvalsh(_hermitian(matrix)), initial=jnp.inf)
        return jnp.where(
            jnp.all(jnp.isfinite(matrix)),
            jnp.maximum(asymmetry, jnp.maximum(self.tolerance - minimum, 0.0)),
            jnp.inf,
        )

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        matrix = self._matrix(point, "HPD point")
        vector = self._matrix(ambient_vector, "HPD tangent")
        _same_shape(vector, matrix, "HPD tangent")
        return _hermitian(vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        matrix = self._matrix(point, "HPD point")
        cotangent = self.project_tangent(matrix, jnp.conj(ambient_cotangent))
        return _hermitian(matrix @ cotangent @ matrix)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = self._matrix(point, "HPD point")
        left = self.project_tangent(matrix, left_tangent)
        right = self.project_tangent(matrix, right_tangent)
        left_solved = jnp.linalg.solve(matrix, left)
        right_solved = jnp.linalg.solve(matrix, right)
        return jnp.real(
            jnp.sum(oe.contract("...ij,...ji->...", left_solved, right_solved))
        )

    def exp(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "HPD point")
        step = self.project_tangent(matrix, tangent)
        root = _hermitian_square_root(matrix)
        inverse_root = _hermitian_inverse_square_root(matrix)
        local = _hermitian(inverse_root @ step @ inverse_root)
        return _hermitian(root @ _matrix_exponential(local) @ root)

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "HPD point")
        target = self._matrix(destination, "HPD destination")
        _same_shape(target, matrix, "HPD destination")
        root = _hermitian_square_root(matrix)
        inverse_root = _hermitian_inverse_square_root(matrix)
        relative = _hermitian(inverse_root @ target @ inverse_root)
        return _hermitian(root @ _hermitian_logarithm(relative) @ root)

    def squared_distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        matrix = self._matrix(left, "HPD point")
        target = self._matrix(right, "HPD destination")
        _same_shape(target, matrix, "HPD destination")
        inverse_root = _hermitian_inverse_square_root(matrix)
        relative = _hermitian(inverse_root @ target @ inverse_root)
        eigenvalues = jnp.linalg.eigvalsh(relative)
        return jnp.sum(jnp.log(eigenvalues) ** 2)

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        return self.exp(point, tangent_step)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = self._matrix(point, "HPD point")
        self.project_tangent(matrix, tangent_step)
        target = self._matrix(destination, "HPD destination")
        vector = self.project_tangent(matrix, tangent)
        _same_shape(target, matrix, "HPD destination")
        inverse_root = _hermitian_inverse_square_root(matrix)
        relative_root = _hermitian_square_root(
            _hermitian(inverse_root @ target @ inverse_root)
        )
        congruence = _hermitian_square_root(matrix) @ relative_root @ inverse_root
        return _hermitian(congruence @ vector @ _adjoint(congruence))


class ComplexStiefelManifold(AbstractRiemannianManifold):
    """Complex matrices with orthonormal columns under the real Frobenius metric."""

    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, rows: int, columns: int, /, *, tolerance: float = 1e-8):
        rows_ = int(rows)
        columns_ = int(columns)
        if not 1 <= columns_ <= rows_:
            raise ValueError("Complex Stiefel requires 1 <= columns <= rows.")
        self.rows = rows_
        self.columns = columns_
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:complex-stiefel:{rows_}:{columns_}"
        self.point_shape = (rows_, columns_)
        self.retraction_method = "complex-qr"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _matrix(self, value: ArrayLike, name: str, /) -> Array:
        matrix = _array_with_trailing_shape(value, self.point_shape, name)
        if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
            raise TypeError(f"{name} must use complex coordinates.")
        return matrix

    def contains(self, point: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "Complex Stiefel point")
        identity = jnp.eye(self.columns, dtype=matrix.dtype)
        residual = jnp.max(jnp.abs(_adjoint(matrix) @ matrix - identity), axis=(-2, -1))
        return jnp.all(jnp.isfinite(matrix)) & jnp.all(residual <= self.tolerance)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "Complex Stiefel point")
        identity = jnp.eye(self.columns, dtype=matrix.dtype)
        return jnp.max(jnp.abs(_adjoint(matrix) @ matrix - identity))

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "Complex Stiefel point")
        vector = self._matrix(ambient_vector, "Complex Stiefel tangent")
        correction = _hermitian(_adjoint(matrix) @ vector)
        return vector - matrix @ correction

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        return self.project_tangent(point, jnp.conj(ambient_cotangent))

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return jnp.real(jnp.vdot(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        matrix = self._matrix(point, "Complex Stiefel point")
        step = self.project_tangent(matrix, tangent_step)
        q, r = jnp.linalg.qr(matrix + step, mode="reduced")
        diagonal = jnp.diag(r)
        phase = jnp.where(jnp.abs(diagonal) > 0.0, diagonal / jnp.abs(diagonal), 1.0)
        return q * jnp.conj(phase)[None, :]

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        self.project_tangent(point, tangent_step)
        return self.project_tangent(destination, tangent)


__all__ = [
    "AffineInvariantHPDManifold",
    "ComplexStiefelManifold",
    "SpecialUnitaryGroup",
    "SpecialUnitaryManifold",
    "UnitaryGroup",
    "UnitaryManifold",
]
