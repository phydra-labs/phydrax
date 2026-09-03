#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._manifold import (
    _array_with_trailing_shape,
    _finite_residual,
    _real_inner,
    _same_shape,
    AbstractGeodesicManifold,
    AbstractRiemannianManifold,
)
from ._state_geometry import (
    SpecialOrthogonalStateGeometry,
    SymmetricPositiveDefiniteStateGeometry,
)


MatrixRetraction: TypeAlias = Literal["exponential", "cayley"]


def _matrix_dimensions(rows: int, columns: int, name: str, /) -> tuple[int, int]:
    n = int(rows)
    p = int(columns)
    if n < 1 or p < 1:
        raise ValueError(f"{name} dimensions must be positive.")
    return n, p


def _tolerance(value: float, name: str, /) -> float:
    tolerance = float(value)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name} tolerance must be finite and positive.")
    return tolerance


def _transpose(value: Array, /) -> Array:
    return jnp.swapaxes(value, -1, -2)


def _symmetric(value: Array, /) -> Array:
    return 0.5 * (value + _transpose(value))


def _skew(value: Array, /) -> Array:
    return 0.5 * (value - _transpose(value))


def _qr_retraction(candidate: Array, /) -> Array:
    orthogonal, triangular = jnp.linalg.qr(candidate, mode="reduced")
    diagonal = jnp.diagonal(triangular, axis1=-2, axis2=-1)
    signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
    return orthogonal * jnp.expand_dims(signs, axis=-2)


def _orthonormal_residual(value: Array, columns: int, /) -> Array:
    identity = jnp.eye(columns, dtype=value.dtype)
    defects = jnp.abs(_transpose(value) @ value - identity)
    return jnp.max(defects, initial=0.0)


def _inverse_congruence(factor: Array, value: Array, /) -> Array:
    left_solved = jnp.linalg.solve(factor, value)
    return _transpose(jnp.linalg.solve(factor, _transpose(left_solved)))


def _symmetric_square_root(value: Array, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetric(value))
    safe = jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny)
    return _symmetric(
        (eigenvectors * jnp.expand_dims(jnp.sqrt(safe), axis=-2))
        @ _transpose(eigenvectors)
    )


class StiefelManifold(AbstractRiemannianManifold):
    """Real Stiefel manifold with induced metric and QR retraction."""

    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, rows: int, columns: int, /, *, tolerance: float = 1e-6):
        n, p = _matrix_dimensions(rows, columns, "Stiefel")
        if p > n:
            raise ValueError("Stiefel columns must not exceed rows.")
        self.rows = n
        self.columns = p
        self.tolerance = _tolerance(tolerance, "Stiefel")
        self.manifold_id = f"manifold:stiefel:{n}x{p}:induced"
        self.point_shape = (n, p)
        self.retraction_method = "qr"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Stiefel point")
        residual = _finite_residual(value, _orthonormal_residual(value, self.columns))
        return residual <= self.tolerance

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Stiefel point")
        return _finite_residual(value, _orthonormal_residual(value, self.columns))

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Stiefel point")
        vector = self._point(ambient_vector, "Stiefel tangent")
        _same_shape(vector, value, "Stiefel tangent")
        return vector - value @ _symmetric(_transpose(value) @ vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Stiefel point")
        left = self._point(left_tangent, "Stiefel left tangent")
        right = self._point(right_tangent, "Stiefel right tangent")
        _same_shape(left, value, "Stiefel left tangent")
        _same_shape(right, value, "Stiefel right tangent")
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Stiefel point")
        step = self._point(tangent_step, "Stiefel tangent step")
        _same_shape(step, value, "Stiefel tangent step")
        return _qr_retraction(value + step)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Stiefel point")
        step = self._point(tangent_step, "Stiefel tangent step")
        target = self._point(destination, "Stiefel destination")
        vector = self._point(tangent, "Stiefel transported tangent")
        _same_shape(step, value, "Stiefel tangent step")
        _same_shape(target, value, "Stiefel destination")
        _same_shape(vector, value, "Stiefel transported tangent")
        return self.project_tangent(target, vector)


class GrassmannManifold(AbstractRiemannianManifold):
    """Real Grassmann manifold represented by orthonormal basis matrices."""

    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, rows: int, columns: int, /, *, tolerance: float = 1e-6):
        n, p = _matrix_dimensions(rows, columns, "Grassmann")
        if p >= n:
            raise ValueError("Grassmann columns must be strictly less than rows.")
        self.rows = n
        self.columns = p
        self.tolerance = _tolerance(tolerance, "Grassmann")
        self.manifold_id = f"manifold:grassmann:{n}x{p}:quotient-induced"
        self.point_shape = (n, p)
        self.retraction_method = "qr"
        self.transport_method = "horizontal-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Grassmann representative")
        residual = _finite_residual(value, _orthonormal_residual(value, self.columns))
        return residual <= self.tolerance

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Grassmann representative")
        return _finite_residual(value, _orthonormal_residual(value, self.columns))

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Grassmann representative")
        vector = self._point(ambient_vector, "Grassmann tangent")
        _same_shape(vector, value, "Grassmann tangent")
        return vector - value @ (_transpose(value) @ vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Grassmann representative")
        left = self._point(left_tangent, "Grassmann left tangent")
        right = self._point(right_tangent, "Grassmann right tangent")
        _same_shape(left, value, "Grassmann left tangent")
        _same_shape(right, value, "Grassmann right tangent")
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Grassmann representative")
        step = self._point(tangent_step, "Grassmann tangent step")
        _same_shape(step, value, "Grassmann tangent step")
        return _qr_retraction(value + step)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Grassmann representative")
        step = self._point(tangent_step, "Grassmann tangent step")
        target = self._point(destination, "Grassmann destination")
        vector = self._point(tangent, "Grassmann transported tangent")
        _same_shape(step, value, "Grassmann tangent step")
        _same_shape(target, value, "Grassmann destination")
        _same_shape(vector, value, "Grassmann transported tangent")
        return self.project_tangent(target, vector)


class ObliqueManifold(AbstractRiemannianManifold):
    """Matrices whose columns are independent unit-sphere points."""

    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, rows: int, columns: int, /, *, tolerance: float = 1e-6):
        n, p = _matrix_dimensions(rows, columns, "Oblique")
        self.rows = n
        self.columns = p
        self.tolerance = _tolerance(tolerance, "Oblique")
        self.manifold_id = f"manifold:oblique:{n}x{p}:induced"
        self.point_shape = (n, p)
        self.retraction_method = "column-normalization"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Oblique point")
        squared_norms = jnp.sum(value * value, axis=-2)
        return _finite_residual(
            value,
            jnp.max(jnp.abs(squared_norms - 1.0), initial=0.0),
        )

    def contains(self, point: ArrayLike, /) -> Array:
        return self.constraint_residual(point) <= self.tolerance

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Oblique point")
        vector = self._point(ambient_vector, "Oblique tangent")
        _same_shape(vector, value, "Oblique tangent")
        radial = jnp.sum(value * vector, axis=-2, keepdims=True)
        return vector - value * radial

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Oblique point")
        left = self._point(left_tangent, "Oblique left tangent")
        right = self._point(right_tangent, "Oblique right tangent")
        _same_shape(left, value, "Oblique left tangent")
        _same_shape(right, value, "Oblique right tangent")
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Oblique point")
        step = self._point(tangent_step, "Oblique tangent step")
        _same_shape(step, value, "Oblique tangent step")
        candidate = value + step
        norms = jnp.linalg.norm(candidate, axis=-2, keepdims=True)
        return candidate / norms

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Oblique point")
        step = self._point(tangent_step, "Oblique tangent step")
        target = self._point(destination, "Oblique destination")
        vector = self._point(tangent, "Oblique transported tangent")
        _same_shape(step, value, "Oblique tangent step")
        _same_shape(target, value, "Oblique destination")
        _same_shape(vector, value, "Oblique transported tangent")
        return self.project_tangent(target, vector)


class FixedRankManifold(AbstractRiemannianManifold):
    """Embedded manifold of matrices with one declared exact rank."""

    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        rows: int,
        columns: int,
        rank: int,
        /,
        *,
        tolerance: float = 1e-6,
    ):
        n, p = _matrix_dimensions(rows, columns, "Fixed-rank")
        rank_value = int(rank)
        if rank_value <= 0 or rank_value >= min(n, p):
            raise ValueError(
                "Fixed-rank rank must be positive and smaller than both dimensions."
            )
        self.rows = n
        self.columns = p
        self.rank = rank_value
        self.tolerance = _tolerance(tolerance, "Fixed-rank")
        self.manifold_id = f"manifold:fixed-rank:{n}x{p}:rank-{rank_value}"
        self.point_shape = (n, p)
        self.retraction_method = "truncated-svd"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def _factors(self, point: Array, /) -> tuple[Array, Array, Array]:
        left, singular_values, right = jnp.linalg.svd(
            point,
            full_matrices=False,
        )
        return (
            left[..., :, : self.rank],
            singular_values[..., : self.rank],
            right[..., : self.rank, :],
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Fixed-rank point")
        singular_values = jnp.linalg.svd(value, compute_uv=False)
        scale = jnp.maximum(singular_values[..., 0], 1.0)
        retained = singular_values[..., self.rank - 1] / scale
        tail = jnp.max(
            singular_values[..., self.rank :] / scale[..., None],
            axis=-1,
            initial=0.0,
        )
        deficiency = jnp.maximum(self.tolerance - retained, 0.0)
        return _finite_residual(value, jnp.max(jnp.maximum(tail, deficiency)))

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Fixed-rank point")
        singular_values = jnp.linalg.svd(value, compute_uv=False)
        scale = jnp.maximum(singular_values[..., 0], 1.0)
        retained = singular_values[..., self.rank - 1] > self.tolerance * scale
        tail = jnp.all(
            singular_values[..., self.rank :] <= self.tolerance * scale[..., None]
        )
        return jnp.all(jnp.isfinite(value)) & jnp.all(retained) & jnp.all(tail)

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Fixed-rank point")
        vector = self._point(ambient_vector, "Fixed-rank tangent")
        _same_shape(vector, value, "Fixed-rank tangent")
        left, _, right = self._factors(value)
        left_projection = left @ _transpose(left)
        right_projection = _transpose(right) @ right
        return (
            left_projection @ vector
            + vector @ right_projection
            - left_projection @ vector @ right_projection
        )

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Fixed-rank point")
        left = self._point(left_tangent, "Fixed-rank left tangent")
        right = self._point(right_tangent, "Fixed-rank right tangent")
        _same_shape(left, value, "Fixed-rank left tangent")
        _same_shape(right, value, "Fixed-rank right tangent")
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Fixed-rank point")
        step = self._point(tangent_step, "Fixed-rank tangent step")
        _same_shape(step, value, "Fixed-rank tangent step")
        left, singular_values, right = self._factors(value + step)
        return (left * singular_values[..., None, :]) @ right

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Fixed-rank point")
        step = self._point(tangent_step, "Fixed-rank tangent step")
        target = self._point(destination, "Fixed-rank destination")
        vector = self._point(tangent, "Fixed-rank transported tangent")
        _same_shape(step, value, "Fixed-rank tangent step")
        _same_shape(target, value, "Fixed-rank destination")
        _same_shape(vector, value, "Fixed-rank transported tangent")
        return self.project_tangent(target, vector)


class SpecialOrthogonalManifold(AbstractRiemannianManifold):
    """SO(n) with induced metric and delegated exponential or Cayley retraction."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    state_geometry: SpecialOrthogonalStateGeometry
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        retraction: MatrixRetraction = "exponential",
        tolerance: float = 1e-6,
    ):
        geometry = SpecialOrthogonalStateGeometry(
            dimension,
            retraction=retraction,
            tolerance=_tolerance(tolerance, "SO(n)"),
        )
        self.dimension = geometry.dimension
        self.tolerance = geometry.tolerance
        self.state_geometry = geometry
        self.manifold_id = f"manifold:so:{self.dimension}:induced:{retraction}"
        self.point_shape = (self.dimension, self.dimension)
        self.retraction_method = geometry.retraction_method
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        return self.state_geometry.contains(self._point(point, "SO(n) point"))

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "SO(n) point")
        identity = jnp.eye(self.dimension, dtype=value.dtype)
        orthogonality = jnp.max(
            jnp.abs(_transpose(value) @ value - identity), initial=0.0
        )
        orientation = jnp.max(jnp.maximum(-jnp.linalg.det(value), 0.0), initial=0.0)
        return _finite_residual(value, jnp.maximum(orthogonality, orientation))

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        return self.state_geometry.project_tangent(point, ambient_vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SO(n) point")
        left = self._point(left_tangent, "SO(n) left tangent")
        right = self._point(right_tangent, "SO(n) right tangent")
        _same_shape(left, value, "SO(n) left tangent")
        _same_shape(right, value, "SO(n) right tangent")
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SO(n) point")
        step = self._point(tangent_step, "SO(n) tangent step")
        _same_shape(step, value, "SO(n) tangent step")
        local = self.state_geometry.to_local(value, step)
        return self.state_geometry.retract(value, local)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SO(n) point")
        step = self._point(tangent_step, "SO(n) tangent step")
        target = self._point(destination, "SO(n) destination")
        vector = self._point(tangent, "SO(n) transported tangent")
        _same_shape(step, value, "SO(n) tangent step")
        _same_shape(target, value, "SO(n) destination")
        _same_shape(vector, value, "SO(n) transported tangent")
        return self.project_tangent(target, vector)


class AffineInvariantSPDManifold(AbstractGeodesicManifold):
    """SPD(n) with the affine-invariant metric and exact geodesic transport."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    state_geometry: SymmetricPositiveDefiniteStateGeometry
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-8):
        geometry = SymmetricPositiveDefiniteStateGeometry(
            dimension,
            tolerance=_tolerance(tolerance, "SPD(n)"),
        )
        self.dimension = geometry.dimension
        self.tolerance = geometry.tolerance
        self.state_geometry = geometry
        self.manifold_id = f"manifold:spd:{self.dimension}:affine-invariant"
        self.point_shape = (self.dimension, self.dimension)
        self.retraction_method = geometry.retraction_method
        self.transport_method = "affine-geodesic"
        self.transport_is_isometric = True
        self.transport_is_parallel = True

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        return self.state_geometry.contains(self._point(point, "SPD(n) point"))

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "SPD(n) point")
        symmetry = jnp.max(jnp.abs(value - _transpose(value)), initial=0.0)
        minimum = jnp.min(jnp.linalg.eigvalsh(_symmetric(value)), initial=jnp.inf)
        positivity = jnp.maximum(self.tolerance - minimum, 0.0)
        return _finite_residual(value, jnp.maximum(symmetry, positivity))

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        return self.state_geometry.project_tangent(point, ambient_vector)

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SPD(n) point")
        cotangent = self._point(ambient_cotangent, "SPD(n) cotangent")
        _same_shape(cotangent, value, "SPD(n) cotangent")
        return _symmetric(value @ _symmetric(cotangent) @ value)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SPD(n) point")
        left = self.project_tangent(value, left_tangent)
        right = self.project_tangent(value, right_tangent)
        left_solved = jnp.linalg.solve(value, left)
        right_solved = jnp.linalg.solve(value, right)
        per_point = ein.contract("...ij,...ji->...", left_solved, right_solved)
        return jnp.real(jnp.sum(per_point))

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SPD(n) point")
        step = self._point(tangent_step, "SPD(n) tangent step")
        _same_shape(step, value, "SPD(n) tangent step")
        local = self.state_geometry.to_local(value, step)
        return self.state_geometry.retract(value, local)

    def exp(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        return self.retract(point, tangent)

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        value = self._point(point, "SPD(n) point")
        target = self._point(destination, "SPD(n) destination")
        _same_shape(target, value, "SPD(n) destination")
        local = self.state_geometry.inverse_retract(value, target)
        return self.state_geometry.from_local(value, local)

    def squared_distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        value = self._point(left, "SPD(n) point")
        target = self._point(right, "SPD(n) destination")
        _same_shape(target, value, "SPD(n) destination")
        local = self.state_geometry.inverse_retract(value, target)
        return jnp.real(jnp.sum(local * local))

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "SPD(n) point")
        step = self._point(tangent_step, "SPD(n) tangent step")
        target = self._point(destination, "SPD(n) destination")
        vector = self._point(tangent, "SPD(n) transported tangent")
        _same_shape(step, value, "SPD(n) tangent step")
        _same_shape(target, value, "SPD(n) destination")
        _same_shape(vector, value, "SPD(n) transported tangent")

        factor = jnp.linalg.cholesky(_symmetric(value))
        relative = _symmetric(_inverse_congruence(factor, target))
        relative_root = _symmetric_square_root(relative)
        left_factor = factor @ relative_root
        congruence = _transpose(
            jnp.linalg.solve(_transpose(factor), _transpose(left_factor))
        )
        return _symmetric(congruence @ _symmetric(vector) @ _transpose(congruence))


__all__ = [
    "AffineInvariantSPDManifold",
    "FixedRankManifold",
    "GrassmannManifold",
    "ObliqueManifold",
    "SpecialOrthogonalManifold",
    "StiefelManifold",
]
