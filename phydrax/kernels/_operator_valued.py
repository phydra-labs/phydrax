#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from abc import abstractmethod
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._base import _as_point, _as_points, AbstractPositiveDefiniteKernel
from ._compact import SphereSpectralKernel


class AbstractOperatorValuedKernel(StrictModule):
    """Positive-definite covariance with a fixed finite output fiber."""

    @abstractmethod
    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return one output-by-output cross-covariance block."""
        raise NotImplementedError

    @abstractmethod
    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return the flattened block covariance matrix."""
        raise NotImplementedError

    @abstractmethod
    def diagonal(self, points: ArrayLike, /) -> Array:
        """Return the flattened scalar diagonal over points and fiber channels."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def kernel_id(self) -> str:
        raise NotImplementedError


def sphere_tangent_projector(
    point: ArrayLike,
    /,
    *,
    membership_tolerance: float = 1e-6,
) -> Array:
    """Return the Euclidean orthogonal projector onto a unit sphere tangent space."""
    tolerance = float(membership_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("membership_tolerance must lie strictly between zero and one.")
    value = _as_point(point, name="sphere point")
    squared_norm = jnp.sum(value * value)
    value = eqx.error_if(
        value,
        jnp.abs(squared_norm - 1.0) > tolerance,
        "sphere_tangent_projector requires a unit vector.",
    )
    value = value / jnp.sqrt(squared_norm)
    return jnp.eye(value.shape[0], dtype=value.dtype) - jnp.outer(value, value)


def _projector(
    projector: Callable[[Array], Array],
    point: Array,
    output_dimension: int,
    /,
) -> Array:
    value = jnp.asarray(projector(point))
    if value.shape != (output_dimension, output_dimension):
        raise ValueError(
            "tangent_projector output must match the declared output dimension."
        )
    return eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)),
        "tangent_projector must return finite matrices.",
    )


def _flatten_blocks(blocks: Array, /) -> Array:
    return jnp.transpose(blocks, (0, 2, 1, 3)).reshape(
        (blocks.shape[0] * blocks.shape[2], blocks.shape[1] * blocks.shape[3])
    )


class ProjectedTangentKernel(AbstractOperatorValuedKernel):
    """Intrinsic vector covariance obtained by projecting ambient latent features."""

    scalar_kernel: AbstractPositiveDefiniteKernel
    tangent_projector: Callable[[Array], Array]
    _output_dimension: int = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)

    def __init__(
        self,
        scalar_kernel: AbstractPositiveDefiniteKernel,
        tangent_projector: Callable[[Array], Array],
        output_dimension: int,
        /,
        *,
        projector_id: str,
    ):
        if not isinstance(scalar_kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("scalar_kernel must be positive definite.")
        if not callable(tangent_projector):
            raise TypeError("tangent_projector must be callable.")
        if int(output_dimension) <= 0:
            raise ValueError("output_dimension must be positive.")
        if not isinstance(projector_id, str) or not projector_id:
            raise ValueError("projector_id must be a nonempty string.")
        self.scalar_kernel = scalar_kernel
        self.tangent_projector = tangent_projector
        self._output_dimension = int(output_dimension)
        self.projector_id = projector_id

    def _projectors(self, points: Array, /) -> Array:
        return jax.vmap(
            lambda point: _projector(
                self.tangent_projector,
                point,
                self.output_dimension,
            )
        )(points)

    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_point(left, name="left")
        right_point = _as_point(right, name="right")
        left_projector = _projector(
            self.tangent_projector, left_point, self.output_dimension
        )
        right_projector = _projector(
            self.tangent_projector, right_point, self.output_dimension
        )
        return (
            self.scalar_kernel.pairwise(left_point, right_point)
            * left_projector
            @ right_projector.T
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _as_points(left, name="left")
        right_points = _as_points(right, name="right")
        left_projectors = self._projectors(left_points)
        right_projectors = self._projectors(right_points)
        projector_blocks = oe.contract("aij,bkj->abik", left_projectors, right_projectors)
        scalar = self.scalar_kernel.matrix(left_points, right_points)
        return _flatten_blocks(scalar[:, :, None, None] * projector_blocks)

    def diagonal(self, points: ArrayLike, /) -> Array:
        point_design = _as_points(points, name="points")
        projectors = self._projectors(point_design)
        fiber_diagonal = jnp.sum(projectors * projectors, axis=-1)
        scalar_diagonal = self.scalar_kernel.diagonal(point_design)
        return (scalar_diagonal[:, None] * fiber_diagonal).reshape((-1,))

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    @property
    def kernel_id(self) -> str:
        return (
            f"ProjectedTangentKernel[{self.scalar_kernel.kernel_id};{self.projector_id}]"
        )


def sphere_tangent_kernel(
    scalar_kernel: SphereSpectralKernel,
    /,
) -> ProjectedTangentKernel:
    """Lift a scalar sphere kernel to a tangent-vector covariance."""
    if not isinstance(scalar_kernel, SphereSpectralKernel):
        raise TypeError("sphere_tangent_kernel requires a SphereSpectralKernel.")
    ambient = scalar_kernel.spectrum.dimension + 1
    return ProjectedTangentKernel(
        scalar_kernel,
        partial(
            sphere_tangent_projector,
            membership_tolerance=scalar_kernel.membership_tolerance,
        ),
        ambient,
        projector_id=f"sphere-S{scalar_kernel.spectrum.dimension}",
    )


def _exterior_power_matrix(
    matrix: Array,
    multi_indices: tuple[tuple[int, ...], ...],
    /,
) -> Array:
    rows = []
    for row_index in multi_indices:
        columns = []
        row = jnp.asarray(row_index, dtype=jnp.int32)
        for column_index in multi_indices:
            column = jnp.asarray(column_index, dtype=jnp.int32)
            minor = matrix[row[:, None], column[None, :]]
            columns.append(jnp.linalg.det(minor))
        rows.append(jnp.stack(columns))
    return jnp.stack(rows)


class ProjectedDifferentialFormKernel(AbstractOperatorValuedKernel):
    """Exterior-power covariance for intrinsic differential forms."""

    scalar_kernel: AbstractPositiveDefiniteKernel
    tangent_projector: Callable[[Array], Array]
    ambient_dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    multi_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)

    def __init__(
        self,
        scalar_kernel: AbstractPositiveDefiniteKernel,
        tangent_projector: Callable[[Array], Array],
        ambient_dimension: int,
        degree: int,
        /,
        *,
        projector_id: str,
    ):
        if not isinstance(scalar_kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("scalar_kernel must be positive definite.")
        if not callable(tangent_projector):
            raise TypeError("tangent_projector must be callable.")
        ambient = int(ambient_dimension)
        resolved_degree = int(degree)
        if ambient <= 0 or resolved_degree <= 0 or resolved_degree > ambient:
            raise ValueError("Form degree must lie in [1, ambient_dimension].")
        if not isinstance(projector_id, str) or not projector_id:
            raise ValueError("projector_id must be a nonempty string.")
        self.scalar_kernel = scalar_kernel
        self.tangent_projector = tangent_projector
        self.ambient_dimension = ambient
        self.degree = resolved_degree
        self.multi_indices = tuple(
            itertools.combinations(range(ambient), resolved_degree)
        )
        self.projector_id = projector_id

    def _form_projector(self, point: Array, /) -> Array:
        tangent = _projector(
            self.tangent_projector,
            point,
            self.ambient_dimension,
        )
        return _exterior_power_matrix(tangent, self.multi_indices)

    def _projectors(self, points: Array, /) -> Array:
        return jax.vmap(self._form_projector)(points)

    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_point(left, name="left")
        right_point = _as_point(right, name="right")
        left_projector = self._form_projector(left_point)
        right_projector = self._form_projector(right_point)
        return (
            self.scalar_kernel.pairwise(left_point, right_point)
            * left_projector
            @ right_projector.T
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _as_points(left, name="left")
        right_points = _as_points(right, name="right")
        left_projectors = self._projectors(left_points)
        right_projectors = self._projectors(right_points)
        projector_blocks = oe.contract("aij,bkj->abik", left_projectors, right_projectors)
        scalar = self.scalar_kernel.matrix(left_points, right_points)
        return _flatten_blocks(scalar[:, :, None, None] * projector_blocks)

    def diagonal(self, points: ArrayLike, /) -> Array:
        point_design = _as_points(points, name="points")
        projectors = self._projectors(point_design)
        fiber_diagonal = jnp.sum(projectors * projectors, axis=-1)
        scalar_diagonal = self.scalar_kernel.diagonal(point_design)
        return (scalar_diagonal[:, None] * fiber_diagonal).reshape((-1,))

    @property
    def output_dimension(self) -> int:
        return math.comb(self.ambient_dimension, self.degree)

    @property
    def kernel_id(self) -> str:
        return (
            f"ProjectedDifferentialFormKernel[{self.scalar_kernel.kernel_id};"
            f"degree={self.degree};{self.projector_id}]"
        )


def sphere_differential_form_kernel(
    scalar_kernel: SphereSpectralKernel,
    degree: int,
    /,
) -> ProjectedDifferentialFormKernel:
    """Lift a scalar sphere kernel to an intrinsic ambient-coordinate form covariance."""
    if not isinstance(scalar_kernel, SphereSpectralKernel):
        raise TypeError(
            "sphere_differential_form_kernel requires a SphereSpectralKernel."
        )
    ambient = scalar_kernel.spectrum.dimension + 1
    if int(degree) > scalar_kernel.spectrum.dimension:
        raise ValueError("Sphere form degree cannot exceed the intrinsic dimension.")
    return ProjectedDifferentialFormKernel(
        scalar_kernel,
        partial(
            sphere_tangent_projector,
            membership_tolerance=scalar_kernel.membership_tolerance,
        ),
        ambient,
        degree,
        projector_id=f"sphere-S{scalar_kernel.spectrum.dimension}",
    )


__all__ = [
    "AbstractOperatorValuedKernel",
    "ProjectedDifferentialFormKernel",
    "ProjectedTangentKernel",
    "sphere_differential_form_kernel",
    "sphere_tangent_kernel",
    "sphere_tangent_projector",
]
