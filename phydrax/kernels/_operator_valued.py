#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ._base import (
    _as_input,
    _as_inputs,
    AbstractPositiveDefiniteKernel,
)
from ._compact import SphereSpectralKernel
from ._finite_feature import (
    kernel_feature_rank,
    kernel_features,
)


class AbstractOperatorValuedKernel(StrictModule):
    """Hermitian positive-definite covariance with a fixed finite output fiber."""

    @abstractmethod
    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return one output-by-output cross-covariance block."""
        raise NotImplementedError

    @abstractmethod
    def blocks(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return point-by-point output covariance blocks without flattening."""
        left_points = _as_inputs(left, input_ndim=self.input_ndim, name="left")
        right_points = _as_inputs(right, input_ndim=self.input_ndim, name="right")
        return jax.vmap(
            lambda point: jax.vmap(lambda other: self.block(point, other))(right_points)
        )(left_points)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return the flattened point-major block covariance matrix."""
        return _flatten_blocks(self.blocks(left, right))

    def diagonal(self, points: ArrayLike, /) -> Array:
        """Return the real point-major Hermitian covariance diagonal."""
        blocks = self.blocks(points, points)
        point_diagonal = jnp.diagonal(blocks, axis1=0, axis2=1)
        point_diagonal = jnp.moveaxis(point_diagonal, -1, 0)
        fiber_diagonal = jnp.real(jnp.diagonal(point_diagonal, axis1=-2, axis2=-1))
        return fiber_diagonal.reshape((-1,))

    @property
    @abstractmethod
    def input_ndim(self) -> int:
        """Number of trailing axes forming one kernel input."""
        return 1

    @property
    @abstractmethod
    def max_derivative_order(self) -> int | None:
        """Largest certified derivative order in either kernel argument."""
        return 0

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
    value = _as_input(point, input_ndim=1, name="sphere point")
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


def _output_names(values: Sequence[str], /) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("output_names must be nonempty and unique.")
    return names


class Coregionalization(StrictModule):
    """Hermitian positive-semidefinite output covariance and explicit factor."""

    weights: Array
    diagonal_scale: Array
    output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        diagonal_scale: ArrayLike,
        /,
        *,
        output_names: Sequence[str],
    ):
        names = _output_names(output_names)
        weights_ = jnp.asarray(weights)
        if not jnp.issubdtype(weights_.dtype, jnp.inexact):
            weights_ = weights_.astype(float)
        diagonal_ = jnp.asarray(diagonal_scale)
        if not jnp.issubdtype(diagonal_.dtype, jnp.inexact):
            diagonal_ = diagonal_.astype(float)
        if jnp.issubdtype(diagonal_.dtype, jnp.complexfloating):
            raise TypeError("diagonal_scale must be real.")
        if weights_.ndim != 2 or weights_.shape[0] != len(names):
            raise ValueError("weights must have shape (output, latent_rank).")
        if int(weights_.shape[1]) <= 0:
            raise ValueError("Coregionalization latent_rank must be positive.")
        if diagonal_.shape != (len(names),):
            raise ValueError("diagonal_scale must contain one value per output.")
        self.weights = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)),
            "Coregionalization weights must be finite.",
        )
        self.diagonal_scale = eqx.error_if(
            diagonal_,
            jnp.any(~jnp.isfinite(diagonal_)) | jnp.any(diagonal_ < 0.0),
            "Coregionalization diagonal scales must be finite and nonnegative.",
        )
        self.output_names = names

    @property
    def factor(self) -> Array:
        diagonal = jnp.diag(self.diagonal_scale).astype(self.weights.dtype)
        return jnp.concatenate((self.weights, diagonal), axis=1)

    @property
    def covariance(self) -> Array:
        factor = self.factor
        return factor @ jnp.conj(factor.T)

    @property
    def num_outputs(self) -> int:
        return len(self.output_names)

    @property
    def factor_rank(self) -> int:
        return int(self.factor.shape[1])

    @property
    def kernel_id(self) -> str:
        return (
            f"Coregionalization[outputs={self.num_outputs};"
            f"factor_rank={self.factor_rank}]"
        )


class IntrinsicCoregionalizationKernel(AbstractOperatorValuedKernel):
    """One scalar input kernel tensored with one output covariance."""

    spatial_kernel: AbstractPositiveDefiniteKernel
    coregionalization: Coregionalization

    def __init__(
        self,
        spatial_kernel: AbstractPositiveDefiniteKernel,
        coregionalization: Coregionalization,
        /,
    ):
        if not isinstance(spatial_kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("spatial_kernel must be a positive-definite kernel.")
        if not isinstance(coregionalization, Coregionalization):
            raise TypeError("coregionalization must be a Coregionalization.")
        self.spatial_kernel = spatial_kernel
        self.coregionalization = coregionalization

    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = _as_input(left, input_ndim=self.input_ndim, name="left")
        right_ = _as_input(right, input_ndim=self.input_ndim, name="right")
        return (
            self.spatial_kernel.pairwise(left_, right_)
            * self.coregionalization.covariance
        )

    def blocks(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        spatial = self.spatial_kernel.matrix(left, right)
        return spatial[:, :, None, None] * self.coregionalization.covariance

    @property
    def input_ndim(self) -> int:
        return self.spatial_kernel.input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        return self.spatial_kernel.max_derivative_order

    @property
    def output_dimension(self) -> int:
        return self.coregionalization.num_outputs

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.coregionalization.output_names

    @property
    def kernel_id(self) -> str:
        return (
            "IntrinsicCoregionalizationKernel["
            f"{self.spatial_kernel.kernel_id};{self.coregionalization.kernel_id}]"
        )


class LinearModelCoregionalizationKernel(AbstractOperatorValuedKernel):
    """Finite sum of scalar-input/coregionalization tensor products."""

    spatial_kernels: tuple[AbstractPositiveDefiniteKernel, ...]
    coregionalizations: tuple[Coregionalization, ...]
    _output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        components: Sequence[tuple[AbstractPositiveDefiniteKernel, Coregionalization]],
        /,
    ):
        components_ = tuple(components)
        if not components_:
            raise ValueError("LinearModelCoregionalizationKernel needs one component.")
        spatial: list[AbstractPositiveDefiniteKernel] = []
        output: list[Coregionalization] = []
        first_names: tuple[str, ...] | None = None
        first_ndim: int | None = None
        for spatial_kernel, coregionalization in components_:
            if not isinstance(spatial_kernel, AbstractPositiveDefiniteKernel):
                raise TypeError("Each spatial component must be a kernel.")
            if not isinstance(coregionalization, Coregionalization):
                raise TypeError("Each output component must be a Coregionalization.")
            if first_names is None:
                first_names = coregionalization.output_names
                first_ndim = spatial_kernel.input_ndim
            elif coregionalization.output_names != first_names:
                raise ValueError("LMC components must use identical output names.")
            elif spatial_kernel.input_ndim != first_ndim:
                raise ValueError("LMC spatial kernels must have equal input_ndim.")
            spatial.append(spatial_kernel)
            output.append(coregionalization)
        if first_names is None:
            raise RuntimeError("LMC construction produced no output vocabulary.")
        self.spatial_kernels = tuple(spatial)
        self.coregionalizations = tuple(output)
        self._output_names = first_names

    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = (
            self.spatial_kernels[0].pairwise(left, right)
            * self.coregionalizations[0].covariance
        )
        for kernel, coregionalization in zip(
            self.spatial_kernels[1:],
            self.coregionalizations[1:],
            strict=True,
        ):
            value = value + kernel.pairwise(left, right) * coregionalization.covariance
        return value

    def blocks(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = (
            self.spatial_kernels[0].matrix(left, right)[:, :, None, None]
            * self.coregionalizations[0].covariance
        )
        for kernel, coregionalization in zip(
            self.spatial_kernels[1:],
            self.coregionalizations[1:],
            strict=True,
        ):
            value = (
                value
                + kernel.matrix(left, right)[:, :, None, None]
                * coregionalization.covariance
            )
        return value

    @property
    def input_ndim(self) -> int:
        return self.spatial_kernels[0].input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        finite_orders = tuple(
            order
            for order in (kernel.max_derivative_order for kernel in self.spatial_kernels)
            if order is not None
        )
        return None if not finite_orders else min(finite_orders)

    @property
    def output_dimension(self) -> int:
        return len(self._output_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def kernel_id(self) -> str:
        components = ",".join(
            f"{kernel.kernel_id}*{coregionalization.kernel_id}"
            for kernel, coregionalization in zip(
                self.spatial_kernels,
                self.coregionalizations,
                strict=True,
            )
        )
        return f"LinearModelCoregionalizationKernel[{components}]"


def operator_kernel_feature_rank(
    kernel: AbstractOperatorValuedKernel,
    /,
) -> int | None:
    """Return the exact operator-valued feature rank when it is available."""
    if isinstance(kernel, IntrinsicCoregionalizationKernel):
        rank = kernel_feature_rank(kernel.spatial_kernel)
        return None if rank is None else int(rank) * kernel.coregionalization.factor_rank
    if isinstance(kernel, LinearModelCoregionalizationKernel):
        ranks = tuple(kernel_feature_rank(child) for child in kernel.spatial_kernels)
        if any(rank is None for rank in ranks):
            return None
        return sum(
            int(rank) * coregionalization.factor_rank
            for rank, coregionalization in zip(
                ranks,
                kernel.coregionalizations,
                strict=True,
            )
            if rank is not None
        )
    if isinstance(kernel, ProjectedTangentKernel):
        rank = kernel_feature_rank(kernel.scalar_kernel)
        return None if rank is None else int(rank) * kernel.output_dimension
    if isinstance(kernel, ProjectedDifferentialFormKernel):
        rank = kernel_feature_rank(kernel.scalar_kernel)
        return None if rank is None else int(rank) * kernel.output_dimension
    return None


def operator_kernel_features(
    kernel: AbstractOperatorValuedKernel,
    points: ArrayLike,
    /,
) -> Array:
    """Evaluate exact features with shape ``(point, fiber, rank)``."""
    rank = operator_kernel_feature_rank(kernel)
    if rank is None:
        raise TypeError(f"{kernel.kernel_id} has no exact finite-feature representation.")
    if isinstance(kernel, IntrinsicCoregionalizationKernel):
        spatial = kernel_features(kernel.spatial_kernel, points)
        features = ein.contract(
            "pr,oq->porq",
            spatial,
            kernel.coregionalization.factor,
        ).reshape((spatial.shape[0], kernel.output_dimension, rank))
    elif isinstance(kernel, LinearModelCoregionalizationKernel):
        components = []
        for spatial_kernel, coregionalization in zip(
            kernel.spatial_kernels,
            kernel.coregionalizations,
            strict=True,
        ):
            spatial = kernel_features(spatial_kernel, points)
            component_rank = int(spatial.shape[1]) * coregionalization.factor_rank
            components.append(
                ein.contract(
                    "pr,oq->porq",
                    spatial,
                    coregionalization.factor,
                ).reshape((spatial.shape[0], kernel.output_dimension, component_rank))
            )
        features = jnp.concatenate(tuple(components), axis=-1)
    elif isinstance(kernel, ProjectedTangentKernel):
        point_design = _as_inputs(
            points,
            input_ndim=kernel.input_ndim,
            name="points",
        )
        spatial = kernel_features(kernel.scalar_kernel, point_design)
        projectors = kernel._projectors(point_design)
        features = ein.contract(
            "pij,pr->pirj",
            projectors,
            spatial,
        ).reshape((spatial.shape[0], kernel.output_dimension, rank))
    elif isinstance(kernel, ProjectedDifferentialFormKernel):
        point_design = _as_inputs(
            points,
            input_ndim=kernel.input_ndim,
            name="points",
        )
        spatial = kernel_features(kernel.scalar_kernel, point_design)
        projectors = kernel._projectors(point_design)
        features = ein.contract(
            "pij,pr->pirj",
            projectors,
            spatial,
        ).reshape((spatial.shape[0], kernel.output_dimension, rank))
    else:
        raise TypeError(f"{kernel.kernel_id} has no exact finite-feature representation.")
    if features.ndim != 3 or features.shape[1:] != (
        kernel.output_dimension,
        rank,
    ):
        raise ValueError("Operator-valued features do not match their declared layout.")
    return features


class ProjectedTangentKernel(AbstractOperatorValuedKernel):
    """Intrinsic vector covariance obtained by projecting ambient latent features."""

    scalar_kernel: AbstractPositiveDefiniteKernel
    tangent_projector: Callable[[Array], Array]
    _output_dimension: int = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    projector_derivative_order: int | None = eqx.field(static=True)

    def __init__(
        self,
        scalar_kernel: AbstractPositiveDefiniteKernel,
        tangent_projector: Callable[[Array], Array],
        output_dimension: int,
        /,
        *,
        projector_id: str,
        projector_derivative_order: int | None = None,
    ):
        if not isinstance(scalar_kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("scalar_kernel must be positive definite.")
        if not callable(tangent_projector):
            raise TypeError("tangent_projector must be callable.")
        if int(output_dimension) <= 0:
            raise ValueError("output_dimension must be positive.")
        if not isinstance(projector_id, str) or not projector_id:
            raise ValueError("projector_id must be a nonempty string.")
        if projector_derivative_order is not None and int(projector_derivative_order) < 0:
            raise ValueError("projector_derivative_order must be nonnegative or None.")
        self.scalar_kernel = scalar_kernel
        self.tangent_projector = tangent_projector
        self._output_dimension = int(output_dimension)
        self.projector_id = projector_id
        self.projector_derivative_order = (
            None
            if projector_derivative_order is None
            else int(projector_derivative_order)
        )

    def _projectors(self, points: Array, /) -> Array:
        return jax.vmap(
            lambda point: _projector(
                self.tangent_projector,
                point,
                self.output_dimension,
            )
        )(points)

    def block(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_input(left, input_ndim=self.input_ndim, name="left")
        right_point = _as_input(right, input_ndim=self.input_ndim, name="right")
        left_projector = _projector(
            self.tangent_projector, left_point, self.output_dimension
        )
        right_projector = _projector(
            self.tangent_projector, right_point, self.output_dimension
        )
        return (
            self.scalar_kernel.pairwise(left_point, right_point)
            * left_projector
            @ jnp.conj(right_projector.T)
        )

    def blocks(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _as_inputs(left, input_ndim=self.input_ndim, name="left")
        right_points = _as_inputs(right, input_ndim=self.input_ndim, name="right")
        left_projectors = self._projectors(left_points)
        right_projectors = self._projectors(right_points)
        projector_blocks = ein.contract(
            "aij,bkj->abik",
            left_projectors,
            jnp.conj(right_projectors),
        )
        scalar = self.scalar_kernel.matrix(left_points, right_points)
        return scalar[:, :, None, None] * projector_blocks

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    @property
    def input_ndim(self) -> int:
        return self.scalar_kernel.input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        kernel_order = self.scalar_kernel.max_derivative_order
        projector_order = self.projector_derivative_order
        if kernel_order is None:
            return projector_order
        if projector_order is None:
            return kernel_order
        return min(kernel_order, projector_order)

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
    projector_derivative_order: int | None = eqx.field(static=True)

    def __init__(
        self,
        scalar_kernel: AbstractPositiveDefiniteKernel,
        tangent_projector: Callable[[Array], Array],
        ambient_dimension: int,
        degree: int,
        /,
        *,
        projector_id: str,
        projector_derivative_order: int | None = None,
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
        if projector_derivative_order is not None and int(projector_derivative_order) < 0:
            raise ValueError("projector_derivative_order must be nonnegative or None.")
        self.scalar_kernel = scalar_kernel
        self.tangent_projector = tangent_projector
        self.ambient_dimension = ambient
        self.degree = resolved_degree
        self.multi_indices = tuple(
            itertools.combinations(range(ambient), resolved_degree)
        )
        self.projector_id = projector_id
        self.projector_derivative_order = (
            None
            if projector_derivative_order is None
            else int(projector_derivative_order)
        )

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
        left_point = _as_input(left, input_ndim=self.input_ndim, name="left")
        right_point = _as_input(right, input_ndim=self.input_ndim, name="right")
        left_projector = self._form_projector(left_point)
        right_projector = self._form_projector(right_point)
        return (
            self.scalar_kernel.pairwise(left_point, right_point)
            * left_projector
            @ jnp.conj(right_projector.T)
        )

    def blocks(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _as_inputs(left, input_ndim=self.input_ndim, name="left")
        right_points = _as_inputs(right, input_ndim=self.input_ndim, name="right")
        left_projectors = self._projectors(left_points)
        right_projectors = self._projectors(right_points)
        projector_blocks = ein.contract(
            "aij,bkj->abik",
            left_projectors,
            jnp.conj(right_projectors),
        )
        scalar = self.scalar_kernel.matrix(left_points, right_points)
        return scalar[:, :, None, None] * projector_blocks

    @property
    def output_dimension(self) -> int:
        return math.comb(self.ambient_dimension, self.degree)

    @property
    def input_ndim(self) -> int:
        return self.scalar_kernel.input_ndim

    @property
    def max_derivative_order(self) -> int | None:
        kernel_order = self.scalar_kernel.max_derivative_order
        projector_order = self.projector_derivative_order
        if kernel_order is None:
            return projector_order
        if projector_order is None:
            return kernel_order
        return min(kernel_order, projector_order)

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
    "Coregionalization",
    "IntrinsicCoregionalizationKernel",
    "LinearModelCoregionalizationKernel",
    "ProjectedDifferentialFormKernel",
    "ProjectedTangentKernel",
    "operator_kernel_feature_rank",
    "operator_kernel_features",
    "sphere_differential_form_kernel",
    "sphere_tangent_kernel",
    "sphere_tangent_projector",
]
