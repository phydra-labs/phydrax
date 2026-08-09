#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ._base import _as_point, _as_points, AbstractPositiveDefiniteKernel


class AbstractFiniteFeatureKernel(AbstractPositiveDefiniteKernel):
    """Positive-definite kernel with an explicit finite real feature map."""

    @abstractmethod
    def features(self, points: ArrayLike, /) -> Array:
        """Evaluate whitened features with shape ``(point, rank)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_rank(self) -> int:
        """Return the static number of whitened features."""
        raise NotImplementedError


class FiniteFeatureKernel(AbstractFiniteFeatureKernel):
    """Finite-rank kernel represented by whitened real feature vectors."""

    feature_map: Callable[[Array], Array]
    feature_factor: Array
    feature_map_id: str = eqx.field(static=True)
    feature_derivative_order: int | None = eqx.field(static=True)

    def __init__(
        self,
        feature_map: Callable[[Array], Array],
        feature_factor: ArrayLike,
        /,
        *,
        feature_map_id: str,
        max_derivative_order: int | None = 0,
    ):
        if not callable(feature_map):
            raise TypeError("feature_map must be callable.")
        if not isinstance(feature_map_id, str) or not feature_map_id:
            raise ValueError("feature_map_id must be a nonempty string.")
        if max_derivative_order is not None and int(max_derivative_order) < 0:
            raise ValueError("max_derivative_order must be nonnegative or None.")
        factor = jnp.asarray(feature_factor, dtype=float)
        if factor.ndim != 2 or factor.shape[0] == 0 or factor.shape[1] == 0:
            raise ValueError("feature_factor must have shape (feature, rank).")
        self.feature_map = feature_map
        self.feature_factor = eqx.error_if(
            factor,
            jnp.any(~jnp.isfinite(factor)),
            "feature_factor must contain only finite values.",
        )
        self.feature_map_id = feature_map_id
        self.feature_derivative_order = (
            None if max_derivative_order is None else int(max_derivative_order)
        )

    @classmethod
    def from_precision_cholesky(
        cls,
        feature_map: Callable[[Array], Array],
        precision_cholesky: ArrayLike,
        /,
        *,
        lower: bool,
        feature_map_id: str,
        max_derivative_order: int | None = 0,
    ) -> FiniteFeatureKernel:
        cholesky = jnp.asarray(precision_cholesky, dtype=float)
        if (
            cholesky.ndim != 2
            or cholesky.shape[0] == 0
            or cholesky.shape[0] != cholesky.shape[1]
        ):
            raise ValueError("precision_cholesky must be a nonempty square matrix.")
        cholesky = eqx.error_if(
            cholesky,
            jnp.any(~jnp.isfinite(cholesky)) | jnp.any(jnp.diag(cholesky) <= 0.0),
            "precision_cholesky must be finite with a positive diagonal.",
        )
        identity = jnp.eye(cholesky.shape[0], dtype=cholesky.dtype)
        inverse_triangular = jsp.linalg.solve_triangular(
            cholesky,
            identity,
            lower=bool(lower),
        )
        factor = inverse_triangular.T if lower else inverse_triangular
        return cls(
            feature_map,
            factor,
            feature_map_id=feature_map_id,
            max_derivative_order=max_derivative_order,
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_feature = self._whitened_feature(_as_point(left, name="left"))
        right_feature = self._whitened_feature(_as_point(right, name="right"))
        return jnp.dot(left_feature, right_feature)

    def features(self, points: ArrayLike, /) -> Array:
        """Evaluate whitened finite features over a point design."""
        point_design = _as_points(points, name="points")
        return jax.vmap(self._whitened_feature)(point_design)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.features(left) @ self.features(right).T

    def diagonal(self, points: ArrayLike, /) -> Array:
        features = self.features(points)
        return jnp.sum(features * features, axis=-1)

    def _whitened_feature(self, point: Array, /) -> Array:
        feature = _as_point(self.feature_map(point), name="feature_map output")
        if feature.shape[0] != self.feature_factor.shape[0]:
            raise ValueError("feature_map output size must match feature_factor rows.")
        return feature @ self.feature_factor

    @property
    def feature_rank(self) -> int:
        return int(self.feature_factor.shape[1])

    @property
    def max_derivative_order(self) -> int | None:
        return self.feature_derivative_order

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"FiniteFeatureKernel[{self.feature_map_id}]"


def kernel_feature_rank(kernel: AbstractPositiveDefiniteKernel, /) -> int | None:
    """Return the exact composed feature rank, or ``None`` when unavailable."""
    from ._algebra import AmplitudeKernel, SumKernel
    from ._transforms import InputTransformedKernel

    if isinstance(kernel, AbstractFiniteFeatureKernel):
        return kernel.feature_rank
    if isinstance(kernel, (AmplitudeKernel, InputTransformedKernel)):
        return kernel_feature_rank(kernel.kernel)
    if isinstance(kernel, SumKernel):
        ranks = tuple(kernel_feature_rank(child) for child in kernel.kernels)
        if any(rank is None for rank in ranks):
            return None
        return sum(int(rank) for rank in ranks if rank is not None)
    return None


def kernel_features(
    kernel: AbstractPositiveDefiniteKernel,
    points: ArrayLike,
    /,
) -> Array:
    """Evaluate exact composed features or reject an unsupported kernel tree."""
    from ._algebra import AmplitudeKernel, SumKernel
    from ._transforms import InputTransformedKernel

    rank = kernel_feature_rank(kernel)
    if rank is None:
        raise TypeError(f"{kernel.kernel_id} has no exact finite-feature representation.")
    if isinstance(kernel, AbstractFiniteFeatureKernel):
        features = kernel.features(points)
    elif isinstance(kernel, AmplitudeKernel):
        features = kernel.amplitude * kernel_features(kernel.kernel, points)
    elif isinstance(kernel, SumKernel):
        features = jnp.concatenate(
            tuple(kernel_features(child, points) for child in kernel.kernels),
            axis=-1,
        )
    elif isinstance(kernel, InputTransformedKernel):
        point_design = _as_points(points, name="points")
        transformed = jax.vmap(kernel._transform_point)(point_design)
        features = kernel_features(kernel.kernel, transformed)
    else:
        raise TypeError(f"{kernel.kernel_id} has no exact finite-feature representation.")
    if features.ndim != 2 or int(features.shape[1]) != rank:
        raise ValueError("Composed kernel features do not match their declared rank.")
    return features


__all__ = [
    "AbstractFiniteFeatureKernel",
    "FiniteFeatureKernel",
    "kernel_feature_rank",
    "kernel_features",
]
