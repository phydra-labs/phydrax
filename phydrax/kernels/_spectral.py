#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..discretization import SpectralDecomposition
from ._base import _as_point, _as_points
from ._finite_feature import AbstractFiniteFeatureKernel


class AbstractSpectralMultiplier(StrictModule):
    """Nonnegative Laplacian covariance law evaluated in log space."""

    @abstractmethod
    def log_weights(
        self,
        eigenvalues: ArrayLike,
        spectral_dimension: float,
        /,
    ) -> Array:
        """Return one finite or negative-infinite log weight per eigenvalue."""
        raise NotImplementedError

    @property
    @abstractmethod
    def multiplier_id(self) -> str:
        """Return stable method provenance."""
        raise NotImplementedError


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        array,
        ~jnp.isfinite(array) | (array <= 0.0),
        f"{name} must be finite and strictly positive.",
    )


class HeatSpectralMultiplier(AbstractSpectralMultiplier):
    """Heat covariance law with weights ``exp(-diffusion_time * eigenvalue)``."""

    diffusion_time: Array

    def __init__(self, diffusion_time: ArrayLike, /):
        value = jnp.asarray(diffusion_time, dtype=float)
        if value.ndim != 0:
            raise ValueError("diffusion_time must be scalar.")
        self.diffusion_time = eqx.error_if(
            value,
            ~jnp.isfinite(value) | (value < 0.0),
            "diffusion_time must be finite and nonnegative.",
        )

    def log_weights(
        self,
        eigenvalues: ArrayLike,
        spectral_dimension: float,
        /,
    ) -> Array:
        del spectral_dimension
        values = jnp.asarray(eigenvalues)
        return -self.diffusion_time * values

    @property
    def multiplier_id(self) -> str:
        return "heat"


class MaternSpectralMultiplier(AbstractSpectralMultiplier):
    """Shifted-power Matérn covariance law relative to its zero mode."""

    length_scale: Array
    smoothness: Array

    def __init__(self, length_scale: ArrayLike, smoothness: ArrayLike, /):
        self.length_scale = _positive_scalar(length_scale, "length_scale")
        self.smoothness = _positive_scalar(smoothness, "smoothness")

    def log_weights(
        self,
        eigenvalues: ArrayLike,
        spectral_dimension: float,
        /,
    ) -> Array:
        values = jnp.asarray(eigenvalues, dtype=float)
        dimension = jnp.asarray(spectral_dimension, dtype=values.dtype)
        safe_values = jnp.where(values > 0.0, values, 1.0)
        log_ratio = (
            2.0 * jnp.log(self.length_scale)
            + jnp.log(safe_values)
            - jnp.log(2.0 * self.smoothness)
        )
        log_ratio = jnp.where(values > 0.0, log_ratio, -jnp.inf)
        return -(self.smoothness + 0.5 * dimension) * jnp.logaddexp(0.0, log_ratio)

    @property
    def multiplier_id(self) -> str:
        return "matern"


class SpectralFeatureKernel(AbstractFiniteFeatureKernel):
    """Finite Laplacian spectral covariance over integer entity identifiers."""

    eigenbasis: SpectralDecomposition
    multiplier: AbstractSpectralMultiplier
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        eigenbasis: SpectralDecomposition,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
    ):
        if not isinstance(eigenbasis, SpectralDecomposition):
            raise TypeError(
                "eigenbasis must be a Laplacian-provenance SpectralDecomposition."
            )
        if eigenbasis.report is None or eigenbasis.spectral_dimension is None:
            raise ValueError(
                "SpectralFeatureKernel requires Laplacian provenance and dimension."
            )
        if not isinstance(multiplier, AbstractSpectralMultiplier):
            raise TypeError("multiplier must be an AbstractSpectralMultiplier.")
        self.eigenbasis = eigenbasis
        self.multiplier = multiplier
        self.normalize = bool(normalize)

    def _sqrt_weights(self) -> Array:
        dimension = self.eigenbasis.spectral_dimension
        if dimension is None:
            raise RuntimeError("Validated Laplacian basis lost its spectral dimension.")
        log_weights = self.multiplier.log_weights(
            self.eigenbasis.eigenvalues,
            dimension,
        )
        if log_weights.shape != self.eigenbasis.eigenvalues.shape:
            raise ValueError("Spectral multiplier output must match the eigenvalues.")
        log_weights = eqx.error_if(
            log_weights,
            jnp.any(jnp.isnan(log_weights)) | jnp.any(log_weights == jnp.inf),
            "Spectral log weights must be finite or negative infinity.",
        )
        if self.normalize:
            maximum = jnp.max(log_weights)
            log_weights = eqx.error_if(
                log_weights,
                ~jnp.isfinite(maximum),
                "Normalized spectral weights cannot all be zero.",
            )
            log_weights = log_weights - (
                maximum + jnp.log(jnp.sum(jnp.exp(log_weights - maximum)))
            )
        return jnp.exp(0.5 * log_weights)

    def _entity_indices(self, points: ArrayLike, /) -> Array:
        design = _as_points(points, name="points")
        if int(design.shape[1]) != 1:
            raise ValueError("Spectral entity inputs must have one coordinate.")
        entity_ids = design[:, 0]
        lower = self.eigenbasis.index_offset
        upper = lower + self.eigenbasis.entity_count
        entity_ids = eqx.error_if(
            entity_ids,
            jnp.any(~jnp.isfinite(entity_ids))
            | jnp.any(entity_ids != jnp.floor(entity_ids))
            | jnp.any(entity_ids < lower)
            | jnp.any(entity_ids >= upper),
            "Spectral entity IDs must be finite in-range integers.",
        )
        return entity_ids.astype(jnp.int32) - lower

    def features(self, points: ArrayLike, /) -> Array:
        indices = self._entity_indices(points)
        return self.eigenbasis.eigenfunctions[indices] * self._sqrt_weights()

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_point(left, name="left")
        right_point = _as_point(right, name="right")
        if left_point.shape != (1,) or right_point.shape != (1,):
            raise ValueError("pairwise requires one spectral entity ID per argument.")
        left_feature = self.features(left_point)[0]
        right_feature = self.features(right_point)[0]
        return jnp.dot(left_feature, right_feature)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_features = self.features(left)
        right_features = self.features(right)
        return left_features @ right_features.T

    def diagonal(self, points: ArrayLike, /) -> Array:
        features = self.features(points)
        return jnp.sum(features * features, axis=-1)

    @property
    def feature_rank(self) -> int:
        return self.eigenbasis.mode_count

    @property
    def max_derivative_order(self) -> int:
        return 0

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return (
            f"SpectralFeatureKernel[{self.eigenbasis.decomposition_id};"
            f"{self.multiplier.multiplier_id};normalize={int(self.normalize)}]"
        )


__all__ = [
    "AbstractSpectralMultiplier",
    "HeatSpectralMultiplier",
    "MaternSpectralMultiplier",
    "SpectralFeatureKernel",
]
