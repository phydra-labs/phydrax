#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import ParameterOwner
from ..discretization import SpectralDecomposition
from ..typing import checked
from ._base import _as_point, _as_points
from ._finite_feature import AbstractFiniteFeatureKernel


class AbstractSpectralMultiplier(StrictModule, ParameterOwner):
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
    array = jnp.asarray(value, dtype=jnp.float64)
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

    def __init__(self, diffusion_time: ArrayLike, /) -> None:
        value = jnp.asarray(diffusion_time, dtype=jnp.float64)
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

    def __init__(self, length_scale: ArrayLike, smoothness: ArrayLike, /) -> None:
        self.length_scale = _positive_scalar(length_scale, "length_scale")
        self.smoothness = _positive_scalar(smoothness, "smoothness")

    def log_weights(
        self,
        eigenvalues: ArrayLike,
        spectral_dimension: float,
        /,
    ) -> Array:
        values = jnp.asarray(eigenvalues, dtype=jnp.float64)
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


@final
class SpectralFeatureKernel(AbstractFiniteFeatureKernel):
    """Finite Laplacian covariance over explicit integer coefficient identifiers."""

    eigenbasis: SpectralDecomposition
    multiplier: AbstractSpectralMultiplier
    normalize: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        eigenbasis: SpectralDecomposition,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
    ) -> None:
        if eigenbasis.report is None or eigenbasis.spectral_dimension is None:
            raise ValueError(
                "SpectralFeatureKernel requires Laplacian provenance and dimension."
            )
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

    def _coefficient_indices(self, points: ArrayLike, /) -> Array:
        design = _as_points(points, name="points")
        if design.shape[1] != 1:
            raise ValueError("Spectral coefficient inputs must have one coordinate.")
        coefficient_ids = design[:, 0]
        lower = self.eigenbasis.index_offset
        upper = lower + self.eigenbasis.num_points
        coefficient_ids = eqx.error_if(
            coefficient_ids,
            jnp.any(~jnp.isfinite(coefficient_ids))
            | jnp.any(coefficient_ids != jnp.floor(coefficient_ids))
            | jnp.any(coefficient_ids < lower)
            | jnp.any(coefficient_ids >= upper),
            "Spectral coefficient IDs must be finite in-range integers.",
        )
        return coefficient_ids.astype(jnp.int32) - lower

    def features(self, points: ArrayLike, /) -> Array:
        indices = self._coefficient_indices(points)
        eigenfunctions = self.eigenbasis.eigenfunctions[indices]
        weights = jnp.asarray(self._sqrt_weights(), dtype=eigenfunctions.dtype)
        return eigenfunctions * weights[None, :]

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_point(left, name="left")
        right_point = _as_point(right, name="right")
        if left_point.shape != (1,) or right_point.shape != (1,):
            raise ValueError(
                "pairwise requires one spectral coefficient ID per argument."
            )
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
