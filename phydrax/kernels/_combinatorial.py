#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import AbstractPositiveDefiniteKernel
from ._spectral import AbstractSpectralMultiplier


def _hamming_points(
    points: ArrayLike,
    dimension: int,
    alphabet_size: int,
    /,
) -> Array:
    array = jnp.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or int(array.shape[1]) != dimension:
        raise ValueError(
            f"Hamming points must have shape ({dimension},) or (point, {dimension})."
        )
    array = eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array))
        | jnp.any(array != jnp.floor(array))
        | jnp.any(array < 0.0)
        | jnp.any(array >= alphabet_size),
        "Hamming coordinates must be finite in-range integers.",
    )
    return array.astype(jnp.int32)


def _stable_level_coefficients(
    multiplier: AbstractSpectralMultiplier,
    eigenvalues: Array,
    spectral_dimension: float,
    log_multiplicities: Array,
    /,
    *,
    normalize: bool,
) -> Array:
    log_weights = multiplier.log_weights(eigenvalues, spectral_dimension)
    if log_weights.shape != eigenvalues.shape:
        raise ValueError("Spectral multiplier output must match Hamming levels.")
    log_coefficients = log_weights + log_multiplicities
    log_coefficients = eqx.error_if(
        log_coefficients,
        jnp.any(jnp.isnan(log_coefficients)) | jnp.any(log_coefficients == jnp.inf),
        "Hamming spectral log coefficients must be finite or negative infinity.",
    )
    if normalize:
        maximum = jnp.max(log_coefficients)
        log_coefficients = eqx.error_if(
            log_coefficients,
            ~jnp.isfinite(maximum),
            "Normalized Hamming spectral coefficients cannot all vanish.",
        )
        log_coefficients = log_coefficients - (
            maximum + jnp.log(jnp.sum(jnp.exp(log_coefficients - maximum)))
        )
    return jnp.exp(log_coefficients)


def _krawtchouk_series(
    distance: Array,
    coefficients: Array,
    dimension: int,
    alphabet_size: int,
    max_level: int,
    /,
) -> Array:
    value = coefficients[0] * jnp.ones_like(distance, dtype=float)
    if max_level == 0:
        return value
    previous = jnp.ones_like(distance, dtype=float)
    current = 1.0 - (alphabet_size * distance / (dimension * (alphabet_size - 1)))
    value = value + coefficients[1] * current
    for level in range(1, max_level):
        denominator = (dimension - level) * (alphabet_size - 1)
        following = (
            1.0 + (level - alphabet_size * distance) / denominator
        ) * current - level / denominator * previous
        value = value + coefficients[level + 1] * following
        previous, current = current, following
    return value


class HammingSpectralKernel(AbstractPositiveDefiniteKernel):
    """Analytic Laplacian spectral kernel on the Hamming scheme H(d, q)."""

    multiplier: AbstractSpectralMultiplier
    dimension: int = eqx.field(static=True)
    alphabet_size: int = eqx.field(static=True)
    max_level: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        alphabet_size: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        max_level: int | None = None,
        normalize: bool = True,
    ):
        resolved_dimension = int(dimension)
        resolved_alphabet = int(alphabet_size)
        if resolved_dimension <= 0:
            raise ValueError("Hamming dimension must be positive.")
        if resolved_alphabet < 2:
            raise ValueError("Hamming alphabet_size must be at least two.")
        if not isinstance(multiplier, AbstractSpectralMultiplier):
            raise TypeError("multiplier must be an AbstractSpectralMultiplier.")
        resolved_level = resolved_dimension if max_level is None else int(max_level)
        if resolved_level < 0 or resolved_level > resolved_dimension:
            raise ValueError("max_level must lie between zero and the Hamming dimension.")
        self.multiplier = multiplier
        self.dimension = resolved_dimension
        self.alphabet_size = resolved_alphabet
        self.max_level = resolved_level
        self.normalize = bool(normalize)

    def _coefficients(self) -> Array:
        levels = tuple(range(self.max_level + 1))
        eigenvalues = jnp.asarray(
            [self.alphabet_size * level for level in levels], dtype=float
        )
        log_multiplicities = jnp.asarray(
            [
                math.log(math.comb(self.dimension, level))
                + level * math.log(self.alphabet_size - 1)
                for level in levels
            ],
            dtype=float,
        )
        return _stable_level_coefficients(
            self.multiplier,
            eigenvalues,
            float(self.dimension),
            log_multiplicities,
            normalize=self.normalize,
        )

    def _distances(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _hamming_points(left, self.dimension, self.alphabet_size)
        right_points = _hamming_points(right, self.dimension, self.alphabet_size)
        return jnp.sum(left_points[:, None, :] != right_points[None, :, :], axis=-1)

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        distances = self._distances(left, right)
        if distances.shape != (1, 1):
            raise ValueError("pairwise requires one Hamming point per argument.")
        return _krawtchouk_series(
            distances[0, 0],
            self._coefficients(),
            self.dimension,
            self.alphabet_size,
            self.max_level,
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return _krawtchouk_series(
            self._distances(left, right),
            self._coefficients(),
            self.dimension,
            self.alphabet_size,
            self.max_level,
        )

    def diagonal(self, points: ArrayLike, /) -> Array:
        point_design = _hamming_points(points, self.dimension, self.alphabet_size)
        return jnp.full(
            (point_design.shape[0],),
            jnp.sum(self._coefficients()),
            dtype=float,
        )

    @property
    def max_derivative_order(self) -> int:
        return 0

    @property
    def is_unit_diagonal(self) -> bool:
        return self.normalize

    @property
    def kernel_id(self) -> str:
        return (
            f"HammingSpectralKernel[H({self.dimension},{self.alphabet_size});"
            f"levels={self.max_level};{self.multiplier.multiplier_id};"
            f"normalize={int(self.normalize)}]"
        )


class HypercubeSpectralKernel(AbstractPositiveDefiniteKernel):
    """Analytic Hamming specialization for the binary hypercube."""

    hamming_kernel: HammingSpectralKernel

    def __init__(
        self,
        dimension: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        max_level: int | None = None,
        normalize: bool = True,
    ):
        self.hamming_kernel = HammingSpectralKernel(
            dimension,
            2,
            multiplier,
            max_level=max_level,
            normalize=normalize,
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.hamming_kernel.pairwise(left, right)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.hamming_kernel.matrix(left, right)

    def diagonal(self, points: ArrayLike, /) -> Array:
        return self.hamming_kernel.diagonal(points)

    @property
    def max_derivative_order(self) -> int:
        return 0

    @property
    def is_unit_diagonal(self) -> bool:
        return self.hamming_kernel.is_unit_diagonal

    @property
    def kernel_id(self) -> str:
        return (
            f"HypercubeSpectralKernel[Q{self.hamming_kernel.dimension};"
            f"levels={self.hamming_kernel.max_level};"
            f"{self.hamming_kernel.multiplier.multiplier_id};"
            f"normalize={int(self.hamming_kernel.normalize)}]"
        )


__all__ = ["HammingSpectralKernel", "HypercubeSpectralKernel"]
