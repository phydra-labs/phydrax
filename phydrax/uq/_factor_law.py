#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._probability import AbstractProbabilityLaw
from ..domain._measure import MeasureKind
from ._gaussian_factor import GaussianFactor


class GaussianFactorLaw(AbstractProbabilityLaw):
    """Gaussian law from a full- or reduced-rank covariance factor."""

    location: Array
    factor: GaussianFactor
    left_vectors: Array
    singular_values: Array
    support_tolerance: Array
    _event_shape: tuple[int, ...] = eqx.field(static=True)
    _rank: int = eqx.field(static=True)
    _measure_kind: MeasureKind = eqx.field(static=True)

    def __init__(
        self,
        location: ArrayLike,
        factor: GaussianFactor,
        /,
        *,
        event_shape,
        support_tolerance: ArrayLike = 1e-8,
    ):
        if not isinstance(factor, GaussianFactor):
            raise TypeError("factor must be a GaussianFactor.")
        events = tuple(int(size) for size in event_shape)
        if not events or any(size <= 0 for size in events):
            raise ValueError("event_shape must contain positive dimensions.")
        size = prod(events)
        mean = jnp.asarray(location)
        if not jnp.issubdtype(mean.dtype, jnp.inexact):
            mean = mean.astype(factor.factor.dtype)
        if jnp.iscomplexobj(mean) or jnp.iscomplexobj(factor.factor):
            raise TypeError("GaussianFactorLaw initially requires real coordinates.")
        if mean.shape != events:
            raise ValueError(
                f"location must have event shape {events}; got {mean.shape}."
            )
        if factor.factor.ndim != 2 or factor.event_size != size:
            raise ValueError("Gaussian factor must be unbatched and match event size.")
        rank = int(jax.device_get(factor.numerical_rank))
        if rank != factor.rank or rank <= 0:
            raise ValueError("Gaussian factor columns must be linearly independent.")
        tolerance = jnp.asarray(support_tolerance, dtype=mean.real.dtype).reshape(())
        if bool(~jnp.isfinite(tolerance)) or float(tolerance) < 0.0:
            raise ValueError("support_tolerance must be finite and nonnegative.")
        left, singular, _ = jnp.linalg.svd(factor.factor, full_matrices=False)
        self.location = mean
        self.factor = factor
        self.left_vectors = left
        self.singular_values = singular
        self.support_tolerance = tolerance
        self._event_shape = events
        self._rank = rank
        self._measure_kind = "lebesgue" if rank == size else "hausdorff"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> MeasureKind:
        return self._measure_kind

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def event_size(self) -> int:
        return prod(self.event_shape)

    def _flat(self, value: ArrayLike, /) -> tuple[Array, tuple[int, ...]]:
        array = jnp.asarray(value, dtype=self.location.dtype)
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError(f"value must end in event shape {self.event_shape}.")
        leading = tuple(array.shape[:-rank])
        return array.reshape(leading + (self.event_size,)), leading

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        noise = jax.random.normal(
            key,
            samples + (self.rank,),
            dtype=self.location.dtype,
        )
        centered = ein.contract("ir,...r->...i", self.factor.factor, noise)
        return (centered + self.location.reshape((self.event_size,))).reshape(
            samples + self.event_shape
        )

    def _coordinates_and_residual(self, value: ArrayLike, /):
        flat, leading = self._flat(value)
        residual = flat - self.location.reshape((self.event_size,))
        projected = ein.contract("ir,...i->...r", self.left_vectors, residual)
        coordinates = projected / self.singular_values
        reconstructed = ein.contract("ir,...r->...i", self.left_vectors, projected)
        orthogonal = residual - reconstructed
        return coordinates, orthogonal, leading

    def contains(self, value: ArrayLike, /) -> Array:
        flat, _ = self._flat(value)
        coordinates, orthogonal, _ = self._coordinates_and_residual(value)
        finite = jnp.all(jnp.isfinite(coordinates), axis=-1)
        residual = jnp.linalg.vector_norm(orthogonal, axis=-1)
        scale = 1.0 + jnp.linalg.vector_norm(flat, axis=-1)
        return finite & (residual <= self.support_tolerance * scale)

    def log_prob(self, value: ArrayLike, /) -> Array:
        flat, _ = self._flat(value)
        coordinates, orthogonal, _ = self._coordinates_and_residual(value)
        support = jnp.linalg.vector_norm(
            orthogonal, axis=-1
        ) <= self.support_tolerance * (1.0 + jnp.linalg.vector_norm(flat, axis=-1))
        quadratic = jnp.sum(coordinates**2, axis=-1)
        log_pseudodeterminant = 2.0 * jnp.sum(jnp.log(self.singular_values))
        density = -0.5 * (
            quadratic
            + log_pseudodeterminant
            + self.rank * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=quadratic.dtype))
        )
        return jnp.where(support & jnp.isfinite(density), density, -jnp.inf)

    def subspace_score(self, value: ArrayLike, /) -> Array:
        flat, leading = self._flat(value)
        residual = flat - self.location.reshape((self.event_size,))
        projected = ein.contract("ir,...i->...r", self.left_vectors, residual)
        weighted = projected / self.singular_values**2
        score = -ein.contract("ir,...r->...i", self.left_vectors, weighted)
        return score.reshape(leading + self.event_shape)

    def score(self, value: ArrayLike, /) -> Array:
        if self.density_measure_kind != "lebesgue":
            raise ValueError(
                "A singular Gaussian has no ambient Lebesgue score; "
                "use subspace_score explicitly."
            )
        return self.subspace_score(value)


__all__ = ["GaussianFactorLaw"]
