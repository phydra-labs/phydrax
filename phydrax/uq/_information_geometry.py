#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._exponential_family import (
    AbstractExponentialFamily,
    ExponentialFamilyConversionResult,
    MeanCoordinates,
    NaturalCoordinates,
)
from .._strict import StrictModule


class ExponentialFamilyInformationGeometry(StrictModule):
    """Dually-flat Fisher geometry of one regular exponential family."""

    family: AbstractExponentialFamily

    def __init__(self, family: AbstractExponentialFamily, /):
        if not isinstance(family, AbstractExponentialFamily):
            raise TypeError("family must implement AbstractExponentialFamily.")
        self.family = family

    def fisher_action(
        self,
        natural: NaturalCoordinates,
        direction: ArrayLike,
        /,
    ) -> Array:
        return self.family.fisher_action(natural, direction)

    def fisher_matrix(self, natural: NaturalCoordinates, /) -> Array:
        """Materialize an unbatched Fisher matrix from exact actions."""
        self.family.natural_domain(natural)
        if natural.batch_shape:
            raise ValueError("fisher_matrix currently requires unbatched coordinates.")
        dimension = natural.values.shape[-1]
        identity = jnp.eye(dimension, dtype=natural.values.dtype)
        columns = jax.vmap(lambda direction: self.fisher_action(natural, direction))(
            identity
        )
        return jnp.swapaxes(columns, -1, -2)

    def natural_gradient(
        self,
        natural: NaturalCoordinates,
        cotangent: ArrayLike,
        /,
        *,
        damping: float = 0.0,
    ) -> Array:
        """Solve the finite-dimensional Fisher duality equation."""
        cotangent_ = jnp.asarray(cotangent)
        if cotangent_.shape != natural.values.shape:
            raise ValueError("Natural-gradient cotangent must match coordinates.")
        if natural.batch_shape:
            raise ValueError("natural_gradient currently requires unbatched coordinates.")
        damping_ = float(damping)
        if damping_ < 0.0:
            raise ValueError("damping must be non-negative.")
        fisher = self.fisher_matrix(natural)
        identity = jnp.eye(fisher.shape[-1], dtype=fisher.dtype)
        return jnp.linalg.solve(fisher + damping_ * identity, cotangent_)

    def dual_coordinates(self, natural: NaturalCoordinates, /) -> MeanCoordinates:
        return self.family.mean_from_natural(natural)

    def kl_divergence(
        self,
        left: NaturalCoordinates,
        right: NaturalCoordinates,
        /,
    ) -> Array:
        return self.family.kl_divergence(left, right)

    def exponential_interpolate(
        self,
        left: NaturalCoordinates,
        right: NaturalCoordinates,
        weight: ArrayLike,
        /,
    ) -> NaturalCoordinates:
        self.family.natural_domain(left)
        self.family.natural_domain(right)
        weight_ = jnp.asarray(weight, dtype=left.values.dtype)
        if weight_.shape != ():
            raise ValueError("Interpolation weight must be scalar.")
        candidate = NaturalCoordinates(
            (1.0 - weight_) * left.values + weight_ * right.values,
            self.family.signature,
        )
        domain = self.family.natural_domain(candidate)
        candidate_values = jnp.where(domain.valid[..., None], candidate.values, jnp.nan)
        return NaturalCoordinates(candidate_values, self.family.signature)

    def mixture_interpolate(
        self,
        left: NaturalCoordinates,
        right: NaturalCoordinates,
        weight: ArrayLike,
        /,
    ) -> ExponentialFamilyConversionResult:
        left_mean = self.family.mean_from_natural(left)
        right_mean = self.family.mean_from_natural(right)
        weight_ = jnp.asarray(weight, dtype=left.values.dtype)
        if weight_.shape != ():
            raise ValueError("Interpolation weight must be scalar.")
        mean = MeanCoordinates(
            (1.0 - weight_) * left_mean.values + weight_ * right_mean.values,
            self.family.signature,
        )
        return self.family.natural_from_mean(mean)


__all__ = ["ExponentialFamilyInformationGeometry"]
