#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._exponential_family import (
    AbstractExponentialFamily,
    ExponentialFamilyConversionResult,
    MeanCoordinates,
    NaturalCoordinates,
)
from .._strict import StrictModule
from ..linalg import LinearSolvePlan, LinearSolvePolicy, LinearSolveResult
from ..metrix import (
    InformationMetricOperator,
    pulled_back_information_operator,
)


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

    def information_operator(
        self,
        natural: NaturalCoordinates,
        /,
        *,
        damping: ArrayLike = 0.0,
    ) -> InformationMetricOperator:
        domain = self.family.natural_domain(natural)
        if natural.batch_shape:
            raise ValueError(
                "information_operator currently requires unbatched coordinates."
            )
        values = jnp.where(domain.valid[..., None], natural.values, jnp.nan)
        return InformationMetricOperator(
            lambda direction: self.family.fisher_action(natural, direction),
            values,
            damping=damping,
            metric_id=f"fisher:{self.family.signature.family_id}",
        )

    def fisher_matrix(
        self,
        natural: NaturalCoordinates,
        /,
        *,
        maximum_size: int = 256,
    ) -> Array:
        """Materialize a bounded unbatched Fisher matrix."""
        return self.information_operator(natural).materialize(maximum_size=maximum_size)

    def natural_gradient_result(
        self,
        natural: NaturalCoordinates,
        cotangent: ArrayLike,
        /,
        *,
        damping: ArrayLike = 0.0,
        policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    ) -> LinearSolveResult:
        cotangent_ = jnp.asarray(cotangent)
        if cotangent_.shape != natural.values.shape:
            raise ValueError("Natural-gradient cotangent must match coordinates.")
        return self.information_operator(natural, damping=damping).solve(
            cotangent_,
            policy=policy,
        )

    def natural_gradient(
        self,
        natural: NaturalCoordinates,
        cotangent: ArrayLike,
        /,
        *,
        damping: ArrayLike = 0.0,
        policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    ) -> Array:
        """Solve the Fisher duality equation through the linear runtime."""
        return self.natural_gradient_result(
            natural,
            cotangent,
            damping=damping,
            policy=policy,
        ).value

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

    def pulled_back_operator(
        self,
        natural_function: Callable[[Array], NaturalCoordinates],
        parameters: ArrayLike,
        /,
        *,
        damping: ArrayLike = 0.0,
    ) -> InformationMetricOperator:
        if not callable(natural_function):
            raise TypeError("natural_function must be callable.")
        parameters_ = jnp.asarray(parameters)
        natural = natural_function(parameters_)
        if not isinstance(natural, NaturalCoordinates):
            raise TypeError("natural_function must return NaturalCoordinates.")
        target = self.information_operator(natural)
        return pulled_back_information_operator(
            lambda value: natural_function(value).values,
            parameters_,
            target,
            damping=damping,
            metric_id=f"pullback-fisher:{self.family.signature.family_id}",
        )


__all__ = ["ExponentialFamilyInformationGeometry"]
