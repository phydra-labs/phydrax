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
from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from ..linalg import LinearSolvePlan, LinearSolvePolicy, LinearSolveResult
from ..metrix import (
    InformationMetricOperator,
    pulled_back_information_operator,
)


class ExponentialFamilyInformationGeometry(StrictModule):
    """Dually-flat Fisher geometry of one regular exponential family."""

    family: AbstractExponentialFamily
    precision: GeometryPrecisionPolicy

    def __init__(
        self,
        family: AbstractExponentialFamily,
        /,
        *,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not isinstance(family, AbstractExponentialFamily):
            raise TypeError("family must implement AbstractExponentialFamily.")
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        self.family = family
        self.precision = precision_

    def _computed_natural(
        self,
        natural: NaturalCoordinates,
        /,
    ) -> NaturalCoordinates:
        self.precision.validate_coordinates(natural.values)
        return NaturalCoordinates(
            self.precision.compute(natural.values),
            self.family.signature,
        )

    def fisher_action(
        self,
        natural: NaturalCoordinates,
        direction: ArrayLike,
        /,
    ) -> Array:
        computed = self._computed_natural(natural)
        value = self.family.fisher_action(
            computed,
            self.precision.compute(direction),
        )
        return self.precision.output(value)

    def information_operator(
        self,
        natural: NaturalCoordinates,
        /,
        *,
        damping: ArrayLike = 0.0,
    ) -> InformationMetricOperator:
        computed = self._computed_natural(natural)
        domain = self.family.natural_domain(computed)
        if computed.batch_shape:
            raise ValueError(
                "information_operator currently requires unbatched coordinates."
            )
        values = jnp.where(domain.valid[..., None], natural.values, jnp.nan)

        def action(direction: Array) -> Array:
            return self.precision.compute(
                self.family.fisher_action(
                    computed,
                    self.precision.compute(direction),
                )
            )

        return InformationMetricOperator(
            action,
            values,
            damping=self.precision.compute(damping),
            metric_id=f"fisher:{self.family.signature.family_id}",
            precision=self.precision,
        )

    def fisher_matrix(
        self,
        natural: NaturalCoordinates,
        /,
        *,
        maximum_size: int = 256,
    ) -> Array:
        """Materialize a bounded unbatched Fisher matrix."""
        matrix = self.information_operator(natural).materialize(maximum_size=maximum_size)
        return self.precision.output(matrix)

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
            self.precision.compute(cotangent_),
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
        result = self.natural_gradient_result(
            natural,
            cotangent,
            damping=damping,
            policy=policy,
        )
        return self.precision.output(result.value)

    def dual_coordinates(self, natural: NaturalCoordinates, /) -> MeanCoordinates:
        computed = self.family.mean_from_natural(self._computed_natural(natural))
        return MeanCoordinates(
            self.precision.output(computed.values),
            self.family.signature,
        )

    def kl_divergence(
        self,
        left: NaturalCoordinates,
        right: NaturalCoordinates,
        /,
    ) -> Array:
        left_ = self._computed_natural(left)
        right_ = self._computed_natural(right)
        return self.precision.decision(self.family.kl_divergence(left_, right_))

    def exponential_interpolate(
        self,
        left: NaturalCoordinates,
        right: NaturalCoordinates,
        weight: ArrayLike,
        /,
    ) -> NaturalCoordinates:
        self.family.natural_domain(left)
        self.family.natural_domain(right)
        self.precision.validate_coordinates(left.values)
        self.precision.validate_coordinates(right.values)
        weight_ = self.precision.compute(jnp.asarray(weight, dtype=left.values.dtype))
        if weight_.shape != ():
            raise ValueError("Interpolation weight must be scalar.")
        candidate = NaturalCoordinates(
            self.precision.output(
                (1.0 - weight_) * self.precision.compute(left.values)
                + weight_ * self.precision.compute(right.values)
            ),
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
        left_mean = self.family.mean_from_natural(self._computed_natural(left))
        right_mean = self.family.mean_from_natural(self._computed_natural(right))
        weight_ = self.precision.compute(jnp.asarray(weight, dtype=left.values.dtype))
        if weight_.shape != ():
            raise ValueError("Interpolation weight must be scalar.")
        mean = MeanCoordinates(
            self.precision.output(
                (1.0 - weight_) * self.precision.compute(left_mean.values)
                + weight_ * self.precision.compute(right_mean.values)
            ),
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
        parameters_ = self.precision.compute(parameters)
        natural = natural_function(parameters_)
        if not isinstance(natural, NaturalCoordinates):
            raise TypeError("natural_function must return NaturalCoordinates.")
        target = self.information_operator(natural)
        return pulled_back_information_operator(
            lambda value: natural_function(value).values,
            parameters_,
            target,
            damping=self.precision.compute(damping),
            metric_id=f"pullback-fisher:{self.family.signature.family_id}",
            precision=self.precision,
        )


__all__ = ["ExponentialFamilyInformationGeometry"]
