#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._doc import DOC_KEY0
from .._term import AbstractEvaluatedScalarTerm, TermEvaluation
from ..domain import DomainFunction
from ..integration import (
    DensityTarget,
    DiscreteMeasureTarget,
    IntegrationRealization,
    WeightedSampleTarget,
)
from ..transport import (
    PreparedUnbalancedSinkhornReference,
    unbalanced_sinkhorn_divergence_against,
)


SpatialMeasure = (
    DiscreteMeasureTarget | WeightedSampleTarget | DensityTarget | IntegrationRealization
)


class SpatialUnbalancedSinkhornDivergenceTerm(AbstractEvaluatedScalarTerm):
    """Training term for unequal-mass physical spatial or intensity measures."""

    objective_vars: tuple[str, ...]
    measure_builder: Callable[[Mapping[str, DomainFunction]], SpatialMeasure]
    reference: PreparedUnbalancedSinkhornReference
    encoder: Callable[[Any], Any] | None
    weight: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        measure_builder: Callable[[Mapping[str, DomainFunction]], SpatialMeasure],
        reference: PreparedUnbalancedSinkhornReference,
        /,
        *,
        encoder: Callable[[Any], Any] | None = None,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not callable(measure_builder):
            raise TypeError("measure_builder must be callable.")
        if not isinstance(reference, PreparedUnbalancedSinkhornReference):
            raise TypeError("reference must be a PreparedUnbalancedSinkhornReference.")
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable or None.")
        weight_ = jnp.asarray(weight, dtype=float)
        if weight_.shape != ():
            raise ValueError("weight must be scalar.")
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.measure_builder = measure_builder
        self.reference = reference
        self.encoder = encoder
        self.weight = weight_
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> TermEvaluation:
        del key, kwargs
        source = self.measure_builder(functions)
        if not isinstance(
            source,
            (
                DiscreteMeasureTarget,
                WeightedSampleTarget,
                DensityTarget,
                IntegrationRealization,
            ),
        ):
            raise TypeError(
                "measure_builder must return a supported physical measure or "
                "IntegrationRealization."
            )
        result = unbalanced_sinkhorn_divergence_against(
            source,
            self.reference,
            encoder=self.encoder,
        )
        value = eqx.error_if(
            result.value,
            ~result.converged,
            "SpatialUnbalancedSinkhornDivergenceTerm transport did not converge.",
        )
        return TermEvaluation(self.weight * value, diagnostics=result)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(functions, key=key, **kwargs).value


__all__ = ["SpatialUnbalancedSinkhornDivergenceTerm"]
