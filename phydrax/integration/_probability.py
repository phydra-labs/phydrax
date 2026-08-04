#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from ..domain._function import DomainFunction
from ..domain._probability import ProbabilityDomain
from ..domain._structure import PointsBatch, ProductStructure
from ._batches import PointIntegrationBatch
from ._estimates import (
    FixedQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import sum_over
from ._plans import FixedQuadraturePlan
from ._rules import interval_rule_data
from ._status import IntegrationStatus
from ._targets import DensityTarget, ProbabilityTarget


def materialize_fixed_probability(
    target: ProbabilityTarget,
    plan: FixedQuadraturePlan,
    /,
) -> PointIntegrationBatch:
    """Map an open canonical interval rule through a probability quantile."""
    data = interval_rule_data(plan.rule)
    unit = 0.5 * (data.nodes + 1.0)
    if bool(jnp.any(unit <= 0.0)) or bool(jnp.any(unit >= 1.0)):
        support = target.probability.distribution.support
        if support is None:
            raise ValueError(
                "Endpoint-inclusive quadrature cannot map an unbounded probability "
                "domain; use GaussLegendreRule or stochastic integration."
            )
    samples = target.probability.distribution.icdf(unit)
    structure = ProductStructure(((target.probability.label,),)).canonicalize(
        (target.probability.label,)
    )
    axis = structure.axis_for(target.probability.label)
    if axis is None:
        raise RuntimeError("Probability quadrature structure has no axis.")
    points = PointsBatch(
        frozendict(
            {target.probability.label: cx.Field(jnp.asarray(samples), dims=(axis,))}
        ),
        structure,
    )
    weights = cx.Field(0.5 * data.weights, dims=(axis,))
    return PointIntegrationBatch(
        points,
        weights,
        axes=(axis,),
        target_mass=jnp.asarray(1.0),
        provenance=f"probability:{type(plan.rule).__name__}",
    )


def _as_function(value: Any, probability: ProbabilityDomain, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=probability, deps=(), func=value)


def integrate_fixed_probability(
    integrand: Any,
    target: ProbabilityTarget | DensityTarget,
    batch: PointIntegrationBatch,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Reduce fixed probability nodes with optional density normalization."""
    callback_kwargs = {} if kwargs is None else kwargs
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ProbabilityTarget):
        raise TypeError("Probability quadrature requires a ProbabilityTarget base.")
    function = _as_function(integrand, base.probability)
    values = function(batch.points, key=key, **callback_kwargs)
    if not isinstance(values, cx.Field):
        raise TypeError("Probability integrands must evaluate to coordax.Field.")
    weights = batch.weights
    if isinstance(target, DensityTarget):
        log_density = _as_function(target.log_density, base.probability)
        log_values = log_density(batch.points, key=key, **callback_kwargs)
        weights = weights * cx.Field(
            jnp.exp(jnp.asarray(log_values.data)), dims=log_values.dims
        )
    numerator = weights * values
    denominator = weights
    for axis in batch.axes:
        numerator = sum_over(numerator, axis)
        denominator = sum_over(denominator, axis)
    normalized = not isinstance(target, DensityTarget) or target.normalized
    numerator_data = jnp.asarray(numerator.data)
    denominator_data = jnp.asarray(denominator.data)
    finite = (
        jnp.all(jnp.isfinite(jnp.asarray(values.data)))
        & jnp.all(jnp.isfinite(jnp.asarray(weights.data)))
        & jnp.all(jnp.isfinite(numerator_data))
    )
    if normalized:
        value = numerator / denominator
        valid_mass = jnp.all(jnp.isfinite(denominator_data)) & jnp.all(
            denominator_data != 0.0
        )
        status = jnp.where(
            valid_mass,
            int(IntegrationStatus.CONVERGED),
            int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
        )
    else:
        value = numerator
        status = jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    count = int(batch.weights.data.size)
    diagnostics = FixedQuadratureDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(count, dtype=jnp.int32),
        target_mass=jnp.asarray(1.0),
        rule=batch.provenance,
    )
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=count,
        error_estimate=None,
        error_kind=None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("fixed", "probability", batch.provenance),
    )


__all__ = ["integrate_fixed_probability", "materialize_fixed_probability"]
