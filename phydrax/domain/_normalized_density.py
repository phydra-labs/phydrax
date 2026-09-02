#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..integration._api import IntegrationRealization, reduce
from ..integration._estimates import IntegrationEstimate
from ._function import DomainFunction
from ._referenced_density import DensityReference, ReferencedDensityField


class _ExponentialFieldEvaluator(StrictModule):
    log_evaluator: Any

    def __call__(self, *args, key=None, **kwargs):
        values = self.log_evaluator(*args, key=key, **kwargs)
        return jnp.exp(values)


class _NormalizedFieldEvaluator(StrictModule):
    log_evaluator: Any
    log_normalizer: Array
    target_mass: Array

    def __call__(self, *args, key=None, **kwargs):
        values = self.log_evaluator(*args, key=key, **kwargs)
        return self.target_mass * jnp.exp(values - self.log_normalizer)


class NormalizedDensityField(StrictModule):
    """Positive field normalized on one frozen represented measure."""

    log_field: DomainFunction
    field: DomainFunction
    referenced: ReferencedDensityField
    realization: IntegrationRealization
    normalization: IntegrationEstimate
    log_normalizer: Array
    target_mass: Array
    finite: Array
    positive: Array
    normalization_id: str = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)

    @property
    def reference(self) -> DensityReference:
        return self.referenced.reference

    @property
    def state_var(self) -> str:
        return self.referenced.state_var

    def __call__(self, *args, key=None, **kwargs):
        return self.field(*args, key=key, **kwargs)


class DensityNormalizationEvidence(StrictModule):
    """Represented mass and quadrature evidence without a continuum exactness claim."""

    normalization: IntegrationEstimate
    represented_mass: Array
    represented_mass_error: Array
    finite: Array
    positive: Array
    valid: Array
    normalization_id: str = eqx.field(static=True)
    evidence_kind: str = eqx.field(static=True)


def _scalar_estimate_value(value: Any, /) -> Array:
    array = jnp.asarray(value.data if isinstance(value, cx.Field) else value)
    if array.shape != ():
        raise ValueError("Density normalization must reduce to one scalar mass.")
    return array


def normalize_density_field(
    log_density: DomainFunction,
    realization: IntegrationRealization,
    /,
    *,
    target_mass: ArrayLike = 1.0,
    reference: DensityReference = "coordinate",
    state_var: str = "x",
    metric=None,
    measure=None,
) -> NormalizedDensityField:
    """Exponentiate and normalize a log field on a frozen realization.

    Unit mass is exact only for the represented reduction.  The retained
    ``IntegrationEstimate`` is the sole continuum/quadrature accuracy evidence.
    Nonfinite values fail rather than being clipped, floored, or repaired.
    """
    if not isinstance(log_density, DomainFunction):
        raise TypeError("log_density must be a DomainFunction.")
    if not isinstance(realization, IntegrationRealization):
        raise TypeError("realization must be an IntegrationRealization.")
    target = jnp.asarray(target_mass)
    if target.shape != () or not isfinite(float(target)) or not float(target) > 0.0:
        raise ValueError("target_mass must be one finite positive scalar.")
    exponential = DomainFunction(
        domain=log_density.domain,
        deps=log_density.deps,
        func=_ExponentialFieldEvaluator(log_density.func),
        metadata={
            **dict(log_density.metadata),
            "density_transform": "strict-exponential",
            "normalization_scope": "represented-realization",
        },
    )
    estimate = reduce(exponential, realization)
    normalizer = _scalar_estimate_value(estimate.value)
    valid = estimate.successful & jnp.isfinite(normalizer) & (normalizer > 0.0)
    checked = eqx.error_if(
        normalizer,
        ~valid,
        "Density normalization requires a finite positive represented mass.",
    )
    log_normalizer = jnp.log(checked)
    normalized_field = DomainFunction(
        domain=log_density.domain,
        deps=log_density.deps,
        func=_NormalizedFieldEvaluator(log_density.func, log_normalizer, target),
        metadata={
            **dict(log_density.metadata),
            "density_transform": "normalized-strict-exponential",
            "normalization_scope": "represented-realization",
            "reference": reference,
        },
    )
    referenced = ReferencedDensityField(
        normalized_field,
        state_var=state_var,
        reference=reference,
        metric=metric,
        measure=measure,
    )
    normalization_id = canonical_fingerprint(
        {
            "kind": "normalized-density-field-v1",
            "domain": tuple(log_density.domain.labels),
            "dependencies": log_density.deps,
            "reference": reference,
            "state_var": state_var,
            "target_mass": float(target),
            "target": type(realization.target).__name__,
            "plan": type(realization.plan).__name__,
        }
    )
    return NormalizedDensityField(
        log_field=log_density,
        field=normalized_field,
        referenced=referenced,
        realization=realization,
        normalization=estimate,
        log_normalizer=log_normalizer,
        target_mass=target,
        finite=valid,
        positive=valid,
        normalization_id=normalization_id,
        approximation_kind="represented-measure-normalization",
    )


def density_normalization_evidence(
    normalized: NormalizedDensityField, /
) -> DensityNormalizationEvidence:
    """Return represented mass and the original integration error evidence."""
    if not isinstance(normalized, NormalizedDensityField):
        raise TypeError("normalized must be a NormalizedDensityField.")
    represented_mass = normalized.target_mass
    error = jnp.asarray(0.0, dtype=represented_mass.dtype)
    valid = (
        normalized.normalization.successful
        & normalized.finite
        & normalized.positive
        & jnp.isfinite(represented_mass)
    )
    return DensityNormalizationEvidence(
        normalization=normalized.normalization,
        represented_mass=represented_mass,
        represented_mass_error=error,
        finite=normalized.finite,
        positive=normalized.positive,
        valid=valid,
        normalization_id=normalized.normalization_id,
        evidence_kind="represented-exact-with-integration-estimate",
    )


__all__ = [
    "DensityNormalizationEvidence",
    "NormalizedDensityField",
    "density_normalization_evidence",
    "normalize_density_field",
]
