#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..metrix import (
    Immersion,
    induced_boundary_density,
    RiemannianMetric,
    WeightedRiemannianMeasure,
)
from ._precision import IntegrationPrecisionPolicy
from ._rules import ReferenceRule
from ._targets import MappedTarget


class _BoundaryMeasureScale(StrictModule):
    density: Any

    def __init__(self, ambient_metric: RiemannianMetric, parameterization: Immersion, /):
        self.density = induced_boundary_density(ambient_metric, parameterization)

    def __call__(self, reference_points: Array, /) -> Array:
        return self.density(reference_points)


class MetricMeasureNormalization(StrictModule):
    """Finite quadrature evidence for one weighted metric measure."""

    mass: Array
    log_mass: Array
    valid: Array
    minimum_log_density: Array
    maximum_log_density: Array
    precision_evidence: PrecisionEvidenceEnvelope
    sample_count: int

    def __init__(
        self,
        *,
        mass: ArrayLike,
        log_mass: ArrayLike,
        valid: ArrayLike,
        minimum_log_density: ArrayLike,
        maximum_log_density: ArrayLike,
        sample_count: int,
        precision_evidence: PrecisionEvidenceEnvelope,
    ):
        self.mass = jnp.asarray(mass)
        self.log_mass = jnp.asarray(log_mass)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.minimum_log_density = jnp.asarray(minimum_log_density)
        self.maximum_log_density = jnp.asarray(maximum_log_density)
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        self.precision_evidence = precision_evidence
        self.sample_count = int(sample_count)


def riemannian_boundary_target(
    reference_rule: ReferenceRule,
    ambient_metric: RiemannianMetric,
    parameterization: Immersion,
    /,
    *,
    mask=None,
    target_mass: ArrayLike | None = None,
) -> MappedTarget:
    """Build a mapped boundary target with induced Riemannian measure."""
    if not isinstance(ambient_metric, RiemannianMetric):
        raise TypeError("ambient_metric must be a RiemannianMetric.")
    if not isinstance(parameterization, Immersion):
        raise TypeError("parameterization must be an Immersion.")
    return MappedTarget(
        reference_rule,
        parameterization,
        _BoundaryMeasureScale(ambient_metric, parameterization),
        mask=mask,
        target_mass=None if target_mass is None else jnp.asarray(target_mass),
    )


def normalize_metric_measure(
    measure: WeightedRiemannianMeasure,
    coordinates: ArrayLike,
    base_weights: ArrayLike,
    /,
    *,
    precision: IntegrationPrecisionPolicy | None = None,
) -> MetricMeasureNormalization:
    """Evaluate finite positive mass under caller-supplied base quadrature weights."""
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
    points = precision_.evaluation(coordinates)
    weights = precision_.accumulation(base_weights)
    if points.ndim != 2 or points.shape[-1] != measure.chart.dimension:
        raise ValueError("coordinates must have shape (samples, chart_dimension).")
    if weights.shape != (points.shape[0],):
        raise ValueError("base_weights must match the coordinate sample axis.")
    log_density = precision_.evaluation(measure.log_coordinate_density(points))
    maximum = precision_.decision(jnp.max(log_density))
    scaled = jnp.sum(
        precision_.accumulation(
            weights * jnp.exp(precision_.accumulation(log_density - maximum))
        )
    )
    log_mass = precision_.decision(maximum + jnp.log(scaled))
    mass = precision_.output(jnp.exp(log_mass))
    valid = (
        jnp.all(jnp.isfinite(points))
        & jnp.all(jnp.isfinite(weights) & (weights >= 0.0))
        & jnp.all(jnp.isfinite(log_density))
        & jnp.isfinite(mass)
        & (mass > 0.0)
    )
    return MetricMeasureNormalization(
        mass=mass,
        log_mass=log_mass,
        valid=valid,
        minimum_log_density=jnp.min(log_density),
        maximum_log_density=maximum,
        sample_count=points.shape[0],
        precision_evidence=precision_.evidence_for(log_density),
    )


__all__ = [
    "MetricMeasureNormalization",
    "normalize_metric_measure",
    "riemannian_boundary_target",
]
