#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..geometry.complex import ProjectiveHypersurface, ProjectiveLineSamples
from ..geometry.complex._hypersurface_patch import HypersurfacePatchGeometry


ProjectiveMeasureKind: TypeAlias = Literal["fubini-study", "canonical"]


class ProjectiveMeasureTarget(StrictModule):
    samples: ProjectiveLineSamples
    log_weights: Array
    normalized_weights: Array
    physical_mass: Array
    effective_sample_size: Array
    valid: Array
    measure_kind: ProjectiveMeasureKind

    def __init__(
        self,
        samples: ProjectiveLineSamples,
        log_weights: ArrayLike,
        /,
        *,
        measure_kind: ProjectiveMeasureKind,
    ):
        weights = jnp.asarray(log_weights)
        if weights.shape != samples.valid.shape:
            raise ValueError("log_weights must match the projective sample axis.")
        masked = jnp.where(samples.valid, weights, -jnp.inf)
        maximum = jnp.max(masked)
        scaled = jnp.where(samples.valid, jnp.exp(masked - maximum), 0.0)
        scaled_mass = jnp.sum(scaled)
        normalized = scaled / scaled_mass
        mass = jnp.exp(maximum) * scaled_mass / float(weights.shape[0])
        ess = 1.0 / jnp.sum(normalized**2)
        self.samples = samples
        self.log_weights = weights
        self.normalized_weights = normalized
        self.physical_mass = mass
        self.effective_sample_size = ess
        self.valid = (
            jnp.any(samples.valid)
            & jnp.all(jnp.isfinite(jnp.where(samples.valid, weights, 0.0)))
            & jnp.isfinite(mass)
            & (mass > 0.0)
        )
        self.measure_kind = measure_kind


class ProjectiveIntegralResult(StrictModule):
    normalized_value: Array
    physical_value: Array
    effective_sample_size: Array
    valid: Array

    def __init__(
        self,
        normalized_value: ArrayLike,
        physical_value: ArrayLike,
        effective_sample_size: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.normalized_value = jnp.asarray(normalized_value)
        self.physical_value = jnp.asarray(physical_value)
        self.effective_sample_size = jnp.asarray(effective_sample_size)
        self.valid = jnp.asarray(valid, dtype=bool)


def projective_measure_target(
    hypersurface: ProjectiveHypersurface,
    samples: ProjectiveLineSamples,
    /,
    *,
    measure_kind: ProjectiveMeasureKind,
) -> ProjectiveMeasureTarget:
    if measure_kind not in ("fubini-study", "canonical"):
        raise ValueError("Unknown projective measure kind.")
    geometry = HypersurfacePatchGeometry(hypersurface)
    log_weights = []
    for index in range(samples.homogeneous_points.shape[0]):
        evaluation = geometry.evaluate(
            samples.homogeneous_points[index],
            chart_index=int(samples.chart_indices[index]),
            pivot_index=int(samples.pivot_indices[index]),
        )
        if measure_kind == "fubini-study":
            log_weight = 0.5 * jnp.linalg.slogdet(evaluation.induced_metric)[1]
        else:
            log_weight = 2.0 * jnp.log(jnp.abs(evaluation.residue_coefficient))
        log_weights.append(log_weight)
    return ProjectiveMeasureTarget(
        samples,
        jnp.stack(log_weights),
        measure_kind=measure_kind,
    )


def integrate_projective_samples(
    target: ProjectiveMeasureTarget,
    function: Callable[[Array], Array],
    /,
) -> ProjectiveIntegralResult:
    if not isinstance(target, ProjectiveMeasureTarget):
        raise TypeError("target must be a ProjectiveMeasureTarget.")
    if not callable(function):
        raise TypeError("function must be callable.")
    values = jax.vmap(function)(target.samples.homogeneous_points)
    if values.shape[0] != target.normalized_weights.shape[0]:
        raise ValueError("Integrand must preserve the projective sample axis.")
    shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    normalized = jnp.sum(target.normalized_weights.reshape(shape) * values, axis=0)
    physical = target.physical_mass * normalized
    valid = target.valid & jnp.all(jnp.isfinite(values))
    return ProjectiveIntegralResult(
        normalized,
        physical,
        target.effective_sample_size,
        valid,
    )


__all__ = [
    "ProjectiveIntegralResult",
    "ProjectiveMeasureKind",
    "ProjectiveMeasureTarget",
    "integrate_projective_samples",
    "projective_measure_target",
]
