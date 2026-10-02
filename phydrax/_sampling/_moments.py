#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import assert_never

import jax.numpy as jnp
from jax import Array

from .._randomized_residual_modes import RealizationSamplingDesign
from ..typing import parse


def validate_sampling_design(
    sampling_design: RealizationSamplingDesign,
    population_size: int | None,
    count: int,
    /,
) -> RealizationSamplingDesign:
    """Validate the declared sampling law, never infer it from realization IDs."""
    design = parse(sampling_design, RealizationSamplingDesign, "sampling_design")
    if count < 1:
        raise ValueError("At least one realization is required.")
    if population_size is not None:
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, Integral)
            or population_size < count
        ):
            raise ValueError(
                "population_size must be an integer at least the realization count."
            )
    match design:
        case "finite_population":
            if population_size is None:
                raise ValueError("finite_population sampling requires population_size.")
        case "iid" | "unknown":
            if population_size is not None:
                raise ValueError(
                    "population_size is only valid for finite_population or exact sampling."
                )
        case "exact":
            pass
        case _:
            assert_never(design)
    return design


def uncertainty_available(design: RealizationSamplingDesign, count: int, /) -> bool:
    return design == "exact" or (design != "unknown" and count > 1)


def realization_moments(
    values: Array,
    sampling_design: RealizationSamplingDesign,
    population_size: int | None = None,
    /,
) -> tuple[Array, Array, Array]:
    """Mean, sample variance, and law-correct mean standard error.

    Exact means have zero sampling uncertainty, including singleton means. Unknown
    laws and nonexact singleton samples cannot identify mean uncertainty; NaN is
    evidence of unavailability, not evidence of a nonfinite estimator value.
    """
    if values.ndim < 1:
        raise ValueError("values must have a realization-first axis.")
    count = values.shape[0]
    design = validate_sampling_design(sampling_design, population_size, count)
    mean = jnp.mean(values, axis=0)
    real_dtype = mean.real.dtype
    if not jnp.issubdtype(real_dtype, jnp.inexact):
        real_dtype = jnp.dtype(jnp.float64)
    if count == 1:
        variance = jnp.full(mean.shape, jnp.nan, dtype=real_dtype)
    else:
        centered = values - mean
        variance = jnp.sum(jnp.abs(centered) ** 2, axis=0) / (count - 1)
    match design:
        case "exact":
            if count == 1:
                variance = jnp.zeros(mean.shape, dtype=real_dtype)
            standard_error = jnp.zeros(mean.shape, dtype=real_dtype)
        case "unknown":
            standard_error = jnp.full(mean.shape, jnp.nan, dtype=real_dtype)
        case "iid" | "finite_population":
            if count == 1:
                standard_error = jnp.full(mean.shape, jnp.nan, dtype=real_dtype)
            else:
                correction = 1.0
                if design == "finite_population":
                    if population_size is None:
                        raise RuntimeError("Validated population size is unavailable.")
                    correction = 1.0 - count / population_size
                standard_error = jnp.sqrt(correction * variance / count)
        case _:
            assert_never(design)
    return mean, variance, standard_error


__all__ = ["realization_moments", "uncertainty_available", "validate_sampling_design"]
