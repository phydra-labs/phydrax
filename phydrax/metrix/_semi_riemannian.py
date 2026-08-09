#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._connection import LeviCivitaConnection
from ._metric import (
    AbstractSemiRiemannianMetric,
    LorentzianMetric,
    SemiRiemannianMetric,
)
from ._operator_kernels import apply_cotangent_map, covariant_symbol_contraction
from ._utils import _pointwise_array


class CausalCharacter(IntEnum):
    """Signature-independent causal classification codes."""

    TIMELIKE = -1
    NULL = 0
    SPACELIKE = 1


class _MetricGradientEvaluator(StrictModule):
    field: Callable[[Array], Array]
    metric: AbstractSemiRiemannianMetric

    def __init__(
        self,
        field: Callable[[Array], Array],
        metric: AbstractSemiRiemannianMetric,
        /,
    ):
        self.field = field
        self.metric = metric

    def __call__(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.field(coordinates))
        if value.shape != ():
            raise ValueError("A signed metric gradient requires a scalar field.")
        return apply_cotangent_map(
            jax.grad(self.field)(coordinates),
            self.metric.inverse(coordinates),
        )


class _InverseMetricMap(StrictModule):
    metric: AbstractSemiRiemannianMetric

    def __init__(self, metric: AbstractSemiRiemannianMetric, /):
        self.metric = metric

    def __call__(self, coordinates: Array, /) -> Array:
        return self.metric.inverse(coordinates)


def _require_signed_metric(
    metric: AbstractSemiRiemannianMetric, /
) -> SemiRiemannianMetric | LorentzianMetric:
    if not isinstance(metric, (SemiRiemannianMetric, LorentzianMetric)):
        raise TypeError(
            "This operator requires a SemiRiemannianMetric or LorentzianMetric."
        )
    return metric


def semi_riemannian_gradient(
    field: Callable[[Array], Array],
    metric: SemiRiemannianMetric | LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Raise ``df`` with a declared non-positive-definite metric."""
    if not callable(field):
        raise TypeError("field must be callable.")
    signed_metric = _require_signed_metric(metric)
    return _pointwise_array(
        _MetricGradientEvaluator(field, signed_metric),
        coordinates,
        signed_metric.chart.dimension,
    )


def dalembertian(
    field: Callable[[Array], Array],
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the Lorentzian wave operator ``gⁱʲ ∇ᵢ∇ⱼ field``."""
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("dalembertian requires a LorentzianMetric.")
    return covariant_symbol_contraction(
        field,
        _InverseMetricMap(metric),
        LeviCivitaConnection(metric),
        coordinates,
    )


def causal_character(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    vector: ArrayLike,
    /,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
) -> Array:
    """Return TIMELIKE, NULL, or SPACELIKE codes using a scale-aware test."""
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("causal_character requires a LorentzianMetric.")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Causal tolerances must be non-negative.")
    vector_array = jnp.asarray(vector)
    if vector_array.shape[-1:] != (metric.chart.dimension,):
        raise ValueError(
            f"Causal vectors must have trailing dimension {metric.chart.dimension}."
        )
    matrix = metric(coordinates)
    quadratic = metric.quadratic_form(vector_array, coordinates)
    scale = jnp.linalg.norm(matrix, axis=(-2, -1)) * jnp.sum(
        jnp.abs(vector_array) ** 2, axis=-1
    )
    threshold = absolute_tolerance + relative_tolerance * scale
    signed_quadratic = metric.timelike_sign * quadratic
    return jnp.where(
        jnp.abs(quadratic) <= threshold,
        int(CausalCharacter.NULL),
        jnp.where(
            signed_quadratic > threshold,
            int(CausalCharacter.TIMELIKE),
            int(CausalCharacter.SPACELIKE),
        ),
    )


def proper_time_rate(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    velocity: ArrayLike,
    /,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
) -> Array:
    """Return ``sqrt(abs(g(v,v)))`` and reject non-timelike velocities."""
    character = causal_character(
        metric,
        coordinates,
        velocity,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    quadratic = metric.quadratic_form(velocity, coordinates)
    quadratic = eqx.error_if(
        quadratic,
        jnp.any(character != int(CausalCharacter.TIMELIKE)),
        "Proper time is defined here only for timelike velocities.",
    )
    return jnp.sqrt(jnp.abs(quadratic))


class TimeOrientation(StrictModule):
    """A declared timelike reference field selecting future and past cones."""

    metric: LorentzianMetric
    vector_field: Callable[[Array], Array]

    def __init__(
        self,
        metric: LorentzianMetric,
        vector_field: Callable[[Array], Array],
        /,
    ):
        if not isinstance(metric, LorentzianMetric):
            raise TypeError("TimeOrientation requires a LorentzianMetric.")
        if not callable(vector_field):
            raise TypeError("Time-orientation vector_field must be callable.")
        self.metric = metric
        self.vector_field = vector_field

    def _oriented_pairing(
        self,
        coordinates: ArrayLike,
        vector: ArrayLike,
        /,
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> Array:
        reference = _pointwise_array(
            self.vector_field,
            coordinates,
            self.metric.chart.dimension,
        )
        if reference.shape[-1:] != (self.metric.chart.dimension,):
            raise ValueError(
                "Time-orientation reference vectors must match the chart dimension."
            )
        reference_character = causal_character(
            self.metric,
            coordinates,
            reference,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        vector_character = causal_character(
            self.metric,
            coordinates,
            vector,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        pairing = self.metric.bilinear(reference, vector, coordinates)
        pairing = eqx.error_if(
            pairing,
            jnp.any(reference_character != int(CausalCharacter.TIMELIKE)),
            "A time-orientation reference field must be timelike.",
        )
        return eqx.error_if(
            pairing,
            jnp.any(vector_character == int(CausalCharacter.SPACELIKE)),
            "Future or past direction is undefined for spacelike vectors.",
        )

    def is_future_directed(
        self,
        coordinates: ArrayLike,
        vector: ArrayLike,
        /,
        *,
        absolute_tolerance: float = 1e-12,
        relative_tolerance: float = 1e-9,
    ) -> Array:
        pairing = self._oriented_pairing(
            coordinates,
            vector,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        return self.metric.timelike_sign * pairing > 0.0

    def is_past_directed(
        self,
        coordinates: ArrayLike,
        vector: ArrayLike,
        /,
        *,
        absolute_tolerance: float = 1e-12,
        relative_tolerance: float = 1e-9,
    ) -> Array:
        pairing = self._oriented_pairing(
            coordinates,
            vector,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        return self.metric.timelike_sign * pairing < 0.0


__all__ = [
    "CausalCharacter",
    "TimeOrientation",
    "causal_character",
    "dalembertian",
    "proper_time_rate",
    "semi_riemannian_gradient",
]
