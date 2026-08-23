#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import RiemannianMetric, WeightedRiemannianMeasure
from ._function import DomainFunction


DensityReference: TypeAlias = Literal[
    "coordinate",
    "riemannian-volume",
    "weighted-riemannian-volume",
]


class ReferencedDensityField(StrictModule):
    """A density field with an explicit reference measure."""

    field: DomainFunction
    metric: RiemannianMetric | None
    measure: WeightedRiemannianMeasure | None
    state_var: str = eqx.field(static=True)
    reference: DensityReference = eqx.field(static=True)

    def __init__(
        self,
        field: DomainFunction,
        /,
        *,
        reference: DensityReference,
        state_var: str,
        metric: RiemannianMetric | None = None,
        measure: WeightedRiemannianMeasure | None = None,
    ):
        if not isinstance(field, DomainFunction):
            raise TypeError("field must be a DomainFunction.")
        if reference not in (
            "coordinate",
            "riemannian-volume",
            "weighted-riemannian-volume",
        ):
            raise ValueError("Unknown density reference.")
        state_var_ = str(state_var)
        if state_var_ not in field.domain.labels:
            raise ValueError("state_var must be one of the density field domain labels.")
        if reference == "coordinate":
            if metric is not None or measure is not None:
                raise ValueError("Coordinate density must not declare a metric measure.")
        elif reference == "riemannian-volume":
            if not isinstance(metric, RiemannianMetric) or measure is not None:
                raise ValueError("Riemannian density requires exactly one metric.")
        elif not isinstance(measure, WeightedRiemannianMeasure) or metric is not None:
            raise ValueError("Weighted density requires exactly one weighted measure.")
        self.field = field
        self.reference = reference
        self.state_var = state_var_
        self.metric = metric
        self.measure = measure

    @property
    def chart_dimension(self) -> int | None:
        if self.metric is not None:
            return self.metric.chart.dimension
        if self.measure is not None:
            return self.measure.chart.dimension
        return None

    def log_reference_density(self, coordinates: ArrayLike, /) -> Array:
        if self.reference == "coordinate":
            points = jnp.asarray(coordinates)
            return jnp.zeros(points.shape[:-1], dtype=points.real.dtype)
        if self.reference == "riemannian-volume":
            assert self.metric is not None
            return self.metric.log_volume_density(coordinates)
        assert self.measure is not None
        return self.measure.log_coordinate_density(coordinates)

    def to_coordinate_value(
        self,
        value: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return jnp.asarray(value) * jnp.exp(self.log_reference_density(coordinates))

    def from_coordinate_value(
        self,
        value: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return jnp.asarray(value) * jnp.exp(-self.log_reference_density(coordinates))


__all__ = ["DensityReference", "ReferencedDensityField"]
