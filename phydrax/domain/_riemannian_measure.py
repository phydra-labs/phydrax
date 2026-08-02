#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .._strict import StrictModule
from ..metrix import RiemannianMetric
from ._base import _AbstractGeometry
from ._components import DomainComponent, Interior
from ._domain import RelabeledDomain
from ._function import DomainFunction


class _MetricVolumeDensityCallable(StrictModule):
    metric: RiemannianMetric

    def __init__(self, metric: RiemannianMetric, /):
        self.metric = metric

    def __call__(self, coordinates: Any, /, *, key=None, **kwargs: Any):
        del key, kwargs
        if isinstance(coordinates, tuple):
            axes = tuple(jnp.asarray(axis).reshape((-1,)) for axis in coordinates)
            if len(axes) != self.metric.chart.dimension:
                raise ValueError(
                    "Coordinate-separable metric weight received the wrong number "
                    "of coordinate axes."
                )
            grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
            return self.metric.volume_density(grid)
        return self.metric.volume_density(coordinates)


def with_riemannian_measure(
    component: DomainComponent,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainComponent:
    """Multiply an interior component's coordinate weight by ``sqrt(det(metric))``."""

    if var is None:
        geometry_labels = []
        for label in component.domain.labels:
            factor = component.domain.factor(label)
            factor = factor.base if isinstance(factor, RelabeledDomain) else factor
            if isinstance(factor, _AbstractGeometry):
                geometry_labels.append(label)
        if len(geometry_labels) != 1:
            raise ValueError(
                "var=None requires exactly one geometry factor; found "
                f"{tuple(geometry_labels)}."
            )
        var_ = geometry_labels[0]
    else:
        var_ = str(var)
    if var_ not in component.domain.labels:
        raise ValueError(
            f"Unknown metric variable {var_!r}; expected one of {component.domain.labels}."
        )
    factor = component.domain.factor(var_)
    factor = factor.base if isinstance(factor, RelabeledDomain) else factor
    if not isinstance(factor, _AbstractGeometry):
        raise ValueError("Riemannian measure requires a geometry domain factor.")
    if int(factor.var_dim) != metric.chart.dimension:
        raise ValueError(
            f"Metric chart dimension {metric.chart.dimension} does not match "
            f"domain variable {var_!r} dimension {factor.var_dim}."
        )
    if not isinstance(component.spec.component_for(var_), Interior):
        raise ValueError(
            "Riemannian measure currently supports interior geometry components only."
        )
    metric_weight = DomainFunction(
        domain=component.domain,
        deps=(var_,),
        func=_MetricVolumeDensityCallable(metric),
        metadata={},
    )
    weight = (
        metric_weight
        if component.weight_all is None
        else component.weight_all * metric_weight
    )
    return DomainComponent(
        domain=component.domain,
        spec=component.spec,
        where=component.where,
        where_all=component.where_all,
        weight_all=weight,
    )
