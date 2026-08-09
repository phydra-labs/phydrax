#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .._strict import StrictModule
from ..metrix import (
    metric_volume_density,
    RiemannianMetric,
    VolumeDensity,
)
from ._base import AbstractGeometry
from ._components import DomainComponent
from ._function import DomainFunction
from ._selection import Interior


class _VolumeDensityCallable(StrictModule):
    density: VolumeDensity

    def __init__(self, density: VolumeDensity, /):
        self.density = density

    def __call__(self, coordinates: Any, /, *, key=None, **kwargs: Any):
        del key, kwargs
        if isinstance(coordinates, tuple):
            axes = tuple(jnp.asarray(axis).reshape((-1,)) for axis in coordinates)
            if len(axes) != self.density.chart.dimension:
                raise ValueError(
                    "Coordinate-separable volume density received the wrong number "
                    "of coordinate axes."
                )
            grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
            return self.density(grid)
        return self.density(coordinates)


def with_volume_density(
    component: DomainComponent,
    density: VolumeDensity,
    /,
    *,
    var: str | None = None,
) -> DomainComponent:
    """Multiply an interior component's coordinate weight by a volume density."""
    if not isinstance(density, VolumeDensity):
        raise TypeError("density must be a VolumeDensity.")
    if var is None:
        geometry_labels = []
        for label in component.domain.labels:
            factor = component.domain.factor(label)
            if isinstance(factor, AbstractGeometry):
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
            f"Unknown density variable {var_!r}; expected one of {component.domain.labels}."
        )
    factor = component.domain.factor(var_)
    if not isinstance(factor, AbstractGeometry):
        raise ValueError("Volume density requires a geometry domain factor.")
    if int(factor.spatial_dim) != density.chart.dimension:
        raise ValueError(
            f"Density chart dimension {density.chart.dimension} does not match "
            f"domain variable {var_!r} dimension {factor.spatial_dim}."
        )
    if not isinstance(component.spec.selection_for(var_), Interior):
        raise ValueError(
            "Volume density currently supports interior geometry components only."
        )
    density_weight = DomainFunction(
        domain=component.domain,
        deps=(var_,),
        func=_VolumeDensityCallable(density),
        metadata={},
    )
    return component.with_density(density_weight)


def with_riemannian_measure(
    component: DomainComponent,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainComponent:
    """Attach the canonical Riemannian volume density to an interior component."""
    return with_volume_density(
        component,
        metric_volume_density(metric),
        var=var,
    )
