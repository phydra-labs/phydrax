#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._density import metric_volume_density, VolumeDensity
from ._map import Immersion
from ._metric import pullback_metric, RiemannianMetric
from ._utils import _pointwise_array


class RiemannianHypersurface(StrictModule):
    """An oriented hypersurface conormal in one Riemannian chart."""

    metric: RiemannianMetric
    conormal_function: Callable[[Array], Array]

    def __init__(
        self,
        metric: RiemannianMetric,
        conormal: Callable[[Array], Array],
        /,
    ):
        if not isinstance(metric, RiemannianMetric):
            raise TypeError("RiemannianHypersurface requires a RiemannianMetric.")
        if not callable(conormal):
            raise TypeError("conormal must be callable.")
        self.metric = metric
        self.conormal_function = conormal

    @property
    def chart(self):
        return self.metric.chart

    def _conormal_point(self, coordinates: Array, /) -> Array:
        covector = jnp.asarray(self.conormal_function(coordinates))
        expected = (self.chart.dimension,)
        if covector.shape != expected:
            raise ValueError(
                f"Pointwise hypersurface conormal must have shape {expected}; "
                f"got {covector.shape}."
            )
        return covector

    def conormal(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._conormal_point,
            coordinates,
            self.chart.dimension,
        )

    def conormal_norm_squared(self, coordinates: ArrayLike, /) -> Array:
        covector = self.conormal(coordinates)
        return oe.contract(
            "...i,...ij,...j->...",
            covector,
            self.metric.inverse(coordinates),
            covector,
        )

    def unit_conormal(self, coordinates: ArrayLike, /) -> Array:
        squared = self.conormal_norm_squared(coordinates)
        squared = eqx.error_if(
            squared,
            jnp.any(~jnp.isfinite(squared) | (squared <= 0)),
            "A Riemannian hypersurface conormal must have positive finite norm.",
        )
        return self.conormal(coordinates) / jnp.sqrt(squared)[..., None]

    def unit_normal(self, coordinates: ArrayLike, /) -> Array:
        return self.metric.sharp(self.unit_conormal(coordinates), coordinates)

    def tangent_projector(self, coordinates: ArrayLike, /) -> Array:
        normal = self.unit_normal(coordinates)
        conormal = self.unit_conormal(coordinates)
        identity = jnp.eye(self.chart.dimension, dtype=normal.dtype)
        return identity - oe.contract("...i,...j->...ij", normal, conormal)

    def project_tangent(
        self,
        vector: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(vector)
        if values.shape[-1:] != (self.chart.dimension,):
            raise ValueError(
                "Hypersurface tangent projection requires the chart trailing dimension."
            )
        return oe.contract(
            "...ij,...j->...i", self.tangent_projector(coordinates), values
        )


def induced_boundary_metric(
    ambient_metric: RiemannianMetric,
    parameterization: Immersion,
    /,
) -> RiemannianMetric:
    """Return the metric induced by a codimension-one parameterization."""
    if not isinstance(ambient_metric, RiemannianMetric):
        raise TypeError("ambient_metric must be a RiemannianMetric.")
    if not isinstance(parameterization, Immersion):
        raise TypeError("parameterization must be an Immersion.")
    if not parameterization.target.compatible_with(ambient_metric.chart):
        raise ValueError("Boundary parameterization target must match the metric chart.")
    if parameterization.source.dimension + 1 != parameterization.target.dimension:
        raise ValueError("Boundary parameterization must have codimension one.")
    return pullback_metric(ambient_metric, parameterization)


def induced_boundary_density(
    ambient_metric: RiemannianMetric,
    parameterization: Immersion,
    /,
) -> VolumeDensity:
    """Return the positive density of an induced boundary metric."""
    return metric_volume_density(
        induced_boundary_metric(ambient_metric, parameterization)
    )


__all__ = [
    "RiemannianHypersurface",
    "induced_boundary_density",
    "induced_boundary_metric",
]
