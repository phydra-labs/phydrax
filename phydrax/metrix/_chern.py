#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._complex import ComplexCoordinateConvention, wirtinger_derivatives
from ._utils import _pointwise_array


class HolomorphicBundleFrame(StrictModule):
    """One local holomorphic frame with a Hermitian fiber metric."""

    convention: ComplexCoordinateConvention
    hermitian_metric_function: Callable[[Array], Array]
    fiber_dimension: int
    frame_id: str

    def __init__(
        self,
        convention: ComplexCoordinateConvention,
        hermitian_metric: Callable[[Array], Array],
        fiber_dimension: int,
        /,
        *,
        frame_id: str,
    ):
        if not isinstance(convention, ComplexCoordinateConvention):
            raise TypeError("convention must be a ComplexCoordinateConvention.")
        if not callable(hermitian_metric):
            raise TypeError("hermitian_metric must be callable.")
        dimension = int(fiber_dimension)
        if dimension < 1:
            raise ValueError("fiber_dimension must be positive.")
        identifier = str(frame_id)
        if not identifier:
            raise ValueError("frame_id must be non-empty.")
        self.convention = convention
        self.hermitian_metric_function = hermitian_metric
        self.fiber_dimension = dimension
        self.frame_id = identifier

    def _metric_point(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.hermitian_metric_function(coordinates))
        expected = (self.fiber_dimension, self.fiber_dimension)
        if value.shape != expected:
            raise ValueError(f"Hermitian bundle metric must have shape {expected}.")
        return value

    def metric(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._metric_point,
            coordinates,
            self.convention.chart.dimension,
        )

    def minimum_eigenvalue(self, coordinates: ArrayLike, /) -> Array:
        metric = self.metric(coordinates)
        hermitian = 0.5 * (metric + jnp.swapaxes(jnp.conj(metric), -1, -2))
        return jnp.min(jnp.linalg.eigvalsh(hermitian), axis=-1)


class ChernConnection(StrictModule):
    """Local Chern connection in a holomorphic bundle frame."""

    frame: HolomorphicBundleFrame

    def __init__(self, frame: HolomorphicBundleFrame, /):
        if not isinstance(frame, HolomorphicBundleFrame):
            raise TypeError("frame must be a HolomorphicBundleFrame.")
        self.frame = frame

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        metric = self.frame._metric_point(coordinates)
        partial_metric, _ = wirtinger_derivatives(
            self.frame._metric_point,
            self.frame.convention,
            coordinates,
        )
        inverse = jnp.linalg.inv(metric)
        return oe.contract("ab,bci->aci", inverse, partial_metric)

    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._coefficients_point,
            coordinates,
            self.frame.convention.chart.dimension,
        )

    def _curvature_point(self, coordinates: Array, /) -> Array:
        _, partial_bar_coefficients = wirtinger_derivatives(
            self._coefficients_point,
            self.frame.convention,
            coordinates,
        )
        return partial_bar_coefficients

    def curvature(self, coordinates: ArrayLike, /) -> Array:
        """Return local `(1,1)` curvature components ``F[a,b,i,jbar]``."""
        return _pointwise_array(
            self._curvature_point,
            coordinates,
            self.frame.convention.chart.dimension,
        )

    def first_chern_form(self, coordinates: ArrayLike, /) -> Array:
        return (
            1j
            * jnp.trace(self.curvature(coordinates), axis1=-4, axis2=-3)
            / (2.0 * jnp.pi)
        )


class HolomorphicBundleTransition(StrictModule):
    """Holomorphic frame transition and Hermitian metric compatibility."""

    source: HolomorphicBundleFrame
    target: HolomorphicBundleFrame
    gauge_function: Callable[[Array], Array]

    def __init__(
        self,
        source: HolomorphicBundleFrame,
        target: HolomorphicBundleFrame,
        gauge: Callable[[Array], Array],
        /,
    ):
        if not isinstance(source, HolomorphicBundleFrame) or not isinstance(
            target, HolomorphicBundleFrame
        ):
            raise TypeError("source and target must be HolomorphicBundleFrame objects.")
        if source.fiber_dimension != target.fiber_dimension:
            raise ValueError("Bundle frame dimensions must match.")
        if not callable(gauge):
            raise TypeError("gauge must be callable.")
        self.source = source
        self.target = target
        self.gauge_function = gauge

    def metric_residual(
        self,
        source_coordinates: ArrayLike,
        target_coordinates: ArrayLike,
        /,
    ) -> Array:
        source_metric = self.source.metric(source_coordinates)
        target_metric = self.target.metric(target_coordinates)
        gauge = jnp.asarray(self.gauge_function(source_coordinates))
        transformed = jnp.swapaxes(jnp.conj(gauge), -1, -2) @ target_metric @ gauge
        return jnp.max(jnp.abs(source_metric - transformed), axis=(-2, -1))


__all__ = [
    "ChernConnection",
    "HolomorphicBundleFrame",
    "HolomorphicBundleTransition",
]
