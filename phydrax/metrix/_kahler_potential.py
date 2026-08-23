#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._complex import ComplexCoordinateConvention, wirtinger_derivatives
from ._metric import RiemannianMetric
from ._utils import _pointwise_array


def _realify_hermitian(matrix: Array, /) -> Array:
    real = jnp.real(matrix)
    imaginary = jnp.imag(matrix)
    return jnp.block([[real, -imaginary], [imaginary, real]])


class _KahlerPotentialMetricMap(StrictModule):
    geometry: KahlerPotentialGeometry

    def __init__(self, geometry: KahlerPotentialGeometry, /):
        self.geometry = geometry

    def __call__(self, coordinates: Array, /) -> Array:
        reference = self.geometry.reference_metric(coordinates)
        correction = self.geometry.complex_hessian(coordinates)
        return reference + _realify_hermitian(correction)


class KahlerPotentialGeometry(StrictModule):
    """Metric correction ``g = g0 + partial partial_bar phi``."""

    reference_metric: RiemannianMetric
    convention: ComplexCoordinateConvention
    potential_function: Callable[[Array], Array]

    def __init__(
        self,
        reference_metric: RiemannianMetric,
        convention: ComplexCoordinateConvention,
        potential: Callable[[Array], Array],
        /,
    ):
        if not isinstance(reference_metric, RiemannianMetric):
            raise TypeError("reference_metric must be a RiemannianMetric.")
        if not isinstance(convention, ComplexCoordinateConvention):
            raise TypeError("convention must be a ComplexCoordinateConvention.")
        if not reference_metric.chart.compatible_with(convention.chart):
            raise ValueError("Reference metric and complex convention charts must match.")
        if not callable(potential):
            raise TypeError("potential must be callable.")
        self.reference_metric = reference_metric
        self.convention = convention
        self.potential_function = potential

    def _potential_point(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.potential_function(coordinates))
        if value.shape != () or jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError("Kähler potential must be real scalar-valued.")
        return value

    def complex_hessian(self, coordinates: ArrayLike, /) -> Array:
        def evaluate(point: Array) -> Array:
            def holomorphic_gradient(local: Array) -> Array:
                return wirtinger_derivatives(
                    self._potential_point, self.convention, local
                )[0]

            return wirtinger_derivatives(holomorphic_gradient, self.convention, point)[1]

        return _pointwise_array(
            evaluate,
            coordinates,
            self.convention.chart.dimension,
        )

    def metric(self) -> RiemannianMetric:
        return RiemannianMetric(
            _KahlerPotentialMetricMap(self), chart=self.convention.chart
        )

    def hermitian_matrix(self, coordinates: ArrayLike, /) -> Array:
        metric = self.metric()(coordinates)
        dimension = self.convention.complex_dimension
        real = metric[..., :dimension, :dimension]
        imaginary = metric[..., dimension:, :dimension]
        return real + 1j * imaginary

    def positivity_margin(self, coordinates: ArrayLike, /) -> Array:
        hermitian = self.hermitian_matrix(coordinates)
        hermitian = 0.5 * (hermitian + jnp.swapaxes(jnp.conj(hermitian), -1, -2))
        return jnp.min(jnp.linalg.eigvalsh(hermitian), axis=-1)

    def log_determinant(self, coordinates: ArrayLike, /) -> Array:
        return jnp.linalg.slogdet(self.hermitian_matrix(coordinates))[1]

    def monge_ampere_residual(
        self,
        target_log_volume: Callable[[Array], Array],
        coordinates: ArrayLike,
        /,
        *,
        normalization: ArrayLike = 0.0,
    ) -> Array:
        if not callable(target_log_volume):
            raise TypeError("target_log_volume must be callable.")
        points = jnp.asarray(coordinates)
        target = _pointwise_array(
            target_log_volume,
            points,
            self.convention.chart.dimension,
        )
        return self.log_determinant(points) - target - jnp.asarray(normalization)


__all__ = ["KahlerPotentialGeometry"]
