#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import CoordinateChart
from ._metric import RiemannianMetric
from ._utils import _pointwise_array, _pointwise_jacfwd


def tangent_projector_from_normal(normal: ArrayLike, /) -> Array:
    """Euclidean tangent projector associated with a nonzero normal vector."""

    values = jnp.asarray(normal)
    if values.ndim < 1:
        raise ValueError("A normal vector must have at least one axis.")
    norm = jnp.linalg.norm(values, axis=-1, keepdims=True)
    values = eqx.error_if(values, jnp.any(norm == 0), "Normal vectors must be nonzero.")
    unit = values / norm
    identity = jnp.eye(values.shape[-1], dtype=values.dtype)
    return identity - oe.contract("...i,...j->...ij", unit, unit)


class _InducedMetricMap(StrictModule):
    embedding: Callable[[Array], Array]

    def __init__(self, embedding: Callable[[Array], Array], /):
        self.embedding = embedding

    def __call__(self, coordinates: Array, /) -> Array:
        jacobian = jax.jacfwd(self.embedding)(coordinates)
        return oe.contract("ai,aj->ij", jacobian, jacobian)


class EmbeddedChart(StrictModule):
    """One local parameterization of a manifold embedded in Euclidean space."""

    chart: CoordinateChart
    embedding: Callable[[Array], Array]
    ambient_dimension: int
    retraction: Callable[[Array], Array] | None

    def __init__(
        self,
        chart: CoordinateChart,
        embedding: Callable[[Array], Array],
        ambient_dimension: int,
        /,
        *,
        retraction: Callable[[Array], Array] | None = None,
    ):
        ambient_dimension_ = int(ambient_dimension)
        if ambient_dimension_ < chart.dimension:
            raise ValueError(
                "Embedded ambient dimension must be at least the chart dimension."
            )
        if not callable(embedding):
            raise TypeError("Embedding must be callable.")
        if retraction is not None and not callable(retraction):
            raise TypeError("Retraction must be callable when supplied.")
        self.chart = chart
        self.embedding = embedding
        self.ambient_dimension = ambient_dimension_
        self.retraction = retraction

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        values = _pointwise_array(
            self.embedding,
            coordinates,
            self.chart.dimension,
        )
        if values.shape[-1:] != (self.ambient_dimension,):
            raise ValueError(
                "Embedding output must have trailing dimension "
                f"{self.ambient_dimension}; got {values.shape}."
            )
        return values

    def tangent_basis(self, coordinates: ArrayLike, /) -> Array:
        """Return embedding Jacobians with shape ``(..., ambient, intrinsic)``."""

        basis = _pointwise_jacfwd(
            self.embedding,
            coordinates,
            self.chart.dimension,
        )
        expected = (self.ambient_dimension, self.chart.dimension)
        if basis.shape[-2:] != expected:
            raise ValueError(
                f"Embedding Jacobian must have trailing shape {expected}; got {basis.shape}."
            )
        return basis

    def embedding_hessian(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            jax.jacfwd(jax.jacfwd(self.embedding)),
            coordinates,
            self.chart.dimension,
        )

    def induced_metric(self) -> RiemannianMetric:
        return RiemannianMetric(
            _InducedMetricMap(self.embedding),
            chart=self.chart,
        )

    def volume_density(self, coordinates: ArrayLike, /) -> Array:
        return self.induced_metric().volume_density(coordinates)

    def tangent_projector(self, coordinates: ArrayLike, /) -> Array:
        basis = self.tangent_basis(coordinates)
        inverse = self.induced_metric().inverse(coordinates)
        return oe.contract("...ai,...ij,...bj->...ab", basis, inverse, basis)

    def normal_projector(self, coordinates: ArrayLike, /) -> Array:
        tangent = self.tangent_projector(coordinates)
        identity = jnp.eye(self.ambient_dimension, dtype=tangent.dtype)
        return identity - tangent

    def project_tangent(
        self,
        vector: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(vector)
        points = jnp.asarray(coordinates)
        expected = points.shape[:-1] + (self.ambient_dimension,)
        if values.shape != expected:
            raise ValueError(
                f"Ambient vector must have shape {expected}; got {values.shape}."
            )
        return oe.contract("...ab,...b->...a", self.tangent_projector(points), values)

    def project_normal(
        self,
        vector: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(vector)
        points = jnp.asarray(coordinates)
        expected = points.shape[:-1] + (self.ambient_dimension,)
        if values.shape != expected:
            raise ValueError(
                f"Ambient vector must have shape {expected}; got {values.shape}."
            )
        return oe.contract("...ab,...b->...a", self.normal_projector(points), values)

    def retract(self, ambient_points: ArrayLike, /) -> Array:
        if self.retraction is None:
            raise ValueError("This embedded chart does not provide a retraction.")
        values = _pointwise_array(
            self.retraction,
            ambient_points,
            self.ambient_dimension,
        )
        if values.shape[-1:] != (self.ambient_dimension,):
            raise ValueError(
                "Retraction output must have trailing dimension "
                f"{self.ambient_dimension}; got {values.shape}."
            )
        return values

    def second_fundamental_form(self, coordinates: ArrayLike, /) -> Array:
        """Normal-valued second fundamental form ``B[..., ambient, i, j]``."""

        hessian = self.embedding_hessian(coordinates)
        normal = self.normal_projector(coordinates)
        return oe.contract("...ab,...bij->...aij", normal, hessian)

    def mean_curvature_vector(self, coordinates: ArrayLike, /) -> Array:
        form = self.second_fundamental_form(coordinates)
        inverse = self.induced_metric().inverse(coordinates)
        trace = oe.contract("...ij,...aij->...a", inverse, form)
        return trace / float(self.chart.dimension)

    def shape_operator(
        self,
        normal: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        """Shape operator associated with a caller-supplied unit normal."""

        normal_ = jnp.asarray(normal)
        points = jnp.asarray(coordinates)
        expected = points.shape[:-1] + (self.ambient_dimension,)
        if normal_.shape != expected:
            raise ValueError(f"Normal must have shape {expected}; got {normal_.shape}.")
        second_form = oe.contract(
            "...a,...aij->...ij",
            normal_,
            self.second_fundamental_form(points),
        )
        return oe.contract(
            "...ik,...kj->...ij",
            self.induced_metric().inverse(points),
            second_form,
        )
