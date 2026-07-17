#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._strict import StrictModule
from ._measure_partition import GeometryMeasurePartition


class CADChartQuadrature(StrictModule):
    """Fixed tensor quadrature mapped through every chart of a CAD atlas."""

    reference_axes: tuple[Array, ...]
    points: Array
    weights: Array
    trim_mask: Array

    def __init__(
        self,
        reference_axes: tuple[Array, ...],
        points: Array,
        weights: Array,
        trim_mask: Array,
    ):
        points_ = jnp.asarray(points, dtype=float)
        weights_ = jnp.asarray(weights, dtype=float)
        trim_ = jnp.asarray(trim_mask, dtype=bool)
        if points_.shape[:-1] != weights_.shape:
            raise ValueError("Chart points and weights must have matching logical shapes.")
        if trim_.shape != weights_.shape:
            raise ValueError("Chart trim mask must match the weight shape.")
        if len(reference_axes) != weights_.ndim - 1:
            raise ValueError("Reference axes must match the chart-local tensor rank.")
        self.reference_axes = tuple(
            jnp.asarray(axis, dtype=float).reshape((-1,)) for axis in reference_axes
        )
        self.points = points_
        self.weights = weights_
        self.trim_mask = trim_

    def integrate(self, values: Array, /) -> Array:
        """Integrate chart-local scalar values, including Jacobian and trim weights."""
        values_ = jnp.asarray(values, dtype=float)
        if values_.shape != self.weights.shape:
            raise ValueError(
                f"Chart values must have shape {self.weights.shape}, got {values_.shape}."
            )
        return jnp.sum(jnp.where(self.trim_mask, self.weights * values_, 0.0))


class CADChartAtlas(StrictModule):
    """Affine segment or Duffy-triangle charts for a CAD boundary mesh."""

    partition: GeometryMeasurePartition

    def __init__(self, partition: GeometryMeasurePartition):
        self.partition = partition

    @property
    def num_charts(self) -> int:
        return self.partition.num_strata

    @property
    def reference_dim(self) -> int:
        return 1 if self.partition.kind == "segment" else 2

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        """Map chart-local coordinates from the unit interval or unit square."""
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=float)
        if reference_.shape[:-1] != indices.shape:
            raise ValueError("chart_indices must match reference leading dimensions.")
        if reference_.shape[-1] != self.reference_dim:
            raise ValueError(
                f"reference must have trailing dimension {self.reference_dim}."
            )
        vertices = self.partition.vertices[indices]
        if self.partition.kind == "segment":
            coordinate = reference_[..., :1]
            return vertices[..., 0, :] + coordinate * (
                vertices[..., 1, :] - vertices[..., 0, :]
            )
        first = reference_[..., :1]
        second = reference_[..., 1:2]
        return (
            vertices[..., 0, :]
            + first * (vertices[..., 1, :] - vertices[..., 0, :])
            + (1.0 - first)
            * second
            * (vertices[..., 2, :] - vertices[..., 0, :])
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        """Return the physical measure Jacobian of the chart map."""
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=float)
        if reference_.shape[:-1] != indices.shape:
            raise ValueError("chart_indices must match reference leading dimensions.")
        measure = self.partition.measures[indices]
        if self.partition.kind == "segment":
            return measure
        return 2.0 * measure * (1.0 - reference_[..., 0])

    def tensor_quadrature(self, order: int, /) -> CADChartQuadrature:
        """Build Gauss-Legendre tensor quadrature on every chart.

        Triangle charts use a Duffy map from the unit square, so every tensor node
        is valid. Jacobian and Gaussian weights are included in the returned rule.
        """
        n = int(order)
        if n <= 0:
            raise ValueError("Chart quadrature order must be positive.")
        nodes, axis_weights_np = np.polynomial.legendre.leggauss(n)
        axis = jnp.asarray(0.5 * (nodes + 1.0), dtype=float)
        axis_weights = jnp.asarray(0.5 * axis_weights_np, dtype=float)
        chart_count = self.num_charts
        if self.partition.kind == "segment":
            reference = jnp.broadcast_to(axis[None, :, None], (chart_count, n, 1))
            indices = jnp.broadcast_to(
                jnp.arange(chart_count, dtype=jnp.int32)[:, None],
                (chart_count, n),
            )
            points = self.map(indices, reference)
            weights = self.jacobian(indices, reference) * axis_weights[None, :]
            trim = jnp.ones(weights.shape, dtype=bool)
            return CADChartQuadrature((axis,), points, weights, trim)

        first, second = jnp.meshgrid(axis, axis, indexing="ij")
        local = jnp.stack((first, second), axis=-1)
        reference = jnp.broadcast_to(
            local[None, ...],
            (chart_count,) + local.shape,
        )
        indices = jnp.broadcast_to(
            jnp.arange(chart_count, dtype=jnp.int32)[:, None, None],
            (chart_count, n, n),
        )
        points = self.map(indices, reference)
        weights = (
            self.jacobian(indices, reference)
            * axis_weights[None, :, None]
            * axis_weights[None, None, :]
        )
        trim = jnp.ones(weights.shape, dtype=bool)
        return CADChartQuadrature((axis, axis), points, weights, trim)


__all__ = ["CADChartAtlas", "CADChartQuadrature"]
