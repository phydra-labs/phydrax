#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._measure_partition import GeometryMeasurePartition


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
            + (1.0 - first) * second * (vertices[..., 2, :] - vertices[..., 0, :])
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


__all__ = ["CADChartAtlas"]
