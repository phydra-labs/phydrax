#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from .._tensor_entities import StructuredAxis, TensorEntityLayout
from ._assignment import (
    _tensor_product_state,
    _uniform_axis_stencil,
    _uniform_spacing,
    AbstractStructuredSplatAssignment,
    SplatAssignmentCapabilities,
    SplatAssignmentState,
)


class TensorBSplineSplatAssignment(AbstractStructuredSplatAssignment):
    """Uniform tensor-product cardinal B-spline assignment of degree one to three."""

    degree: int = eqx.field(static=True)
    capabilities: SplatAssignmentCapabilities = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(self, degree: int, /):
        degree_ = int(degree)
        if degree_ not in (1, 2, 3):
            raise ValueError(
                "Tensor B-spline splatting supports degrees one, two, and three."
            )
        capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1,
            maximum_explicit_derivative_order=1,
            supports_nonuniform=False,
            supports_mixed_entities=True,
        )
        self.degree = degree_
        self.capabilities = capabilities
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "tensor-bspline-splat-assignment",
                "degree": degree_,
                "capabilities": capabilities.capability_id,
            }
        )

    def route_width(self, dimension: int, /) -> int:
        dimension_ = int(dimension)
        if dimension_ <= 0:
            raise ValueError("Splat assignment dimension must be positive.")
        return (self.degree + 1) ** dimension_

    def validate(
        self,
        layout: TensorEntityLayout,
        axes: tuple[StructuredAxis, ...],
        /,
    ) -> None:
        if len(axes) != len(layout.shape):
            raise ValueError("Assignment axes must match the target layout dimension.")
        for coordinates, axis in zip(layout.coordinates_by_axis, axes, strict=True):
            if int(coordinates.size) < self.degree + 1:
                raise ValueError(
                    "B-spline target axes need at least degree plus one entities."
                )
            bounds = (
                float(np.asarray(axis.bounds)[0]),
                float(np.asarray(axis.bounds)[1]),
            )
            _uniform_spacing(coordinates, bounds, axis.periodic)

    def build(
        self,
        layout: TensorEntityLayout,
        axes: tuple[StructuredAxis, ...],
        axis_bounds: tuple[tuple[float, float], ...],
        position: Array,
        active: Array,
        /,
    ) -> SplatAssignmentState:
        stencils = tuple(
            _uniform_axis_stencil(
                self.degree,
                coordinates,
                bounds,
                axis.periodic,
                position[:, index],
                active,
            )
            for index, (coordinates, bounds, axis) in enumerate(
                zip(layout.coordinates_by_axis, axis_bounds, axes, strict=True)
            )
        )
        return _tensor_product_state(stencils, layout.shape, active)


__all__ = ["TensorBSplineSplatAssignment"]
