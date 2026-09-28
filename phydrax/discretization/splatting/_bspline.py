#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from .._tensor_entities import StructuredAxis, TensorEntityLayout
from .._tensor_index import TensorIndexLayout
from ._assignment import (
    _tensor_product_state,
    _uniform_axis_stencil,
    _uniform_spacing,
    AbstractStructuredSplatAssignment,
    SplatAssignmentCapabilities,
    SplatAssignmentState,
)


class TensorBSplineSplatAssignment(AbstractStructuredSplatAssignment):
    """Uniform tensor-product cardinal B-spline assignment of degree zero to three.

    ``degree`` is one degree for every axis or a per-axis degree tuple (the
    mixed-degree tensor splines of spline-Whitney forms). Degree zero is the
    cell box spline, allowed per axis only next to a positive degree.
    """

    degree: int | tuple[int, ...] = eqx.field(static=True)
    capabilities: SplatAssignmentCapabilities = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(self, degree: int | Sequence[int], /) -> None:
        degree_: int | tuple[int, ...] = (
            int(degree)
            if isinstance(degree, (int, np.integer))
            else tuple(int(value) for value in degree)
        )
        degrees = (degree_,) if isinstance(degree_, int) else degree_
        if (
            not degrees
            or any(value not in (0, 1, 2, 3) for value in degrees)
            or max(degrees) == 0
        ):
            raise ValueError(
                "Tensor B-spline splatting supports degrees zero to three with at "
                "least one positive degree."
            )
        capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1,
            maximum_explicit_derivative_order=1,
            supports_nonuniform=False,
            supports_mixed_entities=True,
            maximum_support_radius_cells=0.5 * (max(degrees) + 1),
        )
        self.degree = degree_
        self.capabilities = capabilities
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "tensor-bspline-splat-assignment",
                "degree": degree_ if isinstance(degree_, int) else list(degree_),
                "capabilities": capabilities.capability_id,
            }
        )

    def _axis_degrees(self, dimension: int, /) -> tuple[int, ...]:
        if isinstance(self.degree, int):
            return (self.degree,) * dimension
        if len(self.degree) != dimension:
            raise ValueError("Per-axis B-spline degrees must match the dimension.")
        return self.degree

    def route_width(self, dimension: int, /) -> int:
        dimension_ = int(dimension)
        if dimension_ <= 0:
            raise ValueError("Splat assignment dimension must be positive.")
        return int(np.prod([value + 1 for value in self._axis_degrees(dimension_)]))

    def validate(
        self,
        layout: TensorEntityLayout | TensorIndexLayout,
        axes: tuple[StructuredAxis, ...],
        /,
    ) -> None:
        if len(axes) != len(layout.shape):
            raise ValueError("Assignment axes must match the target layout dimension.")
        degrees = self._axis_degrees(len(axes))
        for coordinates, axis, degree in zip(
            layout.coordinates_by_axis, axes, degrees, strict=True
        ):
            if coordinates.size < degree + 1:
                raise ValueError(
                    "B-spline target axes need at least degree plus one entities."
                )
            bounds = (
                float(np.asarray(axis.bounds)[0]),
                float(np.asarray(axis.bounds)[1]),
            )
            _uniform_spacing(coordinates, bounds, axis.periodic)

    def validate_input(
        self, assignment_input: object, source_count: int, dimension: int, /
    ) -> None:
        del source_count, dimension
        if assignment_input is not None:
            raise ValueError("Tensor B-spline assignment accepts no domain input.")

    def update_input(
        self,
        position: Array,
        deformation_gradient: Array,
        committed_input: object,
        /,
    ) -> object:
        del position, deformation_gradient, committed_input
        return None

    def build(
        self,
        layout: TensorEntityLayout | TensorIndexLayout,
        axes: tuple[StructuredAxis, ...],
        axis_bounds: tuple[tuple[float, float], ...],
        position: Array,
        active: Array,
        *,
        assignment_input: object = None,
    ) -> SplatAssignmentState:
        self.validate_input(assignment_input, position.shape[0], position.shape[1])
        stencils = tuple(
            _uniform_axis_stencil(
                degree,
                coordinates,
                bounds,
                axis.periodic,
                position[:, index],
                active,
            )
            for index, (coordinates, bounds, axis, degree) in enumerate(
                zip(
                    layout.coordinates_by_axis,
                    axis_bounds,
                    axes,
                    self._axis_degrees(len(axes)),
                    strict=True,
                )
            )
        )
        return _tensor_product_state(stencils, layout.shape, active)


__all__ = ["TensorBSplineSplatAssignment"]
