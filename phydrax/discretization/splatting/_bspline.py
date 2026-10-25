#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import comb, factorial
from numbers import Integral

import equinox as eqx
import numpy as np
from jax import Array
from numpy.typing import NDArray

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


_MAXIMUM_DEGREE = 5


def _checked_degree(value: object, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("B-spline degrees must be integers.")
    degree = int(value)
    if not 0 <= degree <= _MAXIMUM_DEGREE:
        raise ValueError(
            "Tensor B-spline splatting supports degrees zero to five (orders one to six)."
        )
    return degree


class TensorBSplineSplatAssignment(AbstractStructuredSplatAssignment):
    """Uniform tensor-product cardinal B-spline assignment of degree zero to five.

    ``degree`` is one degree for every axis or a per-axis degree tuple (the
    mixed-degree tensor splines of spline-Whitney forms). Degree ``p`` is the
    centered cardinal B-spline of order ``p + 1``: ``p + 1`` routes per axis,
    support radius ``(p + 1) / 2`` cells. Degree zero is the cell box spline,
    allowed per axis only next to a positive degree.

    Weights are nonnegative and partition unity. On every positive-degree axis
    the first route moment vanishes, so gathering nodal samples reproduces
    affine fields; a degree-zero axis keeps the nearest-entity offset, so a
    tensor containing one reproduces constants only. ``weight_gradients`` are
    the classical first derivatives inside each polynomial piece. A degree-``p``
    axis basis is ``C^(p-1)`` across its knots (integers for odd ``p``,
    half-integers for even ``p``, relative to the entity spacing), see
    ``continuity_orders``: the ``k``-th position derivative of a deposit or
    gather is continuous only for ``k <= p - 1`` and the ``p``-th jumps at knots.
    No derivative beyond the first is materialized by the assignment.
    """

    degree: int | tuple[int, ...] = eqx.field(static=True)
    capabilities: SplatAssignmentCapabilities = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(self, degree: int | Sequence[int], /) -> None:
        degree_: int | tuple[int, ...]
        if isinstance(degree, Integral) and not isinstance(degree, bool):
            degree_ = _checked_degree(degree)
            degrees: tuple[int, ...] = (degree_,)
        elif isinstance(degree, Sequence) and not isinstance(degree, str):
            degree_ = tuple(_checked_degree(value) for value in degree)
            degrees = degree_
        else:
            raise TypeError(
                "degree must be an integer or a sequence of per-axis integers."
            )
        if not degrees or max(degrees) == 0:
            raise ValueError(
                "Tensor B-spline splatting needs at least one positive degree."
            )
        capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1 if min(degrees) >= 1 else 0,
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

    def continuity_orders(self, dimension: int, /) -> tuple[int, ...]:
        """Per-axis basis continuity ``p - 1`` (``-1``: discontinuous box)."""
        dimension_ = int(dimension)
        if dimension_ <= 0:
            raise ValueError("Splat assignment dimension must be positive.")
        return tuple(degree - 1 for degree in self._axis_degrees(dimension_))

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
                    "B-spline target axes need at least degree plus one entities "
                    f"(degree {degree} has {coordinates.size})."
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


def _cardinal_node_values(
    degree: int, /
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Exact samples ``N_p(j)`` at the integers ``|j| < (p + 1) / 2``."""
    if degree == 0:
        # The truncated-power form is ill-defined at degree zero (0**0).
        return np.zeros((1,), dtype=np.int64), np.ones((1,), dtype=np.float64)
    reach = degree // 2
    offsets = np.arange(-reach, reach + 1, dtype=np.int64)
    half = Fraction(degree + 1, 2)
    values = tuple(
        sum(
            (-1) ** term
            * comb(degree + 1, term)
            * max(int(offset) + half - term, Fraction(0)) ** degree
            for term in range(degree + 2)
        )
        / factorial(degree)
        for offset in offsets
    )
    return offsets, np.asarray([float(value) for value in values], dtype=np.float64)


def cardinal_bspline_periodic_symbol(degree: int, count: int, /) -> NDArray[np.float64]:
    """Discrete Fourier symbol of one periodic cardinal B-spline assignment axis.

    Returns ``S[m] = sum_j N_p(j) cos(2 pi m j / count)`` for ``m`` in
    ``0..count-1`` (``numpy.fft.fftfreq`` order): the exact DFT of what
    ``TensorBSplineSplatAssignment(degree)`` deposits from a unit source on an
    entity of a periodic axis with ``count`` entities. It is real and strictly
    positive, ``S[0] = 1``; particle-mesh Ewald-type influence functions divide
    each axis structure factor by it (the discrete B-spline modulus). Tensor
    symbols are products of axis symbols. Host-side preparation constant.
    """
    degree_ = _checked_degree(degree)
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError("count must be an integer.")
    count_ = int(count)
    if count_ < degree_ + 1:
        raise ValueError(
            "A periodic B-spline axis needs at least degree plus one entities."
        )
    offsets, values = _cardinal_node_values(degree_)
    modes = np.arange(count_, dtype=np.int64)
    phase = 2.0 * np.pi * np.mod(np.outer(modes, offsets), count_) / count_
    return np.cos(phase) @ values


__all__ = ["TensorBSplineSplatAssignment", "cardinal_bspline_periodic_symbol"]
