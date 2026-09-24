#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-difference grid values as evidenced coordinate field views.

Finite-difference plans own derivative stencils but no value reconstruction
between nodes; a view therefore requires an explicit interpolation policy
owned by the native interpolation substrate:

* `MultilinearGridInterpolation` routes the native rectilinear (multilinear)
  gather stencil: `C^0`, piecewise tensor-linear, values only.
* `BSplineGridInterpolation(p)` interpolates the nodal values with a tensor
  B-spline of degree `p` on node-averaged knots (per-axis collocation solved by
  a prepared native factorization) and evaluates native B-spline jets:
  `C^{p-1}`, piecewise tensor degree `p`, exact derivatives through order
  `p - 1`.

Both are linear in the nodal values with exact algebraic transposes. The
support is the node box (the periodic cell on periodic multilinear axes);
points outside it are `OUTSIDE_SUPPORT` and are never extrapolated.
"""

from __future__ import annotations

from math import isfinite, prod
from numbers import Integral
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import (
    bspline_jet_stencil,
    bspline_stencil,
    GatherStencil,
    rectilinear_stencil,
)
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    PreparedFactorization,
    RHSLayout,
)
from ...sparse import linear_apply, linear_transpose_apply
from .._view_support import tensor_box_support_geometry
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTracePolicy,
    FieldTraceSide,
    PreparedFieldReconstruction,
)
from ._plan import PreparedFiniteDifferenceDiscretization


@final
class MultilinearGridInterpolation(StrictModule, NonTrainableState):
    """Tensor multilinear interpolation between grid nodes."""

    policy_id: str = eqx.field(static=True)

    def __init__(self):
        self.policy_id = canonical_fingerprint({"kind": "multilinear-grid-interpolation"})


@final
class BSplineGridInterpolation(StrictModule, NonTrainableState):
    """Interpolating tensor B-spline of one degree through the grid nodes.

    Knots are the node averages of de Boor (open ends at the first and last
    node), so the collocation system satisfies the Schoenberg-Whitney
    conditions and the interpolant reproduces tensor polynomials of degree
    `degree` exactly. Degree one is the multilinear policy.
    """

    degree: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, degree: int, /, *, condition_limit: float = 1.0e12):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("B-spline interpolation degree must be an int.")
        if degree < 2:
            raise ValueError(
                "B-spline grid interpolation degree must be at least two; degree one "
                "is MultilinearGridInterpolation."
            )
        limit = float(condition_limit)
        if not isfinite(limit) or limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        self.degree = int(degree)
        self.condition_limit = limit
        self.policy_id = canonical_fingerprint(
            {
                "kind": "bspline-grid-interpolation",
                "degree": self.degree,
                "condition_limit": limit,
            }
        )


GridInterpolation = MultilinearGridInterpolation | BSplineGridInterpolation


def _averaged_knots(nodes: np.ndarray, degree: int, /) -> np.ndarray:
    count = nodes.size
    knots = np.empty((count + degree + 1,), dtype=np.float64)
    knots[: degree + 1] = nodes[0]
    knots[count:] = nodes[-1]
    for index in range(1, count - degree):
        knots[index + degree] = np.mean(nodes[index : index + degree])
    return knots


def _collocation(
    nodes: np.ndarray, degree: int, condition_limit: float, /
) -> tuple[np.ndarray, PreparedFactorization, float]:
    """Knots, the prepared collocation factorization, and its condition number."""
    count = nodes.size
    if count < degree + 1:
        raise ValueError(
            f"B-spline interpolation of degree {degree} needs at least {degree + 1} "
            "nodes per axis."
        )
    knots = _averaged_knots(nodes, degree)
    stencil = bspline_stencil(knots, nodes, degree=degree)
    matrix = np.zeros((count, count), dtype=np.float64)
    np.add.at(
        matrix,
        (np.arange(count)[:, None], np.asarray(stencil.indices)),
        np.asarray(stencil.weights),
    )
    operator = DenseLinearOperator(jnp.asarray(matrix))
    spectrum = factorize(operator, FactorizationPolicy("svd"))
    singular = np.asarray(spectrum.singular_values())
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else np.inf
    if int(np.asarray(spectrum.rank())) != count or not condition <= condition_limit:
        raise ValueError(
            "The B-spline collocation system is singular or ill-conditioned "
            f"(condition {condition:.6g})."
        )
    return knots, factorize(operator, FactorizationPolicy("lu")), condition


@final
class FiniteDifferenceFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Explicit-policy interpolation of finite-difference nodal values."""

    interpolation: GridInterpolation
    nodes: tuple[Array, ...]
    knots: tuple[Array, ...]
    collocation: tuple[PreparedFactorization, ...]
    box: tuple[tuple[float, float], ...] = eqx.field(static=True)
    condition: float = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        interpolation: GridInterpolation,
        nodes: tuple[np.ndarray, ...],
        lower: np.ndarray,
        upper: np.ndarray,
        periodic: tuple[bool, ...],
        /,
        *,
        field_space_id: str,
    ):
        axes = tuple(np.asarray(values, dtype=np.float64) for values in nodes)
        knots: tuple[np.ndarray, ...] = ()
        collocation: tuple[PreparedFactorization, ...] = ()
        condition = 1.0
        match interpolation:
            case MultilinearGridInterpolation():
                pass
            case BSplineGridInterpolation():
                if any(periodic):
                    raise ValueError(
                        "B-spline grid interpolation is defined on bounded axes; "
                        "periodic axes need MultilinearGridInterpolation."
                    )
                prepared = tuple(
                    _collocation(
                        values, interpolation.degree, interpolation.condition_limit
                    )
                    for values in axes
                )
                knots = tuple(item[0] for item in prepared)
                collocation = tuple(item[1] for item in prepared)
                condition = float(np.prod([item[2] for item in prepared]))
            case _:
                raise TypeError(
                    "interpolation must be MultilinearGridInterpolation or "
                    "BSplineGridInterpolation."
                )
        self.interpolation = interpolation
        self.nodes = tuple(jnp.asarray(values) for values in axes)
        self.knots = tuple(jnp.asarray(values) for values in knots)
        self.collocation = collocation
        self.box = tuple(
            (float(first), float(last)) for first, last in zip(lower, upper, strict=True)
        )
        self.condition = condition
        self.periodic = tuple(bool(value) for value in periodic)
        self.grid_shape = tuple(values.size for values in axes)
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "finite-difference-field-reconstruction-kernel",
                "interpolation": interpolation.policy_id,
                "nodes": [array_tree_fingerprint(values) for values in axes],
                "box": array_tree_fingerprint(np.stack((lower, upper))),
                "periodic": list(self.periodic),
                "field_space": field_space_id,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return 0

    @property
    def support_coverage(self) -> str:
        return "complete"

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[GatherStencil, FieldQueryEvidence]:
        del side
        finite = jnp.all(jnp.isfinite(points), axis=1)
        lower = jnp.asarray([first for first, _ in self.box], dtype=points.dtype)
        upper = jnp.asarray([last for _, last in self.box], dtype=points.dtype)
        # Native stencils refuse non-finite queries; those points are reported
        # NONFINITE and evaluated at an inert in-box coordinate with zero weight.
        query = jnp.where(finite[:, None], points, lower)
        in_box = jnp.all((query >= lower) & (query <= upper), axis=1)
        match self.interpolation:
            case MultilinearGridInterpolation():
                if any(derivative):
                    raise ValueError("Multilinear grid interpolation has no derivatives.")
                indices, weights, support = self._multilinear(query)
            case BSplineGridInterpolation():
                indices, weights, support = self._bspline(query, derivative)
            case _:
                raise TypeError("Unknown finite-difference grid interpolation.")
        status = jnp.where(
            ~finite,
            int(FieldQueryStatus.NONFINITE),
            jnp.where(
                support & in_box,
                int(FieldQueryStatus.VALID),
                int(FieldQueryStatus.OUTSIDE_SUPPORT),
            ),
        ).astype(jnp.int32)
        valid = status == int(FieldQueryStatus.VALID)
        route = GatherStencil(
            indices=indices,
            weights=jnp.where(valid[:, None], weights, 0.0),
            source_size=prod(self.grid_shape),
            support=valid,
        )
        conditioning = jnp.full(points.shape[:1], self.condition, dtype=points.dtype)
        count = jnp.sum(route.weights != 0.0, axis=1, dtype=jnp.int32)
        return route, FieldQueryEvidence(
            status, conditioning, count, kernel_id=self.kernel_id
        )

    def _multilinear(self, query: Array, /) -> tuple[Array, Array, Array]:
        bounds = tuple(
            bound if periodic else None
            for bound, periodic in zip(self.box, self.periodic, strict=True)
        )
        stencil = rectilinear_stencil(
            self.nodes,
            query,
            boundary=tuple(
                "periodic" if value else "constant" for value in self.periodic
            ),
            periods=tuple(
                None if bound is None else bound[1] - bound[0] for bound in bounds
            ),
            axis_bounds=bounds,
        )
        return stencil.indices, stencil.weights, stencil.support

    def _bspline(
        self, query: Array, derivative: tuple[int, ...], /
    ) -> tuple[Array, Array, Array]:
        count = query.shape[0]
        indices = jnp.zeros((count, 1), dtype=jnp.int32)
        weights = jnp.ones((count, 1), dtype=query.dtype)
        support = jnp.ones((count,), dtype=jnp.bool_)
        for axis, (knots, size, order) in enumerate(
            zip(self.knots, self.grid_shape, derivative, strict=True)
        ):
            jet = bspline_jet_stencil(
                knots,
                query[:, axis],
                degree=self.interpolation.degree,
                maximum_order=order,
                bounds="fill",
            )
            indices = (indices[:, :, None] * size + jet.indices[:, None, :]).reshape(
                (count, -1)
            )
            weights = (weights[:, :, None] * jet.jets[:, None, order, :]).reshape(
                (count, -1)
            )
            support = support & jet.support
        return indices, weights, support

    def _solve_axes(self, values: Array, *, transpose: bool) -> Array:
        """Apply the tensor collocation inverse (or its transpose) axis by axis."""
        for axis, factorization in enumerate(self.collocation):
            moved = jnp.moveaxis(values, axis, 0)
            columns = moved.reshape((moved.shape[0], -1))
            layout = RHSLayout((columns.shape[1],))
            solved = (
                factorization.solve_transpose(columns, rhs_layout=layout)
                if transpose
                else factorization.solve(columns, rhs_layout=layout)
            )
            values = jnp.moveaxis(solved.value.reshape(moved.shape), 0, axis)
        return values

    def apply(self, route: GatherStencil, coefficients: Array, /) -> Array:
        values = coefficients
        if isinstance(self.interpolation, BSplineGridInterpolation):
            values = self._solve_axes(values, transpose=False)
        flat = values.reshape(
            (prod(self.grid_shape), *values.shape[len(self.grid_shape) :])
        )
        return linear_apply(route.relation, route.weights, flat)

    def transpose(self, route: GatherStencil, cotangent: Array, /) -> Array:
        scattered = linear_transpose_apply(route.relation, route.weights, cotangent)
        values = scattered.reshape((*self.grid_shape, *cotangent.shape[1:]))
        if isinstance(self.interpolation, BSplineGridInterpolation):
            values = self._solve_axes(values, transpose=True)
        return values

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        del sites, side, cell_ids
        raise ValueError("Single-valued grid interpolations have no trace cells.")


def _centered_space(discretization: PreparedFiniteDifferenceDiscretization, /) -> Any:
    centered = discretization.grid.centered_location.location_id
    for space in discretization.field_spaces:
        if space.layout.location_id == centered:
            return space
    raise ValueError(
        "The finite-difference discretization has no field space at the grid's "
        "primary nodes."
    )


def prepare_finite_difference_field_reconstruction(
    discretization: PreparedFiniteDifferenceDiscretization,
    /,
    *,
    interpolation: GridInterpolation,
    value_port: ValuePort | None = None,
    support_geometry: Any = None,
    support_tolerance: float = 1.0e-9,
) -> PreparedFieldReconstruction:
    """Prepare an evidenced coordinate reconstruction of finite-difference values.

    Coefficients are the nodal values at the grid's primary entities with
    shape `grid.shape + value_port.event_shape`. `interpolation` is required;
    see the module documentation for the regularity and derivative order each
    policy declares. The support is the node box on bounded axes and the
    periodic cell on periodic axes (`support_geometry` defaults to an
    `Orthotope`; an explicit geometry must have the box's bounds and measure).
    """
    if not isinstance(discretization, PreparedFiniteDifferenceDiscretization):
        raise TypeError(
            "discretization must be a PreparedFiniteDifferenceDiscretization."
        )
    space = _centered_space(discretization)
    grid = discretization.grid
    layout = grid.primary_entity_layout
    nodes = tuple(
        np.asarray(values, dtype=np.float64) for values in layout.coordinates_by_axis
    )
    periodic = tuple(axis.periodic for axis in grid.structured_axes)
    lower = np.asarray(
        [
            np.asarray(axis.bounds)[0] if axis.periodic else values[0]
            for axis, values in zip(grid.structured_axes, nodes, strict=True)
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            np.asarray(axis.bounds)[1] if axis.periodic else values[-1]
            for axis, values in zip(grid.structured_axes, nodes, strict=True)
        ],
        dtype=np.float64,
    )
    if value_port is None:
        components: tuple[int, ...] = ()
    elif isinstance(value_port, ValuePort):
        components = value_port.event_shape
    else:
        raise TypeError("value_port must be a ValuePort or None.")
    field_space_id = canonical_fingerprint(
        {
            "kind": "finite-difference-field-space",
            "discretization": discretization.prepared_id,
            "space": space.field_space_id,
            "components": list(components),
        }
    )
    match interpolation:
        case MultilinearGridInterpolation():
            regularity = DerivativeRegularity.piecewise_polynomial(
                continuity=0, degree_bound=len(nodes)
            )
            maximum_order = 0
        case BSplineGridInterpolation():
            degree = interpolation.degree
            regularity = DerivativeRegularity.piecewise_polynomial(
                continuity=degree - 1, degree_bound=degree * len(nodes)
            )
            maximum_order = degree - 1
        case _:
            raise TypeError(
                "interpolation must be MultilinearGridInterpolation or "
                "BSplineGridInterpolation."
            )
    kernel = FiniteDifferenceFieldReconstructionKernel(
        interpolation, nodes, lower, upper, periodic, field_space_id=field_space_id
    )
    support_id = canonical_fingerprint(
        {
            "kind": "finite-difference-node-box-support",
            "support": discretization.support.support_id,
            "box": array_tree_fingerprint(np.stack((lower, upper))),
            "periodic": list(periodic),
        }
    )
    geometry = tensor_box_support_geometry(
        lower,
        upper,
        support_geometry,
        feature_id=f"finite-difference-support:{support_id}",
        tolerance=support_tolerance,
    )
    port = (
        ValuePort(
            space.name,
            event_shape=(),
            component_ids=(space.name,),
            representation="finite-difference-field",
            space_id=field_space_id,
        )
        if value_port is None
        else value_port
    )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=geometry,
        value_port=port,
        regularity=regularity,
        trace_policy=FieldTracePolicy("single-valued"),
        coefficient_shape=(*grid.shape, *components),
        physical_dimension=len(nodes),
        maximum_derivative_order=maximum_order,
        field_space_id=field_space_id,
        support_id=support_id,
    )


__all__ = [
    "BSplineGridInterpolation",
    "FiniteDifferenceFieldReconstructionKernel",
    "MultilinearGridInterpolation",
    "prepare_finite_difference_field_reconstruction",
]
