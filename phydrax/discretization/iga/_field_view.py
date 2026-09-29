#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared point queries and side traces of single-patch isogeometric fields.

A physical query point is mapped back to the patch parameter box by a pure-JAX
Newton inversion of the runtime NURBS map, seeded from host-prepared samples
of every integration-overlay cell. The located parameters select the overlay
cells containing the point (two or more on a knot line) and each candidate is
evaluated on its own spline pieces, so one-sided limits are exact. Values use
the field's rational basis; first physical derivatives solve the transposed
geometry Jacobian against the parameter gradients. Side traces evaluate the
same pieces at facet rule points of the owner cell's local facet with the
physical facet measure (reference weights times the surface Jacobian) and the
outward normal of the traced side cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite, prod
from typing import assert_never, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, SmallLinearSolvePlan, solve_small_linear
from ...typing import parse
from .._integration_domain import IntegrationDomain
from .._reference_cell import FacetShape
from .._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
    SideTraceQuantity,
)
from .._view_support import tensor_box_support_geometry
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldSupportCoverage,
    FieldTracePolicy,
    FieldTraceSide,
    PreparedFieldReconstruction,
)
from ._actions import (
    isogeometric_axis_spans,
    isogeometric_point_jets,
    isogeometric_point_map,
)
from ._basis import TensorSplineBasisSpec
from ._geometry import IsogeometricRuntimeData
from ._plan import PreparedIsogeometricDiscretization


if TYPE_CHECKING:
    from ...geometry import CompiledGeometry


_FACET_SHAPES: dict[int, FacetShape] = {1: "point", 2: "edge", 3: "quadrilateral"}
_TANGENTIAL_DIMENSIONS = (2, 3)
_PARAMETER_TOLERANCE = 1.0e-10
_AFFINE_TOLERANCE = 1.0e-12
_SUPPORT_TOLERANCE = 1.0e-13

# Host preparation evaluates the rational jets as one compiled computation.
_point_jets = eqx.filter_jit(isogeometric_point_jets)
_point_map = eqx.filter_jit(isogeometric_point_map)


def _realized_runtime(
    discretization: PreparedIsogeometricDiscretization,
    runtime: IsogeometricRuntimeData | None,
    /,
) -> IsogeometricRuntimeData:
    realized = discretization.default_runtime if runtime is None else runtime
    discretization.validate_local_runtime(realized)
    return realized


def _field_weights(
    discretization: PreparedIsogeometricDiscretization,
    field_index: int,
    runtime: IsogeometricRuntimeData,
    /,
) -> Array:
    """Rational weights of one field under one geometry realization."""
    field = discretization.fields[field_index]
    if field.weights_from_geometry:
        return runtime.weights
    if field.weights is None:
        return jnp.ones(field.basis.control_shape, dtype=runtime.weights.dtype)
    return jnp.asarray(field.weights, dtype=runtime.weights.dtype)


def _require_volume_patch(
    discretization: PreparedIsogeometricDiscretization,
    runtime: IsogeometricRuntimeData,
    /,
) -> None:
    if runtime.control_points.shape[-1] != discretization.basis.parametric_dimension:
        raise ValueError(
            "IGA field queries and side traces require a patch whose ambient and "
            "parametric dimensions agree; embedded manifold patches have no unique "
            "inverse map or facet normal."
        )


def _span_tables(
    basis: TensorSplineBasisSpec, breaks: tuple[tuple[float, ...], ...], /
) -> tuple[np.ndarray, ...]:
    """Active span of `basis` on every overlay interval of every axis."""
    tables = []
    for axis, values in zip(basis.axes, breaks, strict=True):
        points = np.asarray(values, dtype=np.float64)
        midpoints = 0.5 * (points[:-1] + points[1:])
        spans = np.searchsorted(np.asarray(axis.knots), midpoints, side="right") - 1
        tables.append(spans.astype(np.int32))
    return tuple(tables)


def _break_continuity(
    basis: TensorSplineBasisSpec, breaks: tuple[tuple[float, ...], ...], /
) -> int | None:
    """Smallest continuity of `basis` across an interior overlay break."""
    result: int | None = None
    for axis, values in zip(basis.axes, breaks, strict=True):
        knots = np.asarray(axis.knots)
        for value in values[1:-1]:
            multiplicity = int(np.count_nonzero(knots == value))
            if multiplicity:
                order = axis.degree - multiplicity
                result = order if result is None else min(result, order)
    return result


def _greville(basis: TensorSplineBasisSpec, /) -> np.ndarray:
    axes = []
    for axis in basis.axes:
        knots = np.asarray(axis.knots, dtype=np.float64)
        axes.append(
            np.asarray(
                [
                    np.mean(knots[index + 1 : index + axis.degree + 1])
                    for index in range(axis.control_count)
                ]
            )
        )
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape((-1, basis.parametric_dimension))


def _affine_map(
    basis: TensorSplineBasisSpec, runtime: IsogeometricRuntimeData, /
) -> np.ndarray | None:
    """Linear part `(d + 1, D)` of an exactly affine patch map, else `None`.

    A B-spline map is affine exactly when its weights are uniform and its
    control points are the affine image of the Greville abscissae.
    """
    weights = np.asarray(runtime.weights, dtype=np.float64)
    if np.ptp(weights) > _AFFINE_TOLERANCE * np.max(weights):
        return None
    points = np.asarray(runtime.control_points, dtype=np.float64).reshape(
        (-1, runtime.control_points.shape[-1])
    )
    design = np.concatenate(
        (_greville(basis), np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    solution = np.linalg.lstsq(design, points, rcond=None)[0]
    scale = max(float(np.max(np.abs(points))), 1.0)
    if np.max(np.abs(design @ solution - points)) > _AFFINE_TOLERANCE * scale:
        return None
    return solution


def _uniform(weights: Array, /) -> bool:
    values = np.asarray(weights, dtype=np.float64)
    return bool(np.ptp(values) <= _AFFINE_TOLERANCE * np.max(np.abs(values)))


@dataclass(frozen=True, slots=True)
class _FacetEvaluation:
    """Host sites, measure, outward normals, and trace basis of selected facets."""

    sites: np.ndarray
    weights: np.ndarray
    normals: np.ndarray
    basis: np.ndarray
    dofs: np.ndarray
    tangential_degree: int


def _facet_evaluation(
    discretization: PreparedIsogeometricDiscretization,
    field_basis: TensorSplineBasisSpec,
    field_weights: Array,
    runtime: IsogeometricRuntimeData,
    cells: np.ndarray,
    local_entities: np.ndarray,
    rule: FacetTraceRule,
    /,
) -> _FacetEvaluation:
    """Evaluate the side cells' local facets at the owner-canonical rule points.

    Local entity `2 a + s` is the facet of constant parameter axis `a` at the
    cell's lower (`s = 0`) or upper (`s = 1`) break. Tangential axes keep
    increasing order and map the unit rule parameters affinely onto the cell
    intervals, so owner and neighbor sides of a knot face share parameters.
    """
    breaks = discretization.overlay_breaks
    dimension = len(breaks)
    overlay_shape = tuple(len(values) - 1 for values in breaks)
    reference, rule_weights = rule.reference(_FACET_SHAPES[dimension])
    intervals = np.stack(np.unravel_index(cells, overlay_shape), axis=-1)
    axes = local_entities // 2
    upper = local_entities % 2
    facet_count, point_count = cells.size, rule_weights.size
    parameters = np.empty((facet_count, point_count, dimension), dtype=np.float64)
    lengths = np.ones((facet_count, max(dimension - 1, 1)), dtype=np.float64)
    tangential_degree = 0
    for facet in range(facet_count):
        axis = axes[facet]
        tangential = [index for index in range(dimension) if index != axis]
        low = [breaks[index][intervals[facet, index]] for index in range(dimension)]
        high = [breaks[index][intervals[facet, index] + 1] for index in range(dimension)]
        parameters[facet, :, axis] = high[axis] if upper[facet] else low[axis]
        for slot, index in enumerate(tangential):
            parameters[facet, :, index] = low[index] + reference[:, slot] * (
                high[index] - low[index]
            )
            lengths[facet, slot] = high[index] - low[index]
        tangential_degree = max(
            tangential_degree, sum(field_basis.degrees[index] for index in tangential)
        )
    repeated = np.repeat(intervals, point_count, axis=0)
    flat = jnp.asarray(parameters.reshape((-1, dimension)))

    def spans(basis: TensorSplineBasisSpec) -> Array:
        tables = _span_tables(basis, breaks)
        return jnp.asarray(
            np.stack(
                [tables[index][repeated[:, index]] for index in range(dimension)],
                axis=-1,
            )
        )

    values, _, rows = _point_jets(field_basis, field_weights, flat, spans(field_basis))
    points, jacobian = _point_map(
        discretization.basis,
        runtime.control_points,
        runtime.weights,
        flat,
        spans(discretization.basis),
    )
    covector = jnp.asarray(
        np.eye(dimension, dtype=np.float64)[np.repeat(axes, point_count)]
        * np.where(np.repeat(upper, point_count) > 0, 1.0, -1.0)[:, None]
    )
    solve = solve_small_linear(
        SmallLinearSolvePlan(dimension), jnp.swapaxes(jacobian, -1, -2), covector
    )
    if not bool(jnp.all(solve.successful)):
        raise ValueError("The IGA geometry Jacobian is singular at a facet site.")
    gradient = np.asarray(solve.value, dtype=np.float64)
    normals = gradient / np.linalg.norm(gradient, axis=-1, keepdims=True)
    jacobian_ = np.asarray(jacobian, dtype=np.float64).reshape(
        (facet_count, point_count, dimension, dimension)
    )
    if dimension == 1:
        measure = np.ones((facet_count, point_count), dtype=np.float64)
    else:
        columns = np.stack(
            [
                np.take(
                    jacobian_[facet],
                    [index for index in range(dimension) if index != axes[facet]],
                    axis=-1,
                )
                * lengths[facet]
                for facet in range(facet_count)
            ]
        )
        gram = ein.contract("fqdi,fqdj->fqij", columns, columns)
        measure = np.sqrt(np.linalg.det(gram))
    basis = np.asarray(values, dtype=np.float64).reshape((facet_count, point_count, -1))
    dofs = np.asarray(rows).reshape((facet_count, point_count, -1))[:, 0, :]
    return _FacetEvaluation(
        sites=np.asarray(points, dtype=np.float64).reshape(
            (facet_count, point_count, dimension)
        ),
        weights=rule_weights[None, :] * measure,
        normals=normals.reshape((facet_count, point_count, dimension)),
        basis=basis,
        dofs=dofs,
        tangential_degree=tangential_degree,
    )


class _IsogeometricRoute(StrictModule):
    rows: Array
    weights: Array


@final
class IsogeometricFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Located evaluation of one single-patch IGA field in physical coordinates.

    `locate` inverts the runtime NURBS map by a fixed number of clamped Newton
    steps from the nearest host-prepared seed (pure JAX) and reports
    `OUTSIDE_SUPPORT` for points whose closest parameter lies on the patch box
    with a nonzero residual and `LOCATION_FAILED` for unconverged interior
    iterations. Every integration-overlay cell containing the located parameter
    contributes a candidate evaluated on its own spline pieces; on a knot line
    where a derivative exceeds the field or geometry continuity a trace side is
    required. Coefficients use the field's public tensor layout
    `control_shape + component_shape`.
    """

    field_basis: TensorSplineBasisSpec
    geometry_basis: TensorSplineBasisSpec
    field_weights: Array
    control_points: Array
    geometry_weights: Array
    breaks: tuple[Array, ...]
    field_spans: tuple[Array, ...]
    geometry_spans: tuple[Array, ...]
    seed_parameters: Array
    seed_points: Array
    lower: Array
    upper: Array
    coefficient_shape: tuple[int, ...] = eqx.field(static=True)
    overlay_shape: tuple[int, ...] = eqx.field(static=True)
    continuity: int | None = eqx.field(static=True)
    newton_iterations: int = eqx.field(static=True)
    location_tolerance: float = eqx.field(static=True)
    coordinate_scale: float = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedIsogeometricDiscretization,
        field_index: int,
        runtime: IsogeometricRuntimeData,
        /,
        *,
        newton_iterations: int,
        location_tolerance: float,
        seeds_per_axis: int,
    ) -> None:
        if isinstance(newton_iterations, bool) or not isinstance(newton_iterations, int):
            raise TypeError("newton_iterations must be an int.")
        if newton_iterations < 1:
            raise ValueError("newton_iterations must be positive.")
        if isinstance(seeds_per_axis, bool) or not isinstance(seeds_per_axis, int):
            raise TypeError("seeds_per_axis must be an int.")
        if seeds_per_axis < 1:
            raise ValueError("seeds_per_axis must be positive.")
        tolerance = float(location_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("location_tolerance must be finite and positive.")
        _require_volume_patch(discretization, runtime)
        field = discretization.fields[field_index]
        breaks = discretization.overlay_breaks
        overlay_shape = tuple(len(values) - 1 for values in breaks)
        field_continuity = _break_continuity(field.basis, breaks)
        geometry_continuity = _break_continuity(discretization.basis, breaks)
        known = [
            value
            for value in (field_continuity, geometry_continuity)
            if value is not None
        ]
        control_points = np.asarray(runtime.control_points, dtype=np.float64)
        scale = float(
            np.max(np.ptp(control_points.reshape((-1, control_points.shape[-1])), axis=0))
        )
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("The IGA patch has no positive coordinate extent.")
        offsets = (np.arange(seeds_per_axis, dtype=np.float64) + 0.5) / seeds_per_axis
        seeds = []
        for cell in np.ndindex(overlay_shape):
            axes = [
                breaks[axis][row] + offsets * (breaks[axis][row + 1] - breaks[axis][row])
                for axis, row in enumerate(cell)
            ]
            grid = np.meshgrid(*axes, indexing="ij")
            seeds.append(np.stack(grid, axis=-1).reshape((-1, len(breaks))))
        self.field_basis = field.basis
        self.geometry_basis = discretization.basis
        self.field_weights = _field_weights(discretization, field_index, runtime)
        self.control_points = runtime.control_points
        self.geometry_weights = runtime.weights
        self.breaks = tuple(jnp.asarray(values, dtype=jnp.float64) for values in breaks)
        self.field_spans = tuple(
            jnp.asarray(table) for table in _span_tables(field.basis, breaks)
        )
        self.geometry_spans = tuple(
            jnp.asarray(table) for table in _span_tables(discretization.basis, breaks)
        )
        self.seed_parameters = jnp.asarray(np.concatenate(seeds))
        self.lower = jnp.asarray([values[0] for values in breaks], dtype=jnp.float64)
        self.upper = jnp.asarray([values[-1] for values in breaks], dtype=jnp.float64)
        self.coefficient_shape = field.basis.control_shape + field.component_shape
        self.overlay_shape = overlay_shape
        self.continuity = min(known) if known else None
        self.newton_iterations = newton_iterations
        self.location_tolerance = tolerance
        self.coordinate_scale = scale
        self.seed_points = self._map(self.seed_parameters)[0]
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "isogeometric-field-reconstruction-kernel",
                "prepared": discretization.prepared_id,
                "field": discretization.field_spaces[field_index].field_space_id,
                "runtime": runtime.runtime_id,
                "control_points": array_tree_fingerprint(control_points),
                "weights": array_tree_fingerprint(np.asarray(runtime.weights)),
                "newton_iterations": newton_iterations,
                "location_tolerance": tolerance,
                "seeds_per_axis": seeds_per_axis,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return prod(self.overlay_shape)

    @property
    def support_coverage(self) -> FieldSupportCoverage:
        return "complete"

    @property
    def _dimension(self) -> int:
        return len(self.overlay_shape)

    def _map(self, parameters: Array, /) -> tuple[Array, Array]:
        return _point_map(
            self.geometry_basis,
            self.control_points,
            self.geometry_weights,
            parameters,
            isogeometric_axis_spans(self.geometry_basis, parameters),
        )

    def _invert(self, points: Array, /) -> tuple[Array, Array, Array]:
        """Located parameters, pointwise status, and Jacobian condition numbers."""
        finite = jnp.all(jnp.isfinite(points), axis=-1)
        safe = jnp.where(finite[:, None], points, self.seed_points[0])
        distance = jnp.sum((safe[:, None, :] - self.seed_points[None]) ** 2, axis=-1)
        start = self.seed_parameters[jnp.argmin(distance, axis=1)]
        plan = SmallLinearSolvePlan(self._dimension)

        def step(_: Array, parameters: Array) -> Array:
            mapped, jacobian = self._map(parameters)
            solve = solve_small_linear(plan, jacobian, safe - mapped)
            delta = jnp.where(solve.successful[:, None], solve.value, 0.0)
            return jnp.clip(parameters + delta, self.lower, self.upper)

        parameters = jax.lax.fori_loop(0, self.newton_iterations, step, start)
        mapped, jacobian = self._map(parameters)
        residual = jnp.linalg.norm(safe - mapped, axis=-1)
        singular = jnp.linalg.svd(jacobian, compute_uv=False)
        condition = singular[:, 0] / singular[:, -1]
        tolerance = _PARAMETER_TOLERANCE * (self.upper - self.lower)
        on_box = jnp.any(
            (parameters <= self.lower + tolerance)
            | (parameters >= self.upper - tolerance),
            axis=-1,
        )
        located = residual <= self.location_tolerance * self.coordinate_scale
        status = jnp.where(
            ~finite,
            int(FieldQueryStatus.NONFINITE),
            jnp.where(
                located,
                jnp.where(
                    jnp.isfinite(condition),
                    int(FieldQueryStatus.VALID),
                    int(FieldQueryStatus.ILL_CONDITIONED),
                ),
                jnp.where(
                    on_box,
                    int(FieldQueryStatus.OUTSIDE_SUPPORT),
                    int(FieldQueryStatus.LOCATION_FAILED),
                ),
            ),
        ).astype(jnp.int32)
        return parameters, status, jnp.where(finite, condition, jnp.inf)

    def _candidates(self, parameters: Array, /) -> tuple[Array, Array, Array]:
        """Candidate overlay cells `(n, 2^d)`, their distinctness, and intervals."""
        tolerance = _PARAMETER_TOLERANCE * (self.upper - self.lower)
        lows, highs = [], []
        for axis, breaks in enumerate(self.breaks):
            values = parameters[:, axis]
            count = breaks.shape[0] - 1
            nearest = jnp.argmin(jnp.abs(values[:, None] - breaks[None, :]), axis=1)
            on_break = (
                (jnp.abs(values - breaks[nearest]) <= tolerance[axis])
                & (nearest > 0)
                & (nearest < count)
            )
            inside = jnp.clip(
                jnp.searchsorted(breaks, values, side="right") - 1, 0, count - 1
            )
            lows.append(jnp.where(on_break, nearest - 1, inside))
            highs.append(jnp.where(on_break, nearest, inside))
        intervals, distinct = [], []
        for choice in product((0, 1), repeat=self._dimension):
            intervals.append(
                jnp.stack(
                    [
                        highs[axis] if bit else lows[axis]
                        for axis, bit in enumerate(choice)
                    ],
                    axis=-1,
                )
            )
            separate = jnp.ones(parameters.shape[:1], dtype=jnp.bool_)
            for axis, bit in enumerate(choice):
                if bit:
                    separate = separate & (highs[axis] != lows[axis])
            distinct.append(separate)
        stacked = jnp.stack(intervals, axis=1).astype(jnp.int32)
        strides = np.asarray(
            [prod(self.overlay_shape[axis + 1 :]) for axis in range(self._dimension)],
            dtype=np.int32,
        )
        cells = ein.contract("ncd,d->nc", stacked, jnp.asarray(strides))
        return cells, jnp.stack(distinct, axis=1), stacked

    def _candidate_weights(
        self, parameters: Array, intervals: Array, axis: int | None, /
    ) -> tuple[Array, Array]:
        count, candidates, dimension = intervals.shape
        flat_parameters = jnp.repeat(parameters, candidates, axis=0)
        flat_intervals = intervals.reshape((count * candidates, dimension))
        field_spans = jnp.stack(
            [
                self.field_spans[index][flat_intervals[:, index]]
                for index in range(dimension)
            ],
            axis=-1,
        )
        values, gradients, rows = isogeometric_point_jets(
            self.field_basis, self.field_weights, flat_parameters, field_spans
        )
        if axis is None:
            weights = values
        else:
            geometry_spans = jnp.stack(
                [
                    self.geometry_spans[index][flat_intervals[:, index]]
                    for index in range(dimension)
                ],
                axis=-1,
            )
            _, jacobian = isogeometric_point_map(
                self.geometry_basis,
                self.control_points,
                self.geometry_weights,
                flat_parameters,
                geometry_spans,
            )
            physical = solve_small_linear(
                SmallLinearSolvePlan(dimension),
                jnp.swapaxes(jacobian, -1, -2),
                jnp.swapaxes(gradients, -1, -2),
            )
            weights = physical.value[:, axis, :]
        local = values.shape[-1]
        return (
            rows.reshape((count, candidates, local)),
            weights.reshape((count, candidates, local)),
        )

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[_IsogeometricRoute, FieldQueryEvidence]:
        return _compiled_locate(self, points, derivative, side)

    def _locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[_IsogeometricRoute, FieldQueryEvidence]:
        order = sum(derivative)
        axis = None if order == 0 else derivative.index(1)
        parameters, status, condition = self._invert(points)
        cells, accepted, intervals = self._candidates(parameters)
        if side is not None and side.cell_mask is not None:
            accepted = accepted & side.cell_mask[cells]
        count = jnp.sum(accepted, axis=1, dtype=jnp.int32)
        rows, weights = self._candidate_weights(parameters, intervals, axis)
        if side is not None and side.side == "average":
            share = accepted / jnp.maximum(count, 1)[:, None]
        else:
            first = jnp.argmax(accepted, axis=1)
            share = (jnp.arange(accepted.shape[1])[None, :] == first[:, None]) & accepted
        route = _IsogeometricRoute(
            rows, weights * share.astype(weights.dtype)[:, :, None]
        )
        valid = status == int(FieldQueryStatus.VALID)
        side_status = int(
            FieldQueryStatus.SIDE_REQUIRED
            if side is None
            else FieldQueryStatus.SIDE_UNRESOLVED
        )
        if (
            self.continuity is not None
            and order > self.continuity
            and (side is None or side.side != "average")
        ):
            status = jnp.where(valid & (count > 1), side_status, status)
        status = jnp.where(
            valid & (count == 0), int(FieldQueryStatus.SIDE_UNRESOLVED), status
        )
        return route, FieldQueryEvidence(
            status, condition, count, kernel_id=self.kernel_id
        )

    def apply(self, route: _IsogeometricRoute, coefficients: Array, /) -> Array:
        flat = coefficients.reshape(
            (prod(self.field_basis.control_shape), *self._components)
        )
        return ein.contract("pcl,pcl...->p...", route.weights, flat[route.rows])

    def transpose(self, route: _IsogeometricRoute, cotangent: Array, /) -> Array:
        payload = ein.contract("pcl,p...->pcl...", route.weights, cotangent)
        flat = jnp.zeros(
            (prod(self.field_basis.control_shape), *cotangent.shape[1:]),
            dtype=payload.dtype,
        )
        return (
            flat.at[route.rows]
            .add(payload)
            .reshape(self.field_basis.control_shape + cotangent.shape[1:])
        )

    @property
    def _components(self) -> tuple[int, ...]:
        return self.coefficient_shape[len(self.field_basis.control_shape) :]

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        status, cells_, distinct_ = _compiled_containing(self, jnp.asarray(sites))
        if np.any(np.asarray(status) != int(FieldQueryStatus.VALID)):
            raise ValueError("Every trace site must lie in the IGA patch.")
        cells, distinct = np.asarray(cells_), np.asarray(distinct_)
        containing = np.where(distinct, cells, -1)
        if side == "average":
            if cell_ids is not None:
                raise ValueError("Average traces combine every containing cell.")
            return None, np.full(sites.shape[0], -1, dtype=np.int32)
        if cell_ids is None:
            single = np.sum(distinct, axis=1) == 1
            if side != "owner" or not np.all(single):
                raise ValueError(
                    f"{side!r} traces at sites shared by several overlay cells (or "
                    "without a neighbor cell) require explicit cell_ids."
                )
            site_cells = np.max(containing, axis=1).astype(np.int32)
        else:
            if np.any(np.all(containing != cell_ids[:, None], axis=1)):
                raise ValueError(
                    "cell_ids must name an overlay cell containing each trace site."
                )
            site_cells = cell_ids
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        resolved = distinct & mask[cells]
        if np.any(np.sum(resolved, axis=1) != 1) or np.any(
            np.max(np.where(resolved, cells, -1), axis=1) != site_cells
        ):
            raise ValueError(
                "The side cells do not resolve every trace site to exactly its "
                "declared overlay cell; the side region is ambiguous."
            )
        return mask, site_cells


@eqx.filter_jit
def _compiled_locate(
    kernel: IsogeometricFieldReconstructionKernel,
    points: Array,
    derivative: tuple[int, ...],
    side: FieldSideBinding | None,
    /,
) -> tuple[_IsogeometricRoute, FieldQueryEvidence]:
    return kernel._locate(points, derivative, side)


@eqx.filter_jit
def _compiled_containing(
    kernel: IsogeometricFieldReconstructionKernel, points: Array, /
) -> tuple[Array, Array, Array]:
    """Location status, candidate overlay cells, and their distinctness."""
    parameters, status, _ = kernel._invert(points)
    cells, distinct, _ = kernel._candidates(parameters)
    return status, cells, distinct


def _patch_measure(
    kernel: IsogeometricFieldReconstructionKernel, points_per_axis: int, /
) -> float:
    """Gauss--Legendre measure of the patch on its overlay cells."""
    nodes, weights = np.polynomial.legendre.leggauss(points_per_axis)
    parameters, measures = [], []
    for cell in np.ndindex(kernel.overlay_shape):
        axes, axis_weights = [], []
        for axis, row in enumerate(cell):
            low = float(kernel.breaks[axis][row])
            high = float(kernel.breaks[axis][row + 1])
            axes.append(0.5 * (low + high) + 0.5 * (high - low) * nodes)
            axis_weights.append(0.5 * (high - low) * weights)
        grid = np.meshgrid(*axes, indexing="ij")
        parameters.append(np.stack(grid, axis=-1).reshape((-1, len(cell))))
        combined = axis_weights[0]
        for factor in axis_weights[1:]:
            combined = np.multiply.outer(combined, factor)
        measures.append(np.asarray(combined).reshape((-1,)))
    _, jacobian = kernel._map(jnp.asarray(np.concatenate(parameters)))
    determinant = np.abs(np.linalg.det(np.asarray(jacobian, dtype=np.float64)))
    return float(np.sum(np.concatenate(measures) * determinant))


def _support_geometry(
    discretization: PreparedIsogeometricDiscretization,
    kernel: IsogeometricFieldReconstructionKernel,
    runtime: IsogeometricRuntimeData,
    support_geometry: object,
    /,
    *,
    feature_id: str,
    tolerance: float,
) -> CompiledGeometry:
    """Derived box of an axis-aligned affine patch, or an evidenced explicit region.

    An explicit region is admitted when the patch boundary sites lie on its
    boundary, interior seeds lie strictly inside it, and (when it declares an
    interior measure) its measure equals the patch measure.
    """
    from ...geometry import CompiledGeometry, GeometryCapability, GeometryKind

    dimension = discretization.basis.parametric_dimension
    affine = _affine_map(discretization.basis, runtime)
    if affine is not None:
        linear = affine[:dimension]
        if np.max(np.abs(linear - np.diag(np.diag(linear)))) <= _AFFINE_TOLERANCE * (
            max(float(np.max(np.abs(linear))), 1.0)
        ):
            corners = (
                np.stack([np.asarray(kernel.lower), np.asarray(kernel.upper)]) @ linear
                + affine[dimension]
            )
            return tensor_box_support_geometry(
                np.min(corners, axis=0),
                np.max(corners, axis=0),
                support_geometry,
                feature_id=feature_id,
                tolerance=tolerance,
            )
    if support_geometry is None:
        raise ValueError(
            "The support region of a curved or non-box IGA patch is not derived; pass "
            "an explicit support_geometry covered by the patch."
        )
    if not isinstance(support_geometry, CompiledGeometry):
        raise TypeError("support_geometry must be a CompiledGeometry or None.")
    if support_geometry.kind is not GeometryKind.REGION:
        raise ValueError("support_geometry must be a region geometry.")
    if support_geometry.ambient_dimension != dimension:
        raise ValueError("support_geometry dimension must equal the patch dimension.")
    domain = discretization.exterior_facet_domain
    boundary = _facet_evaluation(
        discretization,
        discretization.basis,
        runtime.weights,
        runtime,
        np.asarray(domain.owner_cells, dtype=np.int32),
        np.asarray(domain.owner_local_entities, dtype=np.int32),
        FacetTraceRule(points=max(discretization.basis.degrees) + 2),
    ).sites.reshape((-1, dimension))
    limit = tolerance * kernel.coordinate_scale
    boundary_field = np.asarray(support_geometry.boundary_field(jnp.asarray(boundary)))
    if np.any(np.abs(boundary_field) > limit):
        raise ValueError("The IGA patch boundary does not lie on the support boundary.")
    interior_field = np.asarray(support_geometry.boundary_field(kernel.seed_points))
    if np.any(interior_field >= 0.0):
        raise ValueError("The IGA patch interior leaves the support geometry.")
    if support_geometry.has_capability(GeometryCapability.INTERIOR_MEASURE):
        patch = _patch_measure(kernel, 2 * max(discretization.basis.degrees) + 4)
        measure = float(np.asarray(support_geometry.measure))
        if abs(measure - patch) > tolerance * max(1.0, measure):
            raise ValueError(
                "The IGA patch measure differs from the support geometry measure."
            )
    return support_geometry


def prepare_isogeometric_field_reconstruction(
    discretization: PreparedIsogeometricDiscretization,
    field_name: str,
    /,
    *,
    runtime: IsogeometricRuntimeData | None = None,
    support_geometry: object = None,
    value_port: ValuePort | None = None,
    support_tolerance: float = 1.0e-9,
    newton_iterations: int = 32,
    location_tolerance: float = 1.0e-11,
    seeds_per_axis: int = 3,
) -> PreparedFieldReconstruction:
    """Prepare an evidenced physical-coordinate reconstruction of one IGA field.

    Physical points are inverted through the runtime NURBS map (see
    `IsogeometricFieldReconstructionKernel`); `location_tolerance` bounds the
    located residual relative to the patch extent. `support_geometry` defaults
    to the box of an axis-aligned affine patch; curved patches require an
    explicit region whose boundary contains the patch boundary, whose interior
    contains the patch interior, and whose declared measure (if any) equals the
    patch measure. Regularity is `C^k` across interior knot lines, `k` the
    smallest field or geometry continuity there, with polynomial pieces for
    polynomial fields on affine patches and smooth rational pieces otherwise.
    Values and first physical derivatives are exact.
    """
    if not isinstance(discretization, PreparedIsogeometricDiscretization):
        raise TypeError("discretization must be a PreparedIsogeometricDiscretization.")
    limit = float(support_tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative.")
    field_index = discretization._field_index(field_name)
    realized = _realized_runtime(discretization, runtime)
    kernel = IsogeometricFieldReconstructionKernel(
        discretization,
        field_index,
        realized,
        newton_iterations=newton_iterations,
        location_tolerance=location_tolerance,
        seeds_per_axis=seeds_per_axis,
    )
    field = discretization.fields[field_index]
    field_space_id = discretization.field_spaces[field_index].field_space_id
    support_id = canonical_fingerprint(
        {
            "kind": "isogeometric-support",
            "support": discretization.support.support_id,
            "runtime": realized.runtime_id,
            "control_points": array_tree_fingerprint(np.asarray(realized.control_points)),
            "weights": array_tree_fingerprint(np.asarray(realized.weights)),
        }
    )
    geometry = _support_geometry(
        discretization,
        kernel,
        realized,
        support_geometry,
        feature_id=f"isogeometric-support:{support_id}",
        tolerance=limit,
    )
    polynomial = (
        _uniform(kernel.field_weights)
        and _affine_map(discretization.basis, realized) is not None
    )
    degree = sum(field.basis.degrees)
    if kernel.continuity is None:
        regularity = DerivativeRegularity.smooth(
            degree_bound=degree if polynomial else None
        )
    elif polynomial:
        regularity = DerivativeRegularity.piecewise_polynomial(
            continuity=kernel.continuity, degree_bound=degree
        )
    else:
        regularity = DerivativeRegularity.piecewise_smooth(continuity=kernel.continuity)
    components = field.component_shape
    port = (
        ValuePort(
            field.name,
            event_shape=components,
            component_ids=(
                (field.name,)
                if not components
                else tuple(f"{field.name}[{index}]" for index in range(prod(components)))
            ),
            representation="isogeometric-field",
            space_id=field_space_id,
        )
        if value_port is None
        else value_port
    )
    if not isinstance(port, ValuePort):
        raise TypeError("value_port must be a ValuePort or None.")
    if port.event_shape != components:
        raise ValueError("value_port event_shape must equal the IGA component shape.")
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=geometry,
        value_port=port,
        regularity=regularity,
        trace_policy=FieldTracePolicy("cell-sided"),
        coefficient_shape=kernel.coefficient_shape,
        physical_dimension=discretization.basis.parametric_dimension,
        maximum_derivative_order=1,
        field_space_id=field_space_id,
        support_id=support_id,
    )


def _require_own_facet_domain(
    domain: IntegrationDomain, base: IntegrationDomain, /
) -> None:
    """Refuse facet domains whose routes differ from this discretization's."""
    facets = np.asarray(domain.entity_indices, dtype=np.int32)
    base_facets = np.asarray(base.entity_indices, dtype=np.int32)
    rows = np.minimum(np.searchsorted(base_facets, facets), max(base_facets.size - 1, 0))
    routes = (
        (domain.owner_cells, base.owner_cells),
        (domain.neighbor_cells, base.neighbor_cells),
        (domain.owner_local_entities, base.owner_local_entities),
        (domain.neighbor_local_entities, base.neighbor_local_entities),
    )
    if (
        domain.support_id != base.support_id
        or domain.entity_set_id != base.entity_set_id
        or base_facets.size == 0
        or not np.array_equal(base_facets[rows], facets)
        or any(
            not np.array_equal(np.asarray(value), np.asarray(expected)[rows])
            for value, expected in routes
        )
    ):
        raise ValueError(
            "The facet domain was not produced by this isogeometric discretization."
        )


def _trace_route(
    evaluation: _FacetEvaluation,
    space: ArraySpace,
    quantity: SideTraceQuantity,
    control_shape: tuple[int, ...],
    /,
) -> SideGatherRoute:
    """Componentwise value route or normal-contracted vector route.

    The facet gathers index flattened control rows; ``control_shape`` declares
    the public tensor layout they flatten (C order, as ``LocalFieldBinding``).
    """
    basis, normals = evaluation.basis, evaluation.normals
    dimension = normals.shape[-1]
    match quantity:
        case "value":
            return SideGatherRoute(
                evaluation.dofs,
                basis.astype(space.dtype),
                coefficient_shape=space.shape,
                value_shape=space.shape[len(control_shape) :],
                row_shape=control_shape,
            )
        case "normal":
            weights = basis[..., None] * normals[:, :, None, :]
            value_shape: tuple[int, ...] = ()
        case "tangential" if dimension == 2:
            tangent = np.stack((-normals[..., 1], normals[..., 0]), axis=-1)
            weights = basis[..., None] * tangent[:, :, None, :]
            value_shape = ()
        case "tangential":
            projector = np.eye(dimension) - normals[..., :, None] * normals[..., None, :]
            weights = basis[..., None, None] * projector[:, :, None]
            value_shape = (dimension,)
        case "conormal-flux":
            raise ValueError("Conormal fluxes are not geometric trace routes.")
        case _:
            assert_never(quantity)
    return SideGatherRoute(
        evaluation.dofs,
        weights.astype(space.dtype),
        coefficient_shape=space.shape,
        mode="contracted",
        value_shape=value_shape,
        row_shape=control_shape,
    )


def prepare_isogeometric_side_trace(
    discretization: PreparedIsogeometricDiscretization,
    field_name: str,
    domain: IntegrationDomain,
    /,
    *,
    rule: FacetTraceRule,
    quantity: SideTraceQuantity = "value",
    side: FieldTraceSide = "owner",
    runtime: IsogeometricRuntimeData | None = None,
) -> PreparedTraceAction:
    """Prepare the exact trace of one IGA field on selected patch facets.

    Facets are the patch boundary faces (`exterior_facet`) or the interior knot
    faces between integration-overlay cells (`interior_facet`) of this
    discretization. Sites are the owner cell's local-facet rule points, shared
    by the `"owner"` and `"neighbor"` sides of a knot face (verified on the
    host); normals point out of the traced side cell and weights are the rule
    weights times the physical surface Jacobian. The trace acts on the compiled
    field's public coefficient layout `(*control_shape, *component_shape)` (the
    field space's vector space); its route flattens the control axes in C order
    exactly as `LocalFieldBinding.flatten`, and `support_rows` are those
    flattened control indices. `"value"` traces act componentwise; `"normal"`
    and `"tangential"` traces of vector fields contract the components against
    the outward normal. Each facet gathers only its side cell's local spline
    functions.
    """
    if not isinstance(discretization, PreparedIsogeometricDiscretization):
        raise TypeError("discretization must be a PreparedIsogeometricDiscretization.")
    if not isinstance(rule, FacetTraceRule):
        raise TypeError("rule must be a FacetTraceRule.")
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be an IntegrationDomain.")
    quantity = parse(quantity, SideTraceQuantity, "quantity")
    side = parse(side, FieldTraceSide, "side")
    field_index = discretization._field_index(field_name)
    field = discretization.fields[field_index]
    realized = _realized_runtime(discretization, runtime)
    _require_volume_patch(discretization, realized)
    dimension = discretization.basis.parametric_dimension
    match quantity:
        case "value":
            pass
        case "normal" | "tangential":
            if field.component_shape != (dimension,):
                raise ValueError(
                    f"{quantity!r} traces require a vector field with component "
                    f"shape ({dimension},)."
                )
            if quantity == "tangential" and dimension not in _TANGENTIAL_DIMENSIONS:
                raise ValueError("Tangential traces require two or three dimensions.")
        case "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners through "
                "prepare_conormal_flux, not by the discretization."
            )
        case _:
            assert_never(quantity)
    if domain.entity_indices.size == 0:
        raise ValueError("A side trace requires at least one selected facet.")
    match domain.kind:
        case "exterior_facet":
            _require_own_facet_domain(domain, discretization.exterior_facet_domain)
        case "interior_facet":
            _require_own_facet_domain(domain, discretization.interior_facet_domain)
        case _:
            raise ValueError("Side traces act on exterior or interior facet domains.")
    owner_cells = np.asarray(domain.owner_cells, dtype=np.int32)
    owner_local = np.asarray(domain.owner_local_entities, dtype=np.int32)
    match side:
        case "owner":
            side_cells, side_local = owner_cells, owner_local
        case "neighbor":
            if domain.kind != "interior_facet":
                raise ValueError(
                    "Exterior facets have no neighbor side; trace the owner side."
                )
            side_cells = np.asarray(domain.neighbor_cells, dtype=np.int32)
            side_local = np.asarray(domain.neighbor_local_entities, dtype=np.int32)
        case "average":
            raise ValueError(
                "Average side traces are not prepared; compose the owner and "
                "neighbor traces of the interior facets instead."
            )
        case _:
            assert_never(side)
    weights = _field_weights(discretization, field_index, realized)
    evaluation = _facet_evaluation(
        discretization, field.basis, weights, realized, side_cells, side_local, rule
    )
    sites, measure = evaluation.sites, evaluation.weights
    if side == "neighbor":
        owner = _facet_evaluation(
            discretization, field.basis, weights, realized, owner_cells, owner_local, rule
        )
        scale = max(
            float(
                np.max(
                    np.ptp(
                        np.asarray(realized.control_points).reshape((-1, dimension)),
                        axis=0,
                    )
                )
            ),
            1.0,
        )
        if np.max(np.abs(owner.sites - evaluation.sites)) > 1.0e-10 * scale:
            raise ValueError(
                "Owner and neighbor embeddings of the facet sites disagree; the "
                "geometry is not continuous across the selected knot faces."
            )
        sites, measure = owner.sites, owner.weights
    polynomial_trace = _uniform(weights)
    match quantity:
        case "value":
            trace_degree = evaluation.tangential_degree if polynomial_trace else None
        case "normal" | "tangential":
            straight = _affine_map(discretization.basis, realized) is not None
            trace_degree = (
                evaluation.tangential_degree if polynomial_trace and straight else None
            )
        case _:
            assert_never(quantity)
    shape = _FACET_SHAPES[dimension]
    space = discretization.field_spaces[field_index].vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError("Isogeometric field coefficients are array valued.")
    magnitude = np.abs(evaluation.basis)
    nonzero = np.max(magnitude, axis=1) > _SUPPORT_TOLERANCE * np.max(magnitude)
    descriptor = SideActionDescriptor(
        owner_id=discretization.prepared_id,
        field_space_id=discretization.field_spaces[field_index].field_space_id,
        quantity=quantity,
        representation="quadrature-values",
        orientation="unoriented" if quantity == "value" else "outward",
        approximation="exact",
        side=side,
        domain=domain,
        revision_id=canonical_fingerprint(
            {
                "kind": "isogeometric-side-geometry",
                "runtime": realized.runtime_id,
                "control_points": array_tree_fingerprint(
                    np.asarray(realized.control_points)
                ),
                "weights": array_tree_fingerprint(np.asarray(realized.weights)),
            }
        ),
        rule=rule,
        trace_degree=trace_degree,
        quadrature_exact_degree=rule.exact_degree(shape),
    )
    return PreparedTraceAction(
        descriptor,
        _trace_route(evaluation, space, quantity, field.basis.control_shape),
        space,
        sites=sites.astype(space.dtype),
        weights=measure.astype(space.dtype),
        normals=evaluation.normals.astype(space.dtype),
        support_rows=np.unique(evaluation.dofs[nonzero]),
    )


__all__ = [
    "IsogeometricFieldReconstructionKernel",
    "prepare_isogeometric_field_reconstruction",
    "prepare_isogeometric_side_trace",
]
