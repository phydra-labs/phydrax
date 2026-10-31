#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Point location, side geometry, and support regions of polygon meshes.

Polygon cells are located through deterministic ear-clipped triangulations.
Exact binary-rational orientation tests establish simple loops and avoid
star-kernel assumptions; a family whose basis is piecewise linear on the fan
of a star-kernel witness locates in that fan instead. Bounded BVH candidates
and fixed-size point batches
keep the query workspace independent of the number of mesh cells. A family
supplies local weights through `AbstractPolygonLocalBasis`, while
`PolygonFieldReconstructionKernel` owns side resolution, pointwise evidence,
and the exact gather/scatter route. Facet helpers prepare edge sites, measures,
outward normals, and geometry revisions for polygon side actions.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import assert_never, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model._ports import ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, positive_integer
from ..exterior._form_type import FormValueSpec
from ..linalg import ArraySpace, inverse_small_linear, SmallLinearSolvePlan
from ..typing import checked, parse
from ._cell_complex import PolygonalConnectivity
from ._cell_mesh import CellMesh
from ._integration_domain import IntegrationDomain
from ._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
    SideOrientation,
    SideTraceQuantity,
)
from ._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldSupportCoverage,
    FieldTraceSide,
)


if TYPE_CHECKING:
    from ..geometry import CompiledGeometry


_VALUE_AXES = "uvw"
_REFERENCE_GRADIENT = ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))


def polygonal_connectivity_of(mesh: CellMesh, /) -> PolygonalConnectivity:
    """Return the polygon edge incidence of a planar polygon mesh."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be a CellMesh.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Polygon side actions require PolygonalConnectivity.")
    return connectivity


def _polygon_twice_area(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
    return sum(
        (
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[index][1] * points[(index + 1) % len(points)][0]
            for index in range(len(points))
        ),
        Fraction(0),
    )


def _orientation(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
    /,
) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _point_on_segment(
    point: tuple[Fraction, Fraction],
    start: tuple[Fraction, Fraction],
    stop: tuple[Fraction, Fraction],
    /,
) -> bool:
    return (
        _orientation(start, stop, point) == 0
        and min(start[0], stop[0]) <= point[0] <= max(start[0], stop[0])
        and min(start[1], stop[1]) <= point[1] <= max(start[1], stop[1])
    )


def _segments_intersect(
    first_start: tuple[Fraction, Fraction],
    first_stop: tuple[Fraction, Fraction],
    second_start: tuple[Fraction, Fraction],
    second_stop: tuple[Fraction, Fraction],
    /,
) -> bool:
    first_left = _orientation(first_start, first_stop, second_start)
    first_right = _orientation(first_start, first_stop, second_stop)
    second_left = _orientation(second_start, second_stop, first_start)
    second_right = _orientation(second_start, second_stop, first_stop)
    if (first_left < 0 < first_right or first_right < 0 < first_left) and (
        second_left < 0 < second_right or second_right < 0 < second_left
    ):
        return True
    return (
        (first_left == 0 and _point_on_segment(second_start, first_start, first_stop))
        or (first_right == 0 and _point_on_segment(second_stop, first_start, first_stop))
        or (
            second_left == 0 and _point_on_segment(first_start, second_start, second_stop)
        )
        or (
            second_right == 0 and _point_on_segment(first_stop, second_start, second_stop)
        )
    )


def _triangulate_simple_polygon(
    coordinates: np.ndarray, vertices: np.ndarray, /
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate one simple polygon with exact binary-rational predicates."""
    points = tuple(
        (
            Fraction.from_float(float(coordinates[int(vertex), 0])),
            Fraction.from_float(float(coordinates[int(vertex), 1])),
        )
        for vertex in vertices
    )
    if len(set(points)) != len(points):
        raise ValueError("Polygon location requires distinct loop vertices.")
    count = len(points)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if second == first_next or (first == 0 and second_next == 0):
                continue
            if _segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
            ):
                raise ValueError("Polygon location requires a simple boundary loop.")
    area = _polygon_twice_area(points)
    if area == 0:
        raise ValueError("Polygon location requires nonzero exact signed area.")
    order = list(range(count)) if area > 0 else list(reversed(range(count)))
    triangles: list[tuple[int, int, int]] = []
    while len(order) > 3:
        for position, current in enumerate(order):
            previous = order[position - 1]
            following = order[(position + 1) % len(order)]
            triangle = (previous, current, following)
            if _orientation(*(points[index] for index in triangle)) <= 0:
                continue
            contains_vertex = any(
                all(
                    _orientation(
                        points[triangle[edge]],
                        points[triangle[(edge + 1) % 3]],
                        points[other],
                    )
                    >= 0
                    for edge in range(3)
                )
                for other in order
                if other not in triangle
            )
            if contains_vertex:
                continue
            triangles.append(triangle)
            del order[position]
            break
        else:
            raise ValueError("Exact ear clipping could not triangulate a simple polygon.")
    final_triangle = (order[0], order[1], order[2])
    if _orientation(*(points[index] for index in final_triangle)) <= 0:
        raise ValueError("Polygon triangulation contains a degenerate final triangle.")
    triangles.append(final_triangle)
    return tuple(triangles)


class PolygonMeshLocation(StrictModule):
    """Every containing candidate cell of each located point.

    `candidate_cells` is `-1` for a bounded candidate that does not contain the
    point; `candidate_triangles` is the first containing triangle of each
    candidate with its `candidate_barycentric` coordinates; and
    `candidate_triangle_vertices` records that triangle's local node slots.
    `candidate_triangle_shared` marks points on an internal triangulation edge.
    """

    candidate_cells: Array
    candidate_triangles: Array
    candidate_barycentric: Array
    candidate_triangle_vertices: Array
    candidate_triangle_shared: Array
    count: Array
    status: Array
    conditioning: Array


@final
class PreparedPolygonMeshLocator(StrictModule, NonTrainableState):
    """Bounded point location in arbitrary simple polygon cells.

    Without `witness`, every boundary loop is validated and triangulated with
    exact predicates over the binary-rational input coordinates; no star-kernel
    witness or guessed polygon diagonal is used. With `witness` (one point per
    cell), each cell is subdivided into its fan triangles `(w, v_i, v_{i+1})`,
    the piecewise-linear pieces of a witness-fan basis; local node slot
    `width` denotes the witness. A fan cell with a singular, inverted, or
    ill-conditioned fan triangle is degenerate. Point queries use the
    nonoverlapping triangles and bounded BVH candidates.
    """

    triangle_vertices: Array
    triangle_origin: Array
    triangle_inverse: Array
    triangle_condition: Array
    triangle_valid: Array
    cell_valid: Array
    lower: Array
    upper: Array
    bvh: PackedBVH
    tolerance: float = eqx.field(static=True)
    padding: float = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        cell_vertices: ArrayLike,
        arity: ArrayLike,
        cell_valid: ArrayLike,
        /,
        *,
        tolerance: float,
        maximum_condition: float,
        maximum_candidates: int = 32,
        batch_size: int = 64,
        witness: ArrayLike | None = None,
    ) -> None:
        points = np.asarray(coordinates)
        cells = np.asarray(cell_vertices, dtype=np.int32)
        arity_ = np.asarray(arity, dtype=np.int32)
        valid = np.asarray(cell_valid, dtype=np.bool_)
        limit = finite_real_scalar(tolerance, "tolerance")
        condition = finite_real_scalar(maximum_condition, "maximum_condition")
        capacity = positive_integer(maximum_candidates, "maximum_candidates")
        batch = positive_integer(batch_size, "batch_size")
        _validate_locator_inputs(points, cells, arity_, valid)
        if limit <= 0.0 or condition <= 1.0:
            raise ValueError("Polygon location tolerances are invalid.")
        width = cells.shape[1]
        if witness is None:
            apex = np.zeros((cells.shape[0], 2), dtype=points.dtype)
            triangle_vertices, triangle_active = _ear_triangles(
                points, cells, arity_, valid
            )
        else:
            apex = np.asarray(witness)
            if apex.shape != (cells.shape[0], 2) or not np.all(np.isfinite(apex)):
                raise ValueError(
                    "Polygon fan witnesses must be finite with shape (cells, 2)."
                )
            triangle_vertices, triangle_active = _fan_triangles(arity_, width)
        nodes = np.concatenate((points[cells], apex[:, None, :]), axis=1)
        triangle_points = nodes[
            np.arange(cells.shape[0])[:, None, None], triangle_vertices
        ]
        triangle_origin = triangle_points[:, :, 0, :]
        jacobians = np.stack(
            (
                triangle_points[:, :, 1, :] - triangle_origin,
                triangle_points[:, :, 2, :] - triangle_origin,
            ),
            axis=-1,
        )
        jacobians = np.where(triangle_active[:, :, None, None], jacobians, np.eye(2))
        inverse = inverse_small_linear(
            SmallLinearSolvePlan(
                2, singular_tolerance=limit, maximum_condition=condition
            ),
            jnp.asarray(jacobians),
        )
        triangle_ok = np.asarray(inverse.successful) & (
            np.asarray(inverse.determinant) > 0.0
        )
        cell_ok = valid & np.all(triangle_ok | ~triangle_active, axis=1)
        slot_valid = np.arange(cells.shape[1])[None, :] < arity_[:, None]
        vertices = points[cells]
        active = np.where(slot_valid[:, :, None], vertices, np.nan)
        lower = np.nanmin(active, axis=1)
        upper = np.nanmax(active, axis=1)
        extent = max(1.0, float(np.max(upper - lower)))
        self.triangle_vertices = jnp.asarray(triangle_vertices)
        self.triangle_origin = jnp.asarray(triangle_origin)
        self.triangle_inverse = inverse.value
        self.triangle_condition = inverse.condition_estimate
        self.triangle_valid = jnp.asarray(triangle_active & triangle_ok)
        self.cell_valid = jnp.asarray(cell_ok)
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)
        self.bvh = prepare_bvh(
            lower,
            upper,
            policy=BVHBuildPolicy(leaf_size=min(8, cells.shape[0])),
            dtype=points.dtype,
        )
        self.tolerance = limit
        self.padding = limit * extent
        self.maximum_candidates = min(capacity, cells.shape[0])
        self.batch_size = batch
        self.locator_id = canonical_fingerprint(
            {
                "kind": "prepared-polygon-mesh-locator",
                "vertices": array_tree_fingerprint(vertices),
                "cells": array_tree_fingerprint(cells),
                "arity": array_tree_fingerprint(arity_),
                "triangles": array_tree_fingerprint(triangle_vertices),
                "witness": None if witness is None else array_tree_fingerprint(apex),
                "cell_valid": array_tree_fingerprint(valid),
                "tolerance": limit,
                "maximum_condition": condition,
                "maximum_candidates": self.maximum_candidates,
                "batch_size": batch,
            }
        )

    @property
    def cell_count(self) -> int:
        return self.cell_valid.shape[0]

    def _locate_point(
        self, point: Array, cells: Array, valid: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        relative = point[None, None, :] - self.triangle_origin[cells]
        reference = ein.contract("ktrd,ktd->ktr", self.triangle_inverse[cells], relative)
        barycentric = jnp.concatenate(
            (1.0 - jnp.sum(reference, axis=-1, keepdims=True), reference), axis=-1
        )
        inside_triangles = (
            self.triangle_valid[cells]
            & valid[:, None]
            & jnp.all(barycentric >= -self.tolerance, axis=-1)
        )
        inside = jnp.any(inside_triangles, axis=1) & self.cell_valid[cells]
        triangles = jnp.argmax(inside_triangles, axis=1).astype(jnp.int32)
        shared = jnp.sum(inside_triangles, axis=1) > 1
        rows = jnp.arange(cells.shape[0])
        in_box = valid & jnp.all(
            (point >= self.lower[cells] - self.padding)
            & (point <= self.upper[cells] + self.padding),
            axis=-1,
        )
        degenerate = jnp.any(in_box & ~self.cell_valid[cells])
        return (
            inside,
            triangles,
            barycentric[rows, triangles],
            self.triangle_vertices[cells, triangles],
            shared & inside,
            degenerate,
            self.triangle_condition[cells, triangles],
        )

    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> PolygonMeshLocation:
        """Locate `(n, 2)` points, optionally restricted to masked cells."""
        values = jnp.asarray(points, dtype=self.triangle_origin.dtype)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Polygon location points must have shape (points, 2).")
        candidates, valid, complete = point_select_leaf_items(
            values,
            bvh=self.bvh,
            maximum_candidates=self.maximum_candidates,
            tolerance=self.padding,
        )
        if cell_mask is not None:
            mask = jnp.asarray(cell_mask)
            if mask.shape != (self.cell_count,) or mask.dtype != jnp.bool_:
                raise ValueError(
                    "cell_mask must be a boolean array with one entry per cell."
                )
            valid = valid & mask[candidates]
        (
            inside,
            triangles,
            barycentric,
            triangle_vertices,
            shared,
            degenerate,
            condition,
        ) = jax.lax.map(
            lambda arguments: self._locate_point(*arguments),
            (values, candidates, valid),
            batch_size=self.batch_size,
        )
        count = jnp.sum(inside, axis=1, dtype=jnp.int32)
        finite = jnp.all(jnp.isfinite(values), axis=1)
        status = jnp.where(
            ~finite,
            int(FieldQueryStatus.NONFINITE),
            jnp.where(
                ~complete | degenerate,
                int(FieldQueryStatus.LOCATION_FAILED),
                jnp.where(
                    count > 0,
                    int(FieldQueryStatus.VALID),
                    int(FieldQueryStatus.OUTSIDE_SUPPORT),
                ),
            ),
        ).astype(jnp.int32)
        conditioning = jnp.where(
            count > 0, jnp.max(jnp.where(inside, condition, 0.0), axis=1), jnp.inf
        )
        return PolygonMeshLocation(
            jnp.where(inside, candidates, -1),
            triangles,
            barycentric,
            triangle_vertices,
            shared,
            count,
            status,
            conditioning,
        )


def _ear_triangles(
    points: np.ndarray, cells: np.ndarray, arity: np.ndarray, valid: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Host `(triangle slots, active)` of the exact ear clipping of valid cells."""
    maximum_triangles = cells.shape[1] - 2
    triangle_vertices = np.zeros((cells.shape[0], maximum_triangles, 3), dtype=np.int32)
    triangle_active = np.zeros((cells.shape[0], maximum_triangles), dtype=np.bool_)
    for cell in range(cells.shape[0]):
        if not valid[cell]:
            continue
        triangulation = _triangulate_simple_polygon(
            points, cells[cell, : int(arity[cell])]
        )
        triangle_vertices[cell, : len(triangulation)] = triangulation
        triangle_active[cell, : len(triangulation)] = True
    return triangle_vertices, triangle_active


def _fan_triangles(arity: np.ndarray, width: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Host `(triangle slots, active)` of the witness fans `(width, i, i + 1)`."""
    slots = np.arange(width, dtype=np.int32)[None, :]
    active = slots < arity[:, None]
    following = np.where(slots + 1 < arity[:, None], slots + 1, 0).astype(np.int32)
    triangle_vertices = np.stack(
        (
            np.full_like(following, width),
            np.broadcast_to(slots, following.shape),
            following,
        ),
        axis=-1,
    )
    return triangle_vertices, active


def _validate_locator_inputs(
    points: np.ndarray,
    cells: np.ndarray,
    arity: np.ndarray,
    valid: np.ndarray,
    /,
) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Polygon locator coordinates must have shape (vertices, 2).")
    if not np.issubdtype(points.dtype, np.floating) or not np.all(np.isfinite(points)):
        raise ValueError("Polygon locator coordinates must be finite floating point.")
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 3:
        raise ValueError("Polygon locator cells must have shape (cells, width >= 3).")
    if (
        arity.shape != cells.shape[:1]
        or np.any(arity < 3)
        or np.any(arity > cells.shape[1])
    ):
        raise ValueError("Polygon arities must lie in [3, width] for every cell.")
    if np.any(cells < 0) or np.any(cells >= points.shape[0]):
        raise ValueError("Polygon locator cells index an undeclared vertex.")
    if valid.shape != cells.shape[:1]:
        raise ValueError("cell_valid must provide one flag per cell.")


class LocatedPolygonPoints(StrictModule):
    """Located points with their cell, selected triangle, and local node slots.

    `barycentric` are the coordinates of `points` in `triangle_vertices` of
    `triangles`; `triangle_inverse` is the selected triangle's inverse
    Jacobian (rows: reference axes, columns: physical axes).
    """

    cells: Array
    triangles: Array
    barycentric: Array
    triangle_vertices: Array
    triangle_inverse: Array
    points: Array


class AbstractPolygonLocalBasis(StrictModule):
    """Family-owned oriented local weights of points located in polygon cells.

    `weights(located, derivative)` returns the weights of every local DOF of
    each point's cell with shape `(points, local_width, *value_shape)` for the
    coordinate multi-index `derivative` (total order at most one); padded
    local slots carry zero weight. `triangulation_continuity` is continuity
    across internal triangulation edges, or `None` for one smooth cell piece.
    """

    @property
    @abc.abstractmethod
    def basis_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def local_width(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def value_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def triangulation_continuity(self) -> int | None:
        raise NotImplementedError

    @abc.abstractmethod
    def weights(
        self, located: LocatedPolygonPoints, derivative: tuple[int, ...], /
    ) -> Array:
        raise NotImplementedError


def triangle_barycentric_gradients(triangle_inverse: Array, /) -> Array:
    """Physical gradients `(points, 3, 2)` of selected triangle coordinates."""
    reference = jnp.asarray(_REFERENCE_GRADIENT, dtype=triangle_inverse.dtype)
    return ein.contract("ar,nrd->nad", reference, triangle_inverse)


class _PolygonRoute(StrictModule):
    dofs: Array
    weights: Array


@final
class PolygonFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Located evaluation of one polygon-mesh field through a family basis.

    Every query point is located in the locator triangles of bounded BVH
    candidate cells. Values and first derivatives come from the family's local
    basis and padded cell DOF routes. Derivatives across an internal
    triangulation edge are unresolved when that basis lacks continuity.
    """

    locator: PreparedPolygonMeshLocator
    basis: AbstractPolygonLocalBasis
    cell_dofs: Array
    continuity: int = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        locator: PreparedPolygonMeshLocator,
        basis: AbstractPolygonLocalBasis,
        cell_dofs: ArrayLike,
        /,
        *,
        continuity: int,
        global_dof_count: int,
        field_space_id: str,
    ) -> None:
        routes = np.asarray(cell_dofs, dtype=np.int32)
        count = positive_integer(global_dof_count, "global_dof_count")
        if routes.shape != (locator.cell_count, basis.local_width):
            raise ValueError("Cell DOF routes must have shape (cells, local_width).")
        if np.any(routes < 0) or np.any(routes >= count):
            raise ValueError("Cell DOF routes lie outside the coefficient rows.")
        match continuity:
            case -1 | 0:
                pass
            case _:
                raise ValueError("Polygon field continuity must be -1 (L2) or 0 (H1).")
        self.locator = locator
        self.basis = basis
        self.cell_dofs = jnp.asarray(routes)
        self.continuity = continuity
        self.global_dof_count = count
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "polygon-field-reconstruction-kernel",
                "locator": locator.locator_id,
                "basis": basis.basis_id,
                "routes": array_tree_fingerprint(routes),
                "field_space": field_space_id,
                "continuity": continuity,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return self.locator.cell_count

    @property
    def support_coverage(self) -> FieldSupportCoverage:
        return "complete"

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[_PolygonRoute, FieldQueryEvidence]:
        mask = None if side is None else side.cell_mask
        location = self.locator.locate(points, cell_mask=mask)
        average = side is not None and side.side == "average"
        cells, triangles, barycentric, triangle_vertices, share = _selected_candidates(
            location, average
        )
        slots = cells.shape[1]
        flat_cells = jnp.maximum(cells, 0).reshape((-1,))
        flat_triangles = triangles.reshape((-1,))
        located = LocatedPolygonPoints(
            flat_cells,
            flat_triangles,
            barycentric.reshape((-1, 3)),
            triangle_vertices.reshape((-1, 3)),
            self.locator.triangle_inverse[flat_cells, flat_triangles],
            jnp.repeat(points, slots, axis=0),
        )
        weights = self.basis.weights(located, derivative)
        weights = weights.reshape((points.shape[0], slots) + weights.shape[1:])
        share = share.astype(weights.dtype).reshape(
            share.shape + (1,) * (weights.ndim - 2)
        )
        route = _PolygonRoute(
            self.cell_dofs[flat_cells].reshape((points.shape[0], slots, -1)),
            weights * share,
        )
        evidence = FieldQueryEvidence(
            self._status(location, sum(derivative), side),
            location.conditioning,
            location.count,
            kernel_id=self.kernel_id,
        )
        return route, evidence

    def _status(
        self,
        location: PolygonMeshLocation,
        order: int,
        side: FieldSideBinding | None,
        /,
    ) -> Array:
        status = location.status
        valid = status == int(FieldQueryStatus.VALID)
        triangulation_continuity = self.basis.triangulation_continuity
        if triangulation_continuity is not None and order > triangulation_continuity:
            on_internal_edge = jnp.any(location.candidate_triangle_shared, axis=1)
            status = jnp.where(
                valid & on_internal_edge, int(FieldQueryStatus.SIDE_UNRESOLVED), status
            )
            valid = status == int(FieldQueryStatus.VALID)
        if side is not None and side.side == "average":
            return status
        side_status = (
            FieldQueryStatus.SIDE_REQUIRED
            if side is None
            else FieldQueryStatus.SIDE_UNRESOLVED
        )
        ambiguous = (location.count > 1) & (order > self.continuity)
        return jnp.where(valid & ambiguous, int(side_status), status)

    def _subscripts(self) -> str:
        return _VALUE_AXES[: len(self.basis.value_shape)]

    def apply(self, route: _PolygonRoute, coefficients: Array, /) -> Array:
        gathered = coefficients[route.dofs]
        if not self.basis.value_shape:
            return ein.contract("psl,psl...->p...", route.weights, gathered)
        values = self._subscripts()
        return ein.contract(f"psl{values},psl->p{values}", route.weights, gathered)

    def transpose(self, route: _PolygonRoute, cotangent: Array, /) -> Array:
        if not self.basis.value_shape:
            payload = ein.contract("psl,p...->psl...", route.weights, cotangent)
        else:
            values = self._subscripts()
            payload = ein.contract(
                f"psl{values},p{values}->psl", route.weights, cotangent
            )
        shape = (self.global_dof_count, *payload.shape[3:])
        return jnp.zeros(shape, dtype=payload.dtype).at[route.dofs].add(payload)

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        location = self.locator.locate(jnp.asarray(sites))
        if np.any(np.asarray(location.status) != int(FieldQueryStatus.VALID)):
            raise ValueError("Every trace site must lie in the polygon support.")
        containing = np.asarray(location.candidate_cells)
        match side:
            case "average":
                if cell_ids is not None:
                    raise ValueError("Average traces combine every containing cell.")
                return None, np.full(sites.shape[0], -1, dtype=np.int32)
            case "owner" | "neighbor":
                site_cells = _side_cells(containing, side, cell_ids, self.cell_count)
            case _:
                assert_never(side)
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        masked = self.locator.locate(jnp.asarray(sites), cell_mask=jnp.asarray(mask))
        if np.any(np.asarray(masked.count) != 1) or np.any(
            np.max(np.asarray(masked.candidate_cells), axis=1) != site_cells
        ):
            raise ValueError(
                "The side cells do not resolve every trace site to exactly its "
                "declared cell; the side region is ambiguous."
            )
        return mask, site_cells


def _selected_candidates(
    location: PolygonMeshLocation, average: bool, /
) -> tuple[Array, Array, Array, Array, Array]:
    accepted = location.candidate_cells >= 0
    if average:
        share = accepted / jnp.maximum(location.count, 1)[:, None]
        return (
            location.candidate_cells,
            location.candidate_triangles,
            location.candidate_barycentric,
            location.candidate_triangle_vertices,
            share,
        )
    first = jnp.argmax(accepted, axis=1)[:, None]
    return (
        jnp.take_along_axis(location.candidate_cells, first, axis=1),
        jnp.take_along_axis(location.candidate_triangles, first, axis=1),
        jnp.take_along_axis(location.candidate_barycentric, first[:, :, None], axis=1),
        jnp.take_along_axis(
            location.candidate_triangle_vertices, first[:, :, None], axis=1
        ),
        jnp.any(accepted, axis=1)[:, None],
    )


def _side_cells(
    containing: np.ndarray,
    side: FieldTraceSide,
    cell_ids: np.ndarray | None,
    cell_count: int,
    /,
) -> np.ndarray:
    if cell_ids is None:
        single = np.sum(containing >= 0, axis=1) == 1
        if side != "owner" or not np.all(single):
            raise ValueError(
                f"{side!r} traces at sites shared by several cells (or without a "
                "neighbor cell) require explicit cell_ids."
            )
        return np.max(containing, axis=1).astype(np.int32)
    if np.any(cell_ids < 0) or np.any(cell_ids >= cell_count):
        raise ValueError("cell_ids must name polygon cells.")
    if np.any(np.all(containing != cell_ids[:, None], axis=1)):
        raise ValueError("cell_ids must name a cell containing each trace site.")
    return cell_ids.astype(np.int32)


def polygon_mesh_support_geometry(
    coordinates: ArrayLike,
    cell_vertices: ArrayLike,
    arity: ArrayLike,
    support_id: str,
    /,
) -> CompiledGeometry:
    """Compile the region covered by exact simple-polygon triangulations."""
    from ..geometry.simplicial._io import planar_region_from_triangles

    points = np.asarray(coordinates, dtype=np.float64)
    cells = np.asarray(cell_vertices, dtype=np.int32)
    arity_ = np.asarray(arity, dtype=np.int32)
    triangles = np.asarray(
        [
            tuple(int(vertices[index]) for index in triangle)
            for cell in range(cells.shape[0])
            for vertices in (cells[cell, : int(arity_[cell])],)
            for triangle in _triangulate_simple_polygon(points, vertices)
        ],
        dtype=np.int32,
    )
    region = planar_region_from_triangles(
        points,
        triangles,
        recenter=False,
        feature_id=f"polygon-mesh-support:{support_id}",
    )
    return region.compile()


def polygon_side_revision(runtime_id: str, coordinates: ArrayLike, /) -> str:
    """Geometry revision of polygon side actions prepared on one runtime."""
    return canonical_fingerprint(
        {
            "kind": "polygon-side-revision",
            "runtime": runtime_id,
            "coordinates": array_tree_fingerprint(np.asarray(coordinates)),
        }
    )


def polygon_value_port(
    name: str,
    event_shape: tuple[int, ...],
    representation: str,
    space_id: str,
    value_port: ValuePort | None,
    /,
    *,
    form: FormValueSpec | None = None,
) -> ValuePort:
    """Return the declared value port of a polygon reconstruction or its default."""
    if value_port is None:
        size = math.prod(event_shape)
        return ValuePort(
            name,
            event_shape=event_shape,
            component_ids=(name,)
            if not event_shape
            else tuple(f"{name}[{index}]" for index in range(size)),
            representation=representation,
            space_id=space_id,
            form=form,
        )
    if not isinstance(value_port, ValuePort):
        raise TypeError("value_port must be a ValuePort or None.")
    if value_port.event_shape != event_shape:
        raise ValueError(
            f"value_port event_shape must equal the reconstruction value shape "
            f"{event_shape}."
        )
    if form is not None:
        if value_port.form is None:
            raise ValueError("value_port must declare the reconstruction form.")
        if value_port.form.value_spec_id != form.value_spec_id:
            raise ValueError("value_port form must match the reconstruction form.")
    return value_port


@final
@dataclass(frozen=True)
class PolygonFacetSites:
    """Host geometry of one side of selected polygon edges.

    `sites` are the physical rule points of each facet in the owner cell's
    local edge parametrization (the same points for either side);
    `canonical_parameters` locate them in `[0, 1]` along the canonical edge
    from its lower to its higher vertex index; `side_signs` are `+1` when the
    side cell traverses the canonical edge direction; `normals` are unit
    outward normals of the side cell and `weights` the physical edge measure.
    """

    side: FieldTraceSide
    edges: np.ndarray
    side_cells: np.ndarray
    side_signs: np.ndarray
    canonical_parameters: np.ndarray
    sites: np.ndarray
    weights: np.ndarray
    normals: np.ndarray
    revision_id: str


def require_polygon_facet_domain(mesh: CellMesh, domain: IntegrationDomain, /) -> None:
    """Refuse a facet domain that is not an edge domain of `mesh`."""
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be an IntegrationDomain.")
    match domain.kind:
        case "exterior_facet" | "interior_facet":
            pass
        case _:
            raise ValueError("Polygon side actions act on exterior or interior facets.")
    if (
        domain.support_id != mesh.support.support_id
        or domain.entity_set_id != mesh.topology.entity_sets[1].entity_set_id
    ):
        raise ValueError(
            "The facet domain belongs to another owner's support; prepare it from "
            "this discretization."
        )


def _require_incidence(
    connectivity: PolygonalConnectivity,
    edges: np.ndarray,
    cells: np.ndarray,
    local: np.ndarray,
    /,
) -> None:
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    arity = np.asarray(connectivity.cell_kinds, dtype=np.int32)
    if np.any(cells < 0) or np.any(cells >= cell_edges.shape[0]):
        raise ValueError("The facet domain routes name undeclared cells.")
    if np.any(local < 0) or np.any(local >= arity[cells]):
        raise ValueError("The facet domain routes name undeclared local edges.")
    if np.any(cell_edges[cells, local] != edges):
        raise ValueError("The facet domain routes do not match the mesh edge incidence.")


def _edge_endpoints(
    connectivity: PolygonalConnectivity,
    points: np.ndarray,
    cells: np.ndarray,
    local: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(connectivity.cell_vertices, dtype=np.int32)
    arity = np.asarray(connectivity.cell_kinds, dtype=np.int32)
    following = (local + 1) % arity[cells]
    return points[vertices[cells, local]], points[vertices[cells, following]]


def _require_matching_neighbor(
    connectivity: PolygonalConnectivity,
    points: np.ndarray,
    domain: IntegrationDomain,
    start: np.ndarray,
    stop: np.ndarray,
    /,
) -> None:
    neighbor_start, neighbor_stop = _edge_endpoints(
        connectivity,
        points,
        np.asarray(domain.neighbor_cells, dtype=np.int32),
        np.asarray(domain.neighbor_local_entities, dtype=np.int32),
    )
    scale = max(1.0, float(np.max(np.abs(points))))
    defect = max(
        float(np.max(np.abs(neighbor_start - stop))),
        float(np.max(np.abs(neighbor_stop - start))),
    )
    if defect > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(
            "The owner and neighbor embeddings of an interior facet disagree; the "
            "side sites are not shared physical points."
        )


def _side_routes(
    domain: IntegrationDomain, side: FieldTraceSide, /
) -> tuple[np.ndarray, np.ndarray]:
    match side:
        case "owner":
            return (
                np.asarray(domain.owner_cells, dtype=np.int32),
                np.asarray(domain.owner_local_entities, dtype=np.int32),
            )
        case "neighbor":
            if domain.kind != "interior_facet":
                raise ValueError(
                    "Exterior facets have no neighbor side; prepare the owner trace."
                )
            return (
                np.asarray(domain.neighbor_cells, dtype=np.int32),
                np.asarray(domain.neighbor_local_entities, dtype=np.int32),
            )
        case "average":
            raise ValueError(
                "Side actions are one-sided; compose the owner and neighbor actions "
                "instead of side='average'."
            )
        case _:
            assert_never(side)


def polygon_side_frame(
    mesh: CellMesh,
    coordinates: ArrayLike,
    domain: IntegrationDomain,
    side: FieldTraceSide,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return host `(side_cells, side_local_edges, unit outward normals)`.

    The normal of each facet is the outward normal of the side cell's own
    counter-clockwise local edge. The domain routes are verified against the
    mesh edge incidence, and the owner and neighbor embeddings of interior
    facets must coincide.
    """
    connectivity = polygonal_connectivity_of(mesh)
    require_polygon_facet_domain(mesh, domain)
    side_ = parse(side, FieldTraceSide, "side")
    side_cells, side_local = _side_routes(domain, side_)
    edges = np.asarray(domain.entity_indices, dtype=np.int32)
    owner = np.asarray(domain.owner_cells, dtype=np.int32)
    owner_local = np.asarray(domain.owner_local_entities, dtype=np.int32)
    _require_incidence(connectivity, edges, owner, owner_local)
    points = np.asarray(coordinates).astype(np.float64)
    if domain.kind == "interior_facet":
        _require_incidence(
            connectivity,
            edges,
            np.asarray(domain.neighbor_cells, dtype=np.int32),
            np.asarray(domain.neighbor_local_entities, dtype=np.int32),
        )
        start, stop = _edge_endpoints(connectivity, points, owner, owner_local)
        _require_matching_neighbor(connectivity, points, domain, start, stop)
    side_start, side_stop = _edge_endpoints(connectivity, points, side_cells, side_local)
    tangent = side_stop - side_start
    lengths = np.sqrt(np.sum(tangent * tangent, axis=-1))
    if np.any(lengths <= 0.0):
        raise ValueError("Polygon side actions require edges of positive length.")
    normals = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1) / lengths[:, None]
    return side_cells, side_local, normals


def prepare_polygon_facet_sites(
    mesh: CellMesh,
    coordinates: ArrayLike,
    runtime_id: str,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    side: FieldTraceSide,
    /,
) -> PolygonFacetSites:
    """Prepare the sites, measures, and outward normals of one facet side."""
    if not isinstance(rule, FacetTraceRule):
        raise TypeError("rule must be a FacetTraceRule.")
    side_cells, side_local, normals = polygon_side_frame(mesh, coordinates, domain, side)
    connectivity = polygonal_connectivity_of(mesh)
    edges = np.asarray(domain.entity_indices, dtype=np.int32)
    owner = np.asarray(domain.owner_cells, dtype=np.int32)
    owner_local = np.asarray(domain.owner_local_entities, dtype=np.int32)
    raw = np.asarray(coordinates)
    start, stop = _edge_endpoints(
        connectivity, raw.astype(np.float64), owner, owner_local
    )
    parameters, reference_weights = rule.reference("edge")
    t = parameters[:, 0]
    lengths = np.sqrt(np.sum((stop - start) ** 2, axis=-1))
    signs = np.asarray(connectivity.cell_edge_signs, dtype=np.float64)
    owner_forward = signs[owner, owner_local] > 0.0
    return PolygonFacetSites(
        side=parse(side, FieldTraceSide, "side"),
        edges=edges,
        side_cells=side_cells,
        side_signs=signs[side_cells, side_local],
        canonical_parameters=np.where(owner_forward[:, None], t[None], 1.0 - t[None]),
        sites=(1.0 - t)[None, :, None] * start[:, None, :]
        + t[None, :, None] * stop[:, None, :],
        weights=lengths[:, None] * reference_weights[None, :],
        normals=np.broadcast_to(normals[:, None, :], (edges.size, t.size, 2)),
        revision_id=polygon_side_revision(runtime_id, raw),
    )


def polygon_trace_action(
    sites: PolygonFacetSites,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    dofs: np.ndarray,
    weights: np.ndarray,
    coefficient_space: ArraySpace,
    /,
    *,
    owner_id: str,
    field_space_id: str,
    quantity: SideTraceQuantity,
    trace_degree: int,
) -> PreparedTraceAction:
    """Assemble an exact polygon edge trace from per-facet gather weights."""
    orientation: SideOrientation
    match quantity:
        case "value":
            orientation = "unoriented"
        case "normal" | "tangential":
            orientation = "outward"
        case "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners, not by "
                "discretization traces."
            )
        case _:
            assert_never(quantity)
    route = SideGatherRoute(
        dofs,
        np.asarray(weights, dtype=coefficient_space.dtype),
        coefficient_shape=coefficient_space.shape,
        mode="componentwise",
        value_shape=coefficient_space.shape[1:],
    )
    descriptor = SideActionDescriptor(
        owner_id=owner_id,
        field_space_id=field_space_id,
        quantity=quantity,
        representation="quadrature-values",
        orientation=orientation,
        approximation="exact",
        side=sites.side,
        domain=domain,
        revision_id=sites.revision_id,
        rule=rule,
        trace_degree=trace_degree,
        quadrature_exact_degree=rule.exact_degree("edge"),
    )
    return PreparedTraceAction(
        descriptor,
        route,
        coefficient_space,
        sites=sites.sites,
        weights=sites.weights,
        normals=sites.normals,
        support_rows=np.unique(dofs),
    )


def subset_polygon_domain(
    base: IntegrationDomain, rows: np.ndarray, /
) -> IntegrationDomain:
    """Restrict an owner integration domain to `rows`, keeping every route."""
    return IntegrationDomain(
        base.kind,
        np.asarray(base.entity_indices)[rows],
        base.support_id,
        base.entity_set_id,
        owner_cells=np.asarray(base.owner_cells)[rows],
        neighbor_cells=np.asarray(base.neighbor_cells)[rows],
        owner_local_entities=np.asarray(base.owner_local_entities)[rows],
        neighbor_local_entities=np.asarray(base.neighbor_local_entities)[rows],
        neighbor_trace_permutations=np.asarray(base.neighbor_trace_permutations)[rows],
        periodic_face_mask=np.asarray(base.periodic_face_mask)[rows],
        selection_id=base.selection_id,
    )


def polygon_facet_domain(
    base: IntegrationDomain, facets: ArrayLike, /
) -> IntegrationDomain:
    """Restrict `base` to `facets`, in the order of `facets`."""
    entities = np.asarray(base.entity_indices, dtype=np.int32)
    selected = np.asarray(facets, dtype=np.int32)
    order = np.argsort(entities, kind="stable")
    positions = np.searchsorted(entities[order], selected)
    positions = np.minimum(positions, max(entities.size - 1, 0))
    rows = order[positions]
    if entities.size == 0 or np.any(entities[rows] != selected):
        raise ValueError("The facets are not selected by the owner facet domain.")
    return subset_polygon_domain(base, rows)


__all__ = [
    "AbstractPolygonLocalBasis",
    "LocatedPolygonPoints",
    "PolygonFacetSites",
    "PolygonFieldReconstructionKernel",
    "PolygonMeshLocation",
    "PreparedPolygonMeshLocator",
    "polygon_facet_domain",
    "polygon_mesh_support_geometry",
    "polygon_side_frame",
    "polygon_side_revision",
    "polygon_trace_action",
    "polygon_value_port",
    "polygonal_connectivity_of",
    "prepare_polygon_facet_sites",
    "require_polygon_facet_domain",
    "subset_polygon_domain",
    "triangle_barycentric_gradients",
]
