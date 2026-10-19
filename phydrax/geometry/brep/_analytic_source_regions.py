#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admit full-turn cylinder and circular disk source regions using rational data."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final, Literal

import equinox as eqx
import numpy as np
import numpy.typing as npt

import phydrax.geometry.brep._boolean_coincidence as boolean_coincidence_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.brep._correspondence as correspondence_owner
import phydrax.geometry.brep._intersection_curve as intersection_curve_owner

from ..._fingerprint import array_tree_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import (
    checked,
    Dim,
    HostBool,
    HostFloat64,
    HostInt32,
    HostInt64,
    parse,
    Scope,
)
from .._atlas import CurveTrimLoop
from .._rational import rational_cross, rational_dot, rational_vector
from ._coordinate_request import coordinate_source_request
from ._intersection import NativePeriodEndpoint
from ._model import BRepGeometry, BRepModel
from ._patches import AbstractCurve, CircleCurve, CylinderPatch, LineCurve, PlanePatch


type _ExactVector = tuple[Fraction, ...]
type _ExactFrame = tuple[_ExactVector, ...]


@final
class CylinderSourceRegion(StrictModule):
    """Represent a capped full-turn cylinder with source heights and a rational frame.

    ``center`` is the midpoint of the finite axial interval in world coordinates;
    ``lower_height`` and ``upper_height`` are coordinates relative to ``patch.origin``.
    The cap face order is lower, upper. Frame rows are radial first, radial second,
    and axial, with the outward lateral orientation retained by the source model.
    """

    lateral_face: int = eqx.field(static=True)
    cap_faces: tuple[int, int] = eqx.field(static=True)
    patch: CylinderPatch
    center: _ExactVector = eqx.field(static=True)
    axes: _ExactFrame = eqx.field(static=True)
    radius: Fraction = eqx.field(static=True)
    halfheight: Fraction = eqx.field(static=True)
    lower_height: Fraction = eqx.field(static=True)
    upper_height: Fraction = eqx.field(static=True)


@final
class DiskSourceRegion(StrictModule):
    """Represent one circular outer loop in a rational orthonormal source plane.

    Plane and circle frame rows retain their independently authored orientations.
    Endpoints are the edge's zero-phase mathematical-turn definitions; the coedge
    carries the corresponding p-curve endpoints and traversal sense.
    """

    face: int = eqx.field(static=True)
    edge: int = eqx.field(static=True)
    patch: PlanePatch
    curve: CircleCurve
    center: _ExactVector = eqx.field(static=True)
    plane_axes: _ExactFrame = eqx.field(static=True)
    circle_axes: _ExactFrame = eqx.field(static=True)
    radius: Fraction = eqx.field(static=True)
    lower_endpoint: NativePeriodEndpoint
    upper_endpoint: NativePeriodEndpoint
    coedge: int = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)


class _AnalyticQueryDim(Dim):
    pass


@final
class AnalyticClosestBatch(StrictModule, NonTrainableState):
    """Immutable host closest-point evidence aligned with one batch of queries."""

    __strict_contract__ = True

    points: HostFloat64[_AnalyticQueryDim, Literal[3]]
    parameters: HostFloat64[_AnalyticQueryDim, Literal[2]]
    distances: HostFloat64[_AnalyticQueryDim]
    faces: HostInt32[_AnalyticQueryDim]
    status: npt.NDArray[np.int8]
    normals: HostFloat64[_AnalyticQueryDim, Literal[3]]
    first_derivatives: HostFloat64[_AnalyticQueryDim, Literal[3]]
    second_derivatives: HostFloat64[_AnalyticQueryDim, Literal[3]]
    stationarity: HostFloat64[_AnalyticQueryDim]
    distance_lower_bounds: HostFloat64[_AnalyticQueryDim]
    unresolved: HostBool[_AnalyticQueryDim]
    refinements: HostInt32[_AnalyticQueryDim]
    parameter_bounds: HostFloat64[_AnalyticQueryDim]
    edge_indices: HostInt32[_AnalyticQueryDim]
    edge_parameters: HostFloat64[_AnalyticQueryDim]
    point_error_bounds: HostFloat64[_AnalyticQueryDim]
    query_operations: HostInt64[_AnalyticQueryDim]

    def __init__(
        self,
        *,
        points: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],
        distances: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int32],
        status: npt.NDArray[np.int8],
        normals: npt.NDArray[np.float64],
        first_derivatives: npt.NDArray[np.float64],
        second_derivatives: npt.NDArray[np.float64],
        stationarity: npt.NDArray[np.float64],
        distance_lower_bounds: npt.NDArray[np.float64],
        unresolved: npt.NDArray[np.bool_],
        refinements: npt.NDArray[np.int32],
        parameter_bounds: npt.NDArray[np.float64],
        edge_indices: npt.NDArray[np.int32],
        edge_parameters: npt.NDArray[np.float64],
        point_error_bounds: npt.NDArray[np.float64],
        query_operations: npt.NDArray[np.int64],
    ) -> None:
        scope = Scope()
        points_ = parse(
            points, HostFloat64[_AnalyticQueryDim, Literal[3]], "points", scope=scope
        )
        parameters_ = parse(
            parameters,
            HostFloat64[_AnalyticQueryDim, Literal[2]],
            "parameters",
            scope=scope,
        )
        distances_ = parse(
            distances, HostFloat64[_AnalyticQueryDim], "distances", scope=scope
        )
        faces_ = parse(faces, HostInt32[_AnalyticQueryDim], "faces", scope=scope)
        if status.dtype != np.dtype(np.int8):
            raise TypeError(
                "AnalyticClosestBatch.status must have dtype int8; "
                f"got dtype {status.dtype}"
            )
        if status.shape != distances_.shape:
            raise ValueError(
                "AnalyticClosestBatch.status must align with queries; "
                f"got shape {status.shape}"
            )
        normals_ = parse(
            normals, HostFloat64[_AnalyticQueryDim, Literal[3]], "normals", scope=scope
        )
        first_ = parse(
            first_derivatives,
            HostFloat64[_AnalyticQueryDim, Literal[3]],
            "first_derivatives",
            scope=scope,
        )
        second_ = parse(
            second_derivatives,
            HostFloat64[_AnalyticQueryDim, Literal[3]],
            "second_derivatives",
            scope=scope,
        )
        stationarity_ = parse(
            stationarity, HostFloat64[_AnalyticQueryDim], "stationarity", scope=scope
        )
        lower_ = parse(
            distance_lower_bounds,
            HostFloat64[_AnalyticQueryDim],
            "distance_lower_bounds",
            scope=scope,
        )
        unresolved_ = parse(
            unresolved, HostBool[_AnalyticQueryDim], "unresolved", scope=scope
        )
        refinements_ = parse(
            refinements, HostInt32[_AnalyticQueryDim], "refinements", scope=scope
        )
        bounds_ = parse(
            parameter_bounds,
            HostFloat64[_AnalyticQueryDim],
            "parameter_bounds",
            scope=scope,
        )
        edges_ = parse(
            edge_indices, HostInt32[_AnalyticQueryDim], "edge_indices", scope=scope
        )
        edge_parameters_ = parse(
            edge_parameters,
            HostFloat64[_AnalyticQueryDim],
            "edge_parameters",
            scope=scope,
        )
        errors_ = parse(
            point_error_bounds,
            HostFloat64[_AnalyticQueryDim],
            "point_error_bounds",
            scope=scope,
        )
        operations_ = parse(
            query_operations,
            HostInt64[_AnalyticQueryDim],
            "query_operations",
            scope=scope,
        )
        for array in (
            points_,
            parameters_,
            distances_,
            faces_,
            status,
            normals_,
            first_,
            second_,
            stationarity_,
            lower_,
            unresolved_,
            refinements_,
            bounds_,
            edges_,
            edge_parameters_,
            errors_,
            operations_,
        ):
            array.setflags(write=False)
        self.points = points_
        self.parameters = parameters_
        self.distances = distances_
        self.faces = faces_
        self.status = status
        self.normals = normals_
        self.first_derivatives = first_
        self.second_derivatives = second_
        self.stationarity = stationarity_
        self.distance_lower_bounds = lower_
        self.unresolved = unresolved_
        self.refinements = refinements_
        self.parameter_bounds = bounds_
        self.edge_indices = edges_
        self.edge_parameters = edge_parameters_
        self.point_error_bounds = errors_
        self.query_operations = operations_


def _orthonormal(axes: _ExactFrame, /) -> bool:
    return all(
        rational_dot(first, second) == int(i == j)
        for i, first in enumerate(axes)
        for j, second in enumerate(axes)
    )


def _unplaced(geometry: BRepGeometry, /) -> bool:
    return len(geometry.occurrences) <= 1 and all(
        occurrence.rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        and occurrence.translation == (0.0, 0.0, 0.0)
        for occurrence in geometry.occurrences
    )


def _face_trims_match(
    model: BRepModel, face: int, /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Bind declared trim carriers to every source coedge, without chord samples."""

    geometry = model.geometry
    domain = model.trim_domains[face]
    if geometry is None or domain is None:
        return False
    loops = geometry.face_loops[face]
    if len(domain.loops) != len(loops):
        return False
    bounds = np.asarray(model.parameter_bounds)[face]
    for declared, coedges in zip(domain.loops, loops, strict=True):
        match declared:
            case CurveTrimLoop():
                if len(declared.curves) != len(coedges):
                    return False
                for trim, coedge in zip(declared.curves, coedges, strict=True):
                    expected = constructors_owner.brep_trim_curve(
                        geometry,
                        coedge,
                        model.patches[face],
                        parameter_bounds=bounds,
                        budget=budget,
                    )
                    if array_tree_fingerprint(trim) != array_tree_fingerprint(expected):
                        return False
            case _:
                return False
    return True


@checked
def native_full_turn(
    model: BRepModel, coedge: int, /, *, zero_phase: bool = True
) -> bool:
    """Prove one authored mathematical turn on an edge and its source use."""
    geometry = model.geometry
    if geometry is None or not 0 <= coedge < len(geometry.coedge_edges):
        return False
    edge = geometry.coedge_edges[coedge]
    carrier = geometry.edge_curves[edge]
    if carrier < 0:
        return False
    for endpoints, curve in (
        (geometry.edge_endpoint_roots[edge], geometry.curves[carrier]),
        (geometry.coedge_endpoint_roots[coedge], geometry.pcurves[coedge]),
    ):
        match endpoints:
            case (NativePeriodEndpoint() as first, NativePeriodEndpoint() as last):
                if (
                    first.rational != last.rational
                    or last.turns - first.turns != 1
                    or (zero_phase and (first.rational != 0 or first.turns != 0))
                    or array_tree_fingerprint(first.curve)
                    != array_tree_fingerprint(curve)
                    or array_tree_fingerprint(last.curve) != array_tree_fingerprint(curve)
                ):
                    return False
            case _:
                return False
    return True


def _disk_bounds_contain(
    bounds: np.ndarray,
    origin: _ExactVector,
    center: _ExactVector,
    plane_axes: _ExactFrame,
    circle_axes: _ExactFrame,
    radius: Fraction,
    /,
) -> bool:
    relative = tuple(c - o for c, o in zip(center, origin, strict=True))
    for index, axis in enumerate(plane_axes):
        coordinate = rational_dot(relative, axis)
        squared_extent = radius**2 * sum(
            (rational_dot(circle_axis, axis) ** 2 for circle_axis in circle_axes),
            Fraction(0),
        )
        lower, upper = (Fraction(float(value)) for value in bounds[:, index])
        if (
            coordinate < lower
            or coordinate > upper
            or (coordinate - lower) ** 2 < squared_extent
            or (upper - coordinate) ** 2 < squared_extent
        ):
            return False
    return True


@checked
def complete_disk_region(
    model: BRepModel, face: int, /, *, budget: CoordinateEnclosureBudget | None = None
) -> DiskSourceRegion | None:
    """Admit a full-turn circular outer boundary with source-curve correspondence."""
    if budget is None:
        with coordinate_source_request(
            (
                model,
                face,
            )
        ) as budget:
            return complete_disk_region(model, face, budget=budget)

    geometry = model.geometry
    if geometry is None or not 0 <= face < len(model.patches) or not _unplaced(geometry):
        return None
    match model.patches[face]:
        case PlanePatch() as patch:
            pass
        case _:
            return None
    loops = geometry.face_loops[face]
    if (
        len(loops) != 1
        or len(loops[0]) != 1
        or not _face_trims_match(model, face, budget=budget)
    ):
        return None
    coedge = loops[0][0]
    edge = geometry.coedge_edges[coedge]
    carrier = geometry.edge_curves[edge]
    if carrier < 0 or not native_full_turn(model, coedge):
        return None
    match geometry.curves[carrier]:
        case CircleCurve() as curve:
            pass
        case _:
            return None
    match geometry.pcurves[coedge]:
        case AbstractCurve() as pcurve:
            pass
        case _:
            return None
    arrays = (
        np.asarray(patch.origin),
        np.asarray(patch.first_axis),
        np.asarray(patch.second_axis),
        np.asarray(curve.center),
        np.asarray(curve.first_axis),
        np.asarray(curve.second_axis),
        np.asarray(curve.radius),
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return None
    first_vertex, last_vertex = geometry.edge_vertices[edge]
    if (
        first_vertex != last_vertex
        or geometry.vertex_roots[first_vertex] is not None
        or not boolean_coincidence_owner.circle_vertex_is_exact(
            curve, np.asarray(geometry.vertex_points[first_vertex])
        )
        or correspondence_owner.circle_correspondence(curve, pcurve, patch) != 0.0
    ):
        return None
    origin, center = rational_vector(arrays[0]), rational_vector(arrays[3])
    plane_axes = rational_vector(arrays[1]), rational_vector(arrays[2])
    circle_axes = rational_vector(arrays[4]), rational_vector(arrays[5])
    radius = Fraction(float(curve.radius))
    if not _orthonormal(plane_axes) or not _orthonormal(circle_axes) or radius <= 0:
        return None
    normal = rational_cross(*plane_axes)
    if (
        rational_dot(tuple(c - o for c, o in zip(center, origin, strict=True)), normal)
        != 0
        or any(rational_dot(axis, normal) != 0 for axis in circle_axes)
        or rational_dot(rational_cross(*circle_axes), normal)
        * geometry.coedge_senses[coedge]
        <= 0
        or not _disk_bounds_contain(
            np.asarray(model.parameter_bounds)[face],
            origin,
            center,
            plane_axes,
            circle_axes,
            radius,
        )
    ):
        return None
    match geometry.edge_endpoint_roots[edge]:
        case (NativePeriodEndpoint() as lower, NativePeriodEndpoint() as upper):
            return DiskSourceRegion(
                face=face,
                edge=edge,
                patch=patch,
                curve=curve,
                center=center,
                plane_axes=plane_axes,
                circle_axes=circle_axes,
                radius=radius,
                lower_endpoint=lower,
                upper_endpoint=upper,
                coedge=coedge,
                reversed=geometry.coedge_senses[coedge] < 0,
            )
        case _:
            return None


def _cylinder_trims(
    model: BRepModel,
    lateral: int,
    patch: CylinderPatch,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction, tuple[int, int]] | None:

    geometry = model.geometry
    if geometry is None or not _face_trims_match(model, lateral, budget=budget):
        return None
    bounds = np.asarray(model.parameter_bounds)[lateral]
    if bounds[0, 0] != 0.0 or bounds[1, 0] != math.tau or bounds[0, 1] >= bounds[1, 1]:
        return None
    lower, upper = Fraction(float(bounds[0, 1])), Fraction(float(bounds[1, 1]))
    loops = geometry.face_loops[lateral]
    if len(loops) != 1 or len(loops[0]) != 4:
        return None
    rings: dict[Fraction, int] = {}
    seams: set[int] = set()
    segments = []
    for coedge in loops[0]:
        edge = geometry.coedge_edges[coedge]
        carrier = geometry.edge_curves[edge]
        if carrier < 0:
            return None
        curve, pcurve = geometry.curves[carrier], geometry.pcurves[coedge]
        factor = intersection_curve_owner.pcurve_periodic_source(pcurve, patch)
        if factor is None:
            return None
        base, shifts = factor
        match base:
            case LineCurve():
                origin = rational_vector(np.asarray(base.origin))
                direction = rational_vector(np.asarray(base.direction))
            case _:
                return None
        match curve:
            case CircleCurve():
                if (
                    shifts != (0, 0)
                    or origin[0] != 0
                    or direction != (Fraction(1), Fraction(0))
                    or not native_full_turn(model, coedge)
                    or correspondence_owner.circle_correspondence(curve, base, patch)
                    != 0.0
                    or geometry.edge_vertices[edge][0] != geometry.edge_vertices[edge][1]
                    or not boolean_coincidence_owner.circle_vertex_is_exact(
                        curve,
                        np.asarray(
                            geometry.vertex_points[geometry.edge_vertices[edge][0]]
                        ),
                    )
                ):
                    return None
                first, last = (0, origin[1]), (1, origin[1])
                rings[origin[1]] = edge
            case LineCurve():
                if (
                    not isinstance(pcurve, intersection_curve_owner.PeriodicPCurve)
                    or shifts[0] not in (0, 1)
                    or shifts[1] != 0
                    or origin[0] != 0
                    or direction[0] != 0
                    or not boolean_coincidence_owner.seam_is_exact(curve, base, patch)
                ):
                    return None
                a, b = (
                    Fraction(float(value))
                    for value in np.asarray(geometry.edge_ranges)[edge]
                )
                first, last = (
                    (shifts[0], origin[1] + a * direction[1]),
                    (shifts[0], origin[1] + b * direction[1]),
                )
                seams.add(edge)
            case _:
                return None
        segments.append(
            (first, last) if geometry.coedge_senses[coedge] > 0 else (last, first)
        )
    corners = ((0, lower), (1, lower), (1, upper), (0, upper))
    expected = tuple((corners[i], corners[(i + 1) % 4]) for i in range(4))
    if (
        tuple(segments) not in tuple(expected[i:] + expected[:i] for i in range(4))
        or set(rings) != {lower, upper}
        or len(seams) != 1
    ):
        return None
    caps = []
    axis = rational_vector(np.asarray(patch.axis))
    for height, side in ((lower, -1), (upper, 1)):
        cap = boolean_coincidence_owner.cap_for_ring(model, lateral, rings[height])
        if cap is None:
            return None
        disk = complete_disk_region(model, cap, budget=budget)
        if (
            disk is None
            or rational_dot(rational_cross(*disk.plane_axes), axis)
            * Fraction(float(model.orientation[cap]))
            * side
            <= 0
        ):
            return None
        caps.append(cap)
    return lower, upper, (caps[0], caps[1])


@checked
def complete_cylinder_region(
    model: BRepModel, /, *, budget: CoordinateEnclosureBudget | None = None
) -> CylinderSourceRegion | None:
    """Admit one unplaced full-turn capped cylinder from rational source data."""
    if budget is None:
        with coordinate_source_request((model,)) as budget:
            return complete_cylinder_region(model, budget=budget)
    geometry = model.geometry
    if (
        geometry is None
        or model.topology.num_solids != 1
        or len(model.patches) != 3
        or geometry.solid_shells != ((0,),)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != {0, 1, 2}
        or geometry.shell_orientations[0] != (1, 1, 1)
        or len(geometry.occurrences) != 1
        or not _unplaced(geometry)
        or any(root is not None for root in geometry.vertex_roots)
    ):
        return None
    candidates: list[tuple[int, CylinderPatch]] = []
    for index, surface in enumerate(model.patches):
        match surface:
            case CylinderPatch():
                candidates.append((index, surface))
            case _:
                pass
    if len(candidates) != 1:
        return None
    lateral, patch = candidates[0]
    axes = tuple(
        rational_vector(np.asarray(axis))
        for axis in (patch.first_axis, patch.second_axis, patch.axis)
    )
    if (
        not _orthonormal(axes)
        or rational_dot(rational_cross(axes[0], axes[1]), axes[2])
        * Fraction(float(model.orientation[lateral]))
        <= 0
    ):
        return None
    interval = _cylinder_trims(model, lateral, patch, budget=budget)
    if interval is None:
        return None
    lower, upper, caps = interval
    middle = (lower + upper) / 2
    center = tuple(
        Fraction(float(value)) + middle * direction
        for value, direction in zip(np.asarray(patch.origin), axes[2], strict=True)
    )
    if any(Fraction(float(value)) != value for value in center):
        return None
    return CylinderSourceRegion(
        lateral_face=lateral,
        cap_faces=caps,
        patch=patch,
        center=center,
        axes=axes,
        radius=Fraction(float(patch.radius)),
        halfheight=(upper - lower) / 2,
        lower_height=lower,
        upper_height=upper,
    )
