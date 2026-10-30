#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Opt-in STEP topology repairs; never change a carrier or a tolerance."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from ..geometry.brep._boolean_core import PeriodicPCurve
from ..geometry.brep._constructors import assemble_brep_model, BRepTessellationPolicy
from ..geometry.brep._measure import brep_shell_volume_sign
from ..geometry.brep._model import BRepGeometry, BRepModel
from ..geometry.brep._patches import (
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    SurfaceIsoparametricCurve,
)
from ._cad import CadInterchangeError, CadStage, refuse, StagedCoedge
from ._report import AdapterLoss


type _Point3 = tuple[Fraction, Fraction, Fraction]
type _Plane = tuple[_Point3, _Point3, int]
type _ConvexShell = tuple[
    tuple[_Point3, ...],
    tuple[_Plane, ...],
    tuple[tuple[_Point3, _Point3], ...],
]


@final
class StepRepairPolicy(StrictModule):
    """Permit only named geometry-preserving repairs.

    All switches default to false. Face-sense repair supports closed convex
    linear polyhedra with rational supporting-plane evidence. Seam insertion
    supports coaxial cylinder bands and native sphere/torus natural charts.
    Other families refuse explicitly. No switch permits moving geometry,
    fitting a repair, or growing tolerance.
    """

    add_missing_seams: bool = eqx.field(static=True)
    reverse_inverted_shells: bool = eqx.field(static=True)
    orient_face_senses: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        add_missing_seams: bool = False,
        reverse_inverted_shells: bool = False,
        orient_face_senses: bool = False,
    ) -> None:
        for value in (add_missing_seams, reverse_inverted_shells, orient_face_senses):
            if not isinstance(value, bool):
                raise TypeError(
                    f"StepRepairPolicy.switches must be bool values; got {value!r}"
                )
        self.add_missing_seams = add_missing_seams
        self.reverse_inverted_shells = reverse_inverted_shells
        self.orient_face_senses = orient_face_senses


def repair_record(entity: str, reason: str, /) -> AdapterLoss:
    """Retain the source entity and the proof authorizing its repair."""
    return AdapterLoss(
        f"geometry.step-repairs.{entity}",
        "import",
        "transformed",
        reason,
        changes_interpretation=False,
    )


def supporting_plane_sense(
    patch: PlanePatch,
    points: tuple[np.ndarray, ...],
    label: str,
    /,
) -> int:
    """Prove outward coorientation of a convex linear shell, using rational dots."""
    sides = _plane_sides(patch, points)
    if sides and all(value <= 0 for value in sides) and any(value < 0 for value in sides):
        return 1
    if sides and all(value >= 0 for value in sides) and any(value > 0 for value in sides):
        return -1
    raise refuse(
        "unsupported-entity",
        label,
        "Face-sense repair requires a strict rational supporting plane of a convex linear shell; "
        "nonconvex, degenerate, curved, or unresolved outward tests refuse.",
    )


def _plane_sides(
    patch: PlanePatch, points: tuple[np.ndarray, ...], /
) -> tuple[Fraction, ...]:
    first = tuple(Fraction(float(value)) for value in np.asarray(patch.first_axis))
    second = tuple(Fraction(float(value)) for value in np.asarray(patch.second_axis))
    origin = tuple(Fraction(float(value)) for value in np.asarray(patch.origin))
    normal = tuple(
        first[(axis + 1) % 3] * second[(axis + 2) % 3]
        - first[(axis + 2) % 3] * second[(axis + 1) % 3]
        for axis in range(3)
    )
    return tuple(
        sum(
            (
                (Fraction(float(point[axis])) - origin[axis]) * normal[axis]
                for axis in range(3)
            ),
            Fraction(0),
        )
        for point in points
    )


def _assemble(
    model: BRepModel,
    geometry: BRepGeometry,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> BRepModel:
    return assemble_brep_model(
        geometry,
        model.patches,
        model.parameter_bounds,
        model.orientation,
        model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        source_id=model.report.source_id,
        source_format=model.report.source_format,
        source_digest=model.report.source_digest,
        import_policy_id=model.report.import_policy_id,
        converted_surface_count=model.report.converted_surface_count,
        curve_surface_tolerance=model.report.curve_surface_tolerance,
        curve_surface_scale=model.report.curve_surface_scale,
        tessellation=tessellation,
        trim_domains=model.trim_domains,
    )


def _shell_polyhedron(
    model: BRepModel, geometry: BRepGeometry, shell: int, /
) -> _ConvexShell:
    vertex_ids = tuple(
        sorted(
            {
                vertex
                for face in geometry.shell_faces[shell]
                for loop in geometry.face_loops[face]
                for coedge in loop
                for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            }
        )
    )
    if not vertex_ids:
        raise ValueError(f"Shell {shell} has no source vertices.")
    exact_points = geometry.vertex_exact_points
    points_list: list[_Point3] = []
    for vertex in vertex_ids:
        exact_point = exact_points[vertex]
        if exact_point is not None:
            points_list.append((exact_point[0], exact_point[1], exact_point[2]))
        else:
            point = np.asarray(geometry.vertex_points[vertex])
            points_list.append(
                (
                    Fraction.from_float(float(point[0])),
                    Fraction.from_float(float(point[1])),
                    Fraction.from_float(float(point[2])),
                )
            )
    points: tuple[_Point3, ...] = tuple(points_list)
    point_by_vertex = dict(zip(vertex_ids, points, strict=True))
    planes: list[_Plane] = []
    edge_ids: set[tuple[int, int]] = set()

    def vector(first: _Point3, second: _Point3, /) -> _Point3:
        return (
            second[0] - first[0],
            second[1] - first[1],
            second[2] - first[2],
        )

    def cross(first: _Point3, second: _Point3, /) -> _Point3:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    def dot(first: _Point3, second: _Point3, /) -> Fraction:
        return sum(
            (first[axis] * second[axis] for axis in range(3)),
            Fraction(0),
        )

    for face in geometry.shell_faces[shell]:
        patch = model.patches[face]
        if not isinstance(patch, PlanePatch):
            raise ValueError(
                f"Convex STEP shell repair requires plane faces; got face {face}."
            )
        loops = geometry.face_loops[face]
        if len(loops) != 1 or len(loops[0]) < 3:
            raise ValueError(
                f"Convex STEP shell repair requires one simple face loop; got face {face}."
            )
        polygon: list[int] = []
        for coedge in loops[0]:
            edge = geometry.coedge_edges[coedge]
            curve_index = geometry.edge_curves[edge]
            if curve_index < 0 or not isinstance(geometry.curves[curve_index], LineCurve):
                raise ValueError(
                    f"Convex STEP shell repair requires linear edges; got edge {edge}."
                )
            first_vertex, second_vertex = geometry.edge_vertices[edge]
            start, end = (
                (first_vertex, second_vertex)
                if geometry.coedge_senses[coedge] > 0
                else (second_vertex, first_vertex)
            )
            polygon.append(start)
            edge_ids.add(
                (min(first_vertex, second_vertex), max(first_vertex, second_vertex))
            )
        origin_values = np.asarray(patch.origin)
        origin: _Point3 = (
            Fraction.from_float(float(origin_values[0])),
            Fraction.from_float(float(origin_values[1])),
            Fraction.from_float(float(origin_values[2])),
        )
        first_axis_values = np.asarray(patch.first_axis)
        first_axis: _Point3 = (
            Fraction.from_float(float(first_axis_values[0])),
            Fraction.from_float(float(first_axis_values[1])),
            Fraction.from_float(float(first_axis_values[2])),
        )
        second_axis_values = np.asarray(patch.second_axis)
        second_axis: _Point3 = (
            Fraction.from_float(float(second_axis_values[0])),
            Fraction.from_float(float(second_axis_values[1])),
            Fraction.from_float(float(second_axis_values[2])),
        )
        normal = cross(first_axis, second_axis)
        if not any(normal):
            raise ValueError(f"Plane face {face} has a degenerate normal.")
        face_points = tuple(point_by_vertex[vertex] for vertex in polygon)
        if any(dot(vector(origin, point), normal) != 0 for point in face_points):
            raise ValueError(
                f"Plane face {face} does not contain its exact source loop vertices."
            )
        turns = []
        for index in range(len(face_points)):
            first = vector(
                face_points[index], face_points[(index + 1) % len(face_points)]
            )
            second = vector(
                face_points[(index + 1) % len(face_points)],
                face_points[(index + 2) % len(face_points)],
            )
            turn = dot(cross(first, second), normal)
            if turn:
                turns.append(1 if turn > 0 else -1)
        if not turns or len(set(turns)) != 1:
            raise ValueError(f"Plane face {face} is not a convex nondegenerate polygon.")
        sides = {
            1 if side > 0 else -1
            for point in points
            if (side := dot(vector(origin, point), normal)) != 0
        }
        if len(sides) != 1:
            raise ValueError(
                f"Plane face {face} is not a strict convex-shell supporting plane."
            )
        planes.append((origin, normal, next(iter(sides))))
    edges = tuple(
        (point_by_vertex[first], point_by_vertex[second])
        for first, second in sorted(edge_ids)
    )
    return points, tuple(planes), edges


def _strictly_inside(points: tuple[_Point3, ...], planes: tuple[_Plane, ...], /) -> bool:
    for origin, normal, interior_sign in planes:
        for point in points:
            side = sum(
                ((point[axis] - origin[axis]) * normal[axis] for axis in range(3)),
                Fraction(0),
            )
            if side == 0 or (1 if side > 0 else -1) != interior_sign:
                return False
    return True


def _shells_separated(first: _ConvexShell, second: _ConvexShell, /) -> bool:
    first_points, first_planes, first_edges = first
    second_points, second_planes, second_edges = second
    axes = [plane[1] for plane in (*first_planes, *second_planes)]
    for first_edge in first_edges:
        first_direction = tuple(
            first_edge[1][axis] - first_edge[0][axis] for axis in range(3)
        )
        for second_edge in second_edges:
            second_direction = tuple(
                second_edge[1][axis] - second_edge[0][axis] for axis in range(3)
            )
            axis = (
                first_direction[1] * second_direction[2]
                - first_direction[2] * second_direction[1],
                first_direction[2] * second_direction[0]
                - first_direction[0] * second_direction[2],
                first_direction[0] * second_direction[1]
                - first_direction[1] * second_direction[0],
            )
            if any(axis):
                axes.append(axis)
    for axis in axes:
        first_projection = tuple(
            sum(
                (point[index] * axis[index] for index in range(3)),
                Fraction(0),
            )
            for point in first_points
        )
        second_projection = tuple(
            sum(
                (point[index] * axis[index] for index in range(3)),
                Fraction(0),
            )
            for point in second_points
        )
        if max(first_projection) < min(second_projection) or max(second_projection) < min(
            first_projection
        ):
            return True
    return False


def _shell_nesting_depth(
    model: BRepModel,
    geometry: BRepGeometry,
    shell: int,
    shells: tuple[int, ...],
    /,
) -> int:
    polyhedra = {item: _shell_polyhedron(model, geometry, item) for item in shells}
    source_points = polyhedra[shell][0]
    depth = 0
    for other in shells:
        if other == shell:
            continue
        other_points, other_planes, _ = polyhedra[other]
        source_inside = _strictly_inside(source_points, other_planes)
        other_inside = _strictly_inside(other_points, polyhedra[shell][1])
        if source_inside and other_inside:
            raise ValueError("Two source shells have identical strict containment.")
        if source_inside:
            depth += 1
        elif not other_inside and not _shells_separated(
            polyhedra[shell], polyhedra[other]
        ):
            raise ValueError(
                f"Shells {shell} and {other} overlap or touch without strict nesting."
            )
    return depth


def _shell_reversal_record(
    policy: StepRepairPolicy, label: str, volume_sign: int, depth: int, /
) -> AdapterLoss | None:
    if volume_sign not in (-1, 1):
        raise ValueError(f"Shell {label} has no certified nonzero volume sign.")
    expected_positive = depth % 2 == 0
    if (volume_sign > 0) == expected_positive:
        return None
    if not policy.reverse_inverted_shells:
        raise refuse(
            "inverted-shell",
            label,
            f"Certified signed volume sign {volume_sign} disagrees with "
            f"nesting depth {depth} ({'outer' if expected_positive else 'cavity'}); "
            "enable StepRepairPolicy(reverse_inverted_shells=True).",
        )
    return repair_record(
        label,
        f"Reversed shell: certified signed volume sign {volume_sign} "
        f"has the wrong sign at certified nesting depth {depth}; carriers and tolerance unchanged.",
    )


def repair_shells(
    model: BRepModel,
    policy: StepRepairPolicy,
    labels: tuple[str, ...],
    /,
) -> tuple[BRepModel, tuple[AdapterLoss, ...]]:
    """Reverse only shells whose certified sign disagrees with strict nesting."""
    geometry = model.geometry
    if not geometry.solid_shells:
        return model, ()
    volume_signs = tuple(
        brep_shell_volume_sign(model, shell) for shell in range(len(geometry.shell_faces))
    )
    signs = list(geometry.shell_orientations)
    repairs: list[AdapterLoss] = []
    for shells in geometry.solid_shells:
        shell_group = tuple(shells)
        for shell in shell_group:
            depth = _shell_nesting_depth(model, geometry, shell, shell_group)
            record = _shell_reversal_record(
                policy, labels[shell], volume_signs[shell], depth
            )
            if record is not None:
                signs[shell] = tuple(-sign for sign in signs[shell])
                repairs.append(record)
    if not repairs:
        return model, ()
    tessellation = None
    if model.tessellation is not None:
        tessellation = BRepTessellationPolicy(
            linear_deflection=model.tessellation.linear_deflection,
            angular_deflection=model.tessellation.angular_deflection,
        )
    return (
        _assemble(
            model,
            replace(geometry, shell_orientations=tuple(signs)),
            tessellation=tessellation,
        ),
        tuple(repairs),
    )


def audit_shells(
    model: BRepModel,
    policy: StepRepairPolicy,
    labels: tuple[str, ...],
    /,
) -> tuple[BRepModel, tuple[AdapterLoss, ...], tuple[str, ...]]:
    """Audit source shell signs; unresolved unsupported evidence stays explicit."""
    try:
        repaired, records = repair_shells(model, policy, labels)
        return repaired, records, ()
    except CadInterchangeError as error:
        if error.refusal.reason == "inverted-shell" or policy.reverse_inverted_shells:
            raise
        reason = str(error)
    except (NotImplementedError, ValueError) as error:
        if policy.reverse_inverted_shells:
            raise refuse("unsupported-entity", "shells", str(error)) from error
        reason = str(error)
    return (
        model,
        (),
        tuple(f"orientation_unresolved {label}: {reason}" for label in labels),
    )


def _cylinder_ring(
    stage: CadStage,
    patch: CylinderPatch,
    edge: int,
    label: str,
    axis: tuple[Fraction, ...],
    origin: tuple[Fraction, ...],
    component: int,
    /,
) -> tuple[float, int, int]:
    """Prove one full cylinder boundary circle and its authored seam endpoint."""
    curve = stage.curves[stage.edge_curves[edge]]
    if (
        not isinstance(curve, CircleCurve)
        or stage.edge_ranges[edge] != (0.0, 2 * np.pi)
        or stage.edge_vertices[edge][0] != stage.edge_vertices[edge][1]
        or not np.array_equal(curve.first_axis, patch.first_axis)
        or not np.array_equal(curve.second_axis, patch.second_axis)
        or float(curve.radius) != float(patch.radius)
    ):
        raise refuse(
            "unsupported-entity",
            label,
            "Cylinder seam bounds must share the native circle frame and radius.",
        )
    center = tuple(Fraction(float(x)) for x in np.asarray(curve.center))
    height = (center[component] - origin[component]) / axis[component]
    vertex = stage.edge_vertices[edge][0]
    expected = tuple(
        center[i] + Fraction(float(patch.radius)) * Fraction(float(patch.first_axis[i]))
        for i in range(3)
    )
    if (
        any(center[i] != origin[i] + height * axis[i] for i in range(3))
        or Fraction(float(height)) != height
        or any(
            Fraction(float(stage.vertices[vertex][i])) != expected[i] for i in range(3)
        )
    ):
        raise refuse(
            "unsupported-entity",
            label,
            "Cylinder seam endpoints lack rational axial incidence; geometry-moving repair refuses.",
        )
    return float(height), edge, vertex


def add_cylinder_seam(
    stage: CadStage,
    patch: CylinderPatch,
    loops: list[list[StagedCoedge]],
    label: str,
    /,
) -> list[StagedCoedge]:
    """Join two authored full axial circle bounds with one longitudinal seam."""
    if len(loops) != 2 or any(len(loop) != 1 for loop in loops):
        raise refuse(
            "unsupported-entity",
            label,
            "Cylinder seam repair requires exactly two full circle bounds.",
        )
    rings: list[tuple[float, int, int]] = []
    axis = tuple(Fraction(float(x)) for x in np.asarray(patch.axis))
    origin = tuple(Fraction(float(x)) for x in np.asarray(patch.origin))
    component = next((i for i, value in enumerate(axis) if value), None)
    if component is None:
        raise refuse("unsupported-entity", label, "Cylinder axis is degenerate.")
    for loop in loops:
        rings.append(
            _cylinder_ring(stage, patch, loop[0].edge, label, axis, origin, component)
        )
    rings.sort()
    (lower, bottom, start), (upper, top, end) = rings
    if not lower < upper:
        raise refuse(
            "unsupported-entity", label, "Cylinder seam requires distinct axial bounds."
        )
    seam_curve = SurfaceIsoparametricCurve(patch, 0, 0.0, parameter_range=(lower, upper))
    seam = stage.edge(seam_curve, start, end, label, parameter_range=(lower, upper))
    source = LineCurve((0.0, 0.0), (0.0, 1.0))
    for edge in (bottom, top):
        stage.native_phase(edge, patch, 0)
    return [
        StagedCoedge(bottom, 1, LineCurve((0.0, lower), (1.0, 0.0)), label),
        StagedCoedge(seam, 1, PeriodicPCurve(source, patch, (1, 0)), label),
        StagedCoedge(top, -1, LineCurve((0.0, upper), (1.0, 0.0)), label),
        StagedCoedge(seam, -1, PeriodicPCurve(source, patch, (0, 0)), label),
    ]
