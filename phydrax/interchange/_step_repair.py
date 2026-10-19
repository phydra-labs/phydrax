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

from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._validation import positive_integer
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ..geometry.brep._constructors import assemble_brep_model, BRepTessellationPolicy
from ..geometry.brep._containment import certify_solid_containment
from ..geometry.brep._intersection_curve import PeriodicPCurve
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


@final
class StepRepairPolicy(StrictModule):
    """Permit only named geometry-preserving repairs, with explicit work limits.

    All switches default to false. Face-sense repair supports closed convex
    linear polyhedra with rational supporting-plane evidence. Seam insertion
    supports coaxial cylinder bands and native sphere/torus natural charts.
    Other families refuse explicitly.
    No switch permits moving geometry, fitting a repair, or growing tolerance.
    """

    add_missing_seams: bool = eqx.field(static=True)
    reverse_inverted_shells: bool = eqx.field(static=True)
    orient_face_senses: bool = eqx.field(static=True)
    subdivisions: int = eqx.field(static=True)
    maximum_boxes: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    audit_maximum_work_units: int = eqx.field(static=True)
    audit_maximum_memory_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        add_missing_seams: bool = False,
        reverse_inverted_shells: bool = False,
        orient_face_senses: bool = False,
        subdivisions: int = 1,
        maximum_boxes: int = 4096,
        maximum_depth: int = 32,
        audit_maximum_work_units: int = 2_000_000,
        audit_maximum_memory_bytes: int = 16 << 20,
    ) -> None:
        for value in (add_missing_seams, reverse_inverted_shells, orient_face_senses):
            if not isinstance(value, bool):
                raise TypeError(
                    f"StepRepairPolicy.switches must be bool values; got {value!r}"
                )
        subdivisions = positive_integer(subdivisions, "StepRepairPolicy.subdivisions")
        maximum_boxes = positive_integer(maximum_boxes, "StepRepairPolicy.maximum_boxes")
        maximum_depth = positive_integer(maximum_depth, "StepRepairPolicy.maximum_depth")
        audit_maximum_work_units = positive_integer(
            audit_maximum_work_units, "StepRepairPolicy.audit_maximum_work_units"
        )
        audit_maximum_memory_bytes = positive_integer(
            audit_maximum_memory_bytes, "StepRepairPolicy.audit_maximum_memory_bytes"
        )
        self.add_missing_seams = add_missing_seams
        self.reverse_inverted_shells = reverse_inverted_shells
        self.orient_face_senses = orient_face_senses
        self.subdivisions = subdivisions
        self.maximum_boxes = maximum_boxes
        self.maximum_depth = maximum_depth
        self.audit_maximum_work_units = audit_maximum_work_units
        self.audit_maximum_memory_bytes = audit_maximum_memory_bytes


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
    budget: CoordinateEnclosureBudget,
    /,
) -> int:
    """Prove outward coorientation of a convex linear shell, using rational dots."""
    budget.reserve(32 * len(points) + 64, 1024 * (len(points) + 1))
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
    budget: CoordinateEnclosureBudget,
    realize: bool = False,
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
        curve_surface_tolerance=model.report.curve_surface_tolerance,
        budget=budget,
        tessellation=BRepTessellationPolicy(
            linear_deflection=model.report.linear_deflection,
            angular_deflection=model.report.angular_deflection,
            trim_samples_per_edge=model.report.trim_samples_per_edge,
            realize=realize,
        ),
    )


def _shell_nesting_point(geometry: BRepGeometry, shell: int, /) -> np.ndarray:
    vertices = sorted(
        {
            vertex
            for face in geometry.shell_faces[shell]
            for loop in geometry.face_loops[face]
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        }
    )
    if not vertices:
        raise ValueError(
            f"BRepGeometry.shell_faces must contain authored nesting vertices; got shell={shell}, vertices={vertices!r}"
        )
    return np.asarray(geometry.vertex_points[vertices[0]])


def _shell_nesting_depth(
    model: BRepModel,
    geometry: BRepGeometry,
    shell: int,
    shells: tuple[int, ...],
    policy: StepRepairPolicy,
    budget: CoordinateEnclosureBudget,
    /,
) -> int:
    point = _shell_nesting_point(geometry, shell)
    depth = 0
    for other in shells:
        if other == shell:
            continue
        evidence = certify_solid_containment(
            model,
            point,
            other,
            maximum_boxes=policy.maximum_boxes,
            maximum_depth=policy.maximum_depth,
            budget=budget,
        )
        if not evidence.complete:
            raise ValueError(
                f"BRepGeometry.shell_faces must admit certified strict nesting; got shell={shell}, other={other}, inside={evidence.inside}, unresolved_faces={evidence.unresolved_faces}"
            )
        depth += evidence.inside
    return depth


def _shell_reversal_record(
    policy: StepRepairPolicy, label: str, enclosure: np.ndarray, depth: int, /
) -> AdapterLoss | None:
    lower, upper = enclosure
    if lower <= 0 <= upper:
        raise ValueError(
            f"BRepGeometry.shell_orientations must admit nonzero signed volume; got shell={label}, enclosure={[lower, upper]!r}"
        )
    expected_positive = depth % 2 == 0
    if (lower > 0) == expected_positive:
        return None
    if not policy.reverse_inverted_shells:
        raise refuse(
            "inverted-shell",
            label,
            f"Certified signed volume [{lower!r}, {upper!r}] disagrees with "
            f"nesting depth {depth} ({'outer' if expected_positive else 'cavity'}); "
            "enable StepRepairPolicy(reverse_inverted_shells=True) with coordinate_budget.",
        )
    return repair_record(
        label,
        f"Reversed shell: certified signed volume [{lower!r}, {upper!r}] "
        f"has the wrong sign at certified nesting depth {depth}; carriers and tolerance unchanged.",
    )


def repair_shells(
    model: BRepModel,
    policy: StepRepairPolicy,
    labels: tuple[str, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[BRepModel, tuple[AdapterLoss, ...]]:
    """Reverse only shells whose certified sign disagrees with certified nesting."""
    from ..geometry.brep._measure import brep_signed_shell_volume_enclosures

    geometry = model.require_geometry()
    if not geometry.solid_shells:
        return model, ()
    volumes = np.asarray(
        brep_signed_shell_volume_enclosures(
            model,
            geometry,
            subdivisions=policy.subdivisions,
            coordinate_budget=budget,
        ).volume_enclosures
    )
    shell_model = model
    if any(len(shells) > 1 for shells in geometry.solid_shells):
        shell_model = _assemble(
            model,
            replace(
                geometry,
                solid_shells=tuple(
                    (shell,) for shell in range(len(geometry.shell_faces))
                ),
                occurrences=None,
                assembly_containers=None,
                budget=budget,
            ),
            budget=budget,
        )
    signs = list(geometry.shell_orientations)
    repairs: list[AdapterLoss] = []
    for shells in geometry.solid_shells:
        for shell in shells:
            depth = _shell_nesting_depth(
                shell_model, geometry, shell, shells, policy, budget
            )
            record = _shell_reversal_record(policy, labels[shell], volumes[shell], depth)
            if record is not None:
                signs[shell] = tuple(-sign for sign in signs[shell])
                repairs.append(record)
    if not repairs:
        return model, ()
    return _assemble(
        model,
        replace(geometry, shell_orientations=tuple(signs), budget=budget),
        budget=budget,
        realize=model.mesh_faces.shape[0] > 0,
    ), tuple(repairs)


def audit_shells(
    model: BRepModel,
    policy: StepRepairPolicy,
    labels: tuple[str, ...],
    budget: CoordinateEnclosureBudget | None,
    /,
) -> tuple[BRepModel, tuple[AdapterLoss, ...], tuple[str, ...]]:
    """Bound default auditing; unresolved evidence never becomes a sign claim."""
    if policy.reverse_inverted_shells:
        if budget is None:
            raise refuse("limit", "DATA", "Shell repairs require coordinate_budget.")
        try:
            repaired, records = repair_shells(model, policy, labels, budget)
        except CadInterchangeError:
            raise
        except CoordinateEnclosureResourceError as error:
            raise refuse("limit", "shells", str(error)) from error
        except ValueError as error:
            raise refuse("unsupported-entity", "shells", str(error)) from error
        return repaired, records, ()
    try:
        with HostBatchTicket(
            policy.audit_maximum_work_units, bytes_upper=policy.audit_maximum_memory_bytes
        ) as ticket:
            audit_budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=policy.audit_maximum_work_units,
                maximum_memory_bytes=policy.audit_maximum_memory_bytes,
            )
            with audit_budget.activate():
                repaired, records = repair_shells(model, policy, labels, audit_budget)
        return repaired, records, ()
    except CadInterchangeError as error:
        if error.refusal.reason == "inverted-shell":
            raise
        reason = str(error)
    except ValueError as error:
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
    budget: CoordinateEnclosureBudget,
    /,
) -> list[StagedCoedge]:
    """Join two authored full axial circle bounds with one longitudinal seam."""
    budget.reserve(512, 8192)
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
