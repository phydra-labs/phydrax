#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native circular plane/quadric blends, with analytic meridian certificates.

Setbacks are ordered (plane distance, curved-face meridian arclength). Admission
requires a full native turn, a single rectangular quadric face and circular
plane loops. No sampled geometry or closeness threshold determines admission.
The retained plane and quadric carriers survive unchanged; only their trims
and the meridian seam are rebuilt. Binary execution coefficients are accompanied
by outward tangency bounds, rather than being called unrounded mathematical roots.
"""

from __future__ import annotations

from fractions import Fraction
from math import atan2, pi
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._interval_enclosure import prepare_interval_function
from ..design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from ._constructors import BRepTessellationPolicy, Builder, finish
from ._coordinate_request import coordinate_source_request
from ._fillet import BRepBlendUnsupported
from ._intersection import NativePeriodEndpoint
from ._intersection_curve import affine_curve_coefficients, PeriodicPCurve
from ._linked_parameters import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
)
from ._model import BRepGeometry, BRepModel, BRepOccurrence, BRepPCurve
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    AbstractCurve,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    exact_frame_orientation,
    LineCurve,
    PlanePatch,
    SpherePatch,
    surface_evaluate_host,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ._request_preparations import TrimRequestPreparations


@final
class ClosedSeamTangencyCertificate(StrictModule):
    """Whole-circle normal mismatch bounds for the represented torus coefficients.

    The meridian construction solves signed-distance equations analytically.
    ``normal_difference_bounds`` encloses each component of the represented
    boundary normal difference in the orthonormal (radial, axial) frame. It is
    not a tolerance, acceptance threshold, or sampled tangency observation.
    """

    normal_difference_bounds: Array
    radius: float = eqx.field(static=True)
    plane_face: int = eqx.field(static=True)
    curved_face: int = eqx.field(static=True)
    plane_parameter: float = eqx.field(static=True)
    curved_parameter: float = eqx.field(static=True)


@final
class _Seam(StrictModule):
    plane: PlanePatch
    curved: CylinderPatch | ConePatch | SpherePatch
    plane_face: int = eqx.field(static=True)
    curved_face: int = eqx.field(static=True)
    plane_coedge: int = eqx.field(static=True)
    curved_coedge: int = eqx.field(static=True)
    seam_edge: int = eqx.field(static=True)
    edge: int = eqx.field(static=True)
    z: float = eqx.field(static=True)
    radial: float = eqx.field(static=True)
    parameter: float = eqx.field(static=True)
    radial_sense: int = eqx.field(static=True)
    axial_sense: int = eqx.field(static=True)


@final
class _Meridian(StrictModule):
    patch: TorusPatch | ConePatch | CylinderPatch
    plane_radius: float = eqx.field(static=True)
    curved_radius: float = eqx.field(static=True)
    curved_z: float = eqx.field(static=True)
    curved_parameter: float = eqx.field(static=True)
    plane_parameter: float = eqx.field(static=True)
    blend_parameter: float = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    contact_bounds: tuple[float, float] = eqx.field(static=True)


def _isoline(pcurve: BRepPCurve) -> tuple[Fraction, Fraction] | None:
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, AbstractCurve):
        return None
    coefficients = affine_curve_coefficients(source, np.zeros(2, dtype=np.float64))
    if coefficients is None or coefficients[1] != (Fraction(1), Fraction(0)):
        return None
    return coefficients[0][0], coefficients[0][1]


def _require_shared_frame(
    plane: PlanePatch, curved: CylinderPatch | ConePatch | SpherePatch
) -> None:
    if (
        exact_frame_orientation(
            np.asarray(curved.first_axis),
            np.asarray(curved.second_axis),
            np.asarray(curved.axis),
        )
        != 1
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: the revolution frame is not exactly right-handed orthonormal"
        )
    if not (
        np.array_equal(plane.first_axis, curved.first_axis)
        and np.array_equal(plane.second_axis, curved.second_axis)
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: plane chart must share the certified circular frame"
        )


def _require_native_turn(geometry: BRepGeometry, edge: int) -> None:
    first, last = geometry.edge_endpoint_roots[edge]
    if not (
        isinstance(first, NativePeriodEndpoint)
        and isinstance(last, NativePeriodEndpoint)
        and first.rational == last.rational == 0
        and last.turns - first.turns == 1
        and first.turns == 0
        and geometry.edge_vertices[edge][0] == geometry.edge_vertices[edge][1]
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: selected edge is not one authored circular native turn"
        )
    carrier_index = geometry.edge_curves[edge]
    if carrier_index < 0 or not isinstance(
        geometry.curves[carrier_index], (CircleCurve, SurfaceIsoparametricCurve)
    ):
        raise BRepBlendUnsupported("closed seam blend: selected carrier is not circular")


def _require_rectangular_chart(
    geometry: BRepGeometry, plane_face: int, curved_face: int, curved_coedge: int
) -> tuple[float, int]:
    offset = _isoline(geometry.pcurves[curved_coedge])
    if offset is None or offset[0] != 0:
        raise BRepBlendUnsupported(
            "closed seam blend: curved boundary is not a native circular isoline"
        )
    loops = geometry.face_loops[curved_face]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise BRepBlendUnsupported(
            "closed seam blend: several-edge curved trims are unsupported"
        )
    counts: dict[int, int] = {}
    for coedge in loops[0]:
        item = geometry.coedge_edges[coedge]
        counts[item] = counts.get(item, 0) + 1
    seams = [item for item, count in counts.items() if count == 2]
    if len(seams) != 1:
        raise BRepBlendUnsupported(
            "closed seam blend: curved face must have one meridian seam"
        )
    if any(len(loop) != 1 for loop in geometry.face_loops[plane_face]):
        raise BRepBlendUnsupported(
            "closed seam blend: noncircular or several-edge plane loops are unsupported"
        )
    return float(offset[1]), seams[0]


def _seam(model: BRepModel, edge: int, budget: CoordinateEnclosureBudget) -> _Seam:
    geometry = model.require_geometry()
    budget.reserve(256, 0)
    if not 0 <= edge < len(geometry.edge_curves):
        raise ValueError(
            f"closed_seam_blend.edge_index must reference the source topology; got {edge}"
        )
    if len(geometry.solid_shells) != 1 or geometry.occurrences != (
        BRepOccurrence(("solid0",), 0),
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: multiple solids or placed occurrences require materialization"
        )
    if any(root is not None for root in geometry.vertex_roots):
        raise BRepBlendUnsupported(
            "closed seam blend: root-authored source vertices are unsupported"
        )
    uses = [
        (face, coedge)
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
        if geometry.coedge_edges[coedge] == edge
    ]
    if len(uses) != 2 or uses[0][0] == uses[1][0]:
        raise BRepBlendUnsupported(
            "closed seam blend: exactly two distinct incident faces are required"
        )
    plane_uses = [
        (face, coedge)
        for face, coedge in uses
        if isinstance(model.patches[face], PlanePatch)
    ]
    if len(plane_uses) != 1:
        raise BRepBlendUnsupported("closed seam blend: one analytic plane is required")
    pf, pc = plane_uses[0]
    cf, cc = next(use for use in uses if use != plane_uses[0])
    plane, curved = model.patches[pf], model.patches[cf]
    if not isinstance(plane, PlanePatch) or not isinstance(
        curved, (CylinderPatch, ConePatch, SpherePatch)
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: curved face is not a cylinder, cone or sphere"
        )
    _require_shared_frame(plane, curved)
    _require_native_turn(geometry, edge)
    parameter, seam_edge = _require_rectangular_chart(geometry, pf, cf, cc)
    center = curved.center if isinstance(curved, SpherePatch) else curved.origin
    relative = np.asarray(plane.origin) - np.asarray(center)
    axis = np.asarray(curved.axis)
    z = float(relative @ axis)
    if not np.array_equal(relative, z * axis):
        raise BRepBlendUnsupported(
            "closed seam blend: plane origin is not on the certified revolution axis"
        )
    point = surface_evaluate_host(curved, np.asarray((0.0, parameter)))
    radial = float((point - np.asarray(center)) @ np.asarray(curved.first_axis))
    return _Seam(
        plane,
        curved,
        pf,
        cf,
        pc,
        cc,
        seam_edge,
        edge,
        z,
        radial,
        parameter,
        int(model.orientation[cf]),
        int(model.orientation[pf]),
    )


def _fillet_values(seam: _Seam, radius: Array) -> Array:
    """(center radial, center axial, contact radial, axial, chart parameter)."""
    q, d, z = seam.radial_sense, seam.axial_sense, seam.z
    zc = z - d * radius
    match seam.curved:
        case CylinderPatch():
            rc = seam.curved.radius - q * radius
            return jnp.stack((rc, zc, seam.curved.radius, zc, zc))
        case ConePatch():
            slope = jnp.tan(seam.curved.semi_angle)
            length = jnp.sqrt(1 + slope * slope)
            rc = seam.curved.reference_radius + slope * zc - q * radius * length
            zt = zc - q * radius * slope / length
            rt = rc + q * radius / length
            return jnp.stack((rc, zc, rt, zt, zt))
        case SpherePatch():
            inner = seam.curved.radius - q * radius
            rc = jnp.sqrt(inner * inner - zc * zc)
            rt = seam.curved.radius * rc / inner
            zt = seam.curved.radius * zc / inner
            # _fillet certifies rt > 0, so latitude stays on the principal
            # arctangent branch without requiring an interval atan2 extension.
            parameter = jnp.arctan(zt / rt)
            return jnp.stack((rc, zc, rt, zt, parameter))


def _fillet(seam: _Seam, radius: float, budget: CoordinateEnclosureBudget) -> _Meridian:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            f"closed_seam_blend.radius must be finite and positive; got {radius}"
        )
    budget.reserve(128, 0)

    def values(point: Array) -> Array:
        return _fillet_values(seam, point[0])

    prepared = prepare_interval_function(values, 1, batch_capacity=1)
    input_ = np.asarray(((radius,),), dtype=np.float64)
    low, high = prepared.evaluate(input_, input_)
    if (
        not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or low[0, 0] <= 0
        or low[0, 2] <= 0
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: rolling-ball center is uncertified"
        )
    rc, zc, rt, zt, parameter = (
        float(value) for value in prepared.evaluate_host_points(input_)[0]
    )
    center = (
        seam.curved.center if isinstance(seam.curved, SpherePatch) else seam.curved.origin
    )
    patch = TorusPatch(
        np.asarray(center) + zc * np.asarray(seam.curved.axis),
        seam.curved.first_axis,
        seam.curved.second_axis,
        seam.curved.axis,
        rc,
        radius,
    )
    plane_angle = seam.axial_sense * pi / 2
    curved_angle = atan2(zt - zc, rt - rc)
    while curved_angle - plane_angle > pi:
        curved_angle -= 2 * pi
    while plane_angle - curved_angle > pi:
        curved_angle += 2 * pi
    # Preserve the chosen short arc while placing its start in the native
    # nonnegative gauge. The chart window may cross the periodic seam.
    phase_shift = 2 * pi if min(plane_angle, curved_angle) < 0 else 0.0
    plane_angle += phase_shift
    curved_angle += phase_shift

    def radial(point: Array) -> Array:
        return patch.major_radius + patch.minor_radius * jnp.cos(point[0])

    radial_plan = prepare_interval_function(radial, 1, batch_capacity=1)
    radial_low, _ = radial_plan.evaluate(
        ((min(plane_angle, curved_angle),),), ((max(plane_angle, curved_angle),),)
    )
    if radial_low[0] <= 0:
        raise BRepBlendUnsupported(
            "closed seam blend: rolling-ball arc reaches the revolution axis"
        )
    return _Meridian(
        patch,
        rc,
        rt,
        zt,
        parameter,
        plane_angle,
        curved_angle,
        min(plane_angle, curved_angle),
        max(plane_angle, curved_angle),
        (float(low[0, 4]), float(high[0, 4])),
    )


def _chamfer(
    seam: _Seam, setbacks: tuple[float, float], budget: CoordinateEnclosureBudget
) -> _Meridian:
    a, b = setbacks
    if any(not np.isfinite(value) or value <= 0 for value in setbacks):
        raise ValueError(
            f"closed_seam_blend.setbacks must be finite and positive; got {setbacks}"
        )
    budget.reserve(128, 0)
    q, d = seam.radial_sense, seam.axial_sense

    def contacts(point: Array) -> Array:
        rp = seam.radial - q * point[0]
        match seam.curved:
            case CylinderPatch():
                parameter = zt = seam.z - d * point[1]
                rt = seam.curved.radius
            case ConePatch():
                slope = jnp.tan(seam.curved.semi_angle)
                parameter = zt = seam.z - d * point[1] / jnp.sqrt(1 + slope * slope)
                rt = seam.curved.reference_radius + slope * zt
            case SpherePatch():
                parameter = seam.parameter - d * point[1] / seam.curved.radius
                zt = seam.curved.radius * jnp.sin(parameter)
                rt = seam.curved.radius * jnp.cos(parameter)
        return jnp.stack((rp, rt, zt, parameter))

    plan = prepare_interval_function(contacts, 2, batch_capacity=1)
    inputs = np.asarray(((a, b),), dtype=np.float64)
    low, high = plan.evaluate(inputs, inputs)
    if (
        not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or min(low[0, :2]) <= 0
        or low[0, 2] <= seam.z <= high[0, 2]
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: chamfer axis avoidance or axial separation is uncertified"
        )
    rp, rt, zt, parameter = (
        float(value) for value in plan.evaluate_host_points(inputs)[0]
    )
    center = (
        seam.curved.center if isinstance(seam.curved, SpherePatch) else seam.curved.origin
    )
    origin = np.asarray(center) + seam.z * np.asarray(seam.curved.axis)
    slope = (rt - rp) / (zt - seam.z)
    if not (high[0, 0] < low[0, 1] or high[0, 1] < low[0, 0]):
        raise BRepBlendUnsupported(
            "closed seam blend: chamfer generatrix slope sign is uncertified"
        )
    patch = ConePatch(
        origin,
        seam.curved.first_axis,
        seam.curved.second_axis,
        seam.curved.axis,
        rp,
        float(np.arctan(slope)),
    )
    last = zt - seam.z
    return _Meridian(
        patch,
        rp,
        rt,
        zt,
        parameter,
        0.0,
        last,
        min(0.0, last),
        max(0.0, last),
        (float(low[0, 3]), float(high[0, 3])),
    )


def _copy_builder(model: BRepModel) -> Builder:
    geometry = model.require_geometry()
    builder = Builder()
    builder.vertices = [np.asarray(point) for point in geometry.vertex_points]
    builder.curves = list(geometry.curves)
    builder.edge_curves = list(geometry.edge_curves)
    builder.edge_ranges = [(float(row[0]), float(row[1])) for row in geometry.edge_ranges]
    builder.edge_vertices = list(geometry.edge_vertices)
    for roots in geometry.edge_endpoint_roots:
        first, last = roots
        if (first is not None and not isinstance(first, NativePeriodEndpoint)) or (
            last is not None and not isinstance(last, NativePeriodEndpoint)
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: nonnative endpoint roots are unsupported"
            )
        builder.edge_endpoint_roots.append((first, last))
    builder.pcurves = list(geometry.pcurves)
    builder.coedge_edges = list(geometry.coedge_edges)
    builder.coedge_senses = list(geometry.coedge_senses)
    builder.coedge_endpoint_roots = list(geometry.coedge_endpoint_roots)
    for face, patch in enumerate(model.patches):
        builder.face(
            patch,
            np.asarray(model.parameter_bounds[face]),
            [list(loop) for loop in geometry.face_loops[face]],
            model.physical_tags[face],
            None,
            orientation=int(model.orientation[face]),
        )
    return builder


def _circle(seam: _Seam, radius: float, z: float) -> CircleCurve:
    center = (
        seam.curved.center if isinstance(seam.curved, SpherePatch) else seam.curved.origin
    )
    return CircleCurve(
        np.asarray(center) + z * np.asarray(seam.curved.axis),
        seam.curved.first_axis,
        seam.curved.second_axis,
        radius,
    )


def _require_clearance(model: BRepModel, seam: _Seam, meridian: _Meridian) -> None:
    box = np.asarray(model.parameter_bounds[seam.curved_face])
    low, high = box[:, 1]
    contact_low, contact_high = meridian.contact_bounds
    if not low < contact_low <= contact_high < high:
        raise BRepBlendUnsupported(
            "closed seam blend: blend reaches another boundary or a pole"
        )
    if seam.parameter not in (low, high) or box[0, 0] != 0:
        raise BRepBlendUnsupported(
            "closed seam blend: selected circle is not an endpoint of the native rectangular chart"
        )
    geometry = model.require_geometry()
    for loop in geometry.face_loops[seam.plane_face]:
        coedge = loop[0]
        if coedge == seam.plane_coedge:
            continue
        pcurve = geometry.pcurves[coedge]
        if not isinstance(pcurve, CircleCurve) or not np.array_equal(
            pcurve.center, np.zeros(2)
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: nonconcentric plane loops are unsupported"
            )
        other = float(pcurve.radius)
        if (
            min(seam.radial, meridian.plane_radius)
            <= other
            <= max(seam.radial, meridian.plane_radius)
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: setback reaches another circular plane trim"
            )


def _stage_boundaries(
    builder: Builder,
    geometry: BRepGeometry,
    seam: _Seam,
    meridian: _Meridian,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int]:
    plane_curve = _circle(seam, meridian.plane_radius, seam.z)
    builder.curves[builder.edge_curves[seam.edge]] = plane_curve
    vertex = builder.edge_vertices[seam.edge][0]
    builder.vertices[vertex] = np.asarray(
        plane_curve.center
    ) + meridian.plane_radius * np.asarray(seam.curved.first_axis)
    builder.pcurves[seam.plane_coedge] = CircleCurve(
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), meridian.plane_radius
    )
    point = surface_evaluate_host(
        seam.curved, np.asarray((0.0, meridian.curved_parameter))
    )
    cv = builder.vertex(point)
    curved_curve = SurfaceIsoparametricCurve(seam.curved, 1, meridian.curved_parameter)
    ce = builder.edge(curved_curve, 0.0, 2 * pi, cv, cv)
    builder.coedge_edges[seam.curved_coedge] = ce
    builder.pcurves[seam.curved_coedge] = curved_curve.p_curve()
    face = builder.faces[seam.curved_face]
    box = face.box.copy()
    endpoint = 0 if seam.parameter == box[0, 1] else 1
    box[endpoint, 1] = meridian.curved_parameter
    face.box = box
    start, end = builder.edge_vertices[seam.seam_edge]
    builder.edge_vertices[seam.seam_edge] = (cv, end) if endpoint == 0 else (start, cv)
    builder.edge_ranges[seam.seam_edge] = (float(box[0, 1]), float(box[1, 1]))
    builder.edge_endpoint_roots[seam.seam_edge] = (None, None)
    builder.curves[builder.edge_curves[seam.seam_edge]] = SurfaceIsoparametricCurve(
        seam.curved, 0, 0.0
    )
    for coedge in geometry.face_loops[seam.curved_face][0]:
        if builder.coedge_edges[coedge] == seam.seam_edge:
            original = geometry.pcurves[coedge]
            shifts = (
                original.period_shifts if isinstance(original, PeriodicPCurve) else (0, 0)
            )
            builder.pcurves[coedge] = PeriodicPCurve(
                LineCurve((0.0, 0.0), (0.0, 1.0)), seam.curved, shifts
            )
            builder.coedge_endpoint_roots[coedge] = (None, None)
    for edge, patch in ((seam.edge, seam.curved), (ce, seam.curved)):
        builder.native_phase(edge, patch, 0, preparations=preparations, budget=budget)
    return seam.edge, ce


def _stage_blend(
    builder: Builder,
    seam: _Seam,
    meridian: _Meridian,
    plane_edge: int,
    curved_edge: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> int:
    patch = meridian.patch
    lower_edge, upper_edge = (
        (plane_edge, curved_edge)
        if meridian.plane_parameter < meridian.blend_parameter
        else (curved_edge, plane_edge)
    )
    start = builder.edge_vertices[lower_edge][0]
    end = builder.edge_vertices[upper_edge][0]
    curve = SurfaceIsoparametricCurve(
        patch, 0, 0.0, parameter_range=(meridian.lower, meridian.upper)
    )
    edge = builder.edge(curve, meridian.lower, meridian.upper, start, end)

    def coedge(item: int, sense: int, pcurve: BRepPCurve) -> int:
        return builder.coedge(
            item, sense, pcurve, preparations=preparations, budget=budget
        )

    low = coedge(lower_edge, 1, LineCurve((0.0, meridian.lower), (1.0, 0.0)))
    right = coedge(edge, 1, PeriodicPCurve(curve.p_curve(), patch, (1, 0)))
    high = coedge(upper_edge, -1, LineCurve((0.0, meridian.upper), (1.0, 0.0)))
    left = coedge(edge, -1, PeriodicPCurve(curve.p_curve(), patch, (0, 0)))
    orientation = 1 if isinstance(patch, TorusPatch) else seam.radial_sense
    return builder.face(
        patch,
        np.asarray(((0.0, meridian.lower), (2 * pi, meridian.upper))),
        [[low, right, high, left]],
        "fillet" if isinstance(patch, TorusPatch) else "chamfer",
        None,
        orientation=orientation,
    )


def closed_seam_blend(
    model: BRepModel,
    edge_index: int,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Replace one full circular plane/quadric edge and publish native topology."""
    if (radius is None) == (setbacks is None):
        raise ValueError(
            "closed_seam_blend.specification must select one radius or setback pair; got an ambiguous selection"
        )
    seam = _seam(model, edge_index, budget)
    if radius is not None:
        meridian = _fillet(seam, radius, budget)
    elif setbacks is not None:
        meridian = _chamfer(seam, setbacks, budget)
    else:
        raise ValueError(
            "closed_seam_blend.specification must select a blend; got no specification"
        )
    _require_clearance(model, seam, meridian)
    geometry = model.require_geometry()
    builder = _copy_builder(model)
    pe, ce = _stage_boundaries(builder, geometry, seam, meridian, preparations, budget)
    blend = _stage_blend(builder, seam, meridian, pe, ce, preparations, budget)
    shells = tuple(
        tuple((*faces, blend)) if seam.curved_face in faces else faces
        for faces in geometry.shell_faces
    )
    orientations = tuple(
        tuple((*row, 1)) if seam.curved_face in faces else row
        for faces, row in zip(
            geometry.shell_faces, geometry.shell_orientations, strict=True
        )
    )
    return finish(
        builder,
        solid=bool(geometry.solid_shells),
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        shells=(shells, orientations, geometry.solid_shells),
        preparations=preparations,
        budget=budget,
    )


def closed_seam_tangency_certificate(
    model: BRepModel,
    edge_index: int,
    radius: float,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> ClosedSeamTangencyCertificate:
    """Enclose analytic normals at both whole-circle fillet boundaries."""
    del preparations
    seam = _seam(model, edge_index, budget)
    meridian = _fillet(seam, radius, budget)

    def differences(angles: Array) -> Array:
        torus_plane = jnp.stack((jnp.cos(angles[0]), jnp.sin(angles[0])))
        torus_curved = jnp.stack((jnp.cos(angles[1]), jnp.sin(angles[1])))
        match seam.curved:
            case CylinderPatch():
                normal = jnp.asarray((seam.radial_sense, 0.0), dtype=jnp.float64)
            case ConePatch():
                slope = jnp.tan(seam.curved.semi_angle)
                normal = (
                    seam.radial_sense
                    * jnp.stack((jnp.ones_like(slope), -slope))
                    / jnp.sqrt(1 + slope * slope)
                )
            case SpherePatch():
                latitude = meridian.curved_parameter
                normal = seam.radial_sense * jnp.stack(
                    (jnp.cos(latitude), jnp.sin(latitude))
                )
        return jnp.stack(
            (torus_plane - jnp.asarray((0.0, seam.axial_sense)), torus_curved - normal)
        )

    budget.reserve(128, 0)
    prepared = prepare_interval_function(differences, 2, batch_capacity=1)
    angles = np.asarray(
        ((meridian.plane_parameter, meridian.blend_parameter),), dtype=np.float64
    )
    low, high = prepared.evaluate(angles, angles)
    return ClosedSeamTangencyCertificate(
        jnp.asarray(np.stack((low[0], high[0]), axis=-1)),
        radius,
        seam.plane_face,
        seam.curved_face,
        meridian.plane_parameter,
        meridian.curved_parameter,
    )


@final
class _ClosedSeamRadiusMap(AbstractBRepParameterMap):
    seam: _Seam
    radius: Array
    result: BRepModel
    angle_shift: float = eqx.field(static=True)
    endpoint: int = eqx.field(static=True)
    other_radius: float = eqx.field(static=True)
    radius_upper: float | None = eqx.field(static=True)

    def compile(
        self, context: ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.model_id != self.result.model_id:
            raise ValueError(
                "closed_seam_radius_map.model must match its certified construction; got a different model identity"
            )
        binding = context.bind(
            ParameterId(model.source_revision, "radius"),
            self.radius,
            role="length",
            unit=context.coordinate_unit,
            physical_scale=float(self.radius),
            bounds=(0.0, self.radius_upper),
            trainable=True,
        )
        return _BoundClosedSeamRadiusMap(
            self.seam,
            self.result,
            binding,
            self.angle_shift,
            self.endpoint,
            self.other_radius,
        )


@final
class _BoundClosedSeamRadiusMap(AbstractBoundBRepParameterMap):
    seam: _Seam
    model: BRepModel
    binding: ParameterBinding = eqx.field(static=True)
    angle_shift: float = eqx.field(static=True)
    endpoint: int = eqx.field(static=True)
    other_radius: float = eqx.field(static=True)

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        # This is the existing native runtime edge owner, not a fitted carrier.
        from ._differentiable import _RealizedTrimEdge

        radius = self.binding.read(state)
        rc, zc, rt, zt, parameter = _fillet_values(self.seam, radius)
        seam, model = self.seam, self.model
        geometry = model.require_geometry()
        old_patch = model.patches[-1]
        if not isinstance(old_patch, TorusPatch):
            raise ValueError(
                f"closed_seam_radius_map.patch must be a torus; got {type(old_patch).__name__}"
            )
        origin = (
            seam.curved.center
            if isinstance(seam.curved, SpherePatch)
            else seam.curved.origin
        )
        patch = eqx.tree_at(
            lambda value: (value.center, value.major_radius, value.minor_radius),
            old_patch,
            (origin + zc * seam.curved.axis, rc, radius),
        )
        patches = (*model.patches[:-1], patch)
        angle = jnp.arctan2(zt - zc, rt - rc) + self.angle_shift
        blend_loop = geometry.face_loops[-1][0]
        plane_blend_coedge = next(
            coedge for coedge in blend_loop if geometry.coedge_edges[coedge] == seam.edge
        )
        fixed_curve = geometry.pcurves[plane_blend_coedge]
        if not isinstance(fixed_curve, LineCurve):
            raise RuntimeError(
                "closed_seam_radius_map.plane_phase must retain its affine chart atom; got another trim family"
            )
        fixed = fixed_curve.origin[1]
        lower, upper = jnp.minimum(angle, fixed), jnp.maximum(angle, fixed)
        bounds = model.parameter_bounds.at[seam.curved_face, self.endpoint, 1].set(
            parameter
        )
        bounds = bounds.at[-1, 0, 1].set(lower).at[-1, 1, 1].set(upper)
        pcurves = list(geometry.pcurves)
        pc = pcurves[seam.plane_coedge]
        if not isinstance(pc, CircleCurve):
            raise ValueError(
                f"closed_seam_radius_map.plane_trim must be a circle; got {type(pc).__name__}"
            )
        pcurves[seam.plane_coedge] = eqx.tree_at(lambda value: value.radius, pc, rc)
        cc = pcurves[seam.curved_coedge]
        if not isinstance(cc, LineCurve):
            raise ValueError(
                f"closed_seam_radius_map.curved_trim must be an affine isoline; got {type(cc).__name__}"
            )
        pcurves[seam.curved_coedge] = eqx.tree_at(
            lambda value: value.origin,
            cc,
            jnp.stack((jnp.zeros_like(parameter), parameter)),
        )
        for coedge, value in ((blend_loop[0], lower), (blend_loop[2], upper)):
            pc = pcurves[coedge]
            if not isinstance(pc, LineCurve):
                raise ValueError(
                    f"closed_seam_radius_map.blend_trim must be an affine isoline; got {type(pc).__name__}"
                )
            pcurves[coedge] = eqx.tree_at(
                lambda item: item.origin,
                pc,
                jnp.stack((jnp.zeros_like(value), value)),
            )
        for coedge in (blend_loop[1], blend_loop[3]):
            pc = pcurves[coedge]
            if not isinstance(pc, PeriodicPCurve):
                raise ValueError(
                    f"closed_seam_radius_map.seam must retain native phase; got {type(pc).__name__}"
                )
            pcurves[coedge] = eqx.tree_at(lambda value: value.patch, pc, patch)
        curves = list(geometry.curves)
        edge_faces = [
            (face, coedge)
            for face, loops in enumerate(geometry.face_loops)
            for loop in loops
            for coedge in loop
        ]
        affected = {
            seam.edge,
            seam.seam_edge,
            geometry.coedge_edges[seam.curved_coedge],
            geometry.coedge_edges[blend_loop[1]],
        }
        for edge in affected:
            face, coedge = next(
                (face, coedge)
                for face, coedge in edge_faces
                if geometry.coedge_edges[coedge] == edge
            )
            curves[geometry.edge_curves[edge]] = _RealizedTrimEdge(
                patches[face], pcurves[coedge]
            )
        ranges = geometry.edge_ranges.at[seam.seam_edge, self.endpoint].set(parameter)
        blend_edge = geometry.coedge_edges[blend_loop[1]]
        ranges = ranges.at[blend_edge].set(jnp.stack((lower, upper)))
        vertices = geometry.vertex_points
        for edge in (seam.edge, geometry.coedge_edges[seam.curved_coedge]):
            curve = curves[geometry.edge_curves[edge]]
            vertices = vertices.at[geometry.edge_vertices[edge][0]].set(
                curve.evaluate(jnp.asarray(0.0))
            )
        realized = eqx.tree_at(
            lambda value: (
                value.curves,
                value.pcurves,
                value.edge_ranges,
                value.vertex_points,
            ),
            geometry,
            (tuple(curves), tuple(pcurves), ranges, vertices),
        )
        uv = _radius_mesh_parameters(
            model, seam, self.other_radius, rc, old_patch.major_radius, bounds
        )
        return BRepParameterGeometry(
            patches=patches,
            geometry=realized,
            parameter_bounds=bounds,
            triangle_parameters=uv,
        )


def _radius_mesh_parameters(
    model: BRepModel,
    seam: _Seam,
    other_radius: float,
    radius: Array,
    reference: Array,
    bounds: Array,
) -> Array:
    """Transport disk/annulus radial coordinates and rectangular curved charts."""
    faces = model.triangle_face_ids
    old_boxes, new_boxes = model.parameter_bounds[faces], bounds[faces]
    fractions = (model.triangle_parameters - old_boxes[:, None, 0]) / (
        old_boxes[:, None, 1] - old_boxes[:, None, 0]
    )
    uv = new_boxes[:, None, 0] + fractions * (
        new_boxes[:, None, 1] - new_boxes[:, None, 0]
    )
    plane_uv = model.triangle_parameters
    rho = jnp.sqrt(jnp.sum(plane_uv * plane_uv, axis=-1))
    new_rho = other_radius + (rho - other_radius) * (radius - other_radius) / (
        reference - other_radius
    )
    transported = plane_uv * (new_rho / jnp.where(rho > 0, rho, 1.0))[..., None]
    return jnp.where((faces == seam.plane_face)[:, None, None], transported, uv)


def _radius_limit(
    source: BRepModel,
    seam: _Seam,
    endpoint: int,
    other: float,
    budget: CoordinateEnclosureBudget,
) -> float | None:
    """Outward lower bound of the first meridian/chart collision radius."""
    far = float(source.parameter_bounds[seam.curved_face, 1 - endpoint, 1])
    q, d, z = seam.radial_sense, seam.axial_sense, seam.z
    sphere_pole = isinstance(seam.curved, SpherePatch) and far == -d * pi / 2 and q == -1
    if sphere_pole and other == 0:
        return None

    def limits(point: Array) -> Array:
        # The dummy point keeps this a source-enclosure map, not a host formula.
        zero = point[0]
        values: list[Array] = []
        match seam.curved:
            case CylinderPatch():
                values.append(jnp.asarray(abs(far - z)) + zero)
                if q == 1:
                    values.append(seam.curved.radius + zero)
                if other > 0:
                    values.append(jnp.asarray(abs(seam.radial - other)) + zero)
            case ConePatch():
                slope = jnp.tan(seam.curved.semi_angle)
                length = jnp.sqrt(1 + slope * slope)
                values.append((far - z) / (-d - q * slope / length) + zero)
                coefficient = -d * slope - q * length
                if q == 1:
                    values.append(-seam.radial / coefficient + zero)
                if other > 0:
                    values.append((other - seam.radial) / coefficient + zero)
            case SpherePatch():
                sphere = seam.curved.radius
                coefficient = 2 * (d * z - q * sphere)
                if q == 1:
                    values.append((z * z - sphere * sphere) / coefficient + zero)
                if not sphere_pole:
                    sine = jnp.sin(far)
                    values.append((z - sphere * sine) / (d - q * sine) + zero)
                if other > 0:
                    values.append(
                        (other * other - sphere * sphere + z * z) / coefficient + zero
                    )
        return jnp.stack(values)

    budget.reserve(128, 0)
    prepared = prepare_interval_function(limits, 1, batch_capacity=1)
    low, _ = prepared.evaluate(((0.0,),), ((0.0,),))
    upper = float(np.min(low))
    if not np.isfinite(upper) or upper <= 0:
        raise BRepBlendUnsupported(
            "closed seam radius map: a positive topology interval is uncertified"
        )
    return float(np.nextafter(upper, -np.inf))


def _chamfer_scale_limit(
    source: BRepModel,
    seam: _Seam,
    endpoint: int,
    other: float,
    setbacks: tuple[float, float],
    budget: CoordinateEnclosureBudget,
    /,
) -> float:
    a, b = setbacks
    far = float(source.parameter_bounds[seam.curved_face, 1 - endpoint, 1])
    q, d = seam.radial_sense, seam.axial_sense
    if isinstance(seam.curved, SpherePatch):
        raise BRepBlendUnsupported(
            "closed seam chamfer range: full-scale spherical generatrix sign is unproved"
        )

    def limits(point: Array) -> Array:
        zero = point[0]
        if isinstance(seam.curved, ConePatch):
            slope = jnp.tan(seam.curved.semi_angle)
            length = jnp.sqrt(1 + slope * slope)
        else:
            slope, length = jnp.asarray(0.0), jnp.asarray(1.0)
        values = [(far - seam.parameter) * length / (-d * b) + zero]
        if q == 1:
            values.append(seam.radial / a + zero)
        if other > 0:
            values.append(abs(seam.radial - other) / a + zero)
        if isinstance(seam.curved, ConePatch) and d * float(seam.curved.semi_angle) > 0:
            values.append(seam.radial * length / (d * b * slope) + zero)
        values.append(q * a - d * b * slope / length + zero)
        return jnp.stack(values)

    budget.reserve(256, 0)
    prepared = prepare_interval_function(limits, 1, batch_capacity=1)
    low, high = prepared.evaluate(((0.0,),), ((0.0,),))
    if low[0, -1] <= 0 <= high[0, -1]:
        raise BRepBlendUnsupported(
            "closed seam chamfer range: full-scale generatrix sign is uncertified"
        )
    upper = float(np.min(low[0, :-1]))
    if not np.isfinite(upper) or upper <= 0:
        raise BRepBlendUnsupported(
            "closed seam chamfer range: positive chart/axis scale interval is uncertified"
        )
    return float(np.nextafter(upper, -np.inf))


def closed_seam_parameter_range(
    source: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Reuse the interval-certified meridian/chart collision bound."""
    parameter = "radius" if setbacks is None else "setback_scale"
    try:
        seam = _seam(source, edge_index, budget)
        bounds = np.asarray(source.parameter_bounds[seam.curved_face])
        if seam.parameter not in (bounds[0, 1], bounds[1, 1]) or bounds[0, 0] != 0:
            raise BRepBlendUnsupported(
                "closed seam range: selected seam is not an endpoint of the native chart"
            )
        endpoint = 0 if seam.parameter == float(bounds[0, 1]) else 1
        other = 0.0
        geometry = source.require_geometry()
        for loop in geometry.face_loops[seam.plane_face]:
            if loop[0] != seam.plane_coedge:
                pcurve = geometry.pcurves[loop[0]]
                if not isinstance(pcurve, CircleCurve) or not np.array_equal(
                    pcurve.center, np.zeros(2)
                ):
                    raise BRepBlendUnsupported(
                        "closed seam blend: nonconcentric plane loops are unsupported"
                    )
                other = float(pcurve.radius)
        upper = (
            _radius_limit(source, seam, endpoint, other, budget)
            if setbacks is None
            else _chamfer_scale_limit(source, seam, endpoint, other, setbacks, budget)
        )
    except BRepBlendUnsupported as error:
        return unknown_parameter_range(parameter, str(error))
    if upper is None:
        return unknown_parameter_range(
            parameter, "closed seam meridian has no proved finite or unbounded full range"
        )
    return CertifiedParameterRange(
        parameter,
        (0.0, upper),
        end_status=("attained_exactly", "certified_inner_bound"),
        constraints=("parameter_domain", "trim_vanishes"),
        entity_ids=(
            (source.edge_ids[edge_index],),
            (source.face_ids[seam.plane_face], source.face_ids[seam.curved_face]),
        ),
        evidence=(
            "Native shared-frame circular seam with rectangular curved chart.",
            "Outward interval evaluation of the first meridian, radial, or chart collision; lower-rounded upper endpoint."
            if setbacks is None
            else f"Scale multiplies authored plane/curved setbacks {setbacks!r}; constant-sign generatrix and chart/radial bounds are interval-certified.",
        ),
    )


def prepare_closed_seam_radius_map(
    source: BRepModel, edge_index: int, radius: float, result: BRepModel
) -> AbstractBRepParameterMap:
    """Prepare the one radius parameter and every correlated native numerical leaf."""
    with coordinate_source_request((source, edge_index, radius, result)) as budget:
        seam = _seam(source, edge_index, budget)
        meridian = _fillet(seam, radius, budget)
        _require_clearance(source, seam, meridian)
    endpoint = (
        0
        if seam.parameter == float(source.parameter_bounds[seam.curved_face, 0, 1])
        else 1
    )
    angle = atan2(
        meridian.curved_z - (seam.z - seam.axial_sense * radius),
        meridian.curved_radius - meridian.plane_radius,
    )
    other = 0.0
    geometry = source.require_geometry()
    for loop in geometry.face_loops[seam.plane_face]:
        if loop[0] != seam.plane_coedge:
            pc = geometry.pcurves[loop[0]]
            if isinstance(pc, CircleCurve):
                other = float(pc.radius)
    with coordinate_source_request((source, seam, radius, result)) as budget:
        radius_upper = _radius_limit(source, seam, endpoint, other, budget)
    if radius_upper is not None and radius >= radius_upper:
        raise BRepBlendUnsupported(
            "closed seam radius map: radius is outside its certified open topology interval"
        )
    return _ClosedSeamRadiusMap(
        seam,
        jnp.asarray(radius, dtype=jnp.float64),
        result,
        meridian.blend_parameter - angle,
        endpoint,
        other,
        radius_upper,
    )
