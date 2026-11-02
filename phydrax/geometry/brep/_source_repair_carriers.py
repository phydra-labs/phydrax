#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Carrier reconstruction for explicit authored-incidence source repair.

The incidence graph, not geometric proximity or source entity identifiers, selects
which surfaces share each carrier.  This module only constructs new carriers;
the source model and its authored occurrences remain untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import atan2, pi
from typing import cast

import numpy as np
from jax import Array

from ...discretization._exact_algebraic_geometry import ExactScalar
from ._boolean_core import pcurve_periodic_source, PeriodicPCurve
from ._correspondence import _affine_bspline_pcurve, torus_rooted_circle_pcurve
from ._model import BRepModel
from ._patches import (
    _validate_source_shadows,
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    LineCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    TorusPatch,
)
from ._source_correspondence import _affine_curve_coefficients
from ._source_repair_roots import _sqrt_rational


@dataclass(frozen=True, slots=True)
class TargetCarriers:
    patches: tuple[AbstractSurfacePatch, ...]
    curves: tuple[AbstractCurve, ...]
    edge_curves: tuple[int, ...]
    pcurves: tuple[AbstractCurve, ...]


@dataclass(frozen=True, slots=True)
class SourceAffineChart:
    """Exact common-parameter affine chart and symbolic native-period gauge."""

    line: LineCurve
    period_shifts: tuple[int, int]
    parameter_range: tuple[float, float]


def source_affine_chart(
    pcurve: AbstractCurve,
    patch: AbstractSurfacePatch,
    parameter_range: Sequence[float],
    /,
) -> SourceAffineChart:
    """Prove an authored affine chart on its occurrence's parameter range.

    Integer period shifts remain symbolic; they are not rounded into the line
    coefficients.  This lets connector proofs use the exact line and account
    independently for its chart sheet.  Unsupported or unbound charts refuse.
    """
    first, last = map(float, parameter_range)
    if not np.isfinite(first) or not np.isfinite(last) or not first < last:
        raise ValueError("An affine source chart requires an increasing finite range.")
    _validate_source_shadows(pcurve, set())
    result = pcurve_periodic_source(pcurve, patch)
    if result is None:
        raise ValueError("The authored p-curve is not bound to its owning patch.")
    source, shifts = result
    # The periodic-source utility can also strip affine transports; those are
    # not identity transports, so only accept actual periodic unwrapping here.
    unwrapped = pcurve
    while isinstance(unwrapped, PeriodicPCurve):
        unwrapped = unwrapped.source_curve
    if source is not unwrapped or source.ambient_dimension != 2:
        raise ValueError("Unsupported authored affine chart transport.")
    domain = source.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        raise ValueError("The occurrence range lies outside its source p-curve.")
    coefficients = _affine_curve_coefficients(source, (Fraction(0), Fraction(0)))
    if coefficients is None:
        normalized = _affine_bspline_pcurve(source, first, last)
        if normalized is not None:
            coefficients = _affine_curve_coefficients(
                normalized, (Fraction(0), Fraction(0))
            )
    if coefficients is None or not any(coefficients[1]):
        raise ValueError("The authored p-curve is not proved affine on its range.")
    return SourceAffineChart(_line(*coefficients), shifts, (first, last))


def _dot(a: Sequence[ExactScalar], b: Sequence[ExactScalar]) -> ExactScalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _sub(a: Sequence[ExactScalar], b: Sequence[ExactScalar]) -> tuple[ExactScalar, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _add(a: Sequence[ExactScalar], b: Sequence[ExactScalar]) -> tuple[ExactScalar, ...]:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _scale(a: Sequence[ExactScalar], s: ExactScalar | int) -> tuple[ExactScalar, ...]:
    return tuple(x * s for x in a)


def _cross(
    a: Sequence[ExactScalar], b: Sequence[ExactScalar]
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rational_vector(array: Array | np.ndarray) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(x)) for x in np.asarray(array))


def _circle(
    center: tuple[ExactScalar, ...],
    first: tuple[ExactScalar, ...],
    second: tuple[ExactScalar, ...],
    radius: ExactScalar,
) -> CircleCurve:
    return CircleCurve(
        tuple(map(float, center)),
        tuple(map(float, first)),
        tuple(map(float, second)),
        float(radius),
        center_exact=center,
        first_axis_exact=first,
        second_axis_exact=second,
        radius_exact=radius,
    )


def _line(
    origin: tuple[ExactScalar, ...], direction: tuple[ExactScalar, ...]
) -> LineCurve:
    return LineCurve(
        tuple(map(float, origin)),
        tuple(map(float, direction)),
        origin_exact=origin,
        direction_exact=direction,
    )


def _plane_frame(
    patch: PlanePatch,
) -> tuple[
    tuple[Fraction, ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
    tuple[ExactScalar, ...],
]:
    origin = _rational_vector(patch.origin)
    a, b = _rational_vector(patch.first_axis), _rational_vector(patch.second_axis)
    normal = _cross(a, b)
    if _dot(normal, normal) == 0:
        raise ValueError("Source repair requires a nondegenerate plane chart.")
    return origin, a, b, normal


def _plane_coordinates(
    vector: tuple[ExactScalar, ...],
    a: tuple[ExactScalar, ...],
    b: tuple[ExactScalar, ...],
) -> tuple[ExactScalar, ExactScalar]:
    aa, ab, bb = _dot(a, a), _dot(a, b), _dot(b, b)
    det = aa * bb - ab * ab
    if det == 0:
        raise ValueError("Source repair requires an independent plane frame.")
    av, bv = _dot(a, vector), _dot(b, vector)
    u, v = (av * bb - bv * ab) / det, (bv * aa - av * ab) / det
    if _add(_scale(a, u), _scale(b, v)) != vector:
        raise ValueError("Repaired circle is not exactly incident to its plane.")
    return (u, v)


def _axis_plane_point(
    cylinder: CylinderPatch, plane: PlanePatch
) -> tuple[ExactScalar, ...]:
    origin, _, _, normal = _plane_frame(plane)
    denominator = _dot(cylinder.axis_exact, normal)
    if denominator == 0 or _cross(cylinder.axis_exact, normal) != (0, 0, 0):
        raise ValueError("Source repair requires a plane normal to the cylinder axis.")
    height = _dot(_sub(origin, cylinder.origin_exact), normal) / denominator
    return _add(cylinder.origin_exact, _scale(cylinder.axis_exact, height))


def _phase(
    first: tuple[ExactScalar, ...],
    second: tuple[ExactScalar, ...],
    a: tuple[ExactScalar, ...],
    b: tuple[ExactScalar, ...],
) -> tuple[Fraction, int]:
    c, s = _dot(first, a), _dot(first, b)
    if (c, s) not in ((1, 0), (0, 1), (-1, 0), (0, -1)) or first != _add(
        _scale(a, c), _scale(b, s)
    ):
        raise ValueError("Torus longitude phase is not an exact cardinal source phase.")
    tangent = _add(_scale(a, -s), _scale(b, c))
    if second == tangent:
        orientation = 1
    elif second == _scale(tangent, -1):
        orientation = -1
    else:
        raise ValueError("Circle frame is not aligned with its target surface.")
    return Fraction(((1, 0), (0, 1), (-1, 0), (0, -1)).index((c, s)), 4), orientation


def _latitude_chart(
    curve: CircleCurve,
    patch: TorusPatch,
    bounds: Sequence[float],
    source_chart: AbstractCurve,
) -> AbstractCurve:
    phase, orientation = _phase(
        curve.first_axis_exact,
        curve.second_axis_exact,
        patch.first_axis_exact,
        patch.second_axis_exact,
    )
    offset = _sub(curve.center_exact, patch.center_exact)
    height = _dot(offset, patch.axis_exact)
    if offset != _scale(patch.axis_exact, height):
        raise ValueError("Torus latitude center is not on its axis.")
    cosine = (curve.radius_exact - patch.major_radius_exact) / patch.minor_radius_exact
    sine = height / patch.minor_radius_exact
    meridian = _circle(
        _add(
            patch.center_exact, _scale(patch.first_axis_exact, patch.major_radius_exact)
        ),
        patch.first_axis_exact,
        patch.axis_exact,
        patch.minor_radius_exact,
    )
    if cosine * cosine + sine * sine != 1:
        raise ValueError("Torus latitude is not exactly incident to its target torus.")
    # The authored chart chooses sheets; exact conic coefficients choose the root.
    sample = (bounds[0] + bounds[1]) / 2
    source_uv = np.asarray(source_chart.evaluate(cast(Array, np.asarray(sample))))
    longitude_turns = (float(source_uv[0]) - orientation * sample) / (2 * pi) - float(
        phase
    )
    longitude_sheet = round(longitude_turns)
    if abs(longitude_turns - longitude_sheet) >= 0.5:
        raise ValueError("The authored torus longitude does not select a unique sheet.")
    phase += longitude_sheet
    principal = atan2(float(sine), float(cosine)) % (2 * pi)
    turns_float = (float(source_uv[1]) - principal) / (2 * pi)
    nearest = round(turns_float)
    if abs(turns_float - nearest) >= 0.5:
        raise ValueError("The authored torus chart does not select a unique sheet.")
    root = ConicParameterRoot(meridian, cosine, sine, turns=int(nearest))
    return torus_rooted_circle_pcurve(root, phase, orientation)


def _plane_chart(
    curve: CircleCurve, plane: PlanePatch, bounds: tuple[float, float]
) -> PlaneCylinderSectionPCurve:
    origin, a, b, _ = _plane_frame(plane)
    center = _plane_coordinates(_sub(curve.center_exact, origin), a, b)
    cosine = _plane_coordinates(_scale(curve.first_axis_exact, curve.radius_exact), a, b)
    sine = _plane_coordinates(_scale(curve.second_axis_exact, curve.radius_exact), a, b)
    zero = (Fraction(0), Fraction(0))
    return PlaneCylinderSectionPCurve(center, zero, cosine, sine, parameter_range=bounds)


def _cylinder_circle_chart(
    curve: CircleCurve, patch: CylinderPatch, bounds: tuple[float, float]
) -> PlaneCylinderSectionPCurve:
    offset = _sub(curve.center_exact, patch.origin_exact)
    height = _dot(offset, patch.axis_exact)
    if (
        offset != _scale(patch.axis_exact, height)
        or curve.radius_exact != patch.radius_exact
    ):
        raise ValueError("Circle is not an exact cylinder section.")
    c = _dot(curve.first_axis_exact, patch.first_axis_exact)
    s = _dot(curve.first_axis_exact, patch.second_axis_exact)
    if (
        curve.first_axis_exact
        != _add(_scale(patch.first_axis_exact, c), _scale(patch.second_axis_exact, s))
        or curve.second_axis_exact
        != _add(_scale(patch.first_axis_exact, -s), _scale(patch.second_axis_exact, c))
        or c * c + s * s != 1
    ):
        raise ValueError("Cylinder section has unsupported authored angular orientation.")
    phase = None
    if (c, s) != (1, 0):
        unit = _circle(
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            Fraction(1),
        )
        phase = ConicParameterRoot(unit, c, s)
    zero = (Fraction(0), Fraction(0))
    return PlaneCylinderSectionPCurve(
        (Fraction(0), height),
        (Fraction(1), Fraction(0)),
        zero,
        zero,
        parameter_range=bounds,
        angle_offset_root=phase,
    )


def _cylinder_line_chart(
    curve: LineCurve, patch: CylinderPatch, bounds: tuple[float, float]
) -> ConicRootLinePCurve:
    offset = _sub(curve.origin_exact, patch.origin_exact)
    height = _dot(offset, patch.axis_exact)
    rate = _dot(curve.direction_exact, patch.axis_exact)
    radial = _sub(offset, _scale(patch.axis_exact, height))
    if (
        rate == 0
        or curve.direction_exact != _scale(patch.axis_exact, rate)
        or _dot(radial, radial) != patch.radius_exact * patch.radius_exact
    ):
        raise ValueError("Repaired seam is not an exact cylinder generator.")
    c = _dot(radial, patch.first_axis_exact) / patch.radius_exact
    s = _dot(radial, patch.second_axis_exact) / patch.radius_exact
    base_circle = _circle(
        patch.origin_exact,
        patch.first_axis_exact,
        patch.second_axis_exact,
        patch.radius_exact,
    )
    return ConicRootLinePCurve(
        ConicParameterRoot(base_circle, c, s), height, rate, parameter_range=bounds
    )


def build_target_carriers(model: BRepModel, /) -> TargetCarriers:
    """Reconstruct shared analytic carriers in authored face/edge/use order.

    Only unambiguous plane/cylinder/torus incidence is supported.  Each torus
    is fitted exactly through its cylinder seam and the authored top ring.
    """
    geometry = model.geometry
    patches = list(model.patches)
    face_uses = [[] for _ in geometry.edge_curves]
    coedge_face = [None] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                if coedge_face[coedge] is not None:
                    raise ValueError("A coedge has ambiguous face ownership.")
                coedge_face[coedge] = face
                face_uses[geometry.coedge_edges[coedge]].append(face)
    if any(face is None for face in coedge_face) or any(
        len(uses) != 2 for uses in face_uses
    ):
        raise ValueError("Repair requires exactly two authored uses of every edge.")
    source_curves = tuple(geometry.curves[index] for index in geometry.edge_curves)
    torus_data = {}
    for face, patch in enumerate(patches):
        if not isinstance(patch, TorusPatch):
            continue
        cylinders, planes = set(), set()
        for uses in face_uses:
            if face not in uses:
                continue
            for other in uses:
                if other == face:
                    continue
                if isinstance(patches[other], CylinderPatch):
                    cylinders.add(other)
                elif isinstance(patches[other], PlanePatch):
                    planes.add(other)
        if len(cylinders) != 1 or len(planes) != 1:
            raise ValueError(
                "A torus must have one authored cylinder and one plane neighbor."
            )
        cylinder = patches[next(iter(cylinders))]
        plane = patches[next(iter(planes))]
        if (
            patch.axis_exact != cylinder.axis_exact
            or _dot(patch.axis_exact, patch.axis_exact) != 1
        ):
            raise ValueError(
                "Torus and cylinder axes must have the same exact orientation."
            )
        axial = _dot(_sub(patch.center_exact, cylinder.origin_exact), cylinder.axis_exact)
        center = _add(cylinder.origin_exact, _scale(cylinder.axis_exact, axial))
        top = _axis_plane_point(cylinder, plane)
        height = _dot(_sub(top, center), cylinder.axis_exact)
        rings = [
            source_curves[i]
            for i, uses in enumerate(face_uses)
            if face in uses and next(iter(planes)) in uses
        ]
        if not rings or any(not isinstance(ring, CircleCurve) for ring in rings):
            raise ValueError("Torus/plane incidence requires circular authored edges.")
        circular_rings = cast(list[CircleCurve], rings)
        radius = circular_rings[0].radius_exact
        if any(ring.radius_exact != radius for ring in circular_rings):
            raise ValueError("Split torus/plane edges disagree on their circle radius.")
        seam = cylinder.radius_exact
        delta = radius - seam
        if (
            not all(
                isinstance(x, Fraction)
                for x in (delta, height, seam, patch.major_radius_exact)
            )
            or delta == 0
        ):
            raise ValueError("Torus fitting requires a unique rational profile.")
        side = (
            1
            if seam > patch.major_radius_exact
            else -1
            if seam < patch.major_radius_exact
            else 0
        )
        if not side:
            raise ValueError("The authored torus does not select a seam side.")
        minor = -(delta * delta + height * height) / (2 * side * delta)
        major = seam - side * minor
        if minor <= 0 or major <= minor:
            raise ValueError("The fitted torus is not a regular positive ring torus.")
        target = TorusPatch(
            tuple(map(float, center)),
            patch.first_axis,
            patch.second_axis,
            patch.axis,
            float(major),
            float(minor),
            center_exact=center,
            first_axis_exact=patch.first_axis_exact,
            second_axis_exact=patch.second_axis_exact,
            axis_exact=patch.axis_exact,
            major_radius_exact=major,
            minor_radius_exact=minor,
        )
        patches[face] = target
        torus_data[face] = (cylinder, plane, top)
    curves = []
    for edge, (source, uses) in enumerate(zip(source_curves, face_uses, strict=True)):
        unique = set(uses)
        if isinstance(source, CircleCurve):
            if len(unique) == 1 and isinstance(patches[uses[0]], TorusPatch):
                patch = patches[uses[0]]
                center = _add(
                    patch.center_exact,
                    _scale(patch.first_axis_exact, patch.major_radius_exact),
                )
                if (
                    source.first_axis_exact != patch.first_axis_exact
                    or source.second_axis_exact != patch.axis_exact
                ):
                    raise ValueError(
                        "Torus seam requires its authored canonical meridian frame."
                    )
                curves.append(
                    _circle(
                        center,
                        source.first_axis_exact,
                        source.second_axis_exact,
                        patch.minor_radius_exact,
                    )
                )
                continue
            cylinders = [
                patches[f] for f in unique if isinstance(patches[f], CylinderPatch)
            ]
            tori = [patches[f] for f in unique if isinstance(patches[f], TorusPatch)]
            planes = [patches[f] for f in unique if isinstance(patches[f], PlanePatch)]
            if len(cylinders) == 1 and len(planes) == 1:
                center, radius = (
                    _axis_plane_point(cylinders[0], planes[0]),
                    cylinders[0].radius_exact,
                )
            elif len(cylinders) == 1 and len(tori) == 1:
                center, radius = tori[0].center_exact, cylinders[0].radius_exact
            elif len(tori) == 1 and len(planes) == 1:
                torus_face = next(f for f in unique if isinstance(patches[f], TorusPatch))
                center, radius = torus_data[torus_face][2], source.radius_exact
            else:
                raise ValueError("Circular edge has ambiguous analytic incidence.")
            curves.append(
                _circle(center, source.first_axis_exact, source.second_axis_exact, radius)
            )
        elif (
            isinstance(source, LineCurve)
            and len(unique) == 1
            and isinstance(patches[uses[0]], CylinderPatch)
        ):
            patch = patches[uses[0]]
            offset = _sub(source.origin_exact, patch.origin_exact)
            height = _dot(offset, patch.axis_exact)
            radial = _sub(offset, _scale(patch.axis_exact, height))
            squared = _dot(radial, radial)
            if not isinstance(squared, Fraction) or squared <= 0:
                raise ValueError(
                    "Cylinder seam has no uniquely defined radial direction."
                )
            length = _sqrt_rational(squared)
            origin = _add(
                _add(patch.origin_exact, _scale(patch.axis_exact, height)),
                _scale(radial, patch.radius_exact / length),
            )
            curves.append(_line(origin, source.direction_exact))
        else:
            raise ValueError("Unsupported authored repair edge carrier or incidence.")
    pcurves = []
    for coedge, edge in enumerate(geometry.coedge_edges):
        face = cast(int, coedge_face[coedge])
        patch, curve = patches[face], curves[edge]
        bounds = tuple(float(x) for x in np.asarray(geometry.edge_ranges[edge]))
        chart_domain = (-1.0e100, 1.0e100)
        if isinstance(curve, CircleCurve) and isinstance(patch, PlanePatch):
            chart = _plane_chart(curve, patch, chart_domain)
        elif isinstance(curve, CircleCurve) and isinstance(patch, CylinderPatch):
            chart = _cylinder_circle_chart(curve, patch, chart_domain)
        elif isinstance(curve, CircleCurve) and isinstance(patch, TorusPatch):
            if len(set(face_uses[edge])) == 1:
                chart = _line((Fraction(0), Fraction(0)), (Fraction(0), Fraction(1)))
            else:
                chart = _latitude_chart(curve, patch, bounds, geometry.pcurves[coedge])
        elif isinstance(curve, LineCurve) and isinstance(patch, CylinderPatch):
            chart = _cylinder_line_chart(curve, patch, chart_domain)
        else:
            raise ValueError("Unsupported target coedge carrier.")
        source_chart = geometry.pcurves[coedge]
        if isinstance(source_chart, PeriodicPCurve):
            # Latitude construction already selected both authored sheets.
            if not (
                isinstance(curve, CircleCurve)
                and isinstance(patch, TorusPatch)
                and len(set(face_uses[edge])) != 1
            ):
                chart = PeriodicPCurve(chart, patch, source_chart.period_shifts)
        pcurves.append(chart)
    return TargetCarriers(
        tuple(patches), tuple(curves), tuple(range(len(curves))), tuple(pcurves)
    )
