#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Occurrence-level chart correspondence for explicit source completion.

These bounds concern the source support chart, not the target surface in world
space.  Surface-support displacement is a separate certificate obligation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np

from ...discretization._exact_algebraic_geometry import ExactScalar
from ._boolean_core import pcurve_periodic_source
from ._model import BRepModel
from ._patches import (
    _validate_source_shadows,
    AbstractSurfacePatch,
    CircleCurve,
    ConicRootLinePCurve,
    CylinderPatch,
    LineCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from ._source_repair_carriers import source_affine_chart
from ._source_repair_trim import (
    chart_connector_bound,
    Interval,
    phase_interval,
    scalar_interval,
)
from ._trim_helpers import period_offset, period_scalar_bounds


_ZERO = Fraction(0)
_ONE = Fraction(1)


def _add(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def _mul(a: Interval, b: Interval) -> Interval:
    products = tuple(x * y for x in a for y in b)
    return min(products), max(products)


def _difference(a: Interval, b: Interval) -> Fraction:
    return max(abs(a[0] - b[1]), abs(a[1] - b[0]))


def _magnitude(value: ExactScalar) -> Fraction:
    lo, hi = scalar_interval(value)
    return max(abs(lo), abs(hi))


def _payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _endpoint(model: BRepModel, edge: int, index: int, *, repaired: bool) -> Interval:
    from ._source_repair_completion import _endpoint_interval

    return _endpoint_interval(model, edge, index, repaired=repaired)


def _frame(
    source: AbstractSurfacePatch, target: AbstractSurfacePatch
) -> tuple[Fraction, Fraction]:
    if type(source) is not type(target):
        raise ValueError("Source and target support charts are not comparable.")
    if isinstance(source, PlanePatch):
        # The source-to-target chart correspondence is the identity, not a
        # nearest-plane projection or an implicit chart transform.
        for name in ("origin", "first_axis", "second_axis"):
            if not np.array_equal(
                np.asarray(getattr(source, name)), np.asarray(getattr(target, name))
            ):
                raise ValueError("Plane completion changed its source chart.")
        axes = tuple(
            tuple(
                Fraction.from_float(float(v)) for v in np.asarray(getattr(source, name))
            )
            for name in ("first_axis", "second_axis")
        )
        return (
            sum((abs(v) for v in axes[0]), _ZERO),
            sum((abs(v) for v in axes[1]), _ZERO),
        )
    if not isinstance(source, (CylinderPatch, TorusPatch)):
        raise ValueError("Unsupported source completion support.")
    axes = (source.first_axis_exact, source.second_axis_exact, source.axis_exact)
    target_support = cast(CylinderPatch | TorusPatch, target)
    target_axes = (
        target_support.first_axis_exact,
        target_support.second_axis_exact,
        target_support.axis_exact,
    )
    if axes != target_axes or any(
        sum((a * b for a, b in zip(axes[i], axes[j], strict=True)), _ZERO) != int(i == j)
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError("Source and target require one exact orthonormal chart frame.")
    if isinstance(source, CylinderPatch):
        return _magnitude(source.radius_exact), _ONE
    return _magnitude(source.major_radius_exact) + _magnitude(
        source.minor_radius_exact
    ), _magnitude(source.minor_radius_exact)


def _source_affine(
    model: BRepModel, face: int, coedge: int, bounds: tuple[float, ...]
) -> tuple[tuple[Interval, ...], tuple[ExactScalar, ...]]:
    chart = source_affine_chart(
        model.geometry.pcurves[coedge], model.patches[face], bounds
    )
    origin = tuple(
        phase_interval(chart.line.origin_exact[i], turns=Fraction(chart.period_shifts[i]))
        for i in range(2)
    )
    return origin, chart.line.direction_exact


def _target_affine(
    model: BRepModel, face: int, coedge: int
) -> tuple[tuple[Interval, ...], tuple[ExactScalar, ...]]:
    patch = model.patches[face]
    pcurve = model.geometry.pcurves[coedge]
    _validate_source_shadows(pcurve, set())
    result = pcurve_periodic_source(pcurve, patch)
    if result is None:
        raise ValueError("Target p-curve is not bound to the owning support.")
    source, shifts = result
    if isinstance(source, TorusRootedCirclePCurve):
        origin = (
            phase_interval(_ZERO, turns=source.turns + shifts[0]),
            phase_interval(source.root_line.root, turns=Fraction(shifts[1])),
        )
        direction = (Fraction(source.orientation), _ZERO)
    elif isinstance(source, ConicRootLinePCurve):
        if source.phase_shift_exact != 0:
            raise ValueError("Unsupported target seam phase shift.")
        origin = (
            phase_interval(source.root, turns=Fraction(shifts[0])),
            phase_interval(source.height_origin_exact, turns=Fraction(shifts[1])),
        )
        direction = (_ZERO, source.height_direction_exact)
    elif isinstance(source, PlaneCylinderSectionPCurve):
        if any(source.cosine_exact) or any(source.sine_exact):
            raise ValueError("A cylinder ring must have an affine target chart.")
        if source.angle_offset_root is None:
            first = phase_interval(source.offset_exact[0], turns=Fraction(shifts[0]))
        else:
            if source.offset_exact[0] != 0:
                raise ValueError("Unsupported target cylinder angular offset.")
            first = phase_interval(source.angle_offset_root, turns=Fraction(shifts[0]))
        origin = (
            first,
            phase_interval(source.offset_exact[1], turns=Fraction(shifts[1])),
        )
        direction = source.linear_exact
    elif isinstance(source, LineCurve):
        origin = tuple(
            phase_interval(source.origin_exact[i], turns=Fraction(shifts[i]))
            for i in range(2)
        )
        direction = source.direction_exact
    else:
        raise ValueError("Unsupported target affine chart carrier.")
    return origin, direction


def _at(
    origin: tuple[Interval, ...], direction: tuple[ExactScalar, ...], parameter: Interval
) -> tuple[Interval, Interval]:
    return (
        _add(origin[0], _mul(scalar_interval(direction[0]), parameter)),
        _add(origin[1], _mul(scalar_interval(direction[1]), parameter)),
    )


def _affine_occurrence(
    source: BRepModel,
    target: BRepModel,
    face: int,
    coedge: int,
    lipschitz: tuple[Fraction, Fraction],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    sg = source.geometry
    edge = sg.coedge_edges[coedge]
    bounds = tuple(float(value) for value in np.asarray(sg.edge_ranges)[edge])
    old_origin, old_direction = _source_affine(source, face, coedge, bounds)
    new_origin, new_direction = _target_affine(target, face, coedge)
    old_axes = tuple(i for i, value in enumerate(old_direction) if value != 0)
    new_axes = tuple(i for i, value in enumerate(new_direction) if value != 0)
    if (
        len(old_axes) != 1
        or old_axes != new_axes
        or old_direction[old_axes[0]] * new_direction[new_axes[0]] <= 0
    ):
        raise ValueError(
            "The chart occurrence does not preserve a unique oriented coordinate."
        )
    old_points = tuple(
        _at(old_origin, old_direction, _endpoint(source, edge, i, repaired=False))
        for i in range(2)
    )
    new_points = tuple(
        _at(new_origin, new_direction, _endpoint(target, edge, i, repaired=True))
        for i in range(2)
    )
    # A strict half-period test proves the same sheet, rather than identifying
    # periodic coordinates by proximity or rounded native-period constants.
    period_lo, _ = period_scalar_bounds(period_offset(_ZERO, _ONE))
    half_period = Fraction.from_float(float(period_lo)) / 2
    for axis, symbol in enumerate(source.patches[face].native_period_symbols()):
        if symbol == "two_pi" and any(
            _difference(a[axis], b[axis]) >= half_period
            for a, b in zip(old_points, new_points, strict=True)
        ):
            raise ValueError("The authored occurrence has an ambiguous periodic sheet.")
    connectors = tuple(
        chart_connector_bound(a, b, *lipschitz)
        for a, b in zip(old_points, new_points, strict=True)
    )
    # Both chart occurrences are affine in their edge parameters, and the
    # endpoint-preserving parameter correspondence is affine.  Convexity thus
    # extends the two endpoint bounds over the entire occurrence.
    return max(connectors), connectors


def _plane_occurrence(
    source: BRepModel,
    target: BRepModel,
    face: int,
    coedge: int,
    lipschitz: tuple[Fraction, Fraction],
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    sg, tg = source.geometry, target.geometry
    edge = sg.coedge_edges[coedge]
    old = sg.pcurves[coedge]
    new = tg.pcurves[coedge]
    _validate_source_shadows(old, set())
    _validate_source_shadows(new, set())
    if (
        not isinstance(old, CircleCurve)
        or old.ambient_dimension != 2
        or not isinstance(new, PlaneCylinderSectionPCurve)
    ):
        raise ValueError("Planar circular chart correspondence is not supported.")
    if new.linear_exact != (_ZERO, _ZERO) or new.angle_offset_root is not None:
        raise ValueError("Target plane chart is not a common-parameter harmonic circle.")
    old_coeff = (
        old.center_exact,
        tuple(old.radius_exact * value for value in old.first_axis_exact),
        tuple(old.radius_exact * value for value in old.second_axis_exact),
    )
    new_coeff = (new.offset_exact, new.cosine_exact, new.sine_exact)
    old_end = tuple(_endpoint(source, edge, i, repaired=False) for i in range(2))
    new_end = tuple(_endpoint(target, edge, i, repaired=True) for i in range(2))
    phase_error = max(_difference(a, b) for a, b in zip(old_end, new_end, strict=True))
    period_lo, _ = period_scalar_bounds(period_offset(_ZERO, _ONE))
    if phase_error >= Fraction.from_float(float(period_lo)) / 2:
        raise ValueError("Plane chart occurrence has an ambiguous phase sheet.")
    coordinate_bounds = []
    for axis in range(2):
        support = sum(
            (_magnitude(old_coeff[k][axis] - new_coeff[k][axis]) for k in range(3)), _ZERO
        )
        speed = _magnitude(old_coeff[1][axis]) + _magnitude(old_coeff[2][axis])
        coordinate_bounds.append(support + speed * phase_error)
    # The harmonic support difference and phase interpolation inequality bound
    # the whole occurrence, and each straight chart endpoint connector.
    bound = sum((lipschitz[i] * coordinate_bounds[i] for i in range(2)), _ZERO)
    return bound, (bound, bound)


def certify_chart_completion(
    source: BRepModel, target: BRepModel, face: int, /, *, tolerance: Fraction
) -> tuple[Fraction, dict[str, object]]:
    """Certify every authored use and chart connector on one completed face.

    This is conditional on the explicit full-strip/annulus completion; it does
    not assert the original authored uses already form a closed trim.
    """
    if not isinstance(tolerance, Fraction) or tolerance <= 0:
        raise ValueError("Chart completion requires a positive exact tolerance.")
    if source.geometry.face_loops[face] != target.geometry.face_loops[face]:
        raise ValueError("Chart completion must preserve every coedge occurrence.")
    lipschitz = _frame(source.patches[face], target.patches[face])
    occurrences = [coedge for loop in source.geometry.face_loops[face] for coedge in loop]
    if not occurrences or len(set(occurrences)) != len(occurrences):
        raise ValueError("Face occurrence ownership is empty or ambiguous.")
    records = []
    maximum = _ZERO
    for coedge in occurrences:
        if (
            source.geometry.coedge_edges[coedge] != target.geometry.coedge_edges[coedge]
            or source.geometry.coedge_senses[coedge]
            != target.geometry.coedge_senses[coedge]
        ):
            raise ValueError("Chart completion changed an authored edge occurrence.")
        if isinstance(source.patches[face], PlanePatch):
            bound, connectors = _plane_occurrence(source, target, face, coedge, lipschitz)
        else:
            bound, connectors = _affine_occurrence(
                source, target, face, coedge, lipschitz
            )
        if bound > tolerance:
            raise ValueError(
                "A chart occurrence completion exceeds the repair tolerance."
            )
        maximum = max(maximum, bound)
        records.append(
            {
                "coedge": coedge,
                "edge": source.geometry.coedge_edges[coedge],
                "occurrence_bound": _payload(bound),
                "connector_bounds": [_payload(value) for value in connectors],
            }
        )
    return maximum, {"face": face, "occurrences": records}
