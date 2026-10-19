#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Whole-edge interval certificates, without sampled convexity decisions."""

from __future__ import annotations

from fractions import Fraction
from math import pi
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._validation import positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._interval_enclosure import interval_divide, interval_multiply, monotone
from ._exact import square_root_interval
from ._model import BRepCurve, BRepEntityId, BRepGeometry, BRepModel
from ._patches import CircleCurve, interval_cross, interval_dot, interval_norm, LineCurve


def _closed_length(
    curve: LineCurve | CircleCurve,
    endpoints: np.ndarray,
    /,
) -> tuple[float, float] | None:
    width = (
        np.nextafter(endpoints[1, 0] - endpoints[0, 1], -np.inf),
        np.nextafter(endpoints[1, 1] - endpoints[0, 0], np.inf),
    )
    if isinstance(curve, LineCurve):
        direction = tuple(Fraction(float(value)) for value in np.asarray(curve.direction))
        root = square_root_interval(
            sum((value * value for value in direction), Fraction(0))
        )
        speed = (np.float64(root[0]), np.float64(root[1]))
    else:
        a = tuple(Fraction(float(value)) for value in np.asarray(curve.first_axis))
        b = tuple(Fraction(float(value)) for value in np.asarray(curve.second_axis))
        norm = sum((value * value for value in a), Fraction(0))
        if sum(
            (x * y for x, y in zip(a, b, strict=True)), Fraction(0)
        ) != 0 or norm != sum((value * value for value in b), Fraction(0)):
            return None
        root = square_root_interval(norm)
        speed = interval_multiply(
            (np.float64(root[0]), np.float64(root[1])),
            (np.float64(curve.radius), np.float64(curve.radius)),
        )
    result = interval_multiply(speed, width)
    return max(0.0, float(result[0])), float(result[1])


BRepEdgeStatus: TypeAlias = Literal["convex", "concave", "sign_change", "unresolved"]


@final
class BRepEdgeClassification(StrictModule):
    """Source-keyed whole-range classification and length/normal-angle enclosures.

    Change intervals enclose every undecided sign cell, not sampled root guesses.
    Unresolved includes nonmanifold incidence and singular surface charts.
    """

    edge_id: BRepEntityId = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    status: BRepEdgeStatus = eqx.field(static=True)
    change_parameters: Array
    dihedral_range: Array
    length_enclosure: Array
    sign_enclosure: Array


def _material_orientations(model: BRepModel, /) -> np.ndarray:
    orientations = np.asarray(model.orientation, dtype=np.float64).copy()
    signs: list[set[int]] = [set() for _ in model.patches]
    for faces, values in zip(
        model.topology.solid_faces, model.topology.solid_face_orientations, strict=True
    ):
        for face, sign in zip(faces, values, strict=True):
            signs[face].add(sign)
    for face, values in enumerate(signs):
        if len(values) > 1:
            orientations[face] = np.nan
        elif values:
            orientations[face] *= next(iter(values))
    return orientations


def _oriented_normal(
    model: BRepModel,
    geometry: BRepGeometry,
    coedge: int,
    first: float,
    last: float,
    budget: CoordinateEnclosureBudget,
    orientations: np.ndarray,
    owners: tuple[int, ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    face = owners[coedge]
    if not np.isfinite(orientations[face]):
        return np.full(3, -np.inf), np.full(3, np.inf)
    uv = geometry.pcurves[coedge].bounding_box(first, last, budget=budget)
    if not model.patches[face].is_c1_on(uv, budget=budget):
        return np.full(3, -np.inf), np.full(3, np.inf)
    lower, upper = model.patches[face].derivative_bounds(uv, budget=budget)
    normal = interval_cross((lower[:, 0], upper[:, 0]), (lower[:, 1], upper[:, 1]))
    if orientations[face] < 0:
        return -normal[1], -normal[0]
    return normal


def _cell_sign_angle(
    model: BRepModel,
    geometry: BRepGeometry,
    edge: int,
    coedges: tuple[int, int],
    first: float,
    last: float,
    budget: CoordinateEnclosureBudget,
    orientations: np.ndarray,
    owners: tuple[int, ...],
    /,
) -> tuple[float, float, float, float, bool]:
    a = _oriented_normal(
        model, geometry, coedges[0], first, last, budget, orientations, owners
    )
    b = _oriented_normal(
        model, geometry, coedges[1], first, last, budget, orientations, owners
    )
    curve = geometry.curves[geometry.edge_curves[edge]]
    tangent = curve.derivative_bounds(first, last, budget=budget)
    face_b = owners[coedges[1]]
    if geometry.coedge_senses[coedges[1]] * orientations[face_b] < 0:
        tangent = (-tangent[1], -tangent[0])
    sign = interval_dot(a, interval_cross(b, tangent))
    an, bn = interval_norm(a), interval_norm(b)
    regular = bool(an[0] > 0 and bn[0] > 0 and interval_norm(tangent)[0] > 0)
    if not regular:
        return -np.inf, np.inf, 0.0, float(np.nextafter(pi, np.inf)), False
    dot = interval_dot(a, b)
    if dot[0] > 0:
        # atan(|a x b| / a.b) avoids acos cancellation at G1 boundaries.
        ratio = interval_divide(interval_norm(interval_cross(a, b)), dot)
        angle = monotone(np.arctan, True, 8)(ratio)
    else:
        cosine = interval_divide(dot, interval_multiply(an, bn))
        angle = monotone(np.arccos, False, 8)(
            (np.clip(cosine[0], -1.0, 1.0), np.clip(cosine[1], -1.0, 1.0))
        )
    angle_lower = max(0.0, float(angle[0]))
    angle_upper = min(float(np.nextafter(pi, np.inf)), float(angle[1]))
    return float(sign[0]), float(sign[1]), angle_lower, angle_upper, True


def _edge_status(signs: list[tuple[float, float]], regular: bool, /) -> BRepEdgeStatus:
    if not regular:
        return "unresolved"
    if all(upper < 0 for _, upper in signs):
        return "convex"
    if all(lower > 0 for lower, _ in signs):
        return "concave"
    if any(upper < 0 for _, upper in signs) and any(lower > 0 for lower, _ in signs):
        return "sign_change"
    return "unresolved"


def _length_enclosure(
    curve: BRepCurve,
    endpoints: np.ndarray,
    cuts: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[float, float]:
    if isinstance(curve, (LineCurve, CircleCurve)):
        closed = _closed_length(curve, endpoints)
        if closed is not None:
            return closed
    lower, upper = 0.0, 0.0
    for start, end in zip(cuts[:-1], cuts[1:], strict=True):
        speed = interval_norm(
            curve.derivative_bounds(float(start), float(end), budget=budget)
        )
        width = (np.nextafter(end - start, -np.inf), np.nextafter(end - start, np.inf))
        contribution = interval_multiply(speed, width)
        lower = float(np.nextafter(lower + float(contribution[0]), -np.inf))
        upper = float(np.nextafter(upper + float(contribution[1]), np.inf))
    if np.any(endpoints[:, 0] != endpoints[:, 1]):
        lower = 0.0
    return max(0.0, lower), max(0.0, upper)


def _edge_record(
    model: BRepModel,
    geometry: BRepGeometry,
    edge: int,
    edge_id: BRepEntityId,
    uses: list[int],
    maximum_cells: int,
    budget: CoordinateEnclosureBudget,
    orientations: np.ndarray,
    owners: tuple[int, ...],
    /,
) -> BRepEdgeClassification:
    budget.reserve(512 * maximum_cells, 8192 * maximum_cells)
    endpoints = geometry.edge_parameter_enclosure(edge, budget=budget)
    cuts = np.linspace(
        endpoints[0, 0], endpoints[1, 1], maximum_cells + 1, dtype=np.float64
    )
    signs: list[tuple[float, float]] = []
    angles: list[tuple[float, float]] = []
    changes: list[tuple[float, float]] = []
    carrier = geometry.edge_curves[edge]
    length = (
        (0.0, 0.0)
        if carrier < 0
        else _length_enclosure(geometry.curves[carrier], endpoints, cuts, budget)
    )
    regular = len(uses) == 2 and carrier >= 0
    if regular:
        regular = geometry.curves[carrier].is_c1_on(
            float(cuts[0]), float(cuts[-1]), budget=budget
        ) and all(
            geometry.pcurves[coedge].is_c1_on(
                float(cuts[0]), float(cuts[-1]), budget=budget
            )
            for coedge in uses
        )
    for start, end in zip(cuts[:-1], cuts[1:], strict=True):
        if len(uses) == 2 and carrier >= 0:
            lo, hi, alo, ahi, valid = _cell_sign_angle(
                model,
                geometry,
                edge,
                (uses[0], uses[1]),
                float(start),
                float(end),
                budget,
                orientations,
                owners,
            )
            regular = regular and valid
        else:
            lo, hi, alo, ahi = -np.inf, np.inf, 0.0, float(np.nextafter(pi, np.inf))
        signs.append((lo, hi))
        angles.append((alo, ahi))
        if lo <= 0 <= hi:
            changes.append((float(start), float(end)))
    status = _edge_status(signs, regular)
    return BRepEdgeClassification(
        edge_id=edge_id,
        model_id=model.model_id,
        status=parse(status, BRepEdgeStatus, "edge.status"),
        change_parameters=jnp.asarray(
            np.asarray(changes, dtype=np.float64).reshape(-1, 2)
        ),
        dihedral_range=jnp.asarray(
            (min(a[0] for a in angles), max(a[1] for a in angles))
        ),
        length_enclosure=jnp.asarray(length),
        sign_enclosure=jnp.asarray((min(s[0] for s in signs), max(s[1] for s in signs))),
    )


@checked
def brep_edge_classification(
    model: BRepModel,
    geometry: BRepGeometry,
    /,
    *,
    maximum_cells: int,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[BRepEdgeClassification, ...]:
    """Enclose every edge in model order using an explicit per-edge cell budget.

    Interval speed integration applies to every supported native curve, including
    ellipse arcs. Opposite strict signs plus regular continuous charts certify a
    sign change; zero-containing cells remain the published change enclosure.
    """
    if not x64_enabled():
        raise ValueError(
            "BRepEdgeClassification.precision must retain binary64 enclosures; got JAX x64 disabled."
        )
    maximum_cells = positive_integer(
        maximum_cells, "BRepEdgeClassification.maximum_cells"
    )
    if model.geometry is not geometry:
        raise ValueError(
            "BRepEdgeClassification.geometry must belong to the model; got a different geometry."
        )
    coordinate_budget.reserve(
        len(geometry.coedge_edges),
        32 * len(geometry.coedge_edges) + 128 * len(model.patches),
    )
    orientations = _material_orientations(model)
    owners = geometry.coedge_faces
    uses: list[list[int]] = [[] for _ in geometry.edge_curves]
    for coedge, edge in enumerate(geometry.coedge_edges):
        uses[edge].append(coedge)
    return tuple(
        _edge_record(
            model,
            geometry,
            edge,
            edge_id,
            uses[edge],
            maximum_cells,
            coordinate_budget,
            orientations,
            owners,
        )
        for edge, edge_id in enumerate(model.edge_ids)
    )
