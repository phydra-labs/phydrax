#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact recognition and distance evidence for circular planar trim loops.

This module does not infer closure from floating point coordinates.  A loop is
accepted only when its authored incidence and exact endpoint authorities prove
that its circular arcs form one simple turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np

from ...discretization._exact_algebraic_geometry import coefficient_interval, ExactScalar
from ...linalg._hermitian_spectral import fraction_sqrt_interval
from .._rational import fraction_endpoint
from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
from ._model import BRepModel
from ._patches import CircleCurve, PlaneCylinderSectionPCurve, PlanePatch


Scalar = ExactScalar
Vector = tuple[Scalar, ...]


def _dot(a: Vector, b: Vector) -> Scalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _angle_order(a: tuple[Scalar, Scalar], b: tuple[Scalar, Scalar]) -> int:
    """Compare nonzero vectors by their exact angle in [0, 2*pi)."""

    def half(v: tuple[Scalar, Scalar]) -> int:
        return int(v[1] < 0 or (v[1] == 0 and v[0] < 0))

    ah, bh = half(a), half(b)
    if ah != bh:
        return -1 if ah < bh else 1
    cross = a[0] * b[1] - a[1] * b[0]
    return -1 if cross > 0 else 1 if cross < 0 else 0


def _endpoint(
    endpoint: object, curve: object
) -> tuple[tuple[Scalar, Scalar], int] | None:
    if isinstance(endpoint, ConicParameterEndpoint):
        if endpoint.carrier is not curve or endpoint.parameter_exact is not None:
            return None
        root = endpoint.root
        return (root.cosine_exact, root.sine_exact), root.turns
    if isinstance(endpoint, NativePeriodEndpoint):
        if (
            endpoint.curve is not curve
            or endpoint.patch is not None
            or endpoint.rational != 0
        ):
            return None
        quarters = endpoint.turns * 4
        if quarters.denominator != 1:
            return None
        n = quarters.numerator
        return (
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            (Fraction(-1), Fraction(0)),
            (Fraction(0), Fraction(-1)),
        )[n % 4], n // 4
    return None


def _harmonics(curve: object) -> tuple[Vector, Vector, Vector] | None:
    if isinstance(curve, CircleCurve) and curve.ambient_dimension == 2:
        return (
            curve.center_exact,
            tuple(curve.radius_exact * x for x in curve.first_axis_exact),
            tuple(curve.radius_exact * x for x in curve.second_axis_exact),
        )
    if isinstance(curve, PlaneCylinderSectionPCurve):
        if curve.linear_exact != (0, 0) or curve.angle_offset_root is not None:
            return None
        return curve.offset_exact, curve.cosine_exact, curve.sine_exact
    return None


def _lift(origin: Vector, first: Vector, second: Vector, uv: Vector) -> Vector:
    return tuple(
        o + uv[0] * a + uv[1] * b for o, a, b in zip(origin, first, second, strict=True)
    )


@dataclass(frozen=True, slots=True)
class AnnulusEvidence:
    """Exact source geometry for an outer loop followed by one inner loop."""

    origin: Vector
    first: Vector
    second: Vector
    center: Vector
    inner_squared: Scalar
    outer_squared: Scalar
    witness: Vector


@dataclass(frozen=True, slots=True)
class AnnulusProjectionEvidence:
    # -1: boundary, 0: outside, 1: inside, for the orthogonal plane projection.
    classification: int
    lower: float
    upper: float
    plane_uv: tuple[Scalar, Scalar]
    plane_point: Vector
    # The upper bound is witnessed by this exact point on the trimmed face.
    witness: Vector


def recognize_annulus(model: BRepModel, face: int, /) -> AnnulusEvidence | None:
    """Prove two concentric, oppositely oriented, single-turn circular loops.

    Unsupported endpoints or carriers and any unproved incidence return None.
    Exact scalars are used throughout; no numeric proximity establishes identity.
    """
    patch = model.patches[face]
    if not isinstance(patch, PlanePatch):
        return None
    geometry = model.geometry
    loops = geometry.face_loops[face]
    if len(loops) != 2 or any(not loop for loop in loops):
        return None
    origin, first, second = (
        tuple(Fraction.from_float(float(x)) for x in np.asarray(v))
        for v in (patch.origin, patch.first_axis, patch.second_axis)
    )
    gram = (
        (_dot(first, first), _dot(first, second)),
        (_dot(first, second), _dot(second, second)),
    )
    if gram[0][0] * gram[1][1] - gram[0][1] ** 2 <= 0:
        return None

    def metric(a: Vector, b: Vector) -> Scalar:
        return a[0] * (gram[0][0] * b[0] + gram[0][1] * b[1]) + a[1] * (
            gram[1][0] * b[0] + gram[1][1] * b[1]
        )

    records = []
    for loop in loops:
        center = radius = None
        direction = None
        reference = None
        starts = []
        ends = []
        vertices = []
        winding = 0
        witness = None
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            pcurve = geometry.pcurves[coedge]
            harmonic = _harmonics(pcurve)
            if harmonic is None:
                return None
            c, a, b = harmonic
            squared = metric(a, a)
            det = a[0] * b[1] - a[1] * b[0]
            if squared <= 0 or metric(b, b) != squared or metric(a, b) != 0 or det == 0:
                return None
            if center is None:
                center, radius, reference = c, squared, (a, b)
            elif c != center or squared != radius:
                return None
            sense = geometry.coedge_senses[coedge]
            orientation = (1 if det > 0 else -1) * sense
            if direction is None:
                direction = orientation
            elif direction != orientation:
                return None
            roots = geometry.coedge_endpoint_roots[coedge]
            endpoints = tuple(_endpoint(root, pcurve) for root in roots)
            if any(item is None for item in endpoints):
                return None
            (p, pt), (q, qt) = cast(
                tuple[tuple[tuple[Scalar, Scalar], int], ...], endpoints
            )
            order = _angle_order(p, q)
            # Strictly positive parameter span, at most one exact turn.
            delta = qt - pt
            if not (delta == 0 and order < 0 or delta == 1 and order >= 0):
                return None
            full = delta == 1 and order == 0
            uv0 = tuple(c[i] + a[i] * p[0] + b[i] * p[1] for i in range(2))
            uv1 = tuple(c[i] + a[i] * q[0] + b[i] * q[1] for i in range(2))
            for root, uv in zip(roots, (uv0, uv1), strict=True):
                if (
                    isinstance(root, ConicParameterEndpoint)
                    and _lift(origin, first, second, uv) != root.point_exact
                ):
                    return None
            if sense < 0:
                uv0, uv1 = uv1, uv0
            v0, v1 = geometry.edge_vertices[edge]
            if sense < 0:
                v0, v1 = v1, v0
            vertices.append((v0, v1))
            starts.append(uv0)
            ends.append(uv1)
            if witness is None:
                witness = _lift(origin, first, second, uv0)
            ra, rb = cast(tuple[Vector, Vector], reference)
            determinant = ra[0] * rb[1] - ra[1] * rb[0]

            def coordinates(uv: Vector) -> tuple[Scalar, Scalar]:
                x, y = uv[0] - center[0], uv[1] - center[1]
                return (
                    (x * rb[1] - y * rb[0]) / determinant,
                    (ra[0] * y - ra[1] * x) / determinant,
                )

            # The reference basis itself may reverse chart orientation.
            relative_direction = orientation * (1 if determinant > 0 else -1)
            cmp = _angle_order(coordinates(uv0), coordinates(uv1))
            if full:
                winding += 1
            elif relative_direction > 0 and cmp > 0 or relative_direction < 0 and cmp < 0:
                winding += 1
        if winding != 1:
            return None
        if any(
            ends[i] != starts[(i + 1) % len(loop)]
            or vertices[i][1] != vertices[(i + 1) % len(loop)][0]
            for i in range(len(loop))
        ):
            return None
        if len(set(v[0] for v in vertices)) != len(vertices):
            return None
        records.append(
            (
                cast(Vector, center),
                cast(Scalar, radius),
                cast(int, direction),
                cast(Vector, witness),
            )
        )
    outer, inner = records
    if outer[0] != inner[0] or outer[1] <= inner[1] or outer[2] != -inner[2]:
        return None
    return AnnulusEvidence(origin, first, second, outer[0], inner[1], outer[1], outer[3])


def project_annulus(
    evidence: AnnulusEvidence, point: np.ndarray, /
) -> AnnulusProjectionEvidence | None:
    """Classify the orthogonal projection and bound distance to the whole face.

    The lower bound combines exact plane distance with a certified radial-gap
    bound.  The upper bound is the distance to an exact authored boundary point.
    """
    values = np.asarray(point, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return None
    query = tuple(Fraction.from_float(float(x)) for x in values)
    a, b = evidence.first, evidence.second
    rel = tuple(x - o for x, o in zip(query, evidence.origin, strict=True))
    aa, ab, bb = _dot(a, a), _dot(a, b), _dot(b, b)
    determinant = aa * bb - ab * ab
    ar, br = _dot(a, rel), _dot(b, rel)
    uv = ((bb * ar - ab * br) / determinant, (aa * br - ab * ar) / determinant)
    plane = _lift(evidence.origin, a, b, uv)
    delta = (uv[0] - evidence.center[0], uv[1] - evidence.center[1])
    radial = aa * delta[0] ** 2 + 2 * ab * delta[0] * delta[1] + bb * delta[1] ** 2
    inner, outer = evidence.inner_squared, evidence.outer_squared
    classification = (
        -1 if radial == inner or radial == outer else int(inner < radial < outer)
    )
    plane_sq = _dot(
        tuple(x - y for x, y in zip(query, plane, strict=True)),
        tuple(x - y for x, y in zip(query, plane, strict=True)),
    )

    def root_interval(value: Scalar) -> tuple[Fraction, Fraction]:
        lo, hi = coefficient_interval(value, bits=128)
        return fraction_sqrt_interval(max(Fraction(0), lo))[0], fraction_sqrt_interval(
            max(Fraction(0), hi)
        )[1]

    rl, rh = root_interval(radial)
    gap = Fraction(0)
    if radial < inner:
        gap = max(Fraction(0), root_interval(inner)[0] - rh)
    elif radial > outer:
        gap = max(Fraction(0), rl - root_interval(outer)[1])
    lower_sq = max(Fraction(0), coefficient_interval(plane_sq, bits=128)[0]) + gap * gap
    lower = fraction_endpoint(
        fraction_sqrt_interval(lower_sq)[0], upper=False, saturate=True
    )
    witness = plane if classification != 0 else evidence.witness
    diff = tuple(x - y for x, y in zip(query, witness, strict=True))
    upper_sq = max(Fraction(0), coefficient_interval(_dot(diff, diff), bits=128)[1])
    upper = fraction_endpoint(
        fraction_sqrt_interval(upper_sq)[1], upper=True, saturate=True
    )
    return AnnulusProjectionEvidence(classification, lower, upper, uv, plane, witness)
