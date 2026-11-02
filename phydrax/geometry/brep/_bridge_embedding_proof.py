#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sufficient exact embedding and full-source separation proof for a bridge.

The caller supplies a constant 2-by-3 projection and two oriented separating
planes, each with an explicitly selected boundary edge on each surface. This
module never infers seam identity, orientation, or correspondence. In
particular, contact containment is not proof that the two edge traces agree.

For each knot rectangle of each of the three patches, the symmetric part of
its projected Jacobian is proved positive definite by Bernstein coefficients
of its leading principal minor and determinant. A common projection per patch
and C1 continuity make that projected map strictly monotone on its entire
convex parameter rectangle, hence injective; its rank is two everywhere.
Positive homogeneous weights justify all cleared denominators. On each
bridge/source pair, the prescribed plane is exactly zero on the selected edges
and strictly signed everywhere else. An additional explicitly supplied plane
strictly separates the two full source images. All decisions use exact
coefficients, not sampling or tolerances. Failure to prove is a refusal, not a
claim that the surface is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ...discretization._coordinate_enclosure import (
    ExactScalar,
    Polynomial,
    add,
    bernstein_coefficients,
    derivative,
    multiply,
    scale,
    subtract,
)
from ...discretization._exact_algebraic_geometry import RadicalElement
from ..._interpolation._bspline_extraction import RationalBSplineSurfaceSpan
from ._spline_projection import _bernstein_power
from ._patches import BSplineSurfacePatch, _validate_surface_source_shadows


@dataclass(frozen=True, slots=True)
class BridgeSeparation:
    """Explicit edge selectors and plane ``normal . x - offset``.

    An edge is ``(parameter_axis, endpoint)``, with axis and endpoint each 0
    or 1. The plane is positive on the bridge and negative on the source,
    except on their respective declared edges.
    """

    bridge_edge: tuple[int, int]
    source_edge: tuple[int, int]
    normal: tuple[ExactScalar, ExactScalar, ExactScalar]
    offset: ExactScalar


@dataclass(frozen=True, slots=True)
class BridgeEmbeddingEvidence:
    source_ids: tuple[str, str, str]
    projections: tuple[tuple[tuple[ExactScalar, ...], ...], ...]
    seams: tuple[BridgeSeparation, BridgeSeparation]
    source_separator: tuple[tuple[ExactScalar, ...], ExactScalar]
    span_counts: tuple[int, int, int]
    principal_controls: tuple[tuple[tuple[ExactScalar, ...], ...], ...]
    determinant_controls: tuple[tuple[tuple[ExactScalar, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class BridgeEmbeddingRefusal:
    reason: str


def _exact(value: object) -> ExactScalar:
    if isinstance(value, (Fraction, RadicalElement)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value)
    raise TypeError("projection and plane coefficients must be exact scalars")


def _sign(value: ExactScalar) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return int(value > 0) - int(value < 0)


def _powers(span: RationalBSplineSurfaceSpan) -> tuple[Polynomial, ...]:
    net = span.homogeneous_bernstein_coefficients
    return tuple(
        _bernstein_power(tuple(tuple(point[component] for point in row) for row in net))
        for component in range(4)
    )


def _weights_positive(spans: tuple[RationalBSplineSurfaceSpan, ...]) -> bool:
    return bool(spans) and all(
        len(point) == 4 and _sign(point[3]) > 0
        for span in spans
        for row in span.homogeneous_bernstein_coefficients
        for point in row
    )


def _c1(patch: BSplineSurfacePatch) -> bool:
    for knots, degree in (
        (patch.exact_u_knots, patch.u_degree),
        (patch.exact_v_knots, patch.v_degree),
    ):
        if degree < 1 or knots is None:
            return False
        start, end = knots[degree], knots[-degree - 1]
        if start >= end:
            return False
        if any(knots.count(knot) >= degree for knot in set(knots) if start < knot < end):
            return False
    return True


def _edge_valid(edge: tuple[int, int]) -> bool:
    return len(edge) == 2 and all(
        type(value) is int and value in (0, 1) for value in edge
    )


def _separated_side(
    spans: tuple[RationalBSplineSurfaceSpan, ...],
    edge: tuple[int, int],
    normal: tuple[ExactScalar, ...],
    offset: ExactScalar,
    direction: int,
) -> bool:
    axis, side = edge
    endpoint = (min if side == 0 else max)(
        span.parameter_bounds[axis][side] for span in spans
    )
    for span in spans:
        net = span.homogeneous_bernstein_coefficients
        on_edge = span.parameter_bounds[axis][side] == endpoint
        last = len(net) - 1 if axis == 0 else len(net[0]) - 1
        for i, row in enumerate(net):
            for j, point in enumerate(row):
                residual = sum(
                    (normal[k] * point[k] for k in range(3)), -offset * point[3]
                )
                expected_zero = on_edge and (i if axis == 0 else j) == (
                    0 if side == 0 else last
                )
                sign = _sign(residual) * direction
                if (expected_zero and sign != 0) or (not expected_zero and sign <= 0):
                    return False
    return True


def _strict_side(
    spans: tuple[RationalBSplineSurfaceSpan, ...],
    normal: tuple[ExactScalar, ...],
    offset: ExactScalar,
    direction: int,
) -> bool:
    return all(
        _sign(sum((normal[k] * point[k] for k in range(3)), -offset * point[3]))
        * direction
        > 0
        for span in spans
        for row in span.homogeneous_bernstein_coefficients
        for point in row
    )


def certify_bridge_embedding(
    bridge: BSplineSurfacePatch,
    source0: BSplineSurfacePatch,
    source1: BSplineSurfacePatch,
    *,
    projection: tuple[
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
    ],
    source_projections: tuple[
        tuple[
            tuple[ExactScalar, ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar, ExactScalar],
        ],
        tuple[
            tuple[ExactScalar, ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar, ExactScalar],
        ],
    ],
    source_separator: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ExactScalar],
    seams: tuple[BridgeSeparation, BridgeSeparation],
) -> BridgeEmbeddingEvidence | BridgeEmbeddingRefusal:
    """Prove regularity, global injectivity and full-source seam containment.

    Exact rational or native algebraic coefficients are required. Refusal means
    only that this sufficient coefficient theorem did not establish the claim.
    """
    try:
        if not all(
            isinstance(p, BSplineSurfacePatch) for p in (bridge, source0, source1)
        ):
            return BridgeEmbeddingRefusal(
                "all three carriers must be B-spline surface patches"
            )
        projections = (projection, *source_projections)
        if (
            len(projections) != 3
            or any(len(m) != 2 or any(len(row) != 3 for row in m) for m in projections)
            or len(seams) != 2
        ):
            return BridgeEmbeddingRefusal("projection or seam extent is invalid")
        matrices = tuple(
            tuple(tuple(_exact(value) for value in row) for row in m) for m in projections
        )
        normal = tuple(_exact(value) for value in source_separator[0])
        offset = _exact(source_separator[1])
        if len(normal) != 3:
            return BridgeEmbeddingRefusal("source separator extent is invalid")
        declared = tuple(
            BridgeSeparation(
                s.bridge_edge,
                s.source_edge,
                (_exact(s.normal[0]), _exact(s.normal[1]), _exact(s.normal[2])),
                _exact(s.offset),
            )
            for s in seams
        )
        if any(
            not _edge_valid(s.bridge_edge)
            or not _edge_valid(s.source_edge)
            or len(s.normal) != 3
            for s in declared
        ):
            return BridgeEmbeddingRefusal("invalid explicit seam selector or plane")
        for patch in (bridge, source0, source1):
            _validate_surface_source_shadows(patch)
        if not all(_c1(p) for p in (bridge, source0, source1)):
            return BridgeEmbeddingRefusal(
                "all patches must have nondegenerate C1 knot domains"
            )
        prepared = tuple(p.prepare_spans() for p in (bridge, source0, source1))
        banks = tuple(item.spans for item in prepared)
        if not all(_weights_positive(spans) for spans in banks):
            return BridgeEmbeddingRefusal("positive homogeneous weights were not proved")
        principals = []
        determinants = []
        for spans, matrix in zip(banks, matrices, strict=True):
            patch_principals = []
            patch_determinants = []
            for span in spans:
                homogeneous = _powers(span)
                weight = homogeneous[3]
                tangents = tuple(
                    tuple(
                        scale(
                            subtract(
                                multiply(derivative(homogeneous[k], a), weight),
                                multiply(homogeneous[k], derivative(weight, a)),
                            ),
                            Fraction(1)
                            / (span.parameter_bounds[a][1] - span.parameter_bounds[a][0]),
                        )
                        for a in range(2)
                    )
                    for k in range(3)
                )
                jacobian = tuple(
                    tuple(
                        add(
                            add(
                                scale(tangents[0][a], matrix[r][0]),
                                scale(tangents[1][a], matrix[r][1]),
                            ),
                            scale(tangents[2][a], matrix[r][2]),
                        )
                        for a in range(2)
                    )
                    for r in range(2)
                )
                cross = scale(add(jacobian[0][1], jacobian[1][0]), Fraction(1, 2))
                determinant = subtract(
                    multiply(jacobian[0][0], jacobian[1][1]), multiply(cross, cross)
                )
                first = bernstein_coefficients(jacobian[0][0], "box", 2)
                second = bernstein_coefficients(determinant, "box", 2)
                if (
                    not first
                    or not second
                    or any(_sign(v) <= 0 for v in (*first, *second))
                ):
                    return BridgeEmbeddingRefusal(
                        "projected symmetric Jacobian positivity was not proved"
                    )
                patch_principals.append(tuple(first))
                patch_determinants.append(tuple(second))
            principals.append(tuple(patch_principals))
            determinants.append(tuple(patch_determinants))
        if not _strict_side(banks[1], normal, offset, 1) or not _strict_side(
            banks[2], normal, offset, -1
        ):
            return BridgeEmbeddingRefusal(
                "full source patches were not strictly separated"
            )
        for index, seam in enumerate(declared):
            if not _separated_side(
                banks[0], seam.bridge_edge, seam.normal, seam.offset, 1
            ):
                return BridgeEmbeddingRefusal(
                    f"bridge side of seam {index} was not proved"
                )
            if not _separated_side(
                banks[index + 1], seam.source_edge, seam.normal, seam.offset, -1
            ):
                return BridgeEmbeddingRefusal(
                    f"source side of seam {index} was not proved"
                )
        return BridgeEmbeddingEvidence(
            (prepared[0].source_id, prepared[1].source_id, prepared[2].source_id),
            matrices,
            (declared[0], declared[1]),
            (normal, offset),
            (len(banks[0]), len(banks[1]), len(banks[2])),
            tuple(principals),
            tuple(determinants),
        )
    except (ArithmeticError, TypeError, ValueError, AttributeError, IndexError) as error:
        return BridgeEmbeddingRefusal(f"exact proof unavailable: {error}")
