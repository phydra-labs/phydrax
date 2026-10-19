#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Evaluate planar predicates and projective material reference maps."""

from fractions import Fraction

from numpy import object_
from numpy.typing import NDArray


type ExactPoint = tuple[Fraction, Fraction]
type ExactTriangle = tuple[ExactPoint, ExactPoint, ExactPoint]
type PointCoordinates = ExactPoint | NDArray[object_]


def orientation(
    first: PointCoordinates, second: PointCoordinates, third: PointCoordinates
) -> Fraction:
    """Return the signed double area without rounding chart coordinates."""
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def contains(
    vertices: ExactTriangle | NDArray[object_],
    point: PointCoordinates,
    *,
    degenerate_message: str,
) -> bool:
    """Include triangle edges and refuse a zero-area material reference."""
    direction = orientation(*vertices)
    if direction == 0:
        raise ValueError(degenerate_message)
    return all(
        orientation(vertices[index], vertices[(index + 1) % 3], point) * direction >= 0
        for index in range(3)
    )


def area(vertices: tuple[ExactPoint, ...]) -> Fraction:
    """Return signed polygon area using rational coordinate arithmetic."""
    return (
        sum(
            (
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(
                    vertices, vertices[1:] + vertices[:1], strict=True
                )
            ),
            Fraction(0),
        )
        / 2
    )


def projective_point(
    coefficients: tuple[tuple[Fraction, ...], ...],
    point: ExactPoint,
    *,
    invalid_message: str,
) -> ExactPoint:
    """Apply a rational chart map inside its positive barycentric cone."""
    if len(coefficients) != 3 or any(len(row) != 3 for row in coefficients):
        raise ValueError(
            f"chart_geometry.projective_point.coefficients must have three rows of three coefficients; got row lengths {tuple(len(row) for row in coefficients)!r}"
        )
    homogeneous = tuple(
        row[0] + row[1] * point[0] + row[2] * point[1] for row in coefficients
    )
    denominator = sum(homogeneous, Fraction(0))
    if denominator <= 0 or any(value < 0 for value in homogeneous):
        raise ValueError(invalid_message)
    return homogeneous[1] / denominator, homogeneous[2] / denominator


def merge_oriented_boundary_chain(
    edges: dict[tuple[ExactPoint, ExactPoint], int],
) -> dict[tuple[ExactPoint, ExactPoint], int]:
    """Merge consecutive collinear edges with equal oriented multiplicity.

    Rational collinearity and a positive direction dot product preserve the
    oriented chain without identifying a turn, reversed segment, or branch.
    """
    outgoing: dict[ExactPoint, dict[ExactPoint, int]] = {}
    incoming: dict[ExactPoint, dict[ExactPoint, int]] = {}
    for (first, last), count in edges.items():
        outgoing.setdefault(first, {})[last] = count
        incoming.setdefault(last, {})[first] = count
    pending = list(set(outgoing) | set(incoming))
    while pending:
        middle = pending.pop()
        before = incoming.get(middle, {})
        after = outgoing.get(middle, {})
        if len(before) != 1 or len(after) != 1:
            continue
        first, first_count = next(iter(before.items()))
        last, last_count = next(iter(after.items()))
        if first_count != last_count or orientation(first, middle, last) != 0:
            continue
        first_direction = middle[0] - first[0], middle[1] - first[1]
        last_direction = last[0] - middle[0], last[1] - middle[1]
        if (
            first_direction[0] * last_direction[0]
            + first_direction[1] * last_direction[1]
            <= 0
        ):
            continue
        del outgoing[first][middle]
        del incoming[last][middle]
        del outgoing[middle]
        del incoming[middle]
        combined = outgoing[first].get(last, 0) + first_count
        outgoing[first][last] = combined
        incoming[last][first] = combined
        pending.extend((first, last))
    return {
        (first, last): count
        for first, destinations in outgoing.items()
        for last, count in destinations.items()
    }
