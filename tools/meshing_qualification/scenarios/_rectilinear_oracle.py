"""Exact authored signed-box references, independent of Boolean classification.

References are analytically clipped, signed boxes supplied by each fixture, not
computed from production output. Integration and the boundary operator are linear.
Plane-grid winding is exhaustive: every output edge must lie on a grid line, so
each open grid rectangle has constant integer winding.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from phydrax.geometry.brep import BRepAffinePlacement, BRepModel
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.linalg._exact import determinant_exact


type Point = tuple[Fraction, Fraction, Fraction]
type Box = tuple[Point, Point]
type BoundaryKey = tuple[int, Fraction, Fraction, Fraction, Fraction, Fraction]
type SignedBoxes = tuple[tuple[int, Box], ...]


@dataclass(frozen=True)
class RectilinearReference:
    volume: Fraction
    first_moments: tuple[Fraction, ...]
    second_moments: tuple[Fraction, ...]
    area: Fraction
    boundary: dict[BoundaryKey, int]


def _source_loops(model: BRepModel) -> tuple[tuple[int, tuple[Point, ...]], ...]:
    geometry = model.geometry
    occurrences = tuple(geometry.iter_occurrences())
    if tuple(sorted(item.solid_index for item in occurrences)) != tuple(
        range(model.topology.num_solids)
    ) or any(item.placement != BRepAffinePlacement.identity() for item in occurrences):
        raise ValueError(
            "The rectilinear oracle requires each unplaced source solid exactly once."
        )
    result: list[tuple[int, tuple[Point, ...]]] = []
    for faces, orientations in zip(
        model.topology.solid_faces, model.topology.solid_face_orientations, strict=True
    ):
        for face, orientation in zip(faces, orientations, strict=True):
            if not isinstance(model.patches[face], PlanePatch):
                raise ValueError("The rectilinear oracle requires exact planar faces.")
            sign = orientation * int(model.orientation[face])
            for loop in geometry.face_loops[face]:
                points: list[Point] = []
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    if not isinstance(
                        geometry.curves[geometry.edge_curves[edge]], LineCurve
                    ):
                        raise ValueError(
                            "The rectilinear oracle requires straight source edges."
                        )
                    vertex = geometry.edge_vertices[edge][
                        0 if geometry.coedge_senses[coedge] == 1 else 1
                    ]
                    lower, upper = geometry.source_vertex_point_bounds(vertex)
                    if lower != upper:
                        raise ValueError(
                            "The rectilinear oracle requires singleton rational source vertices."
                        )
                    points.append(lower)
                for first, last in zip(points, points[1:] + points[:1], strict=True):
                    if sum(a != b for a, b in zip(first, last, strict=True)) != 1:
                        raise ValueError(
                            "Every source edge must be a nonzero axis-aligned segment."
                        )
                result.append((sign, tuple(points)))
    return tuple(result)


def _boundary_key(
    axis: int,
    plane: Fraction,
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
) -> BoundaryKey:
    return axis, plane, first[0], first[1], second[0], second[1]


def _box_boundary(
    boxes: SignedBoxes, grids: tuple[tuple[Fraction, ...], ...]
) -> dict[BoundaryKey, int]:
    boundary: dict[BoundaryKey, int] = {}
    for weight, (low, high) in boxes:
        for axis in range(3):
            first_axis, second_axis = (axis + 1) % 3, (axis + 2) % 3
            first_grid, second_grid = grids[first_axis], grids[second_axis]
            for first, second in product(
                zip(first_grid[:-1], first_grid[1:], strict=True),
                zip(second_grid[:-1], second_grid[1:], strict=True),
            ):
                if not (
                    low[first_axis] <= first[0] < first[1] <= high[first_axis]
                    and low[second_axis] <= second[0] < second[1] <= high[second_axis]
                ):
                    continue
                for plane, orientation in ((low[axis], -1), (high[axis], 1)):
                    key = _boundary_key(axis, plane, first, second)
                    boundary[key] = boundary.get(key, 0) + weight * orientation
    return {key: value for key, value in boundary.items() if value}


def _reference(
    boxes: SignedBoxes, grids: tuple[tuple[Fraction, ...], ...]
) -> RectilinearReference:
    volume = Fraction(0)
    first = [Fraction(0)] * 3
    second = [Fraction(0)] * 9
    for weight, (low, high) in boxes:
        if weight not in (-1, 1) or any(a >= b for a, b in zip(low, high, strict=True)):
            raise ValueError(
                "Authored reference boxes require signed unit weights and positive extents."
            )
        lengths = tuple(b - a for a, b in zip(low, high, strict=True))
        measure = weight * lengths[0] * lengths[1] * lengths[2]
        means = tuple((a + b) / 2 for a, b in zip(low, high, strict=True))
        volume += measure
        for axis in range(3):
            first[axis] += measure * means[axis]
            for other in range(3):
                a, b = low[axis], high[axis]
                value = (
                    (a * a + a * b + b * b) / 3
                    if axis == other
                    else means[axis] * means[other]
                )
                second[3 * axis + other] += measure * value
    boundary = _box_boundary(boxes, grids)
    if any(abs(value) != 1 for value in boundary.values()):
        raise ValueError(
            "The authored signed-box reference has overlapping boundary multiplicity."
        )
    area = sum(((key[3] - key[2]) * (key[5] - key[4]) for key in boundary), Fraction(0))
    return RectilinearReference(volume, tuple(first), tuple(second), area, boundary)


def _winding(points: tuple[Point, ...], axis: int, x: Fraction, y: Fraction) -> int:
    first_axis, second_axis = (axis + 1) % 3, (axis + 2) % 3
    winding = 0
    for first, last in zip(points, points[1:] + points[:1], strict=True):
        a, b = first[first_axis], first[second_axis]
        c, d = last[first_axis], last[second_axis]
        side = (c - a) * (y - b) - (x - a) * (d - b)
        if b <= y < d and side > 0:
            winding += 1
        elif d <= y < b and side < 0:
            winding -= 1
    return winding


def _actual_boundary(
    loops: tuple[tuple[int, tuple[Point, ...]], ...],
    grids: tuple[tuple[Fraction, ...], ...],
) -> dict[BoundaryKey, int]:
    boundary: dict[BoundaryKey, int] = {}
    for sign, points in loops:
        constants = tuple(
            axis
            for axis in range(3)
            if all(point[axis] == points[0][axis] for point in points)
        )
        if len(constants) != 1:
            raise ValueError(
                "Every source loop must lie in one nondegenerate coordinate plane."
            )
        axis = constants[0]
        first_axis, second_axis = (axis + 1) % 3, (axis + 2) % 3
        first_grid, second_grid = grids[first_axis], grids[second_axis]
        for first, second in product(
            zip(first_grid[:-1], first_grid[1:], strict=True),
            zip(second_grid[:-1], second_grid[1:], strict=True),
        ):
            winding = sign * _winding(
                points, axis, (first[0] + first[1]) / 2, (second[0] + second[1]) / 2
            )
            if winding:
                key = _boundary_key(axis, points[0][axis], first, second)
                boundary[key] = boundary.get(key, 0) + winding
    return {key: value for key, value in boundary.items() if value}


def _boundary_moments(
    loops: tuple[tuple[int, tuple[Point, ...]], ...],
) -> tuple[Fraction, tuple[Fraction, ...], tuple[Fraction, ...]]:
    volume = Fraction(0)
    first = [Fraction(0)] * 3
    second = [Fraction(0)] * 9
    for sign, points in loops:
        for index in range(1, len(points) - 1):
            triangle = points[0], points[index], points[index + 1]
            contribution = sign * determinant_exact(triangle) / 6
            sums = tuple(
                sum((point[axis] for point in triangle), Fraction(0)) for axis in range(3)
            )
            volume += contribution
            for axis in range(3):
                first[axis] += contribution * sums[axis] / 4
                for other in range(3):
                    diagonal = sum(
                        (point[axis] * point[other] for point in triangle), Fraction(0)
                    )
                    second[3 * axis + other] += (
                        contribution * (sums[axis] * sums[other] + diagonal) / 20
                    )
    return volume, tuple(first), tuple(second)


def validate_rectilinear_result(
    model: BRepModel, boxes: SignedBoxes
) -> RectilinearReference:
    """Compare exact oriented coverage and moments with an authored signed-box chain."""
    model.require_current()
    loops = _source_loops(model)
    grids = tuple(
        tuple(
            sorted(
                {point[axis] for _, box in boxes for point in box}
                | {point[axis] for _, points in loops for point in points}
            )
        )
        for axis in range(3)
    )
    expected = _reference(boxes, grids)
    if _actual_boundary(loops, grids) != expected.boundary:
        raise ValueError(
            "The source boundary differs from the authored set expression; locality or orientation failed."
        )
    if _boundary_moments(loops) != (
        expected.volume,
        expected.first_moments,
        expected.second_moments,
    ):
        raise ValueError("The independent boundary and volume-integral moments disagree.")
    return expected
