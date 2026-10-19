#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference cells, subdivision, and field-valued embedding image bounds."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from phydrax.discretization._reference_subdivision import (
    prism_child_maps,
    simplex_child_maps,
)

from .._host_table_cache import host_table_cache, logical_table_bytes
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
)
from ..discretization._coordinate_enclosure import (
    affine_arguments,
    BernsteinDomain,
    constant,
    CoordinateSourceBank,
    ExactScalar,
    Expression,
    expression_add,
    expression_bounds as polynomial_bounds,
    expression_scale,
    expression_sum,
    ExpressionComposition as ArgumentComposition,
)
from ..discretization._exact_algebraic_geometry import (
    RadicalElement,
    reserve_algebraic_work,
)
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked
from ._rational import fraction_endpoint, rational_vector


# Named cells plus simplex/tensor dimensions 1 through 6 retain 13568 logical
# numerator/denominator bytes; eight hot sets fit the one-MiB minimum.
_VERTEX_CACHE_BYTES = 1048576


@dataclass(frozen=True, init=False)
class Cell:
    coordinates: tuple[Expression, ...]
    domain: BernsteinDomain
    kind: str
    dimension: int
    vertices: tuple[int, ...]
    reference_vertices: tuple[tuple[Fraction, ...], ...]
    # Images of ``reference_vertices`` for a map multi-affine in its box axes.
    corner_images: CoordinateSourceBank | None = None

    def __init__(
        self,
        coordinates: tuple[Expression, ...],
        domain: BernsteinDomain,
        kind: str,
        dimension: int,
        vertices: tuple[int, ...],
        reference_vertices: tuple[tuple[Fraction, ...], ...],
        corner_images: CoordinateSourceBank | None = None,
    ) -> None:
        if not isinstance(reference_vertices, tuple) or any(
            not isinstance(row, tuple)
            or any(not isinstance(value, Fraction) for value in row)
            for row in reference_vertices
        ):
            raise TypeError(
                "Cell.reference_vertices must contain immutable rational coordinate rows; got another reference bank"
            )
        if corner_images is not None:
            if not isinstance(corner_images, tuple) or any(
                not isinstance(row, tuple)
                or any(not isinstance(value, (Fraction, RadicalElement)) for value in row)
                for row in corner_images
            ):
                raise TypeError(
                    "Cell.corner_images must contain immutable rational or multiquadratic rows; got another physical bank"
                )
            if len(corner_images) != len(reference_vertices) or any(
                len(row) != len(coordinates) for row in corner_images
            ):
                raise ValueError(
                    "Cell.corner_images must align reference corners and ambient coordinates; got incompatible dimensions"
                )
        for name, value in (
            ("coordinates", coordinates),
            ("domain", domain),
            ("kind", kind),
            ("dimension", dimension),
            ("vertices", vertices),
            ("reference_vertices", reference_vertices),
            ("corner_images", corner_images),
        ):
            object.__setattr__(self, name, value)

    @checked
    def composed(
        self, composition: ArgumentComposition, /, *, budget: CoordinateEnclosureBudget
    ) -> Cell:
        """Return a chart-composed cell with its corner-image cache invalidated."""
        return replace(
            self,
            coordinates=tuple(
                composition(value, budget=budget) for value in self.coordinates
            ),
            corner_images=None,
        )


type Chart = tuple[np.ndarray, np.ndarray]
type CellPairPiece = tuple[Chart, Chart, int]


class BisectionWorklist:
    """Own FIFO chart pairs and the live-storage bound of their bisection proof.

    Each entry retains four NumPy owners, two chart tuples, a queue tuple and
    depth. The queue base and one in-flight entry remain charged until release.
    """

    _QUEUE_BYTES = 1024
    _ENTRY_BASE_BYTES = 1024
    _ENTRY_AXIS_BYTES = 32

    def __init__(
        self,
        first_dimension: int,
        second_dimension: int,
        live: CoordinateLiveStorage,
        /,
        *,
        first: Chart | None = None,
        second: Chart | None = None,
    ) -> None:
        self._live = live
        dimension = max(first_dimension, second_dimension)
        self._entry_bytes = self._ENTRY_BASE_BYTES + self._ENTRY_AXIS_BYTES * dimension**2
        live.set_bound(self._QUEUE_BYTES + 2 * self._entry_bytes)
        if first is None:
            first = (
                np.zeros((first_dimension,), dtype=np.float64),
                np.eye(first_dimension, dtype=np.float64),
            )
        if second is None:
            second = (
                np.zeros((second_dimension,), dtype=np.float64),
                np.eye(second_dimension, dtype=np.float64),
            )
        self._active: deque[CellPairPiece] = deque(((first, second, 0),))

    def __bool__(self) -> bool:
        return bool(self._active)

    def __len__(self) -> int:
        return len(self._active)

    def pop(self) -> CellPairPiece:
        return self._active.popleft()

    def reserve_split(self, maximum_children: int, /) -> None:
        """Admit the in-flight parent and child owners before child construction."""
        self._live.set_bound(
            self._QUEUE_BYTES
            + self._entry_bytes * (len(self._active) + 2 + maximum_children)
        )

    def extend(self, pieces: tuple[CellPairPiece, ...], /) -> None:
        self._active.extend(pieces)

    def bound_children(self, count: int, /) -> None:
        """Charge already prepared children before they enter the queue."""
        self._live.set_bound(
            self._QUEUE_BYTES + self._entry_bytes * (len(self._active) + count + 1)
        )

    def release_piece(self) -> None:
        self._live.set_bound(
            self._QUEUE_BYTES + self._entry_bytes * (len(self._active) + 1)
        )


def domain(kind: str) -> BernsteinDomain:
    return (
        "simplex"
        if kind in ("triangle", "tetrahedron")
        else "prism"
        if kind == "prism"
        else "box"
    )


@host_table_cache(
    max_entries=256,
    max_bytes=_VERTEX_CACHE_BYTES,
    weight=lambda value: sum(
        logical_table_bytes((component.numerator, component.denominator))
        for point in value
        for component in point
    ),
)
def vertices(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    """Retain Fraction reference vertices with byte-admitted rational weight."""
    topology = reference_cell_topology(kind)
    if kind == "pyramid":
        return tuple(
            tuple(Fraction(value) for value in point)
            for point in (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            )
        )
    return tuple(rational_vector(point) for point in np.asarray(topology.vertices))


def identity_reference_map(origin: np.ndarray, matrix: np.ndarray) -> bool:
    return (
        not np.any(origin)
        and matrix.shape == (origin.size, origin.size)
        and all(
            value == int(i == j)
            for i, row in enumerate(matrix)
            for j, value in enumerate(row)
        )
    )


@checked
def bounds(
    cell: Cell,
    origin: np.ndarray,
    matrix: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    identity = identity_reference_map(origin, matrix)
    if identity and cell.corner_images is not None:
        # Degree-one Bernstein controls of a multi-affine map are its corner images.

        budget.reserve(len(cell.corner_images) * len(cell.coordinates))
        columns = tuple(zip(*cell.corner_images, strict=True))
        return (
            np.asarray(
                [
                    fraction_endpoint(min(column), upper=False, saturate=True)
                    for column in columns
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    fraction_endpoint(max(column), upper=True, saturate=True)
                    for column in columns
                ],
                dtype=np.float64,
            ),
        )
    composition = (
        None
        if identity
        else ArgumentComposition(
            affine_arguments(origin, matrix, budget=budget), budget=budget
        )
    )
    intervals = np.asarray(
        [
            polynomial_bounds(
                value if composition is None else composition(value, budget=budget),
                cell.domain,
                cell.dimension,
                budget=budget,
            )
            for value in cell.coordinates
        ],
        dtype=np.float64,
    )
    return intervals[:, 0], intervals[:, 1]


@checked
def children(
    cell: Cell, origin: np.ndarray, matrix: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:

    if cell.domain == "simplex":
        origins, matrices = simplex_child_maps(cell.dimension)
    elif cell.domain == "prism":
        origins, matrices = prism_child_maps()
    else:
        # Binary splitting bounds candidate growth and keeps the work budget
        # independent of the ambient dimension.
        axis = int(np.argmax(np.linalg.norm(matrix, axis=0)))
        first = np.eye(cell.dimension, dtype=np.float64)
        first[axis, axis] = 0.5
        second = np.zeros((cell.dimension,), dtype=np.float64)
        second[axis] = 0.5
        origins = np.stack((np.zeros_like(second), second))
        matrices = np.stack((first, first))
    return tuple(
        (origin + matrix @ start, matrix @ child)
        for start, child in zip(origins, matrices, strict=True)
    )


def inside_reference(point: tuple[Fraction, ...], kind: str) -> bool:
    if any(value < 0 or value > 1 for value in point):
        return False
    if kind in ("triangle", "tetrahedron"):
        return sum(point) <= 1
    if kind == "prism":
        return point[0] + point[1] <= 1
    if kind == "pyramid":
        x, y, z = point
        return z / 2 <= x <= 1 - z / 2 and z / 2 <= y <= 1 - z / 2
    return True


@checked
def reference_constraints(
    coordinates: tuple[Expression, ...],
    kind: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    """Return convex source inequalities with their rational denominators retained."""
    if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
        return (
            *coordinates,
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(
                    expression_sum(coordinates, budget=budget), -1, budget=budget
                ),
                budget=budget,
            ),
        )
    if kind == "prism":
        return (
            coordinates[0],
            coordinates[1],
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(
                    expression_add(coordinates[0], coordinates[1], budget=budget),
                    -1,
                    budget=budget,
                ),
                budget=budget,
            ),
            coordinates[2],
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(coordinates[2], -1, budget=budget),
                budget=budget,
            ),
        )
    if kind == "pyramid":
        x, y, z = coordinates
        return (
            z,
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(z, -1, budget=budget),
                budget=budget,
            ),
            expression_add(
                x, expression_scale(z, Fraction(-1, 2), budget=budget), budget=budget
            ),
            expression_add(
                y, expression_scale(z, Fraction(-1, 2), budget=budget), budget=budget
            ),
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(
                    expression_add(
                        x,
                        expression_scale(z, Fraction(1, 2), budget=budget),
                        budget=budget,
                    ),
                    -1,
                    budget=budget,
                ),
                budget=budget,
            ),
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(
                    expression_add(
                        y,
                        expression_scale(z, Fraction(1, 2), budget=budget),
                        budget=budget,
                    ),
                    -1,
                    budget=budget,
                ),
                budget=budget,
            ),
        )
    return (
        *coordinates,
        *(
            expression_add(
                constant(1, dimension, budget=budget),
                expression_scale(value, -1, budget=budget),
                budget=budget,
            )
            for value in coordinates
        ),
    )


def _turn(
    first: tuple[ExactScalar, ...],
    second: tuple[ExactScalar, ...],
    third: tuple[ExactScalar, ...],
) -> ExactScalar:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


@checked
def convex_triangle_union(
    first_shared: tuple[ExactScalar, ...],
    second_shared: tuple[ExactScalar, ...],
    first_outer: tuple[ExactScalar, ...],
    second_outer: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Return whether two field-valued physical triangles form a convex quadrilateral."""
    with budget.temporary_scope():
        points = (first_shared, second_shared, first_outer, second_outer)
        if any(isinstance(value, RadicalElement) for point in points for value in point):
            reserve_algebraic_work(points, 64, 64, 4, budget=budget)
        side = _turn(first_shared, second_shared, first_outer)
        if not side or side * _turn(first_shared, second_shared, second_outer) >= 0:
            return False
        boundary = (first_outer, first_shared, second_outer, second_shared)
        return not any(
            side * _turn(boundary[i], boundary[(i + 1) % 4], boundary[(i + 2) % 4]) < 0
            for i in range(4)
        )
