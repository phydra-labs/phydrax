#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contact exclusion of mapped tetrahedra through their declared shared stratum.

Two polynomial simplices ``A`` and ``B`` in space (triangles or tetrahedra)
sharing the vertex set ``S`` (none, one
vertex, or an edge) with equal exact restrictions on ``S`` meet only in the
image of ``S`` when every pair of reference sub-simplices ``(a, b)`` of an
adaptive longest-edge bisection of ``A`` and ``B`` is excluded by one of two
exact certificates on the Bernstein controls of the pieces:

* hull separation: a plane strictly separates the physical control points of
  ``a`` and ``b``, which enclose the piece images (convex-hull property);
* stratum cone: ``R`` vanishes on the image of ``S`` (the point itself, or two
  planes or the plane and conic residual through an edge trace). The Bernstein
  controls of ``R o a`` vanish exactly on the face of ``a`` spanned by its
  vertices on ``S`` and otherwise lie strictly on the positive side of one
  direction ``eta``; those of ``R o b`` lie strictly on its negative side. Then
  ``eta . R`` is positive on ``a`` and negative on ``b`` off those faces, so a
  common point is the image of a point of ``S`` in both cells; both images then
  lie in the declared shared stratum image.

Pieces are split along their physically longest edge by exact de Casteljau
subdivision until every pair is excluded; a piece below binary64 resolution
leaves the pair unresolved. Face-sharing cells use `glued_face_monotone`.
Floats only propose planes and directions; every sign is decided exactly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
from typing import cast

import numpy as np

from ..discretization._coordinate_enclosure import (
    bernstein_coefficients,
    bernstein_degree,
    BernsteinDomain,
    compose_expression,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..linalg._exact import matmul_exact, symmetric_product_minors_exact
from ._embedding_cells import EmbeddingCell
from ._mapped_embedding import (
    monotone_preconditioner,
    subdivided_jacobian_controls,
    SUBDIVISION_LEVELS,
)


_Index = tuple[int, ...]
_Vector = tuple[ExactScalar, ...]
_Controls = dict[_Index, _Vector]
_Barycentric = tuple[Fraction, ...]
_SIMPLEX_KINDS = ("tetrahedron", "simplex:3")
_PIECE_KINDS = ("triangle", "simplex:2", *_SIMPLEX_KINDS)
# Pieces smaller than this fraction of the cell scale are below the binary64
# resolution of the float guidance; their pair stays unresolved.
_RESOLUTION = 2.0**-44
# This limits only retained subdivision results, never the proof traversal.
# Evicted exact children are deterministically recomputed if needed again.
_CHILD_CACHE_ENTRIES = 256


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return int(value > 0) - int(value < 0)


@dataclass(frozen=True)
class _Piece:
    """Reference sub-cell with exact physical and residual Bernstein controls.

    A piece is a product of barycentric groups: one simplex group for a
    triangle or tetrahedron, a triangle group and a line group for a prism.
    Control indices concatenate the group multi-indices; ``vertices[slot]`` is
    the reference point of that group vertex and ``on_stratum[slot]`` whether it
    lies on the shared stratum (a product of group faces).
    """

    groups: tuple[int, ...]
    vertices: tuple[_Barycentric, ...]
    on_stratum: tuple[bool, ...]
    geometry: _Controls
    geometry_degree: tuple[int, ...]
    residual: _Controls
    residual_degree: tuple[int, ...]
    points: np.ndarray
    corners: dict[tuple[int, ...], np.ndarray]

    @property
    def diameter(self) -> float:
        corners = np.asarray(list(self.corners.values()))
        differences = corners[:, None, :] - corners[None, :, :]
        return float(np.sqrt(np.max(np.sum(differences * differences, axis=-1))))


@dataclass
class StratumSubdivisionCache:
    """Reusable exact pieces and Jacobian controls for a bounded pair workset.

    Roots are keyed by cell and stratum, children additionally by reference
    vertices, and Jacobian controls by cell and uniform subdivision level.
    The child LRU bounds retained subdivision trees without limiting proof work.
    """

    roots: dict[tuple[int, tuple[int, ...]], _Piece | None] = field(default_factory=dict)
    geometry: dict[int, tuple[EmbeddingCell, tuple[int, ...], _Controls]] = field(
        default_factory=dict
    )
    children: OrderedDict[
        tuple[int, tuple[int, ...], tuple[_Barycentric, ...]], tuple[_Piece, _Piece]
    ] = field(default_factory=OrderedDict)
    jacobian_controls: dict[tuple[int, int], tuple[_Matrix, ...] | None] = field(
        default_factory=dict
    )
    simplex_jacobians: dict[int, tuple[EmbeddingCell, tuple[_Matrix, ...] | None]] = (
        field(default_factory=dict)
    )
    reference_edge_maps: dict[
        tuple[tuple[tuple[Fraction, ...], ...], int],
        tuple[_Matrix, _Matrix | None],
    ] = field(default_factory=dict)
    corner_affine_inverses: dict[int, tuple[EmbeddingCell, _Matrix | None]] = field(
        default_factory=dict
    )
    fixed_glue_maps: dict[tuple[object, ...], _Matrix | None] = field(
        default_factory=dict
    )


def _compositions(total: int, parts: int, /) -> tuple[_Index, ...]:
    if parts == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total, -1, -1)
        for rest in _compositions(total - first, parts - 1)
    )


def _group_slots(groups: tuple[int, ...], /) -> tuple[tuple[int, ...], ...]:
    slots: list[tuple[int, ...]] = []
    start = 0
    for size in groups:
        slots.append(tuple(range(start, start + size)))
        start += size
    return tuple(slots)


def _control_indices(
    groups: tuple[int, ...], degree: tuple[int, ...], /
) -> tuple[_Index, ...]:
    """Concatenated group indices in `bernstein_coefficients` order.

    A simplex orders its compositions; a prism orders triangle compositions, then
    line powers ``l`` of the line parameter as the group index ``(L - l, l)``.
    """
    if len(groups) == 1:
        return _compositions(degree[0], groups[0])
    triangle, line = degree
    return tuple(
        (*beta, line - power, power)
        for beta in _compositions(triangle, 3)
        for power in range(line + 1)
    )


def _domain(groups: tuple[int, ...], /) -> BernsteinDomain:
    return "simplex" if len(groups) == 1 else "prism"


def _controls(
    polynomials: tuple[Expression, ...],
    groups: tuple[int, ...],
    degree: tuple[int, ...],
    dimension: int,
    /,
) -> _Controls | None:
    columns: list[tuple[ExactScalar, ...]] = []
    for polynomial in polynomials:
        if not isinstance(polynomial, dict):
            return None
        columns.append(
            bernstein_coefficients(polynomial, _domain(groups), dimension, degree=degree)
        )
    return {
        index: tuple(column[row] for column in columns)
        for row, index in enumerate(_control_indices(groups, degree))
    }


def _float_points(controls: _Controls, /) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in vector] for vector in controls.values()],
        dtype=np.float64,
    )


def _corner_points(
    controls: _Controls, groups: tuple[int, ...], degree: tuple[int, ...], /
) -> dict[tuple[int, ...], np.ndarray]:
    """Corner images keyed by their slot per group (one slot of every group)."""
    group_slots = _group_slots(groups)
    corners: dict[tuple[int, ...], np.ndarray] = {}
    for choice in product(*group_slots):
        index = tuple(
            degree[group] * int(slot in choice)
            for group, slots in enumerate(group_slots)
            for slot in slots
        )
        corners[choice] = np.asarray(
            [float(value) for value in controls[index]], dtype=np.float64
        )
    return corners


def _degree(
    polynomials: tuple[Expression, ...], groups: tuple[int, ...], dimension: int, /
) -> tuple[int, ...] | None:
    degree = [1] * len(groups)
    for polynomial in polynomials:
        if not isinstance(polynomial, dict):
            return None
        least = bernstein_degree(polynomial, _domain(groups), dimension)
        degree = [
            max(value, minimum) for value, minimum in zip(degree, least, strict=True)
        ]
    return tuple(degree)


def _reference_slots(
    cell: EmbeddingCell, /
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    """Piece groups and every corner's slot per group, or ``None``."""
    dimension = cell.dimension
    references = [
        tuple(Fraction(value) for value in point) for point in cell.reference_vertices
    ]
    if any(len(point) != dimension for point in references):
        return None

    def simplex_slot(values: tuple[Fraction, ...], /) -> int | None:
        if not any(values):
            return 0
        if sorted(values) == [0] * (len(values) - 1) + [1]:
            return values.index(Fraction(1)) + 1
        return None

    if cell.kind in _PIECE_KINDS and len(references) == dimension + 1:
        slots = [
            slot
            for slot in (simplex_slot(point) for point in references)
            if slot is not None
        ]
        if sorted(slots) != list(range(dimension + 1)):
            return None
        return (dimension + 1,), tuple((slot,) for slot in slots)
    if cell.kind == "prism" and dimension == 3 and len(references) == 6:
        corners: list[tuple[int, ...]] = []
        for point in references:
            triangle = simplex_slot(point[:2])
            if triangle is None or point[2] not in (0, 1):
                return None
            corners.append((triangle, 3 + int(point[2])))
        if sorted(corners) != sorted(product(range(3), (3, 4))):
            return None
        return (3, 2), tuple(corners)
    return None


def _root_piece(
    cell: EmbeddingCell,
    shared_ids: tuple[int, ...],
    residuals: tuple[Expression, ...],
    /,
    *,
    cache: StratumSubdivisionCache | None = None,
) -> _Piece | None:
    dimension = cell.dimension
    if (
        dimension not in (2, 3)
        or len(cell.coordinates) not in (2, 3)
        or len(cell.coordinates) < dimension
        or len(cell.vertex_ids) != len(cell.reference_vertices)
    ):
        return None
    layout = _reference_slots(cell)
    if layout is None:
        return None
    groups, corner_slots = layout
    slot_count = sum(groups)
    # The stratum must be a product of group faces: a corner lies on it exactly
    # when each of its slots does.
    on_stratum = [False] * slot_count
    corner_on = {
        slots: cell.vertex_ids[local] in shared_ids
        for local, slots in enumerate(corner_slots)
    }
    for slots, on in corner_on.items():
        if on:
            for slot in slots:
                on_stratum[slot] = True
    if any(
        on != all(on_stratum[slot] for slot in slots) for slots, on in corner_on.items()
    ):
        return None
    prepared = None if cache is None else cache.geometry.get(id(cell))
    if prepared is not None and prepared[0] is cell:
        _, geometry_degree, geometry = prepared
    else:
        geometry_degree = _degree(cell.coordinates, groups, dimension)
        if geometry_degree is None:
            return None
        geometry = _controls(cell.coordinates, groups, geometry_degree, dimension)
        if geometry is None:
            return None
        if cache is not None:
            cache.geometry[id(cell)] = (cell, geometry_degree, geometry)
    # An affine physical residual commutes with conversion of the geometry
    # to Bernstein form. This also covers the two planes of a straight edge,
    # without repeatedly composing the cell's polynomial map.
    ambient = len(cell.coordinates)
    zero = (0,) * ambient
    axes = tuple(
        tuple(int(index == axis) for index in range(ambient)) for axis in range(ambient)
    )
    affine_rows: list[tuple[ExactScalar, ...]] = []
    for expression in residuals:
        if not isinstance(expression, dict) or any(
            exponent != zero and exponent not in axes for exponent in expression
        ):
            break
        affine_rows.append(
            (expression.get(zero, Fraction(0)),)
            + tuple(expression.get(axis, Fraction(0)) for axis in axes)
        )
    offsets = (
        tuple(row[0] for row in affine_rows)
        if len(affine_rows) == ambient
        and all(
            row[1:] == tuple(Fraction(int(index == axis)) for index in range(ambient))
            for axis, row in enumerate(affine_rows)
        )
        else None
    )
    if offsets is not None:
        residual_degree = geometry_degree
        residual = {
            index: tuple(
                value + offset for value, offset in zip(vector, offsets, strict=True)
            )
            for index, vector in geometry.items()
        }
    elif residuals and len(affine_rows) == len(residuals):
        residual_degree = geometry_degree
        residual = {
            index: tuple(
                row[0]
                + sum(
                    (
                        coefficient * value
                        for coefficient, value in zip(row[1:], vector, strict=True)
                        if coefficient
                    ),
                    Fraction(0),
                )
                for row in affine_rows
            )
            for index, vector in geometry.items()
        }
    else:
        composed = tuple(
            compose_expression(residual, cell.coordinates) for residual in residuals
        )
        residual_degree = (
            _degree(composed, groups, dimension) if composed else (1,) * len(groups)
        )
        if residual_degree is None:
            return None
        residual = (
            _controls(composed, groups, residual_degree, dimension) if composed else {}
        )
        if residual is None:
            return None
    vertices = tuple(
        tuple(Fraction(int(slot == vertex)) for slot in slots)
        for slots in _group_slots(groups)
        for vertex in slots
    )
    return _Piece(
        groups,
        vertices,
        tuple(on_stratum),
        geometry,
        geometry_degree,
        residual,
        residual_degree,
        _float_points(geometry),
        _corner_points(geometry, groups, geometry_degree),
    )


def _bisect_controls(
    controls: _Controls, first: int, second: int, /
) -> tuple[_Controls, _Controls]:
    """Exact de Casteljau halves along edge ``(first, second)`` of one group.

    The left half keeps vertex ``first`` and moves ``second`` to the midpoint;
    the right half keeps ``second`` and moves ``first`` there.
    """
    left: _Controls = {}
    right: _Controls = {}
    lines: dict[_Index, list[_Index]] = {}
    for index in controls:
        rest = tuple(
            value for slot, value in enumerate(index) if slot not in (first, second)
        )
        lines.setdefault(rest, []).append(index)
    half = Fraction(1, 2)
    for indices in lines.values():
        total = indices[0][first] + indices[0][second]
        ordered = sorted(indices, key=lambda index: index[second])
        levels = [[controls[index] for index in ordered]]
        for _ in range(total):
            previous = levels[-1]
            levels.append(
                [
                    tuple(
                        (left_value + right_value) * half
                        for left_value, right_value in zip(
                            previous[row], previous[row + 1], strict=True
                        )
                    )
                    for row in range(len(previous) - 1)
                ]
            )
        for step, index in enumerate(ordered):
            left[index] = levels[step][0]
            right[index] = levels[total - step][step]
    return left, right


def _bisect(piece: _Piece, first: int, second: int, /) -> tuple[_Piece, _Piece]:
    midpoint = tuple(
        (left + right) / 2
        for left, right in zip(piece.vertices[first], piece.vertices[second], strict=True)
    )
    middle_on_stratum = piece.on_stratum[first] and piece.on_stratum[second]
    geometry = _bisect_controls(piece.geometry, first, second)
    residual = (
        _bisect_controls(piece.residual, first, second) if piece.residual else ({}, {})
    )
    children: list[_Piece] = []
    for moved, geometry_controls, residual_controls in (
        (second, geometry[0], residual[0]),
        (first, geometry[1], residual[1]),
    ):
        vertices = tuple(
            midpoint if vertex == moved else value
            for vertex, value in enumerate(piece.vertices)
        )
        on_stratum = tuple(
            middle_on_stratum if vertex == moved else value
            for vertex, value in enumerate(piece.on_stratum)
        )
        children.append(
            _Piece(
                piece.groups,
                vertices,
                on_stratum,
                geometry_controls,
                piece.geometry_degree,
                residual_controls,
                piece.residual_degree,
                _float_points(geometry_controls),
                _corner_points(geometry_controls, piece.groups, piece.geometry_degree),
            )
        )
    return children[0], children[1]


def _children(
    cache: StratumSubdivisionCache,
    key: tuple[int, tuple[int, ...]],
    piece: _Piece,
    /,
) -> tuple[_Piece, _Piece]:
    memo_key = (*key, piece.vertices)
    cached = cache.children.get(memo_key)
    if cached is not None:
        cache.children.move_to_end(memo_key)
        return cached
    corners = piece.corners

    def length(pair: tuple[int, int], /) -> float:
        # The longest physical image of this group edge over the other groups.
        return max(
            float(
                np.sum(
                    (
                        point
                        - corners[
                            tuple(pair[1] if slot == pair[0] else slot for slot in choice)
                        ]
                    )
                    ** 2
                )
            )
            for choice, point in corners.items()
            if pair[0] in choice
        )

    first, second = max(
        (
            (left, right)
            for slots in _group_slots(piece.groups)
            for left in slots
            for right in slots
            if left < right
        ),
        key=length,
    )
    result = _bisect(piece, first, second)
    cache.children[memo_key] = result
    if len(cache.children) > _CHILD_CACHE_ENTRIES:
        cache.children.popitem(last=False)
    return result


def _strict_direction(vectors: np.ndarray, /) -> np.ndarray | None:
    """A float direction with positive products against every row, if one is found.

    Rows are normalized first. The mean is tried, then Gilbert's least-norm
    iteration (stopping at the first strictly positive direction), then in two
    or three dimensions the sum of the extreme rays of the closed dual cone,
    which is interior to it whenever the open dual cone is nonempty. Rows have
    one to three columns (physical control differences, at most three stratum
    residuals); in one dimension the normalized mean is positive exactly when a
    direction exists, so the routine is complete for every caller.
    """
    if vectors.size == 0 or not np.all(np.isfinite(vectors)):
        return None
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        return None
    rows = vectors / norms[:, None]
    current = rows.mean(axis=0)
    for _ in range(512):
        dots = rows @ current
        row = int(np.argmin(dots))
        if float(dots[row]) > 0.0:
            return current
        step = rows[row] - current
        length = float(step @ step)
        if length == 0.0:
            break
        current = current + min(1.0, max(0.0, -float(current @ step) / length)) * step
        if not np.any(current):
            return None
    dimension = rows.shape[1]
    if dimension == 2:
        perpendicular = np.stack((-rows[:, 1], rows[:, 0]), axis=1)
        rays = np.concatenate((perpendicular, -perpendicular))
    elif dimension == 3:
        first, second = np.triu_indices(rows.shape[0], 1)
        crosses = np.cross(rows[first], rows[second])
        lengths = np.linalg.norm(crosses, axis=1)
        crosses = crosses[lengths > 1.0e-12] / lengths[lengths > 1.0e-12, None]
        rays = np.concatenate((crosses, -crosses))
    else:
        return None
    feasible = rays[np.min(rays @ rows.T, axis=1) >= -1.0e-12]
    if not feasible.size:
        return None
    direction = feasible.sum(axis=0)
    return direction if float(np.min(rows @ direction)) > 0.0 else None


def _integer_direction(direction: np.ndarray, /) -> tuple[int, ...] | None:
    scale = float(np.max(np.abs(direction)))
    if not np.isfinite(scale) or scale == 0.0:
        return None
    values = tuple(int(round(value / scale * 2.0**30)) for value in direction)
    return values if any(values) else None


def _dot(direction: tuple[int, ...], vector: _Vector, /) -> ExactScalar:
    total: ExactScalar = Fraction(0)
    for coefficient, value in zip(direction, vector, strict=True):
        if coefficient:
            total = total + coefficient * value
    return total


def _exactly_positive(direction: np.ndarray | None, vectors: list[_Vector], /) -> bool:
    if direction is None:
        return False
    normal = _integer_direction(direction)
    return normal is not None and all(
        _sign(_dot(normal, vector)) > 0 for vector in vectors
    )


def _hull_separated(first: _Piece, second: _Piece, /) -> bool:
    differences = (first.points[:, None, :] - second.points[None, :, :]).reshape(
        -1, first.points.shape[1]
    )
    direction = _strict_direction(differences)
    if direction is None:
        return False
    normal = _integer_direction(direction)
    if normal is None:
        return False
    first_values = [_dot(normal, vector) for vector in first.geometry.values()]
    second_values = [_dot(normal, vector) for vector in second.geometry.values()]
    lowest = first_values[0]
    for value in first_values[1:]:
        if _sign(value - lowest) < 0:
            lowest = value
    return all(_sign(lowest - value) > 0 for value in second_values)


def _cone_vectors(piece: _Piece, side: int, /) -> list[_Vector] | None:
    """Off-stratum residual controls oriented by ``side``; stratum controls must vanish."""
    vectors: list[_Vector] = []
    for index, vector in piece.residual.items():
        if all(piece.on_stratum[slot] for slot, value in enumerate(index) if value):
            if any(_sign(value) for value in vector):
                return None
            continue
        vectors.append(vector if side > 0 else tuple(-value for value in vector))
    return vectors


def _float_rows(vectors: list[_Vector], /) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in vector] for vector in vectors], dtype=np.float64
    ).reshape(len(vectors), -1)


def _cone_test(first: _Piece, second: _Piece, /) -> tuple[bool, int | None]:
    """Whether a stratum cone excludes the pair; else a piece whose own controls
    leave every open half-space (it must be split before any cone can apply)."""
    first_vectors = _cone_vectors(first, 1)
    second_vectors = _cone_vectors(second, -1)
    if first_vectors is None or second_vectors is None:
        return False, None
    vectors = first_vectors + second_vectors
    if vectors and _exactly_positive(_strict_direction(_float_rows(vectors)), vectors):
        return True, None
    for culprit, own in ((0, first_vectors), (1, second_vectors)):
        if own and _strict_direction(_float_rows(own)) is None:
            return False, culprit
    return False, None


def subdivided_stratum_separation(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    residuals: tuple[Expression, ...],
    cache: StratumSubdivisionCache,
    /,
) -> tuple[int, int] | None:
    """Exclude contact of two polynomial tetrahedra outside their shared stratum.

    ``residuals`` are physical-space polynomials vanishing on the shared stratum
    image (empty without a shared stratum); the caller has proved equal exact
    restrictions on ``shared_ids``. Returns the counts of piece pairs excluded by
    a stratum cone and by hull separation, or ``None`` when a piece reaches
    binary64 resolution first.
    """
    stratum = tuple(sorted(shared_ids))
    pieces: list[_Piece] = []
    for cell in (first, second):
        key = (cell.global_id, stratum)
        if key not in cache.roots:
            cache.roots[key] = _root_piece(cell, stratum, residuals, cache=cache)
        piece = cache.roots[key]
        if piece is None:
            return None
        pieces.append(piece)
    scale = max(
        1.0,
        float(np.max(np.abs(pieces[0].points))),
        float(np.max(np.abs(pieces[1].points))),
    )
    keys = ((first.global_id, stratum), (second.global_id, stratum))
    cones = hulls = 0
    pending = [(pieces[0], pieces[1])]
    while pending:
        pair = pending.pop()
        separated, culprit = _cone_test(*pair) if residuals else (False, None)
        if separated:
            cones += 1
            continue
        if _hull_separated(*pair):
            hulls += 1
            continue
        target = (
            culprit if culprit is not None else int(pair[1].diameter > pair[0].diameter)
        )
        if pair[target].diameter <= _RESOLUTION * scale:
            return None
        for child in _children(cache, keys[target], pair[target]):
            pending.append((child, pair[1]) if target == 0 else (pair[0], child))
    return cones, hulls


_Matrix = tuple[tuple[ExactScalar, ...], ...]


def _determinant(matrix: _Matrix, /) -> ExactScalar:
    if len(matrix) == 2:
        (a, b), (c, d) = matrix
        return a * d - b * c
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _inverse(matrix: _Matrix, /) -> _Matrix | None:
    determinant = _determinant(matrix)
    if not _sign(determinant):
        return None
    if len(matrix) == 2:
        (a, b), (c, d) = matrix
        adjugate: _Matrix = ((d, -b), (-c, a))
    else:
        (a, b, c), (d, e, f), (g, h, i) = matrix
        adjugate = (
            (e * i - f * h, c * h - b * i, b * f - c * e),
            (f * g - d * i, a * i - c * g, c * d - a * f),
            (d * h - e * g, b * g - a * h, a * e - b * d),
        )
    return tuple(tuple(value / determinant for value in row) for row in adjugate)


def _product(first: _Matrix, second: _Matrix, /) -> _Matrix:
    if all(
        isinstance(value, Fraction)
        for matrix in (first, second)
        for row in matrix
        for value in row
    ):
        return matmul_exact(
            cast(tuple[tuple[Fraction, ...], ...], first),
            cast(tuple[tuple[Fraction, ...], ...], second),
        )
    # Algebraic coefficients retain their exact field arithmetic; the integer
    # accumulator owns rational matrices only.
    size = len(first)
    return tuple(
        tuple(
            sum(
                (first[row][inner] * second[inner][column] for inner in range(size)),
                Fraction(0),
            )
            for column in range(size)
        )
        for row in range(size)
    )


def _positive_definite_symmetric_part(*factors: _Matrix) -> bool:
    """Sylvester's criterion without normalizing intermediate rational products."""
    if all(
        isinstance(value, Fraction | int)
        for matrix in factors
        for row in matrix
        for value in row
    ):
        rational = cast(tuple[tuple[tuple[Fraction | int, ...], ...], ...], factors)
        return all(value > 0 for value in symmetric_product_minors_exact(*rational))
    # Ordered algebraic extensions keep their original exact scalar path.
    matrix = factors[-1]
    for factor in reversed(factors[:-1]):
        matrix = _product(factor, matrix)
    size = len(matrix)
    symmetric = tuple(
        tuple((matrix[row][column] + matrix[column][row]) / 2 for column in range(size))
        for row in range(size)
    )
    if _sign(symmetric[0][0]) <= 0:
        return False
    leading = symmetric[0][0] * symmetric[1][1] - symmetric[0][1] * symmetric[1][0]
    if _sign(leading) <= 0:
        return False
    return size == 2 or _sign(_determinant(symmetric)) > 0


def _edge_matrix(points: tuple[tuple[ExactScalar, ...], ...], origin: int, /) -> _Matrix:
    """Columns ``points[k] - points[origin]`` over the other simplex vertices."""
    others = [index for index in range(len(points)) if index != origin]
    return tuple(
        tuple(points[column][row] - points[origin][row] for column in others)
        for row in range(len(points) - 1)
    )


def _jacobian_controls(cell: EmbeddingCell, /) -> tuple[_Matrix, ...] | None:
    """Bernstein controls of the reference Jacobian on the unit reference simplex."""
    dimension = cell.dimension
    entries: list[list[dict[_Index, ExactScalar]]] = []
    for expression in cell.coordinates:
        if not isinstance(expression, dict):
            return None
        row: list[dict[_Index, ExactScalar]] = []
        for axis in range(dimension):
            derivative: dict[_Index, ExactScalar] = {}
            for exponent, coefficient in expression.items():
                if exponent[axis]:
                    lowered = tuple(
                        value - int(index == axis) for index, value in enumerate(exponent)
                    )
                    derivative[lowered] = derivative.get(lowered, Fraction(0)) + (
                        coefficient * exponent[axis]
                    )
            row.append(derivative or {(0,) * dimension: Fraction(0)})
        entries.append(row)
    degree = max(
        1,
        max(
            bernstein_degree(entry, "simplex", dimension)[0]
            for row in entries
            for entry in row
        ),
    )
    columns = [
        [
            bernstein_coefficients(entry, "simplex", dimension, degree=(degree,))
            for entry in row
        ]
        for row in entries
    ]
    count = len(columns[0][0])
    return tuple(
        tuple(
            tuple(columns[row][column][index] for column in range(dimension))
            for row in range(dimension)
        )
        for index in range(count)
    )


def _cached_reference_edges(
    points: tuple[tuple[Fraction, ...], ...],
    origin: int,
    cache: StratumSubdivisionCache | None,
    /,
) -> tuple[_Matrix, _Matrix | None]:
    key = (points, origin)
    if cache is not None and key in cache.reference_edge_maps:
        return cache.reference_edge_maps[key]
    edges = _edge_matrix(points, origin)
    prepared = (edges, _inverse(edges))
    if cache is not None:
        cache.reference_edge_maps[key] = prepared
    return prepared


def _cached_corner_metric(
    cell: EmbeddingCell,
    origin: int,
    inverse: _Matrix,
    cache: StratumSubdivisionCache | None,
    /,
) -> _Matrix | None:
    entry = None if cache is None else cache.corner_affine_inverses.get(id(cell))
    if entry is not None and entry[0] is cell:
        return entry[1]
    metric = _inverse(_product(_edge_matrix(cell.corner_images, origin), inverse))
    if cache is not None:
        cache.corner_affine_inverses[id(cell)] = (cell, metric)
    return metric


def _cached_simplex_jacobians(
    cell: EmbeddingCell, cache: StratumSubdivisionCache | None, /
) -> tuple[_Matrix, ...] | None:
    if cache is None:
        return _jacobian_controls(cell)
    entry = cache.simplex_jacobians.get(id(cell))
    if entry is None or entry[0] is not cell:
        entry = (cell, _jacobian_controls(cell))
        cache.simplex_jacobians[id(cell)] = entry
    return entry[1]


def _fixed_glued_monotone(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared: set[int],
    cache: StratumSubdivisionCache,
    /,
) -> bool:
    """Try the glued-map theorem on a fixed exact convex reference domain."""
    dimension = first.dimension
    first_opposite = next(
        index
        for index, identifier in enumerate(first.vertex_ids)
        if identifier not in shared
    )
    second_opposite = next(
        index
        for index, identifier in enumerate(second.vertex_ids)
        if identifier not in shared
    )
    first_reference = tuple(
        tuple(Fraction(value) for value in point) for point in first.reference_vertices
    )
    second_reference = tuple(
        tuple(Fraction(value) for value in point) for point in second.reference_vertices
    )
    first_edges, first_inverse = _cached_reference_edges(
        first_reference, first_opposite, cache
    )
    _, second_inverse = _cached_reference_edges(second_reference, second_opposite, cache)
    if first_inverse is None or second_inverse is None:
        return False
    metric = _cached_corner_metric(first, first_opposite, first_inverse, cache)
    if metric is None:
        return False
    mapping = tuple(
        first.vertex_ids.index(identifier) if identifier in shared else -1
        for identifier in second.vertex_ids
    )
    key = (first_reference, second_reference, first_opposite, second_opposite, mapping)
    if key not in cache.fixed_glue_maps:
        # All weights are positive and sum to two. The segment joining this
        # opposite vertex to the first opposite vertex crosses the interior
        # of the common face, so their glued union is convex.
        weights = (
            (Fraction(1), Fraction(1))
            if dimension == 2
            else (Fraction(1), Fraction(1, 2), Fraction(1, 2))
        )
        displacement = _apply(first_edges, weights)
        glued = tuple(
            first_reference[first_opposite][axis] + displacement[axis]
            for axis in range(dimension)
        )
        images = tuple(
            tuple(
                first_reference[mapping[column]][row] - glued[row]
                for column in range(dimension + 1)
                if column != second_opposite
            )
            for row in range(dimension)
        )
        cache.fixed_glue_maps[key] = _inverse(_product(images, second_inverse))
    glue = cache.fixed_glue_maps[key]
    if glue is None:
        return False
    first_controls = _cached_simplex_jacobians(first, cache)
    second_controls = _cached_simplex_jacobians(second, cache)
    if first_controls is None or second_controls is None:
        return False
    return all(
        _positive_definite_symmetric_part(metric, control) for control in first_controls
    ) and all(
        _positive_definite_symmetric_part(metric, control, glue)
        for control in second_controls
    )


def _glued_monotone(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared: set[int],
    cache: StratumSubdivisionCache | None,
    /,
) -> bool:
    dimension = first.dimension
    first_opposite = next(
        index
        for index, identifier in enumerate(first.vertex_ids)
        if identifier not in shared
    )
    second_opposite = next(
        index
        for index, identifier in enumerate(second.vertex_ids)
        if identifier not in shared
    )
    first_reference = tuple(
        tuple(Fraction(value) for value in point) for point in first.reference_vertices
    )
    second_reference = tuple(
        tuple(Fraction(value) for value in point) for point in second.reference_vertices
    )

    first_edges, first_inverse = _cached_reference_edges(
        first_reference, first_opposite, cache
    )
    _, second_inverse = _cached_reference_edges(second_reference, second_opposite, cache)
    if first_inverse is None or second_inverse is None:
        return False
    metric = _cached_corner_metric(first, first_opposite, first_inverse, cache)
    if metric is None:
        return False
    # Face weights of the preimage of second's opposite corner under first's
    # corner-affine map, relative to first's opposite corner.
    offset = tuple(
        second.corner_images[second_opposite][axis]
        - first.corner_images[first_opposite][axis]
        for axis in range(dimension)
    )
    weights = _apply(first_inverse, _apply(metric, offset))
    total = sum(weights, Fraction(0))
    if _sign(total - 1) <= 0:
        return False
    # A glued corner across the face plane with nonnegative face weights makes
    # the glued reference domain convex; negative weights are clipped at that
    # depth, which changes only the glue, never the cell maps.
    clipped = tuple(value if _sign(value) > 0 else Fraction(0) for value in weights)
    clipped_total = sum(clipped, Fraction(0))
    if not _sign(clipped_total):
        return False
    scaled = tuple(value * total / clipped_total for value in clipped)
    displacement = _apply(first_edges, scaled)
    glued = tuple(
        first_reference[first_opposite][axis] + displacement[axis]
        for axis in range(dimension)
    )
    reference_by_id = dict(zip(first.vertex_ids, first_reference, strict=True))
    second_others = [index for index in range(dimension + 1) if index != second_opposite]
    images = tuple(
        tuple(
            reference_by_id[second.vertex_ids[column]][row] - glued[row]
            for column in second_others
        )
        for row in range(dimension)
    )
    glue = _inverse(_product(images, second_inverse))
    if glue is None:
        return False

    first_controls = _cached_simplex_jacobians(first, cache)
    second_controls = _cached_simplex_jacobians(second, cache)
    if first_controls is None or second_controls is None:
        return False
    glued_controls = (
        *first_controls,
        *(_product(control, glue) for control in second_controls),
    )
    if all(
        _positive_definite_symmetric_part(metric, control) for control in glued_controls
    ):
        return True
    # Any constant preconditioner proves monotonicity; search one when the
    # corner-affine inverse fails on strongly curved cells.
    return monotone_preconditioner(glued_controls) is not None


def _apply(
    matrix: _Matrix, vector: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ...]:
    return tuple(
        sum(
            (matrix[row][inner] * vector[inner] for inner in range(len(vector))),
            Fraction(0),
        )
        for row in range(len(matrix))
    )


_STANDARD_SIMPLICES = {
    2: ((0, 0), (0, 1), (1, 0)),
    3: ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)),
}


def glued_face_monotone(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    cache: StratumSubdivisionCache | None = None,
    /,
) -> bool:
    """Prove two facet-sharing full-dimensional simplices meet only in their facet.

    Applies to tetrahedra in space and triangles in the plane. One cell's
    reference is glued to the other's across the shared facet by an affine map
    ``phi`` fixing the facet corners and sending the glued cell's opposite
    corner to a point ``P`` across the facet hyperplane whose segment to the
    other opposite corner crosses the closed facet, so the glued reference
    domain ``D`` is convex. A fixed interior-crossing glue is tried first; the
    fallback uses the corner-affine inverse image, clipped into that cone. When
    ``A^{-1} DH`` has a positive definite symmetric part at every Bernstein
    control of the glued map's Jacobian on both cells, then
    ``(x - y) . A^{-1}(H(x) - H(y)) > 0`` along every segment of ``D``; ``H`` is
    injective and the two images meet exactly in the image of the facet. The
    caller has proved equal facet traces; both gluing orders are tried.
    """
    shared = set(shared_ids)
    dimension = first.dimension
    kinds = _SIMPLEX_KINDS if dimension == 3 else ("triangle", "simplex:2")
    if (
        dimension not in _STANDARD_SIMPLICES
        or len(shared) != dimension
        or any(
            cell.kind not in kinds
            or cell.dimension != dimension
            or len(cell.vertex_ids) != dimension + 1
            or len(cell.coordinates) != dimension
            or len(cell.corner_images) != dimension + 1
            or any(len(point) != dimension for point in cell.corner_images)
            or not shared <= set(cell.vertex_ids)
            for cell in (first, second)
        )
    ):
        return False
    standard = _STANDARD_SIMPLICES[dimension]
    if any(
        tuple(
            sorted(
                tuple(Fraction(value) for value in reference)
                for reference in cell.reference_vertices
            )
        )
        != standard
        for cell in (first, second)
    ):
        return False
    workset = cache if cache is not None else StratumSubdivisionCache()
    return (
        _fixed_glued_monotone(first, second, shared, workset)
        or _glued_monotone(first, second, shared, workset)
        or _glued_monotone(second, first, shared, workset)
    )


_Point = tuple[Fraction, ...]


def _plane(points: list[_Point], /) -> tuple[_Point, Fraction] | None:
    """Normal and offset of the hyperplane through ``dimension`` points."""
    origin = points[0]
    if len(origin) == 2:
        edge = tuple(value - base for value, base in zip(points[1], origin, strict=True))
        normal: _Point = (edge[1], -edge[0])
    else:
        first = tuple(value - base for value, base in zip(points[1], origin, strict=True))
        second = tuple(
            value - base for value, base in zip(points[2], origin, strict=True)
        )
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
    if not any(normal):
        return None
    return normal, sum((a * b for a, b in zip(normal, origin, strict=True)), Fraction(0))


def _halfspaces(points: list[_Point], /) -> dict[frozenset[int], tuple[_Point, Fraction]]:
    """Facets ``n . x <= c`` of a convex polytope keyed by their vertex set."""
    facets: dict[frozenset[int], tuple[_Point, Fraction]] = {}
    for subset in combinations(range(len(points)), len(points[0])):
        plane = _plane([points[index] for index in subset])
        if plane is None:
            continue
        normal, offset = plane
        values = [
            sum((a * b for a, b in zip(normal, point, strict=True)), Fraction(0)) - offset
            for point in points
        ]
        if all(value >= 0 for value in values):
            normal, offset = tuple(-value for value in normal), -offset
        elif not all(value <= 0 for value in values):
            continue
        key = frozenset(index for index, value in enumerate(values) if value == 0)
        facets.setdefault(key, (normal, offset))
    return facets


def _side(plane: tuple[_Point, Fraction], point: _Point, /) -> Fraction:
    normal, offset = plane
    return sum((a * b for a, b in zip(normal, point, strict=True)), Fraction(0)) - offset


def _cell_jacobian_controls(
    cell: EmbeddingCell, level: int, cache: StratumSubdivisionCache, /
) -> tuple[_Matrix, ...] | None:
    key = (cell.global_id, level)
    if key not in cache.jacobian_controls:
        jacobian = tuple(
            tuple(derivative(expression, axis) for axis in range(cell.dimension))
            for expression in cell.coordinates
        )
        cache.jacobian_controls[key] = subdivided_jacobian_controls(
            jacobian, cell.proof_domain, cell.dimension, level
        )
    return cache.jacobian_controls[key]


def _glued_facet(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared: tuple[int, ...],
    cache: StratumSubdivisionCache,
    /,
) -> bool:
    dimension = first.dimension
    first_points = [
        tuple(Fraction(value) for value in point) for point in first.reference_vertices
    ]
    second_points = [
        tuple(Fraction(value) for value in point) for point in second.reference_vertices
    ]
    first_by_id = dict(zip(first.vertex_ids, first_points, strict=True))
    second_by_id = dict(zip(second.vertex_ids, second_points, strict=True))
    basis = next(
        (
            subset
            for subset in combinations(shared, dimension)
            if _plane([second_by_id[identifier] for identifier in subset]) is not None
        ),
        None,
    )
    if basis is None:
        return False

    def centroid(points: list[_Point], /) -> _Point:
        return tuple(
            sum(values, Fraction(0)) / len(points) for values in zip(*points, strict=True)
        )

    def image(cell: EmbeddingCell, point: _Point, /) -> np.ndarray:
        return np.asarray(
            [
                float(evaluate_expression(expression, point))
                for expression in cell.coordinates
            ]
        )

    inside = centroid(
        [point for identifier, point in second_by_id.items() if identifier not in shared]
    )
    # The glue sends second's off-facet centroid onto the ray from first's
    # off-facet centroid through the facet centroid: across the facet, and for
    # simplices and prisms with the union convex (an apex over the facet, or a
    # prism over a convex quadrilateral). Floats only scale the ray to match the
    # physical depths of both cells.
    facet_center = centroid([first_by_id[identifier] for identifier in shared])
    first_inside = centroid(
        [point for identifier, point in first_by_id.items() if identifier not in shared]
    )
    center_image = image(first, facet_center)
    depth = float(np.linalg.norm(image(first, first_inside) - center_image))
    if not np.isfinite(depth) or depth == 0.0:
        return False
    ratio = float(np.linalg.norm(image(second, inside) - center_image)) / depth
    if not np.isfinite(ratio) or ratio <= 0.0:
        return False
    scale = Fraction(ratio).limit_denominator(2**20)
    if scale <= 0:
        return False
    glued_inside = tuple(
        center + scale * (center - base)
        for center, base in zip(facet_center, first_inside, strict=True)
    )
    source = _edge_columns([second_by_id[identifier] for identifier in basis], inside)
    destination = _edge_columns(
        [first_by_id[identifier] for identifier in basis], glued_inside
    )
    source_inverse = _inverse(source)
    if source_inverse is None:
        return False
    matrix = _product(destination, source_inverse)
    matrix_inverse = _inverse(matrix)
    if matrix_inverse is None:
        return False
    anchor_source = second_by_id[basis[0]]
    anchor = first_by_id[basis[0]]

    def glue(point: _Point, /) -> _Point:
        moved = _apply(
            matrix,
            tuple(value - base for value, base in zip(point, anchor_source, strict=True)),
        )
        # Rational reference points under a rational glue stay rational.
        return tuple(
            cast(Fraction, base + value)
            for base, value in zip(anchor, moved, strict=True)
        )

    if any(
        glue(second_by_id[identifier]) != first_by_id[identifier] for identifier in shared
    ):
        return False
    glued_points = [glue(point) for point in second_points]
    first_facets = _halfspaces(first_points)
    glued_facets = _halfspaces(glued_points)
    first_shared = frozenset(first.vertex_ids.index(identifier) for identifier in shared)
    glued_shared = frozenset(second.vertex_ids.index(identifier) for identifier in shared)
    if first_shared not in first_facets or glued_shared not in glued_facets:
        return False
    # Opposite sides of the shared facet, and each polytope inside every other
    # facet halfspace of the other: then the union is the convex intersection
    # of those halfspaces.
    if any(
        _side(first_facets[first_shared], glued_points[index]) <= 0
        for index in range(len(glued_points))
        if index not in glued_shared
    ):
        return False
    if any(
        _side(plane, point) > 0
        for key, plane in first_facets.items()
        if key != first_shared
        for point in glued_points
    ) or any(
        _side(plane, point) > 0
        for key, plane in glued_facets.items()
        if key != glued_shared
        for point in first_points
    ):
        return False
    # Whole-cell controls first, then uniformly refined ones, which enclose the
    # Jacobian more tightly (one constant preconditioner serves every piece).
    for level in range(SUBDIVISION_LEVELS + 1):
        first_controls = _cell_jacobian_controls(first, level, cache)
        second_controls = _cell_jacobian_controls(second, level, cache)
        if first_controls is None or second_controls is None:
            return False
        if (
            monotone_preconditioner(
                (
                    *first_controls,
                    *(_product(control, matrix_inverse) for control in second_controls),
                )
            )
            is not None
        ):
            return True
    return False


def _edge_columns(points: list[_Point], extra: _Point, /) -> _Matrix:
    """Columns ``points[k] - points[0]`` (k >= 1) and ``extra - points[0]``."""
    columns = [
        *(
            tuple(a - b for a, b in zip(point, points[0], strict=True))
            for point in points[1:]
        ),
        tuple(a - b for a, b in zip(extra, points[0], strict=True)),
    ]
    return tuple(
        tuple(column[row] for column in columns) for row in range(len(points[0]))
    )


def glued_facet_monotone(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    cache: StratumSubdivisionCache,
    /,
) -> bool:
    """Prove facet-sharing simplices, prisms or hexahedra meet only in their facet.

    An exact affine glue ``phi`` maps the second reference cell onto the far
    side of the shared facet of the first, matching the facet corners by vertex
    identity (the caller has proved equal facet traces, so ``H`` -- the first
    map, and the second composed with ``phi^-1`` -- is continuous). The union
    ``D`` is convex when the two polytopes lie on opposite sides of the facet
    and each satisfies every other facet inequality of the other. A constant
    ``M`` with ``sym(M C)`` positive definite at every Bernstein control ``C``
    of ``DH`` then makes ``H`` strictly monotone, hence injective, on ``D``; the
    two images meet exactly in the image of the facet.
    """
    dimension = first.dimension
    shared = tuple(sorted(set(shared_ids)))
    if (
        dimension not in (2, 3)
        or len(shared) < dimension
        or any(
            cell.dimension != dimension
            or cell.kind not in (*_PIECE_KINDS, "prism", "hexahedron")
            or cell.pyramid_normalizations is not None
            or len(cell.coordinates) != dimension
            or len(cell.reference_vertices) != len(cell.vertex_ids)
            or not set(shared) <= set(cell.vertex_ids)
            or any(not isinstance(expression, dict) for expression in cell.coordinates)
            for cell in (first, second)
        )
    ):
        return False
    return _glued_facet(first, second, shared, cache) or _glued_facet(
        second, first, shared, cache
    )


__all__ = [
    "glued_face_monotone",
    "glued_facet_monotone",
    "StratumSubdivisionCache",
    "subdivided_stratum_separation",
]
