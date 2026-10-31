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

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ..discretization._coordinate_enclosure import (
    bernstein_coefficients,
    bernstein_degree,
    compose_expression,
    ExactScalar,
    Expression,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ._embedding_cells import EmbeddingCell
from ._mapped_embedding import monotone_preconditioner


_Index = tuple[int, ...]
_Vector = tuple[ExactScalar, ...]
_Controls = dict[_Index, _Vector]
_Barycentric = tuple[Fraction, ...]
_SIMPLEX_KINDS = ("tetrahedron", "simplex:3")
_PIECE_KINDS = ("triangle", "simplex:2", *_SIMPLEX_KINDS)
# Pieces smaller than this fraction of the cell scale are below the binary64
# resolution of the float guidance; their pair stays unresolved.
_RESOLUTION = 2.0**-44


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return int(value > 0) - int(value < 0)


@dataclass(frozen=True)
class _Piece:
    """Reference sub-simplex with exact physical and residual Bernstein controls."""

    vertices: tuple[_Barycentric, ...]
    on_stratum: tuple[bool, ...]
    geometry: _Controls
    geometry_degree: int
    residual: _Controls
    residual_degree: int
    points: np.ndarray
    corners: np.ndarray

    @property
    def diameter(self) -> float:
        differences = self.corners[:, None, :] - self.corners[None, :, :]
        return float(np.sqrt(np.max(np.sum(differences * differences, axis=-1))))


@dataclass
class StratumSubdivisionCache:
    """Pieces of one embedding proof, keyed by cell, stratum, and reference vertices."""

    roots: dict[tuple[int, tuple[int, ...]], _Piece | None] = field(default_factory=dict)
    children: dict[
        tuple[int, tuple[int, ...], tuple[_Barycentric, ...]], tuple[_Piece, _Piece]
    ] = field(default_factory=dict)


def _compositions(total: int, parts: int, /) -> tuple[_Index, ...]:
    if parts == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total, -1, -1)
        for rest in _compositions(total - first, parts - 1)
    )


def _controls(
    polynomials: tuple[Expression, ...], degree: int, dimension: int, /
) -> _Controls | None:
    columns: list[tuple[ExactScalar, ...]] = []
    for polynomial in polynomials:
        if not isinstance(polynomial, dict):
            return None
        columns.append(
            bernstein_coefficients(polynomial, "simplex", dimension, degree=(degree,))
        )
    return {
        index: tuple(column[row] for column in columns)
        for row, index in enumerate(_compositions(degree, dimension + 1))
    }


def _float_points(controls: _Controls, /) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in vector] for vector in controls.values()],
        dtype=np.float64,
    )


def _corner_points(controls: _Controls, degree: int, parts: int, /) -> np.ndarray:
    return np.asarray(
        [
            [
                float(value)
                for value in controls[
                    tuple(degree * int(slot == vertex) for slot in range(parts))
                ]
            ]
            for vertex in range(parts)
        ],
        dtype=np.float64,
    )


def _degree(polynomials: tuple[Expression, ...], dimension: int, /) -> int | None:
    degree = 0
    for polynomial in polynomials:
        if not isinstance(polynomial, dict):
            return None
        degree = max(degree, bernstein_degree(polynomial, "simplex", dimension)[0])
    return max(degree, 1)


def _root_piece(
    cell: EmbeddingCell,
    shared_ids: tuple[int, ...],
    residuals: tuple[Expression, ...],
    /,
) -> _Piece | None:
    dimension = cell.dimension
    parts = dimension + 1
    if (
        cell.kind not in _PIECE_KINDS
        or dimension not in (2, 3)
        or len(cell.coordinates) not in (2, 3)
        or len(cell.coordinates) < dimension
        or len(cell.vertex_ids) != parts
    ):
        return None
    slots: list[int] = []
    for reference in cell.reference_vertices:
        values = tuple(Fraction(value) for value in reference)
        if len(values) != dimension:
            return None
        if not any(values):
            slots.append(0)
        elif sorted(values) == [0] * (dimension - 1) + [1]:
            slots.append(values.index(Fraction(1)) + 1)
        else:
            return None
    if sorted(slots) != list(range(parts)):
        return None
    on_stratum = [False] * parts
    for local, slot in enumerate(slots):
        on_stratum[slot] = cell.vertex_ids[local] in shared_ids
    geometry_degree = _degree(cell.coordinates, dimension)
    if geometry_degree is None:
        return None
    geometry = _controls(cell.coordinates, geometry_degree, dimension)
    if geometry is None:
        return None
    composed = tuple(
        compose_expression(residual, cell.coordinates) for residual in residuals
    )
    residual_degree = _degree(composed, dimension) if composed else 1
    if residual_degree is None:
        return None
    residual = _controls(composed, residual_degree, dimension) if composed else {}
    if residual is None:
        return None
    vertices = tuple(
        tuple(Fraction(int(slot == vertex)) for slot in range(parts))
        for vertex in range(parts)
    )
    return _Piece(
        vertices,
        tuple(on_stratum),
        geometry,
        geometry_degree,
        residual,
        residual_degree,
        _float_points(geometry),
        _corner_points(geometry, geometry_degree, parts),
    )


def _bisect_controls(
    controls: _Controls, degree: int, first: int, second: int, /
) -> tuple[_Controls, _Controls]:
    """Exact de Casteljau halves along edge ``(first, second)``.

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
    geometry = _bisect_controls(piece.geometry, piece.geometry_degree, first, second)
    residual = (
        _bisect_controls(piece.residual, piece.residual_degree, first, second)
        if piece.residual
        else ({}, {})
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
                vertices,
                on_stratum,
                geometry_controls,
                piece.geometry_degree,
                residual_controls,
                piece.residual_degree,
                _float_points(geometry_controls),
                _corner_points(
                    geometry_controls, piece.geometry_degree, len(piece.vertices)
                ),
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
        return cached
    corners = piece.corners
    first, second = max(
        (
            (left, right)
            for left in range(len(piece.vertices))
            for right in range(left + 1, len(piece.vertices))
        ),
        key=lambda pair: float(np.sum((corners[pair[0]] - corners[pair[1]]) ** 2)),
    )
    result = _bisect(piece, first, second)
    cache.children[memo_key] = result
    return result


def _strict_direction(vectors: np.ndarray, /) -> np.ndarray | None:
    """A float direction with positive products against every row, if one is found.

    Rows are normalized first. The mean is tried, then Gilbert's least-norm
    iteration (stopping at the first strictly positive direction), then in two
    or three dimensions the sum of the extreme rays of the closed dual cone,
    which is interior to it whenever the open dual cone is nonempty.
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
            cache.roots[key] = _root_piece(cell, stratum, residuals)
        piece = cache.roots[key]
        if piece is None:
            return None
        pieces.append(piece)
    scale = max(
        1.0,
        float(np.max(np.abs(pieces[0].corners))),
        float(np.max(np.abs(pieces[1].corners))),
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


def _positive_definite_symmetric_part(matrix: _Matrix, /) -> bool:
    """Sylvester's criterion on the leading minors of the symmetric part."""
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


def _glued_monotone(
    first: EmbeddingCell, second: EmbeddingCell, shared: set[int], /
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
    first_edges = _edge_matrix(first_reference, first_opposite)
    first_inverse = _inverse(first_edges)
    second_inverse = _inverse(_edge_matrix(second_reference, second_opposite))
    if first_inverse is None or second_inverse is None:
        return False
    metric = _inverse(
        _product(_edge_matrix(first.corner_images, first_opposite), first_inverse)
    )
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
    glued = tuple(
        first_reference[first_opposite][axis] + _apply(first_edges, scaled)[axis]
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
    first_controls = _jacobian_controls(first)
    second_controls = _jacobian_controls(second)
    if first_controls is None or second_controls is None:
        return False
    glued_controls = (
        *first_controls,
        *(_product(control, glue) for control in second_controls),
    )
    if all(
        _positive_definite_symmetric_part(_product(metric, control))
        for control in glued_controls
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
    first: EmbeddingCell, second: EmbeddingCell, shared_ids: tuple[int, ...], /
) -> bool:
    """Prove two facet-sharing full-dimensional simplices meet only in their facet.

    Applies to tetrahedra in space and triangles in the plane. One cell's
    reference is glued to the other's across the shared facet by an affine map
    ``phi`` fixing the facet corners and sending the glued cell's opposite
    corner to a point ``P`` across the facet hyperplane whose segment to the
    other opposite corner crosses the closed facet, so the glued reference
    domain ``D`` is convex (``P`` is the preimage of the opposite corner's image
    under the corner-affine map ``A``, clipped into that cone). When
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
    return _glued_monotone(first, second, shared) or _glued_monotone(
        second, first, shared
    )


__all__ = [
    "glued_face_monotone",
    "StratumSubdivisionCache",
    "subdivided_stratum_separation",
]
