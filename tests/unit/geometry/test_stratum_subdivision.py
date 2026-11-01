from __future__ import annotations

from fractions import Fraction

from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    constant,
    evaluate_expression,
    Expression,
    multiply,
    scale,
)
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._global_embedding import _stratum_residuals
from phydrax.geometry._mapped_embedding import prove_mapped_cell_injective
from phydrax.geometry._stratum_subdivision import (
    glued_facet_monotone,
    StratumSubdivisionCache,
    subdivided_stratum_separation,
)


type _Point = tuple[Fraction, ...]

_PRISM = tuple(
    (Fraction(u), Fraction(v), Fraction(w))
    for w in (0, 1)
    for u, v in ((0, 0), (1, 0), (0, 1))
)
_TETRAHEDRON = tuple(
    (Fraction(u), Fraction(v), Fraction(w))
    for u, v, w in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
)


def _bend(
    shear: Fraction = Fraction(1, 10), lift: Fraction = Fraction(1, 5)
) -> tuple[Expression, ...]:
    """A smooth injective map of space shared by every cell (equal traces)."""
    x, y, z = axes(3)
    return (
        add(x, scale(multiply(y, y), shear)),
        y,
        add(z, scale(multiply(x, y), lift)),
    )


def _cell(
    global_id: int,
    kind: str,
    vertex_ids: tuple[int, ...],
    corners: tuple[_Point, ...],
    physical: tuple[Expression, ...],
    /,
) -> EmbeddingCell:
    """``physical`` composed with the straight cell through ``corners``."""
    references = _PRISM if kind == "prism" else _TETRAHEDRON
    u, v, w = axes(3)
    origin = corners[0]
    if kind == "prism":
        edges = (corners[1], corners[2], corners[3])
    else:
        edges = corners[1:]
    straight = tuple(
        add(
            constant(origin[axis], 3),
            add(
                scale(u, edges[0][axis] - origin[axis]),
                add(
                    scale(v, edges[1][axis] - origin[axis]),
                    scale(w, edges[2][axis] - origin[axis]),
                ),
            ),
        )
        for axis in range(3)
    )
    coordinates = tuple(
        compose_expression(expression, straight) for expression in physical
    )
    images = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    return EmbeddingCell(
        global_id=global_id,
        block_name=kind,
        local_index=global_id,
        kind=kind,
        dimension=3,
        domain="prism" if kind == "prism" else "simplex",
        vertex_indices=tuple(range(len(vertex_ids))),
        vertex_ids=vertex_ids,
        reference_vertices=references,
        coordinates=coordinates,
        corner_images=images,
        affine_offset=None,
        affine_matrix=None,
        image_lower=tuple(
            float(min(point[axis] for point in images)) - 1.0 for axis in range(3)
        ),
        image_upper=tuple(
            float(max(point[axis] for point in images)) + 1.0 for axis in range(3)
        ),
    )


def _p(x: float, y: float, z: float, /) -> _Point:
    return Fraction(x), Fraction(y), Fraction(z)


def _layer_prism(global_id: int, ids: tuple[int, ...], base: tuple, /) -> EmbeddingCell:
    corners = (*(_p(*point, 0) for point in base), *(_p(*point, 1) for point in base))
    return _cell(global_id, "prism", ids, corners, _bend())


_FIRST = _layer_prism(0, (0, 1, 2, 3, 4, 5), ((0, 0), (1, 0), (0, 1)))


def test_curved_prisms_sharing_a_quadrilateral_meet_only_in_it() -> None:
    # The glued second prism completes a convex quadrilateral base: one
    # constant preconditioner makes the glued map monotone.
    neighbour = _layer_prism(1, (1, 6, 2, 4, 7, 5), ((1, 0), (1, 1), (0, 1)))
    assert glued_facet_monotone(_FIRST, neighbour, (1, 2, 4, 5))
    # A positively oriented neighbour folded back over the first prism is refused.
    folded = _layer_prism(2, (1, 2, 6, 4, 5, 7), ((1, 0), (0, 1), (0.25, 0.25)))
    assert not glued_facet_monotone(_FIRST, folded, (1, 2, 4, 5))


def test_curved_tetrahedron_on_a_prism_cap_meets_it_only_in_the_cap() -> None:
    corners = (_p(0, 0, 1), _p(1, 0, 1), _p(0, 1, 1), _p(0.25, 0.25, 1.5))
    cap = _cell(3, "simplex:3", (3, 4, 5, 8), corners, _bend())
    assert glued_facet_monotone(_FIRST, cap, (3, 4, 5))
    # A positively oriented tetrahedron sunk into the prism overlaps it.
    sunk = (corners[0], corners[2], corners[1], _p(0.25, 0.25, 0.5))
    inverted = _cell(4, "simplex:3", (3, 5, 4, 8), sunk, _bend())
    assert not glued_facet_monotone(_FIRST, inverted, (3, 4, 5))


def test_prism_and_tetrahedron_sharing_a_vertex_are_separated_by_subdivision() -> None:
    corners = (_p(1, 0, 1), _p(2, 0, 1), _p(1.5, 0.5, 1), _p(1.5, 0.125, 1.75))
    neighbour = _cell(5, "simplex:3", (4, 9, 10, 11), corners, _bend())
    residuals = _stratum_residuals(_FIRST, (4,))
    assert residuals is not None
    counts = subdivided_stratum_separation(
        _FIRST, neighbour, (4,), residuals, StratumSubdivisionCache()
    )
    assert counts is not None and sum(counts) > 0


def test_sheared_prism_is_injective_by_a_searched_preconditioner() -> None:
    # J = I + 3 u e3 e1^T: the centroid inverse leaves sym(M J) indefinite at
    # u = 1, the searched constant preconditioner J(1/2)^-1 does not.
    x, y, z = axes(3)
    cell = _cell(
        6,
        "prism",
        (0, 1, 2, 3, 4, 5),
        _PRISM,
        (x, y, add(z, scale(multiply(x, x), Fraction(3, 2)))),
    )
    witness = prove_mapped_cell_injective(cell)
    assert witness is not None
    assert witness.projection_matrix != tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )
