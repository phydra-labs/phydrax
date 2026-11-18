from __future__ import annotations

from fractions import Fraction

import pytest

from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    evaluate_expression,
    ExactScalar,
    Expression,
    multiply,
    scale,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._stratum_subdivision import (
    _glued_monotone,
    glued_face_monotone,
    StratumSubdivisionCache,
)


def _cell(
    identifier: int,
    ids: tuple[int, ...],
    coordinates: tuple[Expression, ...],
) -> EmbeddingCell:
    references = tuple(
        tuple(Fraction(value) for value in point)
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    images = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    return EmbeddingCell(
        global_id=identifier,
        block_name="tetrahedron",
        local_index=identifier,
        kind="tetrahedron",
        dimension=3,
        domain="simplex",
        vertex_indices=(0, 1, 2, 3),
        vertex_ids=ids,
        reference_vertices=references,
        coordinate_expressions=coordinates,
        corner_images=images,
        affine_offset=None,
        affine_matrix=None,
        image_lower=(-2.0, -2.0, -2.0),
        image_upper=(2.0, 2.0, 2.0),
    )


@pytest.mark.parametrize(
    "coefficient", [Fraction(1, 5), Fraction(-40), RadicalElement.sqrt(2) / 5]
)
def test_dynamic_glue_preserves_contact_and_overlap_decisions(
    coefficient: ExactScalar,
) -> None:
    x, y, z = axes(3)
    curved: tuple[Expression, ...] = (add(x, scale(multiply(y, y), coefficient)), y, z)
    other_coordinates: tuple[Expression, ...] = (
        {(0, 0, 0): Fraction(1), (0, 1, 0): Fraction(-1), (0, 0, 1): Fraction(-1)},
        {(0, 0, 0): Fraction(1), (1, 0, 0): Fraction(-1), (0, 0, 1): Fraction(-1)},
        {(0, 0, 0): Fraction(1), (1, 0, 0): Fraction(-1), (0, 1, 0): Fraction(-1)},
    )
    first = _cell(0, (0, 1, 2, 3), curved)
    other = _cell(
        1,
        (4, 1, 2, 3),
        tuple(compose_expression(expression, other_coordinates) for expression in curved),
    )
    overlap = _cell(2, (5, 1, 2, 3), curved)
    shared = {1, 2, 3}
    if coefficient == Fraction(-40):
        # At the first corner, A^-1 J has off-diagonal entry 40.
        # Its symmetric leading 2x2 minor is strictly negative, so the
        # corner preconditioner cannot prove monotonicity.
        assert Fraction(1) - Fraction(40, 2) ** 2 == -399
    assert _glued_monotone(first, other, shared, StratumSubdivisionCache())
    assert not _glued_monotone(first, overlap, shared, StratumSubdivisionCache())
    assert glued_face_monotone(first, other, (1, 2, 3))
    assert not glued_face_monotone(first, overlap, (1, 2, 3))
