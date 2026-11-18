from __future__ import annotations

from fractions import Fraction
from random import Random

from phydrax.discretization._coordinate_enclosure import Expression
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._stratum_subdivision import _decoded_controls, _jacobian_controls


def _cell(coordinates: tuple[Expression, ...]) -> EmbeddingCell:
    return EmbeddingCell(
        global_id=0,
        block_name="simplex:3",
        local_index=0,
        kind="simplex:3",
        dimension=3,
        domain="simplex",
        vertex_indices=(0, 1, 2, 3),
        vertex_ids=(0, 1, 2, 3),
        reference_vertices=(),
        coordinate_expressions=coordinates,
        corner_images=(),
        affine_offset=None,
        affine_matrix=None,
        image_lower=(0.0, 0.0, 0.0),
        image_upper=(1.0, 1.0, 1.0),
    )


def test_quadratic_jacobian_controls_match_fraction_derivatives() -> None:
    random = Random(19114)
    indices: list[tuple[int, ...]] = [
        (x, y, z) for x in range(3) for y in range(3 - x) for z in range(3 - x - y)
    ]
    corners = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    for case in range(200):
        rows: list[dict[tuple[int, ...], Fraction]] = [
            {
                index: Fraction(
                    random.randrange(-15, 16) * (1 << 4200 if case == 0 else 1),
                    random.randrange(1, 13),
                )
                for index in indices
                if random.randrange(3) != 0
            }
            for _ in range(3)
        ]
        coordinates: tuple[Expression, ...] = tuple(
            {index: value for index, value in row.items()} for row in rows
        )
        controls = _jacobian_controls(_cell(coordinates))
        assert controls is not None
        actual = _decoded_controls(controls.controls)
        for index, vector in actual.items():
            point = corners[index.index(1)]
            expected = []
            for row in rows:
                for axis in range(3):
                    value = Fraction(0)
                    for exponent, coefficient in row.items():
                        if exponent[axis]:
                            term = coefficient * exponent[axis]
                            for component in range(3):
                                term *= point[component] ** (
                                    exponent[component] - int(component == axis)
                                )
                            value += term
                    expected.append(value)
            assert vector == tuple(expected)


def test_nonquadratic_jacobian_uses_general_exact_controls() -> None:
    coordinates: tuple[Expression, ...] = (
        {(3, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    controls = _jacobian_controls(_cell(coordinates))
    assert controls is not None
    values = _decoded_controls(controls.controls)
    assert len(values) == 10
    assert all(vector[4] == 1 and vector[8] == 1 for vector in values.values())
    assert max(vector[0] for vector in values.values()) == 3
