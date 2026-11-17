# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import random
from fractions import Fraction

from phydrax.discretization._coordinate_enclosure import Expression
from phydrax.geometry import _mapped_embedding as mapped
from phydrax.geometry._embedding_cells import EmbeddingCell


def _cell(index: int, coordinates: tuple[Expression, ...], domain: str) -> EmbeddingCell:
    return EmbeddingCell(
        index,
        "mapped",
        index,
        "interval",
        1,
        domain,
        (0, 1),
        (0, 1),
        ((Fraction(0),), (Fraction(1),)),
        coordinates,
        (),
        None,
        None,
        (0.0,) * len(coordinates),
        (1.0,) * len(coordinates),
    )


def _candidates() -> tuple[EmbeddingCell, ...]:
    rng = random.Random(948351)
    cells = []
    for index in range(12):
        scale = Fraction(rng.randint(1, 6))
        domain = "box" if index % 2 == 0 else "simplex"
        coordinates: tuple[Expression, ...]
        match index % 4:
            case 0:  # The second direct coordinate is monotone.
                coordinates = (
                    {(1,): -scale, (2,): -scale / 8},
                    {(1,): 2 * scale, (2,): scale / 8},
                )
            case 1:  # Only the centroid preconditioner reverses orientation.
                coordinates = ({(1,): -scale, (2,): -scale / 8},)
            case 2:  # Neither coordinate is monotone; their centroid sum is.
                coordinates = (
                    {(1,): -scale / 5, (2,): scale / 2},
                    {(1,): 4 * scale / 5, (2,): -scale / 2},
                )
            case _:  # The whole-cell hull crosses zero; subdivision separates it.
                coordinates = ({(1,): 3 * scale / 10, (2,): -scale / 2, (3,): scale / 3},)
        cells.append(_cell(index, coordinates, domain))
    return tuple(cells)


def _independent(
    cells: tuple[EmbeddingCell, ...],
) -> tuple[mapped.MappedInjectivityWitness | None | ValueError | ZeroDivisionError, ...]:
    results: list[
        mapped.MappedInjectivityWitness | None | ValueError | ZeroDivisionError
    ] = []
    for cell in cells:
        try:
            results.append(mapped.prove_mapped_cell_injective(cell))
        except (ValueError, ZeroDivisionError) as error:
            results.append(error)
    return tuple(results)


def _same(
    actual: tuple[
        mapped.MappedInjectivityWitness | None | ValueError | ZeroDivisionError, ...
    ],
    expected: tuple[
        mapped.MappedInjectivityWitness | None | ValueError | ZeroDivisionError, ...
    ],
) -> None:
    for value, reference in zip(actual, expected, strict=True):
        if isinstance(reference, (ValueError, ZeroDivisionError)):
            assert type(value) is type(reference) and str(value) == str(reference)
        else:
            assert value == reference


def test_randomized_fallback_ladder_witnesses_and_order() -> None:
    cells = _candidates()
    reference = _independent(cells)
    for index, result in enumerate(reference):
        assert isinstance(result, mapped.MappedInjectivityWitness)
        if index % 4 == 0:
            assert result.projection_rows == (1,)
            assert result.projection_matrix == ((Fraction(1),),)
        elif index % 4 == 1:
            assert result.projection_rows == (0,)
            assert result.projection_matrix[0][0] < 0
        elif index % 4 == 2:
            assert result.projection_rows == (0, 1)
        else:
            assert result.projection_matrix[0][0] > 1
    rng = random.Random(52483)
    for _ in range(4):
        order = list(range(len(cells)))
        rng.shuffle(order)
        ordered = tuple(cells[index] for index in order)
        _same(
            mapped.prove_mapped_cells_injective(ordered),
            tuple(reference[index] for index in order),
        )
        midpoint = len(ordered) // 2
        _same(
            mapped.prove_mapped_cells_injective(ordered[:midpoint])
            + mapped.prove_mapped_cells_injective(ordered[midpoint:]),
            tuple(reference[index] for index in order),
        )


def test_centroid_projection_skips_singular_row_subsets() -> None:
    cells = []
    for index in range(6):
        scale = Fraction(index + 1)
        cells.append(
            EmbeddingCell(
                index,
                "mapped",
                index,
                "triangle",
                2,
                "box" if index % 2 == 0 else "simplex",
                (0, 1, 2),
                (0, 1, 2),
                (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(0), Fraction(1)),
                ),
                (
                    {},
                    {(1, 0): -scale, (2, 0): -scale / 8},
                    {(0, 1): -scale, (0, 2): -scale / 8},
                ),
                (),
                None,
                None,
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            )
        )
    candidates = tuple(cells)
    expected = _independent(candidates)
    _same(mapped.prove_mapped_cells_injective(candidates), expected)
    for result in expected:
        assert isinstance(result, mapped.MappedInjectivityWitness)
        assert result.projection_rows == (1, 2)
        assert result.projection_matrix[0][0] < 0
        assert result.projection_matrix[1][1] < 0


def test_malformed_polynomial_item_is_isolated_from_other_proofs() -> None:
    cells = _candidates()
    oversized = _cell(100, ({(1,): Fraction(1 << 4100)},), "box")
    # Differentiation retains the extra exponent, which the batch kernel rejects.
    malformed = _cell(101, ({(1, 0): Fraction(1)},), "box")
    baseline = _independent(cells + (oversized,))
    actual = mapped.prove_mapped_cells_injective(cells + (oversized, malformed))
    _same(actual[:-1], baseline)
    assert all(
        isinstance(result, mapped.MappedInjectivityWitness) for result in actual[:-1]
    )
    assert isinstance(actual[-1], ValueError)
    assert str(actual[-1]) == "Invalid polynomial exponents."


def test_input_perturbation_exposes_noninjective_ambient_map() -> None:
    # Both coordinates of the perturbed map agree at distinct endpoints.
    good = _cell(0, ({(1,): Fraction(1)}, {(1,): Fraction(2)}), "box")
    bad = _cell(
        1,
        (
            {(1,): Fraction(1), (2,): Fraction(-1)},
            {(1,): Fraction(2), (2,): Fraction(-2)},
        ),
        "box",
    )
    for expression in bad.coordinates:
        assert isinstance(expression, dict)
        assert sum(expression.values(), Fraction(0)) == 0
        assert expression.get((0,), Fraction(0)) == 0
    proof, refusal = mapped.prove_mapped_cells_injective((good, bad))
    assert isinstance(proof, mapped.MappedInjectivityWitness)
    assert refusal is None
