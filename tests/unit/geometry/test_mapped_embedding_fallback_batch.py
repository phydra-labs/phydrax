# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from fractions import Fraction

import pytest

from phydrax.discretization._coordinate_enclosure import BernsteinDomain, Expression
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
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


def test_native_refusal_and_item_error_preserve_other_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _candidates() + (
        _cell(100, ({(1,): Fraction(7), (2,): Fraction(1, 4)},), "box"),
    )
    baseline = _independent(cells[:-1])
    original = mapped.symmetric_jacobian_verdicts

    def refuse_and_raise(
        matrices: Sequence[Sequence[Sequence[Mapping[tuple[int, ...], ExactScalar]]]],
        domain: BernsteinDomain,
        dimension: int,
        /,
        *,
        parallelism: int | None = None,
    ) -> list[tuple[tuple[bool, str], ...] | None]:
        if any(
            entry.get((0,)) == 7 for matrix in matrices for row in matrix for entry in row
        ):
            raise ValueError("injected item failure")
        verdicts = original(matrices, domain, dimension, parallelism=parallelism)
        return [
            None
            if any(
                value < 0 for row in matrix for entry in row for value in entry.values()
            )
            else verdict
            for matrix, verdict in zip(matrices, verdicts, strict=True)
        ]

    monkeypatch.setattr(mapped, "symmetric_jacobian_verdicts", refuse_and_raise)
    expected = _independent(cells)
    actual = mapped.prove_mapped_cells_injective(cells)
    _same(actual, expected)
    _same(actual[:-1], baseline)
    assert isinstance(actual[-1], ValueError)
    assert str(actual[-1]) == "injected item failure"
    assert all(
        isinstance(result, mapped.MappedInjectivityWitness) for result in actual[:-1]
    )


def test_corrupted_ambient_projection_digest_changes_consumer_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _candidates()
    expected = _independent(cells)
    original = mapped.symmetric_jacobian_verdicts

    def corrupt(
        matrices: Sequence[Sequence[Sequence[Mapping[tuple[int, ...], ExactScalar]]]],
        domain: BernsteinDomain,
        dimension: int,
        /,
        *,
        parallelism: int | None = None,
    ) -> list[tuple[tuple[bool, str], ...] | None]:
        verdicts = original(matrices, domain, dimension, parallelism=parallelism)
        for index, matrix in enumerate(matrices):
            if matrix == (({(0,): Fraction(1)},),):
                verdicts[index] = ((True, "corrupted"),)
        return verdicts

    monkeypatch.setattr(mapped, "symmetric_jacobian_verdicts", corrupt)
    actual = mapped.prove_mapped_cells_injective(cells)
    for index in (2, 6, 10):
        result = actual[index]
        assert isinstance(result, mapped.MappedInjectivityWitness)
        assert result != expected[index]
        assert result.principal_minor_controls == ("corrupted",)
