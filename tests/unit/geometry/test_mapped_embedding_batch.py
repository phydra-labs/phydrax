# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import random
from dataclasses import replace
from fractions import Fraction

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_enclosure import Expression, RationalPolynomial
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry import (
    _global_embedding as global_embedding,
    _mapped_embedding as mapped,
)
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._mesh_certificates import MeshCertificateBinding


def _cell(index: int, expression: Expression, domain: str = "box") -> EmbeddingCell:
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
        (expression,),
        (),
        None,
        None,
        (0.0,),
        (1.0,),
    )


def _independent(
    cells: tuple[EmbeddingCell, ...], /
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


def _compare(
    actual: tuple[
        mapped.MappedInjectivityWitness | None | ValueError | ZeroDivisionError, ...
    ],
    expected: tuple[
        mapped.MappedInjectivityWitness | None | ValueError | ZeroDivisionError, ...
    ],
) -> None:
    for left, right in zip(actual, expected, strict=True):
        if isinstance(right, (ValueError, ZeroDivisionError)):
            assert type(left) is type(right) and str(left) == str(right)
        else:
            assert left == right


def test_randomized_batch_matches_individual_witnesses_and_refusals() -> None:
    rng = random.Random(58139)
    cells = [
        _cell(
            i,
            {
                (1,): Fraction(rng.choice((-3, -1, 1, 3))),
                (2,): Fraction(rng.randint(-2, 2), 8),
            },
            "simplex" if i % 2 else "box",
        )
        for i in range(24)
    ]
    cells.extend(
        (
            _cell(24, {(2,): Fraction(1)}),  # zero derivative at the boundary
            _cell(25, {(1,): Fraction(1 << 4100)}),  # native range refusal
            _cell(26, {(1,): RadicalElement.sqrt(2)}),
            _cell(
                27,
                RationalPolynomial(
                    {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(1)}
                ),
            ),
            _cell(
                28,
                RationalPolynomial(
                    {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(-2)}
                ),
            ),
            replace(_cell(29, {(1,): Fraction(1)}), kind="pyramid"),
            _cell(30, {(1,): Fraction(1)}, "invalid"),
        )
    )
    cells_tuple = tuple(cells)
    expected = _independent(cells_tuple)
    actual = mapped.prove_mapped_cells_injective(cells_tuple)
    _compare(actual, expected)
    assert actual[24] is None
    assert actual[25] is not None and actual[26] is not None
    rational_witness = actual[27]
    assert isinstance(rational_witness, mapped.MappedInjectivityWitness)
    assert rational_witness.denominator_controls
    assert actual[28] is None and actual[29] is None
    assert isinstance(actual[30], ValueError)


def test_homogeneous_batches_have_exact_positive_diagonal_jacobians() -> None:
    rng = random.Random(316)
    cells = []
    for dimension in (2, 3):
        for index in range(6):
            expressions: tuple[Expression, ...] = tuple(
                {
                    tuple(int(axis == column) for axis in range(dimension)): Fraction(1),
                    tuple(2 * int(axis == column) for axis in range(dimension)): Fraction(
                        rng.randint(1, 4), 16
                    ),
                }
                for column in range(dimension)
            )
            references = (tuple(Fraction(0) for _ in range(dimension)),) + tuple(
                tuple(Fraction(int(axis == column)) for axis in range(dimension))
                for column in range(dimension)
            )
            cells.append(
                EmbeddingCell(
                    dimension * 10 + index,
                    "mapped",
                    index,
                    "simplex",
                    dimension,
                    "simplex",
                    tuple(range(dimension + 1)),
                    tuple(range(dimension + 1)),
                    references,
                    expressions,
                    (),
                    None,
                    None,
                    (0.0,) * dimension,
                    (1.0,) * dimension,
                )
            )
    candidates = tuple(cells)
    expected = _independent(candidates)
    actual = mapped.prove_mapped_cells_injective(candidates)
    _compare(actual, expected)
    for cell, result in zip(candidates, actual, strict=True):
        assert isinstance(result, mapped.MappedInjectivityWitness)
        dimension = cell.dimension
        assert result.projection_rows == tuple(range(dimension))
        assert result.projection_matrix == tuple(
            tuple(Fraction(int(row == column)) for column in range(dimension))
            for row in range(dimension)
        )
        # Every diagonal derivative is 1 + 2*a*x, with a > 0 and x >= 0.
        for axis, expression in enumerate(cell.coordinates):
            assert isinstance(expression, dict)
            linear = tuple(int(index == axis) for index in range(dimension))
            quadratic = tuple(2 * value for value in linear)
            assert expression[linear] == 1 and expression[quadratic] > 0
        assert len(result.principal_minor_controls) == dimension
    changed = replace(
        candidates[0],
        coordinate_expressions=(
            {(1, 0): Fraction(1), (2, 0): Fraction(-1)},
            candidates[0].coordinates[1],
        ),
    )
    # Distinct simplex vertices (0, 0) and (1, 0) have the same image.
    first = changed.coordinates[0]
    assert isinstance(first, dict)
    assert first.get((0, 0), Fraction(0)) == sum(first.values(), Fraction(0))
    assert mapped.prove_mapped_cells_injective((changed,))[0] is None


def test_findings_match_individual_proofs_and_owing_filter() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    binding = MeshCertificateBinding(mesh, CellGeometrySpec.affine(mesh), "affine")
    cells = (
        _cell(0, {(1,): Fraction(1), (2,): Fraction(1, 4)}),
        _cell(1, {(2,): Fraction(1)}),
        _cell(2, {(1,): Fraction(-1)}),
        _cell(3, {(1,): RadicalElement.sqrt(2)}),
        _cell(4, {(1,): Fraction(1)}, "invalid"),
    )
    owing = np.asarray((True, True, False, True, True), dtype=np.bool_)
    actual = global_embedding._cell_map_findings(cells, binding=binding, owing=owing)
    findings, witnesses, injective = actual
    assert injective == {0, 3}
    assert len(findings) == 2 and len(witnesses) == 1
    assert findings[0].entity_ids == (1,)
    assert findings[1].entity_ids == (4,)
    assert all(finding.check == "mapped_cell_injectivity" for finding in findings)
    assert all(finding.status == "unresolved" for finding in findings)
