# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_enclosure import (
    BernsteinDomain,
    Expression,
    RationalPolynomial,
)
from phydrax.discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
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
    # A consumer must not accept a lost or substituted control identity.
    witness = actual[26]
    assert isinstance(witness, mapped.MappedInjectivityWitness)
    assert witness != replace(witness, principal_minor_controls=("incorrect",))


def test_homogeneous_batches_and_corrupted_digest_are_consumer_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    original = mapped.symmetric_jacobian_verdicts

    def corrupt(
        matrices: Sequence[Sequence[Sequence[Mapping[tuple[int, ...], ExactScalar]]]],
        domain: BernsteinDomain,
        dimension: int,
        /,
        *,
        parallelism: int | None = None,
    ) -> list[tuple[tuple[bool, str], ...] | None]:
        decisions = original(matrices, domain, dimension, parallelism=parallelism)
        first = decisions[0]
        if first is None:
            reference = expected[0 if dimension == 2 else 6]
            assert isinstance(reference, mapped.MappedInjectivityWitness)
            first = tuple((True, digest) for digest in reference.principal_minor_controls)
        decisions[0] = ((first[0][0], "incorrect"), *first[1:])
        return decisions

    monkeypatch.setattr(mapped, "symmetric_jacobian_verdicts", corrupt)
    actual = mapped.prove_mapped_cells_injective(candidates)
    assert actual[0] != expected[0] and actual[6] != expected[6]


def test_findings_match_individual_proofs_and_owing_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(global_embedding, "prove_mapped_cells_injective", _independent)
    expected = global_embedding._cell_map_findings(cells, binding=binding, owing=owing)
    assert actual == expected
    findings, witnesses, injective = actual
    assert injective == {0, 3}
    assert len(findings) == 2 and len(witnesses) == 1
    assert findings[0].entity_ids == (1,)
