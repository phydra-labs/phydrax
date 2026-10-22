# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact edge operands share arithmetic, not source acceptance or rounded chords."""

from fractions import Fraction

import numpy as np
import pytest

import phydrax.meshing.qualification._native_lifecycle as owner
from phydrax.discretization import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from tests._support.coordinate_references import coordinate_reference_budget


def _failure(role: str, message: str, context: dict[str, object]) -> Exception:
    return RuntimeError(role, message, context)


def test_exact_norms_share_reversed_and_permuted_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.asarray(((0, 0, 0), (3, 4, 0), (4, 0, 3), (0, 2, 2)), dtype=float)
    mesh = CellMesh.from_tetrahedra(points, np.asarray(((0, 1, 2, 3),)))
    vertices = {row: tuple(Fraction(value) for value in point) for row, point in enumerate(points)}
    edges = np.asarray(((0, 1), (1, 0), (0, 2), (2, 0), (0, 3)))
    roots: list[Fraction] = []
    original = owner.rounded_rational_square_root

    def observed(value: Fraction) -> float:
        roots.append(value)
        return original(value)

    monkeypatch.setattr(owner, "rounded_rational_square_root", observed)
    with coordinate_reference_budget() as budget:
        actual = owner._once_rounded_edge_lengths(mesh, edges, vertices, _failure, budget)
        required = budget.work_units
    np.testing.assert_array_equal(actual, (5.0, 5.0, 5.0, 5.0, original(Fraction(8))))
    assert roots == [Fraction(25), Fraction(8)]
    with coordinate_reference_budget() as budget:
        with budget.bound_stage(required, 10**8):
            np.testing.assert_array_equal(
                owner._once_rounded_edge_lengths(mesh, edges, vertices, _failure, budget),
                actual,
            )
    with coordinate_reference_budget() as budget:
        with pytest.raises(CoordinateEnclosureResourceError):
            with budget.bound_stage(required - 1, 10**8):
                owner._once_rounded_edge_lengths(mesh, edges, vertices, _failure, budget)
    # No request-spanning cache can authorize a changed current exact operand.
    vertices[1] = (Fraction(6), Fraction(8), Fraction(0))
    with coordinate_reference_budget() as budget:
        changed = owner._once_rounded_edge_lengths(mesh, edges, vertices, _failure, budget)
    np.testing.assert_array_equal(changed[:2], (10.0, 10.0))
    np.testing.assert_array_equal(changed[2:], actual[2:])


def test_reversed_scalar_differences_reuse_the_first_exact_square() -> None:
    cache: dict[tuple[Fraction, Fraction], Fraction] = {}
    first, last = (Fraction(1, 3),), (Fraction(7, 11),)
    with coordinate_reference_budget() as budget:
        expected = owner._squared_edge_components(first, last, cache, budget)
        before = budget.work_units
        assert owner._squared_edge_components(last, first, cache, budget) == expected
        assert budget.work_units == before
    assert expected == ((last[0] - first[0]) ** 2,)
    assert len(cache) == 1
