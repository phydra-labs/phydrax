#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Check rational intersection integrals and typed overlap resource refusal."""

from fractions import Fraction

import pytest

from phydrax.geometry._exact_polyhedral_geometry import (
    ExactOverlapRefusal,
    ExactOverlapStatus,
    ExactTetrahedronOverlap,
    StarTetrahedronPreparation,
    tetrahedra_disjoint,
    Tetrahedron,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
)
from tests._support.coordinate_references import coordinate_reference_budget


@pytest.fixture(scope="module")
def unit_tetrahedron() -> Tetrahedron:
    return (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_overlap_retains_rational_integrals_and_oriented_simplices(
    unit_tetrahedron: Tetrahedron,
) -> None:
    with coordinate_reference_budget() as budget:
        overlap = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            second_moments=True,
            retain_simplices=True,
            budget=budget,
        )
    assert isinstance(overlap, ExactTetrahedronOverlap)
    assert overlap.status is ExactOverlapStatus.SUCCESS
    # Uniform simplex barycentrics have E[x_i]=1/4, E[x_i^2]=1/10 and
    # E[x_i*x_j]=1/20 for i != j; the unit tetrahedron has volume 1/6.
    assert overlap.volume == Fraction(1, 6)
    assert overlap.first_moment == (Fraction(1, 24),) * 3
    assert overlap.second_moment == tuple(
        tuple(Fraction(1, 60 if row == column else 120) for column in range(3))
        for row in range(3)
    )
    assert overlap.simplices
    assert overlap.operation_count > 0


def test_overlap_work_refusal_has_typed_status_without_partial_integrals(
    unit_tetrahedron: Tetrahedron,
) -> None:
    with coordinate_reference_budget() as budget:
        starting_work = budget.work_units
        refusal = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            maximum_work=0,
            budget=budget,
        )
    assert isinstance(refusal, ExactOverlapRefusal)
    assert refusal.status is ExactOverlapStatus.RESOURCE_LIMIT
    assert refusal.resource == "coefficient_work"
    assert refusal.operation_count >= 0
    assert refusal.operation_count > 0
    assert refusal.operation_count == budget.work_units - starting_work


@pytest.mark.parametrize("offset", (1, 2), ids=("shared-boundary", "separated"))
def test_disjoint_bounds_and_overlap_agree_on_zero_volume(
    unit_tetrahedron: Tetrahedron,
    offset: int,
) -> None:
    translated = tuple(
        (point[0] + offset, point[1], point[2]) for point in unit_tetrahedron
    )
    second: Tetrahedron = (translated[0], translated[1], translated[2], translated[3])
    assert tetrahedra_disjoint(unit_tetrahedron, second)
    with coordinate_reference_budget() as budget:
        overlap = tetrahedron_overlap(unit_tetrahedron, second, budget=budget)
    assert isinstance(overlap, ExactTetrahedronOverlap)
    assert overlap.status is ExactOverlapStatus.SUCCESS
    assert overlap.volume == 0
    assert overlap.first_moment == (Fraction(0),) * 3


@pytest.mark.parametrize("second_moments", (False, True))
def test_earned_tetrahedron_planes_preserve_integrals_and_reduce_actual_work(
    unit_tetrahedron: Tetrahedron, second_moments: bool
) -> None:
    with coordinate_reference_budget() as budget:
        preparation = TetrahedronPlanePreparation(budget)
        first = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            second_moments=second_moments,
            retain_simplices=True,
            budget=budget,
            preparation=preparation,
        )
        retained = budget.retained_basis_bytes
        second = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            second_moments=second_moments,
            retain_simplices=True,
            budget=budget,
            preparation=preparation,
        )
        assert budget.retained_basis_bytes == retained
    assert isinstance(first, ExactTetrahedronOverlap)
    assert isinstance(second, ExactTetrahedronOverlap)
    assert first.volume == second.volume == Fraction(1, 6)
    assert first.first_moment == second.first_moment
    assert first.second_moment == second.second_moment
    assert first.simplices == second.simplices
    assert 0 < second.operation_count < first.operation_count
    assert retained > 0


def test_earned_tetrahedron_planes_bind_full_coordinates_and_request(
    unit_tetrahedron: Tetrahedron,
) -> None:
    translated = tuple((point[0] + 2, point[1], point[2]) for point in unit_tetrahedron)
    second: Tetrahedron = (translated[0], translated[1], translated[2], translated[3])
    with coordinate_reference_budget() as budget:
        preparation = TetrahedronPlanePreparation(budget)
        first = tetrahedron_overlap(
            unit_tetrahedron, unit_tetrahedron, budget=budget, preparation=preparation
        )
        distinct = tetrahedron_overlap(
            unit_tetrahedron, second, budget=budget, preparation=preparation
        )
        assert isinstance(first, ExactTetrahedronOverlap)
        assert isinstance(distinct, ExactTetrahedronOverlap)
        assert distinct.volume == 0
        assert distinct.first_moment == (Fraction(0),) * 3
    with coordinate_reference_budget() as fresh:
        with pytest.raises(ValueError, match="current request budget"):
            tetrahedron_overlap(
                unit_tetrahedron,
                unit_tetrahedron,
                budget=fresh,
                preparation=preparation,
            )


def test_earned_star_determinant_reuses_only_full_source_piece(
    unit_tetrahedron: Tetrahedron,
) -> None:
    with coordinate_reference_budget() as budget:
        preparation = StarTetrahedronPreparation(budget)
        first = preparation.determinant(unit_tetrahedron)
        work = budget.work_units
        retained = budget.retained_basis_bytes
        assert preparation.determinant(unit_tetrahedron) == first == 1
        assert budget.work_units == work
        assert budget.retained_basis_bytes == retained
        reversed_tet = (
            unit_tetrahedron[0],
            unit_tetrahedron[1],
            unit_tetrahedron[3],
            unit_tetrahedron[2],
        )
        assert preparation.determinant(reversed_tet) == -1
        assert budget.work_units > work
