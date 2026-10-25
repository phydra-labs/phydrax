#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Facet-occurrence classes keep the exact power contact verdict and scope storage."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

import phydrax.geometry._exact_power_certificates as certificates
from phydrax.discretization import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    Facets,
    mesh_facets,
    MeshCertificateLimits,
)
from phydrax.geometry._exact_polyhedral_geometry import (
    triangle_contact,
    TriangleContactPreparation,
    triangulate_loop,
)
from tests._support.coordinate_references import coordinate_reference_budget


_POINTS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (Fraction(1, 4), Fraction(1, 4), -1),
    (1, 1, 1),
    (Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)),
    (Fraction(3, 2), Fraction(-1, 2), Fraction(1, 5)),
    (Fraction(-1, 2), Fraction(3, 2), Fraction(1, 5)),
    (Fraction(1, 5), Fraction(1, 5), Fraction(3, 2)),
)


def _outward_faces(points: np.ndarray, tet: tuple[int, ...]) -> list[np.ndarray]:
    faces = []
    for opposite in range(4):
        face = [vertex for index, vertex in enumerate(tet) if index != opposite]
        a, b, c = (points[index] for index in face)
        d = points[tet[opposite]]
        u, v, w = (
            tuple(q - p for p, q in zip(a, point, strict=True)) for point in (b, c, d)
        )
        determinant = (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )
        faces.append(np.asarray(face if determinant < 0 else face[::-1], dtype=np.int64))
    return faces


def _mesh(tets: tuple[tuple[int, ...], ...]) -> tuple[CellMesh, np.ndarray]:
    exact = np.asarray(
        [[Fraction(value) for value in point] for point in _POINTS], dtype=object
    )
    mesh = CellMesh.from_polyhedra(
        np.asarray(exact, dtype=np.float64),
        [_outward_faces(exact, tet) for tet in tets],
        block_name="tetrahedra",
    )
    return mesh, exact


def _occurrence_verdict(
    points: np.ndarray, facets: Facets, budget: CoordinateEnclosureBudget
) -> tuple[bool, int]:
    """Prior per-occurrence enumeration: every occurrence triangulated and paired."""
    triangles = []
    for occurrence, row in enumerate(facets.rows):
        loop = tuple(row[row >= 0].tolist())
        triangles.extend(
            (occurrence, triangle) for triangle in triangulate_loop(points, loop, budget=budget)
        )
    contact, tested = False, 0
    for (left, first), (right, second) in combinations(triangles, 2):
        if (
            facets.cells[left] == facets.cells[right]
            or facets.group[left] == facets.group[right]
        ):
            continue
        tested += 1
        contact |= triangle_contact(points, first, second, budget=budget)
    return contact, tested


@pytest.mark.parametrize(
    ("tets", "violated"),
    (
        (((0, 1, 2, 3), (0, 2, 1, 4)), False),
        (((0, 1, 2, 3), (0, 2, 1, 4), (1, 2, 3, 5)), False),
        (((0, 1, 2, 3), (0, 2, 1, 4), (6, 7, 8, 9)), True),
    ),
    ids=("shared-face", "fan", "piercing"),
)
def test_facet_classes_keep_contact_verdict_with_fewer_pairs_and_scoped_storage(
    tets: tuple[tuple[int, ...], ...], violated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, points = _mesh(tets)
    calls = []

    def counted(
        points: np.ndarray,
        first: tuple[int, ...],
        second: tuple[int, ...],
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparation: TriangleContactPreparation | None = None,
    ) -> bool:
        calls.append(None)
        return triangle_contact(
            points, first, second, budget=budget, preparation=preparation
        )

    monkeypatch.setattr(certificates, "triangle_contact", counted)
    state = EmbeddingState([], [])
    with coordinate_reference_budget() as budget:
        facets = mesh_facets(mesh, budget=budget)
        classes = certificates._facet_loop_classes(facets, budget=budget)
        temporary = budget.temporary_bytes_upper
        certified = certificates._certify_power_facets(
            state, mesh, points, facets, MeshCertificateLimits(), budget=budget
        )
        assert budget.temporary_bytes_upper == temporary
        expected, occurrence_pairs = _occurrence_verdict(points, facets, budget)
    interior = int(np.sum(facets.counts == 2))
    assert len(classes.representatives) == len(facets.rows) - interior
    assert certified
    assert expected is violated
    assert any(finding.check == "exact_polyhedral_contact" for finding in state.findings) is violated
    assert "exact_polyhedral_contacts" in state.checks
    assert len(calls) <= occurrence_pairs
    if interior > 1:
        # Interior facet pairs were enumerated through three occurrence pairs.
        assert len(calls) < occurrence_pairs


@pytest.mark.parametrize(
    ("first", "second", "same"),
    (
        ((0, 1, 2, 3), (2, 3, 0, 1), True),
        ((0, 1, 2, 3), (3, 2, 1, 0), True),
        ((0, 1, 2, 3), (1, 0, 3, 2), True),
        ((0, 1, 2, 3), (0, 2, 1, 3), False),
        ((0, 1, 2), (0, 1, 2, 3), False),
        ((0, 1, 2), (4, 1, 2), False),
    ),
)
def test_same_cycle_accepts_rotation_and_reversal_only(
    first: tuple[int, ...], second: tuple[int, ...], same: bool
) -> None:
    assert certificates._same_cycle(first, second) is same


@pytest.mark.parametrize(
    ("extra", "second", "contact"),
    (
        (((0, 0, 1),), (0, 1, 3), False),
        (((0, -1, 0),), (0, 1, 3), False),
        (((0, 2, 0),), (0, 1, 3), True),
        (((1, 0, 1), (0, 1, 1)), (0, 3, 4), False),
        (((1, 0, 1), (0, 1, -1)), (0, 3, 4), True),
        (((0, 0, 1), (1, 0, 1), (0, 1, 1)), (3, 4, 5), False),
        (((0, 0, 0), (0, -1, 0), (-1, 0, 0)), (3, 4, 5), True),
    ),
    ids=(
        "noncoplanar-edge", "coplanar-edge", "overlapping-shared-edge",
        "one-sided-vertex", "piercing-shared-vertex", "separated-planes",
        "unidentified-geometric-vertex",
    ),
)
@pytest.mark.parametrize("reverse", (False, True))
def test_triangle_contact_exact_shared_simplex_separation(
    extra: tuple[tuple[int, ...], ...],
    second: tuple[int, ...],
    contact: bool,
    reverse: bool,
) -> None:
    points = np.asarray(
        tuple(tuple(Fraction(value) for value in row) for row in (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), *extra,
        )),
        dtype=object,
    )
    first = (0, 1, 2)
    if reverse:
        first, second = second[::-1], first[::-1]
    with coordinate_reference_budget() as budget:
        assert triangle_contact(points, first, second, budget=budget) is contact


def test_triangle_separation_matches_full_intersection_on_rational_lattice() -> None:
    """The reference always constructs edge hits, without separation shortcuts."""
    from random import Random

    import phydrax.geometry._exact_polyhedral_geometry as exact
    import phydrax.geometry._planar_primitives as planar

    random = Random(20261020)
    compared = 0
    for planar_case in (False, True):
        for _ in range(80):
            points = np.asarray([
                tuple(Fraction(random.randint(-2, 2), 2) for _ in range(3))
                for _ in range(6)
            ], dtype=object)
            if planar_case:
                points[:, 2] = Fraction(0)
            first = (0, 1, 2)
            second = (
                0 if random.random() < 0.5 else 3,
                1 if random.random() < 0.5 else 4,
                5,
            )
            a, b = tuple(tuple(points[i]) for i in first), tuple(tuple(points[i]) for i in second)
            with coordinate_reference_budget() as budget:
                ka, kb = planar.plane_key(a, budget=budget), planar.plane_key(b, budget=budget)
                if ka is None or kb is None:
                    continue
                coplanar = all(
                    sum((ka[0][axis] * point[axis] for axis in range(3)), ka[0][3]) == 0
                    for point in b
                )
                positive_area = coplanar and planar.intersection(
                    planar.project(a, ka[1], budget=budget),
                    planar.project(b, ka[1], budget=budget),
                    budget=budget,
                ) > 0
                hits = set()
                for own, other, key in ((a, b, kb), (b, a, ka)):
                    exact._triangle_edge_hits(own, other, key, coplanar, hits, budget=budget)
                expected = positive_area or exact._contact_outside_shared(
                    points, tuple(sorted(set(first) & set(second))), hits
                )
                assert triangle_contact(points, first, second, budget=budget) is expected
                preparation = TriangleContactPreparation(budget)
                assert triangle_contact(
                    points, first, second, budget=budget, preparation=preparation
                ) is expected
            compared += 1
    assert compared > 100


def test_triangle_preparation_retains_geometry_not_incidence_or_mutable_banks() -> None:
    points = np.asarray([
        tuple(Fraction(value) for value in row) for row in (
            (0, 0, 0), (1, 0, 0), (0, 1, 0),
            (0, 0, 0), (0, -1, 0), (-1, 0, 0),
        )
    ], dtype=object)
    with coordinate_reference_budget() as budget:
        preparation = TriangleContactPreparation(budget)
        assert not triangle_contact(
            points, (0, 1, 2), (0, 4, 5), budget=budget, preparation=preparation
        )
        start, retained = budget.work_units, budget.retained_basis_bytes
        assert not triangle_contact(
            points, (0, 1, 2), (0, 4, 5), budget=budget, preparation=preparation
        )
        assert budget.work_units == start
        assert budget.retained_basis_bytes == retained
        assert triangle_contact(
            points, (0, 1, 2), (3, 4, 5), budget=budget, preparation=preparation
        )
        points[4] = (Fraction(1), Fraction(0), Fraction(0))
        points[5] = (Fraction(0), Fraction(1), Fraction(0))
        assert triangle_contact(
            points, (0, 1, 2), (0, 4, 5), budget=budget, preparation=preparation
        )
    assert retained > 0
