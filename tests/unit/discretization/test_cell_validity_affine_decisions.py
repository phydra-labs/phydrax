#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact affine-simplex validity decisions equal the polynomial route bitwise."""

import sys
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import _cell_geometry_validity as validity
from tests._support.coordinate_references import coordinate_reference_budget


D = phx.discretization


@pytest.mark.parametrize(
    ("kind", "matrix"),
    (
        ("quadrilateral", ((1.0, 0.25), (0.0, 2.0))),
        ("quadrilateral", ((1.0, 0.25), (0.0, 2.0), (0.125, 0.5))),
        ("hexahedron", ((1.0, 0.25, 0.0), (0.0, 2.0, 0.125), (0.0, 0.0, 0.5))),
    ),
)
def test_exact_affine_q1_box_decisions_equal_expanded_polynomials(
    kind: str, matrix: tuple[tuple[float, ...], ...]
) -> None:
    element = D.coordinate_lagrange_element(kind, 1)
    reference = np.asarray(element.reference_nodes)
    points = reference @ np.asarray(matrix).T + 0.125
    reversed_ = points.copy()
    reversed_[:, 0] *= -1
    warped = points.copy()
    warped[2, -1] += 0.125
    banks = tuple(
        tuple(tuple(Fraction(float(value)) for value in row) for row in values)
        for values in (points, reversed_, warped, np.zeros_like(points))
    )
    policy = D.CellValidityPolicy()
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        fast = validity._certify_polynomial_block(
            element, kind, banks, policy, budget=budget
        )
        fast_work = budget.work_units - before
    with coordinate_reference_budget() as budget, pytest.MonkeyPatch.context() as patch:
        patch.setattr(validity, "_affine_box_differential", lambda *args: None)
        before = budget.work_units
        generic = validity._certify_polynomial_block(
            element, kind, banks, policy, budget=budget
        )
        generic_work = budget.work_units - before
    _require_bitwise(fast, generic)
    assert fast_work < generic_work


@pytest.mark.parametrize(
    ("kind", "matrix"),
    (
        ("quadrilateral", ((1.0, 0.25), (0.0, 2.0))),
        ("quadrilateral", ((1.0, 0.25), (0.0, 2.0), (0.125, 0.5))),
        ("hexahedron", ((1.0, 0.25, 0.0), (0.0, 2.0, 0.125), (0.0, 0.0, 0.5))),
    ),
)
def test_affine_q1_block_reuses_canonical_frame_decisions_without_source_loss(
    kind: str, matrix: tuple[tuple[float, ...], ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    element = D.coordinate_lagrange_element(kind, 1)
    points = np.asarray(element.reference_nodes) @ np.asarray(matrix).T
    arrays = tuple(points + shift for shift in range(8))
    bank = tuple(
        tuple(Fraction(float(value)) for value in row)
        for values in arrays for row in values
    )
    route = np.arange(len(bank), dtype=np.int64).reshape(8, element.local_dof_count)
    policy = D.CellValidityPolicy()
    frames = []
    original = validity._affine_simplex_block
    preparations = []
    prepare = validity._prepare_affine_box_corner_action

    def record_prepare(*args: Any) -> Any:
        preparations.append(args[0].element_id)
        return prepare(*args)

    monkeypatch.setattr(validity, "_prepare_affine_box_corner_action", record_prepare)

    def record(*args: Any, **kwargs: Any) -> Any:
        frames.append((args[2].shape[0], len(args[3])))
        return original(*args, **kwargs)

    monkeypatch.setattr(validity, "_affine_simplex_block", record)
    with coordinate_reference_budget() as budget:
        fast = validity._affine_box_block(element, kind, route, bank, policy, budget)
        fast_work = budget.work_units
    assert preparations == [element.element_id]
    with coordinate_reference_budget() as budget:
        oracle = validity._certify_polynomial_block(
            element, kind,
            tuple(tuple(bank[int(index)] for index in row) for row in route),
            policy, budget=budget,
        )
        oracle_work = budget.work_units
    assert fast is not None
    _require_bitwise(fast, oracle)
    assert fast_work < oracle_work
    assert frames == [(1, element.topological_dimension + 1)]

    # The unused frame corners still determine the Q1 law: a single warped
    # coefficient must decline rather than certifying its affine frame only.
    warped = list(bank)
    row = route[-1, 2]
    warped[row] = (*warped[row][:-1], warped[row][-1] + Fraction(1, 8))
    with coordinate_reference_budget() as budget:
        assert validity._affine_box_block(
            element, kind, route, tuple(warped), policy, budget
        ) is None
    changed_rows = route.copy()
    changed_rows[-1, 2] = route[0, 2]
    with coordinate_reference_budget() as budget:
        assert validity._affine_box_block(
            element, kind, changed_rows, bank, policy, budget
        ) is None
        assert validity._affine_box_block(
            D.coordinate_lagrange_element(kind, 2), kind, route, bank, policy, budget
        ) is None

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    with coordinate_reference_budget(
        phx.meshing.MeshingLimits(maximum_work_units=fast_work)
    ) as budget:
        bounded = validity._affine_box_block(element, kind, route, bank, policy, budget)
        assert bounded is not None
        _require_bitwise(bounded, fast)
        assert budget.work_units == fast_work
    with coordinate_reference_budget(
        phx.meshing.MeshingLimits(maximum_work_units=fast_work - 1)
    ) as budget, pytest.raises(CoordinateEnclosureResourceError):
        validity._affine_box_block(element, kind, route, bank, policy, budget)


def _dyadic_cells(
    seed: int, count: int, corners: int, ambient: int, extra: Any
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = np.round(rng.uniform(-4.0, 4.0, size=(count * corners, ambient)) * 64.0)
    points = np.concatenate((points / 64.0, np.asarray(extra, dtype=np.float64)))
    cells = np.arange(points.shape[0], dtype=np.int32).reshape(-1, corners)
    return points, cells


def _decisions(
    mesh: Any, kind: str, policy: Any
) -> tuple[Any, Any, np.ndarray]:
    geometry = D.CellGeometrySpec.affine(mesh)
    with coordinate_reference_budget() as budget:
        elements, routes, _ = geometry.resolve(mesh, budget=budget)
        bank = geometry.source_coordinates(budget=budget)
        route = np.asarray(routes[0], dtype=np.int64)
        element = validity.cell_geometry_owner.require_scalar_coordinate_element(
            elements[0], "Affine decision fixture"
        )
        batch = validity._affine_simplex_block(
            element, kind, route, bank, policy, budget
        )
    # Keep this an independent expanded-polynomial reference, not the shared
    # scalar primitive now also used by differential preparation.
    with coordinate_reference_budget() as budget, pytest.MonkeyPatch.context() as patch:
        patch.setattr(validity, "_affine_source_differential", lambda *args: None)
        patch.setattr(validity, "_affine_corner_differential", lambda *args: None)
        patch.setattr(validity, "_affine_composed_differential", lambda *args: None)
        oracle = validity._certify_polynomial_block(
            element,
            kind,
            tuple(tuple(bank[index] for index in row) for row in route),
            policy,
            budget=budget,
        )
    assert batch is not None
    return batch, oracle, route


def _require_bitwise(batch: Any, oracle: Any) -> None:
    np.testing.assert_array_equal(batch.status, oracle.status)
    np.testing.assert_array_equal(batch.depth, oracle.depth)
    np.testing.assert_array_equal(batch.lower.view(np.uint64), oracle.lower.view(np.uint64))
    np.testing.assert_array_equal(batch.upper.view(np.uint64), oracle.upper.view(np.uint64))
    assert batch.reasons == oracle.reasons


@pytest.mark.parametrize(
    ("kind", "corners", "ambient", "degenerate"),
    (
        ("triangle", 3, 2, ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))),
        (
            "tetrahedron",
            4,
            3,
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        ),
        ("triangle", 3, 3, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))),
    ),
    ids=("planar-triangles", "tetrahedra", "embedded-triangles"),
)
def test_affine_decisions_match_polynomial_route(
    kind: str, corners: int, ambient: int, degenerate: Any
) -> None:
    points, cells = _dyadic_cells(7, 48, corners, ambient, degenerate)
    mesh = D.CellMesh(points, (D.CellBlock("cells", kind, cells),))
    batch, oracle, _ = _decisions(mesh, kind, D.CellValidityPolicy())
    _require_bitwise(batch, oracle)
    statuses = set(np.asarray(batch.status).tolist())
    assert int(D.CellValidityStatus.INVALID) in statuses
    assert int(D.CellValidityStatus.CERTIFIED_VALID) in statuses
    published = D.certify_cell_geometry_validity(mesh)
    np.testing.assert_array_equal(np.asarray(published.status), batch.status)
    np.testing.assert_array_equal(
        np.asarray(published.determinant_lower).view(np.uint64),
        batch.lower.view(np.uint64),
    )


def test_floor_equality_keeps_the_subdivided_polynomial_decision() -> None:
    # sin^2 of the corner angle equals the floor: Gram = floor * |e0|^2 |e1|^2.
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
    mesh = D.CellMesh(points, (D.CellBlock("cells", "triangle", cells),))
    policy = D.CellValidityPolicy(
        maximum_subdivision_depth=2, relative_determinant_floor=0.5
    )
    batch, oracle, _ = _decisions(mesh, "triangle", policy)
    _require_bitwise(batch, oracle)
    assert int(batch.status[0]) == D.CellValidityStatus.UNRESOLVED
    assert int(batch.status[1]) == D.CellValidityStatus.CERTIFIED_VALID


def test_translated_simplex_cells_share_exact_jacobian_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triangle = np.asarray(((0.0, 0.0), (0.5, 0.0), (0.0, 0.25)))
    points = np.concatenate(
        (triangle, triangle + (2, 0), triangle + (4, 1), triangle[(0, 2, 1), :] + (6, 0))
    )
    cells = np.arange(12, dtype=np.int32).reshape(-1, 3)
    mesh = D.CellMesh(points, (D.CellBlock("cells", "triangle", cells),))
    original = validity._simplex_determinant
    decisions = 0

    def determinant(rows: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
        nonlocal decisions
        decisions += 1
        return original(rows)

    monkeypatch.setattr(validity, "_simplex_determinant", determinant)
    batch, oracle, _ = _decisions(mesh, "triangle", D.CellValidityPolicy())
    _require_bitwise(batch, oracle)
    assert decisions == 2
    np.testing.assert_array_equal(
        batch.status,
        (
            D.CellValidityStatus.CERTIFIED_VALID,
            D.CellValidityStatus.CERTIFIED_VALID,
            D.CellValidityStatus.CERTIFIED_VALID,
            D.CellValidityStatus.INVALID,
        ),
    )


def test_embedded_triangles_form_each_shared_edge_norm_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A closed dyadic octahedron: every edge is shared by two cells, and the
    # polarized Gram decision must equal the expanded polynomial route.
    rng = np.random.default_rng(11)
    base = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.float64,
    )
    points = base + np.round(rng.uniform(-0.25, 0.25, size=base.shape) * 64.0) / 64.0
    cells = np.asarray(
        (
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ),
        dtype=np.int32,
    )
    mesh = D.CellMesh(points, (D.CellBlock("cells", "triangle", cells),))
    original = validity.reserve_polynomial
    norms = 0

    def reserve(work: int, *args: Any, **kwargs: Any) -> None:
        nonlocal norms
        norms += int(sys._getframe(1).f_code.co_name == "edge_norm")
        original(work, *args, **kwargs)

    monkeypatch.setattr(validity, "reserve_polynomial", reserve)
    batch, oracle, _ = _decisions(mesh, "triangle", D.CellValidityPolicy())
    _require_bitwise(batch, oracle)
    assert norms == 12
    assert set(np.asarray(batch.status).tolist()) == {
        int(D.CellValidityStatus.CERTIFIED_VALID)
    }


def test_binary64_floor_filter_preserves_exact_bounds_and_near_floor_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    equality = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))
    )
    near = equality.copy()
    near[3, 2] = np.nextafter(1.0, np.inf)
    points = np.concatenate((unit, equality, near, unit[(0, 2, 1, 3), :]))
    mesh = D.CellMesh(
        points,
        (D.CellBlock("cells", "tetrahedron", np.arange(16).reshape(4, 4)),),
    )
    original = validity._binary64_simplex_floor
    filters = []

    def filtered(*args: Any, **kwargs: Any) -> np.ndarray:
        result = original(*args, **kwargs)
        filters.append(result)
        return result

    monkeypatch.setattr(validity, "_binary64_simplex_floor", filtered)
    policy = D.CellValidityPolicy(
        relative_determinant_floor=0.5, maximum_subdivision_depth=2
    )
    batch, oracle, _ = _decisions(mesh, "tetrahedron", policy)
    _require_bitwise(batch, oracle)
    assert len(filters) == 1
    np.testing.assert_array_equal(filters[0], (True, False, False, False))


def test_binary64_floor_filter_does_not_replace_a_nonrepresentable_source() -> None:
    points = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    with coordinate_reference_budget() as budget:
        result = validity._binary64_simplex_floor(
            points, np.asarray(((0, 1, 2, 3),)), 1.0e-12, budget=budget
        )
    np.testing.assert_array_equal(result, (False,))


@pytest.mark.parametrize("dimension", (2, 3))
def test_nonbinary_source_row_does_not_disable_other_exact_binary_cells(
    dimension: int,
) -> None:
    from phydrax.geometry._global_embedding import (
        source_bank_binary64_rows,
        source_bank_predicate_view,
    )

    points = (
        (Fraction(0),) * dimension,
        *(
            tuple(Fraction(int(axis == column)) for column in range(dimension))
            for axis in range(dimension)
        ),
        (Fraction(1, 3),) + (Fraction(0),) * (dimension - 1),
    )
    first = tuple(range(dimension + 1))
    second = (0, dimension + 1, *range(2, dimension + 1))
    with coordinate_reference_budget() as budget:
        result = validity._binary64_simplex_floor(
            points, np.asarray((first, second)), 1.0e-12, budget=budget
        )
        np.testing.assert_array_equal(result, (True, False))
        view, rows = source_bank_binary64_rows(points, budget=budget)
        np.testing.assert_array_equal(rows, (*((True,) * (dimension + 1)), False))
        again, again_rows = source_bank_binary64_rows(points, budget=budget)
        assert again is view and again_rows is rows
        exact, binary = source_bank_predicate_view(points, budget=budget)
        assert not binary and exact.dtype == object
        assert exact[-1, 0] == Fraction(1, 3)
