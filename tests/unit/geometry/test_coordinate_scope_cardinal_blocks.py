#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Block corner binding equals the per-cell coordinate-scope rule."""

from fractions import Fraction
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry import _global_embedding as embedding
from phydrax.geometry._rational import rational_vector
from tests._support.coordinate_references import coordinate_reference_budget


D = phx.discretization
TRIANGLES = np.asarray(((0, 1, 2), (1, 3, 2), (2, 3, 4)), dtype=np.int64)
BANK = (
    (Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(1)),
    (Fraction(1, 3), Fraction(2)),
)


def _inputs(route: np.ndarray, bank: Any, points: np.ndarray) -> Any:
    mesh = D.CellMesh(points, (D.CellBlock("cells", "triangle", TRIANGLES),))
    geometry = D.CellGeometrySpec.affine(mesh)
    inputs = embedding._CoordinateScopeInputs(
        tuple(geometry.elements),
        (route,),
        points,
        points,
        bank,
        (TRIANGLES,),
        (np.ones(TRIANGLES.shape, dtype=np.bool_),),
    )
    return geometry, inputs


def _per_cell(geometry: Any, inputs: Any, target: Any, budget: Any) -> list[int]:
    element = inputs.elements[0]
    reference = tuple(
        rational_vector(point) for point in reference_cell_topology("triangle").vertices
    )
    mapped = []
    for cell, row in enumerate(inputs.routes[0]):
        local = tuple(inputs.source_values[index] for index in row)
        with budget.temporary_scope():
            prepared = embedding._prepare_coordinate_cell_map(
                element, "triangle", local, budget=budget
            )
            assert prepared is not None
            nonbinary = embedding._bind_coordinate_cell_corners(
                prepared,
                "triangle",
                inputs.vertices[0][cell],
                reference,
                inputs,
                target,
                geometry,
                budget=budget,
            )
            if embedding._coordinate_cell_is_mapped(
                prepared, "triangle", nonbinary, geometry, budget=budget
            ):
                mapped.append(cell)
    return mapped


def _outcome(call: Any) -> tuple[str, Any]:
    try:
        return "value", call()
    except ValueError as error:
        return "error", str(error)


def _rounded(bank: Any) -> np.ndarray:
    return np.asarray([[float(value) for value in row] for row in bank], dtype=np.float64)


def _bound(target: Any) -> dict[int, Any]:
    images = (target.get(vertex) for vertex in range(target.identity.size))
    return {vertex: image for vertex, image in enumerate(images) if image is not None}


@pytest.mark.parametrize(
    "case", ("consistent", "shared-conflict", "carrier-mismatch")
)
def test_block_binding_matches_per_cell_rule(case: str) -> None:
    route = TRIANGLES.copy()
    bank = BANK
    points = _rounded(BANK)
    if case == "shared-conflict":
        # Vertex 2 of the last cell is routed to another source row with the
        # same binary64 rounding but a different exact value.
        bank = (*BANK, (Fraction(0), Fraction(1) + Fraction(1, 2**80)))
        route[2, 0] = 5
    elif case == "carrier-mismatch":
        points = points.copy()
        points[3, 0] = np.nextafter(points[3, 0], 2.0)
    geometry, inputs = _inputs(route, bank, points)
    element = inputs.elements[0]
    with coordinate_reference_budget() as budget:
        expected_target = embedding._CornerBindings(
            inputs.source_values, points.shape[0], budget=budget
        )
        expected = _outcome(lambda: _per_cell(geometry, inputs, expected_target, budget))
        actual_target = embedding._CornerBindings(
            inputs.source_values, points.shape[0], budget=budget
        )
        actual = _outcome(
            lambda: embedding._bind_cardinal_simplex_block(
                element,
                "triangle",
                route,
                TRIANGLES,
                inputs,
                actual_target,
                geometry,
                budget=budget,
            )
        )
    if expected[0] == "value":
        assert actual[0] == "value"
        assert actual[1].tolist() == expected[1]
        assert _bound(actual_target) == _bound(expected_target)
    else:
        assert actual == expected
    if case == "consistent":
        # The 1/3 coordinate is not binary64, so its cell is a mapped cell.
        assert expected == ("value", [2])


@pytest.mark.parametrize("case", ("consistent", "shared-conflict", "carrier-mismatch"))
def test_affine_chart_block_binding_matches_per_cell_rule(case: str) -> None:
    route = TRIANGLES.copy()
    bank = BANK
    points = _rounded(bank)
    if case == "shared-conflict":
        bank = (*BANK, (Fraction(0), Fraction(1) + Fraction(1, 2**80)))
        route[2, 0] = 5
    elif case == "carrier-mismatch":
        points[3, 0] = np.nextafter(points[3, 0], 2.0)
    geometry, original = _inputs(route, bank, points)
    # A nontrivial coefficient action, with compensating source routing, has
    # the same physical corner law but cannot use the cardinal-block shortcut.
    element = BarycentricCellGeometryElement(
        original.elements[0],
        np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=np.float64),
    )
    route = route[:, (2, 0, 1)]
    inputs = embedding._CoordinateScopeInputs(
        (element,),
        (route,),
        points,
        points,
        bank,
        (TRIANGLES,),
        original.vertex_valid,
    )
    with coordinate_reference_budget() as budget:
        expected_target = embedding._CornerBindings(bank, points.shape[0], budget=budget)
        expected = _outcome(lambda: _per_cell(geometry, inputs, expected_target, budget))
        actual_target = embedding._CornerBindings(bank, points.shape[0], budget=budget)
        actual = _outcome(
            lambda: embedding._bind_affine_simplex_block(
                element,
                "triangle",
                route,
                TRIANGLES,
                inputs,
                actual_target,
                geometry,
                budget=budget,
            )
        )
    assert actual[0] == expected[0]
    if expected[0] == "value":
        assert actual[1].tolist() == expected[1]
        assert _bound(actual_target) == _bound(expected_target)
        assert actual[1].tolist() == [2]
    else:
        assert actual[1].split(";")[0] == expected[1].split(";")[0]


def test_prepared_vertex_images_bind_live_routes_not_only_layout_ids() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        prepared_coordinate_vertex_images,
    )

    mesh = D.CellMesh.from_triangles(
        np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    with coordinate_reference_budget() as budget:
        geometry = D.CellGeometrySpec.affine(mesh, budget=budget)
        first = prepared_coordinate_vertex_images(
            mesh, geometry, mismatch="unequal corners", budget=budget
        )
        assert first is not None
        work, retained = budget.work_units, budget.retained_basis_bytes
        assert prepared_coordinate_vertex_images(
            mesh, geometry, mismatch="unequal corners", budget=budget
        ) is first
        assert budget.work_units == work
        assert budget.retained_basis_bytes == retained
        changed = eqx.tree_at(
            lambda value: value.geometry_dofs,
            geometry,
            (geometry.geometry_dofs[0][:, (1, 0, 2)],),
        )
        assert changed.geometry_layout_id == geometry.geometry_layout_id
        second = prepared_coordinate_vertex_images(
            mesh, changed, mismatch="unequal corners", budget=budget
        )
        assert second is not None and second is not first
        assert second[0] == first[1] and second[1] == first[0]
        assert budget.work_units > work


def test_binary64_rows_reuse_exact_scalar_conversions_without_approximating_rows() -> None:
    exact = (Fraction(0), Fraction(1))
    nonbinary = (Fraction(1, 3), Fraction(1))
    bank = (exact,) * 64 + (nonbinary,) * 16
    with coordinate_reference_budget() as budget:
        view, mask = embedding.source_bank_binary64_rows(bank, budget=budget)
        first_work = budget.work_units
        assert embedding.source_bank_binary64_rows(bank, budget=budget)[0] is view
        assert budget.work_units == first_work
    np.testing.assert_array_equal(mask, (True,) * 64 + (False,) * 16)
    np.testing.assert_array_equal(view[:64], np.broadcast_to((0.0, 1.0), (64, 2)))
    assert bool(np.all(np.isnan(view[64:])))
    assert first_work < 3 * (64 * 2 + 16)


def test_vertex_images_sum_actual_operation_storage_without_changing_arithmetic_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        polynomial_coefficient_storage_upper,
    )

    count = 64
    arithmetic_calls = []
    original_reserve = CoordinateEnclosureBudget.reserve

    def reserve(self, work, storage_upper=0, **kwargs):
        if sys._getframe(1).f_code.co_name == "coordinate_vertex_images" and work == 4 * count:
            arithmetic_calls.append(work)
        return original_reserve(self, work, storage_upper, **kwargs)

    monkeypatch.setattr(CoordinateEnclosureBudget, "reserve", reserve)
    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1),
        np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1))),
    )
    routes = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
    low = tuple(
        point
        for cell in range(count)
        for point in (
            (Fraction(3 * cell), Fraction(0)),
            (Fraction(3 * cell + 1), Fraction(0)),
            (Fraction(3 * cell), Fraction(1)),
        )
    )
    tiny = Fraction(1, 2**3000)
    high = (*low[:-3], (tiny, Fraction(0)), (1 + tiny, Fraction(0)), low[-1])
    outcomes = []
    for bank in (low, high):
        with coordinate_reference_budget() as budget:
            images = algebra.coordinate_vertex_images(
                ((element, routes, routes),), bank, 3 * count,
                mismatch="unequal corners", budget=budget,
            )
            assert images is not None
            for cell, row in enumerate(routes):
                assert images[3 * cell] == bank[int(row[0])]
                assert images[3 * cell + 1] == tuple(
                    (first + second) / 2
                    for first, second in zip(bank[int(row[0])], bank[int(row[1])], strict=True)
                )
                assert images[3 * cell + 2] == bank[int(row[2])]
            outcomes.append((budget.work_units, budget.peak_bytes_upper))
    assert arithmetic_calls == [4 * count, 4 * count]
    largest_control = max(algebra._scalar_bits(value) for point in high for value in point)
    previous_global_bits = 2 * (largest_control + algebra._scalar_bits(Fraction(1, 2))) + 2
    previous_global_output_bound = 256 + 2 * count * polynomial_coefficient_storage_upper(
        0, previous_global_bits,
    )
    assert outcomes[1][1] < previous_global_output_bound


def _streamed_shared_vertex_case():
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
    )

    base = coordinate_lagrange_element("triangle", 1)
    first = BarycentricCellGeometryElement(
        base, np.asarray(((0.5, 0.5, 0), (0, 1, 0), (0, 0, 1))),
    )
    second = BarycentricCellGeometryElement(
        base, np.asarray(((0.25, 0.75, 0), (0, 1, 0), (0, 0, 1))),
    )
    bank = (
        (Fraction(0), Fraction(0)), (Fraction(2), Fraction(0)), (Fraction(0), Fraction(2)),
        (Fraction(-2), Fraction(0)), (Fraction(2), Fraction(0)), (Fraction(0), Fraction(2)),
    )
    blocks = (
        (first, np.asarray(((0, 1, 2),)), np.asarray(((0, 1, 2),))),
        (second, np.asarray(((3, 4, 5),)), np.asarray(((0, 3, 4),))),
    )
    return blocks, bank


def test_streamed_images_preserve_declared_vertices_and_refuse_r_minus_one() -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.geometry._embedding_state import MeshCertificateLimits

    blocks, bank = _streamed_shared_vertex_case()
    with coordinate_reference_budget() as budget:
        images = algebra.coordinate_vertex_images(
            blocks, bank, 6, mismatch="shared vertex conflict", budget=budget,
        )
        required_work, required_memory = budget.work_units, budget.peak_bytes_upper
    assert images is not None and images[0] == (Fraction(1), Fraction(0))
    assert images[1] is bank[1] and images[3] is bank[4]
    assert images[1] is not images[3]  # Equal coordinates never merge entity identity.
    assert images[5] is None
    for kind in ("work", "memory"):
        limits = MeshCertificateLimits(
            maximum_work_units=required_work - 1 if kind == "work" else required_work,
            maximum_scratch_bytes=required_memory - 1 if kind == "memory" else required_memory,
        )
        with coordinate_reference_budget(limits) as budget, pytest.raises(
            CoordinateEnclosureResourceError
        ):
            algebra.coordinate_vertex_images(
                blocks, bank, 6, mismatch="shared vertex conflict", budget=budget,
            )


def test_streamed_images_compare_full_exact_points_before_reusing_storage() -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra

    blocks, bank = _streamed_shared_vertex_case()
    changed = (*bank[:4], (bank[4][0] + Fraction(1, 2**200), bank[4][1]), bank[5])
    first = (bank[0][0] + bank[1][0]) / 2
    second = changed[3][0] / 4 + 3 * changed[4][0] / 4
    assert first != second and float(first) == float(second)
    with coordinate_reference_budget() as budget, pytest.raises(
        ValueError, match="shared vertex conflict"
    ):
        algebra.coordinate_vertex_images(
            blocks, changed, 6, mismatch="shared vertex conflict", budget=budget,
        )
