#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Chart-factored embedded Gram decisions agree with the polynomial route."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from phydrax.discretization import _cell_geometry_validity as validity
from phydrax.discretization import PolynomialComposedCellGeometryElement
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_enclosure import expression_evaluate
from tests._support.coordinate_references import coordinate_reference_budget


# Dual-quad corner charts of one triangle (corner, edge midpoint, centroid,
# edge midpoint), as extracted from triangle scaffolds.
_DUAL_NUMERATORS = (
    ((0, 0), (1, 0), (1, 1), (0, 1)),
    ((1, 0), (1, 1), (1, 1), (1, 0)),
    ((0, 1), (0, 1), (1, 1), (1, 1)),
)
_DUAL_DENOMINATORS = (
    ((1, 1), (2, 1), (3, 3), (1, 2)),
    ((1, 1), (2, 2), (3, 3), (2, 1)),
    ((1, 1), (1, 2), (3, 3), (2, 2)),
)


def _curved_root_bank() -> tuple[tuple[Fraction, ...], ...]:
    root = coordinate_lagrange_element("quadrilateral", 2)
    nodes = np.asarray(root.reference_nodes)
    return tuple(
        (
            Fraction(float(x)),
            Fraction(float(y)),
            Fraction(float(x)) ** 2 / 4 + Fraction(float(x)) * Fraction(float(y)) / 8,
        )
        for x, y in nodes
    )


def _dual_quads(
    triangle: tuple[tuple[Fraction, Fraction], ...], budget
) -> tuple[PolynomialComposedCellGeometryElement, ...]:
    root = coordinate_lagrange_element("quadrilateral", 2)
    source = PolynomialComposedCellGeometryElement(
        root,
        coordinate_lagrange_element("triangle", 1),
        np.asarray([[value.numerator for value in row] for row in triangle], dtype=object),
        np.asarray(
            [[value.denominator for value in row] for row in triangle], dtype=object
        ),
        budget=budget,
    )
    return tuple(
        PolynomialComposedCellGeometryElement(
            source,
            coordinate_lagrange_element("quadrilateral", 1),
            np.asarray(numerators, dtype=np.int64),
            np.asarray(denominators, dtype=np.int64),
            budget=budget,
        )
        for numerators, denominators in zip(
            _DUAL_NUMERATORS, _DUAL_DENOMINATORS, strict=True
        )
    )


def test_embedded_factor_decisions_match_and_enclose_the_polynomial_route() -> None:
    triangle = (
        (Fraction(1, 8), Fraction(1, 8)),
        (Fraction(7, 8), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(3, 4)),
    )
    bank = _curved_root_bank()
    policy = validity.CellValidityPolicy()
    grid = tuple(Fraction(index, 4) for index in range(5))
    with coordinate_reference_budget() as budget:
        reference = validity._prepare_polynomial_reference_domain(
            "quadrilateral", 2, 2, True
        )
        assert reference is not None
        for quad in _dual_quads(triangle, budget):
            factors = validity._polynomial_chart_factors(quad, 2, budget, embedded=True)
            assert factors is not None
            fast = validity._certify_factored_cell(
                factors, quad, bank, reference, 2, policy, budget
            )
            generic = validity._certify_polynomial_cell(
                quad, bank, reference, 2, True, policy, budget
            )
            assert fast is not None
            assert fast.status == generic.status == validity.CellValidityStatus.CERTIFIED_VALID
            assert fast.depth == 0 and fast.reasons == ()
            gram = validity._prepare_polynomial_differential(
                quad, bank, "quadrilateral", 2, True, budget, retain=False
            ).polynomial
            assert gram is not None
            values = [
                expression_evaluate(gram, point, budget=budget)
                for point in product(grid, repeat=2)
            ]
            # Both enclosures contain every exact Gram value on the cell.
            for lower, upper in ((fast.lower, fast.upper), (generic.lower, generic.upper)):
                assert all(Fraction(lower) <= value <= Fraction(upper) for value in values)
            # The block route takes the factored decision for every cell.
            block = validity._certify_polynomial_block(
                quad, "quadrilateral", (bank,), policy, budget=budget
            )
            assert block.status.tolist() == [fast.status]
            assert block.lower.tolist() == [fast.lower]
            assert block.upper.tolist() == [fast.upper]


def test_embedded_factors_decline_charts_leaving_their_source_cell() -> None:
    # The triangle chart's third corner leaves the root knot span.
    triangle = (
        (Fraction(1, 8), Fraction(1, 8)),
        (Fraction(7, 8), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(5, 4)),
    )
    bank = _curved_root_bank()
    policy = validity.CellValidityPolicy()
    with coordinate_reference_budget() as budget:
        for quad in _dual_quads(triangle, budget):
            assert validity._polynomial_chart_factors(
                quad, 2, budget, embedded=True
            ) is None
            block = validity._certify_polynomial_block(
                quad, "quadrilateral", (bank,), policy, budget=budget
            )
            reference = validity._prepare_polynomial_reference_domain(
                "quadrilateral", 2, 2, True
            )
            assert reference is not None
            generic = validity._certify_polynomial_cell(
                quad, bank, reference, 2, True, policy, budget
            )
            assert block.status.tolist() == [generic.status]
            assert block.lower.view(np.uint64).tolist() == [
                np.float64(generic.lower).view(np.uint64)
            ]


def test_volume_factors_enclose_curved_root_and_reuse_its_determinant() -> None:
    root = coordinate_lagrange_element("hexahedron", 2)
    nodes = tuple(tuple(Fraction(float(value)) for value in row) for row in root.reference_nodes)
    bank = tuple((x, y, z + x * x * y / 8) for x, y, z in nodes)
    chart = coordinate_lagrange_element("hexahedron", 1)
    corners = tuple(
        tuple(Fraction(1, 8) + Fraction(float(value)) / 2 for value in row)
        for row in chart.reference_nodes
    )
    policy = validity.CellValidityPolicy()
    with coordinate_reference_budget() as budget:
        element = PolynomialComposedCellGeometryElement(
            root,
            chart,
            np.asarray([[value.numerator for value in row] for row in corners]),
            np.asarray([[value.denominator for value in row] for row in corners]),
            budget=budget,
        )
        factors = validity._polynomial_chart_factors(element, 3, budget, embedded=False)
        assert factors is not None
        reference = validity._prepare_polynomial_reference_domain("hexahedron", 3, 2, False)
        assert reference is not None
        fast = validity._certify_factored_cell(
            factors, element, bank, reference, 3, policy, budget
        )
        assert fast is not None
        assert fast.status == validity.CellValidityStatus.CERTIFIED_VALID
        assert fast.lower == fast.upper == 0.125
        assert len(budget.volume_determinant_bounds_cache) == 1
        before = budget.work_units
        validity.coordinate_enclosure_owner.coordinate_preparation_key(root, bank, budget=budget)
        authentication_work = budget.work_units - before
        before = budget.work_units
        assert validity._root_differential_bounds(factors, bank, 3, policy, budget) == (
            Fraction(1), Fraction(1)
        )
        assert budget.work_units - before == authentication_work
        assert validity._certify_factored_cell(
            factors,
            element,
            bank,
            reference,
            3,
            validity.CellValidityPolicy(maximum_bernstein_nodes=1),
            budget,
        ) is None
        generic = validity._certify_polynomial_cell(
            element, bank, reference, 3, False, policy, budget
        )
        assert generic.status == fast.status
        determinant = validity._prepare_polynomial_differential(
            element, bank, "hexahedron", 3, False, budget, retain=False
        ).polynomial
        assert determinant is not None
        for point in product((Fraction(0), Fraction(1, 2), Fraction(1)), repeat=3):
            value = expression_evaluate(determinant, point, budget=budget)
            assert Fraction(fast.lower) <= value <= Fraction(fast.upper)
        inverted = tuple((x, y, -z) for x, y, z in bank)
        assert validity._certify_factored_cell(
            factors, element, inverted, reference, 3, policy, budget
        ) is None


@pytest.mark.parametrize("chart_kind", ("anisotropic", "reversed", "permuted", "tiny"))
@pytest.mark.parametrize("cancellation", (False, True))
def test_box_factor_operands_bound_the_independent_expanded_child(
    chart_kind: str, cancellation: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = coordinate_lagrange_element("hexahedron", 2)
    nodes = tuple(tuple(Fraction(float(value)) for value in row) for row in root.reference_nodes)
    bank = tuple(
        (x, y, z + (
            (x - Fraction(1, 2)) ** 2 * (y - Fraction(1, 2)) / 8
            if cancellation else x * x * y / 8
        ))
        for x, y, z in nodes
    )
    chart = coordinate_lagrange_element("hexahedron", 1)
    corners = []
    for row in chart.reference_nodes:
        x, y, z = (Fraction(float(value)) for value in row)
        if chart_kind == "reversed":
            point = (Fraction(3, 4) - x / 2, Fraction(3, 4) - y / 3, Fraction(1, 8) + z / 5)
        elif chart_kind == "permuted":
            point = (Fraction(1, 8) + y / 2, Fraction(1, 8) + z / 3, Fraction(1, 8) + x / 5)
        elif chart_kind == "tiny":
            point = tuple(Fraction(1, 4) + value / (1 << 20) for value in (x, y, z))
        else:
            point = (Fraction(1, 8) + x / 2, Fraction(1, 8) + y / 3, Fraction(1, 8) + z / 5)
        corners.append(point)
    policy = validity.CellValidityPolicy()
    with coordinate_reference_budget() as budget:
        element = PolynomialComposedCellGeometryElement(
            root, chart,
            np.asarray([[value.numerator for value in row] for row in corners]),
            np.asarray([[value.denominator for value in row] for row in corners]),
            budget=budget,
        )
        reference = validity._prepare_polynomial_reference_domain("hexahedron", 3, 2, False)
        assert reference is not None
        factors = validity._polynomial_chart_factors(element, 3, budget, embedded=False)
        assert factors is not None and factors.axis_action is not None
        oracle = validity._certify_polynomial_cell(
            element, bank, reference, 3, False, policy, budget
        )
        jacobian = validity._polynomial_jacobian(element, bank, "hexahedron", 3, budget)
        assert jacobian is not None
        original_magnitudes = validity._expression_magnitudes(jacobian, "box", 3, budget)
        reused_magnitudes = validity._restricted_box_jacobian_magnitudes(
            factors, bank, 3, policy, budget
        )
        assert original_magnitudes is not None and reused_magnitudes is not None
        assert all(
            reused >= original
            for reused_row, original_row in zip(reused_magnitudes, original_magnitudes, strict=True)
            for reused, original in zip(reused_row, original_row, strict=True)
        )
    # A fresh consumer must earn its own root operands, not inherit the oracle.
    with coordinate_reference_budget() as budget:
        factors = validity._polynomial_chart_factors(element, 3, budget, embedded=False)
        assert factors is not None
        original_jacobian = validity._polynomial_jacobian

        def root_only(actual, *args, **kwargs):
            assert actual is not element, "A conclusive root bound expanded the child again"
            return original_jacobian(actual, *args, **kwargs)

        monkeypatch.setattr(validity, "_polynomial_jacobian", root_only)
        result = validity._certify_factored_cell(
            factors, element, bank, reference, 3, policy, budget
        )
        assert result is not None
        assert result.status == oracle.status == validity.CellValidityStatus.CERTIFIED_VALID
        assert result.lower <= oracle.lower <= oracle.upper <= result.upper
        assert len(budget.root_jacobian_bernstein_operands_cache) == 1


def test_loose_root_scale_keeps_the_original_near_floor_child_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = coordinate_lagrange_element("hexahedron", 2)
    bank = tuple(
        (x, y, z + 10 * x * x)
        for x, y, z in (
            tuple(Fraction(float(value)) for value in row) for row in root.reference_nodes
        )
    )
    chart = coordinate_lagrange_element("hexahedron", 1)
    corners = tuple(
        (Fraction(float(x)) / 100, Fraction(float(y)), Fraction(float(z)))
        for x, y, z in chart.reference_nodes
    )
    policy = validity.CellValidityPolicy(relative_determinant_floor=0.1)
    with coordinate_reference_budget() as budget:
        element = PolynomialComposedCellGeometryElement(
            root, chart,
            np.asarray([[value.numerator for value in row] for row in corners]),
            np.asarray([[value.denominator for value in row] for row in corners]),
            budget=budget,
        )
        reference = validity._prepare_polynomial_reference_domain("hexahedron", 3, 2, False)
        assert reference is not None
        factors = validity._polynomial_chart_factors(element, 3, budget, embedded=False)
        assert factors is not None
        magnitude = validity._restricted_box_jacobian_magnitudes(
            factors, bank, 3, policy, budget
        )
        assert magnitude is not None
        assert factors.lower ** 2 <= validity._squared_threshold(magnitude, 3, False, policy)
        original = validity._polynomial_jacobian
        children = []

        def capture(actual, *args, **kwargs):
            if actual is element:
                children.append(actual)
            return original(actual, *args, **kwargs)

        monkeypatch.setattr(validity, "_polynomial_jacobian", capture)
        result = validity._certify_factored_cell(
            factors, element, bank, reference, 3, policy, budget
        )
        assert children
        assert result is not None
        oracle = validity._certify_polynomial_cell(
            element, bank, reference, 3, False, policy, budget
        )
        assert result.status == oracle.status == validity.CellValidityStatus.CERTIFIED_VALID


def test_box_root_operands_reauthenticate_a_mutated_public_fraction_bank() -> None:
    root = coordinate_lagrange_element("hexahedron", 2)
    bank = tuple(
        (x, y, z + x * x * y / 8)
        for x, y, z in (
            tuple(Fraction(float(value)) for value in row) for row in root.reference_nodes
        )
    )
    chart = coordinate_lagrange_element("hexahedron", 1)
    corners = np.asarray(chart.reference_nodes, dtype=np.int64)
    policy = validity.CellValidityPolicy()
    with coordinate_reference_budget() as budget:
        element = PolynomialComposedCellGeometryElement(
            root, chart, corners, np.full_like(corners, 2), budget=budget
        )
        factors = validity._polynomial_chart_factors(element, 3, budget, embedded=False)
        assert factors is not None
        first = validity._restricted_box_jacobian_magnitudes(
            factors, bank, 3, policy, budget
        )
        assert first is not None
        for _, _, value in bank:
            object.__setattr__(value, "_numerator", 2 * value.numerator)
        second = validity._restricted_box_jacobian_magnitudes(
            factors, bank, 3, policy, budget
        )
        assert second is not None
        assert second[2][2] == 2 * first[2][2]
        assert len(budget.root_jacobian_bernstein_operands_cache) == 2
        assert validity._root_differential_bounds(factors, bank, 3, policy, budget) == (
            Fraction(2), Fraction(2)
        )
