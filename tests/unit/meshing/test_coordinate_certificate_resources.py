#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible coordinate certificate resources contracts."""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    exact_orient2d,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization import _coordinate_enclosure as algebra
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    coefficient_profile,
    coordinate_corner_images,
    coordinate_expressions,
    multi_affine_coordinates,
)
from phydrax.geometry._mapped_coverage import (
    chart_flux,
    face_charts,
    nonnegative,
    SubdivisionLedger,
)
from tests._support.domain_coverage import mapped_split_square, split_square_domain
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    reference_budget as reference_budget,
    TETRAHEDRON,
)


@pytest.mark.parametrize("positive", (True, False))
def test_complete_tetrahedral_polynomial_support_subdivides_all_reference_axes(
    positive: bool,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    # The z dependence cannot be projected onto a triangle or replaced by
    # corner values: both signs have nonnegative source corner samples.
    margin = Fraction(1 if positive else -1, 64)
    work = SubdivisionLedger()
    ledger = coordinate_budget(100_000, 8 * 1024**2)
    with ledger.activate():
        z = algebra.axes(3, budget=ledger)[2]
        centered = algebra.add(
            z, algebra.constant(Fraction(-1, 2), 3, budget=ledger), budget=ledger
        )
        polynomial = algebra.add(
            algebra.multiply(centered, centered, budget=ledger),
            algebra.constant(margin, 3, budget=ledger),
            budget=ledger,
        )
        assert (
            nonnegative(polynomial, "simplex", 3, 64, 4, 1_000, work, budget=ledger)
            is positive
        )
    assert work.pieces > 1
    assert work.maximum_depth > 0
    assert ledger.work_units > 0


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_multi_affine_corner_coordinates_equal_the_source_basis_combination(
    kind: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    rng = np.random.default_rng(29)

    def bank(element: Any) -> tuple[tuple[Fraction, ...], ...]:
        rows = rng.uniform(-1.0, 1.0, (np.asarray(element.reference_nodes).shape[0], 3))
        return tuple(tuple(Fraction(float(value)) for value in row) for row in rows)

    linear, quadratic = (
        coordinate_lagrange_element(kind, 1),
        coordinate_lagrange_element(kind, 2),
    )
    local = bank(linear)
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        prepared = multi_affine_coordinates(linear, local, budget=ledger)
        assert multi_affine_coordinates(quadratic, bank(quadratic), budget=ledger) is None
    assert prepared is not None
    polynomials, corners = prepared
    # Independent request owners form the source-basis combination and evaluate it.
    assert polynomials == coordinate_expressions(linear, local, budget=reference_budget)
    assert corners == coordinate_corner_images(linear, local, budget=reference_budget)
    assert ledger.work_units > 0


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
@pytest.mark.parametrize("exact_bank", (False, True))
def test_exact_coordinate_consumers_preserve_complete_signed_source_bank(
    kind: str,
    exact_bank: bool,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    element = coordinate_lagrange_element(kind, 1)
    controls = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    controls[-1, 0] += 0.125
    # Nonbinary signed source rows exercise the immutable rational-bank path too;
    # conversion to a float carrier must never become the source authority.
    local = (
        tuple(
            tuple(Fraction(float(value)) - Fraction(2, 3) for value in row)
            for row in controls
        )
        if exact_bank
        else controls
    )
    expected = algebra.coordinate_expressions(element, local, budget=reference_budget)
    expected_corners = algebra.coordinate_corner_images(
        element, local, budget=reference_budget
    )
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        prepared = algebra.multi_affine_coordinates(element, local, budget=ledger)
        assert prepared == (expected, expected_corners)
        assert algebra.coordinate_expressions(element, local, budget=ledger) == expected
        assert algebra.coordinate_polynomials(element, local, budget=ledger) == expected
        assert (
            algebra.coordinate_corner_images(element, local, budget=ledger)
            == expected_corners
        )


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_coordinate_preparation_rebinds_changed_complete_source_and_reference_map(
    kind: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    element = coordinate_lagrange_element(kind, 1)
    controls = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    dimension = controls.shape[1]
    first = RestrictedCellGeometryElement(
        element,
        kind,
        np.eye(dimension, dtype=np.float64) * 0.5,
        np.zeros(dimension, dtype=np.float64),
    )
    second = RestrictedCellGeometryElement(
        element,
        kind,
        np.eye(dimension, dtype=np.float64) * 0.5,
        np.full(dimension, 0.25, dtype=np.float64),
    )
    expected_first = algebra.coordinate_expressions(
        first, controls, budget=reference_budget
    )
    expected_second = algebra.coordinate_expressions(
        second, controls, budget=reference_budget
    )
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        assert (
            algebra.coordinate_expressions(first, controls, budget=ledger)
            == expected_first
        )
        assert (
            algebra.coordinate_expressions(second, controls, budget=ledger)
            == expected_second
        )
        before = algebra.coordinate_polynomials(element, controls, budget=ledger)
        controls[-1, 0] += 0.125
        after = algebra.coordinate_polynomials(element, controls, budget=ledger)
        changed_restriction = algebra.coordinate_expressions(
            first, controls, budget=ledger
        )
        with pytest.raises(ValueError, match="every source degree of freedom"):
            algebra.coordinate_corner_images(element, controls[:-1], budget=ledger)
    assert before != after
    assert after == algebra.coordinate_polynomials(
        element, controls, budget=reference_budget
    )
    assert changed_restriction == algebra.coordinate_expressions(
        first, controls, budget=reference_budget
    )
    assert expected_first != expected_second


def test_resumed_coordinate_stage_keeps_its_existing_work_deadline(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    ledger = coordinate_budget(10, 1_000_000)
    ledger.reserve(5)
    start = ledger.work_units
    with ledger.bound_stage(2, 1_000_000, starting_work_units=start):
        ledger.reserve(2)
    ledger.reserve(1)
    completed = ledger.work_units
    with pytest.raises(CoordinateEnclosureResourceError) as refusal:
        with ledger.bound_stage(2, 1_000_000, starting_work_units=start):
            pytest.fail("A completed owner's work deadline was renewed.")
    assert refusal.value.resource == "coefficient_work"
    assert refusal.value.limit == start + 2
    assert refusal.value.requested == refusal.value.completed == completed
    assert ledger.work_units == completed
    with ledger.bound_stage(2, 1_000_000):
        ledger.reserve(2)
    assert ledger.work_units == ledger.maximum_work_units


@pytest.mark.parametrize("proof", ("embedding", "coverage"))
def test_public_coordinate_proof_borrows_the_exhausted_request_allowance(
    proof: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry, embedding = mapped_split_square()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    ledger = coordinate_budget(1, 64 * 1024**2)
    ledger.reserve(ledger.maximum_work_units)
    with ledger.activate():
        if proof == "embedding":
            certificate = phx.geometry.certify_global_embedding(
                mesh, geometry, validity, budget=ledger
            )
        else:
            certificate = phx.geometry.certify_domain_coverage(
                mesh,
                geometry,
                split_square_domain(),
                np.asarray((0, 1), dtype=np.int64),
                embedding=embedding,
                budget=ledger,
            )
    assert certificate.status == "unresolved"
    refusal = next(
        finding
        for finding in certificate.findings
        if finding.resource == "coefficient_work"
    )
    requested, achieved = dict(refusal.requested), dict(refusal.achieved)
    assert requested["limit"] == ledger.maximum_work_units
    assert requested["requested"] > requested["limit"]
    assert achieved["completed"] == ledger.work_units == ledger.maximum_work_units == 1
    # Admission refuses before this proof starts; the completed predecessor
    # remains on the request, not on the certificate's phase record.
    assert (
        achieved["source_expression_work_units"]
        == certificate.source_expression_work_units
        == 0
    )


def test_exact_polynomial_zero_annihilates_without_visiting_other_support(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    polynomial: algebra.Polynomial = {(7, 3): Fraction(-2, 3), (0, 0): Fraction(5, 7)}
    original = dict(polynomial)
    ledger = coordinate_budget(0, 0)
    with ledger.activate():
        assert algebra.multiply({}, polynomial, budget=ledger) == {}
        assert algebra.multiply(polynomial, {}, budget=ledger) == {}
        assert algebra.scale(polynomial, 0, budget=ledger) == {}
        assert (
            algebra.math_product_polynomials(
                (polynomial, polynomial, {}), 2, budget=ledger
            )
            == {}
        )
        assert (
            algebra.determinant(((polynomial, polynomial), ({}, {})), budget=ledger) == {}
        )
    assert polynomial == original


def test_exact_reference_late_zero_removes_unexecuted_prefix_powers(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    u, v = algebra.axes(2, budget=reference_budget)
    arguments = (algebra.add(u, v, budget=reference_budget), u, {})
    polynomial: algebra.Polynomial = {(12, 8, 1): Fraction(5, 7)}
    ledger = coordinate_budget(1_000, 1_000_000)
    with ledger.activate():
        composition = algebra.ArgumentComposition(arguments, budget=ledger)
        prepared_work = ledger.work_units
        assert composition(polynomial, budget=ledger) == {}
        assert ledger.work_units == prepared_work
        assert not ledger.reference_monomial_cache


@pytest.mark.parametrize("domain", ("box", "simplex"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_polynomial_flux_exact_zero_coordinate_needs_no_tangent_work(
    domain: algebra.BernsteinDomain,
    axis: int,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    u, v = algebra.axes(2, budget=reference_budget)
    high_order = algebra.add(
        algebra.power(u, 7, 2, budget=reference_budget),
        algebra.power(v, 9, 2, budget=reference_budget),
        budget=reference_budget,
    )
    coordinates = (high_order, algebra.multiply(u, v, budget=reference_budget))
    with_zero = coordinates[:axis] + ({},) + coordinates[axis:]
    ledger = coordinate_budget(0, 0)
    with ledger.activate():
        assert chart_flux(with_zero, domain, budget=ledger) == Fraction(0)
        assert chart_flux(tuple(reversed(with_zero)), domain, budget=ledger) == Fraction(
            0
        )


@pytest.mark.parametrize("reverse", (False, True))
def test_sparse_exact_determinant_preserves_nonzero_signed_terms(
    reverse: bool,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    sign = -1 if reverse else 1
    ledger = coordinate_budget(100_000, 10_000_000)
    with ledger.activate():
        for variables in (2, 3):
            u, v = algebra.axes(variables, budget=ledger)[:2]
            matrix: tuple[tuple[algebra.Polynomial, ...], ...] = (
                (u, u, {}),
                ({}, v, v),
                ({}, {}, algebra.add(u, v, budget=ledger)),
            )
            if reverse:
                matrix = tuple(reversed(matrix))
            suffix = (0,) * (variables - 2)
            expected: algebra.Polynomial = {
                (2, 1) + suffix: Fraction(sign),
                (1, 2) + suffix: Fraction(sign),
            }
            # Polynomial Jacobian/Gram determinants seed n formal variables
            # for an n-by-n matrix. The expression API explicitly owns a
            # separate variable_dimension and also supports two-variable data.
            if variables == len(matrix):
                assert algebra.determinant(matrix, budget=ledger) == expected
            assert (
                algebra.expression_determinant(
                    matrix, variable_dimension=variables, budget=ledger
                )
                == expected
            )


def test_immutable_coordinate_preparation_reuses_only_complete_current_source(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    element = coordinate_lagrange_element("hexahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in row) for row in element.reference_nodes
    )
    ledger = coordinate_budget(100_000, 10_000_000)
    with ledger.activate():
        first = algebra.coordinate_preparation_key(element, bank, budget=ledger)
        first_work = ledger.work_units
        retained = ledger.retained_basis_bytes
        same = tuple(tuple(value for value in row) for row in bank)
        assert algebra.coordinate_preparation_key(element, same, budget=ledger) is first
        assert ledger.work_units == first_work
        assert ledger.retained_basis_bytes == retained > 0
        changed = ((bank[0][0] + 1, *bank[0][1:]), *bank[1:])
        assert (
            algebra.coordinate_preparation_key(element, changed, budget=ledger)[0]
            == changed
        )
        assert ledger.work_units > first_work
        mutable = np.asarray(element.reference_nodes, dtype=np.float64).copy()
        original = algebra.coordinate_preparation_key(element, mutable, budget=ledger)
        before = ledger.work_units
        mutable[0, 0] += 1
        current = algebra.coordinate_preparation_key(element, mutable, budget=ledger)
        assert current[0] != original[0]
        assert ledger.work_units > before


def test_exact_reference_monomials_reuse_immutable_ordered_preparation(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    first: algebra.Polynomial = {(1, 0): Fraction(1), (0, 1): Fraction(1)}
    second: algebra.Polynomial = {(1, 0): Fraction(1)}
    polynomial: algebra.Polynomial = {(3, 2, 0): Fraction(-2, 3)}
    arguments = (first, second, {})
    expected: algebra.Polynomial = {
        (5, 0): Fraction(-2, 3),
        (4, 1): Fraction(-2),
        (3, 2): Fraction(-2),
        (2, 3): Fraction(-2, 3),
    }
    ledger = coordinate_budget(100_000, 10_000_000)
    with ledger.activate():
        original = algebra.ArgumentComposition(arguments, budget=ledger)(
            polynomial, budget=ledger
        )
        first_work = ledger.work_units
        retained = ledger.retained_basis_bytes
        repeated = algebra.ArgumentComposition(
            (dict(first), dict(second), {}), budget=ledger
        )(
            polynomial,
            budget=ledger,
        )
        repeated_work = ledger.work_units - first_work
        assert tuple(original.items()) == tuple(expected.items())
        assert tuple(repeated.items()) == tuple(expected.items())
        assert 0 < repeated_work < first_work
        assert ledger.retained_basis_bytes == retained > 0
        assert len(ledger.reference_monomial_cache) == 1

        # Neither a mutable result nor subsequent source edits may poison the
        # immutable polynomial owner. The ordered support is scientific.
        original[(5, 0)] = Fraction(99)
        pristine: algebra.Polynomial = {(1, 0): Fraction(1), (0, 1): Fraction(1)}
        assert (
            algebra.ArgumentComposition((pristine, second, {}), budget=ledger)(
                polynomial, budget=ledger
            )
            == expected
        )
        first[(0, 1)] = Fraction(-1)
        changed = algebra.ArgumentComposition(arguments, budget=ledger)(
            polynomial, budget=ledger
        )
        assert changed[(4, 1)] == Fraction(2)
        assert len(ledger.reference_monomial_cache) == 2
        reversed_support = dict(reversed(tuple(pristine.items())))
        reordered = algebra.ArgumentComposition(
            (reversed_support, second, {}), budget=ledger
        )(
            polynomial,
            budget=ledger,
        )
        assert reordered == expected
        assert tuple(reordered) == tuple(reversed(tuple(expected)))
        assert len(ledger.reference_monomial_cache) == 3
        lower_power: algebra.Polynomial = {(2, 2, 0): Fraction(1)}
        lower_expected: algebra.Polynomial = {
            (4, 0): Fraction(1),
            (3, 1): Fraction(2),
            (2, 2): Fraction(1),
        }
        assert (
            algebra.ArgumentComposition((pristine, second, {}), budget=ledger)(
                lower_power, budget=ledger
            )
            == lower_expected
        )
        assert len(ledger.reference_monomial_cache) == 4

    # A separate owner earns its own first preparation; no process-global
    # source cache or second/reset ledger erases the first preparation work.
    independent = coordinate_budget(100_000, 10_000_000)
    with independent.activate():
        assert (
            algebra.ArgumentComposition((pristine, second, {}), budget=independent)(
                polynomial, budget=independent
            )
            == expected
        )
    assert independent.work_units > repeated_work
    assert independent.retained_basis_bytes > 0


def test_exact_reference_preparation_preserves_oriented_nonplanar_flux(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    x, y, z = algebra.axes(3, budget=reference_budget)
    coordinates = (
        x,
        y,
        algebra.add(
            z, algebra.multiply(x, y, budget=reference_budget), budget=reference_budget
        ),
    )
    faces = ((0, 3, 2, 1), (0, 1, 2, 3))
    expected = tuple(
        tuple(
            chart_flux(chart, domain, budget=reference_budget)
            for chart, domain in face_charts(
                coordinates, "hexahedron", face, budget=reference_budget
            )
        )
        for face in faces
    )
    assert expected[0] == tuple(-value for value in expected[1] if value is not None)
    assert all(
        value is not None and value != Fraction(0)
        for values in expected
        for value in values
    )
    ledger = coordinate_budget(100_000, 10_000_000)
    with ledger.activate():
        for _ in range(2):
            actual = tuple(
                tuple(
                    chart_flux(chart, domain, budget=ledger)
                    for chart, domain in face_charts(
                        coordinates, "hexahedron", face, budget=ledger
                    )
                )
                for face in faces
            )
            assert actual == expected
        assert ledger.reference_monomial_cache


def test_prepared_reference_action_preserves_current_source_and_independent_ledger(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    source = coordinate_lagrange_element("hexahedron", 1)
    chart = coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(chart.reference_nodes, dtype=np.float64)
    values = nodes.copy()
    values[:, 0] += nodes[:, 0] * (1 - nodes[:, 0]) * nodes[:, 1] / 8

    def action(current: np.ndarray) -> Any:
        exact = tuple(tuple(Fraction(float(value)) for value in row) for row in current)
        return PolynomialComposedCellGeometryElement(
            source,
            chart,
            np.asarray(
                [[value.numerator for value in row] for row in exact], dtype=np.int64
            ),
            np.asarray(
                [[value.denominator for value in row] for row in exact], dtype=np.uint64
            ),
            budget=reference_budget,
        )

    element = action(values)
    original = tuple(
        tuple(Fraction(float(value)) for value in row) for row in source.reference_nodes
    )
    u, v, w = algebra.axes(3, budget=reference_budget)
    expected = (
        algebra.add(
            u,
            algebra.scale(
                algebra.multiply(
                    algebra.multiply(
                        u,
                        algebra.add(
                            algebra.constant(1, 3, budget=reference_budget),
                            algebra.scale(u, -1, budget=reference_budget),
                            budget=reference_budget,
                        ),
                        budget=reference_budget,
                    ),
                    v,
                    budget=reference_budget,
                ),
                Fraction(1, 8),
                budget=reference_budget,
            ),
            budget=reference_budget,
        ),
        v,
        w,
    )
    ledger = coordinate_budget(2_000_000, 81_920_000)
    with ledger.activate():
        prepared = algebra.prepared_reference_arguments(element, budget=ledger)
        first_work = ledger.work_units
        first_retained = ledger.retained_basis_bytes
        repeated = algebra.prepared_reference_arguments(element, budget=ledger)
        assert ledger.work_units == first_work > 0
        assert ledger.retained_basis_bytes == first_retained > 0
        assert (
            tuple(
                algebra.ExpressionComposition(repeated, budget=ledger)(
                    axis, budget=ledger
                )
                for axis in (u, v, w)
            )
            == expected
        )
        assert (
            algebra.coordinate_expressions(element, original, budget=ledger) == expected
        )

        # A current physical coefficient bank remains a fresh binding even
        # while it borrows the same immutable reference preparation.
        moved = tuple((row[0] * 2 + 1, *row[1:]) for row in original)
        assert algebra.coordinate_expressions(element, moved, budget=ledger) == (
            algebra.add(
                algebra.scale(expected[0], 2, budget=ledger),
                algebra.constant(1, 3, budget=ledger),
                budget=ledger,
            ),
            v,
            w,
        )

        # A changed chart earns its own preparation and full map.
        changed_values = values.copy()
        center = np.flatnonzero(np.all(nodes == 0.5, axis=1))[0]
        changed_values[center, 0] += 1 / 16
        changed = action(changed_values)
        before = ledger.work_units
        changed_preparation = algebra.prepared_reference_arguments(changed, budget=ledger)
        assert ledger.work_units > before
        changed_map = tuple(
            algebra.ExpressionComposition(changed_preparation, budget=ledger)(
                axis, budget=ledger
            )
            for axis in (u, v, w)
        )
        assert changed_map != expected
        cold = algebra.reference_composition_arguments(changed, budget=ledger)
        assert changed_map == tuple(
            algebra.ExpressionComposition(cold, budget=ledger)(axis, budget=ledger)
            for axis in (u, v, w)
        )
        assert (
            tuple(
                algebra.ExpressionComposition(prepared, budget=ledger)(
                    axis, budget=ledger
                )
                for axis in (u, v, w)
            )
            == expected
        )

    independent = coordinate_budget(2_000_000, 81_920_000)
    with independent.activate():
        preparation = algebra.prepared_reference_arguments(element, budget=independent)
        assert independent.work_units > 0 and independent.retained_basis_bytes > 0
        assert (
            tuple(
                algebra.ExpressionComposition(preparation, budget=independent)(
                    axis, budget=independent
                )
                for axis in (u, v, w)
            )
            == expected
        )


def test_immutable_support_profiles_preserve_actual_denominators_and_source_mutation(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    source: algebra.Polynomial = {
        (0, 0): Fraction(17, 6),
        (1, 0): Fraction(-19, 10),
        (0, 1): Fraction(23, 15),
    }
    original = dict(source)
    current: algebra.Polynomial = {(0, 0): Fraction(1, 17)}
    ledger = coordinate_budget(100_000, 10_000_000)
    with ledger.activate():
        prepared = algebra.PreparedPolynomialSupport(source, budget=ledger)
        assert algebra.coefficient_profile(
            (prepared,), exact_denominator=True, budget=ledger
        ) == (
            5,
            30,
        )
        for exact in (False, True):
            assert algebra.coefficient_profile(
                (current, prepared),
                exact_denominator=exact,
                budget=ledger,
            ) == (
                algebra.coefficient_profile(
                    (current, original), exact_denominator=exact, budget=ledger
                )
            )
        before = ledger.work_units
        algebra.coefficient_profile((prepared,), budget=ledger)
        reused_work = ledger.work_units - before
        before = ledger.work_units
        algebra.coefficient_profile((original,), budget=ledger)
        assert 0 < reused_work < ledger.work_units - before
        source[(0, 0)] = Fraction(1, 2**70)
        assert dict(prepared) == original
        assert algebra.coefficient_profile(
            (prepared,), exact_denominator=True, budget=ledger
        ) == (
            5,
            30,
        )
        assert algebra.coefficient_profile(
            (source,), exact_denominator=True, budget=ledger
        ) != (5, 30)

    independent = coordinate_budget(100_000, 10_000_000)
    with independent.activate():
        prepared = algebra.PreparedPolynomialSupport(original, budget=independent)
        assert independent.work_units > 0 and independent.retained_basis_bytes > 0
        assert algebra.coefficient_profile(
            (prepared,), exact_denominator=True, budget=independent
        ) == (
            5,
            30,
        )


def test_composition_accumulator_profiles_preserve_coprime_exact_reduction_order(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    u, v = algebra.axes(2, budget=reference_budget)
    arguments = (
        algebra.add(u, v, budget=reference_budget),
        algebra.add(
            u, algebra.scale(v, -1, budget=reference_budget), budget=reference_budget
        ),
    )
    polynomial: algebra.Polynomial = {
        (0, 0): Fraction(1, 17),
        (1, 0): Fraction(-1, 19),
        (0, 1): Fraction(1, 23),
        (2, 0): Fraction(1, 29),
        (1, 1): Fraction(-1, 31),
        (0, 2): Fraction(1, 37),
    }
    expected = algebra.compose(polynomial, arguments, budget=reference_budget)
    ledger = coordinate_budget(100_000, 10_000_000)
    with ledger.activate():
        actual = algebra.compose(polynomial, arguments, budget=ledger)
        assert tuple(actual.items()) == tuple(expected.items())
        assert algebra.coefficient_profile(
            (actual,), exact_denominator=True, budget=ledger
        ) == (
            algebra.coefficient_profile(
                (expected,), exact_denominator=True, budget=ledger
            )
        )
        actual[(0, 0)] = Fraction(99)
        assert tuple(
            algebra.compose(polynomial, arguments, budget=ledger).items()
        ) == tuple(expected.items())
    independent = coordinate_budget(100_000, 10_000_000)
    with independent.activate():
        assert tuple(
            algebra.compose(polynomial, arguments, budget=independent).items()
        ) == tuple(expected.items())
        assert independent.work_units > 0


def test_composition_weighted_reduction_preserves_cancellation_and_order(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    arguments: tuple[algebra.Polynomial, ...] = (
        {(1, 0): Fraction(1), (0, 1): Fraction(1)},
        {(1, 0): Fraction(1), (0, 1): Fraction(-1)},
    )
    polynomial: algebra.Polynomial = {
        (2, 0): Fraction(1),
        (0, 2): Fraction(-1),
        (1, 0): Fraction(1, 17),
        (0, 1): Fraction(-1, 17),
    }
    owner = coordinate_budget(100_000, 10_000_000)
    with owner.activate():
        result = algebra.compose(polynomial, arguments, budget=owner)
    assert tuple(result.items()) == (
        ((1, 1), Fraction(4)),
        ((0, 1), Fraction(2, 17)),
    )
    changed = dict(polynomial)
    changed[(0, 2)] = Fraction(1)
    independent = coordinate_budget(100_000, 10_000_000)
    with independent.activate():
        changed_result = algebra.compose(changed, arguments, budget=independent)
    assert tuple(changed_result.items()) == (
        ((2, 0), Fraction(2)),
        ((0, 2), Fraction(2)),
        ((0, 1), Fraction(2, 17)),
    )


def test_zero_reference_products_do_not_reprepare_a_nonempty_accumulator(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    u, v = algebra.axes(2, budget=reference_budget)
    arguments = (algebra.add(u, v, budget=reference_budget), u, {})
    nonzero: algebra.Polynomial = {(0, 0, 0): Fraction(1, 17), (1, 0, 0): Fraction(2, 19)}
    with_zero = {**nonzero, (12, 8, 1): Fraction(5, 7), (9, 7, 3): Fraction(-11, 13)}
    ledgers = tuple(coordinate_budget(100_000, 10_000_000) for _ in range(2))
    results = []
    for ledger, polynomial in zip(ledgers, (nonzero, with_zero), strict=True):
        with ledger.activate():
            results.append(
                algebra.ArgumentComposition(arguments, budget=ledger)(
                    polynomial, budget=ledger
                )
            )
    assert tuple(results[0].items()) == tuple(results[1].items())
    assert ledgers[0].work_units == ledgers[1].work_units > 0
    assert ledgers[0].retained_basis_bytes == ledgers[1].retained_basis_bytes > 0


def test_sibling_corner_operations_reuse_only_identical_complete_source_inputs(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    source = coordinate_lagrange_element("hexahedron", 1)
    nodes = np.asarray(source.reference_nodes, dtype=np.float64)

    def half(origin: float) -> Any:
        points = nodes / 2 + np.asarray((origin, 0.0, 0.0))
        exact = tuple(tuple(Fraction(float(value)) for value in row) for row in points)
        return PolynomialComposedCellGeometryElement(
            source,
            source,
            np.asarray(
                [[value.numerator for value in row] for row in exact], dtype=np.int64
            ),
            np.asarray(
                [[value.denominator for value in row] for row in exact], dtype=np.uint64
            ),
            budget=reference_budget,
        )

    first, second = half(0.0), half(0.5)
    bank = tuple(tuple(Fraction(float(value)) for value in row) for row in nodes)
    expected_first = algebra.coordinate_corner_images(
        first, bank, budget=reference_budget
    )
    expected_second = algebra.coordinate_corner_images(
        second, bank, budget=reference_budget
    )
    assert expected_first is not None and expected_second is not None
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        assert (
            algebra.coordinate_corner_images(first, bank, budget=ledger) == expected_first
        )
        before = ledger.work_units
        assert (
            algebra.coordinate_corner_images(second, bank, budget=ledger)
            == expected_second
        )
        reused_work = ledger.work_units - before
        moved = tuple((row[0] + Fraction(1, 17), *row[1:]) for row in bank)
        expected_moved = tuple(
            (row[0] + Fraction(1, 17), *row[1:]) for row in expected_second
        )
        assert (
            algebra.coordinate_corner_images(second, moved, budget=ledger)
            == expected_moved
        )
        assert (
            algebra.coordinate_corner_images(second, bank, budget=ledger)
            == expected_second
        )

    independent = coordinate_budget(1_000_000, 64 * 1024**2)
    with independent.activate():
        assert (
            algebra.coordinate_corner_images(second, bank, budget=independent)
            == expected_second
        )
    assert independent.work_units > reused_work > 0
    assert independent.retained_basis_bytes > 0


@pytest.mark.parametrize("resource", ("coefficient_work", "polynomial_storage"))
def test_resumed_coordinate_admission_preserves_already_completed_quantities(
    resource: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    ledger = coordinate_budget(100, 1024)
    ledger.reserve(12, 64)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        with ledger.bound_stage(
            5 if resource == "coefficient_work" else 100,
            32 if resource == "polynomial_storage" else 1024,
            starting_work_units=0,
        ):
            raise AssertionError(
                "fixture.stage must refuse an inadmissible completed prefix; got entered stage"
            )
    failure = caught.value
    assert failure.resource == resource and failure.admission
    assert (
        failure.requested
        == failure.completed
        == (12 if resource == "coefficient_work" else 64)
    )
    assert failure.limit == (5 if resource == "coefficient_work" else 32)
    assert ledger.work_units == 12 and ledger.temporary_bytes_upper == 64


def test_coordinate_refusal_metadata_rejects_false_preoperation_quantities() -> None:

    for limit, requested, completed in (
        (10, 10, 9),
        (10, 11, 12),
        (10, 11, 11),
        (-1, 2, 0),
    ):
        with pytest.raises(
            ValueError, match=r"CoordinateEnclosureResourceError\.quantities"
        ):
            CoordinateEnclosureResourceError(
                "coefficient_work", limit, requested, completed
            )
    with pytest.raises(TypeError, match=r"CoordinateEnclosureResourceError\.quantities"):
        CoordinateEnclosureResourceError("coefficient_work", 10, 11, True)
    failure = CoordinateEnclosureResourceError("coefficient_work", 10, 11, 9)
    assert (failure.limit, failure.requested, failure.completed) == (10, 11, 9)
    assert not failure.admission


def test_native_work_refusal_is_not_relabelled_as_coordinate_storage() -> None:

    native = NativeExecutionScope(NativeExecutionLimits(1, 1, 1, 8192, 120))
    with pytest.raises(MeshcoreError) as caught:
        with native:
            ticket = native.batch(1, bytes_upper=8192)
            ledger = CoordinateEnclosureBudget(
                ticket, maximum_work_units=100, maximum_memory_bytes=8192
            )
            with ticket, ledger.activate():
                ledger.reserve(0, 128)
                with pytest.raises(MeshcoreError) as admission:
                    ledger.reserve(2)
                assert admission.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
                assert admission.value.batch_requested_upper is not None
                assert admission.value.batch_refusal_evidence is not None
                assert admission.value.batch_requested_upper[0] == 2
                assert admission.value.batch_refusal_evidence[4] == 0
                with pytest.raises(RuntimeError, match="retired"):
                    ledger.reserve(0, 1)
    assert caught.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert (
        native.evidence is not None
        and native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    )
    assert (
        caught.value.work_counters is not None
        and caught.value.memory_counters is not None
    )
    assert ledger.work_units == 0 and ledger.temporary_bytes_upper == 128


def test_coordinate_reserve_charges_work_before_native_admission() -> None:

    host_terms: dict[tuple[int, ...], Fraction] = {
        (index,): Fraction(index + 1, 17) for index in range(15)
    }
    first = np.zeros((60, 2), dtype=np.float64)
    second = np.broadcast_to(np.asarray((1.0, 0.0)), first.shape)
    third = np.broadcast_to(np.asarray((0.0, 1.0)), first.shape)
    predecessor = NativeExecutionScope(
        NativeExecutionLimits(1000, 1000, 100, 1024**2, 120)
    )
    with predecessor:
        assert np.all(exact_orient2d(first[:20], second[:20], third[:20]) > 0)
    prior = predecessor.evidence
    assert prior is not None and prior.status == MeshcoreStatus.OK
    native = NativeExecutionScope(NativeExecutionLimits(100, 1000, 100, 1024**2, 120))
    with pytest.raises(MeshcoreError) as caught:
        with native:
            native.import_preparation(
                work=int(prior.work_counters.construction_work),
                geometry_queries=int(prior.work_counters.source_queries),
                elapsed_seconds=prior.elapsed_seconds,
            )
            assert np.all(exact_orient2d(first, second, third) > 0)
            ticket = native.batch(
                native.remaining().remaining_work_units, bytes_upper=1024**2
            )
            ledger = CoordinateEnclosureBudget(
                ticket, maximum_work_units=1000, maximum_memory_bytes=1024**2
            )
            with ticket, ledger.activate():
                assert coefficient_profile((host_terms,), budget=ledger) == (4, 5)
                remaining = ticket.remaining_work_units
                assert remaining == ticket.work_upper - ledger.work_units
                ledger.reserve(remaining + 1)
    assert caught.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert ticket.settled
    # Admitted coordinate work is already unavailable to the next native visit.
    assert (
        ledger.work_units == ledger.native_charged_work_units == ticket.work_units == 15
    )
    assert (
        native.evidence is not None
        and native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    )
    assert caught.value.work_counters is not None
    assert int(caught.value.work_counters.construction_work) == 100 - remaining


def test_coordinate_live_release_does_not_mask_a_primary_callback_error() -> None:

    native = NativeExecutionScope(NativeExecutionLimits(1, 1, 1, 8192, 120))
    with pytest.raises(ValueError, match=r"fixture\.callback") as caught:
        with native:
            ticket = native.batch(1, bytes_upper=8192)
            ledger = CoordinateEnclosureBudget(
                ticket, maximum_work_units=100, maximum_memory_bytes=8192
            )
            with ticket, ledger.activate(), ledger.live_storage() as owner:
                owner.set_bound(128)
                with pytest.raises(MeshcoreError):
                    ledger.reserve(2)
                raise ValueError("fixture.callback must succeed; got injected failure")
    assert str(caught.value) == "fixture.callback must succeed; got injected failure"
    assert owner.bound == 0 and ledger.temporary_bytes_upper == 0
    assert ticket.settled
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence.host_storage_live_bytes_upper == 0
    assert ledger.work_units == 0


def test_certificate_work_measures_its_phase_without_resetting_request(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        TETRAHEDRON, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    reference_budget.reserve(7, 128)
    with reference_budget.temporary_scope():
        reference_budget.reserve(5, 1 << 20)
    predecessor_work = reference_budget.work_units
    predecessor_peak = reference_budget.peak_bytes_upper
    with reference_budget.measurement() as phase:
        certificate = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, budget=reference_budget
        )

    assert certificate.status == "certified"
    assert certificate.source_expression_work_units == phase.work_units > 0
    assert certificate.source_expression_peak_bytes == phase.peak_bytes_upper > 0
    assert phase.peak_bytes_upper < predecessor_peak
    assert reference_budget.work_units == (
        predecessor_work + certificate.source_expression_work_units
    )
    assert reference_budget.peak_bytes_upper == predecessor_peak
    frozen = (phase.work_units, phase.required_work_units, phase.peak_bytes_upper)

    with reference_budget.measurement() as parent:
        reference_budget.reserve(2, 256)
        with reference_budget.measurement() as child:
            with reference_budget.temporary_scope():
                reference_budget.admit_work_bound(6)
                reference_budget.reserve(4, 128)
    assert (parent.work_units, parent.required_work_units, parent.peak_bytes_upper) == (
        6,
        8,
        384,
    )
    assert (child.work_units, child.required_work_units, child.peak_bytes_upper) == (
        4,
        6,
        128,
    )
    reference_budget.reserve(1, 8)
    assert (phase.work_units, phase.required_work_units, phase.peak_bytes_upper) == frozen
    assert reference_budget.work_units == predecessor_work + phase.work_units + 7
