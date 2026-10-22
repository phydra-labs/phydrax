# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Validity reuse identity at serial and owner-local host boundaries."""

from contextlib import contextmanager
from fractions import Fraction

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import _cell_geometry as geometry_owner
from phydrax.discretization import _cell_geometry_validity as validity_owner
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    _validity_request_key,
    CellValidityPolicy,
)
from tests._support.cell_meshes import unit_square


def test_serial_validity_request_key_preserves_full_content_identity() -> None:
    mesh = unit_square()
    geometry = CellGeometrySpec.affine(mesh)
    # Observed full-content digest before the distributed-storage key cutover;
    # lf19_serial_key_identity.log records its byte-identical current value.
    assert _validity_request_key(geometry, mesh, CellValidityPolicy()) == (
        "dd204a6872746f48b9b3bd30b6fecee1240d47585ef85bca54e875d1f44f4b55"
    )


@contextmanager
def _proof_budget():
    work, memory = 2_000_000, 128 * 1024**2
    with HostBatchTicket(work, bytes_upper=memory) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        ).activate() as budget:
            yield budget


@pytest.mark.parametrize(
    "representation", ("polynomial", "barycentric", "homogeneous", "nested-barycentric")
)
@pytest.mark.parametrize(
    ("dimension", "ambient"),
    (
        pytest.param(2, 2, id="signed-triangle"),
        pytest.param(2, 3, id="embedded-triangle"),
        pytest.param(3, 3, id="signed-tetrahedron"),
    ),
)
def test_affine_composed_differential_reuses_exact_roots(
    dimension: int, ambient: int, representation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind = "triangle" if dimension == 2 else "tetrahedron"
    root = geometry_owner.coordinate_lagrange_element(kind, 1)
    identity = tuple(
        tuple(Fraction(int(row == column + 1)) for column in range(dimension))
        for row in range(dimension + 1)
    )
    denominator = 3 if representation == "homogeneous" else 2
    midpoint = tuple(
        (identity[0][axis] + identity[1][axis]) / denominator
        for axis in range(dimension)
    )
    first = (identity[0], midpoint, *identity[2:])
    second = (midpoint, *identity[1:])
    charts = (first, second, (first[1], first[0], *first[2:]), (first[0], first[0], *first[2:]))
    elements = []
    with _proof_budget() as budget:
        for chart in charts:
            if representation == "polynomial":
                element = geometry_owner.PolynomialComposedCellGeometryElement(
                    root,
                    root,
                    np.asarray([[value.numerator for value in row] for row in chart], dtype=object),
                    np.asarray([[value.denominator for value in row] for row in chart], dtype=object),
                    budget=budget,
                )
            else:
                weights = tuple((1 - sum(row, Fraction(0)), *row) for row in chart)
                source = root
                if representation == "nested-barycentric":
                    inner = np.eye(dimension + 1)
                    inner[1, :2] = 0.5
                    source = geometry_owner.BarycentricCellGeometryElement(root, inner)
                element = geometry_owner.BarycentricCellGeometryElement(
                    source,
                    np.asarray(weights, dtype=np.float64),
                    barycentric_homogeneous=(
                        np.asarray(
                            [[float(3 * value) for value in row] for row in weights]
                        )
                        if representation == "homogeneous"
                        else None
                    ),
                )
            elements.append(element)
    rng = np.random.default_rng(8026)
    banks = tuple(
        tuple(tuple(Fraction(float(value)) for value in row) for row in rng.normal(size=(dimension + 1, ambient)))
        for _ in range(12)
    )

    def prepare():
        with _proof_budget() as budget:
            results = tuple(
                validity_owner._prepare_polynomial_differential(
                    element, bank, kind, dimension, ambient != dimension, budget
                )
                for bank in banks
                for element in elements
            )
            return results, budget.work_units

    actual, actual_work = prepare()
    monkeypatch.setattr(validity_owner, "_affine_composed_differential", lambda *args: None)
    expected, expanded_work = prepare()
    for obtained, reference in zip(actual, expected, strict=True):
        assert obtained.jacobian == reference.jacobian
        assert obtained.polynomial == reference.polynomial
    assert actual_work < expanded_work, (actual_work, expanded_work)


@pytest.mark.parametrize(
    "value",
    (pytest.param(-1, id="reversed"), pytest.param(0, id="singular"), pytest.param(2, id="positive")),
)
def test_constant_root_piece_needs_no_identity_composition(
    value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = validity_owner._prepare_polynomial_reference_domain(
        "triangle", 2, 1, False
    )
    assert reference is not None

    def reject_identity(*args, **kwargs):
        raise AssertionError("The earned root polynomial must not be recomposed.")

    monkeypatch.setattr(
        validity_owner.coordinate_enclosure_owner, "expression_compose", reject_identity
    )
    with _proof_budget() as budget:
        result = validity_owner._certify_polynomial_subdivision(
            {(0, 0): Fraction(value)},
            reference.domain,
            2,
            reference.vertices,
            reference.child_origins,
            reference.child_matrices,
            Fraction(1),
            CellValidityPolicy(),
            budget,
        )
    expected = (
        validity_owner.CellValidityStatus.CERTIFIED_VALID
        if value > 0
        else validity_owner.CellValidityStatus.INVALID
    )
    assert result.status == expected
    assert result.lower == result.upper == float(value)
    assert result.depth == 0


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(Fraction(-1), id="negative"),
        pytest.param(Fraction(0), id="zero"),
        pytest.param(Fraction(1, 2), id="below-floor"),
        pytest.param(Fraction(1), id="at-floor"),
        pytest.param(Fraction(2), id="above-floor"),
    ),
)
def test_constant_piece_decisions_match_general_bernstein_route(
    value: Fraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = validity_owner._prepare_polynomial_reference_domain(
        "triangle", 2, 1, False
    )
    assert reference is not None

    def decide():
        with _proof_budget() as budget:
            return validity_owner._certify_polynomial_piece(
                {(0, 0): value}, reference.domain, 2, reference.vertices,
                Fraction(1), 0, CellValidityPolicy(), budget,
            )

    actual = decide()
    monkeypatch.setattr(validity_owner, "_constant_polynomial_value", lambda *args: None)
    expected = decide()
    for name in ("status", "lower", "upper", "evaluated", "terminal", "subdivide", "reason"):
        assert getattr(actual, name) == getattr(expected, name)


def test_nonpartition_coefficient_action_keeps_full_polynomial_proof() -> None:
    root = geometry_owner.coordinate_lagrange_element("triangle", 1)
    weights = np.eye(3)
    weights[0, 0] = 2.0
    element = geometry_owner.BarycentricCellGeometryElement(root, weights)
    bank = tuple(
        tuple(Fraction(value) for value in row)
        for row in ((1, 1), (3, 1), (1, 4))
    )
    with _proof_budget() as budget:
        assert validity_owner._affine_composed_differential(
            element, bank, "triangle", 2, False, budget
        ) is None
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget
        )
    assert result.polynomial == {(0, 0): Fraction(1)}


def test_single_use_differential_retains_only_reusable_roots() -> None:
    root = geometry_owner.coordinate_lagrange_element("triangle", 1)
    element = geometry_owner.BarycentricCellGeometryElement(
        root, np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1)))
    )
    bank = tuple(
        tuple(Fraction(value) for value in row)
        for row in ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    )
    with _proof_budget() as budget:
        with budget.temporary_scope():
            leaf = validity_owner._prepare_polynomial_differential(
                element, bank, "triangle", 2, True, budget, retain=False
            )
            assert leaf.polynomial == {(0, 0): Fraction(1, 4)}
            assert all(value is not leaf for value in budget.polynomial_differential_cache.values())
            assert id(leaf) not in budget.retained_objects
            del leaf
        parent = validity_owner._prepare_polynomial_differential(
            root, bank, "triangle", 2, True, budget
        )
        assert any(value is parent for value in budget.polynomial_differential_cache.values())


@pytest.mark.parametrize(
    "representation", ("barycentric", "nested-barycentric", "nonpartition")
)
@pytest.mark.parametrize(
    ("dimension", "ambient"),
    (
        pytest.param(2, 2, id="signed-triangle"),
        pytest.param(2, 3, id="embedded-triangle"),
        pytest.param(3, 3, id="signed-tetrahedron"),
    ),
)
def test_earned_corner_images_differentiate_affine_actions_exactly(
    dimension: int, ambient: int, representation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind = "triangle" if dimension == 2 else "tetrahedron"
    root = geometry_owner.coordinate_lagrange_element(kind, 1)
    weights = np.eye(dimension + 1)
    weights[1, :2] = 0.5
    if representation == "nonpartition":
        weights[0, 0] = 2.0
    source = root
    if representation == "nested-barycentric":
        inner = np.eye(dimension + 1)
        inner[2, 0], inner[2, 2] = 0.25, 0.75
        source = geometry_owner.BarycentricCellGeometryElement(root, inner)
    element = geometry_owner.BarycentricCellGeometryElement(source, weights)
    rng = np.random.default_rng(8027)
    banks = tuple(
        tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in rng.normal(size=(dimension + 1, ambient))
        )
        for _ in range(6)
    )
    embedded = ambient != dimension

    def reject_composition(*args):
        raise AssertionError("Earned corners must not be recomposed through a chart.")

    with _proof_budget() as budget:
        for bank in banks:
            assert (
                validity_owner.coordinate_enclosure_owner.coordinate_corner_images(
                    element, bank, budget=budget
                )
                is not None
            )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(validity_owner, "_affine_composed_differential", reject_composition)
            actual = tuple(
                validity_owner._prepare_polynomial_differential(
                    element, bank, kind, dimension, embedded, budget, retain=False
                )
                for bank in banks
            )
    monkeypatch.setattr(validity_owner, "_affine_corner_differential", lambda *args: None)
    monkeypatch.setattr(validity_owner, "_affine_composed_differential", lambda *args: None)
    with _proof_budget() as budget:
        expected = tuple(
            validity_owner._prepare_polynomial_differential(
                element, bank, kind, dimension, embedded, budget
            )
            for bank in banks
        )
    for obtained, reference in zip(actual, expected, strict=True):
        assert obtained.jacobian == reference.jacobian
        assert obtained.polynomial == reference.polynomial


def test_uncached_corners_keep_chart_composition() -> None:
    root = geometry_owner.coordinate_lagrange_element("triangle", 1)
    element = geometry_owner.BarycentricCellGeometryElement(
        root, np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1)))
    )
    bank = tuple(
        tuple(Fraction(value) for value in row)
        for row in ((0, 0), (1, 0), (0, 1))
    )
    with _proof_budget() as budget:
        _, key = validity_owner.coordinate_enclosure_owner.coordinate_preparation_key(
            element, bank, budget=budget
        )
        assert validity_owner._affine_corner_differential(
            element, key, "triangle", 2, False, budget
        ) is None
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget
        )
    assert result.polynomial == {(0, 0): Fraction(1, 2)}


def test_affine_chart_validity_reuses_exact_images_for_the_changed_subset() -> None:
    algebra = validity_owner.coordinate_enclosure_owner
    element = geometry_owner.BarycentricCellGeometryElement(
        geometry_owner.coordinate_lagrange_element("triangle", 1),
        np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1)), dtype=np.float64),
    )
    bank = tuple(
        tuple(Fraction(value) for value in point)
        for point in ((0, 0), (2, 0), (0, 2), (3, 0), (5, 0), (3, 2))
    )
    routes = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    policy = CellValidityPolicy()
    with _proof_budget() as preparation:
        images = algebra.coordinate_vertex_images(
            ((element, routes, routes),),
            bank,
            6,
            mismatch="unequal corners",
            budget=preparation,
        )
    assert images is not None and all(image is not None for image in images)
    image_bank = tuple(image for image in images if image is not None)
    with _proof_budget() as budget:
        expected = validity_owner._affine_chart_simplex_block(
            element, "triangle", routes[1:], bank, policy, budget
        )
        expected_work = budget.work_units
    with _proof_budget() as budget:
        actual = validity_owner._affine_chart_simplex_block(
            element,
            "triangle",
            routes[1:],
            bank,
            policy,
            budget,
            corner_bank=image_bank,
            corner_routes=routes[1:],
        )
        actual_work = budget.work_units
    assert actual is not None and expected is not None
    for field in ("status", "lower", "upper", "depth"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.reasons == expected.reasons
    assert actual_work < expected_work


def test_affine_floor_filter_queries_only_distinct_exact_edge_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tuple(
        tuple(Fraction(value) for value in point)
        for point in ((0, 0), (1, 0), (0, 1), (2, 2), (3, 2), (2, 3))
    )
    route = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    observed = []
    original = validity_owner._binary64_simplex_floor

    def filtered(values, rows, floor, *, budget):
        observed.append(rows.shape[0])
        return original(values, rows, floor, budget=budget)

    monkeypatch.setattr(validity_owner, "_binary64_simplex_floor", filtered)
    with _proof_budget() as budget:
        result = validity_owner._affine_simplex_block(
            geometry_owner.coordinate_lagrange_element("triangle", 1),
            "triangle",
            route,
            bank,
            CellValidityPolicy(),
            budget,
        )
    assert result is not None
    assert observed == [1]
    np.testing.assert_array_equal(result.lower, (1.0, 1.0))
    np.testing.assert_array_equal(result.upper, (1.0, 1.0))


def test_affine_scalar_differences_are_computed_once_for_equal_actual_operands() -> None:
    bank = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    element = geometry_owner.coordinate_lagrange_element("triangle", 1)
    observations = []
    for count in (1, 64):
        route = np.tile(np.asarray(((0, 1, 2),), dtype=np.int64), (count, 1))
        with _proof_budget() as budget:
            result = validity_owner._affine_simplex_block(
                element, "triangle", route, bank, CellValidityPolicy(), budget,
            )
            observations.append(budget.work_units)
        assert result is not None
        np.testing.assert_array_equal(result.lower, np.ones(count))
        np.testing.assert_array_equal(result.upper, np.ones(count))
    assert observations[0] == observations[1]


@pytest.mark.parametrize(
    ("frame", "floor", "filtered"),
    (
        (((Fraction(1, 3), Fraction(0)), (Fraction(0), Fraction(1, 7))), 0.9, True),
        (((Fraction(2, 3), Fraction(1, 5)), (Fraction(1, 7), Fraction(3, 5))), 0.3, True),
        (((Fraction(0), Fraction(1, 7)), (Fraction(1, 3), Fraction(0))), 0.0, False),
        (((Fraction(1, 3), Fraction(1, 7)), (Fraction(2, 3), Fraction(2, 7))), 0.0, False),
        (((Fraction(1, 2**900), Fraction(0)), (Fraction(0), Fraction(1, 2**900))), 0.1, False),
        (((Fraction(2**800), Fraction(0)), (Fraction(0), Fraction(2**800))), 0.1, False),
        (((Fraction(1, 10**400), Fraction(0)), (Fraction(0), Fraction(1, 7))), 0.0, False),
    ),
)
def test_rational_planar_floor_filter_has_independent_exact_counterexamples(
    frame: tuple[tuple[Fraction, ...], ...], floor: float, filtered: bool,
) -> None:
    determinant = frame[0][0] * frame[1][1] - frame[0][1] * frame[1][0]
    scale = sum(value * value for value in frame[0]) * sum(
        value * value for value in frame[1]
    )
    exact = determinant > 0 and determinant * determinant > Fraction(floor)**2 * scale
    with _proof_budget() as budget:
        actual = validity_owner._rational_planar_simplex_floor((frame,), floor, budget=budget)
    assert bool(actual[0]) is filtered
    if actual[0]:
        assert exact


@pytest.mark.parametrize("offset", (Fraction(0), Fraction(1, 2**60), Fraction(1, 2**20)))
def test_rational_planar_floor_filter_keeps_near_policy_boundary_exact(
    offset: Fraction,
) -> None:
    floor = 1.0e-12
    coefficient = Fraction(floor) * (1 + offset)
    frame = ((Fraction(1), Fraction(0)), (Fraction(1), coefficient))
    determinant_squared = coefficient * coefficient
    exact = determinant_squared > Fraction(floor)**2 * (1 + determinant_squared)
    with _proof_budget() as budget:
        actual = validity_owner._rational_planar_simplex_floor((frame,), floor, budget=budget)
    if offset == 0:
        assert not exact and not actual[0]
    elif offset == Fraction(1, 2**60):
        assert exact and not actual[0]  # Outward uncertainty retains the exact fallback.
    else:
        assert exact and actual[0]


def test_rational_planar_floor_reuses_real_frames_without_changing_exact_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(0)),
        (Fraction(0), Fraction(1, 7)),
    )
    route = np.asarray(((0, 1, 2),), dtype=np.int64)
    element = geometry_owner.coordinate_lagrange_element("triangle", 1)
    with _proof_budget() as budget:
        actual = validity_owner._affine_simplex_block(
            element, "triangle", route, bank, CellValidityPolicy(), budget,
        )
        filtered_work = budget.work_units
    monkeypatch.setattr(
        validity_owner, "_rational_planar_simplex_floor",
        lambda frames, floor, *, budget: np.zeros(len(frames), dtype=np.bool_),
    )
    with _proof_budget() as budget:
        expected = validity_owner._affine_simplex_block(
            element, "triangle", route, bank, CellValidityPolicy(), budget,
        )
        exact_work = budget.work_units
    assert actual is not None and expected is not None
    for field in ("status", "lower", "upper", "depth"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.reasons == expected.reasons
    assert filtered_work < exact_work


def _signed_differential_fixture(degree: int, sign: int = 1):
    element = geometry_owner.coordinate_lagrange_element("triangle", degree)
    nodes = geometry_owner.coordinate_lagrange_reference_nodes("triangle", degree)
    bank = tuple(
        (x + (x * x / 4 if degree == 2 else 0), sign * y)
        for x, y in nodes
    )
    expected = (
        {(0, 0): Fraction(sign), (1, 0): Fraction(sign, 2)}
        if degree == 2 else {(0, 0): Fraction(sign)}
    ) if sign else {}
    return element, bank, expected


@pytest.mark.parametrize("degree", (1, 2), ids=("affine", "expanded"))
@pytest.mark.parametrize("sign", (-1, 0, 1), ids=("negative", "zero", "positive"))
def test_signed_differential_issues_actual_numeric_output_before_escape(
    degree: int, sign: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands

    element, bank, expected = _signed_differential_fixture(degree, sign)
    algebra = validity_owner.coordinate_enclosure_owner
    arithmetic = []
    issued = []
    owner = validity_owner if degree == 1 else algebra
    name = "_affine_edge_differential" if degree == 1 else "expression_determinant"
    actual_arithmetic = getattr(owner, name)
    actual_issue = operands._remember_signed_determinant_operands

    def compute(*args, **kwargs):
        arithmetic.append(True)
        return actual_arithmetic(*args, **kwargs)

    def issue(key, kind, dimension, polynomial, budget):
        assert arithmetic
        assert kind == "triangle" and dimension == 2
        assert type(polynomial) is dict
        assert all(type(value) is Fraction for value in polynomial.values())
        assert polynomial == expected
        assert (key, kind, dimension, False) not in budget.polynomial_differential_cache
        issued.append((key, tuple(polynomial.items())))
        return actual_issue(key, kind, dimension, polynomial, budget)

    monkeypatch.setattr(owner, name, compute)
    monkeypatch.setattr(operands, "_remember_signed_determinant_operands", issue)
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
        assert result.polynomial == expected
        assert len(issued) == len(arithmetic) == 1
        _, key = algebra.coordinate_preparation_key(element, bank, budget=budget)
        assert issued[0][0] == key
        charged, retained = budget.work_units, budget.retained_basis_bytes
        assert validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        ) is result
        assert budget.work_units > charged
        assert budget.work_units == budget.ticket.work_units
        assert budget.retained_basis_bytes == retained
        assert len(issued) == len(arithmetic) == 1


@pytest.mark.parametrize(
    ("coordinate", "signed"), ((False, False), (False, True), (True, False)),
)
def test_signed_differential_issue_requires_both_operand_opt_ins(
    coordinate: bool, signed: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands

    def unexpected_issue(*args, **kwargs):
        pytest.fail("A noncapturing producer must not issue numeric operands.")

    monkeypatch.setattr(operands, "_remember_signed_determinant_operands", unexpected_issue)
    element, bank, expected = _signed_differential_fixture(1, -1)
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = coordinate
        budget.capture_signed_determinant_operands = signed
        budget.capture_polynomial_operands = True
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
    assert result.polynomial == expected


def test_signed_differential_issue_preserves_current_mutable_source_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands

    element, exact, _ = _signed_differential_fixture(1)
    bank = np.asarray(exact, dtype=np.float64)
    original = bank.copy()
    issued = []
    actual_issue = operands._remember_signed_determinant_operands

    def issue(key, kind, dimension, polynomial, budget):
        issued.append((key, dict(polynomial)))
        return actual_issue(key, kind, dimension, polynomial, budget)

    monkeypatch.setattr(operands, "_remember_signed_determinant_operands", issue)
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        first = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
        np.testing.assert_array_equal(bank, original)
        bank[:, 1] *= -1
        changed = bank.copy()
        second = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
        np.testing.assert_array_equal(bank, changed)
        assert first.polynomial == {(0, 0): Fraction(1)}
        assert second.polynomial == {(0, 0): Fraction(-1)}
        assert issued[0][0] != issued[1][0]
        bank[:] = original
        assert validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        ) is first
        assert len(issued) == 2


@pytest.mark.parametrize("case", ("embedded", "pyramid", "source-dimension"))
def test_signed_differential_issue_declines_unsupported_geometry(
    case: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands

    def unexpected_issue(*args, **kwargs):
        pytest.fail("Unsupported geometry must retain its ordinary differential path.")

    monkeypatch.setattr(operands, "_remember_signed_determinant_operands", unexpected_issue)
    embedded = case == "embedded"
    dimension = 3 if case == "pyramid" else 2
    kind = "pyramid" if case == "pyramid" else "triangle"
    element = geometry_owner.coordinate_lagrange_element(kind, 1)
    bank = geometry_owner.coordinate_lagrange_reference_nodes(kind, 1)
    if embedded:
        bank = tuple((*row, Fraction(0)) for row in bank)
    elif case == "source-dimension":
        root = geometry_owner.coordinate_lagrange_element("tetrahedron", 1)
        element = geometry_owner.RestrictedCellGeometryElement(
            root, "triangle", np.asarray(((1, 0), (0, 1), (0, 0))), np.zeros(3),
        )
        bank = tuple(
            row[:2]
            for row in geometry_owner.coordinate_lagrange_reference_nodes("tetrahedron", 1)
        )
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        result = validity_owner._prepare_polynomial_differential(
            element, bank, kind, dimension, embedded, budget,
        )
        assert result.polynomial is not None
        assert not budget.polynomial_operand_snapshots


@pytest.mark.parametrize("case", ("rational", "algebraic"))
def test_signed_differential_issue_declines_nonrational_polynomial_coefficients(
    case: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    element, bank, _ = _signed_differential_fixture(2)
    algebra = validity_owner.coordinate_enclosure_owner
    zero = (0, 0)
    value = (
        algebra.RationalPolynomial({zero: Fraction(1)}, {zero: Fraction(1), (1, 0): Fraction(1)})
        if case == "rational" else {zero: RadicalElement(terms=((2, Fraction(1)),))}
    )
    jacobian = (({zero: Fraction(1)}, {}), ({}, value))
    monkeypatch.setattr(validity_owner, "_polynomial_jacobian", lambda *args: jacobian)
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
        assert result.polynomial is not None
        assert not budget.polynomial_operand_snapshots
        assert result.jacobian is jacobian


@pytest.mark.parametrize("resource", ("work", "storage"))
@pytest.mark.parametrize("degree", (1, 2), ids=("affine", "expanded"))
def test_signed_differential_issue_refusal_keeps_paid_arithmetic_and_no_publication(
    resource: str, degree: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    element, bank, expected = _signed_differential_fixture(degree, -1)
    actual_issue = operands._remember_signed_determinant_operands
    completed = []

    def issue(key, kind, dimension, polynomial, budget):
        assert polynomial == expected
        assert budget.work_units > 0
        completed.append(budget.work_units)
        with budget.bound_stage(
            0 if resource == "work" else 2_000_000,
            0 if resource == "storage" else 128 * 1024**2,
        ):
            return actual_issue(key, kind, dimension, polynomial, budget)

    monkeypatch.setattr(operands, "_remember_signed_determinant_operands", issue)
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        with pytest.raises(CoordinateEnclosureResourceError):
            validity_owner._prepare_polynomial_differential(
                element, bank, "triangle", 2, False, budget,
            )
        assert completed
        assert budget.work_units >= completed[0]
        assert budget.work_units == budget.ticket.work_units
        assert not budget.polynomial_differential_cache
        assert not budget.polynomial_operand_snapshots


def test_signed_differential_issue_clones_exposed_coefficient_atoms_and_refuses_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operands

    element, bank, _ = _signed_differential_fixture(2)
    algebra = validity_owner.coordinate_enclosure_owner
    zero = (0, 0)
    exposed = Fraction(-3, 7)
    # Actual determinant arithmetic is still required. Its unit-factor shortcut
    # owns the output dictionary but can share the coefficient's Fraction atom.
    jacobian = (({zero: Fraction(1)}, {}), ({}, {zero: exposed}))
    monkeypatch.setattr(validity_owner, "_polynomial_jacobian", lambda *args: jacobian)
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
        _, authority = algebra.coordinate_preparation_key(element, bank, budget=budget)
        role = ("signed-determinant", authority, "triangle", 2)
        snapshot, _ = budget.polynomial_operand_snapshots[role]
        assert len(snapshot) == 1
        assert dict(snapshot[0]) == {zero: Fraction(-3, 7)}
        coefficient = dict(snapshot[0])[zero]
        assert type(coefficient) is Fraction and coefficient is not exposed
        assert isinstance(result.polynomial, dict)
        assert coefficient is not result.polynomial[zero]
        exposed._numerator = 999
        result.polynomial[zero] = Fraction(123)
        assert dict(snapshot[0]) == {zero: Fraction(-3, 7)}
        with pytest.raises(ValueError, match="conflict"):
            operands._remember_signed_determinant_operands(
                authority, "triangle", 2, result.polynomial, budget,
            )
        assert dict(snapshot[0]) == {zero: Fraction(-3, 7)}


def test_signed_differential_issue_does_not_reclassify_uncaptured_history() -> None:
    element, bank, expected = _signed_differential_fixture(1)
    with _proof_budget() as budget:
        result = validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        )
        charged = budget.work_units
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        assert validity_owner._prepare_polynomial_differential(
            element, bank, "triangle", 2, False, budget,
        ) is result
        assert result.polynomial == expected
        assert budget.work_units > charged
        assert not budget.polynomial_operand_snapshots


def test_single_use_signed_differential_issues_numeric_data_without_exporting_differential() -> None:
    element, bank, expected = _signed_differential_fixture(2, -1)
    algebra = validity_owner.coordinate_enclosure_owner
    with _proof_budget() as budget:
        budget.capture_coordinate_operands = True
        budget.capture_signed_determinant_operands = True
        with budget.temporary_scope():
            result = validity_owner._prepare_polynomial_differential(
                element, bank, "triangle", 2, False, budget, retain=False,
            )
            assert result.polynomial == expected
            assert not budget.polynomial_differential_cache
            assert id(result) not in budget.retained_objects
        _, authority = algebra.coordinate_preparation_key(element, bank, budget=budget)
        snapshot, _ = budget.polynomial_operand_snapshots[
            ("signed-determinant", authority, "triangle", 2)
        ]
        assert len(snapshot) == 1 and dict(snapshot[0]) == expected
