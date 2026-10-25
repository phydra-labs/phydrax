# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Native quadratic image boxes may only retire chart pairs the exact boxes separate."""

import itertools
import math
import sys
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

import phydrax as phx
import phydrax._meshcore as meshcore_owner
import phydrax.geometry._embedding_cells as embedding_cells
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from phydrax.discretization._coordinate_enclosure import (
    bernstein_coefficients,
    Expression,
    Polynomial,
    quadratic_simplex_controls,
    quadratic_simplex_matrices,
    quadratic_simplex_matrix,
    quadratic_simplex_storage_upper,
    QuadraticSimplexControls,
    RationalPolynomial,
)
from phydrax.geometry._embedding_cells import (
    bounds,
    Cell,
    chart_barycentric,
    children,
    quadratic_chart_box,
    quadratic_controls,
    vertices,
)
from phydrax.geometry._embedding_state import ContactDecision, MeshCertificateLimits
from phydrax.geometry._mapped_embedding import _bisect_cell_contact, pair_decision
from phydrax.geometry._rational import fraction_endpoint
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mesh_certificate_fixtures import curved_tetrahedral_components


requires_meshcore = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore unavailable"
)


def _value(polynomial: Polynomial, point: tuple[Fraction, ...]) -> Fraction:
    total = Fraction(0)
    for index, coefficient in polynomial.items():
        if not isinstance(coefficient, Fraction):
            raise TypeError("the homogeneous oracle covers rational sources only")
        total += coefficient * math.prod(
            value**power for value, power in zip(point, index, strict=True)
        )
    return total


def _points(dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(value, 7) for value in numerators)
        for numerators in itertools.product(range(8), repeat=dimension)
        if sum(numerators) <= 7
    )


def _quadratic(dimension: int, degree: int, scale: Fraction) -> Polynomial:
    """Nonbinary rational source of the requested total degree, one term per index."""
    terms = [
        index
        for index in itertools.product(range(degree + 1), repeat=dimension)
        if sum(index) <= degree
    ]
    return {
        index: scale * Fraction((-1) ** position * (2 * position + 1), 3 + 4 * position)
        for position, index in enumerate(terms)
    }


def _triangle(x: Polynomial, y: Polynomial, ids: tuple[int, int, int]) -> Cell:
    return Cell((x, y), "simplex", "triangle", 2, ids, vertices("triangle"))


# The first curved triangle bulges across its hypotenuse up to x + y = 1 + kappa/2.
_KAPPA = Fraction(2, 7)
_FIRST = _triangle(
    {(1, 0): Fraction(1), (1, 1): _KAPPA},
    {(0, 1): Fraction(1), (1, 1): _KAPPA},
    (0, 1, 2),
)


def _reflected(shift: Fraction, bow: Fraction, ids: tuple[int, int, int]) -> Cell:
    """x = 1 + shift - v - bow*u*v, y = 1 + shift - u - bow*u*v."""
    return _triangle(
        {(0, 0): 1 + shift, (0, 1): Fraction(-1), (1, 1): -bow},
        {(0, 0): 1 + shift, (1, 0): Fraction(-1), (1, 1): -bow},
        ids,
    )


_PAIRS = {
    # x + y >= 1 + 2/3 - 1/10 on the second image and <= 1 + 1/7 on the first.
    "separated-after-bisection": (
        _FIRST,
        _reflected(Fraction(1, 3), Fraction(1, 5), (3, 4, 5)),
    ),
    "true-overlap": (_FIRST, _reflected(Fraction(-1, 3), Fraction(1, 5), (3, 4, 5))),
    # The translated copy touches the first image only at the shared vertex (1, 0).
    "shared-vertex-contact": (
        _FIRST,
        _triangle(
            {(0, 0): Fraction(1), (1, 0): Fraction(1), (1, 1): _KAPPA},
            {(0, 1): Fraction(1), (1, 1): _KAPPA},
            (1, 6, 7),
        ),
    ),
}


def _controls(cell: Cell, budget: CoordinateEnclosureBudget) -> QuadraticSimplexControls:
    matrices = quadratic_simplex_matrices(cell.coordinates, cell.dimension, budget=budget)
    assert matrices is not None
    return quadratic_simplex_controls(matrices, cell.dimension, budget=budget)


def _charts(cell: Cell, depth: int) -> list[tuple[np.ndarray, np.ndarray]]:
    level = [(np.zeros((cell.dimension,)), np.eye(cell.dimension))]
    charts = list(level)
    for _ in range(depth):
        level = [child for chart in level for child in children(cell, *chart)]
        charts.extend(level)
    return charts


@pytest.mark.parametrize("dimension", (1, 2, 3))
@pytest.mark.parametrize("degree", (0, 1, 2))
def test_quadratic_matrix_is_the_exact_homogeneous_source(
    dimension: int, degree: int
) -> None:
    polynomial = _quadratic(dimension, degree, Fraction(1))
    with coordinate_reference_budget() as budget:
        matrix = quadratic_simplex_matrix(polynomial, dimension, budget=budget)
        controls = bernstein_coefficients(polynomial, "simplex", dimension, budget=budget)
    assert matrix is not None
    size = dimension + 1
    assert all(matrix[i][j] == matrix[j][i] for i in range(size) for j in range(size))
    for point in _points(dimension):
        weights = (1 - sum(point), *point)
        homogeneous = sum(
            (
                weights[i] * matrix[i][j] * weights[j]
                for i in range(size)
                for j in range(size)
            ),
            Fraction(0),
        )
        assert homogeneous == _value(polynomial, point)
    # Elevation never widens the exact Bernstein hull of the source degree.
    entries = [value for row in matrix for value in row]
    assert (min(entries), max(entries)) == (min(controls), max(controls))


@pytest.mark.parametrize(
    "scale",
    (Fraction(1), Fraction(1, 10**320), Fraction(10**320), -Fraction(10**320)),
    ids=("unit", "subnormal", "overflow", "negative-overflow"),
)
def test_controls_round_every_exact_entry_outward(scale: Fraction) -> None:
    polynomials = tuple(_quadratic(3, 2, scale * (axis + 1)) for axis in range(3))
    with coordinate_reference_budget() as budget:
        matrices = quadratic_simplex_matrices(polynomials, 3, budget=budget)
        assert matrices is not None
        controls = quadratic_simplex_controls(matrices, 3, budget=budget)
    assert controls.lower.shape == controls.upper.shape == (1, 3, 4, 4)
    assert controls.lower.flags.c_contiguous and not controls.lower.flags.writeable
    assert not controls.upper.flags.writeable
    retained = (
        sys.getsizeof(controls)
        + sys.getsizeof(controls.lower)
        + sys.getsizeof(controls.upper)
    )
    assert retained <= quadratic_simplex_storage_upper(3, 3)
    for axis, matrix in enumerate(matrices):
        for i, j in itertools.product(range(4), repeat=2):
            exact = matrix[i][j]
            lower = controls.lower[0, axis, i, j]
            upper = controls.upper[0, axis, i, j]
            assert lower == fraction_endpoint(exact, upper=False, saturate=True)
            assert upper == fraction_endpoint(exact, upper=True, saturate=True)
            assert lower != math.inf and upper != -math.inf
            assert math.isinf(lower) or Fraction(lower) <= exact
            assert math.isinf(upper) or exact <= Fraction(upper)


def test_unsupported_sources_have_no_quadratic_controls() -> None:
    cubic: Polynomial = {(3, 0): Fraction(1, 3), (0, 1): Fraction(1)}
    quotient = RationalPolynomial(
        {(1, 0): Fraction(1)}, {(0, 0): Fraction(1), (1, 0): Fraction(1, 8)}
    )
    linear: Polynomial = {(1, 0): Fraction(1)}
    four: Polynomial = {(1, 0, 0, 0): Fraction(1)}
    mismatched: Polynomial = {(1, 0, 0): Fraction(1)}
    unsupported: tuple[tuple[Expression, ...], ...] = (
        (cubic, linear),
        (quotient, linear),
        (linear,) * 4,
    )
    with coordinate_reference_budget() as budget:
        for coordinates in unsupported:
            assert quadratic_simplex_matrices(coordinates, 2, budget=budget) is None
        assert quadratic_simplex_matrices((four,), 4, budget=budget) is None
        assert quadratic_simplex_matrices((mismatched,), 2, budget=budget) is None


def test_control_preparation_refuses_on_its_own_resource_ledger() -> None:
    polynomials = tuple(_quadratic(3, 2, Fraction(1)) for _ in range(3))
    limits = MeshCertificateLimits(maximum_work_units=8)
    with coordinate_reference_budget(limits) as budget:
        with pytest.raises(CoordinateEnclosureResourceError):
            quadratic_simplex_matrices(polynomials, 3, budget=budget)


def test_chart_barycentric_rows_are_exact_or_absent() -> None:
    for kind in ("triangle", "tetrahedron"):
        dimension = 2 if kind == "triangle" else 3
        cell = Cell(
            tuple({(0,) * dimension: Fraction(0)} for _ in range(dimension)),
            "simplex",
            kind,
            dimension,
            tuple(range(dimension + 1)),
            vertices(kind),
        )
        for origin, matrix in _charts(cell, 3 if dimension == 2 else 2):
            rows = chart_barycentric((origin, matrix))
            assert rows is not None
            for k, row in enumerate(rows):
                point = [Fraction(value) for value in origin]
                if k:
                    point = [
                        p + Fraction(value)
                        for p, value in zip(point, matrix[:, k - 1], strict=True)
                    ]
                assert [Fraction(value) for value in row] == [1 - sum(point), *point]
    # 0.1 + 0.2 rounds in binary64; the chart keeps its exact route.
    inexact = (np.asarray([0.1, 0.0]), np.asarray([[0.2, 0.0], [0.0, 0.5]]))
    assert chart_barycentric(inexact) is None
    # A vertex outside the root simplex is not a barycentric enclosure input.
    assert chart_barycentric((np.asarray([-0.5, 0.0]), np.eye(2))) is None
    assert chart_barycentric((np.asarray([0.75, 0.0]), 0.5 * np.eye(2))) is None


@pytest.mark.meshcore
@requires_meshcore
@pytest.mark.parametrize(
    "scale",
    (Fraction(1), Fraction(1, 3 * 10**12), Fraction(7 * 10**12, 3)),
    ids=("unit", "small", "large"),
)
@pytest.mark.parametrize("dimension", (2, 3))
def test_native_chart_boxes_contain_exact_composed_boxes(
    dimension: int, scale: Fraction
) -> None:
    kind = "triangle" if dimension == 2 else "tetrahedron"
    cell = Cell(
        tuple(_quadratic(dimension, 2, scale * (axis + 2)) for axis in range(dimension)),
        "simplex",
        kind,
        dimension,
        tuple(range(dimension + 1)),
        vertices(kind),
    )
    with coordinate_reference_budget() as budget:
        controls = _controls(cell, budget)
        charts = _charts(cell, 3 if dimension == 2 else 2)
        # A near-degenerate deep chart: 2^-30 edges keep exact binary64 vertices.
        deep = (np.full((dimension,), 2.0**-3), 2.0**-30 * np.eye(dimension))
        for chart in (*charts, deep):
            box = quadratic_chart_box(controls, chart, None, budget=budget)
            assert box is not None
            lower, upper = bounds(cell, *chart, budget=budget)
            assert np.all(box.lower <= lower) and np.all(upper <= box.upper)
            assert np.all(np.isfinite(box.lower)) and np.all(np.isfinite(box.upper))


def _bisection(
    first: Cell,
    second: Cell,
    route: str,
    *,
    depth: int = 8,
    maximum_work_units: int = 200_000_000,
) -> tuple[ContactDecision, int, int]:
    """Return a decision, its bisection work, and the request's total work."""
    limits = MeshCertificateLimits(
        maximum_subdivision_depth=depth,
        maximum_subdivision_pieces=50_000,
        maximum_work_units=maximum_work_units,
    )
    with coordinate_reference_budget(limits) as budget:
        # Every route prepares the same root controls, so exact memos agree.
        controls = (_controls(first, budget), _controls(second, budget))
        start = budget.work_units
        quadratic = {"exact": None, "both": controls, "first": (controls[0], None)}[route]
        decision = _bisect_cell_contact(
            first, second, limits, 50_000, 0, quadratic=quadratic, budget=budget
        )
        return decision, budget.work_units - start, budget.work_units


def _decision(contact: ContactDecision) -> tuple[str, int, str | None]:
    return contact.status, contact.work, contact.check


def _tetrahedron(offset: Fraction, ids: tuple[int, ...]) -> Cell:
    # x = u + offset + (v + offset)^2/16, y = v + offset, z = w + offset.
    x: Polynomial = {
        (0, 0, 0): offset + offset**2 / 16,
        (1, 0, 0): Fraction(1),
        (0, 1, 0): offset / 8,
        (0, 2, 0): Fraction(1, 16),
    }
    y: Polynomial = {(0, 0, 0): offset, (0, 1, 0): Fraction(1)}
    z: Polynomial = {(0, 0, 0): offset, (0, 0, 1): Fraction(1)}
    return Cell((x, y, z), "simplex", "tetrahedron", 3, ids, vertices("tetrahedron"))


_SEPARATED = {
    "triangles": _PAIRS["separated-after-bisection"],
    # Every second-image point has y, z >= 2/3; no first-image point has both.
    "tetrahedra": (
        _tetrahedron(Fraction(0), (0, 1, 2, 3)),
        _tetrahedron(Fraction(2, 3), (4, 5, 6, 7)),
    ),
}


@pytest.mark.meshcore
@requires_meshcore
@pytest.mark.parametrize("name", tuple(_PAIRS))
def test_native_filter_preserves_every_exact_bisection_decision(name: str) -> None:
    first, second = _PAIRS[name]
    exact = _bisection(first, second, "exact")[0]
    filtered = _bisection(first, second, "both")[0]
    assert _decision(filtered) == _decision(exact)
    assert exact.status == {
        "separated-after-bisection": "separated",
        "true-overlap": "overlap",
        "shared-vertex-contact": "unresolved",
    }[name]


@pytest.mark.meshcore
@requires_meshcore
@pytest.mark.parametrize("name", tuple(_SEPARATED))
def test_native_filter_separates_within_one_unit_less_than_exact_work(
    name: str,
) -> None:
    first, second = _SEPARATED[name]
    filtered, _, filtered_total = _bisection(first, second, "both")
    exact, _, exact_total = _bisection(first, second, "exact")
    assert exact.status == "separated"
    assert _decision(filtered) == _decision(exact)
    assert filtered_total < exact_total
    with pytest.raises(CoordinateEnclosureResourceError):
        _bisection(first, second, "exact", maximum_work_units=exact_total - 1)
    bounded = _bisection(first, second, "both", maximum_work_units=exact_total - 1)[0]
    assert _decision(bounded) == _decision(exact)


@pytest.mark.meshcore
@requires_meshcore
def test_shared_curved_edge_pair_decision_is_unchanged_by_the_filter() -> None:
    # The second image shares the first's bulged hypotenuse with the same trace.
    second = _triangle(
        {(0, 0): Fraction(1), (1, 0): Fraction(-1), (1, 1): _KAPPA},
        {(0, 0): Fraction(1), (0, 1): Fraction(-1), (1, 1): _KAPPA},
        (8, 2, 1),
    )
    limits = MeshCertificateLimits(
        maximum_subdivision_depth=4, maximum_subdivision_pieces=50_000
    )
    with coordinate_reference_budget(limits) as budget:
        exact_checks: list[str] = []
        exact = pair_decision(
            _FIRST, second, limits, 50_000, checks=exact_checks, budget=budget
        )
        controls = (_controls(_FIRST, budget), _controls(second, budget))
        filtered_checks: list[str] = []
        filtered = pair_decision(
            _FIRST,
            second,
            limits,
            50_000,
            checks=filtered_checks,
            quadratic_controls=lambda: controls,
            budget=budget,
        )
    assert _decision(filtered) == _decision(exact)
    assert filtered_checks == exact_checks


def test_trace_mismatch_refuses_before_any_quadratic_preparation() -> None:
    # Same shared vertices, but the second edge bows by 1/7 instead of 2/7.
    second = _triangle(
        {(0, 0): Fraction(1), (1, 0): Fraction(-1), (1, 1): _KAPPA / 2},
        {(0, 0): Fraction(1), (0, 1): Fraction(-1), (1, 1): _KAPPA / 2},
        (8, 2, 1),
    )

    def prepare() -> tuple[
        QuadraticSimplexControls | None, QuadraticSimplexControls | None
    ]:
        raise AssertionError("a mismatched trace must not prepare controls")

    limits = MeshCertificateLimits(
        maximum_subdivision_depth=4, maximum_subdivision_pieces=50_000
    )
    with coordinate_reference_budget(limits) as budget:
        contact = pair_decision(
            _FIRST, second, limits, 50_000, quadratic_controls=prepare, budget=budget
        )
    assert _decision(contact) == ("unresolved", 0, "mapped_trace_mismatch")


@pytest.mark.parametrize("name", tuple(_SEPARATED))
def test_one_unsupported_side_keeps_the_whole_exact_route(name: str) -> None:
    first, second = _SEPARATED[name]
    exact, exact_work, _ = _bisection(first, second, "exact")
    mixed, mixed_work, _ = _bisection(first, second, "first")
    assert _decision(mixed) == _decision(exact)
    # No native box is charged: the work is exactly the exact route's work.
    assert mixed_work == exact_work


class _AdmissionRefused(Exception):
    pass


class _RefusingLive:
    def __init__(self, admissions: list[int]) -> None:
        self.bound = 0
        self._admissions = admissions

    def set_bound(self, value: int, /) -> None:
        self._admissions.append(value)
        raise _AdmissionRefused


def test_retained_controls_are_admitted_before_any_array_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[int] = []
    admissions: list[int] = []

    def allocate(*_: object, **__: object) -> QuadraticSimplexControls:
        allocations.append(1)
        raise AssertionError("arrays were allocated before admission")

    monkeypatch.setattr(meshcore_owner, "meshcore_available", lambda: True)
    monkeypatch.setattr(embedding_cells, "quadratic_simplex_controls", allocate)
    live = cast(CoordinateLiveStorage, _RefusingLive(admissions))
    with coordinate_reference_budget() as budget:
        with pytest.raises(_AdmissionRefused):
            quadratic_controls(_FIRST, live, budget=budget)
        assert admissions == [quadratic_simplex_storage_upper(2, 2)]
        admissions.clear()
        cubic = _triangle({(3, 0): Fraction(1, 3)}, {(0, 1): Fraction(1)}, (0, 1, 2))
        assert quadratic_controls(cubic, live, budget=budget) is None
    assert admissions == [] and allocations == []


def test_native_refusal_propagates_instead_of_separating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _PAIRS["separated-after-bisection"]
    limits = MeshCertificateLimits(maximum_subdivision_depth=5)

    def refuse(*_: object) -> tuple[np.ndarray, np.ndarray]:
        raise MeshcoreError(MeshcoreStatus.TIMEOUT, "simplex_quadratic_bounds")

    monkeypatch.setattr(meshcore_owner, "simplex_quadratic_bounds", refuse, raising=False)
    with coordinate_reference_budget(limits) as budget:
        quadratic = (_controls(first, budget), _controls(second, budget))
        with pytest.raises(MeshcoreError):
            _bisect_cell_contact(
                first, second, limits, 1_000, 0, quadratic=quadratic, budget=budget
            )


@pytest.mark.meshcore
@requires_meshcore
def test_native_box_refuses_insufficient_native_work() -> None:
    with coordinate_reference_budget() as preparation:
        controls = _controls(_FIRST, preparation)
    # The host reservation fits; the native interval products do not.
    work = 2 * 3 * 3
    limits = NativeExecutionLimits(work, 0, 0, 1_000_000, 30.0)
    scope = NativeExecutionScope(limits)
    with pytest.raises(MeshcoreError) as refusal:
        with scope:
            ticket = scope.batch(work, bytes_upper=1_000_000)
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=1_000_000
            )
            with ticket, budget.activate():
                quadratic_chart_box(
                    controls,
                    (np.zeros((2,)), 0.5 * np.eye(2)),
                    None,
                    budget=budget,
                )
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize(
    ("scale", "offset"),
    ((1.0, 3.0), (1.0, 0.25), (0.1, 0.2)),
    ids=("disjoint-components", "intersecting-components", "contained-component"),
)
def test_p2_mapped_certificate_is_identical_without_the_filter(
    scale: float, offset: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Small genuine P2 mapped diagnostic: the public certificate is unchanged."""
    mesh, geometry, validity = curved_tetrahedral_components(scale, offset, 2)
    filtered = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    monkeypatch.setattr(
        embedding_cells, "quadratic_simplex_matrices", lambda *_, **__: None
    )
    exact = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert (
        filtered.status,
        filtered.findings,
        filtered.evaluated_checks,
        filtered.subdivision_piece_count,
    ) == (
        exact.status,
        exact.findings,
        exact.evaluated_checks,
        exact.subdivision_piece_count,
    )
