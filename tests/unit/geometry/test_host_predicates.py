#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from pathlib import Path

import jax
import numpy as np
import pytest

from phydrax._geometry_predicates import (
    orient2d,
    orient2d_host,
    orient2d_enclosure_host,
    orient3d,
    orient3d_host,
    orient3d_enclosure_host,
    PredicateMode,
    PredicateSign,
)
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import MeshCertificateLimits
from tests._support.coordinate_references import coordinate_reference_budget


def _orientation_sign(points: tuple[np.ndarray, ...]) -> int:
    """Independent Leibniz determinant over the binary64 rational coordinates."""
    origin = tuple(Fraction(float(value)) for value in points[0])
    rows = tuple(
        tuple(
            Fraction(float(value)) - base
            for value, base in zip(point, origin, strict=True)
        )
        for point in points[1:]
    )
    dimension = len(origin)
    determinant = Fraction(0)
    for order in permutations(range(dimension)):
        inversions = sum(
            order[i] > order[j] for i in range(dimension) for j in range(i + 1, dimension)
        )
        term = Fraction((-1) ** inversions)
        for row, column in enumerate(order):
            term *= rows[row][column]
        determinant += term
    return (determinant > 0) - (determinant < 0)


def _cases(dimension: int) -> tuple[np.ndarray, ...]:
    random = np.random.default_rng(19).normal(size=(8, dimension + 1, dimension))
    if dimension == 2:
        adversarial = np.asarray(
            (
                ((0.5, 0.5), (12.0, 12.0), (24.0, 24.0)),
                ((0.5, 0.5), (12.0, 12.0), (24.0, 24.0 + 2.0**-48)),
            ),
            dtype=np.float64,
        )
    else:
        adversarial = np.asarray(
            (
                ((0.5, 0.5, 0.5), (1.0, 12.0, 12.0), (6.0, 2.0, 2.0), (-7.0, 3.0, 3.0)),
                (
                    (0.5, 0.5, 0.5 + 2.0**-52),
                    (1.0, 12.0, 12.0),
                    (6.0, 2.0, 2.0),
                    (-7.0, 3.0, 3.0),
                ),
            ),
            dtype=np.float64,
        )
    rows = np.concatenate((random, adversarial))
    return tuple(rows[:, index] for index in range(dimension + 1))


@pytest.mark.parametrize("dimension", (2, 3), ids=("planar", "spatial"))
@pytest.mark.parametrize("exponent", (-350, 0, 350))
def test_orientation_enclosure_contains_the_exact_binary64_determinant(
    exponent: int, dimension: int,
) -> None:
    points = tuple(np.ldexp(value, exponent) for value in _cases(dimension))
    with coordinate_reference_budget() as budget:
        lower, upper = (
            orient2d_enclosure_host(*points, budget=budget)
            if dimension == 2
            else orient3d_enclosure_host(*points, budget=budget)
        )
    for row in range(points[0].shape[0]):
        origin = tuple(Fraction(float(value)) for value in points[0][row])
        edges = tuple(
            tuple(
                Fraction(float(value)) - base
                for value, base in zip(point[row], origin, strict=True)
            )
            for point in points[1:]
        )
        exact = Fraction(0)
        for order in permutations(range(dimension)):
            inversions = sum(
                order[i] > order[j]
                for i in range(dimension) for j in range(i + 1, dimension)
            )
            term = Fraction((-1) ** inversions)
            for index, column in enumerate(order):
                term *= edges[index][column]
            exact += term
        if np.isfinite(lower[row]):
            assert Fraction(float(lower[row])) <= exact
        else:
            assert lower[row] == -np.inf
        if np.isfinite(upper[row]):
            assert exact <= Fraction(float(upper[row]))
        else:
            assert upper[row] == np.inf


@pytest.mark.parametrize("dimension", (2, 3), ids=("planar", "spatial"))
@pytest.mark.parametrize("mode", (PredicateMode.FILTERED, PredicateMode.EXACT))
def test_host_orientation_matches_device_publication_and_rational_reference(
    dimension: int, mode: PredicateMode
) -> None:
    points = _cases(dimension)
    host = (
        orient2d_host(*points, mode=mode)
        if dimension == 2
        else orient3d_host(*points, mode=mode)
    )
    device = (
        orient2d(*points, mode=mode) if dimension == 2 else orient3d(*points, mode=mode)
    )
    expected = np.asarray(
        tuple(
            _orientation_sign(tuple(point[row] for point in points))
            for row in range(len(points[0]))
        ),
        dtype=np.int8,
    )
    assert isinstance(host.signs, np.ndarray)
    assert isinstance(host.certain, np.ndarray)
    assert isinstance(device.signs, jax.Array)
    assert isinstance(device.certain, jax.Array)
    assert host.signs.dtype == np.dtype(np.int8)
    assert host.certain.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(host.signs, device.signs)
    np.testing.assert_array_equal(host.certain, device.certain)
    np.testing.assert_array_equal(host.signs[host.certain], expected[host.certain])
    np.testing.assert_array_equal(host.signs[~host.certain], PredicateSign.UNCERTAIN)
    if mode is PredicateMode.EXACT:
        assert np.all(host.certain)
        np.testing.assert_array_equal(host.signs, expected)


@pytest.mark.parametrize("dimension", (2, 3), ids=("planar", "spatial"))
def test_host_orientation_dyadic_fallback_retains_exact_signs(
    dimension: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", str(tmp_path / "missing.so"))
    points = _cases(dimension)
    host = (
        orient2d_host(*points, mode=PredicateMode.EXACT)
        if dimension == 2
        else orient3d_host(*points, mode=PredicateMode.EXACT)
    )
    device = (
        orient2d(*points, mode=PredicateMode.EXACT)
        if dimension == 2
        else orient3d(*points, mode=PredicateMode.EXACT)
    )
    expected = np.asarray(
        tuple(
            _orientation_sign(tuple(point[row] for point in points))
            for row in range(len(points[0]))
        ),
        dtype=np.int8,
    )
    assert np.all(host.certain)
    np.testing.assert_array_equal(host.signs, expected)
    np.testing.assert_array_equal(host.signs, device.signs)
    np.testing.assert_array_equal(host.certain, device.certain)


@pytest.mark.parametrize("dimension", (2, 3), ids=("planar", "spatial"))
def test_host_orientation_refuses_device_mode(dimension: int) -> None:
    points = _cases(dimension)
    with pytest.raises(ValueError, match=r"resolve_host_predicate_mode\.mode"):
        if dimension == 2:
            orient2d_host(*points, mode=PredicateMode.FILTERED_DEVICE)
        else:
            orient3d_host(*points, mode=PredicateMode.FILTERED_DEVICE)


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
def test_host_orientation_retains_native_out_of_domain_refusal() -> None:
    tiny = 1.0e-40
    points = tuple(
        np.full(2, factor * tiny, dtype=np.float64) for factor in (1.0, 2.0, 3.0)
    )
    with pytest.raises(
        ValueError, match=r"orient2d\.coordinates.*RANGE_ERROR"
    ) as host_refusal:
        orient2d_host(*points, mode=PredicateMode.EXACT)
    with pytest.raises(
        ValueError, match=r"orient2d\.coordinates.*RANGE_ERROR"
    ) as device_refusal:
        orient2d(*points, mode=PredicateMode.EXACT)
    assert str(host_refusal.value) == str(device_refusal.value)


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
def test_native_orientation_memo_reuses_only_equal_ordered_content() -> None:
    points = tuple(np.repeat(point[-2:-1], 3, axis=0) for point in _cases(3))
    limits = MeshCertificateLimits(maximum_work_units=10**9, maximum_scratch_bytes=10**9)
    with coordinate_reference_budget(limits) as budget:
        first = orient3d_host(*points, mode=PredicateMode.EXACT, budget=budget)
        replay = orient3d_host(
            *(point.copy() for point in points), mode=PredicateMode.EXACT, budget=budget
        )
        assert budget.exact_predicate_evaluations == {"orient3d": 1}
        assert budget.exact_predicate_cache_hits == {"orient3d": 5}
        reordered = (points[1], points[0], points[2], points[3])
        reversed_result = orient3d_host(
            *reordered, mode=PredicateMode.EXACT, budget=budget
        )
        assert budget.exact_predicate_evaluations == {"orient3d": 2}
        assert budget.exact_predicate_cache_hits == {"orient3d": 7}
    expected = _orientation_sign(tuple(point[0] for point in points))
    np.testing.assert_array_equal(first.signs, [expected] * 3)
    np.testing.assert_array_equal(replay.signs, first.signs)
    np.testing.assert_array_equal(reversed_result.signs, [-expected] * 3)
    assert np.all(reversed_result.certain)
    with coordinate_reference_budget(limits) as other:
        orient3d_host(*points, mode=PredicateMode.EXACT, budget=other)
        assert other.exact_predicate_evaluations == {"orient3d": 1}
        assert other.exact_predicate_cache_hits == {"orient3d": 2}


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
@pytest.mark.parametrize("dimension", (2, 3), ids=("planar", "spatial"))
@pytest.mark.parametrize("memoize", (False, True), ids=("unmemoized", "memoized"))
def test_native_orientation_charges_whole_remainder_reservation(
    dimension: int, memoize: bool
) -> None:
    limits = NativeExecutionLimits(1_000_000, 1_000_000, 0, 1_000_000, 30.0)
    points = tuple(point[-2:] for point in _cases(dimension))
    with NativeExecutionScope(limits) as scope:
        remaining = scope.remaining()
        ticket = scope.batch(
            remaining.remaining_work_units,
            queries_upper=remaining.remaining_geometry_queries,
            bytes_upper=remaining.remaining_managed_bytes,
        )
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=(
                limits.maximum_work_units if memoize or dimension == 3 else 0
            ),
            maximum_memory_bytes=(
                limits.maximum_managed_bytes if memoize or dimension == 3 else 0
            ),
        )
        with ticket, budget.activate():
            assert scope.remaining().remaining_work_units == 0
            assert scope.remaining().remaining_managed_bytes == 0
            result = (
                orient2d_host(*points, mode=PredicateMode.EXACT, budget=budget)
                if dimension == 2
                else orient3d_host(
                    *points, mode=PredicateMode.EXACT, budget=budget,
                    memoize_native=memoize,
                )
            )
            assert ticket.work_units == budget.work_units
            evaluations = budget.exact_predicate_evaluations.get(f"orient{dimension}d", 0)
            if memoize:
                assert evaluations > 0
            else:
                assert budget.exact_predicate_evaluations == {}
    expected = [
        _orientation_sign(tuple(point[row] for point in points))
        for row in range(points[0].shape[0])
    ]
    np.testing.assert_array_equal(result.signs, expected)
    assert np.all(result.certain)
    evidence = scope.completed_evidence()
    if memoize:
        assert evidence.native_primitive_queries == evaluations
    else:
        assert evidence.native_primitive_queries > 0
    assert evidence.memory_counters.allocations > 0


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
def test_direct_orientation_preserves_native_telemetry_and_counter_domains() -> None:
    points = tuple(np.repeat(point[-2:-1], 3, axis=0) for point in _cases(3))

    def run(work: int, queries: int) -> tuple[int, int]:
        limits = NativeExecutionLimits(work, queries, 0, 1_000_000, 30.0)
        with NativeExecutionScope(limits) as scope:
            ticket = scope.batch(work, queries_upper=queries, bytes_upper=1_000_000)
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=work,
                maximum_memory_bytes=1_000_000,
            )
            try:
                with ticket, budget.activate():
                    first = orient3d_host(
                        *points, mode=PredicateMode.EXACT, budget=budget,
                        memoize_native=False,
                    )
                    replay = orient3d_host(
                        *points, mode=PredicateMode.EXACT, budget=budget,
                        memoize_native=False,
                    )
                    np.testing.assert_array_equal(first.signs, replay.signs)
                    np.testing.assert_array_equal(
                        first.signs,
                        [_orientation_sign(tuple(point[0] for point in points))] * 3,
                    )
                    assert np.all(first.certain) and np.all(replay.certain)
                    assert budget.exact_native_sign_cache == {}
                    assert budget.exact_predicate_cache_hits == {}
            finally:
                assert ticket.consumed_work_units <= work
                assert ticket.consumed_geometry_queries <= queries
        evidence = scope.completed_evidence()
        assert evidence.native_primitive_queries == 6
        return ticket.consumed_work_units, ticket.consumed_geometry_queries

    work, queries = run(100_000, 100_000)
    # Native primitive telemetry is not construction work or source queries.
    assert (work, queries) == (0, 0)
    assert run(work, queries) == (work, queries)


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
@pytest.mark.parametrize("dimension", (2, 3), ids=("planar", "spatial"))
def test_native_orientation_refuses_insufficient_reserved_storage(dimension: int) -> None:
    limits = NativeExecutionLimits(1_000_000, 1_000_000, 0, 1_000_000, 30.0)
    points = tuple(point[-2:] for point in _cases(dimension))
    scope = NativeExecutionScope(limits)
    with pytest.raises(MeshcoreError) as refusal:
        with scope:
            ticket = scope.batch(0, bytes_upper=0)
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=0, maximum_memory_bytes=0
            )
            with ticket, budget.activate():
                if dimension == 2:
                    orient2d_host(*points, mode=PredicateMode.EXACT, budget=budget)
                else:
                    orient3d_host(*points, mode=PredicateMode.EXACT, budget=budget)
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert scope.completed_evidence().native_primitive_queries == 0


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
@pytest.mark.parametrize("memoize", (False, True))
def test_host_orientation_preadmits_packed_storage_on_original_request(
    memoize: bool,
) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    limits = NativeExecutionLimits(1_000_000, 1_000_000, 0, 512, 30.0)
    points = tuple(point[-2:] for point in _cases(3))
    scope = NativeExecutionScope(limits)
    with pytest.raises(
        CoordinateEnclosureResourceError, match="polynomial_storage.*limit 512"
    ):
        with scope:
            ticket = scope.batch(
                limits.maximum_work_units,
                queries_upper=limits.maximum_geometry_queries,
                bytes_upper=limits.maximum_managed_bytes,
            )
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_managed_bytes,
            )
            with ticket, budget.activate():
                orient3d_host(
                    *points, mode=PredicateMode.EXACT, budget=budget,
                    memoize_native=memoize,
                )
    assert scope.completed_evidence().native_primitive_queries == 0
