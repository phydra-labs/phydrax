#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native exact kernels preserve exact results, enclosure and boundary safety."""

from __future__ import annotations

import random
import weakref
from fractions import Fraction
from inspect import signature
from itertools import product
from typing import Any, get_args, Literal

import mpmath
import numpy as np
import pytest

from phydrax import _bvh, _exact_kernels
from phydrax._exact_kernels import (
    bisect_integer_nets,
    bvh_pair_blocks,
    interval_bernstein_restrict,
    rational_bernstein_nets,
    separation_screen,
    SeparationDecision,
    simplex_blossom_extremes,
)
from phydrax._interpolation import _bspline_extraction as extraction
from phydrax._meshcore import meshcore_available
from phydrax.discretization import _coordinate_enclosure as enclosure
from phydrax.discretization._exact_algebraic_geometry import coefficient_endpoint
from phydrax.geometry import _interval_enclosure as interval_enclosure


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    return [
        (first, *rest)
        for first in range(total, -1, -1)
        for rest in _compositions(total - first, parts - 1)
    ]


def _random_polynomial(
    generator: random.Random, domain: str, dimension: int, degree: int
) -> enclosure.Polynomial:
    polynomial: enclosure.Polynomial = {}
    for _ in range(generator.randint(1, 8)):
        if domain == "simplex":
            index = [0] * dimension
            budget = generator.randint(0, degree)
            for axis in range(dimension):
                index[axis] = generator.randint(0, budget)
                budget -= index[axis]
        elif domain == "box":
            index = [generator.randint(0, degree) for _ in range(dimension)]
        else:
            first = generator.randint(0, degree)
            index = [
                first,
                generator.randint(0, degree - first),
                generator.randint(0, degree),
            ]
        kind = generator.random()
        if kind < 0.4:
            value = Fraction(generator.uniform(-4.0, 4.0))
        elif kind < 0.8:
            value = Fraction(generator.randint(-50, 50), generator.randint(1, 30))
        else:
            value = Fraction(
                generator.randint(-(10**30), 10**30), 3 ** generator.randint(0, 40)
            )
        polynomial[tuple(index)] = value
    return {index: value for index, value in polynomial.items() if value}


@pytest.mark.parametrize(
    ("domain", "dimension"),
    [
        ("simplex", 1),
        ("simplex", 2),
        ("simplex", 3),
        ("box", 2),
        ("box", 3),
        ("prism", 3),
    ],
)
def test_rational_bernstein_nets_equal_fraction_controls(
    domain: Any, dimension: int
) -> None:
    generator = random.Random(f"{domain}:{dimension}")
    polynomials = []
    degrees = []
    for _ in range(60):
        polynomial = _random_polynomial(
            generator, domain, dimension, generator.randint(1, 4)
        )
        least = enclosure.bernstein_degree(polynomial, domain, dimension)
        polynomials.append(polynomial)
        degrees.append(tuple(value + generator.randint(0, 1) for value in least))
    serial = rational_bernstein_nets(
        polynomials, domain, dimension, degrees, parallelism=1
    )
    threaded = rational_bernstein_nets(
        polynomials, domain, dimension, degrees, parallelism=4
    )
    for polynomial, degree, net, other in zip(
        polynomials, degrees, serial, threaded, strict=True
    ):
        reference = enclosure._fraction_controls(polynomial, domain, dimension, degree)
        assert net is not None and other is not None
        assert net.words == other.words and net.scale == other.scale
        controls = net.controls()
        assert controls == reference
        assert all(type(value) is Fraction for value in controls)
        assert net.control(net.least) == min(reference)
        assert net.control(net.greatest) == max(reference)
        least = enclosure._fraction_controls(
            polynomial,
            domain,
            dimension,
            enclosure.bernstein_degree(polynomial, domain, dimension),
        )
        assert enclosure.polynomial_bounds(polynomial, domain, dimension) == (
            coefficient_endpoint(min(least), upper=False, saturate=True),
            coefficient_endpoint(max(least), upper=True, saturate=True),
        )
        assert (
            enclosure.bernstein_coefficients(polynomial, domain, dimension, degree=degree)
            == reference
        )


def test_rational_bernstein_nets_widen_words_and_refuse_beyond_them() -> None:
    wide: enclosure.Polynomial = {
        (index, 12 - index): Fraction(3**600 + index, 2**700) for index in range(13)
    }
    huge: enclosure.Polynomial = {(1, 0): Fraction(2**5000), (0, 1): Fraction(1, 3)}
    plain: enclosure.Polynomial = {(1, 1): Fraction(3, 7), (0, 0): Fraction(-1)}
    nets = rational_bernstein_nets(
        [wide, huge, plain, {(0, 0): 0.5}], "box", 2, [(12, 12), (1, 1), (1, 1), (0, 0)]
    )
    assert nets[0] is not None and nets[0].word_bytes > 32
    assert nets[0].controls() == enclosure._fraction_controls(wide, "box", 2, (12, 12))
    assert nets[1] is None
    assert nets[2] is not None
    assert nets[2].controls() == enclosure._fraction_controls(plain, "box", 2, (1, 1))
    assert nets[3] is None
    # Refused items keep exact Fraction controls on the host path.
    assert enclosure.bernstein_coefficients(
        huge, "box", 2
    ) == enclosure._fraction_controls(huge, "box", 2, (1, 1))


def test_dense_box_bounds_use_wide_words() -> None:
    generator = random.Random(41)
    polynomial: enclosure.Polynomial = {
        (a, b, c): Fraction(generator.randint(-(2**900), 2**900), 2**950)
        for a in range(7)
        for b in range(7)
        for c in range(7)
    }
    net = rational_bernstein_nets([polynomial], "box", 3, [(6, 6, 6)])[0]
    assert net is not None
    reference = enclosure._fraction_controls(polynomial, "box", 3, (6, 6, 6))
    assert net.controls() == reference
    assert enclosure.polynomial_bounds(polynomial, "box", 3) == (
        coefficient_endpoint(min(reference), upper=False, saturate=True),
        coefficient_endpoint(max(reference), upper=True, saturate=True),
    )


def test_interval_restriction_is_bit_identical_to_the_host_steps() -> None:
    generator = random.Random(5)
    noise = np.random.default_rng(5)
    for _ in range(200):
        degree = generator.randint(1, 5)
        width = generator.randint(1, 4)
        lower = noise.normal(size=(degree + 1, width))
        upper = lower + noise.uniform(0.0, 1.0e-3, size=lower.shape)
        if generator.random() < 0.5:
            first = Fraction(generator.randint(0, 5), 8)
            last = first + Fraction(generator.randint(0, 8 - int(first * 8)), 8)
        else:
            first = Fraction(generator.randint(0, 99), 300)
            last = first + Fraction(generator.randint(0, 200), 300)
        reference = extraction._interval_restrict_axis0(lower, upper, first, last)
        parameters = np.zeros((1, 4))
        sides = np.zeros((1,), dtype=np.int32)
        if last < 1:
            parameters[0, :2] = extraction._interval_parameter(last)
            sides[0] |= 1
        if first > 0:
            parameters[0, 2:] = extraction._interval_ratio(first, last)
            sides[0] |= 2
        native = interval_bernstein_restrict(lower[None], upper[None], parameters, sides)
        for computed, expected in zip(native, reference, strict=True):
            assert computed[0].tobytes() == np.ascontiguousarray(expected).tobytes()


def _blossom_reference(
    net: dict[tuple[int, ...], int], corners: list[tuple[int, int, int]], degree: int
) -> list[int]:
    nets: list[tuple[int, dict[tuple[int, ...], int]]] = [(0, net)]
    for remaining in range(degree - 1, -1, -1):
        following = []
        for last, current in nets:
            for corner in range(last, 3):
                w0, w1, w2 = corners[corner]
                following.append(
                    (
                        corner,
                        {
                            index: w0 * current[(index[0] + 1, index[1], index[2])]
                            + w1 * current[(index[0], index[1] + 1, index[2])]
                            + w2 * current[(index[0], index[1], index[2] + 1)]
                            for index in _compositions(remaining, 3)
                        },
                    )
                )
        nets = following
    return [current[(0, 0, 0)] for _, current in nets]


def test_simplex_blossoms_match_the_integer_corner_walk() -> None:
    generator = random.Random(17)
    nets, corners, degrees, references = [], [], [], []
    for _ in range(40):
        degree = generator.randint(1, 5)
        controls = {
            index: generator.randint(-(10**12), 10**12)
            for index in _compositions(degree, 3)
        }
        scale = generator.randint(1, 2**20)
        rows = []
        for _ in range(3):
            first = generator.randint(0, scale)
            second = generator.randint(0, scale - first)
            rows.append((scale - first - second, first, second))
        values = _blossom_reference(controls, rows, degree)
        nets.append([controls[index] for index in _compositions(degree, 3)])
        corners.append(rows)
        degrees.append(degree)
        references.append((min(values), max(values)))
    assert (
        simplex_blossom_extremes(nets, corners, degrees, 3, parallelism=1) == references
    )
    assert (
        simplex_blossom_extremes(nets, corners, degrees, 3, parallelism=4) == references
    )
    wide = [(2**300, 0, 0), (2**299, 2**299, 0), (2**299, 0, 2**299)]
    expected = _blossom_reference(
        dict(zip(_compositions(degrees[0], 3), nets[0], strict=True)), wide, degrees[0]
    )
    assert simplex_blossom_extremes([nets[0]], [wide], [degrees[0]], 3) == [
        (min(expected), max(expected))
    ]
    refused = simplex_blossom_extremes(
        [nets[0]], [[(2**5000, 0, 0)] * 3], [degrees[0]], 3
    )
    assert refused == [None]


def test_bisection_matches_fraction_halving() -> None:
    generator = random.Random(3)
    degree = 3
    indices = _compositions(degree, 4)
    nets = [
        [tuple(generator.randint(-(10**15), 10**15) for _ in range(3)) for _ in indices]
        for _ in range(5)
    ]
    result = bisect_integer_nets(nets, indices, 0, 2, parallelism=4)
    assert result.exponent == degree
    scale = Fraction(1, 2**result.exponent)
    for net, halves in zip(nets, result.halves, strict=True):
        assert halves is not None
        left, right = halves
        lines: dict[tuple[int, ...], list[int]] = {}
        for row, index in enumerate(indices):
            lines.setdefault((index[1], index[3]), []).append(row)
        for rows in lines.values():
            ordered = sorted(rows, key=lambda row: indices[row][2])
            levels = [[tuple(Fraction(value) for value in net[row]) for row in ordered]]
            for _ in range(len(ordered) - 1):
                previous = levels[-1]
                levels.append(
                    [
                        tuple(
                            (a + b) / 2
                            for a, b in zip(previous[r], previous[r + 1], strict=True)
                        )
                        for r in range(len(previous) - 1)
                    ]
                )
            total = len(ordered) - 1
            for step, row in enumerate(ordered):
                assert tuple(value * scale for value in left[row]) == levels[step][0]
                assert (
                    tuple(value * scale for value in right[row])
                    == levels[total - step][step]
                )


def test_separation_screen_decisions_equal_exact_outcome() -> None:
    noise = np.random.default_rng(23)
    first_points, second_points, normals = [], [], []
    first_offsets, second_offsets = [0], [0]
    for _ in range(300):
        first = noise.normal(size=(int(noise.integers(1, 8)), 3))
        second = noise.normal(size=(int(noise.integers(1, 8)), 3)) + noise.normal(
            scale=3.0, size=3
        )
        normal = np.round(noise.normal(size=3) * 2.0**20)
        first_points.append(first)
        second_points.append(second)
        normals.append(normal)
        first_offsets.append(first_offsets[-1] + first.shape[0])
        second_offsets.append(second_offsets[-1] + second.shape[0])
    first_all = np.concatenate(first_points)
    second_all = np.concatenate(second_points)
    decisions = separation_screen(
        first_all,
        first_all,
        np.asarray(first_offsets),
        second_all,
        second_all,
        np.asarray(second_offsets),
        np.asarray(normals),
        parallelism=1,
    )
    threaded = separation_screen(
        first_all,
        first_all,
        np.asarray(first_offsets),
        second_all,
        second_all,
        np.asarray(second_offsets),
        np.asarray(normals),
        parallelism=4,
    )
    assert np.array_equal(decisions, threaded)
    decided = 0
    for decision, first, second, normal in zip(
        decisions, first_points, second_points, normals, strict=True
    ):
        exact_normal = [Fraction(value) for value in normal]
        first_values = [
            sum(n * Fraction(x) for n, x in zip(exact_normal, point, strict=True))
            for point in first
        ]
        second_values = [
            sum(n * Fraction(x) for n, x in zip(exact_normal, point, strict=True))
            for point in second
        ]
        separated = max(second_values) < min(first_values)
        if decision == SeparationDecision.SEPARATED:
            assert separated
        elif decision == SeparationDecision.NOT_SEPARATED:
            assert not separated
        decided += decision != SeparationDecision.UNDECIDED
    # Generic pairs are decided by the binary64 filter alone.
    assert decided >= 0.95 * len(normals)


@pytest.mark.parametrize(
    ("item_block", "node_block", "touching", "tolerance"),
    [(64, 8, True, 0.0), (1 << 20, 1 << 16, False, 0.0), (256, 32, True, 1.0e-3)],
)
def test_bvh_pair_blocks_repeat_the_host_block_sequence(
    item_block: int, node_block: int, touching: bool, tolerance: float, monkeypatch: Any
) -> None:
    monkeypatch.setattr(_bvh, "_HOST_ITEM_PAIR_BLOCK", item_block)
    monkeypatch.setattr(_bvh, "_HOST_NODE_PAIR_BLOCK", node_block)
    noise = np.random.default_rng(item_block)
    lower = noise.uniform(0.0, 20.0, (700, 3))
    upper = lower + noise.uniform(0.01, 1.5, (700, 3))
    other_lower = noise.uniform(0.0, 20.0, (531, 3))
    other_upper = other_lower + noise.uniform(0.01, 1.5, (531, 3))
    first = _bvh.prepare_bvh(lower, upper, dtype=np.float64)
    second = _bvh.prepare_bvh(other_lower, other_upper, dtype=np.float64)
    test = _bvh._validate_host_pair_arguments(first, second, touching, tolerance, 0.0)
    for left, right in ((first, second), (first, first)):
        reference = list(
            _bvh._host_pair_blocks(
                _bvh._host_arrays(left), _bvh._host_arrays(right), test, item_block
            )
        )
        for parallelism in (1, 4):
            native = list(
                bvh_pair_blocks(
                    _bvh._host_arrays(left),
                    _bvh._host_arrays(right),
                    include_touching=touching,
                    absolute_tolerance=tolerance,
                    relative_tolerance=0.0,
                    item_block=item_block,
                    node_block=node_block,
                    parallelism=parallelism,
                )
            )
            assert len(native) == len(reference) > 0
            for (native_first, native_second), (first_ids, second_ids) in zip(
                native, reference, strict=True
            ):
                assert native_first.dtype == first_ids.dtype == np.int64
                assert np.array_equal(native_first, first_ids)
                assert np.array_equal(native_second, second_ids)


def test_bvh_pair_blocks_reference_constants(monkeypatch: Any) -> None:
    monkeypatch.setattr(_bvh, "_HOST_ITEM_PAIR_BLOCK", 64)
    monkeypatch.setattr(_bvh, "_HOST_NODE_PAIR_BLOCK", 8)
    noise = np.random.default_rng(1)
    lower = noise.uniform(0.0, 10.0, (400, 2))
    upper = lower + noise.uniform(0.01, 1.0, (400, 2))
    tree = _bvh.prepare_bvh(lower, upper, dtype=np.float64)
    test = _bvh._validate_host_pair_arguments(tree, tree, True, 0.0, 0.0)
    reference = list(
        _bvh._host_pair_blocks(_bvh._host_arrays(tree), _bvh._host_arrays(tree), test, 64)
    )
    blocks = list(_bvh.bvh_overlap_pair_blocks(tree, tree, include_touching=True))
    assert len(blocks) == len(reference) > 1
    for block, expected in zip(blocks, reference, strict=True):
        assert all(np.array_equal(a, b) for a, b in zip(block, expected, strict=True))
    pairs = _bvh.bvh_overlap_pairs_host(tree, tree, include_touching=True)
    brute = sorted(
        (i, j)
        for i, j in product(range(400), repeat=2)
        if np.all(np.minimum(upper[i], upper[j]) - np.maximum(lower[i], lower[j]) >= 0.0)
    )
    assert list(zip(pairs[0].tolist(), pairs[1].tolist(), strict=True)) == brute


@pytest.mark.parametrize("item_block", (1, 17, 256, 4096))
def test_bvh_pair_schedule_preserves_native_host_sequence(
    item_block: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    lower = np.zeros((64, 3), dtype=np.float64)
    upper = np.ones((64, 3), dtype=np.float64)
    tree = _bvh.prepare_bvh(lower, upper, dtype=np.float64)
    baseline = list(_bvh.bvh_overlap_pair_blocks(tree, tree, include_touching=True))
    native = list(
        _bvh.bvh_overlap_pair_blocks(
            tree, tree, include_touching=True, item_block=item_block
        )
    )
    monkeypatch.setattr(_bvh, "meshcore_available", lambda: False)
    host = list(
        _bvh.bvh_overlap_pair_blocks(
            tree, tree, include_touching=True, item_block=item_block
        )
    )

    def flatten(blocks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, ...]:
        return tuple(np.concatenate([block[axis] for block in blocks]) for axis in (0, 1))

    assert all(
        np.array_equal(a, b) for a, b in zip(flatten(native), flatten(host), strict=True)
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(flatten(native), flatten(baseline), strict=True)
    )


@pytest.mark.parametrize("item_block", (0, -1, True, 1.5))
def test_bvh_pair_schedule_rejects_invalid_execution_count(
    item_block: int | float,
) -> None:
    points = np.zeros((1, 3), dtype=np.float64)
    tree = _bvh.prepare_bvh(points, points, dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        arguments = signature(_bvh.bvh_overlap_pair_blocks).bind(
            tree, tree, include_touching=True, item_block=item_block
        )
        _bvh.bvh_overlap_pair_blocks(*arguments.args, **arguments.kwargs)


def test_float64_interval_kernels_are_bit_identical_to_the_host_operations() -> None:
    noise = np.random.default_rng(13)
    specials = np.asarray(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            5.0e-324,
            -5.0e-324,
            1.0e300,
            -1.0e300,
            2.0**996,
            3.0,
        ]
    )
    count = 4000
    values = noise.normal(size=(4, count)) * 10.0 ** noise.integers(
        -320, 300, size=(4, count)
    )
    values = np.where(
        noise.random((4, count)) < 0.3, noise.choice(specials, (4, count)), values
    )
    first = (np.minimum(values[0], values[1]), np.maximum(values[0], values[1]))
    second = (np.minimum(values[2], values[3]), np.maximum(values[2], values[3]))
    first[0][:7] = np.nan
    for public, host in (
        (
            interval_enclosure.interval_multiply,
            interval_enclosure._host_interval_multiply,
        ),
        (interval_enclosure.interval_add, interval_enclosure._host_interval_add),
        (
            interval_enclosure.interval_subtract,
            interval_enclosure._host_interval_subtract,
        ),
    ):
        expected = host(first, second)
        computed = public(first, second)
        for native, reference in zip(computed, expected, strict=True):
            assert native.dtype == np.float64 and native.shape == reference.shape
            assert native.tobytes() == np.asarray(reference).tobytes()
    # Broadcast operands and 0-d endpoints keep the host shapes.
    scalar = (np.float64(-2.0), np.float64(3.0))
    broadcast = interval_enclosure.interval_multiply(
        scalar, (second[0][:5], second[1][:5])
    )
    reference = interval_enclosure._host_interval_multiply(
        scalar, (second[0][:5], second[1][:5])
    )
    for native, host in zip(broadcast, reference, strict=True):
        assert native.tobytes() == np.asarray(host).tobytes()


@pytest.mark.parametrize("dtype", (np.float32, np.int32, np.float64))
@pytest.mark.parametrize("operation", ("product", "add", "subtract"))
def test_interval_marshalling_normalizes_small_and_strided_buffers(
    dtype: Any, operation: str
) -> None:
    first_lower = np.asarray(((1, 99, 2, 99),), dtype=dtype)[:, ::2]
    first_upper = first_lower + 1
    second_lower = np.asarray(2, dtype=dtype)
    second_upper = np.asarray(((3,), (4,)), dtype=dtype)
    arguments = (first_lower, first_upper, second_lower, second_upper)
    if operation == "product":
        lower, upper = _exact_kernels.float64_interval_product(*arguments)
        expected = (first_lower * second_lower, first_upper * second_upper)
    else:
        subtract = operation == "subtract"
        lower, upper = _exact_kernels.float64_interval_sum(*arguments, subtract=subtract)
        expected = (
            (first_lower - second_upper, first_upper - second_lower)
            if subtract
            else (first_lower + second_lower, first_upper + second_upper)
        )
    assert lower.shape == upper.shape == (2, 2)
    assert lower.dtype == upper.dtype == np.float64
    assert np.array_equal(lower, np.broadcast_to(expected[0], (2, 2)))
    assert np.array_equal(upper, np.broadcast_to(expected[1], (2, 2)))


def _small_host_tree(dimension: int = 2) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((1, dimension)),
        np.ones((1, dimension)),
        np.asarray((-1,), dtype=np.int64),
        np.asarray((-1,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        np.zeros((2, dimension)),
        np.ones((2, dimension)),
    )


def _small_pair_stream(
    first: tuple[np.ndarray, ...], second: tuple[np.ndarray, ...]
) -> Any:
    return bvh_pair_blocks(
        first,
        second,
        include_touching=True,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
        item_block=4,
        node_block=1,
    )


@pytest.mark.parametrize("index", range(8))
def test_bvh_rejects_small_dtypes_before_loading_native(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = list(_small_host_tree())
    tree[index] = tree[index].astype(np.float32 if index in (0, 1, 6, 7) else np.int32)
    monkeypatch.setattr(
        _exact_kernels,
        "load_meshcore",
        lambda: pytest.fail("invalid tree reached native loading"),
    )
    with pytest.raises(ValueError, match="float64.*int64"):
        next(_small_pair_stream(tuple(tree), _small_host_tree()))


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (0, np.zeros(2)),
        (1, np.ones((1, 1))),
        (1, np.ones((2, 2))),
        (2, np.empty(0, dtype=np.int64)),
        (3, np.empty(0, dtype=np.int64)),
        (4, np.empty(0, dtype=np.int64)),
        (2, np.full((1, 1), -1, dtype=np.int64)),
        (5, np.empty((0, 2), dtype=np.int64)),
        (5, np.empty((1, 0), dtype=np.int64)),
        (5, np.asarray((0, 1), dtype=np.int64)),
        (6, np.zeros((2, 1))),
        (6, np.zeros(2)),
        (7, np.ones((1, 2))),
        (5, np.asarray(((0, 2),), dtype=np.int64)),
    ],
)
def test_bvh_rejects_inconsistent_extents_before_loading_native(
    index: int, replacement: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = list(_small_host_tree())
    tree[index] = replacement
    monkeypatch.setattr(
        _exact_kernels,
        "load_meshcore",
        lambda: pytest.fail("invalid tree reached native loading"),
    )
    with pytest.raises(ValueError):
        next(_small_pair_stream(tuple(tree), _small_host_tree()))


def test_bvh_rejects_cross_tree_dimension_before_loading_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _exact_kernels,
        "load_meshcore",
        lambda: pytest.fail("dimension mismatch reached native loading"),
    )
    with pytest.raises(ValueError, match="same dimension"):
        next(_small_pair_stream(_small_host_tree(2), _small_host_tree(1)))


@pytest.mark.parametrize("finish", ("close", "throw", "native_error", "exhaust"))
def test_bvh_stream_retains_owners_and_frees_on_every_exit(
    finish: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _exact_kernels.load_meshcore()
    original_view = _exact_kernels._tree_view
    owners: list[weakref.ReferenceType[np.ndarray]] = []
    freed: list[int] = []
    calls = 0

    def view(tree: tuple[np.ndarray, ...]) -> Any:
        result, arrays = original_view(tree)
        # Both are newly allocated, not kept alive by the caller's source tree.
        owners.extend(weakref.ref(arrays[index]) for index in (0, 2))
        return result, arrays

    def advance(*args: Any) -> int:
        nonlocal calls
        assert all(owner() is not None for owner in owners)
        calls += 1
        if calls == 2 and finish == "native_error":
            return int(_exact_kernels.MeshcoreStatus.ALLOCATION_FAILURE)
        return library["phx_mc_bvh_pair_stream_next"](*args)

    def free(handle: Any) -> None:
        assert all(owner() is not None for owner in owners)
        freed.append(handle.value)
        library["phx_mc_bvh_pair_stream_free"](handle)

    monkeypatch.setattr(_exact_kernels, "_tree_view", view)
    monkeypatch.setattr(
        _exact_kernels,
        "load_meshcore",
        lambda: {
            "phx_mc_bvh_pair_stream_create": library["phx_mc_bvh_pair_stream_create"],
            "phx_mc_bvh_pair_stream_next": advance,
            "phx_mc_bvh_pair_stream_free": free,
        },
    )
    tree = list(_small_host_tree())
    tree[0] = np.zeros((1, 4))[:, ::2]
    stream = _small_pair_stream(tuple(tree), tuple(tree))
    block = next(stream)
    assert list(zip(*block, strict=True)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    saved = tuple(value.copy() for value in block)
    if finish == "close":
        stream.close()
    elif finish == "throw":
        with pytest.raises(RuntimeError, match="consumer failure"):
            stream.throw(RuntimeError("consumer failure"))
    elif finish == "native_error":
        with pytest.raises(RuntimeError, match="bvh_pair_stream_next"):
            next(stream)
    else:
        assert list(stream) == []
    assert len(freed) == 1
    assert all(owner() is None for owner in owners)
    assert all(np.array_equal(a, b) for a, b in zip(block, saved, strict=True))


def test_bvh_yielded_blocks_do_not_alias_reused_output_buffers() -> None:
    tree = _bvh.prepare_bvh(np.zeros((80, 2)), np.ones((80, 2)), dtype=np.float64)
    arrays = _bvh._host_arrays(tree)
    stream = _small_pair_stream(arrays, arrays)
    first = next(stream)
    saved = tuple(value.copy() for value in first)
    count = len(first[0])
    for block in stream:
        count += len(block[0])
        assert not any(np.shares_memory(a, b) for a in first for b in block)
        block[0][:] = -1
    assert count == 80 * 80
    assert all(np.array_equal(a, b) for a, b in zip(first, saved, strict=True))


# WIN-5 audit counterexamples: libm-padded host enclosures that excluded the true
# range. C1: tan boxes straddling a pole; C2: sin/cos boxes around an extremum;
# C3 and the exp point: libm values more than one padded ulp off.
_POLE_BOXES = (
    ("0x1.1abe4b73fefb5p+6", "0x1.1abe4b73fefb6p+6"),
    ("0x1.4d02421c87558p+6", "0x1.4d02421c87559p+6"),
    ("0x1.7f4638c50fafbp+6", "0x1.7f4638c50fafcp+6"),
    ("0x1.0525177794649p+10", "0x1.052517779464ap+10"),
    ("0x1.0dc8c5dc8bd41p+10", "0x1.0dc8c5dc8bd42p+10"),
)
_EXTREMUM_BOXES = (
    ("0x1.22b056b5fff13p+32", "0x1.22b056b5fff14p+32"),
    ("0x1.6161e49e749a0p+32", "0x1.6161e49e749a1p+32"),
    ("0x1.6f477b705c833p+30", "0x1.6f477b705c834p+30"),
    ("0x1.2548680d3aa50p+29", "0x1.2548680d3aa51p+29"),
    ("0x1.7e8b4d8a376cdp+31", "0x1.7e8b4d8a376cep+31"),
    ("0x1.41748d545c13dp+31", "0x1.41748d545c13ep+31"),
    ("0x1.7e8b4d93a42b1p+31", "0x1.7e8b4d93a42b2p+31"),
    ("0x1.41748d5dc8d21p+31", "0x1.41748d5dc8d22p+31"),
    ("0x1.840807e204d00p+31", "0x1.840807e204d01p+31"),
)


def _boxes(rows: tuple[tuple[str, str], ...]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([float.fromhex(row[0]) for row in rows], dtype=np.float64)
    upper = np.asarray([float.fromhex(row[1]) for row in rows], dtype=np.float64)
    return lower, upper


def test_float64_interval_elementary_encloses_tangent_pole_counterexamples() -> None:
    lower, upper = _boxes(_POLE_BOXES)
    with mpmath.workprec(256):
        for low, high in zip(lower, upper, strict=True):
            pole = mpmath.ceil(mpmath.mpf(low) / mpmath.pi - mpmath.mpf(0.5))
            assert mpmath.mpf(low) <= (pole + mpmath.mpf(0.5)) * mpmath.pi <= high
    result = _exact_kernels.float64_interval_elementary("tan", lower, upper)
    assert np.all(result[0] == -np.inf) and np.all(result[1] == np.inf)


@pytest.mark.parametrize("function", ("sin", "cos"))
def test_float64_interval_elementary_encloses_extremum_counterexamples(
    function: Literal["sin", "cos"],
) -> None:
    lower, upper = _boxes(_EXTREMUM_BOXES)
    result = _exact_kernels.float64_interval_elementary(function, lower, upper)
    evaluate = mpmath.sin if function == "sin" else mpmath.cos
    phase = mpmath.mpf(0.5) if function == "sin" else mpmath.mpf(0)
    with mpmath.workprec(256):
        for index, (low, high) in enumerate(zip(lower, upper, strict=True)):
            values = [evaluate(mpmath.mpf(low)), evaluate(mpmath.mpf(high))]
            critical = (
                mpmath.ceil(mpmath.mpf(low) / mpmath.pi - phase) + phase
            ) * mpmath.pi
            if critical <= high:
                values.append(evaluate(critical))
            assert result[0][index] <= min(values) and max(values) <= result[1][index]


@pytest.mark.parametrize(
    ("function", "point"),
    (("tanh", "-0x1.1476b62c72180p-1"), ("exp", "-0x1.dd3739315e97ap-27")),
    ids=("tanh_C3", "exp_point"),
)
def test_float64_interval_elementary_encloses_point_counterexamples(
    function: Literal["tanh", "exp"], point: str
) -> None:
    value = np.asarray(float.fromhex(point), dtype=np.float64)
    lower, upper = _exact_kernels.float64_interval_elementary(function, value, value)
    with mpmath.workprec(256):
        exact = getattr(mpmath, function)(mpmath.mpf(float(value)))
    assert lower <= exact <= upper
    assert np.nextafter(lower, np.inf) >= upper


_INFINITY = np.inf
_SMALLEST = 5.0e-324


@pytest.mark.parametrize(
    ("function", "box", "expected"),
    (
        ("exp", (np.nan, 0.0), (-_INFINITY, _INFINITY)),
        ("sin", (0.0, np.nan), (-_INFINITY, _INFINITY)),
        ("exp", (0.0, 0.0), (1.0, 1.0)),
        ("exp", (-_INFINITY, -_INFINITY), (0.0, _SMALLEST)),
        ("exp", (_INFINITY, _INFINITY), (np.finfo(np.float64).max, _INFINITY)),
        ("log", (1.0, 1.0), (0.0, 0.0)),
        ("log", (-1.0, 1.0), (-_INFINITY, 0.0)),
        ("log", (-2.0, -0.0), (-_INFINITY, _INFINITY)),
        ("sin", (-0.0, -0.0), (-0.0, -0.0)),
        ("cos", (-0.0, 0.0), (1.0, 1.0)),
        ("sin", (0.0, 7.0), (-1.0, 1.0)),
        ("cos", (1.0, _INFINITY), (-1.0, 1.0)),
        ("tan", (1.0, 2.0), (-_INFINITY, _INFINITY)),
        ("atan", (-0.0, -0.0), (-0.0, -0.0)),
        ("tanh", (-_INFINITY, _INFINITY), (-1.0, 1.0)),
    ),
    ids=(
        "nan_exp",
        "nan_sin",
        "exp_zero",
        "exp_minus_infinity",
        "exp_plus_infinity",
        "log_one",
        "log_nonpositive_lower",
        "log_nonpositive_box",
        "sin_negative_zero",
        "cos_signed_zero",
        "sin_wide",
        "cos_unbounded",
        "tan_pole",
        "atan_negative_zero",
        "tanh_unbounded",
    ),
)
def test_float64_interval_elementary_endpoint_conventions(
    function: _exact_kernels.ElementaryFunction,
    box: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    lower, upper = _exact_kernels.float64_interval_elementary(
        function, np.float64(box[0]), np.float64(box[1])
    )
    assert lower.tobytes() == np.float64(expected[0]).tobytes()
    assert upper.tobytes() == np.float64(expected[1]).tobytes()


def test_float64_interval_elementary_broadcasts_and_ignores_parallelism() -> None:
    noise = np.random.default_rng(5)
    lower = noise.normal(size=(3, 700)) * 10.0 ** noise.integers(-8, 8, size=(3, 700))
    upper = lower.max(axis=0, keepdims=True) + np.abs(noise.normal(size=(1, 700)))
    for function in get_args(_exact_kernels.ElementaryFunction.__value__):
        serial = _exact_kernels.float64_interval_elementary(
            function, lower, upper, parallelism=1
        )
        parallel = _exact_kernels.float64_interval_elementary(
            function, lower, upper, parallelism=3
        )
        assert serial[0].shape == lower.shape and serial[0].dtype == np.float64
        for first, second in zip(serial, parallel, strict=True):
            assert first.tobytes() == second.tobytes()
        assert np.all(serial[0] <= serial[1])


def test_float64_interval_elementary_refuses_unknown_functions() -> None:
    with pytest.raises(ValueError, match="function"):
        _exact_kernels.float64_interval_elementary(
            "sinh",  # ty: ignore[invalid-argument-type]
            np.zeros(2),
            np.ones(2),
        )


@pytest.mark.parametrize("function", ("asin", "acos"))
def test_float64_interval_elementary_encloses_inverse_sine_and_cosine(
    function: Literal["asin", "acos"],
) -> None:
    noise = np.random.default_rng(29)
    points = np.concatenate(
        (
            noise.uniform(-1.0, 1.0, size=300),
            noise.uniform(-1.0, 1.0, size=100) * 10.0 ** noise.integers(-20, 0, size=100),
            np.asarray((-1.0, -0.5, -0.0, 0.0, 0.5, 1.0, np.nextafter(1.0, 0.0))),
        )
    )
    lower, upper = _exact_kernels.float64_interval_elementary(function, points, points)
    evaluate = getattr(mpmath, function)
    with mpmath.workprec(256):
        for x, low, high in zip(points, lower, upper, strict=True):
            exact = evaluate(mpmath.mpf(float(x)))
            assert low <= exact <= high
            assert np.nextafter(np.nextafter(low, np.inf), np.inf) >= high


@pytest.mark.parametrize(
    ("function", "box", "expected"),
    (
        ("asin", (-0.0, -0.0), (-0.0, -0.0)),
        ("acos", (1.0, 1.0), (0.0, 0.0)),
        ("asin", (2.0, 3.0), (-_INFINITY, _INFINITY)),
        ("acos", (np.nan, 0.5), (-_INFINITY, _INFINITY)),
        ("asin", (-_INFINITY, 0.0), (np.nextafter(-np.pi / 2, -np.inf), 0.0)),
        ("acos", (0.0, 5.0), (0.0, np.nextafter(np.pi / 2, np.inf))),
    ),
    ids=(
        "asin_negative_zero",
        "acos_one",
        "asin_empty",
        "acos_nan",
        "asin_clipped",
        "acos_clipped",
    ),
)
def test_float64_interval_elementary_inverse_domain_rules(
    function: _exact_kernels.ElementaryFunction,
    box: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    lower, upper = _exact_kernels.float64_interval_elementary(
        function, np.float64(box[0]), np.float64(box[1])
    )
    assert lower.tobytes() == np.float64(expected[0]).tobytes()
    assert upper.tobytes() == np.float64(expected[1]).tobytes()
