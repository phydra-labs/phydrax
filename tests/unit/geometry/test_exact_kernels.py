#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native exact kernels reproduce their host references bit for bit."""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import product
from typing import Any

import numpy as np
import pytest

from phydrax import _bvh
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
                _bvh._host_arrays(left), _bvh._host_arrays(right), test
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
        _bvh._host_pair_blocks(_bvh._host_arrays(tree), _bvh._host_arrays(tree), test)
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
