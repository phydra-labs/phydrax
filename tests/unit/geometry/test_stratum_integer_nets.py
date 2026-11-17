from __future__ import annotations

from fractions import Fraction
from math import comb
from random import Random

import numpy as np
import pytest

from phydrax._exact_kernels import BisectedNets
from phydrax.discretization._coordinate_enclosure import (
    BernsteinDomain,
    ExactScalar,
    Expression,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry import _stratum_subdivision as subdivision
from phydrax.geometry._integer_net_batch import _BatchIntegerNet


def test_integer_bisection_range_refusal_keeps_exact_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = 1 << 4200
    values = {
        (2, 0, 0): (Fraction(huge + 1, 3),),
        (1, 1, 0): (Fraction(-huge, 5),),
        (0, 2, 0): (Fraction(7, 11),),
        (1, 0, 1): (Fraction(2, 7),),
        (0, 1, 1): (Fraction(-3, 13),),
        (0, 0, 2): (Fraction(1, 17),),
    }
    net_values: dict[tuple[int, ...], tuple[ExactScalar | int, ...]] = {
        key: row for key, row in values.items()
    }
    net = subdivision._make_controls(net_values)
    monkeypatch.setattr(subdivision, "meshcore_available", lambda: True)
    monkeypatch.setattr(
        subdivision, "bisect_integer_nets", lambda *args: BisectedNets(2, (None,))
    )
    left, right = subdivision._bisect_controls(net, 0, 1)
    a, b, c = (values[index][0] for index in ((2, 0, 0), (1, 1, 0), (0, 2, 0)))
    midpoint = (a + 2 * b + c) / 4
    assert subdivision._decoded_controls(left) == {
        (2, 0, 0): (a,),
        (1, 1, 0): ((a + b) / 2,),
        (0, 2, 0): (midpoint,),
        (1, 0, 1): (Fraction(2, 7),),
        (0, 1, 1): ((Fraction(2, 7) - Fraction(3, 13)) / 2,),
        (0, 0, 2): (Fraction(1, 17),),
    }
    assert subdivision._decoded_controls(right) == {
        (2, 0, 0): (midpoint,),
        (1, 1, 0): ((b + c) / 2,),
        (0, 2, 0): (c,),
        (1, 0, 1): ((Fraction(2, 7) - Fraction(3, 13)) / 2,),
        (0, 1, 1): (Fraction(-3, 13),),
        (0, 0, 2): (Fraction(1, 17),),
    }


def test_radical_bisection_and_cone_signs_remain_exact() -> None:
    root = RadicalElement.sqrt(2)
    net = subdivision._make_controls({(1, 0): (root,), (0, 1): (-root,)})
    left, right = subdivision._bisect_controls(net, 0, 1)
    assert subdivision._decoded_controls(left)[(0, 1)][0] == 0
    assert subdivision._decoded_controls(right)[(1, 0)][0] == 0
    assert subdivision._exactly_positive(np.array([1.0]), [((root,), 1)])
    assert not subdivision._exactly_positive(np.array([1.0]), [((-root,), 1)])


def test_integer_affine_controls_preserve_distinct_denominators() -> None:
    net = subdivision._make_controls({(1, 0): (Fraction(1, 3), Fraction(-2, 5))})
    result = subdivision._affine_controls(
        net, [(Fraction(1, 7), Fraction(2, 11), Fraction(-3, 13))]
    )
    assert subdivision._decoded_controls(result) == {
        (1, 0): (Fraction(1, 7) + Fraction(2, 33) + Fraction(6, 65),)
    }


@pytest.mark.parametrize(
    ("off_diagonal", "expected"),
    [(Fraction(1, 3), True), (Fraction(1), False)],
    ids=("positive", "corrupted-minor"),
)
def test_scaled_jacobian_controls_decide_exact_positivity(
    off_diagonal: Fraction, expected: bool
) -> None:
    net = subdivision._make_controls(
        {(1, 0, 0): (Fraction(2, 3), off_diagonal, off_diagonal, Fraction(2, 3))}
    )
    controls = subdivision._JacobianControls(2, net)
    metric = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    assert subdivision._jacobian_controls_positive(controls, metric) is expected
    assert expected == (Fraction(4, 9) - off_diagonal * off_diagonal > 0)


def test_native_bisection_matches_unbounded_exact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not subdivision.meshcore_available():
        pytest.skip("Meshcore is unavailable")
    controls = subdivision._make_controls(
        {
            (2, 0, 0): (Fraction(1, 3), Fraction(-4, 7)),
            (1, 1, 0): (Fraction(-2, 5), Fraction(5, 11)),
            (0, 2, 0): (Fraction(7, 13), Fraction(1, 2)),
            (1, 0, 1): (Fraction(1, 17), Fraction(3, 4)),
            (0, 1, 1): (Fraction(2, 19), Fraction(-3, 8)),
            (0, 0, 2): (Fraction(3, 23), Fraction(9, 10)),
        }
    )
    native = subdivision._bisect_controls(controls, 0, 1)
    monkeypatch.setattr(subdivision, "meshcore_available", lambda: False)
    fallback = subdivision._bisect_controls(controls, 0, 1)
    for actual, expected in zip(native, fallback, strict=True):
        assert subdivision._decoded_controls(actual) == subdivision._decoded_controls(
            expected
        )


@pytest.mark.parametrize("radical", [False, True])
def test_control_conversion_retains_exact_refused_coefficients(
    monkeypatch: pytest.MonkeyPatch, radical: bool
) -> None:
    coefficient = RadicalElement.sqrt(2) if radical else Fraction(1 << 4200, 3)
    polynomial: dict[tuple[int, ...], ExactScalar] = {
        (0,): Fraction(1, 7),
        (1,): coefficient,
    }
    monkeypatch.setattr(subdivision, "meshcore_available", lambda: False)
    controls = subdivision._controls((polynomial,), (2,), (1,), 1)
    assert controls is not None
    assert subdivision._decoded_controls(controls) == {
        (1, 0): (Fraction(1, 7),),
        (0, 1): (Fraction(1, 7) + coefficient,),
    }


def test_bisection_matches_fraction_blossoms_for_all_triangle_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subdivision, "meshcore_available", lambda: False)
    random = Random(92643)
    for degree in range(1, 5):
        indices = subdivision._control_indices((3,), (degree,))
        values = {
            index: (Fraction(random.randrange(-30, 31), random.randrange(1, 17)),)
            for index in indices
        }
        net_values: dict[tuple[int, ...], tuple[ExactScalar | int, ...]] = dict(values)
        for first, second in ((0, 1), (0, 2), (1, 2)):
            left, right = subdivision._bisect_controls(
                subdivision._make_controls(net_values), first, second
            )
            left_values = subdivision._decoded_controls(left)
            right_values = subdivision._decoded_controls(right)
            for index in indices:
                total = index[first] + index[second]
                step = index[second]

                def line_value(position: int) -> Fraction:
                    point = list(index)
                    point[first], point[second] = total - position, position
                    return values[tuple(point)][0]

                expected_left = sum(
                    (comb(step, j) * line_value(j) for j in range(step + 1)),
                    Fraction(0),
                ) / (1 << step)
                remaining = total - step
                expected_right = sum(
                    (
                        comb(remaining, j) * line_value(step + j)
                        for j in range(remaining + 1)
                    ),
                    Fraction(0),
                ) / (1 << remaining)
                assert left_values[index] == (expected_left,)
                assert right_values[index] == (expected_right,)


def test_bulk_net_decoding_matches_exact_signed_controls_and_subdivision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random = Random(18271)
    indices = subdivision._control_indices((3,), (3,))
    columns = [
        [-(1 << 63), (1 << 63) - 1, *[random.randrange(-100, 101) for _ in range(8)]],
        [random.randrange(-(1 << 220), 1 << 220) for _ in indices],
        [random.randrange(-1000, 1001) for _ in indices],
        [-(1 << 255), (1 << 255) - 1, *[random.randrange(-50, 51) for _ in range(8)]],
    ]
    scales = (2, 3, 1 << 75, 5)
    widths = (8, 32, 16, 32)
    encoded = [
        b"".join(value.to_bytes(width, "little", signed=True) for value in column)
        for column, width in zip(columns, widths, strict=True)
    ]
    nets = tuple(
        _BatchIntegerNet(
            b"prefix" + words + b"suffix",
            6,
            6 + len(words),
            width,
            len(column),
            scale,
            0,
            0,
        )
        for column, scale, width, words in zip(
            columns, scales, widths, encoded, strict=True
        )
    )
    expected = subdivision._make_controls(
        {
            index: tuple(
                Fraction(column[row], scale)
                for column, scale in zip(columns, scales, strict=True)
            )
            for row, index in enumerate(indices)
        }
    )

    def reject_scalar_decoding(self: _BatchIntegerNet) -> tuple[Fraction, ...]:
        raise AssertionError("bulk conversion must not decode scalar controls")

    monkeypatch.setattr(_BatchIntegerNet, "controls", reject_scalar_decoding)

    def reject_numerators(self: _BatchIntegerNet) -> tuple[int, ...]:
        raise AssertionError("bulk conversion must not decode scalar numerators")

    monkeypatch.setattr(_BatchIntegerNet, "numerators", reject_numerators)
    actual = subdivision._controls_from_nets(({},) * len(nets), (3,), (3,), 2, nets)
    assert actual == expected
    assert actual is not None
    altered = nets[1]
    damaged = _BatchIntegerNet(
        altered.payload[: altered.start]
        + bytes([altered.payload[altered.start] ^ 1])
        + altered.payload[altered.start + 1 :],
        altered.start,
        altered.end,
        altered.word_bytes,
        altered.count,
        altered.scale,
        altered.least,
        altered.greatest,
    )
    corrupted = subdivision._controls_from_nets(
        ({},) * len(nets), (3,), (3,), 2, (nets[0], damaged, *nets[2:])
    )
    assert corrupted != expected
    monkeypatch.setattr(subdivision, "meshcore_available", lambda: False)
    actual_level = [actual]
    expected_level = [expected]
    for edge in ((0, 1), (1, 2), (0, 2)):
        actual_level = [
            child
            for net in actual_level
            for child in subdivision._bisect_controls(net, *edge)
        ]
        expected_level = [
            child
            for net in expected_level
            for child in subdivision._bisect_controls(net, *edge)
        ]
        assert actual_level == expected_level


def test_bulk_net_conversion_is_independent_of_batch_split_and_workers() -> None:
    if not subdivision.meshcore_available():
        pytest.skip("Meshcore is unavailable")
    random = Random(10829)
    requests: list[
        tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]
    ] = [
        (
            tuple(
                {
                    (power,): Fraction(
                        random.randrange(-(1 << bits), 1 << bits), denominator
                    )
                    for power in range(4)
                }
                for bits, denominator in ((12, 3), (190, 7), (60, 11))
            ),
            "simplex",
            1,
            (3,),
        )
        for _ in range(9)
    ]
    expected = [
        subdivision._controls_from_nets(
            polynomials, (2,), degree, dimension, (None,) * len(polynomials)
        )
        for polynomials, _, dimension, degree in requests
    ]
    for workers in (1, 3):
        for split in (1, 4, len(requests)):
            actual = []
            for start in range(0, len(requests), split):
                batch = requests[start : start + split]
                native = subdivision.integer_net_batch(batch, parallelism=workers)
                assert all(net is not None for row in native for net in row)
                control_requests: list[
                    tuple[tuple[Expression, ...], tuple[int, ...], tuple[int, ...], int]
                ] = [
                    (polynomials, (2,), degree, dimension)
                    for polynomials, _, dimension, degree in batch
                ]
                prepared = subdivision._prepare_scaled_columns(control_requests, native)
                actual.extend(
                    subdivision._controls_from_nets(
                        polynomials,
                        (2,),
                        degree,
                        dimension,
                        nets,
                        prepared=prepared.get(position),
                    )
                    for position, (
                        (polynomials, _, dimension, degree),
                        nets,
                    ) in enumerate(zip(batch, native, strict=True))
                )
            assert actual == expected


def test_bulk_control_preparation_preserves_item_local_failure_and_radicals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = RadicalElement.sqrt(2)
    polynomials: tuple[Expression, ...] = ({(0,): Fraction(1, 3), (1,): Fraction(2, 3)},)
    radical: tuple[Expression, ...] = ({(0,): root, (1,): Fraction(1)},)
    requests: list[
        tuple[tuple[Expression, ...], tuple[int, ...], tuple[int, ...], int]
    ] = [
        (polynomials, (2,), (1,), 1),
        (radical, (2,), (1,), 1),
        (polynomials, (2,), (1,), 1),
        (polynomials, (2,), (1,), 1),
    ]
    words = b"".join(value.to_bytes(8, "little", signed=True) for value in (1, 3))
    valid = _BatchIntegerNet(words, 0, len(words), 8, 2, 3, 0, 1)
    invalid = _BatchIntegerNet(words, 0, len(words), 8, 2, 0, 0, 1)
    native: list[tuple[_BatchIntegerNet | None, ...]] = [
        (valid,),
        (None,),
        (invalid,),
        (valid,),
    ]
    expected = [
        subdivision._controls_from_nets(polynomials, (2,), (1,), 1, (None,)),
        subdivision._controls_from_nets(radical, (2,), (1,), 1, (None,)),
        None,
    ]
    expected.append(expected[0])

    def batch(
        items: list[tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]],
        /,
    ) -> list[tuple[_BatchIntegerNet | None, ...]]:
        return native

    monkeypatch.setattr(subdivision, "meshcore_available", lambda: True)
    monkeypatch.setattr(subdivision, "integer_net_batch", batch)
    assert subdivision._controls_many(requests, tolerate_errors=True) == expected
