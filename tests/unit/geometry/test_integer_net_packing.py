from __future__ import annotations

from fractions import Fraction
from random import Random

import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._coordinate_enclosure import BernsteinDomain, Expression
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry import _integer_net_batch as batch


@pytest.mark.parametrize("limbs", [4, 8, 16, 32, 64])
def test_packed_words_match_independent_signed_encoding(limbs: int) -> None:
    random = Random(714 + limbs)
    bits = 64 * limbs
    values = [0, 1, -1, -(1 << 63), (1 << 63) - 1]
    values.extend(random.randrange(-(1 << 63), 1 << 63) for _ in range(37))
    expected = b"".join(
        value.to_bytes(limbs * 8, "little", signed=True) for value in values
    )
    assert batch._pack_words(values, limbs) == expected
    assert batch._pack_words([], limbs) == b""

    values.extend([1 << 100, -(1 << 100), (1 << (bits - 1)) - 1, -(1 << (bits - 1))])
    values.extend(
        random.randrange(-(1 << (bits - 1)), 1 << (bits - 1)) for _ in range(37)
    )
    expected = b"".join(
        value.to_bytes(limbs * 8, "little", signed=True) for value in values
    )
    assert batch._pack_words(values, limbs) == expected


def test_native_packing_exactness_and_item_local_refusal() -> None:
    if not meshcore_available():
        pytest.skip("native meshcore is unavailable")
    random = Random(20261117)
    requests: list[
        tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]
    ] = []
    expected: list[tuple[Fraction, ...]] = []
    for bits in (8, 63, 64, 101, 180, 400, 900, 1900, 3900):
        values = tuple(
            Fraction(random.getrandbits(bits) * sign, denominator)
            for sign, denominator in ((1, 3), (-1, 5), (1, 7))
        )
        polynomials: tuple[Expression, ...] = tuple({(0,): value} for value in values)
        requests.append((polynomials, "simplex", 1, (0,)))
        expected.append(values)
    huge: Expression = {(0,): Fraction(1 << 4200, 11)}
    radical: Expression = {(0,): RadicalElement.sqrt(2)}
    requests.append(
        (
            (huge, radical, {(0,): Fraction(-2, 3)}, {(0,): Fraction(0)}),
            "simplex",
            1,
            (0,),
        )
    )

    def check(rows: list[tuple[batch._BatchIntegerNet | None, ...]]) -> None:
        for nets, values in zip(rows[:-1], expected, strict=True):
            assert all(net is not None for net in nets)
            assert tuple(net.controls() for net in nets if net is not None) == tuple(
                (value,) for value in values
            )
        refused, irrational, finite, zero = rows[-1]
        assert refused is None
        assert irrational is None
        assert finite is not None and finite.controls() == (Fraction(-2, 3),)
        assert zero is not None and zero.controls() == (Fraction(0),)

    for workers in (1, 2):
        check(batch.integer_net_batch(requests, parallelism=workers))
        check(
            batch.integer_net_batch(requests[:4], parallelism=workers)
            + batch.integer_net_batch(requests[4:], parallelism=workers)
        )


def test_exact_controls_detect_input_perturbation() -> None:
    if not meshcore_available():
        pytest.skip("native meshcore is unavailable")
    values = (Fraction(19, 7), Fraction(18, 7))
    requests: list[
        tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]
    ] = [(({(0,): value},), "simplex", 1, (0,)) for value in values]
    rows = batch.integer_net_batch(requests)
    for row, expected in zip(rows, values, strict=True):
        net = row[0]
        assert net is not None
        assert net.controls() == (expected,)
    assert values[0] != values[1]
