#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import pytest

from phydrax.discretization._exact_algebraic_geometry import (
    _prime_uint64,
    _squarefree,
    RadicalElement,
)


def test_word_prime_and_strong_pseudoprimes() -> None:
    assert _prime_uint64(18446744073709551557)
    assert _prime_uint64(2)
    assert not _prime_uint64(0)
    assert not _prime_uint64(1)
    assert not _prime_uint64(341550071728321)
    assert not _prime_uint64(3825123056546413051)
    assert not _prime_uint64((1 << 64) - 1)
    with pytest.raises(ValueError, match="64-bit"):
        _prime_uint64(1 << 64)


def test_squarefree_large_word_prime_and_residual_after_small_factor() -> None:
    prime = 18446744073709551557
    assert _squarefree(prime) == (1, prime)
    assert _squarefree(2 * prime) == (1, 2 * prime)
    assert _squarefree(25 * prime) == (5, prime)
    assert _squarefree(prime * prime) == (prime, 1)
    assert RadicalElement.sqrt(Fraction(prime)).terms == ((prime, Fraction(1)),)
    assert RadicalElement.sqrt(Fraction(25 * prime)).terms == ((prime, Fraction(5)),)
