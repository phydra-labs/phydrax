#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import numpy as np
import pytest

from phydrax.algebraic._exact_integer import exact_integer_determinant


def test_exact_integer_determinant_tracks_row_swaps_and_orientation() -> None:
    # A transposition and a cyclic permutation have independent sign meaning.
    assert exact_integer_determinant(((0, 1, 0), (1, 0, 0), (0, 0, 1))) == -1
    assert exact_integer_determinant(((0, 0, 1), (1, 0, 0), (0, 1, 0))) == 1
    # Requires a pivot swap after the first elimination step.
    assert exact_integer_determinant(((1, 2, 3), (2, 4, 7), (1, 3, 3))) == -1
    assert exact_integer_determinant(np.asarray(((2, 1), (7, 4)), dtype=np.int64)) == 1


def test_exact_integer_determinant_is_arbitrary_precision() -> None:
    large = 10**30
    # det = large * (large + 1) - large * large = large, exceeding int64.
    value = exact_integer_determinant(((large, large), (large, large + 1)))
    assert value == large
    assert isinstance(value, int)


def test_exact_integer_determinant_returns_zero_for_singular_matrices() -> None:
    assert exact_integer_determinant(((1, 2, 3), (2, 4, 6), (0, 1, 5))) == 0
    assert exact_integer_determinant(((0, 0), (0, 3))) == 0


@pytest.mark.parametrize(
    "values",
    (((1.0, 0.0), (0.0, 1.0)), ((True, False), (False, True))),
    ids=("float", "bool"),
)
def test_exact_integer_determinant_refuses_non_integer_kinds(values: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        exact_integer_determinant(values)  # ty: ignore[invalid-argument-type]


def test_exact_integer_determinant_refuses_nonsquare_matrices() -> None:
    with pytest.raises(ValueError, match="square"):
        exact_integer_determinant(((1, 2, 3), (4, 5, 6)))
