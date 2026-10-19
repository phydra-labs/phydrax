"""Binary similarities for bounded B-Rep property tests."""

from itertools import permutations, product
from typing import NamedTuple

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray


class BinarySimilarity(NamedTuple):
    permutation: tuple[int, ...]
    signs: tuple[int, ...]
    exponent: int

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Permute, sign and scale without nonbinary rounding."""
        return np.ldexp(
            points[..., np.asarray(self.permutation)] * np.asarray(self.signs),
            self.exponent,
        )


SIGNED_AXES = tuple(
    (p, s) for p in permutations((0, 1, 2)) for s in product((-1, 1), repeat=3)
)


@st.composite
def binary_similarities(draw: st.DrawFn) -> BinarySimilarity:
    """Draw one of 48 signed permutations and a binary exponent in [-20, 20]."""
    permutation, signs = draw(st.sampled_from(SIGNED_AXES))
    return BinarySimilarity(permutation, signs, draw(st.integers(-20, 20)))


def assert_scaled_bounds(
    source: NDArray[np.float64], target: NDArray[np.float64], exponent: int
) -> None:
    """Compare homogeneous bounds with one outward ulp on each endpoint."""
    expected = np.ldexp(source, exponent)
    assert np.all(np.nextafter(expected, -np.inf) <= target)
    assert np.all(target <= np.nextafter(expected, np.inf))
