"""Independent directed contour-pair oracle for sequence form factors."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.applications.polymer_liquids import SequenceFormFactorPlan
from phydrax.applications.polymer_liquids._mixture import SequenceFormFactorKind


@pytest.mark.parametrize("count", [1, 3, 11, 64])
@pytest.mark.parametrize(
    "kind", ["gaussian-chain", "freely-jointed-chain", "gaussian-ring"]
)
def test_sequence_form_factor_dense_reference(
    count: int, kind: SequenceFormFactorKind
) -> None:
    sequence = (
        np.arange(count, dtype=np.int32) ** 2 + np.arange(count, dtype=np.int32)
    ) % min(count, 3)
    # Keep every declared channel occupied, with unequal multiplicities.
    sequence[: min(count, 3)] = np.arange(min(count, 3), dtype=np.int32)
    sites = min(count, 3)
    length = 1.3
    wave = np.asarray(
        [0.0, 0.17, np.pi / length, 4.5 / length, 7.8 / length], dtype=np.float64
    )
    expected = np.zeros((wave.size, sites, sites), dtype=np.float64)
    for left in range(count):
        for right in range(count):
            distance = abs(left - right)
            if kind == "freely-jointed-chain":
                kernel = np.sinc(wave * length / np.pi) ** distance
            else:
                effective = (
                    distance
                    if kind == "gaussian-chain"
                    else distance * (count - distance) / count
                )
                kernel = np.exp(-(wave**2) * length**2 * effective / 6.0)
            expected[:, sequence[left], sequence[right]] += kernel
    populations = np.bincount(sequence, minlength=sites)
    expected /= np.sqrt(populations[:, None] * populations[None, :])[None, :, :]
    actual = SequenceFormFactorPlan(kind, sequence, length, site_count=sites).evaluate(
        wave
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        np.asarray(actual)[0],
        np.sqrt(populations[:, None] * populations[None, :]),
        rtol=2e-12,
    )
