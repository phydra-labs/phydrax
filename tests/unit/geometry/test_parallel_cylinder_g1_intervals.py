from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from phydrax.geometry._interval_enclosure import (
    interval_asin,
    interval_divide,
    interval_multiply,
)
from phydrax.geometry.brep._fillet import brep_fillet
from tests.unit.geometry.test_brep_blend import _parallel_cylinder_lens


def _contains(bounds: np.ndarray, witness: mp.mpf) -> bool:
    return mp.mpf(float(bounds[0])) <= witness <= mp.mpf(float(bounds[1]))


@pytest.mark.parametrize("radius", (0.25, 0.375))
def test_parallel_cylinder_fillet_g1_encloses_exact_tangency(radius: float) -> None:
    source, edge = _parallel_cylinder_lens()
    result = brep_fillet(source, (source.edge_ids[edge],), radius)
    assert result.status == "certified", result.reason
    bounds = np.asarray(result.g1_angle_bounds)
    assert bounds.shape == (2, 2)
    # The exact cylinder contact normals coincide, so their angle is exactly zero.
    for row in bounds:
        assert _contains(row, mp.mpf(0))
        inward = row.copy()
        inward[0] = np.nextafter(inward[0], np.inf)
        assert not _contains(inward, mp.mpf(0))


def test_inverse_sine_interval_endpoints_and_batch_independence() -> None:
    # These are exact chord lengths for angles 0, pi/3, and pi.
    chords = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
    two = np.asarray(2.0, dtype=np.float64)

    def enclose(values: np.ndarray) -> np.ndarray:
        ratio = interval_divide((values, values), (two, two))
        ratio = (np.maximum(ratio[0], 0.0), np.minimum(ratio[1], 1.0))
        angles = interval_multiply(interval_asin(ratio), (two, two))
        return np.stack(angles, axis=-1)

    batch = enclose(chords)
    with mp.workdps(100):
        witnesses = (mp.mpf(0), mp.pi / 3, mp.pi)
        for index, witness in enumerate(witnesses):
            assert _contains(batch[index], witness)
            single = enclose(chords[index : index + 1])
            np.testing.assert_array_equal(batch[index], single[0])
            inward = batch[index].copy()
            inward[1] = np.nextafter(float(witness), -np.inf)
            assert not _contains(inward, witness)
