#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

from phydrax.domain._sampling import get_sampler_host


def test_hammersley_sampler_is_deterministic_and_bounded():
    sample = get_sampler_host("hammersley", dim=3, seed=0)(16)
    repeat = get_sampler_host("hammersley", dim=3, seed=123)(16)
    assert sample.shape == (16, 3)
    assert np.all(sample >= 0.0)
    assert np.all(sample <= 1.0)
    assert np.allclose(sample, repeat)
    assert np.unique(sample, axis=0).shape[0] == 16


def test_hammersley_first_axis_is_stratified():
    sample = get_sampler_host("hammersley", dim=2, seed=0)(8)
    expected = (np.arange(1, 9) - 0.5) / 8.0
    assert np.allclose(sample[:, 0], expected)
