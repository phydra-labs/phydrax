#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import numpy as np

from phydrax.domain._sampling import get_sampler, get_sampler_host


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


def test_halton_and_sobol_plain_and_scrambled_sequences_are_distinct():
    references = {
        "halton": np.asarray(
            [
                [0.0, 0.0],
                [0.5, 1.0 / 3.0],
                [0.25, 2.0 / 3.0],
                [0.75, 1.0 / 9.0],
            ]
        ),
        "sobol": np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.75],
            ]
        ),
    }

    for name, expected in references.items():
        plain = get_sampler_host(name, dim=2, seed=1)(4)
        plain_other_seed = get_sampler_host(name, dim=2, seed=999)(4)
        scrambled_name = f"{name}_scrambled"
        scrambled = get_sampler_host(scrambled_name, dim=2, seed=7)(4)
        scrambled_repeat = get_sampler_host(
            scrambled_name,
            dim=2,
            seed=7,
        )(4)
        scrambled_other_seed = get_sampler_host(
            scrambled_name,
            dim=2,
            seed=8,
        )(4)

        assert plain.shape == scrambled.shape == (4, 2)
        assert np.all((plain >= 0.0) & (plain <= 1.0))
        assert np.all((scrambled >= 0.0) & (scrambled <= 1.0))
        assert np.allclose(plain, expected)
        assert np.array_equal(plain, plain_other_seed)
        assert np.array_equal(scrambled, scrambled_repeat)
        assert not np.array_equal(scrambled, scrambled_other_seed)
        assert not np.array_equal(plain, scrambled)


def test_qmc_callback_wrappers_preserve_scrambling_semantics():
    for name in ("halton", "sobol"):
        plain = get_sampler(name)
        plain_first = np.asarray(plain(8, 3, jr.key(1)))
        plain_second = np.asarray(plain(8, 3, jr.key(2)))
        scrambled = get_sampler(f"{name}_scrambled")
        scrambled_first = np.asarray(scrambled(8, 3, jr.key(3)))
        scrambled_repeat = np.asarray(scrambled(8, 3, jr.key(3)))
        scrambled_second = np.asarray(scrambled(8, 3, jr.key(4)))

        assert np.array_equal(plain_first, plain_second)
        assert np.array_equal(scrambled_first, scrambled_repeat)
        assert not np.array_equal(scrambled_first, scrambled_second)
        assert not np.array_equal(plain_first, scrambled_first)
