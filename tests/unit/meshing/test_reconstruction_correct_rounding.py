from __future__ import annotations

from typing import Literal

import mpmath
import numpy as np
import pytest
from mpmath.libmp import round_nearest, to_float

from tools.meshing_qualification.scenarios import reconstruction


def _oracle(function: Literal["sin", "cos"], values: np.ndarray, /) -> np.ndarray:
    """Independent oracle: mpmath at 1000 bits on the exact argument, rounded once.

    The 1000-bit value is rounded to binary64 a single time under round to
    nearest, ties to even (``mpmath.libmp.to_float(..., rnd=round_nearest)``).
    """
    evaluate = mpmath.mp.sin if function == "sin" else mpmath.mp.cos
    flat = np.asarray(values, dtype=np.float64).ravel().tolist()
    rounded = []
    with mpmath.mp.workprec(1000):
        for value in flat:
            if value == 0.0:
                rounded.append(value if function == "sin" else 1.0)
                continue
            exact = evaluate(mpmath.mpf(value))
            rounded.append(to_float(exact._mpf_, rnd=round_nearest))
    return np.asarray(rounded, dtype=np.float64).reshape(np.shape(values))


def _torus_angles(resolution: int, seed: int, /) -> tuple[np.ndarray, np.ndarray]:
    major, minor = reconstruction._TORUS_RADII
    samples_per_area = 16.0 / (4.0 * np.pi * 0.2**2)
    area = 4.0 * np.pi**2 * major * minor
    count = int(round(samples_per_area * resolution * resolution * area))
    u, v, _ = np.random.default_rng(seed).random(size=(3, 4 * count))
    return 2.0 * np.pi * u, 2.0 * np.pi * v


_HALF_PI_MULTIPLES = np.asarray(
    [k * (np.pi / 2.0) for k in range(1, 9)], dtype=np.float64
)
_HARD_CASES = np.concatenate(
    (
        _HALF_PI_MULTIPLES,
        np.nextafter(_HALF_PI_MULTIPLES, 0.0),
        np.nextafter(_HALF_PI_MULTIPLES, np.inf),
        np.asarray(
            (
                0.0,
                -0.0,
                5.0e-324,
                2.2250738585072014e-308,
                1.0e-300,
                1.0e-8,
                1.0e22,
                1.0e300,
                1.7976931348623157e308,
                -3.0,
            ),
            dtype=np.float64,
        ),
    )
)


@pytest.mark.parametrize("function", ["sin", "cos"])
def test_sin_cos_are_correctly_rounded_on_sampler_angles_and_hard_cases(
    function: Literal["sin", "cos"],
) -> None:
    u, v = _torus_angles(4, 1711)
    for values in (u, v, _HARD_CASES):
        actual = reconstruction._correctly_rounded_sin_cos(function, values)
        expected = _oracle(function, values)
        assert actual.tobytes() == expected.tobytes()


def test_torus_samples_are_bit_identical_to_the_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = reconstruction._surface_samples("torus", 4, np.random.default_rng(1711))
    monkeypatch.setattr(reconstruction, "_correctly_rounded_sin_cos", _oracle)
    expected = reconstruction._surface_samples("torus", 4, np.random.default_rng(1711))
    assert samples.tobytes() == expected.tobytes()


# Published hardest-to-round binary64 arguments: Lefevre, Ly & Zimmermann,
# "Computing hard-to-round cases of sin, cos, tan in double precision",
# ARITH 2026, Tables V and VI (|x| >= 2**10). Inside the sampler range they come
# from CORE-MATH sin.wc/cos.wc: the m = 65 near-representable sine
# 0x1.fe767739d0f6dp-2, and the closest-to-midpoint listed arguments.
_PUBLISHED_SIN = (
    "0x1.fe767739d0f6dp-2",
    "0x1.598bae9e632f6p-7",
    "0x1.1c63df63d59d0p-2",
    "0x1.d98c4c612718dp-1",
    "0x1.e009c53148be1p+991",
    "0x1.b951f1572eba5p+23",
    "0x1.504cac51f1eafp+131",
    "0x1.6ac5b262ca1ffp+849",
    "0x1.db41f3cb71d7bp+680",
    "0x1.4c96c11134d36p+577",
)
_PUBLISHED_COS = (
    "0x1.34ec2f9fc9c00p+1",
    "0x1.6b8a6273d7c21p+0",
    "0x1.16e534ee36580p-4",
    "0x1.5afb7107105d9p+1006",
    "0x1.e009c53148be1p+992",
    "0x1.b7fe89bf86037p+917",
    "0x1.6ac5b262ca1ffp+852",
)


@pytest.mark.parametrize(
    ("function", "arguments"), [("sin", _PUBLISHED_SIN), ("cos", _PUBLISHED_COS)]
)
def test_published_hardest_cases_round_correctly(
    function: Literal["sin", "cos"], arguments: tuple[str, ...]
) -> None:
    values = np.asarray([float.fromhex(item) for item in arguments], dtype=np.float64)
    evaluate = mpmath.mp.sin if function == "sin" else mpmath.mp.cos
    with mpmath.mp.workprec(4000):
        expected = np.asarray(
            [
                to_float(evaluate(mpmath.mpf(value))._mpf_, rnd=round_nearest)
                for value in values.tolist()
            ],
            dtype=np.float64,
        )
    actual = reconstruction._correctly_rounded_sin_cos(function, values)
    assert actual.tobytes() == expected.tobytes()
