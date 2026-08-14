#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate real-axis Airy Chebyshev coefficients."""

from __future__ import annotations

import argparse

import mpmath as mp


def _positive_normalized_values(
    coordinate: mp.mpf, lower_bound: mp.mpf
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    q = (coordinate + 1) / 2
    x = lower_bound * q ** (-mp.mpf(2) / 3)
    zeta = 2 * x * mp.sqrt(x) / 3
    fourth_root = x ** mp.mpf("0.25")
    sqrt_pi = mp.sqrt(mp.pi)
    ai = mp.airyai(x)
    aip = mp.airyai(x, 1)
    bi = mp.airybi(x)
    bip = mp.airybi(x, 1)
    return (
        2 * sqrt_pi * fourth_root * mp.exp(zeta) * ai,
        -2 * sqrt_pi * mp.exp(zeta) * aip / fourth_root,
        sqrt_pi * fourth_root * mp.exp(-zeta) * bi,
        sqrt_pi * mp.exp(-zeta) * bip / fourth_root,
    )


def _negative_normalized_values(
    coordinate: mp.mpf, lower_bound: mp.mpf
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    q = (coordinate + 1) / 2
    magnitude = lower_bound * q ** (-mp.mpf(2) / 3)
    x = -magnitude
    zeta = 2 * magnitude * mp.sqrt(magnitude) / 3
    phase = zeta + mp.pi / 4
    fourth_root = magnitude ** mp.mpf("0.25")
    sqrt_pi = mp.sqrt(mp.pi)
    ai = sqrt_pi * fourth_root * mp.airyai(x)
    bi = sqrt_pi * fourth_root * mp.airybi(x)
    aip = -sqrt_pi * mp.airyai(x, 1) / fourth_root
    bip = sqrt_pi * mp.airybi(x, 1) / fourth_root
    return (
        mp.sin(phase) * ai + mp.cos(phase) * bi,
        -mp.cos(phase) * ai + mp.sin(phase) * bi,
        mp.cos(phase) * aip + mp.sin(phase) * bip,
        mp.sin(phase) * aip - mp.cos(phase) * bip,
    )


def _print_coefficients(
    *,
    names: tuple[str, ...],
    values: list[tuple[mp.mpf, ...]],
    angles: list[mp.mpf],
) -> None:
    count = len(angles)
    for component, name in enumerate(names):
        coefficients = [
            2
            * mp.fsum(
                values[node][component] * mp.cos(degree * angles[node])
                for node in range(count)
            )
            / count
            for degree in range(count)
        ]
        coefficients[0] /= 2
        print(f"_{name}_COEFFICIENTS = (")
        for coefficient in coefficients:
            print(f"    {float(coefficient):.18e},")
        print(")")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=24)
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--positive-lower-bound", type=str, default="3")
    parser.add_argument("--negative-lower-bound", type=str, default="5")
    args = parser.parse_args()
    mp.mp.dps = args.precision
    positive_lower_bound = mp.mpf(args.positive_lower_bound)
    negative_lower_bound = mp.mpf(args.negative_lower_bound)
    count = args.degree + 1
    angles = [mp.pi * (index + mp.mpf("0.5")) / count for index in range(count)]
    positive_values = [
        _positive_normalized_values(mp.cos(angle), positive_lower_bound)
        for angle in angles
    ]
    _print_coefficients(
        names=("AI_SCALED", "AIP_SCALED", "BI_SCALED", "BIP_SCALED"),
        values=positive_values,
        angles=angles,
    )
    negative_values = [
        _negative_normalized_values(mp.cos(angle), negative_lower_bound)
        for angle in angles
    ]
    _print_coefficients(
        names=(
            "AIRY_NEGATIVE_U",
            "AIRY_NEGATIVE_V",
            "AIRY_NEGATIVE_UD",
            "AIRY_NEGATIVE_VD",
        ),
        values=negative_values,
        angles=angles,
    )


if __name__ == "__main__":
    main()
