#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate the synchrotron-function tables in ``phydrax/special``.

The generated module holds three real functions of ``x > 0``::

    F(x) = x * integral(K_{5/3}(t), t=x..inf)
    G(x) = x * K_{2/3}(x)
    H(x) = x * K_{1/3}(x)

``H`` closes the derivative system used by the package JVPs. Each function is
represented by a convergent small-``x`` power series below ``SERIES_UPPER``,
uniform Chebyshev panels in ``log(x)`` on ``[SERIES_UPPER, ASYMPTOTIC_LOWER)``,
and a Hankel-type asymptotic series above. Panel values are evaluated with
mpmath at ``WORKING_DIGITS`` decimal digits: ``F`` from adaptive
Gauss-Legendre quadrature of the exponentially scaled integral representation

    exp(x) * integral(K_nu(t), t=x..inf)
        = integral(exp(-v**2) * cosh(nu*s) / cosh(s) * 2 / sqrt(v**2 + 2*x), v=0..inf),
    cosh(s) = 1 + v**2 / x,

and ``G``/``H`` from mpmath's Bessel ``K``. Every truncation is measured in
high precision; the recorded bound adds a fixed float64 rounding allowance.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import mpmath as mp


WORKING_DIGITS = 30
SERIES_UPPER = "0.25"
ASYMPTOTIC_LOWER = "64"
PANEL_COUNT = 12
PANEL_COEFFICIENTS = 14
REFERENCE_NODES = 20
SERIES_TERM_CAPACITY = 40
ASYMPTOTIC_TERM_CAPACITY = 60
TRUNCATION_TARGET = "1e-19"
REFERENCE_RESOLUTION = "1e-24"
# Float64 evaluation allowance: log/cbrt/exp argument scaling, Clenshaw or
# Horner accumulation, and the at most fourfold series cancellation at
# SERIES_UPPER stay within 16 units of float64 roundoff.
ROUNDING_ALLOWANCE = 16 * 2.0**-52
OUTPUT = Path("phydrax/special/_synchrotron_data.py")

type Polynomial = tuple[mp.mpf, ...]
type Scaled = Callable[[mp.mpf], mp.mpf]


def _third() -> mp.mpf:
    return mp.mpf(1) / 3


def _bessel_series(order: mp.mpf, terms: int) -> Polynomial:
    """Coefficients ``2**-(2k+order) / (k! Gamma(k+order+1))`` of ``I_order``.

    Coefficient ``k`` multiplies ``x**(2k + order)``.
    """
    return tuple(
        mp.mpf(2) ** (-(2 * k + order)) / (mp.factorial(k) * mp.gamma(k + order + 1))
        for k in range(terms)
    )


def _series_families(name: str, terms: int) -> tuple[Polynomial, ...]:
    """Polynomials ``P_j(x**2)`` with ``f(x) = x**(p/3) sum_j x**(2j/3) P_j(x**2)``.

    ``K_nu = pi / (2 sin(nu pi)) (I_{-nu} - I_nu)`` with ``pi/(2 sin(pi/3)) =
    pi/(2 sin(2 pi/3)) = pi/sqrt(3)``. ``F`` uses
    ``integral(K_{5/3}, x..inf) = 2 K_{2/3}(x) - pi/sqrt(3) + integral(K_{1/3}, 0..x)``;
    its ``x**(5/3 + 2k)`` coefficient combines ``-2 x I_{2/3}`` with
    ``x integral(I_{-1/3})`` in closed form, which vanishes exactly at ``k=0``.
    """
    third = _third()
    scale = mp.pi / mp.sqrt(3)
    zero = mp.mpf(0)
    match name:
        case "G":
            return (
                tuple(scale * value for value in _bessel_series(-2 * third, terms)),
                (zero,),
                tuple(-scale * value for value in _bessel_series(2 * third, terms)),
            )
        case "H":
            return (
                tuple(scale * value for value in _bessel_series(-third, terms)),
                tuple(-scale * value for value in _bessel_series(third, terms)),
            )
        case "F":
            lower = _bessel_series(-third, terms)
            upper = _bessel_series(third, terms)
            return (
                tuple(2 * scale * value for value in _bessel_series(-2 * third, terms)),
                (-scale,),
                tuple(
                    -scale * value * k / ((2 * k + 2 * third) * (k + 2 * third))
                    for k, value in enumerate(lower)
                ),
                tuple(
                    -scale * value / (2 * k + 4 * third) for k, value in enumerate(upper)
                ),
            )
        case _:
            raise ValueError(f"unknown synchrotron function {name!r}")


def _asymptotic_coefficients(name: str, terms: int) -> Polynomial:
    """Coefficients ``c_k`` with ``f(x) ~ sqrt(pi x / 2) exp(-x) sum c_k x**-k``.

    ``K_nu`` uses ``a_k = prod_{j<=k} (4 nu**2 - (2j-1)**2) / (k! 8**k)``. For
    ``F = x J`` with ``J' = -K_{5/3}``, matching powers gives
    ``g_k = a_k(5/3) - (k - 1/2) g_{k-1}``.
    """
    third = _third()
    match name:
        case "G":
            order = 2 * third
        case "H":
            order = third
        case "F":
            order = 5 * third
        case _:
            raise ValueError(f"unknown synchrotron function {name!r}")
    bessel = [mp.mpf(1)]
    for k in range(1, terms):
        bessel.append(bessel[-1] * (4 * order * order - (2 * k - 1) ** 2) / (8 * k))
    if name != "F":
        return tuple(bessel)
    integrated = [mp.mpf(1)]
    for k in range(1, terms):
        integrated.append(bessel[k] - (k - mp.mpf(1) / 2) * integrated[-1])
    return tuple(integrated)


def _scaled_tail_integral(order: mp.mpf, x: mp.mpf) -> mp.mpf:
    """``exp(x) * integral(K_order(t), t=x..inf)`` by high-precision quadrature."""
    two_x = 2 * x

    def integrand(v: mp.mpf) -> mp.mpf:
        v2 = v * v
        hyperbolic = 1 + v2 / x
        return (
            mp.exp(-v2)
            * mp.cosh(order * mp.acosh(hyperbolic))
            * 2
            / (mp.sqrt(v2 + two_x) * hyperbolic)
        )

    cutoff = mp.sqrt(mp.mpf("2.31") * WORKING_DIGITS + 5)
    breakpoints = [mp.mpf(0)]
    breakpoint = mp.sqrt(x) / 4
    while breakpoint < cutoff:
        breakpoints.append(breakpoint)
        breakpoint *= 4
    breakpoints.append(cutoff)
    return mp.quad(integrand, breakpoints, method="gauss-legendre")


def _panel_values(name: str, power_thirds: int) -> Scaled:
    """Return ``phi(x) = f(x) exp(x) x**(-p/3)`` evaluated in high precision."""
    third = _third()
    match name:
        case "F":
            return lambda x: (
                _scaled_tail_integral(5 * third, x) * x ** (1 - power_thirds * third)
            )
        case "G":
            return lambda x: (
                mp.besselk(2 * third, x) * mp.exp(x) * x ** (1 - power_thirds * third)
            )
        case "H":
            return lambda x: (
                mp.besselk(third, x) * mp.exp(x) * x ** (1 - power_thirds * third)
            )
        case _:
            raise ValueError(f"unknown synchrotron function {name!r}")


def _series_value(
    families: tuple[Polynomial, ...], power_thirds: int, x: mp.mpf
) -> mp.mpf:
    root = mp.cbrt(x)
    square = x * x
    return root**power_thirds * mp.fsum(
        root ** (2 * family) * mp.polyval(list(reversed(polynomial)), square)
        for family, polynomial in enumerate(families)
    )


def _series_table(
    name: str, power_thirds: int, upper: mp.mpf, target: mp.mpf
) -> tuple[tuple[Polynomial, ...], mp.mpf]:
    """Truncate the series where the omitted terms stay below ``target`` at ``upper``.

    Every family is a positive-coefficient power series in ``x**2`` times a
    positive power, so omitted terms grow with ``x`` while ``f(x)/x**(p/3)``
    decreases; the relative truncation error is largest at ``upper``.
    """
    full = _series_families(name, SERIES_TERM_CAPACITY)
    value = _series_value(full, power_thirds, upper)
    root = mp.cbrt(upper)
    square = upper * upper
    for terms in range(1, SERIES_TERM_CAPACITY):
        omitted = mp.fsum(
            abs(root ** (power_thirds + 2 * family) * coefficient * square**index)
            for family, polynomial in enumerate(full)
            for index, coefficient in enumerate(polynomial)
            if index >= terms
        )
        relative = omitted / value
        if relative < target:
            return tuple(polynomial[:terms] for polynomial in full), relative
    raise ValueError(f"{name} series does not reach the truncation target")


def _panel_table(
    name: str,
    power_thirds: int,
    lower: mp.mpf,
    width: mp.mpf,
    target: mp.mpf,
    resolution: mp.mpf,
) -> tuple[tuple[Polynomial, ...], mp.mpf]:
    """Truncated Chebyshev series of ``phi`` on uniform ``log(x)`` panels.

    A ``REFERENCE_NODES`` first-kind interpolant resolves ``phi`` to
    ``resolution``; storing its leading ``PANEL_COEFFICIENTS`` terms gives the
    relative truncation bound ``sum(|c_j|, omitted) / min(phi)``.
    """
    values = _panel_values(name, power_thirds)
    count = REFERENCE_NODES
    angles = [mp.pi * (node + mp.mpf(1) / 2) / count for node in range(count)]
    panels = []
    worst = mp.mpf(0)
    for panel in range(PANEL_COUNT):
        center = lower + (panel + mp.mpf(1) / 2) * width
        samples = [values(mp.exp(center + width / 2 * mp.cos(angle))) for angle in angles]
        coefficients = [
            2
            * mp.fsum(
                sample * mp.cos(degree * angle)
                for sample, angle in zip(samples, angles, strict=True)
            )
            / count
            for degree in range(count)
        ]
        coefficients[0] /= 2
        floor = min(abs(sample) for sample in samples)
        if max(abs(value) for value in coefficients[-2:]) > resolution * floor:
            raise ValueError(f"{name} panel {panel} is not resolved by the reference")
        relative = (
            mp.fsum(abs(value) for value in coefficients[PANEL_COEFFICIENTS:]) / floor
        )
        if relative > target:
            raise ValueError(f"{name} panel {panel} exceeds the truncation target")
        worst = max(worst, relative)
        panels.append(tuple(coefficients[:PANEL_COEFFICIENTS]))
    return tuple(panels), worst


def _asymptotic_table(
    name: str, lower: mp.mpf, target: mp.mpf
) -> tuple[Polynomial, mp.mpf]:
    """Truncate before the first term below ``target`` at ``lower``.

    For ``G`` and ``H`` the first omitted term bounds the ``K_nu`` remainder.
    ``F`` has factorially growing coefficients, so its remainder is measured
    against quadrature at ``lower``, where it is largest.
    """
    full = _asymptotic_coefficients(name, ASYMPTOTIC_TERM_CAPACITY)
    for terms in range(1, ASYMPTOTIC_TERM_CAPACITY):
        omitted = abs(full[terms]) / lower**terms
        if omitted < target:
            coefficients = full[:terms]
            series = mp.polyval(list(reversed(coefficients)), 1 / lower)
            if name == "F":
                reference = _scaled_tail_integral(5 * _third(), lower) * mp.sqrt(
                    2 * lower / mp.pi
                )
                omitted = max(omitted, abs(series / reference - 1))
            return coefficients, omitted
    raise ValueError(f"{name} asymptotic series does not reach the truncation target")


def _ceiling(value: mp.mpf) -> float:
    """Round up to two significant decimal digits."""
    exponent = int(mp.floor(mp.log10(value))) - 1
    return float(mp.ceil(value / mp.mpf(10) ** exponent) * mp.mpf(10) ** exponent)


def _number(value: mp.mpf) -> str:
    """Shortest round-trip float64 literal in the formatter's canonical spelling."""
    return repr(float(value)).replace("e+", "e")


def _literal(values: Polynomial, indent: int) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{_number(value)}," for value in values)


def _nested_literal(rows: tuple[Polynomial, ...], indent: int) -> str:
    # One-element rows collapse onto one line, as the formatter spells them.
    prefix = " " * indent
    return "\n".join(
        f"{prefix}({_number(row[0])},),"
        if len(row) == 1
        else f"{prefix}(\n{_literal(row, indent + 4)}\n{prefix}),"
        for row in rows
    )


def _function_block(name: str, power_thirds: int) -> tuple[str, mp.mpf]:
    upper = mp.mpf(SERIES_UPPER)
    lower = mp.mpf(ASYMPTOTIC_LOWER)
    target = mp.mpf(TRUNCATION_TARGET)
    series, series_error = _series_table(name, power_thirds, upper, target)
    width = mp.log(lower / upper) / PANEL_COUNT
    panels, panel_error = _panel_table(
        name,
        power_thirds,
        mp.log(upper),
        width,
        target,
        mp.mpf(REFERENCE_RESOLUTION),
    )
    asymptotic, asymptotic_error = _asymptotic_table(name, lower, target)
    truncation = max(series_error, panel_error, asymptotic_error)
    bound = _ceiling(truncation + ROUNDING_ALLOWANCE)
    block = f"""
{name}_POWER_THIRDS = {power_thirds}
{name}_RELATIVE_ERROR_BOUND = {bound!r}
{name}_SERIES = (
{_nested_literal(series, 4)}
)
{name}_PANELS = (
{_nested_literal(panels, 4)}
)
{name}_ASYMPTOTIC = (
{_literal(asymptotic, 4)}
)
"""
    return block, truncation


def render_module() -> str:
    """Return the deterministic generated module text."""
    with mp.workdps(WORKING_DIGITS):
        upper = mp.mpf(SERIES_UPPER)
        lower = mp.mpf(ASYMPTOTIC_LOWER)
        width = mp.log(lower / upper) / PANEL_COUNT
        blocks = []
        truncations = []
        for name, power_thirds in (("F", 1), ("G", 1), ("H", 2)):
            block, truncation = _function_block(name, power_thirds)
            blocks.append(block)
            truncations.append(f"#   {name}: {mp.nstr(truncation, 3)}")
        body = "".join(blocks)
        return f'''#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Generated by tools/synchrotron_function_tables.py at {WORKING_DIGITS} decimal digits.
# Do not edit by hand. Measured high-precision truncation errors:
{chr(10).join(truncations)}
# Each bound adds a {ROUNDING_ALLOWANCE!r} float64 rounding allowance.
#

"""Generated series, log-x Chebyshev panel, and asymptotic tables."""

from __future__ import annotations


SERIES_UPPER = {float(upper)!r}
ASYMPTOTIC_LOWER = {float(lower)!r}
PANEL_LOG_LOWER = {float(mp.log(upper))!r}
PANEL_LOG_WIDTH = {float(width)!r}
PANEL_COUNT = {PANEL_COUNT}
{body}'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    arguments.output.write_text(render_module())


if __name__ == "__main__":
    main()
