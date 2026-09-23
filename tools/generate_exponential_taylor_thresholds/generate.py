"""Generate the finite-series exponential Taylor backward-error thresholds.

Run from the repository root:
    uv run --project tools/generate_exponential_taylor_thresholds --locked python \
        tools/generate_exponential_taylor_thresholds/generate.py

Adapted from Seth D. Axen's expax theta generator, commit
2f2249110a8ff308588378bd36312b6210815ae9, under the BSD 3-Clause
license reproduced in this directory (Copyright 2026 Seth D. Axen):
https://github.com/sethaxen/expax/blob/2f2249110a8ff308588378bd36312b6210815ae9/tools/generate_theta_values/generate_theta_values.py

The majorant follows Higham and Al-Mohy, Computing Matrix Functions,
Acta Numerica 19 (2010), Appendix A, equation (A.3),
doi:10.1017/S0962492910000036. Precisely, for
T_m(x) = sum(x**j/j!, j=0..m), let h_m(x) = log(exp(-x)*T_m(x))
= sum(c_k*x**k, k=m+1..infinity). We enclose the positive root of
sum(abs(c_k)*x**(k-1), k=1..1224) = 2**-e, for every degree m=1..55 and
e=1..53. This retains 200 more terms than expax's primary 1024-term
majorant. At loose tolerances, the two truncations can have DIFFERENT roots
(for instance, degree 7 at tolerance 1/2); convergence is not assumed.

IMPORTANT: The enclosure certifies only this FINITE 1224-term majorant,
NOT the unbounded series or a bound on the infinite tail. Such a bound
would be required for a rigorous infinite-series backward-error certificate.
"""

import math
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from itertools import accumulate
from operator import mul
from pathlib import Path

import flint
from flint import arb, arb_poly, fmpq, fmpq_series


MAX_DEGREE = 55
TRUNCATION = 1224
ROOT_ACCURACY_BITS = 128
ARB_GUARD_BITS = 128
OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "phydrax/linalg/_generated_exponential_taylor_thresholds.py"
)


@contextmanager
def _workcap(cap: int) -> Generator[None, None, None]:
    previous = flint.ctx.cap
    flint.ctx.cap = cap
    try:
        yield
    finally:
        flint.ctx.cap = previous


def backward_error_series(
    *, max_degree: int, series_degree: int
) -> Iterator[fmpq_series]:
    """Produce rational coefficients of log(exp(-x) T_m(x)), m=1..max_degree."""
    precision = series_degree + 1
    factorials = list(accumulate(range(1, precision), mul, initial=1))
    exp_minus_x = [
        fmpq(-1 if power % 2 else 1, factorials[power]) for power in range(precision)
    ]
    scaled_approximant = exp_minus_x
    for degree in range(1, max_degree + 1):
        for power in range(degree, precision):
            remainder = power - degree
            scaled_approximant[power] += fmpq(
                -1 if remainder % 2 else 1,
                factorials[degree] * factorials[remainder],
            )
        with _workcap(precision):
            yield fmpq_series(scaled_approximant, prec=precision).log()


def majorants(backward_errors: Iterable[fmpq_series]) -> tuple[arb_poly, ...]:
    """Convert exact coefficients to enclosing Arb polynomials immediately."""
    with flint.ctx.workprec(ROOT_ACCURACY_BITS + ARB_GUARD_BITS):
        return tuple(
            arb_poly([abs(error[k]) for k in range(1, TRUNCATION + 1)])
            for error in backward_errors
        )


def _at_most(value: arb, tolerance: fmpq) -> bool:
    """Only classify an interval when its entire enclosure proves the answer."""
    target = arb(tolerance)
    if value.upper() <= target:
        return True
    if value.lower() > target:
        return False
    raise ArithmeticError("Arb enclosure is too wide to classify")


def bracket_root(polynomial: arb_poly, tolerance: fmpq) -> tuple[fmpq, fmpq]:
    lower, upper = fmpq(0), fmpq(1)
    while _at_most(polynomial(arb(upper)), tolerance):
        upper *= 2
    while upper - lower > lower * fmpq(1, 2**ROOT_ACCURACY_BITS):
        midpoint = (lower + upper) / 2
        if _at_most(polynomial(arb(midpoint)), tolerance):
            lower = midpoint
        else:
            upper = midpoint
    return lower, upper


def _rational(value: float) -> fmpq:
    numerator, denominator = value.as_integer_ratio()
    return fmpq(numerator, denominator)


def downward_float64(lower: fmpq, upper: fmpq) -> float:
    candidate = float(lower)
    if _rational(candidate) > lower:
        candidate = math.nextafter(candidate, 0.0)
    successor = math.nextafter(candidate, math.inf)
    if not _rational(candidate) <= lower < upper < _rational(successor):
        raise ArithmeticError("root enclosure does not select a unique float64")
    return candidate


def compute_row(polynomials: tuple[arb_poly, ...], tolerance: fmpq) -> tuple[float, ...]:
    with flint.ctx.workprec(ROOT_ACCURACY_BITS + ARB_GUARD_BITS):
        return tuple(
            downward_float64(*bracket_root(polynomial, tolerance))
            for polynomial in polynomials
        )


def render(rows: tuple[tuple[float, ...], ...]) -> str:
    lines = [
        '"""Generated finite-series Taylor backward-error thresholds; do not edit.',
        "",
        "Regenerate from the repository root with:",
        "uv run --project tools/generate_exponential_taylor_thresholds --locked python \\",
        "    tools/generate_exponential_taylor_thresholds/generate.py",
        "",
        "Row e-1 uses tolerance 2**-e (e=1..53); column m-1 is Taylor degree",
        "m (m=1..55). Each value is a float64 rounded DOWN from the positive",
        "root of the 1224-term finite-series absolute backward-error majorant.",
        "This does NOT certify the unbounded backward-error series tail.",
        "Source: Seth D. Axen, expax, commit 2f2249110a8ff308588378bd36312b6210815ae9,",
        "BSD-3-Clause (Copyright 2026 Seth D. Axen); license in the tool directory.",
        "Higham and Al-Mohy, Acta Numerica 19 (2010), Appendix A (A.3),",
        "doi:10.1017/S0962492910000036.",
        '"""',
        "",
        "import numpy as np",
        "",
        "TAYLOR_THRESHOLDS = np.array(",
        "    [",
    ]
    for row in rows:
        lines.append("        [")
        lines.extend(f"            {value!r}," for value in row)
        lines.append("        ],")
    lines.extend(("    ],", "    dtype=np.float64,", ")", ""))
    return "\n".join(lines)


def main() -> None:
    polynomials = majorants(
        backward_error_series(max_degree=MAX_DEGREE, series_degree=TRUNCATION)
    )
    rows = tuple(
        compute_row(polynomials, fmpq(1, 2**exponent)) for exponent in range(1, 54)
    )
    if any(
        not all(math.isfinite(value) and value > 0 for value in row)
        or any(a > b for a, b in zip(row, row[1:], strict=False))
        for row in rows
    ):
        raise ArithmeticError("thresholds are not positive, finite, monotone in degree")
    if any(
        any(a < b for a, b in zip(rows[index], rows[index + 1], strict=True))
        for index in range(len(rows) - 1)
    ):
        raise ArithmeticError("smaller tolerance produced a larger threshold")
    result = render(rows)
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != result:
        OUTPUT.write_text(result, encoding="utf-8")
    print(
        f"Certified {len(rows)} x {len(rows[0])} finite-series root enclosures: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
