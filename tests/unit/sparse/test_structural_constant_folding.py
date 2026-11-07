#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest

import phydrax as phx


_SOURCE = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
_TARGET = phx.linalg.ArraySpace((1,), dtype=jnp.float64)
_POINT = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
_BINADE_FLOOR = Fraction(2) ** -1022


def _traced_pattern(plan: phx.sparse.SparseDerivativePlan, shape: tuple[int, int]) -> Any:
    relation = plan.coloring.pattern.relation
    pattern = np.zeros(shape, dtype=np.bool_)
    pattern[np.asarray(relation.target_indices), np.asarray(relation.source_indices)] = (
        True
    )
    return pattern


def _branch_selector(predicate: Any) -> Any:
    def function(value: Any, _: Any) -> Any:
        return jnp.where(predicate(), value[0], value[1])[None]

    return function


def test_subnormal_power_predicate_keeps_both_where_branches() -> None:
    # WIN-8 I-49: k**y is a nonzero subnormal. NumPy keeps it, XLA CPU flushes it,
    # so the sign test is decided by backend floating-point behavior. The traced
    # pattern must contain the runtime Jacobian on every backend: both branches.
    base = float.fromhex("0x1.fffe1eff06008p-1")
    exponent = float.fromhex("0x1.813338038fedap+25")
    with mpmath.workdps(40):
        exact = mpmath.power(mpmath.mpf(base), mpmath.mpf(exponent))
        assert 0 < exact < mpmath.mpf(2) ** -1022
    bases = jnp.asarray([base], dtype=jnp.float64)
    exponents = jnp.asarray([exponent], dtype=jnp.float64)
    function = _branch_selector(lambda: jnp.power(bases, exponents)[0] > 0.0)

    plan = phx.sparse.compile_sparse_jacobian(
        function, _POINT, source=_SOURCE, target=_TARGET, compiler="auto"
    )

    assert np.array_equal(_traced_pattern(plan, (1, 2)), [[True, True]])
    runtime = np.asarray(jax.jacfwd(function)(_POINT, None))
    assert np.array_equal(np.asarray(plan.operator(_POINT).as_dense()), runtime)
    assert phx.sparse.verify_sparse_derivative(plan, _POINT, key=jax.random.key(0)).passed


def test_subnormal_float_data_predicate_keeps_both_where_branches() -> None:
    # A subnormal constant is read as zero by denormals-are-zero backends.
    tiny = jnp.asarray([float.fromhex("0x0.0000000000001p-1022")], dtype=jnp.float64)
    assert 0 < Fraction(float(tiny[0])) < _BINADE_FLOOR
    function = _branch_selector(lambda: tiny[0] > 0.0)

    plan = phx.sparse.compile_sparse_jacobian(
        function, _POINT, source=_SOURCE, target=_TARGET, compiler="auto"
    )

    assert np.array_equal(_traced_pattern(plan, (1, 2)), [[True, True]])


def test_normal_float_data_predicate_still_selects_one_branch() -> None:
    # Exact, normal binary64 data compares identically on every IEEE backend.
    level = jnp.asarray([0.75], dtype=jnp.float64)
    function = _branch_selector(lambda: level[0] > 0.5)

    plan = phx.sparse.compile_sparse_jacobian(
        function, _POINT, source=_SOURCE, target=_TARGET, compiler="auto"
    )

    assert np.array_equal(_traced_pattern(plan, (1, 2)), [[True, False]])


def test_rounding_sensitive_float_predicate_keeps_both_branches() -> None:
    # With c = fl(a*b), unfused a*b - c is exactly 0, so "a*b - c >= 0" holds,
    # while the exact (and multiply-add contracted) residual is negative.
    a, b = 0.1, 3.0
    c = a * b
    assert Fraction(a) * Fraction(b) - Fraction(c) < 0
    factors = jnp.asarray([a, b, c], dtype=jnp.float64)
    function = _branch_selector(lambda: factors[0] * factors[1] - factors[2] >= 0.0)

    plan = phx.sparse.compile_sparse_jacobian(
        function, _POINT, source=_SOURCE, target=_TARGET, compiler="auto"
    )

    assert np.array_equal(_traced_pattern(plan, (1, 2)), [[True, True]])


def test_float_arithmetic_far_from_threshold_still_selects_one_branch() -> None:
    # linspace(0, 1, 3)[1] is 1/2 up to a few ulps on every backend.
    function = _branch_selector(lambda: jnp.linspace(0.0, 1.0, 3)[1] > 0.4)

    plan = phx.sparse.compile_sparse_jacobian(
        function, _POINT, source=_SOURCE, target=_TARGET, compiler="auto"
    )

    assert np.array_equal(_traced_pattern(plan, (1, 2)), [[True, False]])


_SIZE = 4
_DIVIDENDS = tuple(range(-3, _SIZE - 3))
# lax.div truncates toward zero; lax.rem takes the dividend's sign.
_QUOTIENTS = tuple(int(Fraction(value, 2)) for value in _DIVIDENDS)
_REMAINDERS = tuple(
    value - 2 * quotient for value, quotient in zip(_DIVIDENDS, _QUOTIENTS, strict=True)
)
_INTEGER_INDEX_CASES: dict[str, tuple[Any, tuple[int, ...]]] = {
    "div": (
        lambda value, _: value[jax.lax.div(jnp.arange(_SIZE) - 3, 2)],
        _QUOTIENTS,
    ),
    "rem": (
        lambda value, _: value[jax.lax.rem(jnp.arange(_SIZE) - 3, 2)],
        _REMAINDERS,
    ),
}


@pytest.mark.parametrize("case", tuple(_INTEGER_INDEX_CASES))
def test_integer_index_arithmetic_folds_with_lax_semantics(case: str) -> None:
    # Negative gather indices wrap, so floor semantics would read the wrong
    # source entries.
    assert _QUOTIENTS == (-1, -1, 0, 0)
    assert _REMAINDERS == (-1, 0, -1, 0)
    function, indices = _INTEGER_INDEX_CASES[case]
    space = phx.linalg.ArraySpace((_SIZE,), dtype=jnp.float64)
    point = jnp.arange(1.0, _SIZE + 1.0, dtype=jnp.float64)
    expected = np.zeros((_SIZE, _SIZE), dtype=np.bool_)
    expected[np.arange(_SIZE), np.asarray(indices) % _SIZE] = True

    plan = phx.sparse.compile_sparse_jacobian(
        function, point, source=space, target=space, compiler="auto"
    )

    assert np.array_equal(_traced_pattern(plan, (_SIZE, _SIZE)), expected)
    runtime = np.asarray(jax.jacfwd(function)(point, None))
    assert np.array_equal(runtime != 0, expected)
