import math
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.special import _synchrotron_data
from tools.synchrotron_function_tables import render_module


type Kernel = Callable[[ArrayLike], Array]

_ROOT = Path(__file__).resolve().parents[3]
_FUNCTIONS = pytest.mark.parametrize(
    "function",
    [phx.special.synchrotron_f, phx.special.synchrotron_g],
    ids=["F", "G"],
)


def _f_reference(x: float) -> float:
    """``F`` from the closed ``1F2`` form of ``integral(K_{1/3}, 0..x)``.

    ``integral(K_{5/3}, x..inf) = 2 K_{2/3}(x) - pi/sqrt(3) + integral(K_{1/3}, 0..x)``;
    the working precision grows with ``x`` to absorb the ``exp(2x)`` cancellation,
    so the order is formed inside that precision.
    """
    with mp.workdps(40 + int(x)):
        argument = mp.mpf(x)
        third = mp.mpf(1) / 3

        def integral_i(order: mp.mpf) -> mp.mpf:
            return (
                argument ** (order + 1)
                / (2**order * (order + 1) * mp.gamma(order + 1))
                * mp.hyp1f2((order + 1) / 2, order + 1, (order + 3) / 2, argument**2 / 4)
            )

        integral_k = (mp.pi / mp.sqrt(3)) * (integral_i(-third) - integral_i(third))
        tail = 2 * mp.besselk(2 * third, argument) - mp.pi / mp.sqrt(3) + integral_k
        return float(argument * tail)


def _scaled_bessel_k(order_thirds: int, x: float) -> float:
    """``x K_{order_thirds/3}(x)`` at 40 digits."""
    with mp.workdps(40):
        return float(mp.mpf(x) * mp.besselk(mp.mpf(order_thirds) / 3, mp.mpf(x)))


def _g_reference(x: float) -> float:
    return _scaled_bessel_k(2, x)


def _reference_grid() -> np.ndarray:
    edges = np.exp(
        _synchrotron_data.PANEL_LOG_LOWER
        + _synchrotron_data.PANEL_LOG_WIDTH * np.arange(_synchrotron_data.PANEL_COUNT + 1)
    )
    edges = edges[edges <= 50.0]
    return np.unique(
        np.concatenate(
            [
                np.geomspace(1e-6, 50.0, 97),
                edges,
                np.nextafter(edges, 0.0),
                np.nextafter(edges, np.inf),
            ]
        )
    )


@pytest.mark.parametrize(
    ("function", "reference", "bound"),
    [
        (
            phx.special.synchrotron_f,
            _f_reference,
            _synchrotron_data.F_RELATIVE_ERROR_BOUND,
        ),
        (
            phx.special.synchrotron_g,
            _g_reference,
            _synchrotron_data.G_RELATIVE_ERROR_BOUND,
        ),
    ],
    ids=["F", "G"],
)
def test_recorded_relative_error_bound_holds_against_high_precision_reference(
    function: Kernel, reference: Callable[[float], float], bound: float
) -> None:
    grid = _reference_grid()
    actual = np.asarray(function(jnp.asarray(grid)))
    expected = np.asarray([reference(float(x)) for x in grid])
    relative = np.abs(actual / expected - 1.0)
    assert bound < 1e-14
    assert relative.max() <= bound, grid[np.argmax(relative)]


def _assert_relative(actual: np.ndarray, expected: np.ndarray, bound: np.ndarray) -> None:
    np.testing.assert_array_less(np.abs(actual / expected - 1.0), bound)


def test_small_argument_asymptotics() -> None:
    x = np.asarray([1e-12, 1e-9, 1e-6, 1e-4])
    f = np.asarray(phx.special.synchrotron_f(jnp.asarray(x)))
    g = np.asarray(phx.special.synchrotron_g(jnp.asarray(x)))
    # F = C x**(1/3) - pi x / sqrt(3) + O(x**(7/3)) with
    # C = 4 pi / (sqrt(3) Gamma(1/3) 2**(1/3)) = 2.1495...
    leading = 4.0 * math.pi / (math.sqrt(3.0) * math.gamma(1.0 / 3.0) * np.cbrt(2.0))
    _assert_relative(f, leading * np.cbrt(x), 0.9 * x ** (2.0 / 3.0))
    _assert_relative(f, leading * np.cbrt(x) - math.pi / math.sqrt(3.0) * x, x**2 + 1e-14)
    # G = Gamma(2/3) (x/2)**(1/3) (1 - 1.18 x**(4/3) + ...).
    g_leading = math.gamma(2.0 / 3.0) * np.cbrt(x / 2.0)
    _assert_relative(g, g_leading, 1.2 * x ** (4.0 / 3.0) + 1e-14)


def test_large_argument_asymptotics() -> None:
    x = np.asarray([100.0, 200.0, 400.0])
    envelope = np.sqrt(0.5 * math.pi * x) * np.exp(-x)
    f = np.asarray(phx.special.synchrotron_f(jnp.asarray(x)))
    g = np.asarray(phx.special.synchrotron_g(jnp.asarray(x)))
    # F ~ sqrt(pi x / 2) exp(-x) (1 + 55/(72x) - 10151/(10368 x**2) + ...),
    # G ~ sqrt(pi x / 2) exp(-x) (1 + 7/(72x) - 455/(10368 x**2) + ...).
    _assert_relative(f, envelope, 0.8 / x)
    _assert_relative(
        f, envelope * (1.0 + 55.0 / (72.0 * x) - 10151.0 / (10368.0 * x**2)), 3.0 / x**3
    )
    _assert_relative(
        g, envelope * (1.0 + 7.0 / (72.0 * x) - 455.0 / (10368.0 * x**2)), 0.1 / x**3
    )


@_FUNCTIONS
def test_argument_derivatives_match_finite_differences(function: Kernel) -> None:
    # Points cover the series, both regime switches, panels, and the asymptotic tail.
    x = jnp.asarray([1e-5, 0.1, 0.25, 0.7, 1.0, 3.0, 12.0, 40.0, 64.0, 90.0])
    # The functions vary on the length min(x, 1): x**(1/3) near zero, exp(-x) beyond.
    length = np.minimum(np.asarray(x), 1.0)
    step = jnp.asarray(1e-5 * length)
    value = np.asarray(function(x))
    gradient = jax.vmap(jax.grad(function))
    first = np.asarray(gradient(x))
    first_difference = (
        np.asarray(function(x + step)) - np.asarray(function(x - step))
    ) / (2.0 * np.asarray(step))
    # The absolute floor uses the natural derivative scale f/length at stationary points.
    scale = np.abs(value) / length
    np.testing.assert_array_less(
        np.abs(first - first_difference), 2e-8 * np.abs(first) + 1e-9 * scale
    )

    second = np.asarray(jax.vmap(jax.grad(jax.grad(function)))(x))
    second_difference = (
        np.asarray(gradient(x + step)) - np.asarray(gradient(x - step))
    ) / (2.0 * np.asarray(step))
    np.testing.assert_array_less(
        np.abs(second - second_difference),
        2e-8 * np.abs(second) + 1e-8 * scale / length,
    )


def test_argument_derivatives_match_bessel_identities() -> None:
    x = [1e-3, 1.5, 5.0, 30.0]
    derivative_f = np.asarray(
        jax.vmap(jax.grad(phx.special.synchrotron_f))(jnp.asarray(x))
    )
    derivative_g = np.asarray(
        jax.vmap(jax.grad(phx.special.synchrotron_g))(jnp.asarray(x))
    )
    # dF/dx = F/x - x K_{5/3}(x); dG/dx differentiates x K_{2/3}(x) numerically.
    expected_f = [_f_reference(value) / value - _scaled_bessel_k(5, value) for value in x]
    with mp.workdps(40):
        order = 2 * mp.mpf(1) / 3
        expected_g = [
            float(mp.diff(lambda t: t * mp.besselk(order, t), mp.mpf(value)))
            for value in x
        ]
    np.testing.assert_allclose(derivative_f, expected_f, rtol=1e-13)
    np.testing.assert_allclose(derivative_g, expected_g, rtol=1e-13)


@_FUNCTIONS
def test_domain_boundaries_and_float64_contract(function: Kernel) -> None:
    values = np.asarray(function(jnp.asarray([0.0, -0.0, jnp.inf, -1.0, jnp.nan])))
    np.testing.assert_array_equal(values[:3], np.zeros(3))
    assert np.isnan(values[3:]).all()
    assert np.isposinf(jax.grad(function)(0.0))
    assert jax.grad(function)(jnp.inf) == 0.0
    assert function(1).dtype == jnp.float64
    with pytest.raises(TypeError, match="requires float64"):
        function(jnp.asarray(1.0, dtype=jnp.float32))
    with pytest.raises(TypeError, match="complex"):
        function(1.0 + 0.0j)

    grid = jnp.geomspace(1e-3, 100.0, 12).reshape(3, 4)
    compiled = jax.jit(function)(grid)
    mapped = jax.vmap(function)(grid)
    assert compiled.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(mapped), rtol=1e-14)


def test_generator_reproduces_committed_tables() -> None:
    committed = (_ROOT / "phydrax" / "special" / "_synchrotron_data.py").read_text()
    assert render_module() == committed
