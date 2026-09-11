import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from phydrax.special._spherical_bessel import (
    _spherical_hankel1_sequence,
    _spherical_i_sequence,
    _spherical_j_sequence,
    _spherical_k_sequence,
    _spherical_sequence_derivative,
    _spherical_y_sequence,
)


def _scipy_sequence(function, maximum_order, argument, **kwargs):
    return np.stack(
        [function(order, argument, **kwargs) for order in range(maximum_order + 1)]
    )


def test_real_sequences_match_scipy_across_series_and_recurrence_regions():
    maximum_order = 12
    argument = np.asarray([0.2, 1.5, 7.0, 21.0])
    cases = (
        (_spherical_j_sequence, scipy.special.spherical_jn),
        (_spherical_y_sequence, scipy.special.spherical_yn),
        (_spherical_i_sequence, scipy.special.spherical_in),
        (_spherical_k_sequence, scipy.special.spherical_kn),
    )
    for actual_function, reference_function in cases:
        actual = np.asarray(actual_function(maximum_order, jnp.asarray(argument)))
        expected = _scipy_sequence(reference_function, maximum_order, argument)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-14)


def test_complex_and_scaled_translation_regimes_match_reference_values():
    maximum_order = 8
    argument = np.asarray([0.4 + 0.3j, 3.0 + 2.0j, 12.0 + 5.0j])
    j_values = np.asarray(_spherical_j_sequence(maximum_order, jnp.asarray(argument)))
    y_values = np.asarray(_spherical_y_sequence(maximum_order, jnp.asarray(argument)))
    expected_j = _scipy_sequence(scipy.special.spherical_jn, maximum_order, argument)
    expected_y = _scipy_sequence(scipy.special.spherical_yn, maximum_order, argument)
    np.testing.assert_allclose(j_values, expected_j, rtol=5.0e-11, atol=2.0e-13)
    np.testing.assert_allclose(y_values, expected_y, rtol=5.0e-11, atol=2.0e-13)

    outgoing = np.asarray(
        _spherical_hankel1_sequence(maximum_order, jnp.asarray(argument), scaled=True)
    )
    expected_outgoing = np.exp(-1j * argument)[None, :] * (expected_j + 1j * expected_y)
    np.testing.assert_allclose(outgoing, expected_outgoing, rtol=8.0e-11, atol=3.0e-13)

    positive = np.asarray([0.5, 20.0, 100.0])
    scaled_i = np.asarray(
        _spherical_i_sequence(maximum_order, jnp.asarray(positive), scaled=True)
    )
    scaled_k = np.asarray(
        _spherical_k_sequence(maximum_order, jnp.asarray(positive), scaled=True)
    )
    expected_i = np.exp(-positive)[None, :] * _scipy_sequence(
        scipy.special.spherical_in, maximum_order, positive
    )
    expected_k = np.exp(positive)[None, :] * _scipy_sequence(
        scipy.special.spherical_kn, maximum_order, positive
    )
    np.testing.assert_allclose(scaled_i, expected_i, rtol=3.0e-11, atol=2.0e-14)
    np.testing.assert_allclose(scaled_k, expected_k, rtol=3.0e-11, atol=2.0e-14)

    far_field = np.asarray(
        _spherical_k_sequence(20, jnp.asarray([400.0, 800.0]), scaled=True)
    )
    assert np.isfinite(far_field).all()


def test_recurrences_and_jy_wronskian_hold_order_by_order():
    argument = jnp.asarray([0.7, 2.5, 11.0])
    order = np.arange(1, 15)[:, None]
    j_values = _spherical_j_sequence(15, argument)
    y_values = _spherical_y_sequence(15, argument)
    i_values = _spherical_i_sequence(15, argument)
    k_values = _spherical_k_sequence(15, argument)

    coefficient = (2.0 * order + 1.0) / np.asarray(argument)[None, :]
    np.testing.assert_allclose(
        np.asarray(j_values[:-2] + j_values[2:]),
        coefficient * np.asarray(j_values[1:-1]),
        rtol=2.0e-11,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(i_values[:-2] - i_values[2:]),
        coefficient * np.asarray(i_values[1:-1]),
        rtol=2.0e-11,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(k_values[2:] - k_values[:-2]),
        coefficient * np.asarray(k_values[1:-1]),
        rtol=2.0e-11,
        atol=2.0e-14,
    )

    j_derivative = _spherical_sequence_derivative(j_values, argument, kind="j")
    y_derivative = _spherical_sequence_derivative(y_values, argument, kind="y")
    wronskian = j_values * y_derivative - j_derivative * y_values
    np.testing.assert_allclose(
        np.asarray(wronskian),
        np.broadcast_to(1.0 / np.asarray(argument) ** 2, wronskian.shape),
        rtol=5.0e-11,
        atol=3.0e-13,
    )


def test_zero_limits_and_tiny_regular_modes_are_not_thresholded_away():
    exact = jnp.asarray([0.0, 1.0e-8])
    j_values = np.asarray(_spherical_j_sequence(6, exact))
    i_values = np.asarray(_spherical_i_sequence(6, exact))
    np.testing.assert_array_equal(j_values[:, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(i_values[:, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.all(j_values[1:, 1] != 0.0)
    assert np.all(i_values[1:, 1] != 0.0)
    np.testing.assert_allclose(
        j_values[:, 1],
        _scipy_sequence(scipy.special.spherical_jn, 6, 1.0e-8),
        rtol=3.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        i_values[:, 1],
        _scipy_sequence(scipy.special.spherical_in, 6, 1.0e-8),
        rtol=3.0e-13,
        atol=0.0,
    )
    assert np.isneginf(np.asarray(_spherical_y_sequence(4, 0.0))).all()
    assert np.isposinf(np.asarray(_spherical_k_sequence(4, 0.0))).all()

    j_derivative = np.asarray(
        _spherical_sequence_derivative(_spherical_j_sequence(4, 0.0), 0.0, kind="j")
    )
    i_derivative = np.asarray(
        _spherical_sequence_derivative(_spherical_i_sequence(4, 0.0), 0.0, kind="i")
    )
    np.testing.assert_array_equal(j_derivative, [0.0, 1.0 / 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(i_derivative, [0.0, 1.0 / 3.0, 0.0, 0.0, 0.0])


def test_neighbor_derivatives_match_scipy_and_differentiate_scaled_primals():
    argument = jnp.asarray([0.15, 1.8, 9.0])
    maximum_order = 9
    cases = (
        ("j", _spherical_j_sequence, scipy.special.spherical_jn),
        ("y", _spherical_y_sequence, scipy.special.spherical_yn),
        ("i", _spherical_i_sequence, scipy.special.spherical_in),
        ("k", _spherical_k_sequence, scipy.special.spherical_kn),
    )
    for kind, function, reference in cases:
        values = function(maximum_order, argument)
        actual = _spherical_sequence_derivative(values, argument, kind=kind)
        expected = _scipy_sequence(
            reference, maximum_order, np.asarray(argument), derivative=True
        )
        np.testing.assert_allclose(actual, expected, rtol=4.0e-10, atol=3.0e-13)

    complex_argument = jnp.asarray([1.2 + 0.4j, 4.0 + 1.5j])
    for kind, function in (
        ("h1", _spherical_hankel1_sequence),
        ("i", _spherical_i_sequence),
        ("k", _spherical_k_sequence),
    ):
        values = function(7, complex_argument, scaled=True)
        analytic = _spherical_sequence_derivative(
            values, complex_argument, kind=kind, scaled=True
        )
        _, tangent = jax.jvp(
            lambda z: function(7, z, scaled=True),
            (complex_argument,),
            (jnp.ones_like(complex_argument),),
        )
        np.testing.assert_allclose(tangent, analytic, rtol=2.0e-10, atol=3.0e-12)


def test_structural_order_validation_rejects_dynamic_or_negative_orders():
    with pytest.raises(TypeError):
        _spherical_j_sequence(jnp.asarray(3), 1.0)
    with pytest.raises(TypeError):
        _spherical_i_sequence(True, 1.0)
    with pytest.raises(ValueError):
        _spherical_k_sequence(-1, 1.0)
