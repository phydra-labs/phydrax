#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control import frequency_response
from phydrax.uq import (
    linear_gaussian_spectral_densities,
    linear_gaussian_transfer_function,
    output_input_cross_spectral_density,
    output_spectral_density,
    state_input_cross_spectral_density,
    state_output_cross_spectral_density,
    state_spectral_density,
)


def test_stochastic_transfer_reuses_control_frequency_response():
    a = jnp.array([[-2.0]])
    b = jnp.array([[3.0]])
    c = jnp.array([[4.0]])
    d = jnp.array([[0.5]])
    frequencies = jnp.array([0.0, 1.0])
    expected = frequency_response(a, b, c, d, frequencies)
    transfer = linear_gaussian_transfer_function(a, b, c, d, frequencies)

    np.testing.assert_allclose(transfer.input_to_state, expected.state_response)
    np.testing.assert_allclose(transfer.input_to_output, expected.response)
    np.testing.assert_allclose(transfer.process_to_state, expected.resolvent)
    np.testing.assert_allclose(
        transfer.process_to_output, c[None, ...] @ expected.resolvent
    )
    assert transfer.diagnostics.method_id == expected.method_id


def test_siso_state_output_and_cross_spectra_match_analytic_values():
    a = jnp.array([[-2.0]])
    b = jnp.array([[1.0]])
    c = jnp.array([[3.0]])
    d = jnp.array([[0.25]])
    frequencies = jnp.array([0.0, 2.0])
    input_psd = jnp.array([[2.0]])
    process_psd = jnp.array([[0.5]])
    measurement_psd = jnp.array([[0.1]])
    spectra = linear_gaussian_spectral_densities(
        a,
        b,
        c,
        d,
        frequencies,
        input_spectrum=input_psd,
        process_spectrum=process_psd,
        measurement_spectrum=measurement_psd,
    )

    resolvent = 1.0 / (1j * frequencies + 2.0)
    h_xu = resolvent
    h_yu = 3.0 * resolvent + 0.25
    h_yw = 3.0 * resolvent
    expected_x = jnp.abs(h_xu) ** 2 * 2.0 + jnp.abs(resolvent) ** 2 * 0.5
    expected_y = jnp.abs(h_yu) ** 2 * 2.0 + jnp.abs(h_yw) ** 2 * 0.5 + 0.1
    expected_xy = h_xu * 2.0 * jnp.conj(h_yu) + resolvent * 0.5 * jnp.conj(h_yw)
    np.testing.assert_allclose(spectra.state_spectrum[:, 0, 0], expected_x)
    np.testing.assert_allclose(spectra.output_spectrum[:, 0, 0], expected_y)
    np.testing.assert_allclose(spectra.state_output_cross_spectrum[:, 0, 0], expected_xy)
    np.testing.assert_allclose(spectra.state_input_cross_spectrum[:, 0, 0], h_xu * 2.0)
    np.testing.assert_allclose(spectra.output_input_cross_spectrum[:, 0, 0], h_yu * 2.0)
    assert bool(jnp.all(spectra.valid))


def test_mimo_spectra_are_hermitian_positive_semidefinite():
    a = jnp.array([[-2.0, 0.5], [-0.25, -1.0]])
    b = jnp.array([[1.0, 0.2], [0.5, -0.4]])
    c = jnp.array([[1.0, 0.5], [-0.25, 1.0]])
    d = jnp.array([[0.1, 0.0], [0.0, -0.2]])
    su = jnp.array([[1.0, 0.25j], [-0.25j, 0.75]])
    sw = jnp.array([[0.4, 0.1], [0.1, 0.3]])
    sv = jnp.array([[0.2, -0.05j], [0.05j, 0.1]])
    spectra = linear_gaussian_spectral_densities(
        a,
        b,
        c,
        d,
        jnp.array([0.0, 0.75, 2.0]),
        input_spectrum=su,
        process_spectrum=sw,
        measurement_spectrum=sv,
    )

    for value in (spectra.state_spectrum, spectra.output_spectrum):
        np.testing.assert_allclose(value, jnp.swapaxes(jnp.conj(value), -1, -2))
        assert bool(jnp.all(jnp.linalg.eigvalsh(value) >= -1.0e-10))
    np.testing.assert_allclose(
        spectra.state_spectrum,
        state_spectral_density(
            a,
            b,
            c,
            d,
            jnp.array([0.0, 0.75, 2.0]),
            input_spectrum=su,
            process_spectrum=sw,
            measurement_spectrum=sv,
        ),
    )
    np.testing.assert_allclose(
        spectra.output_spectrum,
        output_spectral_density(
            a,
            b,
            c,
            d,
            jnp.array([0.0, 0.75, 2.0]),
            input_spectrum=su,
            process_spectrum=sw,
            measurement_spectrum=sv,
        ),
    )
    np.testing.assert_allclose(
        spectra.state_output_cross_spectrum,
        state_output_cross_spectral_density(
            a,
            b,
            c,
            d,
            jnp.array([0.0, 0.75, 2.0]),
            input_spectrum=su,
            process_spectrum=sw,
            measurement_spectrum=sv,
        ),
    )
    np.testing.assert_allclose(
        spectra.state_input_cross_spectrum,
        state_input_cross_spectral_density(
            a,
            b,
            c,
            d,
            jnp.array([0.0, 0.75, 2.0]),
            input_spectrum=su,
            process_spectrum=sw,
            measurement_spectrum=sv,
        ),
    )
    np.testing.assert_allclose(
        spectra.output_input_cross_spectrum,
        output_input_cross_spectral_density(
            a,
            b,
            c,
            d,
            jnp.array([0.0, 0.75, 2.0]),
            input_spectrum=su,
            process_spectrum=sw,
            measurement_spectrum=sv,
        ),
    )


def test_stationary_spectra_reject_unstable_or_ill_shaped_models():
    one = jnp.ones((1, 1))
    zero = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match="stable"):
        linear_gaussian_spectral_densities(
            jnp.array([[0.1]]),
            one,
            one,
            zero,
            jnp.asarray(1.0),
            input_spectrum=one,
            process_spectrum=one,
            measurement_spectrum=one,
        )
    with pytest.raises(ValueError, match="input_spectrum"):
        linear_gaussian_spectral_densities(
            jnp.array([[-1.0]]),
            one,
            one,
            zero,
            jnp.asarray(1.0),
            input_spectrum=jnp.eye(2),
            process_spectrum=one,
            measurement_spectrum=one,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        linear_gaussian_spectral_densities(
            jnp.array([[-1.0]]),
            one,
            one,
            zero,
            jnp.asarray(1.0),
            input_spectrum=-one,
            process_spectrum=one,
            measurement_spectrum=one,
        )
