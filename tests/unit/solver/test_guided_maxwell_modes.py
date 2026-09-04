#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.solver._maxwell_modes import (
    FixedFrequencyGuidedModePlan,
    guided_mode_beta_derivative,
    GuidedModeClassification,
    GuidedModeStatus,
    solve_fixed_frequency_guided_modes,
)


def _plan(gammas, *, target, divergence=None):
    gamma = np.asarray(gammas, dtype=np.complex128)
    count = gamma.size
    identity = np.eye(count, dtype=np.complex128)
    divergence_map = (
        np.zeros((1, count), dtype=np.complex128)
        if divergence is None
        else np.asarray(divergence, dtype=np.complex128)
    )
    return FixedFrequencyGuidedModePlan(
        -np.diag(gamma**2),
        np.zeros((count, count), dtype=np.complex128),
        identity,
        count,
        angular_frequency=8.0,
        right_electric_trace_coefficients=(identity,),
        right_magnetic_trace_coefficients=(identity,),
        left_electric_trace_coefficients=(identity,),
        left_magnetic_trace_coefficients=(identity,),
        divergence_coefficients=(divergence_map,),
        power_pairing=identity,
        target_propagation_constant=target,
        divergence_tolerance=1e-9,
    )


def test_rectangular_waveguide_cutoffs_and_polynomial_mode_evidence():
    vacuum_wavenumber = 4.0
    transverse_wavenumbers = np.sqrt(np.asarray([12.0, 7.0]))
    expected = np.sqrt(vacuum_wavenumber**2 - transverse_wavenumbers**2)
    result = _plan(expected, target=2.1).solve()
    np.testing.assert_allclose(
        np.sort(np.real(result.propagation_constants)), expected, rtol=1e-10, atol=1e-10
    )
    assert int(result.status) == int(GuidedModeStatus.SUCCESS)
    assert np.all(result.polynomial_residuals < 1e-9)
    assert np.all(result.divergence_residuals < 1e-12)
    np.testing.assert_allclose(result.complex_powers, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.biorthogonality_matrix, np.eye(2), atol=1e-9)
    assert np.all(result.derivative_evidence.nearest_absolute_gaps > 0.9)
    assert np.all(result.classifications == int(GuidedModeClassification.PROPAGATING))


def test_cutoff_pml_classification_divergence_and_mode_launch():
    cutoff = _plan([0.0], target=0.0).solve()
    assert int(cutoff.classifications[0]) == int(GuidedModeClassification.CUTOFF)
    assert not bool(cutoff.derivative_evidence.derivative_valid_mask[0])
    assert int(cutoff.derivative_evidence.cluster_multiplicities[0]) == 2

    pml = _plan([2.0 + 0.15j], target=2.0 + 0.15j).solve()
    assert int(pml.classifications[0]) == int(GuidedModeClassification.LEAKY_OR_PML)
    launched = pml.launch(0, amplitude=2j)
    np.testing.assert_allclose(
        launched.electric_trace, 2j * pml.right_electric_traces[:, 0]
    )
    np.testing.assert_allclose(
        launched.magnetic_trace, 2j * pml.right_magnetic_traces[:, 0]
    )
    assert launched.mode_id == pml.mode_ids[0]

    divergent = _plan([2.0], target=2.0, divergence=[[1.0]]).solve()
    assert int(divergent.status) == int(GuidedModeStatus.DIVERGENCE_TOLERANCE_NOT_MET)


def test_isolated_beta_derivative_uses_left_right_polynomial_pairing():
    prepared = _plan([2.0], target=2.0).prepare()
    result = solve_fixed_frequency_guided_modes(prepared)
    derivative = guided_mode_beta_derivative(
        prepared,
        result,
        (
            jnp.asarray([[-4.0]]),
            jnp.asarray([[0.0]]),
            jnp.asarray([[0.0]]),
        ),
    )
    assert bool(derivative.valid_mask[0])
    np.testing.assert_allclose(derivative.values[0], 1.0, rtol=1e-9, atol=1e-9)
