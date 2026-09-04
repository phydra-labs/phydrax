#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.equations.fem._guided_elasticity import guided_elasticity_form
from phydrax.solver._guided_elastic_modes import (
    GuidedElasticModePlan,
    GuidedElasticModeStatus,
)


def test_fixed_q_form_retains_longitudinal_strain_energy():
    form = guided_elasticity_form("u", 1.0, 2.0, 1.0)
    kernel = form.actions[0].kernel
    displacement = jnp.asarray([[[0.0, 0.0, 1.0]]], dtype=jnp.complex128)
    transverse_gradient = jnp.zeros((1, 1, 3, 2), dtype=jnp.complex128)
    residual = kernel(
        (displacement,),
        (transverse_gradient,),
        jnp.zeros((1, 1, 2)),
        jnp.ones((1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 1, 2)),
        None,
    )
    np.testing.assert_allclose(residual[0, 0, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(residual[0, 0, 2], 5.0, atol=1e-12)
    assert form.declared_properties.certifies("self_adjoint")


def test_analytic_guided_elastic_frequencies_and_mass_orthogonality():
    axial_wavenumber = 1.5
    transverse_wavenumbers = np.asarray([2.0, 3.0])
    wave_speeds = np.asarray([4.0, 5.0])
    expected = wave_speeds * np.sqrt(transverse_wavenumbers**2 + axial_wavenumber**2)
    mass = np.diag(np.asarray([2.0, 3.0]))
    stiffness = np.diag(expected**2 * np.diag(mass))
    result = GuidedElasticModePlan(
        stiffness,
        mass,
        2,
        axial_wavenumber=axial_wavenumber,
    ).solve()
    assert int(result.status) == int(GuidedElasticModeStatus.SUCCESS)
    np.testing.assert_allclose(result.angular_frequencies, expected, rtol=1e-10)
    np.testing.assert_allclose(result.modal_masses, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.orthogonality_matrix, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(
        result.modal_stiffnesses,
        result.squared_angular_frequencies,
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.all(result.residuals < 1e-10)
