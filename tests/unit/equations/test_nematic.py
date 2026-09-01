#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
from opt_einsum import contract

import phydrax as phx


def test_nematic_basis_is_orthonormal_symmetric_and_traceless():
    basis = phx.equations.NematicTensorBasis(3)
    compact = jnp.asarray((0.2, -0.1, 0.05, 0.03, -0.02))
    tensor = basis.decode(compact)

    np.testing.assert_allclose(basis.encode(tensor), compact, atol=2e-15)
    np.testing.assert_allclose(tensor, tensor.T, atol=2e-15)
    np.testing.assert_allclose(jnp.trace(tensor), 0.0, atol=2e-15)
    np.testing.assert_allclose(
        contract("aij,bij->ab", basis.matrices, basis.matrices),
        jnp.eye(5),
        atol=2e-15,
    )


def test_landau_de_gennes_molecular_field_and_electric_activity_are_finite():
    basis = phx.equations.NematicTensorBasis(3)
    closure = phx.equations.LandauDeGennesClosure(basis)
    parameters = phx.equations.LandauDeGennesParameters(
        -1.0,
        0.2,
        1.0,
        0.1,
        chiral_wave_number=0.3,
        dielectric_anisotropy=0.4,
    )
    compact = jnp.asarray((0.1, -0.03, 0.02, 0.01, -0.04))
    gradient = jnp.zeros((3, 5))
    laplacian = jnp.zeros((5,))
    fields = closure.evaluate(
        compact,
        gradient,
        laplacian,
        parameters,
        electric_field=jnp.asarray((1.0, 0.0, 0.0)),
    )
    dynamics = phx.equations.BerisEdwardsParameters(
        0.5,
        0.7,
        activity=0.2,
    )
    constitutive = phx.equations.beris_edwards_constitutive_fields(
        basis,
        compact,
        fields.molecular_field,
        jnp.zeros((3, 3)),
        fields.distortion_stress + fields.electric_stress,
        dynamics,
    )

    assert fields.successful
    assert constitutive.successful
    np.testing.assert_allclose(fields.trace_residual, 0.0, atol=2e-15)
    assert jnp.all(jnp.isfinite(constitutive.active_stress))
