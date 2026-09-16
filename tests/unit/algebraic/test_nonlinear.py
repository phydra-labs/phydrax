#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.algebraic import lower_complex_polynomial_root, SparsePolynomialSystem
from phydrax.linalg import DenseLinearOperator
from phydrax.nonlinear import NonlinearSystemProblem


def test_complex_root_lowering_uses_cartesian_residual_and_real_block_jacobian():
    system = SparsePolynomialSystem.from_coo(
        ("z",),
        ("f",),
        (0, 0, 0),
        ((2,), (1,), (0,)),
        jnp.asarray((1.0, 2.0 - 1.0j, 3.0j), dtype=jnp.complex64),
    )
    lowering = lower_complex_polynomial_root(system, complex_dtype=jnp.complex64)
    root = jnp.asarray((1.0 + 2.0j,), dtype=jnp.complex64)
    coordinates = lowering.to_real_coordinates(root)

    assert isinstance(lowering.problem, NonlinearSystemProblem)
    assert coordinates.shape == (2, 1)
    np.testing.assert_allclose(lowering.from_real_coordinates(coordinates), root)
    expected_residual = system.evaluate(root)
    np.testing.assert_allclose(
        lowering.problem.residual(coordinates),
        jnp.stack((jnp.real(expected_residual), jnp.imag(expected_residual)), axis=0),
    )

    operator = lowering.problem.linear_setup(coordinates)
    assert isinstance(operator, DenseLinearOperator)
    expected_block = jnp.asarray(
        ((4.0, -3.0), (3.0, 4.0)),
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(operator.matrix, expected_block, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        lowering.real_block_jacobian(coordinates),
        expected_block,
        rtol=1e-6,
        atol=1e-6,
    )
