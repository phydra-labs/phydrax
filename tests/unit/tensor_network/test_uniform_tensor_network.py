#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.solver._uniform_vumps import (
    solve_uniform_tangent_response,
    solve_uniform_vumps,
    UniformTangentPolicy,
    UniformTangentStatus,
    UniformVUMPSPolicy,
    UniformVUMPSProblem,
)
from phydrax.tensor_network._uniform import (
    uniform_correlation_length,
    uniform_transfer_fixed_points,
    UniformMatrixProductOperator,
    UniformMatrixProductState,
    UniformTransferPolicy,
    UniformTransferStatus,
)


def test_uniform_product_state_fixed_points_and_correlation_length():
    tensor = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)[None, :, None]
    state = UniformMatrixProductState((tensor,))
    fixed = uniform_transfer_fixed_points(state, UniformTransferPolicy(maximum_modes=2))
    assert fixed.successful
    assert fixed.injectivity_gap > 0.99
    assert fixed.dominant_residual < 1e-12
    assert jnp.allclose(fixed.left, jnp.ones((1, 1)))
    assert jnp.allclose(fixed.right, jnp.ones((1, 1)))
    assert jnp.allclose(uniform_correlation_length(fixed, 1), 0.0)


def test_uniform_vumps_reports_projected_residual_for_stationary_state():
    tensor = jnp.asarray([0.0, 1.0], dtype=jnp.complex128)[None, :, None]
    state = UniformMatrixProductState((tensor,))
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))[None, :, :, None]
    hamiltonian = UniformMatrixProductOperator((z,))
    result = solve_uniform_vumps(
        UniformVUMPSProblem(state, hamiltonian),
        UniformVUMPSPolicy(maximum_iterations=3, gradient_step=0.05),
    )
    assert result.successful
    assert result.diagnostics.active_iterations[0]
    assert result.diagnostics.galerkin_residual_history[0] < 1e-10
    assert jnp.allclose(result.energy_density, -1.0, atol=1e-10)


def test_uniform_tangent_excitation_and_response_have_fixed_capacity():
    tensor = jnp.asarray([0.0, 1.0], dtype=jnp.complex128)[None, :, None]
    state = UniformMatrixProductState((tensor,))
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))[None, :, :, None]
    hamiltonian = UniformMatrixProductOperator((z,))
    source = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)[None, :, None]
    result = solve_uniform_tangent_response(
        state,
        hamiltonian,
        source,
        jnp.linspace(0.0, 3.0, 9),
        UniformTangentPolicy(maximum_modes=3),
    )
    assert result.successful
    assert result.excitation_energies.shape == (3,)
    assert result.active_modes.shape == (3,)
    assert jnp.sum(result.active_modes) == 1
    assert jnp.all(jnp.isfinite(result.response))


def test_noninjective_uniform_state_is_explicitly_refused():
    tensor = jnp.zeros((2, 2, 2), dtype=jnp.complex128)
    tensor = tensor.at[0, 0, 0].set(1.0).at[1, 1, 1].set(1.0)
    state = UniformMatrixProductState((tensor,))
    fixed = uniform_transfer_fixed_points(
        state,
        UniformTransferPolicy(maximum_modes=4, injectivity_tolerance=1e-8),
    )
    assert fixed.status == int(UniformTransferStatus.NONINJECTIVE)
    identity = jnp.eye(2, dtype=jnp.complex128)[None, :, :, None]
    tangent = solve_uniform_tangent_response(
        state,
        UniformMatrixProductOperator((identity,)),
        jnp.ones_like(tensor),
        jnp.asarray([0.0, 1.0]),
        UniformTangentPolicy(maximum_modes=2),
    )
    assert tangent.status == int(UniformTangentStatus.NONINJECTIVE)
    assert not tangent.successful
