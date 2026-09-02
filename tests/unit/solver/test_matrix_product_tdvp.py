import jax.numpy as jnp
import jax.scipy as jsp

import phydrax as phx


tn = phx.tensor_network


def test_one_site_real_time_tdvp_matches_exact_evolution():
    pauli_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    state = tn.product_mps(
        jnp.asarray([[1.0, 1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)
    )
    hamiltonian = tn.product_mpo(pauli_z[None, ...])
    problem = phx.solver.FiniteTDVPProblem(state, hamiltonian)
    policy = phx.solver.FiniteTDVPPolicy(
        "real-time",
        step_size=0.05,
        steps=1,
        norm_tolerance=1e-7,
        integrator=phx.linalg.MatrixFunctionPolicy(
            "lanczos", max_dimension=4, error_tolerance=1e-10
        ),
    )
    result = phx.solver.solve_finite_tdvp(problem, policy)
    expected = jsp.linalg.expm(-0.05j * pauli_z) @ state.to_dense()

    assert result.successful
    assert jnp.allclose(result.final_state.to_dense(), expected, atol=1e-8)
    assert jnp.allclose(result.diagnostics.norm_history[:2], 1.0, atol=1e-8)


def test_zero_hamiltonian_preserves_multisite_state_without_normalization():
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    state = tn.product_mps(jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128))
    hamiltonian = tn.product_mpo(jnp.stack((zero, jnp.eye(2, dtype=zero.dtype))))
    result = phx.solver.solve_finite_tdvp(
        phx.solver.FiniteTDVPProblem(state, hamiltonian),
        phx.solver.FiniteTDVPPolicy(
            "real-time", step_size=0.1, steps=1, norm_tolerance=1e-7
        ),
    )

    assert result.successful
    assert jnp.allclose(result.final_state.to_dense(), state.to_dense(), atol=1e-8)
    assert jnp.all(result.diagnostics.local_converged_history[0])
