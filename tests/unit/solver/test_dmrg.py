import jax.numpy as jnp

import phydrax as phx


tn = phx.tensor_network


def _two_site_hamiltonian():
    identity = jnp.eye(2, dtype=jnp.complex128)
    pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    pauli_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    interaction = tn.product_mpo(jnp.stack((-pauli_z, pauli_z)))
    left_field = tn.product_mpo(jnp.stack((-0.5 * pauli_x, identity)))
    right_field = tn.product_mpo(jnp.stack((identity, -0.5 * pauli_x)))
    return tn.add_mpo(tn.add_mpo(interaction, left_field), right_field)


def test_prepared_two_site_dmrg_reaches_dense_ground_state():
    hamiltonian = _two_site_hamiltonian()
    initial = tn.product_mps(jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128))
    problem = phx.solver.DMRGProblem(initial, hamiltonian, problem_id="two-site")
    policy = phx.solver.DMRGPolicy(
        maximum_bond_dimension=2,
        maximum_sweeps=4,
        energy_tolerance=1e-9,
        residual_tolerance=1e-8,
        eigen_policy=phx.linalg.eigen.EigenSolvePolicy(
            phx.linalg.eigen.DenseEigh(), count=1, which="smallest-algebraic"
        ),
    )
    prepared = phx.solver.prepare_dmrg(problem, policy)
    result = phx.solver.solve_dmrg(prepared)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian.to_dense())
    overlap = jnp.abs(jnp.vdot(eigenvectors[:, 0], result.best_state.to_dense()))

    assert result.successful
    assert jnp.allclose(result.energy, eigenvalues[0], atol=1e-8)
    assert jnp.allclose(overlap, 1.0, atol=1e-7)
    assert jnp.any(result.diagnostics.active_sweeps)
    assert jnp.allclose(result.diagnostics.hermiticity_residual, 0.0, atol=1e-10)


def test_dmrg_refresh_preserves_plan_identity_for_numeric_updates():
    hamiltonian = _two_site_hamiltonian()
    initial = tn.product_mps(jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128))
    problem = phx.solver.DMRGProblem(initial, hamiltonian, problem_id="refresh")
    policy = phx.solver.DMRGPolicy(maximum_bond_dimension=2, maximum_sweeps=1)
    prepared = phx.solver.prepare_dmrg(problem, policy)
    updated = phx.solver.DMRGProblem(
        tn.product_mps(jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)),
        hamiltonian,
        problem_id="refresh",
    )
    refreshed = phx.solver.refresh_dmrg(prepared, updated)
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == prepared.numeric_version + 1
