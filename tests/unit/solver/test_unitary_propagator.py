#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.scipy as jsp

import phydrax as phx


def test_constant_hamiltonian_unitary_propagation_and_density_action():
    hamiltonian = jnp.asarray([[0.0, 0.5], [0.5, 0.0]], dtype=complex)
    problem = phx.solver.UnitaryPropagatorProblem(
        lambda time, args: hamiltonian,
        2,
        t0=0.0,
        t1=0.2,
    )
    solution = phx.solver.solve_unitary_propagator(
        problem,
        save_times=jnp.asarray([0.0, 0.2]),
        dt0=0.01,
    )
    expected = jsp.linalg.expm(-1j * 0.2 * hamiltonian)
    assert bool(jnp.all(solution.valid))
    assert jnp.allclose(solution.propagators[-1], expected, atol=2e-5)
    assert solution.maximum_unitarity_residual < 1e-8
    evidence = solution.differential_solution.temporal_evidence
    assert evidence is not None
    assert evidence.state_packing is not None
    assert evidence.state_packing.strategy == "native"

    state = jnp.asarray([1.0 + 0.0j, 0.0j])
    density = jnp.outer(state, jnp.conj(state))
    propagated_state = phx.operators.quantum.apply_unitary_to_state(
        solution.propagators[-1], state
    )
    propagated_density = phx.operators.quantum.conjugate_density(
        solution.propagators[-1], density
    )
    assert jnp.allclose(
        propagated_density,
        jnp.outer(propagated_state, jnp.conj(propagated_state)),
    )
    hermitian, trace, minimum = phx.operators.quantum.density_invariant_residuals(
        propagated_density
    )
    assert hermitian < 1e-8
    assert trace < 1e-8
    assert minimum > -1e-8
