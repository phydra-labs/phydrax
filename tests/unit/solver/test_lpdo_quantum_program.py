import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


Q = phx.operators.quantum
tn = phx.tensor_network


def _product_lpdo(local_states):
    return tn.LocallyPurifiedDensity(
        tuple(value[None, :, None, None] for value in local_states)
    )


def test_lpdo_program_matches_dense_kraus_execution_and_preserves_psd():
    layout = Q.HilbertRegisterLayout(("q0", "q1"), (2, 2))
    gamma = jnp.asarray(0.2)
    kraus = jnp.stack(
        (
            jnp.asarray(
                [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]],
                dtype=jnp.complex128,
            ),
            jnp.asarray(
                [[0.0, jnp.sqrt(gamma)], [0.0, 0.0]],
                dtype=jnp.complex128,
            ),
        )
    )
    program = Q.QuantumProgram(
        layout,
        (Q.LocalKrausChannelOperation(kraus, ("q0",)),),
        state_kind="density-matrix",
    )
    local_states = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    state = _product_lpdo(local_states)
    policy = phx.solver.LPDOQuantumProgramPolicy(
        maximum_bond_dimension=4,
        maximum_purification_dimension=4,
        maximum_discarded_weight=1e-10,
    )
    plan = phx.solver.plan_lpdo_quantum_program(program, state, policy)
    prepared = phx.solver.prepare_lpdo_quantum_program(program, plan)
    result = eqx.filter_jit(phx.solver.execute_lpdo_quantum_program)(prepared, state)
    density = state.to_dense_density()
    dense = phx.solver.execute_dense_quantum_program(
        phx.solver.prepare_dense_quantum_program(program), density
    )
    eigenvalues = jnp.linalg.eigvalsh(result.final_state.to_dense_density())

    assert result.diagnostics.successful
    assert result.diagnostics.positive_semidefinite_by_construction
    assert jnp.allclose(
        result.final_state.to_dense_density(), dense.final_state, atol=1e-9
    )
    assert jnp.min(eigenvalues) >= -1e-10
    assert jnp.allclose(result.final_state.raw_trace(), 1.0, atol=1e-9)


def test_lpdo_program_executes_one_site_identity_kraus_route():
    layout = Q.HilbertRegisterLayout(("q0", "q1"), (2, 2))
    channel = jnp.eye(2, dtype=jnp.complex128)[None, ...]
    program = Q.QuantumProgram(
        layout,
        (Q.LocalKrausChannelOperation(channel, ("q1",)),),
        state_kind="density-matrix",
    )
    state = _product_lpdo(jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128))
    policy = phx.solver.LPDOQuantumProgramPolicy(
        maximum_bond_dimension=4,
        maximum_purification_dimension=4,
    )
    prepared = phx.solver.prepare_lpdo_quantum_program(
        program,
        phx.solver.plan_lpdo_quantum_program(program, state, policy),
    )
    result = phx.solver.execute_lpdo_quantum_program(prepared, state)
    assert result.diagnostics.successful
    assert jnp.allclose(
        result.final_state.to_dense_density(), state.to_dense_density(), atol=1e-9
    )
