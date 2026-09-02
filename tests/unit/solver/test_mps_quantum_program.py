import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


Q = phx.operators.quantum
tn = phx.tensor_network


def test_mps_program_executes_noncontiguous_ordered_target_window():
    layout = Q.HilbertRegisterLayout(("q0", "q1", "q2"), (2, 2, 2))
    swap = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.complex128,
    )
    program = Q.QuantumProgram(
        layout,
        (Q.LocalUnitaryOperation(swap, ("q2", "q0")),),
        state_kind="state-vector",
    )
    state = tn.product_mps(
        jnp.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    )
    policy = phx.solver.MPSQuantumProgramPolicy(
        maximum_bond_dimension=4,
        maximum_window_sites=3,
        maximum_discarded_weight=1e-10,
    )
    plan = phx.solver.plan_mps_quantum_program(program, state, policy)
    prepared = phx.solver.prepare_mps_quantum_program(program, plan)
    result = eqx.filter_jit(phx.solver.execute_mps_quantum_program)(prepared, state)
    dense = phx.solver.execute_dense_quantum_program(
        phx.solver.prepare_dense_quantum_program(program), state.to_dense()
    )

    assert result.diagnostics.successful
    assert jnp.allclose(result.final_state.to_dense(), dense.final_state, atol=1e-9)
    assert plan.routes[0].target_positions == (2, 0)
    assert plan.routes[0].window_start == 0
    assert plan.routes[0].window_stop == 2


def test_mps_program_refresh_preserves_prepared_identity():
    layout = Q.HilbertRegisterLayout(("q",), (2,))
    identity = jnp.eye(2, dtype=jnp.complex128)
    template_program = Q.QuantumProgram(
        layout,
        (Q.LocalUnitaryOperation(identity, ("q",)),),
        state_kind="state-vector",
    )
    state = tn.product_mps(jnp.asarray([[1.0, 0.0]], dtype=jnp.complex128))
    policy = phx.solver.MPSQuantumProgramPolicy(maximum_bond_dimension=1)
    prepared = phx.solver.prepare_mps_quantum_program(
        template_program,
        phx.solver.plan_mps_quantum_program(template_program, state, policy),
    )
    phase = jnp.diag(jnp.asarray([jnp.exp(0.2j), 1.0], dtype=jnp.complex128))
    updated = Q.QuantumProgram(
        layout,
        (Q.LocalUnitaryOperation(phase, ("q",)),),
        state_kind="state-vector",
    )
    refreshed = phx.solver.refresh_mps_quantum_program(prepared, updated)
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == prepared.numeric_version + 1
