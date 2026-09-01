#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


Q = phx.operators.quantum
X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)


def _phase_program(theta, *, state_kind="state-vector"):
    layout = Q.HilbertRegisterLayout(("q",), (2,))
    unitary = jnp.diag(jnp.asarray([jnp.exp(1j * theta), 1.0], dtype=jnp.complex128))
    return Q.QuantumProgram(
        layout,
        (Q.LocalUnitaryOperation(unitary, ("q",)),),
        state_kind=state_kind,
    )


def test_empty_program_is_a_prepared_identity_program():
    layout = Q.HilbertRegisterLayout(("q",), (2,))
    program = Q.QuantumProgram(layout, (), state_kind="state-vector")
    prepared = phx.solver.prepare_dense_quantum_program(program)
    initial = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)

    result = eqx.filter_jit(phx.solver.execute_dense_quantum_program)(prepared, initial)

    assert jnp.allclose(result.final_state, initial)
    assert result.diagnostics.successful
    assert prepared.plan.cost.operation_count == 0
    assert prepared.plan.cost.operation_bytes == 0


def test_density_program_executes_ordered_unitary_and_local_channel():
    layout = Q.HilbertRegisterLayout(("a", "b"), (2, 2))
    gamma = jnp.asarray(0.25)
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
        (
            Q.LocalUnitaryOperation(X, ("b",)),
            Q.LocalKrausChannelOperation(kraus, ("a",)),
        ),
        state_kind="density-matrix",
    )
    prepared = phx.solver.prepare_dense_quantum_program(program)
    ket = jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.complex128)
    density = jnp.outer(ket, jnp.conj(ket))

    result = phx.solver.execute_dense_quantum_program(prepared, density)

    expected = jnp.diag(jnp.asarray([0.0, gamma, 0.0, 1.0 - gamma]))
    assert jnp.allclose(result.final_state, expected)
    assert result.diagnostics.successful
    assert result.diagnostics.positivity_audited
    assert jnp.allclose(result.diagnostics.final_trace_residual, 0.0)
    assert all(item.valid for item in prepared.operation_evidence)
    assert prepared.operation_evidence[1].cp_by_construction


def test_refresh_preserves_prepared_identity_and_supports_real_gradients():
    template = phx.solver.prepare_dense_quantum_program(_phase_program(0.0))
    state = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)

    def objective(theta):
        refreshed = phx.solver.refresh_dense_quantum_program(
            template, _phase_program(theta)
        )
        result = phx.solver.execute_dense_quantum_program(refreshed, state)
        return jnp.real(result.final_state[0])

    theta = jnp.asarray(0.3)
    refreshed = phx.solver.refresh_dense_quantum_program(template, _phase_program(theta))

    assert refreshed.prepared_id == template.prepared_id
    assert refreshed.numeric_version == 1
    assert jnp.allclose(jax.jit(objective)(theta), jnp.cos(theta))
    assert jnp.allclose(jax.grad(objective)(theta), -jnp.sin(theta))
    assert jnp.allclose(
        jax.vmap(objective)(jnp.asarray([0.0, theta])),
        jnp.cos(jnp.asarray([0.0, theta])),
    )


def test_refresh_rejects_every_structural_program_change():
    prepared = phx.solver.prepare_dense_quantum_program(_phase_program(0.0))
    changed_layout = Q.HilbertRegisterLayout(("other",), (2,))
    changed = Q.QuantumProgram(
        changed_layout,
        (Q.LocalUnitaryOperation(jnp.eye(2, dtype=complex), ("other",)),),
        state_kind="state-vector",
    )
    with pytest.raises(ValueError, match="structure changed"):
        phx.solver.refresh_dense_quantum_program(prepared, changed)

    density_program = _phase_program(0.0, state_kind="density-matrix")
    with pytest.raises(ValueError, match="structure changed"):
        phx.solver.refresh_dense_quantum_program(prepared, density_program)


def test_invalid_operations_and_initial_states_fail_closed_with_status():
    layout = Q.HilbertRegisterLayout(("q",), (2,))
    invalid_program = Q.QuantumProgram(
        layout,
        (Q.LocalUnitaryOperation(jnp.zeros((2, 2), dtype=complex), ("q",)),),
        state_kind="state-vector",
    )
    prepared = phx.solver.prepare_dense_quantum_program(invalid_program)
    normalized = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
    invalid_operation = phx.solver.execute_dense_quantum_program(prepared, normalized)

    assert not prepared.operations_valid
    assert invalid_operation.diagnostics.status == int(
        phx.solver.DenseQuantumProgramStatus.INVALID_OPERATION
    )
    assert jnp.allclose(invalid_operation.final_state, normalized)

    valid = phx.solver.prepare_dense_quantum_program(_phase_program(0.0))
    invalid_state = phx.solver.execute_dense_quantum_program(
        valid, jnp.asarray([2.0, 0.0], dtype=jnp.complex128)
    )
    assert invalid_state.diagnostics.status == int(
        phx.solver.DenseQuantumProgramStatus.INVALID_INITIAL_STATE
    )


def test_dense_resource_envelope_rejects_before_execution():
    program = _phase_program(0.0)
    with pytest.raises(MemoryError, match="maximum_state_bytes"):
        phx.solver.plan_dense_quantum_program(
            program,
            phx.solver.DenseQuantumProgramPolicy(maximum_state_bytes=16),
        )
    with pytest.raises(MemoryError, match="maximum_operation_bytes"):
        phx.solver.plan_dense_quantum_program(
            program,
            phx.solver.DenseQuantumProgramPolicy(maximum_operation_bytes=16),
        )
    with pytest.raises(MemoryError, match="maximum_workspace_bytes"):
        phx.solver.plan_dense_quantum_program(
            program,
            phx.solver.DenseQuantumProgramPolicy(maximum_workspace_bytes=16),
        )


def test_construction_audit_carries_cp_tp_closure_without_final_eigensolve():
    layout = Q.HilbertRegisterLayout(("q",), (2,))
    kraus = jnp.eye(2, dtype=jnp.complex128)[None]
    program = Q.QuantumProgram(
        layout,
        (Q.LocalKrausChannelOperation(kraus, ("q",)),),
        state_kind="density-matrix",
    )
    policy = phx.solver.DenseQuantumProgramPolicy(density_positivity_audit="construction")
    prepared = phx.solver.prepare_dense_quantum_program(program, policy)
    density = jnp.diag(jnp.asarray([0.7, 0.3], dtype=jnp.complex128))

    result = phx.solver.execute_dense_quantum_program(prepared, density)

    assert result.diagnostics.successful
    assert not result.diagnostics.positivity_audited
    assert jnp.isnan(result.diagnostics.final_minimum_eigenvalue)
