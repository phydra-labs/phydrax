# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

from phydrax.operators.quantum import (
    HilbertRegisterLayout,
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from phydrax.solver import (
    execute_lpdo_quantum_program,
    execute_mps_quantum_program,
    LPDOQuantumProgramPolicy,
    MPSQuantumProgramPolicy,
    plan_lpdo_quantum_program,
    plan_mps_quantum_program,
    prepare_lpdo_quantum_program,
    prepare_mps_quantum_program,
)
from phydrax.tensor_network import (
    compress_lpdo,
    LocallyPurifiedDensity,
    LPDOCompressionPlan,
    product_mps,
)


def test_canonical_quantum_program_executes_on_mps_without_densification():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    cnot = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=jnp.complex64,
    )
    program = QuantumProgram(
        layout,
        (
            LocalUnitaryOperation(x, ("a",)),
            LocalUnitaryOperation(cnot, ("a", "b")),
        ),
        state_kind="state-vector",
    )
    initial = product_mps(jnp.array([[1, 0], [1, 0]], dtype=jnp.complex64))
    policy = MPSQuantumProgramPolicy(maximum_bond_dimension=4)
    plan = plan_mps_quantum_program(program, initial, policy)
    prepared = prepare_mps_quantum_program(program, plan)
    result = execute_mps_quantum_program(prepared, initial)
    assert bool(result.diagnostics.successful)
    assert jnp.allclose(result.final_state.to_dense(), jnp.array([0, 0, 0, 1]))


def test_tensor_program_executors_reject_nonphysical_local_operations():
    layout = HilbertRegisterLayout(("q",), (2,))
    initial_mps = product_mps(jnp.array([[1, 0]], dtype=jnp.complex64))
    nonunitary = QuantumProgram(
        layout,
        (
            LocalUnitaryOperation(
                2.0 * jnp.eye(2, dtype=jnp.complex64),
                ("q",),
            ),
        ),
        state_kind="state-vector",
    )
    mps_policy = MPSQuantumProgramPolicy(maximum_bond_dimension=2)
    mps_plan = plan_mps_quantum_program(nonunitary, initial_mps, mps_policy)
    mps_prepared = prepare_mps_quantum_program(nonunitary, mps_plan)
    vector_result = execute_mps_quantum_program(mps_prepared, initial_mps)
    assert not bool(mps_prepared.operation_evidence[0].valid)
    assert not bool(vector_result.diagnostics.operations_valid)
    assert not bool(vector_result.diagnostics.successful)

    non_tp = QuantumProgram(
        layout,
        (
            LocalKrausChannelOperation(
                (0.5 * jnp.eye(2, dtype=jnp.complex64))[None, ...],
                ("q",),
            ),
        ),
        state_kind="density-matrix",
    )
    initial_lpdo = LocallyPurifiedDensity(
        (jnp.array([[[[1.0]], [[0.0]]]], dtype=jnp.complex64),)
    )
    lpdo_policy = LPDOQuantumProgramPolicy(
        maximum_bond_dimension=2,
        maximum_purification_dimension=2,
    )
    lpdo_plan = plan_lpdo_quantum_program(non_tp, initial_lpdo, lpdo_policy)
    lpdo_prepared = prepare_lpdo_quantum_program(non_tp, lpdo_plan)
    density_result = execute_lpdo_quantum_program(lpdo_prepared, initial_lpdo)
    assert not bool(lpdo_prepared.operation_evidence[0].valid)
    assert not bool(density_result.diagnostics.operations_valid)
    assert not bool(density_result.diagnostics.successful)


def test_lpdo_compression_remains_psd_by_factor_construction_with_bound():
    tensors = (jnp.array([[[[1.0], [0.1]], [[0.0], [0.2]]]], dtype=jnp.complex64),)
    state = LocallyPurifiedDensity(tensors)
    result = compress_lpdo(
        state,
        LPDOCompressionPlan(
            maximum_bond_dimension=2,
            maximum_purification_dimension=1,
        ),
    )
    density = result.state.to_dense_density()
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (density + jnp.conj(density.T)))
    assert bool(result.valid)
    assert result.positive_by_construction
    assert jnp.min(eigenvalues) >= -1e-6
    assert result.trace_distance_upper_bound >= 0.0
