# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

from phydrax.operators.quantum import (
    HilbertRegisterLayout,
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from phydrax.tensor_network import (
    compress_lpdo,
    execute_tensor_network_quantum_program,
    LocallyPurifiedDensity,
    LPDOCompressionPlan,
    prepare_tensor_network_quantum_program,
    product_mps,
    TensorNetworkQuantumProgramPolicy,
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
    prepared = prepare_tensor_network_quantum_program(
        program,
        TensorNetworkQuantumProgramPolicy(
            maximum_operations=4,
            maximum_bond_dimension=4,
            maximum_purification_dimension=4,
        ),
    )
    initial = product_mps(jnp.array([[1, 0], [1, 0]], dtype=jnp.complex64))
    result = execute_tensor_network_quantum_program(prepared, initial)
    assert bool(result.valid)
    assert jnp.allclose(result.state.to_dense(), jnp.array([0, 0, 0, 1]))


def test_tensor_device_rejects_nonphysical_local_operations():
    layout = HilbertRegisterLayout(("q",), (2,))
    policy = TensorNetworkQuantumProgramPolicy(
        maximum_operations=1,
        maximum_bond_dimension=2,
        maximum_purification_dimension=2,
    )

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
    prepared_nonunitary = prepare_tensor_network_quantum_program(
        nonunitary,
        policy,
    )
    vector_result = execute_tensor_network_quantum_program(
        prepared_nonunitary,
        product_mps(jnp.array([[1, 0]], dtype=jnp.complex64)),
    )
    assert not bool(prepared_nonunitary.operation_evidence[0].valid)
    assert not bool(vector_result.operation_valid[0])
    assert not bool(vector_result.valid)

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
    prepared_non_tp = prepare_tensor_network_quantum_program(non_tp, policy)
    density_result = execute_tensor_network_quantum_program(
        prepared_non_tp,
        LocallyPurifiedDensity((jnp.array([[[[1.0]], [[0.0]]]], dtype=jnp.complex64),)),
    )
    assert not bool(prepared_non_tp.operation_evidence[0].valid)
    assert not bool(density_result.operation_valid[0])
    assert not bool(density_result.valid)


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
