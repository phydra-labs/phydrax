#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.operators.quantum import (
    HilbertRegisterLayout,
    local_density_expectation,
    local_state_expectation,
    LocalObservable,
    LocalUnitaryOperation,
    QuantumProgram,
)
from phydrax.solver import (
    DenseQuantumExpectationStatus,
    evaluate_dense_quantum_observables,
    execute_dense_quantum_program,
    plan_dense_quantum_observables,
    prepare_dense_quantum_program,
)
from phydrax.tensor_network import (
    LocallyPurifiedDensity,
    lpdo_local_observable_expectation,
    MatrixProductState,
    mps_local_observable_expectation,
)


def _paulis():
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    return x, z


def test_local_observable_state_density_and_target_order():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    _, z = _paulis()
    state = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex128)
    density = jnp.outer(state, jnp.conj(state))

    assert jnp.allclose(
        local_state_expectation(layout, LocalObservable(z, ("a",)), state), 1.0
    )
    assert jnp.allclose(
        local_density_expectation(layout, LocalObservable(z, ("b",)), density), -1.0
    )

    ordered = LocalObservable(
        jnp.diag(jnp.asarray([0.0, 1.0, 3.0, 7.0], dtype=jnp.complex128)),
        ("b", "a"),
    )
    assert jnp.allclose(local_state_expectation(layout, ordered, state), 3.0)


def test_dense_observable_plan_groups_targets_and_preserves_output_order():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    x, z = _paulis()
    program = QuantumProgram(
        layout,
        (LocalUnitaryOperation(jnp.eye(2, dtype=jnp.complex128), ("a",)),),
        state_kind="state-vector",
    )
    prepared = prepare_dense_quantum_program(program)
    observables = (
        LocalObservable(z, ("b",)),
        LocalObservable(x, ("a",)),
        LocalObservable(z, ("a",)),
    )
    plan = plan_dense_quantum_observables(prepared, observables)
    state = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)
    result = evaluate_dense_quantum_observables(
        plan,
        execute_dense_quantum_program(prepared, state),
    )

    assert plan.cost.target_group_count == 2
    assert result.diagnostics.status == int(DenseQuantumExpectationStatus.SUCCESS)
    assert jnp.allclose(result.real_values, jnp.asarray([1.0, 0.0, 1.0]))


def test_dense_observable_plan_supports_state_and_density_batches():
    layout = HilbertRegisterLayout(("q",), (2,))
    _, z = _paulis()
    observable = LocalObservable(z, ("q",))
    states = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
    densities = jnp.stack(tuple(jnp.outer(state, jnp.conj(state)) for state in states))

    state_program = QuantumProgram(layout, (), state_kind="state-vector")
    state_prepared = prepare_dense_quantum_program(state_program)
    state_plan = plan_dense_quantum_observables(state_prepared, (observable,))
    state_result = evaluate_dense_quantum_observables(
        state_plan,
        execute_dense_quantum_program(state_prepared, states),
    )

    density_program = QuantumProgram(layout, (), state_kind="density-matrix")
    density_prepared = prepare_dense_quantum_program(density_program)
    density_plan = plan_dense_quantum_observables(density_prepared, (observable,))
    density_result = evaluate_dense_quantum_observables(
        density_plan,
        execute_dense_quantum_program(density_prepared, densities),
    )

    expected = jnp.asarray([[1.0], [-1.0]])
    assert jnp.allclose(state_result.real_values, expected)
    assert jnp.allclose(density_result.real_values, expected)


def test_nonhermitian_observable_is_not_certified_real():
    layout = HilbertRegisterLayout(("q",), (2,))
    raising = LocalObservable(
        jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128),
        ("q",),
    )
    prepared = prepare_dense_quantum_program(
        QuantumProgram(layout, (), state_kind="state-vector")
    )
    plan = plan_dense_quantum_observables(prepared, (raising,))
    state = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
    result = evaluate_dense_quantum_observables(
        plan,
        execute_dense_quantum_program(prepared, state),
    )

    assert not raising.valid
    assert result.diagnostics.status == int(
        DenseQuantumExpectationStatus.INVALID_OBSERVABLE
    )


def test_tensor_network_local_observables_match_dense_and_reject_multisite():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    _, z = _paulis()
    observable = LocalObservable(z, ("b",))
    mps = MatrixProductState(
        (
            jnp.asarray([1.0, 0.0], dtype=jnp.complex128)[None, :, None],
            jnp.asarray([0.0, 1.0], dtype=jnp.complex128)[None, :, None],
        )
    )
    lpdo = LocallyPurifiedDensity(
        (
            jnp.asarray([1.0, 0.0], dtype=jnp.complex128)[None, :, None, None],
            jnp.asarray([0.0, 1.0], dtype=jnp.complex128)[None, :, None, None],
        )
    )

    assert jnp.allclose(mps_local_observable_expectation(mps, layout, observable), -1.0)
    assert jnp.allclose(lpdo_local_observable_expectation(lpdo, layout, observable), -1.0)

    two_site = LocalObservable(jnp.eye(4, dtype=jnp.complex128), ("a", "b"))
    with pytest.raises(ValueError, match="one site"):
        mps_local_observable_expectation(mps, layout, two_site)
    with pytest.raises(ValueError, match="one site"):
        lpdo_local_observable_expectation(lpdo, layout, two_site)
