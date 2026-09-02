# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.operators.quantum import (
    apply_finite_cptp,
    compose_finite_cptp,
    finite_cptp_from_kraus,
    finite_cptp_from_local_kraus_operation,
    finite_cptp_from_unitary,
    HilbertRegisterLayout,
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from phydrax.solver._quantum_measurement import (
    execute_mid_circuit_quantum_plan,
    measure_dense_quantum_program,
    MidCircuitQuantumPlan,
    QuantumMeasurementPlan,
)
from phydrax.solver._quantum_program import (
    DenseQuantumProgramPolicy,
    execute_dense_quantum_program,
    prepare_dense_quantum_program,
)


def _amplitude_damping(probability=0.25):
    return jnp.array(
        [
            [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - probability)]],
            [[0.0, jnp.sqrt(probability)], [0.0, 0.0]],
        ],
        dtype=jnp.complex64,
    )


def test_finite_cptp_representations_apply_and_compose_without_state_repair():
    damping = finite_cptp_from_kraus(_amplitude_damping())
    identity = finite_cptp_from_unitary(jnp.eye(2, dtype=jnp.complex64))
    excited = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
    output = apply_finite_cptp(damping, excited)
    assert bool(damping.valid)
    assert jnp.allclose(jnp.trace(output), 1.0)
    assert jnp.allclose(jnp.real(output[0, 0]), 0.25)
    composed = compose_finite_cptp(damping, identity)
    assert jnp.allclose(apply_finite_cptp(composed, excited), output)


def test_rectangular_finite_channel_has_explicit_input_output_dimensions():
    kraus = jnp.zeros((3, 2, 3), dtype=jnp.complex64)
    kraus = kraus.at[0, 0, 0].set(1.0)
    kraus = kraus.at[1, 1, 1].set(1.0)
    kraus = kraus.at[2, 0, 2].set(1.0)
    channel = finite_cptp_from_kraus(kraus)
    assert channel.input_dimension == 3
    assert channel.output_dimension == 2
    assert bool(channel.valid)
    assert apply_finite_cptp(channel, jnp.eye(3, dtype=jnp.complex64) / 3).shape == (2, 2)


def test_canonical_program_adapter_measurement_replay_and_midcircuit_branches():
    layout = HilbertRegisterLayout(("q",), (2,))
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    state_program = QuantumProgram(
        layout,
        (LocalUnitaryOperation(x, ("q",)),),
        state_kind="state-vector",
    )
    policy = DenseQuantumProgramPolicy(compute_dtype="complex64")
    prepared = prepare_dense_quantum_program(state_program, policy)
    result = execute_dense_quantum_program(
        prepared, jnp.array([1, 0], dtype=jnp.complex64)
    )
    effects = jnp.stack(
        (
            jnp.diag(jnp.array([1, 0], dtype=jnp.complex64)),
            jnp.diag(jnp.array([0, 1], dtype=jnp.complex64)),
        )
    )
    measurement = QuantumMeasurementPlan(
        effects, shots=16, measurement_id="computational"
    )
    first = measure_dense_quantum_program(prepared, result, measurement, key=jr.key(4))
    second = measure_dense_quantum_program(prepared, result, measurement, key=jr.key(4))
    assert jnp.array_equal(first.sampled_outcomes, second.sampled_outcomes)
    assert jnp.array_equal(first.counts, jnp.array([0, 16]))

    density_program = QuantumProgram(layout, (), state_kind="density-matrix")
    branch = prepare_dense_quantum_program(density_program, policy)
    mid = MidCircuitQuantumPlan(
        prepared,
        effects,
        (branch, branch),
        plan_id="bounded-control",
    )
    branched = execute_mid_circuit_quantum_plan(
        mid, jnp.array([1, 0], dtype=jnp.complex64)
    )
    assert bool(branched.valid)
    assert jnp.allclose(branched.outcome_probabilities, jnp.array([0.0, 1.0]))

    operation = LocalKrausChannelOperation(_amplitude_damping(), ("q",))
    adapted = finite_cptp_from_local_kraus_operation(operation)
    assert bool(adapted.valid)


def test_midcircuit_branches_require_the_prefix_hilbert_layout_identity():
    prefix_layout = HilbertRegisterLayout(("qubit", "qutrit"), (2, 3))
    matching_layout = HilbertRegisterLayout(("qubit", "qutrit"), (2, 3))
    reordered_layout = HilbertRegisterLayout(("qubit", "qutrit"), (3, 2))
    policy = DenseQuantumProgramPolicy(compute_dtype="complex64")
    prefix = prepare_dense_quantum_program(
        QuantumProgram(prefix_layout, (), state_kind="state-vector"),
        policy,
    )
    matching_branch = prepare_dense_quantum_program(
        QuantumProgram(matching_layout, (), state_kind="density-matrix"),
        policy,
    )
    reordered_branch = prepare_dense_quantum_program(
        QuantumProgram(reordered_layout, (), state_kind="density-matrix"),
        policy,
    )
    measurement_kraus = jnp.eye(6, dtype=jnp.complex64)[None, :, :]

    plan = MidCircuitQuantumPlan(
        prefix,
        measurement_kraus,
        (matching_branch,),
        plan_id="matching-composite-layout",
    )
    result = execute_mid_circuit_quantum_plan(
        plan,
        jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.complex64),
    )
    assert bool(result.valid)
    assert jnp.array_equal(result.outcome_probabilities, jnp.array([1.0]))

    with pytest.raises(ValueError, match="same Hilbert layout"):
        MidCircuitQuantumPlan(
            prefix,
            measurement_kraus,
            (reordered_branch,),
            plan_id="reordered-composite-layout",
        )


def test_midcircuit_validity_requires_only_successful_active_branches():
    layout = HilbertRegisterLayout(("q",), (2,))
    policy = DenseQuantumProgramPolicy(compute_dtype="complex64")
    prefix = prepare_dense_quantum_program(
        QuantumProgram(layout, (), state_kind="state-vector"),
        policy,
    )
    valid_branch = prepare_dense_quantum_program(
        QuantumProgram(layout, (), state_kind="density-matrix"),
        policy,
    )
    invalid_branch = prepare_dense_quantum_program(
        QuantumProgram(
            layout,
            (
                LocalUnitaryOperation(
                    2.0 * jnp.eye(2, dtype=jnp.complex64),
                    ("q",),
                ),
            ),
            state_kind="density-matrix",
        ),
        policy,
    )
    measurement_kraus = jnp.stack(
        (
            jnp.diag(jnp.array([1, 0], dtype=jnp.complex64)),
            jnp.diag(jnp.array([0, 1], dtype=jnp.complex64)),
        )
    )
    initial = jnp.array([1, 0], dtype=jnp.complex64)

    inactive_failure = execute_mid_circuit_quantum_plan(
        MidCircuitQuantumPlan(
            prefix,
            measurement_kraus,
            (valid_branch, invalid_branch),
            plan_id="inactive-invalid-branch",
        ),
        initial,
    )
    assert bool(inactive_failure.valid)
    assert jnp.array_equal(
        inactive_failure.valid_branches,
        jnp.array([True, False]),
    )

    active_failure = execute_mid_circuit_quantum_plan(
        MidCircuitQuantumPlan(
            prefix,
            measurement_kraus,
            (invalid_branch, valid_branch),
            plan_id="active-invalid-branch",
        ),
        initial,
    )
    assert not bool(active_failure.valid)
    assert jnp.array_equal(
        active_failure.valid_branches,
        jnp.array([False, False]),
    )
