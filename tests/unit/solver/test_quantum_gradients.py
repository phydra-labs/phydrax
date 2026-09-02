#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.operators.quantum import (
    HilbertRegisterLayout,
    LocalObservable,
    materialize_quantum_program,
    PauliRotationInstruction,
    QuantumProgramTemplate,
)
from phydrax.solver import (
    evaluate_parameter_shift_jacobian,
    execute_dense_quantum_template,
    plan_dense_quantum_observables,
    plan_parameter_shift,
    prepare_dense_quantum_template,
)


def _one_qubit_problem(*, occurrences=1):
    layout = HilbertRegisterLayout(("q",), (2,))
    template = QuantumProgramTemplate(
        layout,
        tuple(PauliRotationInstruction(("X",), ("q",), 0) for _ in range(occurrences)),
        state_kind="state-vector",
    )
    prepared = prepare_dense_quantum_template(template)
    z = LocalObservable(
        jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128),
        ("q",),
    )
    observable_plan = plan_dense_quantum_observables(
        prepared.prepared_program,
        (z,),
    )
    state = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
    return template, prepared, observable_plan, state


def test_template_materialization_preserves_program_schema():
    template, prepared, _observable_plan, state = _one_qubit_problem()
    first = materialize_quantum_program(template, jnp.asarray([0.1]))
    second = materialize_quantum_program(template, jnp.asarray([0.7]))

    assert first.program_id == second.program_id
    first_result = execute_dense_quantum_template(
        prepared,
        jnp.asarray([0.1]),
        state,
    )
    second_result = execute_dense_quantum_template(
        prepared,
        jnp.asarray([0.7]),
        state,
    )
    assert first_result.prepared_id == second_result.prepared_id
    assert int(first_result.numeric_version) == 1
    assert int(second_result.numeric_version) == 1


def test_parameter_shift_matches_analytic_and_autodiff_derivatives():
    _template, prepared, observable_plan, state = _one_qubit_problem()
    shift_plan = plan_parameter_shift(prepared.template)
    theta = jnp.asarray([0.37], dtype=jnp.float64)
    shifted = evaluate_parameter_shift_jacobian(
        prepared,
        observable_plan,
        shift_plan,
        theta,
        state,
    )

    def expectation(angle):
        program_result = execute_dense_quantum_template(prepared, angle, state)
        from phydrax.solver import evaluate_dense_quantum_observables

        return evaluate_dense_quantum_observables(
            observable_plan,
            program_result,
        ).real_values[0]

    autodiff = jax.grad(expectation)(theta)[0]
    assert shifted.evaluation_count == 2
    assert jnp.allclose(shifted.baseline.real_values[0], jnp.cos(theta[0]))
    assert jnp.allclose(shifted.jacobian[0, 0], -jnp.sin(theta[0]))
    assert jnp.allclose(shifted.jacobian[0, 0], autodiff)


def test_parameter_shift_sums_shared_angle_occurrences():
    _template, prepared, observable_plan, state = _one_qubit_problem(occurrences=2)
    shift_plan = plan_parameter_shift(prepared.template)
    theta = jnp.asarray([0.23], dtype=jnp.float64)
    shifted = evaluate_parameter_shift_jacobian(
        prepared,
        observable_plan,
        shift_plan,
        theta,
        state,
    )

    assert shift_plan.occurrence_count == 2
    assert shift_plan.evaluation_count == 4
    assert jnp.allclose(shifted.baseline.real_values[0], jnp.cos(2.0 * theta[0]))
    assert jnp.allclose(
        shifted.jacobian[0, 0],
        -2.0 * jnp.sin(2.0 * theta[0]),
    )


def test_template_validation_rejects_invalid_bindings_and_qudit_rotations():
    layout = HilbertRegisterLayout(("q",), (2,))
    with pytest.raises(ValueError, match="contiguous"):
        QuantumProgramTemplate(
            layout,
            (PauliRotationInstruction(("X",), ("q",), 1),),
            state_kind="state-vector",
        )

    template = QuantumProgramTemplate(
        layout,
        (PauliRotationInstruction(("X",), ("q",), 0),),
        state_kind="state-vector",
    )
    with pytest.raises(ValueError, match="exact shape"):
        materialize_quantum_program(template, jnp.asarray([0.1, 0.2]))
    with pytest.raises(TypeError, match="precisions"):
        materialize_quantum_program(
            template,
            jnp.asarray([0.1], dtype=jnp.float32),
        )

    qutrit = HilbertRegisterLayout(("t",), (3,))
    with pytest.raises(ValueError, match="two-dimensional"):
        QuantumProgramTemplate(
            qutrit,
            (PauliRotationInstruction(("Z",), ("t",), 0),),
            state_kind="state-vector",
        )
