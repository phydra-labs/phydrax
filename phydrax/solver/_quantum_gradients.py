#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..operators.quantum._parameterized import (
    _materialize_quantum_program,
    materialize_quantum_program,
    QuantumProgramTemplate,
)
from ._quantum_expectation import (
    DenseQuantumExpectationResult,
    DenseQuantumObservablePlan,
    evaluate_dense_quantum_observables,
)
from ._quantum_program import (
    DenseQuantumProgramPolicy,
    DenseQuantumProgramResult,
    execute_dense_quantum_program,
    prepare_dense_quantum_program,
    PreparedDenseQuantumProgram,
    refresh_dense_quantum_program,
)


class PreparedDenseQuantumTemplate(StrictModule, NonTrainableState):
    """One content-identified angle template bound to a dense program plan."""

    template: QuantumProgramTemplate
    prepared_program: PreparedDenseQuantumProgram
    prepared_template_id: str = eqx.field(static=True)


class ParameterShiftPlan(StrictModule, NonTrainableState):
    """Linear-size exact shift schedule for all Pauli-rotation occurrences."""

    shifted_occurrences: Array
    shifted_angle_indices: Array
    shifts: Array
    coefficients: Array
    angle_count: int = eqx.field(static=True)
    occurrence_count: int = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    template_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ParameterShiftJacobianResult(StrictModule):
    baseline: DenseQuantumExpectationResult
    jacobian: Array
    shifted_values: Array
    shifted_successful: Array
    evaluation_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def prepare_dense_quantum_template(
    template: QuantumProgramTemplate,
    policy: DenseQuantumProgramPolicy | None = None,
    /,
) -> PreparedDenseQuantumTemplate:
    """Bind one parameterized template to a reusable dense execution plan."""
    if not isinstance(template, QuantumProgramTemplate):
        raise TypeError("template must be a QuantumProgramTemplate.")
    selected = (
        DenseQuantumProgramPolicy(compute_dtype=template.dtype)
        if policy is None
        else policy
    )
    if not isinstance(selected, DenseQuantumProgramPolicy):
        raise TypeError("policy must be DenseQuantumProgramPolicy or None.")
    if selected.compute_dtype != template.dtype:
        raise TypeError("Template and dense-policy dtypes must match exactly.")
    angles = jnp.zeros((template.angle_count,), dtype=template.angle_dtype)
    program = materialize_quantum_program(template, angles)
    prepared = prepare_dense_quantum_program(program, selected)
    prepared_template_id = canonical_fingerprint(
        {
            "kind": "prepared-dense-quantum-template",
            "template": template.template_id,
            "policy": selected.policy_id,
        }
    )
    return PreparedDenseQuantumTemplate(template, prepared, prepared_template_id)


def execute_dense_quantum_template(
    prepared: PreparedDenseQuantumTemplate,
    angles: ArrayLike,
    initial_state: ArrayLike,
    /,
) -> DenseQuantumProgramResult:
    """Materialize, refresh, and execute one dense angle-template instance."""
    if not isinstance(prepared, PreparedDenseQuantumTemplate):
        raise TypeError("prepared must be PreparedDenseQuantumTemplate.")
    program = materialize_quantum_program(prepared.template, angles)
    refreshed = refresh_dense_quantum_program(prepared.prepared_program, program)
    return execute_dense_quantum_program(refreshed, initial_state)


def plan_parameter_shift(template: QuantumProgramTemplate, /) -> ParameterShiftPlan:
    """Plan exact two-point shifts for every Pauli-rotation occurrence."""
    if not isinstance(template, QuantumProgramTemplate):
        raise TypeError("template must be a QuantumProgramTemplate.")
    occurrence_count = template.parameterized_operation_count
    evaluation_count = 2 * occurrence_count
    occurrences = jnp.repeat(
        jnp.arange(occurrence_count, dtype=jnp.int32),
        2,
    )
    angle_indices = jnp.repeat(
        jnp.asarray(template.occurrence_angle_indices, dtype=jnp.int32),
        2,
    )
    signs = jnp.tile(
        jnp.asarray([1.0, -1.0], dtype=template.angle_dtype),
        occurrence_count,
    )
    shifts = signs * jnp.asarray(0.5 * jnp.pi, dtype=template.angle_dtype)
    coefficients = signs * jnp.asarray(0.5, dtype=template.angle_dtype)
    plan_id = canonical_fingerprint(
        {
            "kind": "pauli-parameter-shift-plan",
            "template": template.template_id,
            "occurrence_angle_indices": template.occurrence_angle_indices,
        }
    )
    return ParameterShiftPlan(
        occurrences,
        angle_indices,
        shifts,
        coefficients,
        template.angle_count,
        occurrence_count,
        evaluation_count,
        template.template_id,
        plan_id,
    )


def _validate_shift_inputs(
    prepared: PreparedDenseQuantumTemplate,
    observable_plan: DenseQuantumObservablePlan,
    shift_plan: ParameterShiftPlan,
    angles: ArrayLike,
    initial_state: ArrayLike,
    /,
) -> tuple[Array, Array]:
    if not isinstance(prepared, PreparedDenseQuantumTemplate):
        raise TypeError("prepared must be PreparedDenseQuantumTemplate.")
    if not isinstance(observable_plan, DenseQuantumObservablePlan):
        raise TypeError("observable_plan must be DenseQuantumObservablePlan.")
    if not isinstance(shift_plan, ParameterShiftPlan):
        raise TypeError("shift_plan must be ParameterShiftPlan.")
    if shift_plan.template_id != prepared.template.template_id:
        raise ValueError("Shift plan and prepared template IDs must match.")
    if observable_plan.prepared_id != prepared.prepared_program.prepared_id:
        raise ValueError("Observable plan and prepared template IDs must match.")
    values = jnp.asarray(angles)
    if values.shape != (shift_plan.angle_count,):
        raise ValueError("angles must have exact shape (shift_plan.angle_count,).")
    if values.dtype != prepared.template.angle_dtype:
        raise TypeError("Angle and template precisions must match exactly.")
    state = jnp.asarray(initial_state)
    dimension = prepared.template.layout.dimension
    expected_shape = (
        (dimension,)
        if prepared.template.state_kind == "state-vector"
        else (dimension, dimension)
    )
    if state.shape != expected_shape:
        raise ValueError("Parameter-shift evaluation requires one unbatched state.")
    return values, state


def _shifted_expectations(
    prepared: PreparedDenseQuantumTemplate,
    observable_plan: DenseQuantumObservablePlan,
    shift_plan: ParameterShiftPlan,
    angles: Array,
    initial_state: Array,
    /,
) -> tuple[Array, Array]:
    observable_count = observable_plan.cost.observable_count
    if shift_plan.evaluation_count == 0:
        return (
            jnp.empty((0, observable_count), dtype=prepared.template.angle_dtype),
            jnp.empty((0,), dtype=jnp.bool_),
        )

    def evaluate(occurrence: Array, shift: Array) -> tuple[Array, Array]:
        program = _materialize_quantum_program(
            prepared.template,
            angles,
            shifted_occurrence=occurrence,
            shift=shift,
        )
        refreshed = refresh_dense_quantum_program(
            prepared.prepared_program,
            program,
        )
        program_result = execute_dense_quantum_program(refreshed, initial_state)
        expectation_result = evaluate_dense_quantum_observables(
            observable_plan,
            program_result,
        )
        return (
            jnp.real(expectation_result.complex_values),
            expectation_result.diagnostics.successful,
        )

    return jax.vmap(evaluate)(shift_plan.shifted_occurrences, shift_plan.shifts)


def _assemble_jacobian(
    shift_plan: ParameterShiftPlan,
    shifted_values: Array,
    observable_count: int,
    /,
) -> Array:
    contributions = shift_plan.coefficients[:, None] * shifted_values
    by_angle = jnp.zeros(
        (shift_plan.angle_count, observable_count),
        dtype=shifted_values.dtype,
    )
    by_angle = by_angle.at[shift_plan.shifted_angle_indices].add(contributions)
    return jnp.swapaxes(by_angle, -1, -2)


def evaluate_parameter_shift_jacobian(
    prepared: PreparedDenseQuantumTemplate,
    observable_plan: DenseQuantumObservablePlan,
    shift_plan: ParameterShiftPlan,
    angles: ArrayLike,
    initial_state: ArrayLike,
    /,
) -> ParameterShiftJacobianResult:
    """Evaluate exact expectation values and their angle Jacobian by shifts."""
    values, state = _validate_shift_inputs(
        prepared,
        observable_plan,
        shift_plan,
        angles,
        initial_state,
    )
    baseline_program = execute_dense_quantum_template(prepared, values, state)
    baseline = evaluate_dense_quantum_observables(
        observable_plan,
        baseline_program,
    )
    shifted_values, shifted_successful = _shifted_expectations(
        prepared,
        observable_plan,
        shift_plan,
        values,
        state,
    )
    jacobian = _assemble_jacobian(
        shift_plan,
        shifted_values,
        observable_plan.cost.observable_count,
    )
    return ParameterShiftJacobianResult(
        baseline,
        jacobian,
        shifted_values,
        shifted_successful,
        shift_plan.evaluation_count,
        shift_plan.plan_id,
    )


@eqx.filter_custom_vjp
def _parameter_shift_expectation_values(
    angles: Array,
    prepared: PreparedDenseQuantumTemplate,
    observable_plan: DenseQuantumObservablePlan,
    shift_plan: ParameterShiftPlan,
    initial_state: Array,
    /,
) -> Array:
    program_result = execute_dense_quantum_template(prepared, angles, initial_state)
    return evaluate_dense_quantum_observables(
        observable_plan,
        program_result,
    ).real_values


def _parameter_shift_expectation_values_fwd(
    perturbed: Array,
    angles: Array,
    prepared: PreparedDenseQuantumTemplate,
    observable_plan: DenseQuantumObservablePlan,
    shift_plan: ParameterShiftPlan,
    initial_state: Array,
    /,
) -> tuple[Array, Array]:
    del perturbed
    values = _parameter_shift_expectation_values(
        angles,
        prepared,
        observable_plan,
        shift_plan,
        initial_state,
    )
    return values, angles


def _parameter_shift_expectation_values_bwd(
    residual: Array,
    output_cotangent: Array,
    perturbed: Array,
    angles: Array,
    prepared: PreparedDenseQuantumTemplate,
    observable_plan: DenseQuantumObservablePlan,
    shift_plan: ParameterShiftPlan,
    initial_state: Array,
    /,
) -> Array:
    del perturbed, angles
    shifted_values, shifted_successful = _shifted_expectations(
        prepared,
        observable_plan,
        shift_plan,
        residual,
        initial_state,
    )
    shifted_values = eqx.error_if(
        shifted_values,
        ~jnp.all(shifted_successful),
        "A parameter-shift circuit evaluation was invalid.",
    )
    jacobian = _assemble_jacobian(
        shift_plan,
        shifted_values,
        observable_plan.cost.observable_count,
    )
    return ein.contract("oa,o->a", jacobian, output_cotangent)


_parameter_shift_expectation_values.def_fwd(_parameter_shift_expectation_values_fwd)
_parameter_shift_expectation_values.def_bwd(_parameter_shift_expectation_values_bwd)


__all__ = [
    "ParameterShiftJacobianResult",
    "ParameterShiftPlan",
    "PreparedDenseQuantumTemplate",
    "evaluate_parameter_shift_jacobian",
    "execute_dense_quantum_template",
    "plan_parameter_shift",
    "prepare_dense_quantum_template",
]
