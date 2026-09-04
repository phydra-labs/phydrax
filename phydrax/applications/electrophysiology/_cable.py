#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit cable integration through the PhydraX linear algebra runtime."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._mechanisms import (
    evaluate_membrane_program,
    initialize_membrane_program,
    MechanismStatus,
    MembraneEvaluation,
    MembraneProgram,
    MembraneProgramState,
    update_membrane_program,
)
from ._morphology import PreparedCellMorphology
from ._units import ELECTROPHYSIOLOGY_UNITS


CableScheme = Literal["backward-euler", "crank-nicolson"]


class CableSolveStatus(IntFlag):
    """Fail-closed bitwise cable-step status."""

    SUCCESS = 0
    NONFINITE = 1
    RESIDUAL_FAILURE = 2
    MECHANISM_FAILURE = 4
    INVALID_INPUT = 8


class CableSolverPlan(StrictModule, NonTrainableState):
    """Immutable time-integration and residual acceptance plan."""

    dt_ms: float = eqx.field(static=True)
    scheme: CableScheme = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dt_ms: float,
        /,
        *,
        scheme: CableScheme = "backward-euler",
        residual_tolerance: float = 1.0e-5,
    ):
        if isinstance(dt_ms, bool):
            raise TypeError("dt_ms must be a real scalar, not bool.")
        step = float(dt_ms)
        tolerance = float(residual_tolerance)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("dt_ms must be finite and positive.")
        if scheme not in ("backward-euler", "crank-nicolson"):
            raise ValueError("scheme must be 'backward-euler' or 'crank-nicolson'.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        self.dt_ms = step
        self.scheme = scheme
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-cable-solver-v1",
                "dt_ms": step,
                "scheme": scheme,
                "residual_tolerance": tolerance,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )

    def prepare(
        self,
        morphology: PreparedCellMorphology,
        program: MembraneProgram,
        /,
    ) -> PreparedCableSolver:
        """Bind one fixed morphology and ordered mechanism program."""
        return prepare_cable_solver(self, morphology, program)


class PreparedCableSolver(StrictModule, NonTrainableState):
    """Fixed-shape cable runtime with reusable geometry and policy identity."""

    plan: CableSolverPlan
    morphology: PreparedCellMorphology
    program: MembraneProgram
    theta: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CableSolverPlan,
        morphology: PreparedCellMorphology,
        program: MembraneProgram,
        /,
    ):
        self.plan = plan
        self.morphology = morphology
        self.program = program
        self.theta = 1.0 if plan.scheme == "backward-euler" else 0.5
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-electrophysiology-cable-v1",
                "plan": plan.plan_id,
                "morphology": morphology.runtime_id,
                "program": program.program_id,
            }
        )


class CableState(StrictModule):
    """Complete fixed-shape single-cell state at a cable time boundary."""

    voltage_mV: Array
    membrane: MembraneProgramState
    intracellular_mM: Array
    extracellular_mM: Array
    time_ms: Array
    step_index: Array


class CableStepInputs(StrictModule):
    """Affine synaptic/stimulus terms and exact Dirichlet voltage clamps."""

    injected_current_nA: Array
    synaptic_conductance_uS: Array
    synaptic_current_offset_nA: Array
    voltage_clamp_mask: Array
    voltage_clamp_target_mV: Array


class CableSolveEvidence(StrictModule):
    """Residual, Kirchhoff, clamp-current, finiteness, and routing evidence."""

    residual_norm: Array
    relative_residual: Array
    kirchhoff_residual_nA: Array
    charge_balance_residual_nA: Array
    clamp_current_nA: Array
    status: Array
    successful: Array
    finite: Array
    nonlinear_mechanism_routed: Array


class CableStepResult(StrictModule):
    """Accepted state, rejected candidate, coefficients, and solve evidence."""

    state: CableState
    candidate_voltage_mV: Array
    membrane_evaluation: MembraneEvaluation
    evidence: CableSolveEvidence


def prepare_cable_solver(
    plan: CableSolverPlan,
    morphology: PreparedCellMorphology,
    program: MembraneProgram,
    /,
) -> PreparedCableSolver:
    """Prepare cable integration after validating all fixed-shape dependencies."""
    if not isinstance(plan, CableSolverPlan):
        raise TypeError("plan must be a CableSolverPlan.")
    if not isinstance(morphology, PreparedCellMorphology):
        raise TypeError("morphology must be a PreparedCellMorphology.")
    if not isinstance(program, MembraneProgram):
        raise TypeError("program must be a MembraneProgram.")
    return PreparedCableSolver(plan, morphology, program)


def initialize_cable_state(
    runtime: PreparedCableSolver,
    voltage_mV: Array,
    /,
    *,
    intracellular_mM: Array | None = None,
    extracellular_mM: Array | None = None,
) -> CableState:
    """Create a shape-checked cable state and steady-state mechanism gates."""
    voltage = jnp.asarray(voltage_mV)
    count = runtime.morphology.plan.compartment_count
    if voltage.shape != (count,):
        raise ValueError(f"voltage_mV must have shape {(count,)}.")
    if intracellular_mM is None:
        intracellular = jnp.empty((0, count), dtype=voltage.dtype)
    else:
        intracellular = jnp.asarray(intracellular_mM, dtype=voltage.dtype)
    if extracellular_mM is None:
        extracellular = jnp.empty((0, count), dtype=voltage.dtype)
    else:
        extracellular = jnp.asarray(extracellular_mM, dtype=voltage.dtype)
    if intracellular.shape != extracellular.shape:
        raise ValueError(
            "Intracellular and extracellular concentrations must match shape."
        )
    if intracellular.ndim != 2 or intracellular.shape[1] != count:
        raise ValueError("Ion concentrations must have shape [species, compartment].")
    return CableState(
        voltage,
        initialize_membrane_program(runtime.program, voltage),
        intracellular,
        extracellular,
        jnp.asarray(0.0, dtype=voltage.dtype),
        jnp.asarray(0, dtype=jnp.int32),
    )


def zero_cable_inputs(runtime: PreparedCableSolver, /, *, dtype=None) -> CableStepInputs:
    """Return neutral fixed-shape cable inputs."""
    count = runtime.morphology.plan.compartment_count
    resolved_dtype = runtime.morphology.capacitance_nF.dtype if dtype is None else dtype
    zeros = jnp.zeros((count,), dtype=resolved_dtype)
    return CableStepInputs(zeros, zeros, zeros, jnp.zeros((count,), dtype=bool), zeros)


def _native_dense_solve(matrix: Array, right_hand_side: Array, /) -> Array:
    result = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right_hand_side,
        policy=LinearSolvePolicy(DenseLU()),
    )
    return result.value


def differentiable_dense_solve(
    matrix: Array,
    right_hand_side: Array,
    /,
) -> Array:
    """Solve ``A x = b`` with the native forward/reverse implicit derivative."""
    return _native_dense_solve(matrix, right_hand_side)


def tree_elimination_solve(
    diagonal: Array,
    right_hand_side: Array,
    morphology: PreparedCellMorphology,
    /,
) -> Array:
    """Solve a tree matrix with off-diagonals from prepared axial edges."""
    diagonal_ = jnp.asarray(diagonal)
    right = jnp.asarray(right_hand_side)
    count = morphology.plan.compartment_count
    if diagonal_.shape != (count,) or right.shape != (count,):
        raise ValueError("Tree solve diagonal and right-hand side must match morphology.")

    def eliminate(position, carry):
        reduced_diagonal, reduced_right = carry
        child = morphology.elimination_order[position]
        parent = morphology.parent_index[child]
        conductance = morphology.edge_conductance_uS[child]
        pivot = reduced_diagonal[child]
        reduced_diagonal = reduced_diagonal.at[parent].add(
            -(conductance * conductance) / pivot
        )
        reduced_right = reduced_right.at[parent].add(
            conductance * reduced_right[child] / pivot
        )
        return reduced_diagonal, reduced_right

    reduced_diagonal, reduced_right = jax.lax.fori_loop(
        0,
        morphology.elimination_order.shape[0],
        eliminate,
        (diagonal_, right),
    )
    root = morphology.root_index
    solution = (
        jnp.zeros_like(right).at[root].set(reduced_right[root] / reduced_diagonal[root])
    )

    def substitute(position, value):
        child = morphology.back_substitution_order[position]
        parent = morphology.parent_index[child]
        conductance = morphology.edge_conductance_uS[child]
        return value.at[child].set(
            (reduced_right[child] + conductance * value[parent]) / reduced_diagonal[child]
        )

    return jax.lax.fori_loop(
        0,
        morphology.back_substitution_order.shape[0],
        substitute,
        solution,
    )


def assemble_cable_system(
    runtime: PreparedCableSolver,
    state: CableState,
    evaluation: MembraneEvaluation,
    inputs: CableStepInputs,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Assemble the exact affine theta-method system and physical operator."""
    count = runtime.morphology.plan.compartment_count
    expected = (count,)
    arrays = (
        inputs.injected_current_nA,
        inputs.synaptic_conductance_uS,
        inputs.synaptic_current_offset_nA,
        inputs.voltage_clamp_mask,
        inputs.voltage_clamp_target_mV,
    )
    if any(value.shape != expected for value in arrays):
        raise ValueError(f"Every cable input must have shape {expected}.")
    conductance = evaluation.conductance_uS + inputs.synaptic_conductance_uS
    offset = evaluation.current_offset_nA + inputs.synaptic_current_offset_nA
    physical_operator = runtime.morphology.axial_laplacian_uS + jnp.diag(conductance)
    capacitance_rate = runtime.morphology.capacitance_nF / runtime.plan.dt_ms
    theta = runtime.theta
    matrix = jnp.diag(capacitance_rate) + theta * physical_operator
    right = (
        (jnp.diag(capacitance_rate) - (1.0 - theta) * physical_operator)
        @ state.voltage_mV
        + inputs.injected_current_nA
        - offset
    )
    identity = jnp.eye(count, dtype=matrix.dtype)
    matrix = jnp.where(inputs.voltage_clamp_mask[:, None], identity, matrix)
    right = jnp.where(
        inputs.voltage_clamp_mask,
        inputs.voltage_clamp_target_mV,
        right,
    )
    return matrix, right, physical_operator, offset


def step_cable(
    runtime: PreparedCableSolver,
    state: CableState,
    inputs: CableStepInputs,
    /,
) -> CableStepResult:
    """Advance one implicit cable step and fail closed on invalid evidence."""
    evaluation = evaluate_membrane_program(
        runtime.program,
        state.membrane,
        runtime.morphology,
        state.voltage_mV,
        state.intracellular_mM,
        state.extracellular_mM,
    )
    matrix, right, physical_operator, offset = assemble_cable_system(
        runtime, state, evaluation, inputs
    )
    candidate = differentiable_dense_solve(matrix, right)
    residual = matrix @ candidate - right
    residual_norm = jnp.linalg.norm(residual)
    denominator = jnp.maximum(jnp.linalg.norm(right), jnp.finfo(candidate.dtype).tiny)
    relative_residual = residual_norm / denominator
    theta_voltage = runtime.theta * candidate + (1.0 - runtime.theta) * state.voltage_mV
    physical_residual = (
        runtime.morphology.capacitance_nF
        * (candidate - state.voltage_mV)
        / runtime.plan.dt_ms
        + physical_operator @ theta_voltage
        + offset
        - inputs.injected_current_nA
    )
    clamp_current = jnp.where(inputs.voltage_clamp_mask, physical_residual, 0.0)
    kirchhoff = physical_residual - clamp_current
    charge_balance = jnp.sum(kirchhoff)
    input_finite = (
        jnp.all(jnp.isfinite(inputs.injected_current_nA))
        & jnp.all(jnp.isfinite(inputs.synaptic_conductance_uS))
        & jnp.all(jnp.isfinite(inputs.synaptic_current_offset_nA))
        & jnp.all(jnp.isfinite(inputs.voltage_clamp_target_mV))
    )
    updated_membrane = update_membrane_program(
        runtime.program,
        state.membrane,
        candidate,
        jnp.asarray(runtime.plan.dt_ms, dtype=candidate.dtype),
    )
    gates_finite = jnp.all(jnp.isfinite(updated_membrane.gates))
    finite = (
        input_finite
        & jnp.all(evaluation.finite)
        & jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(relative_residual)
        & jnp.all(jnp.isfinite(kirchhoff))
        & gates_finite
    )
    residual_ok = relative_residual <= runtime.plan.residual_tolerance
    mechanism_ok = jnp.all(
        (
            evaluation.status
            & int(MechanismStatus.NONFINITE | MechanismStatus.INVALID_CONCENTRATION)
        )
        == 0
    )
    status = jnp.asarray(int(CableSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(CableSolveStatus.NONFINITE)),
    )
    status = jnp.where(
        residual_ok,
        status,
        jnp.bitwise_or(status, int(CableSolveStatus.RESIDUAL_FAILURE)),
    )
    status = jnp.where(
        mechanism_ok,
        status,
        jnp.bitwise_or(status, int(CableSolveStatus.MECHANISM_FAILURE)),
    )
    status = jnp.where(
        input_finite,
        status,
        jnp.bitwise_or(status, int(CableSolveStatus.INVALID_INPUT)),
    )
    successful = finite & residual_ok & mechanism_ok & input_finite
    proposed_state = CableState(
        candidate,
        updated_membrane,
        state.intracellular_mM,
        state.extracellular_mM,
        state.time_ms + runtime.plan.dt_ms,
        state.step_index + jnp.asarray(1, dtype=state.step_index.dtype),
    )
    accepted_state = jax.tree.map(
        lambda proposed, prior: jnp.where(successful, proposed, prior),
        proposed_state,
        state,
    )
    evidence = CableSolveEvidence(
        residual_norm,
        relative_residual,
        kirchhoff,
        charge_balance,
        clamp_current,
        status,
        successful,
        finite,
        jnp.any(evaluation.nonlinear_routed),
    )
    return CableStepResult(accepted_state, candidate, evaluation, evidence)


__all__ = [
    "CableScheme",
    "CableSolveEvidence",
    "CableSolveStatus",
    "CableSolverPlan",
    "CableState",
    "CableStepInputs",
    "CableStepResult",
    "PreparedCableSolver",
    "assemble_cable_system",
    "differentiable_dense_solve",
    "initialize_cable_state",
    "prepare_cable_solver",
    "step_cable",
    "tree_elimination_solve",
    "zero_cable_inputs",
]
