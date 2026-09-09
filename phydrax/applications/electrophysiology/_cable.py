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
from ...linalg import (
    LinearSolvePolicy,
    LinearSystem,
    solve,
    StructuredDirect,
    TolerancePolicy,
    TreeLinearOperator,
)
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
    LINEAR_FAILURE = 16


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
    linear_solve_status: Array


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


def assemble_cable_system(
    runtime: PreparedCableSolver,
    state: CableState,
    evaluation: MembraneEvaluation,
    inputs: CableStepInputs,
    /,
    *,
    elapsed_ms: Array | None = None,
) -> tuple[TreeLinearOperator, Array, TreeLinearOperator, Array]:
    """Assemble a linear-storage theta system and the unchanged physical operator.

    Exact Dirichlet row replacement affects only the solve operator. The returned
    physical operator retains every edge for Kirchhoff and clamp-current evidence.
    """
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
    morphology = runtime.morphology
    physical_operator = TreeLinearOperator(
        morphology.axial_diagonal_uS + conductance,
        -morphology.edge_conductance_uS,
        -morphology.edge_conductance_uS,
        morphology.topology,
    )
    elapsed = _cable_elapsed(runtime, state, elapsed_ms)
    capacitance_rate = morphology.capacitance_nF / elapsed
    theta = runtime.theta
    right = (
        capacitance_rate * state.voltage_mV
        - (1.0 - theta) * physical_operator.mv(state.voltage_mV)
        + inputs.injected_current_nA
        - offset
    )
    mask = inputs.voltage_clamp_mask
    parent = jnp.maximum(morphology.topology.parent_index, 0)
    operator = TreeLinearOperator(
        jnp.where(mask, 1.0, capacitance_rate + theta * physical_operator.diagonal),
        jnp.where(mask, 0.0, theta * physical_operator.lower),
        jnp.where(mask[parent], 0.0, theta * physical_operator.upper),
        morphology.topology,
    )
    right = jnp.where(
        inputs.voltage_clamp_mask,
        inputs.voltage_clamp_target_mV,
        right,
    )
    return operator, right, physical_operator, offset


def _cable_elapsed(runtime, state, elapsed_ms, /) -> Array:
    elapsed = jnp.asarray(
        runtime.plan.dt_ms if elapsed_ms is None else elapsed_ms,
        dtype=state.voltage_mV.dtype,
    )
    if elapsed.shape != ():
        raise ValueError("elapsed_ms must be a scalar.")
    return elapsed


def step_cable(
    runtime: PreparedCableSolver,
    state: CableState,
    inputs: CableStepInputs,
    /,
    *,
    elapsed_ms: Array | None = None,
) -> CableStepResult:
    """Advance an implicit step, accepting every coupled field or rolling back.

    ``elapsed_ms`` is a dynamic positive interval; when omitted the prepared plan's
    interval is used. Voltage, gate, charge, and time updates share this interval.
    """
    elapsed = _cable_elapsed(runtime, state, elapsed_ms)
    evaluation = evaluate_membrane_program(
        runtime.program,
        state.membrane,
        runtime.morphology,
        state.voltage_mV,
        state.intracellular_mM,
        state.extracellular_mM,
    )
    operator, right, physical_operator, offset = assemble_cable_system(
        runtime, state, evaluation, inputs, elapsed_ms=elapsed
    )
    solved = solve(
        LinearSystem(operator),
        right,
        policy=LinearSolvePolicy(
            StructuredDirect(),
            tolerance=TolerancePolicy(
                relative=runtime.plan.residual_tolerance, absolute=0.0
            ),
        ),
    )
    candidate = solved.value
    residual_norm = solved.diagnostics.residual_norm
    relative_residual = solved.diagnostics.relative_residual
    linear_ok = jnp.all(solved.successful)
    theta_voltage = runtime.theta * candidate + (1.0 - runtime.theta) * state.voltage_mV
    physical_residual = (
        runtime.morphology.capacitance_nF * (candidate - state.voltage_mV) / elapsed
        + physical_operator.mv(theta_voltage)
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
        & jnp.isfinite(elapsed)
        & (elapsed > 0.0)
    )
    updated_membrane = update_membrane_program(
        runtime.program,
        state.membrane,
        candidate,
        elapsed,
    )
    gates_finite = jnp.asarray(True)
    for gates in updated_membrane.gates:
        gates_finite = gates_finite & jnp.all(jnp.isfinite(gates))
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
    status = jnp.where(
        linear_ok,
        status,
        jnp.bitwise_or(status, int(CableSolveStatus.LINEAR_FAILURE)),
    )
    successful = finite & residual_ok & mechanism_ok & input_finite & linear_ok
    proposed_state = CableState(
        candidate,
        updated_membrane,
        state.intracellular_mM,
        state.extracellular_mM,
        state.time_ms + elapsed,
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
        solved.status,
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
    "initialize_cable_state",
    "prepare_cable_solver",
    "step_cable",
    "zero_cable_inputs",
]
