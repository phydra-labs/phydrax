#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._operators import DenseLinearOperator
from ..linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenResourcePolicy,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
)
from ..tensor_network._uniform import (
    uniform_correlation_length,
    uniform_transfer_fixed_points,
    UniformMatrixProductOperator,
    UniformMatrixProductState,
    UniformTransferFixedPoints,
    UniformTransferPolicy,
    UniformTransferStatus,
)


class UniformVUMPSStatus(IntEnum):
    SUCCESS = 0
    NONINJECTIVE = 1
    NONFINITE = 2
    MAXIMUM_ITERATIONS_REACHED = 3
    ENERGY_INCREASE = 4


class UniformVUMPSProblem(StrictModule):
    initial_state: UniformMatrixProductState
    hamiltonian: UniformMatrixProductOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self, initial_state, hamiltonian, /, *, problem_id: str = "uniform-vumps"
    ):
        if not isinstance(initial_state, UniformMatrixProductState) or not isinstance(
            hamiltonian, UniformMatrixProductOperator
        ):
            raise TypeError("Uniform VUMPS requires a uniform MPS and MPO.")
        if (
            initial_state.unit_cell_size != hamiltonian.unit_cell_size
            or initial_state.physical_dimensions != hamiltonian.input_dimensions
        ):
            raise ValueError(
                "Uniform state and operator unit-cell dimensions must match."
            )
        if hamiltonian.output_dimensions != hamiltonian.input_dimensions:
            raise ValueError("Uniform VUMPS requires a square operator.")
        if initial_state.precision.policy_id != hamiltonian.precision.policy_id:
            raise ValueError("Uniform state and operator precision policies must match.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.initial_state, self.hamiltonian, self.problem_id = (
            initial_state,
            hamiltonian,
            identifier,
        )


class UniformVUMPSPolicy(StrictModule):
    maximum_iterations: int = eqx.field(static=True)
    gradient_step: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    injectivity_tolerance: float = eqx.field(static=True)
    maximum_transfer_elements: int = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 64,
        gradient_step: float = 0.05,
        residual_tolerance: float = 1e-8,
        energy_tolerance: float = 1e-10,
        injectivity_tolerance: float = 1e-8,
        maximum_transfer_elements: int = 10_000_000,
        maximum_history_elements: int = 1_000_000,
    ):
        iterations, step = int(maximum_iterations), float(gradient_step)
        tolerances = (
            float(residual_tolerance),
            float(energy_tolerance),
            float(injectivity_tolerance),
        )
        resources = (int(maximum_transfer_elements), int(maximum_history_elements))
        if (
            iterations < 1
            or not isfinite(step)
            or step <= 0.0
            or any(not isfinite(x) or x < 0.0 for x in tolerances)
        ):
            raise ValueError("Uniform VUMPS iteration and tolerance policy is invalid.")
        if resources[0] < 1 or resources[1] < 4 * iterations + 2:
            raise ValueError("Uniform VUMPS resource capacities are insufficient.")
        self.maximum_iterations, self.gradient_step = iterations, step
        self.residual_tolerance, self.energy_tolerance, self.injectivity_tolerance = (
            tolerances
        )
        self.maximum_transfer_elements, self.maximum_history_elements = resources
        self.policy_id = canonical_fingerprint(
            {
                "kind": "uniform-vumps-policy",
                "iterations": iterations,
                "gradient_step": step,
                "tolerances": tolerances,
                "resources": resources,
            }
        )


class UniformVUMPSCostEstimate(StrictModule):
    norm_transfer_elements: int = eqx.field(static=True)
    operator_transfer_elements: int = eqx.field(static=True)
    tensor_elements: int = eqx.field(static=True)
    history_elements: int = eqx.field(static=True)


class UniformVUMPSPlan(StrictModule):
    policy: UniformVUMPSPolicy
    cost: UniformVUMPSCostEstimate
    state_structure_id: str = eqx.field(static=True)
    hamiltonian_structure_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedUniformVUMPS(StrictModule):
    problem: UniformVUMPSProblem
    plan: UniformVUMPSPlan
    fixed_points: UniformTransferFixedPoints
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class UniformVUMPSDiagnostics(StrictModule):
    energy_history: Array
    galerkin_residual_history: Array
    injectivity_gap_history: Array
    active_iterations: Array
    initial_fixed_point_residual: Array
    status: Array

    @property
    def successful(self):
        return self.status == int(UniformVUMPSStatus.SUCCESS)


class UniformVUMPSResult(StrictModule):
    state: UniformMatrixProductState
    best_state: UniformMatrixProductState
    energy_density: Array
    correlation_length: Array
    fixed_points: UniformTransferFixedPoints
    diagnostics: UniformVUMPSDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self):
        return self.diagnostics.successful


def _operator_transfer(bra, operator, ket, /):
    bond, operator_bond = int(ket.shape[0]), int(operator.shape[0])
    return ein.contract("apr,wpqx,bqs->awbrxs", jnp.conj(bra), operator, ket).reshape(
        (bond * operator_bond * bond, bond * operator_bond * bond)
    )


def _cell_matrix_element(bra_tensors, operator_tensors, ket_tensors, /):
    bond, operator_bond = int(ket_tensors[0].shape[0]), int(operator_tensors[0].shape[0])
    transfer = jnp.eye(
        bond * operator_bond * bond,
        dtype=jnp.result_type(*bra_tensors, *operator_tensors, *ket_tensors),
    )
    for bra, operator, ket in zip(
        bra_tensors, operator_tensors, ket_tensors, strict=True
    ):
        transfer = transfer @ _operator_transfer(bra, operator, ket)
    return jnp.trace(transfer)


def _cell_overlap(bra_tensors, ket_tensors, /):
    bond = int(ket_tensors[0].shape[0])
    transfer = jnp.eye(bond * bond, dtype=jnp.result_type(*bra_tensors, *ket_tensors))
    for bra, ket in zip(bra_tensors, ket_tensors, strict=True):
        transfer = transfer @ ein.contract("apr,bps->abrs", jnp.conj(bra), ket).reshape(
            (bond * bond, bond * bond)
        )
    return jnp.trace(transfer)


def _cell_energy(tensors, operators, /):
    return jnp.real(
        _cell_matrix_element(tensors, operators, tensors)
        / _cell_overlap(tensors, tensors)
    ) / len(tensors)


def _transfer_policy(policy):
    return UniformTransferPolicy(
        maximum_modes=2,
        injectivity_tolerance=policy.injectivity_tolerance,
        maximum_transfer_elements=policy.maximum_transfer_elements,
    )


def _normalize_state(state, fixed):
    factor = jnp.abs(fixed.eigenvalues[0]) ** (-0.5 / float(state.unit_cell_size))
    return UniformMatrixProductState(
        tuple(tensor * factor for tensor in state.tensors), precision=state.precision
    )


def plan_uniform_vumps(problem, policy, /):
    if not isinstance(problem, UniformVUMPSProblem) or not isinstance(
        policy, UniformVUMPSPolicy
    ):
        raise TypeError("plan_uniform_vumps requires a uniform problem and policy.")
    bond, op_bond = (
        problem.initial_state.bond_dimension,
        problem.hamiltonian.bond_dimension,
    )
    norm_elements, operator_elements = (bond * bond) ** 2, (bond * op_bond * bond) ** 2
    tensor_elements = sum(int(tensor.size) for tensor in problem.initial_state.tensors)
    history = 4 * policy.maximum_iterations + 2
    if (
        max(norm_elements, operator_elements) > policy.maximum_transfer_elements
        or history > policy.maximum_history_elements
    ):
        raise MemoryError("Uniform VUMPS resource policy is exceeded.")
    plan_id = canonical_fingerprint(
        {
            "kind": "uniform-vumps-plan",
            "problem": problem.problem_id,
            "state": problem.initial_state.structure_id,
            "hamiltonian": problem.hamiltonian.structure_id,
            "policy": policy.policy_id,
        }
    )
    return UniformVUMPSPlan(
        policy,
        UniformVUMPSCostEstimate(
            norm_elements, operator_elements, tensor_elements, history
        ),
        problem.initial_state.structure_id,
        problem.hamiltonian.structure_id,
        problem.problem_id,
        plan_id,
    )


def _validate(problem, plan):
    if (
        problem.problem_id != plan.problem_id
        or problem.initial_state.structure_id != plan.state_structure_id
        or problem.hamiltonian.structure_id != plan.hamiltonian_structure_id
    ):
        raise ValueError("Uniform VUMPS structure changed; replan is required.")


def prepare_uniform_vumps(problem, plan_or_policy, /):
    if not isinstance(problem, UniformVUMPSProblem):
        raise TypeError("problem must be UniformVUMPSProblem.")
    if not isinstance(plan_or_policy, (UniformVUMPSPlan, UniformVUMPSPolicy)):
        raise TypeError("plan_or_policy must be a uniform VUMPS plan or policy.")
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, UniformVUMPSPlan)
        else plan_uniform_vumps(problem, plan_or_policy)
    )
    _validate(problem, plan)
    fixed = uniform_transfer_fixed_points(
        problem.initial_state, _transfer_policy(plan.policy)
    )
    return PreparedUniformVUMPS(
        problem,
        plan,
        fixed,
        jnp.asarray(0, dtype=jnp.int32),
        canonical_fingerprint({"kind": "prepared-uniform-vumps", "plan": plan.plan_id}),
    )


def refresh_uniform_vumps(prepared, problem, /):
    if not isinstance(prepared, PreparedUniformVUMPS) or not isinstance(
        problem, UniformVUMPSProblem
    ):
        raise TypeError("refresh_uniform_vumps requires prepared and problem values.")
    _validate(problem, prepared.plan)
    fixed = uniform_transfer_fixed_points(
        problem.initial_state, _transfer_policy(prepared.plan.policy)
    )
    return PreparedUniformVUMPS(
        problem, prepared.plan, fixed, prepared.numeric_version + 1, prepared.prepared_id
    )


def _projected_gradient(tensors, operators, /):
    gradient = jax.grad(_cell_energy)(tensors, operators)
    projected = []
    for tensor, value in zip(tensors, gradient, strict=True):
        scale = jnp.real(jnp.vdot(tensor, value)) / jnp.maximum(
            jnp.real(jnp.vdot(tensor, tensor)), jnp.finfo(tensor.real.dtype).tiny
        )
        projected.append(value - scale * tensor)
    return tuple(projected), jnp.sqrt(
        sum(jnp.real(jnp.vdot(value, value)) for value in projected)
    )


def solve_uniform_vumps(problem_or_prepared, policy=None, /):
    if isinstance(problem_or_prepared, PreparedUniformVUMPS):
        if policy is not None:
            raise ValueError("policy must be omitted for prepared uniform VUMPS.")
        prepared = problem_or_prepared
    else:
        if policy is None:
            raise ValueError("policy is required for unprepared uniform VUMPS.")
        prepared = prepare_uniform_vumps(problem_or_prepared, policy)
    selected, state, fixed = (
        prepared.plan.policy,
        prepared.problem.initial_state,
        prepared.fixed_points,
    )
    iterations, real_dtype = selected.maximum_iterations, state.tensors[0].real.dtype
    energies = jnp.full((iterations + 1,), jnp.nan, dtype=real_dtype)
    residuals = jnp.full((iterations,), jnp.nan, dtype=real_dtype)
    gaps = jnp.full((iterations + 1,), jnp.nan, dtype=real_dtype)
    active = jnp.zeros((iterations,), dtype=bool)
    energy = _cell_energy(state.tensors, prepared.problem.hamiltonian.tensors)
    energies, gaps = energies.at[0].set(energy), gaps.at[0].set(fixed.injectivity_gap)
    best_state, best_energy = state, energy
    status = UniformVUMPSStatus.MAXIMUM_ITERATIONS_REACHED
    if int(fixed.status) == int(UniformTransferStatus.NONINJECTIVE):
        status = UniformVUMPSStatus.NONINJECTIVE
    elif not bool(fixed.successful):
        status = UniformVUMPSStatus.NONFINITE
    else:
        state = _normalize_state(state, fixed)
        for iteration in range(iterations):
            projected, residual = _projected_gradient(
                state.tensors, prepared.problem.hamiltonian.tensors
            )
            candidate = UniformMatrixProductState(
                tuple(
                    tensor - selected.gradient_step * value
                    for tensor, value in zip(state.tensors, projected, strict=True)
                ),
                precision=state.precision,
            )
            candidate_fixed = uniform_transfer_fixed_points(
                candidate, _transfer_policy(selected)
            )
            candidate = _normalize_state(candidate, candidate_fixed)
            candidate_energy = _cell_energy(
                candidate.tensors, prepared.problem.hamiltonian.tensors
            )
            residuals, energies, gaps, active = (
                residuals.at[iteration].set(residual),
                energies.at[iteration + 1].set(candidate_energy),
                gaps.at[iteration + 1].set(candidate_fixed.injectivity_gap),
                active.at[iteration].set(True),
            )
            if not bool(
                jnp.isfinite(candidate_energy)
                & jnp.isfinite(residual)
                & jnp.isfinite(candidate_fixed.injectivity_gap)
            ):
                status = UniformVUMPSStatus.NONFINITE
                break
            if int(candidate_fixed.status) == int(UniformTransferStatus.NONINJECTIVE):
                status = UniformVUMPSStatus.NONINJECTIVE
                break
            if float(candidate_energy - energy) > selected.energy_tolerance:
                status = UniformVUMPSStatus.ENERGY_INCREASE
                break
            state, fixed = candidate, candidate_fixed
            if float(candidate_energy) < float(best_energy):
                best_state, best_energy = candidate, candidate_energy
            if (
                float(residual) <= selected.residual_tolerance
                and float(jnp.abs(candidate_energy - energy)) <= selected.energy_tolerance
            ):
                status = UniformVUMPSStatus.SUCCESS
                break
            energy = candidate_energy
    diagnostics = UniformVUMPSDiagnostics(
        energies,
        residuals,
        gaps,
        active,
        prepared.fixed_points.dominant_residual,
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    return UniformVUMPSResult(
        state,
        best_state,
        best_energy,
        uniform_correlation_length(fixed, state.unit_cell_size),
        fixed,
        diagnostics,
        prepared.numeric_version,
        prepared.prepared_id,
    )


class UniformTangentStatus(IntEnum):
    SUCCESS = 0
    NONINJECTIVE = 1
    INVALID_METRIC = 2
    EIGENSOLVE_FAILED = 3
    NONFINITE = 4


class UniformTangentPolicy(StrictModule):
    maximum_modes: int = eqx.field(static=True)
    broadening: float = eqx.field(static=True)
    metric_tolerance: float = eqx.field(static=True)
    maximum_tangent_elements: int = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_modes: int,
        broadening: float = 1e-2,
        metric_tolerance: float = 1e-9,
        maximum_tangent_elements: int = 10_000_000,
    ):
        modes = int(maximum_modes)
        broadening_ = float(broadening)
        tolerance = float(metric_tolerance)
        elements = int(maximum_tangent_elements)
        if modes < 1 or elements < 1 or not isfinite(broadening_) or broadening_ <= 0.0:
            raise ValueError("Tangent capacities and broadening must be positive.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("metric_tolerance must be finite and nonnegative.")
        self.maximum_modes = modes
        self.broadening = broadening_
        self.metric_tolerance = tolerance
        self.maximum_tangent_elements = elements


class UniformTangentResponse(StrictModule):
    excitation_energies: Array
    spectral_weights: Array
    active_modes: Array
    frequencies: Array
    response: Array
    metric_floor: Array
    fixed_point_status: Array
    status: Array

    @property
    def successful(self):
        return self.status == int(UniformTangentStatus.SUCCESS)


def solve_uniform_tangent_response(
    state,
    hamiltonian,
    source_tangent: ArrayLike,
    frequencies: ArrayLike,
    policy,
    /,
    *,
    site: int = 0,
):
    if not isinstance(policy, UniformTangentPolicy):
        raise TypeError("policy must be UniformTangentPolicy.")
    if not isinstance(state, UniformMatrixProductState) or not isinstance(
        hamiltonian, UniformMatrixProductOperator
    ):
        raise TypeError("Uniform tangent response requires a uniform MPS and MPO.")
    if (
        state.unit_cell_size != hamiltonian.unit_cell_size
        or state.physical_dimensions != hamiltonian.input_dimensions
        or hamiltonian.output_dimensions != hamiltonian.input_dimensions
        or state.precision.policy_id != hamiltonian.precision.policy_id
    ):
        raise ValueError("Uniform tangent state and operator are incompatible.")
    site_ = int(site)
    if not 0 <= site_ < state.unit_cell_size:
        raise ValueError("Tangent site is outside the unit cell.")
    source = jnp.asarray(source_tangent)
    frequency_values = jnp.asarray(frequencies)
    if source.shape != state.tensors[site_].shape:
        raise ValueError("source_tangent must match the selected uniform tensor.")
    if (
        frequency_values.ndim != 1
        or frequency_values.size < 1
        or not bool(jnp.all(jnp.isfinite(frequency_values)))
    ):
        raise ValueError("frequencies must be a nonempty finite vector.")
    fixed = uniform_transfer_fixed_points(state, UniformTransferPolicy(maximum_modes=2))
    tangent_dimension = int(source.size - 1)
    capacity = policy.maximum_modes
    complex_dtype = jnp.result_type(source, jnp.complex64)
    energies = jnp.full((capacity,), jnp.nan, dtype=source.real.dtype)
    weights = jnp.full((capacity,), jnp.nan, dtype=source.real.dtype)
    active = jnp.zeros((capacity,), dtype=bool)
    response = jnp.full(frequency_values.shape, jnp.nan + 0j, dtype=complex_dtype)
    metric_floor = jnp.asarray(jnp.nan, dtype=source.real.dtype)
    status = UniformTangentStatus.SUCCESS
    if int(fixed.status) == int(UniformTransferStatus.NONINJECTIVE):
        status = UniformTangentStatus.NONINJECTIVE
    elif not bool(fixed.successful):
        status = UniformTangentStatus.NONFINITE
    elif tangent_dimension < 1:
        status = UniformTangentStatus.INVALID_METRIC
    else:
        flattened = state.tensors[site_].reshape(-1)
        _, _, vh = jnp.linalg.svd(flattened[None, :], full_matrices=True)
        basis = jnp.conj(vh[1:].T)
        workspace = int(basis.size + 2 * tangent_dimension * tangent_dimension)
        if workspace > policy.maximum_tangent_elements:
            raise MemoryError(
                "Uniform tangent workspace exceeds maximum_tangent_elements."
            )
        tangent_tensors = tuple(
            basis[:, index].reshape(source.shape) for index in range(tangent_dimension)
        )
        hessian = jnp.zeros((tangent_dimension, tangent_dimension), dtype=complex_dtype)
        metric = jnp.zeros((tangent_dimension, tangent_dimension), dtype=complex_dtype)
        ground_energy = _cell_energy(state.tensors, hamiltonian.tensors)
        for row in range(tangent_dimension):
            bra = list(state.tensors)
            bra[site_] = tangent_tensors[row]
            for column in range(tangent_dimension):
                ket = list(state.tensors)
                ket[site_] = tangent_tensors[column]
                overlap = _cell_overlap(tuple(bra), tuple(ket))
                matrix = _cell_matrix_element(tuple(bra), hamiltonian.tensors, tuple(ket))
                metric = metric.at[row, column].set(overlap)
                hessian = hessian.at[row, column].set(matrix - ground_energy * overlap)
        metric = 0.5 * (metric + jnp.conj(metric.T))
        hessian = 0.5 * (hessian + jnp.conj(hessian.T))
        metric_floor = jnp.min(jnp.linalg.eigvalsh(metric))
        if float(metric_floor) <= policy.metric_tolerance:
            status = UniformTangentStatus.INVALID_METRIC
        else:
            mode_count = min(capacity, tangent_dimension)
            solve = general_eigensolve(
                GeneralEigenproblem(
                    DenseLinearOperator(hessian),
                    DenseLinearOperator(metric),
                    problem_id="uniform-single-particle-tangent",
                ),
                policy=GeneralEigenSolvePolicy(
                    DenseSchurQZ(),
                    selection=GeneralEigenSelection("smallest-real", count=mode_count),
                    resources=GeneralEigenResourcePolicy(max_dimension=tangent_dimension),
                ),
            )
            if not bool(solve.successful):
                status = UniformTangentStatus.EIGENSOLVE_FAILED
            else:
                computed = jnp.real(solve.eigenvalues[:mode_count])
                source_coordinates = jnp.conj(basis.T) @ source.reshape(-1)
                vectors = solve.right_eigenvector_coordinates[:, :mode_count]
                computed_weights = jnp.square(
                    jnp.abs(jnp.conj(source_coordinates) @ (metric @ vectors))
                )
                energies = energies.at[:mode_count].set(computed)
                weights = weights.at[:mode_count].set(computed_weights)
                active = active.at[:mode_count].set(True)
                response = ein.contract(
                    "m,wm->w",
                    computed_weights,
                    1.0
                    / (
                        frequency_values[:, None]
                        - computed[None, :]
                        + 1j * policy.broadening
                    ),
                )
                finite = (
                    jnp.all(jnp.isfinite(computed))
                    & jnp.all(jnp.isfinite(computed_weights))
                    & jnp.all(jnp.isfinite(response))
                )
                if not bool(finite):
                    status = UniformTangentStatus.NONFINITE
    return UniformTangentResponse(
        energies,
        weights,
        active,
        frequency_values,
        response,
        metric_floor,
        fixed.status,
        jnp.asarray(int(status), dtype=jnp.int32),
    )


__all__ = [
    "PreparedUniformVUMPS",
    "UniformTangentPolicy",
    "UniformTangentResponse",
    "UniformTangentStatus",
    "UniformVUMPSCostEstimate",
    "UniformVUMPSDiagnostics",
    "UniformVUMPSPlan",
    "UniformVUMPSPolicy",
    "UniformVUMPSProblem",
    "UniformVUMPSResult",
    "UniformVUMPSStatus",
    "plan_uniform_vumps",
    "prepare_uniform_vumps",
    "refresh_uniform_vumps",
    "solve_uniform_tangent_response",
    "solve_uniform_vumps",
]
