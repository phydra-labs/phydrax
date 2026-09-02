#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    eigen as eigen_linalg,
    FunctionLinearOperator,
    OperatorProperties,
)
from ..tensor_network import (
    build_mps_mpo_environments,
    canonicalize_mps,
    MatrixProductOperator,
    MatrixProductState,
    mpo_hermiticity_residual,
    mps_mpo_expectation,
)
from ..tensor_network._environments import (
    _left_mps_mpo_step,
    _right_mps_mpo_step,
)
from ..tensor_network._split import truncated_svd


class DMRGStatus(IntEnum):
    SUCCESS = 0
    INVALID_HAMILTONIAN = 1
    LOCAL_SOLVE_FAILED = 2
    NONFINITE_ITERATE = 3
    MAXIMUM_SWEEPS_REACHED = 4
    ENERGY_INCREASE = 5


class DMRGProblem(StrictModule):
    initial_state: MatrixProductState
    hamiltonian: MatrixProductOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_state: MatrixProductState,
        hamiltonian: MatrixProductOperator,
        /,
        *,
        problem_id: str = "matrix-product-ground-state",
    ):
        if not isinstance(initial_state, MatrixProductState):
            raise TypeError("initial_state must be a MatrixProductState.")
        if not isinstance(hamiltonian, MatrixProductOperator):
            raise TypeError("hamiltonian must be a MatrixProductOperator.")
        if initial_state.site_count < 2:
            raise ValueError("Two-site DMRG requires at least two sites.")
        if hamiltonian.output_dimensions != hamiltonian.input_dimensions:
            raise ValueError("DMRG requires a square MPO.")
        if initial_state.physical_dimensions != hamiltonian.input_dimensions:
            raise ValueError("MPS and MPO physical dimensions must match.")
        if initial_state.precision.policy_id != hamiltonian.precision.policy_id:
            raise ValueError("MPS and MPO precision policies must match.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.problem_id = identifier


class DMRGPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_sweeps: int = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    canonical_tolerance: float = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    maximum_environment_elements: int = eqx.field(static=True)
    maximum_local_elements: int = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    eigen_policy: eigen_linalg.EigenSolvePolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_bond_dimension: int,
        maximum_sweeps: int = 8,
        energy_tolerance: float = 1e-8,
        residual_tolerance: float = 1e-7,
        canonical_tolerance: float = 1e-7,
        hermiticity_tolerance: float = 1e-9,
        maximum_environment_elements: int = 100_000_000,
        maximum_local_elements: int = 10_000_000,
        maximum_history_elements: int = 10_000_000,
        eigen_policy: eigen_linalg.EigenSolvePolicy | None = None,
    ):
        bond = int(maximum_bond_dimension)
        sweeps = int(maximum_sweeps)
        if bond < 1 or sweeps < 1:
            raise ValueError("DMRG bond dimension and sweep count must be positive.")
        resource_limits = (
            int(maximum_environment_elements),
            int(maximum_local_elements),
            int(maximum_history_elements),
        )
        if any(value < 1 for value in resource_limits):
            raise ValueError("DMRG resource limits must be positive.")
        tolerances = tuple(
            float(value)
            for value in (
                energy_tolerance,
                residual_tolerance,
                canonical_tolerance,
                hermiticity_tolerance,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("DMRG tolerances must be finite and nonnegative.")
        selected = (
            eigen_linalg.EigenSolvePolicy(
                count=1,
                which="smallest-algebraic",
                max_steps=80,
                tolerance=eigen_linalg.EigenTolerancePolicy(
                    relative=1e-8,
                    absolute=1e-10,
                    orthogonality=1e-8,
                ),
            )
            if eigen_policy is None
            else eigen_policy
        )
        if not isinstance(selected, eigen_linalg.EigenSolvePolicy):
            raise TypeError("eigen_policy must be EigenSolvePolicy or None.")
        if selected.count != 1 or selected.which != "smallest-algebraic":
            raise ValueError("DMRG eigen_policy must request one smallest eigenpair.")
        if selected.differentiation != "none":
            raise ValueError("DMRG does not differentiate through local eigensolves.")
        self.maximum_bond_dimension = bond
        self.maximum_sweeps = sweeps
        self.energy_tolerance = tolerances[0]
        self.residual_tolerance = tolerances[1]
        self.canonical_tolerance = tolerances[2]
        self.hermiticity_tolerance = tolerances[3]
        self.maximum_environment_elements = resource_limits[0]
        self.maximum_local_elements = resource_limits[1]
        self.maximum_history_elements = resource_limits[2]
        self.eigen_policy = selected
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dmrg-policy",
                "maximum_bond_dimension": bond,
                "maximum_sweeps": sweeps,
                "energy_tolerance": tolerances[0],
                "residual_tolerance": tolerances[1],
                "canonical_tolerance": tolerances[2],
                "hermiticity_tolerance": tolerances[3],
                "resource_limits": resource_limits,
                "eigen_method": selected.method.name,
                "eigen_max_steps": selected.max_steps,
            }
        )


class DMRGCostEstimate(StrictModule):
    environment_elements: int = eqx.field(static=True)
    maximum_local_elements: int = eqx.field(static=True)
    history_elements: int = eqx.field(static=True)


class DMRGPlan(StrictModule):
    policy: DMRGPolicy
    cost: DMRGCostEstimate
    state_structure_id: str = eqx.field(static=True)
    hamiltonian_structure_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedDMRG(StrictModule):
    problem: DMRGProblem
    plan: DMRGPlan
    initial_state: MatrixProductState
    hermiticity_residual: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class DMRGDiagnostics(StrictModule):
    energy_history: Array
    energy_change_history: Array
    local_residual_history: Array
    discarded_weight_history: Array
    canonical_residual_history: Array
    active_sweeps: Array
    hermiticity_residual: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(DMRGStatus.SUCCESS)


class DMRGResult(StrictModule):
    final_state: MatrixProductState
    best_state: MatrixProductState
    energy: Array
    diagnostics: DMRGDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.diagnostics.successful


def plan_dmrg(
    problem: DMRGProblem,
    policy: DMRGPolicy,
    /,
) -> DMRGPlan:
    if not isinstance(problem, DMRGProblem):
        raise TypeError("problem must be a DMRGProblem.")
    if not isinstance(policy, DMRGPolicy):
        raise TypeError("policy must be a DMRGPolicy.")
    state = problem.initial_state
    operator = problem.hamiltonian
    environment_elements = sum(
        int(a * b * c)
        for a, b, c in zip(
            (1,) + state.bond_dimensions,
            (1,) + operator.bond_dimensions,
            (1,) + state.bond_dimensions,
            strict=True,
        )
    )
    maximum_local = max(
        int(
            state.tensors[index].shape[0]
            * state.physical_dimensions[index]
            * state.physical_dimensions[index + 1]
            * state.tensors[index + 1].shape[-1]
        )
        for index in range(state.site_count - 1)
    )
    updates = 2 * (state.site_count - 1)
    history = policy.maximum_sweeps * (2 * updates + 3) + policy.maximum_sweeps + 1
    cost = DMRGCostEstimate(environment_elements, maximum_local, history)
    if environment_elements > policy.maximum_environment_elements:
        raise MemoryError("DMRG environments exceed maximum_environment_elements.")
    if maximum_local > policy.maximum_local_elements:
        raise MemoryError("DMRG local problem exceeds maximum_local_elements.")
    if history > policy.maximum_history_elements:
        raise MemoryError("DMRG histories exceed maximum_history_elements.")
    plan_id = canonical_fingerprint(
        {
            "kind": "dmrg-plan",
            "problem": problem.problem_id,
            "state": state.structure_id,
            "hamiltonian": operator.structure_id,
            "policy": policy.policy_id,
        }
    )
    return DMRGPlan(
        policy,
        cost,
        state.structure_id,
        operator.structure_id,
        problem.problem_id,
        plan_id,
    )


def _validate_dmrg_structure(problem: DMRGProblem, plan: DMRGPlan, /) -> None:
    if problem.problem_id != plan.problem_id:
        raise ValueError("DMRG problem identity changed; replan is required.")
    if problem.initial_state.structure_id != plan.state_structure_id:
        raise ValueError("DMRG state structure changed; replan is required.")
    if problem.hamiltonian.structure_id != plan.hamiltonian_structure_id:
        raise ValueError("DMRG Hamiltonian structure changed; replan is required.")


def prepare_dmrg(
    problem: DMRGProblem,
    plan_or_policy: DMRGPlan | DMRGPolicy,
    /,
) -> PreparedDMRG:
    if not isinstance(problem, DMRGProblem):
        raise TypeError("problem must be a DMRGProblem.")
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, DMRGPlan)
        else plan_dmrg(problem, plan_or_policy)
    )
    _validate_dmrg_structure(problem, plan)
    state, _ = canonicalize_mps(problem.initial_state, center=0, normalize=True)
    hermiticity = mpo_hermiticity_residual(problem.hamiltonian)
    prepared_id = canonical_fingerprint({"kind": "prepared-dmrg", "plan": plan.plan_id})
    return PreparedDMRG(
        problem,
        plan,
        state,
        hermiticity,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_dmrg(prepared: PreparedDMRG, problem: DMRGProblem, /) -> PreparedDMRG:
    if not isinstance(prepared, PreparedDMRG):
        raise TypeError("prepared must be PreparedDMRG.")
    if not isinstance(problem, DMRGProblem):
        raise TypeError("problem must be DMRGProblem.")
    _validate_dmrg_structure(problem, prepared.plan)
    state, _ = canonicalize_mps(problem.initial_state, center=0, normalize=True)
    return PreparedDMRG(
        problem,
        prepared.plan,
        state,
        mpo_hermiticity_residual(problem.hamiltonian),
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


class _TwoSiteEffectiveAction(StrictModule):
    left_environment: Array
    left_operator: Array
    right_operator: Array
    right_environment: Array

    def __call__(self, vector: Array, /) -> Array:
        return oe.contract(
            "abc,bpim,mqjn,dne,cije->apqd",
            self.left_environment,
            self.left_operator,
            self.right_operator,
            self.right_environment,
            vector,
        )


def _local_eigen_policy(
    policy: eigen_linalg.EigenSolvePolicy,
    initial: Array,
    /,
) -> eigen_linalg.EigenSolvePolicy:
    return eigen_linalg.EigenSolvePolicy(
        policy.method,
        count=1,
        which="smallest-algebraic",
        max_steps=policy.max_steps,
        tolerance=policy.tolerance,
        resources=policy.resources,
        materialization=policy.materialization,
        initial_basis=initial.reshape((-1, 1)),
        key=policy.key,
        preconditioning=policy.preconditioning,
        differentiation="none",
        failure=policy.failure,
    )


def _solve_two_site(
    state: MatrixProductState,
    hamiltonian: MatrixProductOperator,
    left_environment: Array,
    right_environment: Array,
    bond: int,
    policy: DMRGPolicy,
    direction: int,
    /,
):
    precision = state.precision
    left = precision.contraction(state.tensors[bond])
    right = precision.contraction(state.tensors[bond + 1])
    theta = oe.contract("lpi,iqr->lpqr", left, right)
    action = _TwoSiteEffectiveAction(
        precision.accumulation(left_environment),
        precision.accumulation(hamiltonian.tensors[bond]),
        precision.accumulation(hamiltonian.tensors[bond + 1]),
        precision.accumulation(right_environment),
    )
    space = ArraySpace(theta.shape, dtype=theta.dtype)
    operator = FunctionLinearOperator(
        action,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "verified"},
        ),
        operator_id=f"dmrg-effective-bond-{bond}",
    )
    solve = eigen_linalg.eigensolve(
        eigen_linalg.Eigenproblem(operator, problem_id=f"dmrg-local-bond-{bond}"),
        policy=_local_eigen_policy(policy.eigen_policy, theta),
    )
    optimized = solve.eigenvectors[..., 0]
    matrix = optimized.reshape(
        (
            optimized.shape[0] * optimized.shape[1],
            optimized.shape[2] * optimized.shape[3],
        )
    )
    first, second, truncation = truncated_svd(
        matrix,
        maximum_rank=policy.maximum_bond_dimension,
        absorb="right" if direction > 0 else "left",
        precision=precision,
        evidence_source=state.tensors,
        evidence_children={"input-state": state.precision_evidence},
    )
    retained = truncation.retained_rank
    tensors = list(state.tensors)
    tensors[bond] = first.reshape((optimized.shape[0], optimized.shape[1], retained))
    tensors[bond + 1] = second.reshape((retained, optimized.shape[2], optimized.shape[3]))
    updated = MatrixProductState(tuple(tensors), precision=precision)
    return (
        updated,
        solve.eigenvalues[0],
        solve.diagnostics.residual_norms[0],
        solve.successful,
        truncation,
    )


def _energy(state: MatrixProductState, hamiltonian: MatrixProductOperator, /) -> Array:
    return mps_mpo_expectation(state, hamiltonian)


def solve_dmrg(
    problem_or_prepared: DMRGProblem | PreparedDMRG,
    policy: DMRGPolicy | None = None,
    /,
) -> DMRGResult:
    if isinstance(problem_or_prepared, PreparedDMRG):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared DMRG solve.")
        prepared = problem_or_prepared
    else:
        if policy is None:
            raise ValueError("policy is required for an unprepared DMRG problem.")
        prepared = prepare_dmrg(problem_or_prepared, policy)

    plan_policy = prepared.plan.policy
    state = prepared.initial_state
    hamiltonian = prepared.problem.hamiltonian
    sweeps = plan_policy.maximum_sweeps
    updates = 2 * (state.site_count - 1)
    real_dtype = state.tensors[0].real.dtype
    energy_history = jnp.full((sweeps + 1,), jnp.nan, dtype=real_dtype)
    energy_change_history = jnp.full((sweeps,), jnp.nan, dtype=real_dtype)
    local_residual_history = jnp.full((sweeps, updates), jnp.nan, dtype=real_dtype)
    discarded_history = jnp.full((sweeps, updates), jnp.nan, dtype=real_dtype)
    canonical_history = jnp.full((sweeps,), jnp.nan, dtype=real_dtype)
    active_sweeps = jnp.zeros((sweeps,), dtype=bool)

    initial_energy = _energy(state, hamiltonian)
    energy_history = energy_history.at[0].set(jnp.real(initial_energy))
    best_state = state
    best_energy = jnp.real(initial_energy)
    previous_energy = best_energy
    status = DMRGStatus.MAXIMUM_SWEEPS_REACHED

    if not bool(
        jax.device_get(
            jnp.isfinite(prepared.hermiticity_residual)
            & (prepared.hermiticity_residual <= plan_policy.hermiticity_tolerance)
        )
    ):
        status = DMRGStatus.INVALID_HAMILTONIAN
    else:
        for sweep in range(sweeps):
            residuals = []
            discarded = []
            local_success = True
            left_envs, right_envs = build_mps_mpo_environments(state, hamiltonian, state)
            left_values = list(left_envs)
            for bond in range(state.site_count - 1):
                state, _, residual, successful, truncation = _solve_two_site(
                    state,
                    hamiltonian,
                    left_values[bond],
                    right_envs[bond + 2],
                    bond,
                    plan_policy,
                    +1,
                )
                residuals.append(residual)
                discarded.append(truncation.discarded_weight)
                local_success = local_success and bool(jax.device_get(successful))
                left_values[bond + 1] = _left_mps_mpo_step(
                    left_values[bond],
                    state.precision.accumulation(state.tensors[bond]),
                    state.precision.accumulation(hamiltonian.tensors[bond]),
                    state.precision.accumulation(state.tensors[bond]),
                )
                if not local_success:
                    break

            if local_success:
                left_envs, right_envs = build_mps_mpo_environments(
                    state, hamiltonian, state
                )
                right_values = list(right_envs)
                for bond in range(state.site_count - 2, -1, -1):
                    state, _, residual, successful, truncation = _solve_two_site(
                        state,
                        hamiltonian,
                        left_envs[bond],
                        right_values[bond + 2],
                        bond,
                        plan_policy,
                        -1,
                    )
                    residuals.append(residual)
                    discarded.append(truncation.discarded_weight)
                    local_success = local_success and bool(jax.device_get(successful))
                    right_values[bond + 1] = _right_mps_mpo_step(
                        right_values[bond + 2],
                        state.precision.accumulation(state.tensors[bond + 1]),
                        state.precision.accumulation(hamiltonian.tensors[bond + 1]),
                        state.precision.accumulation(state.tensors[bond + 1]),
                    )
                    if not local_success:
                        break

            if not local_success:
                status = DMRGStatus.LOCAL_SOLVE_FAILED
                break

            state, canonical = canonicalize_mps(state, center=0, normalize=True)
            energy_value = _energy(state, hamiltonian)
            energy = jnp.real(energy_value)
            energy_change = jnp.abs(energy - previous_energy)
            energy_increase = energy - previous_energy
            residual_array = jnp.stack(residuals)
            discarded_array = jnp.stack(discarded)
            canonical_residual = jnp.maximum(
                jnp.max(canonical.left_residuals),
                jnp.max(canonical.right_residuals),
            )
            energy_history = energy_history.at[sweep + 1].set(energy)
            energy_change_history = energy_change_history.at[sweep].set(energy_change)
            local_residual_history = local_residual_history.at[sweep].set(residual_array)
            discarded_history = discarded_history.at[sweep].set(discarded_array)
            canonical_history = canonical_history.at[sweep].set(canonical_residual)
            active_sweeps = active_sweeps.at[sweep].set(True)

            finite = (
                jnp.isfinite(energy)
                & jnp.all(jnp.isfinite(residual_array))
                & jnp.all(jnp.isfinite(discarded_array))
                & jnp.isfinite(canonical_residual)
            )
            if not bool(jax.device_get(finite)):
                status = DMRGStatus.NONFINITE_ITERATE
                break
            if float(jax.device_get(energy)) < float(jax.device_get(best_energy)):
                best_energy = energy
                best_state = state
            if float(jax.device_get(energy_increase)) > plan_policy.energy_tolerance:
                status = DMRGStatus.ENERGY_INCREASE
                break

            truncation_floor = jnp.sqrt(jnp.max(discarded_array))
            residual_threshold = jnp.maximum(
                plan_policy.residual_tolerance, truncation_floor
            )
            converged = (
                (jnp.max(residual_array) <= residual_threshold)
                & (energy_change <= plan_policy.energy_tolerance)
                & (canonical_residual <= plan_policy.canonical_tolerance)
            )
            previous_energy = energy
            if bool(jax.device_get(converged)):
                status = DMRGStatus.SUCCESS
                break

    diagnostics = DMRGDiagnostics(
        energy_history,
        energy_change_history,
        local_residual_history,
        discarded_history,
        canonical_history,
        active_sweeps,
        prepared.hermiticity_residual,
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    return DMRGResult(
        state,
        best_state,
        best_energy,
        diagnostics,
        prepared.numeric_version,
        prepared.prepared_id,
        prepared.problem.problem_id,
    )


__all__ = [
    "DMRGCostEstimate",
    "DMRGDiagnostics",
    "DMRGPlan",
    "DMRGPolicy",
    "DMRGProblem",
    "DMRGResult",
    "DMRGStatus",
    "PreparedDMRG",
    "plan_dmrg",
    "prepare_dmrg",
    "refresh_dmrg",
    "solve_dmrg",
]
