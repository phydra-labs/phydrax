#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    matrix_exponential_action,
    MatrixFunctionPolicy,
    OperatorProperties,
)
from ..tensor_network._canonical import canonicalize_mps
from ..tensor_network._core import MatrixProductOperator, MatrixProductState
from ..tensor_network._environments import (
    _left_mps_mpo_step,
    _right_mps_mpo_step,
    BondOverlapEffectiveAction,
    mpo_hermiticity_residual,
    mps_mpo_expectation,
    mps_norm_squared,
    OneSiteMPOEffectiveAction,
    prepare_chain_environments,
    TwoSiteMPOEffectiveAction,
)
from ..tensor_network._models import FixedStructureMPOCoefficients
from ..tensor_network._split import truncated_svd


FiniteTDVPMode: TypeAlias = Literal["real-time", "imaginary-time"]
FiniteTDVPAlgorithm: TypeAlias = Literal["one-site", "two-site"]


class FiniteTDVPStatus(IntEnum):
    SUCCESS = 0
    INVALID_HAMILTONIAN = 1
    LOCAL_EXPONENTIAL_FAILED = 2
    NONFINITE_ITERATE = 3
    NORM_DRIFT = 4
    ENERGY_INCREASE = 5


class FiniteTDVPProblem(StrictModule):
    initial_state: MatrixProductState
    hamiltonian: MatrixProductOperator | FixedStructureMPOCoefficients
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_state: MatrixProductState,
        hamiltonian: MatrixProductOperator | FixedStructureMPOCoefficients,
        /,
        *,
        problem_id: str = "finite-matrix-product-tdvp",
    ):
        if not isinstance(initial_state, MatrixProductState):
            raise TypeError("initial_state must be a MatrixProductState.")
        if not isinstance(
            hamiltonian, (MatrixProductOperator, FixedStructureMPOCoefficients)
        ):
            raise TypeError(
                "hamiltonian must be an MPO or FixedStructureMPOCoefficients."
            )
        reference = (
            hamiltonian
            if isinstance(hamiltonian, MatrixProductOperator)
            else hamiltonian.operator_at(0)
        )
        if reference.output_dimensions != reference.input_dimensions:
            raise ValueError("Finite TDVP requires a square MPO.")
        if initial_state.physical_dimensions != reference.input_dimensions:
            raise ValueError("MPS and MPO physical dimensions must match.")
        if initial_state.precision.policy_id != reference.precision.policy_id:
            raise ValueError("MPS and MPO precision policies must match.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.problem_id = identifier

    def hamiltonian_at(self, step: int, /) -> MatrixProductOperator:
        if isinstance(self.hamiltonian, MatrixProductOperator):
            return self.hamiltonian
        return self.hamiltonian.operator_at(step)


class FiniteTDVPPolicy(StrictModule):
    mode: FiniteTDVPMode = eqx.field(static=True)
    algorithm: FiniteTDVPAlgorithm = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    norm_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    maximum_environment_elements: int = eqx.field(static=True)
    maximum_local_elements: int = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    integrator: MatrixFunctionPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: FiniteTDVPMode,
        /,
        *,
        step_size: float,
        steps: int,
        algorithm: FiniteTDVPAlgorithm = "one-site",
        maximum_bond_dimension: int = 1,
        normalize: bool | None = None,
        norm_tolerance: float = 1e-7,
        energy_tolerance: float = 1e-7,
        hermiticity_tolerance: float = 1e-9,
        maximum_environment_elements: int = 100_000_000,
        maximum_local_elements: int = 10_000_000,
        maximum_history_elements: int = 10_000_000,
        integrator: MatrixFunctionPolicy | None = None,
    ):
        if mode not in ("real-time", "imaginary-time"):
            raise ValueError("Unknown finite TDVP mode.")
        if algorithm not in ("one-site", "two-site"):
            raise ValueError("Unknown finite TDVP algorithm.")
        step = float(step_size)
        count = int(steps)
        bond = int(maximum_bond_dimension)
        tolerances = (
            float(norm_tolerance),
            float(energy_tolerance),
            float(hermiticity_tolerance),
        )
        resources = (
            int(maximum_environment_elements),
            int(maximum_local_elements),
            int(maximum_history_elements),
        )
        if not isfinite(step) or step <= 0.0 or count < 0 or bond < 1:
            raise ValueError("TDVP step, step count, and bond capacity are invalid.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("TDVP tolerances must be finite and nonnegative.")
        if any(value < 1 for value in resources):
            raise ValueError("TDVP resource capacities must be positive.")
        selected = MatrixFunctionPolicy("lanczos") if integrator is None else integrator
        if not isinstance(selected, MatrixFunctionPolicy):
            raise TypeError("integrator must be MatrixFunctionPolicy or None.")
        normalize_ = mode == "imaginary-time" if normalize is None else bool(normalize)
        self.mode = mode
        self.algorithm = algorithm
        self.step_size = step
        self.steps = count
        self.maximum_bond_dimension = bond
        self.normalize = normalize_
        self.norm_tolerance = tolerances[0]
        self.energy_tolerance = tolerances[1]
        self.hermiticity_tolerance = tolerances[2]
        self.maximum_environment_elements = resources[0]
        self.maximum_local_elements = resources[1]
        self.maximum_history_elements = resources[2]
        self.integrator = selected
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-tdvp-policy",
                "mode": mode,
                "algorithm": algorithm,
                "step_size": step,
                "steps": count,
                "maximum_bond_dimension": bond,
                "normalize": normalize_,
                "tolerances": tolerances,
                "resources": resources,
                "integrator": selected.method,
                "integrator_dimension": selected.max_dimension,
            }
        )


class FiniteTDVPCostEstimate(StrictModule):
    environment_elements: int = eqx.field(static=True)
    maximum_local_elements: int = eqx.field(static=True)
    history_elements: int = eqx.field(static=True)


class FiniteTDVPPlan(StrictModule):
    policy: FiniteTDVPPolicy
    cost: FiniteTDVPCostEstimate
    state_structure_id: str = eqx.field(static=True)
    hamiltonian_structure_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedFiniteTDVP(StrictModule):
    problem: FiniteTDVPProblem
    plan: FiniteTDVPPlan
    initial_state: MatrixProductState
    hermiticity_history: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class FiniteTDVPDiagnostics(StrictModule):
    time_history: Array
    norm_history: Array
    normalized_energy_history: Array
    normalization_factor_history: Array
    local_error_history: Array
    local_residual_history: Array
    local_converged_history: Array
    truncation_history: Array
    canonical_residual_history: Array
    active_steps: Array
    hermiticity_history: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(FiniteTDVPStatus.SUCCESS)


class FiniteTDVPCheckpoint(StrictModule):
    state: MatrixProductState
    completed_steps: Array
    time: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FiniteTDVPResult(StrictModule):
    final_state: MatrixProductState
    diagnostics: FiniteTDVPDiagnostics
    checkpoint: FiniteTDVPCheckpoint
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.diagnostics.successful


def plan_finite_tdvp(
    problem: FiniteTDVPProblem, policy: FiniteTDVPPolicy, /
) -> FiniteTDVPPlan:
    if not isinstance(problem, FiniteTDVPProblem) or not isinstance(
        policy, FiniteTDVPPolicy
    ):
        raise TypeError("plan_finite_tdvp requires a finite problem and policy.")
    if (
        isinstance(problem.hamiltonian, FixedStructureMPOCoefficients)
        and policy.steps > problem.hamiltonian.step_count
    ):
        raise ValueError("TDVP policy steps exceed the fixed coefficient schedule.")
    state = problem.initial_state
    operator = problem.hamiltonian_at(0)
    if policy.algorithm == "two-site" and state.site_count < 2:
        raise ValueError("Two-site finite TDVP requires at least two sites.")
    state_cuts = (
        (1,) + state.bond_dimensions
        if policy.algorithm == "one-site"
        else (1,) + (policy.maximum_bond_dimension,) * (state.site_count - 1)
    )
    environment_elements = sum(
        int(a * b * c)
        for a, b, c in zip(
            state_cuts,
            (1,) + operator.bond_dimensions,
            state_cuts,
            strict=True,
        )
    )
    if policy.algorithm == "one-site":
        maximum_local = max(int(value.size) for value in state.tensors)
        local_count = 4 * state.site_count - 2
    else:
        maximum_local = max(
            int(
                policy.maximum_bond_dimension
                * state.physical_dimensions[index]
                * state.physical_dimensions[index + 1]
                * policy.maximum_bond_dimension
            )
            for index in range(state.site_count - 1)
        )
        local_count = 2 * (state.site_count - 1)
    history = (policy.steps + 1) * 3 + policy.steps * (4 * local_count + 2)
    if environment_elements > policy.maximum_environment_elements:
        raise MemoryError("Finite TDVP environments exceed maximum_environment_elements.")
    if maximum_local > policy.maximum_local_elements:
        raise MemoryError("Finite TDVP local action exceeds maximum_local_elements.")
    if history > policy.maximum_history_elements:
        raise MemoryError("Finite TDVP histories exceed maximum_history_elements.")
    structure = (
        operator.structure_id
        if isinstance(problem.hamiltonian, MatrixProductOperator)
        else problem.hamiltonian.structure_id
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "finite-tdvp-plan",
            "problem": problem.problem_id,
            "state": state.structure_id,
            "hamiltonian": structure,
            "policy": policy.policy_id,
        }
    )
    return FiniteTDVPPlan(
        policy,
        FiniteTDVPCostEstimate(environment_elements, maximum_local, history),
        state.structure_id,
        structure,
        problem.problem_id,
        plan_id,
    )


def _problem_hamiltonian_structure(problem: FiniteTDVPProblem, /) -> str:
    if isinstance(problem.hamiltonian, MatrixProductOperator):
        return problem.hamiltonian.structure_id
    return problem.hamiltonian.structure_id


def _validate_structure(problem: FiniteTDVPProblem, plan: FiniteTDVPPlan, /) -> None:
    if problem.problem_id != plan.problem_id:
        raise ValueError("Finite TDVP problem identity changed; replan is required.")
    if problem.initial_state.structure_id != plan.state_structure_id:
        raise ValueError("Finite TDVP state structure changed; replan is required.")
    if _problem_hamiltonian_structure(problem) != plan.hamiltonian_structure_id:
        raise ValueError("Finite TDVP Hamiltonian structure changed; replan is required.")


def prepare_finite_tdvp(
    problem: FiniteTDVPProblem,
    plan_or_policy: FiniteTDVPPlan | FiniteTDVPPolicy,
    /,
) -> PreparedFiniteTDVP:
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, FiniteTDVPPlan)
        else plan_finite_tdvp(problem, plan_or_policy)
    )
    _validate_structure(problem, plan)
    state, _ = canonicalize_mps(problem.initial_state, center=0, normalize=True)
    if plan.policy.mode == "real-time" and not jnp.issubdtype(
        state.tensors[0].dtype, jnp.complexfloating
    ):
        raise TypeError("Real-time finite TDVP requires complex MPS storage.")
    hermiticity = jnp.stack(
        [
            mpo_hermiticity_residual(problem.hamiltonian_at(step))
            for step in range(plan.policy.steps + 1)
        ]
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-finite-tdvp", "plan": plan.plan_id}
    )
    return PreparedFiniteTDVP(
        problem,
        plan,
        state,
        hermiticity,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_finite_tdvp(
    prepared: PreparedFiniteTDVP, problem: FiniteTDVPProblem, /
) -> PreparedFiniteTDVP:
    if not isinstance(prepared, PreparedFiniteTDVP) or not isinstance(
        problem, FiniteTDVPProblem
    ):
        raise TypeError("refresh_finite_tdvp requires prepared and problem values.")
    _validate_structure(problem, prepared.plan)
    state, _ = canonicalize_mps(problem.initial_state, center=0, normalize=True)
    hermiticity = jnp.stack(
        [
            mpo_hermiticity_residual(problem.hamiltonian_at(step))
            for step in range(prepared.plan.policy.steps + 1)
        ]
    )
    return PreparedFiniteTDVP(
        problem,
        prepared.plan,
        state,
        hermiticity,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _self_adjoint_operator(action, shape, dtype, identifier):
    space = ArraySpace(shape, dtype=dtype)
    return FunctionLinearOperator(
        action,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "verified"}
        ),
        operator_id=identifier,
    )


def _evolve(action, vector, scale, policy, identifier):
    operator = _self_adjoint_operator(action, vector.shape, vector.dtype, identifier)
    return matrix_exponential_action(operator, vector, scale, policy=policy.integrator)


def _step_scale(policy: FiniteTDVPPolicy, dtype):
    step = jnp.asarray(policy.step_size, dtype=jnp.dtype(dtype).type(0).real.dtype)
    return -1j * step if policy.mode == "real-time" else -step


def _one_site_step(state, hamiltonian, policy):
    precision = state.precision
    base_scale = _step_scale(policy, state.tensors[0].dtype)
    errors = []
    residuals = []
    converged = []
    environments = prepare_chain_environments(state, hamiltonian, state)
    left_values = list(environments.left)
    right_envs = environments.right
    tensors = list(state.tensors)
    for site in range(state.site_count - 1):
        center = precision.factorization(tensors[site])
        evolved = _evolve(
            OneSiteMPOEffectiveAction(
                precision.accumulation(left_values[site]),
                precision.accumulation(hamiltonian.tensors[site]),
                precision.accumulation(right_envs[site + 1]),
            ),
            center,
            0.5 * base_scale,
            policy,
            f"finite-tdvp-site-{site}",
        )
        errors.append(evolved.error_estimate)
        residuals.append(evolved.residual_estimate)
        converged.append(evolved.converged)
        matrix = precision.factorization(evolved.value).reshape(
            (-1, evolved.value.shape[-1])
        )
        q, bond = jnp.linalg.qr(matrix)
        rank = q.shape[-1]
        left_core = q.reshape(evolved.value.shape[:-1] + (rank,))
        left_values[site + 1] = _left_mps_mpo_step(
            precision.accumulation(left_values[site]),
            precision.accumulation(left_core),
            precision.accumulation(hamiltonian.tensors[site]),
            precision.accumulation(left_core),
        )
        bond_evolved = _evolve(
            BondOverlapEffectiveAction(
                precision.accumulation(left_values[site + 1]),
                precision.accumulation(right_envs[site + 1]),
            ),
            bond,
            -0.5 * base_scale,
            policy,
            f"finite-tdvp-bond-{site}",
        )
        errors.append(bond_evolved.error_estimate)
        residuals.append(bond_evolved.residual_estimate)
        converged.append(bond_evolved.converged)
        tensors[site] = precision.storage(left_core)
        tensors[site + 1] = precision.storage(
            ein.contract(
                "ab,bpr->apr",
                precision.contraction(bond_evolved.value),
                precision.contraction(tensors[site + 1]),
            )
        )
    last = state.site_count - 1
    evolved = _evolve(
        OneSiteMPOEffectiveAction(
            precision.accumulation(left_values[last]),
            precision.accumulation(hamiltonian.tensors[last]),
            precision.accumulation(right_envs[last + 1]),
        ),
        precision.factorization(tensors[last]),
        0.5 * base_scale,
        policy,
        f"finite-tdvp-site-{last}",
    )
    tensors[last] = precision.storage(evolved.value)
    errors.append(evolved.error_estimate)
    residuals.append(evolved.residual_estimate)
    converged.append(evolved.converged)
    state = MatrixProductState(tuple(tensors), precision=precision)
    environments = prepare_chain_environments(state, hamiltonian, state)
    left_envs = environments.left
    right_values = list(environments.right)
    tensors = list(state.tensors)
    for site in range(state.site_count - 1, 0, -1):
        center = precision.factorization(tensors[site])
        evolved = _evolve(
            OneSiteMPOEffectiveAction(
                precision.accumulation(left_envs[site]),
                precision.accumulation(hamiltonian.tensors[site]),
                precision.accumulation(right_values[site + 1]),
            ),
            center,
            0.5 * base_scale,
            policy,
            f"finite-tdvp-site-{site}",
        )
        errors.append(evolved.error_estimate)
        residuals.append(evolved.residual_estimate)
        converged.append(evolved.converged)
        matrix = precision.factorization(evolved.value).reshape(
            (evolved.value.shape[0], -1)
        )
        q, triangular = jnp.linalg.qr(matrix.T)
        rank = q.shape[-1]
        right_core = q.T.reshape((rank,) + evolved.value.shape[1:])
        bond = triangular.T
        right_values[site] = _right_mps_mpo_step(
            precision.accumulation(right_values[site + 1]),
            precision.accumulation(right_core),
            precision.accumulation(hamiltonian.tensors[site]),
            precision.accumulation(right_core),
        )
        bond_evolved = _evolve(
            BondOverlapEffectiveAction(
                precision.accumulation(left_envs[site]),
                precision.accumulation(right_values[site]),
            ),
            bond,
            -0.5 * base_scale,
            policy,
            f"finite-tdvp-bond-{site - 1}",
        )
        errors.append(bond_evolved.error_estimate)
        residuals.append(bond_evolved.residual_estimate)
        converged.append(bond_evolved.converged)
        tensors[site] = precision.storage(right_core)
        tensors[site - 1] = precision.storage(
            ein.contract(
                "lpa,ab->lpb",
                precision.contraction(tensors[site - 1]),
                precision.contraction(bond_evolved.value),
            )
        )
    evolved = _evolve(
        OneSiteMPOEffectiveAction(
            precision.accumulation(left_envs[0]),
            precision.accumulation(hamiltonian.tensors[0]),
            precision.accumulation(right_values[1]),
        ),
        precision.factorization(tensors[0]),
        0.5 * base_scale,
        policy,
        "finite-tdvp-site-0",
    )
    tensors[0] = precision.storage(evolved.value)
    errors.append(evolved.error_estimate)
    residuals.append(evolved.residual_estimate)
    converged.append(evolved.converged)
    count = 4 * state.site_count - 2
    return (
        MatrixProductState(tuple(tensors), precision=precision),
        jnp.stack(errors),
        jnp.stack(residuals),
        jnp.stack(converged),
        jnp.zeros((count,), dtype=state.tensors[0].real.dtype),
    )


def _two_site_step(state, hamiltonian, policy):
    precision = state.precision
    half_scale = 0.5 * _step_scale(policy, state.tensors[0].dtype)
    errors = []
    residuals = []
    converged = []
    discarded = []
    for direction in (1, -1):
        environments = prepare_chain_environments(state, hamiltonian, state)
        bonds = (
            range(state.site_count - 1)
            if direction > 0
            else range(state.site_count - 2, -1, -1)
        )
        for bond in bonds:
            left = precision.contraction(state.tensors[bond])
            right = precision.contraction(state.tensors[bond + 1])
            theta = ein.contract("lpi,iqr->lpqr", left, right)
            evolved = _evolve(
                TwoSiteMPOEffectiveAction(
                    precision.accumulation(environments.left[bond]),
                    precision.accumulation(hamiltonian.tensors[bond]),
                    precision.accumulation(hamiltonian.tensors[bond + 1]),
                    precision.accumulation(environments.right[bond + 2]),
                ),
                theta,
                half_scale,
                policy,
                f"finite-two-site-tdvp-{direction}-{bond}",
            )
            matrix = evolved.value.reshape(
                (
                    evolved.value.shape[0] * evolved.value.shape[1],
                    evolved.value.shape[2] * evolved.value.shape[3],
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
            tensors[bond] = first.reshape(
                (evolved.value.shape[0], evolved.value.shape[1], retained)
            )
            tensors[bond + 1] = second.reshape(
                (retained, evolved.value.shape[2], evolved.value.shape[3])
            )
            state = MatrixProductState(tuple(tensors), precision=precision)
            errors.append(evolved.error_estimate)
            residuals.append(evolved.residual_estimate)
            converged.append(evolved.converged)
            discarded.append(truncation.discarded_weight)
            environments = prepare_chain_environments(state, hamiltonian, state)
    return (
        state,
        jnp.stack(errors),
        jnp.stack(residuals),
        jnp.stack(converged),
        jnp.stack(discarded),
    )


def _normalized_energy(
    state: MatrixProductState, hamiltonian: MatrixProductOperator, /
) -> Array:
    return jnp.real(mps_mpo_expectation(state, hamiltonian) / mps_norm_squared(state))


def solve_finite_tdvp(
    problem_or_prepared: FiniteTDVPProblem | PreparedFiniteTDVP,
    policy: FiniteTDVPPolicy | None = None,
    /,
) -> FiniteTDVPResult:
    if isinstance(problem_or_prepared, PreparedFiniteTDVP):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared finite TDVP solve.")
        prepared = problem_or_prepared
    else:
        if policy is None:
            raise ValueError("policy is required for an unprepared finite TDVP problem.")
        prepared = prepare_finite_tdvp(problem_or_prepared, policy)
    plan_policy = prepared.plan.policy
    state = prepared.initial_state
    count = plan_policy.steps
    local_count = (
        4 * state.site_count - 2
        if plan_policy.algorithm == "one-site"
        else 2 * (state.site_count - 1)
    )
    real_dtype = state.tensors[0].real.dtype
    times = plan_policy.step_size * jnp.arange(count + 1, dtype=real_dtype)
    norms = jnp.full((count + 1,), jnp.nan, dtype=real_dtype)
    energies = jnp.full((count + 1,), jnp.nan, dtype=real_dtype)
    normalization = jnp.ones((count,), dtype=real_dtype)
    errors = jnp.full((count, local_count), jnp.nan, dtype=real_dtype)
    residuals = jnp.full((count, local_count), jnp.nan, dtype=real_dtype)
    local_converged = jnp.zeros((count, local_count), dtype=bool)
    truncation = jnp.full((count, local_count), jnp.nan, dtype=real_dtype)
    canonical_residuals = jnp.full((count,), jnp.nan, dtype=real_dtype)
    active = jnp.zeros((count,), dtype=bool)
    norms = norms.at[0].set(state.norm())
    energies = energies.at[0].set(
        _normalized_energy(state, prepared.problem.hamiltonian_at(0))
    )
    status = FiniteTDVPStatus.SUCCESS
    completed_steps = 0
    valid_hamiltonian = jnp.all(
        jnp.isfinite(prepared.hermiticity_history)
        & (prepared.hermiticity_history <= plan_policy.hermiticity_tolerance)
    )
    if not bool(jax.device_get(valid_hamiltonian)):
        status = FiniteTDVPStatus.INVALID_HAMILTONIAN
    else:
        for step in range(count):
            hamiltonian = prepared.problem.hamiltonian_at(step)
            previous_energy = energies[step]
            if plan_policy.algorithm == "one-site":
                state, local_errors, local_residuals, converged, discarded = (
                    _one_site_step(state, hamiltonian, plan_policy)
                )
            else:
                state, local_errors, local_residuals, converged, discarded = (
                    _two_site_step(state, hamiltonian, plan_policy)
                )
            pre_normalization_norm = state.norm()
            normalization = normalization.at[step].set(pre_normalization_norm)
            if plan_policy.normalize:
                state = state.normalized()
            state, canonical = canonicalize_mps(state, center=0, normalize=False)
            norm = state.norm()
            next_hamiltonian = prepared.problem.hamiltonian_at(step + 1)
            energy = _normalized_energy(state, next_hamiltonian)
            canonical_residual = jnp.maximum(
                jnp.max(canonical.left_residuals), jnp.max(canonical.right_residuals)
            )
            errors = errors.at[step].set(local_errors)
            residuals = residuals.at[step].set(local_residuals)
            local_converged = local_converged.at[step].set(converged)
            truncation = truncation.at[step].set(discarded)
            canonical_residuals = canonical_residuals.at[step].set(canonical_residual)
            norms = norms.at[step + 1].set(norm)
            energies = energies.at[step + 1].set(energy)
            active = active.at[step].set(True)
            completed_steps = step + 1
            finite = (
                jnp.isfinite(norm)
                & jnp.isfinite(energy)
                & jnp.isfinite(pre_normalization_norm)
                & jnp.all(jnp.isfinite(local_errors))
                & jnp.all(jnp.isfinite(local_residuals))
                & jnp.all(jnp.isfinite(discarded))
                & jnp.isfinite(canonical_residual)
            )
            if not bool(jax.device_get(finite)):
                status = FiniteTDVPStatus.NONFINITE_ITERATE
                break
            if not bool(jax.device_get(jnp.all(converged))):
                status = FiniteTDVPStatus.LOCAL_EXPONENTIAL_FAILED
                break
            if (
                plan_policy.mode == "real-time"
                and not plan_policy.normalize
                and float(jax.device_get(jnp.abs(norm - norms[0])))
                > plan_policy.norm_tolerance
            ):
                status = FiniteTDVPStatus.NORM_DRIFT
                break
            if (
                plan_policy.mode == "imaginary-time"
                and float(jax.device_get(energy - previous_energy))
                > plan_policy.energy_tolerance
            ):
                status = FiniteTDVPStatus.ENERGY_INCREASE
                break
    diagnostics = FiniteTDVPDiagnostics(
        times,
        norms,
        energies,
        normalization,
        errors,
        residuals,
        local_converged,
        truncation,
        canonical_residuals,
        active,
        prepared.hermiticity_history,
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    checkpoint = FiniteTDVPCheckpoint(
        state,
        jnp.asarray(completed_steps, dtype=jnp.int32),
        jnp.asarray(completed_steps * plan_policy.step_size, dtype=real_dtype),
        prepared.numeric_version,
        prepared.prepared_id,
        prepared.problem.problem_id,
        prepared.plan.plan_id,
    )
    return FiniteTDVPResult(
        state,
        diagnostics,
        checkpoint,
        prepared.problem.problem_id,
        plan_policy.policy_id,
    )


__all__ = [
    "FiniteTDVPAlgorithm",
    "FiniteTDVPCheckpoint",
    "FiniteTDVPCostEstimate",
    "FiniteTDVPDiagnostics",
    "FiniteTDVPMode",
    "FiniteTDVPPlan",
    "FiniteTDVPPolicy",
    "FiniteTDVPProblem",
    "FiniteTDVPResult",
    "FiniteTDVPStatus",
    "PreparedFiniteTDVP",
    "plan_finite_tdvp",
    "prepare_finite_tdvp",
    "refresh_finite_tdvp",
    "solve_finite_tdvp",
]
