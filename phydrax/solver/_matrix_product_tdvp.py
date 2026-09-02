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
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    matrix_exponential_action,
    MatrixFunctionPolicy,
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


MatrixProductTDVPMode: TypeAlias = Literal["real-time", "imaginary-time"]


class MatrixProductTDVPStatus(IntEnum):
    SUCCESS = 0
    INVALID_HAMILTONIAN = 1
    LOCAL_EXPONENTIAL_FAILED = 2
    NONFINITE_ITERATE = 3
    NORM_DRIFT = 4
    ENERGY_INCREASE = 5


class MatrixProductTDVPProblem(StrictModule):
    initial_state: MatrixProductState
    hamiltonian: MatrixProductOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_state: MatrixProductState,
        hamiltonian: MatrixProductOperator,
        /,
        *,
        problem_id: str = "matrix-product-tdvp",
    ):
        if not isinstance(initial_state, MatrixProductState):
            raise TypeError("initial_state must be a MatrixProductState.")
        if not isinstance(hamiltonian, MatrixProductOperator):
            raise TypeError("hamiltonian must be a MatrixProductOperator.")
        if hamiltonian.output_dimensions != hamiltonian.input_dimensions:
            raise ValueError("TDVP requires a square MPO.")
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


class MatrixProductTDVPPolicy(StrictModule):
    mode: MatrixProductTDVPMode = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    norm_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    integrator: MatrixFunctionPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: MatrixProductTDVPMode,
        /,
        *,
        step_size: float,
        steps: int,
        normalize: bool = False,
        norm_tolerance: float = 1e-7,
        energy_tolerance: float = 1e-7,
        hermiticity_tolerance: float = 1e-9,
        integrator: MatrixFunctionPolicy | None = None,
    ):
        if mode not in ("real-time", "imaginary-time"):
            raise ValueError("Unknown matrix-product TDVP mode.")
        step = float(step_size)
        count = int(steps)
        tolerances = (
            float(norm_tolerance),
            float(energy_tolerance),
            float(hermiticity_tolerance),
        )
        if not isfinite(step) or step <= 0.0 or count < 0:
            raise ValueError("TDVP step size must be positive and steps nonnegative.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("TDVP tolerances must be finite and nonnegative.")
        selected = MatrixFunctionPolicy("lanczos") if integrator is None else integrator
        if not isinstance(selected, MatrixFunctionPolicy):
            raise TypeError("integrator must be MatrixFunctionPolicy or None.")
        self.mode = mode
        self.step_size = step
        self.steps = count
        self.normalize = bool(normalize)
        self.norm_tolerance = tolerances[0]
        self.energy_tolerance = tolerances[1]
        self.hermiticity_tolerance = tolerances[2]
        self.integrator = selected
        self.policy_id = canonical_fingerprint(
            {
                "kind": "matrix-product-tdvp-policy",
                "mode": mode,
                "step_size": step,
                "steps": count,
                "normalize": bool(normalize),
                "norm_tolerance": tolerances[0],
                "energy_tolerance": tolerances[1],
                "hermiticity_tolerance": tolerances[2],
                "integrator": selected.method,
                "integrator_dimension": selected.max_dimension,
            }
        )


class MatrixProductTDVPDiagnostics(StrictModule):
    time_history: Array
    norm_history: Array
    energy_history: Array
    local_error_history: Array
    local_residual_history: Array
    local_converged_history: Array
    canonical_residual_history: Array
    active_steps: Array
    hermiticity_residual: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(MatrixProductTDVPStatus.SUCCESS)


class MatrixProductTDVPResult(StrictModule):
    final_state: MatrixProductState
    diagnostics: MatrixProductTDVPDiagnostics
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.diagnostics.successful


class _SiteEffectiveAction(StrictModule):
    left_environment: Array
    operator_tensor: Array
    right_environment: Array

    def __call__(self, vector: Array, /) -> Array:
        return oe.contract(
            "abc,bpqe,def,cqf->apd",
            self.left_environment,
            self.operator_tensor,
            self.right_environment,
            vector,
        )


class _BondEffectiveAction(StrictModule):
    left_environment: Array
    right_environment: Array

    def __call__(self, vector: Array, /) -> Array:
        return oe.contract(
            "abc,dbf,cf->ad",
            self.left_environment,
            self.right_environment,
            vector,
        )


def _self_adjoint_operator(action, shape, dtype, identifier):
    space = ArraySpace(shape, dtype=dtype)
    return FunctionLinearOperator(
        action,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "verified"},
        ),
        operator_id=identifier,
    )


def _evolve(action, vector, scale, policy, identifier):
    operator = _self_adjoint_operator(action, vector.shape, vector.dtype, identifier)
    return matrix_exponential_action(
        operator,
        vector,
        scale,
        policy=policy.integrator,
    )


def _site_action(left, operator, right):
    return _SiteEffectiveAction(left, operator, right)


def _bond_action(left, right):
    return _BondEffectiveAction(left, right)


def _step_scale(policy: MatrixProductTDVPPolicy, dtype):
    step = jnp.asarray(policy.step_size, dtype=jnp.dtype(dtype).type(0).real.dtype)
    return -1j * step if policy.mode == "real-time" else -step


def _tdvp_step(state, hamiltonian, policy):
    precision = state.precision
    base_scale = _step_scale(policy, state.tensors[0].dtype)
    errors = []
    residuals = []
    converged = []

    left_envs, right_envs = build_mps_mpo_environments(state, hamiltonian, state)
    left_values = list(left_envs)
    tensors = list(state.tensors)
    for site in range(state.site_count - 1):
        center = precision.factorization(tensors[site])
        evolved = _evolve(
            _site_action(
                precision.accumulation(left_values[site]),
                precision.accumulation(hamiltonian.tensors[site]),
                precision.accumulation(right_envs[site + 1]),
            ),
            center,
            0.5 * base_scale,
            policy,
            f"tdvp-site-{site}",
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
            _bond_action(
                precision.accumulation(left_values[site + 1]),
                precision.accumulation(right_envs[site + 1]),
            ),
            bond,
            -0.5 * base_scale,
            policy,
            f"tdvp-bond-{site}",
        )
        errors.append(bond_evolved.error_estimate)
        residuals.append(bond_evolved.residual_estimate)
        converged.append(bond_evolved.converged)
        tensors[site] = precision.storage(left_core)
        tensors[site + 1] = precision.storage(
            oe.contract(
                "ab,bpr->apr",
                precision.contraction(bond_evolved.value),
                precision.contraction(tensors[site + 1]),
            )
        )

    last = state.site_count - 1
    evolved = _evolve(
        _site_action(
            precision.accumulation(left_values[last]),
            precision.accumulation(hamiltonian.tensors[last]),
            precision.accumulation(right_envs[last + 1]),
        ),
        precision.factorization(tensors[last]),
        0.5 * base_scale,
        policy,
        f"tdvp-site-{last}",
    )
    tensors[last] = precision.storage(evolved.value)
    errors.append(evolved.error_estimate)
    residuals.append(evolved.residual_estimate)
    converged.append(evolved.converged)
    state = MatrixProductState(tuple(tensors), precision=precision)

    left_envs, right_envs = build_mps_mpo_environments(state, hamiltonian, state)
    right_values = list(right_envs)
    tensors = list(state.tensors)
    for site in range(state.site_count - 1, 0, -1):
        center = precision.factorization(tensors[site])
        evolved = _evolve(
            _site_action(
                precision.accumulation(left_envs[site]),
                precision.accumulation(hamiltonian.tensors[site]),
                precision.accumulation(right_values[site + 1]),
            ),
            center,
            0.5 * base_scale,
            policy,
            f"tdvp-site-{site}",
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
            _bond_action(
                precision.accumulation(left_envs[site]),
                precision.accumulation(right_values[site]),
            ),
            bond,
            -0.5 * base_scale,
            policy,
            f"tdvp-bond-{site - 1}",
        )
        errors.append(bond_evolved.error_estimate)
        residuals.append(bond_evolved.residual_estimate)
        converged.append(bond_evolved.converged)
        tensors[site] = precision.storage(right_core)
        tensors[site - 1] = precision.storage(
            oe.contract(
                "lpa,ab->lpb",
                precision.contraction(tensors[site - 1]),
                precision.contraction(bond_evolved.value),
            )
        )

    evolved = _evolve(
        _site_action(
            precision.accumulation(left_envs[0]),
            precision.accumulation(hamiltonian.tensors[0]),
            precision.accumulation(right_values[1]),
        ),
        precision.factorization(tensors[0]),
        0.5 * base_scale,
        policy,
        "tdvp-site-0",
    )
    tensors[0] = precision.storage(evolved.value)
    errors.append(evolved.error_estimate)
    residuals.append(evolved.residual_estimate)
    converged.append(evolved.converged)
    return (
        MatrixProductState(tuple(tensors), precision=precision),
        jnp.stack(errors),
        jnp.stack(residuals),
        jnp.stack(converged),
    )


def solve_matrix_product_tdvp(
    problem: MatrixProductTDVPProblem,
    policy: MatrixProductTDVPPolicy,
    /,
) -> MatrixProductTDVPResult:
    if not isinstance(problem, MatrixProductTDVPProblem):
        raise TypeError("problem must be MatrixProductTDVPProblem.")
    if not isinstance(policy, MatrixProductTDVPPolicy):
        raise TypeError("policy must be MatrixProductTDVPPolicy.")
    state, _ = canonicalize_mps(problem.initial_state, center=0, normalize=True)
    if policy.mode == "real-time" and not jnp.issubdtype(
        state.tensors[0].dtype, jnp.complexfloating
    ):
        raise TypeError("Real-time matrix-product TDVP requires complex MPS storage.")
    hermiticity = mpo_hermiticity_residual(problem.hamiltonian)
    count = policy.steps
    local_count = 4 * state.site_count - 2
    real_dtype = state.tensors[0].real.dtype
    times = policy.step_size * jnp.arange(count + 1, dtype=real_dtype)
    norms = jnp.full((count + 1,), jnp.nan, dtype=real_dtype)
    energies = jnp.full((count + 1,), jnp.nan, dtype=real_dtype)
    errors = jnp.full((count, local_count), jnp.nan, dtype=real_dtype)
    residuals = jnp.full((count, local_count), jnp.nan, dtype=real_dtype)
    local_converged = jnp.zeros((count, local_count), dtype=bool)
    canonical_residuals = jnp.full((count,), jnp.nan, dtype=real_dtype)
    active = jnp.zeros((count,), dtype=bool)
    norms = norms.at[0].set(state.norm())
    energies = energies.at[0].set(
        jnp.real(mps_mpo_expectation(state, problem.hamiltonian))
    )
    status = MatrixProductTDVPStatus.SUCCESS

    if not bool(
        jax.device_get(
            jnp.isfinite(hermiticity) & (hermiticity <= policy.hermiticity_tolerance)
        )
    ):
        status = MatrixProductTDVPStatus.INVALID_HAMILTONIAN
    else:
        for step in range(count):
            state, local_errors, local_residuals, converged = _tdvp_step(
                state, problem.hamiltonian, policy
            )
            if policy.normalize:
                state = state.normalized()
            state, canonical = canonicalize_mps(state, center=0, normalize=False)
            norm = state.norm()
            energy = jnp.real(mps_mpo_expectation(state, problem.hamiltonian))
            canonical_residual = jnp.maximum(
                jnp.max(canonical.left_residuals),
                jnp.max(canonical.right_residuals),
            )
            errors = errors.at[step].set(local_errors)
            residuals = residuals.at[step].set(local_residuals)
            local_converged = local_converged.at[step].set(converged)
            canonical_residuals = canonical_residuals.at[step].set(canonical_residual)
            norms = norms.at[step + 1].set(norm)
            energies = energies.at[step + 1].set(energy)
            active = active.at[step].set(True)

            finite = (
                jnp.isfinite(norm)
                & jnp.isfinite(energy)
                & jnp.all(jnp.isfinite(local_errors))
                & jnp.all(jnp.isfinite(local_residuals))
                & jnp.isfinite(canonical_residual)
            )
            if not bool(jax.device_get(finite)):
                status = MatrixProductTDVPStatus.NONFINITE_ITERATE
                break
            if not bool(jax.device_get(jnp.all(converged))):
                status = MatrixProductTDVPStatus.LOCAL_EXPONENTIAL_FAILED
                break
            if (
                policy.mode == "real-time"
                and not policy.normalize
                and float(jax.device_get(jnp.abs(norm - norms[0])))
                > policy.norm_tolerance
            ):
                status = MatrixProductTDVPStatus.NORM_DRIFT
                break
            if (
                policy.mode == "imaginary-time"
                and float(jax.device_get(energy - energies[step]))
                > policy.energy_tolerance
            ):
                status = MatrixProductTDVPStatus.ENERGY_INCREASE
                break

    diagnostics = MatrixProductTDVPDiagnostics(
        times,
        norms,
        energies,
        errors,
        residuals,
        local_converged,
        canonical_residuals,
        active,
        hermiticity,
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    return MatrixProductTDVPResult(
        state,
        diagnostics,
        problem.problem_id,
        policy.policy_id,
    )


__all__ = [
    "MatrixProductTDVPDiagnostics",
    "MatrixProductTDVPMode",
    "MatrixProductTDVPPolicy",
    "MatrixProductTDVPProblem",
    "MatrixProductTDVPResult",
    "MatrixProductTDVPStatus",
    "solve_matrix_product_tdvp",
]
