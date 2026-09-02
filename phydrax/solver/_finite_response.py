#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import MatrixFunctionPolicy
from ..tensor_network._core import MatrixProductOperator, MatrixProductState
from ..tensor_network._environments import (
    mpo_hermiticity_residual,
    mps_inner,
    mps_mpo_expectation,
    mps_norm_squared,
)
from ..tensor_network._mpo import (
    add_mpo,
    apply_mpo_exact,
    compress_mps,
    scale_mpo,
)
from ._dmrg import (
    FiniteDMRGPolicy,
    FiniteDMRGProblem,
    FiniteDMRGResult,
    solve_finite_dmrg,
)
from ._matrix_product_tdvp import (
    FiniteTDVPPolicy,
    FiniteTDVPProblem,
    solve_finite_tdvp,
)


class FiniteExcitedStateResult(StrictModule):
    dmrg: FiniteDMRGResult
    reference_overlaps: Array
    projector_hermiticity_residuals: Array
    penalty_energy: Array
    successful: Array


def mps_projector_mpo(state: MatrixProductState, /) -> MatrixProductOperator:
    """Build the normalized exact rank-one projector |psi><psi| as an MPO."""
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be MatrixProductState.")
    norm = mps_norm_squared(state)
    norm = eqx.error_if(
        norm,
        ~jnp.isfinite(norm) | (norm <= 0.0),
        "Projector state norm must be positive.",
    )
    tensors = []
    for index, tensor in enumerate(state.tensors):
        combined = oe.contract("apr,bqs->abpqrs", tensor, jnp.conj(tensor))
        value = combined.reshape(
            (
                tensor.shape[0] * tensor.shape[0],
                tensor.shape[1],
                tensor.shape[1],
                tensor.shape[-1] * tensor.shape[-1],
            )
        )
        tensors.append(value / norm if index == 0 else value)
    return MatrixProductOperator(tuple(tensors), precision=state.precision)


def solve_finite_excited_state(
    problem: FiniteDMRGProblem,
    reference_states: Sequence[MatrixProductState],
    penalties: ArrayLike,
    policy: FiniteDMRGPolicy,
    /,
) -> FiniteExcitedStateResult:
    """Target an excited state by adding explicit positive rank-one projectors."""
    references = tuple(reference_states)
    penalty_values = jnp.asarray(penalties)
    if not references or any(
        not isinstance(state, MatrixProductState) for state in references
    ):
        raise ValueError("reference_states must contain at least one MPS.")
    if penalty_values.shape != (len(references),):
        raise ValueError("penalties must have one value per reference state.")
    if not bool(jnp.all(jnp.isfinite(penalty_values) & (penalty_values > 0.0))):
        raise ValueError("Excited-state penalties must be finite and positive.")
    penalized = problem.hamiltonian
    hermiticity = []
    for index, reference in enumerate(references):
        if reference.physical_dimensions != problem.initial_state.physical_dimensions:
            raise ValueError("Reference-state dimensions must match the DMRG problem.")
        projector = mps_projector_mpo(reference)
        hermiticity.append(mpo_hermiticity_residual(projector))
        penalized = add_mpo(penalized, scale_mpo(projector, penalty_values[index]))
    targeted = FiniteDMRGProblem(
        problem.initial_state,
        penalized,
        problem_id=f"{problem.problem_id}-projector-target",
    )
    result = solve_finite_dmrg(targeted, policy)
    target_norm = mps_norm_squared(result.best_state)
    overlaps = jnp.stack(
        [
            jnp.square(jnp.abs(mps_inner(reference, result.best_state)))
            / (mps_norm_squared(reference) * target_norm)
            for reference in references
        ]
    )
    penalty_energy = oe.contract("i,i->", penalty_values, overlaps)
    successful = result.successful & jnp.all(jnp.isfinite(overlaps))
    return FiniteExcitedStateResult(
        result,
        overlaps,
        jnp.stack(hermiticity),
        penalty_energy,
        successful,
    )


class FiniteResponseStatus(IntEnum):
    SUCCESS = 0
    INVALID_OPERATOR = 1
    EVOLUTION_FAILED = 2
    NONFINITE = 3


class FiniteResponseProblem(StrictModule):
    ground_state: MatrixProductState
    hamiltonian: MatrixProductOperator
    excitation: MatrixProductOperator
    frequencies: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        ground_state: MatrixProductState,
        hamiltonian: MatrixProductOperator,
        excitation: MatrixProductOperator,
        frequencies: ArrayLike,
        /,
        *,
        problem_id: str = "finite-time-domain-response",
    ):
        values = jnp.asarray(frequencies)
        if not isinstance(ground_state, MatrixProductState):
            raise TypeError("ground_state must be MatrixProductState.")
        if not isinstance(hamiltonian, MatrixProductOperator) or not isinstance(
            excitation, MatrixProductOperator
        ):
            raise TypeError("hamiltonian and excitation must be MPO values.")
        if values.ndim != 1 or values.size < 1 or not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("frequencies must be a nonempty finite vector.")
        if (
            ground_state.physical_dimensions != hamiltonian.input_dimensions
            or ground_state.physical_dimensions != excitation.input_dimensions
            or hamiltonian.output_dimensions != hamiltonian.input_dimensions
            or excitation.output_dimensions != excitation.input_dimensions
        ):
            raise ValueError("Finite response dimensions must be square and compatible.")
        self.ground_state = ground_state
        self.hamiltonian = hamiltonian
        self.excitation = excitation
        self.frequencies = values
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.problem_id = identifier


class FiniteResponsePolicy(StrictModule):
    step_size: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    tdvp_algorithm: str = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    maximum_state_elements: int = eqx.field(static=True)
    integrator: MatrixFunctionPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        step_size: float,
        steps: int,
        maximum_bond_dimension: int,
        tdvp_algorithm: str = "two-site",
        damping: float = 0.0,
        hermiticity_tolerance: float = 1e-9,
        maximum_history_elements: int = 10_000_000,
        maximum_state_elements: int = 10_000_000,
        integrator: MatrixFunctionPolicy | None = None,
    ):
        step = float(step_size)
        count = int(steps)
        bond = int(maximum_bond_dimension)
        damping_ = float(damping)
        tolerance = float(hermiticity_tolerance)
        history = int(maximum_history_elements)
        state_elements = int(maximum_state_elements)
        if not isfinite(step) or step <= 0.0 or count < 1 or bond < 1:
            raise ValueError("Response step, count, and bond capacity are invalid.")
        if tdvp_algorithm not in ("one-site", "two-site"):
            raise ValueError("tdvp_algorithm must be one-site or two-site.")
        if (
            not isfinite(damping_)
            or damping_ < 0.0
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Response damping and tolerance are invalid.")
        if history < 4 * (count + 1) or state_elements < 1:
            raise ValueError("Finite response history/state capacities are insufficient.")
        selected = MatrixFunctionPolicy("lanczos") if integrator is None else integrator
        if not isinstance(selected, MatrixFunctionPolicy):
            raise TypeError("integrator must be MatrixFunctionPolicy or None.")
        self.step_size = step
        self.steps = count
        self.maximum_bond_dimension = bond
        self.tdvp_algorithm = tdvp_algorithm
        self.damping = damping_
        self.hermiticity_tolerance = tolerance
        self.maximum_state_elements = state_elements
        self.maximum_history_elements = history
        self.integrator = selected
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-response-policy",
                "step": step,
                "steps": count,
                "bond": bond,
                "algorithm": tdvp_algorithm,
                "damping": damping_,
                "tolerance": tolerance,
                "integrator": selected.method,
            }
        )


class FiniteResponseEvidence(StrictModule):
    active_steps: Array
    tdvp_status_history: Array
    truncation_history: Array
    excitation_discarded_weight: Array
    approximate_sum_rule: Array
    sum_rule: Array
    zero_time_correlation: Array
    sum_rule_residual: Array
    fourier_sum: Array
    fourier_sum_rule_residual: Array
    hamiltonian_hermiticity_residual: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(FiniteResponseStatus.SUCCESS)


class FiniteResponseResult(StrictModule):
    times: Array
    correlations: Array
    frequencies: Array
    spectrum: Array
    evidence: FiniteResponseEvidence
    policy_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


def solve_finite_response(
    problem: FiniteResponseProblem,
    policy: FiniteResponsePolicy,
    /,
) -> FiniteResponseResult:
    if not isinstance(problem, FiniteResponseProblem) or not isinstance(
        policy, FiniteResponsePolicy
    ):
        raise TypeError("solve_finite_response requires a finite problem and policy.")
    required = (policy.steps + 1) * (4 + int(problem.frequencies.size))
    if required > policy.maximum_history_elements:
        raise MemoryError("Finite response histories exceed maximum_history_elements.")
    hermiticity = mpo_hermiticity_residual(problem.hamiltonian)
    exact_excited = apply_mpo_exact(problem.excitation, problem.ground_state)
    if (
        sum(int(tensor.size) for tensor in exact_excited.tensors)
        > policy.maximum_state_elements
    ):
        raise MemoryError("Exact response excitation exceeds maximum_state_elements.")
    ground_norm = mps_norm_squared(problem.ground_state)
    sum_rule = mps_norm_squared(exact_excited) / ground_norm
    excited, excitation_evidence = compress_mps(
        exact_excited,
        maximum_bond_dimension=policy.maximum_bond_dimension,
        normalize=False,
    )
    excited_norm = mps_norm_squared(excited)
    approximate_sum_rule = excited_norm / ground_norm
    excited = excited.normalized()
    state = excited
    energy = jnp.real(
        mps_mpo_expectation(problem.ground_state, problem.hamiltonian) / ground_norm
    )
    real_dtype = problem.ground_state.tensors[0].real.dtype
    complex_dtype = jnp.result_type(problem.ground_state.tensors[0], jnp.complex64)
    times = policy.step_size * jnp.arange(policy.steps + 1, dtype=real_dtype)
    correlations = jnp.full((policy.steps + 1,), jnp.nan + 0j, dtype=complex_dtype)
    zero = approximate_sum_rule * mps_inner(excited, excited)
    correlations = correlations.at[0].set(zero)
    active = jnp.zeros((policy.steps,), dtype=bool)
    tdvp_status = jnp.full((policy.steps,), -1, dtype=jnp.int32)
    truncation = jnp.full((policy.steps,), jnp.nan, dtype=real_dtype)
    status = FiniteResponseStatus.SUCCESS
    if not bool(
        jnp.isfinite(hermiticity) & (hermiticity <= policy.hermiticity_tolerance)
    ):
        status = FiniteResponseStatus.INVALID_OPERATOR
    else:
        for step in range(policy.steps):
            tdvp_policy = FiniteTDVPPolicy(
                "real-time",
                step_size=policy.step_size,
                steps=1,
                algorithm=policy.tdvp_algorithm,
                maximum_bond_dimension=policy.maximum_bond_dimension,
                normalize=False,
                integrator=policy.integrator,
            )
            evolved = solve_finite_tdvp(
                FiniteTDVPProblem(
                    state,
                    problem.hamiltonian,
                    problem_id=f"{problem.problem_id}-step-{step}",
                ),
                tdvp_policy,
            )
            state = evolved.final_state
            tdvp_status = tdvp_status.at[step].set(evolved.diagnostics.status)
            truncation = truncation.at[step].set(
                jnp.nansum(evolved.diagnostics.truncation_history[0])
            )
            active = active.at[step].set(True)
            phase = jnp.exp(1j * energy * times[step + 1])
            correlations = correlations.at[step + 1].set(
                approximate_sum_rule * mps_inner(excited, state) * phase
            )
            if not bool(evolved.successful):
                status = FiniteResponseStatus.EVOLUTION_FAILED
                break
    window = jnp.exp(-policy.damping * times)
    phases = jnp.exp(1j * problem.frequencies[:, None] * times[None, :])
    weights = (
        jnp.ones((policy.steps + 1,), dtype=real_dtype).at[0].set(0.5).at[-1].set(0.5)
    )
    spectrum = policy.step_size * oe.contract(
        "wt,t,t,t->w", phases, window, weights, correlations
    )
    if problem.frequencies.size > 1:
        differences = jnp.diff(problem.frequencies)
        uniform_spacing = jnp.max(jnp.abs(differences - differences[0]))
        spacing = differences[0]
        fourier_sum = jnp.real(jnp.sum(spectrum) * spacing / (2.0 * pi))
        fourier_residual = jnp.where(
            uniform_spacing <= 1e-8 * jnp.maximum(jnp.abs(spacing), 1.0),
            jnp.abs(fourier_sum - sum_rule),
            jnp.asarray(jnp.nan, dtype=real_dtype),
        )
    else:
        fourier_sum = jnp.asarray(jnp.nan, dtype=real_dtype)
        fourier_residual = jnp.asarray(jnp.nan, dtype=real_dtype)
    sum_rule_residual = jnp.abs(zero - sum_rule)
    correlation_mask = jnp.concatenate((jnp.asarray([True]), active))
    finite = jnp.all(
        jnp.where(correlation_mask, jnp.isfinite(correlations), True)
    ) & jnp.all(jnp.isfinite(spectrum))
    if status == FiniteResponseStatus.SUCCESS and not bool(finite):
        status = FiniteResponseStatus.NONFINITE
    evidence = FiniteResponseEvidence(
        active,
        tdvp_status,
        truncation,
        excitation_evidence.accumulated_discarded_weight,
        approximate_sum_rule,
        sum_rule,
        zero,
        sum_rule_residual,
        fourier_sum,
        fourier_residual,
        hermiticity,
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    return FiniteResponseResult(
        times,
        correlations,
        problem.frequencies,
        spectrum,
        evidence,
        policy.policy_id,
        problem.problem_id,
    )


__all__ = [
    "FiniteExcitedStateResult",
    "FiniteResponseEvidence",
    "FiniteResponsePolicy",
    "FiniteResponseProblem",
    "FiniteResponseResult",
    "FiniteResponseStatus",
    "mps_projector_mpo",
    "solve_finite_excited_state",
    "solve_finite_response",
]
