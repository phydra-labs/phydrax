#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bounds import Bounds
from ..._strict import StrictModule
from ...optim import (
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    LinearProgram,
    solve_linear_program,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._network import StoichiometricNetwork


class FluxStatus(IntEnum):
    """Portable flux-analysis status aligned with native convex-program evidence."""

    OPTIMAL = 0
    ITERATION_LIMIT = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    NONFINITE_INPUT = 4
    NONFINITE_OUTPUT = 5
    NUMERICAL_FAILURE = 6
    BACKEND_FAILED = 7
    INVALID_PROBLEM = 8


class FluxCapacityError(ValueError):
    """Raised before FVA when the requested complete solve family exceeds capacity."""


class AlternateOptimumEvidence(StrictModule):
    """Complete per-reaction optimal-face ranges used to detect alternate optima."""

    minimum_fluxes: Array
    maximum_fluxes: Array
    minimum_statuses: Array
    maximum_statuses: Array
    maximum_span: Array
    alternate_optimum: Array
    available: Array
    objective_tolerance: Array
    complete: bool = eqx.field(static=True)


class FluxBalanceEvidence(StrictModule):
    """Native primal/dual/KKT result plus optimal-face evidence."""

    native_result: ConvexProgramResult
    mass_balance_residual: Array
    objective_coefficients: Array
    alternate: AlternateOptimumEvidence
    exact_model: bool = eqx.field(static=True)
    solver_runtime: str = eqx.field(static=True)


class FluxBalanceResult(StrictModule):
    """One FBA optimum or an auditable infeasible/unbounded terminal result."""

    valid: Array
    status: Array
    fluxes: Array
    objective_value: Array
    evidence: FluxBalanceEvidence
    method_contract: BioinformaticsMethodContract
    network_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(FluxStatus.OPTIMAL))


class FluxVariabilityEvidence(StrictModule):
    """All native endpoint solves and retained-objective semantics for complete FVA."""

    minimum_results: tuple[ConvexProgramResult, ...]
    maximum_results: tuple[ConvexProgramResult, ...]
    minimum_statuses: Array
    maximum_statuses: Array
    retained_objective: Array
    objective_fraction: float = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    solver_runtime: str = eqx.field(static=True)


class FluxVariabilityResult(StrictModule):
    """Complete reaction-wise minimum and maximum fluxes on an objective face."""

    valid: Array
    status: Array
    minimum_fluxes: Array
    maximum_fluxes: Array
    evidence: FluxVariabilityEvidence
    method_contract: BioinformaticsMethodContract
    network_id: str = eqx.field(static=True)

    @property
    def spans(self) -> Array:
        return self.maximum_fluxes - self.minimum_fluxes


def _flux_contract(
    policy: ConvexSolvePolicy, method_name: str, /
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.ITERATIVE_TOLERANCE,
        DifferentiationKind.IMPLICIT,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "The LP optimum and feasibility certificates are conditioned by the "
            "stoichiometric equality matrix and active variable bounds."
        ),
        truncation_statement="No feasible reactions or auxiliary extrema are truncated.",
        capacity_semantics=(
            "Native optimization preflights factorization/workspace resources; FVA "
            "also preflights its complete two-solves-per-reaction family."
        ),
        assumptions=(
            "Internal metabolites satisfy steady-state mass balance.",
            "Reaction bounds and the linear objective define the flux state.",
        ),
        nondifferentiable_outputs=(
            "status",
            "valid",
            "alternate_optimum",
            "infeasibility and recession certificates",
        ),
        absolute_tolerance=policy.termination.absolute,
        relative_tolerance=policy.termination.relative,
    )


def _policy(policy: ConvexSolvePolicy | None, /) -> ConvexSolvePolicy:
    if policy is None:
        return ConvexSolvePolicy()
    if not isinstance(policy, ConvexSolvePolicy):
        raise TypeError("policy must be a ConvexSolvePolicy or None.")
    return policy


def _flux_status(result: ConvexProgramResult, /) -> Array:
    """Reconcile solver termination with independently audited LP ray certificates."""

    return jnp.where(
        result.certificate.dual_ray_valid,
        int(FluxStatus.INFEASIBLE),
        jnp.where(
            result.certificate.primal_ray_valid,
            int(FluxStatus.UNBOUNDED),
            result.status,
        ),
    ).astype(jnp.int32)


def _objective(network: StoichiometricNetwork, objective: ArrayLike | None, /) -> Array:
    values = (
        network.objective_coefficients
        if objective is None
        else jnp.asarray(objective, dtype=network.stoichiometric_matrix.dtype)
    )
    if values.shape != (network.num_reactions,):
        raise ValueError("objective must have one coefficient per reaction.")
    return eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        "Flux objective coefficients must be finite.",
    )


def _bounds(
    network: StoichiometricNetwork,
    active_genes: Iterable[str] | None,
    /,
) -> tuple[Array, Array]:
    return (
        (network.lower_bounds, network.upper_bounds)
        if active_genes is None
        else network.bounds_for_active_genes(active_genes)
    )


def _program(
    network: StoichiometricNetwork,
    linear: Array,
    lower: Array,
    upper: Array,
    /,
    *,
    objective_coefficients: Array | None = None,
    retained_objective: Array | None = None,
    objective_tolerance: float = 0.0,
    problem_id: str,
) -> LinearProgram:
    equality_matrix = network.steady_state_matrix
    equality_rhs = jnp.zeros((equality_matrix.shape[0],), dtype=linear.dtype)
    if objective_coefficients is None:
        inequality_matrix = None
        inequality_rhs = None
    else:
        target = jnp.asarray(retained_objective, dtype=linear.dtype)
        tolerance = jnp.asarray(objective_tolerance, dtype=linear.dtype)
        if network.objective_sense == "maximize":
            inequality_matrix = -objective_coefficients[None, :]
            inequality_rhs = -(target - tolerance)[None]
        else:
            inequality_matrix = objective_coefficients[None, :]
            inequality_rhs = (target + tolerance)[None]
    return LinearProgram(
        linear,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_rhs,
        bounds=Bounds(lower, upper),
        problem_id=problem_id,
    )


def _audit_primal_recession(
    network: StoichiometricNetwork,
    linear: Array,
    lower: Array,
    upper: Array,
    policy: ConvexSolvePolicy,
    native: ConvexProgramResult,
    /,
) -> ConvexProgramResult:
    """Recover a finite recession certificate when the primary iterate is non-finite."""

    direction_lower = jnp.where(jnp.isneginf(lower), -1.0, 0.0).astype(linear.dtype)
    direction_upper = jnp.where(jnp.isposinf(upper), 1.0, 0.0).astype(linear.dtype)
    direction_program = _program(
        network,
        linear,
        direction_lower,
        direction_upper,
        problem_id=f"{network.network_id}:recession-audit",
    )
    direction_result = solve_linear_program(direction_program, policy=policy)
    direction = direction_result.primal
    objective = jnp.sum(linear * direction)
    residual = jnp.max(
        jnp.abs(network.steady_state_matrix @ direction),
        initial=0.0,
    )
    tolerance = jnp.asarray(
        max(policy.termination.absolute, policy.termination.relative, 1.0e-8),
        dtype=linear.dtype,
    )
    valid = (
        direction_result.valid
        & jnp.all(jnp.isfinite(direction))
        & (residual <= tolerance)
        & (objective < -tolerance)
    )
    certificate = eqx.tree_at(
        lambda value: (
            value.primal_ray,
            value.primal_ray_residual_norm,
            value.primal_ray_objective,
            value.primal_ray_valid,
        ),
        native.certificate,
        (direction, residual, objective, valid),
    )
    return eqx.tree_at(
        lambda value: (value.status, value.certificate),
        native,
        (
            jnp.where(
                valid,
                int(ConvexProgramStatus.DUAL_INFEASIBLE),
                native.status,
            ).astype(jnp.int32),
            certificate,
        ),
    )


def _solve_endpoint_family(
    network: StoichiometricNetwork,
    lower: Array,
    upper: Array,
    objective: Array,
    retained_objective: Array,
    policy: ConvexSolvePolicy,
    /,
    *,
    objective_tolerance: float,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    tuple[ConvexProgramResult, ...],
    tuple[ConvexProgramResult, ...],
]:
    minima = []
    maxima = []
    minimum_statuses = []
    maximum_statuses = []
    minimum_results = []
    maximum_results = []
    basis = jnp.eye(network.num_reactions, dtype=objective.dtype)
    for index in range(network.num_reactions):
        minimum_program = _program(
            network,
            basis[index],
            lower,
            upper,
            objective_coefficients=objective,
            retained_objective=retained_objective,
            objective_tolerance=objective_tolerance,
            problem_id=f"{network.network_id}:fva:min:{index}",
        )
        maximum_program = _program(
            network,
            -basis[index],
            lower,
            upper,
            objective_coefficients=objective,
            retained_objective=retained_objective,
            objective_tolerance=objective_tolerance,
            problem_id=f"{network.network_id}:fva:max:{index}",
        )
        minimum_result = solve_linear_program(minimum_program, policy=policy)
        maximum_result = solve_linear_program(maximum_program, policy=policy)
        minimum_results.append(minimum_result)
        maximum_results.append(maximum_result)
        minimum_status = _flux_status(minimum_result)
        maximum_status = _flux_status(maximum_result)
        minimum_statuses.append(minimum_status)
        maximum_statuses.append(maximum_status)
        minimum_value = jnp.where(
            minimum_status == int(FluxStatus.OPTIMAL),
            minimum_result.primal[index],
            jnp.where(
                minimum_status == int(FluxStatus.UNBOUNDED),
                -jnp.inf,
                jnp.nan,
            ),
        )
        maximum_value = jnp.where(
            maximum_status == int(FluxStatus.OPTIMAL),
            maximum_result.primal[index],
            jnp.where(
                maximum_status == int(FluxStatus.UNBOUNDED),
                jnp.inf,
                jnp.nan,
            ),
        )
        minima.append(minimum_value)
        maxima.append(maximum_value)
    return (
        jnp.stack(tuple(minima)),
        jnp.stack(tuple(maxima)),
        jnp.stack(tuple(minimum_statuses)).astype(jnp.int32),
        jnp.stack(tuple(maximum_statuses)).astype(jnp.int32),
        tuple(minimum_results),
        tuple(maximum_results),
    )


def flux_balance_analysis(
    network: StoichiometricNetwork,
    /,
    *,
    objective: ArrayLike | None = None,
    active_genes: Iterable[str] | None = None,
    policy: ConvexSolvePolicy | None = None,
    detect_alternate_optima: bool = True,
    alternate_tolerance: float = 1.0e-6,
    max_auxiliary_solves: int = 8192,
) -> FluxBalanceResult:
    """Solve FBA through the native LP lifecycle and audit the full optimal face."""

    if not isinstance(network, StoichiometricNetwork):
        raise TypeError("network must be a StoichiometricNetwork.")
    policy_ = _policy(policy)
    objective_ = _objective(network, objective)
    lower, upper = _bounds(network, active_genes)
    tolerance = float(alternate_tolerance)
    capacity = int(max_auxiliary_solves)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("alternate_tolerance must be finite and non-negative.")
    if capacity < 0:
        raise ValueError("max_auxiliary_solves must be non-negative.")
    if detect_alternate_optima and 2 * network.num_reactions > capacity:
        raise FluxCapacityError(
            "Complete alternate-optimum analysis requires "
            f"{2 * network.num_reactions} solves; capacity is {capacity}."
        )
    linear = -objective_ if network.objective_sense == "maximize" else objective_
    program = _program(
        network,
        linear,
        lower,
        upper,
        problem_id=f"{network.network_id}:fba",
    )
    native = solve_linear_program(program, policy=policy_)
    if bool(np.asarray(~native.valid & jnp.any(jnp.isinf(lower) | jnp.isinf(upper)))):
        native = _audit_primal_recession(
            network,
            linear,
            lower,
            upper,
            policy_,
            native,
        )
    status = _flux_status(native)
    valid = native.valid & (status == int(FluxStatus.OPTIMAL))
    biological_objective = jnp.sum(objective_ * native.primal)
    objective_value = jnp.where(
        status == int(FluxStatus.OPTIMAL),
        biological_objective,
        jnp.where(
            status == int(FluxStatus.UNBOUNDED),
            jnp.inf if network.objective_sense == "maximize" else -jnp.inf,
            jnp.nan,
        ),
    )
    solved = bool(np.asarray(valid))
    if detect_alternate_optima and solved:
        (
            minimum_fluxes,
            maximum_fluxes,
            minimum_statuses,
            maximum_statuses,
            _,
            _,
        ) = _solve_endpoint_family(
            network,
            lower,
            upper,
            objective_,
            objective_value,
            policy_,
            objective_tolerance=max(
                tolerance,
                policy_.termination.absolute
                + policy_.termination.relative * abs(float(np.asarray(objective_value))),
            ),
        )
        endpoints_valid = jnp.all(
            (minimum_statuses == int(FluxStatus.OPTIMAL))
            & (maximum_statuses == int(FluxStatus.OPTIMAL))
        )
        span = jnp.max(maximum_fluxes - minimum_fluxes, initial=0.0)
        alternate = endpoints_valid & (span > tolerance)
        available = endpoints_valid
    else:
        minimum_fluxes = jnp.full((network.num_reactions,), jnp.nan)
        maximum_fluxes = jnp.full((network.num_reactions,), jnp.nan)
        minimum_statuses = jnp.full((network.num_reactions,), status, dtype=jnp.int32)
        maximum_statuses = minimum_statuses
        span = jnp.asarray(jnp.nan)
        alternate = jnp.asarray(False)
        available = jnp.asarray(False)
    alternate_evidence = AlternateOptimumEvidence(
        minimum_fluxes=minimum_fluxes,
        maximum_fluxes=maximum_fluxes,
        minimum_statuses=minimum_statuses,
        maximum_statuses=maximum_statuses,
        maximum_span=span,
        alternate_optimum=alternate,
        available=available,
        objective_tolerance=jnp.asarray(tolerance),
        complete=bool(detect_alternate_optima and solved),
    )
    evidence = FluxBalanceEvidence(
        native_result=native,
        mass_balance_residual=network.steady_state_matrix @ native.primal,
        objective_coefficients=objective_,
        alternate=alternate_evidence,
        exact_model=True,
        solver_runtime="phydrax.optim",
    )
    return FluxBalanceResult(
        valid=valid,
        status=status,
        fluxes=native.primal,
        objective_value=objective_value,
        evidence=evidence,
        method_contract=_flux_contract(policy_, "flux-balance-analysis"),
        network_id=network.network_id,
    )


def flux_variability_analysis(
    network: StoichiometricNetwork,
    /,
    *,
    objective: ArrayLike | None = None,
    objective_fraction: float = 1.0,
    active_genes: Iterable[str] | None = None,
    policy: ConvexSolvePolicy | None = None,
    objective_tolerance: float = 1.0e-7,
    max_auxiliary_solves: int = 8192,
) -> FluxVariabilityResult:
    """Compute complete reaction-wise extrema while retaining an objective fraction."""

    if not isinstance(network, StoichiometricNetwork):
        raise TypeError("network must be a StoichiometricNetwork.")
    fraction = float(objective_fraction)
    tolerance = float(objective_tolerance)
    capacity = int(max_auxiliary_solves)
    if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("objective_fraction must lie in [0, 1].")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and non-negative.")
    required = 2 * network.num_reactions
    if capacity < required:
        raise FluxCapacityError(
            f"Complete FVA requires {required} endpoint solves; capacity is {capacity}."
        )
    policy_ = _policy(policy)
    objective_ = _objective(network, objective)
    lower, upper = _bounds(network, active_genes)
    linear = -objective_ if network.objective_sense == "maximize" else objective_
    optimum_result = solve_linear_program(
        _program(
            network,
            linear,
            lower,
            upper,
            problem_id=f"{network.network_id}:fva:reference",
        ),
        policy=policy_,
    )
    optimum_status = _flux_status(optimum_result)
    optimum_valid = bool(
        np.asarray(optimum_result.valid & (optimum_status == int(FluxStatus.OPTIMAL)))
    )
    optimum = jnp.sum(objective_ * optimum_result.primal)
    retained = (
        optimum - (1.0 - fraction) * jnp.abs(optimum)
        if network.objective_sense == "maximize"
        else optimum + (1.0 - fraction) * jnp.abs(optimum)
    )
    if optimum_valid:
        (
            minima,
            maxima,
            minimum_statuses,
            maximum_statuses,
            minimum_results,
            maximum_results,
        ) = _solve_endpoint_family(
            network,
            lower,
            upper,
            objective_,
            retained,
            policy_,
            objective_tolerance=tolerance,
        )
        endpoint_valid = (minimum_statuses == int(FluxStatus.OPTIMAL)) & (
            maximum_statuses == int(FluxStatus.OPTIMAL)
        )
        valid = jnp.all(endpoint_valid)
        status = jnp.where(
            valid,
            int(FluxStatus.OPTIMAL),
            jnp.max(
                jnp.where(
                    endpoint_valid,
                    0,
                    jnp.maximum(minimum_statuses, maximum_statuses),
                )
            ),
        ).astype(jnp.int32)
    else:
        minima = jnp.full((network.num_reactions,), jnp.nan)
        maxima = jnp.full((network.num_reactions,), jnp.nan)
        minimum_statuses = jnp.full(
            (network.num_reactions,), optimum_status, dtype=jnp.int32
        )
        maximum_statuses = minimum_statuses
        minimum_results = ()
        maximum_results = ()
        valid = jnp.asarray(False)
        status = optimum_status
    evidence = FluxVariabilityEvidence(
        minimum_results=minimum_results,
        maximum_results=maximum_results,
        minimum_statuses=minimum_statuses,
        maximum_statuses=maximum_statuses,
        retained_objective=retained,
        objective_fraction=fraction,
        complete=optimum_valid,
        solver_runtime="phydrax.optim",
    )
    return FluxVariabilityResult(
        valid=valid,
        status=status,
        minimum_fluxes=minima,
        maximum_fluxes=maxima,
        evidence=evidence,
        method_contract=_flux_contract(policy_, "flux-variability-analysis"),
        network_id=network.network_id,
    )


__all__ = [
    "flux_balance_analysis",
    "flux_variability_analysis",
    "AlternateOptimumEvidence",
    "FluxBalanceEvidence",
    "FluxBalanceResult",
    "FluxCapacityError",
    "FluxStatus",
    "FluxVariabilityEvidence",
    "FluxVariabilityResult",
]
