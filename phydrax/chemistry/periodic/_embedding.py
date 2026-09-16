#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Narrow normal orthonormal one-orbital finite-bath single-site DMFT."""

from __future__ import annotations

from math import isfinite
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.dlr import matsubara_frequencies
from ...nonlinear import NonlinearSystemProblem
from ...operators.quantum._impurity import ImpurityEnvironment, MatsubaraHybridization
from ...operators.quantum._thermal_green import (
    MatsubaraGreenFunction,
    MatsubaraSelfEnergy,
)
from ...solver._impurity import (
    AbstractImpurityProvider,
    AndersonBathFitPlan,
    AndersonBathFitResult,
    EDImpurityPolicy,
    ExactDiagonalizationImpurityProvider,
    fit_causal_anderson_bath,
    ImpuritySolveRequest,
    ImpuritySolveResult,
)


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


class SingleSiteDMFTPlan(StrictModule, NonTrainableState):
    """One normal orthonormal orbital, one local self-energy, finite temperature."""

    lattice_energies: Array
    lattice_weights: Array
    indices: Array
    bath_fit: AndersonBathFitPlan
    impurity_policy: EDImpurityPolicy
    onsite_energy: float = eqx.field(static=True)
    interaction: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    target_density: float = eqx.field(static=True)
    self_energy_mixing: float = eqx.field(static=True)
    chemical_potential_step: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    fixed_point_tolerance: float = eqx.field(static=True)
    density_tolerance: float = eqx.field(static=True)
    bath_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice_energies: ArrayLike,
        lattice_weights: ArrayLike,
        indices: ArrayLike,
        bath_fit: AndersonBathFitPlan,
        /,
        *,
        onsite_energy: float,
        interaction: float,
        beta: float,
        target_density: float,
        impurity_policy: EDImpurityPolicy | None = None,
        self_energy_mixing: float = 0.5,
        chemical_potential_step: float = 0.5,
        maximum_iterations: int = 64,
        fixed_point_tolerance: float = 1e-5,
        density_tolerance: float = 1e-5,
        bath_tolerance: float = 2e-2,
    ):
        energies = jnp.asarray(lattice_energies)
        weights = jnp.asarray(lattice_weights)
        labels = jnp.asarray(indices)
        if energies.ndim != 1 or energies.size == 0 or weights.shape != energies.shape:
            raise ValueError("One lattice weight is required per scalar band energy.")
        if jnp.issubdtype(energies.dtype, jnp.complexfloating) or jnp.issubdtype(
            weights.dtype, jnp.complexfloating
        ):
            raise TypeError(
                "The orthonormal normal profile requires real energies/weights."
            )
        energies_host = np.asarray(energies)
        weights_host = np.asarray(weights)
        if (
            not np.all(np.isfinite(energies_host))
            or not np.all(np.isfinite(weights_host))
            or np.any(weights_host < 0.0)
            or not np.isclose(np.sum(weights_host), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError(
                "Lattice weights must be finite, non-negative, and sum to one."
            )
        if (
            labels.ndim != 1
            or labels.size == 0
            or not jnp.issubdtype(labels.dtype, jnp.integer)
        ):
            raise TypeError("indices must be one nonempty rank-one integer array.")
        if not isinstance(bath_fit, AndersonBathFitPlan):
            raise TypeError("bath_fit must be AndersonBathFitPlan.")
        policy = EDImpurityPolicy() if impurity_policy is None else impurity_policy
        if not isinstance(policy, EDImpurityPolicy):
            raise TypeError("impurity_policy must be EDImpurityPolicy or None.")
        onsite = float(onsite_energy)
        interaction_ = float(interaction)
        density = float(target_density)
        if not isfinite(onsite) or not isfinite(interaction_) or interaction_ < 0.0:
            raise ValueError(
                "DMFT onsite energy and non-negative interaction must be finite."
            )
        if not isfinite(density) or not 0.0 <= density <= 2.0:
            raise ValueError("target_density must lie between zero and two.")
        mixing = float(self_energy_mixing)
        mu_step = _positive(chemical_potential_step, "chemical_potential_step")
        if not isfinite(mixing) or not 0.0 < mixing <= 1.0:
            raise ValueError("self_energy_mixing must lie in (0, 1].")
        beta_ = _positive(beta, "beta")
        iterations = _positive_int(maximum_iterations, "maximum_iterations")
        fixed = _positive(fixed_point_tolerance, "fixed_point_tolerance")
        density_tolerance_ = _positive(density_tolerance, "density_tolerance")
        bath_tolerance_ = _positive(bath_tolerance, "bath_tolerance")
        self.lattice_energies = energies
        self.lattice_weights = weights
        self.indices = labels.astype(jnp.int32)
        self.bath_fit = bath_fit
        self.impurity_policy = policy
        self.onsite_energy = onsite
        self.interaction = interaction_
        self.beta = beta_
        self.target_density = density
        self.self_energy_mixing = mixing
        self.chemical_potential_step = mu_step
        self.maximum_iterations = iterations
        self.fixed_point_tolerance = fixed
        self.density_tolerance = density_tolerance_
        self.bath_tolerance = bath_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "normal-orthonormal-one-orbital-finite-bath-dmft",
                "lattice": array_tree_fingerprint(
                    {"energies": energies, "weights": weights}
                ),
                "indices": array_tree_fingerprint(labels),
                "bath_fit": bath_fit.plan_id,
                "onsite": onsite,
                "interaction": interaction_,
                "beta": beta_,
                "target_density": density,
                "iterations": iterations,
                "tolerances": (fixed, density_tolerance_, bath_tolerance_),
            }
        )


class DMFTState(StrictModule):
    self_energy: Array
    chemical_potential: Array

    def __init__(self, self_energy: ArrayLike, chemical_potential: ArrayLike, /):
        sigma = jnp.asarray(self_energy)
        chemical = jnp.asarray(chemical_potential)
        if sigma.ndim != 1 or chemical.shape != ():
            raise ValueError(
                "DMFT state is a self-energy vector and scalar chemical potential."
            )
        self.self_energy = sigma
        self.chemical_potential = chemical


class DMFTResidual(StrictModule):
    self_energy: Array
    density: Array


class DMFTResidualArguments(StrictModule):
    plan: SingleSiteDMFTPlan
    provider: AbstractImpurityProvider

    def __init__(self, plan: SingleSiteDMFTPlan, provider: AbstractImpurityProvider, /):
        if not isinstance(plan, SingleSiteDMFTPlan):
            raise TypeError("plan must be SingleSiteDMFTPlan.")
        if not isinstance(provider, AbstractImpurityProvider):
            raise TypeError("provider must be AbstractImpurityProvider.")
        self.plan = plan
        self.provider = provider


class DMFTIterationEvaluation(StrictModule):
    residual: DMFTResidual
    lattice_green: MatsubaraGreenFunction
    target_hybridization: MatsubaraHybridization
    bath_fit: AndersonBathFitResult
    impurity: ImpuritySolveResult


class DMFTEvidence(StrictModule):
    """Causality, moments, Dyson, density, bath, and fixed-point errors are distinct."""

    causality_residual: Array
    moment_residual: Array
    dyson_residual: Array
    density_error: Array
    bath_fit_error: Array
    finite_bath_error: Array
    fixed_point_error: Array
    finite: Array
    converged: Array
    valid: Array
    iteration_count: Array


class SingleSiteDMFTResult(StrictModule):
    state: DMFTState
    lattice_green: MatsubaraGreenFunction
    impurity: ImpuritySolveResult
    bath_fit: AndersonBathFitResult
    evidence: DMFTEvidence
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class DMFTImplicitDerivativeEvidence(StrictModule):
    """Independent smooth-branch and Jacobian admission evidence."""

    branch_gap: Array
    jacobian_condition: Array
    smooth_branch: Array
    jacobian_resolved: Array
    admitted: Array


def _evaluate_dmft(
    state: DMFTState, arguments: DMFTResidualArguments, /
) -> DMFTIterationEvaluation:
    plan = arguments.plan
    if state.self_energy.shape != plan.indices.shape:
        raise ValueError("DMFT self-energy shape does not match the plan.")
    frequency = matsubara_frequencies(
        plan.indices, beta=plan.beta, statistics="fermionic"
    )
    sigma = MatsubaraSelfEnergy(plan.beta, plan.indices, state.self_energy)
    denominator = (
        1j * frequency[:, None]
        + state.chemical_potential
        - plan.lattice_energies[None, :]
        - sigma.values[:, None]
    )
    lattice_values = jnp.sum(plan.lattice_weights[None, :] / denominator, axis=1)
    lattice_green = MatsubaraGreenFunction(plan.beta, plan.indices, lattice_values)
    target_values = (
        1j * frequency
        + state.chemical_potential
        - plan.onsite_energy
        - sigma.values
        - jnp.reciprocal(lattice_values)
    )
    target = MatsubaraHybridization(plan.beta, plan.indices, target_values)
    bath_fit = fit_causal_anderson_bath(plan.bath_fit, target)
    request = ImpuritySolveRequest(
        plan.onsite_energy,
        plan.interaction,
        float(state.chemical_potential),
        plan.beta,
        plan.indices,
        ImpurityEnvironment(bath=bath_fit.bath),
    )
    impurity = arguments.provider.solve(request)
    residual = DMFTResidual(
        sigma.values - impurity.self_energy.values,
        impurity.density - plan.target_density,
    )
    return DMFTIterationEvaluation(residual, lattice_green, target, bath_fit, impurity)


def dmft_physical_residual(
    state: DMFTState, arguments: DMFTResidualArguments, /
) -> DMFTResidual:
    """The one physical closure used by both primal iteration and root formulation."""

    return _evaluate_dmft(state, arguments).residual


def dmft_nonlinear_problem(arguments: DMFTResidualArguments, /) -> NonlinearSystemProblem:
    """Expose the exact primal residual to the native implicit-root substrate."""

    if not isinstance(arguments, DMFTResidualArguments):
        raise TypeError("arguments must be DMFTResidualArguments.")
    return NonlinearSystemProblem(
        dmft_physical_residual,
        problem_id=f"single-site-dmft:{arguments.plan.plan_id}",
    )


def solve_single_site_dmft(
    plan: SingleSiteDMFTPlan,
    /,
    *,
    initial_self_energy: ArrayLike | None = None,
    initial_chemical_potential: float = 0.0,
    provider: AbstractImpurityProvider | None = None,
) -> SingleSiteDMFTResult:
    """Run bounded damped fixed-point iteration on the shared physical residual."""

    provider_ = (
        ExactDiagonalizationImpurityProvider(plan.impurity_policy)
        if provider is None
        else provider
    )
    arguments = DMFTResidualArguments(plan, provider_)
    initial_sigma = (
        jnp.zeros(plan.indices.shape, dtype=jnp.complex128)
        if initial_self_energy is None
        else jnp.asarray(initial_self_energy)
    )
    if initial_sigma.shape != plan.indices.shape:
        raise ValueError("initial_self_energy must match the Matsubara indices.")
    chemical = float(initial_chemical_potential)
    if not isfinite(chemical):
        raise ValueError("initial_chemical_potential must be finite.")
    state = DMFTState(initial_sigma, jnp.asarray(chemical))
    evaluation = _evaluate_dmft(state, arguments)
    completed = 0
    for iteration in range(plan.maximum_iterations):
        residual = evaluation.residual
        fixed_error = jnp.max(jnp.abs(residual.self_energy), initial=0.0)
        density_error = jnp.abs(residual.density)
        bath_error = evaluation.bath_fit.evidence.relative_fit_residual
        completed = iteration + 1
        if bool(
            (fixed_error <= plan.fixed_point_tolerance)
            & (density_error <= plan.density_tolerance)
            & (bath_error <= plan.bath_tolerance)
            & evaluation.impurity.evidence.valid
        ):
            break
        state = DMFTState(
            state.self_energy - plan.self_energy_mixing * residual.self_energy,
            state.chemical_potential - plan.chemical_potential_step * residual.density,
        )
        evaluation = _evaluate_dmft(state, arguments)
    residual = evaluation.residual
    fixed_error = jnp.max(jnp.abs(residual.self_energy), initial=0.0)
    density_error = jnp.abs(residual.density)
    bath_error = evaluation.bath_fit.evidence.relative_fit_residual
    finite_bath_error = evaluation.bath_fit.evidence.finite_bath_error
    impurity_evidence = evaluation.impurity.evidence
    finite = (
        impurity_evidence.finite
        & evaluation.bath_fit.evidence.finite
        & jnp.isfinite(fixed_error)
        & jnp.isfinite(density_error)
    )
    converged = (
        (fixed_error <= plan.fixed_point_tolerance)
        & (density_error <= plan.density_tolerance)
        & (bath_error <= plan.bath_tolerance)
    )
    valid = (
        finite & converged & impurity_evidence.valid & evaluation.bath_fit.evidence.valid
    )
    evidence = DMFTEvidence(
        impurity_evidence.causality_residual,
        impurity_evidence.moment_residual,
        impurity_evidence.dyson_residual,
        density_error,
        bath_error,
        finite_bath_error,
        fixed_error,
        finite,
        converged,
        valid,
        jnp.asarray(completed, dtype=jnp.int32),
    )
    result_id = canonical_fingerprint(
        {
            "kind": "single-site-finite-bath-dmft-result",
            "plan": plan.plan_id,
            "impurity": evaluation.impurity.result_id,
            "bath_fit": evaluation.bath_fit.fit_id,
            "iterations": completed,
        }
    )
    return SingleSiteDMFTResult(
        state,
        evaluation.lattice_green,
        evaluation.impurity,
        evaluation.bath_fit,
        evidence,
        plan.plan_id,
        result_id,
    )


def admit_dmft_implicit_derivative(
    result: SingleSiteDMFTResult,
    arguments: DMFTResidualArguments,
    /,
    *,
    branch_gap: ArrayLike,
    jacobian_condition: ArrayLike,
    minimum_branch_gap: float = 1e-6,
    maximum_jacobian_condition: float = 1e8,
) -> tuple[NonlinearSystemProblem, DMFTImplicitDerivativeEvidence]:
    """Fail closed unless convergence, bath branch, and Jacobian are all certified."""

    if not isinstance(result, SingleSiteDMFTResult):
        raise TypeError("result must be SingleSiteDMFTResult.")
    if result.plan_id != arguments.plan.plan_id:
        raise ValueError("DMFT result and residual arguments use different plans.")
    gap = jnp.asarray(branch_gap)
    condition = jnp.asarray(jacobian_condition)
    if gap.shape != () or condition.shape != ():
        raise ValueError("branch_gap and jacobian_condition must be scalar.")
    minimum = _positive(minimum_branch_gap, "minimum_branch_gap")
    maximum = _positive(maximum_jacobian_condition, "maximum_jacobian_condition")
    smooth = jnp.isfinite(gap) & (gap >= minimum)
    resolved = jnp.isfinite(condition) & (condition <= maximum)
    admitted = (
        result.evidence.valid
        & result.bath_fit.evidence.valid
        & arguments.provider.differentiable
        & smooth
        & resolved
    )
    if not bool(admitted):
        raise ValueError(
            "Implicit DMFT derivative rejected: convergence, differentiable "
            "provider, finite-bath branch, or Jacobian evidence failed."
        )
    evidence = DMFTImplicitDerivativeEvidence(gap, condition, smooth, resolved, admitted)
    return dmft_nonlinear_problem(arguments), evidence


__all__ = [
    "DMFTEvidence",
    "DMFTImplicitDerivativeEvidence",
    "DMFTResidual",
    "DMFTResidualArguments",
    "DMFTState",
    "SingleSiteDMFTPlan",
    "SingleSiteDMFTResult",
    "admit_dmft_implicit_derivative",
    "dmft_nonlinear_problem",
    "dmft_physical_residual",
    "solve_single_site_dmft",
]
