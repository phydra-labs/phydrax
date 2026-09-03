#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Local identifiability, profile, derivative, and experiment-design utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    DenseLinearOperator,
    eigen as eigen_api,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
    pseudoinverse,
    RankPolicy,
    svd as svd_api,
)
from ....optim import (
    AbstractMinimizationMethod,
    Bounds,
    MinimizationProblem,
    OptimizationStatus,
    OptimizationTermination,
    ProjectedLBFGS,
    StateDesignResult,
)


class SensitivitySVDResult(StrictModule):
    """Scaled local sensitivity SVD and explicit confounding subspace."""

    jacobian: Array
    scaled_jacobian: Array
    singular_values: Array
    right_vectors: Array
    nullspace_basis: Array
    nullity: Array
    rank: Array
    condition_number: Array
    finite: Array
    confounded: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SensitivitySVDPlan(StrictModule, NonTrainableState):
    """Differentiate one fixed-topology observable map and prepare native SVD rank evidence."""

    forward: Any
    parameter_scale: Array
    observation_scale: Array
    relative_rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: Callable[[Array, Any], ArrayLike],
        parameter_scale: ArrayLike,
        observation_scale: ArrayLike,
        /,
        *,
        relative_rank_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ):
        if not callable(forward):
            raise TypeError("forward must be callable.")
        parameter = jax.lax.stop_gradient(
            jnp.asarray(parameter_scale, dtype=float).reshape(-1)
        )
        observation = jax.lax.stop_gradient(
            jnp.asarray(observation_scale, dtype=float).reshape(-1)
        )
        tolerance = float(relative_rank_tolerance)
        if parameter.size == 0 or observation.size == 0:
            raise ValueError("Sensitivity scales must be non-empty.")
        if bool(jnp.any(~jnp.isfinite(parameter) | (parameter <= 0.0))):
            raise ValueError("parameter_scale must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(observation) | (observation <= 0.0))):
            raise ValueError("observation_scale must be finite and positive.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("relative_rank_tolerance must be finite and non-negative.")
        derived = canonical_fingerprint(
            {
                "kind": "cardiovascular-sensitivity-svd-plan",
                "parameter_scale": array_tree_fingerprint(parameter),
                "observation_scale": array_tree_fingerprint(observation),
                "relative_rank_tolerance": tolerance,
            }
        )
        identifier = derived if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.forward = forward
        self.parameter_scale = parameter
        self.observation_scale = observation
        self.relative_rank_tolerance = tolerance
        self.plan_id = identifier

    def evaluate(
        self, parameters: ArrayLike, args: Any = None, /
    ) -> SensitivitySVDResult:
        point = jnp.asarray(parameters, dtype=self.parameter_scale.dtype)
        if point.shape != self.parameter_scale.shape:
            raise ValueError("parameters must match parameter_scale shape.")

        def flattened(candidate):
            output = jnp.asarray(self.forward(candidate, args))
            if jnp.issubdtype(output.dtype, jnp.complexfloating):
                raise TypeError("Sensitivity observables must be real-valued.")
            return output.reshape(-1)

        value = flattened(point)
        if value.shape != self.observation_scale.shape:
            raise ValueError("forward output size must match observation_scale.")
        jacobian = jax.jacrev(flattened)(point).reshape(
            (self.observation_scale.size, self.parameter_scale.size)
        )
        scaled = (
            jacobian * self.parameter_scale[None, :] / self.observation_scale[:, None]
        )
        operator = DenseLinearOperator(scaled)
        rank_policy = RankPolicy(relative_cutoff=self.relative_rank_tolerance)
        singular_decomposition = svd_api.svd(
            svd_api.SVDProblem(operator),
            policy=svd_api.SVDSolvePolicy(
                count=min(scaled.shape),
                rank=rank_policy,
            ),
        )
        decomposition = factorize(
            operator,
            FactorizationPolicy("svd", rank=rank_policy),
        )
        singular_values = singular_decomposition.singular_values
        rank = decomposition.rank().astype(jnp.int32)
        nullspace = decomposition.right_nullspace()
        right_vectors = singular_decomposition.right_vectors
        maximum_rank = min(scaled.shape)
        full_column_rank = rank == self.parameter_scale.size
        smallest = singular_values[jnp.maximum(jnp.asarray(maximum_rank - 1), 0)]
        condition = jnp.where(
            full_column_rank & (smallest > 0.0),
            singular_values[0] / smallest,
            jnp.inf,
        )
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(jacobian))
            & jnp.all(jnp.isfinite(singular_values))
            & jnp.all(jnp.isfinite(nullspace.basis))
        )
        confounded = rank < self.parameter_scale.size
        resolved_threshold = self.relative_rank_tolerance * jnp.maximum(
            singular_values[0], jnp.finfo(singular_values.dtype).tiny
        )
        resolved_modes = singular_values > resolved_threshold
        decomposition_accepted = jnp.all(
            jnp.where(resolved_modes, singular_decomposition.converged, True)
        )
        return SensitivitySVDResult(
            jacobian=jacobian,
            scaled_jacobian=scaled,
            singular_values=singular_values,
            right_vectors=right_vectors,
            nullspace_basis=nullspace.basis,
            nullity=nullspace.dimension,
            rank=rank,
            condition_number=condition,
            finite=finite,
            confounded=confounded,
            successful=finite & decomposition_accepted,
            plan_id=self.plan_id,
        )


class PositiveSemidefiniteEvidence(StrictModule):
    """Native eigensolve evidence for one symmetric PSD information input."""

    symmetry_error: Array
    minimum_eigenvalue: Array
    tolerance: Array
    finite: Array
    accepted: Array


def _positive_semidefinite_evidence(
    matrix: Array,
    name: str,
    /,
) -> PositiveSemidefiniteEvidence:
    symmetry_error = jnp.max(jnp.abs(matrix - matrix.T))
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), 1.0)
    tolerance = 64.0 * jnp.finfo(matrix.dtype).eps * matrix.shape[0] * scale
    if bool(symmetry_error > tolerance):
        raise ValueError(f"{name} must be symmetric.")
    properties = OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "verified"},
    )
    eigen_result = eigen_api.eigensolve(
        eigen_api.Eigenproblem(
            DenseLinearOperator(0.5 * (matrix + matrix.T), properties=properties)
        ),
        policy=eigen_api.EigenSolvePolicy(
            eigen_api.DenseEigh(),
            count=matrix.shape[0],
            which="smallest-algebraic",
        ),
    )
    minimum = jnp.min(eigen_result.eigenvalues)
    finite = jnp.all(jnp.isfinite(eigen_result.eigenvalues))
    accepted = eigen_result.successful & finite & (minimum >= -tolerance)
    if not bool(accepted):
        raise ValueError(f"{name} must be positive semidefinite.")
    return PositiveSemidefiniteEvidence(
        symmetry_error=symmetry_error,
        minimum_eigenvalue=minimum,
        tolerance=tolerance,
        finite=finite,
        accepted=accepted,
    )


class FisherLocalResult(StrictModule):
    """Local Fisher information, pseudocovariance, and correlation diagnostics."""

    information: Array
    covariance: Array
    standard_errors: Array
    correlation: Array
    observation_precision_evidence: PositiveSemidefiniteEvidence
    prior_information_evidence: PositiveSemidefiniteEvidence
    rank: Array
    confounded: Array
    finite: Array
    successful: Array


def fisher_local_diagnostics(
    jacobian: ArrayLike,
    /,
    *,
    observation_precision: ArrayLike | None = None,
    prior_information: ArrayLike | None = None,
    relative_rank_tolerance: float = 1.0e-8,
) -> FisherLocalResult:
    sensitivity = jnp.asarray(jacobian, dtype=float)
    if sensitivity.ndim != 2 or sensitivity.shape[0] == 0 or sensitivity.shape[1] == 0:
        raise ValueError("jacobian must be a non-empty matrix.")
    if bool(jnp.any(~jnp.isfinite(sensitivity))):
        raise ValueError("jacobian must be finite.")
    observations, parameters = sensitivity.shape
    precision = (
        jnp.eye(observations, dtype=sensitivity.dtype)
        if observation_precision is None
        else jnp.asarray(observation_precision, dtype=sensitivity.dtype)
    )
    if precision.shape != (observations, observations):
        raise ValueError("observation_precision shape does not match jacobian rows.")
    prior = (
        jnp.zeros((parameters, parameters), dtype=sensitivity.dtype)
        if prior_information is None
        else jnp.asarray(prior_information, dtype=sensitivity.dtype)
    )
    if prior.shape != (parameters, parameters):
        raise ValueError("prior_information shape does not match jacobian columns.")
    if bool(jnp.any(~jnp.isfinite(precision))) or bool(jnp.any(~jnp.isfinite(prior))):
        raise ValueError("Fisher precision inputs must be finite.")
    precision_evidence = _positive_semidefinite_evidence(
        precision, "observation_precision"
    )
    prior_evidence = _positive_semidefinite_evidence(prior, "prior_information")
    information = contract("oi,op,pj->ij", sensitivity, precision, sensitivity) + prior
    information = 0.5 * (information + information.T)
    decomposition = factorize(
        DenseLinearOperator(information),
        FactorizationPolicy(
            "svd", rank=RankPolicy(relative_cutoff=relative_rank_tolerance)
        ),
    )
    covariance_result = pseudoinverse(
        DenseLinearOperator(information),
        FactorizationPolicy(
            "svd", rank=RankPolicy(relative_cutoff=relative_rank_tolerance)
        ),
    )
    covariance = covariance_result.value
    variance = jnp.maximum(jnp.diag(covariance), 0.0)
    standard_errors = jnp.sqrt(variance)
    denominator = standard_errors[:, None] * standard_errors[None, :]
    correlation = jnp.where(denominator > 0.0, covariance / denominator, 0.0)
    rank = decomposition.rank().astype(jnp.int32)
    finite = (
        jnp.all(jnp.isfinite(information))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.isfinite(correlation))
    )
    return FisherLocalResult(
        information=information,
        covariance=covariance,
        standard_errors=standard_errors,
        correlation=correlation,
        observation_precision_evidence=precision_evidence,
        prior_information_evidence=prior_evidence,
        rank=rank,
        confounded=rank < parameters,
        finite=finite,
        successful=finite & covariance_result.successful,
    )


class ProfileLikelihoodResult(StrictModule):
    """Fixed-grid profiled objective with per-grid optimizer status."""

    grid: Array
    optimized_parameters: Array
    objectives: Array
    delta_objective: Array
    statuses: Array
    successful: Array
    all_successful: Array
    plan_id: str = eqx.field(static=True)


class ProfileLikelihoodPlan(StrictModule, NonTrainableState):
    """Profile one parameter while optimizing every declared nuisance coordinate."""

    objective: Any
    parameter_index: int = eqx.field(static=True)
    grid: Array
    bounds: Bounds | None
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array, Any], ArrayLike],
        parameter_index: int,
        grid: ArrayLike,
        /,
        *,
        bounds: Bounds | None = None,
        method: AbstractMinimizationMethod | None = None,
        termination: OptimizationTermination | None = None,
        plan_id: str | None = None,
    ):
        if not callable(objective):
            raise TypeError("objective must be callable.")
        index = int(parameter_index)
        if index < 0:
            raise ValueError("parameter_index must be non-negative.")
        grid_ = jax.lax.stop_gradient(jnp.asarray(grid, dtype=float).reshape(-1))
        if grid_.size < 2 or bool(jnp.any(~jnp.isfinite(grid_))):
            raise ValueError("Profile grid must contain at least two finite values.")
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds or None.")
        method_ = ProjectedLBFGS() if method is None else method
        termination_ = (
            OptimizationTermination(maximum_steps=100)
            if termination is None
            else termination
        )
        if not isinstance(method_, AbstractMinimizationMethod):
            raise TypeError("method must be AbstractMinimizationMethod or None.")
        if not isinstance(termination_, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination or None.")
        derived = canonical_fingerprint(
            {
                "kind": "cardiovascular-profile-likelihood-plan",
                "parameter_index": index,
                "grid": array_tree_fingerprint(grid_),
                "method": method_.method_id,
            }
        )
        identifier = derived if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.objective = objective
        self.parameter_index = index
        self.grid = grid_
        self.bounds = bounds
        self.method = method_
        self.termination = termination_
        self.plan_id = identifier

    def evaluate(
        self, initial_parameters: ArrayLike, args: Any = None, /
    ) -> ProfileLikelihoodResult:
        initial = jnp.asarray(initial_parameters, dtype=self.grid.dtype).reshape(-1)
        if self.parameter_index >= initial.size:
            raise ValueError("parameter_index lies outside initial_parameters.")
        if bool(jnp.any(~jnp.isfinite(initial))):
            raise ValueError("initial_parameters must be finite.")
        free = np.asarray(
            [index for index in range(initial.size) if index != self.parameter_index],
            dtype=np.int32,
        )
        free_indices = jnp.asarray(free)
        lower = upper = None
        if self.bounds is not None:
            materialized_lower, materialized_upper = self.bounds.materialize(initial)
            lower = materialized_lower[free_indices]
            upper = materialized_upper[free_indices]
        warm = initial[free_indices]
        candidates: list[Array] = []
        objectives: list[Array] = []
        statuses: list[Array] = []
        accepted: list[Array] = []
        for fixed_value in self.grid:
            if free.size == 0:
                candidate = initial.at[self.parameter_index].set(fixed_value)
                value = jnp.asarray(self.objective(candidate, args)).reshape(())
                success = jnp.isfinite(value)
                status = jnp.where(
                    success,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.NONFINITE_EVALUATION),
                ).astype(jnp.int32)
            else:

                def assemble(free_values):
                    return (
                        initial.at[free_indices]
                        .set(free_values)
                        .at[self.parameter_index]
                        .set(fixed_value)
                    )

                problem = MinimizationProblem(
                    lambda free_values, dynamic_args: self.objective(
                        assemble(free_values), dynamic_args
                    ),
                    bounds=None
                    if lower is None or upper is None
                    else Bounds(lower, upper),
                    problem_id=f"{self.plan_id}/grid",
                )
                result = self.method.solve(
                    problem,
                    warm,
                    termination=self.termination,
                    args=args,
                )
                candidate = assemble(result.parameters)
                value = result.objective
                success = result.successful & jnp.isfinite(value)
                status = result.status
                if bool(success):
                    warm = result.parameters
            candidates.append(candidate)
            objectives.append(value)
            statuses.append(status)
            accepted.append(success)
        objective_array = jnp.stack(objectives)
        successful = jnp.stack(accepted)
        accepted_values = jnp.where(successful, objective_array, jnp.inf)
        minimum = jnp.min(accepted_values)
        delta = jnp.where(successful, objective_array - minimum, jnp.inf)
        return ProfileLikelihoodResult(
            grid=self.grid,
            optimized_parameters=jnp.stack(candidates),
            objectives=objective_array,
            delta_objective=delta,
            statuses=jnp.stack(statuses),
            successful=successful,
            all_successful=jnp.all(successful),
            plan_id=self.plan_id,
        )


class DirectionalDerivativeCheck(StrictModule):
    """AD-versus-centered-difference evidence at one fixed-topology point."""

    automatic: Array
    finite_difference: Array
    absolute_error: Array
    relative_error: Array
    finite: Array
    accepted: Array


def check_directional_derivative(
    function: Callable[[Array], ArrayLike],
    point: ArrayLike,
    direction: ArrayLike,
    /,
    *,
    step: float = 1.0e-4,
    relative_tolerance: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-6,
) -> DirectionalDerivativeCheck:
    if not callable(function):
        raise TypeError("function must be callable.")
    point_ = jnp.asarray(point, dtype=float)
    direction_ = jnp.asarray(direction, dtype=point_.dtype)
    if direction_.shape != point_.shape:
        raise ValueError("direction must match point shape.")
    step_ = float(step)
    relative = float(relative_tolerance)
    absolute = float(absolute_tolerance)
    if any(
        not math.isfinite(value) or value <= 0.0 for value in (step_, relative, absolute)
    ):
        raise ValueError(
            "Derivative-check step and tolerances must be finite and positive."
        )

    def scalar(value):
        result = jnp.asarray(function(value))
        if result.shape != ():
            raise ValueError("Directional derivative checks require a scalar function.")
        return result

    _, automatic = jax.jvp(scalar, (point_,), (direction_,))
    finite_difference = (
        scalar(point_ + step_ * direction_) - scalar(point_ - step_ * direction_)
    ) / (2.0 * step_)
    absolute_error = jnp.abs(automatic - finite_difference)
    scale = jnp.maximum(jnp.maximum(jnp.abs(automatic), jnp.abs(finite_difference)), 1.0)
    relative_error = absolute_error / scale
    finite = jnp.isfinite(automatic) & jnp.isfinite(finite_difference)
    accepted = finite & (absolute_error <= absolute + relative * scale)
    return DirectionalDerivativeCheck(
        automatic=automatic,
        finite_difference=finite_difference,
        absolute_error=absolute_error,
        relative_error=relative_error,
        finite=finite,
        accepted=accepted,
    )


class ForwardAdjointEvidence(StrictModule):
    """Evidence required before a derivative contributes to experiment design."""

    forward_accepted: Array
    adjoint_accepted: Array
    fixed_topology: Array
    derivative_finite: Array
    successful: Array

    def __init__(
        self,
        forward_accepted: ArrayLike,
        adjoint_accepted: ArrayLike,
        fixed_topology: ArrayLike,
        derivative_finite: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (
                forward_accepted,
                adjoint_accepted,
                fixed_topology,
                derivative_finite,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Forward/adjoint evidence values must be scalar.")
        self.forward_accepted = values[0]
        self.adjoint_accepted = values[1]
        self.fixed_topology = values[2]
        self.derivative_finite = values[3]
        self.successful = values[0] & values[1] & values[2] & values[3]

    @classmethod
    def from_state_design(
        cls,
        result: StateDesignResult,
        /,
        *,
        fixed_topology: ArrayLike,
        derivative_finite: ArrayLike,
    ) -> "ForwardAdjointEvidence":
        if not isinstance(result, StateDesignResult):
            raise TypeError("result must be a StateDesignResult.")
        adjoint = (
            jnp.asarray(False)
            if result.adjoint_acceptance is None
            else result.adjoint_acceptance.accepted
        )
        return cls(
            result.state_acceptance.accepted,
            adjoint,
            fixed_topology,
            derivative_finite,
        )


class ExperimentDesignCandidate(StrictModule, NonTrainableState):
    """One auditable experiment's local Fisher-information contribution."""

    sensitivity: Array
    noise_precision: Array
    information: Array
    evidence: ForwardAdjointEvidence
    cost: float = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_id: str,
        sensitivity: ArrayLike,
        noise_precision: ArrayLike,
        evidence: ForwardAdjointEvidence,
        /,
        *,
        cost: float = 1.0,
    ):
        identifier = str(candidate_id)
        if not identifier or identifier != identifier.strip():
            raise ValueError("candidate_id must be non-empty and canonical.")
        sensitivity_ = jax.lax.stop_gradient(jnp.asarray(sensitivity, dtype=float))
        if sensitivity_.ndim != 2 or min(sensitivity_.shape) < 1:
            raise ValueError("Candidate sensitivity must be a non-empty matrix.")
        precision = jax.lax.stop_gradient(
            jnp.asarray(noise_precision, dtype=sensitivity_.dtype)
        )
        if precision.shape != (sensitivity_.shape[0], sensitivity_.shape[0]):
            raise ValueError("noise_precision must match sensitivity rows.")
        if bool(jnp.any(~jnp.isfinite(sensitivity_))) or bool(
            jnp.any(~jnp.isfinite(precision))
        ):
            raise ValueError("Candidate derivative and precision must be finite.")
        if bool(jnp.any(jnp.abs(precision - precision.T) > 1.0e-10)):
            raise ValueError("Candidate noise_precision must be symmetric.")
        if not isinstance(evidence, ForwardAdjointEvidence):
            raise TypeError("evidence must be ForwardAdjointEvidence.")
        cost_ = float(cost)
        if not math.isfinite(cost_) or cost_ <= 0.0:
            raise ValueError("Candidate cost must be finite and positive.")
        information = contract("oi,op,pj->ij", sensitivity_, precision, sensitivity_)
        self.sensitivity = sensitivity_
        self.noise_precision = precision
        self.information = 0.5 * (information + information.T)
        self.evidence = evidence
        self.cost = cost_
        self.candidate_id = identifier

    @property
    def parameter_count(self) -> int:
        return int(self.sensitivity.shape[1])


class ExperimentDesignCriterion(Enum):
    D_OPTIMAL = "d-optimal"
    A_OPTIMAL = "a-optimal"
    E_OPTIMAL = "e-optimal"


class ExperimentDesignPlan(StrictModule, NonTrainableState):
    """Finite candidate set, positive prior information, and a hard resource budget."""

    candidates: tuple[ExperimentDesignCandidate, ...]
    prior_information: Array
    criterion: ExperimentDesignCriterion = eqx.field(static=True)
    maximum_experiments: int = eqx.field(static=True)
    budget: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidates: Sequence[ExperimentDesignCandidate],
        prior_information: ArrayLike,
        /,
        *,
        criterion: ExperimentDesignCriterion = ExperimentDesignCriterion.D_OPTIMAL,
        maximum_experiments: int = 1,
        budget: float = math.inf,
    ):
        resolved = tuple(candidates)
        if not resolved:
            raise ValueError("ExperimentDesignPlan requires candidates.")
        if any(
            not isinstance(candidate, ExperimentDesignCandidate) for candidate in resolved
        ):
            raise TypeError("candidates must contain ExperimentDesignCandidate values.")
        identifiers = tuple(candidate.candidate_id for candidate in resolved)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Experiment candidate IDs must be unique.")
        parameter_count = resolved[0].parameter_count
        if any(candidate.parameter_count != parameter_count for candidate in resolved):
            raise ValueError(
                "All experiment candidates must use one parameter dimension."
            )
        prior = jax.lax.stop_gradient(jnp.asarray(prior_information, dtype=float))
        if prior.shape != (parameter_count, parameter_count):
            raise ValueError("prior_information shape must match candidate parameters.")
        if bool(jnp.any(~jnp.isfinite(prior))) or bool(
            jnp.any(jnp.abs(prior - prior.T) > 1.0e-10)
        ):
            raise ValueError("prior_information must be finite and symmetric.")
        if not isinstance(criterion, ExperimentDesignCriterion):
            raise TypeError("criterion must be an ExperimentDesignCriterion.")
        maximum = int(maximum_experiments)
        budget_ = float(budget)
        if maximum < 1 or maximum > len(resolved):
            raise ValueError("maximum_experiments must lie within the candidate count.")
        if math.isnan(budget_) or budget_ <= 0.0:
            raise ValueError("budget must be positive and not NaN.")
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "construction", "positive_definite": "asserted"},
        )
        prior_factor = factorize(
            DenseLinearOperator(prior, properties=properties),
            FactorizationPolicy("cholesky"),
        )
        if not bool(prior_factor.materialize_inverse().successful):
            raise ValueError("prior_information must be positive definite.")
        self.candidates = resolved
        self.prior_information = prior
        self.criterion = criterion
        self.maximum_experiments = maximum
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-experiment-design-plan",
                "candidates": list(identifiers),
                "candidate_information": [
                    array_tree_fingerprint(candidate.information)
                    for candidate in resolved
                ],
                "prior_information": array_tree_fingerprint(prior),
                "criterion": criterion.value,
                "maximum_experiments": maximum,
                "budget": "infinity" if math.isinf(budget_) else budget_,
            }
        )

    def prepare(self, /) -> "PreparedExperimentDesign":
        return PreparedExperimentDesign(self)


class ExperimentDesignResult(StrictModule):
    """Greedy information design with fixed-capacity selection evidence."""

    selected_indices: Array
    selected_mask: Array
    selected_count: Array
    score_history: Array
    total_cost: Array
    final_information: Array
    candidate_evidence_accepted: Array
    budget_satisfied: Array
    finite: Array
    successful: Array
    runtime_id: str = eqx.field(static=True)


class PreparedExperimentDesign(StrictModule, NonTrainableState):
    plan: ExperimentDesignPlan
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: ExperimentDesignPlan, /):
        if not isinstance(plan, ExperimentDesignPlan):
            raise TypeError("plan must be an ExperimentDesignPlan.")
        self.plan = plan
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-cardiovascular-experiment-design", "plan": plan.plan_id}
        )

    def _score(self, information: Array, /) -> Array:
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "construction", "positive_definite": "asserted"},
        )
        if self.plan.criterion is ExperimentDesignCriterion.E_OPTIMAL:
            decomposition = factorize(
                DenseLinearOperator(information, properties=properties),
                FactorizationPolicy("svd"),
            )
            values = decomposition.singular_values()
            return jnp.min(values)
        decomposition = factorize(
            DenseLinearOperator(information, properties=properties),
            FactorizationPolicy("cholesky"),
        )
        if self.plan.criterion is ExperimentDesignCriterion.D_OPTIMAL:
            return decomposition.log_abs_determinant()
        inverse = decomposition.materialize_inverse()
        return -jnp.trace(inverse.value)

    def select(self, /) -> ExperimentDesignResult:
        selected = np.zeros(len(self.plan.candidates), dtype=bool)
        indices = np.full(self.plan.maximum_experiments, -1, dtype=np.int32)
        scores = np.full(self.plan.maximum_experiments + 1, np.nan, dtype=float)
        information = self.plan.prior_information
        current_score = self._score(information)
        scores[0] = float(current_score)
        total_cost = 0.0
        selected_count = 0
        for slot in range(self.plan.maximum_experiments):
            best_index = -1
            best_gain = -math.inf
            best_score = current_score
            for index, candidate in enumerate(self.plan.candidates):
                if selected[index] or not bool(candidate.evidence.successful):
                    continue
                if total_cost + candidate.cost > self.plan.budget:
                    continue
                candidate_score = self._score(information + candidate.information)
                gain = float((candidate_score - current_score) / candidate.cost)
                if math.isfinite(gain) and gain > best_gain:
                    best_index = index
                    best_gain = gain
                    best_score = candidate_score
            if best_index < 0:
                break
            selected[best_index] = True
            indices[slot] = best_index
            candidate = self.plan.candidates[best_index]
            information = information + candidate.information
            total_cost += candidate.cost
            selected_count += 1
            current_score = best_score
            scores[slot + 1] = float(current_score)
        evidence = jnp.asarray(
            [bool(candidate.evidence.successful) for candidate in self.plan.candidates]
        )
        selected_mask = jnp.asarray(selected)
        selected_evidence = jnp.all(jnp.where(selected_mask, evidence, True))
        budget_satisfied = jnp.asarray(total_cost <= self.plan.budget)
        finite = jnp.all(jnp.isfinite(information)) & jnp.isfinite(current_score)
        successful = (
            jnp.asarray(selected_count > 0)
            & selected_evidence
            & budget_satisfied
            & finite
        )
        return ExperimentDesignResult(
            selected_indices=jnp.asarray(indices),
            selected_mask=selected_mask,
            selected_count=jnp.asarray(selected_count, dtype=jnp.int32),
            score_history=jnp.asarray(scores, dtype=information.dtype),
            total_cost=jnp.asarray(total_cost, dtype=information.dtype),
            final_information=information,
            candidate_evidence_accepted=evidence,
            budget_satisfied=budget_satisfied,
            finite=finite,
            successful=successful,
            runtime_id=self.runtime_id,
        )


__all__ = [
    "DirectionalDerivativeCheck",
    "ExperimentDesignCandidate",
    "ExperimentDesignCriterion",
    "ExperimentDesignPlan",
    "ExperimentDesignResult",
    "FisherLocalResult",
    "ForwardAdjointEvidence",
    "PreparedExperimentDesign",
    "PositiveSemidefiniteEvidence",
    "ProfileLikelihoodPlan",
    "ProfileLikelihoodResult",
    "SensitivitySVDPlan",
    "SensitivitySVDResult",
    "check_directional_derivative",
    "fisher_local_diagnostics",
]
