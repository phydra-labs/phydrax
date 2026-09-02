#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._factorizations import FactorizationPolicy, factorize, PreparedFactorization
from ..linalg._operators import DenseLinearOperator
from ..linalg._properties import OperatorProperties
from ..stochastic._euler_maruyama import EulerMaruyamaTransitionKernel
from ..stochastic._state_space import (
    AbstractObservationModel,
    AbstractTransitionKernel,
    GaussianStatePrior,
    StateSpaceModel,
    StateSpaceProblem,
    TransitionSample,
)


SINGTransitionMethod: TypeAlias = Literal[
    "euler-factor", "local-linearization", "ensemble-moments"
]
SINGObjectiveKind: TypeAlias = Literal[
    "elbo", "surrogate_elbo", "unnormalized_variational"
]


class SINGSupportPlan(StrictModule):
    """Declared constant-rank affine support with Hausdorff reference density."""

    constraints: Array
    offset: Array
    tangent_basis: Array
    origin: Array
    constraint_factorization: PreparedFactorization
    solve_residual: Array
    nullspace_residual: Array
    constraint_rank: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    reference: Literal["hausdorff"] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        constraints: ArrayLike,
        tangent_basis: ArrayLike,
        /,
        *,
        offset: ArrayLike | None = None,
        reference: Literal["hausdorff"] = "hausdorff",
        rank: int,
        support_id: str,
        tolerance: float = 1.0e-8,
    ):
        matrix = np.asarray(constraints, dtype=float)
        basis = np.asarray(tangent_basis, dtype=float)
        fixed_rank = int(rank)
        threshold = float(tolerance)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or basis.ndim != 2
            or matrix.shape[1] != basis.shape[0]
        ):
            raise ValueError(
                "constraints must be nonempty and share the ambient dimension "
                "with tangent_basis."
            )
        if basis.shape[1] != fixed_rank or fixed_rank <= 0:
            raise ValueError("tangent_basis trailing dimension must equal positive rank.")
        shift = (
            np.zeros((matrix.shape[0],), dtype=float)
            if offset is None
            else np.asarray(offset, dtype=float)
        )
        if shift.shape != (matrix.shape[0],):
            raise ValueError("offset must contain one value per affine constraint.")
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if reference != "hausdorff":
            raise ValueError("Singular SING support density is Hausdorff only.")
        if (
            not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(basis))
            or not np.all(np.isfinite(shift))
        ):
            raise ValueError("support arrays must be finite.")
        constraint_factorization = factorize(
            DenseLinearOperator(
                jnp.asarray(matrix),
                operator_id=f"sing-support:{support_id}:constraints",
            ),
            FactorizationPolicy("svd"),
        )
        constraint_rank = int(jax.device_get(constraint_factorization.rank()))
        if constraint_rank != matrix.shape[1] - fixed_rank:
            raise ValueError(
                "Constraint numerical rank does not match the declared tangent rank."
            )
        origin_solve = constraint_factorization.solve(jnp.asarray(shift))
        origin = jnp.asarray(origin_solve.value)
        solve_residual = jnp.asarray(origin_solve.diagnostics.residual_norm)
        nullspace = constraint_factorization.right_nullspace()
        null_basis = jnp.asarray(nullspace.basis)
        declared_projector = jnp.asarray(basis) @ jnp.asarray(basis).T
        computed_projector = null_basis @ null_basis.T
        nullspace_residual = jnp.sqrt(
            jnp.sum((declared_projector - computed_projector) ** 2)
        )
        if (
            int(nullspace.dimension) != fixed_rank
            or not bool(origin_solve.successful)
            or not bool(solve_residual <= threshold)
            or not bool(nullspace_residual <= 10.0 * threshold)
        ):
            raise ValueError(
                "Affine support solve/nullspace evidence is inconsistent with tangent_basis."
            )
        if not np.allclose(
            basis.T @ basis, np.eye(fixed_rank), rtol=threshold, atol=threshold
        ):
            raise ValueError("tangent_basis must be orthonormal.")
        if not isinstance(support_id, str) or not support_id:
            raise ValueError("support_id must be non-empty.")
        self.constraints = jnp.asarray(matrix)
        self.offset = jnp.asarray(shift)
        self.tangent_basis = jnp.asarray(basis)
        self.origin = origin
        self.constraint_factorization = constraint_factorization
        self.solve_residual = solve_residual
        self.nullspace_residual = nullspace_residual
        self.constraint_rank = constraint_rank
        self.rank = fixed_rank
        self.reference = reference
        self.support_id = support_id
        self.tolerance = threshold

    def residual(self, state: ArrayLike, /) -> Array:
        flat = jnp.asarray(state).reshape((-1,))
        return self.constraints @ flat - self.offset


class SINGTransitionPlan(StrictModule):
    """Exact Euler factor or explicitly labeled finite Gaussian surrogate."""

    support: SINGSupportPlan | None
    ensemble_plan: Any
    surrogate_provider: Any
    method: SINGTransitionMethod = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    approximation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SINGTransitionMethod = "euler-factor",
        /,
        *,
        support: SINGSupportPlan | None = None,
        ensemble_plan: Any = None,
        surrogate_provider: Any = None,
        rank_tolerance: float = 1.0e-8,
        approximation_tolerance: float = 1.0e-4,
    ):
        if method not in ("euler-factor", "local-linearization", "ensemble-moments"):
            raise ValueError("Unknown SING transition method.")
        if support is not None and not isinstance(support, SINGSupportPlan):
            raise TypeError("support must be SINGSupportPlan or None.")
        if method == "euler-factor" and surrogate_provider is not None:
            raise ValueError("euler-factor does not consume a surrogate_provider.")
        if method != "euler-factor" and not callable(surrogate_provider):
            raise TypeError(
                "solver-backed methods require an explicit surrogate_provider."
            )
        if method == "ensemble-moments" and ensemble_plan is None:
            raise ValueError("ensemble-moments requires a finite ensemble_plan.")
        rank_threshold = float(rank_tolerance)
        approximation_threshold = float(approximation_tolerance)
        if not all(
            isfinite(value) and value > 0.0
            for value in (rank_threshold, approximation_threshold)
        ):
            raise ValueError("transition tolerances must be finite and positive.")
        self.method = method
        self.support = support
        self.ensemble_plan = ensemble_plan
        self.surrogate_provider = surrogate_provider
        self.rank_tolerance = rank_threshold
        self.approximation_tolerance = approximation_threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sing-transition-plan-v1",
                "method": method,
                "support": None if support is None else support.support_id,
                "ensemble": None
                if ensemble_plan is None
                else type(ensemble_plan).__name__,
                "rank_tolerance": rank_threshold,
                "approximation_tolerance": approximation_threshold,
            }
        )


class SINGTransitionEvaluation(StrictModule):
    mean: Array
    covariance_factor: Array
    log_density: Array
    support_residual: Array
    rank: Array
    valid: Array
    approximation_error: Array
    status: Array
    approximation_kind: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)


def _gaussian_data(
    mean: Array,
    covariance: Array,
    next_state: Array,
    tolerance: float,
    /,
):
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    valid = (
        jnp.all(jnp.isfinite(symmetric))
        & jnp.all(eigenvalues > tolerance)
        & jnp.all(jnp.isfinite(mean))
    )
    safe = jnp.where(valid, symmetric, jnp.eye(symmetric.shape[0], dtype=symmetric.dtype))
    factor = jnp.linalg.cholesky(safe)
    difference = next_state - mean
    solution = jnp.linalg.solve(factor, difference)
    log_density = -0.5 * (
        mean.size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=mean.dtype))
        + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
        + ein.contract("i,i->", solution, solution)
    )
    return factor, log_density, eigenvalues, valid


def evaluate_sing_transition(
    transition: Any,
    state: ArrayLike,
    next_state: ArrayLike,
    t0: ArrayLike,
    t1: ArrayLike,
    context: Any,
    plan: SINGTransitionPlan,
    /,
    *,
    key: Any = None,
) -> SINGTransitionEvaluation:
    """Evaluate one exact represented factor or one explicit Gaussian surrogate."""
    if not isinstance(plan, SINGTransitionPlan):
        raise TypeError("plan must be a SINGTransitionPlan.")
    source = jnp.asarray(state)
    target = jnp.asarray(next_state)
    if source.shape != target.shape:
        raise ValueError("state and next_state must have one shape.")
    if plan.method == "euler-factor":
        if not isinstance(transition, EulerMaruyamaTransitionKernel):
            raise TypeError("euler-factor requires EulerMaruyamaTransitionKernel.")
        mean = transition.mean(source, t0, t1, context).reshape((-1,))
        covariance = transition.covariance(source, t0, t1, context)
        approximation_error = jnp.asarray(0.0, dtype=mean.dtype)
        approximation_kind = "exact-represented-euler"
    else:
        mean, covariance, approximation_error = plan.surrogate_provider(
            transition,
            source,
            jnp.asarray(t0),
            jnp.asarray(t1),
            context,
            key,
            plan.ensemble_plan,
        )
        mean = jnp.asarray(mean).reshape((-1,))
        covariance = jnp.asarray(covariance)
        approximation_error = jnp.asarray(approximation_error)
        approximation_kind = plan.method
    flat_target = target.reshape((-1,))
    dimension = int(flat_target.size)
    if covariance.shape != (dimension, dimension):
        raise ValueError("transition covariance must be ambient state_size square.")
    support = plan.support
    if support is None:
        factor, log_density, eigenvalues, factor_valid = _gaussian_data(
            mean,
            covariance,
            flat_target,
            plan.rank_tolerance,
        )
        residual = jnp.zeros((0,), dtype=mean.dtype)
        rank = jnp.sum(eigenvalues > plan.rank_tolerance).astype(jnp.int32)
        support_valid = rank == dimension
        reference = "lebesgue"
    else:
        if support.tangent_basis.shape[0] != dimension:
            raise ValueError("SING support ambient dimension does not match state.")
        basis = support.tangent_basis.astype(covariance.dtype)
        projected_mean = basis.T @ mean
        projected_target = basis.T @ flat_target
        projected_covariance = basis.T @ covariance @ basis
        factor, log_density, eigenvalues, factor_valid = _gaussian_data(
            projected_mean,
            projected_covariance,
            projected_target,
            plan.rank_tolerance,
        )
        residual = jnp.concatenate(
            (support.residual(mean), support.residual(flat_target))
        )
        rank = jnp.sum(eigenvalues > plan.rank_tolerance).astype(jnp.int32)
        support_valid = (rank == support.rank) & jnp.all(
            jnp.abs(residual) <= support.tolerance
        )
        reference = "hausdorff"
    approximation_valid = jnp.isfinite(approximation_error) & (
        approximation_error <= plan.approximation_tolerance
    )
    valid = factor_valid & support_valid & approximation_valid & jnp.isfinite(log_density)
    status = jnp.where(
        ~support_valid,
        2,
        jnp.where(~factor_valid, 1, jnp.where(~approximation_valid, 3, 0)),
    ).astype(jnp.int32)
    transition_id = canonical_fingerprint(
        {
            "kind": "sing-transition-evaluation-v1",
            "transition": transition.process_id,
            "plan": plan.plan_id,
            "reference": reference,
        }
    )
    return SINGTransitionEvaluation(
        mean=mean.reshape(source.shape),
        covariance_factor=factor,
        log_density=jnp.where(valid, log_density, -jnp.inf),
        support_residual=residual,
        rank=rank,
        valid=valid,
        approximation_error=approximation_error,
        status=status,
        approximation_kind=approximation_kind,
        transition_id=transition_id,
        reference_measure=reference,
    )


class _ProjectedEulerTransition(AbstractTransitionKernel):
    ambient: EulerMaruyamaTransitionKernel
    support: SINGSupportPlan
    wiener_terms: tuple[Any, ...]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        ambient: EulerMaruyamaTransitionKernel,
        support: SINGSupportPlan,
        /,
    ):
        self.ambient = ambient
        self.support = support
        self.wiener_terms = ambient.wiener_terms
        self.state_shape = (support.rank,)
        self.process_id = f"{ambient.process_id}:affine-hausdorff:{support.support_id}"
        self.approximation_id = "exact-affine-hausdorff-euler"
        self.has_log_density = True

    def lift(self, reduced: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(reduced).reshape((self.support.rank,))
        ambient = self.support.origin + self.support.tangent_basis @ coordinates
        return ambient.reshape(self.ambient.state_shape)

    def drift(self, time, state, context, /) -> Array:
        ambient = self.lift(state)
        value = self.ambient.drift(time, ambient, context).reshape((-1,))
        return self.support.tangent_basis.T @ value

    def dispersion(self, time, state, context, /) -> Array:
        ambient = self.lift(state)
        value = self.ambient.dispersion(time, ambient, context)
        return self.support.tangent_basis.T @ value

    def mean(self, state, t0, t1, context, /) -> Array:
        value = jnp.asarray(state)
        return value + (jnp.asarray(t1) - jnp.asarray(t0)) * self.drift(
            t0, value, context
        )

    def covariance(self, state, t0, t1, context, /) -> Array:
        coefficient = self.dispersion(t0, state, context)
        return (jnp.asarray(t1) - jnp.asarray(t0)) * (coefficient @ coefficient.T)

    def _factorization(self, state, t0, t1, context, /):
        covariance = self.covariance(state, t0, t1, context)
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        positive_definite = jnp.all(jnp.isfinite(symmetric)) & jnp.all(eigenvalues > 0.0)
        safe_covariance = jnp.where(
            positive_definite,
            symmetric,
            jnp.eye(symmetric.shape[0], dtype=symmetric.dtype),
        )
        prepared = factorize(
            DenseLinearOperator(
                safe_covariance,
                operator_id=f"{self.process_id}:covariance",
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={"positive_definite": "verified"},
                ),
            ),
            FactorizationPolicy("cholesky"),
        )
        return prepared, positive_definite

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        mean = self.mean(state, t0, t1, context)
        coefficient = self.dispersion(t0, state, context)
        noise = jr.normal(key, (coefficient.shape[-1],), dtype=mean.dtype)
        value = mean + jnp.sqrt(jnp.asarray(t1) - jnp.asarray(t0)) * (coefficient @ noise)
        valid = jnp.all(jnp.isfinite(value))
        return TransitionSample(
            values=value,
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        mean = self.mean(state, t0, t1, context)
        difference = jnp.asarray(next_state) - mean
        prepared, covariance_valid = self._factorization(state, t0, t1, context)
        solved = prepared.solve(difference)
        rank_valid = prepared.rank() == self.support.rank
        valid = (
            covariance_valid
            & solved.successful
            & rank_valid
            & jnp.isfinite(prepared.log_abs_determinant())
        )
        quadratic = ein.contract("i,i->", difference, solved.value)
        value = -0.5 * (
            self.support.rank * jnp.log(2.0 * jnp.pi)
            + prepared.log_abs_determinant()
            + quadratic
        )
        return jnp.where(valid, value, -jnp.inf)


class _ProjectedObservationModel(AbstractObservationModel):
    ambient: AbstractObservationModel
    support: SINGSupportPlan
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(self, ambient: AbstractObservationModel, support: SINGSupportPlan, /):
        self.ambient = ambient
        self.support = support
        self.state_shape = (support.rank,)
        self.observation_shape = ambient.observation_shape
        self.observation_id = f"{ambient.observation_id}:support:{support.support_id}"

    def lift(self, state):
        flat = self.support.origin + self.support.tangent_basis @ jnp.asarray(state)
        return flat.reshape(self.ambient.state_shape)

    def location(self, state, time, context, /):
        return self.ambient.location(self.lift(state), time, context)

    def log_prob(self, value, state, time, mask, context, /):
        return self.ambient.log_prob(value, self.lift(state), time, mask, context)

    def sample(self, key, state, time, context, sample_shape=()):
        return self.ambient.sample(
            key,
            self.lift(state),
            time,
            context,
            sample_shape,
        )


class SINGConstrainedResult(StrictModule):
    """Gaussian-chain posterior normalized on one fixed affine Hausdorff support."""

    reduced_problem: StateSpaceProblem
    reduced_result: Any
    ambient_means: Array
    ambient_covariances: Array
    ambient_transition_cross_covariances: Array
    support_residuals: Array
    valid: Array
    status: Array
    solve_evidence: tuple[str, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)

    @property
    def state(self):
        return self.reduced_result.state

    @property
    def objective(self):
        return self.reduced_result.elbo.total_elbo


def _projected_prior(
    prior: GaussianStatePrior,
    support: SINGSupportPlan,
    /,
):
    state_size = support.tangent_basis.shape[0]
    case_shape = prior.batch_shape
    case_count = int(np.prod(case_shape)) if case_shape else 1
    means = prior.mean.reshape((case_count, state_size))
    covariances = prior.covariance.reshape((case_count, state_size, state_size))
    if not prior.has_log_density:
        raise ValueError(
            "Affine-Hausdorff SING requires a positive-definite ambient prior."
        )
    reduced_means = []
    reduced_covariances = []
    evidence = []
    basis = support.tangent_basis
    identity = jnp.eye(support.rank, dtype=basis.dtype)
    for case_index in range(case_count):
        covariance = 0.5 * (covariances[case_index] + covariances[case_index].T)
        covariance_factorization = factorize(
            DenseLinearOperator(
                covariance,
                operator_id=(
                    f"sing-support:{support.support_id}:prior-covariance:{case_index}"
                ),
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={"positive_definite": "verified"},
                ),
            ),
            FactorizationPolicy("cholesky"),
        )
        solved_basis = jnp.stack(
            tuple(
                covariance_factorization.solve(basis[:, column]).value
                for column in range(support.rank)
            ),
            axis=-1,
        )
        solved_difference = covariance_factorization.solve(
            means[case_index] - support.origin
        )
        reduced_precision = 0.5 * (basis.T @ solved_basis + solved_basis.T @ basis)
        reduced_factorization = factorize(
            DenseLinearOperator(
                reduced_precision,
                operator_id=(
                    f"sing-support:{support.support_id}:prior-precision:{case_index}"
                ),
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={"positive_definite": "construction"},
                ),
            ),
            FactorizationPolicy("cholesky"),
        )
        covariance_columns = tuple(
            reduced_factorization.solve(identity[:, column]).value
            for column in range(support.rank)
        )
        reduced_covariance = jnp.stack(covariance_columns, axis=-1)
        information = basis.T @ solved_difference.value
        reduced_mean = reduced_covariance @ information
        valid = (covariance_factorization.rank() == state_size) & (
            reduced_factorization.rank() == support.rank
        )
        if not bool(valid):
            raise ValueError("Constrained prior factorization is rank deficient.")
        reduced_means.append(reduced_mean)
        reduced_covariances.append(reduced_covariance)
        evidence.extend(
            (
                covariance_factorization.factorization_id,
                reduced_factorization.factorization_id,
            )
        )
    mean = jnp.stack(reduced_means).reshape(case_shape + (support.rank,))
    covariance = jnp.stack(reduced_covariances).reshape(
        case_shape + (support.rank, support.rank)
    )
    return (
        GaussianStatePrior(
            mean,
            covariance,
            state_shape=(support.rank,),
            prior_id=f"{prior.prior_id}:support:{support.support_id}",
        ),
        tuple(evidence),
    )


def _projected_sing_problem(
    problem: StateSpaceProblem,
    support: SINGSupportPlan,
    /,
):
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    transition = problem.model.transition
    if not isinstance(transition, EulerMaruyamaTransitionKernel):
        raise TypeError("Affine-Hausdorff SING requires an Euler transition.")
    if any(term.structure != "additive" for term in transition.wiener_terms):
        raise ValueError(
            "Affine-Hausdorff SING requires state-independent additive diffusion."
        )
    ambient_size = int(np.prod(transition.state_shape))
    if support.tangent_basis.shape[0] != ambient_size:
        raise ValueError("Support ambient dimension differs from model state size.")
    context = problem.step_context(0, 0)
    origin_state = support.origin.reshape(transition.state_shape)

    def flat_drift(flat):
        return transition.drift(
            problem.initial_time.reshape((-1,))[0],
            flat.reshape(transition.state_shape),
            context,
        ).reshape((-1,))

    drift_origin = flat_drift(support.origin)
    drift_jacobian = jax.jacfwd(flat_drift)(support.origin)
    coefficient_origin = transition.dispersion(
        problem.initial_time.reshape((-1,))[0],
        origin_state,
        context,
    )
    tangent_residual = jnp.maximum(
        jnp.max(jnp.abs(support.constraints @ drift_origin)),
        jnp.max(jnp.abs(support.constraints @ coefficient_origin)),
    )
    affine_residual = jnp.asarray(0.0, dtype=drift_origin.dtype)
    diffusion_residual = jnp.asarray(0.0, dtype=drift_origin.dtype)
    for column in range(support.rank):
        probe = support.origin + support.tangent_basis[:, column]
        probe_drift = flat_drift(probe)
        predicted = drift_origin + drift_jacobian @ (probe - support.origin)
        affine_residual = jnp.maximum(
            affine_residual, jnp.max(jnp.abs(probe_drift - predicted))
        )
        probe_coefficient = transition.dispersion(
            problem.initial_time.reshape((-1,))[0],
            probe.reshape(transition.state_shape),
            context,
        )
        diffusion_residual = jnp.maximum(
            diffusion_residual,
            jnp.max(jnp.abs(probe_coefficient - coefficient_origin)),
        )
    if not bool(
        (tangent_residual <= support.tolerance)
        & (affine_residual <= support.tolerance)
        & (diffusion_residual <= support.tolerance)
    ):
        raise ValueError(
            "Transition is not affine/tangent with constant diffusion on the support."
        )
    reduced_prior, prior_evidence = _projected_prior(problem.model.prior, support)
    reduced_transition = _ProjectedEulerTransition(transition, support)
    reduced_observation = _ProjectedObservationModel(
        problem.model.observation,
        support,
    )
    reduced_model = StateSpaceModel(
        reduced_prior,
        reduced_transition,
        reduced_observation,
        model_id=f"{problem.model.model_id}:support:{support.support_id}",
        parameter_id=problem.model.parameter_id,
        basis_id=problem.model.basis_id,
        discretization_id=problem.model.discretization_id,
        metadata={
            **dict(problem.model.metadata),
            "reference_measure": "hausdorff",
            "support_id": support.support_id,
        },
    )
    reduced_problem = StateSpaceProblem(
        reduced_model,
        problem.observations,
        initial_time=problem.initial_time,
        problem_id=f"{problem.problem_id}:support:{support.support_id}",
        args=problem.args,
        input_signal=problem.input_signal,
    )
    evidence = (
        support.constraint_factorization.factorization_id,
        *prior_evidence,
        f"support-solve-residual:{float(support.solve_residual)}",
        f"support-nullspace-residual:{float(support.nullspace_residual)}",
    )
    return reduced_problem, evidence


def sing_constrained_smoother(
    problem: StateSpaceProblem,
    /,
    *,
    transition_plan: SINGTransitionPlan,
    **kwargs,
) -> SINGConstrainedResult:
    """Run SING in fixed affine coordinates and lift Hausdorff posterior moments."""
    from ._sing import sing_smoother

    if not isinstance(transition_plan, SINGTransitionPlan):
        raise TypeError("transition_plan must be a SINGTransitionPlan.")
    support = transition_plan.support
    if support is None:
        raise ValueError("sing_constrained_smoother requires a support plan.")
    if transition_plan.method != "euler-factor":
        raise ValueError(
            "Affine-Hausdorff smoothing currently requires exact represented Euler."
        )
    reduced_problem, evidence = _projected_sing_problem(problem, support)
    reduced = sing_smoother(reduced_problem, **kwargs)
    reduced_means = reduced.means
    ambient_flat = support.origin + ein.contract(
        "...r,dr->...d", reduced_means, support.tangent_basis
    )
    ambient_means = ambient_flat.reshape(
        reduced.case_shape + (reduced.moments.num_nodes,) + problem.model.state_shape
    )
    covariances = ein.contract(
        "dr,...rs,es->...de",
        support.tangent_basis,
        reduced.covariances,
        support.tangent_basis,
    )
    cross = ein.contract(
        "dr,...rs,es->...de",
        support.tangent_basis,
        reduced.transition_cross_covariances,
        support.tangent_basis,
    )
    residuals = (
        ein.contract("cd,...d->...c", support.constraints, ambient_flat) - support.offset
    )
    valid = (
        reduced.valid
        & jnp.all(jnp.abs(residuals) <= support.tolerance)
        & jnp.all(jnp.isfinite(covariances))
        & jnp.all(jnp.isfinite(cross))
    )
    return SINGConstrainedResult(
        reduced_problem=reduced_problem,
        reduced_result=reduced,
        ambient_means=ambient_means,
        ambient_covariances=covariances,
        ambient_transition_cross_covariances=cross,
        support_residuals=residuals,
        valid=valid,
        status=jnp.where(valid, 0, 1).astype(jnp.int32),
        solve_evidence=evidence,
        support_id=support.support_id,
        reference_measure="hausdorff",
        approximation_kind="exact-affine-hausdorff-euler",
    )


class SINGObjectiveResult(StrictModule):
    objective: Array
    decomposition: Any
    valid: Array
    audited: Array
    objective_kind: SINGObjectiveKind = eqx.field(static=True)
    evidence_available: bool = eqx.field(static=True)
    transition_semantics: str = eqx.field(static=True)


def sing_objective(
    problem: Any,
    state: Any,
    /,
    *,
    transition_plan: SINGTransitionPlan,
    observation_factor: Any = None,
    batch: Any = None,
) -> SINGObjectiveResult:
    """Evaluate SING with explicit transition and observation semantics.

    Factor batches are accepted only for optimization bookkeeping here; a value is
    evidence-bearing only after the caller supplies a full audit (``batch=None``).
    """
    from ._sing import sing_elbo

    if not isinstance(transition_plan, SINGTransitionPlan):
        raise TypeError("transition_plan must be a SINGTransitionPlan.")
    if transition_plan.support is None:
        objective_problem = problem
        transition_semantics = transition_plan.method
    else:
        objective_problem, _ = _projected_sing_problem(problem, transition_plan.support)
        transition_semantics = "affine-hausdorff"
    decomposition = sing_elbo(objective_problem, state)
    semantics = (
        "normalized" if observation_factor is None else observation_factor.semantics
    )
    if semantics not in ("normalized", "unnormalized_potential"):
        raise ValueError("Unknown observation factor semantics.")
    if semantics == "unnormalized_potential":
        kind: SINGObjectiveKind = "unnormalized_variational"
        evidence_available = False
    elif transition_plan.method == "euler-factor":
        kind = "elbo"
        evidence_available = batch is None
    else:
        kind = "surrogate_elbo"
        evidence_available = False
    audited = jnp.asarray(batch is None)
    return SINGObjectiveResult(
        objective=decomposition.total_elbo,
        decomposition=decomposition,
        valid=decomposition.valid,
        audited=audited,
        objective_kind=kind,
        evidence_available=evidence_available,
        transition_semantics=transition_semantics,
    )


__all__ = [
    "SINGObjectiveKind",
    "SINGObjectiveResult",
    "SINGConstrainedResult",
    "SINGSupportPlan",
    "SINGTransitionEvaluation",
    "SINGTransitionMethod",
    "SINGTransitionPlan",
    "evaluate_sing_transition",
    "sing_constrained_smoother",
    "sing_objective",
]
