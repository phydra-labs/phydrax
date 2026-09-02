#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-budget mixed-domain q-batch Gaussian-process optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import opt_einsum as oe
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule
from ..optim import FiniteProductSpace
from ._gp_kernel_fit import (
    fit_gaussian_process_kernel,
    GaussianProcessKernelFitPolicy,
    GaussianProcessKernelFitResult,
)
from ._gp_likelihood import GaussianProcessLikelihoodState


BAYESIAN_OPTIMIZATION_INITIAL = 0
BAYESIAN_OPTIMIZATION_ACQUISITION = 1
BAYESIAN_OPTIMIZATION_FEASIBILITY_FIRST = 2
BAYESIAN_OPTIMIZATION_SPACE_FILLING = 3


class BayesianOptimizationPoint(StrictModule):
    """Decoded continuous PyTree and finite categorical payload."""

    continuous: PyTree[Any]
    categorical: PyTree[Any] | None
    categorical_index: Array
    encoded: Array


class BayesianOptimizationDomain(StrictModule):
    """Explicit bounded continuous PyTree and optional finite categorical product."""

    continuous_template: PyTree[Any] | None
    lower_bounds: PyTree[Any] | None
    upper_bounds: PyTree[Any] | None
    continuous_initial: Array
    continuous_lower: Array
    continuous_upper: Array
    categorical: FiniteProductSpace | None
    _unravel: Any = eqx.field(static=True)
    continuous_dimension: int = eqx.field(static=True)
    categorical_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        continuous_template: PyTree[Any] | None = None,
        /,
        *,
        lower_bounds: PyTree[Any] | None = None,
        upper_bounds: PyTree[Any] | None = None,
        categorical: FiniteProductSpace | None = None,
    ):
        if continuous_template is None:
            if lower_bounds is not None or upper_bounds is not None:
                raise ValueError("Continuous bounds require a continuous_template.")
            initial = jnp.empty((0,), dtype=float)
            lower = initial
            upper = initial
            unravel = lambda _: None
        else:
            if lower_bounds is None or upper_bounds is None:
                raise ValueError("Both lower_bounds and upper_bounds are required.")
            initial, unravel = ravel_pytree(continuous_template)
            lower, lower_unravel = ravel_pytree(lower_bounds)
            upper, upper_unravel = ravel_pytree(upper_bounds)
            if jax.tree_util.tree_structure(
                continuous_template
            ) != jax.tree_util.tree_structure(
                lower_bounds
            ) or jax.tree_util.tree_structure(
                continuous_template
            ) != jax.tree_util.tree_structure(upper_bounds):
                raise ValueError("Continuous template and bounds need equal PyTrees.")
            del lower_unravel, upper_unravel
            dtype = jnp.result_type(initial, lower, upper)
            if not jnp.issubdtype(dtype, jnp.floating):
                raise TypeError("Continuous domain leaves must be real-valued.")
            initial = initial.astype(dtype)
            lower = lower.astype(dtype)
            upper = upper.astype(dtype)
            host_initial = np.asarray(jax.device_get(initial))
            host_lower = np.asarray(jax.device_get(lower))
            host_upper = np.asarray(jax.device_get(upper))
            if (
                not np.all(np.isfinite(host_initial))
                or not np.all(np.isfinite(host_lower))
                or not np.all(np.isfinite(host_upper))
            ):
                raise ValueError("Continuous templates and bounds must be finite.")
            if np.any(host_lower >= host_upper):
                raise ValueError(
                    "Every continuous lower bound must be below its upper bound."
                )
            if np.any((host_initial < host_lower) | (host_initial > host_upper)):
                raise ValueError("continuous_template must lie inside its bounds.")
        if categorical is not None and not isinstance(categorical, FiniteProductSpace):
            raise TypeError("categorical must be a FiniteProductSpace or None.")
        if int(initial.shape[0]) == 0 and categorical is None:
            raise ValueError("A Bayesian-optimization domain cannot be empty.")
        self.continuous_template = continuous_template
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.continuous_initial = initial
        self.continuous_lower = lower
        self.continuous_upper = upper
        self.categorical = categorical
        self._unravel = unravel
        self.continuous_dimension = int(initial.shape[0])
        self.categorical_dimension = (
            0 if categorical is None else len(categorical.product_shape)
        )

    @property
    def encoded_dimension(self) -> int:
        return self.continuous_dimension + self.categorical_dimension

    def to_unit(self, continuous: PyTree[Any] | None, /) -> Array:
        if self.continuous_dimension == 0:
            if continuous is not None:
                raise ValueError("This domain has no continuous coordinates.")
            return self.continuous_initial
        vector, _ = ravel_pytree(continuous)
        if vector.shape != (self.continuous_dimension,):
            raise ValueError("Continuous point does not match the domain template.")
        return (vector - self.continuous_lower) / (
            self.continuous_upper - self.continuous_lower
        )

    def decode(
        self, unit_continuous: ArrayLike, categorical_flat_index: ArrayLike = 0, /
    ) -> BayesianOptimizationPoint:
        unit = jnp.asarray(unit_continuous, dtype=self.continuous_initial.dtype)
        if unit.shape != (self.continuous_dimension,):
            raise ValueError("unit_continuous has an incompatible shape.")
        unit = eqx.error_if(
            unit,
            jnp.any(~jnp.isfinite(unit)) | jnp.any(unit < 0.0) | jnp.any(unit > 1.0),
            "Continuous unit coordinates must lie in [0, 1].",
        )
        physical_vector = self.continuous_lower + unit * (
            self.continuous_upper - self.continuous_lower
        )
        continuous = (
            None if self.continuous_dimension == 0 else self._unravel(physical_vector)
        )
        if self.categorical is None:
            category_indices = jnp.empty((0,), dtype=jnp.int32)
            categorical_payload = None
        else:
            flat = jnp.asarray(categorical_flat_index)
            if flat.ndim != 0 or not jnp.issubdtype(flat.dtype, jnp.integer):
                raise TypeError("categorical_flat_index must be an integer scalar.")
            category_indices = jnp.stack(
                tuple(
                    index.astype(jnp.int32)
                    for index in self.categorical.unravel_index(flat)
                )
            )
            categorical_payload = self.categorical.take(flat)
        encoded = jnp.concatenate((unit, category_indices.astype(unit.dtype)))
        return BayesianOptimizationPoint(
            continuous=continuous,
            categorical=categorical_payload,
            categorical_index=category_indices,
            encoded=encoded,
        )


class BayesianOptimizationProblem(StrictModule):
    """One real scalar objective and fixed scalar inequality constraints g(x) <= 0."""

    domain: BayesianOptimizationDomain
    objective: Callable[[BayesianOptimizationPoint], ArrayLike] = eqx.field(static=True)
    constraints: tuple[Callable[[BayesianOptimizationPoint], ArrayLike], ...] = eqx.field(
        static=True
    )
    pending: tuple[BayesianOptimizationPoint, ...]

    def __init__(
        self,
        objective: Callable[[BayesianOptimizationPoint], ArrayLike],
        domain: BayesianOptimizationDomain,
        /,
        *,
        constraints: Sequence[Callable[[BayesianOptimizationPoint], ArrayLike]] = (),
        pending: Sequence[BayesianOptimizationPoint] = (),
    ):
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if not isinstance(domain, BayesianOptimizationDomain):
            raise TypeError("domain must be a BayesianOptimizationDomain.")
        constraint_tuple = tuple(constraints)
        if any(not callable(constraint) for constraint in constraint_tuple):
            raise TypeError("Every constraint must be callable.")
        pending_tuple = tuple(pending)
        if any(
            not isinstance(point, BayesianOptimizationPoint) for point in pending_tuple
        ):
            raise TypeError("pending must contain BayesianOptimizationPoint values.")
        if any(
            point.encoded.shape != (domain.encoded_dimension,) for point in pending_tuple
        ):
            raise ValueError("Every pending point must match the optimization domain.")
        if any(not bool(jnp.all(jnp.isfinite(point.encoded))) for point in pending_tuple):
            raise ValueError("Pending point encodings must be finite.")
        self.domain = domain
        self.objective = objective
        self.constraints = constraint_tuple
        self.pending = pending_tuple


class GaussianProcessBayesianOptimization(StrictModule):
    """Fixed-budget q-batch Monte Carlo acquisition over finite candidate pools."""

    objective_surrogate: GaussianProcessLikelihoodState
    constraint_surrogates: tuple[GaussianProcessLikelihoodState, ...]
    kernel_fit: GaussianProcessKernelFitPolicy | None
    max_evaluations: int = eqx.field(static=True)
    initial_evaluations: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    candidate_tuple_count: int = eqx.field(static=True)
    fantasy_count: int = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)

    def __init__(
        self,
        max_evaluations: int,
        /,
        *,
        objective_surrogate: GaussianProcessLikelihoodState,
        constraint_surrogates: Sequence[GaussianProcessLikelihoodState] = (),
        kernel_fit: GaussianProcessKernelFitPolicy | None = None,
        initial_evaluations: int = 8,
        batch_size: int = 1,
        candidate_tuple_count: int = 256,
        fantasy_count: int = 128,
        minimum_separation: float = 1e-6,
    ):
        maximum = _positive_integer(max_evaluations, name="max_evaluations")
        initial = _positive_integer(initial_evaluations, name="initial_evaluations")
        batch = _positive_integer(batch_size, name="batch_size")
        candidates = _positive_integer(
            candidate_tuple_count, name="candidate_tuple_count"
        )
        fantasies = _positive_integer(fantasy_count, name="fantasy_count")
        if fantasies < 2:
            raise ValueError("fantasy_count must be at least 2.")
        if initial > maximum:
            raise ValueError("initial_evaluations cannot exceed max_evaluations.")
        if batch > maximum:
            raise ValueError("batch_size cannot exceed max_evaluations.")
        if not isinstance(objective_surrogate, GaussianProcessLikelihoodState):
            raise TypeError(
                "objective_surrogate must be a GaussianProcessLikelihoodState."
            )
        constraints = tuple(constraint_surrogates)
        if any(
            not isinstance(state, GaussianProcessLikelihoodState) for state in constraints
        ):
            raise TypeError("constraint_surrogates must contain GP likelihood states.")
        if any(
            state.noise_scale.ndim != 0 for state in (objective_surrogate,) + constraints
        ):
            raise ValueError(
                "Bayesian-optimization surrogate noise_scale must be scalar."
            )
        if kernel_fit is not None and not isinstance(
            kernel_fit, GaussianProcessKernelFitPolicy
        ):
            raise TypeError(
                "kernel_fit must be a GaussianProcessKernelFitPolicy or None."
            )
        separation = _positive_real(minimum_separation, name="minimum_separation")
        self.objective_surrogate = objective_surrogate
        self.constraint_surrogates = constraints
        self.kernel_fit = kernel_fit
        self.max_evaluations = maximum
        self.initial_evaluations = initial
        self.batch_size = batch
        self.candidate_tuple_count = candidates
        self.fantasy_count = fantasies
        self.minimum_separation = separation


class BayesianOptimizationResult(StrictModule):
    """Complete finite-budget observations, acquisition error, and key provenance."""

    evaluated_encoded: Array
    evaluated_continuous_unit: Array
    evaluated_categorical_indices: Array
    objectives: Array
    constraints: Array
    valid: Array
    feasible: Array
    proposal_kinds: Array
    incumbent_history: Array
    acquisition_estimates: Array
    acquisition_standard_errors: Array
    decoded_points: tuple[BayesianOptimizationPoint, ...]
    key: Array
    fantasy_keys: Array
    best_point: BayesianOptimizationPoint
    best_objective: Array
    kernel_fit_results: tuple[GaussianProcessKernelFitResult, ...]
    evaluation_count: int = eqx.field(static=True)
    invalid_evaluation_count: int = eqx.field(static=True)
    pending_count: int = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    globally_optimal: bool = eqx.field(static=True)


def bayesian_optimize(
    problem: BayesianOptimizationProblem,
    plan: GaussianProcessBayesianOptimization,
    key: Array,
    /,
) -> BayesianOptimizationResult:
    """Run deterministic-key mixed constrained q-batch Bayesian optimization."""
    if not isinstance(problem, BayesianOptimizationProblem):
        raise TypeError("problem must be a BayesianOptimizationProblem.")
    if not isinstance(plan, GaussianProcessBayesianOptimization):
        raise TypeError("plan must be a GaussianProcessBayesianOptimization.")
    if plan.objective_surrogate.kernel.input_ndim != 1:
        raise ValueError("BO surrogate kernels must consume encoded vectors.")
    if problem.constraints:
        if len(plan.constraint_surrogates) != len(problem.constraints):
            raise ValueError("constraint_surrogates must align with problem constraints.")
    elif plan.constraint_surrogates:
        raise ValueError("constraint_surrogates require problem constraints.")
    root_key = jnp.asarray(key)
    domain = problem.domain
    key, initial_key = jr.split(root_key)
    initial_units, initial_categories = _initial_candidate_points(
        domain, initial_key, problem.pending, plan
    )
    pending_encoded = _pending_encodings(domain, problem.pending)
    encoded_rows: list[Array] = []
    unit_rows: list[Array] = []
    categorical_rows: list[Array] = []
    points: list[BayesianOptimizationPoint] = []
    objectives: list[Array] = []
    constraint_rows: list[Array] = []
    proposal_kinds: list[int] = []
    acquisition_estimates: list[Array] = []
    acquisition_errors: list[Array] = []
    fantasy_keys: list[Array] = []
    kernel_fit_results: list[GaussianProcessKernelFitResult] = []
    objective_surrogate = plan.objective_surrogate

    def evaluate(unit: Array, category: Array, proposal_kind: int) -> None:
        point = domain.decode(unit, category)
        objective = jnp.asarray(problem.objective(point), dtype=unit.dtype)
        if objective.ndim != 0 or jnp.issubdtype(objective.dtype, jnp.complexfloating):
            raise ValueError(
                "Bayesian-optimization objective must return one real scalar."
            )
        if problem.constraints:
            constraint = jnp.stack(
                tuple(
                    jnp.asarray(function(point), dtype=unit.dtype).reshape(())
                    for function in problem.constraints
                )
            )
        else:
            constraint = jnp.empty((0,), dtype=unit.dtype)
        encoded_rows.append(point.encoded)
        unit_rows.append(unit)
        categorical_rows.append(point.categorical_index)
        points.append(point)
        objectives.append(objective)
        constraint_rows.append(constraint)
        proposal_kinds.append(proposal_kind)

    for index in range(plan.initial_evaluations):
        evaluate(
            initial_units[index],
            initial_categories[index],
            BAYESIAN_OPTIMIZATION_INITIAL,
        )

    while len(points) < plan.max_evaluations:
        epoch_index = len(fantasy_keys)
        if (
            plan.kernel_fit is not None
            and len(points) >= plan.kernel_fit.minimum_data_count
            and epoch_index % plan.kernel_fit.refit_interval == 0
        ):
            fit_mask = np.asarray(
                jax.device_get(jnp.isfinite(jnp.stack(tuple(objectives))))
            )
            if int(np.sum(fit_mask)) >= plan.kernel_fit.minimum_data_count:
                fit_result = fit_gaussian_process_kernel(
                    jnp.stack(tuple(encoded_rows))[fit_mask],
                    jnp.stack(tuple(objectives))[fit_mask],
                    plan.kernel_fit,
                    previous_state=objective_surrogate,
                )
                kernel_fit_results.append(fit_result)
                objective_surrogate = fit_result.state
        epoch_plan = eqx.tree_at(
            lambda value: value.objective_surrogate,
            plan,
            objective_surrogate,
        )
        key, candidate_key, fantasy_key = jr.split(key, 3)
        fantasy_keys.append(fantasy_key)
        tuple_units, tuple_categories = _candidate_tuples(domain, candidate_key, plan)
        encoded = jnp.stack(tuple(encoded_rows))
        occupied = jnp.concatenate((encoded, pending_encoded), axis=0)
        encoded_tuples = _encode_tuples(domain, tuple_units, tuple_categories)
        separated = _separated_tuples(
            encoded_tuples,
            occupied,
            minimum_separation=plan.minimum_separation,
        )
        objective_values = jnp.stack(tuple(objectives))
        constraint_values = jnp.stack(tuple(constraint_rows))
        objective_valid = jnp.isfinite(objective_values)
        constraints_valid = (
            jnp.all(jnp.isfinite(constraint_values), axis=1)
            if problem.constraints
            else jnp.ones_like(objective_valid)
        )
        observed_feasible = objective_valid & constraints_valid
        if problem.constraints:
            observed_feasible = observed_feasible & jnp.all(
                constraint_values <= 0.0, axis=1
            )
        has_feasible = bool(jnp.any(observed_feasible))
        if int(jnp.sum(objective_valid)) >= 2:
            scores, errors = _acquisition_scores(
                encoded,
                objective_values,
                constraint_values,
                objective_valid,
                constraints_valid,
                observed_feasible,
                tuple_units,
                tuple_categories,
                domain,
                problem.pending,
                epoch_plan,
                fantasy_key,
            )
            scores = jnp.where(separated & jnp.isfinite(scores), scores, -jnp.inf)
            usable = bool(jnp.any(jnp.isfinite(scores)))
        else:
            scores = jnp.full((plan.candidate_tuple_count,), -jnp.inf)
            errors = jnp.full_like(scores, jnp.nan)
            usable = False
        if usable:
            selected = int(jnp.argmax(scores))
            proposal = (
                BAYESIAN_OPTIMIZATION_ACQUISITION
                if has_feasible or not problem.constraints
                else BAYESIAN_OPTIMIZATION_FEASIBILITY_FIRST
            )
            acquisition_estimates.append(scores[selected])
            acquisition_errors.append(errors[selected])
        else:
            if not bool(jnp.any(separated)):
                raise ValueError(
                    "The BO candidate pool has no tuple separated from observations "
                    "and pending points."
                )
            selected = int(
                _space_filling_tuple(encoded_tuples, occupied, eligible=separated)
            )
            proposal = BAYESIAN_OPTIMIZATION_SPACE_FILLING
            acquisition_estimates.append(
                jnp.asarray(jnp.nan, dtype=objective_values.dtype)
            )
            acquisition_errors.append(jnp.asarray(jnp.nan, dtype=objective_values.dtype))
        remaining = plan.max_evaluations - len(points)
        take = min(plan.batch_size, remaining)
        for member in range(take):
            evaluate(
                tuple_units[selected, member],
                tuple_categories[selected, member],
                proposal,
            )

    objective_array = jnp.stack(tuple(objectives))
    constraint_array = jnp.stack(tuple(constraint_rows))
    valid = jnp.isfinite(objective_array)
    if problem.constraints:
        valid = valid & jnp.all(jnp.isfinite(constraint_array), axis=1)
        feasible = valid & jnp.all(constraint_array <= 0.0, axis=1)
    else:
        feasible = valid
    eligible = jnp.where(feasible, objective_array, jnp.inf)
    has_result = bool(jnp.any(feasible))
    best_index = int(jnp.argmin(eligible)) if has_result else 0
    incumbent = jnp.minimum.accumulate(eligible)
    acquisition_array = (
        jnp.stack(tuple(acquisition_estimates))
        if acquisition_estimates
        else jnp.empty((0,), dtype=objective_array.dtype)
    )
    error_array = (
        jnp.stack(tuple(acquisition_errors))
        if acquisition_errors
        else jnp.empty((0,), dtype=objective_array.dtype)
    )
    fantasy_array = (
        jnp.stack(tuple(fantasy_keys))
        if fantasy_keys
        else jnp.empty((0,) + root_key.shape, dtype=root_key.dtype)
    )
    return BayesianOptimizationResult(
        evaluated_encoded=jnp.stack(tuple(encoded_rows)),
        evaluated_continuous_unit=jnp.stack(tuple(unit_rows)),
        evaluated_categorical_indices=jnp.stack(tuple(categorical_rows)),
        objectives=objective_array,
        constraints=constraint_array,
        valid=valid,
        feasible=feasible,
        proposal_kinds=jnp.asarray(proposal_kinds, dtype=jnp.int32),
        incumbent_history=incumbent,
        acquisition_estimates=acquisition_array,
        acquisition_standard_errors=error_array,
        decoded_points=tuple(points),
        key=root_key,
        fantasy_keys=fantasy_array,
        best_point=points[best_index],
        best_objective=(
            objective_array[best_index]
            if has_result
            else jnp.asarray(jnp.nan, dtype=objective_array.dtype)
        ),
        kernel_fit_results=tuple(kernel_fit_results),
        evaluation_count=plan.max_evaluations,
        invalid_evaluation_count=int(jnp.sum(~valid)),
        pending_count=len(problem.pending),
        termination_reason=(
            "evaluation_budget_exhausted"
            if has_result
            else "no_feasible_finite_evaluation"
        ),
        globally_optimal=False,
    )


def _candidate_points(
    domain: BayesianOptimizationDomain, key: Array, /, *, count: int
) -> tuple[Array, Array]:
    continuous_key, category_key = jr.split(key)
    units = jr.uniform(
        continuous_key,
        (count, domain.continuous_dimension),
        dtype=domain.continuous_initial.dtype,
    )
    categories = (
        jnp.zeros((count,), dtype=jnp.int32)
        if domain.categorical is None
        else jr.randint(
            category_key, (count,), 0, domain.categorical.size, dtype=jnp.int32
        )
    )
    return units, categories


def _candidate_tuples(
    domain: BayesianOptimizationDomain,
    key: Array,
    plan: GaussianProcessBayesianOptimization,
    /,
) -> tuple[Array, Array]:
    units, categories = _candidate_points(
        domain,
        key,
        count=plan.candidate_tuple_count * plan.batch_size,
    )
    return (
        units.reshape(
            (plan.candidate_tuple_count, plan.batch_size, domain.continuous_dimension)
        ),
        categories.reshape((plan.candidate_tuple_count, plan.batch_size)),
    )


def _pending_encodings(
    domain: BayesianOptimizationDomain,
    pending: tuple[BayesianOptimizationPoint, ...],
    /,
) -> Array:
    if not pending:
        return jnp.empty(
            (0, domain.encoded_dimension), dtype=domain.continuous_initial.dtype
        )
    return jnp.stack(tuple(point.encoded for point in pending))


def _initial_candidate_points(
    domain: BayesianOptimizationDomain,
    key: Array,
    pending: tuple[BayesianOptimizationPoint, ...],
    plan: GaussianProcessBayesianOptimization,
    /,
) -> tuple[Array, Array]:
    if domain.continuous_dimension == 0:
        pool_count = domain.categorical.size
        units = jnp.zeros((pool_count, 0), dtype=domain.continuous_initial.dtype)
        categories = jnp.arange(pool_count, dtype=jnp.int32)
    else:
        pool_count = max(
            plan.candidate_tuple_count,
            plan.initial_evaluations + len(pending),
        )
        units, categories = _candidate_points(domain, key, count=pool_count)
        units = units.at[0].set(domain.to_unit(domain.continuous_template))
        if domain.categorical is not None:
            categories = categories.at[0].set(0)
    candidate_encoded = _encode_tuples(domain, units[:, None, :], categories[:, None])
    occupied = _pending_encodings(domain, pending)
    selected: list[int] = []
    for _ in range(plan.initial_evaluations):
        separated = _separated_tuples(
            candidate_encoded,
            occupied,
            minimum_separation=plan.minimum_separation,
        )
        if not bool(jnp.any(separated)):
            raise ValueError(
                "The BO domain has too few separated points for the initial design "
                "after accounting for pending points."
            )
        index = int(jnp.argmax(separated))
        selected.append(index)
        occupied = jnp.concatenate(
            (occupied, candidate_encoded[index, 0][None, :]), axis=0
        )
    indices = jnp.asarray(selected, dtype=jnp.int32)
    return units[indices], categories[indices]


def _encode_tuples(
    domain: BayesianOptimizationDomain, units: Array, categories: Array, /
) -> Array:
    if domain.categorical is None:
        return units
    flat = categories.reshape((-1,))
    indices = jnp.stack(domain.categorical.unravel_index(flat), axis=1).astype(
        units.dtype
    )
    return jnp.concatenate((units.reshape((flat.shape[0], -1)), indices), axis=1).reshape(
        units.shape[:2] + (domain.encoded_dimension,)
    )


def _gp_posterior(
    train_points: Array,
    train_values: Array,
    query_points: Array,
    valid: Array,
    state: GaussianProcessLikelihoodState,
    /,
) -> tuple[Array, Array, Array]:
    host_valid = np.asarray(jax.device_get(valid))
    points = train_points[host_valid]
    values = train_values[host_valid]
    noise = jnp.broadcast_to(state.noise_scale, (train_points.shape[0],))[host_valid]
    covariance = state.kernel.matrix(points, points) + jnp.diag(
        noise * noise + state.jitter
    )
    cholesky = jnp.linalg.cholesky(covariance)
    cross = state.kernel.matrix(query_points, points)
    alpha = jsp.linalg.solve_triangular(cholesky, values, lower=True)
    alpha = jsp.linalg.solve_triangular(cholesky.T, alpha, lower=False)
    mean = cross @ alpha
    whitened = jsp.linalg.solve_triangular(cholesky, cross.T, lower=True)
    query_covariance = (
        state.kernel.matrix(query_points, query_points) - whitened.T @ whitened
    )
    query_covariance = 0.5 * (query_covariance + query_covariance.T)
    usable = jnp.all(jnp.isfinite(cholesky)) & jnp.all(jnp.isfinite(query_covariance))
    return mean, query_covariance, usable


def _acquisition_scores(
    encoded: Array,
    objectives: Array,
    constraints: Array,
    objective_valid: Array,
    constraints_valid: Array,
    feasible: Array,
    tuple_units: Array,
    tuple_categories: Array,
    domain: BayesianOptimizationDomain,
    pending: tuple[BayesianOptimizationPoint, ...],
    plan: GaussianProcessBayesianOptimization,
    key: Array,
    /,
) -> tuple[Array, Array]:
    candidate_encoded = _encode_tuples(domain, tuple_units, tuple_categories)
    flat_candidates = candidate_encoded.reshape((-1, domain.encoded_dimension))
    pending_encoded = (
        jnp.stack(tuple(point.encoded for point in pending))
        if pending
        else jnp.empty((0, domain.encoded_dimension), dtype=encoded.dtype)
    )
    tuple_count = plan.candidate_tuple_count
    q = plan.batch_size
    keys = jr.split(key, 2 * (1 + len(plan.constraint_surrogates)))
    objective_means, objective_covariance, objective_usable = _pending_fantasy_posterior(
        encoded,
        objectives,
        flat_candidates,
        pending_encoded,
        objective_valid,
        plan.objective_surrogate,
        keys[0],
        fantasy_count=plan.fantasy_count,
    )
    objective_means = objective_means.reshape((plan.fantasy_count, tuple_count, q))
    objective_covariance = objective_covariance.reshape((tuple_count, q, tuple_count, q))
    objective_blocks = jnp.stack(
        tuple(objective_covariance[index, :, index, :] for index in range(tuple_count))
    )
    normal = jr.normal(
        keys[1],
        (plan.fantasy_count, tuple_count, q),
        dtype=objectives.dtype,
    )
    objective_factor = jax.vmap(jnp.linalg.cholesky)(objective_blocks)
    objective_samples = objective_means + oe.contract(
        "tij,ftj->fti", objective_factor, normal
    )
    has_feasible = jnp.any(feasible)
    incumbent = jnp.min(jnp.where(feasible, objectives, jnp.inf))
    improvement = jnp.maximum(incumbent - jnp.min(objective_samples, axis=-1), 0.0)
    sample_feasible = jnp.ones((plan.fantasy_count, tuple_count), dtype=bool)
    usable = objective_usable
    for index, state in enumerate(plan.constraint_surrogates):
        means, covariance, child_usable = _pending_fantasy_posterior(
            encoded,
            constraints[:, index],
            flat_candidates,
            pending_encoded,
            constraints_valid,
            state,
            keys[2 + 2 * index],
            fantasy_count=plan.fantasy_count,
        )
        means = means.reshape((plan.fantasy_count, tuple_count, q))
        covariance = covariance.reshape((tuple_count, q, tuple_count, q))
        blocks = jnp.stack(
            tuple(covariance[item, :, item, :] for item in range(tuple_count))
        )
        factor = jax.vmap(jnp.linalg.cholesky)(blocks)
        child_normal = jr.normal(
            keys[3 + 2 * index],
            (plan.fantasy_count, tuple_count, q),
            dtype=objectives.dtype,
        )
        samples = means + oe.contract("tij,ftj->fti", factor, child_normal)
        sample_feasible = sample_feasible & jnp.all(samples <= 0.0, axis=-1)
        usable = usable & child_usable
    values = jnp.where(
        has_feasible | (len(plan.constraint_surrogates) == 0),
        jnp.where(sample_feasible, improvement, 0.0),
        sample_feasible.astype(objectives.dtype),
    )
    estimates = jnp.mean(values, axis=0)
    errors = jnp.std(values, axis=0, ddof=1) / jnp.sqrt(plan.fantasy_count)
    return (
        jnp.where(usable, estimates, -jnp.inf),
        jnp.where(usable, errors, jnp.nan),
    )


def _pending_fantasy_posterior(
    train_points: Array,
    train_values: Array,
    query_points: Array,
    pending_points: Array,
    valid: Array,
    state: GaussianProcessLikelihoodState,
    key: Array,
    /,
    *,
    fantasy_count: int,
) -> tuple[Array, Array, Array]:
    if pending_points.shape[0] == 0:
        mean, covariance, usable = _gp_posterior(
            train_points, train_values, query_points, valid, state
        )
        return (
            jnp.broadcast_to(mean, (fantasy_count,) + mean.shape),
            covariance,
            usable,
        )
    joint = jnp.concatenate((pending_points, query_points), axis=0)
    mean, covariance, usable = _gp_posterior(
        train_points, train_values, joint, valid, state
    )
    pending_count = int(pending_points.shape[0])
    pending_mean = mean[:pending_count]
    query_mean = mean[pending_count:]
    pending_covariance = covariance[:pending_count, :pending_count]
    pending_noise = jnp.broadcast_to(state.noise_scale, (pending_count,))
    pending_covariance = pending_covariance + jnp.diag(pending_noise * pending_noise)
    query_pending = covariance[pending_count:, :pending_count]
    query_covariance = covariance[pending_count:, pending_count:]
    factor = jnp.linalg.cholesky(pending_covariance)
    fantasy_values = pending_mean + oe.contract(
        "ij,fj->fi",
        factor,
        jr.normal(
            key,
            (fantasy_count, pending_count),
            dtype=mean.dtype,
        ),
    )
    centered = fantasy_values - pending_mean
    solved_centered = jsp.linalg.solve_triangular(factor, centered.T, lower=True)
    solved_centered = jsp.linalg.solve_triangular(
        factor.T, solved_centered, lower=False
    ).T
    conditional_means = query_mean + oe.contract(
        "qp,fp->fq", query_pending, solved_centered
    )
    solved_cross = jsp.linalg.solve_triangular(factor, query_pending.T, lower=True)
    conditional_covariance = query_covariance - solved_cross.T @ solved_cross
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    child_usable = (
        usable
        & jnp.all(jnp.isfinite(factor))
        & jnp.all(jnp.isfinite(conditional_means))
        & jnp.all(jnp.isfinite(conditional_covariance))
    )
    return conditional_means, conditional_covariance, child_usable


def _separated_tuples(
    candidates: Array, occupied: Array, /, *, minimum_separation: float
) -> Array:
    if occupied.shape[0] == 0:
        external = jnp.ones((candidates.shape[0],), dtype=bool)
    else:
        squared = jnp.sum(
            (candidates[:, :, None, :] - occupied[None, None, :, :]) ** 2,
            axis=-1,
        )
        external = jnp.all(jnp.min(squared, axis=-1) > minimum_separation**2, axis=-1)
    if candidates.shape[1] == 1:
        return external
    pairwise = jnp.sum(
        (candidates[:, :, None, :] - candidates[:, None, :, :]) ** 2, axis=-1
    )
    diagonal = jnp.eye(candidates.shape[1], dtype=bool)[None, :, :]
    internal = jnp.all(
        jnp.where(diagonal, jnp.inf, pairwise) > minimum_separation**2, axis=(1, 2)
    )
    return external & internal


def _space_filling_tuple(
    candidates: Array, occupied: Array, /, *, eligible: Array
) -> Array:
    squared = jnp.sum(
        jnp.square(candidates[:, :, None, :] - occupied[None, None, :, :]), axis=-1
    )
    distances = jnp.min(squared, axis=(1, 2))
    return jnp.argmax(jnp.where(eligible, distances, -jnp.inf))


def _positive_integer(value: int, /, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_real(value: Real, /, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


__all__ = [
    "BAYESIAN_OPTIMIZATION_ACQUISITION",
    "BAYESIAN_OPTIMIZATION_FEASIBILITY_FIRST",
    "BAYESIAN_OPTIMIZATION_INITIAL",
    "BAYESIAN_OPTIMIZATION_SPACE_FILLING",
    "BayesianOptimizationDomain",
    "BayesianOptimizationPoint",
    "BayesianOptimizationProblem",
    "BayesianOptimizationResult",
    "GaussianProcessBayesianOptimization",
    "bayesian_optimize",
]
