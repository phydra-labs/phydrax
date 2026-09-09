# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Bounded exact-GP noisy qHVI for two or three correlated objectives.

Observation kernels and noise scales describe the original physical outputs.
Directions and positive scales affect only Pareto geometry. Constraint GPs are
explicitly independent of each other and of the correlated objective GP.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any, cast, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
from phydrax.linalg import HermitianSpectrum

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..optim._pareto import _hypervolume, nondominated_mask
from ._bayesian_optimization import (
    _candidate_tuples,
    _encode_tuples,
    _initial_candidate_points,
    _pending_encodings,
    _positive_integer,
    _positive_real,
    _separated_tuples,
    BAYESIAN_OPTIMIZATION_ACQUISITION,
    BAYESIAN_OPTIMIZATION_INITIAL,
    BayesianOptimizationDomain,
    BayesianOptimizationPoint,
)
from ._gp_computation_backend import _factorize_positive, _solve_columns, _solve_vector
from ._gp_likelihood import GaussianProcessLikelihoodState
from ._gp_multioutput import (
    MultiOutputDesign,
    MultiOutputGaussianProcessDiscrepancy,
    MultiOutputGaussianProcessLikelihoodState,
)


class MultiObjectiveBayesianOptimizationProblem(StrictModule):
    """Physical vector observations with stable geometry and keyed evaluators.

    ``objective(point, key)`` returns a vector in ``objective_names`` order;
    each ``constraint(point, key)`` returns a scalar inequality g <= 0. Keys
    are independent children of each recorded evaluation key. An optional
    ``validity(point)`` guard prevents invalid physical evaluations entirely.
    Nonfinite observations are recorded but never become GP training points.
    ``reference`` is in physical units, and directions are ``min`` or ``max``.
    Pending points are unevaluated members of the sampled attained set.
    """

    domain: BayesianOptimizationDomain
    objective: Callable = eqx.field(static=True)
    constraints: tuple[Callable, ...] = eqx.field(static=True)
    validity: Callable | None = eqx.field(static=True)
    objective_names: tuple[str, ...] = eqx.field(static=True)
    directions: tuple[str, ...] = eqx.field(static=True)
    scales: Array
    reference: Array
    pending: tuple[BayesianOptimizationPoint, ...]

    def __init__(
        self,
        objective: Callable[[BayesianOptimizationPoint, Array], ArrayLike],
        domain: BayesianOptimizationDomain,
        /,
        *,
        objective_names: Sequence[str],
        directions: Sequence[Literal["min", "max"]],
        scales: ArrayLike,
        reference: ArrayLike,
        constraints: Sequence[
            Callable[[BayesianOptimizationPoint, Array], ArrayLike]
        ] = (),
        validity: Callable[[BayesianOptimizationPoint], ArrayLike] | None = None,
        pending: Sequence[BayesianOptimizationPoint] = (),
    ):
        names, direction_tuple = tuple(objective_names), tuple(directions)
        if len(names) not in (2, 3):
            raise ValueError(
                "Multiobjective BO supports exactly two or three objectives."
            )
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("objective_names must be distinct nonempty strings.")
        if len(direction_tuple) != len(names) or any(
            value not in ("min", "max") for value in direction_tuple
        ):
            raise ValueError("directions must contain min or max for every objective.")
        if not isinstance(domain, BayesianOptimizationDomain):
            raise TypeError("domain must be a BayesianOptimizationDomain.")
        constraint_tuple = tuple(constraints)
        if not callable(objective) or any(
            not callable(value) for value in constraint_tuple
        ):
            raise TypeError("Objective and constraints must be callable.")
        if validity is not None and not callable(validity):
            raise TypeError("validity must be callable or None.")
        scale_array, ref = (
            jnp.asarray(scales, dtype=float),
            jnp.asarray(reference, dtype=float),
        )
        if scale_array.shape != (len(names),) or ref.shape != (len(names),):
            raise ValueError("scales and reference must align with objective_names.")
        if not bool(jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))):
            raise ValueError("scales must be finite and strictly positive.")
        if not bool(jnp.all(jnp.isfinite(ref))):
            raise ValueError("reference must be finite.")
        if not bool(jnp.all(jnp.isfinite(ref / scale_array))):
            raise ValueError("reference must remain finite after objective scaling.")
        pending_tuple = tuple(pending)
        for point in pending_tuple:
            if not isinstance(point, BayesianOptimizationPoint):
                raise TypeError("pending must contain BayesianOptimizationPoint values.")
            _validate_pending(domain, point)
        self.domain = domain
        self.objective = objective
        self.constraints = constraint_tuple
        self.validity = validity
        self.objective_names = names
        self.directions = direction_tuple
        self.scales = scale_array
        self.reference = ref
        self.pending = pending_tuple

    def canonical(self, values: ArrayLike, /) -> Array:
        """Convert physical values to the fixed dimensionless minimization frame."""
        signs = jnp.asarray(
            tuple(1.0 if value == "min" else -1.0 for value in self.directions)
        )
        return jnp.asarray(values) * signs / self.scales


class GaussianProcessMultiObjectiveBayesianOptimization(StrictModule):
    """Finite candidate-tuple search with shared noisy-baseline Monte Carlo draws.

    All capacities are hard preflight limits, not truncation policies. The byte
    bound is a conservative bound on numeric arrays owned by this algorithm
    (including covariance/solve/spectral scratch), not the JAX runtime allocator
    or executable cache. ``psd_tolerance`` is an explicit relative spectral rank
    tolerance; latent sampling adds no jitter. Only the likelihood states'
    declared jitter regularizes the observed-data factorization.
    """

    objective_surrogate: MultiOutputGaussianProcessLikelihoodState
    constraint_surrogates: tuple[GaussianProcessLikelihoodState, ...]
    max_evaluations: int = eqx.field(static=True)
    initial_evaluations: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    candidate_tuple_count: int = eqx.field(static=True)
    fantasy_count: int = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    psd_tolerance: float = eqx.field(static=True)
    max_training_points: int = eqx.field(static=True)
    max_pending_points: int = eqx.field(static=True)
    max_baseline_points: int = eqx.field(static=True)
    max_hypervolume_points: int = eqx.field(static=True)
    max_working_bytes: int = eqx.field(static=True)
    max_hypervolume_work: int = eqx.field(static=True)

    def __init__(
        self,
        max_evaluations: int,
        /,
        *,
        objective_surrogate: MultiOutputGaussianProcessLikelihoodState,
        constraint_surrogates: Sequence[GaussianProcessLikelihoodState] = (),
        initial_evaluations: int = 8,
        batch_size: int = 1,
        candidate_tuple_count: int = 128,
        fantasy_count: int = 128,
        minimum_separation: float = 1e-6,
        psd_tolerance: float = 1e-6,
        max_training_points: int = 128,
        max_pending_points: int = 32,
        max_baseline_points: int = 160,
        max_hypervolume_points: int = 192,
        max_working_bytes: int = 256 * 1024 * 1024,
        max_hypervolume_work: int = 1_000_000_000,
    ):
        self.max_evaluations = _positive_integer(max_evaluations, name="max_evaluations")
        self.initial_evaluations = _positive_integer(
            initial_evaluations, name="initial_evaluations"
        )
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        self.candidate_tuple_count = _positive_integer(
            candidate_tuple_count, name="candidate_tuple_count"
        )
        self.fantasy_count = _positive_integer(fantasy_count, name="fantasy_count")
        self.max_training_points = _positive_integer(
            max_training_points, name="max_training_points"
        )
        self.max_pending_points = _positive_integer(
            max_pending_points, name="max_pending_points"
        )
        self.max_baseline_points = _positive_integer(
            max_baseline_points, name="max_baseline_points"
        )
        self.max_hypervolume_points = _positive_integer(
            max_hypervolume_points, name="max_hypervolume_points"
        )
        self.max_working_bytes = _positive_integer(
            max_working_bytes, name="max_working_bytes"
        )
        self.max_hypervolume_work = _positive_integer(
            max_hypervolume_work, name="max_hypervolume_work"
        )
        self.minimum_separation = _positive_real(
            cast(Real, minimum_separation),
            name="minimum_separation",
        )
        self.psd_tolerance = _positive_real(
            cast(Real, psd_tolerance),
            name="psd_tolerance",
        )
        if self.fantasy_count < 2:
            raise ValueError("fantasy_count must be at least two for Monte Carlo error.")
        if (
            self.initial_evaluations > self.max_evaluations
            or self.batch_size > self.max_evaluations
        ):
            raise ValueError("Initial and batch counts cannot exceed max_evaluations.")
        if not isinstance(objective_surrogate, MultiOutputGaussianProcessLikelihoodState):
            raise TypeError(
                "objective_surrogate must be a multi-output GP likelihood state."
            )
        if objective_surrogate.noise_layout != "output":
            raise ValueError("Objective noise must use the fixed output layout.")
        constraints = tuple(constraint_surrogates)
        if any(
            not isinstance(state, GaussianProcessLikelihoodState) for state in constraints
        ):
            raise TypeError(
                "constraint_surrogates must contain scalar GP likelihood states."
            )
        if any(
            state.noise_scale.ndim != 0 or state.kernel.input_ndim != 1
            for state in constraints
        ):
            raise ValueError("Constraint GPs need scalar noise and vector-input kernels.")
        self.objective_surrogate = objective_surrogate
        self.constraint_surrogates = constraints


class MultiObjectiveBayesianOptimizationObservation(StrictModule):
    """One attempted physical evaluation; invalid rows are never training data."""

    point: BayesianOptimizationPoint
    objectives: Array
    constraints: Array
    valid: Array
    feasible: Array
    evaluation_key: Array
    proposal_kind: int = eqx.field(static=True)


class MultiObjectiveBayesianOptimizationResult(StrictModule):
    """Noisy observed Pareto set, not a latent-front or global-optimum certificate."""

    observations: tuple[MultiObjectiveBayesianOptimizationObservation, ...]
    evaluated_encoded: Array
    objectives: Array
    constraints: Array
    valid: Array
    feasible: Array
    observed_pareto_mask: Array
    observed_pareto_objectives: Array
    observed_pareto_points: tuple[BayesianOptimizationPoint, ...]
    observed_hypervolume: Array
    acquisition_estimates: Array
    acquisition_standard_errors: Array
    acquisition_batch_sizes: Array
    key: Array
    final_key: Array
    initial_key: Array
    evaluation_keys: Array
    candidate_keys: Array
    fantasy_keys: Array
    pending_encoded: Array
    objective_names: tuple[str, ...] = eqx.field(static=True)
    directions: tuple[str, ...] = eqx.field(static=True)
    scales: Array
    reference: Array
    objective_surrogate: MultiOutputGaussianProcessLikelihoodState
    constraint_surrogates: tuple[GaussianProcessLikelihoodState, ...]
    pending_id: str = eqx.field(static=True)
    work_id: str = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    invalid_evaluation_count: int = eqx.field(static=True)
    pending_count: int = eqx.field(static=True)
    scored_tuple_count: int = eqx.field(static=True)
    estimated_peak_bytes: int = eqx.field(static=True)
    hypervolume_work_bound: int = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    globally_optimal: bool = eqx.field(static=True)


def _validate_pending(
    domain: BayesianOptimizationDomain, point: BayesianOptimizationPoint, /
) -> None:
    encoded = point.encoded
    if encoded.shape != (domain.encoded_dimension,) or not bool(
        jnp.all(jnp.isfinite(encoded))
    ):
        raise ValueError("Pending encodings must be finite and match the domain.")
    continuous = encoded[: domain.continuous_dimension]
    if not bool(jnp.all((continuous >= 0) & (continuous <= 1))):
        raise ValueError("Pending continuous coordinates must lie in the unit box.")
    tolerance = 8 * jnp.finfo(continuous.dtype).eps
    if not bool(
        jnp.allclose(
            domain.to_unit(point.continuous), continuous, rtol=tolerance, atol=tolerance
        )
    ):
        raise ValueError("Pending physical and encoded continuous coordinates disagree.")
    categorical = encoded[domain.continuous_dimension :]
    if point.categorical_index.shape != (domain.categorical_dimension,) or not bool(
        jnp.all(categorical == point.categorical_index)
    ):
        raise ValueError("Pending categorical indices disagree with their encoding.")
    if domain.categorical is not None:
        sizes = jnp.asarray(domain.categorical.product_shape)
        if not bool(
            jnp.all(
                (categorical >= 0)
                & (categorical < sizes)
                & (categorical == jnp.floor(categorical))
            )
        ):
            raise ValueError(
                "Pending categorical indices must belong to the finite product."
            )


def _preflight(problem, plan, /) -> tuple[int, int]:
    """Bound the largest epoch before candidate generation or physical calls."""
    if plan.objective_surrogate.kernel.output_names != problem.objective_names:
        raise ValueError("Objective surrogate output names/order must match the problem.")
    if len(plan.constraint_surrogates) != len(problem.constraints):
        raise ValueError("constraint_surrogates must align with problem constraints.")
    n, p, m, c = (
        plan.max_evaluations,
        len(problem.pending),
        len(problem.objective_names),
        len(problem.constraints),
    )
    b, h, q, t, s = (
        n + p,
        n + p,
        plan.batch_size,
        plan.candidate_tuple_count,
        plan.fantasy_count,
    )
    for count, limit, name in (
        (n, plan.max_training_points, "training"),
        (p, plan.max_pending_points, "pending"),
        (b, plan.max_baseline_points, "baseline"),
        (h, plan.max_hypervolume_points, "hypervolume"),
    ):
        if count > limit:
            raise ValueError(f"Multiobjective BO {name} point capacity exceeded.")
    d = problem.domain.encoded_dimension
    initial_pool = (
        problem.domain.categorical.size
        if problem.domain.continuous_dimension == 0
        else max(t, plan.initial_evaluations + p)
    )
    # Owned GP matrices, Cholesky/solve/eigensystem scratch, shared latent draws,
    # one tuple's conditional geometry, and candidate-separation temporaries.
    numeric_entries = (
        16 * (m * m + c) * (n * n + b * b + q * q + n * b + b * q)
        + 8 * s * (b + q) * (m + c)
        + 8 * t * q * (b + q) * max(d, 1)
        + 8 * initial_pool * (plan.initial_evaluations + p + 1) * max(d, 1)
        + 8 * n * (d + m + c + n)
    )
    itemsize = max(
        problem.domain.continuous_initial.dtype.itemsize, problem.scales.dtype.itemsize, 8
    )
    peak = numeric_entries * itemsize
    if peak > plan.max_working_bytes:
        raise ValueError(
            f"Multiobjective BO requires at most {peak} owned working bytes, exceeding max_working_bytes."
        )
    epochs = (n - plan.initial_evaluations + q - 1) // q
    # Each attained-set HV uses O(h log h) in 2D or O(h squared) in 3D.
    per_hv = h * max(1, h.bit_length()) if m == 2 else h * h
    work = s * (t + 1) * epochs * per_hv
    if work > plan.max_hypervolume_work:
        raise ValueError("Multiobjective BO max_hypervolume_work exceeded.")
    return peak, work


class _GPQuery(NamedTuple):
    design: object
    cross: Array
    solved: Array
    mean: Array
    covariance: Array


class _PreparedGP:
    """Exact multi-output/scalar GP using one native observation factor per epoch."""

    def __init__(self, points, values, state, max_bytes):
        self.state = state
        self.multioutput = isinstance(state, MultiOutputGaussianProcessLikelihoodState)
        self.train = self.design(points)
        if self.multioutput:
            discrepancy = MultiOutputGaussianProcessDiscrepancy(self.train, values)
            residual = discrepancy.residual(jnp.zeros_like(discrepancy.observations))
            noise = state.observation_noise(self.train)
        else:
            residual = values
            noise = jnp.broadcast_to(state.noise_scale, values.shape)
        covariance = state.kernel.matrix(self.train, self.train) + jnp.diag(
            noise * noise + state.jitter
        )
        self.factor = _factorize_positive(
            covariance,
            name="multiobjective-bo-observations",
            max_factorization_bytes=max_bytes,
        )
        self.alpha = _solve_vector(self.factor, residual)

    def design(self, points):
        return (
            MultiOutputDesign.from_dense(
                points, output_names=self.state.kernel.output_names
            )
            if self.multioutput
            else points
        )

    def query(self, points):
        design = self.design(points)
        cross = self.state.kernel.matrix(design, self.train)
        solved, successful = _solve_columns(self.factor, cross.T)
        solved = eqx.error_if(
            solved, ~successful, "Multiobjective GP posterior solve failed."
        )
        mean = ein.contract("qn,n->q", cross, self.alpha)
        covariance = self.state.kernel.matrix(design, design) - cross @ solved
        return _GPQuery(design, cross, solved, mean, 0.5 * (covariance + covariance.T))

    def cross_covariance(self, left, right):
        return (
            self.state.kernel.matrix(left.design, right.design)
            - left.cross @ right.solved
        )


def _psd_factors(
    covariance: Array,
    tolerance: float,
    /,
    *,
    reference_scale: Array | None = None,
) -> tuple[Array, Array]:
    """Rank-revealing square root and inverse-root; no added latent variance."""
    spectrum = HermitianSpectrum(covariance, tolerance=tolerance)
    eigenvalues = spectrum.eigenvalues
    spectral_scale = jnp.max(jnp.abs(eigenvalues))
    scale = (
        spectral_scale
        if reference_scale is None
        else jnp.maximum(spectral_scale, jnp.abs(reference_scale))
    )
    threshold = tolerance * scale
    eigenvalues = eqx.error_if(
        eigenvalues,
        ~spectrum.valid | (jnp.min(eigenvalues) < -threshold),
        "Latent GP covariance is not positive semidefinite within psd_tolerance.",
    )
    active = eigenvalues > threshold
    roots = jnp.sqrt(jnp.where(active, eigenvalues, 1.0))
    factor = spectrum.eigenvectors * jnp.where(active, roots, 0.0)[None, :]
    inverse_root = spectrum.eigenvectors * jnp.where(active, 1.0 / roots, 0.0)[None, :]
    return factor, inverse_root


class _SampledBaseline(NamedTuple):
    query: _GPQuery
    draws: Array
    inverse_root: Array


def _sample_baseline(gp, points, key, plan):
    query = gp.query(points)
    factor, inverse_root = _psd_factors(query.covariance, plan.psd_tolerance)
    noise = jr.normal(
        key, (plan.fantasy_count, query.mean.shape[0]), dtype=query.mean.dtype
    )
    return _SampledBaseline(query, query.mean + noise @ factor.T, inverse_root)


def _sample_conditional(gp, baseline, points, noise, plan):
    query = gp.query(points)
    cross = gp.cross_covariance(query, baseline.query)
    projected = cross @ baseline.inverse_root
    projected_covariance = projected @ projected.T
    conditional_covariance = query.covariance - projected_covariance
    reference_scale = jnp.maximum(
        jnp.max(jnp.abs(query.covariance)),
        jnp.max(jnp.abs(projected_covariance)),
    )
    factor, _ = _psd_factors(
        conditional_covariance,
        plan.psd_tolerance,
        reference_scale=reference_scale,
    )
    regression = projected @ baseline.inverse_root.T
    conditional_mean = query.mean + (baseline.draws - baseline.query.mean) @ regression.T
    return conditional_mean + noise @ factor.T


def _sample_hvi(
    baseline,
    candidates,
    baseline_feasible,
    candidate_feasible,
    reference,
    baseline_hv=None,
):
    """Feasibility filters individual points, never gates an entire q tuple."""
    if baseline_hv is None:
        baseline_hv = jax.vmap(_hypervolume, in_axes=(0, None, 0))(
            baseline, reference, baseline_feasible
        )
    attained = jnp.concatenate((baseline, candidates), axis=1)
    feasible = jnp.concatenate((baseline_feasible, candidate_feasible), axis=1)
    total = jax.vmap(_hypervolume, in_axes=(0, None, 0))(attained, reference, feasible)
    return jnp.maximum(total - baseline_hv, 0.0)


def _acquisition_scores(
    problem, plan, encoded, objectives, constraints, valid, candidates, key
):
    """Shared latent B union P draws, then only q-by-q conditional covariance."""
    active = np.asarray(jax.device_get(valid), dtype=bool)
    train_points, train_values = encoded[active], objectives[active]
    pending = _pending_encodings(problem.domain, problem.pending)
    baseline_points = jnp.concatenate((train_points, pending), axis=0)
    b, m, q = baseline_points.shape[0], len(problem.objective_names), candidates.shape[1]
    gp = _PreparedGP(
        train_points, train_values, plan.objective_surrogate, plan.max_working_bytes
    )
    baseline_key, candidate_key = jr.split(key)
    baseline = _sample_baseline(gp, baseline_points, jr.fold_in(baseline_key, 0), plan)
    baseline_objectives = problem.canonical(
        baseline.draws.reshape((plan.fantasy_count, b, m))
    )
    baseline_feasible = jnp.ones((plan.fantasy_count, b), dtype=bool)
    constraint_models = []
    for index, state in enumerate(plan.constraint_surrogates):
        constraint_gp = _PreparedGP(
            train_points, constraints[active, index], state, plan.max_working_bytes
        )
        constraint_baseline = _sample_baseline(
            constraint_gp, baseline_points, jr.fold_in(baseline_key, index + 1), plan
        )
        constraint_models.append((constraint_gp, constraint_baseline))
        baseline_feasible = baseline_feasible & (constraint_baseline.draws <= 0.0)
    reference = problem.canonical(problem.reference)
    baseline_hv = jax.vmap(_hypervolume, in_axes=(0, None, 0))(
        baseline_objectives, reference, baseline_feasible
    )
    # Common residual normals reduce comparison noise, while each tuple keeps its
    # own conditional covariance and correlation with the same attained-set draws.
    objective_noise = jr.normal(
        jr.fold_in(candidate_key, 0),
        (plan.fantasy_count, q * m),
        dtype=baseline.draws.dtype,
    )
    constraint_noise = tuple(
        jr.normal(
            jr.fold_in(candidate_key, index + 1),
            (plan.fantasy_count, q),
            dtype=baseline.draws.dtype,
        )
        for index in range(len(constraint_models))
    )
    estimates, errors = [], []
    for points in candidates:
        draws = _sample_conditional(gp, baseline, points, objective_noise, plan)
        candidate_objectives = problem.canonical(
            draws.reshape((plan.fantasy_count, q, m))
        )
        candidate_feasible = jnp.ones((plan.fantasy_count, q), dtype=bool)
        for (constraint_gp, constraint_baseline), noise in zip(
            constraint_models, constraint_noise, strict=True
        ):
            constraint_draws = _sample_conditional(
                constraint_gp, constraint_baseline, points, noise, plan
            )
            candidate_feasible = candidate_feasible & (constraint_draws <= 0.0)
        improvement = _sample_hvi(
            baseline_objectives,
            candidate_objectives,
            baseline_feasible,
            candidate_feasible,
            reference,
            baseline_hv,
        )
        estimates.append(jnp.mean(improvement))
        errors.append(jnp.std(improvement, ddof=1) / jnp.sqrt(plan.fantasy_count))
    return jnp.stack(estimates), jnp.stack(errors)


def _evaluate(problem, unit, category, key, proposal):
    point = problem.domain.decode(unit, category)
    physical_valid = jnp.asarray(
        True if problem.validity is None else problem.validity(point)
    )
    if physical_valid.shape != () or physical_valid.dtype != jnp.bool_:
        raise ValueError("validity must return a boolean scalar.")
    dtype = problem.scales.dtype
    if bool(physical_valid):
        values = jnp.asarray(problem.objective(point, jr.fold_in(key, 0)))
        if values.shape != (len(problem.objective_names),) or jnp.issubdtype(
            values.dtype, jnp.complexfloating
        ):
            raise ValueError(
                "objective must return a real vector in objective_names order."
            )
        values = values.astype(dtype)
        constraint_values = []
        for index, constraint in enumerate(problem.constraints):
            value = jnp.asarray(constraint(point, jr.fold_in(key, index + 1)))
            if value.shape != () or jnp.issubdtype(value.dtype, jnp.complexfloating):
                raise ValueError("Each constraint must return one real scalar.")
            constraint_values.append(value.astype(dtype))
        constraints = (
            jnp.stack(constraint_values)
            if constraint_values
            else jnp.empty((0,), dtype=dtype)
        )
    else:
        values = jnp.full((len(problem.objective_names),), jnp.nan, dtype=dtype)
        constraints = jnp.full((len(problem.constraints),), jnp.nan, dtype=dtype)
    valid = (
        physical_valid
        & jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(constraints))
    )
    return MultiObjectiveBayesianOptimizationObservation(
        point=point,
        objectives=values,
        constraints=constraints,
        valid=valid,
        feasible=valid & jnp.all(constraints <= 0.0),
        evaluation_key=key,
        proposal_kind=proposal,
    )


def multiobjective_bayesian_optimize(
    problem: MultiObjectiveBayesianOptimizationProblem,
    plan: GaussianProcessMultiObjectiveBayesianOptimization,
    key: Array,
    /,
) -> MultiObjectiveBayesianOptimizationResult:
    """Search a finite budget; report noisy observations without an optimum claim."""
    if not isinstance(problem, MultiObjectiveBayesianOptimizationProblem):
        raise TypeError("problem must be a MultiObjectiveBayesianOptimizationProblem.")
    if not isinstance(plan, GaussianProcessMultiObjectiveBayesianOptimization):
        raise TypeError(
            "plan must be a GaussianProcessMultiObjectiveBayesianOptimization."
        )
    peak_bytes, hv_work = _preflight(problem, plan)
    root_key = key
    key, initial_key = jr.split(key)
    units, categories = _initial_candidate_points(
        problem.domain,
        initial_key,
        problem.pending,
        cast(Any, plan),
    )
    records = []
    for unit, category in zip(units, categories, strict=True):
        key, evaluation_key = jr.split(key)
        records.append(
            _evaluate(
                problem, unit, category, evaluation_key, BAYESIAN_OPTIMIZATION_INITIAL
            )
        )
    estimates, errors, batch_sizes, candidate_keys, fantasy_keys = [], [], [], [], []
    pending = _pending_encodings(problem.domain, problem.pending)
    termination, scored_count = "evaluation_budget_exhausted", 0
    while len(records) < plan.max_evaluations:
        valid = jnp.stack(tuple(record.valid for record in records))
        if not bool(jnp.any(valid)):
            termination = "no_valid_training_observations"
            break
        encoded = jnp.stack(tuple(record.point.encoded for record in records))
        objectives = jnp.stack(tuple(record.objectives for record in records))
        constraints = jnp.stack(tuple(record.constraints for record in records))
        key, candidate_key, fantasy_key = jr.split(key, 3)
        candidate_keys.append(candidate_key)
        fantasy_keys.append(fantasy_key)
        tuple_units, tuple_categories = _candidate_tuples(
            problem.domain,
            candidate_key,
            cast(Any, plan),
        )
        q = min(plan.batch_size, plan.max_evaluations - len(records))
        tuple_units, tuple_categories = tuple_units[:, :q], tuple_categories[:, :q]
        candidates = _encode_tuples(problem.domain, tuple_units, tuple_categories)
        occupied = jnp.concatenate((encoded, pending), axis=0)
        separated = _separated_tuples(
            candidates, occupied, minimum_separation=plan.minimum_separation
        )
        eligible = np.flatnonzero(np.asarray(jax.device_get(separated)))
        if eligible.size == 0:
            termination = "candidate_pool_exhausted"
            break
        scores, standard_errors = _acquisition_scores(
            problem,
            plan,
            encoded,
            objectives,
            constraints,
            valid,
            candidates[eligible],
            fantasy_key,
        )
        scored_count += int(eligible.size)
        if not bool(
            jnp.all(jnp.isfinite(scores)) & jnp.all(jnp.isfinite(standard_errors))
        ):
            raise ValueError(
                "Nonfinite multiobjective acquisition; no physical proposal was evaluated."
            )
        selected_local = int(jnp.argmax(scores))
        selected = int(eligible[selected_local])
        estimates.append(scores[selected_local])
        errors.append(standard_errors[selected_local])
        batch_sizes.append(q)
        for member in range(q):
            key, evaluation_key = jr.split(key)
            records.append(
                _evaluate(
                    problem,
                    tuple_units[selected, member],
                    tuple_categories[selected, member],
                    evaluation_key,
                    BAYESIAN_OPTIMIZATION_ACQUISITION,
                )
            )
    objectives = jnp.stack(tuple(record.objectives for record in records))
    constraints = jnp.stack(tuple(record.constraints for record in records))
    valid = jnp.stack(tuple(record.valid for record in records))
    feasible = jnp.stack(tuple(record.feasible for record in records))
    canonical = problem.canonical(objectives)
    pareto = nondominated_mask(canonical, feasible)
    indices = np.flatnonzero(np.asarray(jax.device_get(pareto)))
    pending_id = canonical_fingerprint(array_tree_fingerprint(problem.pending))
    work_id = canonical_fingerprint(
        {
            "key": array_tree_fingerprint(jr.key_data(root_key)),
            "domain": array_tree_fingerprint(
                (problem.domain.continuous_lower, problem.domain.continuous_upper)
            ),
            "categorical_domain": None
            if problem.domain.categorical is None
            else problem.domain.categorical.space_id,
            "objectives": problem.objective_names,
            "directions": problem.directions,
            "geometry": array_tree_fingerprint((problem.scales, problem.reference)),
            "surrogates": array_tree_fingerprint(
                (plan.objective_surrogate, plan.constraint_surrogates)
            ),
            "objective_kernel": plan.objective_surrogate.kernel.kernel_id,
            "constraint_kernels": tuple(
                state.kernel_id for state in plan.constraint_surrogates
            ),
            "pending": pending_id,
            "evaluations": len(records),
            "scored_tuples": scored_count,
            "observations": array_tree_fingerprint(
                (
                    tuple(record.point.encoded for record in records),
                    objectives,
                    constraints,
                    valid,
                )
            ),
            "plan": (
                plan.max_evaluations,
                plan.initial_evaluations,
                plan.batch_size,
                plan.candidate_tuple_count,
                plan.fantasy_count,
                plan.minimum_separation,
                plan.psd_tolerance,
            ),
            "limits": (
                plan.max_training_points,
                plan.max_pending_points,
                plan.max_baseline_points,
                plan.max_hypervolume_points,
                plan.max_working_bytes,
                plan.max_hypervolume_work,
            ),
        }
    )

    def stack_keys(values):
        return jnp.stack(values) if values else jr.split(root_key, 0)

    return MultiObjectiveBayesianOptimizationResult(
        observations=tuple(records),
        evaluated_encoded=jnp.stack(tuple(record.point.encoded for record in records)),
        objectives=objectives,
        constraints=constraints,
        valid=valid,
        feasible=feasible,
        observed_pareto_mask=pareto,
        observed_pareto_objectives=objectives[indices],
        observed_pareto_points=tuple(records[index].point for index in indices),
        observed_hypervolume=_hypervolume(
            canonical, problem.canonical(problem.reference), feasible
        ),
        acquisition_estimates=jnp.stack(estimates)
        if estimates
        else jnp.empty((0,), dtype=objectives.dtype),
        acquisition_standard_errors=jnp.stack(errors)
        if errors
        else jnp.empty((0,), dtype=objectives.dtype),
        acquisition_batch_sizes=jnp.asarray(batch_sizes, dtype=jnp.int32),
        key=root_key,
        final_key=key,
        initial_key=initial_key,
        evaluation_keys=jnp.stack(tuple(record.evaluation_key for record in records)),
        candidate_keys=stack_keys(candidate_keys),
        fantasy_keys=stack_keys(fantasy_keys),
        pending_encoded=pending,
        objective_names=problem.objective_names,
        directions=problem.directions,
        scales=problem.scales,
        reference=problem.reference,
        objective_surrogate=plan.objective_surrogate,
        constraint_surrogates=plan.constraint_surrogates,
        pending_id=pending_id,
        work_id=work_id,
        evaluation_count=len(records),
        invalid_evaluation_count=int(jnp.sum(~valid)),
        pending_count=len(problem.pending),
        scored_tuple_count=scored_count,
        estimated_peak_bytes=peak_bytes,
        hypervolume_work_bound=hv_work,
        termination_reason=termination,
        globally_optimal=False,
    )


__all__ = [
    "GaussianProcessMultiObjectiveBayesianOptimization",
    "MultiObjectiveBayesianOptimizationObservation",
    "MultiObjectiveBayesianOptimizationProblem",
    "MultiObjectiveBayesianOptimizationResult",
    "multiobjective_bayesian_optimize",
]
