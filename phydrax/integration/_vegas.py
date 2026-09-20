#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive stratified VEGAS with an immutable production grid."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import linear_interpolate
from .._strict import StrictModule
from .._trainable import NonTrainableState


class VegasStatus(IntEnum):
    """Terminal status for VEGAS preparation and production."""

    CONVERGED = 0
    NONFINITE_INTEGRAND = 1
    INVALID_GRID = 2
    INSUFFICIENT_PRODUCTION_REPLICATES = 3


class VegasPlan(StrictModule, NonTrainableState):
    """Static resource and adaptation policy for one rectangular integral."""

    lower: Array
    upper: Array
    dimension: int = eqx.field(static=True)
    bins: int = eqx.field(static=True)
    adaptation_iterations: int = eqx.field(static=True)
    adaptation_samples: int = eqx.field(static=True)
    production_iterations: int = eqx.field(static=True)
    production_samples: int = eqx.field(static=True)
    adaptation_power: float = eqx.field(static=True)
    minimum_bin_fraction: float = eqx.field(static=True)
    max_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        bins: int = 32,
        adaptation_iterations: int = 5,
        adaptation_samples: int = 4096,
        production_iterations: int = 8,
        production_samples: int = 8192,
        adaptation_power: float = 0.5,
        minimum_bin_fraction: float = 1.0e-6,
        max_evaluations: int = 1_000_000,
    ):
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if lower_.ndim != 1 or lower_.size == 0 or upper_.shape != lower_.shape:
            raise ValueError("VEGAS bounds must be nonempty aligned rank-one arrays.")
        if np.any(~np.isfinite(lower_)) or np.any(~np.isfinite(upper_)):
            raise ValueError("VEGAS bounds must be finite.")
        if np.any(upper_ <= lower_):
            raise ValueError("Every VEGAS upper bound must exceed its lower bound.")
        bins_ = int(bins)
        adapt_iterations = int(adaptation_iterations)
        adapt_samples = int(adaptation_samples)
        production_iterations_ = int(production_iterations)
        production_samples_ = int(production_samples)
        maximum = int(max_evaluations)
        if bins_ < 2:
            raise ValueError("VEGAS requires at least two bins per dimension.")
        if adapt_iterations < 0 or adapt_samples < bins_:
            raise ValueError("Adaptation samples must cover every bin.")
        if production_iterations_ < 1 or production_samples_ < bins_:
            raise ValueError("Production samples must cover every bin.")
        power = float(adaptation_power)
        floor = float(minimum_bin_fraction)
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError("adaptation_power must be finite and positive.")
        if not math.isfinite(floor) or floor <= 0.0 or floor * bins_ >= 1.0:
            raise ValueError("minimum_bin_fraction is inconsistent with bin capacity.")
        evaluations = (
            adapt_iterations * adapt_samples
            + production_iterations_ * production_samples_
        )
        if maximum < 1 or evaluations > maximum:
            raise ValueError("VEGAS evaluation request exceeds max_evaluations.")
        payload = {
            "kind": "adaptive-stratified-vegas",
            "lower": array_tree_fingerprint(lower_),
            "upper": array_tree_fingerprint(upper_),
            "bins": bins_,
            "adaptation_iterations": adapt_iterations,
            "adaptation_samples": adapt_samples,
            "production_iterations": production_iterations_,
            "production_samples": production_samples_,
            "adaptation_power": power,
            "minimum_bin_fraction": floor,
            "max_evaluations": maximum,
        }
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.dimension = lower_.size
        self.bins = bins_
        self.adaptation_iterations = adapt_iterations
        self.adaptation_samples = adapt_samples
        self.production_iterations = production_iterations_
        self.production_samples = production_samples_
        self.adaptation_power = power
        self.minimum_bin_fraction = floor
        self.max_evaluations = maximum
        self.plan_id = canonical_fingerprint(payload)


class VegasPreparationEvidence(StrictModule):
    """Fixed-shape evidence emitted while adapting a VEGAS grid."""

    iteration_estimates: Array
    marginal_weights: Array
    finite_iterations: Array
    status: Array
    num_evaluations: Array


class FrozenVegasGrid(StrictModule, NonTrainableState):
    """Prepared piecewise-linear importance transform used for production."""

    edges: Array
    evidence: VegasPreparationEvidence
    dimension: int = eqx.field(static=True)
    bins: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        edges: ArrayLike,
        evidence: VegasPreparationEvidence,
        /,
        *,
        plan_id: str,
    ):
        edges_ = np.asarray(edges, dtype=np.float64)
        if edges_.ndim != 2 or edges_.shape[1] < 3:
            raise ValueError("VEGAS edges must have shape (dimension, bins + 1).")
        if np.any(~np.isfinite(edges_)) or np.any(np.diff(edges_, axis=1) <= 0.0):
            raise ValueError("Frozen VEGAS grid edges must be finite and increasing.")
        if not plan_id:
            raise ValueError("Frozen VEGAS grids require a nonempty plan identity.")
        self.edges = jnp.asarray(edges_)
        self.evidence = evidence
        self.dimension = edges_.shape[0]
        self.bins = edges_.shape[1] - 1
        self.plan_id = str(plan_id)
        self.grid_id = canonical_fingerprint(
            {
                "kind": "frozen-vegas-grid",
                "plan_id": plan_id,
                "edges": array_tree_fingerprint(edges_),
            }
        )


class VegasSampleBatch(StrictModule):
    """Samples and exact inverse-proposal Jacobians from a frozen grid."""

    points: Array
    jacobians: Array
    unit_points: Array
    bin_indices: Array
    grid_id: str = eqx.field(static=True)


class VegasResult(StrictModule):
    """Production estimate with between-replicate uncertainty."""

    value: Array
    standard_error: Array
    iteration_estimates: Array
    iteration_variances: Array
    chi_square: Array
    status: Array
    num_evaluations: Array
    grid: FrozenVegasGrid
    method: str = eqx.field(static=True)


class PreparedVegas(StrictModule, NonTrainableState):
    """Immutable boundary between adaptive preparation and production runtime."""

    plan: VegasPlan
    grid: FrozenVegasGrid
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: VegasPlan, grid: FrozenVegasGrid, /):
        if grid.plan_id != plan.plan_id:
            raise ValueError("Frozen VEGAS grid was prepared for a different plan.")
        self.plan = plan
        self.grid = grid
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vegas",
                "plan_id": plan.plan_id,
                "grid_id": grid.grid_id,
            }
        )


def _stratified_unit_points(key: Key[Array, ""], count: int, dimension: int, /) -> Array:
    """Latin-stratified points: every one-dimensional stratum is occupied."""
    keys = jr.split(key, 2 * dimension)
    base = jnp.arange(count, dtype=jnp.float64)
    coordinates = []
    for axis in range(dimension):
        jitter = jr.uniform(keys[2 * axis], (count,))
        coordinate = (base + jitter) / float(count)
        coordinates.append(jr.permutation(keys[2 * axis + 1], coordinate))
    return jnp.stack(coordinates, axis=-1)


def _transform_vegas_edges(
    edges: Array, unit: Array, grid_id: str, /
) -> VegasSampleBatch:
    dimension = edges.shape[0]
    bins = edges.shape[1] - 1
    scaled = jnp.clip(unit, 0.0, jnp.nextafter(1.0, 0.0)) * bins
    indices = jnp.floor(scaled).astype(jnp.int32)
    fractions = scaled - indices
    axis = jnp.arange(dimension)[None, :]
    left = edges[axis, indices]
    right = edges[axis, indices + 1]
    widths = right - left
    points = left + fractions * widths
    jacobians = jnp.prod(float(bins) * widths, axis=-1)
    return VegasSampleBatch(points, jacobians, unit, indices, grid_id)


def transform_frozen_vegas(
    grid: FrozenVegasGrid, unit_points: ArrayLike, /
) -> VegasSampleBatch:
    """Map unit points through ``grid`` and return exact ``dx / du``."""
    unit = jnp.asarray(unit_points)
    if unit.ndim != 2 or unit.shape[1] != grid.dimension:
        raise ValueError("unit_points must have shape (samples, grid.dimension).")
    return _transform_vegas_edges(grid.edges, unit, grid.grid_id)


def sample_frozen_vegas(
    grid: FrozenVegasGrid,
    key: Key[Array, ""],
    count: int,
    /,
) -> VegasSampleBatch:
    """Draw a fixed-size stratified batch from a production grid."""
    count_ = int(count)
    if count_ < grid.bins:
        raise ValueError("Frozen VEGAS sampling must cover every grid bin.")
    unit = _stratified_unit_points(key, count_, grid.dimension)
    return transform_frozen_vegas(grid, unit)


def _integrand_values(integrand: Callable[[Array], Array], points: Array, /) -> Array:
    values = jnp.asarray(integrand(points))
    if values.shape[:1] != points.shape[:1]:
        raise ValueError("VEGAS integrands must preserve the leading sample axis.")
    return values


def _iteration_estimate(values: Array, jacobians: Array, /) -> Array:
    shape = (jacobians.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.mean(values * jacobians.reshape(shape), axis=0)


def _adapt_edges(
    edges: Array,
    indices: Array,
    importance: Array,
    power: float,
    minimum_fraction: float,
    /,
) -> tuple[Array, Array]:
    bins = edges.shape[1] - 1
    dimensions = edges.shape[0]
    marginals = []
    adapted = []
    targets = jnp.linspace(0.0, 1.0, bins + 1)
    for axis in range(dimensions):
        membership = jax.nn.one_hot(indices[:, axis], bins, dtype=importance.dtype)
        marginal = jnp.sum(membership * importance[:, None], axis=0)
        marginal = (jnp.roll(marginal, 1) + 2.0 * marginal + jnp.roll(marginal, -1)) / 4.0
        marginal = jnp.power(
            jnp.maximum(marginal, 0.0) + jnp.finfo(marginal.dtype).tiny, power
        )
        marginal = marginal / jnp.sum(marginal)
        marginal = jnp.maximum(marginal, minimum_fraction)
        marginal = marginal / jnp.sum(marginal)
        cumulative = jnp.concatenate(
            (jnp.zeros((1,), dtype=marginal.dtype), jnp.cumsum(marginal))
        )
        adapted.append(linear_interpolate(cumulative, edges[axis], targets).values)
        marginals.append(marginal)
    return jnp.stack(adapted), jnp.stack(marginals)


def prepare_vegas(
    integrand: Callable[[Array], Array],
    plan: VegasPlan,
    key: Key[Array, ""],
    /,
) -> PreparedVegas:
    """Adapt a grid using pilot evaluations and freeze it for later production."""
    if not isinstance(plan, VegasPlan):
        raise TypeError("plan must be VegasPlan.")
    edges = (
        plan.lower[:, None]
        + (plan.upper - plan.lower)[:, None]
        * jnp.linspace(0.0, 1.0, plan.bins + 1)[None, :]
    )
    estimates: list[Array] = []
    finite: list[Array] = []
    marginal_records: list[Array] = []
    marginals = jnp.full((plan.dimension, plan.bins), 1.0 / plan.bins)
    keys = jr.split(key, max(plan.adaptation_iterations, 1))
    for iteration in range(plan.adaptation_iterations):
        unit = _stratified_unit_points(
            keys[iteration],
            plan.adaptation_samples,
            plan.dimension,
        )
        batch = _transform_vegas_edges(edges, unit, plan.plan_id)
        values = _integrand_values(integrand, batch.points)
        estimate = _iteration_estimate(values, batch.jacobians)
        absolute_values = jnp.abs(values.reshape((values.shape[0], -1)))
        scalar_importance = (
            jnp.sqrt(jnp.sum(absolute_values**2, axis=1)) * batch.jacobians
        )
        candidate_edges, candidate_marginals = _adapt_edges(
            edges,
            batch.bin_indices,
            scalar_importance,
            plan.adaptation_power,
            plan.minimum_bin_fraction,
        )
        iteration_finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(batch.jacobians))
            & jnp.all(jnp.isfinite(candidate_edges))
            & jnp.all(jnp.isfinite(candidate_marginals))
        )
        edges = jnp.where(iteration_finite, candidate_edges, edges)
        marginals = jnp.where(iteration_finite, candidate_marginals, marginals)
        marginal_records.append(marginals)
        estimates.append(estimate)
        finite.append(iteration_finite)
    estimate_array = (
        jnp.stack(estimates)
        if estimates
        else jnp.zeros((0,), dtype=jnp.result_type(plan.lower, jnp.float64))
    )
    finite_array = jnp.stack(finite) if finite else jnp.ones((0,), dtype=jnp.bool_)
    marginal_history = (
        jnp.stack(marginal_records)
        if marginal_records
        else jnp.zeros((0, plan.dimension, plan.bins), dtype=plan.lower.dtype)
    )
    status = jnp.where(
        jnp.all(finite_array),
        int(VegasStatus.CONVERGED),
        int(VegasStatus.NONFINITE_INTEGRAND),
    )
    evidence = VegasPreparationEvidence(
        estimate_array,
        marginal_history,
        finite_array,
        status.astype(jnp.int32),
        jnp.asarray(
            plan.adaptation_iterations * plan.adaptation_samples, dtype=jnp.int32
        ),
    )
    return PreparedVegas(plan, FrozenVegasGrid(edges, evidence, plan_id=plan.plan_id))


def run_vegas(
    integrand: Callable[[Array], Array],
    prepared: PreparedVegas,
    key: Key[Array, ""],
    /,
) -> VegasResult:
    """Run independent production replicates without mutating the frozen grid."""
    if not isinstance(prepared, PreparedVegas):
        raise TypeError("prepared must be PreparedVegas.")
    plan = prepared.plan
    keys = jr.split(key, plan.production_iterations)
    estimates = []
    estimate_variances = []
    finite = []
    for iteration in range(plan.production_iterations):
        batch = sample_frozen_vegas(
            prepared.grid, keys[iteration], plan.production_samples
        )
        values = _integrand_values(integrand, batch.points)
        estimate = _iteration_estimate(values, batch.jacobians)
        weight_shape = (batch.jacobians.shape[0],) + (1,) * (values.ndim - 1)
        contributions = values * batch.jacobians.reshape(weight_shape)
        centered_samples = contributions - estimate
        sample_variance = jnp.sum(
            jnp.real(centered_samples * jnp.conj(centered_samples)), axis=0
        ) / max(plan.production_samples - 1, 1)
        estimates.append(estimate)
        estimate_variances.append(sample_variance / plan.production_samples)
        finite.append(
            jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(batch.jacobians))
        )
    iteration_estimates = jnp.stack(estimates)
    iteration_variances = jnp.stack(estimate_variances)
    value = jnp.mean(iteration_estimates, axis=0)
    centered = iteration_estimates - value
    replicate_count = plan.production_iterations
    variance = jnp.sum(jnp.real(centered * jnp.conj(centered)), axis=0) / max(
        replicate_count - 1, 1
    )
    standard_error = jnp.sqrt(variance / replicate_count)
    scale = jnp.maximum(
        iteration_variances,
        jnp.finfo(iteration_variances.dtype).tiny,
    )
    chi_square = jnp.sum(jnp.real(centered * jnp.conj(centered)) / scale, axis=0)
    all_finite = jnp.all(jnp.stack(finite)) & (
        prepared.grid.evidence.status == int(VegasStatus.CONVERGED)
    )
    status = jnp.where(
        all_finite,
        int(
            VegasStatus.CONVERGED
            if replicate_count >= 2
            else VegasStatus.INSUFFICIENT_PRODUCTION_REPLICATES
        ),
        int(VegasStatus.NONFINITE_INTEGRAND),
    )
    return VegasResult(
        value,
        standard_error,
        iteration_estimates,
        iteration_variances,
        chi_square,
        status.astype(jnp.int32),
        jnp.asarray(replicate_count * plan.production_samples, dtype=jnp.int32),
        prepared.grid,
        "adaptive-stratified-vegas-frozen-production",
    )


def vegas_integrate(
    integrand: Callable[[Array], Array],
    plan: VegasPlan,
    adaptation_key: Key[Array, ""],
    production_key: Key[Array, ""],
    /,
) -> VegasResult:
    """Prepare and run VEGAS with disjoint caller-owned random keys."""
    return run_vegas(
        integrand, prepare_vegas(integrand, plan, adaptation_key), production_key
    )


__all__ = [
    "FrozenVegasGrid",
    "PreparedVegas",
    "VegasPlan",
    "VegasPreparationEvidence",
    "VegasResult",
    "VegasSampleBatch",
    "VegasStatus",
    "prepare_vegas",
    "run_vegas",
    "sample_frozen_vegas",
    "transform_frozen_vegas",
    "vegas_integrate",
]
