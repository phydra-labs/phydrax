#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..integration._api import IntegrationRealization
from ..integration._targets import DiscreteMeasureTarget, WeightedSampleTarget
from ._costs import (
    AbstractGroundCost,
    SquaredEuclideanCost,
    WeightedSquaredEuclideanCost,
)
from ._measure import EventEncoder, lower_transport_measure
from ._status import TransportStatus


BarycenterMeasure = DiscreteMeasureTarget | WeightedSampleTarget | IntegrationRealization


class BarycenterProblemProvenance(StrictModule):
    """Static identities of the measures and declared barycenter support."""

    measures: tuple[str, ...] = eqx.field(static=True)
    support: str = eqx.field(static=True)
    cost: str = eqx.field(static=True)

    def __init__(
        self,
        measures: tuple[str, ...],
        support: str,
        cost: str,
        /,
    ):
        self.measures = tuple(str(item) for item in measures)
        self.support = str(support)
        self.cost = str(cost)


class FixedSupportBarycenterProblem(StrictModule):
    """A finite balanced barycenter problem on one declared support.

    Input measures are lowered through the existing integration contracts and are
    stored in a padded, explicitly masked representation. Padding never contributes
    mass or cost to a solve.
    """

    measure_points: Array
    measure_probabilities: Array
    measure_active: Array
    support_points: Array
    support_probabilities: Array
    support_active: Array
    measure_weights: Array
    mass: Array
    cost: AbstractGroundCost
    provenance: BarycenterProblemProvenance
    measure_atom_counts: tuple[int, ...] = eqx.field(static=True)
    measure_event_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    support_event_shape: tuple[int, ...] = eqx.field(static=True)
    mass_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        measures: tuple[BarycenterMeasure, ...],
        support: BarycenterMeasure,
        /,
        *,
        measure_weights: ArrayLike,
        cost: AbstractGroundCost,
        encoders: tuple[EventEncoder | None, ...] | None = None,
        support_encoder: EventEncoder | None = None,
        mass_tolerance: float = 1e-8,
    ):
        if not isinstance(measures, tuple) or not measures:
            raise TypeError(
                "measures must be a nonempty tuple of finite integration measures."
            )
        if not isinstance(cost, AbstractGroundCost):
            raise TypeError("cost must be an AbstractGroundCost.")
        tolerance = float(mass_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("mass_tolerance must be finite and nonnegative.")
        if encoders is None:
            resolved_encoders: tuple[EventEncoder | None, ...] = (None,) * len(measures)
        else:
            resolved_encoders = tuple(encoders)
            if len(resolved_encoders) != len(measures):
                raise ValueError("encoders must contain one entry per measure.")
            if any(item is not None and not callable(item) for item in resolved_encoders):
                raise TypeError("Each encoder must be callable or None.")
        if support_encoder is not None and not callable(support_encoder):
            raise TypeError("support_encoder must be callable or None.")

        lowered = tuple(
            lower_transport_measure(
                _finite_target(measure, name=f"measures[{index}]"),
                encoder=encoder,
                name=f"measures[{index}]",
            )
            for index, (measure, encoder) in enumerate(
                zip(measures, resolved_encoders, strict=True)
            )
        )
        lowered_support = lower_transport_measure(
            _finite_target(support, name="support"),
            encoder=support_encoder,
            name="support",
        )
        feature_size = lowered_support.feature_size
        if any(measure.feature_size != feature_size for measure in lowered):
            raise ValueError(
                "All encoded measures and support must have a common feature size."
            )
        weights = jnp.asarray(measure_weights, dtype=float)
        if weights.shape != (len(lowered),):
            raise ValueError("measure_weights must contain one value per measure.")
        weights = eqx.error_if(
            weights,
            jnp.any(~jnp.isfinite(weights))
            | jnp.any(weights <= 0.0)
            | ~jnp.isclose(jnp.sum(weights), 1.0, rtol=1e-8, atol=1e-10),
            "measure_weights must be finite, strictly positive, and sum to one.",
        )
        masses = jnp.stack(
            tuple(measure.mass for measure in lowered) + (lowered_support.mass,)
        )
        common_mass = eqx.error_if(
            masses[0],
            jnp.any(
                ~jnp.isclose(
                    masses,
                    masses[0],
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "Barycenter measures and declared support must have common physical mass.",
        )
        maximum_atoms = max(measure.num_atoms for measure in lowered)
        points = []
        probabilities = []
        active = []
        for measure in lowered:
            padding = maximum_atoms - measure.num_atoms
            points.append(jnp.pad(measure.points, ((0, padding), (0, 0))))
            probabilities.append(jnp.pad(measure.probabilities, (0, padding)))
            active.append(jnp.pad(measure.active, (0, padding)))

        self.measure_points = jnp.stack(points)
        self.measure_probabilities = jnp.stack(probabilities)
        self.measure_active = jnp.stack(active)
        self.support_points = lowered_support.points
        self.support_probabilities = lowered_support.probabilities
        self.support_active = lowered_support.active
        self.measure_weights = weights
        self.mass = common_mass
        self.cost = cost
        self.provenance = BarycenterProblemProvenance(
            tuple(measure.provenance for measure in lowered),
            lowered_support.provenance,
            cost.cost_id,
        )
        self.measure_atom_counts = tuple(measure.num_atoms for measure in lowered)
        self.measure_event_shapes = tuple(measure.event_shape for measure in lowered)
        self.support_event_shape = lowered_support.event_shape
        self.mass_tolerance = tolerance

    @property
    def num_measures(self) -> int:
        return len(self.measure_atom_counts)

    @property
    def padded_atom_count(self) -> int:
        return int(self.measure_points.shape[1])

    @property
    def support_atom_count(self) -> int:
        return int(self.support_points.shape[0])

    @property
    def feature_size(self) -> int:
        return int(self.support_points.shape[1])

    def cost_matrices(self) -> Array:
        """Materialize padded measure-to-support cost matrices."""
        return jax.vmap(lambda points: self.cost.matrix(points, self.support_points))(
            self.measure_points
        )

    def with_support_points(self, points: ArrayLike, /) -> FixedSupportBarycenterProblem:
        """Return the same declared problem at new support coordinates."""
        values = jnp.asarray(points, dtype=self.support_points.dtype)
        if values.shape != self.support_points.shape:
            raise ValueError("New support points must preserve declared support shape.")
        values = eqx.error_if(
            values,
            jnp.any(self.support_active[:, None] & ~jnp.isfinite(values)),
            "Active barycenter support points must be finite.",
        )
        values = jnp.where(self.support_active[:, None], values, self.support_points)
        return eqx.tree_at(lambda item: item.support_points, self, values)


def fixed_support_barycenter_problem(
    measures: tuple[BarycenterMeasure, ...],
    support: BarycenterMeasure,
    /,
    *,
    measure_weights: ArrayLike,
    cost: AbstractGroundCost,
    encoders: tuple[EventEncoder | None, ...] | None = None,
    support_encoder: EventEncoder | None = None,
    mass_tolerance: float = 1e-8,
) -> FixedSupportBarycenterProblem:
    """Construct a fixed-support barycenter from finite integration measures."""
    return FixedSupportBarycenterProblem(
        measures,
        support,
        measure_weights=measure_weights,
        cost=cost,
        encoders=encoders,
        support_encoder=support_encoder,
        mass_tolerance=mass_tolerance,
    )


class BarycenterProvenance(StrictModule):
    """Static numerical provenance for a fixed-support barycenter solve."""

    method: str = eqx.field(static=True)
    cost: str = eqx.field(static=True)
    execution: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    measures: tuple[str, ...] = eqx.field(static=True)
    support: str = eqx.field(static=True)
    approximate: bool = eqx.field(static=True)


class BarycenterDiagnostics(StrictModule):
    """Fixed-structure diagnostics for log-domain Sinkhorn barycenters."""

    status: Array
    per_measure_status: Array
    num_iterations: Array
    first_converged_iteration: Array
    normalized_marginal_residual: Array
    physical_marginal_residual: Array
    per_measure_marginal_residual: Array
    consensus_residual: Array
    dual_residual: Array
    num_checks: Array
    residual_history: Array
    per_measure_residual_history: Array


class BarycenterResult(StrictModule):
    """A fixed-support entropic barycenter and all native coupling data."""

    problem: FixedSupportBarycenterProblem
    probabilities: Array
    measure_potentials: Array
    support_potentials: Array
    epsilon: Array
    per_measure_transport_costs: Array
    per_measure_regularizations: Array
    per_measure_objectives: Array
    objective: Array
    diagnostics: BarycenterDiagnostics
    provenance: BarycenterProvenance
    block_size: int | None = eqx.field(static=True)

    @property
    def converged(self) -> Array:
        """Whether every marginal and the common support law converged."""
        return self.diagnostics.status == int(TransportStatus.CONVERGED)

    @property
    def approximate(self) -> bool:
        """Whether the execution deliberately approximated the declared problem."""
        return self.provenance.approximate

    def padded_couplings(self) -> Array:
        """Return all physical couplings in the padded measure representation."""
        return self.problem.mass * _couplings(
            self.problem,
            self.measure_potentials,
            self.support_potentials,
            self.probabilities,
            self.epsilon,
            self.problem.cost_matrices(),
        )

    def coupling(self, measure_index: int, /) -> Array:
        """Return one unpadded physical measure-to-barycenter coupling."""
        index = int(measure_index)
        if index < 0 or index >= self.problem.num_measures:
            raise IndexError("measure_index is out of range.")
        count = self.problem.measure_atom_counts[index]
        return self.padded_couplings()[index, :count]

    def as_target(
        self,
        /,
        *,
        axis: str = "barycenter_atom",
        provenance: str = "sinkhorn-barycenter",
    ) -> DiscreteMeasureTarget:
        """Return the barycenter as a physical-mass discrete integration target."""
        axis_ = str(axis)
        if not axis_:
            raise ValueError("axis must be nonempty.")
        event_shape = self.problem.support_event_shape
        points = (
            self.problem.support_points.reshape(
                (self.problem.support_atom_count,) + event_shape
            )
            if event_shape
            else self.problem.support_points[:, 0]
        )
        return DiscreteMeasureTarget(
            points,
            cx.Field(self.problem.mass * self.probabilities, dims=(axis_,)),
            axes=axis_,
            mask=cx.Field(self.problem.support_active, dims=(axis_,)),
            normalized=False,
            target_mass=self.problem.mass,
            provenance=provenance,
        )


class SinkhornBarycenter(StrictModule):
    """Stabilized log-domain solver for balanced fixed-support barycenters."""

    epsilon: Array
    tolerance: Array
    stagnation_tolerance: Array
    max_iterations: int = eqx.field(static=True)
    min_iterations: int = eqx.field(static=True)
    check_every: int = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    stagnation_patience: int = eqx.field(static=True)
    early_stop: bool = eqx.field(static=True)
    store_history: bool = eqx.field(static=True)

    def __init__(
        self,
        epsilon: ArrayLike,
        /,
        *,
        max_iterations: int = 500,
        min_iterations: int = 1,
        tolerance: ArrayLike = 1e-7,
        check_every: int = 5,
        block_size: int | None = None,
        stagnation_patience: int = 0,
        stagnation_tolerance: ArrayLike = 1e-5,
        early_stop: bool = False,
        store_history: bool = False,
    ):
        maximum = int(max_iterations)
        minimum = int(min_iterations)
        interval = int(check_every)
        patience = int(stagnation_patience)
        if maximum < 1:
            raise ValueError("max_iterations must be positive.")
        if minimum < 0 or minimum > maximum:
            raise ValueError("min_iterations must lie in [0, max_iterations].")
        if interval < 1:
            raise ValueError("check_every must be positive.")
        if block_size is not None and int(block_size) < 1:
            raise ValueError("block_size must be positive or None.")
        if patience < 0:
            raise ValueError("stagnation_patience must be nonnegative.")
        epsilon_ = jnp.asarray(epsilon, dtype=float).reshape(())
        tolerance_ = jnp.asarray(tolerance, dtype=float).reshape(())
        stagnation_ = jnp.asarray(stagnation_tolerance, dtype=float).reshape(())
        self.epsilon = eqx.error_if(
            epsilon_,
            ~jnp.isfinite(epsilon_) | (epsilon_ <= 0.0),
            "epsilon must be finite and positive.",
        )
        self.tolerance = eqx.error_if(
            tolerance_,
            ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
            "tolerance must be finite and nonnegative.",
        )
        self.stagnation_tolerance = eqx.error_if(
            stagnation_,
            ~jnp.isfinite(stagnation_) | (stagnation_ < 0.0) | (stagnation_ >= 1.0),
            "stagnation_tolerance must be finite and lie in [0, 1).",
        )
        self.max_iterations = maximum
        self.min_iterations = minimum
        self.check_every = interval
        self.block_size = None if block_size is None else int(block_size)
        self.stagnation_patience = patience
        self.early_stop = bool(early_stop)
        self.store_history = bool(store_history)

    def __call__(
        self,
        problem: FixedSupportBarycenterProblem,
        /,
        *,
        initial_potentials: tuple[ArrayLike, ArrayLike] | None = None,
        initial_probabilities: ArrayLike | None = None,
    ) -> BarycenterResult:
        if not isinstance(problem, FixedSupportBarycenterProblem):
            raise TypeError("problem must be a FixedSupportBarycenterProblem.")
        dtype = jnp.result_type(
            problem.measure_points,
            problem.support_points,
            self.epsilon,
        )
        measure_shape = (problem.num_measures, problem.padded_atom_count)
        support_shape = (problem.num_measures, problem.support_atom_count)
        if initial_potentials is None:
            measure_potentials = jnp.zeros(measure_shape, dtype=dtype)
            support_potentials = jnp.zeros(support_shape, dtype=dtype)
        else:
            measure_potentials = jnp.asarray(initial_potentials[0], dtype=dtype)
            support_potentials = jnp.asarray(initial_potentials[1], dtype=dtype)
            if measure_potentials.shape != measure_shape:
                raise ValueError("Initial measure potentials have incompatible shape.")
            if support_potentials.shape != support_shape:
                raise ValueError("Initial support potentials have incompatible shape.")
            measure_potentials = eqx.error_if(
                measure_potentials,
                jnp.any(problem.measure_active & ~jnp.isfinite(measure_potentials)),
                "Active initial measure potentials must be finite.",
            )
            support_potentials = eqx.error_if(
                support_potentials,
                jnp.any(
                    problem.support_active[None, :] & ~jnp.isfinite(support_potentials)
                ),
                "Active initial support potentials must be finite.",
            )
        probabilities = (
            problem.support_probabilities
            if initial_probabilities is None
            else jnp.asarray(initial_probabilities, dtype=dtype)
        )
        if probabilities.shape != (problem.support_atom_count,):
            raise ValueError("Initial probabilities must match support atom count.")
        probabilities = eqx.error_if(
            probabilities,
            jnp.any(~jnp.isfinite(probabilities))
            | jnp.any(problem.support_active & (probabilities <= 0.0))
            | jnp.any(~problem.support_active & (probabilities != 0.0))
            | ~jnp.isclose(jnp.sum(probabilities), 1.0, rtol=1e-8, atol=1e-10),
            "Initial support probabilities must be positive on active support and sum to one.",
        )
        epsilon = self.epsilon.astype(dtype)
        tolerance = self.tolerance.astype(dtype)
        costs = problem.cost_matrices().astype(dtype)
        initial_residuals = jnp.full((problem.num_measures,), jnp.inf, dtype=dtype)
        initial_carry = (
            measure_potentials,
            support_potentials,
            probabilities,
            jnp.asarray(jnp.inf, dtype=dtype),
            initial_residuals,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(-1, dtype=jnp.int32),
        )

        def step(carry, index):
            (
                current_f,
                current_g,
                current_q,
                current_residual,
                current_per_measure,
                current_dual_residual,
                first_converged,
                converged,
                failed,
                best_residual,
                stagnant_checks,
                stagnated,
                terminal_iteration,
            ) = carry
            frozen = failed | ((converged | stagnated) & self.early_stop)

            def update(_):
                next_f_parts = []
                log_support_parts = []
                for measure_index in range(problem.num_measures):
                    active = problem.measure_active[measure_index]
                    row_normalizer = _row_logsumexp(
                        costs[measure_index],
                        current_g[measure_index] / epsilon,
                        epsilon,
                        active,
                        problem.support_active,
                        block_size=self.block_size,
                    )
                    next_f_part = epsilon * (
                        jnp.where(
                            active,
                            _safe_log(problem.measure_probabilities[measure_index]),
                            0.0,
                        )
                        - jnp.where(active, row_normalizer, 0.0)
                    )
                    log_support_part = _column_logsumexp(
                        costs[measure_index],
                        next_f_part / epsilon,
                        epsilon,
                        problem.measure_active[measure_index],
                        problem.support_active,
                        block_size=self.block_size,
                    )
                    next_f_parts.append(next_f_part)
                    log_support_parts.append(log_support_part)
                next_f = jnp.stack(next_f_parts)
                log_support = jnp.stack(log_support_parts)
                log_q_unnormalized = jnp.sum(
                    problem.measure_weights[:, None] * log_support,
                    axis=0,
                )
                log_q_unnormalized = jnp.where(
                    problem.support_active, log_q_unnormalized, -jnp.inf
                )
                log_q_normalizer = logsumexp(log_q_unnormalized)
                next_q = jnp.where(
                    problem.support_active,
                    jnp.exp(log_q_unnormalized - log_q_normalizer),
                    0.0,
                )
                next_g = epsilon * (_safe_log(next_q)[None, :] - log_support)
                next_g = jnp.where(problem.support_active[None, :], next_g, 0.0)
                finite = (
                    jnp.all(jnp.isfinite(jnp.where(problem.measure_active, next_f, 0.0)))
                    & jnp.all(
                        jnp.isfinite(
                            jnp.where(problem.support_active[None, :], next_g, 0.0)
                        )
                    )
                    & jnp.all(jnp.isfinite(next_q))
                    & jnp.isfinite(log_q_normalizer)
                )
                next_f = jnp.where(finite, next_f, current_f)
                next_g = jnp.where(finite, next_g, current_g)
                next_q = jnp.where(finite, next_q, current_q)
                potential_change = (
                    jnp.maximum(
                        jnp.max(jnp.abs(next_f - current_f)),
                        jnp.max(jnp.abs(next_g - current_g)),
                    )
                    / epsilon
                )
                probability_change = jnp.sum(jnp.abs(next_q - current_q))
                return (
                    next_f,
                    next_g,
                    next_q,
                    jnp.maximum(potential_change, probability_change),
                    ~finite,
                )

            def retain(_):
                return current_f, current_g, current_q, current_dual_residual, failed

            next_f, next_g, next_q, next_dual, next_failed = jax.lax.cond(
                frozen,
                retain,
                update,
                operand=None,
            )
            iteration = index + 1
            should_check = (
                (iteration % self.check_every == 0)
                | (iteration == self.max_iterations)
                | (iteration == self.min_iterations)
            )

            def check(_):
                per_measure, consensus, finite = _marginal_residuals(
                    problem,
                    next_f,
                    next_g,
                    next_q,
                    epsilon,
                    costs,
                )
                aggregate = jnp.maximum(jnp.max(per_measure), consensus)
                return (
                    jnp.where(finite, aggregate, jnp.inf),
                    jnp.where(finite, per_measure, jnp.inf),
                    ~finite,
                )

            def no_check(_):
                return current_residual, current_per_measure, jnp.asarray(False)

            next_residual, next_per_measure, objective_failed = jax.lax.cond(
                should_check & ~next_failed,
                check,
                no_check,
                operand=None,
            )
            next_failed = next_failed | objective_failed
            eligible = (
                should_check
                & (iteration >= self.min_iterations)
                & (next_residual <= tolerance)
                & ~next_failed
            )
            next_first = jnp.where(
                (first_converged < 0) & eligible,
                iteration.astype(jnp.int32),
                first_converged,
            )
            next_converged = converged | eligible
            improved = next_residual < best_residual * (
                1.0 - self.stagnation_tolerance.astype(dtype)
            )
            checked_stagnant = jnp.where(
                improved,
                jnp.asarray(0, dtype=jnp.int32),
                stagnant_checks + 1,
            )
            next_stagnant_checks = jnp.where(
                should_check & ~next_failed,
                checked_stagnant,
                stagnant_checks,
            )
            next_best = jnp.where(
                should_check & ~next_failed,
                jnp.minimum(best_residual, next_residual),
                best_residual,
            )
            detected_stagnation = (
                (self.stagnation_patience > 0)
                & should_check
                & (next_stagnant_checks >= self.stagnation_patience)
                & ~eligible
                & ~next_failed
            )
            next_stagnated = stagnated | detected_stagnation
            terminal = next_failed | eligible | detected_stagnation
            next_terminal = jnp.where(
                (terminal_iteration < 0) & terminal,
                iteration.astype(jnp.int32),
                terminal_iteration,
            )
            return (
                next_f,
                next_g,
                next_q,
                next_residual,
                next_per_measure,
                next_dual,
                next_first,
                next_converged,
                next_failed,
                next_best,
                next_stagnant_checks,
                next_stagnated,
                next_terminal,
            ), (next_residual, next_per_measure)

        final_carry, histories = jax.lax.scan(
            step,
            initial_carry,
            jnp.arange(self.max_iterations, dtype=jnp.int32),
        )
        (
            measure_potentials,
            support_potentials,
            probabilities,
            _,
            _,
            dual_residual,
            first_converged,
            _,
            failed,
            _,
            _,
            stagnated,
            terminal_iteration,
        ) = final_carry
        couplings = _couplings(
            problem,
            measure_potentials,
            support_potentials,
            probabilities,
            epsilon,
            costs,
        )
        per_measure_residual, consensus_residual, marginal_finite = (
            _residuals_from_couplings(problem, couplings, probabilities)
        )
        final_residual = jnp.maximum(jnp.max(per_measure_residual), consensus_residual)
        transport_costs, regularizations, objectives, objective_finite = _objectives(
            problem,
            couplings,
            measure_potentials,
            support_potentials,
            epsilon,
            costs,
        )
        objective = jnp.sum(problem.measure_weights * objectives)
        finite = marginal_finite & objective_finite & jnp.isfinite(objective)
        final_converged = (
            (final_residual <= tolerance)
            & (self.max_iterations >= self.min_iterations)
            & ~failed
            & finite
        )
        status = jnp.where(
            failed,
            int(TransportStatus.NONFINITE_ITERATE),
            jnp.where(
                ~finite,
                int(TransportStatus.NONFINITE_OBJECTIVE),
                jnp.where(
                    final_converged,
                    int(TransportStatus.CONVERGED),
                    jnp.where(
                        stagnated,
                        int(TransportStatus.MARGINAL_STAGNATION),
                        int(TransportStatus.MAXIMUM_ITERATIONS_REACHED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        per_measure_status = jnp.where(
            failed,
            int(TransportStatus.NONFINITE_ITERATE),
            jnp.where(
                ~finite,
                int(TransportStatus.NONFINITE_OBJECTIVE),
                jnp.where(
                    per_measure_residual <= tolerance,
                    int(TransportStatus.CONVERGED),
                    jnp.where(
                        stagnated,
                        int(TransportStatus.MARGINAL_STAGNATION),
                        int(TransportStatus.MAXIMUM_ITERATIONS_REACHED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        check_indices = tuple(
            index
            for index in range(self.max_iterations)
            if (index + 1) % self.check_every == 0
            or (index + 1) == self.max_iterations
            or (index + 1) == self.min_iterations
        )
        if self.store_history:
            indices = jnp.asarray(check_indices, dtype=jnp.int32)
            residual_history = histories[0][indices]
            per_measure_history = jnp.swapaxes(histories[1][indices], 0, 1)
        else:
            residual_history = jnp.empty((0,), dtype=dtype)
            per_measure_history = jnp.empty((problem.num_measures, 0), dtype=dtype)
        actual_iterations = jnp.where(
            self.early_stop & (terminal_iteration >= 0),
            terminal_iteration,
            self.max_iterations,
        ).astype(jnp.int32)
        diagnostics = BarycenterDiagnostics(
            status=status,
            per_measure_status=per_measure_status,
            num_iterations=actual_iterations,
            first_converged_iteration=first_converged,
            normalized_marginal_residual=final_residual,
            physical_marginal_residual=problem.mass * final_residual,
            per_measure_marginal_residual=per_measure_residual,
            consensus_residual=consensus_residual,
            dual_residual=dual_residual,
            num_checks=jnp.asarray(len(check_indices), dtype=jnp.int32),
            residual_history=residual_history,
            per_measure_residual_history=per_measure_history,
        )
        provenance = BarycenterProvenance(
            method="sinkhorn-barycenter",
            cost=problem.provenance.cost,
            execution="dense" if self.block_size is None else "blockwise",
            differentiation="unrolled",
            measures=problem.provenance.measures,
            support=problem.provenance.support,
            approximate=False,
        )
        return BarycenterResult(
            problem=problem,
            probabilities=probabilities,
            measure_potentials=measure_potentials,
            support_potentials=support_potentials,
            epsilon=epsilon,
            per_measure_transport_costs=problem.mass * transport_costs,
            per_measure_regularizations=problem.mass * regularizations,
            per_measure_objectives=objectives,
            objective=objective,
            diagnostics=diagnostics,
            provenance=provenance,
            block_size=self.block_size,
        )


class FreeSupportBarycenterProvenance(StrictModule):
    """Static provenance for an explicitly initialized local barycenter search."""

    method: str = eqx.field(static=True)
    cost: str = eqx.field(static=True)
    execution: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    initialization: str = eqx.field(static=True)
    local_optimization: bool = eqx.field(static=True)
    retained_inner_solves: int = eqx.field(static=True)
    approximate: bool = eqx.field(static=True)


class FreeSupportBarycenterDiagnostics(StrictModule):
    """Outer alternating-minimization diagnostics for a free support."""

    status: Array
    num_iterations: Array
    first_converged_iteration: Array
    objective_history: Array
    support_displacement_history: Array
    inner_status_history: Array
    collapse_iteration: Array
    stagnation_iteration: Array
    failure_iteration: Array


class FreeSupportBarycenterResult(StrictModule):
    """A local free-support solution retaining every inner fixed-support solve."""

    initial_problem: FixedSupportBarycenterProblem
    barycenter: BarycenterResult
    inner_results: tuple[BarycenterResult, ...]
    diagnostics: FreeSupportBarycenterDiagnostics
    provenance: FreeSupportBarycenterProvenance

    @property
    def converged(self) -> Array:
        return self.diagnostics.status == int(TransportStatus.CONVERGED)

    @property
    def local_optimum(self) -> Array:
        """Whether the alternating solve met its declared local stationarity test."""
        return self.converged

    @property
    def approximate(self) -> bool:
        return self.provenance.approximate

    def as_target(
        self,
        /,
        *,
        axis: str = "barycenter_atom",
        provenance: str = "free-support-barycenter",
    ) -> DiscreteMeasureTarget:
        """Return the terminal local barycenter as a physical discrete measure."""
        return self.barycenter.as_target(axis=axis, provenance=provenance)


class FreeSupportBarycenter(StrictModule):
    """Explicit alternating free-support barycenter solver for quadratic costs."""

    inner_solver: SinkhornBarycenter
    tolerance: Array
    collapse_tolerance: Array
    stagnation_tolerance: Array
    max_iterations: int = eqx.field(static=True)
    stagnation_patience: int = eqx.field(static=True)

    def __init__(
        self,
        inner_solver: SinkhornBarycenter,
        /,
        *,
        max_iterations: int = 20,
        tolerance: ArrayLike = 1e-6,
        collapse_tolerance: ArrayLike = 1e-10,
        stagnation_patience: int = 0,
        stagnation_tolerance: ArrayLike = 1e-6,
    ):
        if not isinstance(inner_solver, SinkhornBarycenter):
            raise TypeError("inner_solver must be a SinkhornBarycenter.")
        maximum = int(max_iterations)
        patience = int(stagnation_patience)
        if maximum < 1:
            raise ValueError("max_iterations must be positive.")
        if patience < 0:
            raise ValueError("stagnation_patience must be nonnegative.")
        tolerance_ = jnp.asarray(tolerance, dtype=float).reshape(())
        collapse_ = jnp.asarray(collapse_tolerance, dtype=float).reshape(())
        stagnation_ = jnp.asarray(stagnation_tolerance, dtype=float).reshape(())
        self.inner_solver = inner_solver
        self.tolerance = eqx.error_if(
            tolerance_,
            ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
            "tolerance must be finite and nonnegative.",
        )
        self.collapse_tolerance = eqx.error_if(
            collapse_,
            ~jnp.isfinite(collapse_) | (collapse_ < 0.0),
            "collapse_tolerance must be finite and nonnegative.",
        )
        self.stagnation_tolerance = eqx.error_if(
            stagnation_,
            ~jnp.isfinite(stagnation_) | (stagnation_ < 0.0) | (stagnation_ >= 1.0),
            "stagnation_tolerance must be finite and lie in [0, 1).",
        )
        self.max_iterations = maximum
        self.stagnation_patience = patience

    def __call__(
        self,
        problem: FixedSupportBarycenterProblem,
        /,
    ) -> FreeSupportBarycenterResult:
        if not isinstance(problem, FixedSupportBarycenterProblem):
            raise TypeError("problem must be a FixedSupportBarycenterProblem.")
        if not isinstance(
            problem.cost,
            (SquaredEuclideanCost, WeightedSquaredEuclideanCost),
        ):
            raise TypeError(
                "FreeSupportBarycenter requires squared or weighted squared Euclidean cost."
            )
        current_problem = problem
        previous_result: BarycenterResult | None = None
        inner_results: list[BarycenterResult] = []
        objective_history: list[Array] = []
        displacement_history: list[Array] = []
        inner_status_history: list[Array] = []
        frozen = jnp.asarray(False)
        failed_status = jnp.asarray(
            int(TransportStatus.MAXIMUM_ITERATIONS_REACHED), dtype=jnp.int32
        )
        first_converged = jnp.asarray(-1, dtype=jnp.int32)
        collapse_iteration = jnp.asarray(-1, dtype=jnp.int32)
        stagnation_iteration = jnp.asarray(-1, dtype=jnp.int32)
        failure_iteration = jnp.asarray(-1, dtype=jnp.int32)
        previous_objective = jnp.asarray(jnp.inf, dtype=problem.support_points.dtype)
        stagnant_steps = jnp.asarray(0, dtype=jnp.int32)

        for outer_index in range(self.max_iterations):
            if previous_result is None:
                result = self.inner_solver(current_problem)
            else:
                result = self.inner_solver(
                    current_problem,
                    initial_potentials=(
                        previous_result.measure_potentials,
                        previous_result.support_potentials,
                    ),
                    initial_probabilities=previous_result.probabilities,
                )
            couplings = result.padded_couplings()
            weighted_couplings = problem.measure_weights[:, None, None] * couplings
            numerator = jnp.sum(
                weighted_couplings[..., None] * problem.measure_points[:, :, None, :],
                axis=(0, 1),
            )
            denominator = jnp.sum(weighted_couplings, axis=(0, 1))
            safe_denominator = jnp.where(
                denominator > self.collapse_tolerance,
                denominator,
                1.0,
            )
            proposal = numerator / safe_denominator[:, None]
            proposal = jnp.where(
                problem.support_active[:, None],
                proposal,
                current_problem.support_points,
            )
            mass_collapse = jnp.any(
                problem.support_active & (denominator <= self.collapse_tolerance)
            )
            pair_difference = proposal[:, None, :] - proposal[None, :, :]
            pair_distance_squared = jnp.sum(pair_difference * pair_difference, axis=-1)
            pair_valid = (
                problem.support_active[:, None]
                & problem.support_active[None, :]
                & jnp.triu(
                    jnp.ones(
                        (problem.support_atom_count, problem.support_atom_count),
                        dtype=bool,
                    ),
                    k=1,
                )
            )
            support_collapse = jnp.any(
                pair_valid
                & (
                    pair_distance_squared
                    <= self.collapse_tolerance * self.collapse_tolerance
                )
            )
            finite = jnp.all(
                jnp.isfinite(jnp.where(problem.support_active[:, None], proposal, 0.0))
            )
            collapsed = (mass_collapse | support_collapse) & result.converged
            displacement = jnp.max(
                jnp.where(
                    problem.support_active,
                    jnp.linalg.norm(proposal - current_problem.support_points, axis=1),
                    0.0,
                )
            )
            iteration = jnp.asarray(outer_index + 1, dtype=jnp.int32)
            converged = (
                result.converged & (displacement <= self.tolerance) & ~collapsed & finite
            )
            objective_improvement = previous_objective - result.objective
            improved = jnp.isinf(previous_objective) | (
                objective_improvement
                > self.stagnation_tolerance
                * jnp.maximum(1.0, jnp.abs(previous_objective))
            )
            stagnant_steps = jnp.where(improved, 0, stagnant_steps + 1)
            stagnated = (
                (self.stagnation_patience > 0)
                & (stagnant_steps >= self.stagnation_patience)
                & ~converged
            )
            collapse_iteration = jnp.where(
                (collapse_iteration < 0) & collapsed & ~frozen,
                iteration,
                collapse_iteration,
            )
            stagnation_iteration = jnp.where(
                (stagnation_iteration < 0) & stagnated & ~frozen,
                iteration,
                stagnation_iteration,
            )
            first_converged = jnp.where(
                (first_converged < 0) & converged & ~frozen,
                iteration,
                first_converged,
            )
            inner_failure = ~result.converged
            failure_iteration = jnp.where(
                (failure_iteration < 0) & (inner_failure | ~finite) & ~frozen,
                iteration,
                failure_iteration,
            )
            failed_status = jnp.where(
                ~frozen & ~finite,
                int(TransportStatus.NONFINITE_ITERATE),
                jnp.where(
                    ~frozen & inner_failure,
                    result.diagnostics.status,
                    failed_status,
                ),
            )
            terminal = converged | collapsed | ~finite | inner_failure | stagnated
            next_frozen = frozen | terminal
            accept_update = ~next_frozen & (outer_index + 1 < self.max_iterations)
            next_points = jnp.where(
                accept_update,
                proposal,
                current_problem.support_points,
            )
            current_problem = current_problem.with_support_points(next_points)
            previous_objective = jnp.where(frozen, previous_objective, result.objective)
            previous_result = result
            frozen = next_frozen
            inner_results.append(result)
            objective_history.append(result.objective)
            displacement_history.append(displacement)
            inner_status_history.append(result.diagnostics.status)

        terminal_result = inner_results[-1]
        status = jnp.where(
            first_converged >= 0,
            int(TransportStatus.CONVERGED),
            jnp.where(
                collapse_iteration >= 0,
                int(TransportStatus.SUPPORT_COLLAPSE),
                jnp.where(
                    stagnation_iteration >= 0,
                    int(TransportStatus.MARGINAL_STAGNATION),
                    failed_status,
                ),
            ),
        ).astype(jnp.int32)
        terminal_candidates = jnp.stack(
            (
                jnp.where(first_converged >= 0, first_converged, self.max_iterations),
                jnp.where(
                    collapse_iteration >= 0, collapse_iteration, self.max_iterations
                ),
                jnp.where(
                    stagnation_iteration >= 0,
                    stagnation_iteration,
                    self.max_iterations,
                ),
                jnp.where(
                    failure_iteration >= 0,
                    failure_iteration,
                    self.max_iterations,
                ),
            )
        )
        diagnostics = FreeSupportBarycenterDiagnostics(
            status=status,
            num_iterations=jnp.min(terminal_candidates).astype(jnp.int32),
            first_converged_iteration=first_converged,
            objective_history=jnp.stack(objective_history),
            support_displacement_history=jnp.stack(displacement_history),
            inner_status_history=jnp.stack(inner_status_history),
            collapse_iteration=collapse_iteration,
            stagnation_iteration=stagnation_iteration,
            failure_iteration=failure_iteration,
        )
        provenance = FreeSupportBarycenterProvenance(
            method="free-support-barycenter",
            cost=problem.provenance.cost,
            execution="dense" if self.inner_solver.block_size is None else "blockwise",
            differentiation="unrolled-alternating",
            initialization=problem.provenance.support,
            local_optimization=True,
            retained_inner_solves=len(inner_results),
            approximate=False,
        )
        return FreeSupportBarycenterResult(
            initial_problem=problem,
            barycenter=terminal_result,
            inner_results=tuple(inner_results),
            diagnostics=diagnostics,
            provenance=provenance,
        )


def require_barycenter_converged(
    result: BarycenterResult | FreeSupportBarycenterResult,
    /,
) -> BarycenterResult | FreeSupportBarycenterResult:
    """Raise a JAX-compatible error unless a barycenter solve converged."""
    if not isinstance(result, (BarycenterResult, FreeSupportBarycenterResult)):
        raise TypeError("result must be a barycenter result.")
    if isinstance(result, BarycenterResult):
        checked = eqx.error_if(
            result.probabilities,
            ~result.converged,
            "Native barycenter transport did not converge.",
        )
        return eqx.tree_at(lambda item: item.probabilities, result, checked)
    checked = eqx.error_if(
        result.barycenter.probabilities,
        ~result.converged,
        "Native free-support barycenter transport did not converge.",
    )
    return eqx.tree_at(
        lambda item: item.barycenter.probabilities,
        result,
        checked,
    )


def _finite_target(
    measure: BarycenterMeasure,
    /,
    *,
    name: str,
) -> DiscreteMeasureTarget | WeightedSampleTarget:
    if isinstance(measure, (DiscreteMeasureTarget, WeightedSampleTarget)):
        return measure
    if isinstance(measure, IntegrationRealization):
        if isinstance(measure.target, (DiscreteMeasureTarget, WeightedSampleTarget)):
            return measure.target
        raise TypeError(
            f"{name} realization must materialize a DiscreteMeasureTarget or "
            "WeightedSampleTarget for finite barycenter transport."
        )
    raise TypeError(
        f"{name} must be a DiscreteMeasureTarget, WeightedSampleTarget, or "
        "finite IntegrationRealization."
    )


def _safe_log(values: Array, /) -> Array:
    positive = values > 0.0
    return jnp.where(
        positive,
        jnp.log(jnp.where(positive, values, 1.0)),
        -jnp.inf,
    )


def _row_logsumexp(
    costs: Array,
    values: Array,
    epsilon: Array,
    row_active: Array,
    column_active: Array,
    /,
    *,
    block_size: int | None,
) -> Array:
    if block_size is None:
        terms = values[None, :] - costs / epsilon
        terms = jnp.where(row_active[:, None] & column_active[None, :], terms, -jnp.inf)
        return logsumexp(terms, axis=1)
    size = int(block_size)
    row_count, column_count = costs.shape
    row_blocks = (row_count + size - 1) // size
    column_blocks = (column_count + size - 1) // size
    padded_rows = row_blocks * size
    padded_columns = column_blocks * size
    padded_costs = jnp.pad(
        costs,
        ((0, padded_rows - row_count), (0, padded_columns - column_count)),
        constant_values=jnp.inf,
    )
    padded_values = jnp.pad(
        values, (0, padded_columns - column_count), constant_values=-jnp.inf
    )
    padded_row_active = jnp.pad(row_active, (0, padded_rows - row_count))
    padded_column_active = jnp.pad(column_active, (0, padded_columns - column_count))
    output = jnp.full((padded_rows,), -jnp.inf, dtype=values.dtype)

    def row_body(row_block, result):
        row_start = row_block * size
        row_mask = jax.lax.dynamic_slice(padded_row_active, (row_start,), (size,))
        accumulator = jnp.full((size,), -jnp.inf, dtype=values.dtype)

        def column_body(column_block, current):
            column_start = column_block * size
            block = jax.lax.dynamic_slice(
                padded_costs, (row_start, column_start), (size, size)
            )
            block_values = jax.lax.dynamic_slice(padded_values, (column_start,), (size,))
            column_mask = jax.lax.dynamic_slice(
                padded_column_active, (column_start,), (size,)
            )
            terms = block_values[None, :] - block / epsilon
            terms = jnp.where(row_mask[:, None] & column_mask[None, :], terms, -jnp.inf)
            return jnp.logaddexp(current, logsumexp(terms, axis=1))

        accumulator = jax.lax.fori_loop(0, column_blocks, column_body, accumulator)
        return jax.lax.dynamic_update_slice(
            result, jnp.where(row_mask, accumulator, -jnp.inf), (row_start,)
        )

    return jax.lax.fori_loop(0, row_blocks, row_body, output)[:row_count]


def _column_logsumexp(
    costs: Array,
    values: Array,
    epsilon: Array,
    row_active: Array,
    column_active: Array,
    /,
    *,
    block_size: int | None,
) -> Array:
    return _row_logsumexp(
        costs.T,
        values,
        epsilon,
        column_active,
        row_active,
        block_size=block_size,
    )


def _couplings(
    problem: FixedSupportBarycenterProblem,
    measure_potentials: Array,
    support_potentials: Array,
    probabilities: Array,
    epsilon: Array,
    costs: Array,
    /,
) -> Array:
    del probabilities
    log_ratio = (
        measure_potentials[:, :, None] + support_potentials[:, None, :] - costs
    ) / epsilon
    valid = problem.measure_active[:, :, None] & problem.support_active[None, None, :]
    return jnp.where(valid, jnp.exp(log_ratio), 0.0)


def _residuals_from_couplings(
    problem: FixedSupportBarycenterProblem,
    couplings: Array,
    probabilities: Array,
    /,
) -> tuple[Array, Array, Array]:
    source_marginals = jnp.sum(couplings, axis=2)
    support_marginals = jnp.sum(couplings, axis=1)
    source_residual = jnp.sum(
        jnp.abs(source_marginals - problem.measure_probabilities), axis=1
    )
    support_residual = jnp.sum(
        jnp.abs(support_marginals - probabilities[None, :]), axis=1
    )
    per_measure = jnp.maximum(source_residual, support_residual)
    weighted_support = jnp.sum(
        problem.measure_weights[:, None] * support_marginals,
        axis=0,
    )
    consensus = jnp.sum(jnp.abs(weighted_support - probabilities))
    finite = (
        jnp.all(jnp.isfinite(source_marginals))
        & jnp.all(jnp.isfinite(support_marginals))
        & jnp.all(jnp.isfinite(per_measure))
        & jnp.isfinite(consensus)
    )
    return per_measure, consensus, finite


def _marginal_residuals(
    problem: FixedSupportBarycenterProblem,
    measure_potentials: Array,
    support_potentials: Array,
    probabilities: Array,
    epsilon: Array,
    costs: Array,
    /,
) -> tuple[Array, Array, Array]:
    couplings = _couplings(
        problem,
        measure_potentials,
        support_potentials,
        probabilities,
        epsilon,
        costs,
    )
    return _residuals_from_couplings(problem, couplings, probabilities)


def _objectives(
    problem: FixedSupportBarycenterProblem,
    couplings: Array,
    measure_potentials: Array,
    support_potentials: Array,
    epsilon: Array,
    costs: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    valid = problem.measure_active[:, :, None] & problem.support_active[None, None, :]
    log_ratio = (
        measure_potentials[:, :, None] + support_potentials[:, None, :] - costs
    ) / epsilon
    safe_ratio = jnp.where(valid & jnp.isfinite(log_ratio), log_ratio, 0.0)
    transport = jnp.sum(couplings * costs, axis=(1, 2))
    negative_entropy = jnp.sum(couplings * safe_ratio, axis=(1, 2))
    regularization = epsilon * negative_entropy
    probability_objectives = transport + regularization
    physical_objectives = problem.mass * probability_objectives
    finite = (
        jnp.all(jnp.isfinite(transport))
        & jnp.all(jnp.isfinite(regularization))
        & jnp.all(jnp.isfinite(physical_objectives))
    )
    return transport, regularization, physical_objectives, finite


__all__ = [
    "BarycenterDiagnostics",
    "BarycenterProblemProvenance",
    "BarycenterProvenance",
    "BarycenterResult",
    "FixedSupportBarycenterProblem",
    "FreeSupportBarycenter",
    "FreeSupportBarycenterDiagnostics",
    "FreeSupportBarycenterProvenance",
    "FreeSupportBarycenterResult",
    "SinkhornBarycenter",
    "fixed_support_barycenter_problem",
    "require_barycenter_converged",
]
