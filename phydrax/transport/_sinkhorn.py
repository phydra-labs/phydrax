#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._blocks import column_logsumexp, coupling_statistics, row_logsumexp
from ._problem import DiscreteTransportProblem
from ._results import (
    AbstractBalancedTransportSolver,
    SinkhornDiagnostics,
    SinkhornResult,
    TransportProvenance,
)
from ._status import TransportStatus


class Sinkhorn(AbstractBalancedTransportSolver):
    """Stabilized log-domain solver for balanced entropic transport."""

    epsilon: Array
    tolerance: Array
    max_iterations: int = eqx.field(static=True)
    min_iterations: int = eqx.field(static=True)
    check_every: int = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
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
        early_stop: bool = False,
        store_history: bool = False,
    ):
        maximum = int(max_iterations)
        minimum = int(min_iterations)
        interval = int(check_every)
        if maximum < 1:
            raise ValueError("max_iterations must be positive.")
        if minimum < 0 or minimum > maximum:
            raise ValueError("min_iterations must lie in [0, max_iterations].")
        if interval < 1:
            raise ValueError("check_every must be positive.")
        if block_size is not None and int(block_size) < 1:
            raise ValueError("block_size must be positive or None.")
        epsilon_ = jnp.asarray(epsilon, dtype=float).reshape(())
        tolerance_ = jnp.asarray(tolerance, dtype=float).reshape(())
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
        self.max_iterations = maximum
        self.min_iterations = minimum
        self.check_every = interval
        self.block_size = None if block_size is None else int(block_size)
        self.early_stop = bool(early_stop)
        self.store_history = bool(store_history)

    def __call__(
        self,
        problem: DiscreteTransportProblem,
        /,
        *,
        initial_potentials: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> SinkhornResult:
        if not isinstance(problem, DiscreteTransportProblem):
            raise TypeError("problem must be a DiscreteTransportProblem.")
        source_count, target_count = problem.shape
        dtype = jnp.result_type(
            problem.source.points,
            problem.target.points,
            self.epsilon,
        )
        if initial_potentials is None:
            source_initial = jnp.zeros((source_count,), dtype=dtype)
            target_initial = jnp.zeros((target_count,), dtype=dtype)
        else:
            source_initial = jnp.asarray(initial_potentials[0], dtype=dtype)
            target_initial = jnp.asarray(initial_potentials[1], dtype=dtype)
            if source_initial.shape != (source_count,):
                raise ValueError(
                    "Initial source potential must match source atom count."
                )
            if target_initial.shape != (target_count,):
                raise ValueError(
                    "Initial target potential must match target atom count."
                )
            source_initial = eqx.error_if(
                source_initial,
                jnp.any(~jnp.isfinite(source_initial)),
                "Initial source potential must be finite.",
            )
            target_initial = eqx.error_if(
                target_initial,
                jnp.any(~jnp.isfinite(target_initial)),
                "Initial target potential must be finite.",
            )
        log_source = _safe_log(problem.source_probabilities)
        log_target = _safe_log(problem.target_probabilities)
        epsilon = self.epsilon.astype(dtype)
        tolerance = self.tolerance.astype(dtype)
        initial_carry = (
            source_initial,
            target_initial,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
        )

        def step(carry, index):
            (
                source_potential,
                target_potential,
                marginal_residual,
                dual_residual,
                first_converged,
                converged,
                failed,
            ) = carry
            frozen = failed | (converged if self.early_stop else False)

            def update(_):
                next_source = -epsilon * row_logsumexp(
                    problem,
                    log_target + target_potential / epsilon,
                    epsilon,
                    block_size=self.block_size,
                )
                next_target = -epsilon * column_logsumexp(
                    problem,
                    log_source + next_source / epsilon,
                    epsilon,
                    block_size=self.block_size,
                )
                source_mean = jnp.sum(
                    problem.source_probabilities * next_source
                )
                target_mean = jnp.sum(
                    problem.target_probabilities * next_target
                )
                shift = 0.5 * (target_mean - source_mean)
                next_source = next_source + shift
                next_target = next_target - shift
                finite = jnp.all(jnp.isfinite(next_source)) & jnp.all(
                    jnp.isfinite(next_target)
                )
                next_source = jnp.where(finite, next_source, source_potential)
                next_target = jnp.where(finite, next_target, target_potential)
                next_dual_residual = jnp.maximum(
                    jnp.max(jnp.abs(next_source - source_potential)),
                    jnp.max(jnp.abs(next_target - target_potential)),
                ) / epsilon
                return next_source, next_target, next_dual_residual, ~finite

            def keep(_):
                return source_potential, target_potential, dual_residual, failed

            (
                next_source,
                next_target,
                next_dual_residual,
                next_failed,
            ) = jax.lax.cond(frozen, keep, update, operand=None)
            iteration = index + 1
            should_check = (
                (iteration % self.check_every == 0)
                | (iteration == self.max_iterations)
                | (iteration == self.min_iterations)
            )

            def check(_):
                source_marginal, target_marginal, _, _, _, finite = (
                    coupling_statistics(
                        problem,
                        next_source,
                        next_target,
                        epsilon,
                        block_size=self.block_size,
                    )
                )
                residual = jnp.maximum(
                    jnp.sum(
                        jnp.abs(
                            source_marginal - problem.source_probabilities
                        )
                    ),
                    jnp.sum(
                        jnp.abs(
                            target_marginal - problem.target_probabilities
                        )
                    ),
                )
                return jnp.where(finite, residual, jnp.inf), ~finite

            def retain(_):
                return marginal_residual, jnp.asarray(False)

            next_residual, objective_failed = jax.lax.cond(
                should_check & ~next_failed,
                check,
                retain,
                operand=None,
            )
            eligible = (
                should_check
                & (iteration >= self.min_iterations)
                & (next_residual <= tolerance)
                & ~next_failed
                & ~objective_failed
            )
            next_first = jnp.where(
                (first_converged < 0) & eligible,
                iteration.astype(jnp.int32),
                first_converged,
            )
            next_converged = converged | eligible
            next_failed = next_failed | objective_failed
            next_carry = (
                next_source,
                next_target,
                next_residual,
                next_dual_residual,
                next_first,
                next_converged,
                next_failed,
            )
            return next_carry, next_residual

        final_carry, residuals = jax.lax.scan(
            step,
            initial_carry,
            jnp.arange(self.max_iterations, dtype=jnp.int32),
        )
        (
            source_potential,
            target_potential,
            final_residual,
            dual_residual,
            first_converged,
            converged_during_scan,
            failed,
        ) = final_carry
        (
            source_marginal,
            target_marginal,
            transport_cost_probability,
            kl,
            plan_mass,
            objective_finite,
        ) = coupling_statistics(
            problem,
            source_potential,
            target_potential,
            epsilon,
            block_size=self.block_size,
        )
        final_residual = jnp.maximum(
            jnp.sum(jnp.abs(source_marginal - problem.source_probabilities)),
            jnp.sum(jnp.abs(target_marginal - problem.target_probabilities)),
        )
        regularization_probability = epsilon * kl
        primal_probability = transport_cost_probability + regularization_probability
        dual_probability = (
            jnp.sum(problem.source_probabilities * source_potential)
            + jnp.sum(problem.target_probabilities * target_potential)
            - epsilon * (plan_mass - 1.0)
        )
        transport_cost = problem.mass * transport_cost_probability
        regularization = problem.mass * regularization_probability
        regularized_cost = problem.mass * primal_probability
        dual_cost = problem.mass * dual_probability
        finite_objective = (
            objective_finite
            & jnp.isfinite(regularized_cost)
            & jnp.isfinite(dual_cost)
        )
        final_converged = (
            (final_residual <= tolerance)
            & (self.max_iterations >= self.min_iterations)
            & ~failed
            & finite_objective
        )
        status = jnp.where(
            failed,
            int(TransportStatus.NONFINITE_ITERATE),
            jnp.where(
                ~finite_objective,
                int(TransportStatus.NONFINITE_OBJECTIVE),
                jnp.where(
                    final_converged,
                    int(TransportStatus.CONVERGED),
                    int(TransportStatus.MAXIMUM_ITERATIONS_REACHED),
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
            history = residuals[jnp.asarray(check_indices, dtype=jnp.int32)]
        else:
            history = jnp.empty((0,), dtype=dtype)
        actual_iterations = jnp.where(
            self.early_stop & (first_converged >= 0),
            first_converged,
            self.max_iterations,
        ).astype(jnp.int32)
        diagnostics = SinkhornDiagnostics(
            status=status,
            num_iterations=actual_iterations,
            first_converged_iteration=first_converged,
            normalized_marginal_residual=final_residual,
            physical_marginal_residual=problem.mass * final_residual,
            dual_residual=dual_residual,
            primal_dual_gap=jnp.abs(regularized_cost - dual_cost),
            num_checks=jnp.asarray(len(check_indices), dtype=jnp.int32),
            residual_history=history,
        )
        provenance = TransportProvenance(
            "sinkhorn",
            problem.provenance.cost,
            "dense" if self.block_size is None else "blockwise",
            "unrolled",
            problem.provenance.source,
            problem.provenance.target,
        )
        return SinkhornResult(
            problem=problem,
            source_potential=source_potential,
            target_potential=target_potential,
            epsilon=epsilon,
            transport_cost=transport_cost,
            regularization=regularization,
            regularized_cost=regularized_cost,
            dual_cost=dual_cost,
            diagnostics=diagnostics,
            provenance=provenance,
            block_size=self.block_size,
        )


def _safe_log(values: Array, /) -> Array:
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


__all__ = ["Sinkhorn"]
