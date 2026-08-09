#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._geometry import column_logsumexp, row_logsumexp
from ._results import TransportProvenance
from ._status import TransportStatus
from ._unbalanced_blocks import coupling_statistics, generalized_kl
from ._unbalanced_problem import UnbalancedTransportProblem
from ._unbalanced_results import (
    UnbalancedSinkhornDiagnostics,
    UnbalancedSinkhornResult,
)


class UnbalancedSinkhorn(StrictModule):
    """Stabilized generalized log-domain Sinkhorn with two KL penalties."""

    epsilon: Array
    tolerance: Array
    mass_collapse_tolerance: Array
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
        mass_collapse_tolerance: ArrayLike = 0.0,
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
        collapse_ = jnp.asarray(mass_collapse_tolerance, dtype=float).reshape(())
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
        self.mass_collapse_tolerance = eqx.error_if(
            collapse_,
            ~jnp.isfinite(collapse_) | (collapse_ < 0.0),
            "mass_collapse_tolerance must be finite and nonnegative.",
        )
        self.max_iterations = maximum
        self.min_iterations = minimum
        self.check_every = interval
        self.block_size = None if block_size is None else int(block_size)
        self.early_stop = bool(early_stop)
        self.store_history = bool(store_history)

    def __call__(
        self,
        problem: UnbalancedTransportProblem,
        /,
        *,
        initial_potentials: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> UnbalancedSinkhornResult:
        if not isinstance(problem, UnbalancedTransportProblem):
            raise TypeError("problem must be an UnbalancedTransportProblem.")
        source_count, target_count = problem.shape
        dtype = jnp.result_type(
            problem.source.points,
            problem.target.points,
            self.epsilon,
            problem.source_marginal_penalty,
            problem.target_marginal_penalty,
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
        epsilon = self.epsilon.astype(dtype)
        tolerance = self.tolerance.astype(dtype)
        source_penalty = problem.source_marginal_penalty.astype(dtype)
        target_penalty = problem.target_marginal_penalty.astype(dtype)
        source_exponent = source_penalty / (source_penalty + epsilon)
        target_exponent = target_penalty / (target_penalty + epsilon)
        log_source = _safe_log(problem.source_weights)
        log_target = _safe_log(problem.target_weights)
        initial_carry = (
            source_initial,
            target_initial,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
        )

        def fixed_point(source_potential, target_potential):
            next_source = -source_exponent * epsilon * row_logsumexp(
                problem.cost,
                problem.source.points,
                problem.target.points,
                log_target + target_potential / epsilon,
                epsilon,
                block_size=self.block_size,
            )
            next_target = -target_exponent * epsilon * column_logsumexp(
                problem.cost,
                problem.source.points,
                problem.target.points,
                log_source + next_source / epsilon,
                epsilon,
                block_size=self.block_size,
            )
            log_source_relaxed_mass = logsumexp(
                log_source - next_source / source_penalty
            )
            log_target_relaxed_mass = logsumexp(
                log_target - next_target / target_penalty
            )
            shift = (
                source_penalty
                * target_penalty
                / (source_penalty + target_penalty)
                * (log_source_relaxed_mass - log_target_relaxed_mass)
            )
            return next_source + shift, next_target - shift

        def step(carry, index):
            (
                source_potential,
                target_potential,
                fixed_residual,
                first_converged,
                converged,
                failed,
            ) = carry
            frozen = failed | (converged if self.early_stop else False)

            def update(_):
                next_source, next_target = fixed_point(
                    source_potential,
                    target_potential,
                )
                finite = jnp.all(jnp.isfinite(next_source)) & jnp.all(
                    jnp.isfinite(next_target)
                )
                next_source = jnp.where(finite, next_source, source_potential)
                next_target = jnp.where(finite, next_target, target_potential)
                residual = jnp.maximum(
                    jnp.max(jnp.abs(next_source - source_potential)),
                    jnp.max(jnp.abs(next_target - target_potential)),
                ) / epsilon
                return next_source, next_target, residual, ~finite

            def keep(_):
                return source_potential, target_potential, fixed_residual, failed

            next_source, next_target, next_residual, next_failed = jax.lax.cond(
                frozen,
                keep,
                update,
                operand=None,
            )
            iteration = index + 1
            should_check = (
                (iteration % self.check_every == 0)
                | (iteration == self.max_iterations)
                | (iteration == self.min_iterations)
            )
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
            return (
                next_source,
                next_target,
                next_residual,
                next_first,
                converged | eligible,
                next_failed,
            ), jnp.where(should_check, next_residual, jnp.inf)

        final_carry, residuals = jax.lax.scan(
            step,
            initial_carry,
            jnp.arange(self.max_iterations, dtype=jnp.int32),
        )
        (
            source_potential,
            target_potential,
            _,
            first_converged,
            _,
            failed,
        ) = final_carry
        mapped_source, mapped_target = fixed_point(
            source_potential,
            target_potential,
        )
        fixed_residual = jnp.maximum(
            jnp.max(jnp.abs(mapped_source - source_potential)),
            jnp.max(jnp.abs(mapped_target - target_potential)),
        ) / epsilon
        (
            source_marginal,
            target_marginal,
            transport_cost,
            entropy_kl,
            transported_mass,
            objective_finite,
        ) = coupling_statistics(
            problem,
            source_potential,
            target_potential,
            epsilon,
            block_size=self.block_size,
        )
        source_marginal_kl = generalized_kl(
            source_marginal,
            problem.source_weights,
        )
        target_marginal_kl = generalized_kl(
            target_marginal,
            problem.target_weights,
        )
        entropy_regularization = epsilon * entropy_kl
        source_regularization = source_penalty * source_marginal_kl
        target_regularization = target_penalty * target_marginal_kl
        regularized_cost = (
            transport_cost
            + entropy_regularization
            + source_regularization
            + target_regularization
        )
        source_dual = source_penalty * jnp.sum(
            problem.source_weights
            * (1.0 - jnp.exp(-source_potential / source_penalty))
        )
        target_dual = target_penalty * jnp.sum(
            problem.target_weights
            * (1.0 - jnp.exp(-target_potential / target_penalty))
        )
        entropy_dual = epsilon * (
            problem.source_mass * problem.target_mass - transported_mass
        )
        dual_cost = source_dual + target_dual + entropy_dual
        source_stationarity = _stationarity_residual(
            source_potential,
            source_marginal,
            problem.source_weights,
            source_penalty,
        )
        target_stationarity = _stationarity_residual(
            target_potential,
            target_marginal,
            problem.target_weights,
            target_penalty,
        )
        finite_objective = (
            objective_finite
            & jnp.isfinite(source_marginal_kl)
            & jnp.isfinite(target_marginal_kl)
            & jnp.isfinite(regularized_cost)
            & jnp.isfinite(dual_cost)
            & jnp.isfinite(fixed_residual)
            & jnp.isfinite(source_stationarity)
            & jnp.isfinite(target_stationarity)
        )
        mass_collapsed = (
            transported_mass <= self.mass_collapse_tolerance.astype(dtype)
        )
        final_converged = (
            (fixed_residual <= tolerance)
            & (self.max_iterations >= self.min_iterations)
            & ~failed
            & finite_objective
            & ~mass_collapsed
        )
        status = jnp.where(
            failed,
            int(TransportStatus.NONFINITE_ITERATE),
            jnp.where(
                mass_collapsed,
                int(TransportStatus.TRANSPORT_MASS_COLLAPSED),
                jnp.where(
                    ~finite_objective,
                    int(TransportStatus.NONFINITE_OBJECTIVE),
                    jnp.where(
                        final_converged,
                        int(TransportStatus.CONVERGED),
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
            history = residuals[jnp.asarray(check_indices, dtype=jnp.int32)]
        else:
            history = jnp.empty((0,), dtype=dtype)
        actual_iterations = jnp.where(
            self.early_stop & (first_converged >= 0),
            first_converged,
            self.max_iterations,
        ).astype(jnp.int32)
        diagnostics = UnbalancedSinkhornDiagnostics(
            status=status,
            num_iterations=actual_iterations,
            first_converged_iteration=first_converged,
            fixed_point_residual=fixed_residual,
            source_stationarity_residual=source_stationarity,
            target_stationarity_residual=target_stationarity,
            primal_dual_gap=jnp.abs(regularized_cost - dual_cost),
            transported_mass=transported_mass,
            mass_collapsed=mass_collapsed,
            num_checks=jnp.asarray(len(check_indices), dtype=jnp.int32),
            residual_history=history,
        )
        provenance = TransportProvenance(
            "unbalanced_sinkhorn",
            problem.provenance.cost,
            "dense" if self.block_size is None else "blockwise",
            "unrolled",
            problem.provenance.source,
            problem.provenance.target,
        )
        return UnbalancedSinkhornResult(
            problem=problem,
            source_potential=source_potential,
            target_potential=target_potential,
            epsilon=epsilon,
            transported_mass=transported_mass,
            transport_cost=transport_cost,
            entropy_kl=entropy_kl,
            source_marginal_kl=source_marginal_kl,
            target_marginal_kl=target_marginal_kl,
            entropy_regularization=entropy_regularization,
            source_marginal_regularization=source_regularization,
            target_marginal_regularization=target_regularization,
            regularized_cost=regularized_cost,
            dual_cost=dual_cost,
            diagnostics=diagnostics,
            provenance=provenance,
            block_size=self.block_size,
        )


def _stationarity_residual(
    potential: Array,
    marginal: Array,
    reference: Array,
    penalty: Array,
    /,
) -> Array:
    active = reference > 0.0
    positive = marginal > 0.0
    residual = potential + penalty * (
        jnp.log(jnp.where(active & positive, marginal, 1.0))
        - jnp.log(jnp.where(active, reference, 1.0))
    )
    return jnp.max(
        jnp.where(active & positive, jnp.abs(residual), jnp.where(active, jnp.inf, 0.0))
    )


def _safe_log(values: Array, /) -> Array:
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


__all__ = ["UnbalancedSinkhorn"]
