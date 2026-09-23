#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Mass-conserving fixed-pivot population-balance integration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract


@dataclass(frozen=True, slots=True)
class SectionalPopulationState:
    cell_number: Array
    overflow_number: Array
    overflow_first_moment: Array

    @classmethod
    def create(
        cls,
        cell_number: ArrayLike,
        /,
        *,
        overflow_number: ArrayLike = 0.0,
        overflow_first_moment: ArrayLike = 0.0,
    ) -> SectionalPopulationState:
        number = np.asarray(cell_number, dtype=np.float64)
        if number.ndim != 1 or np.any(number < 0) or not np.all(np.isfinite(number)):
            raise ValueError(
                "Sectional particle numbers must be a finite non-negative vector."
            )
        overflow_number_ = float(overflow_number)
        overflow_moment_ = float(overflow_first_moment)
        if (
            not np.isfinite(overflow_number_)
            or not np.isfinite(overflow_moment_)
            or overflow_number_ < 0
            or overflow_moment_ < 0
            or ((overflow_number_ == 0) != (overflow_moment_ == 0))
        ):
            raise ValueError(
                "Population overflow reservoirs must be finite, nonnegative, and jointly empty/nonempty."
            )
        return cls(
            jnp.asarray(number),
            jnp.asarray(overflow_number_),
            jnp.asarray(overflow_moment_),
        )


@dataclass(frozen=True, slots=True)
class SectionalPopulationRate:
    cell_number_rate: Array
    overflow_number_rate: Array
    overflow_first_moment_rate: Array
    first_moment_residual: Array


@dataclass(frozen=True, slots=True)
class SectionalPopulationStep:
    state: SectionalPopulationState
    internal_steps: int
    first_moment_residual: Array
    minimum_cell_number: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class ConservativeSectionalSolver:
    """Fixed-pivot aggregation and conservative daughter redistribution.

    The primary unknown is particle number in each section, not density per unit
    internal coordinate. Aggregates beyond the largest pivot enter explicit
    overflow reservoirs, so no first moment disappears at the numerical boundary.
    """

    pivots: Array
    aggregation_kernel: Array
    breakage_frequency_s_inv: Array
    daughter_number: Array

    @classmethod
    def create(
        cls,
        pivots: ArrayLike,
        aggregation_kernel: ArrayLike,
        /,
        *,
        breakage_frequency_s_inv: ArrayLike | None = None,
        daughter_number: ArrayLike | None = None,
        conservation_tolerance: float = 1e-10,
    ) -> ConservativeSectionalSolver:
        centers = np.asarray(pivots, dtype=np.float64)
        kernel = np.asarray(aggregation_kernel, dtype=np.float64)
        if not isfinite(conservation_tolerance) or conservation_tolerance <= 0:
            raise ValueError(
                "Population conservation tolerance must be finite and positive."
            )
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(kernel)):
            raise ValueError("Population pivots and aggregation kernel must be finite.")
        if (
            centers.ndim != 1
            or centers.size < 2
            or np.any(centers <= 0)
            or np.any(np.diff(centers) <= 0)
        ):
            raise ValueError("Population pivots must be positive and increasing.")
        if kernel.shape != (centers.size, centers.size) or np.any(kernel < 0):
            raise ValueError("Aggregation kernel must be non-negative and square.")
        if not np.allclose(kernel, kernel.T, atol=conservation_tolerance, rtol=0):
            raise ValueError("Aggregation kernel must be symmetric.")
        frequency = (
            np.zeros_like(centers)
            if breakage_frequency_s_inv is None
            else np.asarray(breakage_frequency_s_inv, dtype=np.float64)
        )
        daughters = (
            np.eye(centers.size)
            if daughter_number is None
            else np.asarray(daughter_number, dtype=np.float64)
        )
        if (
            frequency.shape != centers.shape
            or not np.all(np.isfinite(frequency))
            or np.any(frequency < 0)
        ):
            raise ValueError(
                "Breakage frequencies must be finite, non-negative and aligned."
            )
        if (
            daughters.shape != kernel.shape
            or not np.all(np.isfinite(daughters))
            or np.any(daughters < 0)
        ):
            raise ValueError(
                "Daughter-number matrix must be finite, non-negative and square."
            )
        active_columns = frequency > 0
        daughter_moment = centers @ daughters
        if np.any(
            np.abs(daughter_moment[active_columns] - centers[active_columns])
            > conservation_tolerance * np.maximum(centers[active_columns], 1.0)
        ):
            raise ValueError("Each active daughter distribution must conserve size.")
        return cls(
            jnp.asarray(centers),
            jnp.asarray(kernel),
            jnp.asarray(frequency),
            jnp.asarray(daughters),
        )

    def moments(self, state: SectionalPopulationState, orders: ArrayLike, /) -> Array:
        powers = jnp.asarray(orders)
        moments = contract(
            "b,kb->k", state.cell_number, self.pivots[None, :] ** powers[:, None]
        )
        return (
            moments
            + jnp.where(powers == 0, state.overflow_number, 0)
            + jnp.where(powers == 1, state.overflow_first_moment, 0)
        )

    def rate(self, state: SectionalPopulationState, /) -> SectionalPopulationRate:
        number = state.cell_number
        if number.shape != self.pivots.shape:
            raise ValueError("Sectional state does not match population pivots.")
        bins = self.pivots.size
        loss = number * (self.aggregation_kernel @ number)
        birth = jnp.zeros_like(number)
        overflow_number_rate = jnp.asarray(0.0, dtype=number.dtype)
        overflow_moment_rate = jnp.asarray(0.0, dtype=number.dtype)
        for left in range(bins):
            for right in range(bins):
                product = self.pivots[left] + self.pivots[right]
                event_rate = (
                    0.5
                    * self.aggregation_kernel[left, right]
                    * number[left]
                    * number[right]
                )
                is_overflow = product > self.pivots[-1]
                upper = jnp.clip(
                    jnp.searchsorted(self.pivots, product, side="left"),
                    1,
                    bins - 1,
                )
                lower = upper - 1
                denominator = self.pivots[upper] - self.pivots[lower]
                upper_weight = jnp.clip(
                    (product - self.pivots[lower]) / denominator,
                    0.0,
                    1.0,
                )
                bounded = jnp.where(is_overflow, 0.0, event_rate)
                birth = birth.at[lower].add((1.0 - upper_weight) * bounded)
                birth = birth.at[upper].add(upper_weight * bounded)
                overflow_number_rate = overflow_number_rate + jnp.where(
                    is_overflow, event_rate, 0.0
                )
                overflow_moment_rate = overflow_moment_rate + jnp.where(
                    is_overflow, product * event_rate, 0.0
                )

        breaking = self.breakage_frequency_s_inv * number
        breakage_rate = self.daughter_number @ breaking - breaking
        cell_rate = birth - loss + breakage_rate
        first_residual = contract("b,b->", self.pivots, cell_rate) + overflow_moment_rate
        return SectionalPopulationRate(
            cell_rate,
            overflow_number_rate,
            overflow_moment_rate,
            first_residual,
        )

    def advance(
        self,
        state: SectionalPopulationState,
        step_size_s: float,
        /,
        *,
        maximum_fractional_depletion: float = 0.2,
        maximum_internal_steps: int = 10000,
    ) -> SectionalPopulationStep:
        if (
            not isfinite(step_size_s)
            or step_size_s <= 0
            or not isfinite(maximum_fractional_depletion)
            or not 0 < maximum_fractional_depletion <= 1
            or isinstance(maximum_internal_steps, bool)
            or not isinstance(maximum_internal_steps, int)
            or maximum_internal_steps <= 0
        ):
            raise ValueError("Population time-integration controls are invalid.")
        number = jnp.asarray(state.cell_number)
        if number.shape != self.pivots.shape:
            raise ValueError("Sectional state does not match population pivots.")
        number = eqx.error_if(
            number,
            jnp.any(~jnp.isfinite(number) | (number < 0))
            | ~jnp.isfinite(state.overflow_number)
            | ~jnp.isfinite(state.overflow_first_moment)
            | (state.overflow_number < 0)
            | (state.overflow_first_moment < 0),
            "Sectional population state must be finite and nonnegative.",
        )
        source = SectionalPopulationState(
            number,
            jnp.asarray(state.overflow_number),
            jnp.asarray(state.overflow_first_moment),
        )
        initial_first = self.moments(source, jnp.asarray((0, 1)))[1]
        dt = jnp.asarray(step_size_s, dtype=number.dtype)

        def continue_loop(loop_state):
            elapsed, steps, _, _, _, failed = loop_state
            return (elapsed < dt) & (steps < maximum_internal_steps) & ~failed

        def advance_loop(loop_state):
            elapsed, steps, current_number, overflow_number, overflow_moment, failed = (
                loop_state
            )
            current = SectionalPopulationState(
                current_number, overflow_number, overflow_moment
            )
            rate = self.rate(current)
            fractional_loss = jnp.where(
                (current_number > 0) & (rate.cell_number_rate < 0),
                -rate.cell_number_rate
                / jnp.maximum(current_number, jnp.finfo(current_number.dtype).tiny),
                0,
            )
            largest_loss = jnp.max(fractional_loss)
            remaining = dt - elapsed
            internal_step = jnp.where(
                largest_loss > 0,
                jnp.minimum(
                    remaining,
                    maximum_fractional_depletion / largest_loss,
                ),
                remaining,
            )
            candidate_number = current_number + internal_step * rate.cell_number_rate
            candidate_overflow_number = (
                overflow_number + internal_step * rate.overflow_number_rate
            )
            candidate_overflow_moment = (
                overflow_moment + internal_step * rate.overflow_first_moment_rate
            )
            valid = (
                jnp.all(jnp.isfinite(candidate_number))
                & jnp.all(candidate_number >= -1e-12)
                & jnp.isfinite(candidate_overflow_number)
                & (candidate_overflow_number >= 0)
                & jnp.isfinite(candidate_overflow_moment)
                & (candidate_overflow_moment >= 0)
                & jnp.isfinite(internal_step)
                & (internal_step > 0)
            )
            return (
                jnp.where(valid, elapsed + internal_step, elapsed),
                steps + 1,
                jnp.where(valid, jnp.maximum(candidate_number, 0), current_number),
                jnp.where(valid, candidate_overflow_number, overflow_number),
                jnp.where(valid, candidate_overflow_moment, overflow_moment),
                failed | ~valid,
            )

        (
            elapsed,
            steps,
            final_number,
            final_overflow_number,
            final_overflow_moment,
            failed,
        ) = jax.lax.while_loop(
            continue_loop,
            advance_loop,
            (
                jnp.asarray(0.0, dtype=number.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                source.cell_number,
                source.overflow_number,
                source.overflow_first_moment,
                jnp.asarray(False),
            ),
        )
        successful = ~failed & (elapsed >= dt)
        candidate = SectionalPopulationState(
            final_number, final_overflow_number, final_overflow_moment
        )
        accepted = SectionalPopulationState(
            jnp.where(successful, candidate.cell_number, source.cell_number),
            jnp.where(successful, candidate.overflow_number, source.overflow_number),
            jnp.where(
                successful,
                candidate.overflow_first_moment,
                source.overflow_first_moment,
            ),
        )
        final_first = self.moments(accepted, jnp.asarray((0, 1)))[1]
        return SectionalPopulationStep(
            accepted,
            steps,
            final_first - initial_first,
            jnp.min(accepted.cell_number),
            successful,
        )


__all__ = [
    "ConservativeSectionalSolver",
    "SectionalPopulationRate",
    "SectionalPopulationState",
    "SectionalPopulationStep",
]
