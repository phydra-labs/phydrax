#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Mass-conserving fixed-pivot population-balance integration."""

from __future__ import annotations

from dataclasses import dataclass

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
        if float(overflow_number) < 0 or float(overflow_first_moment) < 0:
            raise ValueError("Population overflow reservoirs must be non-negative.")
        return cls(
            jnp.asarray(number),
            jnp.asarray(overflow_number),
            jnp.asarray(overflow_first_moment),
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
        if frequency.shape != centers.shape or np.any(frequency < 0):
            raise ValueError("Breakage frequencies must be non-negative and aligned.")
        if daughters.shape != kernel.shape or np.any(daughters < 0):
            raise ValueError("Daughter-number matrix must be non-negative and square.")
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
        centers = np.asarray(self.pivots)
        for left in range(bins):
            for right in range(bins):
                product = centers[left] + centers[right]
                event_rate = (
                    0.5
                    * self.aggregation_kernel[left, right]
                    * number[left]
                    * number[right]
                )
                if product > centers[-1]:
                    overflow_number_rate = overflow_number_rate + event_rate
                    overflow_moment_rate = overflow_moment_rate + product * event_rate
                    continue
                upper = int(np.searchsorted(centers, product, side="left"))
                if upper == 0 or centers[upper] == product:
                    birth = birth.at[upper].add(event_rate)
                    continue
                lower = upper - 1
                upper_weight = (product - centers[lower]) / (
                    centers[upper] - centers[lower]
                )
                birth = birth.at[lower].add((1 - upper_weight) * event_rate)
                birth = birth.at[upper].add(upper_weight * event_rate)

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
        if step_size_s <= 0 or not 0 < maximum_fractional_depletion <= 1:
            raise ValueError("Population time-integration controls are invalid.")
        initial_first = self.moments(state, jnp.asarray((0, 1)))[1]
        current = state
        elapsed = 0.0
        steps = 0
        while elapsed < step_size_s and steps < maximum_internal_steps:
            rate = self.rate(current)
            number = current.cell_number
            fractional_loss = jnp.where(
                (number > 0) & (rate.cell_number_rate < 0),
                -rate.cell_number_rate
                / jnp.maximum(number, jnp.finfo(number.dtype).tiny),
                0,
            )
            largest_loss = float(np.asarray(jnp.max(fractional_loss)))
            remaining = step_size_s - elapsed
            internal_step = (
                remaining
                if largest_loss == 0
                else min(remaining, maximum_fractional_depletion / largest_loss)
            )
            next_number = number + internal_step * rate.cell_number_rate
            if bool(jnp.any(next_number < -1e-12)):
                raise RuntimeError("Population positivity control failed.")
            current = SectionalPopulationState(
                jnp.maximum(next_number, 0),
                current.overflow_number + internal_step * rate.overflow_number_rate,
                current.overflow_first_moment
                + internal_step * rate.overflow_first_moment_rate,
            )
            elapsed += internal_step
            steps += 1
        if elapsed < step_size_s:
            raise RuntimeError("Population integration exceeded maximum internal steps.")
        final_first = self.moments(current, jnp.asarray((0, 1)))[1]
        return SectionalPopulationStep(
            current,
            steps,
            final_first - initial_first,
            jnp.min(current.cell_number),
        )


__all__ = [
    "ConservativeSectionalSolver",
    "SectionalPopulationRate",
    "SectionalPopulationState",
    "SectionalPopulationStep",
]
