#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._keldysh import (
    ClosedTimePathGrid,
    KeldyshTwoPointFunctions,
    NonequilibriumStatus,
)


TwoPITruncation = Literal["free", "hartree", "basketball"]


class TwoPISelfEnergy(StrictModule):
    statistical: Array
    spectral: Array
    retarded: Array
    support_mask: Array
    truncation: TwoPITruncation = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class MemorySupportEvidence(StrictModule):
    statistical_outside_support_residual: Array
    retarded_outside_support_residual: Array
    retarded_acausal_residual: Array
    spectral_diagonal_residual: Array
    finite: Array
    causal: Array
    prepared_id: str = eqx.field(static=True)


class Conserving2PIDiagnostics(StrictModule):
    discrete_energy: Array
    relative_energy_drift: Array
    statistical_symmetry_residual: Array
    spectral_antisymmetry_residual: Array
    equal_time_spectral_residual: Array
    memory: MemorySupportEvidence
    finite: Array
    conserved: Array
    status: Array
    truncation: TwoPITruncation = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class KadanoffBaymResult(StrictModule):
    two_point: KeldyshTwoPointFunctions
    self_energy: TwoPISelfEnergy
    diagnostics: Conserving2PIDiagnostics
    prepared_id: str = eqx.field(static=True)


class KadanoffBaym2PIPlan(StrictModule, NonTrainableState):
    """Uniform-grid scalar 2PI Kadanoff--Baym evolution with finite memory."""

    grid: ClosedTimePathGrid
    coupling: float = eqx.field(static=True)
    memory_steps: int = eqx.field(static=True)
    truncation: TwoPITruncation = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    maximum_work_elements: int = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: ClosedTimePathGrid,
        /,
        *,
        coupling: float,
        memory_steps: int,
        truncation: TwoPITruncation = "basketball",
        maximum_modes: int = 1024,
        maximum_work_elements: int = 20_000_000,
        energy_tolerance: float = 5.0e-3,
    ):
        if not isinstance(grid, ClosedTimePathGrid):
            raise TypeError("grid must be ClosedTimePathGrid.")
        coupling_ = float(coupling)
        memory = int(memory_steps)
        mode_capacity = int(maximum_modes)
        work_capacity = int(maximum_work_elements)
        tolerance = float(energy_tolerance)
        times = np.asarray(grid.plan.time_nodes)
        steps = np.diff(times)
        if (
            not np.isfinite(coupling_)
            or coupling_ < 0.0
            or memory < 1
            or memory >= times.size
            or truncation not in ("free", "hartree", "basketball")
            or mode_capacity <= 0
            or work_capacity <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-14)
        ):
            raise ValueError(
                "Kadanoff--Baym truncation, grid, or work budget is invalid."
            )
        if truncation == "free" and coupling_ != 0.0:
            raise ValueError("The free 2PI truncation requires zero coupling.")
        self.grid = grid
        self.coupling = coupling_
        self.memory_steps = memory
        self.truncation = truncation
        self.maximum_modes = mode_capacity
        self.maximum_work_elements = work_capacity
        self.energy_tolerance = tolerance
        self.time_step = float(steps[0])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-memory-kadanoff-baym-2pi",
                "grid": grid.grid_id,
                "coupling": coupling_,
                "memory_steps": memory,
                "truncation": truncation,
                "maximum_modes": mode_capacity,
                "maximum_work_elements": work_capacity,
                "energy_tolerance": tolerance,
                "time_integrator": "centered-second-order-causal-march",
            }
        )

    def prepare(self, frequencies: ArrayLike, /) -> "PreparedKadanoffBaym2PI":
        return PreparedKadanoffBaym2PI(self, frequencies)


class PreparedKadanoffBaym2PI(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: KadanoffBaym2PIPlan
    frequencies: Array
    support_mask: Array
    retarded_support_mask: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: KadanoffBaym2PIPlan, frequencies: ArrayLike, /):
        if not isinstance(plan, KadanoffBaym2PIPlan):
            raise TypeError("plan must be KadanoffBaym2PIPlan.")
        frequency = np.asarray(frequencies, dtype=np.float64)
        time_count = plan.grid.plan.time_nodes.size
        work = time_count * time_count * frequency.size * 4
        if (
            frequency.ndim != 1
            or frequency.size == 0
            or frequency.size > plan.maximum_modes
            or np.any(~np.isfinite(frequency))
            or np.any(frequency <= 0.0)
            or work > plan.maximum_work_elements
            or time_count * time_count * frequency.size
            > plan.grid.plan.maximum_two_point_elements
            or plan.time_step * float(np.max(frequency)) >= 2.0
        ):
            raise ValueError(
                "Kadanoff--Baym modes exceed stability or allocation bounds."
            )
        index = np.arange(time_count)
        support = np.abs(index[:, None] - index[None, :]) <= plan.memory_steps
        retarded = support & (index[:, None] >= index[None, :])
        self.plan = plan
        self.frequencies = jnp.asarray(frequency)
        self.support_mask = jnp.asarray(support)
        self.retarded_support_mask = jnp.asarray(retarded)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-memory-kadanoff-baym-2pi",
                "plan": plan.plan_id,
                "frequencies": array_tree_fingerprint(frequency),
                "support": array_tree_fingerprint(support),
            }
        )

    def _self_energy(self, statistical: Array, spectral: Array, /) -> tuple[Array, Array]:
        if self.plan.truncation != "basketball":
            return jnp.zeros_like(statistical), jnp.zeros_like(spectral)
        coupling_squared = self.plan.coupling**2
        sigma_statistical = (
            -coupling_squared / 6.0 * statistical * (statistical**2 - 0.75 * spectral**2)
        )
        sigma_spectral = (
            -0.5 * coupling_squared * spectral * (statistical**2 - spectral**2 / 12.0)
        )
        return sigma_statistical, sigma_spectral

    def evolve(self, occupations: ArrayLike, /) -> KadanoffBaymResult:
        occupation = jnp.asarray(occupations, dtype=self.frequencies.dtype)
        if occupation.shape != self.frequencies.shape:
            raise ValueError(
                "Occupations must provide one value per Kadanoff--Baym mode."
            )
        time_count = self.plan.grid.plan.time_nodes.size
        mode_count = self.frequencies.size
        dt = self.plan.time_step
        statistical = jnp.zeros(
            (time_count, time_count, mode_count), dtype=occupation.dtype
        )
        spectral = jnp.zeros_like(statistical)
        equal_time = (occupation + 0.5) / self.frequencies
        initial_mass = self.frequencies**2
        if self.plan.truncation != "free":
            initial_mass = initial_mass + 0.5 * self.plan.coupling * equal_time
        verlet_cosine = 1.0 - 0.5 * dt**2 * initial_mass
        first_statistical = equal_time * verlet_cosine
        statistical = statistical.at[0, 0].set(equal_time)
        statistical = statistical.at[1, 0].set(first_statistical)
        statistical = statistical.at[0, 1].set(first_statistical)
        statistical = statistical.at[1, 1].set(equal_time)
        spectral = spectral.at[1, 0].set(dt)
        spectral = spectral.at[0, 1].set(-dt)

        for current_index in range(1, time_count - 1):
            start = max(0, current_index - self.plan.memory_steps)
            source_indices = jnp.arange(start, current_index + 1)
            target_indices = jnp.arange(current_index + 1)
            current_statistical = statistical[current_index, : current_index + 1]
            current_spectral = spectral[current_index, : current_index + 1]
            previous_statistical = statistical[current_index - 1, : current_index + 1]
            previous_spectral = spectral[current_index - 1, : current_index + 1]
            source_statistical = statistical[current_index, start : current_index + 1]
            source_spectral = spectral[current_index, start : current_index + 1]
            sigma_statistical, sigma_spectral = self._self_energy(
                source_statistical, source_spectral
            )
            statistical_block = statistical[
                start : current_index + 1, : current_index + 1
            ]
            spectral_block = spectral[start : current_index + 1, : current_index + 1]
            first_statistical_memory = dt * contract(
                "s k,s t k->t k", sigma_spectral, statistical_block
            )
            earlier_mask = source_indices[:, None] <= target_indices[None, :]
            second_statistical_memory = dt * contract(
                "s k,s t k,s t->t k",
                sigma_statistical,
                spectral_block,
                earlier_mask,
            )
            spectral_mask = source_indices[:, None] >= target_indices[None, :]
            spectral_memory = dt * contract(
                "s k,s t k,s t->t k",
                sigma_spectral,
                spectral_block,
                spectral_mask,
            )
            effective_mass = self.frequencies**2
            if self.plan.truncation != "free":
                effective_mass = effective_mass + (
                    0.5 * self.plan.coupling * statistical[current_index, current_index]
                )
            statistical_acceleration = (
                -effective_mass * current_statistical
                - first_statistical_memory
                + second_statistical_memory
            )
            spectral_acceleration = -effective_mass * current_spectral - spectral_memory
            next_statistical = (
                2.0 * current_statistical
                - previous_statistical
                + dt**2 * statistical_acceleration
            )
            next_spectral = (
                2.0 * current_spectral - previous_spectral + dt**2 * spectral_acceleration
            )
            statistical = statistical.at[current_index + 1, : current_index + 1].set(
                next_statistical
            )
            statistical = statistical.at[: current_index + 1, current_index + 1].set(
                next_statistical
            )
            spectral = spectral.at[current_index + 1, : current_index + 1].set(
                next_spectral
            )
            spectral = spectral.at[: current_index + 1, current_index + 1].set(
                -next_spectral
            )
            future_statistical = next_statistical[start : current_index + 1]
            future_spectral = -next_spectral[start : current_index + 1]
            future_first_memory = dt * contract(
                "s k,s k->k", sigma_spectral, future_statistical
            )
            future_second_memory = dt * contract(
                "s k,s k->k", sigma_statistical, future_spectral
            )
            future_acceleration = (
                -effective_mass * next_statistical[current_index]
                - future_first_memory
                + future_second_memory
            )
            diagonal = (
                2.0 * next_statistical[current_index]
                - next_statistical[current_index - 1]
                + dt**2 * future_acceleration
            )
            statistical = statistical.at[current_index + 1, current_index + 1].set(
                diagonal
            )

        raw_statistical, raw_spectral = self._self_energy(statistical, spectral)
        sigma_statistical = jnp.where(self.support_mask[..., None], raw_statistical, 0.0)
        sigma_spectral = jnp.where(self.support_mask[..., None], raw_spectral, 0.0)
        sigma_retarded = jnp.where(
            self.retarded_support_mask[..., None], raw_spectral, 0.0
        )
        causal = self.plan.grid.causal_mask[..., None]
        retarded = jnp.where(causal, spectral, 0.0)
        advanced = -jnp.swapaxes(retarded, 0, 1)
        derivative = self._spectral_derivative(spectral)
        finite_two_point = (
            jnp.all(jnp.isfinite(occupation))
            & jnp.all(occupation >= 0.0)
            & jnp.all(jnp.isfinite(statistical))
            & jnp.all(jnp.isfinite(spectral))
        )
        two_point_status = jnp.where(
            finite_two_point,
            int(NonequilibriumStatus.SUCCESS),
            int(NonequilibriumStatus.NONFINITE),
        ).astype(jnp.int32)
        two_point = KeldyshTwoPointFunctions(
            statistical,
            spectral,
            retarded,
            advanced,
            derivative,
            finite_two_point,
            two_point_status,
            self.plan.grid.grid_id,
            self.prepared_id,
        )
        self_energy = TwoPISelfEnergy(
            sigma_statistical,
            sigma_spectral,
            sigma_retarded,
            self.support_mask,
            self.plan.truncation,
            self.prepared_id,
        )
        memory = self._memory_evidence(self_energy)
        diagnostics = self._diagnostics(two_point, self_energy, memory)
        return KadanoffBaymResult(two_point, self_energy, diagnostics, self.prepared_id)

    def _spectral_derivative(self, spectral: Array, /) -> Array:
        dt = self.plan.time_step
        derivative = jnp.zeros_like(spectral)
        derivative = derivative.at[1:-1].set((spectral[2:] - spectral[:-2]) / (2.0 * dt))
        derivative = derivative.at[0].set((spectral[1] - spectral[0]) / dt)
        derivative = derivative.at[-1].set((spectral[-1] - spectral[-2]) / dt)
        return derivative

    def _memory_evidence(self, self_energy: TwoPISelfEnergy, /) -> MemorySupportEvidence:
        outside = ~self.support_mask
        retarded_outside = ~self.retarded_support_mask
        acausal = ~self.plan.grid.causal_mask
        statistical_residual = jnp.max(
            jnp.abs(jnp.where(outside[..., None], self_energy.statistical, 0.0))
        )
        retarded_residual = jnp.max(
            jnp.abs(jnp.where(retarded_outside[..., None], self_energy.retarded, 0.0))
        )
        acausal_residual = jnp.max(
            jnp.abs(jnp.where(acausal[..., None], self_energy.retarded, 0.0))
        )
        diagonal = jnp.max(jnp.abs(jnp.diagonal(self_energy.spectral, axis1=0, axis2=1)))
        residuals = jnp.stack(
            (statistical_residual, retarded_residual, acausal_residual, diagonal)
        )
        finite = jnp.all(jnp.isfinite(residuals))
        causal_support = finite & (
            jnp.max(residuals) <= 64.0 * jnp.finfo(residuals.dtype).eps
        )
        return MemorySupportEvidence(
            statistical_residual,
            retarded_residual,
            acausal_residual,
            diagonal,
            finite,
            causal_support,
            self.prepared_id,
        )

    def _diagnostics(
        self,
        two_point: KeldyshTwoPointFunctions,
        self_energy: TwoPISelfEnergy,
        memory: MemorySupportEvidence,
        /,
    ) -> Conserving2PIDiagnostics:
        statistical = two_point.statistical
        spectral = two_point.spectral
        dt = self.plan.time_step
        diagonal = jnp.diagonal(statistical, axis1=0, axis2=1).T
        adjacent = jnp.stack(
            [statistical[index + 1, index] for index in range(diagonal.shape[0] - 1)]
        )
        kinetic = (diagonal[1:] + diagonal[:-1] - 2.0 * adjacent) / dt**2
        quadratic = self.frequencies**2 * adjacent
        midpoint_variance = 0.5 * (diagonal[1:] + diagonal[:-1])
        interaction = jnp.zeros_like(midpoint_variance)
        if self.plan.truncation != "free":
            interaction = self.plan.coupling / 8.0 * midpoint_variance**2
        memory_energy_rows = []
        for index in range(1, diagonal.shape[0]):
            start = max(0, index - self.plan.memory_steps)
            memory_density = (
                self_energy.retarded[index, start : index + 1]
                * statistical[index, start : index + 1]
                - self_energy.statistical[index, start : index + 1]
                * spectral[index, start : index + 1]
            )
            memory_energy_rows.append(-0.5 * dt * jnp.sum(memory_density, axis=0))
        memory_energy = jnp.stack(memory_energy_rows)
        energy = jnp.sum(
            0.5 * (kinetic + quadratic) + interaction + memory_energy,
            axis=-1,
        )
        energy_drift = jnp.max(jnp.abs(energy - energy[0])) / jnp.maximum(
            jnp.abs(energy[0]), 1.0
        )
        statistical_symmetry = jnp.max(
            jnp.abs(statistical - jnp.swapaxes(statistical, 0, 1))
        )
        spectral_antisymmetry = jnp.max(jnp.abs(spectral + jnp.swapaxes(spectral, 0, 1)))
        spectral_diagonal = jnp.max(jnp.abs(jnp.diagonal(spectral, axis1=0, axis2=1)))
        finite = (
            two_point.finite
            & memory.finite
            & jnp.all(jnp.isfinite(energy))
            & jnp.isfinite(energy_drift)
        )
        identity_scale = jnp.maximum(
            jnp.maximum(statistical_symmetry, spectral_antisymmetry),
            spectral_diagonal,
        )
        conserved = (
            finite
            & memory.causal
            & (identity_scale <= 1.0e-10)
            & (energy_drift <= self.plan.energy_tolerance)
        )
        status = jnp.where(
            conserved,
            int(NonequilibriumStatus.SUCCESS),
            jnp.where(
                finite,
                int(NonequilibriumStatus.CONSTRAINT_VIOLATION),
                int(NonequilibriumStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return Conserving2PIDiagnostics(
            energy,
            energy_drift,
            statistical_symmetry,
            spectral_antisymmetry,
            spectral_diagonal,
            memory,
            finite,
            conserved,
            status,
            self.plan.truncation,
            self.prepared_id,
        )


__all__ = [
    "Conserving2PIDiagnostics",
    "KadanoffBaym2PIPlan",
    "KadanoffBaymResult",
    "MemorySupportEvidence",
    "PreparedKadanoffBaym2PI",
    "TwoPISelfEnergy",
    "TwoPITruncation",
]
