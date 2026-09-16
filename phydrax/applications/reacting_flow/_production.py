#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class EnergyDepositionEvaluation(StrictModule):
    enthalpy_density_rate: Array
    instantaneous_power: Array
    cumulative_work: Array
    active: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class EnergyDepositionSourcePlan(StrictModule, NonTrainableState):
    cell_volumes: Array
    spatial_density: Array
    total_energy: float = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    end_time: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_volumes: ArrayLike,
        spatial_weights: ArrayLike,
        /,
        *,
        total_energy: float,
        start_time: float,
        end_time: float,
        source_id: str,
    ):
        volumes = np.asarray(cell_volumes, dtype=float)
        weights = np.asarray(spatial_weights, dtype=float)
        energy, start, end = map(float, (total_energy, start_time, end_time))
        identifier = str(source_id).strip()
        if (
            volumes.shape != weights.shape
            or volumes.size == 0
            or np.any(~np.isfinite(volumes))
            or np.any(volumes <= 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not isfinite(energy)
            or energy < 0.0
            or not isfinite(start)
            or not isfinite(end)
            or end <= start
            or not identifier
        ):
            raise ValueError(
                "Energy-deposition geometry, schedule, or identity is invalid."
            )
        normalization = float(np.sum(volumes * weights))
        if normalization <= 0.0:
            raise ValueError("Energy-deposition spatial weights have zero measure.")
        density = weights / normalization
        self.cell_volumes = jnp.asarray(volumes)
        self.spatial_density = jnp.asarray(density)
        self.total_energy = energy
        self.start_time = start
        self.end_time = end
        self.source_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-energy-deposition-source",
                "volumes": array_tree_fingerprint(volumes),
                "spatial_density": array_tree_fingerprint(density),
                "total_energy": energy,
                "start_time": start,
                "end_time": end,
                "source_id": identifier,
            }
        )

    def evaluate(self, time: ArrayLike, /) -> EnergyDepositionEvaluation:
        time_ = jnp.asarray(time, dtype=self.cell_volumes.dtype)
        if time_.shape != ():
            raise ValueError("Energy-deposition time must be scalar.")
        duration = self.end_time - self.start_time
        tau = jnp.clip((time_ - self.start_time) / duration, 0.0, 1.0)
        active = (time_ >= self.start_time) & (time_ <= self.end_time)
        envelope = jnp.where(
            active,
            1.0 - jnp.cos(2.0 * jnp.pi * tau),
            0.0,
        )
        power = self.total_energy / duration * envelope
        cumulative_fraction = tau - jnp.sin(2.0 * jnp.pi * tau) / (2.0 * jnp.pi)
        cumulative = self.total_energy * cumulative_fraction
        rate = power * self.spatial_density
        reconstructed = jnp.sum(self.cell_volumes * rate)
        finite = (
            jnp.isfinite(time_)
            & jnp.isfinite(power)
            & jnp.isfinite(cumulative)
            & jnp.all(jnp.isfinite(rate))
        )
        tolerance = 128.0 * jnp.finfo(rate.dtype).eps * jnp.maximum(jnp.abs(power), 1.0)
        successful = finite & (jnp.abs(reconstructed - power) <= tolerance)
        return EnergyDepositionEvaluation(
            rate, power, cumulative, active, finite, successful, self.plan_id
        )


class ReactingALERemapEvidence(StrictModule):
    species_extensive_defect: Array
    enthalpy_extensive_defect: Array
    volume_change: Array
    geometric_conservation_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactingALERemapResult(StrictModule):
    species_density: Array
    enthalpy_density: Array
    evidence: ReactingALERemapEvidence
    plan_id: str = eqx.field(static=True)


class FixedConnectivityReactingALERemapPlan(StrictModule, NonTrainableState):
    old_cell_volumes: Array
    new_cell_volumes: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, old_cell_volumes: ArrayLike, new_cell_volumes: ArrayLike, /):
        old = np.asarray(old_cell_volumes, dtype=float)
        new = np.asarray(new_cell_volumes, dtype=float)
        if (
            old.shape != new.shape
            or old.size == 0
            or np.any(~np.isfinite(old))
            or np.any(~np.isfinite(new))
            or np.any(old <= 0.0)
            or np.any(new <= 0.0)
        ):
            raise ValueError("ALE old/new cell volumes must be matching and positive.")
        self.old_cell_volumes = jnp.asarray(old)
        self.new_cell_volumes = jnp.asarray(new)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-connectivity-reacting-ale-remap",
                "old_volumes": array_tree_fingerprint(old),
                "new_volumes": array_tree_fingerprint(new),
            }
        )

    def remap(
        self, species_density: ArrayLike, enthalpy_density: ArrayLike, /
    ) -> ReactingALERemapResult:
        species = jnp.asarray(species_density, dtype=self.old_cell_volumes.dtype)
        enthalpy = jnp.asarray(enthalpy_density, dtype=self.old_cell_volumes.dtype)
        if (
            species.shape[:-1] != self.old_cell_volumes.shape
            or enthalpy.shape != self.old_cell_volumes.shape
        ):
            raise ValueError("ALE reacting fields must align with cell volumes.")
        ratio = self.old_cell_volumes / self.new_cell_volumes
        mapped_species = species * ratio[..., None]
        mapped_enthalpy = enthalpy * ratio
        species_before = jnp.sum(
            self.old_cell_volumes[..., None] * species,
            axis=tuple(range(species.ndim - 1)),
        )
        species_after = jnp.sum(
            self.new_cell_volumes[..., None] * mapped_species,
            axis=tuple(range(species.ndim - 1)),
        )
        enthalpy_before = jnp.sum(self.old_cell_volumes * enthalpy)
        enthalpy_after = jnp.sum(self.new_cell_volumes * mapped_enthalpy)
        species_defect = species_after - species_before
        enthalpy_defect = enthalpy_after - enthalpy_before
        volume_change = self.new_cell_volumes - self.old_cell_volumes
        gcl_residual = (
            mapped_species * self.new_cell_volumes[..., None]
            - species * self.old_cell_volumes[..., None]
        )
        finite = (
            jnp.all(jnp.isfinite(mapped_species))
            & jnp.all(jnp.isfinite(mapped_enthalpy))
            & jnp.all(jnp.isfinite(gcl_residual))
        )
        scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(species_before)), jnp.abs(enthalpy_before)), 1.0
        )
        tolerance = 256.0 * jnp.finfo(species.dtype).eps * scale
        successful = (
            finite
            & (jnp.max(jnp.abs(species_defect), initial=0.0) <= tolerance)
            & (jnp.abs(enthalpy_defect) <= tolerance)
            & (jnp.max(jnp.abs(gcl_residual), initial=0.0) <= tolerance)
        )
        evidence = ReactingALERemapEvidence(
            species_defect,
            enthalpy_defect,
            volume_change,
            gcl_residual,
            finite,
            successful,
            self.plan_id,
        )
        return ReactingALERemapResult(
            mapped_species, mapped_enthalpy, evidence, self.plan_id
        )


class ChemistryWorkScheduleState(StrictModule):
    estimated_cost: Array
    worker_assignment: Array
    accepted_epoch: Array
    plan_id: str = eqx.field(static=True)


class ChemistryWorkScheduleCandidate(StrictModule):
    proposed_state: ChemistryWorkScheduleState
    worker_load: Array
    load_imbalance: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ChemistryWorkSchedulePlan(StrictModule, NonTrainableState):
    block_count: int = eqx.field(static=True)
    worker_count: int = eqx.field(static=True)
    smoothing: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, block_count: int, worker_count: int, /, *, smoothing: float = 0.5):
        blocks, workers = int(block_count), int(worker_count)
        smoothing_ = float(smoothing)
        if blocks <= 0 or workers <= 0 or not 0.0 < smoothing_ <= 1.0:
            raise ValueError("Chemistry schedule capacities or smoothing are invalid.")
        self.block_count = blocks
        self.worker_count = workers
        self.smoothing = smoothing_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chemistry-work-schedule",
                "block_count": blocks,
                "worker_count": workers,
                "smoothing": smoothing_,
            }
        )

    def initialize(self, dtype=jnp.float64) -> ChemistryWorkScheduleState:
        costs = jnp.ones((self.block_count,), dtype=dtype)
        assignment, _ = self._assign(costs)
        return ChemistryWorkScheduleState(
            costs, assignment, jnp.asarray(0, dtype=jnp.int32), self.plan_id
        )

    def _assign(self, cost: Array, /) -> tuple[Array, Array]:
        order = jnp.argsort(-cost, stable=True)
        assignment = -jnp.ones((self.block_count,), dtype=jnp.int32)
        loads = jnp.zeros((self.worker_count,), dtype=cost.dtype)

        def body(index, carry):
            assigned, worker_load = carry
            block = order[index]
            worker = jnp.argmin(worker_load).astype(jnp.int32)
            return (
                assigned.at[block].set(worker),
                worker_load.at[worker].add(cost[block]),
            )

        return jax.lax.fori_loop(0, self.block_count, body, (assignment, loads))

    def propose(
        self,
        accepted: ChemistryWorkScheduleState,
        rhs_call_count: ArrayLike,
        /,
    ) -> ChemistryWorkScheduleCandidate:
        if (
            not isinstance(accepted, ChemistryWorkScheduleState)
            or accepted.plan_id != self.plan_id
        ):
            raise TypeError("accepted must belong to this chemistry schedule.")
        observed = jnp.asarray(rhs_call_count, dtype=accepted.estimated_cost.dtype)
        if observed.shape != (self.block_count,):
            raise ValueError("rhs_call_count must contain one value per block.")
        estimated = (
            1.0 - self.smoothing
        ) * accepted.estimated_cost + self.smoothing * observed
        assignment, loads = self._assign(estimated)
        mean = jnp.mean(loads)
        imbalance = (jnp.max(loads) - jnp.min(loads)) / jnp.maximum(mean, 1.0)
        finite = jnp.all(jnp.isfinite(observed)) & jnp.all(jnp.isfinite(loads))
        successful = finite & jnp.all(observed >= 0.0) & jnp.all(assignment >= 0)
        proposed = ChemistryWorkScheduleState(
            estimated,
            assignment,
            accepted.accepted_epoch + 1,
            self.plan_id,
        )
        return ChemistryWorkScheduleCandidate(
            proposed, loads, imbalance, finite, successful, self.plan_id
        )

    def commit(
        self,
        accepted: ChemistryWorkScheduleState,
        candidate: ChemistryWorkScheduleCandidate,
        commit: ArrayLike,
        /,
    ) -> ChemistryWorkScheduleState:
        if accepted.plan_id != self.plan_id or candidate.plan_id != self.plan_id:
            raise ValueError("Chemistry scheduling values belong to another plan.")
        decision = jnp.asarray(commit, dtype=bool)
        if decision.shape != ():
            raise ValueError("Chemistry schedule commit must be scalar.")
        return jax.tree.map(
            lambda proposed, prior: jnp.where(decision, proposed, prior),
            candidate.proposed_state,
            accepted,
        )


__all__ = [
    "ChemistryWorkScheduleCandidate",
    "ChemistryWorkSchedulePlan",
    "ChemistryWorkScheduleState",
    "EnergyDepositionEvaluation",
    "EnergyDepositionSourcePlan",
    "FixedConnectivityReactingALERemapPlan",
    "ReactingALERemapEvidence",
    "ReactingALERemapResult",
]
