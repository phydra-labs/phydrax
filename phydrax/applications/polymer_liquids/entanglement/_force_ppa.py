#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._snapshot import PreparedPrimitivePathSnapshot, PrimitivePathSnapshot


class ForcePrimitivePathPlan(StrictModule, NonTrainableState):
    bond_tension: float = eqx.field(static=True)
    excluded_volume_energy: float = eqx.field(static=True)
    excluded_volume_sigma: float = eqx.field(static=True)
    contact_distance: float = eqx.field(static=True)
    minimum_interchain_distance: float = eqx.field(static=True)
    maximum_bond_length: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    maximum_node_displacement: float = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    backtracking_steps: int = eqx.field(static=True)
    maximum_contacts: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        bond_tension: float,
        excluded_volume_energy: float,
        excluded_volume_sigma: float,
        contact_distance: float,
        minimum_interchain_distance: float,
        maximum_bond_length: float,
        step_size: float = 1.0e-3,
        maximum_node_displacement: float = 0.02,
        gradient_tolerance: float = 1.0e-6,
        maximum_iterations: int = 4096,
        backtracking_steps: int = 12,
        maximum_contacts: int = 100_000,
    ):
        values = tuple(
            float(value)
            for value in (
                bond_tension,
                excluded_volume_energy,
                excluded_volume_sigma,
                contact_distance,
                minimum_interchain_distance,
                maximum_bond_length,
                step_size,
                maximum_node_displacement,
                gradient_tolerance,
            )
        )
        iterations = int(maximum_iterations)
        backtracking = int(backtracking_steps)
        contacts = int(maximum_contacts)
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in values)
            or values[4] >= values[3]
            or iterations <= 0
            or backtracking <= 0
            or contacts <= 0
        ):
            raise ValueError("Force primitive-path controls are invalid.")
        (
            self.bond_tension,
            self.excluded_volume_energy,
            self.excluded_volume_sigma,
            self.contact_distance,
            self.minimum_interchain_distance,
            self.maximum_bond_length,
            self.step_size,
            self.maximum_node_displacement,
            self.gradient_tolerance,
        ) = values
        self.maximum_iterations = iterations
        self.backtracking_steps = backtracking
        self.maximum_contacts = contacts
        self.plan_id = canonical_fingerprint(
            {
                "kind": "force-primitive-path-plan",
                "values": values,
                "maximum_iterations": iterations,
                "backtracking_steps": backtracking,
                "maximum_contacts": contacts,
                "endpoint_policy": "fixed-open-chain-ends",
                "intrachain_excluded_volume": False,
                "interchain_uncrossability": "repulsive-bead-barrier",
            }
        )

    def prepare(
        self, snapshot: PreparedPrimitivePathSnapshot, /
    ) -> PreparedForcePrimitivePath:
        return PreparedForcePrimitivePath(self, snapshot)


class PrimitivePathContactState(StrictModule):
    left_chain: Array
    left_contour: Array
    right_chain: Array
    right_contour: Array
    distance: Array
    active: Array
    count: Array
    overflow: Array


class ForcePrimitivePathResult(StrictModule):
    primitive_positions: Array
    contour_lengths: Array
    contact_state: PrimitivePathContactState
    objective_history: Array
    history_mask: Array
    iterations: Array
    final_objective: Array
    gradient_norm: Array
    maximum_bond_length: Array
    minimum_interchain_distance: Array
    endpoint_residual: Array
    converged: Array
    uncrossability_valid: Array
    finite: Array
    successful: Array
    source_snapshot_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedForcePrimitivePath(StrictModule, NonTrainableState):
    plan: ForcePrimitivePathPlan
    snapshot: PreparedPrimitivePathSnapshot
    particle_chain: Array
    particle_contour: Array
    endpoint_mask: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ForcePrimitivePathPlan,
        snapshot: PreparedPrimitivePathSnapshot,
        /,
    ):
        if not isinstance(plan, ForcePrimitivePathPlan):
            raise TypeError("plan must be ForcePrimitivePathPlan.")
        if not isinstance(snapshot, PreparedPrimitivePathSnapshot):
            raise TypeError("snapshot must be PreparedPrimitivePathSnapshot.")
        capacity = snapshot.dynamics.system.capacity
        particle_chain = np.full((capacity,), -1, dtype=np.int32)
        particle_contour = np.full((capacity,), -1, dtype=np.int32)
        endpoint = np.zeros((capacity,), dtype=bool)
        indices = np.asarray(snapshot.layout.particle_indices)
        mask = np.asarray(snapshot.layout.chain_mask)
        for chain_index, (row, active) in enumerate(zip(indices, mask, strict=True)):
            chain = row[active]
            particle_chain[chain] = chain_index
            particle_contour[chain] = np.arange(chain.size, dtype=np.int32)
            endpoint[chain[0]] = True
            endpoint[chain[-1]] = True
        self.plan = plan
        self.snapshot = snapshot
        self.particle_chain = jnp.asarray(particle_chain)
        self.particle_contour = jnp.asarray(particle_contour)
        self.endpoint_mask = jnp.asarray(endpoint)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-force-primitive-path",
                "plan": plan.plan_id,
                "snapshot": snapshot.prepared_id,
            }
        )

    def _pair_displacement(self, positions: Array, cell_vectors: Array, /) -> Array:
        displacement = positions[:, None, :] - positions[None, :, :]
        cell = self.snapshot.dynamics.system.cell
        if cell is None:
            return displacement
        return cell.minimum_image_with_vectors(displacement, cell_vectors)

    def _objective(self, positions: Array, cell_vectors: Array, /) -> Array:
        indices = self.snapshot.expected_bonds
        bond = positions[indices[:, 1]] - positions[indices[:, 0]]
        bond_energy = 0.5 * self.plan.bond_tension * jnp.sum(bond * bond)
        displacement = self._pair_displacement(positions, cell_vectors)
        squared = jnp.sum(displacement * displacement, axis=-1)
        active = self.snapshot.dynamics.system.active_mask
        different_chain = (self.particle_chain[:, None] >= 0) & (
            self.particle_chain[:, None] != self.particle_chain[None, :]
        )
        pair_mask = (
            active[:, None]
            & active[None, :]
            & different_chain
            & jnp.triu(jnp.ones_like(squared, dtype=bool), 1)
        )
        distance = jnp.sqrt(jnp.where(pair_mask & (squared > 0.0), squared, 1.0))
        cutoff = 2.0 ** (1.0 / 6.0) * self.plan.excluded_volume_sigma
        ratio6 = (self.plan.excluded_volume_sigma / distance) ** 6
        wca = self.plan.excluded_volume_energy * (4.0 * (ratio6 * ratio6 - ratio6) + 1.0)
        return bond_energy + jnp.sum(jnp.where(pair_mask & (distance < cutoff), wca, 0.0))

    def evaluate(self, source: PrimitivePathSnapshot, /) -> ForcePrimitivePathResult:
        if not isinstance(source, PrimitivePathSnapshot):
            raise TypeError("source must be PrimitivePathSnapshot.")
        if source.prepared_id != self.snapshot.prepared_id:
            raise ValueError("Source snapshot belongs to another prepared analysis.")
        original = source.unwrapped_positions
        current = original
        objective_and_gradient = jax.value_and_grad(
            lambda value: self._objective(value, source.cell_vectors)
        )
        history: list[Array] = []
        iteration = 0
        converged = False
        for iteration in range(self.plan.maximum_iterations):
            objective, gradient = objective_and_gradient(current)
            gradient = jnp.where(self.endpoint_mask[:, None], 0.0, gradient)
            gradient_norm = jnp.sqrt(jnp.sum(gradient * gradient))
            history.append(objective)
            if bool(np.asarray(gradient_norm <= self.plan.gradient_tolerance)):
                converged = True
                break
            raw_step = -self.plan.step_size * gradient
            maximum = jnp.max(jnp.sqrt(jnp.sum(raw_step * raw_step, axis=-1)))
            trial_scale = jnp.minimum(
                1.0,
                self.plan.maximum_node_displacement / jnp.maximum(maximum, 1.0e-30),
            )
            accepted = False
            for _ in range(self.plan.backtracking_steps):
                candidate = current + trial_scale * raw_step
                candidate = jnp.where(self.endpoint_mask[:, None], original, candidate)
                candidate_objective = self._objective(candidate, source.cell_vectors)
                if bool(
                    np.asarray(
                        jnp.isfinite(candidate_objective)
                        & (candidate_objective <= objective)
                    )
                ):
                    current = candidate
                    accepted = True
                    break
                trial_scale = 0.5 * trial_scale
            if not accepted:
                break
        final_objective, gradient = objective_and_gradient(current)
        gradient = jnp.where(self.endpoint_mask[:, None], 0.0, gradient)
        gradient_norm = jnp.sqrt(jnp.sum(gradient * gradient))
        chain_positions = current[source.chain_indices]
        segment = chain_positions[:, 1:, :] - chain_positions[:, :-1, :]
        segment_mask = source.chain_mask[:, 1:] & source.chain_mask[:, :-1]
        bond_lengths = jnp.sqrt(jnp.sum(segment * segment, axis=-1))
        contour_lengths = jnp.sum(jnp.where(segment_mask, bond_lengths, 0.0), axis=-1)
        maximum_bond = jnp.max(jnp.where(segment_mask, bond_lengths, 0.0))
        endpoint_residual = jnp.max(
            jnp.abs(jnp.where(self.endpoint_mask[:, None], current - original, 0.0))
        )
        displacement = self._pair_displacement(current, source.cell_vectors)
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        active = self.snapshot.dynamics.system.active_mask
        different_chain = (self.particle_chain[:, None] >= 0) & (
            self.particle_chain[:, None] != self.particle_chain[None, :]
        )
        pair_mask = (
            active[:, None]
            & active[None, :]
            & different_chain
            & jnp.triu(jnp.ones_like(distance, dtype=bool), 1)
        )
        minimum_distance = jnp.min(jnp.where(pair_mask, distance, jnp.inf))
        contact_mask = pair_mask & (distance <= self.plan.contact_distance)
        pair_indices = jnp.argwhere(
            contact_mask, size=self.plan.maximum_contacts, fill_value=-1
        )
        contact_count = jnp.sum(contact_mask, dtype=jnp.int32)
        overflow = contact_count > self.plan.maximum_contacts
        safe_left = jnp.clip(pair_indices[:, 0], 0, current.shape[0] - 1)
        safe_right = jnp.clip(pair_indices[:, 1], 0, current.shape[0] - 1)
        contact_active = jnp.arange(self.plan.maximum_contacts) < jnp.minimum(
            contact_count, self.plan.maximum_contacts
        )
        contacts = PrimitivePathContactState(
            jnp.where(contact_active, self.particle_chain[safe_left], -1),
            jnp.where(contact_active, self.particle_contour[safe_left], -1),
            jnp.where(contact_active, self.particle_chain[safe_right], -1),
            jnp.where(contact_active, self.particle_contour[safe_right], -1),
            jnp.where(contact_active, distance[safe_left, safe_right], jnp.nan),
            contact_active,
            contact_count,
            overflow,
        )
        history_array = (
            jnp.full((self.plan.maximum_iterations,), jnp.nan, dtype=current.dtype)
            .at[: len(history)]
            .set(jnp.stack(history))
        )
        history_mask = jnp.arange(self.plan.maximum_iterations) < len(history)
        finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.isfinite(final_objective)
            & jnp.isfinite(gradient_norm)
            & jnp.all(jnp.isfinite(contour_lengths))
        )
        uncrossability = (
            (minimum_distance >= self.plan.minimum_interchain_distance)
            & (maximum_bond <= self.plan.maximum_bond_length)
            & ~overflow
        )
        successful = (
            jnp.asarray(converged) & finite & uncrossability & (endpoint_residual == 0.0)
        )
        return ForcePrimitivePathResult(
            current,
            contour_lengths,
            contacts,
            history_array,
            history_mask,
            jnp.asarray(iteration + 1, dtype=jnp.int32),
            final_objective,
            gradient_norm,
            maximum_bond,
            minimum_distance,
            endpoint_residual,
            jnp.asarray(converged),
            uncrossability,
            finite,
            successful,
            source.snapshot_id,
            self.prepared_id,
        )


__all__ = [
    "ForcePrimitivePathPlan",
    "ForcePrimitivePathResult",
    "PreparedForcePrimitivePath",
    "PrimitivePathContactState",
]
