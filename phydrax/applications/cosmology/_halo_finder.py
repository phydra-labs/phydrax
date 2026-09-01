#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FoFFinderResult(StrictModule):
    group_labels: Array
    group_ids: Array
    group_masses: Array
    group_positions: Array
    group_velocities: Array
    group_counts: Array
    group_active: Array
    linking_edges: Array
    converged: Array
    finite: Array
    successful: Array


class PeriodicFoFFinderPlan(StrictModule, NonTrainableState):
    box_size: tuple[float, float, float] = eqx.field(static=True)
    linking_length: float = eqx.field(static=True)
    maximum_groups: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, float, float],
        linking_length: float,
        maximum_groups: int,
        /,
    ):
        lengths = tuple(float(value) for value in box_size)
        linking = float(linking_length)
        groups = int(maximum_groups)
        if (
            len(lengths) != 3
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(linking)
            or linking <= 0.0
            or groups <= 0
        ):
            raise ValueError("FoF finder policy is invalid.")
        self.box_size = lengths
        self.linking_length = linking
        self.maximum_groups = groups
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-fof-finder",
                "box_size": list(lengths),
                "linking_length": linking,
                "maximum_groups": groups,
            }
        )

    def find(
        self,
        particle_ids: ArrayLike,
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike,
        active_mask: ArrayLike,
        /,
    ) -> FoFFinderResult:
        ids = jnp.asarray(particle_ids)
        position = jnp.asarray(positions)
        velocity = jnp.asarray(velocities, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        if (
            ids.ndim != 1
            or position.shape != velocity.shape
            or position.shape != (ids.size, 3)
            or mass.shape != ids.shape
            or active.shape != ids.shape
        ):
            raise ValueError("FoF particle arrays are inconsistent.")
        box = jnp.asarray(self.box_size, dtype=position.dtype)
        displacement = position[:, None, :] - position[None, :, :]
        displacement = displacement - box * jnp.round(displacement / box)
        distance_squared = jnp.sum(displacement**2, axis=-1)
        adjacency = (
            (distance_squared <= self.linking_length**2)
            & active[:, None]
            & active[None, :]
        )
        labels = jnp.where(active, jnp.arange(ids.size, dtype=jnp.int32), ids.size)

        def propagate(_, current):
            candidate = jnp.where(adjacency, current[None, :], ids.size)
            return jnp.minimum(current, jnp.min(candidate, axis=1))

        labels = jax.lax.fori_loop(0, ids.size, propagate, labels)
        checked = propagate(0, labels)
        converged = jnp.all(checked == labels)
        unique = jnp.unique(labels, size=self.maximum_groups, fill_value=ids.size)
        group_active = unique < ids.size
        membership = labels[None, :] == unique[:, None]
        group_counts = jnp.sum(membership & active[None, :], axis=1)
        group_masses = jnp.sum(jnp.where(membership, mass[None, :], 0.0), axis=1)
        safe_mass = jnp.where(group_masses > 0.0, group_masses, 1.0)
        angle = 2.0 * jnp.pi * position / box
        cosine = jnp.sum(
            jnp.where(
                membership[..., None],
                mass[None, :, None] * jnp.cos(angle)[None, :, :],
                0.0,
            ),
            axis=1,
        )
        sine = jnp.sum(
            jnp.where(
                membership[..., None],
                mass[None, :, None] * jnp.sin(angle)[None, :, :],
                0.0,
            ),
            axis=1,
        )
        group_angle = jnp.mod(jnp.arctan2(sine, cosine), 2.0 * jnp.pi)
        group_position = box[None, :] * group_angle / (2.0 * jnp.pi)
        group_velocity = (
            jnp.sum(
                jnp.where(
                    membership[..., None], mass[None, :, None] * velocity[None, :, :], 0.0
                ),
                axis=1,
            )
            / safe_mass[:, None]
        )
        group_ids = jnp.min(
            jnp.where(membership, ids[None, :], jnp.iinfo(ids.dtype).max), axis=1
        )
        finite = jnp.all(jnp.isfinite(group_masses)) & jnp.all(
            jnp.isfinite(group_position)
        )
        return FoFFinderResult(
            labels,
            group_ids,
            group_masses,
            group_position,
            group_velocity,
            group_counts,
            group_active,
            jnp.sum(jnp.triu(adjacency, k=1)),
            converged,
            finite,
            converged & finite,
        )


class HaloUnbindingResult(StrictModule):
    bound_mask: Array
    specific_energy: Array
    bulk_velocity: Array
    iterations: Array
    finite: Array
    successful: Array


class DirectHaloUnbindingPlan(StrictModule, NonTrainableState):
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        /,
        *,
        softening: float,
        maximum_iterations: int = 32,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Halo unbinding policy is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.maximum_iterations = iterations

    def unbind(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike,
        candidate_mask: ArrayLike,
        /,
    ) -> HaloUnbindingResult:
        position = jnp.asarray(positions)
        velocity = jnp.asarray(velocities, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        initial = jnp.asarray(candidate_mask, dtype=bool)

        def update(_, state):
            mask, iteration = state
            total_mass = jnp.sum(jnp.where(mask, mass, 0.0))
            bulk = jnp.sum(
                jnp.where(mask[:, None], mass[:, None] * velocity, 0.0), axis=0
            ) / jnp.maximum(total_mass, 1.0)
            displacement = position[None, :, :] - position[:, None, :]
            radius = jnp.sqrt(jnp.sum(displacement**2, axis=-1) + self.softening**2)
            pair = mask[None, :] & ~jnp.eye(position.shape[0], dtype=bool)
            potential = -self.gravitational_constant * jnp.sum(
                jnp.where(pair, mass[None, :] / radius, 0.0), axis=1
            )
            kinetic = 0.5 * jnp.sum((velocity - bulk) ** 2, axis=-1)
            energy = kinetic + potential
            next_mask = mask & (energy <= 0.0)
            changed = jnp.any(next_mask != mask)
            return next_mask, iteration + changed.astype(jnp.int32)

        bound, iterations = jax.lax.fori_loop(
            0, self.maximum_iterations, update, (initial, jnp.asarray(0, dtype=jnp.int32))
        )
        total_mass = jnp.sum(jnp.where(bound, mass, 0.0))
        bulk = jnp.sum(
            jnp.where(bound[:, None], mass[:, None] * velocity, 0.0), axis=0
        ) / jnp.maximum(total_mass, 1.0)
        displacement = position[None, :, :] - position[:, None, :]
        radius = jnp.sqrt(jnp.sum(displacement**2, axis=-1) + self.softening**2)
        potential = -self.gravitational_constant * jnp.sum(
            jnp.where(
                bound[None, :] & ~jnp.eye(position.shape[0], dtype=bool),
                mass[None, :] / radius,
                0.0,
            ),
            axis=1,
        )
        energy = 0.5 * jnp.sum((velocity - bulk) ** 2, axis=-1) + potential
        finite = jnp.all(jnp.isfinite(energy))
        return HaloUnbindingResult(bound, energy, bulk, iterations, finite, finite)


class HaloPropertyResult(StrictModule):
    mass_200m: Array
    radius_200m: Array
    center: Array
    bulk_velocity: Array
    particle_count: Array
    successful: Array


class HaloPropertyPlan(StrictModule, NonTrainableState):
    mean_density: float = eqx.field(static=True)

    def __init__(self, mean_density: float, /):
        density = float(mean_density)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("Halo mean density must be finite and positive.")
        self.mean_density = density

    def evaluate(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike,
        bound_mask: ArrayLike,
        center: ArrayLike,
        /,
    ) -> HaloPropertyResult:
        position = jnp.asarray(positions)
        velocity = jnp.asarray(velocities, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        bound = jnp.asarray(bound_mask, dtype=bool)
        center_ = jnp.asarray(center, dtype=position.dtype)
        radius = jnp.sqrt(jnp.sum((position - center_) ** 2, axis=-1))
        order = jnp.argsort(radius)
        sorted_radius = radius[order]
        sorted_mass = jnp.where(bound[order], mass[order], 0.0)
        enclosed = jnp.cumsum(sorted_mass)
        volume = 4.0 * jnp.pi * jnp.maximum(sorted_radius, 1.0e-12) ** 3 / 3.0
        overdensity = enclosed / volume / self.mean_density
        inside = overdensity >= 200.0
        index = jnp.maximum(jnp.sum(inside.astype(jnp.int32)) - 1, 0)
        mass_200m = enclosed[index]
        radius_200m = sorted_radius[index]
        count = jnp.sum(bound & (radius <= radius_200m))
        bulk = jnp.sum(
            jnp.where(
                (bound & (radius <= radius_200m))[:, None], mass[:, None] * velocity, 0.0
            ),
            axis=0,
        ) / jnp.maximum(mass_200m, 1.0)
        successful = jnp.isfinite(mass_200m) & (mass_200m > 0.0)
        return HaloPropertyResult(
            mass_200m, radius_200m, center_, bulk, count, successful
        )


class SubstructureCandidateResult(StrictModule):
    peak_mask: Array
    parent_peak: Array
    density: Array
    peak_count: Array
    successful: Array


class DensityPeakSubstructurePlan(StrictModule, NonTrainableState):
    neighbour_count: int = eqx.field(static=True)

    def __init__(self, neighbour_count: int = 16):
        count = int(neighbour_count)
        if count < 2:
            raise ValueError("Substructure neighbour count must be at least two.")
        self.neighbour_count = count

    def identify(
        self, positions: ArrayLike, masses: ArrayLike, host_mask: ArrayLike, /
    ) -> SubstructureCandidateResult:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        host = jnp.asarray(host_mask, dtype=bool)
        displacement = position[:, None, :] - position[None, :, :]
        distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
        distance = jnp.where(
            host[None, :] & ~jnp.eye(position.shape[0], dtype=bool), distance, jnp.inf
        )
        neighbours = jnp.sort(distance, axis=1)[:, : self.neighbour_count]
        smoothing = neighbours[:, -1]
        density = (
            self.neighbour_count
            * mass
            / (4.0 * jnp.pi * jnp.maximum(smoothing, 1.0e-12) ** 3 / 3.0)
        )
        higher = density[None, :] > density[:, None]
        nearest_higher = jnp.argmin(
            jnp.where(higher & host[None, :], distance, jnp.inf), axis=1
        )
        has_higher = jnp.any(higher & host[None, :], axis=1)
        peak = host & ~has_higher
        parent = jnp.where(peak, jnp.arange(position.shape[0]), nearest_higher)
        successful = jnp.all(jnp.isfinite(jnp.where(host, density, 0.0)))
        return SubstructureCandidateResult(
            peak, parent, density, jnp.sum(peak), successful
        )


class MergerMatchResult(StrictModule):
    descendant_indices: Array
    merits: Array
    overlap_counts: Array
    matched: Array
    successful: Array


class ParticleCoreOverlapTreePlan(StrictModule, NonTrainableState):
    core_size: int = eqx.field(static=True)
    minimum_overlap: int = eqx.field(static=True)

    def __init__(self, core_size: int, minimum_overlap: int, /):
        core = int(core_size)
        overlap = int(minimum_overlap)
        if core <= 0 or overlap <= 0 or overlap > core:
            raise ValueError("Merger matcher core/overlap policy is invalid.")
        self.core_size = core
        self.minimum_overlap = overlap

    def match(
        self,
        source_members: ArrayLike,
        source_binding_rank: ArrayLike,
        target_members: ArrayLike,
        /,
    ) -> MergerMatchResult:
        source = jnp.asarray(source_members)
        rank = jnp.asarray(source_binding_rank)
        target = jnp.asarray(target_members)
        if source.ndim != 2 or target.ndim != 2 or rank.shape != source.shape:
            raise ValueError("Merger membership/rank arrays are invalid.")
        core_order = jnp.argsort(rank, axis=1)[:, : self.core_size]
        core_ids = jnp.take_along_axis(source, core_order, axis=1)
        overlaps = jnp.sum(
            core_ids[:, None, :, None] == target[None, :, None, :], axis=(2, 3)
        )
        target_count = jnp.sum(target >= 0, axis=1)
        merit = overlaps**2 / jnp.maximum(self.core_size * target_count[None, :], 1)
        descendant = jnp.argmax(merit, axis=1)
        best_overlap = jnp.take_along_axis(overlaps, descendant[:, None], axis=1)[:, 0]
        best_merit = jnp.take_along_axis(merit, descendant[:, None], axis=1)[:, 0]
        matched = best_overlap >= self.minimum_overlap
        return MergerMatchResult(
            jnp.where(matched, descendant, -1),
            best_merit,
            best_overlap,
            matched,
            jnp.all(jnp.isfinite(best_merit)),
        )


__all__ = [
    "DensityPeakSubstructurePlan",
    "DirectHaloUnbindingPlan",
    "FoFFinderResult",
    "HaloPropertyPlan",
    "HaloPropertyResult",
    "HaloUnbindingResult",
    "MergerMatchResult",
    "ParticleCoreOverlapTreePlan",
    "PeriodicFoFFinderPlan",
    "SubstructureCandidateResult",
]
