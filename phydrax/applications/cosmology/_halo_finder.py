#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spatial import MortonAddressPlan, MortonRadiusRelationPlan
from ...sparse import EdgeRelation, KeyGroupPlan


FoFRealization: TypeAlias = Literal["direct", "cell_list", "morton_plane"]


class FoFFinderEvidence(NonTrainableState, StrictModule):
    """Capacity, convergence, and completeness evidence for one FoF catalogue."""

    required_groups: Array
    group_capacity: Array
    group_overflow: Array
    required_links: Array
    link_capacity: Array
    link_overflow: Array
    topology_complete: Array
    stable_ids_unique: Array
    converged: Array
    finite: Array
    successful: Array


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
    evidence: FoFFinderEvidence


def _pack_links(
    left: Array,
    right: Array,
    left_ids: Array,
    right_ids: Array,
    valid: Array,
    *,
    particle_count: int,
    capacity: int,
    topology_complete: Array,
) -> tuple[EdgeRelation, Array, Array]:
    flat_left = jnp.asarray(left, dtype=jnp.int32).reshape((-1,))
    flat_right = jnp.asarray(right, dtype=jnp.int32).reshape((-1,))
    flat_left_ids = jnp.asarray(left_ids).reshape((-1,))
    flat_right_ids = jnp.asarray(right_ids).reshape((-1,))
    flat_valid = jnp.asarray(valid, dtype=bool).reshape((-1,))
    order = jnp.lexsort(
        (
            flat_right_ids,
            flat_left_ids,
            (~flat_valid).astype(jnp.int32),
        )
    )
    selected = order[:capacity]
    required = jnp.sum(flat_valid, dtype=jnp.int32)
    overflow = required > capacity
    route_valid = (
        (jnp.arange(capacity, dtype=jnp.int32) < required) & topology_complete & ~overflow
    )
    selected_left = flat_left[selected]
    selected_right = flat_right[selected]
    return (
        EdgeRelation(
            jnp.where(route_valid, selected_left, 0),
            jnp.where(route_valid, selected_right, 0),
            source_size=particle_count,
            target_size=particle_count,
            valid=route_valid,
        ),
        required,
        overflow,
    )


class PeriodicFoFFinderPlan(StrictModule, NonTrainableState):
    box_size: tuple[float, float, float] = eqx.field(static=True)
    linking_length: float = eqx.field(static=True)
    maximum_groups: int = eqx.field(static=True)
    realization: FoFRealization = eqx.field(static=True)
    maximum_links: int = eqx.field(static=True)
    maximum_particles_per_cell: int = eqx.field(static=True)
    morton_maximum_depth: int = eqx.field(static=True)
    morton_maximum_nodes: int | None = eqx.field(static=True)
    morton_leaf_occupancy: int = eqx.field(static=True)
    cell_shape: tuple[int, int, int] = eqx.field(static=True)
    cell_strides: Array
    neighbor_offsets: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, float, float],
        linking_length: float,
        maximum_groups: int,
        /,
        *,
        realization: FoFRealization = "direct",
        maximum_links: int | None = None,
        maximum_particles_per_cell: int = 64,
        morton_maximum_depth: int = 21,
        morton_maximum_nodes: int | None = None,
        morton_leaf_occupancy: int = 32,
    ):
        lengths = tuple(float(value) for value in box_size)
        linking = float(linking_length)
        groups = int(maximum_groups)
        links = 0 if maximum_links is None else int(maximum_links)
        cell_occupancy = int(maximum_particles_per_cell)
        morton_depth = int(morton_maximum_depth)
        morton_nodes = None if morton_maximum_nodes is None else int(morton_maximum_nodes)
        morton_occupancy = int(morton_leaf_occupancy)
        if (
            len(lengths) != 3
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(linking)
            or linking <= 0.0
            or groups <= 0
        ):
            raise ValueError("FoF finder policy is invalid.")
        if realization not in ("direct", "cell_list", "morton_plane"):
            raise ValueError(
                "realization must be 'direct', 'cell_list', or 'morton_plane'."
            )
        if realization != "direct" and links <= 0:
            raise ValueError("Non-direct FoF realizations require maximum_links.")
        if cell_occupancy <= 0 or morton_occupancy <= 0:
            raise ValueError("FoF cell and Morton occupancies must be positive.")
        if morton_depth < 1 or morton_depth > 21:
            raise ValueError("morton_maximum_depth must lie in [1, 21].")
        shape = tuple(max(int(np.floor(length / linking)), 1) for length in lengths)
        strides = tuple(prod(shape[axis + 1 :]) for axis in range(3))
        offsets = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int32)
        self.box_size = lengths
        self.linking_length = linking
        self.maximum_groups = groups
        self.realization = realization
        self.maximum_links = links
        self.maximum_particles_per_cell = cell_occupancy
        self.morton_maximum_depth = morton_depth
        self.morton_maximum_nodes = morton_nodes
        self.morton_leaf_occupancy = morton_occupancy
        self.cell_shape = shape
        self.cell_strides = jnp.asarray(strides, dtype=jnp.int32)
        self.neighbor_offsets = jnp.asarray(offsets)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-fof-finder",
                "box_size": list(lengths),
                "linking_length": linking,
                "maximum_groups": groups,
                "realization": realization,
                "maximum_links": links,
                "maximum_particles_per_cell": cell_occupancy,
                "morton_maximum_depth": morton_depth,
                "morton_maximum_nodes": morton_nodes,
                "morton_leaf_occupancy": morton_occupancy,
            }
        )

    def _direct_links(
        self,
        ids: Array,
        position: Array,
        active: Array,
        finite_particles: Array,
    ) -> tuple[EdgeRelation, Array, Array, Array, Array]:
        first, second = np.triu_indices(int(ids.size), k=1)
        left = jnp.asarray(first, dtype=jnp.int32)
        right = jnp.asarray(second, dtype=jnp.int32)
        box = jnp.asarray(self.box_size, dtype=position.dtype)
        displacement = position[left] - position[right]
        displacement = displacement - box * jnp.round(displacement / box)
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        valid = (
            active[left]
            & active[right]
            & finite_particles[left]
            & finite_particles[right]
            & (distance_squared <= self.linking_length**2)
        )
        topology_complete = jnp.all(finite_particles | ~active)
        relation, required, overflow = _pack_links(
            left,
            right,
            ids[left],
            ids[right],
            valid,
            particle_count=int(ids.size),
            capacity=int(left.size),
            topology_complete=topology_complete,
        )
        return (
            relation,
            required,
            jnp.asarray(int(left.size), dtype=jnp.int32),
            overflow,
            topology_complete,
        )

    def _cell_list_links(
        self,
        ids: Array,
        position: Array,
        active: Array,
        finite_particles: Array,
    ) -> tuple[EdgeRelation, Array, Array, Array, Array]:
        particle_count = int(ids.size)
        box = jnp.asarray(self.box_size, dtype=position.dtype)
        safe_position = jnp.where(finite_particles[:, None], position, 0)
        wrapped = jnp.mod(safe_position, box)
        cell_width = box / jnp.asarray(self.cell_shape, dtype=position.dtype)
        coordinates = jnp.floor(wrapped / cell_width).astype(jnp.int32)
        coordinates = jnp.minimum(
            coordinates, jnp.asarray(self.cell_shape, dtype=jnp.int32) - 1
        )
        cell_ids = jnp.sum(coordinates * self.cell_strides, axis=-1)
        active_valid = active & finite_particles
        cell_count = prod(self.cell_shape)
        groups = KeyGroupPlan(
            particle_count,
            min(particle_count, cell_count),
            cell_count - 1,
            maximum_group_size=self.maximum_particles_per_cell,
        ).build(cell_ids, active_valid, stable_ids=ids)

        neighbor_coordinates = coordinates[:, None, :] + self.neighbor_offsets[None]
        neighbor_coordinates = jnp.mod(
            neighbor_coordinates,
            jnp.asarray(self.cell_shape, dtype=jnp.int32),
        )
        neighbor_ids = jnp.sum(
            neighbor_coordinates * self.cell_strides,
            axis=-1,
        )
        neighbor_order = jnp.argsort(neighbor_ids, axis=1)
        neighbor_ids = jnp.take_along_axis(neighbor_ids, neighbor_order, axis=1)
        previous = jnp.concatenate(
            (
                jnp.full((particle_count, 1), cell_count, dtype=jnp.int32),
                neighbor_ids[:, :-1],
            ),
            axis=1,
        )
        neighbor_valid = active_valid[:, None] & (neighbor_ids != previous)
        lookup = groups.lookup(neighbor_ids, valid=neighbor_valid)
        group_starts = groups.group_starts[lookup.group_slots]
        group_counts = groups.group_counts[lookup.group_slots]
        member_rank = jnp.arange(self.maximum_particles_per_cell, dtype=jnp.int32)
        storage = jnp.clip(
            group_starts[..., None] + member_rank,
            0,
            particle_count - 1,
        )
        right = groups.storage_to_logical[storage]
        right_valid = lookup.supported[..., None] & (
            member_rank < group_counts[..., None]
        )
        left = jnp.broadcast_to(
            jnp.arange(particle_count, dtype=jnp.int32)[:, None, None],
            right.shape,
        )
        left_ids = ids[left]
        right_ids = ids[right]
        candidate = active_valid[:, None, None] & right_valid & (left_ids < right_ids)
        displacement = wrapped[left] - wrapped[right]
        displacement = displacement - box * jnp.round(displacement / box)
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        candidate = candidate & (distance_squared <= self.linking_length**2)
        topology_complete = (
            jnp.all(finite_particles | ~active)
            & ~groups.evidence.group_overflow
            & ~groups.evidence.member_overflow
            & (groups.evidence.duplicate_stable_ids == 0)
        )
        relation, required, overflow = _pack_links(
            left,
            right,
            left_ids,
            right_ids,
            candidate,
            particle_count=particle_count,
            capacity=self.maximum_links,
            topology_complete=topology_complete,
        )
        return (
            relation,
            required,
            jnp.asarray(self.maximum_links, dtype=jnp.int32),
            overflow,
            topology_complete,
        )

    def _morton_links(
        self,
        ids: Array,
        position: Array,
        active: Array,
    ) -> tuple[EdgeRelation, Array, Array, Array, Array]:
        particle_count = int(ids.size)
        address = MortonAddressPlan(
            (0.0, 0.0, 0.0),
            self.box_size,
            self.morton_maximum_depth,
            periodic_axes=(True, True, True),
        )
        query = MortonRadiusRelationPlan(
            address,
            particle_count,
            particle_count,
            self.maximum_links,
            inclusive=True,
            maximum_nodes=self.morton_maximum_nodes,
            maximum_leaf_occupancy=self.morton_leaf_occupancy,
        ).query(
            position,
            position,
            self.linking_length,
            source_mask=active,
            target_mask=active,
            source_stable_ids=ids,
            target_stable_ids=ids,
            exclude_self=True,
            pair_once=True,
        )
        return (
            query.relation,
            query.evidence.required_pairs,
            query.evidence.pair_capacity,
            query.evidence.pair_overflow,
            query.evidence.topology_successful,
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
            or not jnp.issubdtype(ids.dtype, jnp.integer)
            or position.shape != velocity.shape
            or position.shape != (ids.size, 3)
            or mass.shape != ids.shape
            or active.shape != ids.shape
        ):
            raise ValueError("FoF particle arrays are inconsistent.")

        finite_particles = (
            jnp.all(jnp.isfinite(position), axis=1)
            & jnp.all(jnp.isfinite(velocity), axis=1)
            & jnp.isfinite(mass)
        )
        maximum_id = jnp.iinfo(ids.dtype).max
        identifier_order = jnp.lexsort((ids, (~active).astype(jnp.int32)))
        ordered_ids = ids[identifier_order]
        ordered_active = active[identifier_order]
        stable_ids_unique = ~jnp.any(
            ordered_active[1:]
            & ordered_active[:-1]
            & (ordered_ids[1:] == ordered_ids[:-1])
        ) & jnp.all(~active | (ids < maximum_id))
        if self.realization == "direct":
            relation, required_links, link_capacity, link_overflow, topology_complete = (
                self._direct_links(ids, position, active, finite_particles)
            )
        elif self.realization == "cell_list":
            relation, required_links, link_capacity, link_overflow, topology_complete = (
                self._cell_list_links(ids, position, active, finite_particles)
            )
        else:
            relation, required_links, link_capacity, link_overflow, topology_complete = (
                self._morton_links(ids, position, active)
            )
        topology_complete = topology_complete & stable_ids_unique

        labels = jnp.where(active, ids, maximum_id)

        def propagate(_, current):
            left = relation.source_indices
            right = relation.target_indices
            valid = relation.valid
            candidates = current
            candidates = candidates.at[left].min(
                jnp.where(valid, current[right], maximum_id)
            )
            candidates = candidates.at[right].min(
                jnp.where(valid, current[left], maximum_id)
            )
            return jnp.where(active, jnp.minimum(current, candidates), maximum_id)

        labels = jax.lax.fori_loop(0, ids.size, propagate, labels)
        checked = propagate(0, labels)
        converged = jnp.all(checked == labels)
        sorted_labels = jnp.sort(jnp.where(active, labels, maximum_id))
        label_boundary = (sorted_labels < maximum_id) & jnp.concatenate(
            (
                jnp.asarray([True]),
                sorted_labels[1:] != sorted_labels[:-1],
            )
        )
        required_groups = jnp.sum(label_boundary, dtype=jnp.int32)
        group_overflow = required_groups > self.maximum_groups
        group_positions = jnp.nonzero(
            label_boundary,
            size=self.maximum_groups,
            fill_value=0,
        )[0]
        group_ids = sorted_labels[group_positions]
        group_rank = jnp.arange(self.maximum_groups, dtype=jnp.int32)
        group_active = (
            (group_rank < required_groups)
            & ~group_overflow
            & topology_complete
            & ~link_overflow
        )
        group_ids = jnp.where(group_active, group_ids, maximum_id)
        membership = (
            group_active[:, None]
            & active[None, :]
            & (labels[None, :] == group_ids[:, None])
        )
        group_counts = jnp.sum(membership, axis=1, dtype=jnp.int32)
        group_masses = jnp.sum(jnp.where(membership, mass[None, :], 0), axis=1)
        safe_mass = jnp.where(group_masses > 0, group_masses, 1)
        box = jnp.asarray(self.box_size, dtype=position.dtype)
        angle = 2 * jnp.pi * position / box
        cosine = jnp.sum(
            jnp.where(
                membership[..., None],
                mass[None, :, None] * jnp.cos(angle)[None],
                0,
            ),
            axis=1,
        )
        sine = jnp.sum(
            jnp.where(
                membership[..., None],
                mass[None, :, None] * jnp.sin(angle)[None],
                0,
            ),
            axis=1,
        )
        group_angle = jnp.mod(jnp.arctan2(sine, cosine), 2 * jnp.pi)
        group_position = jnp.where(
            group_active[:, None],
            box[None] * group_angle / (2 * jnp.pi),
            0,
        )
        group_velocity = jnp.where(
            group_active[:, None],
            jnp.sum(
                jnp.where(
                    membership[..., None],
                    mass[None, :, None] * velocity[None],
                    0,
                ),
                axis=1,
            )
            / safe_mass[:, None],
            0,
        )
        compact_labels = jnp.searchsorted(group_ids, labels).astype(jnp.int32)
        compact_labels = jnp.where(
            active & ~group_overflow & topology_complete & ~link_overflow,
            compact_labels,
            self.maximum_groups,
        )
        finite = (
            jnp.all(finite_particles | ~active)
            & jnp.all(jnp.isfinite(group_masses))
            & jnp.all(jnp.isfinite(group_position))
            & jnp.all(jnp.isfinite(group_velocity))
        )
        successful = (
            converged
            & finite
            & stable_ids_unique
            & topology_complete
            & ~group_overflow
            & ~link_overflow
        )
        evidence = FoFFinderEvidence(
            required_groups=required_groups,
            group_capacity=jnp.asarray(self.maximum_groups, dtype=jnp.int32),
            group_overflow=group_overflow,
            required_links=required_links,
            link_capacity=link_capacity,
            link_overflow=link_overflow,
            topology_complete=topology_complete,
            stable_ids_unique=stable_ids_unique,
            converged=converged,
            finite=finite,
            successful=successful,
        )
        return FoFFinderResult(
            compact_labels,
            group_ids,
            group_masses,
            group_position,
            group_velocity,
            group_counts,
            group_active,
            required_links,
            converged,
            finite,
            successful,
            evidence,
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
    "FoFFinderEvidence",
    "FoFFinderResult",
    "FoFRealization",
    "HaloPropertyPlan",
    "HaloPropertyResult",
    "HaloUnbindingResult",
    "MergerMatchResult",
    "ParticleCoreOverlapTreePlan",
    "PeriodicFoFFinderPlan",
    "SubstructureCandidateResult",
]
