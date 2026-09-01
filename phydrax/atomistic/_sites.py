#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DenseParticleNeighborhoodPlan,
    ParticleDiscretization,
    ParticleNeighborhoodState,
    ParticleSetPlan,
)
from ..discretization.particle._periodic_cell import ParticleCell


class AtomisticSiteDomain(StrEnum):
    DOF_ATOMS = "dof-atoms"
    PHYSICAL_ATOMS = "physical-atoms"
    INTERACTION_SITES = "interaction-sites"


class VirtualSiteKind(StrEnum):
    WEIGHTED = "weighted"
    LOCAL_FRAME = "local-frame"


class VirtualSiteRule(StrictModule, NonTrainableState):
    kind: VirtualSiteKind = eqx.field(static=True)
    site_id: int = eqx.field(static=True)
    parent_ids: Array
    coefficients: Array
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: VirtualSiteKind,
        site_id: int,
        parent_ids: ArrayLike,
        coefficients: ArrayLike,
        /,
    ):
        if not isinstance(kind, VirtualSiteKind):
            raise TypeError("kind must be VirtualSiteKind.")
        parents = np.asarray(parent_ids)
        values = np.asarray(coefficients, dtype=float)
        if parents.ndim != 1 or not np.issubdtype(parents.dtype, np.integer):
            raise TypeError("parent_ids must be an integer vector.")
        if len(set(int(value) for value in parents)) != parents.size:
            raise ValueError("Virtual-site parents must be unique.")
        if kind is VirtualSiteKind.WEIGHTED:
            if parents.size < 1 or values.shape != (parents.size,):
                raise ValueError("Weighted sites require one coefficient per parent.")
            if not np.isclose(np.sum(values), 1.0):
                raise ValueError("Weighted-site coefficients must sum to one.")
        elif parents.size != 3 or values.shape != (3,):
            raise ValueError(
                "Local-frame sites require three parents and xyz coefficients."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("Virtual-site coefficients must be finite.")
        identifier = int(site_id)
        if identifier in set(int(value) for value in parents):
            raise ValueError("A virtual site cannot be its own parent.")
        self.kind = kind
        self.site_id = identifier
        self.parent_ids = jnp.asarray(parents, dtype=jnp.int64)
        self.coefficients = jnp.asarray(values)
        self.rule_id = canonical_fingerprint(
            {
                "kind": "atomistic-virtual-site-rule",
                "site_kind": kind.value,
                "site_id": identifier,
                "arrays": array_tree_fingerprint(
                    {"parent_ids": parents, "coefficients": values}
                ),
            }
        )


class AtomisticInteractionSitePlan(StrictModule, NonTrainableState):
    site_ids: Array
    atomic_numbers: Array
    site_type_ids: Array
    charges: Array
    active_mask: Array
    physical_mask: Array
    output_mask: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_ids: ArrayLike,
        atomic_numbers: ArrayLike,
        site_type_ids: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        physical_mask: ArrayLike | None = None,
        output_mask: ArrayLike | None = None,
    ):
        ids = np.asarray(site_ids)
        numbers = np.asarray(atomic_numbers)
        types = np.asarray(site_type_ids)
        charge = np.asarray(charges, dtype=float)
        if ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("site_ids must be a non-empty integer vector.")
        expected = ids.shape
        if (
            numbers.shape != expected
            or types.shape != expected
            or charge.shape != expected
            or not np.issubdtype(numbers.dtype, np.integer)
            or not np.issubdtype(types.dtype, np.integer)
        ):
            raise TypeError("Interaction-site properties must align with site_ids.")
        if np.unique(ids).size != ids.size:
            raise ValueError("Interaction-site IDs must be unique.")
        active = (
            np.ones(expected, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        physical = (
            numbers > 0
            if physical_mask is None
            else np.asarray(physical_mask, dtype=bool)
        )
        output = (
            physical.copy()
            if output_mask is None
            else np.asarray(output_mask, dtype=bool)
        )
        if (
            active.shape != expected
            or physical.shape != expected
            or output.shape != expected
        ):
            raise ValueError("Interaction-site masks must align with site_ids.")
        if np.any(physical & (numbers <= 0)) or np.any(~physical & (numbers != 0)):
            raise ValueError(
                "Physical sites require atomic numbers; virtual sites use zero."
            )
        if np.any(types[active] < 0) or np.any(~np.isfinite(charge[active])):
            raise ValueError("Active interaction-site types and charges are invalid.")
        arrays = {
            "site_ids": ids.astype(np.int64, copy=False),
            "atomic_numbers": numbers.astype(np.int32, copy=False),
            "site_type_ids": types.astype(np.int32, copy=False),
            "charges": np.where(active, charge, 0.0),
            "active_mask": active,
            "physical_mask": physical & active,
            "output_mask": output & active,
        }
        for name, value in arrays.items():
            setattr(self, name, jnp.asarray(value))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-interaction-site-plan",
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    @property
    def capacity(self) -> int:
        return int(self.site_ids.size)


class AtomisticInteractionSiteState(StrictModule):
    positions: Array
    active_mask: Array
    physical_mask: Array
    output_mask: Array
    frame_margin: Array
    successful: Array
    coordinate_map_id: str = eqx.field(static=True)


class AbstractAtomisticCoordinateMapPlan(StrictModule, NonTrainableState):
    plan_id: AbstractAttribute[str]
    sites: AbstractAttribute[AtomisticInteractionSitePlan]

    @abc.abstractmethod
    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> "PreparedAtomisticCoordinateMap":
        raise NotImplementedError


class AtomisticCoordinateMapPlan(AbstractAtomisticCoordinateMapPlan):
    dof_particle_ids: Array
    sites: AtomisticInteractionSitePlan
    physical_dof_indices: Array
    virtual_rules: tuple[VirtualSiteRule, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dof_particle_ids: ArrayLike,
        sites: AtomisticInteractionSitePlan,
        physical_dof_indices: ArrayLike,
        /,
        *,
        virtual_rules: tuple[VirtualSiteRule, ...] = (),
    ):
        dof_ids = np.asarray(dof_particle_ids)
        physical = np.asarray(physical_dof_indices)
        if dof_ids.ndim != 1 or not np.issubdtype(dof_ids.dtype, np.integer):
            raise TypeError("dof_particle_ids must be an integer vector.")
        if physical.shape != (sites.capacity,) or not np.issubdtype(
            physical.dtype, np.integer
        ):
            raise TypeError("physical_dof_indices must align with interaction sites.")
        rules = tuple(virtual_rules)
        if any(not isinstance(rule, VirtualSiteRule) for rule in rules):
            raise TypeError("virtual_rules must contain VirtualSiteRule values.")
        virtual_by_id = {rule.site_id: rule for rule in rules}
        site_ids = np.asarray(sites.site_ids, dtype=np.int64)
        physical_mask = np.asarray(sites.physical_mask, dtype=bool)
        if any(
            int(site_ids[index]) not in virtual_by_id
            for index in np.flatnonzero(~physical_mask & np.asarray(sites.active_mask))
        ):
            raise ValueError("Every active virtual site requires one virtual-site rule.")
        if any(
            identifier not in set(int(value) for value in site_ids)
            for identifier in virtual_by_id
        ):
            raise ValueError("Virtual-site rule references an unknown site ID.")
        if np.any(physical_mask & ((physical < 0) | (physical >= dof_ids.size))):
            raise ValueError("Physical sites require valid DOF indices.")
        if np.any(~physical_mask & (physical != -1)):
            raise ValueError("Virtual sites use physical_dof_indices=-1.")
        self.dof_particle_ids = jnp.asarray(dof_ids, dtype=jnp.int64)
        self.sites = sites
        self.physical_dof_indices = jnp.asarray(physical, dtype=jnp.int32)
        self.virtual_rules = rules
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-coordinate-map-plan",
                "dof_ids": array_tree_fingerprint(dof_ids),
                "sites": sites.plan_id,
                "physical_dof_indices": array_tree_fingerprint(physical),
                "virtual_rules": [rule.rule_id for rule in rules],
            }
        )

    @classmethod
    def identity(
        cls,
        particle_ids: ArrayLike,
        atomic_numbers: ArrayLike,
        site_type_ids: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> "AtomisticCoordinateMapPlan":
        ids = np.asarray(particle_ids)
        active = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        sites = AtomisticInteractionSitePlan(
            ids,
            atomic_numbers,
            site_type_ids,
            charges,
            active_mask=active,
            physical_mask=active,
            output_mask=active,
        )
        indices = np.where(active, np.arange(ids.size), 0)
        return cls(ids, sites, indices)

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> "PreparedAtomisticCoordinateMap":
        return PreparedAtomisticCoordinateMap(self, particles)


class PreparedAtomisticCoordinateMap(StrictModule, NonTrainableState):
    plan: AtomisticCoordinateMapPlan
    particles: ParticleDiscretization
    site_particles: ParticleDiscretization
    physical_dof_indices: Array
    virtual_site_indices: Array
    virtual_parent_indices: tuple[Array, ...]
    pair_left: Array
    pair_right: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: AtomisticCoordinateMapPlan, particles: ParticleDiscretization, /
    ):
        if not isinstance(plan, AtomisticCoordinateMapPlan):
            raise TypeError("plan must be AtomisticCoordinateMapPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if not np.array_equal(
            np.asarray(plan.dof_particle_ids), np.asarray(particles.particle_ids)
        ):
            raise ValueError("Coordinate-map DOF identity does not match particles.")
        site_particles = ParticleSetPlan(
            plan.sites.site_ids,
            np.ones(
                (plan.sites.capacity,), dtype=np.asarray(particles.safe_masses).dtype
            ),
            ambient_dimension=3,
            active_mask=plan.sites.active_mask,
            name="atomistic-interaction-sites",
            domain_labels=("atomistic", "interaction_site"),
            coordinate_dtype=particles.safe_masses.dtype,
        ).prepare(numeric_version=particles.numeric_version)
        dof_rank = {
            int(value): index
            for index, value in enumerate(np.asarray(particles.particle_ids))
        }
        site_rank = {
            int(value): index
            for index, value in enumerate(np.asarray(plan.sites.site_ids))
        }
        parent_indices = []
        virtual_indices = []
        for rule in plan.virtual_rules:
            resolved = []
            for identifier in np.asarray(rule.parent_ids):
                key = int(identifier)
                if key not in dof_rank:
                    raise ValueError(
                        f"Virtual-site parent ID {key} is not a DOF particle."
                    )
                resolved.append(dof_rank[key])
            parent_indices.append(jnp.asarray(resolved, dtype=jnp.int32))
            virtual_indices.append(site_rank[rule.site_id])
        left, right = np.triu_indices(plan.sites.capacity, 1)
        self.plan = plan
        self.particles = particles
        self.site_particles = site_particles
        self.physical_dof_indices = plan.physical_dof_indices
        self.virtual_site_indices = jnp.asarray(virtual_indices, dtype=jnp.int32)
        self.virtual_parent_indices = tuple(parent_indices)
        self.pair_left = jnp.asarray(left, dtype=jnp.int32)
        self.pair_right = jnp.asarray(right, dtype=jnp.int32)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-coordinate-map",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "sites": site_particles.prepared_id,
            }
        )

    def realize(
        self,
        dof_positions: ArrayLike,
        /,
        *,
        cell: ParticleCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> AtomisticInteractionSiteState:
        positions = jnp.asarray(dof_positions, dtype=self.particles.safe_masses.dtype)
        expected = (self.particles.capacity, 3)
        if positions.shape != expected:
            raise ValueError(f"dof_positions must have shape {expected}.")
        dynamic_fractional = (
            None
            if fractional_positions is None
            else jnp.asarray(fractional_positions, dtype=positions.dtype)
        )
        dynamic_vectors = (
            None
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=positions.dtype)
        )
        if (dynamic_fractional is None) != (dynamic_vectors is None):
            raise ValueError(
                "Dynamic coordinate-map geometry requires fractions and cell vectors."
            )
        if dynamic_fractional is not None and (
            cell is None
            or dynamic_fractional.shape != expected
            or dynamic_vectors.shape != (3, 3)
        ):
            raise ValueError("Dynamic coordinate-map cell geometry is invalid.")
        safe_indices = jnp.maximum(self.physical_dof_indices, 0)
        sites = positions[safe_indices]
        sites = jnp.where(self.plan.sites.physical_mask[:, None], sites, 0.0)
        margin = jnp.asarray(jnp.inf, dtype=positions.dtype)
        successful = jnp.all(
            jnp.isfinite(jnp.where(self.particles.active_mask[:, None], positions, 0.0))
        )
        for index, rule in enumerate(self.plan.virtual_rules):
            parent_indices = self.virtual_parent_indices[index]
            parents = positions[parent_indices]
            dynamic_parents = (
                None if dynamic_fractional is None else dynamic_fractional[parent_indices]
            )
            if rule.kind is VirtualSiteKind.WEIGHTED:
                anchor = parents[0]
                if dynamic_parents is not None:
                    relative_fractional = dynamic_parents - dynamic_parents[0]
                    central = jax.lax.stop_gradient(
                        jnp.round(relative_fractional).astype(jnp.int32)
                    )
                    central = jnp.where(cell.periodic_mask, central, 0)
                    parents = anchor + contract(
                        "pi,ij->pj",
                        relative_fractional - central.astype(positions.dtype),
                        dynamic_vectors,
                    )
                elif cell is not None:
                    relative = cell.minimum_image(parents - anchor)
                    parents = anchor + relative
                virtual = contract(
                    "p,pd->d", rule.coefficients.astype(positions.dtype), parents
                )
                local_margin = (
                    jnp.min(jnp.sqrt(jnp.sum((parents[1:] - parents[:1]) ** 2, axis=-1)))
                    if parents.shape[0] > 1
                    else jnp.asarray(jnp.inf, dtype=positions.dtype)
                )
            else:
                origin = parents[0]
                if dynamic_parents is not None:
                    relative_fractional = dynamic_parents[1:3] - dynamic_parents[0]
                    central = jax.lax.stop_gradient(
                        jnp.round(relative_fractional).astype(jnp.int32)
                    )
                    central = jnp.where(cell.periodic_mask, central, 0)
                    relative = contract(
                        "pi,ij->pj",
                        relative_fractional - central.astype(positions.dtype),
                        dynamic_vectors,
                    )
                    first, second = relative[0], relative[1]
                else:
                    first = parents[1] - origin
                    second = parents[2] - origin
                    if cell is not None:
                        first = cell.minimum_image(first)
                        second = cell.minimum_image(second)
                first_norm = jnp.sqrt(jnp.sum(first * first))
                x_axis = first / jnp.where(first_norm > 0.0, first_norm, 1.0)
                normal = jnp.cross(first, second)
                normal_norm = jnp.sqrt(jnp.sum(normal * normal))
                z_axis = normal / jnp.where(normal_norm > 0.0, normal_norm, 1.0)
                y_axis = jnp.cross(z_axis, x_axis)
                virtual = (
                    origin
                    + rule.coefficients[0] * x_axis
                    + rule.coefficients[1] * y_axis
                    + rule.coefficients[2] * z_axis
                )
                local_margin = jnp.minimum(first_norm, normal_norm)
                successful = successful & (first_norm > 0.0) & (normal_norm > 0.0)
            sites = sites.at[self.virtual_site_indices[index]].set(virtual)
            margin = jnp.minimum(margin, local_margin)
        sites = jnp.where(self.plan.sites.active_mask[:, None], sites, 0.0)
        successful = successful & jnp.all(jnp.isfinite(sites))
        return AtomisticInteractionSiteState(
            sites,
            self.plan.sites.active_mask,
            self.plan.sites.physical_mask,
            self.plan.sites.output_mask,
            margin,
            successful,
            self.prepared_id,
        )

    def force_pullback(
        self,
        dof_positions: ArrayLike,
        site_forces: ArrayLike,
        /,
        *,
        cell: ParticleCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> Array:
        positions = jnp.asarray(dof_positions, dtype=self.particles.safe_masses.dtype)
        forces = jnp.asarray(site_forces, dtype=positions.dtype)
        if forces.shape != (self.plan.sites.capacity, 3):
            raise ValueError("site_forces must match interaction-site capacity.")
        _, pullback = jax.vjp(
            lambda value: (
                self.realize(
                    value,
                    cell=cell,
                    fractional_positions=fractional_positions,
                    cell_vectors=cell_vectors,
                ).positions
            ),
            positions,
        )
        return pullback(forces)[0]

    def dense_neighborhood(
        self,
        site_state: AtomisticInteractionSiteState,
        /,
        *,
        cell: ParticleCell | None = None,
    ) -> ParticleNeighborhoodState:
        pair_capacity = self.plan.sites.capacity * (self.plan.sites.capacity - 1) // 2
        plan = DenseParticleNeighborhoodPlan(pair_capacity, box=cell)
        return plan.prepare(self.site_particles).build(site_state.positions)


class AtomisticNeighborhoodBundleState(StrictModule):
    physical_atoms: ParticleNeighborhoodState
    interaction_sites: ParticleNeighborhoodState
    physical_and_site_aliased: bool = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)


__all__ = [
    "AbstractAtomisticCoordinateMapPlan",
    "AtomisticCoordinateMapPlan",
    "AtomisticInteractionSitePlan",
    "AtomisticInteractionSiteState",
    "AtomisticNeighborhoodBundleState",
    "AtomisticSiteDomain",
    "PreparedAtomisticCoordinateMap",
    "VirtualSiteKind",
    "VirtualSiteRule",
]
