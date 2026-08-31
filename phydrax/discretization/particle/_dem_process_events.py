#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ._dem import DEMRuntimeState, PreparedSoftSphereDEMDynamics
from ._particle_internal_mesh import PreparedParticleInternalBatch
from ._particle_internal_state import ParticleInternalBatchState
from ._particle_morphology import ParticleDynamicBodyProperties
from ._rigid_sphere import RigidSphereKinematics


class ReactiveParticleTemplatePlan(StrictModule, NonTrainableState):
    radius: float = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    material_id: int = eqx.field(static=True)
    velocity: Array
    angular_velocity: Array
    internal_energy: Array
    species_amount: Array
    porosity: Array
    internal_surface_area: Array
    outer_scale: float = eqx.field(static=True)
    reaction_front: Array
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        mass: float,
        material_id: int,
        velocity: ArrayLike,
        angular_velocity: ArrayLike,
        internal_energy: ArrayLike,
        species_amount: ArrayLike,
        porosity: ArrayLike,
        internal_surface_area: ArrayLike,
        /,
        *,
        outer_scale: float | None = None,
        reaction_front: ArrayLike = (),
        template_id: str | None = None,
    ):
        radius_ = float(radius)
        mass_ = float(mass)
        material = int(material_id)
        velocity_ = np.asarray(velocity, dtype=float)
        angular_ = np.asarray(angular_velocity, dtype=float)
        energy = np.asarray(internal_energy, dtype=float)
        species = np.asarray(species_amount, dtype=float)
        pore = np.asarray(porosity, dtype=float)
        area = np.asarray(internal_surface_area, dtype=float)
        front = np.asarray(reaction_front, dtype=float)
        scale = radius_ if outer_scale is None else float(outer_scale)
        if (
            not np.isfinite(radius_)
            or radius_ <= 0.0
            or not np.isfinite(mass_)
            or mass_ <= 0.0
            or material < 0
            or velocity_.ndim != 1
            or velocity_.size not in (2, 3)
            or angular_.shape != (1 if velocity_.size == 2 else 3,)
            or energy.ndim != 1
            or energy.size == 0
            or species.ndim != 2
            or species.shape[0] != energy.size
            or pore.shape != energy.shape
            or area.shape != energy.shape
            or front.ndim != 1
            or np.any(~np.isfinite(velocity_))
            or np.any(~np.isfinite(angular_))
            or np.any(~np.isfinite(energy))
            or np.any(~np.isfinite(species))
            or np.any(species < 0.0)
            or np.any(~np.isfinite(pore))
            or np.any((pore < 0.0) | (pore >= 1.0))
            or np.any(~np.isfinite(area))
            or np.any(area < 0.0)
            or not np.isfinite(scale)
            or scale <= 0.0
            or np.any(~np.isfinite(front))
            or np.any((front < 0.0) | (front > 1.0))
        ):
            raise ValueError("Reactive particle template is invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "reactive-particle-template",
                "radius": radius_,
                "mass": mass_,
                "material_id": material,
                "values": array_tree_fingerprint(
                    {
                        "velocity": velocity_,
                        "angular_velocity": angular_,
                        "internal_energy": energy,
                        "species_amount": species,
                        "porosity": pore,
                        "internal_surface_area": area,
                        "outer_scale": np.asarray(scale),
                        "reaction_front": front,
                    }
                ),
            }
        )
        self.radius = radius_
        self.mass = mass_
        self.material_id = material
        self.velocity = jnp.asarray(velocity_)
        self.angular_velocity = jnp.asarray(angular_)
        self.internal_energy = jnp.asarray(energy)
        self.species_amount = jnp.asarray(species)
        self.porosity = jnp.asarray(pore)
        self.internal_surface_area = jnp.asarray(area)
        self.outer_scale = scale
        self.reaction_front = jnp.asarray(front)
        self.template_id = generated if template_id is None else str(template_id)
        if not self.template_id:
            raise ValueError("template_id must be nonempty.")


class ReactiveParticleTemplateDistributionPlan(StrictModule, NonTrainableState):
    templates: tuple[ReactiveParticleTemplatePlan, ...]
    probabilities: Array
    distribution_id: str = eqx.field(static=True)

    def __init__(self, templates, probabilities: ArrayLike, /):
        values = tuple(templates)
        probability = np.asarray(probabilities, dtype=float)
        if not values or any(
            not isinstance(value, ReactiveParticleTemplatePlan) for value in values
        ):
            raise TypeError("templates must contain ReactiveParticleTemplatePlan values.")
        if (
            probability.shape != (len(values),)
            or np.any(~np.isfinite(probability))
            or np.any(probability < 0.0)
            or not np.isclose(np.sum(probability), 1.0)
        ):
            raise ValueError("Template probabilities must be a probability vector.")
        first = values[0]
        if any(
            value.velocity.shape != first.velocity.shape
            or value.internal_energy.shape != first.internal_energy.shape
            or value.species_amount.shape != first.species_amount.shape
            or value.reaction_front.shape != first.reaction_front.shape
            for value in values[1:]
        ):
            raise ValueError("Distributed templates must share static schemas.")
        self.templates = values
        self.probabilities = jnp.asarray(probability)
        self.distribution_id = canonical_fingerprint(
            {
                "kind": "reactive-particle-template-distribution",
                "templates": [value.template_id for value in values],
                "probabilities": array_tree_fingerprint(probability),
            }
        )

    def sample(self, key: Key[Array, ""], count: int, /) -> Array:
        return jr.choice(
            key,
            len(self.templates),
            shape=(int(count),),
            p=self.probabilities,
        ).astype(jnp.int32)


class ParticleInsertionPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    requested_count: int = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    all_inside: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        requested_count: int,
        /,
        *,
        maximum_attempts: int = 32,
        all_inside: bool = True,
    ):
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        count = int(requested_count)
        attempts = int(maximum_attempts)
        if (
            lower_.shape != upper_.shape
            or lower_.ndim != 1
            or lower_.size not in (2, 3)
            or np.any(~np.isfinite(lower_))
            or np.any(~np.isfinite(upper_))
            or np.any(upper_ <= lower_)
            or count <= 0
            or attempts <= 0
        ):
            raise ValueError("Particle insertion region/count controls are invalid.")
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.requested_count = count
        self.maximum_attempts = attempts
        self.all_inside = bool(all_inside)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-insertion-plan",
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
                "requested_count": count,
                "maximum_attempts": attempts,
                "all_inside": bool(all_inside),
            }
        )


class ParticleInsertionResult(StrictModule):
    candidate_dem_state: DEMRuntimeState
    accepted_dem_state: DEMRuntimeState
    candidate_internal_state: ParticleInternalBatchState
    accepted_internal_state: ParticleInternalBatchState
    owner_slots: Array
    template_indices: Array
    inserted: Array
    attempt_indices: Array
    capacity_available: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def insert_reactive_particles(
    plan: ParticleInsertionPlan,
    distribution: ReactiveParticleTemplateDistributionPlan,
    dynamics: PreparedSoftSphereDEMDynamics,
    dem_state: DEMRuntimeState,
    internal_batch: PreparedParticleInternalBatch,
    internal_state: ParticleInternalBatchState,
    molar_masses: ArrayLike,
    key: Key[Array, ""],
    time: Array,
    /,
    *,
    args=None,
) -> ParticleInsertionResult:
    if not isinstance(plan, ParticleInsertionPlan):
        raise TypeError("plan must be a ParticleInsertionPlan.")
    if not isinstance(distribution, ReactiveParticleTemplateDistributionPlan):
        raise TypeError("distribution must be a template distribution.")
    if plan.lower.shape[0] != dynamics.bodies.ambient_dimension:
        raise ValueError("Insertion region dimension does not match DEM.")
    if internal_batch.particles.prepared_id != dynamics.bodies.particles.prepared_id:
        raise ValueError("Insertion internal and DEM populations do not match.")
    molar = jnp.asarray(molar_masses, dtype=internal_state.species_amount.dtype)
    if molar.shape != (internal_batch.species_count,):
        raise ValueError("molar_masses must have internal species shape.")
    inactive = dynamics.bodies.particles.active_mask & ~dem_state.body_properties.active
    slots = jnp.nonzero(
        inactive,
        size=plan.requested_count,
        fill_value=-1,
    )[0].astype(jnp.int32)
    capacity_available = jnp.sum(inactive, dtype=jnp.int32) >= plan.requested_count
    template_key, position_key = jr.split(key)
    template_indices = distribution.sample(template_key, plan.requested_count)
    candidates = jr.uniform(
        position_key,
        shape=(
            plan.requested_count,
            plan.maximum_attempts,
            dynamics.bodies.ambient_dimension,
        ),
        minval=plan.lower,
        maxval=plan.upper,
        dtype=dem_state.kinematics.position.dtype,
    )
    position = dem_state.kinematics.position
    velocity = dem_state.kinematics.velocity
    angular = dem_state.kinematics.angular_velocity
    properties = dem_state.body_properties
    active = properties.active
    radii = properties.radii
    masses = properties.masses
    inverse_masses = properties.inverse_masses
    inertias = properties.inertias
    inverse_inertias = properties.inverse_inertias
    energy = internal_state.internal_energy
    species = internal_state.species_amount
    pore = internal_state.porosity
    internal_area = internal_state.internal_surface_area
    outer_scale = internal_state.outer_scale
    front = internal_state.reaction_front
    internal_active = internal_state.active
    inserted = jnp.zeros((plan.requested_count,), dtype=bool)
    selected_attempt = -jnp.ones((plan.requested_count,), dtype=jnp.int32)
    for insertion_index in range(plan.requested_count):
        slot = jnp.maximum(slots[insertion_index], 0)
        template_index = template_indices[insertion_index]
        template_values = tuple(distribution.templates)
        radius_values = jnp.asarray([value.radius for value in template_values])
        mass_values = jnp.asarray([value.mass for value in template_values])
        radius = radius_values[template_index]
        mass = mass_values[template_index]
        candidate = candidates[insertion_index]
        lower_ok = (
            candidate - radius >= plan.lower
            if plan.all_inside
            else candidate >= plan.lower
        )
        upper_ok = (
            candidate + radius <= plan.upper
            if plan.all_inside
            else candidate <= plan.upper
        )
        region_ok = jnp.all(lower_ok & upper_ok, axis=-1)
        displacement = candidate[:, None, :] - position[None, :, :]
        distance = jnp.linalg.norm(displacement, axis=-1)
        overlap_free = jnp.all(
            ~active[None, :] | (distance >= radius + radii[None, :]),
            axis=-1,
        )
        admissible = region_ok & overlap_free & (slots[insertion_index] >= 0)
        has_candidate = jnp.any(admissible)
        attempt = jnp.argmax(admissible.astype(jnp.int32))
        chosen = candidate[attempt]
        use = capacity_available & has_candidate
        template_velocity = jnp.stack(tuple(value.velocity for value in template_values))[
            template_index
        ]
        template_angular = jnp.stack(
            tuple(value.angular_velocity for value in template_values)
        )[template_index]
        position = position.at[slot].set(jnp.where(use, chosen, position[slot]))
        velocity = velocity.at[slot].set(
            jnp.where(use, template_velocity, velocity[slot])
        )
        angular = angular.at[slot].set(jnp.where(use, template_angular, angular[slot]))
        active = active.at[slot].set(use | active[slot])
        radii = radii.at[slot].set(jnp.where(use, radius, radii[slot]))
        masses = masses.at[slot].set(jnp.where(use, mass, masses[slot]))
        inverse_masses = inverse_masses.at[slot].set(
            jnp.where(use, 1.0 / mass, inverse_masses[slot])
        )
        inertia = (
            (0.5 if dynamics.bodies.ambient_dimension == 2 else 0.4) * mass * radius**2
        )
        inertias = inertias.at[slot].set(jnp.where(use, inertia, inertias[slot]))
        inverse_inertias = inverse_inertias.at[slot].set(
            jnp.where(use, 1.0 / inertia, inverse_inertias[slot])
        )
        local_matches = internal_batch.owner_indices == slot
        local_exists = jnp.any(local_matches)
        local = jnp.argmax(local_matches.astype(jnp.int32))
        template_energy = jnp.stack(
            tuple(value.internal_energy for value in template_values)
        )[template_index]
        template_species = jnp.stack(
            tuple(value.species_amount for value in template_values)
        )[template_index]
        template_pore = jnp.stack(tuple(value.porosity for value in template_values))[
            template_index
        ]
        template_area = jnp.stack(
            tuple(value.internal_surface_area for value in template_values)
        )[template_index]
        template_scale = jnp.asarray([value.outer_scale for value in template_values])[
            template_index
        ]
        template_front = jnp.stack(
            tuple(value.reaction_front for value in template_values)
        )[template_index]
        use_internal = use & local_exists
        energy = energy.at[local].set(
            jnp.where(use_internal, template_energy, energy[local])
        )
        species = species.at[local].set(
            jnp.where(use_internal, template_species, species[local])
        )
        pore = pore.at[local].set(jnp.where(use_internal, template_pore, pore[local]))
        internal_area = internal_area.at[local].set(
            jnp.where(use_internal, template_area, internal_area[local])
        )
        outer_scale = outer_scale.at[local].set(
            jnp.where(use_internal, template_scale, outer_scale[local])
        )
        front = front.at[local].set(jnp.where(use_internal, template_front, front[local]))
        internal_active = internal_active.at[local].set(
            use_internal | internal_active[local]
        )
        inserted = inserted.at[insertion_index].set(use & local_exists)
        selected_attempt = selected_attempt.at[insertion_index].set(
            jnp.where(use & local_exists, attempt, -1)
        )
    body_properties = ParticleDynamicBodyProperties(
        masses,
        inverse_masses,
        radii,
        inertias,
        inverse_inertias,
        active,
    )
    raw_dem = DEMRuntimeState(
        RigidSphereKinematics(position, velocity, angular),
        body_properties,
        dem_state.particle_history,
        dem_state.boundary_histories,
        dem_state.neighborhood_cache,
        dem_state.loads,
        dem_state.energy,
    )
    body_update = dynamics.apply_body_properties(
        time,
        raw_dem,
        body_properties,
        jnp.asarray(True),
        args=args,
    )
    candidate_internal = ParticleInternalBatchState(
        energy,
        species,
        pore,
        internal_area,
        outer_scale,
        front,
        internal_active,
        internal_state.batch_id,
    )
    template_mass = jnp.stack(
        tuple(jnp.sum(value.species_amount * molar) for value in distribution.templates)
    )
    mass_consistent = jnp.all(
        jnp.abs(
            template_mass[template_indices]
            - jnp.asarray([value.mass for value in distribution.templates])[
                template_indices
            ]
        )
        <= 1.0e-10 * jnp.maximum(template_mass[template_indices], 1.0)
    )
    successful = (
        capacity_available & jnp.all(inserted) & mass_consistent & body_update.successful
    )
    accepted_dem = tree_where(successful, body_update.candidate_state, dem_state)
    accepted_internal = tree_where(successful, candidate_internal, internal_state)
    return ParticleInsertionResult(
        body_update.candidate_state,
        accepted_dem,
        candidate_internal,
        accepted_internal,
        slots,
        template_indices,
        inserted,
        selected_attempt,
        capacity_available,
        successful,
        plan.plan_id,
    )


class ParticleResidenceState(StrictModule):
    inside: Array
    residence_time: Array
    entry_count: Array
    exit_count: Array


class ParticleRegionPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    region_id: str = eqx.field(static=True)

    def __init__(
        self, lower: ArrayLike, upper: ArrayLike, /, *, region_id: str | None = None
    ):
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        if lower_.shape != upper_.shape or lower_.ndim != 1 or np.any(upper_ <= lower_):
            raise ValueError("Particle region bounds are invalid.")
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        generated = canonical_fingerprint(
            {
                "kind": "particle-region",
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
            }
        )
        self.region_id = generated if region_id is None else str(region_id)

    def contains(self, positions: ArrayLike, /) -> Array:
        value = jnp.asarray(positions)
        return jnp.all((value >= self.lower) & (value <= self.upper), axis=-1)

    def initialize_residence(self, positions: ArrayLike, /) -> ParticleResidenceState:
        inside = self.contains(positions)
        return ParticleResidenceState(
            inside,
            jnp.zeros(inside.shape, dtype=jnp.asarray(positions).dtype),
            inside.astype(jnp.int32),
            jnp.zeros(inside.shape, dtype=jnp.int32),
        )

    def update_residence(
        self,
        state: ParticleResidenceState,
        positions: ArrayLike,
        step_size: Array,
        /,
    ) -> ParticleResidenceState:
        inside = self.contains(positions)
        entered = inside & ~state.inside
        exited = ~inside & state.inside
        return ParticleResidenceState(
            inside,
            state.residence_time + jnp.where(inside, step_size, 0.0),
            state.entry_count + entered.astype(jnp.int32),
            state.exit_count + exited.astype(jnp.int32),
        )


class MassFlowSurfacePlan(StrictModule, NonTrainableState):
    point: Array
    normal: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, point: ArrayLike, normal: ArrayLike, /):
        point_ = np.asarray(point, dtype=float)
        normal_ = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal_)
        if (
            point_.shape != normal_.shape
            or point_.ndim != 1
            or not np.isfinite(norm)
            or norm <= 0.0
        ):
            raise ValueError("Mass-flow surface point/normal are invalid.")
        self.point = jnp.asarray(point_)
        self.normal = jnp.asarray(normal_ / norm)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mass-flow-surface",
                "point": point_.tolist(),
                "normal": (normal_ / norm).tolist(),
            }
        )

    def crossed_mass(
        self,
        previous_position: ArrayLike,
        current_position: ArrayLike,
        masses: ArrayLike,
        active: ArrayLike,
        /,
    ) -> Array:
        previous = jnp.sum(
            (jnp.asarray(previous_position) - self.point) * self.normal, axis=-1
        )
        current = jnp.sum(
            (jnp.asarray(current_position) - self.point) * self.normal, axis=-1
        )
        crossed = (previous < 0.0) & (current >= 0.0) & jnp.asarray(active, dtype=bool)
        return jnp.sum(jnp.where(crossed, jnp.asarray(masses), 0.0))


class ParticleRemovalResult(StrictModule):
    candidate_dem_state: DEMRuntimeState
    accepted_dem_state: DEMRuntimeState
    candidate_internal_state: ParticleInternalBatchState
    accepted_internal_state: ParticleInternalBatchState
    removed: Array
    released_mass: Array
    released_internal_energy: Array
    released_species_amount: Array
    successful: Array
    region_id: str = eqx.field(static=True)


def remove_particles_in_region(
    region: ParticleRegionPlan,
    dynamics: PreparedSoftSphereDEMDynamics,
    dem_state: DEMRuntimeState,
    internal_batch: PreparedParticleInternalBatch,
    internal_state: ParticleInternalBatchState,
    time: Array,
    /,
    *,
    remove_inside: bool = True,
    args=None,
) -> ParticleRemovalResult:
    if not isinstance(region, ParticleRegionPlan):
        raise TypeError("region must be a ParticleRegionPlan.")
    inside = region.contains(dem_state.kinematics.position)
    removed = (
        inside if bool(remove_inside) else ~inside
    ) & dem_state.body_properties.active
    coverage = jnp.zeros_like(removed, dtype=jnp.int32)
    coverage = coverage.at[internal_batch.owner_indices].add(1)
    local_removed = removed[internal_batch.owner_indices]
    released_mass = jnp.sum(jnp.where(removed, dem_state.body_properties.masses, 0.0))
    released_energy = jnp.sum(
        jnp.where(
            local_removed[:, None],
            internal_state.internal_energy,
            0.0,
        )
    )
    released_species = jnp.sum(
        jnp.where(
            local_removed[:, None, None],
            internal_state.species_amount,
            0.0,
        ),
        axis=(0, 1),
    )
    candidate_internal = ParticleInternalBatchState(
        jnp.where(local_removed[:, None], 0.0, internal_state.internal_energy),
        jnp.where(local_removed[:, None, None], 0.0, internal_state.species_amount),
        jnp.where(local_removed[:, None], 0.0, internal_state.porosity),
        jnp.where(
            local_removed[:, None],
            0.0,
            internal_state.internal_surface_area,
        ),
        jnp.where(local_removed, 1.0, internal_state.outer_scale),
        jnp.where(local_removed[:, None], 0.0, internal_state.reaction_front),
        internal_state.active & ~local_removed,
        internal_state.batch_id,
    )
    properties = ParticleDynamicBodyProperties(
        jnp.where(removed, 0.0, dem_state.body_properties.masses),
        jnp.where(removed, 0.0, dem_state.body_properties.inverse_masses),
        jnp.where(removed, 0.0, dem_state.body_properties.radii),
        jnp.where(removed, 1.0, dem_state.body_properties.inertias),
        jnp.where(removed, 0.0, dem_state.body_properties.inverse_inertias),
        dem_state.body_properties.active & ~removed,
    )
    raw_state = DEMRuntimeState(
        RigidSphereKinematics(
            dem_state.kinematics.position,
            jnp.where(removed[:, None], 0.0, dem_state.kinematics.velocity),
            jnp.where(
                removed[:, None],
                0.0,
                dem_state.kinematics.angular_velocity,
            ),
        ),
        properties,
        dem_state.particle_history,
        dem_state.boundary_histories,
        dem_state.neighborhood_cache,
        dem_state.loads,
        dem_state.energy,
    )
    body_update = dynamics.apply_body_properties(
        time,
        raw_state,
        properties,
        jnp.any(removed),
        args=args,
    )
    successful = body_update.successful & jnp.all(~removed | (coverage == 1))
    return ParticleRemovalResult(
        body_update.candidate_state,
        tree_where(successful, body_update.candidate_state, dem_state),
        candidate_internal,
        tree_where(successful, candidate_internal, internal_state),
        removed,
        released_mass,
        released_energy,
        released_species,
        successful,
        region.region_id,
    )


__all__ = [
    "MassFlowSurfacePlan",
    "ParticleInsertionPlan",
    "ParticleInsertionResult",
    "ParticleRegionPlan",
    "ParticleResidenceState",
    "ReactiveParticleTemplateDistributionPlan",
    "ReactiveParticleTemplatePlan",
    "ParticleRemovalResult",
    "remove_particles_in_region",
    "insert_reactive_particles",
]
