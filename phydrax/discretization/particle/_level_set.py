#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Voxel LEVEL_SET rigid shapes and symmetric physical-area contact quadrature.

The field is the trilinear interpolant, not an exact signed distance. Each
surface direction contributes half its physical area; its conservative force
is the derivative of that discrete energy using the raw field gradient.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...ein import contract
from ...geometry._certificate import DistanceSemantics
from ...geometry._voxel_sampling import PreparedVoxelGeometrySamples
from ...typing import parse
from ._core import ParticleDiscretization
from ._pair_state import ParticlePairKeys, ParticlePairKeySpace
from ._pairwise import ParticlePairGeometry, ParticlePairRelation
from ._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodySetPlan,
)
from ._rigid_contact import RigidContactGeometry
from ._rigid_shape import (
    AbstractPreparedRigidContactShape,
    AbstractRigidContactShapePlan,
    CompliantContactShapeKind,
    RigidContactGeometryBatch,
)
from ._signed_field import posed_signed_field_coordinates, PosedSignedFieldQueryResult


@final
class LevelSetMassPropertyEvidence(StrictModule):
    active_voxels: Array
    boundary_cells: Array
    spacing: Array
    quadrature_volume: Array
    volume_error_bound: Array
    relative_volume_error_bound: Array
    center_of_mass: Array
    inertia_per_mass: Array
    center_of_mass_error_bound: Array
    inertia_per_mass_error_bound: Array
    finite: Array
    positive_definite: Array
    successful: Array
    exact_properties: bool = eqx.field(static=True)


@final
class LevelSetShapeTemplatePlan(StrictModule):
    """Immutable field, quadrature and mass-property preparation.

    The deliberately straight-line host constructor retains validation,
    quadrature bounds, exact-property admission and identity in one transaction:
    no intermediate template can publish mismatched mass or field evidence.
    None of this host preparation runs inside a compiled contact evaluation.
    """

    samples: PreparedVoxelGeometrySamples
    surface_node_ids: Array
    surface_node_positions: Array
    physical_surface_area_weights: Array
    center_of_mass: Array
    inertia_per_mass: Array
    volume: Array
    bounding_radius: Array
    lipschitz_bound: Array
    evaluation_error: Array
    evidence: LevelSetMassPropertyEvidence
    material_id: int = eqx.field(static=True)
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        samples: PreparedVoxelGeometrySamples,
        surface_node_ids: ArrayLike,
        surface_node_positions: ArrayLike,
        physical_surface_area_weights: ArrayLike,
        material_id: int,
        /,
        *,
        volume_tolerance: float = 1.0e-8,
        exact_volume: float | None = None,
        exact_center_of_mass: ArrayLike | None = None,
        exact_inertia_per_mass: ArrayLike | None = None,
        template_id: str | None = None,
    ) -> None:
        if samples.certificate.distance_semantics != DistanceSemantics.LEVEL_SET:
            raise ValueError("Level-set templates require LEVEL_SET sampling semantics.")
        grid = samples.field.grid
        if (
            grid.dimension != 3
            or samples.field.values.ndim != 2
            or any(grid.address_plan.periodic_axes)
        ):
            raise ValueError(
                "Level-set templates require nonperiodic scalar 3-D voxel fields."
            )
        ids = np.asarray(surface_node_ids)
        nodes = np.asarray(surface_node_positions, dtype=np.float64)
        weights = np.asarray(physical_surface_area_weights, dtype=np.float64)
        if ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Surface node IDs must be a nonempty integer vector.")
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError("Surface node IDs must be unique and nonnegative.")
        if nodes.shape != (ids.size, 3) or weights.shape != ids.shape:
            raise ValueError("Surface nodes and physical area weights must match IDs.")
        if (
            np.any(~np.isfinite(nodes))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "Surface nodes must be finite and area weights positive in m²."
            )
        if material_id < 0 or not np.isfinite(volume_tolerance) or volume_tolerance < 0.0:
            raise ValueError("Material ID and volume tolerance must be nonnegative.")
        resolution = grid.address_plan.resolution
        active = np.asarray(grid.voxel_active)
        if np.count_nonzero(active) != resolution**3 or not bool(
            np.asarray(grid.evidence.successful)
        ):
            raise ValueError(
                "Mass quadrature requires the complete volume-domain voxel support."
            )
        centers = np.asarray(grid.voxel_centers())[active]
        values = np.asarray(samples.field.values)[active]
        if np.any(~np.isfinite(values)):
            raise ValueError("Volume-domain field samples must be finite.")
        lower = np.asarray(grid.address_plan.lower)
        upper = np.asarray(grid.address_plan.upper)
        spacing = (upper - lower) / resolution
        voxel_volume = float(np.prod(spacing))
        inside = values < 0.0
        count = np.count_nonzero(inside)
        if count == 0:
            raise ValueError("Level-set template has no resolved interior volume.")
        volume = count * voxel_volume
        center = np.mean(centers[inside], axis=0)
        offsets = centers[inside] - center
        inertia = (
            np.eye(3) * np.mean(np.sum(offsets**2, axis=1)) - offsets.T @ offsets / count
        )
        # Finite differences bound each component of the trilinear gradient.
        integer = np.rint((centers - lower) / spacing - 0.5).astype(np.int64)
        dense = np.empty((resolution,) * 3, dtype=np.float64)
        dense[integer[:, 0], integer[:, 1], integer[:, 2]] = values
        slopes = np.array(
            [np.max(np.abs(np.diff(dense, axis=a))) / spacing[a] for a in range(3)]
        )
        lipschitz = float(np.sqrt(np.sum(slopes**2)))
        field_error = samples.certificate.evaluation_error
        source_error = 0.0 if field_error is None else field_error
        uncertainty = lipschitz * 0.5 * float(np.sqrt(np.sum(spacing**2))) + source_error
        boundary = np.abs(values) <= uncertainty
        boundary_count = np.count_nonzero(boundary)
        volume_error = boundary_count * voxel_volume
        boundary_faces = np.concatenate(
            [np.take(dense, i, axis=a).ravel() for a in range(3) for i in (0, -1)]
        )
        if np.any(boundary_faces <= uncertainty):
            raise ValueError(
                "The resolved interior must be enclosed by a positive field margin."
            )
        provided = (
            exact_volume is not None,
            exact_center_of_mass is not None,
            exact_inertia_per_mass is not None,
        )
        if any(provided) and not all(provided):
            raise ValueError(
                "Exact volume, COM, and inertia per mass must be supplied together."
            )
        exact = all(provided)
        quadrature_center = center.copy()
        quadrature_inertia = inertia.copy()
        if exact:
            if (
                exact_volume is None
                or exact_center_of_mass is None
                or exact_inertia_per_mass is None
            ):
                raise ValueError("Exact mass properties are incomplete.")
            center = np.asarray(exact_center_of_mass, dtype=np.float64)
            inertia = np.asarray(exact_inertia_per_mass, dtype=np.float64)
            supplied_volume = float(exact_volume)
        else:
            supplied_volume = volume
            if volume_error / volume > volume_tolerance:
                raise ValueError(
                    "Voxel volume error bound exceeds volume_tolerance; refine or supply exact properties."
                )
        if (
            center.shape != (3,)
            or inertia.shape != (3, 3)
            or not np.allclose(inertia, inertia.T, rtol=0.0, atol=1e-14)
        ):
            raise ValueError(
                "COM and symmetric inertia per mass have invalid shapes or symmetry."
            )
        if (
            not np.isfinite(supplied_volume)
            or supplied_volume <= 0.0
            or np.any(~np.isfinite(center))
            or np.any(~np.isfinite(inertia))
            or np.min(np.linalg.eigvalsh(inertia)) <= 0.0
        ):
            raise ValueError(
                "Mass properties require positive volume and finite SPD inertia."
            )
        query = samples.field.sample_multilinear_with_gradient(jnp.asarray(nodes))
        if not bool(np.all(np.asarray(query.supported))):
            raise ValueError("Surface nodes require complete interpolation stencils.")
        dtype = samples.field.values.dtype
        self.samples = samples
        self.surface_node_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.surface_node_positions = jnp.asarray(nodes, dtype=dtype)
        self.physical_surface_area_weights = jnp.asarray(weights, dtype=dtype)
        self.center_of_mass = jnp.asarray(center, dtype=dtype)
        self.inertia_per_mass = jnp.asarray(inertia, dtype=dtype)
        self.volume = jnp.asarray(supplied_volume, dtype=dtype)
        # Entire represented negative region plus the uncertain boundary shell is enclosed.
        represented = inside | boundary
        radius = max(
            float(np.max(np.sqrt(np.sum((centers[represented] - center) ** 2, axis=1))))
            + float(np.sqrt(np.sum(spacing**2))),
            float(np.max(np.sqrt(np.sum((nodes - center) ** 2, axis=1)))),
        )
        self.bounding_radius = jnp.asarray(radius, dtype=dtype)
        self.lipschitz_bound = jnp.asarray(lipschitz, dtype=dtype)
        # Source enclosure width also bounds center-interpolation error; absent
        # source evidence is infinity, never an exact-distance certificate.
        self.evaluation_error = samples.evidence.maximum_enclosure_width.astype(dtype)
        lower_volume = volume - volume_error
        com_error = volume_error * radius / lower_volume if lower_volume > 0.0 else np.inf
        inertia_error = (
            4.0 * radius**2 * volume_error / lower_volume
            + 2.0 * radius * com_error
            + com_error**2
            + float(np.sum(spacing**2)) / 6.0
            if lower_volume > 0.0
            else np.inf
        )
        self.evidence = LevelSetMassPropertyEvidence(
            jnp.asarray(count, dtype=jnp.int32),
            jnp.asarray(boundary_count, dtype=jnp.int32),
            jnp.asarray(spacing, dtype=dtype),
            jnp.asarray(volume, dtype=dtype),
            jnp.asarray(volume_error, dtype=dtype),
            jnp.asarray(volume_error / volume, dtype=dtype),
            jnp.asarray(quadrature_center, dtype=dtype),
            jnp.asarray(quadrature_inertia, dtype=dtype),
            jnp.asarray(com_error, dtype=dtype),
            jnp.asarray(inertia_error, dtype=dtype),
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(True),
            exact,
        )
        self.material_id = material_id
        generated = canonical_fingerprint(
            {
                "kind": "level-set-template",
                "sampling": samples.sampling_id,
                "field": array_tree_fingerprint(samples.field.values),
                "properties": array_tree_fingerprint(
                    {
                        "ids": ids,
                        "nodes": nodes,
                        "areas": weights,
                        "volume": supplied_volume,
                        "center": center,
                        "inertia": inertia,
                    }
                ),
                "material": material_id,
                "tolerance": volume_tolerance,
                "exact": exact,
            }
        )
        self.template_id = generated if template_id is None else template_id
        if not self.template_id:
            raise ValueError("template_id must be nonempty.")


@final
class LevelSetRigidShapeSetPlan(AbstractRigidContactShapePlan):
    templates: tuple[LevelSetShapeTemplatePlan, ...]
    owner_template_ids: Array
    owner_material_ids: Array
    fixed_mask: Array
    maximum_contact_routes: int = eqx.field(static=True)
    maximum_candidate_slots: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    shape_kind: CompliantContactShapeKind = eqx.field(static=True)

    def __init__(
        self,
        templates: tuple[LevelSetShapeTemplatePlan, ...],
        owner_template_ids: ArrayLike,
        owner_material_ids: ArrayLike,
        /,
        *,
        fixed_mask: ArrayLike | None = None,
        maximum_contact_routes: int,
        maximum_candidate_slots: int,
        plan_id: str | None = None,
    ) -> None:
        if not templates or maximum_contact_routes < 0 or maximum_candidate_slots < 0:
            raise ValueError("Templates are required and capacities must be nonnegative.")
        ids = np.asarray(owner_template_ids)
        materials = np.asarray(owner_material_ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or materials.shape != ids.shape
            or not np.issubdtype(materials.dtype, np.integer)
        ):
            raise ValueError(
                "Owner template and material IDs must be aligned integer vectors."
            )
        if np.any(ids < 0) or np.any(ids >= len(templates)) or np.any(materials < 0):
            raise ValueError("Owner IDs are out of range.")
        schema = templates[0].samples.field.grid.grid_id
        if any(
            t.samples.field.grid.grid_id != schema
            or t.surface_node_ids.shape != templates[0].surface_node_ids.shape
            or t.samples.field.values.dtype != templates[0].samples.field.values.dtype
            for t in templates
        ):
            raise ValueError(
                "Templates require shared voxel topology, dtype and node capacity."
            )
        if len(set(t.template_id for t in templates)) != len(templates):
            raise ValueError("Identical templates must share one template slot.")
        fixed = (
            np.zeros(ids.shape, dtype=np.bool_)
            if fixed_mask is None
            else np.asarray(fixed_mask, dtype=np.bool_)
        )
        if fixed.shape != ids.shape:
            raise ValueError("fixed_mask must match owner capacity.")
        self.templates = templates
        self.owner_template_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.owner_material_ids = jnp.asarray(materials, dtype=jnp.int32)
        self.fixed_mask = jnp.asarray(fixed)
        self.maximum_contact_routes = maximum_contact_routes
        self.maximum_candidate_slots = maximum_candidate_slots
        self.shape_kind = parse("level_set", CompliantContactShapeKind, "shape_kind")
        generated = canonical_fingerprint(
            {
                "kind": "level-set-shape-set",
                "templates": [t.template_id for t in templates],
                "owners": array_tree_fingerprint(
                    {"templates": ids, "materials": materials, "fixed": fixed}
                ),
                "routes": maximum_contact_routes,
                "scratch": maximum_candidate_slots,
            }
        )
        self.plan_id = generated if plan_id is None else plan_id
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def rigid_body_plan(self, particles: ParticleDiscretization, /) -> RigidBodySetPlan:
        if particles.ambient_dimension != 3 or self.owner_template_ids.shape != (
            particles.capacity,
        ):
            raise ValueError("Level-set bodies require matching 3-D particle capacity.")
        inertia = (
            jnp.stack([t.inertia_per_mass for t in self.templates])[
                self.owner_template_ids
            ]
            * particles.safe_masses[:, None, None]
        )
        return RigidBodySetPlan(
            self.owner_material_ids,
            inertia,
            fixed_mask=self.fixed_mask,
            name="level-set-bodies",
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedLevelSetRigidShapeSet:
        return PreparedLevelSetRigidShapeSet(self, particles)


@final
class PreparedLevelSetRigidShapeSet(AbstractPreparedRigidContactShape):
    plan: LevelSetRigidShapeSetPlan
    bodies: PreparedRigidBodySet
    bounding_radii: Array
    material_ids: Array
    contact_material_ids: Array
    volumes: Array
    shape_kind: CompliantContactShapeKind = eqx.field(static=True)
    shape_id: str = eqx.field(static=True)
    routes_per_pair: int = eqx.field(static=True)

    def __init__(
        self, plan: LevelSetRigidShapeSetPlan, particles: ParticleDiscretization, /
    ) -> None:
        self.plan = plan
        self.bodies = plan.rigid_body_plan(particles).prepare(particles)
        self.bounding_radii = jnp.stack([t.bounding_radius for t in plan.templates])[
            plan.owner_template_ids
        ]
        self.material_ids = plan.owner_material_ids
        self.contact_material_ids = self.material_ids
        self.volumes = jnp.stack([t.volume for t in plan.templates])[
            plan.owner_template_ids
        ]
        self.shape_kind = plan.shape_kind
        self.routes_per_pair = 2 * plan.templates[0].surface_node_ids.size
        self.shape_id = canonical_fingerprint(
            {
                "kind": "prepared-level-set-shapes",
                "plan": plan.plan_id,
                "bodies": self.bodies.prepared_id,
                "measure": "symmetric-half-physical-area",
            }
        )

    def contact_capacity(self, pair_capacity: int, maximum_contact_routes: int, /) -> int:
        if pair_capacity * self.routes_per_pair > self.plan.maximum_candidate_slots:
            raise ValueError(
                "Level-set candidate scratch exceeds maximum_candidate_slots."
            )
        return min(
            super().contact_capacity(pair_capacity, maximum_contact_routes),
            self.plan.maximum_contact_routes,
        )

    def evaluate_signed_field_query(
        self,
        kinematics: RigidBodyKinematics,
        body_indices: Array,
        world_points: Array,
        /,
        *,
        shape_scale: Array | None = None,
    ) -> PosedSignedFieldQueryResult:
        query = evaluate_level_set_point_query(
            self, kinematics, body_indices, world_points, shape_scale=shape_scale
        )
        ids = self.plan.owner_template_ids[body_indices]
        bounds = jnp.stack([t.lipschitz_bound for t in self.plan.templates])[ids]
        errors = jnp.stack([t.evaluation_error for t in self.plan.templates])[ids]
        scale = (
            jnp.ones((self.bodies.capacity,), dtype=world_points.dtype)
            if shape_scale is None
            else shape_scale
        )
        supported = (
            query.stencil_complete
            & jnp.isfinite(errors)
            & jnp.isfinite(query.field_value)
        )
        return PosedSignedFieldQueryResult(
            jnp.where(supported, query.field_value, jnp.nan),
            jnp.where(supported, bounds, jnp.inf),
            jnp.where(supported, errors * scale[body_indices], jnp.inf),
            supported,
            jnp.all(supported),
            self.shape_id,
        )

    def safe_query_points(
        self,
        kinematics: RigidBodyKinematics,
        body_indices: Array,
        /,
        *,
        shape_scale: Array | None = None,
    ) -> Array:
        templates = self.plan.owner_template_ids[body_indices]
        local = jnp.stack(
            [t.surface_node_positions[0] - t.center_of_mass for t in self.plan.templates]
        )[templates]
        scale = (
            jnp.ones((self.bodies.capacity,), dtype=kinematics.position.dtype)
            if shape_scale is None
            else shape_scale
        )
        rotation = quaternion_rotation_matrix(kinematics.orientation)[body_indices]
        return kinematics.position[body_indices] + contract(
            "bij,bj->bi", rotation, local * scale[body_indices, None]
        )

    def pair_geometry(
        self,
        kinematics: RigidBodyKinematics,
        shape_scale: Array,
        pairs: ParticlePairRelation,
        pair_key_space: ParticlePairKeySpace,
        /,
        *,
        maximum_contact_routes: int,
        center_geometry: ParticlePairGeometry | None = None,
        distance_tolerance: float = 1.0e-12,
    ) -> RigidContactGeometryBatch:
        del center_geometry, distance_tolerance
        scratch = pairs.valid.size * self.routes_per_pair
        if scratch > self.plan.maximum_candidate_slots:
            raise ValueError(
                "Level-set pair quadrature exceeds candidate scratch capacity."
            )
        node_count = self.routes_per_pair // 2
        source = jnp.repeat(
            jnp.stack((pairs.left_indices, pairs.right_indices), axis=1).reshape(-1),
            node_count,
        )
        target = jnp.repeat(
            jnp.stack((pairs.right_indices, pairs.left_indices), axis=1).reshape(-1),
            node_count,
        )
        node = jnp.tile(jnp.arange(node_count, dtype=jnp.int32), pairs.valid.size * 2)
        pair_valid = jnp.repeat(pairs.valid, self.routes_per_pair)
        template = self.plan.owner_template_ids[source]
        nodes = jnp.stack(
            [
                t.surface_node_positions - t.center_of_mass[None, :]
                for t in self.plan.templates
            ]
        )[template, node]
        rotation = quaternion_rotation_matrix(kinematics.orientation)
        world = kinematics.position[source] + contract(
            "bij,bj->bi", rotation[source], nodes * shape_scale[source, None]
        )
        bounds = self.bounding_radii * shape_scale
        candidate = pair_valid & (
            jnp.sum((world - kinematics.position[target]) ** 2, axis=1)
            <= bounds[target] ** 2
        )
        # Complete positive boundary cells certify exterior nodes as noncontacts;
        # no field sample or gradient is extrapolated outside the interpolation box.
        target_template = self.plan.owner_template_ids[target]
        target_com = jnp.stack([t.center_of_mass for t in self.plan.templates])[
            target_template
        ]
        local_query = (
            contract("bji,bj->bi", rotation[target], world - kinematics.position[target])
            / shape_scale[target, None]
            + target_com
        )
        grid = self.plan.templates[0].samples.field.grid
        lower = jnp.asarray(grid.address_plan.lower, dtype=world.dtype)
        upper = jnp.asarray(grid.address_plan.upper, dtype=world.dtype)
        half_spacing = 0.5 * (upper - lower) / grid.address_plan.resolution
        candidate = candidate & jnp.all(
            (local_query >= (lower + half_spacing)[None, :])
            & (local_query < (upper - half_spacing)[None, :]),
            axis=1,
        )
        safe_world = jnp.where(
            candidate[:, None],
            world,
            self.safe_query_points(kinematics, target, shape_scale=shape_scale),
        )
        query = evaluate_level_set_point_query(
            self, kinematics, target, safe_world, shape_scale=shape_scale
        )
        valid = candidate & query.valid
        contact_point = world
        source_arm = contact_point - kinematics.position[source]
        target_arm = contact_point - kinematics.position[target]
        relative = (
            kinematics.velocity[source]
            + jnp.cross(kinematics.angular_velocity[source], source_arm)
            - kinematics.velocity[target]
            - jnp.cross(kinematics.angular_velocity[target], target_arm)
        )
        normal_velocity = jnp.sum(relative * query.unit_normal, axis=1)
        area = (
            0.5
            * jnp.stack([t.physical_surface_area_weights for t in self.plan.templates])[
                template, node
            ]
            * shape_scale[source] ** 2
        )
        base = pair_key_space.keys(pairs)
        body_keys = jnp.repeat(base.keys, self.routes_per_pair, axis=0)
        side = (
            self.bodies.particles.particle_ids[source]
            > self.bodies.particles.particle_ids[target]
        ).astype(jnp.int64)
        node_ids = jnp.stack([t.surface_node_ids for t in self.plan.templates])[
            template, node
        ]
        keys = jnp.stack(
            (jnp.full_like(side, 4), body_keys[:, 1], body_keys[:, 2], side, node_ids),
            axis=1,
        )
        # Histories live on penetrating nodes, remapped by stable identity each evaluation.
        key_valid = valid & (query.gap < 0.0)
        count = jnp.sum(key_valid, dtype=jnp.int32)
        active = jnp.sum(valid & (query.gap < 0.0), dtype=jnp.int32)
        capacity = min(scratch, maximum_contact_routes, self.plan.maximum_contact_routes)
        overflow = count > capacity
        successful = base.successful & ~jnp.any(candidate & ~query.valid) & ~overflow
        geometry = RigidContactGeometry(
            query.unit_normal,
            query.gap,
            jnp.maximum(-query.gap, 0.0),
            jnp.zeros_like(query.gap),
            contact_point,
            source_arm,
            target_arm,
            source_arm,
            target_arm,
            relative,
            normal_velocity,
            relative - normal_velocity[:, None] * query.unit_normal,
            kinematics.angular_velocity[source],
            kinematics.angular_velocity[target],
            keys,
            node_ids,
            jnp.zeros_like(node),
            key_valid,
            (candidate & ~query.valid).astype(jnp.int32),
            query.gradient_margin,
            successful,
            "rigid-contact:level-set",
            source,
            target,
            query.raw_gradient,
            area,
            jnp.zeros_like(valid),
            query.gradient_margin,
            jnp.zeros_like(query.gap),
        )
        if capacity < scratch:
            order = jax.lax.stop_gradient(jnp.argsort(~key_valid, stable=True)[:capacity])

            def gather(value: Array) -> Array:
                return value[order] if value.ndim > 0 else value

            geometry = jax.tree.map(gather, geometry)
            keys = keys[order]
            key_valid = key_valid[order]
            source = source[order]
            target = target[order]
        keys = jnp.where(key_valid[:, None], keys, -1)
        return RigidContactGeometryBatch(
            geometry,
            ParticlePairKeys(keys, key_valid, successful, pair_key_space.key_space_id),
            count,
            active,
            capacity,
            overflow,
            successful,
            self.shape_id,
            self.material_ids[source],
            self.material_ids[target],
        )


@final
class LevelSetPointQueryResult(StrictModule):
    field_value: Array
    gap: Array
    raw_gradient: Array
    unit_normal: Array
    contact_point: Array
    mean_curvature: Array
    effective_radius: Array
    stencil_complete: Array
    gradient_margin: Array
    curvature_valid: Array
    valid: Array
    successful: Array
    field_id: str = eqx.field(static=True)


def evaluate_level_set_point_query(
    shape: PreparedLevelSetRigidShapeSet,
    kinematics: RigidBodyKinematics,
    body_indices: Array,
    world_points: Array,
    /,
    *,
    shape_scale: Array | None = None,
) -> LevelSetPointQueryResult:
    """Shared DEM/MPM posed query; no extrapolation or fabricated normal."""
    local, indices, route_scale, pose_supported = posed_signed_field_coordinates(
        kinematics,
        body_indices,
        world_points,
        shape.bodies.particles.active_mask,
        shape_scale=shape_scale,
        rotate=False,
    )
    rotation = quaternion_rotation_matrix(kinematics.orientation)[indices]
    local = contract("bji,bj->bi", rotation, local)
    template_ids = shape.plan.owner_template_ids[indices]
    values = jnp.zeros((world_points.shape[0],), dtype=world_points.dtype)
    gradient = jnp.zeros_like(world_points)
    complete = jnp.zeros(values.shape, dtype=jnp.bool_)
    for index, template in enumerate(shape.plan.templates):
        result = template.samples.field.sample_multilinear_with_gradient(
            local + template.center_of_mass[None, :]
        )
        selected = template_ids == index
        values = jnp.where(selected, result.values, values)
        gradient = jnp.where(selected[:, None], result.gradient, gradient)
        complete = jnp.where(selected, result.stencil_complete, complete)
    raw = contract("bij,bj->bi", rotation, gradient)
    squared = jnp.sum(raw**2, axis=1)
    norm = jnp.sqrt(jnp.where(squared > 0.0, squared, 1.0))
    bound = jnp.stack([t.lipschitz_bound for t in shape.plan.templates])[template_ids]
    margin = (
        jnp.where(squared > 0.0, norm, 0.0) - jnp.sqrt(jnp.finfo(raw.dtype).eps) * bound
    )
    complete = complete & pose_supported
    valid = (
        complete
        & (margin > 0.0)
        & jnp.isfinite(values)
        & jnp.all(jnp.isfinite(raw), axis=1)
        & (route_scale > 0.0)
    )
    normal = jnp.where(valid[:, None], raw / norm[:, None], 0.0)
    value = values * route_scale
    return LevelSetPointQueryResult(
        value,
        value,
        raw,
        normal,
        world_points,
        jnp.zeros_like(value),
        jnp.zeros_like(value),
        complete,
        margin,
        jnp.zeros_like(valid),
        valid,
        jnp.all(valid),
        shape.shape_id,
    )


__all__ = [
    "LevelSetMassPropertyEvidence",
    "LevelSetShapeTemplatePlan",
    "LevelSetRigidShapeSetPlan",
    "PreparedLevelSetRigidShapeSet",
    "LevelSetPointQueryResult",
    "evaluate_level_set_point_query",
]
