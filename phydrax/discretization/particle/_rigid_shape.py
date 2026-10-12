#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._pair_state import ParticlePairKeys, ParticlePairKeySpace
from ._pairwise import ParticlePairGeometry, ParticlePairRelation
from ._rigid_body import PreparedRigidBodySet, RigidBodyKinematics, RigidBodySetPlan
from ._rigid_contact import RigidContactGeometry
from ._signed_field import PosedSignedFieldQueryResult


type CompliantContactShapeKind = Literal[
    "sphere", "clump", "superquadric", "convex", "level_set"
]


@final
class RigidContactGeometryBatch(StrictModule):
    geometry: RigidContactGeometry
    keys: ParticlePairKeys
    candidate_count: Array
    active_count: Array
    capacity: int = eqx.field(static=True)
    overflow: Array
    successful: Array
    shape_id: str = eqx.field(static=True)
    left_material_ids: Array
    right_material_ids: Array


class AbstractRigidContactShapePlan(StrictModule, NonTrainableState):
    shape_kind: eqx.AbstractVar[CompliantContactShapeKind]

    @abc.abstractmethod
    def rigid_body_plan(self, particles: ParticleDiscretization, /) -> RigidBodySetPlan:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> AbstractPreparedRigidContactShape:
        raise NotImplementedError


class AbstractPreparedRigidContactShape(StrictModule, NonTrainableState):
    bodies: eqx.AbstractVar[PreparedRigidBodySet]
    bounding_radii: eqx.AbstractVar[Array]
    material_ids: eqx.AbstractVar[Array]
    contact_material_ids: eqx.AbstractVar[Array]
    volumes: eqx.AbstractVar[Array]
    shape_kind: eqx.AbstractVar[CompliantContactShapeKind]
    shape_id: eqx.AbstractVar[str]
    routes_per_pair: eqx.AbstractVar[int]

    def contact_capacity(self, pair_capacity: int, maximum_contact_routes: int, /) -> int:
        if pair_capacity < 0 or maximum_contact_routes < 0:
            raise ValueError("Contact capacities must be nonnegative.")
        return min(pair_capacity * self.routes_per_pair, maximum_contact_routes)

    @abc.abstractmethod
    def evaluate_signed_field_query(
        self,
        kinematics: RigidBodyKinematics,
        body_indices: Array,
        world_points: Array,
        /,
        *,
        shape_scale: Array | None = None,
    ) -> PosedSignedFieldQueryResult:
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError


def bounded_contact_geometry(
    geometry: RigidContactGeometry,
    keys: ParticlePairKeys,
    maximum_contact_routes: int,
    shape_id: str,
    /,
    *,
    left_material_ids: Array,
    right_material_ids: Array,
) -> RigidContactGeometryBatch:
    """Bound all possible law routes, including finite-gap cohesive contacts."""
    if maximum_contact_routes < 0:
        raise ValueError("maximum_contact_routes must be nonnegative.")
    active = geometry.valid & (geometry.overlap > 0.0)
    candidate_count = jnp.sum(keys.valid, dtype=jnp.int32)
    active_count = jnp.sum(active, dtype=jnp.int32)
    capacity = min(geometry.valid.shape[0], maximum_contact_routes)
    overflow = candidate_count > capacity
    if capacity < geometry.valid.shape[0]:
        order = jax.lax.stop_gradient(jnp.argsort(~keys.valid, stable=True)[:capacity])

        def gather(value: Array) -> Array:
            return value[order] if value.ndim > 0 else value

        geometry = jax.tree.map(gather, geometry)
        keys = ParticlePairKeys(
            keys.keys[order], keys.valid[order], keys.successful, keys.key_space_id
        )
        left_material_ids = left_material_ids[order]
        right_material_ids = right_material_ids[order]
    successful = geometry.successful & keys.successful & ~overflow
    return RigidContactGeometryBatch(
        geometry,
        keys,
        candidate_count,
        active_count,
        capacity,
        overflow,
        successful,
        shape_id,
        left_material_ids,
        right_material_ids,
    )


__all__ = [
    "AbstractPreparedRigidContactShape",
    "AbstractRigidContactShapePlan",
    "CompliantContactShapeKind",
    "RigidContactGeometryBatch",
]
