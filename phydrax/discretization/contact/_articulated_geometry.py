#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Body-attached analytic collision geometry and lifted articulated paths.

The first native articulated hard-contact profile admits positive-radius
spheres and capsules attached to the bodies of one prepared fixed-base reduced
articulation, plus static planes. Geometry positions are exactly native forward
kinematics evaluated on the binding lifted path ``q(s) = q0 + s h v1``.
"""

from __future__ import annotations

from enum import IntEnum
from math import pi
from numbers import Integral
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import ConvertibleToArray
from ..particle._reduced_articulation import PreparedReducedArticulation
from ..particle._rigid_body import quaternion_rotation_matrix
from ..particle._rigid_joints import RigidJointKind
from ._guarantee import ContactGuaranteeLevel
from ._implicit_geometry import PlaneContactGeometry
from ._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
    CollisionSurfacePlan,
    ContactPairPolicy,
)
from ._trajectory import (
    _interval_times,
    _revision,
    AbstractContactTrajectory,
    ContactPathEnclosure,
    ContactPathMidpointBound,
)


class ArticulatedPrimitiveKind(IntEnum):
    """Admitted body-attached analytic primitive kinds."""

    SPHERE = 0
    CAPSULE = 1


# Scene feature IDs are ``3 * primitive_id + role`` with role 0 (sphere center
# or capsule first endpoint), 1 (capsule second endpoint), and 2 (capsule axis).
_FEATURE_STRIDE = 3
_MAXIMUM_PRIMITIVE_ID = (np.iinfo(np.int64).max - 2) // _FEATURE_STRIDE


def _integer_ids(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim != 1 or (values.size and not np.issubdtype(values.dtype, np.integer)):
        raise TypeError(f"{name} must be one integer vector.")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.unique(values).size != values.size:
        raise ValueError(f"{name} must be unique and nonnegative.")
    return values


def _real_matrix(
    value: ConvertibleToArray, shape: tuple[int, ...], name: str, /
) -> np.ndarray:
    values = np.asarray(value)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real numbers.")
    values = values.astype(np.float64, copy=False)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    return values


def _feature_labels(
    value: ConvertibleToArray | None, count: int, name: str, /
) -> np.ndarray:
    if value is None:
        return np.zeros((count,), dtype=np.int64)
    values = np.asarray(value)
    if values.shape != (count,) or (
        count and not np.issubdtype(values.dtype, np.integer)
    ):
        raise TypeError(f"{name} must contain one integer per declared item.")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return values


class _HostPrimitives(NamedTuple):
    """Validated host primitive declaration, in declared or canonical order."""

    ids: np.ndarray
    body_ids: np.ndarray
    kinds: np.ndarray
    first: np.ndarray
    second: np.ndarray
    radii: np.ndarray
    materials: np.ndarray


class _HostBodies(NamedTuple):
    """Articulation body-ID map used to attach primitives."""

    index_by_body: dict[int, int]
    root_index: int
    root_body_id: int


class _HostPlanes(NamedTuple):
    """Validated static plane declaration."""

    ids: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    materials: np.ndarray


class _HostLayout(NamedTuple):
    """Scene vertex/edge layout of canonically ordered primitives."""

    vertex_primitives: np.ndarray
    vertex_roles: np.ndarray
    vertex_body_indices: np.ndarray
    edges: np.ndarray
    edge_ids: np.ndarray


class _HostTopology(NamedTuple):
    """Static articulation edge topology in topological order."""

    parents: tuple[int, ...]
    children: tuple[int, ...]
    kinds: tuple[int, ...]
    dofs: tuple[int, ...]


class _PathAncestry(NamedTuple):
    """Per-body ancestor rate sums along one path interval (body capacity)."""

    prismatic: Array
    hinge: Array
    moment: Array
    rotating: Array
    coordinate_error: Array


def _box_distance_upper(
    first_center: Array,
    first_radius: Array,
    second_center: Array,
    second_radius: Array,
    /,
) -> Array:
    """Upper bound of the distance between points of two axis-aligned boxes."""
    reach = jnp.abs(first_center - second_center) + first_radius + second_radius
    return jnp.sqrt(jnp.sum(reach * reach, axis=-1)) * (
        1.0 + 4.0 * jnp.finfo(reach.dtype).eps
    )


def _declared_primitives(
    primitive_ids: ConvertibleToArray,
    primitive_body_ids: ConvertibleToArray,
    primitive_kinds: tuple[ArticulatedPrimitiveKind, ...],
    local_first: ConvertibleToArray,
    local_second: ConvertibleToArray,
    radii: ConvertibleToArray,
    primitive_material_ids: ConvertibleToArray | None,
    /,
) -> _HostPrimitives:
    identifiers = _integer_ids(primitive_ids, "primitive_ids")
    count = identifiers.size
    if count == 0:
        raise ValueError("Articulated collision geometry requires a primitive.")
    if np.any(identifiers > _MAXIMUM_PRIMITIVE_ID):
        raise ValueError("primitive_ids exceed the injective scene feature-ID range.")
    bodies = np.asarray(primitive_body_ids)
    if bodies.shape != (count,) or not np.issubdtype(bodies.dtype, np.integer):
        raise TypeError("primitive_body_ids must contain one integer per primitive.")
    kinds_raw = tuple(primitive_kinds)
    if len(kinds_raw) != count or not all(
        isinstance(kind, ArticulatedPrimitiveKind) for kind in kinds_raw
    ):
        raise TypeError(
            "primitive_kinds must contain one ArticulatedPrimitiveKind per primitive."
        )
    first = _real_matrix(local_first, (count, 3), "local_first")
    second = _real_matrix(local_second, (count, 3), "local_second")
    radius = _real_matrix(radii, (count,), "radii")
    if np.any(radius <= 0.0):
        raise ValueError("Primitive radii must be positive.")
    kinds = np.asarray([int(kind) for kind in kinds_raw], dtype=np.int32)
    sphere = kinds == int(ArticulatedPrimitiveKind.SPHERE)
    if np.any(sphere & np.any(first != second, axis=-1)):
        raise ValueError("Sphere primitives require local_second equal to local_first.")
    lengths = np.linalg.norm(second - first, axis=-1)
    if np.any(~sphere & ~(lengths > 0.0)):
        raise ValueError("Capsule primitives require distinct axis endpoints.")
    materials = _feature_labels(primitive_material_ids, count, "primitive_material_ids")
    return _HostPrimitives(
        identifiers,
        bodies.astype(np.int64, copy=False),
        kinds,
        first,
        second,
        radius,
        materials,
    )


def _articulation_bodies(
    articulation: PreparedReducedArticulation, primitive_bodies: np.ndarray, /
) -> _HostBodies:
    body_ids = np.asarray(articulation.body_ids, dtype=np.int64)
    body_indices = np.asarray(articulation.body_indices, dtype=np.int64)
    index_by_body = {
        int(body): int(index) for body, index in zip(body_ids, body_indices, strict=True)
    }
    missing = sorted({int(body) for body in primitive_bodies} - set(index_by_body))
    if missing:
        raise ValueError(f"Primitive body IDs are not articulation bodies: {missing}.")
    root_index = int(articulation.root_index)
    root_body_id = int(body_ids[body_indices == root_index][0])
    return _HostBodies(index_by_body, root_index, root_body_id)


def _declared_planes(
    plane_ids: ConvertibleToArray,
    plane_normals: ConvertibleToArray | None,
    plane_offsets: ConvertibleToArray | None,
    plane_material_ids: ConvertibleToArray | None,
    /,
) -> _HostPlanes:
    identifiers = _integer_ids(plane_ids, "plane_ids")
    count = identifiers.size
    normals = _real_matrix(
        np.empty((0, 3)) if plane_normals is None else plane_normals,
        (count, 3),
        "plane_normals",
    )
    offsets = _real_matrix(
        np.empty((0,)) if plane_offsets is None else plane_offsets,
        (count,),
        "plane_offsets",
    )
    if np.any(~(np.linalg.norm(normals, axis=-1) > 0.0)):
        raise ValueError("Plane normals must be nonzero.")
    materials = _feature_labels(plane_material_ids, count, "plane_material_ids")
    return _HostPlanes(identifiers, normals, offsets, materials)


def _excluded_body_pairs(
    excluded_body_pairs: ConvertibleToArray | None,
    index_by_body: dict[int, int],
    /,
) -> tuple[tuple[int, int], ...]:
    raw = (
        np.empty((0, 2), dtype=np.int64)
        if excluded_body_pairs is None
        else np.asarray(excluded_body_pairs)
    )
    if raw.size == 0:
        raw = np.empty((0, 2), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("excluded_body_pairs must be an integer (pairs, 2) array.")
    pairs = np.sort(raw.astype(np.int64, copy=False), axis=1)
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Excluded body pairs must join distinct bodies.")
    unknown = sorted({int(body) for body in pairs.reshape(-1)} - set(index_by_body))
    if unknown:
        raise ValueError(f"Excluded body IDs are not articulation bodies: {unknown}.")
    return tuple(sorted({(int(left), int(right)) for left, right in pairs}))


def _canonical_primitives(declared: _HostPrimitives, /) -> _HostPrimitives:
    """Order primitives by scientific primitive ID (stable)."""
    order = np.argsort(declared.ids, kind="stable")
    return _HostPrimitives(*(values[order] for values in declared))


def _scene_layout(primitives: _HostPrimitives, body_index: np.ndarray, /) -> _HostLayout:
    """One vertex per sphere; two vertices and one axis edge per capsule."""
    vertex_primitives: list[int] = []
    vertex_roles: list[int] = []
    edges: list[tuple[int, int]] = []
    edge_ids: list[int] = []
    for primitive, (identifier, kind) in enumerate(
        zip(primitives.ids, primitives.kinds, strict=True)
    ):
        vertex_primitives.append(primitive)
        vertex_roles.append(0)
        if kind == int(ArticulatedPrimitiveKind.CAPSULE):
            vertex_primitives.append(primitive)
            vertex_roles.append(1)
            edges.append((len(vertex_primitives) - 2, len(vertex_primitives) - 1))
            edge_ids.append(_FEATURE_STRIDE * int(identifier) + 2)
    vertex_primitive_array = np.asarray(vertex_primitives, dtype=np.int64)
    return _HostLayout(
        vertex_primitive_array,
        np.asarray(vertex_roles, dtype=np.int64),
        body_index[vertex_primitive_array],
        np.asarray(edges, dtype=np.int32).reshape((-1, 2)),
        np.asarray(edge_ids, dtype=np.int64),
    )


def _excluded_vertex_pairs(
    vertex_body_ids: np.ndarray, excluded: tuple[tuple[int, int], ...], /
) -> np.ndarray:
    """Vertex pairs on the same body or on a declared excluded body pair."""
    excluded_set = set(excluded)
    count = vertex_body_ids.size
    pairs = [
        (left, right)
        for left in range(count)
        for right in range(left + 1, count)
        if vertex_body_ids[left] == vertex_body_ids[right]
        or (
            min(int(vertex_body_ids[left]), int(vertex_body_ids[right])),
            max(int(vertex_body_ids[left]), int(vertex_body_ids[right])),
        )
        in excluded_set
    ]
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def _surface_plan(
    articulation: PreparedReducedArticulation,
    primitives: _HostPrimitives,
    layout: _HostLayout,
    excluded: tuple[tuple[int, int], ...],
    participant: int,
    root_index: int,
    /,
) -> tuple[CollisionSurfacePlan, str]:
    """Collision surface plan and feature provenance of the primitive scene."""
    vertices = layout.vertex_primitives
    vertex_count = vertices.size
    vertex_body_ids = primitives.body_ids[vertices]
    provenance = canonical_fingerprint(
        {
            "kind": "articulated-collision-features",
            "articulation": articulation.prepared_id,
            "primitive_ids": primitives.ids.tolist(),
            "primitive_kinds": primitives.kinds.tolist(),
            "primitive_body_ids": primitives.body_ids.tolist(),
        }
    )
    surface_plan = CollisionSurfacePlan(
        _FEATURE_STRIDE * primitives.ids[vertices] + layout.vertex_roles,
        ambient_dimension=3,
        edges=layout.edges,
        edge_ids=layout.edge_ids,
        codimensional_mask=np.ones((vertex_count,), dtype=np.bool_),
        pair_policy=ContactPairPolicy(
            vertex_count,
            excluded_vertex_pairs=_excluded_vertex_pairs(vertex_body_ids, excluded),
        ),
        participant_ids=participant,
        body_ids=vertex_body_ids,
        material_ids=primitives.materials[vertices],
        static_mask=layout.vertex_body_indices == root_index,
        physical_radius=primitives.radii[vertices],
        feature_provenance_id=provenance,
        allow_isolated_vertices=True,
    )
    return surface_plan, provenance


def _static_planes(
    planes: _HostPlanes, participant: int, root_body_id: int, provenance: str, /
) -> tuple[PlaneContactGeometry, ...]:
    """Static world planes as analytic features attached to the fixed root."""
    return tuple(
        PlaneContactGeometry(
            planes.normals[index],
            planes.offsets[index],
            feature_policy=CollisionFeaturePolicy(
                np.asarray((planes.ids[index],), dtype=np.int64),
                np.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=np.int32),
                participant_ids=participant,
                body_ids=root_body_id,
                material_ids=int(planes.materials[index]),
                patch_ids=0,
                static_mask=True,
                provenance_id=f"{provenance}:plane:{int(planes.ids[index])}",
            ),
        )
        for index in range(planes.ids.size)
    )


def _articulation_topology(articulation: PreparedReducedArticulation, /) -> _HostTopology:
    """Static joint edges (topological order) and their configuration DOFs."""
    parents = np.asarray(articulation.parent_indices, dtype=np.int64)
    children = np.asarray(articulation.child_indices, dtype=np.int64)
    joint_kinds = np.asarray(articulation.joint_kinds, dtype=np.int64)
    offsets = np.asarray(articulation.configuration_offsets, dtype=np.int64)
    return _HostTopology(
        tuple(int(value) for value in parents),
        tuple(int(value) for value in children),
        tuple(int(value) for value in joint_kinds),
        tuple(
            -1 if int(kind) == int(RigidJointKind.FIXED) else int(offsets[edge])
            for edge, kind in enumerate(joint_kinds)
        ),
    )


class ArticulatedCollisionGeometry(StrictModule, NonTrainableState):
    """Spheres/capsules attached to articulation bodies and static world planes.

    Primitives are canonically ordered by scientific primitive ID. The contact
    scene layout is: one vertex per sphere center; two vertices (first, second
    endpoint in body-attached order) and one edge per capsule. Physical radii are
    feature physical radii. Primitives attached to the fixed articulation root
    are static. Same-body pairs and declared excluded body pairs are excluded
    through vertex-pair exclusions; static-static pairs are excluded by the
    shared static rule. Numerical local points are dynamic leaves; IDs, kinds,
    body attachment and articulation topology form the static identity.
    """

    surface_plan: CollisionSurfacePlan
    planes: tuple[PlaneContactGeometry, ...]
    primitive_body_indices: Array
    local_first: Array
    local_second: Array
    primitive_ids: tuple[int, ...] = eqx.field(static=True)
    primitive_body_ids: tuple[int, ...] = eqx.field(static=True)
    primitive_kinds: tuple[ArticulatedPrimitiveKind, ...] = eqx.field(static=True)
    plane_ids: tuple[int, ...] = eqx.field(static=True)
    vertex_primitives: tuple[int, ...] = eqx.field(static=True)
    vertex_roles: tuple[int, ...] = eqx.field(static=True)
    vertex_bodies: tuple[int, ...] = eqx.field(static=True)
    excluded_body_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    edge_parents: tuple[int, ...] = eqx.field(static=True)
    edge_children: tuple[int, ...] = eqx.field(static=True)
    edge_kinds: tuple[int, ...] = eqx.field(static=True)
    edge_dofs: tuple[int, ...] = eqx.field(static=True)
    participant_id: int = eqx.field(static=True)
    articulation_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        articulation: PreparedReducedArticulation,
        /,
        *,
        primitive_ids: ConvertibleToArray,
        primitive_body_ids: ConvertibleToArray,
        primitive_kinds: tuple[ArticulatedPrimitiveKind, ...],
        local_first: ConvertibleToArray,
        local_second: ConvertibleToArray,
        radii: ConvertibleToArray,
        plane_ids: ConvertibleToArray = (),
        plane_normals: ConvertibleToArray | None = None,
        plane_offsets: ConvertibleToArray | None = None,
        excluded_body_pairs: ConvertibleToArray | None = None,
        primitive_material_ids: ConvertibleToArray | None = None,
        plane_material_ids: ConvertibleToArray | None = None,
        participant_id: int = 0,
    ) -> None:
        if not isinstance(articulation, PreparedReducedArticulation):
            raise TypeError("articulation must be a PreparedReducedArticulation.")
        declared = _declared_primitives(
            primitive_ids,
            primitive_body_ids,
            primitive_kinds,
            local_first,
            local_second,
            radii,
            primitive_material_ids,
        )
        bodies = _articulation_bodies(articulation, declared.body_ids)
        planes_declared = _declared_planes(
            plane_ids, plane_normals, plane_offsets, plane_material_ids
        )
        excluded = _excluded_body_pairs(excluded_body_pairs, bodies.index_by_body)
        if isinstance(participant_id, bool) or not isinstance(participant_id, Integral):
            raise TypeError("participant_id must be an integer.")
        participant = int(participant_id)
        if participant < 0:
            raise ValueError("participant_id must be nonnegative.")
        primitives = _canonical_primitives(declared)
        body_index = np.asarray(
            [bodies.index_by_body[int(body)] for body in primitives.body_ids],
            dtype=np.int64,
        )
        layout = _scene_layout(primitives, body_index)
        surface_plan, provenance = _surface_plan(
            articulation,
            primitives,
            layout,
            excluded,
            participant,
            bodies.root_index,
        )
        planes = _static_planes(
            planes_declared, participant, bodies.root_body_id, provenance
        )
        topology = _articulation_topology(articulation)
        if articulation.nq != articulation.nv:
            raise ValueError("Lifted articulated paths require scalar joint coordinates.")
        self.surface_plan = surface_plan
        self.planes = planes
        self.primitive_body_indices = jnp.asarray(body_index, dtype=jnp.int32)
        self.local_first = jnp.asarray(primitives.first)
        self.local_second = jnp.asarray(primitives.second)
        self.primitive_ids = tuple(int(value) for value in primitives.ids)
        self.primitive_body_ids = tuple(int(value) for value in primitives.body_ids)
        self.primitive_kinds = tuple(
            ArticulatedPrimitiveKind(int(value)) for value in primitives.kinds
        )
        self.plane_ids = tuple(int(value) for value in planes_declared.ids)
        self.vertex_primitives = tuple(int(value) for value in layout.vertex_primitives)
        self.vertex_roles = tuple(int(value) for value in layout.vertex_roles)
        self.vertex_bodies = tuple(int(value) for value in layout.vertex_body_indices)
        self.excluded_body_pairs = excluded
        self.edge_parents = topology.parents
        self.edge_children = topology.children
        self.edge_kinds = topology.kinds
        self.edge_dofs = topology.dofs
        self.participant_id = participant
        self.articulation_id = articulation.prepared_id
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "articulated-collision-geometry",
                "articulation": articulation.prepared_id,
                "surface": surface_plan.topology_id,
                "plane_ids": planes_declared.ids.tolist(),
                "plane_geometry": [plane.geometry_id for plane in planes],
                "excluded_body_pairs": [list(pair) for pair in excluded],
            }
        )

    @property
    def primitive_count(self) -> int:
        return len(self.primitive_ids)

    @property
    def vertex_count(self) -> int:
        return len(self.vertex_primitives)

    @property
    def plane_count(self) -> int:
        return len(self.planes)

    @property
    def vertex_body_indices(self) -> Array:
        """Articulation body-capacity index of every scene vertex."""
        return jnp.asarray(self.vertex_bodies, dtype=jnp.int32)

    @property
    def vertex_local_positions(self) -> Array:
        primitives = jnp.asarray(self.vertex_primitives, dtype=jnp.int32)
        second = jnp.asarray(self.vertex_roles, dtype=jnp.bool_)[:, None]
        return jnp.where(
            second, self.local_second[primitives], self.local_first[primitives]
        )

    def _check(self, articulation: PreparedReducedArticulation, /) -> None:
        if (
            not isinstance(articulation, PreparedReducedArticulation)
            or articulation.prepared_id != self.articulation_id
        ):
            raise ValueError("Articulation does not match the collision geometry.")

    def vertex_positions(
        self, articulation: PreparedReducedArticulation, configuration: ArrayLike, /
    ) -> Array:
        """World scene-vertex positions by native FK at a lifted configuration."""
        self._check(articulation)
        transforms = articulation.body_transforms(configuration)
        bodies = self.vertex_body_indices
        local = self.vertex_local_positions.astype(transforms.dtype)
        rotation = transforms[bodies, :3, :3]
        return transforms[bodies, :3, 3] + (rotation @ local[..., None])[..., 0]

    def point_velocities(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        generalized_velocity: ArrayLike,
        body_indices: ArrayLike,
        points: ArrayLike,
        /,
    ) -> Array:
        """Velocities of world points rigidly attached to articulation bodies.

        A negative body index denotes static world geometry with zero velocity.
        """
        self._check(articulation)
        configuration_ = jnp.asarray(configuration)
        poses = articulation.body_transforms(configuration_)
        body_velocity = articulation.body_velocity_action(
            configuration_, generalized_velocity
        )
        indices = jnp.asarray(body_indices, dtype=jnp.int32)
        world_points = jnp.asarray(points, dtype=body_velocity.dtype)
        if world_points.shape != indices.shape + (3,):
            raise ValueError("points must have shape body_indices.shape + (3,).")
        attached = indices >= 0
        safe = jnp.where(attached, indices, 0)
        linear = body_velocity[safe, :3]
        angular = body_velocity[safe, 3:]
        arm = world_points - poses[safe, :3, 3]
        velocity = linear + jnp.cross(angular, arm)
        return jnp.where(attached[..., None], velocity, 0.0)

    def point_force_pullback(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        body_indices: ArrayLike,
        points: ArrayLike,
        forces: ArrayLike,
        /,
    ) -> Array:
        """Generalized effort of world forces applied at body-attached points.

        This is the exact transpose of :meth:`point_velocities` at fixed
        configuration, so impulse power is preserved.
        """
        self._check(articulation)
        configuration_ = jnp.asarray(configuration)
        template = jnp.zeros((articulation.nv,), dtype=configuration_.dtype)

        def action(velocity: Array, /) -> Array:
            return self.point_velocities(
                articulation, configuration_, velocity, body_indices, points
            )

        force = jnp.asarray(forces, dtype=configuration_.dtype)
        return jax.linear_transpose(action, template)(force)[0]

    def vertex_velocities(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        generalized_velocity: ArrayLike,
        /,
    ) -> Array:
        positions = self.vertex_positions(articulation, configuration)
        return self.point_velocities(
            articulation,
            configuration,
            generalized_velocity,
            self.vertex_body_indices,
            positions,
        )

    def vertex_force_pullback(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        forces: ArrayLike,
        /,
    ) -> Array:
        positions = self.vertex_positions(articulation, configuration)
        return self.point_force_pullback(
            articulation,
            configuration,
            self.vertex_body_indices,
            positions,
            forces,
        )


class ArticulatedLiftedPath(StrictModule):
    """The binding segment path ``q(s) = q0 + s h v`` in lifted scalar coordinates.

    No principal-angle wrapping is applied: endpoints modulo ``2π`` do not define
    the path, and hinge winding is owned by the plant history.
    """

    articulation: PreparedReducedArticulation
    start_configuration: Array
    velocity: Array
    step_size: Array
    path_revision: Array

    def __init__(
        self,
        articulation: PreparedReducedArticulation,
        start_configuration: ArrayLike,
        velocity: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        revision: ArrayLike = 0,
    ) -> None:
        if not isinstance(articulation, PreparedReducedArticulation):
            raise TypeError("articulation must be a PreparedReducedArticulation.")
        start = jnp.asarray(
            start_configuration, dtype=articulation.reference_position.dtype
        )
        rate = jnp.asarray(velocity, dtype=start.dtype)
        step = jnp.asarray(step_size, dtype=start.dtype)
        if articulation.nq != articulation.nv:
            raise ValueError("Lifted articulated paths require scalar joint coordinates.")
        if start.shape != (articulation.nq,) or rate.shape != (articulation.nv,):
            raise ValueError("Lifted path configuration/velocity shapes are invalid.")
        if step.shape != ():
            raise ValueError("step_size must be scalar.")
        self.articulation = articulation
        self.start_configuration = start
        self.velocity = rate
        self.step_size = step
        self.path_revision = _revision(revision)

    @property
    def increment(self) -> Array:
        return self.step_size * self.velocity

    def configuration(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=self.start_configuration.dtype)
        return self.start_configuration + time_ * self.increment

    def end_configuration(self) -> Array:
        return self.configuration(1.0)


def _interval_product(
    left_center: Array, left_radius: Array, right_center: Array, right_radius: Array, /
) -> tuple[Array, Array]:
    """Midpoint-radius enclosure of an interval matrix product.

    The radius includes a modeled floating-point allowance; it is not a
    directed-rounding certificate.
    """
    center = left_center @ right_center
    absolute_left = jnp.abs(left_center)
    absolute_right = jnp.abs(right_center)
    radius = (
        absolute_left @ right_radius
        + left_radius @ absolute_right
        + left_radius @ right_radius
    )
    gamma = 16.0 * jnp.finfo(center.dtype).eps
    return center, radius + gamma * (absolute_left @ absolute_right + radius)


def _sine_cosine_enclosure(
    lower: Array, upper: Array, /
) -> tuple[Array, Array, Array, Array]:
    """Enclose ``sin`` and ``cos`` over ``[lower, upper]`` with interior extrema.

    Returns midpoint/radius pairs. Intervals of width at least ``2π`` give the
    full range ``[-1, 1]``. Argument-reduction and evaluation errors are covered
    by a modeled allowance proportional to the angle magnitude.
    """
    eps = jnp.finfo(lower.dtype).eps
    allowance = 16.0 * eps * (1.0 + jnp.maximum(jnp.abs(lower), jnp.abs(upper)))
    first = lower - allowance
    last = upper + allowance
    period = 2.0 * pi

    def contains(phase: float, /) -> Array:
        turns = jnp.ceil((first - phase) / period)
        return phase + period * turns <= last + allowance

    full = last - first >= period
    sine_first, sine_last = jnp.sin(first), jnp.sin(last)
    cosine_first, cosine_last = jnp.cos(first), jnp.cos(last)
    sine_low = jnp.where(contains(-0.5 * pi), -1.0, jnp.minimum(sine_first, sine_last))
    sine_high = jnp.where(contains(0.5 * pi), 1.0, jnp.maximum(sine_first, sine_last))
    cosine_low = jnp.where(contains(pi), -1.0, jnp.minimum(cosine_first, cosine_last))
    cosine_high = jnp.where(contains(0.0), 1.0, jnp.maximum(cosine_first, cosine_last))
    sine_low = jnp.clip(jnp.where(full, -1.0, sine_low) - 4.0 * eps, -1.0, 1.0)
    sine_high = jnp.clip(jnp.where(full, 1.0, sine_high) + 4.0 * eps, -1.0, 1.0)
    cosine_low = jnp.clip(jnp.where(full, -1.0, cosine_low) - 4.0 * eps, -1.0, 1.0)
    cosine_high = jnp.clip(jnp.where(full, 1.0, cosine_high) + 4.0 * eps, -1.0, 1.0)
    return (
        0.5 * (sine_low + sine_high),
        0.5 * (sine_high - sine_low),
        0.5 * (cosine_low + cosine_high),
        0.5 * (cosine_high - cosine_low),
    )


def _skew(vector: Array, /) -> Array:
    x, y, z = vector[0], vector[1], vector[2]
    zero = jnp.zeros_like(x)
    return jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape((3, 3))


class ArticulatedPathFrames(StrictModule):
    """Interval enclosures of articulated body frames over a path interval.

    Rotations and positions are midpoint/radius interval enclosures of every
    body transform for all lifted configurations on the interval; rotation
    entries are intersected with ``[-1, 1]``. ``anchor_center``/``anchor_radius``
    enclose each joint anchor in world coordinates.
    """

    rotation_center: Array
    rotation_radius: Array
    position_center: Array
    position_radius: Array
    anchor_center: Array
    anchor_radius: Array


class ArticulatedContactTrajectory(AbstractContactTrajectory):
    """Scene-vertex trajectory of articulated primitives along a lifted path.

    ``evaluate(s)`` is exact native FK of ``q(s)``. Interval bounds intersect
    (1) an interval forward-kinematics box built from sine/cosine enclosures of
    every ancestor hinge (including interior extrema and full turns) and
    prismatic offsets, and (2) a midpoint ball whose radius bounds the
    Jacobian-column speed ``sum |h v_j| |a_j| dist(x, anchor_j)`` over the
    interval. Both contract with the interval. For a vertex whose ancestor hinges
    all have exactly zero increment, every ancestor rotation is constant and the
    world path is exactly affine in ``s``; its bounds are additionally
    intersected with the endpoint hull, which keeps cancellation between
    prismatic ancestors (zero net angular velocity of moving hinges is not
    sufficient and keeps the generic bounds). The declared guarantee is
    ``ENCLOSURE_CONSERVATIVE``: analytic enclosures with a modeled, not
    directed-rounding, floating-point allowance.
    """

    geometry: ArticulatedCollisionGeometry
    path: ArticulatedLiftedPath
    _trajectory_id: str = eqx.field(static=True)

    def __init__(
        self, geometry: ArticulatedCollisionGeometry, path: ArticulatedLiftedPath, /
    ) -> None:
        if not isinstance(geometry, ArticulatedCollisionGeometry):
            raise TypeError("geometry must be ArticulatedCollisionGeometry.")
        if not isinstance(path, ArticulatedLiftedPath):
            raise TypeError("path must be ArticulatedLiftedPath.")
        geometry._check(path.articulation)
        self.geometry = geometry
        self.path = path
        self._trajectory_id = canonical_fingerprint(
            {
                "kind": "articulated-lifted-contact-trajectory",
                "geometry": geometry.geometry_id,
                "articulation": path.articulation.prepared_id,
                "dtype": str(path.start_configuration.dtype),
            }
        )

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def guarantee_level(self) -> ContactGuaranteeLevel:
        return ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE

    @property
    def revision(self) -> Array:
        return self.path.path_revision

    def evaluate(self, time: ArrayLike, /) -> Array:
        return self.geometry.vertex_positions(
            self.path.articulation, self.path.configuration(time)
        )

    def frames(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> ArticulatedPathFrames:
        """Enclose every body frame and joint anchor over ``[lower, upper]``."""
        articulation = self.path.articulation
        geometry = self.geometry
        dtype = self.path.start_configuration.dtype
        lower, upper, _ = _interval_times(lower_time, upper_time, dtype)
        eps = jnp.finfo(dtype).eps
        first = self.path.configuration(lower)
        last = self.path.configuration(upper)
        coordinate_allowance = (
            8.0
            * eps
            * (jnp.abs(self.path.start_configuration) + jnp.abs(self.path.increment))
        )
        coordinate_low = jnp.minimum(first, last) - coordinate_allowance
        coordinate_high = jnp.maximum(first, last) + coordinate_allowance
        rotation_center = quaternion_rotation_matrix(articulation.reference_orientation)
        rotation_radius = jnp.zeros_like(rotation_center)
        position_center = articulation.reference_position
        position_radius = jnp.zeros_like(position_center)
        identity = jnp.eye(3, dtype=dtype)
        zero_matrix = jnp.zeros((3, 3), dtype=dtype)
        zero_vector = jnp.zeros((3,), dtype=dtype)
        anchors_center: list[Array] = []
        anchors_radius: list[Array] = []
        for edge, (parent, child, kind, dof) in enumerate(
            zip(
                geometry.edge_parents,
                geometry.edge_children,
                geometry.edge_kinds,
                geometry.edge_dofs,
                strict=True,
            )
        ):
            parent_rotation = (rotation_center[parent], rotation_radius[parent])
            parent_position = (position_center[parent], position_radius[parent])
            anchor_local = articulation.parent_anchors[edge]
            anchor, anchor_spread = _interval_product(
                parent_rotation[0], parent_rotation[1], anchor_local, zero_vector
            )
            anchors_center.append(parent_position[0] + anchor)
            anchors_radius.append(parent_position[1] + anchor_spread)
            reference_rotation = articulation.parent_reference_rotation[edge]
            reference_translation = articulation.parent_reference_translation[edge]
            axis = articulation.parent_axes[edge]
            if kind == int(RigidJointKind.HINGE):
                axis_norm = jnp.sqrt(jnp.sum(axis * axis))
                skew = _skew(axis / axis_norm)
                skew_squared = skew @ skew
                sine, sine_radius, cosine, cosine_radius = _sine_cosine_enclosure(
                    axis_norm * coordinate_low[dof], axis_norm * coordinate_high[dof]
                )
                joint_center = identity + sine * skew + (1.0 - cosine) * skew_squared
                joint_radius = sine_radius * jnp.abs(skew) + cosine_radius * jnp.abs(
                    skew_squared
                )
                joint_radius = joint_radius + 16.0 * eps * (
                    jnp.abs(identity) + jnp.abs(skew) + jnp.abs(skew_squared)
                )
                relative_rotation = _interval_product(
                    joint_center, joint_radius, reference_rotation, zero_matrix
                )
                arm, arm_radius = _interval_product(
                    joint_center,
                    joint_radius,
                    reference_translation - anchor_local,
                    zero_vector,
                )
                relative_translation = (anchor_local + arm, arm_radius)
            elif kind == int(RigidJointKind.PRISMATIC):
                offset_center = 0.5 * (coordinate_low[dof] + coordinate_high[dof])
                offset_radius = 0.5 * (coordinate_high[dof] - coordinate_low[dof])
                relative_rotation = (reference_rotation, zero_matrix)
                relative_translation = (
                    reference_translation + axis * offset_center,
                    jnp.abs(axis) * offset_radius
                    + 8.0 * eps * jnp.abs(reference_translation + axis * offset_center),
                )
            else:
                relative_rotation = (reference_rotation, zero_matrix)
                relative_translation = (reference_translation, zero_vector)
            child_rotation, child_rotation_radius = _interval_product(
                parent_rotation[0],
                parent_rotation[1],
                relative_rotation[0],
                relative_rotation[1],
            )
            child_offset, child_offset_radius = _interval_product(
                parent_rotation[0],
                parent_rotation[1],
                relative_translation[0],
                relative_translation[1],
            )
            rotation_low = jnp.maximum(child_rotation - child_rotation_radius, -1.0)
            rotation_high = jnp.minimum(child_rotation + child_rotation_radius, 1.0)
            rotation_center = rotation_center.at[child].set(
                0.5 * (rotation_low + rotation_high)
            )
            rotation_radius = rotation_radius.at[child].set(
                0.5 * (rotation_high - rotation_low)
            )
            position_center = position_center.at[child].set(
                parent_position[0] + child_offset
            )
            position_radius = position_radius.at[child].set(
                parent_position[1]
                + child_offset_radius
                + 8.0 * eps * jnp.abs(parent_position[0] + child_offset)
            )
        anchor_center = (
            jnp.stack(anchors_center)
            if anchors_center
            else jnp.zeros((0, 3), dtype=dtype)
        )
        anchor_radius = (
            jnp.stack(anchors_radius)
            if anchors_radius
            else jnp.zeros((0, 3), dtype=dtype)
        )
        return ArticulatedPathFrames(
            rotation_center,
            rotation_radius,
            position_center,
            position_radius,
            anchor_center,
            anchor_radius,
        )

    def _ancestry(self, frames: ArticulatedPathFrames, /) -> _PathAncestry:
        """Accumulate per-body ancestry quantities by one root-to-leaf pass.

        Edges are in topological order, so each child reads its parent's
        already-accumulated values. Work and workspace are ``O(edges + body
        capacity)``; no vertex-by-edge array is formed.

        For body ``b`` with origin ``o_b``, ancestor hinges ``j`` with anchors
        ``a_j`` and rates ``w_j = |h v_j| |axis_j|``, and ancestor prismatic rates
        ``u_j``, the triangle inequality gives for any point ``x`` rigid on ``b``
        ``|dx/ds| <= P_b + M_b + H_b |x - o_b|`` with ``P_b = sum u_j``,
        ``H_b = sum w_j`` and ``M_b = sum w_j |o_b - a_j|``. Distances are upper
        bounds between interval boxes, propagated by
        ``|o_c - a_j| <= |o_c - o_p| + |o_p - a_j|``.

        ``rotating[b]`` is true when any ancestor hinge has a nonzero
        increment (zero net angular velocity is not sufficient), and
        ``coordinate_error[b]`` bounds the world effect of rounding the lifted
        prismatic coordinates of the ancestors.
        """
        geometry = self.geometry
        articulation = self.path.articulation
        dtype = self.path.start_configuration.dtype
        eps = jnp.finfo(dtype).eps
        capacity = frames.position_center.shape[0]
        prismatic = jnp.zeros((capacity,), dtype=dtype)
        hinge = jnp.zeros((capacity,), dtype=dtype)
        moment = jnp.zeros((capacity,), dtype=dtype)
        coordinate_error = jnp.zeros((capacity,), dtype=dtype)
        rotating = jnp.zeros((capacity,), dtype=jnp.bool_)
        for edge, (parent, child, kind, dof) in enumerate(
            zip(
                geometry.edge_parents,
                geometry.edge_children,
                geometry.edge_kinds,
                geometry.edge_dofs,
                strict=True,
            )
        ):
            hop = _box_distance_upper(
                frames.position_center[child],
                frames.position_radius[child],
                frames.position_center[parent],
                frames.position_radius[parent],
            )
            child_prismatic = prismatic[parent]
            child_hinge = hinge[parent]
            child_moment = moment[parent] + hinge[parent] * hop
            child_error = coordinate_error[parent]
            child_rotating = rotating[parent]
            if dof >= 0:
                increment = self.path.increment[dof]
                axis = articulation.parent_axes[edge]
                axis_norm = jnp.sqrt(jnp.sum(axis * axis))
                rate = jnp.abs(increment) * axis_norm
                if kind == int(RigidJointKind.HINGE):
                    arm = _box_distance_upper(
                        frames.position_center[child],
                        frames.position_radius[child],
                        frames.anchor_center[edge],
                        frames.anchor_radius[edge],
                    )
                    child_hinge = child_hinge + rate
                    child_moment = child_moment + rate * arm
                    child_rotating = child_rotating | (increment != 0.0)
                else:
                    span = jnp.abs(self.path.start_configuration[dof]) + jnp.abs(
                        increment
                    )
                    child_prismatic = child_prismatic + rate
                    child_error = child_error + 8.0 * eps * axis_norm * span
            prismatic = prismatic.at[child].set(child_prismatic)
            hinge = hinge.at[child].set(child_hinge)
            moment = moment.at[child].set(child_moment)
            coordinate_error = coordinate_error.at[child].set(child_error)
            rotating = rotating.at[child].set(child_rotating)
        return _PathAncestry(prismatic, hinge, moment, rotating, coordinate_error)

    def interval_bounds(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> tuple[ContactPathEnclosure, ContactPathMidpointBound]:
        geometry = self.geometry
        dtype = self.path.start_configuration.dtype
        lower, upper, interval_valid = _interval_times(lower_time, upper_time, dtype)
        eps = jnp.finfo(dtype).eps
        frames = self.frames(lower, upper)
        bodies = geometry.vertex_body_indices
        local = geometry.vertex_local_positions.astype(dtype)
        local_absolute = jnp.abs(local)
        rotation = frames.rotation_center[bodies]
        rotation_radius = frames.rotation_radius[bodies]
        box_center = (
            frames.position_center[bodies] + (rotation @ local[..., None])[..., 0]
        )
        box_radius = (
            frames.position_radius[bodies]
            + (rotation_radius @ local_absolute[..., None])[..., 0]
            + 16.0
            * eps
            * (
                (jnp.abs(rotation) @ local_absolute[..., None])[..., 0]
                + jnp.abs(frames.position_center[bodies])
            )
        )
        midpoint = 0.5 * (lower + upper)
        positions = self.evaluate(midpoint)
        edge_count = len(geometry.edge_kinds)
        scale = (
            jnp.max(jnp.abs(positions), axis=-1)
            + jnp.sqrt(jnp.sum(local * local, axis=-1))
            + jnp.max(jnp.abs(frames.position_center))
        )
        evaluation_allowance = 64.0 * (edge_count + 1) * eps * scale
        ancestry = self._ancestry(frames)
        origin_distance = _box_distance_upper(
            box_center,
            box_radius,
            frames.position_center[bodies],
            frames.position_radius[bodies],
        )
        speed = (
            ancestry.prismatic[bodies]
            + ancestry.moment[bodies]
            + ancestry.hinge[bodies] * origin_distance
        )
        displacement = (
            0.5 * (upper - lower) * speed * (1.0 + 16.0 * eps)
            + 2.0 * evaluation_allowance
        )
        enclosure_lower = jnp.maximum(
            positions - displacement[:, None],
            box_center - box_radius - evaluation_allowance[:, None],
        )
        enclosure_upper = jnp.minimum(
            positions + displacement[:, None],
            box_center + box_radius + evaluation_allowance[:, None],
        )
        # Constant-rotation ancestry: when every ancestor hinge increment is
        # exactly zero, all ancestor rotations are fixed and the vertex path is
        # exactly affine in s, so it lies in the hull of its endpoint positions.
        # Summed independent joint intervals cannot express cancellation between
        # prismatic ancestors moving in opposite directions.
        affine = ~ancestry.rotating[bodies]
        coordinate_error = ancestry.coordinate_error[bodies]
        first = self.evaluate(lower)
        last = self.evaluate(upper)
        affine_scale = (
            jnp.maximum(
                jnp.maximum(
                    jnp.max(jnp.abs(first), axis=-1), jnp.max(jnp.abs(last), axis=-1)
                ),
                jnp.max(jnp.abs(positions), axis=-1),
            )
            + jnp.sqrt(jnp.sum(local * local, axis=-1))
            + jnp.max(jnp.abs(frames.position_center))
        )
        affine_allowance = 64.0 * (edge_count + 1) * eps * affine_scale + coordinate_error
        affine_displacement = (
            0.5 * jnp.sqrt(jnp.sum((last - first) ** 2, axis=-1)) * (1.0 + 16.0 * eps)
            + 2.0 * affine_allowance
        )
        displacement = jnp.where(
            affine, jnp.minimum(displacement, affine_displacement), displacement
        )
        enclosure_lower = jnp.where(
            affine[:, None],
            jnp.maximum(
                enclosure_lower, jnp.minimum(first, last) - affine_allowance[:, None]
            ),
            enclosure_lower,
        )
        enclosure_upper = jnp.where(
            affine[:, None],
            jnp.minimum(
                enclosure_upper, jnp.maximum(first, last) + affine_allowance[:, None]
            ),
            enclosure_upper,
        )
        finite = (
            jnp.all(jnp.isfinite(enclosure_lower))
            & jnp.all(jnp.isfinite(enclosure_upper))
            & jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(displacement))
        )
        successful = finite & interval_valid & jnp.all(enclosure_lower <= enclosure_upper)
        level = jnp.asarray(int(self.guarantee_level), dtype=jnp.int32)
        return (
            ContactPathEnclosure(
                enclosure_lower,
                enclosure_upper,
                lower,
                upper,
                level,
                self.revision,
                finite,
                successful,
                self.trajectory_id,
            ),
            ContactPathMidpointBound(
                midpoint,
                positions,
                displacement,
                lower,
                upper,
                level,
                self.revision,
                finite,
                successful,
                self.trajectory_id,
            ),
        )

    def bounds(
        self, lower_time: ArrayLike, upper_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        enclosure, _ = self.interval_bounds(lower_time, upper_time)
        return enclosure.lower, enclosure.upper

    def enclosure(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> ContactPathEnclosure:
        return self.interval_bounds(lower_time, upper_time)[0]

    def midpoint_bound(
        self, lower_time: ArrayLike = 0.0, upper_time: ArrayLike = 1.0, /
    ) -> ContactPathMidpointBound:
        return self.interval_bounds(lower_time, upper_time)[1]


__all__ = [
    "ArticulatedCollisionGeometry",
    "ArticulatedContactTrajectory",
    "ArticulatedLiftedPath",
    "ArticulatedPathFrames",
    "ArticulatedPrimitiveKind",
]
