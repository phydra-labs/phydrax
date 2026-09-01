#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    FunctionLinearOperator,
)
from .._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from .._cell_mesh import CellMesh
from ._precision import ContactPrecisionPolicy


def _canonical_edges(faces: np.ndarray, /) -> np.ndarray:
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    local = faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2))
    return np.unique(np.sort(local, axis=1), axis=0).astype(np.int32, copy=False)


def _sorted_edges(
    edges: np.ndarray, vertex_ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    canonical = np.sort(edges, axis=1)
    keys = np.sort(vertex_ids[canonical], axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    sorted_edges = canonical[order]
    sorted_keys = keys[order]
    if sorted_keys.shape[0] > 1 and np.any(
        np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    ):
        raise ValueError("Collision surface edges must be unique.")
    return sorted_edges, np.arange(sorted_edges.shape[0], dtype=np.int64)


def _sorted_faces(
    faces: np.ndarray, vertex_ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0:
        return faces.astype(np.int32, copy=False), np.empty((0,), dtype=np.int64)
    keys = np.sort(vertex_ids[faces], axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_faces = faces[order].astype(np.int32, copy=False)
    sorted_keys = keys[order]
    if sorted_keys.shape[0] > 1 and np.any(
        np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    ):
        raise ValueError("Collision surface faces must be unique.")
    return sorted_faces, np.arange(sorted_faces.shape[0], dtype=np.int64)


class ContactPairPolicy(StrictModule, NonTrainableState):
    """Stable vertex labels and explicit pair exclusions for one collision surface."""

    body_ids: Array
    patch_ids: Array
    static_mask: Array
    excluded_vertex_pairs: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_count: int,
        /,
        *,
        body_ids: ArrayLike | None = None,
        patch_ids: ArrayLike | None = None,
        static_mask: ArrayLike | None = None,
        excluded_vertex_pairs: ArrayLike | None = None,
    ):
        count = int(vertex_count)
        if count <= 0:
            raise ValueError("Contact pair policy requires a positive vertex count.")
        body = (
            np.zeros((count,), dtype=np.int64)
            if body_ids is None
            else np.asarray(body_ids)
        )
        patch = (
            np.zeros((count,), dtype=np.int64)
            if patch_ids is None
            else np.asarray(patch_ids)
        )
        static = (
            np.zeros((count,), dtype=bool)
            if static_mask is None
            else np.asarray(static_mask, dtype=bool)
        )
        excluded = (
            np.empty((0, 2), dtype=np.int64)
            if excluded_vertex_pairs is None
            else np.asarray(excluded_vertex_pairs)
        )
        if body.shape != (count,) or patch.shape != (count,):
            raise ValueError("Contact body and patch IDs must have vertex shape.")
        if not np.issubdtype(body.dtype, np.integer) or not np.issubdtype(
            patch.dtype, np.integer
        ):
            raise TypeError("Contact body and patch IDs must contain integers.")
        if static.shape != (count,):
            raise ValueError("Contact static_mask must have vertex shape.")
        if (
            excluded.ndim != 2
            or excluded.shape[1:] != (2,)
            or not np.issubdtype(excluded.dtype, np.integer)
        ):
            raise TypeError("excluded_vertex_pairs must be an integer (pairs, 2) array.")
        if np.any(body < 0) or np.any(patch < 0):
            raise ValueError("Contact body and patch IDs must be nonnegative.")
        excluded = np.sort(excluded.astype(np.int64, copy=False), axis=1)
        if excluded.size:
            if (
                np.any(excluded < 0)
                or np.any(excluded >= count)
                or np.any(excluded[:, 0] == excluded[:, 1])
            ):
                raise ValueError("Excluded contact vertex pairs are invalid.")
            excluded = np.unique(excluded, axis=0)
        self.body_ids = jnp.asarray(body, dtype=jnp.int64)
        self.patch_ids = jnp.asarray(patch, dtype=jnp.int64)
        self.static_mask = jnp.asarray(static)
        self.excluded_vertex_pairs = jnp.asarray(excluded, dtype=jnp.int64)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "contact-pair-policy",
                "body_ids": array_tree_fingerprint(body),
                "patch_ids": array_tree_fingerprint(patch),
                "static_mask": array_tree_fingerprint(static),
                "excluded": array_tree_fingerprint(excluded),
            }
        )


class CollisionSurfacePlan(StrictModule, NonTrainableState):
    """Immutable segment or triangle collision topology with stable feature IDs."""

    vertex_ids: Array
    vertex_ordinals: Array
    edges: Array
    edge_ids: Array
    faces: Array
    face_ids: Array
    orientable_mask: Array
    codimensional_mask: Array
    pair_policy: ContactPairPolicy
    ambient_dimension: int = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_ids: ArrayLike,
        /,
        *,
        ambient_dimension: int,
        edges: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        orientable_mask: ArrayLike | None = None,
        codimensional_mask: ArrayLike | None = None,
        pair_policy: ContactPairPolicy | None = None,
        minimum_separation: float = 0.0,
        topology_id: str | None = None,
    ):
        identifiers = np.asarray(vertex_ids)
        dimension = int(ambient_dimension)
        if (
            identifiers.ndim != 1
            or identifiers.size == 0
            or not np.issubdtype(identifiers.dtype, np.integer)
        ):
            raise TypeError("vertex_ids must be one nonempty integer vector.")
        identifiers = identifiers.astype(np.int64, copy=False)
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("Collision vertex IDs must be unique and nonnegative.")
        if dimension not in (2, 3):
            raise ValueError("Collision surfaces require ambient dimension two or three.")
        face_array = (
            np.empty((0, 3), dtype=np.int32) if faces is None else np.asarray(faces)
        )
        if (
            face_array.ndim != 2
            or face_array.shape[1:] != (3,)
            or not np.issubdtype(face_array.dtype, np.integer)
        ):
            raise TypeError("faces must be an integer (faces, 3) array.")
        face_array = face_array.astype(np.int32, copy=False)
        edge_array = _canonical_edges(face_array) if edges is None else np.asarray(edges)
        if (
            edge_array.ndim != 2
            or edge_array.shape[1:] != (2,)
            or not np.issubdtype(edge_array.dtype, np.integer)
        ):
            raise TypeError("edges must be an integer (edges, 2) array.")
        edge_array = edge_array.astype(np.int32, copy=False)
        if edge_array.shape[0] == 0:
            raise ValueError("Collision surfaces require at least one edge.")
        if dimension == 2 and face_array.shape[0] != 0:
            raise ValueError("Two-dimensional collision surfaces use edges, not faces.")
        if dimension == 3 and face_array.shape[0] == 0:
            intrinsic = 1
        else:
            intrinsic = dimension - 1
        count = identifiers.size
        for name, topology in (("edges", edge_array), ("faces", face_array)):
            if np.any(topology < 0) or np.any(topology >= count):
                raise ValueError(f"Collision {name} index an undeclared vertex.")
            if topology.size and np.any(np.diff(np.sort(topology, axis=1), axis=1) == 0):
                raise ValueError(f"Collision {name} contain repeated vertices.")
        edge_array, edge_ids = _sorted_edges(edge_array, identifiers)
        face_array, face_ids = _sorted_faces(face_array, identifiers)
        ranks = np.empty((count,), dtype=np.int64)
        ranks[np.argsort(identifiers, kind="stable")] = np.arange(count, dtype=np.int64)
        orientable = (
            np.ones((count,), dtype=bool)
            if orientable_mask is None
            else np.asarray(orientable_mask, dtype=bool)
        )
        codimensional = (
            np.zeros((count,), dtype=bool)
            if codimensional_mask is None
            else np.asarray(codimensional_mask, dtype=bool)
        )
        if orientable.shape != (count,) or codimensional.shape != (count,):
            raise ValueError(
                "Collision orientation/codimension masks must have vertex shape."
            )
        policy = ContactPairPolicy(count) if pair_policy is None else pair_policy
        if not isinstance(policy, ContactPairPolicy) or policy.body_ids.shape != (count,):
            raise TypeError("pair_policy must match the collision vertex count.")
        separation = float(minimum_separation)
        if not np.isfinite(separation) or separation < 0.0:
            raise ValueError("minimum_separation must be finite and nonnegative.")
        generated = canonical_fingerprint(
            {
                "kind": "collision-surface-topology",
                "ambient_dimension": dimension,
                "intrinsic_dimension": intrinsic,
                "vertex_ids": array_tree_fingerprint(identifiers),
                "edges": array_tree_fingerprint(identifiers[edge_array]),
                "faces": array_tree_fingerprint(identifiers[face_array]),
                "orientable": array_tree_fingerprint(orientable),
                "codimensional": array_tree_fingerprint(codimensional),
                "pair_policy": policy.policy_id,
                "minimum_separation": separation.hex(),
            }
        )
        identifier = generated if topology_id is None else str(topology_id)
        if not identifier:
            raise ValueError("topology_id must be nonempty or None.")
        self.vertex_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.vertex_ordinals = jnp.asarray(ranks, dtype=jnp.int64)
        self.edges = jnp.asarray(edge_array, dtype=jnp.int32)
        self.edge_ids = jnp.asarray(edge_ids, dtype=jnp.int64)
        self.faces = jnp.asarray(face_array, dtype=jnp.int32)
        self.face_ids = jnp.asarray(face_ids, dtype=jnp.int64)
        self.orientable_mask = jnp.asarray(orientable)
        self.codimensional_mask = jnp.asarray(codimensional)
        self.pair_policy = policy
        self.ambient_dimension = dimension
        self.intrinsic_dimension = intrinsic
        self.minimum_separation = separation
        self.topology_id = identifier

    @property
    def vertex_count(self) -> int:
        return int(self.vertex_ids.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edges.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


class CollisionMapEvidence(StrictModule):
    primal_pairing: Array
    transpose_pairing: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array


class PreparedCollisionSurface(StrictModule, NonTrainableState):
    """Numeric collision surface and exact mechanics-to-surface linear map."""

    plan: CollisionSurfacePlan
    rest_positions: Array
    displacement_operator: AbstractLinearOperator
    precision: ContactPrecisionPolicy
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CollisionSurfacePlan,
        rest_positions: ArrayLike,
        displacement_operator: AbstractLinearOperator,
        /,
        *,
        precision: ContactPrecisionPolicy | None = None,
        prepared_id: str | None = None,
    ):
        if not isinstance(plan, CollisionSurfacePlan):
            raise TypeError("plan must be CollisionSurfacePlan.")
        if not isinstance(displacement_operator, AbstractLinearOperator):
            raise TypeError("displacement_operator must be AbstractLinearOperator.")
        policy = ContactPrecisionPolicy() if precision is None else precision
        if not isinstance(policy, ContactPrecisionPolicy):
            raise TypeError("precision must be ContactPrecisionPolicy or None.")
        positions = np.asarray(rest_positions, dtype=policy.geometry_dtype)
        expected = (plan.vertex_count, plan.ambient_dimension)
        if positions.shape != expected or np.any(~np.isfinite(positions)):
            raise ValueError(
                f"rest_positions must be one finite array of shape {expected}."
            )
        target = displacement_operator.target
        if not isinstance(target, ArraySpace) or target.shape != expected:
            raise ValueError(
                "Collision displacement operator target must match surface vertices."
            )
        if target.dtype != policy.geometry_dtype:
            raise TypeError(
                "Collision displacement map and geometry precision must agree."
            )
        selected_edges = positions[np.asarray(plan.edges)]
        edge_length = np.sqrt(
            np.sum((selected_edges[:, 1] - selected_edges[:, 0]) ** 2, axis=-1)
        )
        if np.any(edge_length <= 0.0):
            raise ValueError("Collision surface contains a zero-length rest edge.")
        if plan.face_count:
            selected_faces = positions[np.asarray(plan.faces)]
            double_area = np.sqrt(
                np.sum(
                    np.cross(
                        selected_faces[:, 1] - selected_faces[:, 0],
                        selected_faces[:, 2] - selected_faces[:, 0],
                    )
                    ** 2,
                    axis=-1,
                )
            )
            if np.any(double_area <= 0.0):
                raise ValueError("Collision surface contains a degenerate rest face.")
        generated = canonical_fingerprint(
            {
                "kind": "prepared-collision-surface",
                "topology": plan.topology_id,
                "rest_positions": array_tree_fingerprint(positions),
                "operator": displacement_operator.operator_id,
                "precision": policy.policy_id,
            }
        )
        identifier = generated if prepared_id is None else str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be nonempty or None.")
        self.plan = plan
        self.rest_positions = jnp.asarray(positions, dtype=policy.geometry_dtype)
        self.displacement_operator = displacement_operator
        self.precision = policy
        self.prepared_id = identifier

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.displacement_operator.source

    @property
    def target_space(self) -> ArraySpace:
        target = self.displacement_operator.target
        if not isinstance(target, ArraySpace):
            raise TypeError("Prepared collision surface lost its array target space.")
        return target

    def map_values(self, state: PyTree[Any], /) -> Array:
        """Map a mechanics-space vector without adding reference positions."""
        return self.precision.geometry(self.displacement_operator.mv(state))

    def positions(self, state: PyTree[Any], /) -> Array:
        return self.precision.geometry(self.rest_positions + self.map_values(state))

    def pullback(self, surface_dual: ArrayLike, /) -> PyTree[Array]:
        value = self.target_space.validate(
            jnp.asarray(surface_dual, dtype=self.target_space.dtype)
        )
        return self.displacement_operator.transpose_mv(value)

    def duality_evidence(
        self,
        state: PyTree[Any],
        surface_dual: ArrayLike,
        /,
    ) -> CollisionMapEvidence:
        state_ = self.source_space.validate(state)
        dual = self.target_space.validate(
            jnp.asarray(surface_dual, dtype=self.target_space.dtype)
        )
        mapped = self.displacement_operator.mv(state_)
        pulled = self.displacement_operator.transpose_mv(dual)
        primal = self.target_space.inner(mapped, dual)
        transpose = self.source_space.inner(state_, pulled)
        residual = primal - transpose
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(primal), jnp.abs(transpose)))
        tolerance = jnp.finfo(self.precision.certification_dtype).eps * max(
            32, 4 * self.plan.vertex_count * self.plan.ambient_dimension
        )
        finite = jnp.all(jnp.isfinite(jnp.stack((primal, transpose, residual, scale))))
        valid = finite & (jnp.abs(residual) <= tolerance * scale)
        return CollisionMapEvidence(primal, transpose, residual, scale, finite, valid)


def selection_collision_operator(
    source_space: ArraySpace,
    vertex_indices: ArrayLike,
    /,
    *,
    operator_id: str | None = None,
) -> FunctionLinearOperator:
    """Create an exact vertex gather/scatter from an array-valued mechanics space."""
    if not isinstance(source_space, ArraySpace) or len(source_space.shape) != 2:
        raise TypeError("Collision vertex selection requires a rank-two ArraySpace.")
    indices = np.asarray(vertex_indices)
    if (
        indices.ndim != 1
        or indices.size == 0
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise TypeError("vertex_indices must be one nonempty integer vector.")
    indices = indices.astype(np.int32, copy=False)
    if np.any(indices < 0) or np.any(indices >= source_space.shape[0]):
        raise ValueError("Collision vertex selection index is out of bounds.")
    dimension = source_space.shape[1]
    target = ArraySpace((indices.size, dimension), dtype=source_space.dtype)
    indices_array = jnp.asarray(indices, dtype=jnp.int32)

    def gather(value):
        return value[indices_array]

    def scatter(value):
        return (
            jnp.zeros(source_space.shape, dtype=value.dtype).at[indices_array].add(value)
        )

    return FunctionLinearOperator(
        gather,
        source=source_space,
        target=target,
        transpose_action=scatter,
        operator_id=operator_id,
    )


def static_collision_operator(
    source_space: AbstractVectorSpace,
    vertex_count: int,
    ambient_dimension: int,
    /,
    *,
    dtype: Any = np.float64,
    operator_id: str | None = None,
) -> FunctionLinearOperator:
    """Create a zero displacement map for a static collision surface."""
    if not isinstance(source_space, AbstractVectorSpace):
        raise TypeError("source_space must be AbstractVectorSpace.")
    target = ArraySpace((int(vertex_count), int(ambient_dimension)), dtype=dtype)

    def zero(_):
        return target.zeros()

    def zero_transpose(_):
        return source_space.zeros()

    return FunctionLinearOperator(
        zero,
        source=source_space,
        target=target,
        transpose_action=zero_transpose,
        operator_id=operator_id,
    )


class PreparedCollisionScene(StrictModule, NonTrainableState):
    """Several collision surfaces sharing one mechanics state space."""

    surfaces: tuple[PreparedCollisionSurface, ...]
    vertex_offsets: tuple[int, ...] = eqx.field(static=True)
    edge_offsets: tuple[int, ...] = eqx.field(static=True)
    face_offsets: tuple[int, ...] = eqx.field(static=True)
    scene_id: str = eqx.field(static=True)

    def __init__(self, surfaces: Sequence[PreparedCollisionSurface], /):
        values = tuple(surfaces)
        if not values or not all(
            isinstance(value, PreparedCollisionSurface) for value in values
        ):
            raise TypeError("surfaces must contain PreparedCollisionSurface values.")
        source = values[0].source_space
        dimension = values[0].plan.ambient_dimension
        dtype = values[0].precision.geometry_dtype
        for value in values[1:]:
            if not source.compatible(value.source_space):
                raise ValueError(
                    "Collision scene surfaces must share one mechanics state space."
                )
            if (
                value.plan.ambient_dimension != dimension
                or value.precision.geometry_dtype != dtype
            ):
                raise ValueError(
                    "Collision scene surfaces must share dimension and geometry dtype."
                )
        vertex_offsets = [0]
        edge_offsets = [0]
        face_offsets = [0]
        for value in values:
            vertex_offsets.append(vertex_offsets[-1] + value.plan.vertex_count)
            edge_offsets.append(edge_offsets[-1] + value.plan.edge_count)
            face_offsets.append(face_offsets[-1] + value.plan.face_count)
        self.surfaces = values
        self.vertex_offsets = tuple(vertex_offsets)
        self.edge_offsets = tuple(edge_offsets)
        self.face_offsets = tuple(face_offsets)
        self.scene_id = canonical_fingerprint(
            {
                "kind": "prepared-collision-scene",
                "surfaces": [value.prepared_id for value in values],
            }
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.surfaces[0].source_space

    @property
    def ambient_dimension(self) -> int:
        return self.surfaces[0].plan.ambient_dimension

    @property
    def vertex_count(self) -> int:
        return self.vertex_offsets[-1]

    @property
    def edge_count(self) -> int:
        return self.edge_offsets[-1]

    @property
    def face_count(self) -> int:
        return self.face_offsets[-1]

    @property
    def edges(self) -> Array:
        values = tuple(
            surface.plan.edges + self.vertex_offsets[index]
            for index, surface in enumerate(self.surfaces)
        )
        return jnp.concatenate(values, axis=0)

    @property
    def faces(self) -> Array:
        values = tuple(
            surface.plan.faces + self.vertex_offsets[index]
            for index, surface in enumerate(self.surfaces)
            if surface.plan.face_count
        )
        if not values:
            return jnp.empty((0, 3), dtype=jnp.int32)
        return jnp.concatenate(values, axis=0)

    @property
    def vertex_body_ids(self) -> Array:
        return jnp.concatenate(
            tuple(value.plan.pair_policy.body_ids for value in self.surfaces)
        )

    @property
    def vertex_patch_ids(self) -> Array:
        return jnp.concatenate(
            tuple(value.plan.pair_policy.patch_ids for value in self.surfaces)
        )

    @property
    def vertex_static_mask(self) -> Array:
        return jnp.concatenate(
            tuple(value.plan.pair_policy.static_mask for value in self.surfaces)
        )

    @property
    def minimum_separation(self) -> Array:
        return jnp.concatenate(
            tuple(
                jnp.full(
                    (value.plan.vertex_count,),
                    value.plan.minimum_separation,
                    dtype=value.precision.geometry_dtype,
                )
                for value in self.surfaces
            )
        )

    def map_values(self, state: PyTree[Any], /) -> Array:
        """Map a shared mechanics-space vector to concatenated surface values."""
        return jnp.concatenate(
            tuple(value.map_values(state) for value in self.surfaces), axis=0
        )

    def positions(self, state: PyTree[Any], /) -> Array:
        return jnp.concatenate(
            tuple(value.positions(state) for value in self.surfaces), axis=0
        )

    def pullback(self, scene_dual: ArrayLike, /) -> PyTree[Array]:
        dual = jnp.asarray(scene_dual, dtype=self.surfaces[0].precision.geometry_dtype)
        expected = (self.vertex_count, self.ambient_dimension)
        if dual.shape != expected:
            raise ValueError(f"scene_dual must have shape {expected}.")
        contributions = tuple(
            surface.pullback(
                dual[self.vertex_offsets[index] : self.vertex_offsets[index + 1]]
            )
            for index, surface in enumerate(self.surfaces)
        )
        return jax.tree.map(lambda *values: sum(values), *contributions)


def prepare_cell_mesh_collision_surface(
    mesh: CellMesh,
    source_space: ArraySpace,
    /,
    *,
    body_id: int = 0,
    patch_id: int = 0,
    static: bool = False,
    minimum_separation: float = 0.0,
    precision: ContactPrecisionPolicy | None = None,
) -> PreparedCollisionSurface:
    """Extract a compact segment/triangle boundary from a nodal T3/T4 mesh."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(source_space, ArraySpace):
        raise TypeError("source_space must be ArraySpace.")
    expected = (mesh.coordinates.shape[0], mesh.ambient_dimension)
    if source_space.shape != expected:
        raise ValueError(
            "Direct cell-mesh collision extraction requires one nodal vector "
            "unknown per mesh coordinate."
        )
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        boundary_edges = np.asarray(connectivity.edges)[
            np.asarray(connectivity.boundary_edges, dtype=bool)
        ]
        boundary_vertices = np.unique(boundary_edges.reshape((-1,)))
        local = np.full((mesh.coordinates.shape[0],), -1, dtype=np.int32)
        local[boundary_vertices] = np.arange(boundary_vertices.size, dtype=np.int32)
        edges = local[boundary_edges]
        faces = None
    elif isinstance(connectivity, TetrahedralConnectivity):
        boundary_faces = np.asarray(connectivity.faces)[
            np.asarray(connectivity.boundary_faces, dtype=bool)
        ]
        boundary_vertices = np.unique(boundary_faces.reshape((-1,)))
        local = np.full((mesh.coordinates.shape[0],), -1, dtype=np.int32)
        local[boundary_vertices] = np.arange(boundary_vertices.size, dtype=np.int32)
        faces = local[boundary_faces]
        edges = None
    else:
        raise TypeError(
            "Certified direct collision extraction currently supports polygonal "
            "2-D meshes and tetrahedral 3-D meshes."
        )
    count = int(boundary_vertices.size)
    policy = ContactPairPolicy(
        count,
        body_ids=np.full((count,), int(body_id), dtype=np.int64),
        patch_ids=np.full((count,), int(patch_id), dtype=np.int64),
        static_mask=np.full((count,), bool(static)),
    )
    plan = CollisionSurfacePlan(
        np.asarray(mesh.vertex_global_ids)[boundary_vertices],
        ambient_dimension=mesh.ambient_dimension,
        edges=edges,
        faces=faces,
        pair_policy=policy,
        minimum_separation=minimum_separation,
    )
    operator = (
        static_collision_operator(
            source_space,
            count,
            mesh.ambient_dimension,
            dtype=source_space.dtype,
        )
        if static
        else selection_collision_operator(source_space, boundary_vertices)
    )
    return PreparedCollisionSurface(
        plan,
        np.asarray(mesh.coordinates)[boundary_vertices],
        operator,
        precision=precision,
    )


__all__ = [
    "CollisionMapEvidence",
    "CollisionSurfacePlan",
    "ContactPairPolicy",
    "PreparedCollisionScene",
    "PreparedCollisionSurface",
    "prepare_cell_mesh_collision_surface",
    "selection_collision_operator",
    "static_collision_operator",
]
