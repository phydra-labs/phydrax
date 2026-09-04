#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
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
    dual_transpose,
    DualSpace,
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
    return sorted_edges, order


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
    return sorted_faces, order


class CollisionFeatureKind(IntEnum):
    VERTEX = 0
    EDGE = 1
    FACE = 2
    ANALYTIC = 3


def _feature_integer_values(
    value: ArrayLike | int,
    count: int,
    name: str,
    /,
) -> np.ndarray:
    values = np.asarray(value)
    if values.shape == ():
        values = np.full((count,), values, dtype=values.dtype)
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must be an integer scalar or feature vector.")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return values


def _feature_metric_values(
    value: ArrayLike | float,
    count: int,
    name: str,
    /,
) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.shape == ():
        values = np.full((count,), float(values), dtype=float)
    if values.shape != (count,):
        raise ValueError(f"{name} must be scalar or have one value per feature.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} values must be finite and nonnegative.")
    return values


class CollisionFeaturePolicy(StrictModule, NonTrainableState):
    """Immutable provenance and physical policy for collision primitives."""

    feature_ids: Array
    feature_kinds: Array
    participant_ids: Array
    body_ids: Array
    material_ids: Array
    patch_ids: Array
    static_mask: Array
    physical_radius: Array
    solver_clearance: Array
    proxy_error: Array
    vertex_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    analytic_count: int = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_ids: ArrayLike,
        feature_kinds: ArrayLike,
        /,
        *,
        participant_ids: ArrayLike | int,
        body_ids: ArrayLike | int,
        material_ids: ArrayLike | int,
        patch_ids: ArrayLike | int,
        static_mask: ArrayLike | bool = False,
        physical_radius: ArrayLike | float = 0.0,
        solver_clearance: ArrayLike | float = 0.0,
        proxy_error: ArrayLike | float = 0.0,
        provenance_id: str,
    ):
        identifiers = np.asarray(feature_ids)
        kinds = np.asarray(feature_kinds)
        if (
            identifiers.ndim != 1
            or identifiers.size == 0
            or not np.issubdtype(identifiers.dtype, np.integer)
        ):
            raise TypeError("feature_ids must be one nonempty integer vector.")
        if kinds.shape != identifiers.shape or not np.issubdtype(kinds.dtype, np.integer):
            raise TypeError("feature_kinds must be one integer per feature.")
        identifiers = identifiers.astype(np.int64, copy=False)
        kinds = kinds.astype(np.int32, copy=False)
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("Collision feature IDs must be unique and nonnegative.")
        valid_kinds = np.arange(len(CollisionFeatureKind), dtype=np.int32)
        if np.any(~np.isin(kinds, valid_kinds)):
            raise ValueError("Collision feature kind is invalid.")
        counts = tuple(
            int(np.count_nonzero(kinds == int(kind))) for kind in CollisionFeatureKind
        )
        expected_kinds = np.concatenate(
            tuple(
                np.full((counts[int(kind)],), int(kind), dtype=np.int32)
                for kind in CollisionFeatureKind
            )
        )
        if not np.array_equal(kinds, expected_kinds):
            raise ValueError(
                "Collision features must use canonical vertex/edge/face/analytic order."
            )
        count = int(identifiers.size)
        participant = _feature_integer_values(participant_ids, count, "participant_ids")
        body = _feature_integer_values(body_ids, count, "body_ids")
        material = _feature_integer_values(material_ids, count, "material_ids")
        patch = _feature_integer_values(patch_ids, count, "patch_ids")
        static = np.asarray(static_mask, dtype=bool)
        if static.shape == ():
            static = np.full((count,), bool(static), dtype=bool)
        if static.shape != (count,):
            raise ValueError("static_mask must be scalar or have feature shape.")
        radius = _feature_metric_values(physical_radius, count, "physical_radius")
        clearance = _feature_metric_values(solver_clearance, count, "solver_clearance")
        error = _feature_metric_values(proxy_error, count, "proxy_error")
        provenance = str(provenance_id)
        if not provenance:
            raise ValueError("provenance_id must be nonempty.")
        self.feature_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.feature_kinds = jnp.asarray(kinds, dtype=jnp.int32)
        self.participant_ids = jnp.asarray(participant, dtype=jnp.int64)
        self.body_ids = jnp.asarray(body, dtype=jnp.int64)
        self.material_ids = jnp.asarray(material, dtype=jnp.int64)
        self.patch_ids = jnp.asarray(patch, dtype=jnp.int64)
        self.static_mask = jnp.asarray(static)
        self.physical_radius = jnp.asarray(radius)
        self.solver_clearance = jnp.asarray(clearance)
        self.proxy_error = jnp.asarray(error)
        (
            self.vertex_count,
            self.edge_count,
            self.face_count,
            self.analytic_count,
        ) = counts
        self.provenance_id = provenance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "collision-feature-policy",
                "features": array_tree_fingerprint((identifiers, kinds)),
                "participant_ids": array_tree_fingerprint(participant),
                "body_ids": array_tree_fingerprint(body),
                "material_ids": array_tree_fingerprint(material),
                "patch_ids": array_tree_fingerprint(patch),
                "static_mask": array_tree_fingerprint(static),
                "physical_radius": array_tree_fingerprint(radius),
                "solver_clearance": array_tree_fingerprint(clearance),
                "proxy_error": array_tree_fingerprint(error),
                "provenance_id": provenance,
            }
        )

    @property
    def contact_extent(self) -> Array:
        return self.physical_radius + self.solver_clearance + self.proxy_error

    @property
    def vertex_slice(self) -> slice:
        return slice(0, self.vertex_count)

    @property
    def edge_slice(self) -> slice:
        return slice(self.vertex_count, self.vertex_count + self.edge_count)

    @property
    def face_slice(self) -> slice:
        start = self.vertex_count + self.edge_count
        return slice(start, start + self.face_count)

    @property
    def analytic_slice(self) -> slice:
        start = self.vertex_count + self.edge_count + self.face_count
        return slice(start, start + self.analytic_count)

    def with_proxy_error(
        self,
        proxy_error: ArrayLike,
        /,
        *,
        provenance_id: str | None = None,
    ) -> CollisionFeaturePolicy:
        return CollisionFeaturePolicy(
            self.feature_ids,
            self.feature_kinds,
            participant_ids=self.participant_ids,
            body_ids=self.body_ids,
            material_ids=self.material_ids,
            patch_ids=self.patch_ids,
            static_mask=self.static_mask,
            physical_radius=self.physical_radius,
            solver_clearance=self.solver_clearance,
            proxy_error=proxy_error,
            provenance_id=(
                self.provenance_id if provenance_id is None else provenance_id
            ),
        )


class ContactPairPolicy(StrictModule, NonTrainableState):
    """Explicit reciprocal participant pairs and local self-contact exclusions."""

    allowed_participant_pairs: Array
    excluded_vertex_pairs: Array
    unrestricted: bool = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_count: int,
        /,
        *,
        allowed_participant_pairs: ArrayLike | None = None,
        excluded_vertex_pairs: ArrayLike | None = None,
    ):
        count = int(vertex_count)
        if count <= 0:
            raise ValueError("Contact pair policy requires a positive vertex count.")
        unrestricted = allowed_participant_pairs is None
        allowed = (
            np.empty((0, 2), dtype=np.int64)
            if unrestricted
            else np.asarray(allowed_participant_pairs)
        )
        if (
            allowed.ndim != 2
            or allowed.shape[1:] != (2,)
            or not np.issubdtype(allowed.dtype, np.integer)
        ):
            raise TypeError(
                "allowed_participant_pairs must be an integer (pairs, 2) array."
            )
        allowed = np.sort(allowed.astype(np.int64, copy=False), axis=1)
        if np.any(allowed < 0):
            raise ValueError("Allowed participant IDs must be nonnegative.")
        allowed = np.unique(allowed, axis=0)
        excluded = (
            np.empty((0, 2), dtype=np.int64)
            if excluded_vertex_pairs is None
            else np.asarray(excluded_vertex_pairs)
        )
        if (
            excluded.ndim != 2
            or excluded.shape[1:] != (2,)
            or not np.issubdtype(excluded.dtype, np.integer)
        ):
            raise TypeError("excluded_vertex_pairs must be an integer (pairs, 2) array.")
        excluded = np.sort(excluded.astype(np.int64, copy=False), axis=1)
        if excluded.size:
            if (
                np.any(excluded < 0)
                or np.any(excluded >= count)
                or np.any(excluded[:, 0] == excluded[:, 1])
            ):
                raise ValueError("Excluded contact vertex pairs are invalid.")
            excluded = np.unique(excluded, axis=0)
        self.allowed_participant_pairs = jnp.asarray(allowed, dtype=jnp.int64)
        self.excluded_vertex_pairs = jnp.asarray(excluded, dtype=jnp.int64)
        self.unrestricted = unrestricted
        self.vertex_count = count
        self.policy_id = canonical_fingerprint(
            {
                "kind": "contact-pair-policy",
                "vertex_count": count,
                "allowed": (
                    "unrestricted" if unrestricted else array_tree_fingerprint(allowed)
                ),
                "excluded": array_tree_fingerprint(excluded),
            }
        )

    def allows(self, left_participant: int, right_participant: int, /) -> bool:
        if self.unrestricted:
            return True
        pair = np.sort(np.asarray((left_participant, right_participant), dtype=np.int64))
        allowed = np.asarray(self.allowed_participant_pairs)
        return bool(allowed.size and np.any(np.all(allowed == pair, axis=1)))


def _surface_feature_values(
    vertex_values: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    name: str,
    /,
) -> np.ndarray:
    primitive_values: list[np.ndarray] = [vertex_values]
    for topology in (edges, faces):
        if not topology.size:
            continue
        endpoint_values = vertex_values[topology]
        agreement = np.all(endpoint_values == endpoint_values[:, :1], axis=1)
        if not np.all(agreement):
            raise ValueError(
                f"Collision {name} labels must agree on every primitive endpoint."
            )
        primitive_values.append(endpoint_values[:, 0])
    return np.concatenate(tuple(primitive_values))


def _surface_feature_static(
    vertex_values: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    /,
) -> np.ndarray:
    primitive_values: list[np.ndarray] = [vertex_values]
    for topology in (edges, faces):
        if topology.size:
            primitive_values.append(np.all(vertex_values[topology], axis=1))
    return np.concatenate(tuple(primitive_values))


def _surface_feature_metrics(
    vertex_values: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    /,
) -> np.ndarray:
    primitive_values: list[np.ndarray] = [vertex_values]
    for topology in (edges, faces):
        if topology.size:
            primitive_values.append(np.max(vertex_values[topology], axis=1))
    return np.concatenate(tuple(primitive_values))


def _default_surface_feature_policy(
    vertex_ids: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    edge_ids: np.ndarray | None,
    face_ids: np.ndarray | None,
    participant_ids: ArrayLike | int,
    body_ids: ArrayLike | int,
    material_ids: ArrayLike | int | None,
    patch_ids: ArrayLike | int,
    static_mask: ArrayLike | bool,
    physical_radius: ArrayLike | float,
    solver_clearance: ArrayLike | float,
    proxy_error: ArrayLike | float,
    provenance_id: str | None,
) -> CollisionFeaturePolicy:
    vertex_count = int(vertex_ids.size)
    edge_count = int(edges.shape[0])
    face_count = int(faces.shape[0])
    next_identifier = int(np.max(vertex_ids)) + 1
    edge_identifiers = (
        np.arange(next_identifier, next_identifier + edge_count, dtype=np.int64)
        if edge_ids is None
        else np.asarray(edge_ids)
    )
    next_identifier += edge_count
    face_identifiers = (
        np.arange(next_identifier, next_identifier + face_count, dtype=np.int64)
        if face_ids is None
        else np.asarray(face_ids)
    )
    if (
        edge_identifiers.shape != (edge_count,)
        or face_identifiers.shape != (face_count,)
        or not np.issubdtype(edge_identifiers.dtype, np.integer)
        or not np.issubdtype(face_identifiers.dtype, np.integer)
    ):
        raise TypeError("edge_ids and face_ids must match canonical primitive counts.")
    feature_ids = np.concatenate(
        (
            vertex_ids,
            edge_identifiers.astype(np.int64, copy=False),
            face_identifiers.astype(np.int64, copy=False),
        )
    )
    feature_kinds = np.concatenate(
        (
            np.full((vertex_count,), int(CollisionFeatureKind.VERTEX), dtype=np.int32),
            np.full((edge_count,), int(CollisionFeatureKind.EDGE), dtype=np.int32),
            np.full((face_count,), int(CollisionFeatureKind.FACE), dtype=np.int32),
        )
    )
    participant = _feature_integer_values(
        participant_ids, vertex_count, "participant_ids"
    )
    body = _feature_integer_values(body_ids, vertex_count, "body_ids")
    material = (
        body
        if material_ids is None
        else _feature_integer_values(material_ids, vertex_count, "material_ids")
    )
    patch = _feature_integer_values(patch_ids, vertex_count, "patch_ids")
    static = np.asarray(static_mask, dtype=bool)
    if static.shape == ():
        static = np.full((vertex_count,), bool(static), dtype=bool)
    if static.shape != (vertex_count,):
        raise ValueError("static_mask must be scalar or have vertex shape.")
    radius = _feature_metric_values(physical_radius, vertex_count, "physical_radius")
    clearance = _feature_metric_values(solver_clearance, vertex_count, "solver_clearance")
    error = _feature_metric_values(proxy_error, vertex_count, "proxy_error")
    provenance = (
        canonical_fingerprint(
            {
                "kind": "generated-collision-feature-provenance",
                "vertex_ids": array_tree_fingerprint(vertex_ids),
                "edges": array_tree_fingerprint(vertex_ids[edges]),
                "faces": array_tree_fingerprint(vertex_ids[faces]),
            }
        )
        if provenance_id is None
        else str(provenance_id)
    )
    return CollisionFeaturePolicy(
        feature_ids,
        feature_kinds,
        participant_ids=_surface_feature_values(participant, edges, faces, "participant"),
        body_ids=_surface_feature_values(body, edges, faces, "body"),
        material_ids=_surface_feature_values(material, edges, faces, "material"),
        patch_ids=_surface_feature_values(patch, edges, faces, "patch"),
        static_mask=_surface_feature_static(static, edges, faces),
        physical_radius=_surface_feature_metrics(radius, edges, faces),
        solver_clearance=_surface_feature_metrics(clearance, edges, faces),
        proxy_error=_surface_feature_metrics(error, edges, faces),
        provenance_id=provenance,
    )


class CollisionSurfacePlan(StrictModule, NonTrainableState):
    """Immutable segment or triangle topology with provenance-bound features."""

    vertex_ordinals: Array
    edges: Array
    faces: Array
    orientable_mask: Array
    codimensional_mask: Array
    feature_policy: CollisionFeaturePolicy
    pair_policy: ContactPairPolicy
    ambient_dimension: int = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    allow_isolated_vertices: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_ids: ArrayLike,
        /,
        *,
        ambient_dimension: int,
        edges: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        edge_ids: ArrayLike | None = None,
        face_ids: ArrayLike | None = None,
        orientable_mask: ArrayLike | None = None,
        codimensional_mask: ArrayLike | None = None,
        feature_policy: CollisionFeaturePolicy | None = None,
        pair_policy: ContactPairPolicy | None = None,
        participant_ids: ArrayLike | int = 0,
        body_ids: ArrayLike | int = 0,
        material_ids: ArrayLike | int | None = None,
        patch_ids: ArrayLike | int = 0,
        static_mask: ArrayLike | bool = False,
        physical_radius: float | ArrayLike = 0.0,
        solver_clearance: float | ArrayLike = 0.0,
        proxy_error: float | ArrayLike = 0.0,
        feature_provenance_id: str | None = None,
        allow_isolated_vertices: bool = False,
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
        allow_isolated = bool(allow_isolated_vertices)
        if edge_array.shape[0] == 0 and not allow_isolated:
            raise ValueError(
                "Collision surfaces require an edge unless isolated vertices "
                "are explicitly enabled."
            )
        if dimension == 2 and face_array.shape[0] != 0:
            raise ValueError("Two-dimensional collision surfaces use edges, not faces.")
        if edge_array.shape[0] == 0:
            intrinsic = 0
        elif dimension == 3 and face_array.shape[0] == 0:
            intrinsic = 1
        else:
            intrinsic = dimension - 1
        count = identifiers.size
        for name, topology in (("edges", edge_array), ("faces", face_array)):
            if np.any(topology < 0) or np.any(topology >= count):
                raise ValueError(f"Collision {name} index an undeclared vertex.")
            if topology.size and np.any(np.diff(np.sort(topology, axis=1), axis=1) == 0):
                raise ValueError(f"Collision {name} contain repeated vertices.")
        edge_array, edge_order = _sorted_edges(edge_array, identifiers)
        face_array, face_order = _sorted_faces(face_array, identifiers)
        sorted_edge_ids = None if edge_ids is None else np.asarray(edge_ids)[edge_order]
        sorted_face_ids = None if face_ids is None else np.asarray(face_ids)[face_order]
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
        pair = ContactPairPolicy(count) if pair_policy is None else pair_policy
        if not isinstance(pair, ContactPairPolicy) or pair.vertex_count != count:
            raise TypeError("pair_policy must match the collision vertex count.")
        if feature_policy is None:
            features = _default_surface_feature_policy(
                identifiers,
                edge_array,
                face_array,
                edge_ids=sorted_edge_ids,
                face_ids=sorted_face_ids,
                participant_ids=participant_ids,
                body_ids=body_ids,
                material_ids=material_ids,
                patch_ids=patch_ids,
                static_mask=static_mask,
                physical_radius=physical_radius,
                solver_clearance=solver_clearance,
                proxy_error=proxy_error,
                provenance_id=feature_provenance_id,
            )
        else:
            if not isinstance(feature_policy, CollisionFeaturePolicy):
                raise TypeError("feature_policy must be CollisionFeaturePolicy or None.")
            if (
                feature_policy.vertex_count != count
                or feature_policy.edge_count != edge_array.shape[0]
                or feature_policy.face_count != face_array.shape[0]
                or feature_policy.analytic_count != 0
            ):
                raise ValueError(
                    "feature_policy primitive counts do not match collision topology."
                )
            features = feature_policy
            if not np.array_equal(
                np.asarray(features.feature_ids[features.vertex_slice]), identifiers
            ):
                raise ValueError(
                    "Collision vertex IDs and feature-policy vertex IDs must agree."
                )
            for name, values in (
                ("participant", features.participant_ids),
                ("body", features.body_ids),
                ("material", features.material_ids),
                ("patch", features.patch_ids),
            ):
                expected = _surface_feature_values(
                    np.asarray(values[features.vertex_slice]),
                    edge_array,
                    face_array,
                    name,
                )
                if not np.array_equal(np.asarray(values), expected):
                    raise ValueError(
                        f"Collision {name} feature labels must equal their endpoints."
                    )
            expected_static = _surface_feature_static(
                np.asarray(features.static_mask[features.vertex_slice]),
                edge_array,
                face_array,
            )
            if not np.array_equal(np.asarray(features.static_mask), expected_static):
                raise ValueError(
                    "Collision static feature labels must equal endpoint conjunctions."
                )
        generated = canonical_fingerprint(
            {
                "kind": "collision-surface-topology",
                "ambient_dimension": dimension,
                "intrinsic_dimension": intrinsic,
                "edges": array_tree_fingerprint(identifiers[edge_array]),
                "faces": array_tree_fingerprint(identifiers[face_array]),
                "orientable": array_tree_fingerprint(orientable),
                "codimensional": array_tree_fingerprint(codimensional),
                "feature_policy": features.policy_id,
                "pair_policy": pair.policy_id,
                "allow_isolated_vertices": allow_isolated,
            }
        )
        identifier = generated if topology_id is None else str(topology_id)
        if not identifier:
            raise ValueError("topology_id must be nonempty or None.")
        self.vertex_ordinals = jnp.asarray(ranks, dtype=jnp.int64)
        self.edges = jnp.asarray(edge_array, dtype=jnp.int32)
        self.faces = jnp.asarray(face_array, dtype=jnp.int32)
        self.orientable_mask = jnp.asarray(orientable)
        self.codimensional_mask = jnp.asarray(codimensional)
        self.feature_policy = features
        self.pair_policy = pair
        self.ambient_dimension = dimension
        self.intrinsic_dimension = intrinsic
        self.allow_isolated_vertices = allow_isolated
        self.topology_id = identifier

    @property
    def vertex_count(self) -> int:
        return self.feature_policy.vertex_count

    @property
    def edge_count(self) -> int:
        return self.feature_policy.edge_count

    @property
    def face_count(self) -> int:
        return self.feature_policy.face_count

    @property
    def vertex_ids(self) -> Array:
        return self.feature_policy.feature_ids[self.feature_policy.vertex_slice]

    @property
    def edge_ids(self) -> Array:
        return self.feature_policy.feature_ids[self.feature_policy.edge_slice]

    @property
    def face_ids(self) -> Array:
        return self.feature_policy.feature_ids[self.feature_policy.face_slice]

    @property
    def vertex_physical_radius(self) -> Array:
        return self.feature_policy.physical_radius[self.feature_policy.vertex_slice]

    @property
    def vertex_solver_clearance(self) -> Array:
        return self.feature_policy.solver_clearance[self.feature_policy.vertex_slice]

    @property
    def vertex_proxy_error(self) -> Array:
        return self.feature_policy.proxy_error[self.feature_policy.vertex_slice]

    @property
    def physical_radius(self) -> float:
        return float(np.max(np.asarray(self.feature_policy.physical_radius), initial=0.0))

    @property
    def solver_clearance(self) -> float:
        return float(
            np.max(np.asarray(self.feature_policy.solver_clearance), initial=0.0)
        )

    @property
    def proxy_error(self) -> float:
        return float(np.max(np.asarray(self.feature_policy.proxy_error), initial=0.0))


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
    def tangent_space(self) -> AbstractVectorSpace:
        return self.displacement_operator.source

    @property
    def effort_space(self) -> DualSpace:
        return DualSpace(self.tangent_space)

    @property
    def contact_velocity_space(self) -> ArraySpace:
        target = self.displacement_operator.target
        if not isinstance(target, ArraySpace):
            raise TypeError("Prepared collision surface lost its array target space.")
        return target

    @property
    def contact_effort_space(self) -> DualSpace:
        return DualSpace(self.contact_velocity_space)

    def map_values(self, state: PyTree[Any], /) -> Array:
        """Map a mechanics-space vector without adding reference positions."""
        return self.precision.geometry(self.displacement_operator.mv(state))

    def positions(self, state: PyTree[Any], /) -> Array:
        return self.precision.geometry(self.rest_positions + self.map_values(state))

    def effort_pullback(self, surface_effort: ArrayLike, /) -> PyTree[Array]:
        effort = self.contact_effort_space.validate(
            jnp.asarray(surface_effort, dtype=self.contact_velocity_space.dtype)
        )
        return self.effort_space.validate(
            dual_transpose(self.displacement_operator).mv(effort)
        )

    def duality_evidence(
        self,
        state: PyTree[Any],
        surface_effort: ArrayLike,
        /,
    ) -> CollisionMapEvidence:
        state_ = self.tangent_space.validate(state)
        effort = self.contact_effort_space.validate(
            jnp.asarray(surface_effort, dtype=self.contact_velocity_space.dtype)
        )
        mapped = self.displacement_operator.mv(state_)
        pulled = self.effort_pullback(effort)
        primal = self.contact_effort_space.pair(effort, mapped)
        transpose = self.effort_space.pair(pulled, state_)
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
        for feature_slice, kind_name in (
            ("vertex_slice", "vertex"),
            ("edge_slice", "edge"),
            ("face_slice", "face"),
        ):
            identifiers = np.concatenate(
                tuple(
                    np.asarray(surface.plan.feature_policy.feature_ids)[
                        getattr(surface.plan.feature_policy, feature_slice)
                    ]
                    for surface in values
                )
            )
            if np.unique(identifiers).size != identifiers.size:
                raise ValueError(
                    f"Collision scene {kind_name} feature IDs must be globally unique."
                )
        reference_pair_policy = values[0].plan.pair_policy
        for value in values[1:]:
            candidate = value.plan.pair_policy
            if (
                candidate.unrestricted != reference_pair_policy.unrestricted
                or not np.array_equal(
                    np.asarray(candidate.allowed_participant_pairs),
                    np.asarray(reference_pair_policy.allowed_participant_pairs),
                )
            ):
                raise ValueError(
                    "Collision scene participant-pair policies must agree reciprocally."
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

    def _feature_values(self, name: str, feature_slice: str, /) -> Array:
        return jnp.concatenate(
            tuple(
                getattr(surface.plan.feature_policy, name)[
                    getattr(surface.plan.feature_policy, feature_slice)
                ]
                for surface in self.surfaces
            )
        )

    @property
    def vertex_feature_ids(self) -> Array:
        return self._feature_values("feature_ids", "vertex_slice")

    @property
    def edge_feature_ids(self) -> Array:
        return self._feature_values("feature_ids", "edge_slice")

    @property
    def face_feature_ids(self) -> Array:
        return self._feature_values("feature_ids", "face_slice")

    @property
    def feature_ids(self) -> Array:
        return jnp.concatenate(
            (
                self.vertex_feature_ids,
                self.edge_feature_ids,
                self.face_feature_ids,
            )
        )

    @property
    def vertex_participant_ids(self) -> Array:
        return self._feature_values("participant_ids", "vertex_slice")

    @property
    def edge_participant_ids(self) -> Array:
        return self._feature_values("participant_ids", "edge_slice")

    @property
    def face_participant_ids(self) -> Array:
        return self._feature_values("participant_ids", "face_slice")

    @property
    def feature_participant_ids(self) -> Array:
        return jnp.concatenate(
            (
                self.vertex_participant_ids,
                self.edge_participant_ids,
                self.face_participant_ids,
            )
        )

    @property
    def vertex_body_ids(self) -> Array:
        return self._feature_values("body_ids", "vertex_slice")

    @property
    def feature_body_ids(self) -> Array:
        return jnp.concatenate(
            (
                self.vertex_body_ids,
                self._feature_values("body_ids", "edge_slice"),
                self._feature_values("body_ids", "face_slice"),
            )
        )

    @property
    def vertex_material_ids(self) -> Array:
        return self._feature_values("material_ids", "vertex_slice")

    @property
    def feature_material_ids(self) -> Array:
        return jnp.concatenate(
            (
                self.vertex_material_ids,
                self._feature_values("material_ids", "edge_slice"),
                self._feature_values("material_ids", "face_slice"),
            )
        )

    @property
    def vertex_patch_ids(self) -> Array:
        return self._feature_values("patch_ids", "vertex_slice")

    @property
    def feature_patch_ids(self) -> Array:
        return jnp.concatenate(
            (
                self.vertex_patch_ids,
                self._feature_values("patch_ids", "edge_slice"),
                self._feature_values("patch_ids", "face_slice"),
            )
        )

    @property
    def vertex_static_mask(self) -> Array:
        return self._feature_values("static_mask", "vertex_slice")

    @property
    def feature_static_mask(self) -> Array:
        return jnp.concatenate(
            (
                self.vertex_static_mask,
                self._feature_values("static_mask", "edge_slice"),
                self._feature_values("static_mask", "face_slice"),
            )
        )

    @property
    def feature_physical_radius(self) -> Array:
        return jnp.concatenate(
            (
                self._feature_values("physical_radius", "vertex_slice"),
                self._feature_values("physical_radius", "edge_slice"),
                self._feature_values("physical_radius", "face_slice"),
            )
        )

    @property
    def feature_solver_clearance(self) -> Array:
        return jnp.concatenate(
            (
                self._feature_values("solver_clearance", "vertex_slice"),
                self._feature_values("solver_clearance", "edge_slice"),
                self._feature_values("solver_clearance", "face_slice"),
            )
        )

    @property
    def feature_proxy_error(self) -> Array:
        return jnp.concatenate(
            (
                self._feature_values("proxy_error", "vertex_slice"),
                self._feature_values("proxy_error", "edge_slice"),
                self._feature_values("proxy_error", "face_slice"),
            )
        )

    @property
    def feature_contact_extent(self) -> Array:
        return (
            self.feature_physical_radius
            + self.feature_solver_clearance
            + self.feature_proxy_error
        )

    @property
    def pair_policy(self) -> ContactPairPolicy:
        return self.surfaces[0].plan.pair_policy

    def map_values(self, state: PyTree[Any], /) -> Array:
        """Map a shared mechanics-space vector to concatenated surface values."""
        return jnp.concatenate(
            tuple(value.map_values(state) for value in self.surfaces), axis=0
        )

    def positions(self, state: PyTree[Any], /) -> Array:
        return jnp.concatenate(
            tuple(value.positions(state) for value in self.surfaces), axis=0
        )

    def effort_pullback(self, scene_effort: ArrayLike, /) -> PyTree[Array]:
        effort = jnp.asarray(
            scene_effort, dtype=self.surfaces[0].precision.geometry_dtype
        )
        expected = (self.vertex_count, self.ambient_dimension)
        if effort.shape != expected:
            raise ValueError(f"scene_effort must have shape {expected}.")
        contributions = tuple(
            surface.effort_pullback(
                effort[self.vertex_offsets[index] : self.vertex_offsets[index + 1]]
            )
            for index, surface in enumerate(self.surfaces)
        )
        return DualSpace(self.source_space).validate(
            jax.tree.map(lambda *values: sum(values), *contributions)
        )


def prepare_cell_mesh_collision_surface(
    mesh: CellMesh,
    source_space: ArraySpace,
    /,
    *,
    participant_id: int = 0,
    body_id: int = 0,
    material_id: int | None = None,
    patch_id: int = 0,
    static: bool = False,
    physical_radius: float = 0.0,
    solver_clearance: float = 0.0,
    proxy_error: float = 0.0,
    feature_provenance_id: str | None = None,
    pair_policy: ContactPairPolicy | None = None,
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
    policy = ContactPairPolicy(count) if pair_policy is None else pair_policy
    plan = CollisionSurfacePlan(
        np.asarray(mesh.vertex_global_ids)[boundary_vertices],
        ambient_dimension=mesh.ambient_dimension,
        edges=edges,
        faces=faces,
        pair_policy=policy,
        participant_ids=int(participant_id),
        body_ids=int(body_id),
        material_ids=(int(body_id) if material_id is None else int(material_id)),
        patch_ids=int(patch_id),
        static_mask=bool(static),
        physical_radius=physical_radius,
        solver_clearance=solver_clearance,
        proxy_error=proxy_error,
        feature_provenance_id=feature_provenance_id,
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
    "CollisionFeatureKind",
    "CollisionFeaturePolicy",
    "CollisionMapEvidence",
    "CollisionSurfacePlan",
    "ContactPairPolicy",
    "PreparedCollisionScene",
    "PreparedCollisionSurface",
    "prepare_cell_mesh_collision_surface",
    "selection_collision_operator",
    "static_collision_operator",
]
