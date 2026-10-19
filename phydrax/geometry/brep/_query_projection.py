#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Occurrence-qualified B-Rep projection and solid-location publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.geometry.brep._stationary_isolation as stationary_isolation_owner

from ..._fingerprint import canonical_fingerprint
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._planar_embedding import PlanarEmbedding
from ._coordinate_request import coordinate_source_request
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._projection_contracts import (
    AbstractBRepProjection,
    BRepEntityDimension,
    BRepProjectionPolicy,
    BRepProjectionResult,
    BRepProjectionStatus,
    closure_codes,
)
from ._query import (
    BRepQueryPolicy,
    BRepSurfaceQueryResult,
    certified_edge_query,
    PreparedBRepQuery,
)
from ._query_types import BRepQueryBudget, EdgeHit, LowerEntityHit, ProjectionHit
from ._request_preparations import TrimRequestPreparations


_UNIQUE = int(BRepProjectionStatus.UNIQUE)
_AMBIGUOUS = int(BRepProjectionStatus.AMBIGUOUS)
_FAILED = int(BRepProjectionStatus.FAILED)


@eqx.filter_jit
def _place_surface_result(
    result: BRepSurfaceQueryResult, rotation: Array, translation: Array, placement: int, /
) -> BRepSurfaceQueryResult:
    return eqx.tree_at(
        lambda value: (
            value.points,
            value.normals,
            value.first_derivatives,
            value.second_derivatives,
            value.occurrences,
        ),
        result,
        (
            result.points @ rotation.T + translation,
            result.normals @ rotation.T,
            result.first_derivatives @ rotation.T,
            result.second_derivatives @ rotation.T,
            jnp.full(result.distances.shape, placement, dtype=jnp.int32),
        ),
    )


@eqx.filter_jit
def _merge_surface_results(
    hits: tuple[BRepSurfaceQueryResult, ...],
    namespaces: tuple[Array, ...],
    tolerance: float,
    /,
) -> tuple[BRepSurfaceQueryResult, Array]:
    distance = jnp.stack([hit.distances for hit in hits])
    best = jnp.argmin(distance, axis=0)
    rows = jnp.arange(hits[0].distances.shape[0])
    result = jax.tree_util.tree_map(lambda *values: jnp.stack(values)[best, rows], *hits)
    selected_path = jnp.stack(namespaces)[best, rows]
    lower = jnp.min(jnp.stack([hit.distance_lower_bounds for hit in hits]), axis=0)
    unresolved = result.unresolved | (result.distances - lower > tolerance)
    tied_path = jnp.any(
        (jnp.stack(namespaces) != selected_path[None])
        & (distance <= result.distances[None] + tolerance),
        axis=0,
    )
    status = jnp.where(
        unresolved, _FAILED, jnp.where(tied_path, _AMBIGUOUS, result.status)
    ).astype(jnp.int8)
    return eqx.tree_at(
        lambda value: (value.status, value.unresolved, value.distance_lower_bounds),
        result,
        (status, unresolved, lower),
    ), selected_path


@dataclass(frozen=True, slots=True)
class _HostLowerEntityHit:
    """Host projection fields shared by vertex and edge strategies."""

    point: np.ndarray
    distance: np.ndarray
    parameter: np.ndarray
    tangent: np.ndarray
    status: np.ndarray
    lower_bound: np.ndarray
    occurrence: np.ndarray

    @classmethod
    def from_edge(cls, hit: EdgeHit, namespace: int, /) -> _HostLowerEntityHit:
        point, distance, parameter, tangent, status, lower = jax.device_get(
            (
                hit.point,
                hit.distance,
                hit.parameter,
                hit.tangent,
                hit.status,
                hit.lower_bound,
            )
        )
        return cls(
            point=point,
            distance=distance,
            parameter=parameter,
            tangent=tangent,
            status=status,
            lower_bound=lower,
            occurrence=np.full(distance.shape, namespace, dtype=np.int32),
        )


def _merge_lower_entity_hits(
    hits: list[_HostLowerEntityHit], tolerance: float, /
) -> LowerEntityHit:
    distance = np.stack([hit.distance for hit in hits])
    best = np.argmin(distance, axis=0)
    rows = np.arange(distance.shape[1])
    points = np.stack([hit.point for hit in hits])
    statuses = np.stack([hit.status for hit in hits])
    occurrences = np.stack([hit.occurrence for hit in hits])
    selected_point, selected_distance = points[best, rows], distance[best, rows]
    selected_occurrence = occurrences[best, rows]
    lower_bounds = np.stack([hit.lower_bound for hit in hits])
    lower = np.min(lower_bounds, axis=0)
    different = np.linalg.norm(points - selected_point[None], axis=-1) > tolerance
    distinct_occurrence = occurrences != selected_occurrence[None]
    tied = np.any(
        (different | distinct_occurrence)
        & (distance <= selected_distance[None] + tolerance),
        axis=0,
    )
    unresolved = selected_distance - lower > tolerance
    return LowerEntityHit(
        point=jnp.asarray(selected_point),
        distance=jnp.asarray(selected_distance),
        parameter=jnp.asarray(np.stack([hit.parameter for hit in hits])[best, rows]),
        tangent=jnp.asarray(np.stack([hit.tangent for hit in hits])[best, rows]),
        status=jnp.asarray(
            np.where(
                unresolved, _FAILED, np.where(tied, _AMBIGUOUS, statuses[best, rows])
            ),
            dtype=jnp.int8,
        ),
        lower_bound=jnp.asarray(lower_bounds[best, rows]),
        occurrence=jnp.asarray(selected_occurrence),
    )


def _group_size(count: int, /) -> int:
    """Power-of-two padding bounds recompilation across batch sizes."""
    return 1 << max(0, (count - 1).bit_length())


@dataclass(frozen=True, slots=True)
class _QualifiedProjection:
    paths: tuple[tuple[str, ...], ...]
    entity_paths: tuple[tuple[int, ...], ...]
    placements: tuple[tuple[int, ...], ...]
    codes: np.ndarray
    boxes: np.ndarray
    closure: np.ndarray
    world_boxes: np.ndarray


def _qualified_projection(
    geometry: BRepGeometry, query: PreparedBRepQuery, definition_boxes: np.ndarray, /
) -> _QualifiedProjection:
    records = geometry.qualified_entity_incidence(query.source_revision)
    identities = {record.container for record in records} | {
        record.member for record in records
    }
    paths = tuple(
        sorted(
            {
                identity.occurrence_path
                for identity in identities
                if identity.occurrence_path
            }
        )
    )
    path_indices = {path: index for index, path in enumerate(paths)}
    counts = (
        geometry.vertex_points.shape[0],
        len(geometry.edge_curves),
        len(geometry.face_loops),
        len(geometry.solid_shells),
    )
    offsets = np.cumsum((0,) + counts)
    total = sum(counts)
    qualified_total = total * (len(paths) + 1)

    def code(identity: BRepEntityId) -> int:
        dimension = BRepEntityDimension[identity.kind.upper()].value
        namespace = (
            path_indices[identity.occurrence_path] + 1 if identity.occurrence_path else 0
        )
        return int(offsets[dimension]) + identity.index + namespace * total

    adjacency: dict[int, set[int]] = {}
    for record in records:
        adjacency.setdefault(code(record.container), set()).add(code(record.member))
    # Definition closure and occurrence-qualified closure are distinct
    # namespaces. The authored occurrence graph need not repeat unplaced
    # definitions, but public definition queries retain their source topology.
    definition_closure = closure_codes(
        geometry.topology(),
        np.asarray(geometry.edge_vertices, dtype=np.int64).reshape(-1, 2),
        counts,
    )
    codes = sorted({code(identity) for identity in identities})
    reach = {divmod(int(relation), total) for relation in definition_closure}
    for container in codes:
        pending, seen = [container], set()
        while pending:
            member = pending.pop()
            if member in seen:
                continue
            seen.add(member)
            reach.add((container, member))
            pending.extend(adjacency.get(member, ()))
    solid_codes = tuple(
        code(
            BRepEntityId(
                query.source_revision, "solid", occurrence.solid, occurrence.path
            )
        )
        for occurrence in geometry.occurrences
    )
    placements, boxes = [], []
    entity_paths: list[list[int]] = [[] for _ in range(total)]
    world_boxes = np.full(definition_boxes.shape, np.inf, dtype=np.float64)
    world_boxes[:, 1] = -np.inf
    rotations, translations = jax.device_get(
        (query.occurrence_rotations, query.occurrence_translations)
    )
    for qualified in codes:
        base, namespace = qualified % total, qualified // total
        indices = (
            (-1,)
            if namespace == 0
            else tuple(
                index
                for index, solid in enumerate(solid_codes)
                if (solid, qualified) in reach
            )
        )
        if not indices:
            raise ValueError(
                f"NativeBRepProjection.qualified_entity must have physical placement; got entity code {qualified} with none"
            )
        placed = []
        for index in indices:
            rotation = np.eye(3, dtype=np.float64) if index < 0 else rotations[index]
            translation = (
                np.zeros((3,), dtype=np.float64) if index < 0 else translations[index]
            )
            products = np.asarray(rotation) * definition_boxes[base, :, None, :]
            box = np.stack(
                (
                    np.sum(np.min(products, axis=0), axis=-1),
                    np.sum(np.max(products, axis=0), axis=-1),
                )
            ) + np.asarray(translation)
            margin = (
                256.0
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    float(np.max(np.abs(definition_boxes[base]))),
                    float(np.max(np.abs(box))),
                )
            )
            placed.append(np.stack((box[0] - margin, box[1] + margin)))
        placed_boxes = np.stack(placed)
        box = np.stack(
            (np.min(placed_boxes[:, 0], axis=0), np.max(placed_boxes[:, 1], axis=0))
        )
        boxes.append(box)
        placements.append(indices)
        entity_paths[base].append(namespace - 1)
        world_boxes[base, 0] = np.minimum(world_boxes[base, 0], box[0])
        world_boxes[base, 1] = np.maximum(world_boxes[base, 1], box[1])
    return _QualifiedProjection(
        paths=paths,
        entity_paths=tuple(tuple(indices) for indices in entity_paths),
        placements=tuple(placements),
        codes=np.asarray(codes, dtype=np.int64),
        boxes=np.stack(boxes),
        closure=np.asarray(
            sorted(container * qualified_total + member for container, member in reach),
            dtype=np.int64,
        ),
        world_boxes=world_boxes,
    )


@final
class NativeBRepProjection(AbstractBRepProjection):
    """Native closest-point, classification and solid location of one revision.

    Construct with :func:`prepare_brep_projection` from a model that carries
    native curves, p-curves, and trim loops. Entity indices are the model's ``BRepEntityId``
    indices; a planar ``embedding`` binds face-only planar revisions to
    two-dimensional coordinates.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    policy: BRepProjectionPolicy
    embedding: PlanarEmbedding | None = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    entity_counts: tuple[int, int, int, int] = eqx.field(static=True)
    closure_codes: Array
    entity_boxes: Array
    edge_degenerate: Array
    edge_vertices: Array
    edge_closed: Array
    vertex_points: Array
    query: PreparedBRepQuery
    projection_id: str = eqx.field(static=True)
    entity_occurrences: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    qualified_entity_codes: Array
    qualified_entity_boxes: Array
    qualified_closure_codes: Array
    qualified_placements: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativeBRepProjection], value: NativeBRepProjection, /
    ) -> NativeBRepProjection:
        """Rebuild projection metadata through its model and policy preparation."""
        rebuilt = cls(
            value.query.model, value.policy, value.embedding, value.query.policy
        )
        if (
            rebuilt.projection_id != value.projection_id
            or rebuilt.model_id != value.model_id
            or rebuilt.source_revision != value.source_revision
        ):
            raise ValueError(
                "NativeBRepProjection.stored_fields must preserve projection and source identity; "
                f"got projection={value.projection_id!r}, model={value.model_id!r}, revision={value.source_revision!r}."
            )
        return rebuilt

    @checked
    def __init__(
        self,
        model: BRepModel,
        policy: BRepProjectionPolicy,
        embedding: PlanarEmbedding | None,
        query_policy: BRepQueryPolicy,
        /,
    ) -> None:
        geometry = model.require_geometry()
        query = PreparedBRepQuery(model, query_policy, policy)
        topology = model.topology
        points = np.asarray(geometry.vertex_points)
        edge_vertices = np.asarray(geometry.edge_vertices, dtype=np.int64).reshape(-1, 2)
        counts = (
            topology.num_vertices,
            topology.num_edges,
            topology.num_faces,
            topology.num_solids,
        )
        face_boxes = np.asarray(query.face_boxes)
        solid_boxes = np.asarray(
            [
                np.stack(
                    (
                        np.min(face_boxes[list(faces), 0], axis=0),
                        np.max(face_boxes[list(faces), 1], axis=0),
                    )
                )
                for faces in topology.solid_faces
            ],
            dtype=np.float64,
        ).reshape(-1, 2, 3)
        ambient = 3
        if embedding is not None:
            if topology.num_solids:
                raise ValueError(
                    f"NativeBRepProjection.embedding must bind face-only revisions; got {topology.num_solids} solids"
                )
            if (
                np.max(np.abs(embedding.plane_residual(points)))
                > policy.classifier_tolerance
            ):
                raise ValueError(
                    f"NativeBRepProjection.embedding must contain revision vertices within classifier_tolerance; got plane residual above {policy.classifier_tolerance}"
                )
            ambient = 2
        self.source_id = model.source_id
        self.source_revision = model.source_revision
        self.model_id = model.model_id
        self.policy = policy
        self.embedding = embedding
        self.ambient_dimension = ambient
        self.entity_counts = counts
        self.closure_codes = jnp.asarray(closure_codes(topology, edge_vertices, counts))
        definition_boxes = np.concatenate(
            (
                np.stack((points, points), axis=1),
                np.asarray(query.edge_boxes),
                face_boxes,
                solid_boxes,
            )
        )
        qualified = _qualified_projection(geometry, query, definition_boxes)
        self.entity_boxes = jnp.asarray(qualified.world_boxes, dtype=jnp.float64)
        self.entity_occurrences = qualified.entity_paths
        self.occurrence_paths = qualified.paths
        self.qualified_entity_codes = jnp.asarray(qualified.codes, dtype=jnp.int64)
        self.qualified_entity_boxes = jnp.asarray(qualified.boxes, dtype=jnp.float64)
        self.qualified_closure_codes = jnp.asarray(qualified.closure, dtype=jnp.int64)
        self.qualified_placements = qualified.placements
        self.edge_degenerate = jnp.asarray(
            np.asarray(geometry.degenerate_edges, dtype=np.bool_)
        )
        self.edge_vertices = jnp.asarray(edge_vertices.astype(np.int32))
        self.edge_closed = jnp.asarray(
            np.asarray([edge.closed for edge in query.edges], dtype=np.bool_)
        )
        self.vertex_points = jnp.asarray(points)
        self.query = query
        self.projection_id = canonical_fingerprint(
            {
                "kind": "native-brep-projection",
                "source_revision": model.source_revision,
                "model_id": model.model_id,
                "policy": policy.policy_id,
                "query": query.query_id,
                "embedding": None if embedding is None else embedding.embedding_id,
            }
        )

    def _entity_targets(
        self, dimension: int, entity: int, namespace: int | None, /
    ) -> tuple[tuple[int, int], ...]:
        base = sum(self.entity_counts[:dimension]) + entity
        paths = self.entity_occurrences[base] if namespace is None else (namespace,)
        published = np.asarray(self.qualified_entity_codes)
        targets = []
        for path in paths:
            code = base + (path + 1) * sum(self.entity_counts)
            position = int(np.searchsorted(published, code))
            if position >= published.size or published[position] != code:
                raise ValueError(
                    f"NativeBRepProjection.entity must have authored identity at the requested occurrence path; got entity code {code} without identity"
                )
            targets.extend(
                (path, placement) for placement in self.qualified_placements[position]
            )
        return tuple(targets)

    def _placed_lower_entity(
        self,
        points: Array,
        dimension: BRepEntityDimension,
        entity: int,
        /,
        occurrence: int | None = None,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> LowerEntityHit:
        occurrences = self._entity_targets(dimension, entity, occurrence)
        if not occurrences:
            raise ValueError(
                "NativeBRepProjection.entity must have placed occurrences; got none"
            )
        host_points = np.asarray(jax.device_get(points))
        rotations, translations = jax.device_get(
            (self.query.occurrence_rotations, self.query.occurrence_translations)
        )
        hits: list[_HostLowerEntityHit] = []
        for namespace, index in occurrences:
            world = self.query.world_query
            if world is not None:
                geometry = world.model.require_geometry()
                path = () if index < 0 else self.query.occurrences[index].path
                match dimension:
                    case BRepEntityDimension.VERTEX:
                        kind, sources = "vertex", self.query.world_vertex_sources
                    case BRepEntityDimension.EDGE:
                        kind, sources = "edge", self.query.world_edge_sources
                    case _:
                        raise ValueError(
                            f"NativeBRepProjection.dimension must select a vertex or edge; got {dimension}"
                        )
                candidates = {
                    record.member.index
                    for record in geometry.qualified_entity_incidence(
                        world.source_revision
                    )
                    if record.member.kind == kind
                    and record.member.occurrence_path == path
                    and sources[record.member.index] == entity
                }
                if len(candidates) != 1:
                    raise RuntimeError(
                        "NativeBRepProjection.world_entity must have unique qualified source incidence; got ambiguous incidence"
                    )
                target = candidates.pop()
                if kind == "edge":
                    edge_hit = certified_edge_query(
                        world.edges[target],
                        points,
                        world.tolerances,
                        world.policy,
                        endpoint_roots=geometry.edge_endpoint_roots[target],
                        preparations=preparations,
                        budget=budget,
                    )
                    hit = _HostLowerEntityHit.from_edge(edge_hit, namespace)
                else:
                    vertex = np.asarray(world.vertex_points[target])
                    projected = np.broadcast_to(vertex, host_points.shape)
                    root = geometry.vertex_roots[target]
                    enclosure = (
                        np.stack((vertex, vertex))
                        if root is None
                        else root.point_enclosure(
                            preparations=preparations, budget=budget
                        )
                    )
                    lower = np.asarray(
                        [
                            stationary_isolation_owner.source_distance_bounds(
                                enclosure, point
                            )[0]
                            for point in host_points
                        ],
                        dtype=np.float64,
                    )
                    distance = np.linalg.norm(host_points - projected, axis=-1)
                    hit = _HostLowerEntityHit(
                        point=projected,
                        distance=distance,
                        parameter=np.full(distance.shape, np.nan, dtype=np.float64),
                        tangent=np.full(host_points.shape, np.nan, dtype=np.float64),
                        status=np.where(
                            distance - lower <= self.policy.ambiguity_tolerance,
                            _UNIQUE,
                            _FAILED,
                        ).astype(np.int8),
                        lower_bound=lower,
                        occurrence=np.full(distance.shape, namespace, dtype=np.int32),
                    )
                hits.append(hit)
                continue
            rotation = np.eye(3, dtype=np.float64) if index < 0 else rotations[index]
            translation = (
                np.zeros((3,), dtype=np.float64) if index < 0 else translations[index]
            )
            local = (host_points - translation) @ rotation
            match dimension:
                case BRepEntityDimension.VERTEX:
                    projected = np.broadcast_to(
                        np.asarray(self.vertex_points[entity]), host_points.shape
                    )
                    distance = np.linalg.norm(local - projected, axis=-1)
                    hit = _HostLowerEntityHit(
                        point=projected,
                        distance=distance,
                        parameter=np.full(distance.shape, np.nan, dtype=np.float64),
                        tangent=np.full(host_points.shape, np.nan, dtype=np.float64),
                        status=np.full(distance.shape, _UNIQUE, dtype=np.int8),
                        lower_bound=distance,
                        occurrence=np.full(distance.shape, namespace, dtype=np.int32),
                    )
                case BRepEntityDimension.EDGE:
                    geometry = self.query.model.require_geometry()
                    edge_hit = certified_edge_query(
                        self.query.edges[entity],
                        jnp.asarray(local),
                        self.query.tolerances,
                        self.query.policy,
                        endpoint_roots=geometry.edge_endpoint_roots[entity],
                        preparations=preparations,
                        budget=budget,
                    )
                    hit = _HostLowerEntityHit.from_edge(edge_hit, namespace)
                case _:
                    raise ValueError(
                        f"NativeBRepProjection.dimension must select a vertex or edge; got {dimension}"
                    )
            hits.append(
                _HostLowerEntityHit(
                    point=hit.point @ rotation.T + translation,
                    distance=hit.distance,
                    parameter=hit.parameter,
                    tangent=hit.tangent @ rotation.T,
                    status=hit.status,
                    lower_bound=hit.lower_bound,
                    occurrence=hit.occurrence,
                )
            )
        return _merge_lower_entity_hits(hits, self.policy.ambiguity_tolerance)

    def _placed_face(
        self,
        points: Array,
        entity: int,
        occurrence: int | None,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[BRepSurfaceQueryResult, Array]:
        hits, namespaces = [], []
        host_points = np.asarray(jax.device_get(points))
        rotations, translations = jax.device_get(
            (self.query.occurrence_rotations, self.query.occurrence_translations)
        )
        for namespace, placement in self._entity_targets(2, entity, occurrence):
            if self.query.world_query is not None:
                result = self.query.world_closest(
                    points,
                    (entity,),
                    placement,
                    preparations=preparations,
                    coordinate_budget=budget,
                )
            else:
                rotation = (
                    np.eye(3, dtype=np.float64) if placement < 0 else rotations[placement]
                )
                translation = (
                    np.zeros((3,), dtype=np.float64)
                    if placement < 0
                    else translations[placement]
                )
                preparation = self.query.prepare_placed_query(
                    placement, (entity,), budget=budget
                )
                local = (
                    points
                    if preparation is not None
                    else jnp.asarray((host_points - translation) @ rotation)
                )
                result = self.query.closest_definition(
                    local,
                    (entity,),
                    affine_preparation=preparation,
                    preparations=preparations,
                    coordinate_budget=budget,
                )
                if preparation is not None:
                    rotation, translation = (
                        np.eye(3, dtype=np.float64),
                        np.zeros((3,), dtype=np.float64),
                    )
                result = _place_surface_result(
                    result, jnp.asarray(rotation), jnp.asarray(translation), placement
                )
            hits.append(result)
            namespaces.append(
                jnp.full(result.distances.shape, namespace, dtype=jnp.int32)
            )
        return _merge_surface_results(
            tuple(hits), tuple(namespaces), self.policy.ambiguity_tolerance
        )

    def project(
        self,
        points: ArrayLike,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> BRepProjectionResult:
        """Closest point of each query on its own vertex, edge, or face."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    points,
                    dimensions,
                    indices,
                )
            ) as budget:
                return self.project(
                    points,
                    dimensions,
                    indices,
                    occurrence_paths=occurrence_paths,
                    preparations=preparations,
                    budget=budget,
                )
        queries = self._queries(points, "Projection queries")
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError(
                f"NativeBRepProjection.entity_targets must use integer dimensions and indices; got dtypes {(dims.dtype, rows.dtype)}"
            )
        if dims.shape != (queries.shape[0],) or rows.shape != dims.shape:
            raise ValueError(
                f"NativeBRepProjection.entity_targets must supply one dimension and index per query; got shapes {(dims.shape, rows.shape)} for {queries.shape[0]} queries"
            )
        if np.any((dims < 0) | (dims > 2)):
            raise ValueError(
                "NativeBRepProjection.dimensions must select vertices, edges, or faces; got dimensions outside [0, 2]"
            )
        limits = np.asarray(self.entity_counts[:3])[dims]
        if np.any((rows < 0) | (rows >= limits)):
            raise ValueError(
                "NativeBRepProjection.indices must lie within their entity dimensions; got out-of-range indices"
            )
        world = self._world(queries)
        count = queries.shape[0]
        closest = np.full((count, 3), np.nan, dtype=np.float64)
        parameters = np.full((count, 2), np.nan, dtype=np.float64)
        residuals = np.full((count,), np.inf, dtype=np.float64)
        status = np.full((count,), _FAILED, dtype=np.int8)
        normals = np.full((count, 3), np.nan, dtype=np.float64)
        first = np.full((count, 3), np.nan, dtype=np.float64)
        second = np.full((count, 3), np.nan, dtype=np.float64)
        occurrences = np.full((count,), -1, dtype=np.int32)
        codes = self.entity_codes(dims, rows, occurrence_paths=occurrence_paths)
        order = np.argsort(codes, kind="stable")
        groups, starts = np.unique(codes[order], return_index=True)
        boundaries = np.append(starts, order.size)
        for code, start, stop in zip(
            groups.tolist(), boundaries[:-1], boundaries[1:], strict=True
        ):
            dimensions_, indices_ = self.code_entities(
                np.asarray((code,), dtype=np.int64)
            )
            dimension, entity = BRepEntityDimension(int(dimensions_[0])), int(indices_[0])
            occurrence = (
                None if occurrence_paths is None else code // sum(self.entity_counts) - 1
            )
            selected = order[start:stop]
            size = _group_size(selected.size)
            batch = jax.device_put(
                np.concatenate(
                    (
                        world[selected],
                        np.broadcast_to(world[selected[:1]], (size - selected.size, 3)),
                    )
                )
            )
            match dimension:
                case BRepEntityDimension.VERTEX:
                    hit = self._placed_lower_entity(
                        batch,
                        dimension,
                        entity,
                        occurrence,
                        preparations=preparations,
                        budget=budget,
                    )
                case BRepEntityDimension.EDGE:
                    hit = self._placed_lower_entity(
                        batch,
                        dimension,
                        entity,
                        occurrence,
                        preparations=preparations,
                        budget=budget,
                    )
                    parameters[selected, 0] = np.asarray(hit.parameter)[: selected.size]
                    first[selected] = np.asarray(hit.tangent)[: selected.size]
                case BRepEntityDimension.FACE:
                    result, selected_paths = self._placed_face(
                        batch,
                        entity,
                        occurrence,
                        preparations=preparations,
                        budget=budget,
                    )
                    hit = ProjectionHit(
                        point=result.points,
                        distance=result.distances,
                        status=result.status,
                        occurrence=selected_paths,
                    )
                    parameters[selected] = np.asarray(result.parameters)[: selected.size]
                    normals[selected] = np.asarray(result.normals)[: selected.size]
                    first[selected] = np.asarray(result.first_derivatives)[
                        : selected.size
                    ]
                    second[selected] = np.asarray(result.second_derivatives)[
                        : selected.size
                    ]
                case _:
                    raise ValueError(
                        f"NativeBRepProjection.dimensions must select vertices, edges, or faces; got {dimension!r}"
                    )
            closest[selected] = np.asarray(hit.point)[: selected.size]
            residuals[selected] = np.asarray(hit.distance)[: selected.size]
            status[selected] = np.asarray(hit.status)[: selected.size]
            occurrences[selected] = np.asarray(hit.occurrence)[: selected.size]
        return self._result(
            dims,
            rows,
            queries,
            closest,
            parameters,
            residuals,
            status,
            normals,
            first,
            second,
            occurrences,
        )

    def _result(
        self,
        dimensions: np.ndarray,
        indices: np.ndarray,
        queries: np.ndarray,
        closest: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        status: np.ndarray,
        normals: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        occurrences: np.ndarray,
        /,
    ) -> BRepProjectionResult:
        count = dimensions.size
        tangents = np.full((count, 2, 3), np.nan, dtype=np.float64)
        face = (dimensions == BRepEntityDimension.FACE) & np.all(np.isfinite(normals), 1)
        first_norm = np.linalg.norm(first, axis=1)
        usable = np.isfinite(first_norm) & (first_norm > 0.0)
        tangents[usable, 0] = first[usable] / first_norm[usable, None]
        # At a pole the first derivative vanishes: seed the frame with the other.
        second_norm = np.linalg.norm(second, axis=1)
        swap = face & ~usable & np.isfinite(second_norm) & (second_norm > 0.0)
        tangents[swap, 0] = second[swap] / second_norm[swap, None]
        frame = face & np.all(np.isfinite(tangents[:, 0]), axis=1)
        tangents[frame, 0] -= (
            np.sum(tangents[frame, 0] * normals[frame], axis=1)[:, None] * normals[frame]
        )
        tangents[frame, 0] /= np.linalg.norm(tangents[frame, 0], axis=1)[:, None]
        tangents[frame, 1] = np.cross(normals[frame], tangents[frame, 0])
        found = np.all(np.isfinite(closest), axis=1)
        ambient_points = self._ambient(np.where(found[:, None], closest, 0.0))
        ambient_points[~found] = np.nan
        ambient_normals = (
            normals
            if self.embedding is None
            else np.full((count, 2), np.nan, dtype=np.float64)
        )
        return BRepProjectionResult(
            self.source_revision,
            dimensions,
            indices,
            queries,
            ambient_points,
            parameters,
            residuals,
            status,
            ambient_normals,
            self._directions(tangents),
            occurrence_indices=occurrences,
            occurrence_paths=self.occurrence_paths,
        )

    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> BRepProjectionResult:
        """Locate definition solids while retaining the selected occurrence path."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if coordinate_budget is None:
            with coordinate_source_request(
                (
                    self,
                    points,
                )
            ) as coordinate_budget:
                return self.locate_solids(
                    points,
                    occurrence_paths=occurrence_paths,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
        queries = self._queries(points, "Solid location queries")
        if self.entity_counts[3] == 0:
            raise ValueError("NativeBRepProjection.model must contain solids; got 0")
        solid_paths = tuple(occurrence.path for occurrence in self.query.occurrences)
        if occurrence_paths is not None:
            if len(occurrence_paths) != queries.shape[0] or any(
                path not in solid_paths for path in occurrence_paths
            ):
                raise ValueError(
                    "NativeBRepProjection.occurrence_paths must supply one authored solid path per query; got misaligned or unknown paths"
                )
        targets = tuple(
            (index, occurrence.solid)
            for index, occurrence in enumerate(self.query.occurrences)
        )
        if not targets:
            raise ValueError(
                "NativeBRepProjection.model must contain placed solid occurrences; got 0"
            )
        count = queries.shape[0]
        inside = np.zeros((count, len(targets)), dtype=np.bool_)
        unresolved = np.zeros((count, len(targets)), dtype=np.bool_)
        exhausted = np.zeros((count, len(targets)), dtype=np.bool_)
        world = self._world(queries)
        budget = BRepQueryBudget(
            self.query.policy.maximum_operations,
            self.query.policy.maximum_points,
            self.query.policy.maximum_scratch_bytes,
        )
        for column, (index, _solid) in enumerate(targets):
            path = self.query.occurrences[index].path
            active = (
                np.ones((count,), dtype=np.bool_)
                if occurrence_paths is None
                else np.asarray(
                    [value == path for value in occurrence_paths], dtype=np.bool_
                )
            )
            if not np.any(active):
                continue
            # Use the public occurrence route: its world-space affine preparation
            # preserves reflected coorientation and avoids lossy inverse placement
            # at large translations. All occurrences share one query allowance.
            result = self.query.contains(
                world[active],
                path=path,
                budget=budget,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
            inside[active, column] = np.asarray(result.inside)
            unresolved[active, column] = np.asarray(result.unresolved)
            exhausted[active, column] = np.asarray(result.resource_exhausted)
        hits = np.count_nonzero(inside, axis=1)
        best = np.argmax(inside, axis=1)
        solids = np.asarray([solid for _, solid in targets], dtype=np.int32)
        occurrences = np.asarray(
            [
                self.occurrence_paths.index(self.query.occurrences[index].path)
                for index, _ in targets
            ],
            dtype=np.int32,
        )
        selected = np.where(hits > 0, occurrences[best], -1)
        if occurrence_paths is not None:
            selected = np.where(
                hits > 0,
                selected,
                np.asarray(
                    [
                        self.occurrence_paths.index(path) if path else -1
                        for path in occurrence_paths
                    ],
                    dtype=np.int32,
                ),
            )
        state = np.where(
            np.any(exhausted, axis=1),
            _FAILED,
            np.where(
                np.any(unresolved, axis=1) | (hits > 1),
                _AMBIGUOUS,
                np.where(hits == 0, _FAILED, _UNIQUE),
            ),
        ).astype(np.int8)
        return BRepProjectionResult(
            self.source_revision,
            np.where(hits > 0, 3, -1).astype(np.int8),
            np.where(hits > 0, solids[best], -1),
            queries,
            np.where(hits[:, None] > 0, queries, np.nan),
            np.full((count, 2), np.nan, dtype=np.float64),
            np.where(hits > 0, 0.0, np.inf),
            state,
            np.full((count, self.ambient_dimension), np.nan, dtype=np.float64),
            np.full((count, 2, self.ambient_dimension), np.nan, dtype=np.float64),
            occurrence_indices=selected,
            occurrence_paths=self.occurrence_paths,
        )


__all__ = ["NativeBRepProjection"]
