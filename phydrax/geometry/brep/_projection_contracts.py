#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Engine-neutral B-Rep projection status, evidence, and query contracts.

These contracts bind closest-point and classification queries to one exact
B-Rep revision without depending on the geometry engine that answers them.
The entity closure relation is pure topology over ``BRepTopology`` and is
shared by every concrete projection.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from enum import IntEnum
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import bvh_overlap_pairs_host, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from .._planar_embedding import PlanarEmbedding
from ._model import BRepEntityId, BRepTopology


class BRepEntityDimension(IntEnum):
    """Topological dimension of one B-Rep entity kind."""

    VERTEX = 0
    EDGE = 1
    FACE = 2
    SOLID = 3


_KINDS = ("vertex", "edge", "face", "solid")


class BRepProjectionStatus(IntEnum):
    """Outcome of one closest-point or classification query.

    ``UNIQUE``: one closest point with unique parameters. ``SEAM``: one closest
    point whose parameters are not unique (periodic seam or singular pole); the
    reported parameters are the lexicographically lowest representative.
    ``AMBIGUOUS``: distinct closest points (or distinct entities) tie within the
    ambiguity tolerance, including continua such as a sphere center. ``FAILED``:
    the query engine produced no closest point, or no entity satisfies the query.
    """

    UNIQUE = 0
    SEAM = 1
    AMBIGUOUS = 2
    FAILED = 3


def brep_entity_id(
    source_revision: str,
    dimension: int,
    index: int,
    /,
    *,
    occurrence_path: tuple[str, ...] = (),
) -> str:
    """Encode exact definition identity and, when supplied, its scientific path."""
    if dimension not in range(len(_KINDS)):
        raise ValueError("dimension must identify a vertex, edge, face, or solid.")
    entity = BRepEntityId(source_revision, _KINDS[dimension], index, occurrence_path)
    definition = f"{entity.source_revision}:{entity.kind}:{entity.index}"
    if not entity.occurrence_path:
        return definition
    path = json.dumps(entity.occurrence_path, ensure_ascii=True, separators=(",", ":"))
    return f"{definition}:occurrence:{path}"


@final
class BRepProjectionPolicy(StrictModule, NonTrainableState):
    """Tie, seam, and trimmed-region tolerances of B-Rep closest-point queries.

    ``ambiguity_tolerance`` is absolute: closest points farther apart are distinct,
    and distinct minima whose distances differ by at most it tie.
    ``parametric_tolerance`` is relative to a periodic parameter's period (seam
    detection). ``classifier_tolerance`` is the absolute tolerance of the
    trimmed-face and solid classifiers.
    """

    ambiguity_tolerance: float = eqx.field(static=True)
    parametric_tolerance: float = eqx.field(static=True)
    classifier_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ambiguity_tolerance: float = 1.0e-9,
        parametric_tolerance: float = 1.0e-9,
        classifier_tolerance: float = 1.0e-7,
    ) -> None:
        ambiguity = positive_finite_float(ambiguity_tolerance, "ambiguity_tolerance")
        parametric = positive_finite_float(parametric_tolerance, "parametric_tolerance")
        classifier = positive_finite_float(classifier_tolerance, "classifier_tolerance")
        self.ambiguity_tolerance = ambiguity
        self.parametric_tolerance = parametric
        self.classifier_tolerance = classifier
        self.policy_id = canonical_fingerprint(
            {
                "kind": "brep-projection-policy",
                "ambiguity_tolerance": ambiguity,
                "parametric_tolerance": parametric,
                "classifier_tolerance": classifier,
            }
        )


@final
class BRepProjectionResult(StrictModule, NonTrainableState):
    """Closest points of query points on explicit B-Rep entities.

    Rows align with the queries; points, normals, and tangents are in the
    projection's ambient coordinates. ``parameters`` hold ``(u, v)`` on faces,
    ``(t, nan)`` on edges, and NaN on vertices and solids. ``normals`` are unit
    oriented face normals (NaN where undefined or not representable in a planar
    ambient); ``tangents[:, 0]`` is the unit edge tangent, ``tangents[:, :2]`` an
    orthonormal face tangent frame. ``dimensions == -1`` marks unclassified rows.
    """

    source_revision: str = eqx.field(static=True)
    dimensions: Array
    indices: Array
    queries: Array
    points: Array
    parameters: Array
    residuals: Array
    status: Array
    normals: Array
    tangents: Array
    result_id: str = eqx.field(static=True)
    occurrence_indices: Array
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        source_revision: str,
        dimensions: ArrayLike,
        indices: ArrayLike,
        queries: ArrayLike,
        points: ArrayLike,
        parameters: ArrayLike,
        residuals: ArrayLike,
        status: ArrayLike,
        normals: ArrayLike,
        tangents: ArrayLike,
        /,
        *,
        occurrence_indices: ArrayLike | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        revision = str(source_revision).strip()
        dims = np.asarray(dimensions, dtype=np.int8)
        rows = np.asarray(indices, dtype=np.int32)
        query = np.asarray(queries, dtype=np.float64)
        closest = np.asarray(points, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        distance = np.asarray(residuals, dtype=np.float64)
        state = np.asarray(status, dtype=np.int8)
        normal = np.asarray(normals, dtype=np.float64)
        tangent = np.asarray(tangents, dtype=np.float64)
        count = dims.shape[0]
        occurrences = (
            np.full(dims.shape, -1, dtype=np.int32)
            if occurrence_indices is None
            else np.asarray(occurrence_indices, dtype=np.int32)
        )
        paths = tuple(occurrence_paths)
        if any(
            not path or any(not isinstance(name, str) or not name for name in path)
            for path in paths
        ) or len(set(paths)) != len(paths):
            raise ValueError("Occurrence path tables must contain unique nonempty paths.")
        if occurrences.shape != dims.shape or np.any(
            (occurrences < -1) | (occurrences >= len(paths))
        ):
            raise ValueError(
                "Occurrence indices must be row-aligned and reference the path table or -1."
            )
        ambient = query.shape[-1] if query.ndim == 2 else -1
        if not revision:
            raise ValueError("source_revision must be non-empty.")
        if (
            dims.ndim != 1
            or ambient not in (2, 3)
            or rows.shape != dims.shape
            or distance.shape != dims.shape
            or state.shape != dims.shape
            or query.shape != (count, ambient)
            or closest.shape != (count, ambient)
            or params.shape != (count, 2)
            or normal.shape != (count, ambient)
            or tangent.shape != (count, 2, ambient)
        ):
            raise ValueError("B-Rep projection arrays must be row-aligned.")
        if not np.all(np.isin(state, [int(value) for value in BRepProjectionStatus])):
            raise ValueError("Projection status values must be BRepProjectionStatus.")
        if np.any((dims < -1) | (dims > 3)):
            raise ValueError("Projection entity dimensions must lie in [-1, 3].")
        self.source_revision = revision
        self.dimensions = jnp.asarray(dims)
        self.indices = jnp.asarray(rows)
        self.queries = jnp.asarray(query)
        self.points = jnp.asarray(closest)
        self.parameters = jnp.asarray(params)
        self.residuals = jnp.asarray(distance)
        self.status = jnp.asarray(state)
        self.normals = jnp.asarray(normal)
        self.tangents = jnp.asarray(tangent)
        self.occurrence_indices = jnp.asarray(occurrences, dtype=jnp.int32)
        self.occurrence_paths = paths
        self.result_id = canonical_fingerprint(
            {
                "kind": "brep-projection-result",
                "source_revision": revision,
                "dimensions": array_tree_fingerprint(dims),
                "indices": array_tree_fingerprint(rows),
                "queries": array_tree_fingerprint(query),
                "points": array_tree_fingerprint(closest),
                "parameters": array_tree_fingerprint(params),
                "status": array_tree_fingerprint(state),
                "occurrence_indices": array_tree_fingerprint(occurrences),
                "occurrence_paths": [list(path) for path in paths],
            }
        )

    @property
    def resolved(self) -> Array:
        """Rows with one closest point (parameters may be seam-duplicated)."""
        return (self.status == BRepProjectionStatus.UNIQUE) | (
            self.status == BRepProjectionStatus.SEAM
        )

    @property
    def source_occurrence_paths(self) -> tuple[tuple[str, ...], ...]:
        """Row-aligned scientific occurrence identity; ``()`` denotes the definition."""
        return tuple(
            self.occurrence_paths[index] if index >= 0 else ()
            for index in np.asarray(self.occurrence_indices).tolist()
        )

    def entity_ids(self) -> tuple[str, ...]:
        """Revision-qualified entity identities; empty for unclassified rows."""
        return tuple(
            brep_entity_id(self.source_revision, dimension, index, occurrence_path=path)
            if dimension >= 0
            else ""
            for dimension, index, path in zip(
                np.asarray(self.dimensions).tolist(),
                np.asarray(self.indices).tolist(),
                self.source_occurrence_paths,
                strict=True,
            )
        )


def _matches(sorted_keys: np.ndarray, queries: np.ndarray, /) -> Any:
    """``(query_row, position)`` of every sorted key equal to each query."""
    start = np.searchsorted(sorted_keys, queries, side="left")
    counts = np.searchsorted(sorted_keys, queries, side="right") - start
    owner = np.repeat(np.arange(queries.size), counts)
    position = np.repeat(start - (np.cumsum(counts) - counts), counts)
    return owner, position + np.arange(owner.size)


def _closure_codes(
    topology: BRepTopology,
    edge_vertices: np.ndarray,
    counts: tuple[int, int, int, int],
    /,
) -> np.ndarray:
    """Sorted codes ``container * total + member`` of the reflexive closure relation."""

    vertex_count, edge_count, face_count, solid_count = counts
    offsets = np.cumsum((0, vertex_count, edge_count, face_count))
    total = int(offsets[-1] + solid_count)
    valid = (edge_vertices >= 0).reshape(-1)
    edge_rows = np.repeat(np.arange(edge_count), 2)[valid]
    edge_vertex = np.stack((edge_rows, edge_vertices.reshape(-1)[valid]), axis=1)
    face_edge = np.asarray(
        [
            (face, edge)
            for face, edges in enumerate(topology.face_edges)
            for edge in edges
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    face_vertex = np.concatenate(
        [
            np.stack((face_edge[:, 0], edge_vertices[face_edge[:, 1], side]), axis=1)
            for side in (0, 1)
        ]
    )
    face_vertex = face_vertex[face_vertex[:, 1] >= 0]
    solid_face = np.asarray(
        [
            (solid, face)
            for solid, faces in enumerate(topology.solid_faces)
            for face in faces
        ],
        dtype=np.int64,
    ).reshape(-1, 2)

    def through(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
        """Compose (a, b) with (b, c) pairs into (a, c)."""
        order = np.argsort(second[:, 0], kind="stable")
        owner, position = _matches(second[order, 0], first[:, 1])
        return np.stack((first[owner, 0], second[order[position], 1]), axis=1)

    pairs = (
        np.stack((np.arange(total), np.arange(total)), axis=1),
        np.stack((offsets[1] + edge_vertex[:, 0], edge_vertex[:, 1]), axis=1),
        np.stack((offsets[2] + face_edge[:, 0], offsets[1] + face_edge[:, 1]), axis=1),
        np.stack((offsets[2] + face_vertex[:, 0], face_vertex[:, 1]), axis=1),
        np.stack((offsets[3] + solid_face[:, 0], offsets[2] + solid_face[:, 1]), axis=1),
        np.stack(
            (
                offsets[3] + through(solid_face, face_edge)[:, 0],
                offsets[1] + through(solid_face, face_edge)[:, 1],
            ),
            axis=1,
        ),
        np.stack(
            (
                offsets[3] + through(solid_face, face_vertex)[:, 0],
                through(solid_face, face_vertex)[:, 1],
            ),
            axis=1,
        ),
    )
    stacked = np.concatenate(pairs).astype(np.int64)
    return np.unique(stacked[:, 0] * total + stacked[:, 1])


class AbstractBRepProjection(StrictModule, NonTrainableState):
    """Closest-point, classification, and closure queries of one exact B-Rep revision.

    ``(dimension, index)`` pairs are the ``BRepEntityId`` identities of
    ``source_revision``. With a planar ``embedding`` queries and results use the
    embedding's two-dimensional frame. ``closure_codes`` is the sorted reflexive
    closure relation over dense entity codes (vertices, edges, faces, solids),
    ``entity_boxes`` the world-frame bounding boxes of those entities in code
    order, and ``edge_degenerate`` flags degenerate edges. Topology queries and
    tolerance classification are shared by every concrete projection; concrete
    projections supply the geometric closest-point and solid-location engines.
    """

    source_id: eqx.AbstractVar[str]
    source_revision: eqx.AbstractVar[str]
    model_id: eqx.AbstractVar[str]
    policy: eqx.AbstractVar[BRepProjectionPolicy]
    embedding: eqx.AbstractVar[PlanarEmbedding | None]
    ambient_dimension: eqx.AbstractVar[int]
    entity_counts: eqx.AbstractVar[tuple[int, int, int, int]]
    closure_codes: eqx.AbstractVar[Array]
    entity_boxes: eqx.AbstractVar[Array]
    edge_degenerate: eqx.AbstractVar[Array]
    projection_id: eqx.AbstractVar[str]
    occurrence_paths: eqx.AbstractVar[tuple[tuple[str, ...], ...]]
    entity_occurrences: eqx.AbstractVar[tuple[tuple[int, ...], ...]]
    qualified_entity_codes: eqx.AbstractVar[Array]
    qualified_entity_boxes: eqx.AbstractVar[Array]
    qualified_closure_codes: eqx.AbstractVar[Array]

    # -- topology ---------------------------------------------------------------

    @property
    def region_dimension(self) -> int:
        """Dimension of the highest entities present (solids, else faces)."""
        return 3 if self.entity_counts[3] else 2

    def entity_codes(
        self,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> np.ndarray:
        """Dense definition code with a separate occurrence namespace."""
        offsets = np.cumsum((0,) + self.entity_counts)
        base = offsets[np.asarray(dimensions, dtype=np.int64)] + np.asarray(
            indices, dtype=np.int64
        )
        if occurrence_paths is None:
            return base
        if len(occurrence_paths) != base.size:
            raise ValueError("One occurrence path is required per entity.")
        namespaces = np.zeros(base.shape, dtype=np.int64)
        for row, path in enumerate(occurrence_paths):
            if path:
                if path not in self.occurrence_paths:
                    raise ValueError(
                        "An occurrence path is not part of this source revision."
                    )
                index = self.occurrence_paths.index(path)
                if index not in self.entity_occurrences[int(base[row])]:
                    raise ValueError(
                        "The entity does not belong to the requested occurrence."
                    )
                namespaces[row] = index + 1
        return base + namespaces * sum(self.entity_counts)

    def code_entities(self, codes: ArrayLike, /) -> tuple[np.ndarray, np.ndarray]:
        """Decode definition dimension/index without discarding the code's namespace."""
        values = np.asarray(codes, dtype=np.int64) % sum(self.entity_counts)
        bounds = np.cumsum((0,) + self.entity_counts)
        dimensions = np.searchsorted(bounds, values, side="right") - 1
        return dimensions, values - bounds[dimensions]

    def code_occurrence_paths(self, codes: ArrayLike, /) -> tuple[tuple[str, ...], ...]:
        """Decode the scientific occurrence namespace of dense entity codes."""
        namespaces = np.asarray(codes, dtype=np.int64) // sum(self.entity_counts)
        if np.any((namespaces < 0) | (namespaces > len(self.occurrence_paths))):
            raise ValueError("Entity codes reference an absent occurrence namespace.")
        return tuple(
            self.occurrence_paths[index - 1] if index > 0 else ()
            for index in namespaces.tolist()
        )

    def contains(
        self,
        container_dimensions: ArrayLike,
        container_indices: ArrayLike,
        member_dimensions: ArrayLike,
        member_indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        member_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> np.ndarray:
        """Whether each member entity lies in the closure of its container entity."""
        containers = self.entity_codes(
            container_dimensions, container_indices, occurrence_paths=occurrence_paths
        )
        members = self.entity_codes(
            member_dimensions, member_indices, occurrence_paths=member_occurrence_paths
        )
        total = sum(self.entity_counts) * (len(self.occurrence_paths) + 1)
        codes = np.asarray(self.qualified_closure_codes)
        query = containers * total + members
        position = np.minimum(np.searchsorted(codes, query), codes.size - 1)
        return codes[position] == query

    def containers(
        self,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
        """Qualified containers, including authored shared-stratum path transitions."""
        total = sum(self.entity_counts) * (len(self.occurrence_paths) + 1)
        codes = np.asarray(self.qualified_closure_codes)
        order = np.argsort(codes % total, kind="stable")
        qualified = self.entity_codes(
            dimensions, indices, occurrence_paths=occurrence_paths
        )
        owner, position = _matches((codes % total)[order], qualified)
        found = codes[order[position]] // total
        found_dimensions, found_indices = self.code_entities(found)
        return owner, found_dimensions, found_indices, self.code_occurrence_paths(found)

    def members(
        self,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
        """Qualified closure members with their authored occurrence paths."""
        total = sum(self.entity_counts) * (len(self.occurrence_paths) + 1)
        codes = np.asarray(self.qualified_closure_codes)
        qualified = self.entity_codes(
            dimensions, indices, occurrence_paths=occurrence_paths
        )
        owner, position = _matches(codes // total, qualified)
        found = codes[position] % total
        found_dimensions, found_indices = self.code_entities(found)
        return owner, found_dimensions, found_indices, self.code_occurrence_paths(found)

    # -- coordinates ------------------------------------------------------------

    def _world(self, points: np.ndarray, /) -> np.ndarray:
        return points if self.embedding is None else self.embedding.to_world(points)

    def _ambient(self, points: np.ndarray, /) -> np.ndarray:
        if self.embedding is None:
            return points
        return (points - np.asarray(self.embedding.origin)) @ np.stack(
            (self.embedding.x_axis, self.embedding.y_axis), axis=1
        )

    def _directions(self, vectors: np.ndarray, /) -> np.ndarray:
        if self.embedding is None:
            return vectors
        return vectors @ np.stack((self.embedding.x_axis, self.embedding.y_axis), axis=1)

    def _queries(self, points: ArrayLike, name: str, /) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != self.ambient_dimension
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"{name} must be finite (n, {self.ambient_dimension}) ambient points."
            )
        return values

    # -- geometric queries ------------------------------------------------------

    @abstractmethod
    def project(
        self,
        points: ArrayLike,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Closest point of each query on its own vertex, edge, or face."""
        raise NotImplementedError

    def classify(
        self,
        points: ArrayLike,
        /,
        *,
        tolerance: float,
        maximum_dimension: int = 2,
        groups: ArrayLike | None = None,
        allowed_groups: ArrayLike | None = None,
        allowed_dimensions: ArrayLike | None = None,
        allowed_indices: ArrayLike | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        allowed_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Lowest-dimensional entity within ``tolerance`` of each point.

        Candidates are the entities of dimension at most ``maximum_dimension``
        (vertices, edges, faces; degenerate edges excluded) whose bounding boxes
        meet the tolerance box of the point (packed BVH overlap). Optional
        ``groups`` assign each query a group; ``allowed_*`` then lists the
        admissible ``(group, dimension, index)`` entities of each group, which
        is how classification transfer restricts candidates. Rows without a
        candidate within the tolerance are FAILED with dimension -1; distinct
        lowest-dimensional entities with distinct closest points are AMBIGUOUS.
        """
        queries = self._queries(points, "Classification queries")
        radius = positive_finite_float(tolerance, "tolerance")
        top = int(maximum_dimension)
        if top not in (0, 1, 2):
            raise ValueError("maximum_dimension must be 0, 1, or 2.")
        world = self._world(queries)
        all_codes = np.asarray(self.qualified_entity_codes)
        all_dimensions, _ = self.code_entities(all_codes)
        active = all_dimensions <= top
        candidate_codes = all_codes[active]
        boxes = np.asarray(self.qualified_entity_boxes)[active]
        first = prepare_bvh(world - radius, world + radius, dtype=jnp.float64)
        second = prepare_bvh(boxes[:, 0], boxes[:, 1], dtype=jnp.float64)
        query_rows, entity_rows = bvh_overlap_pairs_host(
            first, second, include_touching=True
        )
        pair_codes = candidate_codes[entity_rows]
        pair_dimensions, pair_indices = self.code_entities(pair_codes)
        pair_paths = self.code_occurrence_paths(pair_codes)
        edge_rows = np.where(pair_dimensions == BRepEntityDimension.EDGE, pair_indices, 0)
        keep = ~(
            (pair_dimensions == BRepEntityDimension.EDGE)
            & np.asarray(self.edge_degenerate)[edge_rows]
        )
        if occurrence_paths is not None:
            if len(occurrence_paths) != queries.shape[0]:
                raise ValueError(
                    "One occurrence path is required per classification query."
                )
            keep &= np.asarray(
                [
                    path == occurrence_paths[row]
                    for row, path in zip(query_rows.tolist(), pair_paths, strict=True)
                ],
                dtype=np.bool_,
            )
        if groups is not None:
            keep &= self._allowed(
                np.asarray(groups, dtype=np.int64)[query_rows],
                pair_dimensions,
                pair_indices,
                allowed_groups,
                allowed_dimensions,
                allowed_indices,
                pair_paths,
                allowed_occurrence_paths,
            )
        query_rows = query_rows[keep]
        candidates = self.project(
            queries[query_rows],
            pair_dimensions[keep],
            pair_indices[keep],
            occurrence_paths=tuple(
                path
                for path, selected in zip(pair_paths, keep.tolist(), strict=True)
                if selected
            ),
        )
        within = np.asarray(candidates.residuals) <= radius
        return self._lowest(
            queries, query_rows[within], candidates, np.flatnonzero(within)
        )

    def _allowed(
        self,
        query_groups: np.ndarray,
        dimensions: np.ndarray,
        indices: np.ndarray,
        allowed_groups: ArrayLike | None,
        allowed_dimensions: ArrayLike | None,
        allowed_indices: ArrayLike | None,
        /,
        occurrence_paths: tuple[tuple[str, ...], ...],
        allowed_occurrence_paths: tuple[tuple[str, ...], ...] | None,
    ) -> np.ndarray:
        if (
            allowed_groups is None
            or allowed_dimensions is None
            or allowed_indices is None
        ):
            raise ValueError("Grouped classification requires the allowed entity table.")
        total = sum(self.entity_counts) * (len(self.occurrence_paths) + 1)
        table = np.unique(
            np.asarray(allowed_groups, dtype=np.int64) * total
            + self.entity_codes(
                allowed_dimensions,
                allowed_indices,
                occurrence_paths=allowed_occurrence_paths,
            )
        )
        query = query_groups * total + self.entity_codes(
            dimensions, indices, occurrence_paths=occurrence_paths
        )
        if table.size == 0:
            return np.zeros(query.shape, dtype=np.bool_)
        position = np.minimum(np.searchsorted(table, query), table.size - 1)
        return (query_groups >= 0) & (table[position] == query)

    def _lowest(
        self,
        queries: np.ndarray,
        owners: np.ndarray,
        candidates: BRepProjectionResult,
        rows: np.ndarray,
        /,
    ) -> BRepProjectionResult:
        count = queries.shape[0]
        ambient = self.ambient_dimension
        dimensions = np.asarray(candidates.dimensions)[rows].astype(np.int64)
        lowest = np.full((count,), 3, dtype=np.int64)
        np.minimum.at(lowest, owners, dimensions)
        chosen = dimensions == lowest[owners]
        owners, rows = owners[chosen], rows[chosen]
        residuals = np.asarray(candidates.residuals)[rows]
        order = np.lexsort((residuals, owners))
        owners, rows = owners[order], rows[order]
        selected_owners, first = np.unique(owners, return_index=True)
        selected_rows = rows[first]
        points = np.asarray(candidates.points)
        leader = points[selected_rows][np.searchsorted(selected_owners, owners)]
        spread = np.zeros((count,), dtype=np.float64)
        np.maximum.at(spread, owners, np.linalg.norm(points[rows] - leader, axis=1))
        ambiguous = spread > self.policy.ambiguity_tolerance
        occurrence_indices = np.asarray(candidates.occurrence_indices)
        leader_occurrence = occurrence_indices[selected_rows][
            np.searchsorted(selected_owners, owners)
        ]
        identity_spread = np.zeros((count,), dtype=np.bool_)
        np.logical_or.at(
            identity_spread, owners, occurrence_indices[rows] != leader_occurrence
        )
        ambiguous |= identity_spread

        def gather(values: Array, fill: float, shape: tuple[int, ...] = ()) -> Any:
            output = np.full((count,) + shape, fill, dtype=np.float64)
            output[selected_owners] = np.asarray(values)[selected_rows]
            return output

        status = gather(candidates.status, BRepProjectionStatus.FAILED).astype(np.int8)
        status[ambiguous] = BRepProjectionStatus.AMBIGUOUS
        return BRepProjectionResult(
            self.source_revision,
            gather(candidates.dimensions, -1).astype(np.int8),
            gather(candidates.indices, -1).astype(np.int32),
            queries,
            gather(candidates.points, np.nan, (ambient,)),
            gather(candidates.parameters, np.nan, (2,)),
            gather(candidates.residuals, np.inf),
            status,
            gather(candidates.normals, np.nan, (ambient,)),
            gather(candidates.tangents, np.nan, (2, ambient)),
            occurrence_indices=gather(candidates.occurrence_indices, -1).astype(np.int32),
            occurrence_paths=candidates.occurrence_paths,
        )

    @abstractmethod
    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Solid containing each point; interface and multiply contained points are AMBIGUOUS."""
        raise NotImplementedError


__all__ = [
    "AbstractBRepProjection",
    "BRepEntityDimension",
    "BRepProjectionPolicy",
    "BRepProjectionResult",
    "BRepProjectionStatus",
    "brep_entity_id",
]
