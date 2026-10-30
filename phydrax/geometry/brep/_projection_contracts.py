#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...typing import checked
from .._planar_embedding import PlanarEmbedding
from .._source_identity import source_entity_id
from ._model import BRepTopology


class BRepEntityDimension(IntEnum):
    """Topological dimension of a B-Rep entity kind."""

    VERTEX = 0
    EDGE = 1
    FACE = 2
    SOLID = 3

class BRepProjectionStatus(IntEnum):
    """Pointwise query outcome, preserving unresolved and non-isolated cases."""

    UNIQUE = 0
    SEAM = 1
    AMBIGUOUS = 2
    FAILED = 3
    UNRESOLVED = 4
    NON_ISOLATED = 5
    SINGULAR = 6


_KINDS = ("vertex", "edge", "face", "solid")


def brep_entity_id(
    source_id: str,
    source_revision: str,
    dimension: int,
    index: int,
    /,
    *,
    occurrence_path: tuple[str, ...] = (),
) -> str:
    """Return the canonical owner/revision/occurrence key for a B-Rep entity."""
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer.")
    if dimension not in range(len(_KINDS)):
        raise ValueError("dimension must select a B-Rep entity kind.")
    return source_entity_id(
        source_id,
        source_revision,
        _KINDS[dimension],
        index,
        occurrence_path=occurrence_path,
    )


@final
class BRepProjectionPolicy(StrictModule, NonTrainableState):
    """Physical and parameter tolerances for native B-Rep projection.

    ``ambiguity_tolerance`` is a physical Euclidean-distance tie band: distinct
    source minima within it remain ambiguous even when their distance bounds are narrow.
    """

    ambiguity_tolerance: float = eqx.field(static=True)
    parametric_tolerance: float = eqx.field(static=True)
    classifier_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
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
    """Row-aligned closest-point evidence bound to one physical source revision.

    `residuals` and `distance_lower_bounds` are certified distance bounds, not
    merely norms of the published floating-point representative. A non-isolated
    result retains its source entity and distance interval without selecting a
    point or unique parameter.
    `occurrence_paths` is row-aligned; `()` explicitly denotes definition-only
    geometry.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    dimensions: Array
    indices: Array
    queries: Array
    points: Array
    parameters: Array
    residuals: Array
    distance_lower_bounds: Array
    status: Array
    normals: Array
    tangents: Array
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_id: str,
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
        distance_lower_bounds: ArrayLike | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...],
        frame_id: str,
    ) -> None:
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string.")
        if not source_id or source_id.strip() != source_id:
            raise ValueError("source_id must be nonempty and whitespace-trimmed.")
        if not isinstance(source_revision, str):
            raise TypeError("source_revision must be a string.")
        revision = source_revision
        if not revision or revision.strip() != revision:
            raise ValueError("source_revision must be nonempty and whitespace-trimmed.")
        if not isinstance(frame_id, str):
            raise TypeError("frame_id must be a string.")
        if not frame_id or frame_id.strip() != frame_id:
            raise ValueError("frame_id must be nonempty and whitespace-trimmed.")
        raw_dims = np.asarray(dimensions)
        raw_rows = np.asarray(indices)
        raw_state = np.asarray(status)
        if any(
            not np.issubdtype(value.dtype, np.integer)
            for value in (raw_dims, raw_rows, raw_state)
        ):
            raise TypeError("Projection dimensions, indices, and status must be integer arrays.")
        query = np.asarray(queries, dtype=np.float64)
        closest = np.asarray(points, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        upper = np.asarray(residuals, dtype=np.float64)
        normal = np.asarray(normals, dtype=np.float64)
        tangent = np.asarray(tangents, dtype=np.float64)
        count = raw_dims.shape[0] if raw_dims.ndim == 1 else -1
        ambient = query.shape[1] if query.ndim == 2 else -1
        lower = (
            np.zeros(upper.shape, dtype=np.float64)
            if distance_lower_bounds is None
            else np.asarray(distance_lower_bounds, dtype=np.float64)
        )
        paths = occurrence_paths
        if not isinstance(paths, tuple):
            raise TypeError("occurrence_paths must be a tuple of row-aligned paths.")
        if raw_dims.ndim == 1 and np.any((raw_dims < -1) | (raw_dims > 3)):
            raise ValueError("Projection dimensions must lie in [-1, 3].")
        if raw_rows.ndim == 1 and np.any(
            (raw_rows < np.iinfo(np.int32).min) | (raw_rows > np.iinfo(np.int32).max)
        ):
            raise ValueError("Projection indices must fit in int32.")
        if raw_state.ndim == 1 and np.any(
            ~np.isin(raw_state, [int(value) for value in BRepProjectionStatus])
        ):
            raise ValueError("Projection status values are invalid.")
        if len(paths) != count or any(
            not isinstance(path, tuple)
            or any(not isinstance(name, str) or not name for name in path)
            for path in paths
        ):
            raise ValueError(
                "occurrence_paths must contain one exact path tuple per projection row."
            )
        dims = raw_dims.astype(np.int8)
        rows = raw_rows.astype(np.int32)
        state = raw_state.astype(np.int8)
        if (
            dims.ndim != 1
            or ambient not in (2, 3)
            or rows.shape != dims.shape
            or upper.shape != dims.shape
            or lower.shape != dims.shape
            or state.shape != dims.shape
            or query.shape != (count, ambient)
            or closest.shape != (count, ambient)
            or params.shape != (count, 2)
            or normal.shape != (count, ambient)
            or tangent.shape != (count, 2, ambient)
        ):
            raise ValueError("Projection arrays must be row-aligned with one ambient dimension.")
        if not np.all(np.isfinite(query)):
            raise ValueError("Projection queries must be finite.")
        if np.any(np.isinf(closest)) or np.any(np.isinf(params)):
            raise ValueError("Projection points and parameters may contain NaN but not infinity.")
        if np.any(np.isinf(normal)) or np.any(np.isinf(tangent)):
            raise ValueError("Projection frames may contain NaN but not infinity.")
        if np.any(np.isnan(upper) | (upper < 0.0)) or np.any(
            ~np.isfinite(lower) | (lower < 0.0) | (lower > upper)
        ):
            raise ValueError("Projection distance bounds must be ordered and nonnegative.")
        resolved = np.isin(
            state,
            (
                BRepProjectionStatus.UNIQUE,
                BRepProjectionStatus.SEAM,
                BRepProjectionStatus.AMBIGUOUS,
                BRepProjectionStatus.SINGULAR,
            ),
        )
        if np.any(
            resolved
            & (
                (dims < 0)
                | (rows < 0)
                | ~np.isfinite(upper)
                | ~np.all(np.isfinite(closest), axis=1)
            )
        ):
            raise ValueError("Resolved projection rows require finite points, bounds, and source identity.")
        parametric = np.isin(dims, (BRepEntityDimension.EDGE, BRepEntityDimension.FACE))
        if np.any(resolved & parametric & ~np.isfinite(params[:, 0])):
            raise ValueError("Resolved edge and face rows require a finite first parameter.")
        if np.any(resolved & (dims == BRepEntityDimension.FACE) & ~np.isfinite(params[:, 1])):
            raise ValueError("Resolved face rows require two finite parameters.")
        unclassified = dims < 0
        if np.any(unclassified & (rows >= 0)):
            raise ValueError("Unclassified projection rows cannot carry entity indices.")
        fiber = state == BRepProjectionStatus.NON_ISOLATED
        if np.any(fiber) and (
            np.any(~np.isin(dims[fiber], (1, 2)))
            or np.any(rows[fiber] < 0)
            or np.any(~np.isfinite(upper[fiber]))
            or np.any(~np.all(np.isnan(closest[fiber]), axis=1))
            or np.any(~np.any(np.isnan(params[fiber]), axis=1))
        ):
            raise ValueError("NON_ISOLATED rows must retain bounded source identity without a selected point.")
        self.source_id = source_id
        self.frame_id = frame_id
        self.source_revision = revision
        self.dimensions = jnp.asarray(dims)
        self.indices = jnp.asarray(rows)
        self.queries = jnp.asarray(query)
        self.points = jnp.asarray(closest)
        self.parameters = jnp.asarray(params)
        self.residuals = jnp.asarray(upper)
        self.distance_lower_bounds = jnp.asarray(lower)
        self.status = jnp.asarray(state)
        self.normals = jnp.asarray(normal)
        self.tangents = jnp.asarray(tangent)
        self.occurrence_paths = paths
        self.result_id = canonical_fingerprint(
            {
                "kind": "brep-projection-result",
                "source_id": source_id,
                "frame_id": frame_id,
                "source_revision": revision,
                "indices": array_tree_fingerprint(rows),
                "queries": array_tree_fingerprint(query),
                "points": array_tree_fingerprint(closest),
                "parameters": array_tree_fingerprint(params),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
                "status": array_tree_fingerprint(state),
                "normals": array_tree_fingerprint(normal),
                "tangents": array_tree_fingerprint(tangent),
                "occurrence_paths": paths,
            }
        )

    @property
    def resolved(self) -> Array:
        return (self.status == BRepProjectionStatus.UNIQUE) | (
            self.status == BRepProjectionStatus.SEAM
        )


    def entity_ids(self) -> tuple[str, ...]:
        return tuple(
            brep_entity_id(
                self.source_id,
                self.source_revision,
                int(dimension),
                int(index),
                occurrence_path=path,
            )
            if dimension >= 0
            else ""
            for dimension, index, path in zip(
                np.asarray(self.dimensions).tolist(),
                np.asarray(self.indices).tolist(),
                self.occurrence_paths,
                strict=True,
            )
        )


def _matches(sorted_keys: np.ndarray, queries: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    start = np.searchsorted(sorted_keys, queries, side="left")
    counts = np.searchsorted(sorted_keys, queries, side="right") - start
    owners = np.repeat(np.arange(queries.size), counts)
    positions = np.repeat(start - (np.cumsum(counts) - counts), counts)
    return owners, positions + np.arange(owners.size)


@checked
def closure_codes(
    topology: BRepTopology,
    edge_vertices: tuple[tuple[int, int], ...],
    counts: tuple[int, int, int, int],
    /,
) -> np.ndarray:
    """Sorted reflexive closure codes over vertex, edge, face, and solid banks."""
    vertex_count, edge_count, face_count, solid_count = counts
    if vertex_count != topology.num_vertices or edge_count != topology.num_edges:
        raise ValueError("Entity counts must agree with B-Rep topology.")
    offsets = np.cumsum((0, vertex_count, edge_count, face_count))
    total = int(offsets[-1] + solid_count)
    endpoints = np.asarray(edge_vertices, dtype=np.int64)
    if endpoints.shape != (edge_count, 2):
        raise ValueError("edge_vertices must contain two endpoint IDs per edge.")
    edge_vertex = np.stack(
        (np.repeat(np.arange(edge_count), 2), endpoints.reshape(-1)), axis=1
    )
    edge_vertex = edge_vertex[edge_vertex[:, 1] >= 0]
    face_edge = np.asarray(
        [(face, edge) for face, edges in enumerate(topology.face_edges) for edge in edges],
        dtype=np.int64,
    ).reshape((-1, 2))
    face_vertex = np.concatenate(
        [
            np.stack((face_edge[:, 0], endpoints[face_edge[:, 1], side]), axis=1)
            for side in (0, 1)
        ],
        axis=0,
    )
    face_vertex = face_vertex[face_vertex[:, 1] >= 0]
    solid_face = np.asarray(
        [(solid, face) for solid, faces in enumerate(topology.solid_faces) for face in faces],
        dtype=np.int64,
    ).reshape((-1, 2))

    def through(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
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
    encoded = np.concatenate(pairs).astype(np.int64)
    return np.unique(encoded[:, 0] * total + encoded[:, 1])


class AbstractBRepProjection(StrictModule, NonTrainableState):
    """Engine-neutral source projection and solid-location contract."""

    source_id: eqx.AbstractVar[str]
    frame_id: eqx.AbstractVar[str]
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
        raise NotImplementedError

    def entity_codes(self, dimensions: ArrayLike, indices: ArrayLike, /) -> np.ndarray:
        """Encode aligned definition-entity rows in the canonical topology bank."""
        raise NotImplementedError

    def code_entities(self, codes: ArrayLike, /) -> tuple[np.ndarray, np.ndarray]:
        """Decode topology-bank codes to dimensions and row indices."""
        raise NotImplementedError

    def members(
        self, dimensions: ArrayLike, indices: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return definition-topology closure rows for each input entity."""
        raise NotImplementedError

    def containers(
        self, dimensions: ArrayLike, indices: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return definition-topology containers for each input entity."""
        raise NotImplementedError

    def contains(
        self,
        container_dimensions: ArrayLike,
        container_indices: ArrayLike,
        child_dimensions: ArrayLike,
        child_indices: ArrayLike,
        /,
    ) -> np.ndarray:
        """Test aligned definition-topology containment, excluding occurrences."""
        raise NotImplementedError

    def occurrence_paths_for_entities(
        self,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        """Return source occurrence paths that own each definition entity."""
        raise NotImplementedError

    @abstractmethod
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
        allowed_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        raise NotImplementedError

    @abstractmethod
    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        raise NotImplementedError


__all__ = [
    "AbstractBRepProjection",
    "BRepEntityDimension",
    "BRepProjectionPolicy",
    "BRepProjectionResult",
    "BRepProjectionStatus",
    "brep_entity_id",
    "closure_codes",
]
