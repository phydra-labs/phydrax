#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit OCCT comparison queries against one persisted B-Rep revision.

Every query needs one host call into OCCT extrema; that per-point call is the
external-provider boundary and cannot be vectorized. Queries are grouped by
entity so each face and edge prepares its surface/curve projector,
trimmed-face classifier, parameter bounds, and seam/pole data exactly once.
Bookkeeping around the calls (grouping, BVH candidate search, closure
relations, frames, and status reduction) is vectorized NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._bvh import bvh_overlap_pairs_host, prepare_bvh
from .._fingerprint import canonical_fingerprint
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry.brep._model import BRepModel
from ..geometry.brep._projection_contracts import (
    _closure_codes,
    AbstractBRepProjection,
    BRepEntityDimension,
    BRepProjectionPolicy,
    BRepProjectionResult,
    BRepProjectionStatus,
)
from ._cad_occt import _explore_unique, _shape_digest, _shape_index, read_occt_shape


@dataclass(frozen=True, slots=True, eq=False)
class _Occt:
    """Host OCCT handles of one revision (identity-compared, never fingerprinted)."""

    faces: tuple[Any, ...]
    edges: tuple[Any, ...]
    solids: tuple[Any, ...]
    surfaces: tuple[Any, ...]
    curves: tuple[Any, ...]


@dataclass(slots=True)
class _Rows:
    """Per-query accumulation of one projection batch (world coordinates)."""

    points: np.ndarray
    parameters: np.ndarray
    residuals: np.ndarray
    status: np.ndarray
    first: np.ndarray
    second: np.ndarray
    normals: np.ndarray

    @classmethod
    def empty(cls, count: int, /) -> _Rows:
        return cls(
            np.full((count, 3), np.nan),
            np.full((count, 2), np.nan),
            np.full((count,), np.inf),
            np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8),
            np.full((count, 3), np.nan),
            np.full((count, 3), np.nan),
            np.full((count, 3), np.nan),
        )


def _xyz(value: Any, /) -> np.ndarray:
    return np.asarray((value.X(), value.Y(), value.Z()), dtype=np.float64)


def _box(shape: Any, /) -> np.ndarray:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    # OCP 8 no longer converts ``Bnd_Box.Get``'s ``Limits`` struct; read the corners.
    return np.stack((_xyz(box.CornerMin()), _xyz(box.CornerMax())))


def _bound_shape(model: BRepModel, source: Any, /) -> Any:
    """Load the OCCT shape and prove it is the model's exact persisted revision."""
    from OCP.TopoDS import TopoDS_Shape

    if isinstance(source, (str, Path)):
        shape, _, digest = read_occt_shape(source)
    elif isinstance(source, TopoDS_Shape):
        if source.IsNull():
            raise ValueError("Cannot bind a null OCCT shape.")
        shape, digest = source, _shape_digest(source)
    else:
        raise TypeError("source must be an OCCT TopoDS_Shape or a CAD file path.")
    if digest != model.source_digest:
        raise ValueError(
            "The OCCT source does not match the B-Rep model's persisted revision digest."
        )
    return shape


@final
class PreparedOcctProjection(AbstractBRepProjection):
    """Host OCCT projectors and classifiers of one revision.

    Entity indices follow the B-Rep model (first-seen unique OCCT exploration),
    so ``(dimension, index)`` pairs are the ``BRepEntityId`` identities of
    ``source_revision``. With a planar ``embedding`` queries and results use the
    embedding's two-dimensional frame. Construct with
    :func:`prepare_occt_projection`.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    policy: BRepProjectionPolicy
    embedding: PlanarEmbedding | None = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    vertex_points: Array
    edge_vertices: Array
    edge_degenerate: Array
    edge_closed: Array
    edge_ranges: Array
    face_bounds: Array
    face_orientation: Array
    face_periods: Array
    entity_counts: tuple[int, int, int, int] = eqx.field(static=True)
    closure_codes: Array
    entity_boxes: Array
    face_poles: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    occt: _Occt = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    entity_occurrences: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    qualified_entity_codes: Array
    qualified_entity_boxes: Array
    qualified_closure_codes: Array

    def __init__(
        self,
        model: BRepModel,
        shape: Any,
        policy: BRepProjectionPolicy,
        embedding: PlanarEmbedding | None,
        /,
    ) -> None:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCP.BRepTools import BRepTools
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
        from OCP.TopExp import TopExp
        from OCP.TopoDS import TopoDS

        faces = tuple(_explore_unique(shape, TopAbs_FACE, TopoDS.Face))
        edges = tuple(_explore_unique(shape, TopAbs_EDGE, TopoDS.Edge))
        vertices = _explore_unique(shape, TopAbs_VERTEX, TopoDS.Vertex)
        solids = tuple(_explore_unique(shape, TopAbs_SOLID, TopoDS.Solid))
        topology = model.topology
        if (len(faces), len(edges), len(vertices), len(solids)) != (
            topology.num_faces,
            topology.num_edges,
            topology.num_vertices,
            topology.num_solids,
        ):
            raise ValueError("The OCCT shape topology does not match the B-Rep model.")
        vertex_points = np.asarray(
            [_xyz(BRep_Tool.Pnt_s(vertex)) for vertex in vertices], dtype=np.float64
        ).reshape(-1, 3)
        edge_vertices = np.full((len(edges), 2), -1, dtype=np.int64)
        degenerate = np.zeros((len(edges),), dtype=np.bool_)
        closed = np.zeros((len(edges),), dtype=np.bool_)
        ranges = np.zeros((len(edges), 2), dtype=np.float64)
        curves = []
        for row, edge in enumerate(edges):
            first, last = TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)
            if not first.IsNull():
                edge_vertices[row, 0] = _shape_index(vertices, first)
            if not last.IsNull():
                edge_vertices[row, 1] = _shape_index(vertices, last)
            degenerate[row] = BRep_Tool.Degenerated_s(edge)
            adaptor = BRepAdaptor_Curve(edge)
            ranges[row] = (adaptor.FirstParameter(), adaptor.LastParameter())
            closed[row] = (
                not degenerate[row]
                and edge_vertices[row, 0] == edge_vertices[row, 1]
                and adaptor.IsClosed()
            )
            curves.append(
                None
                if degenerate[row]
                else BRep_Tool.Curve_s(edge, float(ranges[row, 0]), float(ranges[row, 1]))
            )
        surfaces, bounds, periods, poles = [], [], [], []
        for face_index, face in enumerate(faces):
            surfaces.append(BRep_Tool.Surface_s(face))
            u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
            bounds.append((u0, u1, v0, v1))
            adaptor = BRepAdaptor_Surface(face, True)
            u_period = adaptor.UPeriod() if adaptor.IsUPeriodic() else np.nan
            v_period = adaptor.VPeriod() if adaptor.IsVPeriodic() else np.nan
            # Only a face spanning its full period has a parametric seam.
            periods.append(
                (
                    u_period if u1 - u0 >= u_period * (1.0 - 1.0e-9) else np.nan,
                    v_period if v1 - v0 >= v_period * (1.0 - 1.0e-9) else np.nan,
                )
            )
            poles.append(
                tuple(
                    sorted(
                        {
                            int(edge_vertices[edge, 0])
                            for edge in topology.face_edges[face_index]
                            if degenerate[edge] and edge_vertices[edge, 0] >= 0
                        }
                    )
                )
            )
        counts = (len(vertices), len(edges), len(faces), len(solids))
        boxes = np.concatenate(
            (
                np.stack((vertex_points, vertex_points), axis=1),
                np.asarray([_box(edge) for edge in edges]).reshape(-1, 2, 3),
                np.asarray([_box(face) for face in faces]).reshape(-1, 2, 3),
                np.asarray([_box(solid) for solid in solids]).reshape(-1, 2, 3),
            )
        )
        ambient = 3
        if embedding is not None:
            if not isinstance(embedding, PlanarEmbedding):
                raise TypeError("embedding must be PlanarEmbedding or None.")
            if solids:
                raise ValueError("A planar embedding binds face-only B-Rep revisions.")
            residual = np.max(np.abs(embedding.plane_residual(vertex_points)))
            if residual > policy.classifier_tolerance:
                raise ValueError(
                    "The B-Rep revision does not lie on its planar embedding."
                )
            ambient = 2
        self.source_id = model.source_id
        self.source_revision = model.source_revision
        self.model_id = model.model_id
        self.policy = policy
        self.embedding = embedding
        self.ambient_dimension = ambient
        self.vertex_points = jnp.asarray(vertex_points)
        self.edge_vertices = jnp.asarray(edge_vertices.astype(np.int32))
        self.edge_degenerate = jnp.asarray(degenerate)
        self.edge_closed = jnp.asarray(closed)
        self.edge_ranges = jnp.asarray(ranges)
        self.face_bounds = jnp.asarray(
            np.asarray(bounds, dtype=np.float64).reshape(-1, 4)
        )
        self.face_orientation = jnp.asarray(np.asarray(model.orientation))
        self.face_periods = jnp.asarray(
            np.asarray(periods, dtype=np.float64).reshape(-1, 2)
        )
        self.entity_counts = counts
        self.closure_codes = jnp.asarray(_closure_codes(topology, edge_vertices, counts))
        self.entity_boxes = jnp.asarray(boxes)
        self.occurrence_paths = ()
        self.entity_occurrences = tuple((-1,) for _ in range(sum(counts)))
        self.qualified_entity_codes = jnp.arange(sum(counts), dtype=jnp.int64)
        self.qualified_entity_boxes = jnp.asarray(boxes, dtype=jnp.float64)
        self.qualified_closure_codes = self.closure_codes
        self.face_poles = tuple(poles)
        self.occt = _Occt(faces, edges, solids, tuple(surfaces), tuple(curves))
        self.projection_id = canonical_fingerprint(
            {
                "kind": "prepared-brep-projection",
                "source_revision": model.source_revision,
                "model_id": model.model_id,
                "policy": policy.policy_id,
                "embedding": None if embedding is None else embedding.embedding_id,
            }
        )

    # -- projection -------------------------------------------------------------

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
        queries = self._queries(points, "Projection queries")
        if occurrence_paths is not None and (
            len(occurrence_paths) != queries.shape[0] or any(occurrence_paths)
        ):
            raise ValueError(
                "The external comparison shape exposes only root-qualified entities."
            )
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError("Projection entity dimensions and indices must be integers.")
        if dims.shape != (queries.shape[0],) or rows.shape != dims.shape:
            raise ValueError("One entity dimension and index is required per query.")
        if np.any((dims < 0) | (dims > 2)):
            raise ValueError("Projection targets vertices, edges, or faces only.")
        limits = np.asarray(self.entity_counts[:3])[dims]
        if np.any((rows < 0) | (rows >= limits)):
            raise ValueError("Projection entity indices are out of range.")
        world = self._world(queries)
        state = _Rows.empty(queries.shape[0])
        vertex = np.flatnonzero(dims == BRepEntityDimension.VERTEX)
        state.points[vertex] = np.asarray(self.vertex_points)[rows[vertex]]
        state.residuals[vertex] = np.linalg.norm(
            world[vertex] - state.points[vertex], axis=1
        )
        state.status[vertex] = BRepProjectionStatus.UNIQUE
        codes = dims.astype(np.int64) * (1 << 32) + rows
        order = np.argsort(codes, kind="stable")
        groups, starts = np.unique(codes[order], return_index=True)
        stops = np.append(starts[1:], order.size)[: starts.size]
        for code, start, stop in zip(groups.tolist(), starts, stops, strict=True):
            dimension, entity = code >> 32, code & 0xFFFFFFFF
            selected = order[start:stop]
            match dimension:
                case BRepEntityDimension.VERTEX:
                    continue
                case BRepEntityDimension.EDGE:
                    self._project_edge(entity, world, selected, state)
                case BRepEntityDimension.FACE:
                    self._project_face(entity, world, selected, state)
                case _:
                    raise ValueError("Unsupported projection entity dimension.")
        return self._result(dims, rows, queries, state)

    def _project_edge(
        self, edge: int, queries: np.ndarray, selected: np.ndarray, state: _Rows, /
    ) -> None:
        from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
        from OCP.gp import gp_Pnt, gp_Vec

        tolerance = self.policy.ambiguity_tolerance
        if bool(np.asarray(self.edge_degenerate)[edge]):
            # A degenerate edge is geometrically its single vertex.
            point = np.asarray(self.vertex_points)[int(self.edge_vertices[edge, 0])]
            state.points[selected] = point
            state.residuals[selected] = np.linalg.norm(queries[selected] - point, axis=1)
            state.status[selected] = BRepProjectionStatus.UNIQUE
            return
        curve = self.occt.curves[edge]
        first, last = (float(value) for value in np.asarray(self.edge_ranges)[edge])
        projector = GeomAPI_ProjectPointOnCurve()
        projector.Init(curve, first, last)
        ends = np.stack((_xyz(curve.Value(first)), _xyz(curve.Value(last))))
        closed = bool(np.asarray(self.edge_closed)[edge])
        span = last - first
        for row in selected:
            query = queries[row]
            projector.Perform(gp_Pnt(*query))
            count = projector.NbPoints()
            # Interior extrema plus the trimmed end points bound the closest point.
            candidates = np.concatenate(
                (
                    np.asarray(
                        [_xyz(projector.Point(i)) for i in range(1, count + 1)]
                    ).reshape(-1, 3),
                    ends,
                )
            )
            parameters = np.append(
                np.asarray([projector.Parameter(i) for i in range(1, count + 1)]),
                (first, last),
            )
            best, status = _select(query, candidates, tolerance)
            same = np.linalg.norm(candidates - candidates[best], axis=1) <= tolerance
            parameter = float(np.min(parameters[same]))
            if status == BRepProjectionStatus.UNIQUE and count == 0:
                # No isolated extremum: a constant distance around the answer is a
                # continuum of minima (e.g. a circle's center).
                probes = parameter + 1.0e-3 * span * np.asarray((-1.0, 1.0))
                probes = probes[(probes >= first) & (probes <= last)]
                distances = [np.linalg.norm(_xyz(curve.Value(p)) - query) for p in probes]
                if _flat(
                    distances,
                    float(np.linalg.norm(candidates[best] - query)),
                    tolerance,
                ):
                    status = BRepProjectionStatus.AMBIGUOUS
            if status == BRepProjectionStatus.UNIQUE and closed:
                seam = min(parameter - first, last - parameter) <= (
                    self.policy.parametric_tolerance * span
                )
                status = BRepProjectionStatus.SEAM if seam else status
            point = gp_Pnt()
            derivative = gp_Vec()
            curve.D1(parameter, point, derivative)
            state.points[row] = _xyz(point)
            state.parameters[row, 0] = parameter
            state.residuals[row] = float(np.linalg.norm(query - state.points[row]))
            state.status[row] = status
            state.first[row] = _xyz(derivative)

    def _project_face(
        self, face: int, queries: np.ndarray, selected: np.ndarray, state: _Rows, /
    ) -> None:
        from OCP.BRepClass import BRepClass_FaceClassifier
        from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCP.GeomLProp import GeomLProp_SLProps
        from OCP.gp import gp_Pnt, gp_Pnt2d
        from OCP.ShapeAnalysis import ShapeAnalysis_Surface
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON

        tolerance = self.policy.ambiguity_tolerance
        classifier_tolerance = self.policy.classifier_tolerance
        shape = self.occt.faces[face]
        surface = self.occt.surfaces[face]
        u0, u1, v0, v1 = (float(value) for value in np.asarray(self.face_bounds)[face])
        projector = GeomAPI_ProjectPointOnSurf()
        projector.Init(surface, u0, u1, v0, v1, classifier_tolerance)
        classifier = BRepClass_FaceClassifier()
        analysis = ShapeAnalysis_Surface(surface)
        orientation = float(np.asarray(self.face_orientation)[face])
        for row in selected:
            query = queries[row]
            projector.Perform(gp_Pnt(*query))
            candidates, parameters = [], []
            for i in range(1, projector.NbPoints() + 1):
                u, v = projector.Parameters(i)
                classifier.Perform(shape, gp_Pnt2d(u, v), classifier_tolerance)
                if classifier.State() in (TopAbs_IN, TopAbs_ON):
                    candidates.append(_xyz(projector.Point(i)))
                    parameters.append((u, v))
            if not candidates:
                # No surface extremum inside the trim (or a continuum of minima):
                # the exact trimmed-face distance comes from BRepExtrema.
                candidates, parameters = _trimmed_candidates(
                    shape, analysis, query, classifier_tolerance
                )
            if not candidates:
                continue
            points = np.asarray(candidates, dtype=np.float64)
            params = np.asarray(parameters, dtype=np.float64)
            best, status = _select(query, points, tolerance)
            same = np.linalg.norm(points - points[best], axis=1) <= tolerance
            u, v = params[same][np.lexsort(params[same].T[::-1])[0]]
            if status == BRepProjectionStatus.UNIQUE and projector.NbPoints() == 0:
                # No isolated surface extremum: constant distance along a parameter
                # direction is a continuum of minima (e.g. a point on a cylinder axis).
                distance = float(np.linalg.norm(points[best] - query))
                for step in ((1.0e-3 * (u1 - u0), 0.0), (0.0, 1.0e-3 * (v1 - v0))):
                    probes = [
                        _xyz(surface.Value(u + sign * step[0], v + sign * step[1]))
                        for sign in (-1.0, 1.0)
                    ]
                    if _flat(
                        [np.linalg.norm(probe - query) for probe in probes],
                        distance,
                        tolerance,
                    ):
                        status = BRepProjectionStatus.AMBIGUOUS
            status = self._face_status(face, points[best], u, v, status)
            properties = GeomLProp_SLProps(surface, u, v, 1, tolerance)
            state.points[row] = points[best]
            state.parameters[row] = (u, v)
            state.residuals[row] = float(np.linalg.norm(query - points[best]))
            state.status[row] = status
            state.first[row] = _xyz(properties.D1U())
            state.second[row] = _xyz(properties.D1V())
            if properties.IsNormalDefined():
                state.normals[row] = _xyz(properties.Normal()) * orientation

    def _face_status(
        self, face: int, point: np.ndarray, u: float, v: float, status: int, /
    ) -> int:
        if status != BRepProjectionStatus.UNIQUE:
            return status
        u0, u1, v0, v1 = np.asarray(self.face_bounds)[face]
        periods = np.asarray(self.face_periods)[face]
        tolerance = self.policy.parametric_tolerance
        seam = bool(
            (np.isfinite(periods[0]) and min(u - u0, u1 - u) <= tolerance * periods[0])
            or (np.isfinite(periods[1]) and min(v - v0, v1 - v) <= tolerance * periods[1])
        )
        poles = np.asarray(self.vertex_points)[list(self.face_poles[face])]
        singular = poles.size > 0 and bool(
            np.any(
                np.linalg.norm(poles - point, axis=1) <= self.policy.ambiguity_tolerance
            )
        )
        return BRepProjectionStatus.SEAM if seam or singular else status

    def _result(
        self,
        dimensions: np.ndarray,
        indices: np.ndarray,
        queries: np.ndarray,
        state: _Rows,
        /,
    ) -> BRepProjectionResult:
        count = dimensions.size
        tangents = np.full((count, 2, 3), np.nan)
        first_norm = np.linalg.norm(state.first, axis=1)
        usable = first_norm > 0.0
        tangents[usable, 0] = state.first[usable] / first_norm[usable, None]
        normal_norm = np.linalg.norm(state.normals, axis=1)
        face = (dimensions == BRepEntityDimension.FACE) & (normal_norm > 0.0)
        normals = np.full((count, 3), np.nan)
        normals[face] = state.normals[face] / normal_norm[face, None]
        # At a pole the first derivative vanishes: seed the frame with the other.
        second_norm = np.linalg.norm(state.second, axis=1)
        swap = face & ~usable & (second_norm > 0.0)
        tangents[swap, 0] = state.second[swap] / second_norm[swap, None]
        frame = face & np.all(np.isfinite(tangents[:, 0]), axis=1)
        tangents[frame, 0] -= (
            np.sum(tangents[frame, 0] * normals[frame], axis=1)[:, None] * normals[frame]
        )
        tangents[frame, 0] /= np.linalg.norm(tangents[frame, 0], axis=1)[:, None]
        tangents[frame, 1] = np.cross(normals[frame], tangents[frame, 0])
        points = np.where(
            np.isfinite(state.points), state.points, 0.0
        )  # unresolved rows stay NaN below
        ambient_points = self._ambient(points)
        ambient_points[~np.all(np.isfinite(state.points), axis=1)] = np.nan
        if self.embedding is None:
            ambient_normals = normals
        else:
            # Face normals leave the plane: not representable in planar ambient.
            ambient_normals = np.full((count, 2), np.nan)
        return BRepProjectionResult(
            self.source_revision,
            dimensions,
            indices,
            queries,
            ambient_points,
            state.parameters,
            state.residuals,
            state.status,
            ambient_normals,
            self._directions(tangents),
        )

    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Solid containing each point (OCCT solid classification).

        Points on a shared solid interface or in several solids are AMBIGUOUS;
        points in no solid are FAILED. Residuals are zero for contained points.
        """
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON

        queries = self._queries(points, "Solid location queries")
        if occurrence_paths is not None and (
            len(occurrence_paths) != queries.shape[0] or any(occurrence_paths)
        ):
            raise ValueError(
                "The external comparison shape exposes only root-qualified entities."
            )
        if self.entity_counts[3] == 0:
            raise ValueError("The B-Rep revision has no solids.")
        count = queries.shape[0]
        boxes = np.asarray(self.entity_boxes)[sum(self.entity_counts[:3]) :]
        tolerance = self.policy.classifier_tolerance
        first = prepare_bvh(queries - tolerance, queries + tolerance, dtype=jnp.float64)
        second = prepare_bvh(boxes[:, 0], boxes[:, 1], dtype=jnp.float64)
        query_rows, solid_rows = bvh_overlap_pairs_host(
            first, second, include_touching=True
        )
        classifier = BRepClass3d_SolidClassifier()
        states = np.empty((query_rows.size,), dtype=np.int8)
        for pair, (row, solid) in enumerate(zip(query_rows, solid_rows, strict=True)):
            classifier.Load(self.occt.solids[int(solid)])
            classifier.Perform(gp_Pnt(*queries[row]), tolerance)
            state = classifier.State()
            states[pair] = 0 if state == TopAbs_IN else (1 if state == TopAbs_ON else 2)
        inside = states <= 1
        hits = np.bincount(query_rows[inside], minlength=count)
        on = np.bincount(query_rows[states == 1], minlength=count)
        index = np.full((count,), -1, dtype=np.int64)
        index[query_rows[inside]] = solid_rows[inside]
        status = np.where(
            hits == 0,
            BRepProjectionStatus.FAILED,
            np.where(
                (hits > 1) | (on > 0),
                BRepProjectionStatus.AMBIGUOUS,
                BRepProjectionStatus.UNIQUE,
            ),
        ).astype(np.int8)
        return BRepProjectionResult(
            self.source_revision,
            np.where(hits > 0, 3, -1).astype(np.int8),
            index.astype(np.int32),
            queries,
            np.where(hits[:, None] > 0, queries, np.nan),
            np.full((count, 2), np.nan),
            np.where(hits > 0, 0.0, np.inf),
            status,
            np.full((count, 3), np.nan),
            np.full((count, 2, 3), np.nan),
        )


def _flat(distances: list[float], reference: float, tolerance: float, /) -> bool:
    """Whether every probe distance equals the minimum (a continuum of minima)."""
    return bool(distances) and all(
        abs(float(value) - reference) <= tolerance for value in distances
    )


def _select(query: np.ndarray, candidates: np.ndarray, tolerance: float, /) -> Any:
    """Nearest candidate and UNIQUE/AMBIGUOUS status among distinct tied minima."""
    distances = np.linalg.norm(candidates - query, axis=1)
    best = int(np.argmin(distances))
    tied = distances <= distances[best] + tolerance
    distinct = np.linalg.norm(candidates[tied] - candidates[best], axis=1) > tolerance
    status = (
        BRepProjectionStatus.AMBIGUOUS
        if np.any(distinct)
        else BRepProjectionStatus.UNIQUE
    )
    return best, status


def _trimmed_candidates(
    face: Any, analysis: Any, query: np.ndarray, tolerance: float, /
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    extrema = BRepExtrema_DistShapeShape(
        BRepBuilderAPI_MakeVertex(gp_Pnt(*query)).Vertex(), face
    )
    if not extrema.IsDone():
        return [], []
    points, parameters = [], []
    for solution in range(1, extrema.NbSolution() + 1):
        point = extrema.PointOnShape2(solution)
        uv = analysis.ValueOfUV(point, tolerance)
        points.append(_xyz(point))
        parameters.append((uv.X(), uv.Y()))
    return points, parameters


def prepare_occt_projection(
    model: BRepModel,
    source: Any,
    /,
    *,
    policy: BRepProjectionPolicy | None = None,
    embedding: PlanarEmbedding | None = None,
) -> PreparedOcctProjection:
    """Bind explicit external queries to the verified persisted source revision."""
    if not isinstance(model, BRepModel):
        raise TypeError("model must be BRepModel.")
    policy_ = BRepProjectionPolicy() if policy is None else policy
    if not isinstance(policy_, BRepProjectionPolicy):
        raise TypeError("policy must be BRepProjectionPolicy or None.")
    return PreparedOcctProjection(model, _bound_shape(model, source), policy_, embedding)


__all__ = [
    "PreparedOcctProjection",
    "prepare_occt_projection",
]
